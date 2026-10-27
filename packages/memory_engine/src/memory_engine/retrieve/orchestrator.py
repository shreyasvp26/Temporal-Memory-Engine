"""Retrieval orchestrator coordinating query planning, parallel lanes, fusion, and assembly."""

import asyncio
import logging
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any
from zoneinfo import ZoneInfo

from memory_engine.domain.enums import Intent
from memory_engine.domain.errors import AllLanesFailed, InvariantViolation
from memory_engine.domain.ids import UserId
from memory_engine.domain.models import Entity, Message
from memory_engine.ingest.time_resolver import TimeResolver
from memory_engine.ports.clock import Clock, SystemClock
from memory_engine.ports.embed import Embedder
from memory_engine.ports.llm import ChatMessage, LLMGateway, LLMRequest, PlannerLLMOutput
from memory_engine.ports.repositories import FilterMode, TemporalFilter
from memory_engine.ports.uow import UnitOfWorkFactory
from memory_engine.retrieve.abstain import AbstentionDecision, AbstentionPolicy
from memory_engine.retrieve.assembler import AssembledContext, ContextAssembler
from memory_engine.retrieve.decay import DecayScorer
from memory_engine.retrieve.fusion import Hit, RankedItem, RrfFuser
from memory_engine.retrieve.lanes.fts import FtsLane
from memory_engine.retrieve.lanes.graph import GraphLane
from memory_engine.retrieve.lanes.temporal import TemporalLane
from memory_engine.retrieve.lanes.vector import VectorLane
from memory_engine.retrieve.planner.rules import QueryPlan, RulesPlanner, needs_llm_planner
from memory_engine.retrieve.rerank import NoopReranker, Reranker

if TYPE_CHECKING:
    from memory_engine.domain.intervals import TimeInterval

__all__ = [
    'RetrievalOrchestrator',
    'RetrievalResult',
]

_logger = logging.getLogger(__name__)

PLANNER_SYSTEM_PROMPT = """You are a retrieval query planner for a personal memory assistant.
Analyze the user question and determine retrieval intent, exact temporal cues, entity names,
predicates, and whether multi-hop reasoning is needed.

Available intents:
- current_state: Questions about current state, location, job, or attributes
- historical: Questions about the past across all history
- point_in_time: Questions as of a specific date or time
- interval: Questions about a date range or interval
- aggregation: Counting or listing facts
- duration_or_order: Questions asking about durations, order of events, or which came first
- open: Questions without specific temporal bounds

Rules:
1. Identify the intent from the allowed intents list.
2. Extract exact time phrases verbatim from the question (do not parse or convert to dates).
3. Extract mentioned entity names (people, places, organizations). First person words refer to User.
4. Extract relevant predicates or topics.
5. Set multi_hop to true if answering requires connecting multiple entities or facts.
Return only valid JSON matching the PlannerLLMOutput schema."""


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    """Consolidated retrieval result per LLD Section 9.1 and 9.4 [FROZEN]."""

    plan: QueryPlan
    ranked_items: list[RankedItem]
    context: AssembledContext
    abstention: AbstentionDecision
    lane_counts: dict[str, int]
    degraded_lanes: list[str]

    @property
    def items(self) -> list[RankedItem]:
        """Convenience property for downstream consumers expecting .items."""
        return self.ranked_items

    @property
    def has_structured_hit(self) -> bool:
        """Indicate whether any structured or temporal lane contributed a hit."""
        return any(
            'structured' in item.lanes or 'temporal_facts' in item.lanes
            for item in self.ranked_items
        )


class RetrievalOrchestrator:
    """Coordinates query planning, parallel lanes, RRF fusion, decay, rerank, and assembly."""

    def __init__(
        self,
        uow_factory: UnitOfWorkFactory | None = None,
        embedder: Embedder | None = None,
        planner: RulesPlanner | None = None,
        llm_gateway: LLMGateway | None = None,
        fuser: RrfFuser | None = None,
        decay: DecayScorer | None = None,
        reranker: Reranker | None = None,
        assembler: ContextAssembler | None = None,
        abstention: AbstentionPolicy | None = None,
        lanes: Sequence[Any] | None = None,
        clock: Clock | None = None,
        cfg: Any = None,
        enable_llm_planner: bool | None = None,
        lane_timeout_s: float | None = None,
        *,
        uows: UnitOfWorkFactory | None = None,
        rules_planner: RulesPlanner | None = None,
    ) -> None:
        resolved_uows = uow_factory or uows
        if resolved_uows is None:
            raise InvariantViolation('uow_factory is required for RetrievalOrchestrator')
        self.uow_factory = resolved_uows
        self.embedder = embedder
        self.rules_planner = planner or rules_planner or RulesPlanner()
        self.llm_gateway = llm_gateway
        self.fuser = fuser or RrfFuser()
        self.decay = decay or DecayScorer(cfg=cfg)
        self.reranker = reranker or NoopReranker()
        self.assembler = assembler or ContextAssembler(cfg=cfg)
        self.abstention = abstention or AbstentionPolicy(cfg=cfg)
        self.clock = clock or SystemClock()
        self.cfg = cfg

        if enable_llm_planner is not None:
            self.enable_llm_planner = enable_llm_planner
        elif cfg is not None and hasattr(cfg, 'enable_llm_planner'):
            self.enable_llm_planner = bool(cfg.enable_llm_planner)
        else:
            self.enable_llm_planner = False

        if lane_timeout_s is not None:
            self.lane_timeout_s = lane_timeout_s
        elif cfg is not None and hasattr(cfg, 'lane_timeout_ms'):
            self.lane_timeout_s = float(cfg.lane_timeout_ms) / 1000.0
        else:
            self.lane_timeout_s = 1.2

        if lanes is not None:
            self.lanes = list(lanes)
        else:
            self.lanes = [
                VectorLane(),
                FtsLane(),
                GraphLane(),
                TemporalLane(),
            ]

    async def retrieve(
        self,
        user_id: UserId,
        query: str | QueryPlan,
        question_ts: datetime | None = None,
        tz: str | ZoneInfo | None = None,
        known_entities: Sequence[Entity] = (),
        recent: Sequence[Message] = (),
        pending: Sequence[Message] = (),
        statement_only: bool = False,
    ) -> RetrievalResult:
        """Execute full retrieval pipeline per LLD Section 9.1-9.6."""
        # 1. Query planning
        if isinstance(query, QueryPlan):
            plan = query
        else:
            now = question_ts or self.clock.now()
            tz_str = str(tz) if tz is not None else 'UTC'
            aliases = {e.canonical_name: e.id for e in known_entities}
            plan = self.rules_planner.plan(
                question=query,
                question_ts=now,
                tz=tz_str,
                user_id=user_id,
                aliases=aliases,
            )

        # Check LLM planner fallback if triggers fire and LLM planner is enabled
        if needs_llm_planner(plan) and self.enable_llm_planner and self.llm_gateway is not None:
            try:
                plan = await self._run_llm_planner(plan)
            except Exception as exc:
                _logger.warning('LLM planner fallback failed: %s; falling back to rules plan', exc)
                plan = QueryPlan(
                    question=plan.question,
                    question_ts=plan.question_ts,
                    tz=plan.tz,
                    intent=plan.intent,
                    as_of=plan.as_of,
                    time_range=plan.time_range,
                    entity_ids=plan.entity_ids,
                    entity_names=plan.entity_names,
                    predicates=plan.predicates,
                    multi_hop=plan.multi_hop,
                    expects_count=plan.expects_count,
                    ambiguous_time=plan.ambiguous_time,
                    planner_path='fallback',
                    confidence=plan.confidence,
                )

        # 2. Query embedding outside DB transaction
        embedding: Sequence[float] | None = None
        degraded_lanes: list[str] = []

        has_vector_lane = any(
            isinstance(lane, VectorLane)
            or getattr(lane, 'name', '') in ('vector', 'vector_facts', 'source_vector')
            for lane in self.lanes
        )

        if has_vector_lane:
            if self.embedder is not None:
                try:
                    if hasattr(self.embedder, 'embed_query'):
                        embedding = await self.embedder.embed_query(plan.question)
                    else:
                        vecs = await self.embedder.embed_queries([plan.question])
                        embedding = vecs[0]
                except Exception as exc:
                    _logger.warning('Query embedder failed: %s; skipping vector lanes', exc)
                    embedding = None
                    degraded_lanes.append('no_vector')
            else:
                degraded_lanes.append('no_vector')

        # 3. Filter preparation
        flt = TemporalFilter.from_plan(plan)

        # 4. Identify applicable lanes
        applicable_lanes: list[Any] = []
        for lane in self.lanes:
            is_vector = isinstance(lane, VectorLane) or getattr(lane, 'name', '') in (
                'vector',
                'vector_facts',
                'source_vector',
            )
            if is_vector and embedding is None:
                continue
            if hasattr(lane, 'applies') and not lane.applies(plan, self.cfg):
                continue
            applicable_lanes.append(lane)

        if not applicable_lanes and self.lanes:
            raise AllLanesFailed(
                f'All retrieval lanes failed: no applicable lanes ({degraded_lanes})'
            )

        # 5. Parallel lane execution with per-lane read-only UoW sessions (R-9)
        async def _execute_lane(lane: Any, current_flt: TemporalFilter) -> list[Hit]:
            async with self.uow_factory.for_user(user_id) as uow:
                if isinstance(lane, VectorLane):
                    return await lane.execute(
                        uow=uow,
                        user_id=user_id,
                        plan=plan,
                        flt=current_flt,
                        embedding=embedding,
                    )
                if hasattr(lane, 'execute'):
                    res_exec = await lane.execute(
                        uow=uow,
                        user_id=user_id,
                        plan=plan,
                        flt=current_flt,
                    )
                    return list(res_exec)
                if hasattr(lane, 'search'):

                    class _Ctx:
                        user_id: UserId
                        plan: QueryPlan
                        flt: TemporalFilter
                        limit: int
                        uows: UnitOfWorkFactory
                        query_vector: Any

                    ctx = _Ctx()
                    ctx.user_id = user_id
                    ctx.plan = plan
                    ctx.flt = current_flt
                    ctx.limit = 40
                    ctx.uows = self.uow_factory
                    ctx.query_vector = lambda: embedding
                    res_search = await lane.search(ctx)
                    return list(res_search)
                raise InvariantViolation(f'Lane {lane} does not implement execute or search')

        async def _execute_lane_with_timeout(lane: Any, current_flt: TemporalFilter) -> list[Hit]:
            return await asyncio.wait_for(
                _execute_lane(lane, current_flt),
                timeout=self.lane_timeout_s,
            )

        tasks = [_execute_lane_with_timeout(lane, flt) for lane in applicable_lanes]
        gather_results = await asyncio.gather(*tasks, return_exceptions=True)

        all_failed = True
        lane_hits_by_lane: dict[str, list[Hit]] = defaultdict(list)
        lane_counts: dict[str, int] = {
            'vector_facts': 0,
            'fts_facts': 0,
            'graph_expand': 0,
            'temporal_facts': 0,
            'vector_sources': 0,
        }

        for lane, res in zip(applicable_lanes, gather_results, strict=False):
            lane_name = getattr(lane, 'name', lane.__class__.__name__)
            if isinstance(res, BaseException):
                _logger.warning('Retrieval lane %s failed: %s', lane_name, res)
                degraded_lanes.append(lane_name)
            else:
                all_failed = False
                hits: list[Hit] = res
                for hit in hits:
                    lane_hits_by_lane[hit.lane].append(hit)
                    lane_counts[hit.lane] = lane_counts.get(hit.lane, 0) + 1

        if applicable_lanes and all_failed:
            msg = f'All {len(applicable_lanes)} retrieval lanes failed: {degraded_lanes}'
            raise AllLanesFailed(msg)

        # 6. Temporal filter fallback: if AS_OF or OVERLAP yields 0 fact hits,
        # retry with ANY_HISTORY
        has_fact_hits = any(
            hit.kind == 'fact' for hits in lane_hits_by_lane.values() for hit in hits
        )
        is_scoped_mode = flt.mode in (FilterMode.AS_OF, FilterMode.OVERLAP)
        if is_scoped_mode and not has_fact_hits and not all_failed:
            fallback_flt = TemporalFilter(mode=FilterMode.ANY_HISTORY)
            fact_lanes = [
                lane
                for lane in applicable_lanes
                if not (isinstance(lane, VectorLane) and embedding is None)
            ]
            if fact_lanes:
                fallback_tasks = [
                    _execute_lane_with_timeout(lane, fallback_flt) for lane in fact_lanes
                ]
                fallback_results = await asyncio.gather(*fallback_tasks, return_exceptions=True)
                any_new_fact = False
                for _lane, res in zip(fact_lanes, fallback_results, strict=False):
                    if not isinstance(res, BaseException):
                        for hit in res:
                            if hit.kind == 'fact':
                                lane_hits_by_lane[hit.lane].append(hit)
                                lane_counts[hit.lane] = lane_counts.get(hit.lane, 0) + 1
                                any_new_fact = True
                if any_new_fact:
                    degraded_lanes.append('temporal_fallback')

        # 7. Reciprocal Rank Fusion
        ranked_items = self.fuser.fuse(lane_hits_by_lane)

        # 8. Multiplicative Decay
        decayed_items = self.decay.apply(plan, ranked_items)

        # 9. Reranking
        reranked_items = await self.reranker.rerank(plan.question, decayed_items)

        # 10. Context Assembly
        async with self.uow_factory.for_user(user_id) as uow:
            context = await self.assembler.build(
                uow=uow,
                plan=plan,
                items=reranked_items,
                recent=recent,
                pending=pending,
                user_id=user_id,
            )

        # 11. Abstention Decision
        preliminary_result = RetrievalResult(
            plan=plan,
            ranked_items=reranked_items,
            context=context,
            abstention=AbstentionDecision.proceed(),
            lane_counts=dict(lane_counts),
            degraded_lanes=list(degraded_lanes),
        )

        decision = self.abstention.evaluate(
            plan=plan,
            result=preliminary_result,
            ctx=context,
            statement_only=statement_only,
        )

        return RetrievalResult(
            plan=plan,
            ranked_items=reranked_items,
            context=context,
            abstention=decision,
            lane_counts=dict(lane_counts),
            degraded_lanes=list(degraded_lanes),
        )

    async def _run_llm_planner(self, plan: QueryPlan) -> QueryPlan:
        """Execute LLM query planner fallback per LLD Section 9.2."""
        assert self.llm_gateway is not None
        user_prompt = (
            f'QUESTION: {plan.question}\n\nPlan the memory retrieval query and return JSON.'
        )
        req = LLMRequest(
            purpose='plan',
            tier='fast',
            messages=[
                ChatMessage(role='system', content=PLANNER_SYSTEM_PROMPT),
                ChatMessage(
                    role='user',
                    content=user_prompt,
                ),
            ],
            prompt_version='plan.v1',
            temperature=0.0,
            max_tokens=512,
            json_mode=True,
            priority='interactive',
            timeout_s=1.5,
        )
        output = await self.llm_gateway.complete_json(req, PlannerLLMOutput)

        # Re-resolve time phrases with TimeResolver (never trust model-computed dates)
        time_resolver = TimeResolver()
        resolved_as_of: datetime | None = None
        resolved_range: TimeInterval | None = None
        try:
            zone = ZoneInfo(plan.tz)
        except Exception:
            zone = ZoneInfo('UTC')

        for phrase in output.time_phrases:
            r = time_resolver.resolve_range(phrase, plan.question_ts, zone)
            if r is not None:
                resolved_range = r
                if output.intent == Intent.POINT_IN_TIME:
                    resolved_as_of = r.start

        return QueryPlan(
            question=plan.question,
            question_ts=plan.question_ts,
            tz=plan.tz,
            intent=output.intent,
            as_of=resolved_as_of if output.intent == Intent.POINT_IN_TIME else plan.as_of,
            time_range=resolved_range
            if output.intent in (Intent.INTERVAL, Intent.AGGREGATION, Intent.DURATION_OR_ORDER)
            else plan.time_range,
            entity_ids=plan.entity_ids,
            entity_names=tuple(output.entity_names) if output.entity_names else plan.entity_names,
            predicates=tuple(output.predicates) if output.predicates else plan.predicates,
            multi_hop=output.multi_hop,
            expects_count=output.intent == Intent.AGGREGATION or plan.expects_count,
            ambiguous_time=plan.ambiguous_time,
            planner_path='llm',
            confidence=0.9,
        )
