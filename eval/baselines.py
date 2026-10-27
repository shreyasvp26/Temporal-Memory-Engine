"""LongMemEval and ablation baselines per LLD Section 13.2–13.3 [FROZEN].

Provides:
- Evaluation domain entities: EvalTurn, EvalSession, EvalCase, SystemAnswer, JudgeResult.
- Protocols: MemorySystem, Reader.
- Reader implementations: SimpleReader, LLMReader.
- Ablation flags and configuration: AblationFlags, BaselineConfig.
- Baseline conflict registry: build_baseline_conflict_registry (SingleAsMultiStrategy).
- Baseline orchestrator builder: build_baseline_orchestrator, BaselineRetrievalOrchestrator.
- Comparative baselines:
  1. FullContextBaseline
  2. VectorOnlyBaseline (only vector lane, no FTS/graph, ANY_HISTORY)
  3. Bm25OnlyBaseline (only FTS lane)
  4. HybridNoTemporalBaseline (vector + FTS + RRF, no supersession, ANY_HISTORY)
  5. BitemporalNoDecayBaseline (full bitemporal engine, decay disabled)
  6. FullBitemporalBaseline / TemporalMemoryBaseline (5 lanes, S1-S10, filter, decay)
"""

import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any, ClassVar, Protocol
from uuid import uuid5
from zoneinfo import ZoneInfo

from memory_engine.domain.enums import Cardinality, Intent
from memory_engine.domain.ids import UserId
from memory_engine.domain.models import Candidate, Entity, Fact, JudgeHint, Message
from memory_engine.ingest.conflict.event import EventStrategy
from memory_engine.ingest.conflict.multi import MultiValuedStrategy
from memory_engine.ingest.conflict.plan import (
    ResolutionContext,
    ResolutionPlan,
    ResolutionStrategy,
)
from memory_engine.ingest.conflict.registry import StrategyRegistry
from memory_engine.ingest.conflict.single import SingleValuedStrategy
from memory_engine.ports.clock import Clock
from memory_engine.ports.embed import Embedder
from memory_engine.ports.llm import ChatMessage, LLMGateway, LLMRequest
from memory_engine.ports.repositories import FilterMode
from memory_engine.ports.uow import UnitOfWorkFactory
from memory_engine.retrieve.abstain import AbstentionPolicy
from memory_engine.retrieve.assembler import ContextAssembler
from memory_engine.retrieve.decay import DecayScorer
from memory_engine.retrieve.lanes.fts import FtsLane
from memory_engine.retrieve.lanes.graph import GraphLane
from memory_engine.retrieve.lanes.temporal import TemporalLane
from memory_engine.retrieve.lanes.vector import VectorLane
from memory_engine.retrieve.orchestrator import RetrievalOrchestrator, RetrievalResult
from memory_engine.retrieve.planner.rules import QueryPlan
from memory_engine.retrieve.rerank import NoopReranker

__all__ = [
    'BASELINE_CONFIGS',
    'READER_V1_PROMPT',
    'AblationFlags',
    'BaselineConfig',
    'BaselineRetrievalOrchestrator',
    'BitemporalNoDecayBaseline',
    'Bm25OnlyBaseline',
    'EvalCase',
    'EvalSession',
    'EvalTurn',
    'FullBitemporalBaseline',
    'FullContextBaseline',
    'HybridNoTemporalBaseline',
    'JudgeResult',
    'LLMReader',
    'MemorySystem',
    'Reader',
    'RetrievalBaseline',
    'SimpleReader',
    'SingleAsMultiStrategy',
    'SystemAnswer',
    'TemporalMemoryBaseline',
    'VectorOnlyBaseline',
    'build_baseline_conflict_registry',
    'build_baseline_orchestrator',
    'create_baseline',
    'get_baseline_config',
]

EVAL_NS = uuid.UUID('7b1e4c82-963a-4a25-8321-4d2a84d28471')

READER_V1_PROMPT = """You are a helpful assistant answering strictly from the provided context.
Answer the question using only the facts in the context.
If the information is not in the context, say that the information is not available.
For questions about dates or times, explicitly state the relevant dates.
Be concise and truthful."""


@dataclass(frozen=True, slots=True)
class EvalTurn:
    """Single turn within an evaluation session."""

    role: str
    content: str
    has_answer: bool = False


@dataclass(frozen=True, slots=True)
class EvalSession:
    """A dated session containing turns."""

    session_id: str
    ts: datetime
    turns: tuple[EvalTurn, ...]

    @property
    def timestamp(self) -> datetime:
        """Alias for session timestamp."""
        return self.ts


@dataclass(frozen=True, slots=True)
class EvalCase:
    """Benchmark evaluation case per LLD Section 13.2."""

    case_id: str
    qtype: str
    question: str
    gold: str
    question_ts: datetime
    sessions: tuple[EvalSession, ...]
    evidence_session_ids: frozenset[str]
    is_abstention: bool


@dataclass(frozen=True, slots=True)
class SystemAnswer:
    """Standardized answer returned by any MemorySystem."""

    text: str
    retrieved_session_ids: tuple[str, ...]
    context_tokens: int
    reader_tokens: int
    latency_ms: float
    extras: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class JudgeResult:
    """Judge evaluation outcome."""

    correct: bool
    raw: str
    judge_model: str


class MemorySystem(Protocol):
    """Protocol for comparative memory systems per LLD Section 13.2."""

    name: str

    async def prepare(self, case: EvalCase) -> None:
        """Ingest or prepare history for an evaluation case."""
        ...

    async def answer(self, case: EvalCase) -> SystemAnswer:
        """Answer the question in the evaluation case."""
        ...

    async def cleanup(self, case: EvalCase) -> None:
        """Clean up tenant data for the evaluation case."""
        ...


class Reader(Protocol):
    """Protocol for answering questions given retrieved context per LLD Section 13.3."""

    async def answer(
        self,
        question: str,
        context: str,
        question_ts: datetime | None = None,
    ) -> tuple[str, int, int]:
        """Answer question from context. Returns (answer_text, context_tokens, reader_tokens)."""
        ...


class SimpleReader:
    """Deterministic, socket-free reader for unit tests and offline testing."""

    def __init__(self, default_response: str | None = None) -> None:
        self.default_response = default_response

    async def answer(
        self,
        question: str,
        context: str,
        question_ts: datetime | None = None,
    ) -> tuple[str, int, int]:
        if self.default_response is not None:
            resp = self.default_response
        elif not context.strip():
            resp = 'The information is not available.'
        else:
            resp = f'Answer based on context: {context[:60]}'
        ctx_tokens = max(1, len(context.split()))
        rdr_tokens = max(1, len(resp.split()))
        return (resp, ctx_tokens, rdr_tokens)


class LLMReader:
    """LLM-backed reader using LLMGateway and READER_V1_PROMPT per LLD Section 13.3."""

    def __init__(
        self,
        llm_gateway: LLMGateway,
        system_prompt: str = READER_V1_PROMPT,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> None:
        self.llm_gateway = llm_gateway
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.max_tokens = max_tokens

    async def answer(
        self,
        question: str,
        context: str,
        question_ts: datetime | None = None,
    ) -> tuple[str, int, int]:
        ts_hint = f'\nCurrent Reference Time: {question_ts.isoformat()}' if question_ts else ''
        user_prompt = f'Context:\n{context}\n{ts_hint}\n\nQuestion: {question}'
        req = LLMRequest(
            purpose='synth',
            tier='synth',
            prompt_version='reader.v1',
            messages=[
                ChatMessage(role='system', content=self.system_prompt),
                ChatMessage(role='user', content=user_prompt),
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        resp = await self.llm_gateway.complete_text(req)
        ctx_tokens = resp.prompt_tokens or len(user_prompt.split())
        rdr_tokens = resp.completion_tokens or len(resp.text.split())
        return (resp.text, ctx_tokens, rdr_tokens)


@dataclass(frozen=True, slots=True)
class AblationFlags:
    """Ablation toggles per LLD Section 13.3 [FROZEN].

    All default True except rerank which defaults to False.
    """

    supersession: bool = True
    temporal_filter: bool = True
    decay: bool = True
    graph: bool = True
    source_lane: bool = True
    evidence: bool = True
    llm_planner: bool = True
    rerank: bool = False


@dataclass(frozen=True, slots=True)
class BaselineConfig:
    """Configuration descriptor for baseline or ablation system."""

    name: str
    description: str
    ablation_flags: AblationFlags
    enabled_lanes: tuple[str, ...]
    filter_mode_override: FilterMode | None = None
    max_context_tokens: int = 3000


class SingleAsMultiStrategy(ResolutionStrategy):
    """Resolution strategy mapping SINGLE cardinality to MULTI behavior for ablations."""

    cardinality: ClassVar[Cardinality] = Cardinality.SINGLE

    def __init__(self, delegate: MultiValuedStrategy | None = None) -> None:
        self._delegate = delegate or MultiValuedStrategy()

    def plan(
        self,
        c: Candidate,
        timeline: Sequence[Fact],
        ctx: ResolutionContext,
        hint: JudgeHint | None = None,
    ) -> ResolutionPlan:
        return self._delegate.plan(c, timeline, ctx, hint=hint)


def build_baseline_conflict_registry(flags: AblationFlags) -> StrategyRegistry:
    """Build a StrategyRegistry adhering to AblationFlags."""
    registry = StrategyRegistry()
    multi = MultiValuedStrategy()
    event = EventStrategy()
    if not flags.supersession:
        registry.register(SingleAsMultiStrategy(multi))
    else:
        registry.register(SingleValuedStrategy())
    registry.register(multi)
    registry.register(event)
    return registry


class BaselineRetrievalOrchestrator(RetrievalOrchestrator):
    """RetrievalOrchestrator configured for ablation baselines with optional filter override."""

    def __init__(
        self,
        *args: Any,
        filter_mode_override: FilterMode | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.filter_mode_override = filter_mode_override

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
        if self.filter_mode_override is not None:
            if isinstance(query, QueryPlan):
                base_plan = query
            else:
                now = question_ts or self.clock.now()
                tz_str = str(tz) if tz is not None else 'UTC'
                aliases = {e.canonical_name: e.id for e in known_entities}
                base_plan = self.rules_planner.plan(
                    question=query,
                    question_ts=now,
                    tz=tz_str,
                    user_id=user_id,
                    aliases=aliases,
                )

            forced_plan = QueryPlan(
                question=base_plan.question,
                question_ts=base_plan.question_ts,
                tz=base_plan.tz,
                intent=(
                    Intent.HISTORICAL
                    if self.filter_mode_override == FilterMode.ANY_HISTORY
                    else base_plan.intent
                ),
                as_of=(
                    None if self.filter_mode_override == FilterMode.ANY_HISTORY else base_plan.as_of
                ),
                time_range=(
                    None
                    if self.filter_mode_override == FilterMode.ANY_HISTORY
                    else base_plan.time_range
                ),
                entity_ids=base_plan.entity_ids,
                entity_names=base_plan.entity_names,
                predicates=base_plan.predicates,
                multi_hop=base_plan.multi_hop,
                expects_count=base_plan.expects_count,
                ambiguous_time=base_plan.ambiguous_time,
                planner_path=base_plan.planner_path,
                confidence=base_plan.confidence,
            )
            return await super().retrieve(
                user_id=user_id,
                query=forced_plan,
                question_ts=question_ts,
                tz=tz,
                known_entities=known_entities,
                recent=recent,
                pending=pending,
                statement_only=statement_only,
            )

        return await super().retrieve(
            user_id=user_id,
            query=query,
            question_ts=question_ts,
            tz=tz,
            known_entities=known_entities,
            recent=recent,
            pending=pending,
            statement_only=statement_only,
        )


def build_baseline_orchestrator(
    config: BaselineConfig,
    uow_factory: UnitOfWorkFactory,
    embedder: Embedder | None = None,
    clock: Clock | None = None,
    llm_gateway: LLMGateway | None = None,
    cfg: Any = None,
) -> BaselineRetrievalOrchestrator:
    """Build a RetrievalOrchestrator instance wired for a specific BaselineConfig."""
    flags = config.ablation_flags
    lanes: list[Any] = []

    if 'vector_facts' in config.enabled_lanes or 'vector_sources' in config.enabled_lanes:
        inc_sources = flags.source_lane and ('vector_sources' in config.enabled_lanes)
        lanes.append(VectorLane(include_sources=inc_sources))

    if 'fts_facts' in config.enabled_lanes:
        lanes.append(FtsLane())

    if 'graph_expand' in config.enabled_lanes and flags.graph:
        lanes.append(GraphLane())

    if 'temporal_facts' in config.enabled_lanes:
        lanes.append(TemporalLane())

    decay = DecayScorer(cfg=cfg, decay_enabled=flags.decay)
    reranker = NoopReranker()
    assembler = ContextAssembler(cfg=cfg, fact_tokens=config.max_context_tokens)
    abstention = AbstentionPolicy(cfg=cfg)

    return BaselineRetrievalOrchestrator(
        uow_factory=uow_factory,
        embedder=embedder,
        llm_gateway=llm_gateway,
        decay=decay,
        reranker=reranker,
        assembler=assembler,
        abstention=abstention,
        lanes=lanes,
        clock=clock,
        cfg=cfg,
        enable_llm_planner=flags.llm_planner,
        filter_mode_override=config.filter_mode_override,
    )


BASELINE_CONFIGS: dict[str, BaselineConfig] = {
    'vector_only': BaselineConfig(
        name='vector_only',
        description='Vector lane only with ANY_HISTORY temporal filter and no decay/graph',
        ablation_flags=AblationFlags(
            supersession=True,
            temporal_filter=False,
            decay=False,
            graph=False,
            source_lane=False,
            evidence=False,
            llm_planner=False,
            rerank=False,
        ),
        enabled_lanes=('vector_facts',),
        filter_mode_override=FilterMode.ANY_HISTORY,
    ),
    'bm25_only': BaselineConfig(
        name='bm25_only',
        description='BM25 / Full-Text Search lane only with ANY_HISTORY filter and no decay/graph',
        ablation_flags=AblationFlags(
            supersession=True,
            temporal_filter=False,
            decay=False,
            graph=False,
            source_lane=False,
            evidence=False,
            llm_planner=False,
            rerank=False,
        ),
        enabled_lanes=('fts_facts',),
        filter_mode_override=FilterMode.ANY_HISTORY,
    ),
    'hybrid_no_bitemporal': BaselineConfig(
        name='hybrid_no_bitemporal',
        description='Hybrid Vector + BM25 + RRF without conflict supersession or temporal filter',
        ablation_flags=AblationFlags(
            supersession=False,
            temporal_filter=False,
            decay=False,
            graph=False,
            source_lane=True,
            evidence=True,
            llm_planner=False,
            rerank=False,
        ),
        enabled_lanes=('vector_facts', 'fts_facts', 'vector_sources'),
        filter_mode_override=FilterMode.ANY_HISTORY,
    ),
    'bitemporal_no_decay': BaselineConfig(
        name='bitemporal_no_decay',
        description='Full bitemporal engine with all 5 lanes and conflict resolution, decay off',
        ablation_flags=AblationFlags(
            supersession=True,
            temporal_filter=True,
            decay=False,
            graph=True,
            source_lane=True,
            evidence=True,
            llm_planner=True,
            rerank=False,
        ),
        enabled_lanes=(
            'vector_facts',
            'fts_facts',
            'graph_expand',
            'temporal_facts',
            'vector_sources',
        ),
        filter_mode_override=None,
    ),
    'full_bitemporal': BaselineConfig(
        name='full_bitemporal',
        description='Full bitemporal system with all 5 lanes, S1-S10 resolution, filter, and decay',
        ablation_flags=AblationFlags(
            supersession=True,
            temporal_filter=True,
            decay=True,
            graph=True,
            source_lane=True,
            evidence=True,
            llm_planner=True,
            rerank=False,
        ),
        enabled_lanes=(
            'vector_facts',
            'fts_facts',
            'graph_expand',
            'temporal_facts',
            'vector_sources',
        ),
        filter_mode_override=None,
    ),
    'full_context': BaselineConfig(
        name='full_context',
        description='Full chronological history reference baseline',
        ablation_flags=AblationFlags(),
        enabled_lanes=(),
        filter_mode_override=None,
    ),
}

# Standard aliases per LLD Section 13.3
BASELINE_CONFIGS['tmem'] = BASELINE_CONFIGS['full_bitemporal']
BASELINE_CONFIGS['hybrid'] = BASELINE_CONFIGS['hybrid_no_bitemporal']
BASELINE_CONFIGS['bm25'] = BASELINE_CONFIGS['bm25_only']
BASELINE_CONFIGS['naive_turn'] = BASELINE_CONFIGS['vector_only']
BASELINE_CONFIGS['naive_session'] = BASELINE_CONFIGS['vector_only']

# Specific ablation configs
BASELINE_CONFIGS['tmem_no_supersession'] = BaselineConfig(
    name='tmem_no_supersession',
    description='Full bitemporal system without single-valued predicate supersession',
    ablation_flags=AblationFlags(supersession=False),
    enabled_lanes=(
        'vector_facts',
        'fts_facts',
        'graph_expand',
        'temporal_facts',
        'vector_sources',
    ),
)
BASELINE_CONFIGS['tmem_no_temporal_filter'] = BaselineConfig(
    name='tmem_no_temporal_filter',
    description='Full bitemporal system forcing ANY_HISTORY retrieval filter',
    ablation_flags=AblationFlags(temporal_filter=False),
    enabled_lanes=(
        'vector_facts',
        'fts_facts',
        'graph_expand',
        'temporal_facts',
        'vector_sources',
    ),
    filter_mode_override=FilterMode.ANY_HISTORY,
)
BASELINE_CONFIGS['tmem_no_decay'] = BASELINE_CONFIGS['bitemporal_no_decay']
BASELINE_CONFIGS['tmem_no_graph'] = BaselineConfig(
    name='tmem_no_graph',
    description='Full bitemporal system without GraphExpansionLane',
    ablation_flags=AblationFlags(graph=False),
    enabled_lanes=(
        'vector_facts',
        'fts_facts',
        'temporal_facts',
        'vector_sources',
    ),
)
BASELINE_CONFIGS['tmem_no_source_lane'] = BaselineConfig(
    name='tmem_no_source_lane',
    description='Full bitemporal system without raw source vector/FTS lanes',
    ablation_flags=AblationFlags(source_lane=False),
    enabled_lanes=(
        'vector_facts',
        'fts_facts',
        'graph_expand',
        'temporal_facts',
    ),
)
BASELINE_CONFIGS['tmem_no_evidence'] = BaselineConfig(
    name='tmem_no_evidence',
    description='Full bitemporal system without evidence_for attachment',
    ablation_flags=AblationFlags(evidence=False),
    enabled_lanes=(
        'vector_facts',
        'fts_facts',
        'graph_expand',
        'temporal_facts',
        'vector_sources',
    ),
)
BASELINE_CONFIGS['tmem_rules_only'] = BaselineConfig(
    name='tmem_rules_only',
    description='Full bitemporal system using RulesPlanner only (no LLM planner fallback)',
    ablation_flags=AblationFlags(llm_planner=False),
    enabled_lanes=(
        'vector_facts',
        'fts_facts',
        'graph_expand',
        'temporal_facts',
        'vector_sources',
    ),
)


def get_baseline_config(name: str) -> BaselineConfig:
    """Retrieve BaselineConfig by name or alias."""
    norm = name.strip().lower()
    if norm not in BASELINE_CONFIGS:
        raise KeyError(f'Unknown baseline {name!r}. Available: {sorted(BASELINE_CONFIGS.keys())}')
    return BASELINE_CONFIGS[norm]


class FullContextBaseline:
    """Full-context baseline providing entire chronological history to reader."""

    def __init__(
        self,
        config: BaselineConfig | None = None,
        reader: Reader | None = None,
        max_context_tokens: int = 3000,
    ) -> None:
        self.config = config or get_baseline_config('full_context')
        self.name = self.config.name
        self.reader = reader or SimpleReader()
        self.max_context_tokens = max_context_tokens
        self._cases: dict[str, EvalCase] = {}

    def has_case(self, case_id: str) -> bool:
        """Check if case is present in memory."""
        return case_id in self._cases

    async def prepare(self, case: EvalCase) -> None:
        """Store case history for context construction."""
        self._cases[case.case_id] = case

    async def answer(self, case: EvalCase) -> SystemAnswer:
        """Construct full chronological history context and invoke reader."""
        t0 = time.perf_counter()
        sessions = sorted(case.sessions, key=lambda s: s.ts)
        lines: list[str] = []
        retrieved_session_ids: list[str] = []

        for session in sessions:
            retrieved_session_ids.append(session.session_id)
            dt_str = session.ts.strftime('%Y-%m-%d %H:%M:%S')
            lines.append(f'=== Session: {session.session_id} ({dt_str}) ===')
            for turn in session.turns:
                lines.append(f'{turn.role.capitalize()}: {turn.content}')

        raw_context = '\n'.join(lines)
        words = raw_context.split()
        if len(words) > self.max_context_tokens:
            truncated_words = words[-self.max_context_tokens :]
            context = '... [truncated older history] ...\n' + ' '.join(truncated_words)
        else:
            context = raw_context

        answer_text, ctx_tokens, rdr_tokens = await self.reader.answer(
            question=case.question,
            context=context,
            question_ts=case.question_ts,
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0

        return SystemAnswer(
            text=answer_text,
            retrieved_session_ids=tuple(retrieved_session_ids),
            context_tokens=ctx_tokens,
            reader_tokens=rdr_tokens,
            latency_ms=latency_ms,
            extras={'raw_context_words': len(words), 'system': self.name},
        )

    async def cleanup(self, case: EvalCase) -> None:
        """Clean up stored case state."""
        self._cases.pop(case.case_id, None)


class RetrievalBaseline:
    """Base class for retrieval-driven memory system baselines."""

    def __init__(
        self,
        config: BaselineConfig,
        orchestrator: RetrievalOrchestrator | None = None,
        reader: Reader | None = None,
        uow_factory: UnitOfWorkFactory | None = None,
        embedder: Embedder | None = None,
    ) -> None:
        self.config = config
        self.name = config.name
        self.reader = reader or SimpleReader()
        self.uow_factory = uow_factory
        self.embedder = embedder

        if orchestrator is not None:
            self.orchestrator: RetrievalOrchestrator | None = orchestrator
        elif uow_factory is not None:
            self.orchestrator = build_baseline_orchestrator(
                config=config,
                uow_factory=uow_factory,
                embedder=embedder,
            )
        else:
            self.orchestrator = None

    async def prepare(self, case: EvalCase) -> None:
        """Prepare tenant data (implemented by runner/e2e integration)."""
        return None

    async def answer(self, case: EvalCase) -> SystemAnswer:
        """Retrieve context via orchestrator and generate answer via reader."""
        t0 = time.perf_counter()
        user_id = UserId(uuid5(EVAL_NS, f'{self.name}:{case.case_id}'))
        retrieved_ids: list[str] = []
        context_text = ''
        extras: dict[str, Any] = {'config': self.config.name}

        if self.orchestrator is not None:
            retrieval_res = await self.orchestrator.retrieve(
                user_id=user_id,
                query=case.question,
                question_ts=case.question_ts,
            )
            context_text = retrieval_res.context.text
            extras['lane_counts'] = retrieval_res.lane_counts
            extras['degraded_lanes'] = retrieval_res.degraded_lanes
            extras['abstention'] = retrieval_res.abstention.should_abstain

            for fid in retrieval_res.context.used_fact_ids:
                retrieved_ids.append(str(fid))

        answer_text, ctx_tokens, rdr_tokens = await self.reader.answer(
            question=case.question,
            context=context_text,
            question_ts=case.question_ts,
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0

        return SystemAnswer(
            text=answer_text,
            retrieved_session_ids=tuple(retrieved_ids),
            context_tokens=ctx_tokens,
            reader_tokens=rdr_tokens,
            latency_ms=latency_ms,
            extras=extras,
        )

    async def cleanup(self, case: EvalCase) -> None:
        """Clean up tenant state for case."""
        return None


class VectorOnlyBaseline(RetrievalBaseline):
    """Vector lane only baseline with ANY_HISTORY filter and no decay/graph."""

    def __init__(
        self,
        config: BaselineConfig | None = None,
        orchestrator: RetrievalOrchestrator | None = None,
        reader: Reader | None = None,
        uow_factory: UnitOfWorkFactory | None = None,
        embedder: Embedder | None = None,
    ) -> None:
        cfg = config or get_baseline_config('vector_only')
        super().__init__(
            cfg,
            orchestrator=orchestrator,
            reader=reader,
            uow_factory=uow_factory,
            embedder=embedder,
        )


class Bm25OnlyBaseline(RetrievalBaseline):
    """BM25 / Full-Text Search lane only baseline with ANY_HISTORY filter."""

    def __init__(
        self,
        config: BaselineConfig | None = None,
        orchestrator: RetrievalOrchestrator | None = None,
        reader: Reader | None = None,
        uow_factory: UnitOfWorkFactory | None = None,
        embedder: Embedder | None = None,
    ) -> None:
        cfg = config or get_baseline_config('bm25_only')
        super().__init__(
            cfg,
            orchestrator=orchestrator,
            reader=reader,
            uow_factory=uow_factory,
            embedder=embedder,
        )


class HybridNoTemporalBaseline(RetrievalBaseline):
    """Hybrid (Vector + FTS + RRF) baseline without conflict supersession or temporal filtering."""

    def __init__(
        self,
        config: BaselineConfig | None = None,
        orchestrator: RetrievalOrchestrator | None = None,
        reader: Reader | None = None,
        uow_factory: UnitOfWorkFactory | None = None,
        embedder: Embedder | None = None,
    ) -> None:
        cfg = config or get_baseline_config('hybrid_no_bitemporal')
        super().__init__(
            cfg,
            orchestrator=orchestrator,
            reader=reader,
            uow_factory=uow_factory,
            embedder=embedder,
        )


class BitemporalNoDecayBaseline(RetrievalBaseline):
    """Full bitemporal engine with all 5 lanes and conflict resolution, decay disabled."""

    def __init__(
        self,
        config: BaselineConfig | None = None,
        orchestrator: RetrievalOrchestrator | None = None,
        reader: Reader | None = None,
        uow_factory: UnitOfWorkFactory | None = None,
        embedder: Embedder | None = None,
    ) -> None:
        cfg = config or get_baseline_config('bitemporal_no_decay')
        super().__init__(
            cfg,
            orchestrator=orchestrator,
            reader=reader,
            uow_factory=uow_factory,
            embedder=embedder,
        )


class FullBitemporalBaseline(RetrievalBaseline):
    """Full bitemporal engine with all 5 lanes, S1-S10 conflict resolution, filter, and decay."""

    def __init__(
        self,
        config: BaselineConfig | None = None,
        orchestrator: RetrievalOrchestrator | None = None,
        reader: Reader | None = None,
        uow_factory: UnitOfWorkFactory | None = None,
        embedder: Embedder | None = None,
    ) -> None:
        cfg = config or get_baseline_config('full_bitemporal')
        super().__init__(
            cfg,
            orchestrator=orchestrator,
            reader=reader,
            uow_factory=uow_factory,
            embedder=embedder,
        )


TemporalMemoryBaseline = FullBitemporalBaseline


def create_baseline(
    name: str,
    reader: Reader | None = None,
    uow_factory: UnitOfWorkFactory | None = None,
    embedder: Embedder | None = None,
    orchestrator: RetrievalOrchestrator | None = None,
) -> MemorySystem:
    """Instantiate a baseline MemorySystem by name or alias."""
    norm = name.strip().lower()
    if norm == 'full_context':
        return FullContextBaseline(reader=reader)
    if norm in ('vector_only', 'naive_turn', 'naive_session'):
        return VectorOnlyBaseline(
            reader=reader,
            uow_factory=uow_factory,
            embedder=embedder,
            orchestrator=orchestrator,
        )
    if norm in ('bm25_only', 'bm25'):
        return Bm25OnlyBaseline(
            reader=reader,
            uow_factory=uow_factory,
            embedder=embedder,
            orchestrator=orchestrator,
        )
    if norm in ('hybrid_no_bitemporal', 'hybrid'):
        return HybridNoTemporalBaseline(
            reader=reader,
            uow_factory=uow_factory,
            embedder=embedder,
            orchestrator=orchestrator,
        )
    if norm in ('bitemporal_no_decay', 'tmem_no_decay'):
        return BitemporalNoDecayBaseline(
            reader=reader,
            uow_factory=uow_factory,
            embedder=embedder,
            orchestrator=orchestrator,
        )
    if norm in ('full_bitemporal', 'tmem'):
        return FullBitemporalBaseline(
            reader=reader,
            uow_factory=uow_factory,
            embedder=embedder,
            orchestrator=orchestrator,
        )

    # Specific ablation name or custom baseline
    cfg = get_baseline_config(name)
    return RetrievalBaseline(
        cfg,
        orchestrator=orchestrator,
        reader=reader,
        uow_factory=uow_factory,
        embedder=embedder,
    )
