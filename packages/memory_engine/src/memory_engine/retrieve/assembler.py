"""Context assembler, token counter, and line formatting per LLD Section 9.5 [FROZEN]."""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from uuid import UUID
from zoneinfo import ZoneInfo

from memory_engine.domain.enums import Cardinality, FactStatus, Intent, TimePrecision
from memory_engine.domain.ids import FactId, SourceId, UserId
from memory_engine.domain.models import Fact, Message, Source
from memory_engine.ports.repositories import EvidenceSnippet
from memory_engine.ports.uow import UnitOfWork
from memory_engine.retrieve.fusion import RankedItem
from memory_engine.retrieve.planner import QueryPlan

__all__ = [
    'AssembledContext',
    'ContextAssembler',
    'TokenCounter',
    'TraceItem',
]


class TokenCounter:
    """Token estimation heuristic per LLD Section 9.5 [FROZEN].

    Default is ceil(len(text) / 3.6).
    """

    def count(self, text: str) -> int:
        if not text:
            return 0
        return math.ceil(len(text) / 3.6)


@dataclass(frozen=True, slots=True)
class TraceItem:
    """Trace entry for an item included in the assembled prompt context."""

    label: str
    kind: str
    id: UUID
    score: float | None = None
    detail: str = ''


@dataclass(frozen=True, slots=True)
class AssembledContext:
    """Final assembled context block with prompt text and traceability metadata [FROZEN]."""

    text: str
    fact_labels: dict[str, FactId]
    traces: tuple[TraceItem, ...]
    used_fact_ids: tuple[FactId, ...]
    source_labels: dict[str, SourceId] = field(default_factory=dict)


def _format_date(dt: datetime, precision: TimePrecision) -> str:
    """Format UTC datetime per precision: YYYY, YYYY-MM, or YYYY-MM-DD."""
    if precision == TimePrecision.YEAR:
        return dt.strftime('%Y')
    if precision == TimePrecision.MONTH:
        return dt.strftime('%Y-%m')
    return dt.strftime('%Y-%m-%d')


def _format_state(fact: Fact) -> str:
    """Format fact state: current, no longer true since D, past event, archived."""
    if fact.status == FactStatus.ARCHIVED:
        return 'archived'
    if fact.cardinality == Cardinality.EVENT:
        return 'past event'
    if fact.is_current:
        return 'current'
    end_dt = fact.valid_to or fact.invalidated_at
    if end_dt is not None:
        d = _format_date(end_dt, TimePrecision.DAY)
        return f'no longer true since {d}'
    return 'current'


def _format_validity(fact: Fact) -> str:
    """Format validity interval with precision."""
    start_str = _format_date(fact.valid_from, fact.time_precision)
    if fact.valid_to is None:
        return f'valid {start_str} to present'
    end_precision = (
        fact.time_precision if fact.cardinality == Cardinality.EVENT else TimePrecision.DAY
    )
    end_str = _format_date(fact.valid_to, end_precision)
    return f'valid {start_str} to {end_str}'


def _format_evidence(evidence_snippets: Sequence[EvidenceSnippet] | None) -> str:
    """Format evidence snippet part: | evidence (YYYY-MM-DD, role): 'snippet'."""
    if not evidence_snippets:
        return ''
    snip = evidence_snippets[0]
    cleaned = ' '.join(snip.snippet.split())
    date_str = snip.source_ts.strftime('%Y-%m-%d')
    role_str = snip.role.value
    return f" | evidence ({date_str}, {role_str}): '{cleaned}'"


def _format_fact_line(
    label: str,
    fact: Fact,
    evidence_snippets: Sequence[EvidenceSnippet] | None = None,
) -> str:
    """Format structured fact line per LLD Section 9.5 [FROZEN].

    Example:
      [F1] User lives in Delhi | valid 2023-06 to 2026-10-01 | no longer true since 2026-10-01
           | evidence (...)
    """
    validity = _format_validity(fact)
    state = _format_state(fact)
    evidence_part = _format_evidence(evidence_snippets)
    return f'[{label}] {fact.fact_text} | {validity} | {state}{evidence_part}'


def _format_source_line(label: str, source: Source) -> str:
    """Format raw source snippet line per LLD Section 9.5 [FROZEN].

    [S1] (2026-08-02, user) ...raw excerpt...
    """
    date_str = source.source_ts.strftime('%Y-%m-%d')
    role_str = source.role.value
    excerpt = source.content[:400]
    return f'[{label}] ({date_str}, {role_str}) {excerpt}'


class ContextAssembler:
    """Assembles prompt context blocks, enforcing budgets and line formats [FROZEN]."""

    def __init__(
        self,
        tokens: TokenCounter | None = None,
        cfg: Any = None,
        *,
        fact_tokens: int | None = None,
        source_tokens: int | None = None,
    ) -> None:
        self.tokens = tokens or TokenCounter()

        if fact_tokens is not None:
            self.fact_tokens = fact_tokens
        elif cfg is not None and hasattr(cfg, 'ctx_fact_tokens'):
            self.fact_tokens = int(cfg.ctx_fact_tokens)
        elif cfg is not None and hasattr(cfg, 'fact_tokens'):
            self.fact_tokens = int(cfg.fact_tokens)
        else:
            self.fact_tokens = 2200

        if source_tokens is not None:
            self.source_tokens = source_tokens
        elif cfg is not None and hasattr(cfg, 'ctx_source_tokens'):
            self.source_tokens = int(cfg.ctx_source_tokens)
        elif cfg is not None and hasattr(cfg, 'source_tokens'):
            self.source_tokens = int(cfg.source_tokens)
        else:
            self.source_tokens = 800

    def assemble_pure(
        self,
        plan: QueryPlan,
        items: Sequence[RankedItem],
        evidence: Mapping[FactId, Sequence[EvidenceSnippet]] | None = None,
        chain_neighbors: Sequence[Fact] | None = None,
        recent: Sequence[Message] = (),
        pending: Sequence[Message] = (),
    ) -> AssembledContext:
        """Pure formatting and selection logic for context assembly."""
        evidence_map = evidence or {}
        is_aggregation = plan.intent == Intent.AGGREGATION

        # Budget and count limits per Section 9.5
        fact_budget = 3500 if is_aggregation else self.fact_tokens
        max_facts = 100 if is_aggregation else 15
        max_sources = 5
        source_budget = self.source_tokens

        # 1. Select fact items in score order until budget or max_facts is reached
        # Truncation never cuts mid-fact.
        candidate_fact_items = [
            item for item in items if item.kind == 'fact' and item.fact is not None
        ]
        selected_facts: list[tuple[RankedItem, Fact]] = []
        selected_fact_ids: set[FactId] = set()
        accumulated_fact_tokens = 0

        for item in candidate_fact_items:
            fact = item.fact
            assert fact is not None  # for mypy
            if len(selected_facts) >= max_facts:
                break
            # Tentative line to evaluate token consumption
            test_line = _format_fact_line('F1', fact, evidence_map.get(fact.id))
            line_tokens = self.tokens.count(test_line)
            if accumulated_fact_tokens + line_tokens > fact_budget and selected_facts:
                break
            selected_facts.append((item, fact))
            selected_fact_ids.add(fact.id)
            accumulated_fact_tokens += line_tokens

        # 3. Add chain neighbors (at most 6, marked as chain context)
        chain_facts_to_add: list[Fact] = []
        if chain_neighbors:
            for neighbor in chain_neighbors:
                if neighbor.id not in selected_fact_ids and len(chain_facts_to_add) < 6:
                    chain_facts_to_add.append(neighbor)
                    selected_fact_ids.add(neighbor.id)

        # 4. Source items whose id equals an evidence source of a selected fact are dropped.
        # Up to 5 of the remaining sources are added within SB_CTX_SOURCE_TOKENS (800).
        evidence_source_ids: set[SourceId] = {
            snip.source_id for snips in evidence_map.values() for snip in snips
        }

        candidate_sources: list[tuple[RankedItem, Source]] = [
            (item, item.source)
            for item in items
            if item.kind == 'source'
            and item.source is not None
            and item.source.id not in evidence_source_ids
        ]

        selected_sources: list[tuple[RankedItem, Source]] = []
        accumulated_source_tokens = 0
        for item, src in candidate_sources:
            if len(selected_sources) >= max_sources:
                break
            test_src_line = _format_source_line(f'S{len(selected_sources) + 1}', src)
            src_tokens = self.tokens.count(test_src_line)
            if accumulated_source_tokens + src_tokens > source_budget and selected_sources:
                break
            selected_sources.append((item, src))
            accumulated_source_tokens += src_tokens

        # 5. Sort fact lines chronologically by valid_from
        facts_with_origin: list[tuple[Fact, float | None, bool]] = []
        for item, f in selected_facts:
            facts_with_origin.append((f, item.score, False))
        for cf in chain_facts_to_add:
            facts_with_origin.append((cf, None, True))

        # Sort chronologically by valid_from, then by id for determinism
        facts_with_origin.sort(key=lambda t: (t[0].valid_from, str(t[0].id)))

        # Format fact lines
        fact_lines: list[str] = []
        fact_labels: dict[str, FactId] = {}
        used_fact_ids: list[FactId] = []
        traces: list[TraceItem] = []

        for idx, (fact, score, is_chain) in enumerate(facts_with_origin, start=1):
            lbl = f'F{idx}'
            fact_labels[lbl] = fact.id
            used_fact_ids.append(fact.id)
            line = _format_fact_line(lbl, fact, evidence_map.get(fact.id))
            fact_lines.append(line)
            traces.append(
                TraceItem(
                    label=lbl,
                    kind='chain' if is_chain else 'fact',
                    id=fact.id,
                    score=score,
                    detail='chain_context' if is_chain else 'selected',
                )
            )

        # Format source lines
        source_lines: list[str] = []
        source_labels: dict[str, SourceId] = {}
        for idx, (item, src) in enumerate(selected_sources, start=1):
            lbl = f'S{idx}'
            source_labels[lbl] = src.id
            line = _format_source_line(lbl, src)
            source_lines.append(line)
            traces.append(
                TraceItem(
                    label=lbl,
                    kind='source',
                    id=src.id,
                    score=item.score,
                    detail='source',
                )
            )

        # Format recent turns
        recent_lines: list[str] = []
        for idx, msg in enumerate(recent, start=1):
            lbl = f'T{idx}'
            recent_lines.append(f'[{lbl}] {msg.role.value}: {msg.content}')
            traces.append(
                TraceItem(
                    label=lbl,
                    kind='recent',
                    id=msg.id,
                    detail=msg.role.value,
                )
            )

        # Format pending messages
        pending_lines: list[str] = []
        for idx, msg in enumerate(pending, start=1):
            lbl = f'P{idx}'
            pending_lines.append(f'[{lbl}] {msg.content}')
            traces.append(
                TraceItem(
                    label=lbl,
                    kind='pending',
                    id=msg.id,
                    detail='pending',
                )
            )

        # Compute "Today: YYYY-MM-DD"
        try:
            today_dt = plan.question_ts.astimezone(ZoneInfo(plan.tz))
        except Exception:
            today_dt = plan.question_ts
        today_str = today_dt.strftime('%Y-%m-%d')

        # Assemble full prompt block
        block_parts: list[str] = [f'Today: {today_str}']
        if fact_lines or source_lines:
            block_parts.append('MEMORY:')
            block_parts.extend(fact_lines)
            block_parts.extend(source_lines)
        if recent_lines:
            block_parts.append('RECENT TURNS:')
            block_parts.extend(recent_lines)
        if pending_lines:
            block_parts.append('NOT YET PROCESSED:')
            block_parts.extend(pending_lines)

        full_text = '\n'.join(block_parts)

        return AssembledContext(
            text=full_text,
            fact_labels=fact_labels,
            traces=tuple(traces),
            used_fact_ids=tuple(used_fact_ids),
            source_labels=source_labels,
        )

    async def build(
        self,
        uow: UnitOfWork | None,
        plan: QueryPlan,
        items: Sequence[RankedItem],
        recent: Sequence[Message] = (),
        pending: Sequence[Message] = (),
        user_id: UserId | None = None,
    ) -> AssembledContext:
        """Asynchronously build prompt context by fetching evidence and chain neighbors."""
        # Derive user_id if not explicitly provided
        resolved_user_id: UserId | None = user_id
        if resolved_user_id is None:
            for item in items:
                if item.fact is not None:
                    resolved_user_id = item.fact.user_id
                    break
                if item.source is not None:
                    resolved_user_id = item.source.user_id
                    break
            if resolved_user_id is None and recent:
                resolved_user_id = recent[0].user_id
            if resolved_user_id is None and pending:
                resolved_user_id = pending[0].user_id
            if resolved_user_id is None and hasattr(plan, 'user_id'):
                plan_uid = plan.user_id
                if isinstance(plan_uid, UUID):
                    resolved_user_id = UserId(plan_uid)

        # Gather candidate fact IDs to fetch evidence for
        is_aggregation = plan.intent == Intent.AGGREGATION
        max_facts = 100 if is_aggregation else 15
        candidate_facts = [
            item.fact for item in items if item.kind == 'fact' and item.fact is not None
        ][:max_facts]
        fact_ids: list[FactId] = [f.id for f in candidate_facts]

        evidence_map: dict[FactId, list[EvidenceSnippet]] = {}
        chain_neighbors: list[Fact] = []

        if uow is not None and resolved_user_id is not None and fact_ids:
            evidence_map = await uow.search.evidence_for(resolved_user_id, fact_ids, per_fact=1)

            # Step 3: For HISTORICAL, DURATION_OR_ORDER and POINT_IN_TIME plans add chain_neighbors
            if plan.intent in (
                Intent.HISTORICAL,
                Intent.DURATION_OR_ORDER,
                Intent.POINT_IN_TIME,
            ):
                chain_neighbors = await uow.search.chain_neighbors(resolved_user_id, fact_ids)
                if chain_neighbors:
                    chain_ids = [cf.id for cf in chain_neighbors[:6]]
                    chain_ev = await uow.search.evidence_for(
                        resolved_user_id, chain_ids, per_fact=1
                    )
                    evidence_map.update(chain_ev)

        return self.assemble_pure(
            plan=plan,
            items=items,
            evidence=evidence_map,
            chain_neighbors=chain_neighbors,
            recent=recent,
            pending=pending,
        )
