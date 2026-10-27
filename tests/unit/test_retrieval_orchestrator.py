"""Unit tests for RetrievalOrchestrator and search lanes (M5-01).

Tests:
  1. Full 5-lane retrieval orchestration (vector_facts, fts_facts, graph_expand,
     temporal_facts, vector_sources) and context assembly.
  2. Partial lane failure degradation (T-RET: 1 lane raises TransientStorageError,
     orchestrator succeeds with remaining lanes and records degraded_lanes).
  3. Total lane failure raising AllLanesFailed.
  4. Embedder failure degradation (flags no_vector in degraded_lanes).
  5. LLM planner fallback when needs_llm_planner triggers.
  6. LLM planner failure falling back to rules plan with planner_path='fallback'.
  7. Temporal filter fallback retrying with ANY_HISTORY and recording temporal_fallback.
  8. Abstention evaluation when no evidence is found.
  9. Statement-only messages never abstaining.
  10. Raw string question planning and retrieval orchestration.
"""

from collections.abc import AsyncIterator, Sequence
from datetime import UTC, datetime
from types import TracebackType
from typing import Any, TypeVar, cast
from uuid import uuid4

import pytest
from memory_engine.domain.enums import (
    Cardinality,
    EntityType,
    FactOrigin,
    FactStatus,
    Intent,
    Role,
    SourceKind,
    TimePrecision,
)
from memory_engine.domain.errors import (
    AllLanesFailed,
    LLMTimeout,
    TransientStorageError,
)
from memory_engine.domain.ids import EntityId, FactId, SourceId, UserId
from memory_engine.domain.models import Entity, Fact, Source
from memory_engine.ports.clock import FixedClock
from memory_engine.ports.llm import LLMRequest, LLMResponse, PlannerLLMOutput
from memory_engine.ports.repositories import (
    EvidenceSnippet,
    FilterMode,
    ScoredFact,
    ScoredSource,
    TemporalFilter,
)
from memory_engine.ports.uow import UnitOfWork
from memory_engine.retrieve.orchestrator import RetrievalOrchestrator
from memory_engine.retrieve.planner.rules import QueryPlan

T = TypeVar('T')


def utc(
    year: int,
    month: int = 1,
    day: int = 1,
    hour: int = 0,
    minute: int = 0,
    second: int = 0,
) -> datetime:
    """Helper creating timezone-aware UTC datetime."""
    return datetime(year, month, day, hour, minute, second, tzinfo=UTC)


def make_fact(
    *,
    fact_id: FactId | None = None,
    user_id: UserId | None = None,
    subject_id: EntityId | None = None,
    predicate: str = 'lives_in',
    object_norm: str = 'pune',
    fact_text: str = 'User lives in Pune',
    valid_from: datetime | None = None,
    status: FactStatus = FactStatus.ACTIVE,
) -> Fact:
    """Helper factory for valid domain Fact instances."""
    return Fact(
        id=fact_id or FactId(uuid4()),
        user_id=user_id or UserId(uuid4()),
        subject_id=subject_id or EntityId(uuid4()),
        predicate=predicate,
        cardinality=Cardinality.SINGLE,
        object_entity_id=None,
        object_value=object_norm,
        object_norm=object_norm,
        fact_text=fact_text,
        valid_from=valid_from if valid_from is not None else utc(2026, 1, 1),
        valid_to=None,
        time_precision=TimePrecision.DAY,
        valid_from_explicit=True,
        recorded_at=utc(2026, 1, 1),
        invalidated_at=None,
        confidence=0.95,
        salience=1.0,
        access_count=0,
        last_accessed_at=None,
        status=status,
        origin=FactOrigin.EXTRACTED,
        event_hash=None,
    )


def make_source(
    *,
    source_id: SourceId | None = None,
    user_id: UserId | None = None,
    content: str = 'I live in Pune',
    source_ts: datetime | None = None,
) -> Source:
    """Helper factory for valid domain Source instances."""
    return Source(
        id=source_id or SourceId(uuid4()),
        user_id=user_id or UserId(uuid4()),
        kind=SourceKind.CHAT,
        role=Role.USER,
        session_id='test-session',
        turn_index=1,
        content=content,
        source_ts=source_ts if source_ts is not None else utc(2026, 1, 1),
        content_hash='test-hash',
    )


class FakeSearchRepository:
    """Configurable in-memory search repository for retrieval tests."""

    def __init__(self) -> None:
        self.vector_facts_results: list[ScoredFact] = []
        self.fts_facts_results: list[ScoredFact] = []
        self.temporal_facts_results: list[ScoredFact] = []
        self.graph_expand_results: list[ScoredFact] = []
        self.vector_sources_results: list[ScoredSource] = []
        self.evidence_map: dict[FactId, list[EvidenceSnippet]] = {}
        self.chain_neighbors_list: list[Fact] = []
        self.fallback_fn: Any = None

        # Exceptions to raise per method
        self.fail_vector_facts: Exception | None = None
        self.fail_fts_facts: Exception | None = None
        self.fail_temporal_facts: Exception | None = None
        self.fail_graph_expand: Exception | None = None
        self.fail_vector_sources: Exception | None = None

        # Call tracking
        self.called_methods: list[str] = []

    async def vector_facts(
        self,
        user_id: UserId,
        embedding: Sequence[float],
        flt: TemporalFilter,
        limit: int,
    ) -> list[ScoredFact]:
        self.called_methods.append('vector_facts')
        if self.fail_vector_facts is not None:
            raise self.fail_vector_facts
        return list(self.vector_facts_results[:limit])

    async def keyword_facts(
        self,
        user_id: UserId,
        query: str,
        flt: TemporalFilter,
        limit: int,
    ) -> list[ScoredFact]:
        return await self.fts_facts(user_id, query, flt, limit)

    async def fts_facts(
        self,
        user_id: UserId,
        query: str,
        flt: TemporalFilter,
        limit: int,
    ) -> list[ScoredFact]:
        self.called_methods.append('fts_facts')
        if self.fallback_fn is not None:
            return await self.fallback_fn(user_id, query, flt, limit)  # type: ignore[no-any-return]
        if self.fail_fts_facts is not None:
            raise self.fail_fts_facts
        return list(self.fts_facts_results[:limit])

    async def structured_facts(
        self,
        user_id: UserId,
        subject_ids: Sequence[EntityId] | None,
        predicates: Sequence[str] | None,
        flt: TemporalFilter,
        limit: int,
    ) -> list[ScoredFact]:
        return await self.temporal_facts(user_id, subject_ids, predicates, flt, limit)

    async def temporal_facts(
        self,
        user_id: UserId,
        subject_ids: Sequence[EntityId] | None,
        predicates: Sequence[str] | None,
        flt: TemporalFilter,
        limit: int,
    ) -> list[ScoredFact]:
        self.called_methods.append('temporal_facts')
        if self.fail_temporal_facts is not None:
            raise self.fail_temporal_facts
        return list(self.temporal_facts_results[:limit])

    async def vector_sources(
        self,
        user_id: UserId,
        embedding: Sequence[float],
        limit: int,
        before: datetime | None,
    ) -> list[ScoredSource]:
        self.called_methods.append('vector_sources')
        if self.fail_vector_sources is not None:
            raise self.fail_vector_sources
        return list(self.vector_sources_results[:limit])

    async def keyword_sources(
        self,
        user_id: UserId,
        query: str,
        limit: int,
        before: datetime | None,
    ) -> list[ScoredSource]:
        return []

    async def expand_graph(
        self,
        user_id: UserId,
        seed_entity_ids: Sequence[EntityId],
        depth: int,
        flt: TemporalFilter,
        limit: int,
    ) -> list[ScoredFact]:
        return await self.graph_expand(user_id, seed_entity_ids, depth, flt, limit)

    async def graph_expand(
        self,
        user_id: UserId,
        seed_entity_ids: Sequence[EntityId],
        depth: int,
        flt: TemporalFilter,
        limit: int,
    ) -> list[ScoredFact]:
        self.called_methods.append('graph_expand')
        if self.fail_graph_expand is not None:
            raise self.fail_graph_expand
        return list(self.graph_expand_results[:limit])

    async def evidence_for(
        self,
        user_id: UserId,
        fact_ids: Sequence[FactId],
        per_fact: int,
    ) -> dict[FactId, list[EvidenceSnippet]]:
        return {fid: self.evidence_map.get(fid, []) for fid in fact_ids}

    async def chain_neighbors(
        self,
        user_id: UserId,
        fact_ids: Sequence[FactId],
    ) -> list[Fact]:
        return list(self.chain_neighbors_list)


class FakeEntityRepository:
    """In-memory entity repository for alias resolution."""

    def __init__(self) -> None:
        self.entities_by_norm: dict[str, Entity] = {}

    async def find_by_norm_name(
        self,
        user_id: UserId,
        name_or_type: str | EntityType,
        alias_or_type: str | EntityType | None = None,
        entity_type: EntityType | None = None,
    ) -> Entity | None:
        key = str(name_or_type).strip().lower()
        return self.entities_by_norm.get(key)


class FakeUnitOfWork:
    """Fake UnitOfWork implementing search and entities repositories."""

    def __init__(
        self,
        search_repo: FakeSearchRepository,
        entity_repo: FakeEntityRepository,
    ) -> None:
        self.search = search_repo
        self.entities = entity_repo
        self.sources: Any = None
        self.facts: Any = None
        self.jobs: Any = None
        self.messages: Any = None
        self.audit: Any = None
        self.profiles: Any = None
        self.locked_user: UserId | None = None

    async def lock_user(self, user_id: UserId | None = None) -> None:
        self.locked_user = user_id

    async def __aenter__(self) -> 'FakeUnitOfWork':
        return self

    async def __aexit__(
        self,
        et: type[BaseException] | None,
        ev: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        pass


class FakeUnitOfWorkFactory:
    """Fake UnitOfWorkFactory providing per-user FakeUnitOfWork sessions."""

    def __init__(
        self,
        search_repo: FakeSearchRepository | None = None,
        entity_repo: FakeEntityRepository | None = None,
    ) -> None:
        self.search = search_repo or FakeSearchRepository()
        self.entities = entity_repo or FakeEntityRepository()

    def begin(self, user_id: UserId | None, *, readonly: bool = False) -> UnitOfWork:
        return cast('UnitOfWork', FakeUnitOfWork(self.search, self.entities))

    def for_user(self, user_id: UserId, *, readonly: bool = False) -> UnitOfWork:
        return cast('UnitOfWork', FakeUnitOfWork(self.search, self.entities))

    def system(self, *, readonly: bool = False) -> UnitOfWork:
        return cast('UnitOfWork', FakeUnitOfWork(self.search, self.entities))


class FakeEmbedder:
    """In-memory embedder for retrieval tests."""

    def __init__(self, dim: int = 384, fail: bool = False) -> None:
        self.dim = dim
        self.fail = fail
        self.model_id = 'test-embedder'

    async def embed_query(self, text: str) -> list[float]:
        if self.fail:
            raise RuntimeError('Embedder model crashed')
        return [0.05] * self.dim

    async def embed_queries(self, texts: Sequence[str]) -> list[list[float]]:
        if self.fail:
            raise RuntimeError('Embedder model crashed')
        return [[0.05] * self.dim for _ in texts]

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [[0.05] * self.dim for _ in texts]


class FakeLLMGateway:
    """In-memory LLM gateway for query planner fallback tests."""

    def __init__(
        self,
        planner_output: PlannerLLMOutput | None = None,
        fail: bool = False,
    ) -> None:
        self.planner_output = planner_output
        self.fail = fail
        self.requests: list[LLMRequest] = []

    async def complete_text(self, req: LLMRequest) -> LLMResponse:
        self.requests.append(req)
        return LLMResponse(
            text='ok',
            model='test',
            provider='test',
            prompt_tokens=10,
            completion_tokens=5,
            latency_ms=10.0,
        )

    async def complete_json(self, req: LLMRequest, schema: type[T]) -> T:
        self.requests.append(req)
        if self.fail:
            raise LLMTimeout('LLM planner request timed out')
        if self.planner_output is not None and issubclass(schema, PlannerLLMOutput):
            return self.planner_output  # type: ignore[return-value]
        return schema()

    async def stream_text(self, req: LLMRequest) -> AsyncIterator[str]:
        yield 'ok'


# ==============================================================================
# TESTS
# ==============================================================================


@pytest.mark.asyncio
async def test_full_5_lane_retrieval_orchestration_and_context_assembly() -> None:
    """Full 5-lane retrieval orchestration (vector_facts, fts_facts, graph_expand,

    temporal_facts, vector_sources) and context assembly.
    """
    user_id = UserId(uuid4())
    search = FakeSearchRepository()
    entities = FakeEntityRepository()

    # Create distinct facts and sources for all 5 lanes
    fact_vec = make_fact(user_id=user_id, fact_text='User lives in Zurich', predicate='lives_in')
    fact_fts = make_fact(user_id=user_id, fact_text='User works at Google', predicate='works_at')
    fact_grp = make_fact(user_id=user_id, fact_text='User knows Alice', predicate='knows_colleague')
    fact_tmp = make_fact(
        user_id=user_id,
        fact_text='User moved to Zurich in 2024',
        predicate='moved_to',
        valid_from=utc(2024, 1, 1),
    )
    src_vec = make_source(
        user_id=user_id, content='I just moved to Zurich last month', source_ts=utc(2026, 9, 1)
    )

    search.vector_facts_results = [ScoredFact(fact=fact_vec, score=0.95)]
    search.fts_facts_results = [ScoredFact(fact=fact_fts, score=0.88)]
    search.graph_expand_results = [ScoredFact(fact=fact_grp, score=0.78)]
    search.temporal_facts_results = [ScoredFact(fact=fact_tmp, score=0.85)]
    search.vector_sources_results = [ScoredSource(source=src_vec, score=0.72)]

    # Provide evidence snippet
    search.evidence_map[fact_vec.id] = [
        EvidenceSnippet(
            source_id=src_vec.id,
            source_ts=src_vec.source_ts,
            role=src_vec.role,
            snippet=src_vec.content,
            fact_id=fact_vec.id,
        )
    ]

    # Set up entity hint resolution
    alice_id = EntityId(uuid4())
    entities.entities_by_norm['alice'] = Entity(
        id=alice_id,
        user_id=user_id,
        canonical_name='Alice',
        type=EntityType.PERSON,
        is_self=False,
    )

    uow_factory = FakeUnitOfWorkFactory(search_repo=search, entity_repo=entities)
    embedder = FakeEmbedder()
    clock = FixedClock(utc(2026, 10, 27))

    orchestrator = RetrievalOrchestrator(
        uow_factory=uow_factory,
        embedder=embedder,
        clock=clock,
    )

    # Query plan that activates all 5 lanes:
    # - vector_facts: always
    # - fts_facts: always
    # - graph_expand: multi_hop or entity hints ('alice')
    # - temporal_facts: intent DURATION_OR_ORDER or as_of/time_range
    # - vector_sources: included by VectorLane
    plan = QueryPlan(
        question='Where did I live and work with Alice in 2024?',
        question_ts=utc(2026, 10, 27),
        tz='UTC',
        intent=Intent.DURATION_OR_ORDER,
        as_of=utc(2024, 1, 1),
        time_range=None,
        entity_ids=(),
        entity_names=('alice',),
        predicates=('lives_in', 'works_at'),
        multi_hop=True,
        expects_count=False,
        ambiguous_time=False,
        planner_path='rules',
        confidence=0.9,
    )

    result = await orchestrator.retrieve(user_id=user_id, query=plan)

    # Verify lane counts and 5-lane execution
    assert result.lane_counts['vector_facts'] == 1
    assert result.lane_counts['fts_facts'] == 1
    assert result.lane_counts['graph_expand'] == 1
    assert result.lane_counts['temporal_facts'] == 1
    assert result.lane_counts['vector_sources'] == 1
    assert result.degraded_lanes == []

    # Verify ranked items from fusion
    assert len(result.ranked_items) == 5
    ranked_ids = {item.id for item in result.ranked_items}
    assert fact_vec.id in ranked_ids
    assert fact_fts.id in ranked_ids
    assert fact_grp.id in ranked_ids
    assert fact_tmp.id in ranked_ids
    assert src_vec.id in ranked_ids

    # Verify context assembly
    assert 'MEMORY:' in result.context.text
    assert 'Zurich' in result.context.text
    assert 'Google' in result.context.text
    assert len(result.context.fact_labels) > 0
    assert not result.abstention.should_abstain


@pytest.mark.asyncio
async def test_partial_lane_failure_degradation_t_ret() -> None:
    """T-RET: 1 lane raises TransientStorageError, orchestrator succeeds with

    remaining lanes and records degraded_lanes.
    """
    user_id = UserId(uuid4())
    search = FakeSearchRepository()
    entities = FakeEntityRepository()

    fact_vec = make_fact(user_id=user_id, fact_text='User lives in Boston')
    search.vector_facts_results = [ScoredFact(fact=fact_vec, score=0.9)]

    # Make FTS lane fail with TransientStorageError
    search.fail_fts_facts = TransientStorageError('Postgres FTS query timeout')

    uow_factory = FakeUnitOfWorkFactory(search_repo=search, entity_repo=entities)
    embedder = FakeEmbedder()
    clock = FixedClock(utc(2026, 10, 27))

    orchestrator = RetrievalOrchestrator(
        uow_factory=uow_factory,
        embedder=embedder,
        clock=clock,
    )

    plan = QueryPlan(
        question='Where do I live?',
        question_ts=utc(2026, 10, 27),
        tz='UTC',
        intent=Intent.CURRENT_STATE,
        as_of=None,
        time_range=None,
        entity_ids=(),
        entity_names=(),
        predicates=('lives_in',),
        multi_hop=False,
        expects_count=False,
        ambiguous_time=False,
        planner_path='rules',
        confidence=0.9,
    )

    result = await orchestrator.retrieve(user_id=user_id, query=plan)

    # Partial degradation succeeds:
    assert 'fts_facts' in result.degraded_lanes
    assert result.lane_counts['vector_facts'] == 1
    assert result.lane_counts['fts_facts'] == 0
    assert len(result.ranked_items) >= 1
    assert result.ranked_items[0].fact is not None
    assert result.ranked_items[0].fact.fact_text == 'User lives in Boston'
    assert not result.abstention.should_abstain


@pytest.mark.asyncio
async def test_total_lane_failure_raises_all_lanes_failed() -> None:
    """When ALL executed lanes fail with exceptions, orchestrator raises AllLanesFailed."""
    user_id = UserId(uuid4())
    search = FakeSearchRepository()
    entities = FakeEntityRepository()

    # Make all search methods fail
    search.fail_vector_facts = TransientStorageError('Connection reset by peer')
    search.fail_vector_sources = TransientStorageError('Connection reset by peer')
    search.fail_fts_facts = TransientStorageError('Connection reset by peer')
    search.fail_temporal_facts = TransientStorageError('Connection reset by peer')
    search.fail_graph_expand = TransientStorageError('Connection reset by peer')

    uow_factory = FakeUnitOfWorkFactory(search_repo=search, entity_repo=entities)
    embedder = FakeEmbedder()
    clock = FixedClock(utc(2026, 10, 27))

    orchestrator = RetrievalOrchestrator(
        uow_factory=uow_factory,
        embedder=embedder,
        clock=clock,
    )

    plan = QueryPlan(
        question='Where do I live?',
        question_ts=utc(2026, 10, 27),
        tz='UTC',
        intent=Intent.CURRENT_STATE,
        as_of=None,
        time_range=None,
        entity_ids=(),
        entity_names=(),
        predicates=(),
        multi_hop=False,
        expects_count=False,
        ambiguous_time=False,
        planner_path='rules',
        confidence=0.9,
    )

    with pytest.raises(AllLanesFailed) as exc_info:
        await orchestrator.retrieve(user_id=user_id, query=plan)

    assert 'All' in str(exc_info.value)
    assert 'failed' in str(exc_info.value)


@pytest.mark.asyncio
async def test_embedder_failure_skips_vector_lanes_and_flags_no_vector() -> None:
    """When Embedder raises an exception, vector lanes are skipped and no_vector is flagged."""
    user_id = UserId(uuid4())
    search = FakeSearchRepository()
    fact_fts = make_fact(user_id=user_id, fact_text='User likes photography')
    search.fts_facts_results = [ScoredFact(fact=fact_fts, score=0.8)]

    uow_factory = FakeUnitOfWorkFactory(search_repo=search)
    embedder = FakeEmbedder(fail=True)  # Embedder will raise RuntimeError
    clock = FixedClock(utc(2026, 10, 27))

    orchestrator = RetrievalOrchestrator(
        uow_factory=uow_factory,
        embedder=embedder,
        clock=clock,
    )

    plan = QueryPlan(
        question='What are my hobbies?',
        question_ts=utc(2026, 10, 27),
        tz='UTC',
        intent=Intent.OPEN,
        as_of=None,
        time_range=None,
        entity_ids=(),
        entity_names=(),
        predicates=(),
        multi_hop=False,
        expects_count=False,
        ambiguous_time=False,
        planner_path='rules',
        confidence=0.9,
    )

    result = await orchestrator.retrieve(user_id=user_id, query=plan)

    assert 'no_vector' in result.degraded_lanes
    assert result.lane_counts['vector_facts'] == 0
    assert result.lane_counts['fts_facts'] == 1
    assert len(result.ranked_items) == 1
    assert result.ranked_items[0].fact == fact_fts


@pytest.mark.asyncio
async def test_llm_planner_fallback_triggered() -> None:
    """When needs_llm_planner triggers and enable_llm_planner is True, LLM planner runs."""
    user_id = UserId(uuid4())
    search = FakeSearchRepository()
    fact = make_fact(user_id=user_id, fact_text='User worked at Acme Corp')
    search.fts_facts_results = [ScoredFact(fact=fact, score=0.85)]

    uow_factory = FakeUnitOfWorkFactory(search_repo=search)
    embedder = FakeEmbedder()
    clock = FixedClock(utc(2026, 10, 27))

    # Fake LLM returns structured planner output
    llm_output = PlannerLLMOutput(
        intent=Intent.INTERVAL,
        time_phrases=['in 2024'],
        entity_names=['Acme Corp'],
        predicates=['works_at'],
        multi_hop=False,
    )
    llm_gateway = FakeLLMGateway(planner_output=llm_output)

    orchestrator = RetrievalOrchestrator(
        uow_factory=uow_factory,
        embedder=embedder,
        llm_gateway=llm_gateway,
        clock=clock,
        enable_llm_planner=True,
    )

    # A question with 2 distinct time cues triggers needs_llm_planner
    question = 'Where was I living in 2022 and what was my role in 2024?'
    result = await orchestrator.retrieve(user_id=user_id, query=question)

    assert len(llm_gateway.requests) == 1
    assert result.plan.planner_path == 'llm'
    assert result.plan.intent == Intent.INTERVAL
    assert 'Acme Corp' in result.plan.entity_names
    assert result.plan.time_range is not None
    assert result.plan.time_range.start == utc(2024, 1, 1)


@pytest.mark.asyncio
async def test_llm_planner_fallback_on_llm_error() -> None:
    """When LLM planner raises an error, orchestrator falls back to rules plan (fallback)."""
    user_id = UserId(uuid4())
    search = FakeSearchRepository()
    fact = make_fact(user_id=user_id, fact_text='User lives in Berlin')
    search.fts_facts_results = [ScoredFact(fact=fact, score=0.85)]

    uow_factory = FakeUnitOfWorkFactory(search_repo=search)
    embedder = FakeEmbedder()
    clock = FixedClock(utc(2026, 10, 27))

    # Gateway fails with timeout
    llm_gateway = FakeLLMGateway(fail=True)

    orchestrator = RetrievalOrchestrator(
        uow_factory=uow_factory,
        embedder=embedder,
        llm_gateway=llm_gateway,
        clock=clock,
        enable_llm_planner=True,
    )

    question = 'Where was I living in 2022 and what was my role in 2024?'
    result = await orchestrator.retrieve(user_id=user_id, query=question)

    assert len(llm_gateway.requests) == 1
    assert result.plan.planner_path == 'fallback'
    assert len(result.ranked_items) == 1


@pytest.mark.asyncio
async def test_temporal_filter_fallback_retries_with_any_history() -> None:
    """When AS_OF / OVERLAP search returns 0 facts, retries once with ANY_HISTORY."""
    user_id = UserId(uuid4())
    search = FakeSearchRepository()

    fact = make_fact(user_id=user_id, fact_text='User lived in Paris formerly')

    # Simulate: first call under AS_OF returns empty list,
    # second call under ANY_HISTORY returns fact
    call_count = 0

    async def fts_with_fallback(
        uid: UserId, query: str, flt: TemporalFilter, limit: int
    ) -> list[ScoredFact]:
        nonlocal call_count
        call_count += 1
        if flt.mode == FilterMode.ANY_HISTORY:
            return [ScoredFact(fact=fact, score=0.8)]
        return []

    search.fallback_fn = fts_with_fallback

    uow_factory = FakeUnitOfWorkFactory(search_repo=search)
    embedder = FakeEmbedder()
    clock = FixedClock(utc(2026, 10, 27))

    orchestrator = RetrievalOrchestrator(
        uow_factory=uow_factory,
        embedder=embedder,
        clock=clock,
    )

    plan = QueryPlan(
        question='Where was I in 2020?',
        question_ts=utc(2026, 10, 27),
        tz='UTC',
        intent=Intent.POINT_IN_TIME,
        as_of=utc(2020, 1, 1),
        time_range=None,
        entity_ids=(),
        entity_names=(),
        predicates=(),
        multi_hop=False,
        expects_count=False,
        ambiguous_time=False,
        planner_path='rules',
        confidence=0.9,
    )

    result = await orchestrator.retrieve(user_id=user_id, query=plan)

    assert 'temporal_fallback' in result.degraded_lanes
    assert len(result.ranked_items) == 1
    assert result.ranked_items[0].fact == fact


@pytest.mark.asyncio
async def test_abstention_when_no_evidence_found() -> None:
    """When all lanes return 0 hits, context has no evidence and abstention triggers."""
    user_id = UserId(uuid4())
    search = FakeSearchRepository()

    uow_factory = FakeUnitOfWorkFactory(search_repo=search)
    embedder = FakeEmbedder()
    clock = FixedClock(utc(2026, 10, 27))

    orchestrator = RetrievalOrchestrator(
        uow_factory=uow_factory,
        embedder=embedder,
        clock=clock,
    )

    plan = QueryPlan(
        question='What is my secret nickname?',
        question_ts=utc(2026, 10, 27),
        tz='UTC',
        intent=Intent.OPEN,
        as_of=None,
        time_range=None,
        entity_ids=(),
        entity_names=(),
        predicates=(),
        multi_hop=False,
        expects_count=False,
        ambiguous_time=False,
        planner_path='rules',
        confidence=0.9,
    )

    result = await orchestrator.retrieve(user_id=user_id, query=plan)

    assert len(result.ranked_items) == 0
    assert result.abstention.should_abstain is True
    assert result.abstention.reason == 'no_evidence'
    assert result.abstention.reply == "I don't have anything in your memory about that yet."


@pytest.mark.asyncio
async def test_statement_only_never_abstains() -> None:
    """Statement-only inputs never trigger abstention even with zero retrieved items."""
    user_id = UserId(uuid4())
    search = FakeSearchRepository()

    uow_factory = FakeUnitOfWorkFactory(search_repo=search)
    embedder = FakeEmbedder()
    clock = FixedClock(utc(2026, 10, 27))

    orchestrator = RetrievalOrchestrator(
        uow_factory=uow_factory,
        embedder=embedder,
        clock=clock,
    )

    plan = QueryPlan(
        question='I am learning Italian.',
        question_ts=utc(2026, 10, 27),
        tz='UTC',
        intent=Intent.OPEN,
        as_of=None,
        time_range=None,
        entity_ids=(),
        entity_names=(),
        predicates=(),
        multi_hop=False,
        expects_count=False,
        ambiguous_time=False,
        planner_path='rules',
        confidence=0.9,
    )

    result = await orchestrator.retrieve(user_id=user_id, query=plan, statement_only=True)

    assert len(result.ranked_items) == 0
    assert result.abstention.should_abstain is False


@pytest.mark.asyncio
async def test_retrieve_with_raw_string_question() -> None:
    """RetrievalOrchestrator accepts raw string questions and plans via RulesPlanner."""
    user_id = UserId(uuid4())
    search = FakeSearchRepository()
    fact = make_fact(user_id=user_id, fact_text='User works at Acme')
    search.vector_facts_results = [ScoredFact(fact=fact, score=0.92)]

    uow_factory = FakeUnitOfWorkFactory(search_repo=search)
    embedder = FakeEmbedder()
    clock = FixedClock(utc(2026, 10, 27, 12, 0))

    orchestrator = RetrievalOrchestrator(
        uow_factory=uow_factory,
        embedder=embedder,
        clock=clock,
    )

    result = await orchestrator.retrieve(user_id=user_id, query='What is my current job?')

    assert result.plan.intent == Intent.CURRENT_STATE
    assert result.plan.planner_path == 'rules'
    assert len(result.ranked_items) >= 1
    assert result.ranked_items[0].fact == fact
    assert not result.abstention.should_abstain
