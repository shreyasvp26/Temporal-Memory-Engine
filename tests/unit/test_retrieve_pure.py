"""Unit tests for pure retrieval logic per LLD Sections 9.4, 9.5, and 9.6."""

from datetime import UTC, datetime
from uuid import UUID, uuid4

import pytest
from memory_engine.domain.enums import (
    Cardinality,
    DecayClass,
    FactOrigin,
    FactStatus,
    Intent,
    ObjectKind,
    Role,
    SourceKind,
    TimePrecision,
)
from memory_engine.domain.ids import EntityId, FactId, SourceId, UserId
from memory_engine.domain.intervals import TimeInterval
from memory_engine.domain.models import Fact, Message, Predicate, Source
from memory_engine.ports.repositories import EvidenceSnippet
from memory_engine.retrieve.abstain import AbstentionPolicy
from memory_engine.retrieve.assembler import (
    AssembledContext,
    ContextAssembler,
    TokenCounter,
    TraceItem,
)
from memory_engine.retrieve.decay import DecayScorer
from memory_engine.retrieve.fusion import Hit, RankedItem, RrfFuser
from memory_engine.retrieve.planner import QueryPlan
from memory_engine.retrieve.rerank import NoopReranker


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
    cardinality: Cardinality = Cardinality.SINGLE,
    object_norm: str = 'pune',
    fact_text: str = 'User lives in Pune',
    valid_from: datetime | None = None,
    valid_to: datetime | None = None,
    time_precision: TimePrecision = TimePrecision.DAY,
    recorded_at: datetime | None = None,
    invalidated_at: datetime | None = None,
    last_accessed_at: datetime | None = None,
    status: FactStatus = FactStatus.ACTIVE,
) -> Fact:
    """Helper factory for valid domain Fact instances."""
    return Fact(
        id=fact_id or FactId(uuid4()),
        user_id=user_id or UserId(uuid4()),
        subject_id=subject_id or EntityId(uuid4()),
        predicate=predicate,
        cardinality=cardinality,
        object_entity_id=None,
        object_value=object_norm,
        object_norm=object_norm,
        fact_text=fact_text,
        valid_from=valid_from if valid_from is not None else utc(2026, 1, 1),
        valid_to=valid_to,
        time_precision=time_precision,
        valid_from_explicit=True,
        recorded_at=recorded_at if recorded_at is not None else utc(2026, 1, 1),
        invalidated_at=invalidated_at,
        confidence=0.95,
        salience=1.0,
        access_count=0,
        last_accessed_at=last_accessed_at,
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
    role: Role = Role.USER,
) -> Source:
    """Helper factory for valid domain Source instances."""
    return Source(
        id=source_id or SourceId(uuid4()),
        user_id=user_id or UserId(uuid4()),
        kind=SourceKind.CHAT,
        role=role,
        session_id='test-session',
        turn_index=1,
        content=content,
        source_ts=source_ts if source_ts is not None else utc(2026, 1, 1),
        content_hash='test-hash',
    )


def make_query_plan(
    *,
    question: str = 'Where do I live?',
    question_ts: datetime | None = None,
    tz: str = 'UTC',
    intent: Intent = Intent.CURRENT_STATE,
    as_of: datetime | None = None,
    time_range: TimeInterval | None = None,
) -> QueryPlan:
    """Helper factory for QueryPlan instances."""
    return QueryPlan(
        question=question,
        question_ts=question_ts if question_ts is not None else utc(2026, 10, 21),
        tz=tz,
        intent=intent,
        as_of=as_of,
        time_range=time_range,
        entity_ids=(),
        entity_names=(),
        predicates=('lives_in',),
        multi_hop=False,
        expects_count=False,
        ambiguous_time=False,
        planner_path='rules',
        confidence=1.0,
    )


class DummyRegistry:
    """Mock predicate registry for decay class lookup."""

    def __init__(self, mapping: dict[str, DecayClass]) -> None:
        self.mapping = mapping

    def get(self, name: str) -> Predicate:
        dc = self.mapping.get(name, DecayClass.NORMAL)
        return Predicate(
            name=name,
            cardinality=Cardinality.SINGLE,
            decay_class=dc,
            object_kind=ObjectKind.LITERAL,
            template='{subject} ' + name + ' {object}',
        )


# ==============================================================================
# RRF Fusion & Tie-Breaking Tests (Section 9.4)
# ==============================================================================


def test_rrf_fusion_multiple_lanes() -> None:
    """Items appearing in multiple lanes accumulate reciprocal rank scores."""
    f1 = make_fact(fact_text='Fact 1')
    f2 = make_fact(fact_text='Fact 2')
    fuser = RrfFuser(rrf_k=60)

    # Lane 1: fact_vector (weight 1.0)
    # rank 1: f1 -> 1.0 / (60 + 1) = 1/61
    # rank 2: f2 -> 1.0 / (60 + 2) = 1/62
    lane_vec = [
        Hit(kind='fact', id=f1.id, lane='fact_vector', rank=1, raw_score=0.9, fact=f1),
        Hit(kind='fact', id=f2.id, lane='fact_vector', rank=2, raw_score=0.8, fact=f2),
    ]

    # Lane 2: fact_keyword (weight 0.8)
    # rank 1: f2 -> 0.8 / (60 + 1) = 0.8/61
    # rank 2: f1 -> 0.8 / (60 + 2) = 0.8/62
    lane_kw = [
        Hit(kind='fact', id=f2.id, lane='fact_keyword', rank=1, raw_score=0.85, fact=f2),
        Hit(kind='fact', id=f1.id, lane='fact_keyword', rank=2, raw_score=0.75, fact=f1),
    ]

    fused = fuser.fuse({'fact_vector': lane_vec, 'fact_keyword': lane_kw})

    expected_f1_score = (1.0 / 61) + (0.8 / 62)
    expected_f2_score = (1.0 / 62) + (0.8 / 61)

    assert len(fused) == 2
    f1_item = next(it for it in fused if it.id == f1.id)
    f2_item = next(it for it in fused if it.id == f2.id)

    assert pytest.approx(f1_item.score, rel=1e-5) == expected_f1_score
    assert pytest.approx(f2_item.score, rel=1e-5) == expected_f2_score

    # Check that highest score is ranked first
    if expected_f1_score > expected_f2_score:
        assert fused[0].id == f1.id
    else:
        assert fused[0].id == f2.id


def test_rrf_tie_breaking_by_newer_valid_from() -> None:
    """Facts with identical fused scores break ties by newer valid_from."""
    older_fact = make_fact(valid_from=utc(2024, 1, 1), fact_text='Older fact')
    newer_fact = make_fact(valid_from=utc(2026, 1, 1), fact_text='Newer fact')

    fuser = RrfFuser(rrf_k=60)
    # Both at rank 1 in single-item separate runs, or same lane with same weight
    lane1 = [
        Hit(
            kind='fact',
            id=older_fact.id,
            lane='fact_vector',
            rank=1,
            raw_score=1.0,
            fact=older_fact,
        )
    ]
    lane2 = [
        Hit(
            kind='fact',
            id=newer_fact.id,
            lane='fact_vector',
            rank=1,
            raw_score=1.0,
            fact=newer_fact,
        )
    ]

    fused = fuser.fuse([lane1, lane2])

    assert len(fused) == 2
    assert fused[0].score == fused[1].score
    # Newer valid_from wins the tie-break
    assert fused[0].id == newer_fact.id
    assert fused[1].id == older_fact.id


def test_rrf_tie_breaking_by_newer_source_ts() -> None:
    """Sources with identical fused scores break ties by newer source_ts."""
    older_source = make_source(source_ts=utc(2024, 6, 1), content='Older source')
    newer_source = make_source(source_ts=utc(2026, 6, 1), content='Newer source')

    fuser = RrfFuser(rrf_k=60)
    lane1 = [
        Hit(
            kind='source',
            id=older_source.id,
            lane='source_vector',
            rank=1,
            raw_score=1.0,
            source=older_source,
        )
    ]
    lane2 = [
        Hit(
            kind='source',
            id=newer_source.id,
            lane='source_vector',
            rank=1,
            raw_score=1.0,
            source=newer_source,
        )
    ]

    fused = fuser.fuse([lane1, lane2])

    assert len(fused) == 2
    assert fused[0].score == fused[1].score
    assert fused[0].id == newer_source.id
    assert fused[1].id == older_source.id


def test_rrf_tie_breaking_by_id_determinism() -> None:
    """Items with identical score AND identical timestamp break ties by UUID string."""
    same_ts = utc(2026, 1, 1)
    id_a = FactId(UUID('00000000-0000-0000-0000-000000000001'))
    id_b = FactId(UUID('00000000-0000-0000-0000-000000000002'))

    fact_a = make_fact(fact_id=id_a, valid_from=same_ts)
    fact_b = make_fact(fact_id=id_b, valid_from=same_ts)

    fuser = RrfFuser(rrf_k=60)
    lane_b = [Hit(kind='fact', id=id_b, lane='fact_vector', rank=1, raw_score=1.0, fact=fact_b)]
    lane_a = [Hit(kind='fact', id=id_a, lane='fact_vector', rank=1, raw_score=1.0, fact=fact_a)]

    # Pass lane_b first, then lane_a
    fused = fuser.fuse([lane_b, lane_a])

    assert len(fused) == 2
    assert fused[0].id == id_a
    assert fused[1].id == id_b


# ==============================================================================
# Decay Scorer & Time-Scoped Bypass Tests (Section 9.4, T-DECAY)
# ==============================================================================


@pytest.mark.parametrize(
    'bypassing_intent',
    [
        Intent.HISTORICAL,
        Intent.POINT_IN_TIME,
        Intent.INTERVAL,
        Intent.AGGREGATION,
        Intent.DURATION_OR_ORDER,
    ],
)
def test_decay_bypassed_for_non_current_state_intents(bypassing_intent: Intent) -> None:
    """Decay is strictly bypassed for all historical and time-scoped intents (T-DECAY)."""
    fact = make_fact(valid_from=utc(2020, 1, 1))
    plan = make_query_plan(intent=bypassing_intent, question_ts=utc(2026, 10, 21))

    item = RankedItem(kind='fact', id=fact.id, score=1.0, fact=fact)
    scorer = DecayScorer()
    result = scorer.score(plan, [item])

    assert result[0].score == 1.0


def test_decay_bypassed_when_time_range_or_as_of_present() -> None:
    """Decay is strictly bypassed whenever time_range or as_of is specified (T-DECAY)."""
    fact = make_fact(valid_from=utc(2020, 1, 1))

    # Test with as_of set
    plan_as_of = make_query_plan(
        intent=Intent.CURRENT_STATE,
        as_of=utc(2025, 1, 1),
        question_ts=utc(2026, 10, 21),
    )
    scorer = DecayScorer()
    item = RankedItem(kind='fact', id=fact.id, score=1.0, fact=fact)
    assert scorer.score(plan_as_of, [item])[0].score == 1.0

    # Test with time_range set
    plan_range = make_query_plan(
        intent=Intent.OPEN,
        time_range=TimeInterval(start=utc(2024, 1, 1), end=utc(2025, 1, 1)),
        question_ts=utc(2026, 10, 21),
    )
    assert scorer.score(plan_range, [item])[0].score == 1.0


def test_decay_never_decays_source_items() -> None:
    """Source items are never decayed under any circumstances."""
    src = make_source(source_ts=utc(2020, 1, 1))
    plan = make_query_plan(intent=Intent.CURRENT_STATE, question_ts=utc(2026, 10, 21))

    item = RankedItem(kind='source', id=src.id, score=1.0, source=src)
    scorer = DecayScorer()
    result = scorer.score(plan, [item])

    assert result[0].score == 1.0


def test_decay_formula_values_and_classes() -> None:
    """Verify exact numerical decay formula and decay classes (T-DECAY)."""
    # floor = 0.3, half_life: slow 365, normal 90, fast 14, permanent: 1.0
    registry = DummyRegistry(
        {
            'perm': DecayClass.PERMANENT,
            'slow': DecayClass.SLOW,
            'norm': DecayClass.NORMAL,
            'fast': DecayClass.FAST,
        }
    )
    scorer = DecayScorer(registry=registry)  # type: ignore[arg-type]

    q_ts = utc(2026, 10, 1)

    # 1. Permanent fact: factor = 1.0 regardless of age
    fact_perm = make_fact(predicate='perm', valid_from=utc(2020, 1, 1))
    item_perm = RankedItem(kind='fact', id=fact_perm.id, score=1.0, fact=fact_perm)
    plan = make_query_plan(intent=Intent.CURRENT_STATE, question_ts=q_ts)
    assert scorer.score(plan, [item_perm])[0].score == 1.0

    # 2. Age 0: factor = 1.0 (no decay)
    fact_zero = make_fact(predicate='norm', valid_from=q_ts)
    item_zero = RankedItem(kind='fact', id=fact_zero.id, score=1.0, fact=fact_zero)
    assert scorer.score(plan, [item_zero])[0].score == 1.0

    # 3. Normal fact at exactly 1 half-life (90 days):
    # factor = 0.3 + 0.7 * (0.5 ** 1) = 0.3 + 0.35 = 0.65
    from datetime import timedelta

    fact_90d = make_fact(predicate='norm', valid_from=q_ts - timedelta(days=90))
    item_90d = RankedItem(kind='fact', id=fact_90d.id, score=1.0, fact=fact_90d)
    res_90d = scorer.score(plan, [item_90d])[0]
    assert pytest.approx(res_90d.score, rel=1e-5) == 0.65

    # 4. Fast fact at exactly 1 half-life (14 days): factor = 0.65
    fact_14d = make_fact(predicate='fast', valid_from=q_ts - timedelta(days=14))
    item_14d = RankedItem(kind='fact', id=fact_14d.id, score=1.0, fact=fact_14d)
    res_14d = scorer.score(plan, [item_14d])[0]
    assert pytest.approx(res_14d.score, rel=1e-5) == 0.65

    # 5. Slow fact at exactly 1 half-life (365 days): factor = 0.65
    fact_365d = make_fact(predicate='slow', valid_from=q_ts - timedelta(days=365))
    item_365d = RankedItem(kind='fact', id=fact_365d.id, score=1.0, fact=fact_365d)
    res_365d = scorer.score(plan, [item_365d])[0]
    assert pytest.approx(res_365d.score, rel=1e-5) == 0.65

    # 6. last_accessed_at refreshes anchor:
    # valid_from 365 days ago, but accessed 0 days ago -> age 0 -> factor 1.0
    fact_refreshed = make_fact(
        predicate='norm',
        valid_from=q_ts - timedelta(days=365),
        last_accessed_at=q_ts,
    )
    item_refreshed = RankedItem(kind='fact', id=fact_refreshed.id, score=1.0, fact=fact_refreshed)
    assert scorer.score(plan, [item_refreshed])[0].score == 1.0


# ==============================================================================
# Reranker Protocol & NoopReranker Tests
# ==============================================================================


@pytest.mark.asyncio
async def test_noop_reranker() -> None:
    """NoopReranker passes items through or truncates to top_n."""
    items = [
        RankedItem(kind='fact', id=uuid4(), score=0.9, fact=make_fact()),
        RankedItem(kind='fact', id=uuid4(), score=0.8, fact=make_fact()),
        RankedItem(kind='fact', id=uuid4(), score=0.7, fact=make_fact()),
    ]
    reranker = NoopReranker()
    out_all = await reranker.rerank('test query', items)
    assert len(out_all) == 3

    out_top2 = await reranker.rerank('test query', items, top_n=2)
    assert len(out_top2) == 2
    assert out_top2[0] == items[0]
    assert out_top2[1] == items[1]


# ==============================================================================
# Context Assembler Line Formatting & Truncation Tests (Section 9.5)
# ==============================================================================


def test_token_counter() -> None:
    """TokenCounter uses ceil(len(text) / 3.6)."""
    tc = TokenCounter()
    assert tc.count('') == 0
    # len = 10 -> 10 / 3.6 = 2.777 -> ceil = 3
    assert tc.count('0123456789') == 3
    # len = 4 -> 4 / 3.6 = 1.111 -> ceil = 2
    assert tc.count('1234') == 2


def test_context_assembler_line_formats() -> None:
    """ContextAssembler produces the exact frozen line formats from Section 9.5."""
    plan = make_query_plan(question_ts=utc(2026, 10, 21), tz='UTC')

    # Delhi fact: valid 2023-06 to 2026-10-01 | no longer true since 2026-10-01
    delhi = make_fact(
        fact_text='User lives in Delhi',
        valid_from=utc(2023, 6, 1),
        valid_to=utc(2026, 10, 1),
        time_precision=TimePrecision.MONTH,
        status=FactStatus.ACTIVE,
    )
    # Pune fact: valid 2026-10-01 to present | current
    pune = make_fact(
        fact_text='User lives in Pune',
        valid_from=utc(2026, 10, 1),
        valid_to=None,
        time_precision=TimePrecision.DAY,
        status=FactStatus.ACTIVE,
    )
    # Raw source
    src = make_source(source_ts=utc(2026, 8, 2), content='raw excerpt content')

    evidence = {
        delhi.id: [
            EvidenceSnippet(
                source_id=SourceId(uuid4()),
                source_ts=utc(2026, 1, 10),
                role=Role.USER,
                snippet='I live in Delhi',
            )
        ],
        pune.id: [
            EvidenceSnippet(
                source_id=SourceId(uuid4()),
                source_ts=utc(2026, 10, 12),
                role=Role.USER,
                snippet='I just moved to Pune',
            )
        ],
    }

    recent_msg = Message(
        id=uuid4(),
        conversation_id=uuid4(),
        user_id=delhi.user_id,
        role=Role.USER,
        content='What is my address?',
        client_message_id=None,
        source_id=None,
        partial=False,
        created_at=utc(2026, 10, 21),
    )

    pending_msg = Message(
        id=uuid4(),
        conversation_id=uuid4(),
        user_id=delhi.user_id,
        role=Role.USER,
        content='Remember to check mail',
        client_message_id=None,
        source_id=None,
        partial=False,
        created_at=utc(2026, 10, 21),
    )

    assembler = ContextAssembler()
    items = [
        RankedItem(kind='fact', id=pune.id, score=0.9, fact=pune),
        RankedItem(kind='fact', id=delhi.id, score=0.8, fact=delhi),
        RankedItem(kind='source', id=src.id, score=0.7, source=src),
    ]

    ctx = assembler.assemble_pure(
        plan=plan,
        items=items,
        evidence=evidence,
        recent=[recent_msg],
        pending=[pending_msg],
    )

    delhi_line = (
        '[F1] User lives in Delhi | valid 2023-06 to 2026-10-01 | '
        "no longer true since 2026-10-01 | evidence (2026-01-10, user): 'I live in Delhi'\n"
    )
    pune_line = (
        '[F2] User lives in Pune | valid 2026-10-01 to present | '
        "current | evidence (2026-10-12, user): 'I just moved to Pune'\n"
    )
    expected = (
        'Today: 2026-10-21\n'
        'MEMORY:\n'
        f'{delhi_line}'
        f'{pune_line}'
        '[S1] (2026-08-02, user) raw excerpt content\n'
        'RECENT TURNS:\n'
        '[T1] user: What is my address?\n'
        'NOT YET PROCESSED:\n'
        '[P1] Remember to check mail'
    )

    assert ctx.text == expected
    assert ctx.fact_labels == {'F1': delhi.id, 'F2': pune.id}
    assert ctx.used_fact_ids == (delhi.id, pune.id)


def test_context_assembler_chronological_sort_by_valid_from() -> None:
    """Facts are sorted chronologically by valid_from (Section 9.5 Step 5)."""
    f_early = make_fact(fact_text='Early', valid_from=utc(2022, 1, 1))
    f_mid = make_fact(fact_text='Mid', valid_from=utc(2024, 1, 1))
    f_late = make_fact(fact_text='Late', valid_from=utc(2026, 1, 1))

    # Input items in arbitrary/score order
    items = [
        RankedItem(kind='fact', id=f_late.id, score=0.9, fact=f_late),
        RankedItem(kind='fact', id=f_early.id, score=0.7, fact=f_early),
        RankedItem(kind='fact', id=f_mid.id, score=0.8, fact=f_mid),
    ]

    plan = make_query_plan()
    assembler = ContextAssembler()
    ctx = assembler.assemble_pure(plan=plan, items=items)

    assert ctx.fact_labels['F1'] == f_early.id
    assert ctx.fact_labels['F2'] == f_mid.id
    assert ctx.fact_labels['F3'] == f_late.id


def test_token_truncation_never_cuts_mid_fact() -> None:
    """Fact budget truncation stops before cutting a fact in half."""
    f1 = make_fact(fact_text='Short fact 1')
    f2 = make_fact(fact_text='Second fact that requires more tokens than the tight budget allows')

    plan = make_query_plan()
    tc = TokenCounter()
    f1_tokens = tc.count('[F1] Short fact 1 | valid 2026-01-01 to present | current')

    # Set budget just enough for f1, but not for f1 + f2
    assembler = ContextAssembler(fact_tokens=f1_tokens + 5)
    items = [
        RankedItem(kind='fact', id=f1.id, score=0.9, fact=f1),
        RankedItem(kind='fact', id=f2.id, score=0.8, fact=f2),
    ]

    ctx = assembler.assemble_pure(plan=plan, items=items)

    # f1 is included completely; f2 is entirely excluded (never cut mid-fact)
    assert len(ctx.used_fact_ids) == 1
    assert ctx.used_fact_ids[0] == f1.id
    assert '[F1]' in ctx.text
    assert '[F2]' not in ctx.text


def test_source_deduplication_against_selected_fact_evidence() -> None:
    """Evidence sources of selected facts are dropped from source items (Section 9.5)."""
    ev_source_id = SourceId(uuid4())
    other_source_id = SourceId(uuid4())

    fact = make_fact(fact_text='Fact with evidence')
    evidence = {
        fact.id: [
            EvidenceSnippet(
                source_id=ev_source_id,
                source_ts=utc(2026, 1, 1),
                role=Role.USER,
                snippet='Evidence snippet',
            )
        ]
    }

    src_dup = make_source(source_id=ev_source_id, content='Duplicate source text')
    src_other = make_source(source_id=other_source_id, content='Non-duplicate source text')

    items = [
        RankedItem(kind='fact', id=fact.id, score=0.9, fact=fact),
        RankedItem(kind='source', id=src_dup.id, score=0.8, source=src_dup),
        RankedItem(kind='source', id=src_other.id, score=0.7, source=src_other),
    ]

    assembler = ContextAssembler()
    ctx = assembler.assemble_pure(plan=make_query_plan(), items=items, evidence=evidence)

    # src_dup is dropped because its id is in fact evidence
    # src_other is retained as S1
    assert 'Non-duplicate source text' in ctx.text
    assert 'Duplicate source text' not in ctx.text
    assert ctx.source_labels == {'S1': other_source_id}


@pytest.mark.asyncio
async def test_chain_neighbors_added_for_historical_queries() -> None:
    """Chain neighbors are fetched and added for historical/order queries (Section 9.5 Step 3)."""
    f_main = make_fact(fact_text='Main fact')
    f_neighbor = make_fact(fact_text='Predecessor fact', valid_from=utc(2024, 1, 1))

    class MockSearchRepo:
        async def evidence_for(
            self, user_id: UserId, fact_ids: list[FactId], per_fact: int
        ) -> dict[FactId, list[EvidenceSnippet]]:
            return {}

        async def chain_neighbors(self, user_id: UserId, fact_ids: list[FactId]) -> list[Fact]:
            return [f_neighbor]

    class MockUoW:
        search = MockSearchRepo()

    plan = make_query_plan(intent=Intent.HISTORICAL)
    assembler = ContextAssembler()
    items = [RankedItem(kind='fact', id=f_main.id, score=0.9, fact=f_main)]

    ctx = await assembler.build(
        uow=MockUoW(),  # type: ignore[arg-type]
        plan=plan,
        items=items,
        user_id=f_main.user_id,
    )

    assert len(ctx.used_fact_ids) == 2
    assert f_neighbor.id in ctx.used_fact_ids
    # Predecessor has earlier valid_from, so it is F1, and Main is F2
    assert ctx.fact_labels['F1'] == f_neighbor.id
    assert ctx.fact_labels['F2'] == f_main.id
    chain_trace = next(t for t in ctx.traces if t.id == f_neighbor.id)
    assert chain_trace.kind == 'chain'
    assert chain_trace.detail == 'chain_context'


# ==============================================================================
# Abstention Policy Tests (Section 9.6)
# ==============================================================================


def test_abstention_statement_only_never_abstains() -> None:
    """Statement-only input never triggers abstention (Section 9.6 Rule 1)."""
    policy = AbstentionPolicy(min_score=0.5)
    plan = make_query_plan()
    empty_ctx = AssembledContext(
        text='Today: 2026-10-21',
        fact_labels={},
        traces=(),
        used_fact_ids=(),
    )

    decision = policy.decide(plan, result=None, ctx=empty_ctx, statement_only=True)
    assert not decision.should_abstain
    assert not decision.abstained
    assert decision.reason == 'none'


def test_abstention_no_evidence_gives_canned_reply() -> None:
    """No facts and no sources in context gives abstain(no_evidence) (Section 9.6 Rule 2)."""
    policy = AbstentionPolicy()
    plan = make_query_plan()
    empty_ctx = AssembledContext(
        text='Today: 2026-10-21',
        fact_labels={},
        traces=(),
        used_fact_ids=(),
    )

    decision = policy.decide(plan, result=None, ctx=empty_ctx, statement_only=False)
    assert decision.should_abstain
    assert decision.reason == 'no_evidence'
    assert decision.reply == "I don't have anything in your memory about that yet."


def test_abstention_low_evidence_when_min_score_exceeded() -> None:
    """When min_score > 0, low best score without StructuredLane gives abstain(low_evidence)."""
    policy = AbstentionPolicy(min_score=0.5)
    plan = make_query_plan()
    fact_id = FactId(uuid4())

    ctx_with_fact = AssembledContext(
        text='Today: 2026-10-21\nMEMORY:\n[F1] ...',
        fact_labels={'F1': fact_id},
        traces=(TraceItem(label='F1', kind='fact', id=fact_id),),
        used_fact_ids=(fact_id,),
    )

    class MockRetrievalResult:
        def __init__(self, score: float, lanes: tuple[str, ...]) -> None:
            self.items = [
                RankedItem(kind='fact', id=fact_id, score=score, fact=make_fact(), lanes=lanes)
            ]
            self.has_structured_hit = 'structured' in lanes

    # Score 0.2 < min_score 0.5, no structured lane hit -> abstain(low_evidence)
    res_low = MockRetrievalResult(score=0.2, lanes=('fact_vector',))
    dec_low = policy.decide(plan, result=res_low, ctx=ctx_with_fact, statement_only=False)
    assert dec_low.should_abstain
    assert dec_low.reason == 'low_evidence'
    assert dec_low.reply == "I don't have anything in your memory about that yet."

    # Score 0.2 < min_score 0.5, BUT structured lane hit -> proceed!
    res_struct = MockRetrievalResult(score=0.2, lanes=('fact_vector', 'structured'))
    dec_struct = policy.decide(plan, result=res_struct, ctx=ctx_with_fact, statement_only=False)
    assert not dec_struct.should_abstain
    assert dec_struct.reason == 'none'

    # Score 0.8 >= min_score 0.5 -> proceed!
    res_high = MockRetrievalResult(score=0.8, lanes=('fact_vector',))
    dec_high = policy.decide(plan, result=res_high, ctx=ctx_with_fact, statement_only=False)
    assert not dec_high.should_abstain
