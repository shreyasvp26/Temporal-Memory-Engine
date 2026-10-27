"""Unit tests for evaluation metrics and ablation baselines (M6-01).

Tests:
1. normalize_answer: lowercase, remove punctuation, remove articles, collapse whitespace.
2. exact_match: normalized exact equality, whitespace/punctuation insensitivity.
3. token_f1: precision, recall, harmonic mean, duplicate handling, edge cases.
4. compute_latency_stats: empty, single item, percentiles (p50, p95, p99), mean.
5. paired_bootstrap_ci: validation, zero difference, constant delta, determinism, bounds.
6. percentile_bootstrap_ci & abstention_accuracy: single CI and abstention classification.
7. Baseline configs: comparative table entries (vector_only, bm25_only, hybrid_no_bitemporal,
   bitemporal_no_decay, full_bitemporal, full_context) and ablation flags.
8. SingleAsMultiStrategy & conflict registry: supersession ablation behavior.
9. Baseline orchestrator builder: correct lane assembly and filter overrides.
10. Baseline system wrappers: FullContextBaseline and RetrievalBaseline with SimpleReader.
"""

from datetime import UTC, datetime
from types import TracebackType
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from memory_engine.domain.enums import (
    Cardinality,
    FactOrigin,
    FactStatus,
    TimePrecision,
)
from memory_engine.domain.ids import EntityId, FactId, SourceId, UserId
from memory_engine.domain.models import Candidate, Fact
from memory_engine.ingest.conflict.plan import Outcome, ResolutionContext
from memory_engine.ingest.conflict.single import SingleValuedStrategy
from memory_engine.ports.repositories import FilterMode
from memory_engine.retrieve.lanes.fts import FtsLane
from memory_engine.retrieve.lanes.graph import GraphLane
from memory_engine.retrieve.lanes.vector import VectorLane

if TYPE_CHECKING:
    from memory_engine.ports.uow import UnitOfWorkFactory

from eval.baselines import (
    AblationFlags,
    BaselineRetrievalOrchestrator,
    BitemporalNoDecayBaseline,
    Bm25OnlyBaseline,
    EvalCase,
    EvalSession,
    EvalTurn,
    FullBitemporalBaseline,
    FullContextBaseline,
    HybridNoTemporalBaseline,
    SimpleReader,
    SingleAsMultiStrategy,
    VectorOnlyBaseline,
    build_baseline_conflict_registry,
    build_baseline_orchestrator,
    create_baseline,
    get_baseline_config,
)
from eval.metrics import (
    LatencyStats,
    abstention_accuracy,
    compute_latency_stats,
    exact_match,
    is_abstention_text,
    normalize_answer,
    paired_bootstrap_ci,
    percentile_bootstrap_ci,
    token_f1,
)


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


class DummyUowContext:
    """Async context manager for fake UnitOfWork."""

    def __init__(self, uow: Any) -> None:
        self.uow = uow

    async def __aenter__(self) -> Any:
        return self.uow

    async def __aexit__(
        self,
        et: type[BaseException] | None,
        ev: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        pass


class DummyUowFactory:
    """Fake UnitOfWorkFactory for orchestrator instantiation."""

    def __init__(self, uow: Any | None = None) -> None:
        self.uow = uow or AsyncMock()

    def begin(self, user_id: UserId | None, *, readonly: bool = False) -> Any:
        return DummyUowContext(self.uow)

    def for_user(self, user_id: UserId, *, readonly: bool = False) -> Any:
        return DummyUowContext(self.uow)

    def system(self, *, readonly: bool = False) -> Any:
        return DummyUowContext(self.uow)


# ============================================================================
# 1. Answer Normalization Tests
# ============================================================================


def test_normalize_answer_casing_and_whitespace() -> None:
    assert normalize_answer('HELLO WORLD') == 'hello world'
    assert normalize_answer('  hello   \n\t world  ') == 'hello world'
    assert normalize_answer('') == ''
    assert normalize_answer('   ') == ''


def test_normalize_answer_punctuation() -> None:
    raw = 'Hello, world! Is this (great)? Yes: absolutely; 100%.'
    expected = 'hello world is this great yes absolutely 100'
    assert normalize_answer(raw) == expected
    assert normalize_answer('!?.--;:,') == ''


def test_normalize_answer_articles() -> None:
    raw = 'The quick brown fox jumped over a lazy dog in an orchard.'
    expected = 'quick brown fox jumped over lazy dog in orchard'
    assert normalize_answer(raw) == expected
    assert normalize_answer('A An The') == ''


# ============================================================================
# 2. Exact Match Tests
# ============================================================================


def test_exact_match_basic() -> None:
    assert exact_match('Paris', 'Paris') is True
    assert exact_match('The Eiffel Tower.', 'eiffel tower') is True
    assert exact_match('42', '42') is True
    assert exact_match('Paris', 'London') is False
    assert exact_match('', '') is True
    assert exact_match('Paris', '') is False


# ============================================================================
# 3. Token F1 Tests
# ============================================================================


def test_token_f1_identical_and_empty() -> None:
    assert token_f1('Paris', 'Paris') == 1.0
    assert token_f1('The Eiffel Tower', 'eiffel tower') == 1.0
    assert token_f1('', '') == 1.0
    assert token_f1('Paris', '') == 0.0
    assert token_f1('', 'Paris') == 0.0


def test_token_f1_partial_overlap() -> None:
    # pred: ["san", "francisco", "ca"], gold: ["san", "francisco"]
    # overlap: 2, prec: 2/3, rec: 2/2 = 1.0, f1 = 2 * (2/3 * 1) / (2/3 + 1) = 0.8
    f1 = token_f1('San Francisco, CA', 'San Francisco')
    assert pytest.approx(f1, abs=1e-4) == 0.8


def test_token_f1_zero_overlap() -> None:
    assert token_f1('Tokyo, Japan', 'Berlin, Germany') == 0.0


def test_token_f1_duplicate_tokens() -> None:
    # pred: ["cat", "cat", "dog"], gold: ["cat", "dog"]
    # common: cat: 1, dog: 1 -> num_same = 2
    # prec = 2/3, rec = 2/2 = 1.0, f1 = 0.8
    f1 = token_f1('cat cat dog', 'cat dog')
    assert pytest.approx(f1, abs=1e-4) == 0.8


# ============================================================================
# 4. Latency Statistics Tests
# ============================================================================


def test_compute_latency_stats_empty_and_single() -> None:
    empty_stats = compute_latency_stats([])
    assert empty_stats == LatencyStats(p50=0.0, p95=0.0, p99=0.0, mean=0.0)

    single_stats = compute_latency_stats([42.5])
    assert single_stats == LatencyStats(p50=42.5, p95=42.5, p99=42.5, mean=42.5)


def test_compute_latency_stats_percentiles() -> None:
    data = [float(x) for x in range(1, 101)]  # 1.0 to 100.0
    stats = compute_latency_stats(data)

    assert pytest.approx(stats.mean, abs=1e-4) == 50.5
    assert pytest.approx(stats.p50, abs=1e-4) == 50.5
    assert pytest.approx(stats.p95, abs=1e-4) == 95.05
    assert pytest.approx(stats.p99, abs=1e-4) == 99.01


# ============================================================================
# 5. Paired Bootstrap CI Tests
# ============================================================================


def test_paired_bootstrap_ci_validation() -> None:
    with pytest.raises(ValueError, match='same length'):
        paired_bootstrap_ci([1.0, 0.5], [1.0])


def test_paired_bootstrap_ci_empty() -> None:
    diff, low, high = paired_bootstrap_ci([], [])
    assert diff == 0.0
    assert low == 0.0
    assert high == 0.0


def test_paired_bootstrap_ci_identical_scores() -> None:
    scores = [0.8, 0.9, 0.5, 0.7, 1.0]
    diff, low, high = paired_bootstrap_ci(scores, scores, n_resamples=500, seed=1337)
    assert diff == 0.0
    assert low == 0.0
    assert high == 0.0


def test_paired_bootstrap_ci_constant_delta() -> None:
    scores_b = [0.2, 0.4, 0.6, 0.8]
    scores_a = [x + 0.15 for x in scores_b]
    diff, low, high = paired_bootstrap_ci(scores_a, scores_b, n_resamples=500, seed=1337)
    assert pytest.approx(diff, abs=1e-5) == 0.15
    assert pytest.approx(low, abs=1e-5) == 0.15
    assert pytest.approx(high, abs=1e-5) == 0.15


def test_paired_bootstrap_ci_determinism_and_bounds() -> None:
    scores_a = [1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0]
    scores_b = [0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]

    res1 = paired_bootstrap_ci(scores_a, scores_b, n_resamples=1000, seed=42)
    res2 = paired_bootstrap_ci(scores_a, scores_b, n_resamples=1000, seed=42)
    assert res1 == res2

    mean_diff, ci_low, ci_high = res1
    assert mean_diff == pytest.approx(0.4, abs=1e-4)
    assert ci_low <= mean_diff <= ci_high
    assert ci_low >= 0.0  # System A is noticeably better


# ============================================================================
# 6. Percentile Bootstrap & Abstention Accuracy Tests
# ============================================================================


def test_percentile_bootstrap_ci_and_abstention_accuracy() -> None:
    scores = [0.5, 0.6, 0.7, 0.8]
    mean_val, low, high = percentile_bootstrap_ci(scores, n_resamples=500, seed=42)
    assert pytest.approx(mean_val, abs=1e-4) == 0.65
    assert low <= mean_val <= high

    # Abstention phrases
    assert is_abstention_text('The information is not available in the notes.') is True
    assert is_abstention_text('I do not know the answer.') is True
    assert is_abstention_text('He lives in Seattle, Washington.') is False

    preds: list[str | bool] = [
        'The information is not available.',  # abstained (True)
        'User was in Paris on Tuesday.',  # answered (False)
        True,  # abstained (True)
        False,  # answered (False)
    ]
    golds = [True, False, True, False]
    assert abstention_accuracy(preds, golds) == 1.0

    mismatched_golds = [True, True, True, False]
    assert abstention_accuracy(preds, mismatched_golds) == 0.75

    with pytest.raises(ValueError, match='same length'):
        abstention_accuracy(['yes'], [True, False])


# ============================================================================
# 7. Baseline Configuration Preset Tests
# ============================================================================


def test_comparative_table_baseline_configs() -> None:
    # 1. vector_only
    v_cfg = get_baseline_config('vector_only')
    assert v_cfg.name == 'vector_only'
    assert v_cfg.filter_mode_override == FilterMode.ANY_HISTORY
    assert v_cfg.enabled_lanes == ('vector_facts',)
    assert v_cfg.ablation_flags.decay is False
    assert v_cfg.ablation_flags.graph is False

    # 2. bm25_only
    bm_cfg = get_baseline_config('bm25_only')
    assert bm_cfg.name == 'bm25_only'
    assert bm_cfg.filter_mode_override == FilterMode.ANY_HISTORY
    assert bm_cfg.enabled_lanes == ('fts_facts',)
    assert bm_cfg.ablation_flags.decay is False

    # 3. hybrid_no_bitemporal
    hy_cfg = get_baseline_config('hybrid_no_bitemporal')
    assert hy_cfg.name == 'hybrid_no_bitemporal'
    assert hy_cfg.filter_mode_override == FilterMode.ANY_HISTORY
    assert hy_cfg.ablation_flags.supersession is False
    assert hy_cfg.ablation_flags.temporal_filter is False
    assert 'vector_facts' in hy_cfg.enabled_lanes
    assert 'fts_facts' in hy_cfg.enabled_lanes

    # 4. bitemporal_no_decay
    nd_cfg = get_baseline_config('bitemporal_no_decay')
    assert nd_cfg.name == 'bitemporal_no_decay'
    assert nd_cfg.ablation_flags.decay is False
    assert nd_cfg.ablation_flags.temporal_filter is True
    assert nd_cfg.ablation_flags.supersession is True
    assert len(nd_cfg.enabled_lanes) == 5

    # 5. full_bitemporal
    fb_cfg = get_baseline_config('full_bitemporal')
    assert fb_cfg.name == 'full_bitemporal'
    assert fb_cfg.ablation_flags.decay is True
    assert fb_cfg.ablation_flags.temporal_filter is True
    assert fb_cfg.ablation_flags.supersession is True
    assert fb_cfg.filter_mode_override is None
    assert len(fb_cfg.enabled_lanes) == 5

    # 6. full_context
    fc_cfg = get_baseline_config('full_context')
    assert fc_cfg.name == 'full_context'


def test_baseline_aliases_and_unknown() -> None:
    assert get_baseline_config('tmem') == get_baseline_config('full_bitemporal')
    assert get_baseline_config('hybrid') == get_baseline_config('hybrid_no_bitemporal')
    assert get_baseline_config('bm25') == get_baseline_config('bm25_only')

    with pytest.raises(KeyError, match='Unknown baseline'):
        get_baseline_config('non_existent_baseline')


# ============================================================================
# 8. SingleAsMultiStrategy & Conflict Registry Tests
# ============================================================================


def test_build_baseline_conflict_registry() -> None:
    full_flags = AblationFlags(supersession=True)
    full_reg = build_baseline_conflict_registry(full_flags)
    assert isinstance(full_reg.for_cardinality(Cardinality.SINGLE), SingleValuedStrategy)

    ablation_flags = AblationFlags(supersession=False)
    ablation_reg = build_baseline_conflict_registry(ablation_flags)
    strategy = ablation_reg.for_cardinality(Cardinality.SINGLE)
    assert isinstance(strategy, SingleAsMultiStrategy)

    # Verify SingleAsMultiStrategy does NOT close previous fact
    user_id = UserId(uuid4())
    subject_id = EntityId(uuid4())
    existing_fact = Fact(
        id=FactId(uuid4()),
        user_id=user_id,
        subject_id=subject_id,
        predicate='lives_in',
        cardinality=Cardinality.SINGLE,
        object_entity_id=None,
        object_value='Seattle',
        object_norm='seattle',
        fact_text='User lives in Seattle',
        valid_from=utc(2026, 1, 1),
        valid_to=None,
        time_precision=TimePrecision.DAY,
        valid_from_explicit=True,
        recorded_at=utc(2026, 1, 1),
        invalidated_at=None,
        confidence=0.9,
        salience=1.0,
        access_count=0,
        last_accessed_at=None,
        origin=FactOrigin.EXTRACTED,
        status=FactStatus.ACTIVE,
        event_hash=None,
    )
    new_candidate = Candidate(
        user_id=user_id,
        subject_id=subject_id,
        predicate='lives_in',
        cardinality=Cardinality.SINGLE,
        object_entity_id=None,
        object_value='San Francisco',
        object_norm='san francisco',
        fact_text='User lives in San Francisco',
        valid_from=utc(2026, 6, 1),
        valid_to=None,
        time_precision=TimePrecision.DAY,
        valid_from_explicit=True,
        confidence=0.95,
        is_retraction=False,
        source_id=SourceId(uuid4()),
        span=(0, 20),
        source_ts=utc(2026, 6, 1),
        event_hash=None,
        embedding=[0.1, 0.2, 0.3],
    )
    ctx = ResolutionContext(now=utc(2026, 6, 1))

    plan = strategy.plan(new_candidate, [existing_fact], ctx)
    assert plan.outcome == Outcome.INSERTED
    # SingleValuedStrategy would close existing fact; MultiValuedStrategy does not
    op_names = [op.__class__.__name__ for op in plan.ops]
    assert 'InsertFact' in op_names
    assert 'CloseFact' not in op_names


# ============================================================================
# 9. Baseline Orchestrator Builder Tests
# ============================================================================


def test_build_baseline_orchestrator_lane_wiring() -> None:
    uow_factory = cast('UnitOfWorkFactory', DummyUowFactory())

    # Vector only
    v_orch = build_baseline_orchestrator(get_baseline_config('vector_only'), uow_factory)
    assert isinstance(v_orch, BaselineRetrievalOrchestrator)
    assert len(v_orch.lanes) == 1
    assert isinstance(v_orch.lanes[0], VectorLane)
    assert v_orch.lanes[0].include_sources is False
    assert v_orch.filter_mode_override == FilterMode.ANY_HISTORY
    assert v_orch.decay.decay_enabled is False

    # BM25 only
    bm_orch = build_baseline_orchestrator(get_baseline_config('bm25_only'), uow_factory)
    assert len(bm_orch.lanes) == 1
    assert isinstance(bm_orch.lanes[0], FtsLane)
    assert bm_orch.filter_mode_override == FilterMode.ANY_HISTORY

    # Full bitemporal
    fb_orch = build_baseline_orchestrator(get_baseline_config('full_bitemporal'), uow_factory)
    assert len(fb_orch.lanes) == 4  # Vector, FTS, Graph, Temporal
    assert fb_orch.filter_mode_override is None
    assert fb_orch.decay.decay_enabled is True

    # tmem_no_graph ablation
    ng_orch = build_baseline_orchestrator(get_baseline_config('tmem_no_graph'), uow_factory)
    assert not any(isinstance(lane, GraphLane) for lane in ng_orch.lanes)


# ============================================================================
# 10. Baseline System Wrappers (MemorySystem Protocol) Tests
# ============================================================================


@pytest.mark.asyncio
async def test_full_context_baseline_prepare_and_answer() -> None:
    reader = SimpleReader(default_response='Alice lives in Seattle.')
    baseline = FullContextBaseline(reader=reader, max_context_tokens=100)

    case = EvalCase(
        case_id='case_001',
        qtype='current_state',
        question='Where does Alice live?',
        gold='Seattle',
        question_ts=utc(2026, 6, 1),
        sessions=(
            EvalSession(
                session_id='s1',
                ts=utc(2026, 1, 1),
                turns=(
                    EvalTurn(role='user', content='I moved to Seattle.'),
                    EvalTurn(role='assistant', content='Great! How is Seattle?'),
                ),
            ),
            EvalSession(
                session_id='s2',
                ts=utc(2026, 3, 1),
                turns=(EvalTurn(role='user', content='Still enjoying Seattle.'),),
            ),
        ),
        evidence_session_ids=frozenset({'s1'}),
        is_abstention=False,
    )

    await baseline.prepare(case)
    answer = await baseline.answer(case)

    assert answer.text == 'Alice lives in Seattle.'
    assert answer.retrieved_session_ids == ('s1', 's2')
    assert answer.context_tokens > 0
    assert answer.latency_ms >= 0.0

    assert baseline.has_case('case_001')
    await baseline.cleanup(case)
    assert not baseline.has_case('case_001')


@pytest.mark.asyncio
async def test_create_baseline_factory() -> None:
    fc = create_baseline('full_context')
    assert isinstance(fc, FullContextBaseline)

    vo = create_baseline('vector_only')
    assert isinstance(vo, VectorOnlyBaseline)

    bm = create_baseline('bm25_only')
    assert isinstance(bm, Bm25OnlyBaseline)

    hy = create_baseline('hybrid_no_bitemporal')
    assert isinstance(hy, HybridNoTemporalBaseline)

    nd = create_baseline('bitemporal_no_decay')
    assert isinstance(nd, BitemporalNoDecayBaseline)

    fb = create_baseline('full_bitemporal')
    assert isinstance(fb, FullBitemporalBaseline)

    # Offline answer without orchestrator uses SimpleReader fallback
    case = EvalCase(
        case_id='case_offline',
        qtype='open',
        question='What is my hobby?',
        gold='hiking',
        question_ts=utc(2026, 6, 1),
        sessions=(),
        evidence_session_ids=frozenset(),
        is_abstention=True,
    )
    res = await vo.answer(case)
    assert 'information is not available' in res.text
    assert res.extras['config'] == 'vector_only'
