"""Unit tests for ports, repository DTOs, protocols, and TemporalFilter."""

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import uuid4

import pytest
from memory_engine.domain.enums import (
    FactStatus,
    Intent,
    JobType,
    Role,
    SourceKind,
)
from memory_engine.domain.errors import InvariantViolation
from memory_engine.domain.ids import (
    FactId,
    JobId,
    SourceId,
    UserId,
)
from memory_engine.domain.intervals import TimeInterval
from memory_engine.domain.models import Source
from memory_engine.ports import (
    AuditRepository,
    ChatMessage,
    Clock,
    Embedder,
    EntityRepository,
    ErrorInfo,
    EvidenceSnippet,
    FactListQuery,
    FactRepository,
    FilterMode,
    FixedClock,
    InsertedMessage,
    InsertedSource,
    JobRepository,
    LLMClient,
    LLMGateway,
    LLMRequest,
    LLMResponse,
    MessageRepository,
    NewExtractJob,
    NewJob,
    NewMessage,
    NewSource,
    Page,
    PredicateRegistry,
    ProfileRepository,
    ScoredSource,
    SearchRepository,
    SourceRepository,
    SourceWindow,
    SystemClock,
    TemporalFilter,
    UnitOfWork,
    UnitOfWorkFactory,
    UpsertResult,
)


@dataclass(frozen=True, slots=True)
class _DummyPlan:
    question: str = 'Where do I live?'
    question_ts: datetime = datetime(2026, 10, 27, 12, 0, tzinfo=UTC)
    intent: Intent | str = Intent.CURRENT_STATE
    as_of: datetime | None = None
    time_range: TimeInterval | None = None


def test_protocols_are_importable_and_types() -> None:
    protocols: list[type[Any]] = [
        SourceRepository,
        EntityRepository,
        FactRepository,
        SearchRepository,
        JobRepository,
        MessageRepository,
        PredicateRegistry,
        AuditRepository,
        ProfileRepository,
        UnitOfWork,
        UnitOfWorkFactory,
        LLMClient,
        LLMGateway,
        Embedder,
        Clock,
        FixedClock,
        SystemClock,
    ]
    for p in protocols:
        assert isinstance(p, type)


def test_temporal_filter_current() -> None:
    flt = TemporalFilter(mode=FilterMode.CURRENT)
    frag = flt.to_sql('f', first_param=1)
    assert frag.sql == "f.status = 'active' and f.valid_to is null and f.invalidated_at is null"
    assert frag.text == frag.sql
    assert frag.params == ()


def test_temporal_filter_as_of() -> None:
    now = datetime(2026, 10, 27, 12, 0, tzinfo=UTC)
    flt = TemporalFilter(mode=FilterMode.AS_OF, as_of=now)
    frag = flt.to_sql('f', first_param=3)
    assert frag.sql == (
        "f.status in ('active','archived') and f.valid_from <= $3 "
        'and (f.valid_to is null or $3 < f.valid_to)'
    )
    assert frag.params == (now,)


def test_temporal_filter_as_of_missing_timestamp_raises() -> None:
    flt = TemporalFilter(mode=FilterMode.AS_OF, as_of=None)
    with pytest.raises(InvariantViolation, match='as_of must be provided'):
        flt.to_sql('f', first_param=1)


def test_temporal_filter_overlap_both_bounds() -> None:
    t1 = datetime(2026, 1, 1, 0, 0, tzinfo=UTC)
    t2 = datetime(2026, 6, 1, 0, 0, tzinfo=UTC)
    flt = TemporalFilter(mode=FilterMode.OVERLAP, range=TimeInterval(start=t1, end=t2))
    frag = flt.to_sql('f', first_param=2)
    assert frag.sql == (
        "f.status in ('active','archived') and f.valid_from < $2 "
        'and (f.valid_to is null or f.valid_to > $3)'
    )
    assert frag.params == (t2, t1)


def test_temporal_filter_overlap_end_only() -> None:
    t2 = datetime(2026, 6, 1, 0, 0, tzinfo=UTC)
    flt = TemporalFilter(mode=FilterMode.OVERLAP, range=TimeInterval(start=None, end=t2))
    frag = flt.to_sql('f', first_param=4)
    assert frag.sql == "f.status in ('active','archived') and f.valid_from < $4"
    assert frag.params == (t2,)


def test_temporal_filter_overlap_start_only() -> None:
    t1 = datetime(2026, 1, 1, 0, 0, tzinfo=UTC)
    flt = TemporalFilter(mode=FilterMode.OVERLAP, range=TimeInterval(start=t1, end=None))
    frag = flt.to_sql('f', first_param=1)
    assert (
        frag.sql == "f.status in ('active','archived') and (f.valid_to is null or f.valid_to > $1)"
    )
    assert frag.params == (t1,)


def test_temporal_filter_overlap_no_bounds() -> None:
    flt = TemporalFilter(mode=FilterMode.OVERLAP, range=TimeInterval(start=None, end=None))
    frag = flt.to_sql('f', first_param=1)
    assert frag.sql == "f.status in ('active','archived')"
    assert frag.params == ()

    flt_none = TemporalFilter(mode=FilterMode.OVERLAP, range=None)
    frag_none = flt_none.to_sql('f', first_param=1)
    assert frag_none.sql == "f.status in ('active','archived')"
    assert frag_none.params == ()


def test_temporal_filter_any_history() -> None:
    flt = TemporalFilter(mode=FilterMode.ANY_HISTORY)
    frag = flt.to_sql('f', first_param=1)
    assert frag.sql == "f.status in ('active','archived')"
    assert frag.params == ()


def test_temporal_filter_open_with_and_without_cutoff() -> None:
    cutoff = datetime(2025, 10, 27, 0, 0, tzinfo=UTC)
    flt_with = TemporalFilter(mode=FilterMode.OPEN, cutoff=cutoff)
    frag_with = flt_with.to_sql('f', first_param=5)
    assert frag_with.sql == "f.status = 'active' and (f.valid_to is null or f.valid_to > $5)"
    assert frag_with.params == (cutoff,)

    flt_without = TemporalFilter(mode=FilterMode.OPEN, cutoff=None)
    frag_without = flt_without.to_sql('f', first_param=1)
    assert frag_without.sql == "f.status = 'active' and f.valid_to is null"
    assert frag_without.params == ()


def test_temporal_filter_from_plan_mapping_priority() -> None:
    t_start = datetime(2026, 1, 1, tzinfo=UTC)
    t_end = datetime(2026, 5, 1, tzinfo=UTC)
    interval = TimeInterval(start=t_start, end=t_end)
    as_of = datetime(2026, 3, 1, tzinfo=UTC)

    # Priority 1: time_range forces OVERLAP even if as_of is set
    plan1 = _DummyPlan(intent=Intent.CURRENT_STATE, as_of=as_of, time_range=interval)
    f1 = TemporalFilter.from_plan(plan1)
    assert f1.mode == FilterMode.OVERLAP
    assert f1.range == interval

    # Priority 2: as_of forces AS_OF if time_range is None
    plan2 = _DummyPlan(intent=Intent.HISTORICAL, as_of=as_of, time_range=None)
    f2 = TemporalFilter.from_plan(plan2)
    assert f2.mode == FilterMode.AS_OF
    assert f2.as_of == as_of

    # Priority 3: Intent defaults
    p_cur = _DummyPlan(intent=Intent.CURRENT_STATE)
    assert TemporalFilter.from_plan(p_cur).mode == FilterMode.CURRENT

    p_point = _DummyPlan(intent=Intent.POINT_IN_TIME)
    assert TemporalFilter.from_plan(p_point).mode == FilterMode.AS_OF

    p_int = _DummyPlan(intent=Intent.INTERVAL)
    assert TemporalFilter.from_plan(p_int).mode == FilterMode.OVERLAP

    p_hist = _DummyPlan(intent=Intent.HISTORICAL)
    assert TemporalFilter.from_plan(p_hist).mode == FilterMode.ANY_HISTORY

    p_dur = _DummyPlan(intent=Intent.DURATION_OR_ORDER)
    assert TemporalFilter.from_plan(p_dur).mode == FilterMode.ANY_HISTORY

    p_agg = _DummyPlan(intent=Intent.AGGREGATION)
    assert TemporalFilter.from_plan(p_agg).mode == FilterMode.ANY_HISTORY

    # Open intent computes cutoff = question_ts - 365 days
    q_ts = datetime(2026, 10, 27, 0, 0, tzinfo=UTC)
    p_open = _DummyPlan(intent=Intent.OPEN, question_ts=q_ts)
    f_open = TemporalFilter.from_plan(p_open)
    assert f_open.mode == FilterMode.OPEN
    assert f_open.cutoff == q_ts - timedelta(days=365)


def test_temporal_filter_utc_enforcement() -> None:
    naive_dt = datetime(2026, 1, 1, 0, 0)
    with pytest.raises(InvariantViolation, match='must be timezone-aware UTC'):
        TemporalFilter(mode=FilterMode.AS_OF, as_of=naive_dt)

    with pytest.raises(InvariantViolation, match='must be timezone-aware UTC'):
        TemporalFilter(mode=FilterMode.OPEN, cutoff=naive_dt)


def test_filter_mode_case_insensitivity() -> None:
    assert FilterMode('current') == FilterMode.CURRENT
    assert FilterMode('CURRENT') == FilterMode.CURRENT
    assert FilterMode('As_Of') == FilterMode.AS_OF
    assert FilterMode('ANY_HISTORY') == FilterMode.ANY_HISTORY


def test_dto_instantiation_and_properties() -> None:
    u_id = UserId(uuid4())
    s_id = SourceId(uuid4())
    f_id = FactId(uuid4())
    j_id = JobId(uuid4())
    now = datetime(2026, 10, 27, 12, 0, tzinfo=UTC)

    # NewSource
    ns = NewSource(
        user_id=u_id,
        kind=SourceKind.CHAT,
        role=Role.USER,
        session_id='sess-1',
        turn_index=0,
        content='I live in Tokyo.',
        source_ts=now,
        content_hash='hash-123',
    )
    assert ns.turn_index == 0
    assert ns.id is None

    # NewSource naive timestamp rejection
    with pytest.raises(InvariantViolation, match='source_ts must be timezone-aware UTC'):
        NewSource(
            user_id=u_id,
            kind=SourceKind.CHAT,
            role=Role.USER,
            session_id='sess-1',
            turn_index=0,
            content='I live in Tokyo.',
            source_ts=datetime(2026, 1, 1),
            content_hash='hash-123',
        )

    # InsertedSource
    ins = InsertedSource(id=s_id, created=True)
    assert ins.id == s_id
    assert ins.created is True

    # SourceWindow
    src = Source(
        id=s_id,
        user_id=u_id,
        kind=SourceKind.CHAT,
        role=Role.USER,
        session_id='sess-1',
        turn_index=0,
        content='I live in Tokyo.',
        source_ts=now,
        content_hash='hash-123',
    )
    win = SourceWindow(targets=[src], context=[])
    assert len(win.targets) == 1
    assert len(win.context) == 0

    # FactListQuery
    flq = FactListQuery(status=FactStatus.ACTIVE, as_of=now, page_size=10)
    assert flq.page_size == 10
    with pytest.raises(InvariantViolation, match='as_of must be timezone-aware UTC'):
        FactListQuery(as_of=datetime(2026, 1, 1))

    # Page
    page = Page[str](items=['a', 'b'], next_cursor='cursor-1')
    assert len(page.items) == 2
    assert page.next_cursor == 'cursor-1'

    # ScoredSource
    ss = ScoredSource(source=src, score=0.95)
    assert ss.score == 0.95
    assert ss.source == src

    # EvidenceSnippet
    ev = EvidenceSnippet(
        source_id=s_id,
        source_ts=now,
        role=Role.USER,
        snippet='live in Tokyo',
        fact_id=f_id,
    )
    assert ev.snippet == 'live in Tokyo'
    with pytest.raises(InvariantViolation, match='source_ts must be timezone-aware UTC'):
        EvidenceSnippet(
            source_id=s_id,
            source_ts=datetime(2026, 1, 1),
            role=Role.USER,
            snippet='test',
        )

    # NewExtractJob
    nej = NewExtractJob(
        user_id=u_id,
        session_id='sess-1',
        source_id=s_id,
        idempotency_key='ik-1',
        coalesce_key='ck-1',
    )
    assert nej.idempotency_key == 'ik-1'

    # UpsertResult
    ur = UpsertResult(job_id=j_id, created=True, sealed=False)
    assert ur.job_id == j_id
    assert ur.created is True
    assert ur.sealed is False

    # NewJob
    nj = NewJob(
        user_id=u_id,
        type=JobType.EXTRACT,
        payload={'k': 'v'},
        idempotency_key='ik-job',
        run_after=now,
    )
    assert nj.max_attempts == 3
    with pytest.raises(InvariantViolation, match='run_after must be timezone-aware UTC'):
        NewJob(
            user_id=u_id,
            type=JobType.EXTRACT,
            payload={},
            idempotency_key='ik',
            run_after=datetime(2026, 1, 1),
        )

    # ErrorInfo
    err = ErrorInfo(message='lease expired', code='E-JOB-LEASE')
    assert err.last_error == 'lease expired'
    assert err.error_code == 'E-JOB-LEASE'

    # NewMessage & InsertedMessage
    msg_id = uuid4()
    nm = NewMessage(
        conversation_id=uuid4(),
        user_id=u_id,
        role=Role.USER,
        content='Hello',
    )
    assert nm.partial is False
    im = InsertedMessage(id=msg_id, created=True)
    assert im.id == msg_id

    # ChatMessage, LLMRequest, LLMResponse
    cm = ChatMessage(role='user', content='Hello')
    req = LLMRequest(
        purpose='extract',
        tier='fast',
        messages=[cm],
        prompt_version='v1',
    )
    assert req.temperature == 0.0
    res = LLMResponse(
        text='response',
        model='llama-3.3-70b',
        provider='groq',
        prompt_tokens=10,
        completion_tokens=20,
        latency_ms=150.0,
    )
    assert res.provider == 'groq'
