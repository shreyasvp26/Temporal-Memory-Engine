"""Unit tests for domain entity dataclasses, boundary models, and invariants."""

from dataclasses import FrozenInstanceError
from datetime import UTC, datetime, timedelta, timezone
from typing import Any, Literal, cast
from uuid import UUID, uuid4

import pytest
from memory_engine.domain.enums import (
    Cardinality,
    DecayClass,
    EntityType,
    FactOrigin,
    FactStatus,
    JobStatus,
    JobType,
    ObjectKind,
    RelationType,
    Role,
    SourceKind,
    TimePrecision,
)
from memory_engine.domain.errors import InvariantViolation
from memory_engine.domain.ids import (
    EntityId,
    FactId,
    JobId,
    SourceId,
    UserId,
)
from memory_engine.domain.models import (
    Candidate,
    Entity,
    ExtractedFact,
    ExtractionResult,
    Fact,
    FactEvidence,
    FactRelation,
    Job,
    JudgeHint,
    Message,
    Predicate,
    Source,
)
from pydantic import ValidationError


def _utc(
    year: int,
    month: int,
    day: int,
    hour: int = 0,
    minute: int = 0,
    second: int = 0,
    microsecond: int = 0,
) -> datetime:
    """Helper creating timezone-aware UTC datetime."""
    return datetime(year, month, day, hour, minute, second, microsecond, tzinfo=UTC)


def _sample_user_id() -> UserId:
    return UserId(UUID('11111111-1111-1111-1111-111111111111'))


def _sample_entity_id() -> EntityId:
    return EntityId(UUID('22222222-2222-2222-2222-222222222222'))


def _sample_fact_id() -> FactId:
    return FactId(UUID('33333333-3333-3333-3333-333333333333'))


def _sample_source_id() -> SourceId:
    return SourceId(UUID('44444444-4444-4444-4444-444444444444'))


def _sample_job_id() -> JobId:
    return JobId(UUID('55555555-5555-5555-5555-555555555555'))


def _make_valid_fact(
    *,
    fact_id: FactId | None = None,
    user_id: UserId | None = None,
    subject_id: EntityId | None = None,
    predicate: str = 'lives_in',
    cardinality: Cardinality = Cardinality.SINGLE,
    object_entity_id: EntityId | None = None,
    object_value: str | None = 'Berlin',
    object_norm: str = 'berlin',
    fact_text: str = 'User lives in Berlin',
    valid_from: datetime | None = None,
    valid_to: datetime | None = None,
    time_precision: TimePrecision = TimePrecision.DAY,
    valid_from_explicit: bool = True,
    recorded_at: datetime | None = None,
    invalidated_at: datetime | None = None,
    confidence: float = 0.95,
    salience: float = 1.0,
    access_count: int = 0,
    last_accessed_at: datetime | None = None,
    status: FactStatus = FactStatus.ACTIVE,
    origin: FactOrigin = FactOrigin.USER_EDIT,
    event_hash: str | None = None,
) -> Fact:
    """Helper creating a valid Fact instance with default UTC timestamps."""
    return Fact(
        id=fact_id or _sample_fact_id(),
        user_id=user_id or _sample_user_id(),
        subject_id=subject_id or _sample_entity_id(),
        predicate=predicate,
        cardinality=cardinality,
        object_entity_id=object_entity_id,
        object_value=object_value,
        object_norm=object_norm,
        fact_text=fact_text,
        valid_from=valid_from or _utc(2026, 1, 1),
        valid_to=valid_to,
        time_precision=time_precision,
        valid_from_explicit=valid_from_explicit,
        recorded_at=recorded_at or _utc(2026, 1, 2),
        invalidated_at=invalidated_at,
        confidence=confidence,
        salience=salience,
        access_count=access_count,
        last_accessed_at=last_accessed_at,
        status=status,
        origin=origin,
        event_hash=event_hash,
    )


def _make_valid_candidate(
    *,
    user_id: UserId | None = None,
    subject_id: EntityId | None = None,
    predicate: str = 'lives_in',
    cardinality: Cardinality = Cardinality.SINGLE,
    object_entity_id: EntityId | None = None,
    object_value: str | None = 'Berlin',
    object_norm: str = 'berlin',
    fact_text: str = 'User lives in Berlin',
    valid_from: datetime | None = None,
    valid_to: datetime | None = None,
    time_precision: TimePrecision = TimePrecision.DAY,
    valid_from_explicit: bool = True,
    confidence: float = 0.9,
    is_retraction: bool = False,
    source_id: SourceId | None = None,
    span: tuple[int, int] | None = (0, 20),
    source_ts: datetime | None = None,
    event_hash: str | None = None,
    embedding: list[float] | None = None,
    hint: JudgeHint | None = None,
) -> Candidate:
    """Helper creating a valid Candidate instance."""
    return Candidate(
        user_id=user_id or _sample_user_id(),
        subject_id=subject_id or _sample_entity_id(),
        predicate=predicate,
        cardinality=cardinality,
        object_entity_id=object_entity_id,
        object_value=object_value,
        object_norm=object_norm,
        fact_text=fact_text,
        valid_from=valid_from or _utc(2026, 1, 1),
        valid_to=valid_to,
        time_precision=time_precision,
        valid_from_explicit=valid_from_explicit,
        confidence=confidence,
        is_retraction=is_retraction,
        source_id=source_id or _sample_source_id(),
        span=span,
        source_ts=source_ts or _utc(2026, 1, 2),
        event_hash=event_hash,
        embedding=embedding if embedding is not None else [0.1, 0.2, 0.3],
        hint=hint,
    )


# ============================================================================
# Valid entity construction tests
# ============================================================================


def test_source_valid_construction() -> None:
    """Verify Source entity constructs properly and respects immutability."""
    source_ts = _utc(2026, 3, 1, 12, 0)
    source = Source(
        id=_sample_source_id(),
        user_id=_sample_user_id(),
        kind=SourceKind.CHAT,
        role=Role.USER,
        session_id='session-123',
        turn_index=0,
        content='I moved to Berlin.',
        source_ts=source_ts,
        content_hash='hash123',
        deleted_at=None,
    )
    assert source.id == _sample_source_id()
    assert source.role == Role.USER
    assert source.turn_index == 0
    assert source.deleted_at is None

    with pytest.raises(FrozenInstanceError):
        setattr(source, 'content', 'changed')  # noqa: B010


def test_entity_valid_construction() -> None:
    """Verify Entity constructs properly."""
    ent = Entity(
        id=_sample_entity_id(),
        user_id=_sample_user_id(),
        canonical_name='Shreyas',
        type=EntityType.PERSON,
        is_self=True,
    )
    assert ent.canonical_name == 'Shreyas'
    assert ent.type == EntityType.PERSON
    assert ent.is_self is True


def test_predicate_valid_construction() -> None:
    """Verify Predicate constructs properly."""
    pred = Predicate(
        name='lives_in',
        cardinality=Cardinality.SINGLE,
        decay_class=DecayClass.SLOW,
        object_kind=ObjectKind.ENTITY,
        template='{subject} lives in {object}',
        synonyms=('resides_in', 'dwells_in'),
    )
    assert pred.name == 'lives_in'
    assert pred.synonyms == ('resides_in', 'dwells_in')


def test_fact_evidence_valid_construction() -> None:
    """Verify FactEvidence constructs properly."""
    fe = FactEvidence(
        fact_id=_sample_fact_id(),
        source_id=_sample_source_id(),
        span_start=10,
        span_end=25,
    )
    assert fe.fact_id == _sample_fact_id()
    assert fe.span_start == 10
    assert fe.span_end == 25


def test_fact_relation_valid_construction() -> None:
    """Verify FactRelation constructs properly."""
    f1 = FactId(uuid4())
    f2 = FactId(uuid4())
    fr = FactRelation(
        from_fact_id=f1,
        to_fact_id=f2,
        relation=RelationType.SUPERSEDES,
    )
    assert fr.relation == RelationType.SUPERSEDES
    assert fr.from_fact_id == f1
    assert fr.to_fact_id == f2


def test_job_valid_construction() -> None:
    """Verify Job constructs properly."""
    worker = uuid4()
    job = Job(
        id=_sample_job_id(),
        user_id=_sample_user_id(),
        type=JobType.EXTRACT,
        payload={'source_id': str(_sample_source_id())},
        idempotency_key='idem-1',
        coalesce_key='coalesce-1',
        status=JobStatus.QUEUED,
        attempts=0,
        max_attempts=3,
        deferrals=0,
        run_after=_utc(2026, 3, 1),
        worker_id=worker,
        lease_expires_at=_utc(2026, 3, 1, 1),
        last_error=None,
    )
    assert job.status == JobStatus.QUEUED
    assert job.worker_id == worker


def test_message_valid_construction() -> None:
    """Verify Message constructs properly."""
    msg_id = uuid4()
    conv_id = uuid4()
    msg = Message(
        id=msg_id,
        conversation_id=conv_id,
        user_id=_sample_user_id(),
        role=Role.USER,
        content='Hello assistant',
        client_message_id='cli-1',
        source_id=_sample_source_id(),
        partial=False,
        created_at=_utc(2026, 3, 1),
    )
    assert msg.id == msg_id
    assert msg.conversation_id == conv_id
    assert msg.content == 'Hello assistant'


def test_judge_hint_valid_construction() -> None:
    """Verify JudgeHint accepts valid verdicts and rejects invalid ones."""
    verdicts: tuple[Literal['same', 'supersedes', 'coexists'], ...] = (
        'same',
        'supersedes',
        'coexists',
    )
    for verdict in verdicts:
        hint = JudgeHint(target_fact_id=_sample_fact_id(), verdict=verdict)
        assert hint.verdict == verdict

    with pytest.raises(InvariantViolation, match='verdict must be one of'):
        JudgeHint(target_fact_id=_sample_fact_id(), verdict=cast('Any', 'invalid_verdict'))


def test_candidate_valid_construction() -> None:
    """Verify Candidate constructs properly."""
    candidate = _make_valid_candidate()
    assert candidate.predicate == 'lives_in'
    assert candidate.object_value == 'Berlin'
    assert candidate.hint is None


# ============================================================================
# Fact.is_current and Fact.interval tests
# ============================================================================


def test_fact_is_current() -> None:
    """Verify is_current logic: valid_to is None and invalidated_at is None and status == ACTIVE."""
    # Current fact: open, not invalidated, active
    fact = _make_valid_fact(valid_to=None, invalidated_at=None, status=FactStatus.ACTIVE)
    assert fact.is_current is True

    # Closed by valid_to
    closed_fact = _make_valid_fact(
        valid_from=_utc(2025, 1, 1),
        valid_to=_utc(2026, 1, 1),
        invalidated_at=None,
        status=FactStatus.ACTIVE,
    )
    assert closed_fact.is_current is False

    # Invalidated
    invalidated_fact = _make_valid_fact(
        valid_to=None,
        invalidated_at=_utc(2026, 2, 1),
        status=FactStatus.ACTIVE,
    )
    assert invalidated_fact.is_current is False

    # Archived status
    archived_fact = _make_valid_fact(
        valid_to=None,
        invalidated_at=None,
        status=FactStatus.ARCHIVED,
    )
    assert archived_fact.is_current is False

    # Deleted status
    deleted_fact = _make_valid_fact(
        valid_to=None,
        invalidated_at=None,
        status=FactStatus.DELETED,
    )
    assert deleted_fact.is_current is False


def test_fact_interval() -> None:
    """Verify Fact.interval() returns expected TimeInterval."""
    start = _utc(2026, 1, 1)
    end = _utc(2026, 6, 1)
    fact = _make_valid_fact(valid_from=start, valid_to=end)
    interval = fact.interval()

    assert interval.start == start
    assert interval.end == end
    assert interval.contains(_utc(2026, 3, 1))
    assert not interval.contains(_utc(2026, 7, 1))


# ============================================================================
# Invariant I-1: All datetimes timezone-aware UTC
# ============================================================================


def test_invariant_i1_fact_naive_datetimes() -> None:
    """Verify naive datetimes trigger InvariantViolation on Fact."""
    naive_dt = datetime(2026, 1, 1, 0, 0, 0)

    with pytest.raises(InvariantViolation, match='valid_from must be timezone-aware UTC'):
        _make_valid_fact(valid_from=naive_dt)

    with pytest.raises(InvariantViolation, match='valid_to must be timezone-aware UTC'):
        _make_valid_fact(valid_to=naive_dt)

    with pytest.raises(InvariantViolation, match='recorded_at must be timezone-aware UTC'):
        _make_valid_fact(recorded_at=naive_dt)

    with pytest.raises(InvariantViolation, match='invalidated_at must be timezone-aware UTC'):
        _make_valid_fact(invalidated_at=naive_dt)

    with pytest.raises(InvariantViolation, match='last_accessed_at must be timezone-aware UTC'):
        _make_valid_fact(last_accessed_at=naive_dt)


def test_invariant_i1_fact_non_utc_datetimes() -> None:
    """Verify non-UTC timezone-aware datetimes trigger InvariantViolation on Fact."""
    ist = timezone(timedelta(hours=5, minutes=30))
    non_utc_dt = datetime(2026, 1, 1, 5, 30, tzinfo=ist)

    with pytest.raises(InvariantViolation, match='valid_from must be timezone-aware UTC'):
        _make_valid_fact(valid_from=non_utc_dt)

    with pytest.raises(InvariantViolation, match='valid_to must be timezone-aware UTC'):
        _make_valid_fact(valid_to=non_utc_dt)


def test_invariant_i1_other_entities() -> None:
    """Verify UTC enforcement on Source, Job, Message, Candidate."""
    naive_dt = datetime(2026, 1, 1, 0, 0, 0)

    with pytest.raises(InvariantViolation, match='source_ts must be timezone-aware UTC'):
        Source(
            id=_sample_source_id(),
            user_id=_sample_user_id(),
            kind=SourceKind.CHAT,
            role=Role.USER,
            session_id='s',
            turn_index=0,
            content='test',
            source_ts=naive_dt,
            content_hash='h',
        )

    with pytest.raises(InvariantViolation, match='run_after must be timezone-aware UTC'):
        Job(
            id=_sample_job_id(),
            user_id=_sample_user_id(),
            type=JobType.EXTRACT,
            payload={},
            idempotency_key='k',
            coalesce_key=None,
            status=JobStatus.QUEUED,
            attempts=0,
            max_attempts=3,
            deferrals=0,
            run_after=naive_dt,
        )

    with pytest.raises(InvariantViolation, match='created_at must be timezone-aware UTC'):
        Message(
            id=uuid4(),
            conversation_id=uuid4(),
            user_id=_sample_user_id(),
            role=Role.USER,
            content='hi',
            client_message_id=None,
            source_id=None,
            partial=False,
            created_at=naive_dt,
        )

    with pytest.raises(InvariantViolation, match='valid_from must be timezone-aware UTC'):
        _make_valid_candidate(valid_from=naive_dt)


# ============================================================================
# Invariant I-2: valid_to is None or valid_to > valid_from
# ============================================================================


def test_invariant_i2_valid_to_greater_than_valid_from() -> None:
    """Verify valid_to <= valid_from triggers InvariantViolation on Fact and Candidate."""
    t1 = _utc(2026, 5, 1)
    t2 = _utc(2026, 4, 1)  # earlier than t1

    # valid_to < valid_from
    with pytest.raises(InvariantViolation, match='valid_to must be strictly greater than'):
        _make_valid_fact(valid_from=t1, valid_to=t2)

    # valid_to == valid_from
    with pytest.raises(InvariantViolation, match='valid_to must be strictly greater than'):
        _make_valid_fact(valid_from=t1, valid_to=t1)

    # Candidate check
    with pytest.raises(InvariantViolation, match='valid_to must be strictly greater than'):
        _make_valid_candidate(valid_from=t1, valid_to=t2)

    # valid_to > valid_from succeeds
    fact = _make_valid_fact(valid_from=t1, valid_to=_utc(2026, 6, 1))
    assert fact.valid_to == _utc(2026, 6, 1)


# ============================================================================
# Invariant I-3: Exactly one of object_entity_id or object_value is set
# ============================================================================


def test_invariant_i3_object_entity_xor_value() -> None:
    """Verify exactly one of object_entity_id or object_value must be non-None."""
    # Neither set
    with pytest.raises(
        InvariantViolation, match='Exactly one of object_entity_id or object_value must be set'
    ):
        _make_valid_fact(object_entity_id=None, object_value=None)

    # Both set
    with pytest.raises(
        InvariantViolation, match='Exactly one of object_entity_id or object_value must be set'
    ):
        _make_valid_fact(object_entity_id=_sample_entity_id(), object_value='Berlin')

    # Only object_entity_id set -> succeeds
    fact_ent = _make_valid_fact(object_entity_id=_sample_entity_id(), object_value=None)
    assert fact_ent.object_entity_id == _sample_entity_id()
    assert fact_ent.object_value is None

    # Only object_value set -> succeeds
    fact_val = _make_valid_fact(object_entity_id=None, object_value='Berlin')
    assert fact_val.object_entity_id is None
    assert fact_val.object_value == 'Berlin'

    # Candidate check
    with pytest.raises(
        InvariantViolation, match='Exactly one of object_entity_id or object_value must be set'
    ):
        _make_valid_candidate(object_entity_id=None, object_value=None)

    with pytest.raises(
        InvariantViolation, match='Exactly one of object_entity_id or object_value must be set'
    ):
        _make_valid_candidate(object_entity_id=_sample_entity_id(), object_value='Berlin')


# ============================================================================
# Invariant I-5: Event facts always have a finite valid_to
# ============================================================================


def test_invariant_i5_event_facts_finite_valid_to() -> None:
    """Verify Cardinality.EVENT requires a finite valid_to on Fact and Candidate."""
    # Fact with Cardinality.EVENT and valid_to=None -> raises
    with pytest.raises(InvariantViolation, match='Event facts must have a finite valid_to'):
        _make_valid_fact(cardinality=Cardinality.EVENT, valid_to=None)

    # Fact with Cardinality.EVENT and finite valid_to -> succeeds
    event_fact = _make_valid_fact(
        cardinality=Cardinality.EVENT,
        valid_from=_utc(2026, 1, 1, 10, 0),
        valid_to=_utc(2026, 1, 1, 12, 0),
    )
    assert event_fact.valid_to == _utc(2026, 1, 1, 12, 0)

    # Candidate check
    with pytest.raises(InvariantViolation, match='Event facts must have a finite valid_to'):
        _make_valid_candidate(cardinality=Cardinality.EVENT, valid_to=None)


# ============================================================================
# Invariant I-7: confidence in [0.0, 1.0]
# ============================================================================


def test_invariant_i7_confidence_bounds() -> None:
    """Verify confidence must be in range [0.0, 1.0]."""
    # Negative confidence
    with pytest.raises(InvariantViolation, match='confidence must be between 0.0 and 1.0'):
        _make_valid_fact(confidence=-0.001)

    # > 1.0 confidence
    with pytest.raises(InvariantViolation, match='confidence must be between 0.0 and 1.0'):
        _make_valid_fact(confidence=1.001)

    # NaN confidence
    with pytest.raises(InvariantViolation, match='confidence must be between 0.0 and 1.0'):
        _make_valid_fact(confidence=float('nan'))

    # Boundaries 0.0 and 1.0 succeed
    f0 = _make_valid_fact(confidence=0.0)
    assert f0.confidence == 0.0

    f1 = _make_valid_fact(confidence=1.0)
    assert f1.confidence == 1.0

    # Candidate check
    with pytest.raises(InvariantViolation, match='confidence must be between 0.0 and 1.0'):
        _make_valid_candidate(confidence=-0.1)

    with pytest.raises(InvariantViolation, match='confidence must be between 0.0 and 1.0'):
        _make_valid_candidate(confidence=1.5)


# ============================================================================
# ExtractedFact & ExtractionResult Pydantic Boundary Tests
# ============================================================================


def test_extracted_fact_valid_construction() -> None:
    """Verify ExtractedFact initializes properly with default values."""
    ef = ExtractedFact(
        subject='I',
        predicate='lives_in',
        object='Berlin',
        confidence=0.9,
        evidence_span='I live in Berlin',
        source_turn=1,
    )
    assert ef.subject == 'I'
    assert ef.subject_type == EntityType.PERSON
    assert ef.predicate == 'lives_in'
    assert ef.object == 'Berlin'
    assert ef.object_type == 'literal'
    assert ef.valid_from_text is None
    assert ef.valid_to_text is None
    assert ef.is_retraction is False
    assert ef.confidence == 0.9
    assert ef.evidence_span == 'I live in Berlin'
    assert ef.source_turn == 1


def test_extracted_fact_frozen() -> None:
    """Verify ExtractedFact is frozen and rejects mutation."""
    ef = ExtractedFact(
        subject='I',
        predicate='lives_in',
        object='Berlin',
        confidence=0.9,
        evidence_span='I live in Berlin',
        source_turn=1,
    )
    with pytest.raises(ValidationError):
        setattr(ef, 'subject', 'Someone else')  # noqa: B010


def test_extracted_fact_extra_forbid() -> None:
    """Verify extra attributes are forbidden on ExtractedFact."""
    with pytest.raises(ValidationError):
        ExtractedFact.model_validate(
            {
                'subject': 'I',
                'predicate': 'lives_in',
                'object': 'Berlin',
                'confidence': 0.9,
                'evidence_span': 'I live in Berlin',
                'source_turn': 1,
                'unexpected_field': 'not allowed',
            }
        )


def test_extracted_fact_field_boundaries() -> None:
    """Verify length and numeric constraints on ExtractedFact fields."""
    # subject min_length=1, max_length=120
    with pytest.raises(ValidationError):
        ExtractedFact(
            subject='',
            predicate='lives_in',
            object='Berlin',
            confidence=0.9,
            evidence_span='test span',
            source_turn=0,
        )

    with pytest.raises(ValidationError):
        ExtractedFact(
            subject='x' * 121,
            predicate='lives_in',
            object='Berlin',
            confidence=0.9,
            evidence_span='test span',
            source_turn=0,
        )

    # predicate min_length=1, max_length=64
    with pytest.raises(ValidationError):
        ExtractedFact(
            subject='I',
            predicate='',
            object='Berlin',
            confidence=0.9,
            evidence_span='test span',
            source_turn=0,
        )

    with pytest.raises(ValidationError):
        ExtractedFact(
            subject='I',
            predicate='p' * 65,
            object='Berlin',
            confidence=0.9,
            evidence_span='test span',
            source_turn=0,
        )

    # object min_length=1, max_length=300
    with pytest.raises(ValidationError):
        ExtractedFact(
            subject='I',
            predicate='lives_in',
            object='',
            confidence=0.9,
            evidence_span='test span',
            source_turn=0,
        )

    with pytest.raises(ValidationError):
        ExtractedFact(
            subject='I',
            predicate='lives_in',
            object='o' * 301,
            confidence=0.9,
            evidence_span='test span',
            source_turn=0,
        )

    # evidence_span min_length=1, max_length=400
    with pytest.raises(ValidationError):
        ExtractedFact(
            subject='I',
            predicate='lives_in',
            object='Berlin',
            confidence=0.9,
            evidence_span='',
            source_turn=0,
        )

    with pytest.raises(ValidationError):
        ExtractedFact(
            subject='I',
            predicate='lives_in',
            object='Berlin',
            confidence=0.9,
            evidence_span='e' * 401,
            source_turn=0,
        )

    # confidence ge=0.0, le=1.0
    with pytest.raises(ValidationError):
        ExtractedFact(
            subject='I',
            predicate='lives_in',
            object='Berlin',
            confidence=-0.1,
            evidence_span='test span',
            source_turn=0,
        )

    with pytest.raises(ValidationError):
        ExtractedFact(
            subject='I',
            predicate='lives_in',
            object='Berlin',
            confidence=1.1,
            evidence_span='test span',
            source_turn=0,
        )


def test_extracted_fact_object_type_union() -> None:
    """Verify object_type supports EntityType members as well as 'literal'."""
    ef1 = ExtractedFact(
        subject='I',
        predicate='lives_in',
        object='Berlin',
        object_type=EntityType.PLACE,
        confidence=0.9,
        evidence_span='test span',
        source_turn=0,
    )
    assert ef1.object_type == EntityType.PLACE

    ef2 = ExtractedFact(
        subject='I',
        predicate='job_title',
        object='Engineer',
        object_type='literal',
        confidence=0.9,
        evidence_span='test span',
        source_turn=0,
    )
    assert ef2.object_type == 'literal'

    with pytest.raises(ValidationError):
        ExtractedFact.model_validate(
            {
                'subject': 'I',
                'predicate': 'job_title',
                'object': 'Engineer',
                'object_type': 'unsupported_type',
                'confidence': 0.9,
                'evidence_span': 'test span',
                'source_turn': 0,
            }
        )


def test_extraction_result_validation() -> None:
    """Verify ExtractionResult enforces max_length=30 on facts and extra='forbid'."""
    # Default empty list
    res_empty = ExtractionResult()
    assert res_empty.facts == []

    # Up to 30 facts succeed
    ef = ExtractedFact(
        subject='I',
        predicate='lives_in',
        object='Berlin',
        confidence=0.9,
        evidence_span='test span',
        source_turn=0,
    )
    res_30 = ExtractionResult(facts=[ef] * 30)
    assert len(res_30.facts) == 30

    # 31 facts fail max_length=30
    with pytest.raises(ValidationError):
        ExtractionResult(facts=[ef] * 31)

    # Extra fields forbidden
    with pytest.raises(ValidationError):
        ExtractionResult.model_validate({'extra_field': 'not allowed'})

    # Immutability
    with pytest.raises(ValidationError):
        setattr(res_30, 'facts', [])  # noqa: B010
