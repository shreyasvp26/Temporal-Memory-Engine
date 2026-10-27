"""Domain entity dataclasses and boundary models."""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

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
from memory_engine.domain.intervals import TimeInterval


def _validate_utc_datetime(dt: datetime, name: str) -> None:
    if not isinstance(dt, datetime):
        raise InvariantViolation(f'{name} must be a datetime, got {type(dt)!r}')
    if dt.tzinfo is None or dt.utcoffset() != timedelta(0):
        raise InvariantViolation(f'{name} must be timezone-aware UTC, got {dt!r}')


@dataclass(frozen=True, slots=True)
class Source:
    id: SourceId
    user_id: UserId
    kind: SourceKind
    role: Role
    session_id: str
    turn_index: int
    content: str
    source_ts: datetime
    content_hash: str
    deleted_at: datetime | None = None

    def __post_init__(self) -> None:
        _validate_utc_datetime(self.source_ts, 'source_ts')
        if self.deleted_at is not None:
            _validate_utc_datetime(self.deleted_at, 'deleted_at')


@dataclass(frozen=True, slots=True)
class Entity:
    id: EntityId
    user_id: UserId
    canonical_name: str
    type: EntityType
    is_self: bool


@dataclass(frozen=True, slots=True)
class Predicate:
    name: str
    cardinality: Cardinality
    decay_class: DecayClass
    object_kind: ObjectKind
    template: str
    synonyms: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class Fact:
    id: FactId
    user_id: UserId
    subject_id: EntityId
    predicate: str
    cardinality: Cardinality
    object_entity_id: EntityId | None
    object_value: str | None
    object_norm: str
    fact_text: str
    valid_from: datetime
    valid_to: datetime | None
    time_precision: TimePrecision
    valid_from_explicit: bool
    recorded_at: datetime
    invalidated_at: datetime | None
    confidence: float
    salience: float
    access_count: int
    last_accessed_at: datetime | None
    status: FactStatus
    origin: FactOrigin
    event_hash: str | None

    def __post_init__(self) -> None:
        # I-1: All datetimes timezone-aware UTC
        _validate_utc_datetime(self.valid_from, 'valid_from')
        if self.valid_to is not None:
            _validate_utc_datetime(self.valid_to, 'valid_to')
        _validate_utc_datetime(self.recorded_at, 'recorded_at')
        if self.invalidated_at is not None:
            _validate_utc_datetime(self.invalidated_at, 'invalidated_at')
        if self.last_accessed_at is not None:
            _validate_utc_datetime(self.last_accessed_at, 'last_accessed_at')

        # I-2: valid_to is None or greater than valid_from
        if self.valid_to is not None and self.valid_to <= self.valid_from:
            raise InvariantViolation(
                'valid_to must be strictly greater than valid_from, '
                f'got valid_from={self.valid_from.isoformat()}, '
                f'valid_to={self.valid_to.isoformat()}'
            )

        # I-3: Exactly one of object_entity_id or object_value is set
        has_entity = self.object_entity_id is not None
        has_value = self.object_value is not None
        if has_entity == has_value:
            raise InvariantViolation('Exactly one of object_entity_id or object_value must be set')

        # I-5: Event facts always have a finite valid_to
        if self.cardinality == Cardinality.EVENT and self.valid_to is None:
            raise InvariantViolation('Event facts must have a finite valid_to')

        # I-7: confidence in [0, 1]
        if not (0.0 <= self.confidence <= 1.0):
            raise InvariantViolation(
                f'confidence must be between 0.0 and 1.0, got {self.confidence}'
            )

    @property
    def is_current(self) -> bool:
        return (
            self.valid_to is None
            and self.invalidated_at is None
            and self.status == FactStatus.ACTIVE
        )

    def interval(self) -> TimeInterval:
        return TimeInterval(start=self.valid_from, end=self.valid_to)


@dataclass(frozen=True, slots=True)
class FactEvidence:
    fact_id: FactId
    source_id: SourceId
    span_start: int | None = None
    span_end: int | None = None


@dataclass(frozen=True, slots=True)
class FactRelation:
    from_fact_id: FactId
    to_fact_id: FactId
    relation: RelationType


@dataclass(frozen=True, slots=True)
class Job:
    id: JobId
    user_id: UserId
    type: JobType
    payload: dict[str, Any]
    idempotency_key: str
    coalesce_key: str | None
    status: JobStatus
    attempts: int
    max_attempts: int
    deferrals: int
    run_after: datetime
    worker_id: UUID | None = None
    lease_expires_at: datetime | None = None
    last_error: str | None = None

    def __post_init__(self) -> None:
        _validate_utc_datetime(self.run_after, 'run_after')
        if self.lease_expires_at is not None:
            _validate_utc_datetime(self.lease_expires_at, 'lease_expires_at')


@dataclass(frozen=True, slots=True)
class Message:
    id: UUID
    conversation_id: UUID
    user_id: UserId
    role: Role
    content: str
    client_message_id: str | None
    source_id: SourceId | None
    partial: bool
    created_at: datetime

    def __post_init__(self) -> None:
        _validate_utc_datetime(self.created_at, 'created_at')


class ExtractedFact(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)

    subject: str = Field(min_length=1, max_length=120)
    subject_type: EntityType = EntityType.PERSON
    predicate: str = Field(min_length=1, max_length=64)
    object: str = Field(min_length=1, max_length=300)
    object_type: EntityType | Literal['literal'] = 'literal'
    valid_from_text: str | None = None
    valid_to_text: str | None = None
    is_retraction: bool = False
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_span: str = Field(min_length=1, max_length=400)
    source_turn: int


class ExtractionResult(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)

    facts: list[ExtractedFact] = Field(default_factory=list, max_length=30)


@dataclass(frozen=True, slots=True)
class JudgeHint:
    target_fact_id: FactId
    verdict: Literal['same', 'supersedes', 'coexists']

    def __post_init__(self) -> None:
        if self.verdict not in ('same', 'supersedes', 'coexists'):
            raise InvariantViolation(
                f'verdict must be one of "same", "supersedes", "coexists", got {self.verdict!r}'
            )


@dataclass(frozen=True, slots=True)
class Candidate:
    user_id: UserId
    subject_id: EntityId
    predicate: str
    cardinality: Cardinality
    object_entity_id: EntityId | None
    object_value: str | None
    object_norm: str
    fact_text: str
    valid_from: datetime
    valid_to: datetime | None
    time_precision: TimePrecision
    valid_from_explicit: bool
    confidence: float
    is_retraction: bool
    source_id: SourceId
    span: tuple[int, int] | None
    source_ts: datetime
    event_hash: str | None
    embedding: list[float]
    hint: JudgeHint | None = None

    def __post_init__(self) -> None:
        _validate_utc_datetime(self.valid_from, 'valid_from')
        if self.valid_to is not None:
            _validate_utc_datetime(self.valid_to, 'valid_to')
            if self.valid_to <= self.valid_from:
                raise InvariantViolation(
                    'valid_to must be strictly greater than valid_from, '
                    f'got valid_from={self.valid_from.isoformat()}, '
                    f'valid_to={self.valid_to.isoformat()}'
                )
        _validate_utc_datetime(self.source_ts, 'source_ts')
        has_entity = self.object_entity_id is not None
        has_value = self.object_value is not None
        if has_entity == has_value:
            raise InvariantViolation('Exactly one of object_entity_id or object_value must be set')
        if self.cardinality == Cardinality.EVENT and self.valid_to is None:
            raise InvariantViolation('Event facts must have a finite valid_to')
        if not (0.0 <= self.confidence <= 1.0):
            raise InvariantViolation(
                f'confidence must be between 0.0 and 1.0, got {self.confidence}'
            )
