"""Resolution plan, operation commands, and strategy abstract base class."""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum
from typing import ClassVar

from memory_engine.domain.enums import Cardinality, FactStatus, RelationType
from memory_engine.domain.errors import InvariantViolation
from memory_engine.domain.ids import FactId
from memory_engine.domain.models import Candidate, Fact, JudgeHint


def _validate_utc_datetime(dt: datetime, name: str) -> None:
    """Validate that dt is timezone-aware UTC."""
    if not isinstance(dt, datetime):
        raise InvariantViolation(f'{name} must be a datetime, got {type(dt)!r}')
    if dt.tzinfo is None or dt.utcoffset() != timedelta(0):
        raise InvariantViolation(f'{name} must be timezone-aware UTC, got {dt!r}')


@dataclass(frozen=True, slots=True)
class NewRef:
    """Placeholder for a newly inserted fact whose ID is assigned during execution."""


FactRef = FactId | NewRef


@dataclass(frozen=True, slots=True)
class InsertFact:
    """Command to insert a new fact with given temporal boundaries and status."""

    valid_from: datetime
    valid_to: datetime | None
    status: FactStatus

    def __post_init__(self) -> None:
        _validate_utc_datetime(self.valid_from, 'valid_from')
        if self.valid_to is not None:
            _validate_utc_datetime(self.valid_to, 'valid_to')
            if self.valid_to <= self.valid_from:
                raise InvariantViolation(
                    f'valid_to must be strictly greater than valid_from, '
                    f'got valid_from={self.valid_from.isoformat()}, '
                    f'valid_to={self.valid_to.isoformat()}'
                )


@dataclass(frozen=True, slots=True)
class CloseFact:
    """Command to set valid_to on an open existing fact."""

    fact_id: FactId
    valid_to: datetime

    def __post_init__(self) -> None:
        _validate_utc_datetime(self.valid_to, 'valid_to')


@dataclass(frozen=True, slots=True)
class ExtendValidFrom:
    """Command to move valid_from of an existing fact to an earlier timestamp."""

    fact_id: FactId
    valid_from: datetime

    def __post_init__(self) -> None:
        _validate_utc_datetime(self.valid_from, 'valid_from')


@dataclass(frozen=True, slots=True)
class SetStatus:
    """Command to update status of an existing fact."""

    fact_id: FactId
    status: FactStatus


@dataclass(frozen=True, slots=True)
class AddEvidence:
    """Command to record supporting evidence for a fact reference."""

    ref: FactRef


@dataclass(frozen=True, slots=True)
class BumpConfidence:
    """Command to bump confidence score on an existing fact."""

    fact_id: FactId
    confidence: float

    def __post_init__(self) -> None:
        if not (0.0 <= self.confidence <= 1.0):
            raise InvariantViolation(
                f'confidence must be between 0.0 and 1.0, got {self.confidence}'
            )


@dataclass(frozen=True, slots=True)
class AddRelation:
    """Command to create a typed relationship edge between two fact references."""

    from_ref: FactRef
    to_ref: FactRef
    relation: RelationType


Op = (
    InsertFact
    | CloseFact
    | ExtendValidFrom
    | SetStatus
    | AddEvidence
    | BumpConfidence
    | AddRelation
)


class Outcome(StrEnum):
    """Result category of a conflict resolution plan."""

    INSERTED = 'inserted'
    SUPERSEDED = 'superseded'
    MERGED = 'merged'
    BACKFILLED = 'backfilled'
    REVIEW = 'review'
    RETRACTED = 'retracted'
    NOOP = 'noop'


def _op_sort_key(op: Op) -> int:
    """Canonical ordering key for plan operations.

    In every plan, CloseFact and ExtendValidFrom ops come before InsertFact.
    AddEvidence and AddRelation come last.
    """
    if isinstance(op, (CloseFact, ExtendValidFrom, SetStatus)):
        return 0
    if isinstance(op, BumpConfidence):
        return 1
    if isinstance(op, InsertFact):
        return 2
    if isinstance(op, AddEvidence):
        return 3
    if isinstance(op, AddRelation):
        return 4
    return 5


@dataclass(frozen=True, slots=True)
class ResolutionPlan:
    """Sequence of operations produced by a ResolutionStrategy."""

    ops: tuple[Op, ...]
    outcome: Outcome
    reason: str

    def __post_init__(self) -> None:
        ordered = tuple(sorted(self.ops, key=_op_sort_key))
        object.__setattr__(self, 'ops', ordered)


@dataclass(frozen=True, slots=True)
class ResolutionContext:
    """Context parameters for candidate conflict resolution."""

    now: datetime
    merge_gap: timedelta = timedelta(days=1)

    def __post_init__(self) -> None:
        _validate_utc_datetime(self.now, 'now')
        if self.merge_gap <= timedelta(0):
            raise InvariantViolation('merge_gap must be strictly positive')


class ResolutionStrategy(ABC):
    """Abstract base class for cardinality-specific conflict resolution strategies."""

    cardinality: ClassVar[Cardinality]

    @abstractmethod
    def plan(
        self,
        c: Candidate,
        timeline: Sequence[Fact],
        ctx: ResolutionContext,
        hint: JudgeHint | None = None,
    ) -> ResolutionPlan:
        """Compute an execution plan to resolve the candidate against the existing timeline."""
