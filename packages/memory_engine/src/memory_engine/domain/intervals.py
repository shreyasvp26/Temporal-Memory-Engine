"""Time interval and time point value objects."""

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from memory_engine.domain.enums import TimePrecision
from memory_engine.domain.errors import InvariantViolation


def _validate_utc_datetime(dt: datetime, name: str) -> None:
    if not isinstance(dt, datetime):
        raise InvariantViolation(f'{name} must be a datetime, got {type(dt)!r}')
    if dt.tzinfo is None or dt.utcoffset() != timedelta(0):
        raise InvariantViolation(f'{name} must be timezone-aware UTC, got {dt!r}')


@dataclass(frozen=True, slots=True)
class TimeInterval:
    """Half-open time interval [start, end).

    start: inclusive lower bound, or None for unbounded past (-inf).
    end: exclusive upper bound, or None for unbounded future (+inf).
    """

    start: datetime | None = None
    end: datetime | None = None

    def __post_init__(self) -> None:
        """Validate invariant I-1 (UTC awareness) and I-2 (start < end)."""
        if self.start is not None:
            _validate_utc_datetime(self.start, 'start')
        if self.end is not None:
            _validate_utc_datetime(self.end, 'end')
        if self.start is not None and self.end is not None and self.start >= self.end:
            raise InvariantViolation(
                f'start must be strictly less than end, got '
                f'start={self.start.isoformat()}, end={self.end.isoformat()}'
            )

    def contains(self, t: datetime) -> bool:
        """Return True if t falls within the half-open interval [start, end)."""
        _validate_utc_datetime(t, 't')
        lower_ok = self.start is None or t >= self.start
        upper_ok = self.end is None or t < self.end
        return lower_ok and upper_ok

    def overlaps(self, other: 'TimeInterval') -> bool:
        """Return True if self and other have a non-empty half-open intersection."""
        if not isinstance(other, TimeInterval):
            raise InvariantViolation(f'other must be a TimeInterval, got {type(other)!r}')
        disjoint_left = self.start is not None and other.end is not None and self.start >= other.end
        disjoint_right = (
            other.start is not None and self.end is not None and other.start >= self.end
        )
        return not (disjoint_left or disjoint_right)

    def is_open(self) -> bool:
        """Return True if the interval has no upper bound (end is None)."""
        return self.end is None


@dataclass(frozen=True, slots=True)
class TimePoint:
    """Time point with a specific granularity/precision."""

    value: datetime
    precision: TimePrecision

    def __post_init__(self) -> None:
        """Enforce invariant I-1 (UTC awareness) and validate precision."""
        _validate_utc_datetime(self.value, 'value')
        if isinstance(self.precision, str) and not isinstance(self.precision, TimePrecision):
            try:
                object.__setattr__(self, 'precision', TimePrecision(self.precision))
            except ValueError as err:
                raise InvariantViolation(f'Invalid precision: {self.precision!r}') from err
        elif not isinstance(self.precision, TimePrecision):
            raise InvariantViolation(f'Invalid precision: {self.precision!r}')

    def span_end(self) -> datetime:
        """Return exclusive UTC end of the period.

        DAY: +1 day
        MONTH: 1st of next month at 00:00 UTC
        YEAR: Jan 1 of next year at 00:00 UTC
        EXACT: +1 second
        """
        match self.precision:
            case TimePrecision.DAY:
                return self.value + timedelta(days=1)
            case TimePrecision.MONTH:
                if self.value.month == 12:
                    return datetime(self.value.year + 1, 1, 1, 0, 0, 0, tzinfo=UTC)
                return datetime(self.value.year, self.value.month + 1, 1, 0, 0, 0, tzinfo=UTC)
            case TimePrecision.YEAR:
                return datetime(self.value.year + 1, 1, 1, 0, 0, 0, tzinfo=UTC)
            case TimePrecision.EXACT:
                return self.value + timedelta(seconds=1)
            case _:
                raise InvariantViolation(f'Unsupported precision: {self.precision!r}')
