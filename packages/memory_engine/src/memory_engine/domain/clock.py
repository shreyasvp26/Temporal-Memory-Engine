"""Clock protocol and implementations in domain."""

from datetime import UTC, datetime, timedelta
from typing import Protocol, runtime_checkable

from memory_engine.domain.errors import InvariantViolation


def _validate_utc_datetime(dt: datetime, name: str) -> None:
    if not isinstance(dt, datetime):
        raise InvariantViolation(f'{name} must be a datetime, got {type(dt)!r}')
    if dt.tzinfo is None or dt.utcoffset() != timedelta(0):
        raise InvariantViolation(f'{name} must be timezone-aware UTC, got {dt!r}')


@runtime_checkable
class Clock(Protocol):
    """Protocol for providers of current UTC time."""

    def now(self) -> datetime:
        """Return the current time as timezone-aware UTC datetime."""
        ...


class SystemClock:
    """Production clock returning the real system time in UTC."""

    __slots__ = ()

    def now(self) -> datetime:
        """Return the current system time in UTC."""
        return datetime.now(UTC)


class FixedClock:
    """Deterministic clock for testing and evaluation."""

    __slots__ = ('_now',)

    def __init__(self, fixed_now: datetime) -> None:
        _validate_utc_datetime(fixed_now, 'fixed_now')
        self._now = fixed_now

    def now(self) -> datetime:
        """Return the current fixed time in UTC."""
        return self._now

    def advance(self, delta: timedelta) -> datetime:
        """Advance the clock by delta and return the new UTC time."""
        self._now = self._now + delta
        return self._now

    def set(self, new_now: datetime) -> None:
        """Set the clock to a specific UTC datetime."""
        _validate_utc_datetime(new_now, 'new_now')
        self._now = new_now
