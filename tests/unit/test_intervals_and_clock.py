"""Unit tests for TimeInterval, TimePoint, and Clock implementations."""

from datetime import UTC, datetime, timedelta, timezone

import pytest
from memory_engine.domain.clock import Clock, FixedClock, SystemClock
from memory_engine.domain.enums import TimePrecision
from memory_engine.domain.errors import InvariantViolation
from memory_engine.domain.intervals import TimeInterval, TimePoint
from memory_engine.ports.clock import (
    Clock as PortClock,
)
from memory_engine.ports.clock import (
    FixedClock as PortFixedClock,
)
from memory_engine.ports.clock import (
    SystemClock as PortSystemClock,
)


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


# ============================================================================
# TimeInterval tests
# ============================================================================


def test_time_interval_valid_construction() -> None:
    """Verify valid TimeInterval instances construct with expected attributes."""
    t1 = _utc(2026, 1, 1, 0, 0, 0)
    t2 = _utc(2026, 1, 2, 0, 0, 0)

    interval = TimeInterval(start=t1, end=t2)
    assert interval.start == t1
    assert interval.end == t2
    assert not interval.is_open()

    open_ended = TimeInterval(start=t1, end=None)
    assert open_ended.start == t1
    assert open_ended.end is None
    assert open_ended.is_open()

    open_start = TimeInterval(start=None, end=t2)
    assert open_start.start is None
    assert open_start.end == t2
    assert not open_start.is_open()

    fully_open = TimeInterval(start=None, end=None)
    assert fully_open.start is None
    assert fully_open.end is None
    assert fully_open.is_open()


def test_time_interval_naive_datetime_rejection() -> None:
    """Verify naive datetimes in start or end raise InvariantViolation."""
    naive_dt = datetime(2026, 1, 1, 0, 0, 0)
    aware_utc = _utc(2026, 1, 2, 0, 0, 0)

    with pytest.raises(InvariantViolation, match='start must be timezone-aware UTC'):
        TimeInterval(start=naive_dt, end=aware_utc)

    with pytest.raises(InvariantViolation, match='end must be timezone-aware UTC'):
        TimeInterval(start=aware_utc, end=naive_dt)

    with pytest.raises(InvariantViolation, match='start must be timezone-aware UTC'):
        TimeInterval(start=naive_dt, end=None)


def test_time_interval_non_utc_tz_rejection() -> None:
    """Verify non-UTC timezones in start or end raise InvariantViolation."""
    ist_tz = timezone(timedelta(hours=5, minutes=30))
    ist_dt = datetime(2026, 1, 1, 0, 0, 0, tzinfo=ist_tz)
    aware_utc = _utc(2026, 1, 2, 0, 0, 0)

    with pytest.raises(InvariantViolation, match='start must be timezone-aware UTC'):
        TimeInterval(start=ist_dt, end=aware_utc)

    with pytest.raises(InvariantViolation, match='end must be timezone-aware UTC'):
        TimeInterval(start=aware_utc, end=ist_dt)


def test_time_interval_invalid_types_rejected() -> None:
    """Verify non-datetime types raise InvariantViolation."""
    aware_utc = _utc(2026, 1, 2, 0, 0, 0)

    with pytest.raises(InvariantViolation, match='start must be a datetime'):
        TimeInterval(start='2026-01-01', end=aware_utc)  # type: ignore[arg-type]

    with pytest.raises(InvariantViolation, match='end must be a datetime'):
        TimeInterval(start=aware_utc, end=12345)  # type: ignore[arg-type]


def test_time_interval_start_greater_or_equal_to_end_rejection() -> None:
    """Verify start >= end raises InvariantViolation (Invariant I-2)."""
    t1 = _utc(2026, 1, 1, 12, 0, 0)
    t2 = _utc(2026, 1, 1, 13, 0, 0)

    # start == end must be rejected
    with pytest.raises(InvariantViolation, match='start must be strictly less than end'):
        TimeInterval(start=t1, end=t1)

    # start > end must be rejected
    with pytest.raises(InvariantViolation, match='start must be strictly less than end'):
        TimeInterval(start=t2, end=t1)


def test_time_interval_contains_half_open() -> None:
    """Verify contains obeys half-open [start, end) semantics."""
    t_start = _utc(2026, 1, 1, 10, 0, 0)
    t_end = _utc(2026, 1, 1, 12, 0, 0)
    interval = TimeInterval(start=t_start, end=t_end)

    # Exact start is included [start
    assert interval.contains(t_start)

    # Just after start is included
    assert interval.contains(t_start + timedelta(microseconds=1))

    # Midpoint is included
    assert interval.contains(_utc(2026, 1, 1, 11, 0, 0))

    # Just before end is included
    assert interval.contains(t_end - timedelta(microseconds=1))

    # Exact end is excluded end)
    assert not interval.contains(t_end)

    # Past end is excluded
    assert not interval.contains(t_end + timedelta(seconds=1))

    # Before start is excluded
    assert not interval.contains(t_start - timedelta(seconds=1))


def test_time_interval_contains_open_bounds() -> None:
    """Verify contains with unbounded past, future, or both."""
    t1 = _utc(2026, 1, 1, 0, 0, 0)
    t2 = _utc(2026, 6, 1, 0, 0, 0)

    # Open future [t1, None)
    open_future = TimeInterval(start=t1, end=None)
    assert not open_future.contains(_utc(2025, 12, 31, 23, 59, 59))
    assert open_future.contains(t1)
    assert open_future.contains(_utc(2099, 1, 1, 0, 0, 0))

    # Open past (None, t2)
    open_past = TimeInterval(start=None, end=t2)
    assert open_past.contains(_utc(1970, 1, 1, 0, 0, 0))
    assert open_past.contains(_utc(2026, 5, 31, 23, 59, 59))
    assert not open_past.contains(t2)
    assert not open_past.contains(_utc(2026, 6, 1, 0, 0, 1))

    # Unbounded (None, None)
    unbounded = TimeInterval(start=None, end=None)
    assert unbounded.contains(_utc(1900, 1, 1, 0, 0, 0))
    assert unbounded.contains(_utc(2026, 1, 1, 0, 0, 0))
    assert unbounded.contains(_utc(3000, 1, 1, 0, 0, 0))


def test_time_interval_contains_rejects_invalid_datetime() -> None:
    """Verify contains raises InvariantViolation for naive or non-UTC datetimes."""
    interval = TimeInterval(start=_utc(2026, 1, 1), end=_utc(2026, 1, 2))

    naive_dt = datetime(2026, 1, 1, 12, 0, 0)
    with pytest.raises(InvariantViolation, match='t must be timezone-aware UTC'):
        interval.contains(naive_dt)

    ist_tz = timezone(timedelta(hours=5, minutes=30))
    ist_dt = datetime(2026, 1, 1, 12, 0, 0, tzinfo=ist_tz)
    with pytest.raises(InvariantViolation, match='t must be timezone-aware UTC'):
        interval.contains(ist_dt)


def test_time_interval_overlaps_bounded() -> None:
    """Verify overlaps for bounded half-open intervals."""
    # [10:00, 11:00) and [11:00, 12:00) abut, so intersection is empty
    a = TimeInterval(_utc(2026, 1, 1, 10, 0, 0), _utc(2026, 1, 1, 11, 0, 0))
    b = TimeInterval(_utc(2026, 1, 1, 11, 0, 0), _utc(2026, 1, 1, 12, 0, 0))
    assert not a.overlaps(b)
    assert not b.overlaps(a)

    # Overlap by 1 microsecond: [10:00, 11:00:00.000001) and [11:00, 12:00)
    a_overlap = TimeInterval(_utc(2026, 1, 1, 10, 0, 0), _utc(2026, 1, 1, 11, 0, 0, 1))
    assert a_overlap.overlaps(b)
    assert b.overlaps(a_overlap)

    # Disjoint intervals
    c = TimeInterval(_utc(2026, 1, 1, 13, 0, 0), _utc(2026, 1, 1, 14, 0, 0))
    assert not a.overlaps(c)
    assert not c.overlaps(a)

    # Nested intervals
    outer = TimeInterval(_utc(2026, 1, 1, 9, 0, 0), _utc(2026, 1, 1, 17, 0, 0))
    inner = TimeInterval(_utc(2026, 1, 1, 12, 0, 0), _utc(2026, 1, 1, 13, 0, 0))
    assert outer.overlaps(inner)
    assert inner.overlaps(outer)

    # Identical intervals
    assert a.overlaps(a)


def test_time_interval_overlaps_unbounded() -> None:
    """Verify overlaps with unbounded intervals (None start and/or end)."""
    t10 = _utc(2026, 1, 1, 10, 0, 0)
    t11 = _utc(2026, 1, 1, 11, 0, 0)
    t12 = _utc(2026, 1, 1, 12, 0, 0)

    open_future_10 = TimeInterval(start=t10, end=None)
    open_future_12 = TimeInterval(start=t12, end=None)
    assert open_future_10.overlaps(open_future_12)
    assert open_future_12.overlaps(open_future_10)

    # [10, None) and [5, 10) abut at 10 -> no overlap
    t05 = _utc(2026, 1, 1, 5, 0, 0)
    bounded_05_10 = TimeInterval(start=t05, end=t10)
    assert not open_future_10.overlaps(bounded_05_10)
    assert not bounded_05_10.overlaps(open_future_10)

    # [10, None) and [5, 11) overlap in [10, 11)
    bounded_05_11 = TimeInterval(start=t05, end=t11)
    assert open_future_10.overlaps(bounded_05_11)
    assert bounded_05_11.overlaps(open_future_10)

    # Open past [None, 10) and [10, 12) abut at 10 -> no overlap
    open_past_10 = TimeInterval(start=None, end=t10)
    bounded_10_12 = TimeInterval(start=t10, end=t12)
    assert not open_past_10.overlaps(bounded_10_12)
    assert not bounded_10_12.overlaps(open_past_10)

    # Both open past [None, 10) and [None, 12) overlap in (-inf, 10)
    open_past_12 = TimeInterval(start=None, end=t12)
    assert open_past_10.overlaps(open_past_12)
    assert open_past_12.overlaps(open_past_10)

    # Unbounded (None, None) overlaps with any interval
    unbounded = TimeInterval(start=None, end=None)
    assert unbounded.overlaps(bounded_05_10)
    assert bounded_05_10.overlaps(unbounded)
    assert unbounded.overlaps(unbounded)


def test_time_interval_overlaps_type_check() -> None:
    """Verify overlaps raises InvariantViolation if other is not a TimeInterval."""
    interval = TimeInterval(_utc(2026, 1, 1), _utc(2026, 1, 2))
    with pytest.raises(InvariantViolation, match='other must be a TimeInterval'):
        interval.overlaps('not-an-interval')  # type: ignore[arg-type]


# ============================================================================
# TimePoint tests
# ============================================================================


def test_time_point_construction_and_tz_enforcement() -> None:
    """Verify TimePoint enforces UTC awareness and validates precision."""
    valid_dt = _utc(2026, 3, 15, 10, 0, 0)
    tp = TimePoint(value=valid_dt, precision=TimePrecision.DAY)
    assert tp.value == valid_dt
    assert tp.precision == TimePrecision.DAY

    # StrEnum string conversion accepted
    tp_str = TimePoint(value=valid_dt, precision='day')  # type: ignore[arg-type]
    assert tp_str.precision == TimePrecision.DAY

    # Naive datetime rejected
    naive_dt = datetime(2026, 3, 15, 10, 0, 0)
    with pytest.raises(InvariantViolation, match='value must be timezone-aware UTC'):
        TimePoint(value=naive_dt, precision=TimePrecision.DAY)

    # Non-UTC timezone rejected
    ist_tz = timezone(timedelta(hours=5, minutes=30))
    ist_dt = datetime(2026, 3, 15, 10, 0, 0, tzinfo=ist_tz)
    with pytest.raises(InvariantViolation, match='value must be timezone-aware UTC'):
        TimePoint(value=ist_dt, precision=TimePrecision.DAY)

    # Invalid precision rejected
    with pytest.raises(InvariantViolation, match='Invalid precision'):
        TimePoint(value=valid_dt, precision='century')  # type: ignore[arg-type]


def test_time_point_span_end_exact() -> None:
    """Verify EXACT precision adds exactly 1 second."""
    dt = _utc(2026, 10, 26, 12, 30, 45)
    tp = TimePoint(value=dt, precision=TimePrecision.EXACT)
    expected = _utc(2026, 10, 26, 12, 30, 46)
    assert tp.span_end() == expected


def test_time_point_span_end_day_regular_and_leap_year() -> None:
    """Verify DAY precision adds 1 day, handling leap year and month boundaries."""
    # Standard day
    tp1 = TimePoint(value=_utc(2026, 3, 15, 0, 0, 0), precision=TimePrecision.DAY)
    assert tp1.span_end() == _utc(2026, 3, 16, 0, 0, 0)

    # Leap year Feb 28 -> Feb 29 (2024 is a leap year)
    tp_leap1 = TimePoint(value=_utc(2024, 2, 28, 0, 0, 0), precision=TimePrecision.DAY)
    assert tp_leap1.span_end() == _utc(2024, 2, 29, 0, 0, 0)

    # Leap year Feb 29 -> Mar 1 (2024 is a leap year)
    tp_leap2 = TimePoint(value=_utc(2024, 2, 29, 0, 0, 0), precision=TimePrecision.DAY)
    assert tp_leap2.span_end() == _utc(2024, 3, 1, 0, 0, 0)

    # Non-leap year Feb 28 -> Mar 1 (2023 is not a leap year)
    tp_non_leap = TimePoint(value=_utc(2023, 2, 28, 0, 0, 0), precision=TimePrecision.DAY)
    assert tp_non_leap.span_end() == _utc(2023, 3, 1, 0, 0, 0)

    # Year rollover Dec 31 -> Jan 1
    tp_dec31 = TimePoint(value=_utc(2026, 12, 31, 0, 0, 0), precision=TimePrecision.DAY)
    assert tp_dec31.span_end() == _utc(2027, 1, 1, 0, 0, 0)


def test_time_point_span_end_month_regular_dec_to_jan_and_leap() -> None:
    """Verify MONTH precision returns the 1st of next month at 00:00 UTC."""
    # Regular month
    tp_apr = TimePoint(value=_utc(2026, 4, 15, 14, 30, 0), precision=TimePrecision.MONTH)
    assert tp_apr.span_end() == _utc(2026, 5, 1, 0, 0, 0)

    # Dec -> Jan rollover
    tp_dec = TimePoint(value=_utc(2026, 12, 31, 23, 59, 59), precision=TimePrecision.MONTH)
    assert tp_dec.span_end() == _utc(2027, 1, 1, 0, 0, 0)

    # February in leap year 2024 -> March 1
    tp_feb_leap = TimePoint(value=_utc(2024, 2, 10, 8, 0, 0), precision=TimePrecision.MONTH)
    assert tp_feb_leap.span_end() == _utc(2024, 3, 1, 0, 0, 0)

    # February in non-leap year 2023 -> March 1
    tp_feb_non_leap = TimePoint(value=_utc(2023, 2, 10, 8, 0, 0), precision=TimePrecision.MONTH)
    assert tp_feb_non_leap.span_end() == _utc(2023, 3, 1, 0, 0, 0)


def test_time_point_span_end_year_regular_and_leap() -> None:
    """Verify YEAR precision returns Jan 1 of next year at 00:00 UTC."""
    # Leap year 2024
    tp_leap = TimePoint(value=_utc(2024, 2, 29, 12, 0, 0), precision=TimePrecision.YEAR)
    assert tp_leap.span_end() == _utc(2025, 1, 1, 0, 0, 0)

    # Non-leap year 2026
    tp_2026 = TimePoint(value=_utc(2026, 7, 4, 0, 0, 0), precision=TimePrecision.YEAR)
    assert tp_2026.span_end() == _utc(2027, 1, 1, 0, 0, 0)


# ============================================================================
# Clock tests
# ============================================================================


def test_system_clock() -> None:
    """Verify SystemClock satisfies Clock protocol and returns current UTC datetime."""
    clock = SystemClock()
    assert isinstance(clock, Clock)
    assert isinstance(clock, PortClock)

    now = clock.now()
    assert now.tzinfo is not None
    assert now.utcoffset() == timedelta(0)

    # Check that system time is within 2 seconds of datetime.now(UTC)
    sys_now = datetime.now(UTC)
    assert abs((now - sys_now).total_seconds()) < 2.0


def test_fixed_clock_initialization_and_utc_enforcement() -> None:
    """Verify FixedClock validates UTC awareness at construction."""
    fixed_time = _utc(2026, 10, 26, 12, 0, 0)
    clock = FixedClock(fixed_time)
    assert isinstance(clock, Clock)
    assert isinstance(clock, PortClock)
    assert clock.now() == fixed_time

    # Naive datetime rejected
    naive_dt = datetime(2026, 10, 26, 12, 0, 0)
    with pytest.raises(InvariantViolation, match='fixed_now must be timezone-aware UTC'):
        FixedClock(naive_dt)

    # Non-UTC timezone rejected
    ist_tz = timezone(timedelta(hours=5, minutes=30))
    ist_dt = datetime(2026, 10, 26, 12, 0, 0, tzinfo=ist_tz)
    with pytest.raises(InvariantViolation, match='fixed_now must be timezone-aware UTC'):
        FixedClock(ist_dt)


def test_fixed_clock_advance_and_set() -> None:
    """Verify FixedClock advance and set methods."""
    start_time = _utc(2026, 10, 26, 12, 0, 0)
    clock = FixedClock(start_time)

    # Advance by 1 hour
    advanced = clock.advance(timedelta(hours=1))
    assert advanced == _utc(2026, 10, 26, 13, 0, 0)
    assert clock.now() == _utc(2026, 10, 26, 13, 0, 0)

    # Advance by 1 day
    clock.advance(timedelta(days=1))
    assert clock.now() == _utc(2026, 10, 27, 13, 0, 0)

    # Explicit set
    new_time = _utc(2027, 1, 1, 0, 0, 0)
    clock.set(new_time)
    assert clock.now() == new_time

    # Set rejects naive datetime
    with pytest.raises(InvariantViolation, match='new_now must be timezone-aware UTC'):
        clock.set(datetime(2027, 1, 1))

    # Set rejects non-UTC timezone
    ist_tz = timezone(timedelta(hours=5, minutes=30))
    with pytest.raises(InvariantViolation, match='new_now must be timezone-aware UTC'):
        clock.set(datetime(2027, 1, 1, tzinfo=ist_tz))


def test_port_clock_exports() -> None:
    """Verify ports.clock re-exports match domain.clock implementations."""
    assert PortClock is Clock
    assert PortFixedClock is FixedClock
    assert PortSystemClock is SystemClock
