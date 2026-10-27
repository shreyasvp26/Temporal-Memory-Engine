"""Unit tests for TimeResolver and ResolvedCandidateTime per LLD Section 8.5."""

from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest
from memory_engine.domain.enums import Cardinality, TimePrecision
from memory_engine.domain.errors import InvariantViolation
from memory_engine.ingest.time_resolver import ResolvedCandidateTime, TimeResolver


@pytest.fixture
def resolver() -> TimeResolver:
    return TimeResolver()


@pytest.fixture
def utc_anchor_oct_2026() -> datetime:
    # Anchor: 2026-10-15 12:00:00 UTC
    return datetime(2026, 10, 15, 12, 0, 0, tzinfo=UTC)


@pytest.fixture
def utc_anchor_mar_2026() -> datetime:
    # Anchor: 2026-03-15 12:00:00 UTC
    return datetime(2026, 3, 15, 12, 0, 0, tzinfo=UTC)


# =============================================================================
# Row 1: ISO date or datetime
# =============================================================================


def test_row_1_iso_date(resolver: TimeResolver, utc_anchor_oct_2026: datetime) -> None:
    tz = ZoneInfo('UTC')
    pt = resolver.resolve_point('2026-10-01', utc_anchor_oct_2026, tz)
    assert pt is not None
    assert pt.value == datetime(2026, 10, 1, 0, 0, 0, tzinfo=UTC)
    assert pt.precision == TimePrecision.DAY


def test_row_1_iso_datetime(resolver: TimeResolver, utc_anchor_oct_2026: datetime) -> None:
    tz = ZoneInfo('UTC')
    pt1 = resolver.resolve_point('2026-10-01T14:30:00Z', utc_anchor_oct_2026, tz)
    assert pt1 is not None
    assert pt1.value == datetime(2026, 10, 1, 14, 30, 0, tzinfo=UTC)
    assert pt1.precision == TimePrecision.EXACT

    pt2 = resolver.resolve_point('2026-10-01 14:30:00+00:00', utc_anchor_oct_2026, tz)
    assert pt2 is not None
    assert pt2.value == datetime(2026, 10, 1, 14, 30, 0, tzinfo=UTC)
    assert pt2.precision == TimePrecision.EXACT


# =============================================================================
# Row 2: Day and month name, optional year
# =============================================================================


def test_row_2_day_month_with_year(resolver: TimeResolver, utc_anchor_oct_2026: datetime) -> None:
    tz = ZoneInfo('UTC')
    for phrase in ['1 Oct 2026', 'October 1st, 2026', '1st of October 2026', 'October 1 2026']:
        pt = resolver.resolve_point(phrase, utc_anchor_oct_2026, tz)
        assert pt is not None, f'Failed for phrase: {phrase}'
        assert pt.value == datetime(2026, 10, 1, 0, 0, 0, tzinfo=UTC)
        assert pt.precision == TimePrecision.DAY


def test_row_2_day_month_without_year(
    resolver: TimeResolver, utc_anchor_oct_2026: datetime
) -> None:
    tz = ZoneInfo('UTC')
    # Anchor is 2026-10-15.
    # "October 1st" <= 2026-10-15 -> year 2026
    pt1 = resolver.resolve_point('October 1st', utc_anchor_oct_2026, tz)
    assert pt1 is not None
    assert pt1.value == datetime(2026, 10, 1, 0, 0, 0, tzinfo=UTC)
    assert pt1.precision == TimePrecision.DAY

    # "October 20th" > 2026-10-15 -> most recent date not after anchor is 2025-10-20
    pt2 = resolver.resolve_point('October 20th', utc_anchor_oct_2026, tz)
    assert pt2 is not None
    assert pt2.value == datetime(2025, 10, 20, 0, 0, 0, tzinfo=UTC)
    assert pt2.precision == TimePrecision.DAY


def test_row_2_leap_year(resolver: TimeResolver) -> None:
    tz = ZoneInfo('UTC')
    # Anchor is Feb 28, 2027 (non-leap year). "29 Feb" should resolve to 2024-02-29
    anchor_2027 = datetime(2027, 2, 28, 12, 0, 0, tzinfo=UTC)
    pt = resolver.resolve_point('29 Feb', anchor_2027, tz)
    assert pt is not None
    assert pt.value == datetime(2024, 2, 29, 0, 0, 0, tzinfo=UTC)
    assert pt.precision == TimePrecision.DAY


# =============================================================================
# Row 3: Month name and year
# =============================================================================


def test_row_3_month_and_year(resolver: TimeResolver, utc_anchor_oct_2026: datetime) -> None:
    tz = ZoneInfo('UTC')
    for phrase in ['March 2025', 'in March 2025', 'Mar 2025', 'March, 2025']:
        pt = resolver.resolve_point(phrase, utc_anchor_oct_2026, tz)
        assert pt is not None, f'Failed for phrase: {phrase}'
        assert pt.value == datetime(2025, 3, 1, 0, 0, 0, tzinfo=UTC)
        assert pt.precision == TimePrecision.MONTH


# =============================================================================
# Row 4: Bare month (most recent such month whose start is <= anchor month start)
# =============================================================================


def test_row_4_bare_month(
    resolver: TimeResolver, utc_anchor_oct_2026: datetime, utc_anchor_mar_2026: datetime
) -> None:
    tz = ZoneInfo('UTC')
    # Anchor Oct 2026:
    # "March" (month 3 <= 10) -> March 2026
    pt_mar = resolver.resolve_point('March', utc_anchor_oct_2026, tz)
    assert pt_mar is not None
    assert pt_mar.value == datetime(2026, 3, 1, 0, 0, 0, tzinfo=UTC)
    assert pt_mar.precision == TimePrecision.MONTH

    # "in October" (month 10 <= 10) -> October 2026
    pt_oct = resolver.resolve_point('in October', utc_anchor_oct_2026, tz)
    assert pt_oct is not None
    assert pt_oct.value == datetime(2026, 10, 1, 0, 0, 0, tzinfo=UTC)
    assert pt_oct.precision == TimePrecision.MONTH

    # "November" (month 11 > 10) -> November 2025
    pt_nov = resolver.resolve_point('November', utc_anchor_oct_2026, tz)
    assert pt_nov is not None
    assert pt_nov.value == datetime(2025, 11, 1, 0, 0, 0, tzinfo=UTC)
    assert pt_nov.precision == TimePrecision.MONTH

    # Anchor Mar 2026:
    # "March" (month 3 <= 3) -> March 2026
    pt_mar_anchor_mar = resolver.resolve_point('March', utc_anchor_mar_2026, tz)
    assert pt_mar_anchor_mar is not None
    assert pt_mar_anchor_mar.value == datetime(2026, 3, 1, 0, 0, 0, tzinfo=UTC)


# =============================================================================
# Row 5: last MONTH (latest such month whose start is strictly < anchor month start)
# Specific verification for LLD examples: Oct 2026 -> Mar 2026; Mar 2026 -> Mar 2025
# =============================================================================


def test_row_5_last_month_anchors(
    resolver: TimeResolver, utc_anchor_oct_2026: datetime, utc_anchor_mar_2026: datetime
) -> None:
    tz = ZoneInfo('UTC')
    # Anchor Oct 2026 -> last March gives March 2026 (month 3 < 10)
    pt1 = resolver.resolve_point('last March', utc_anchor_oct_2026, tz)
    assert pt1 is not None
    assert pt1.value == datetime(2026, 3, 1, 0, 0, 0, tzinfo=UTC)
    assert pt1.precision == TimePrecision.MONTH

    # Anchor Mar 2026 -> last March gives March 2025 (month 3 < 3 is False -> 2025)
    pt2 = resolver.resolve_point('last March', utc_anchor_mar_2026, tz)
    assert pt2 is not None
    assert pt2.value == datetime(2025, 3, 1, 0, 0, 0, tzinfo=UTC)
    assert pt2.precision == TimePrecision.MONTH

    # Anchor Oct 2026 -> last October gives October 2025 (month 10 < 10 is False -> 2025)
    pt3 = resolver.resolve_point('last October', utc_anchor_oct_2026, tz)
    assert pt3 is not None
    assert pt3.value == datetime(2025, 10, 1, 0, 0, 0, tzinfo=UTC)
    assert pt3.precision == TimePrecision.MONTH


# =============================================================================
# Row 6: this MONTH (month within anchor year)
# =============================================================================


def test_row_6_this_month(resolver: TimeResolver, utc_anchor_oct_2026: datetime) -> None:
    tz = ZoneInfo('UTC')
    pt_mar = resolver.resolve_point('this March', utc_anchor_oct_2026, tz)
    assert pt_mar is not None
    assert pt_mar.value == datetime(2026, 3, 1, 0, 0, 0, tzinfo=UTC)
    assert pt_mar.precision == TimePrecision.MONTH

    pt_dec = resolver.resolve_point('this December', utc_anchor_oct_2026, tz)
    assert pt_dec is not None
    assert pt_dec.value == datetime(2026, 12, 1, 0, 0, 0, tzinfo=UTC)
    assert pt_dec.precision == TimePrecision.MONTH


# =============================================================================
# Row 7: Year only
# =============================================================================


def test_row_7_year_only(resolver: TimeResolver, utc_anchor_oct_2026: datetime) -> None:
    tz = ZoneInfo('UTC')
    for phrase in ['2025', 'in 2025']:
        pt = resolver.resolve_point(phrase, utc_anchor_oct_2026, tz)
        assert pt is not None, f'Failed for phrase: {phrase}'
        assert pt.value == datetime(2025, 1, 1, 0, 0, 0, tzinfo=UTC)
        assert pt.precision == TimePrecision.YEAR


# =============================================================================
# Row 8: today, tonight, this morning, yesterday, the day before yesterday
# =============================================================================


def test_row_8_relative_days(resolver: TimeResolver, utc_anchor_oct_2026: datetime) -> None:
    tz = ZoneInfo('UTC')
    for phrase in ['today', 'tonight', 'this morning', 'this afternoon', 'this evening']:
        pt = resolver.resolve_point(phrase, utc_anchor_oct_2026, tz)
        assert pt is not None, f'Failed for phrase: {phrase}'
        assert pt.value == datetime(2026, 10, 15, 0, 0, 0, tzinfo=UTC)
        assert pt.precision == TimePrecision.DAY

    pt_yest = resolver.resolve_point('yesterday', utc_anchor_oct_2026, tz)
    assert pt_yest is not None
    assert pt_yest.value == datetime(2026, 10, 14, 0, 0, 0, tzinfo=UTC)
    assert pt_yest.precision == TimePrecision.DAY

    pt_dby = resolver.resolve_point('the day before yesterday', utc_anchor_oct_2026, tz)
    assert pt_dby is not None
    assert pt_dby.value == datetime(2026, 10, 13, 0, 0, 0, tzinfo=UTC)
    assert pt_dby.precision == TimePrecision.DAY


# =============================================================================
# Row 9: last week, last month, last year, this week, this month, this year
# =============================================================================


def test_row_9_periods(resolver: TimeResolver, utc_anchor_oct_2026: datetime) -> None:
    tz = ZoneInfo('UTC')
    # 2026-10-15 is Thursday (weekday 3).
    # Monday of this week: 2026-10-12
    pt_tw = resolver.resolve_point('this week', utc_anchor_oct_2026, tz)
    assert pt_tw is not None
    assert pt_tw.value == datetime(2026, 10, 12, 0, 0, 0, tzinfo=UTC)
    assert pt_tw.precision == TimePrecision.DAY

    # Monday of last week: 2026-10-05
    pt_lw = resolver.resolve_point('last week', utc_anchor_oct_2026, tz)
    assert pt_lw is not None
    assert pt_lw.value == datetime(2026, 10, 5, 0, 0, 0, tzinfo=UTC)
    assert pt_lw.precision == TimePrecision.DAY

    # this month: 2026-10-01
    pt_tm = resolver.resolve_point('this month', utc_anchor_oct_2026, tz)
    assert pt_tm is not None
    assert pt_tm.value == datetime(2026, 10, 1, 0, 0, 0, tzinfo=UTC)
    assert pt_tm.precision == TimePrecision.MONTH

    # last month: 2026-09-01
    pt_lm = resolver.resolve_point('last month', utc_anchor_oct_2026, tz)
    assert pt_lm is not None
    assert pt_lm.value == datetime(2026, 9, 1, 0, 0, 0, tzinfo=UTC)
    assert pt_lm.precision == TimePrecision.MONTH

    # this year: 2026-01-01
    pt_ty = resolver.resolve_point('this year', utc_anchor_oct_2026, tz)
    assert pt_ty is not None
    assert pt_ty.value == datetime(2026, 1, 1, 0, 0, 0, tzinfo=UTC)
    assert pt_ty.precision == TimePrecision.YEAR

    # last year: 2025-01-01
    pt_ly = resolver.resolve_point('last year', utc_anchor_oct_2026, tz)
    assert pt_ly is not None
    assert pt_ly.value == datetime(2025, 1, 1, 0, 0, 0, tzinfo=UTC)
    assert pt_ly.precision == TimePrecision.YEAR


# =============================================================================
# Row 10: N units ago
# =============================================================================


def test_row_10_units_ago(resolver: TimeResolver, utc_anchor_oct_2026: datetime) -> None:
    tz = ZoneInfo('UTC')
    # 3 days ago / three days ago
    pt_3d = resolver.resolve_point('3 days ago', utc_anchor_oct_2026, tz)
    assert pt_3d is not None
    assert pt_3d.value == datetime(2026, 10, 12, 0, 0, 0, tzinfo=UTC)
    assert pt_3d.precision == TimePrecision.DAY

    pt_3d_word = resolver.resolve_point('three days ago', utc_anchor_oct_2026, tz)
    assert pt_3d_word is not None
    assert pt_3d_word.value == datetime(2026, 10, 12, 0, 0, 0, tzinfo=UTC)
    assert pt_3d_word.precision == TimePrecision.DAY

    # 2 weeks ago
    pt_2w = resolver.resolve_point('2 weeks ago', utc_anchor_oct_2026, tz)
    assert pt_2w is not None
    assert pt_2w.value == datetime(2026, 10, 1, 0, 0, 0, tzinfo=UTC)
    assert pt_2w.precision == TimePrecision.DAY

    # 4 months ago (from 2026-10 -> 2026-06)
    pt_4m = resolver.resolve_point('4 months ago', utc_anchor_oct_2026, tz)
    assert pt_4m is not None
    assert pt_4m.value == datetime(2026, 6, 1, 0, 0, 0, tzinfo=UTC)
    assert pt_4m.precision == TimePrecision.MONTH

    # 2 years ago (from 2026 -> 2024)
    pt_2y = resolver.resolve_point('two years ago', utc_anchor_oct_2026, tz)
    assert pt_2y is not None
    assert pt_2y.value == datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC)
    assert pt_2y.precision == TimePrecision.YEAR


# =============================================================================
# Row 11: since X, from X
# =============================================================================


def test_row_11_since_and_from(resolver: TimeResolver, utc_anchor_oct_2026: datetime) -> None:
    tz = ZoneInfo('UTC')
    pt1 = resolver.resolve_point('since 2024', utc_anchor_oct_2026, tz)
    assert pt1 is not None
    assert pt1.value == datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC)
    assert pt1.precision == TimePrecision.YEAR

    pt2 = resolver.resolve_point('from March 2025', utc_anchor_oct_2026, tz)
    assert pt2 is not None
    assert pt2.value == datetime(2025, 3, 1, 0, 0, 0, tzinfo=UTC)
    assert pt2.precision == TimePrecision.MONTH


# =============================================================================
# Row 12: until X, till X, through X
# =============================================================================


def test_row_12_until_till_through(resolver: TimeResolver, utc_anchor_oct_2026: datetime) -> None:
    tz = ZoneInfo('UTC')
    # until 2024 -> span_end of 2024 is 2025-01-01
    pt1 = resolver.resolve_point('until 2024', utc_anchor_oct_2026, tz)
    assert pt1 is not None
    assert pt1.value == datetime(2025, 1, 1, 0, 0, 0, tzinfo=UTC)

    # through March 2025 -> span_end of March 2025 is 2025-04-01
    pt2 = resolver.resolve_point('through March 2025', utc_anchor_oct_2026, tz)
    assert pt2 is not None
    assert pt2.value == datetime(2025, 4, 1, 0, 0, 0, tzinfo=UTC)

    # till yesterday -> span_end of yesterday (2026-10-14) is 2026-10-15
    pt3 = resolver.resolve_point('till yesterday', utc_anchor_oct_2026, tz)
    assert pt3 is not None
    assert pt3.value == datetime(2026, 10, 15, 0, 0, 0, tzinfo=UTC)


# =============================================================================
# Row 13: Seasons, recently, a while ago, long ago -> unresolved (None)
# =============================================================================


def test_row_13_unresolved_phrases(resolver: TimeResolver, utc_anchor_oct_2026: datetime) -> None:
    tz = ZoneInfo('UTC')
    vague_phrases = [
        'spring',
        'in summer',
        'last winter',
        'autumn 2024',
        'recently',
        'a while ago',
        'long ago',
        'just now',
        'lately',
        'earlier',
    ]
    for phrase in vague_phrases:
        pt = resolver.resolve_point(phrase, utc_anchor_oct_2026, tz)
        assert pt is None, f'Expected None for vague phrase: {phrase}'


# =============================================================================
# Row 14: Fallback dateparser
# =============================================================================


def test_row_14_fallback_dateparser(resolver: TimeResolver, utc_anchor_oct_2026: datetime) -> None:
    tz = ZoneInfo('UTC')
    # Phrase with digits that doesn't match earlier rules
    pt = resolver.resolve_point('2024/05/15', utc_anchor_oct_2026, tz)
    assert pt is not None
    assert pt.value == datetime(2024, 5, 15, 0, 0, 0, tzinfo=UTC)
    assert pt.precision == TimePrecision.DAY

    # Phrase without digits or month names must be rejected
    rejected = resolver.resolve_point('some random text without numbers', utc_anchor_oct_2026, tz)
    assert rejected is None

    # Invalid nonsense with numbers
    assert resolver.resolve_point('random 99999999', utc_anchor_oct_2026, tz) is None


# =============================================================================
# Range Resolution: resolve_range
# =============================================================================


def test_resolve_range(resolver: TimeResolver, utc_anchor_oct_2026: datetime) -> None:
    tz = ZoneInfo('UTC')
    # between X and Y
    r1 = resolver.resolve_range('between 2020 and 2022', utc_anchor_oct_2026, tz)
    assert r1 is not None
    assert r1.start == datetime(2020, 1, 1, 0, 0, 0, tzinfo=UTC)
    assert r1.end == datetime(2023, 1, 1, 0, 0, 0, tzinfo=UTC)

    # from X to Y
    r2 = resolver.resolve_range('from March 2025 to May 2025', utc_anchor_oct_2026, tz)
    assert r2 is not None
    assert r2.start == datetime(2025, 3, 1, 0, 0, 0, tzinfo=UTC)
    assert r2.end == datetime(2025, 6, 1, 0, 0, 0, tzinfo=UTC)

    # since X -> [X.value, None)
    r3 = resolver.resolve_range('since 2024', utc_anchor_oct_2026, tz)
    assert r3 is not None
    assert r3.start == datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC)
    assert r3.end is None

    # until X -> [None, X.span_end())
    r4 = resolver.resolve_range('until 2024', utc_anchor_oct_2026, tz)
    assert r4 is not None
    assert r4.start is None
    assert r4.end == datetime(2025, 1, 1, 0, 0, 0, tzinfo=UTC)

    # bare point -> [point.value, point.span_end())
    r5 = resolver.resolve_range('March 2025', utc_anchor_oct_2026, tz)
    assert r5 is not None
    assert r5.start == datetime(2025, 3, 1, 0, 0, 0, tzinfo=UTC)
    assert r5.end == datetime(2025, 4, 1, 0, 0, 0, tzinfo=UTC)

    # Inverted range -> None
    assert resolver.resolve_range('between 2025 and 2020', utc_anchor_oct_2026, tz) is None

    # Unresolved range -> None
    assert resolver.resolve_range('recently', utc_anchor_oct_2026, tz) is None
    assert resolver.resolve_range(None, utc_anchor_oct_2026, tz) is None


# =============================================================================
# Fact Bounds & Future Guard: resolve_fact_bounds
# =============================================================================


def test_resolve_fact_bounds_past_explicit(
    resolver: TimeResolver, utc_anchor_oct_2026: datetime
) -> None:
    tz = ZoneInfo('UTC')
    res = resolver.resolve_fact_bounds(
        valid_from_text='October 1st',
        valid_to_text=None,
        source_ts=utc_anchor_oct_2026,
        tz=tz,
        predicate='lives_in',
        cardinality=Cardinality.SINGLE,
    )
    assert res.valid_from == datetime(2026, 10, 1, 0, 0, 0, tzinfo=UTC)
    assert res.valid_to is None
    assert res.time_precision == TimePrecision.DAY
    assert res.valid_from_explicit is True
    assert res.confidence_multiplier == 1.0


def test_resolve_fact_bounds_future_guard_clamping(
    resolver: TimeResolver, utc_anchor_oct_2026: datetime
) -> None:
    tz = ZoneInfo('UTC')
    # Point is in 2027 (future compared to anchor 2026-10-15) and predicate is NOT plans_to
    res = resolver.resolve_fact_bounds(
        valid_from_text='2027-01-01',
        valid_to_text=None,
        source_ts=utc_anchor_oct_2026,
        tz=tz,
        predicate='lives_in',
        cardinality=Cardinality.SINGLE,
    )
    # Clamped to source_ts, explicit False, multiplier 0.9
    assert res.valid_from == utc_anchor_oct_2026
    assert res.valid_to is None
    assert res.time_precision == TimePrecision.DAY
    assert res.valid_from_explicit is False
    assert res.confidence_multiplier == 0.9


def test_resolve_fact_bounds_future_guard_plans_to(
    resolver: TimeResolver, utc_anchor_oct_2026: datetime
) -> None:
    tz = ZoneInfo('UTC')
    # Point is in 2027 and predicate IS plans_to -> NOT clamped
    res = resolver.resolve_fact_bounds(
        valid_from_text='2027-01-01',
        valid_to_text=None,
        source_ts=utc_anchor_oct_2026,
        tz=tz,
        predicate='plans_to',
        cardinality=Cardinality.SINGLE,
    )
    assert res.valid_from == datetime(2027, 1, 1, 0, 0, 0, tzinfo=UTC)
    assert res.valid_to is None
    assert res.time_precision == TimePrecision.DAY
    assert res.valid_from_explicit is True
    assert res.confidence_multiplier == 1.0


def test_resolve_fact_bounds_unresolved_or_none(
    resolver: TimeResolver, utc_anchor_oct_2026: datetime
) -> None:
    tz = ZoneInfo('UTC')
    res = resolver.resolve_fact_bounds(
        valid_from_text=None,
        valid_to_text=None,
        source_ts=utc_anchor_oct_2026,
        tz=tz,
        predicate='likes',
        cardinality=Cardinality.MULTI,
    )
    assert res.valid_from == utc_anchor_oct_2026
    assert res.valid_to is None
    assert res.time_precision == TimePrecision.EXACT
    assert res.valid_from_explicit is False
    assert res.confidence_multiplier == 1.0


def test_resolve_fact_bounds_event_finite_valid_to(
    resolver: TimeResolver, utc_anchor_oct_2026: datetime
) -> None:
    tz = ZoneInfo('UTC')
    # EVENT with DAY precision valid_from: valid_to is span_end (1 day later)
    res = resolver.resolve_fact_bounds(
        valid_from_text='2026-10-01',
        valid_to_text=None,
        source_ts=utc_anchor_oct_2026,
        tz=tz,
        predicate='attended',
        cardinality=Cardinality.EVENT,
    )
    assert res.valid_from == datetime(2026, 10, 1, 0, 0, 0, tzinfo=UTC)
    assert res.valid_to == datetime(2026, 10, 2, 0, 0, 0, tzinfo=UTC)
    assert res.time_precision == TimePrecision.DAY
    assert res.valid_from_explicit is True
    assert res.confidence_multiplier == 1.0


def test_resolve_fact_bounds_event_unresolved(
    resolver: TimeResolver, utc_anchor_oct_2026: datetime
) -> None:
    tz = ZoneInfo('UTC')
    # EVENT without valid_from_text -> valid_from = source_ts (EXACT), valid_to = source_ts + 1s
    res = resolver.resolve_fact_bounds(
        valid_from_text=None,
        valid_to_text=None,
        source_ts=utc_anchor_oct_2026,
        tz=tz,
        predicate='attended',
        cardinality=Cardinality.EVENT,
    )
    assert res.valid_from == utc_anchor_oct_2026
    assert res.valid_to == utc_anchor_oct_2026 + timedelta(seconds=1)
    assert res.time_precision == TimePrecision.EXACT
    assert res.valid_from_explicit is False
    assert res.confidence_multiplier == 1.0


def test_resolve_fact_bounds_drop_invalid_valid_to(
    resolver: TimeResolver, utc_anchor_oct_2026: datetime
) -> None:
    tz = ZoneInfo('UTC')
    # valid_to <= valid_from -> valid_to must be dropped
    res = resolver.resolve_fact_bounds(
        valid_from_text='2026-10-01',
        valid_to_text='2025-01-01',
        source_ts=utc_anchor_oct_2026,
        tz=tz,
        predicate='worked_at',
        cardinality=Cardinality.SINGLE,
    )
    assert res.valid_from == datetime(2026, 10, 1, 0, 0, 0, tzinfo=UTC)
    assert res.valid_to is None


# =============================================================================
# Timezone Boundary Conversion
# =============================================================================


def test_timezone_boundary_conversion(resolver: TimeResolver) -> None:
    ny_tz = ZoneInfo('America/New_York')
    # In New York in October (EDT, UTC-4), midnight 2026-10-01 EDT is 2026-10-01 04:00:00 UTC
    anchor = datetime(2026, 10, 15, 12, 0, 0, tzinfo=UTC)

    pt = resolver.resolve_point('2026-10-01', anchor, ny_tz)
    assert pt is not None
    assert pt.value == datetime(2026, 10, 1, 4, 0, 0, tzinfo=UTC)
    assert pt.precision == TimePrecision.DAY

    # span_end in TimePoint should be +1 day: 2026-10-02 04:00:00 UTC
    assert pt.span_end() == datetime(2026, 10, 2, 4, 0, 0, tzinfo=UTC)


# =============================================================================
# Invariant Violations
# =============================================================================


def test_non_utc_anchor_raises_invariant_violation(resolver: TimeResolver) -> None:
    naive_anchor = datetime(2026, 10, 15, 12, 0, 0)
    with pytest.raises(InvariantViolation, match='anchor must be timezone-aware UTC'):
        resolver.resolve_point('today', naive_anchor, ZoneInfo('UTC'))


def test_resolved_candidate_time_post_init_validation() -> None:
    t1 = datetime(2026, 10, 1, 0, 0, 0, tzinfo=UTC)
    t0 = datetime(2026, 9, 1, 0, 0, 0, tzinfo=UTC)
    with pytest.raises(
        InvariantViolation, match='valid_to must be strictly greater than valid_from'
    ):
        ResolvedCandidateTime(
            valid_from=t1,
            valid_to=t0,
            time_precision=TimePrecision.DAY,
            valid_from_explicit=True,
            confidence_multiplier=1.0,
        )
