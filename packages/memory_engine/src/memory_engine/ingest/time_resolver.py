"""Time resolver implementation per LLD Section 8.5."""

import re
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import structlog
from dateparser.date import DateDataParser

from memory_engine.domain.enums import Cardinality, TimePrecision
from memory_engine.domain.errors import InvariantViolation
from memory_engine.domain.intervals import TimeInterval, TimePoint

__all__ = [
    'ResolvedCandidateTime',
    'TimeResolver',
]

_logger = structlog.get_logger(__name__)


def _validate_utc_datetime(dt: datetime, name: str) -> None:
    if not isinstance(dt, datetime):
        raise InvariantViolation(f'{name} must be a datetime, got {type(dt)!r}')
    if dt.tzinfo is None or dt.utcoffset() != timedelta(0):
        raise InvariantViolation(f'{name} must be timezone-aware UTC, got {dt!r}')


@dataclass(frozen=True, slots=True)
class ResolvedCandidateTime:
    """Resolved candidate fact time bounds per LLD Section 8.5."""

    valid_from: datetime
    valid_to: datetime | None
    time_precision: TimePrecision
    valid_from_explicit: bool
    confidence_multiplier: float

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


_MONTH_NAMES: dict[str, int] = {
    'january': 1,
    'jan': 1,
    'february': 2,
    'feb': 2,
    'march': 3,
    'mar': 3,
    'april': 4,
    'apr': 4,
    'may': 5,
    'june': 6,
    'jun': 6,
    'july': 7,
    'jul': 7,
    'august': 8,
    'aug': 8,
    'september': 9,
    'sep': 9,
    'sept': 9,
    'october': 10,
    'oct': 10,
    'november': 11,
    'nov': 11,
    'december': 12,
    'dec': 12,
}

_WORD_NUMBERS: dict[str, int] = {
    'a': 1,
    'an': 1,
    'one': 1,
    'two': 2,
    'three': 3,
    'four': 4,
    'five': 5,
    'six': 6,
    'seven': 7,
    'eight': 8,
    'nine': 9,
    'ten': 10,
    'eleven': 11,
    'twelve': 12,
    'thirteen': 13,
    'fourteen': 14,
    'fifteen': 15,
    'sixteen': 16,
    'seventeen': 17,
    'eighteen': 18,
    'nineteen': 19,
    'twenty': 20,
    'thirty': 30,
    'forty': 40,
    'fifty': 50,
    'sixty': 60,
    'seventy': 70,
    'eighty': 80,
    'ninety': 90,
    'hundred': 100,
}

# Row patterns
_ROW_1_ISO_DATE = re.compile(r'^\s*(\d{4})-(\d{2})-(\d{2})\s*$')
_ROW_1_ISO_DATETIME = re.compile(
    r'^\s*(\d{4})-(\d{2})-(\d{2})[T ](\d{2}):(\d{2})'
    r'(?::(\d{2})(?:\.(\d+))?)?(?:\s*(Z|[+-]\d{2}:?\d{2}))?\s*$'
)

_ROW_2_DAY_FIRST = re.compile(
    r'^\s*(?:(?:on|the)\s+)?(\d{1,2})(?:st|nd|rd|th)?(?:\s+of)?\s+([A-Za-z]+)\.?(?:[,\s]+(\d{4}))?\s*$',
    re.IGNORECASE,
)
_ROW_2_MONTH_FIRST = re.compile(
    r'^\s*(?:(?:on|the)\s+)?([A-Za-z]+)\.?\s+(\d{1,2})(?:st|nd|rd|th)?(?:[,\s]+(\d{4}))?\s*$',
    re.IGNORECASE,
)

_ROW_3 = re.compile(r'^\s*(?:in\s+)?([A-Za-z]+)\.?[,\s]+(\d{4})\s*$', re.IGNORECASE)
_ROW_4 = re.compile(r'^\s*(?:in\s+)?([A-Za-z]+)\.?\s*$', re.IGNORECASE)
_ROW_5 = re.compile(r'^\s*(?:in\s+)?last\s+([A-Za-z]+)\.?\s*$', re.IGNORECASE)
_ROW_6 = re.compile(r'^\s*(?:in\s+)?this\s+([A-Za-z]+)\.?\s*$', re.IGNORECASE)
_ROW_7 = re.compile(r'^\s*(?:in\s+)?([12]\d{3})\s*$', re.IGNORECASE)

_ROW_8 = re.compile(
    r'^\s*(today|tonight|this\s+morning|this\s+afternoon|this\s+evening|yesterday|the\s+day\s+before\s+yesterday)\s*$',
    re.IGNORECASE,
)
_ROW_9 = re.compile(r'^\s*(last|this)\s+(week|month|year)\s*$', re.IGNORECASE)
_ROW_10 = re.compile(
    r'^\s*(\d+|[a-zA-Z]+)\s+(days?|weeks?|months?|years?)\s+ago\s*$',
    re.IGNORECASE,
)

_ROW_11 = re.compile(r'^\s*(?:since|from)\s+(.+)$', re.IGNORECASE)
_ROW_12 = re.compile(r'^\s*(?:until|till|through|to|before)\s+(.+)$', re.IGNORECASE)

_ROW_13_SEASONS = re.compile(
    r'^\s*(?:in\s+)?(?:this\s+|last\s+)?(?:spring|summer|autumn|fall|winter)(?:\s+\d{4})?\s*$',
    re.IGNORECASE,
)
_ROW_13_RELATIVE = re.compile(
    r'^\s*(?:recently|a\s+while\s+ago|long\s+ago|just\s+now|lately|earlier|some\s+time\s+ago)\s*$',
    re.IGNORECASE,
)

_ROW_14_DIGIT_OR_MONTH = re.compile(
    r'\d|\b(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|sep(?:t|tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\b',
    re.IGNORECASE,
)

_RANGE_BETWEEN = re.compile(
    r'^\s*(?:between\s+(.+?)\s+and\s+(.+)|from\s+(.+?)\s+(?:to|until|till)\s+(.+))\s*$',
    re.IGNORECASE,
)
_RANGE_SINCE = re.compile(r'^\s*(?:since|from)\s+(.+)$', re.IGNORECASE)
_RANGE_UNTIL = re.compile(r'^\s*(?:until|till|through|to|before)\s+(.+)$', re.IGNORECASE)


class TimeResolver:
    """Deterministic temporal resolver per LLD Section 8.5."""

    def resolve_point(
        self,
        phrase: str | None,
        anchor: datetime,
        tz: ZoneInfo,
    ) -> TimePoint | None:
        """Resolve a temporal phrase to a TimePoint in UTC."""
        _validate_utc_datetime(anchor, 'anchor')
        if phrase is None:
            return None
        p = phrase.strip().strip('"\'')
        if not p:
            return None

        anchor_local = anchor.astimezone(tz)

        # Row 1: ISO date or datetime
        if (res := self._match_row_1(p, tz)) is not None:
            return res

        # Row 2: Day and month name, optional year
        if (res := self._match_row_2(p, anchor_local, tz)) is not None:
            return res

        # Row 3: Month name and year
        if (res := self._match_row_3(p, tz)) is not None:
            return res

        # Row 4: Bare month
        if (res := self._match_row_4(p, anchor_local, tz)) is not None:
            return res

        # Row 5: last MONTH
        if (res := self._match_row_5(p, anchor_local, tz)) is not None:
            return res

        # Row 6: this MONTH
        if (res := self._match_row_6(p, anchor_local, tz)) is not None:
            return res

        # Row 7: Year only
        if (res := self._match_row_7(p, tz)) is not None:
            return res

        # Row 8: today, tonight, this morning, yesterday, the day before yesterday
        if (res := self._match_row_8(p, anchor_local, tz)) is not None:
            return res

        # Row 9: last week, last month, last year, this week, this month, this year
        if (res := self._match_row_9(p, anchor_local, tz)) is not None:
            return res

        # Row 10: N units ago
        if (res := self._match_row_10(p, anchor_local, tz)) is not None:
            return res

        # Row 11: since X, from X
        if (res := self._match_row_11(p, anchor, tz)) is not None:
            return res

        # Row 12: until X, till X, through X
        if (res := self._match_row_12(p, anchor, tz)) is not None:
            return res

        # Row 13: Seasons, recently, a while ago, long ago
        if self._match_row_13(p):
            return None

        # Row 14: Fallback: dateparser
        return self._match_row_14(p, anchor, tz)

    def resolve_range(
        self,
        phrase: str | None,
        anchor: datetime,
        tz: ZoneInfo,
    ) -> TimeInterval | None:
        """Resolve a temporal range phrase to a half-open TimeInterval."""
        _validate_utc_datetime(anchor, 'anchor')
        if phrase is None:
            return None
        p = phrase.strip().strip('"\'')
        if not p:
            return None

        if m := _RANGE_BETWEEN.match(p):
            x_text = m.group(1) or m.group(3)
            y_text = m.group(2) or m.group(4)
            x_pt = self.resolve_point(x_text, anchor, tz)
            y_pt = self.resolve_point(y_text, anchor, tz)
            if x_pt is not None and y_pt is not None:
                start = x_pt.value
                end = y_pt.span_end()
                if start < end:
                    return TimeInterval(start=start, end=end)
            return None

        if m := _RANGE_SINCE.match(p):
            x_text = m.group(1)
            x_pt = self.resolve_point(x_text, anchor, tz)
            if x_pt is not None:
                return TimeInterval(start=x_pt.value, end=None)
            return None

        if m := _RANGE_UNTIL.match(p):
            x_text = m.group(1)
            x_pt = self.resolve_point(x_text, anchor, tz)
            if x_pt is not None:
                return TimeInterval(start=None, end=x_pt.span_end())
            return None

        pt = self.resolve_point(p, anchor, tz)
        if pt is not None:
            return TimeInterval(start=pt.value, end=pt.span_end())

        return None

    def resolve_fact_bounds(
        self,
        valid_from_text: str | None,
        valid_to_text: str | None,
        source_ts: datetime,
        tz: ZoneInfo,
        predicate: str,
        cardinality: Cardinality,
    ) -> ResolvedCandidateTime:
        """Resolve fact candidate valid_from and valid_to with future guard."""
        _validate_utc_datetime(source_ts, 'source_ts')

        from_pt = (
            self.resolve_point(valid_from_text, anchor=source_ts, tz=tz)
            if valid_from_text is not None and valid_from_text.strip()
            else None
        )

        if from_pt is not None:
            valid_from = from_pt.value
            precision = from_pt.precision
            valid_from_explicit = True
            confidence_multiplier = 1.0
            from_pt_event = from_pt

            # Future guard: a resolved point after the anchor is clamped to the anchor
            # and treated as not explicit, unless the predicate is plans_to.
            # Confidence is multiplied by 0.9.
            if valid_from > source_ts and predicate != 'plans_to':
                valid_from = source_ts
                valid_from_explicit = False
                confidence_multiplier = 0.9
                from_pt_event = TimePoint(value=source_ts, precision=precision)
        else:
            # Unresolved or absent phrases give valid_from = source_ts,
            # valid_from_explicit = false, precision exact.
            valid_from = source_ts
            precision = TimePrecision.EXACT
            valid_from_explicit = False
            confidence_multiplier = 1.0
            from_pt_event = TimePoint(value=source_ts, precision=TimePrecision.EXACT)

        # valid_to determination
        if cardinality == Cardinality.EVENT:
            valid_to: datetime | None = from_pt_event.span_end()
        else:
            if valid_to_text is not None and valid_to_text.strip():
                to_pt = self.resolve_point(valid_to_text, anchor=source_ts, tz=tz)
                valid_to = to_pt.value if to_pt is not None else None
            else:
                valid_to = None

        # If valid_to <= valid_from after resolution the valid_to is dropped and logged.
        if valid_to is not None and valid_to <= valid_from:
            _logger.warning(
                'dropped_invalid_valid_to',
                valid_from=valid_from.isoformat(),
                valid_to=valid_to.isoformat(),
                predicate=predicate,
                cardinality=str(cardinality),
            )
            valid_to = None

        return ResolvedCandidateTime(
            valid_from=valid_from,
            valid_to=valid_to,
            time_precision=precision,
            valid_from_explicit=valid_from_explicit,
            confidence_multiplier=confidence_multiplier,
        )

    # -------------------------------------------------------------------------
    # Grammar Row Matchers
    # -------------------------------------------------------------------------

    def _match_row_1(self, p: str, tz: ZoneInfo) -> TimePoint | None:
        """Row 1: ISO date or datetime."""
        if m := _ROW_1_ISO_DATE.match(p):
            y, mth, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
            try:
                dt_tz = datetime(y, mth, d, 0, 0, 0, tzinfo=tz)
                return TimePoint(value=dt_tz.astimezone(UTC), precision=TimePrecision.DAY)
            except ValueError:
                return None

        if _ROW_1_ISO_DATETIME.match(p):
            try:
                dt = datetime.fromisoformat(p)
                if dt.tzinfo is None:
                    dt_tz = dt.replace(tzinfo=tz)
                    dt_utc = dt_tz.astimezone(UTC)
                else:
                    dt_utc = dt.astimezone(UTC)
                return TimePoint(value=dt_utc, precision=TimePrecision.EXACT)
            except ValueError:
                return None

        return None

    def _match_row_2(self, p: str, anchor_local: datetime, tz: ZoneInfo) -> TimePoint | None:
        """Row 2: Day and month name, optional year."""
        m_day_first = _ROW_2_DAY_FIRST.match(p)
        m_month_first = _ROW_2_MONTH_FIRST.match(p)

        if not m_day_first and not m_month_first:
            return None

        if m_day_first:
            day_str, month_str, year_str = (
                m_day_first.group(1),
                m_day_first.group(2),
                m_day_first.group(3),
            )
        else:
            assert m_month_first is not None
            month_str, day_str, year_str = (
                m_month_first.group(1),
                m_month_first.group(2),
                m_month_first.group(3),
            )

        month_key = month_str.lower()
        if month_key not in _MONTH_NAMES:
            return None

        mth = _MONTH_NAMES[month_key]
        d = int(day_str)

        if year_str is not None:
            target_year = int(year_str)
            try:
                dt_local = datetime(target_year, mth, d, 0, 0, 0, tzinfo=tz)
                return TimePoint(value=dt_local.astimezone(UTC), precision=TimePrecision.DAY)
            except ValueError:
                return None

        # Year missing: most recent such date not after the anchor
        anchor_date = anchor_local.date()
        candidate_year = anchor_local.year
        for _ in range(8):
            try:
                cand_date = date(candidate_year, mth, d)
                if cand_date <= anchor_date:
                    dt_local = datetime(candidate_year, mth, d, 0, 0, 0, tzinfo=tz)
                    return TimePoint(value=dt_local.astimezone(UTC), precision=TimePrecision.DAY)
                candidate_year -= 1
            except ValueError:
                candidate_year -= 1

        return None

    def _match_row_3(self, p: str, tz: ZoneInfo) -> TimePoint | None:
        """Row 3: Month name and year."""
        if m := _ROW_3.match(p):
            month_key = m.group(1).lower()
            if month_key in _MONTH_NAMES:
                mth = _MONTH_NAMES[month_key]
                y = int(m.group(2))
                try:
                    dt_local = datetime(y, mth, 1, 0, 0, 0, tzinfo=tz)
                    return TimePoint(value=dt_local.astimezone(UTC), precision=TimePrecision.MONTH)
                except ValueError:
                    return None
        return None

    def _match_row_4(self, p: str, anchor_local: datetime, tz: ZoneInfo) -> TimePoint | None:
        """Row 4: Bare month (most recent such month whose start is <= anchor month start)."""
        if m := _ROW_4.match(p):
            month_key = m.group(1).lower()
            if month_key in _MONTH_NAMES:
                mth = _MONTH_NAMES[month_key]
                y = anchor_local.year if mth <= anchor_local.month else anchor_local.year - 1
                dt_local = datetime(y, mth, 1, 0, 0, 0, tzinfo=tz)
                return TimePoint(value=dt_local.astimezone(UTC), precision=TimePrecision.MONTH)
        return None

    def _match_row_5(self, p: str, anchor_local: datetime, tz: ZoneInfo) -> TimePoint | None:
        """Row 5: last MONTH (latest such month whose start is strictly < anchor month start)."""
        if m := _ROW_5.match(p):
            month_key = m.group(1).lower()
            if month_key in _MONTH_NAMES:
                mth = _MONTH_NAMES[month_key]
                y = anchor_local.year if mth < anchor_local.month else anchor_local.year - 1
                dt_local = datetime(y, mth, 1, 0, 0, 0, tzinfo=tz)
                return TimePoint(value=dt_local.astimezone(UTC), precision=TimePrecision.MONTH)
        return None

    def _match_row_6(self, p: str, anchor_local: datetime, tz: ZoneInfo) -> TimePoint | None:
        """Row 6: this MONTH (month within the anchor year)."""
        if m := _ROW_6.match(p):
            month_key = m.group(1).lower()
            if month_key in _MONTH_NAMES:
                mth = _MONTH_NAMES[month_key]
                y = anchor_local.year
                dt_local = datetime(y, mth, 1, 0, 0, 0, tzinfo=tz)
                return TimePoint(value=dt_local.astimezone(UTC), precision=TimePrecision.MONTH)
        return None

    def _match_row_7(self, p: str, tz: ZoneInfo) -> TimePoint | None:
        """Row 7: Year only."""
        if m := _ROW_7.match(p):
            y = int(m.group(1))
            dt_local = datetime(y, 1, 1, 0, 0, 0, tzinfo=tz)
            return TimePoint(value=dt_local.astimezone(UTC), precision=TimePrecision.YEAR)
        return None

    def _match_row_8(self, p: str, anchor_local: datetime, tz: ZoneInfo) -> TimePoint | None:
        """Row 8: today, tonight, this morning, yesterday, the day before yesterday."""
        if m := _ROW_8.match(p):
            token = re.sub(r'\s+', ' ', m.group(1).lower())
            anchor_date = anchor_local.date()
            if token in ('today', 'tonight', 'this morning', 'this afternoon', 'this evening'):
                target_date = anchor_date
            elif token == 'yesterday':
                target_date = anchor_date - timedelta(days=1)
            elif token == 'the day before yesterday':
                target_date = anchor_date - timedelta(days=2)
            else:
                return None
            dt_local = datetime(
                target_date.year, target_date.month, target_date.day, 0, 0, 0, tzinfo=tz
            )
            return TimePoint(value=dt_local.astimezone(UTC), precision=TimePrecision.DAY)
        return None

    def _match_row_9(self, p: str, anchor_local: datetime, tz: ZoneInfo) -> TimePoint | None:
        """Row 9: last/this week, month, year."""
        if m := _ROW_9.match(p):
            modifier = m.group(1).lower()
            unit = m.group(2).lower()
            if unit == 'week':
                this_monday = anchor_local.date() - timedelta(days=anchor_local.weekday())
                target_monday = (
                    this_monday if modifier == 'this' else this_monday - timedelta(days=7)
                )
                dt_local = datetime(
                    target_monday.year,
                    target_monday.month,
                    target_monday.day,
                    0,
                    0,
                    0,
                    tzinfo=tz,
                )
                return TimePoint(value=dt_local.astimezone(UTC), precision=TimePrecision.DAY)
            elif unit == 'month':
                if modifier == 'this':
                    y, mth = anchor_local.year, anchor_local.month
                else:
                    if anchor_local.month > 1:
                        y, mth = anchor_local.year, anchor_local.month - 1
                    else:
                        y, mth = anchor_local.year - 1, 12
                dt_local = datetime(y, mth, 1, 0, 0, 0, tzinfo=tz)
                return TimePoint(value=dt_local.astimezone(UTC), precision=TimePrecision.MONTH)
            elif unit == 'year':
                y = anchor_local.year if modifier == 'this' else anchor_local.year - 1
                dt_local = datetime(y, 1, 1, 0, 0, 0, tzinfo=tz)
                return TimePoint(value=dt_local.astimezone(UTC), precision=TimePrecision.YEAR)
        return None

    def _match_row_10(self, p: str, anchor_local: datetime, tz: ZoneInfo) -> TimePoint | None:
        """Row 10: N units ago (days, weeks, months, years; words or digits)."""
        if m := _ROW_10.match(p):
            num_str = m.group(1).lower()
            unit_str = m.group(2).lower()
            if num_str.isdigit():
                n = int(num_str)
            elif num_str in _WORD_NUMBERS:
                n = _WORD_NUMBERS[num_str]
            else:
                return None

            if unit_str.startswith('day'):
                target_date = anchor_local.date() - timedelta(days=n)
                dt_local = datetime(
                    target_date.year, target_date.month, target_date.day, 0, 0, 0, tzinfo=tz
                )
                return TimePoint(value=dt_local.astimezone(UTC), precision=TimePrecision.DAY)
            elif unit_str.startswith('week'):
                target_date = anchor_local.date() - timedelta(days=7 * n)
                dt_local = datetime(
                    target_date.year, target_date.month, target_date.day, 0, 0, 0, tzinfo=tz
                )
                return TimePoint(value=dt_local.astimezone(UTC), precision=TimePrecision.DAY)
            elif unit_str.startswith('month'):
                total_months = anchor_local.year * 12 + (anchor_local.month - 1) - n
                y = total_months // 12
                mth = (total_months % 12) + 1
                dt_local = datetime(y, mth, 1, 0, 0, 0, tzinfo=tz)
                return TimePoint(value=dt_local.astimezone(UTC), precision=TimePrecision.MONTH)
            elif unit_str.startswith('year'):
                y = anchor_local.year - n
                dt_local = datetime(y, 1, 1, 0, 0, 0, tzinfo=tz)
                return TimePoint(value=dt_local.astimezone(UTC), precision=TimePrecision.YEAR)

        return None

    def _match_row_11(self, p: str, anchor: datetime, tz: ZoneInfo) -> TimePoint | None:
        """Row 11: since X, from X (X as valid_from)."""
        if m := _ROW_11.match(p):
            x_phrase = m.group(1).strip()
            return self.resolve_point(x_phrase, anchor, tz)
        return None

    def _match_row_12(self, p: str, anchor: datetime, tz: ZoneInfo) -> TimePoint | None:
        """Row 12: until X, till X, through X (X as valid_to: exclusive end of X's period)."""
        if m := _ROW_12.match(p):
            x_phrase = m.group(1).strip()
            pt = self.resolve_point(x_phrase, anchor, tz)
            if pt is not None:
                return TimePoint(value=pt.span_end(), precision=pt.precision)
        return None

    def _match_row_13(self, p: str) -> bool:
        """Row 13: Seasons, recently, a while ago, long ago (unresolved: returns True to reject)."""
        return bool(_ROW_13_SEASONS.match(p) or _ROW_13_RELATIVE.match(p))

    def _match_row_14(self, p: str, anchor: datetime, tz: ZoneInfo) -> TimePoint | None:
        """Row 14: Fallback dateparser."""
        if not _ROW_14_DIGIT_OR_MONTH.search(p):
            return None

        try:
            settings: dict[str, Any] = {
                'RELATIVE_BASE': anchor.astimezone(tz).replace(tzinfo=None),
                'TIMEZONE': str(tz),
                'TO_TIMEZONE': 'UTC',
                'RETURN_AS_TIMEZONE_AWARE': True,
                'PREFER_DATES_FROM': 'past',
            }
            ddp = DateDataParser(settings=settings)
            data = ddp.get_date_data(p)
            if data is None or data.date_obj is None:
                return None

            dt_utc = data.date_obj.astimezone(UTC)
            local_dt = dt_utc.astimezone(tz)
            period = data.period

            if period == 'year':
                boundary = datetime(local_dt.year, 1, 1, 0, 0, 0, tzinfo=tz).astimezone(UTC)
                return TimePoint(value=boundary, precision=TimePrecision.YEAR)
            elif period == 'month':
                dt_month = datetime(local_dt.year, local_dt.month, 1, 0, 0, 0, tzinfo=tz)
                boundary = dt_month.astimezone(UTC)
                return TimePoint(value=boundary, precision=TimePrecision.MONTH)
            elif period == 'week':
                monday = local_dt.date() - timedelta(days=local_dt.weekday())
                boundary = datetime(
                    monday.year, monday.month, monday.day, 0, 0, 0, tzinfo=tz
                ).astimezone(UTC)
                return TimePoint(value=boundary, precision=TimePrecision.DAY)
            elif period == 'day':
                boundary = datetime(
                    local_dt.year, local_dt.month, local_dt.day, 0, 0, 0, tzinfo=tz
                ).astimezone(UTC)
                return TimePoint(value=boundary, precision=TimePrecision.DAY)
            else:
                return TimePoint(value=dt_utc, precision=TimePrecision.EXACT)
        except Exception:
            return None
