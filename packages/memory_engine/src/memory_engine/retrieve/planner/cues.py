"""Temporal cue extraction for retrieval query planning per LLD Section 9.2."""

import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum
from zoneinfo import ZoneInfo

from memory_engine.domain.enums import TimePrecision
from memory_engine.domain.errors import InvariantViolation
from memory_engine.domain.intervals import TimeInterval
from memory_engine.ingest.time_resolver import TimeResolver

__all__ = [
    'CueKind',
    'TemporalCueExtractor',
    'TimeCue',
]


class CueKind(StrEnum):
    """Temporal cue kind: POINT for day or exact precision, RANGE otherwise."""

    POINT = 'point'
    RANGE = 'range'


@dataclass(frozen=True, slots=True)
class TimeCue:
    """Extracted temporal cue with resolved bounds."""

    text: str
    kind: CueKind
    interval: TimeInterval | None = None

    @property
    def start(self) -> datetime | None:
        return self.interval.start if self.interval is not None else None


def _validate_utc_datetime(dt: datetime, name: str) -> None:
    if not isinstance(dt, datetime):
        raise InvariantViolation(f'{name} must be a datetime, got {type(dt)!r}')
    if dt.tzinfo is None or dt.utcoffset() != timedelta(0):
        raise InvariantViolation(f'{name} must be timezone-aware UTC, got {dt!r}')


_RANGE_PREFIX_WORDS = (
    'since',
    'until',
    'till',
    'through',
    'before',
    'after',
    'between',
    'from',
)

_MONTH_NAMES_PATTERN = (
    r'(?:january|jan|february|feb|march|mar|april|apr|may|'
    r'june|jun|july|jul|august|aug|september|sep|sept|'
    r'october|oct|november|nov|december|dec)'
)

_CUE_PATTERNS: tuple[re.Pattern[str], ...] = (
    # 1. Complex ranges
    re.compile(
        r'\bbetween\s+[^,?.!]+?\s+and\s+[^,?.!]+?(?=\s+(?:and|or|was|were|did|is|are|in|at|to|with)\b|[?.!,]|$)',
        re.IGNORECASE,
    ),
    re.compile(
        r'\bfrom\s+[^,?.!]+?\s+(?:to|until|till)\s+[^,?.!]+?(?=\s+(?:and|or|was|were|did|is|are|in|at|to|with)\b|[?.!,]|$)',
        re.IGNORECASE,
    ),
    re.compile(
        r'\b(?:since|until|till|through|before|after)\s+[^,?.!]+?(?=\s+(?:and|or|was|were|did|is|are|in|at|to|with)\b|[?.!,]|$)',
        re.IGNORECASE,
    ),
    # 2. Point prefix cues
    re.compile(
        r'\bas\s+of\s+[^,?.!]+?(?=\s+(?:and|or|was|were|did|is|are|in|at|to|with)\b|[?.!,]|$)',
        re.IGNORECASE,
    ),
    re.compile(
        r'\bback\s+in\s+[^,?.!]+?(?=\s+(?:and|or|was|were|did|is|are|in|at|to|with)\b|[?.!,]|$)',
        re.IGNORECASE,
    ),
    # 3. ISO date / datetime
    re.compile(
        r'\b\d{4}-\d{2}-\d{2}(?:[T\s]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?)?\b',
        re.IGNORECASE,
    ),
    # 4. Day + Month + Year
    re.compile(
        r'\b(?:\d{1,2}(?:st|nd|rd|th)?\s+(?:of\s+)?'
        + _MONTH_NAMES_PATTERN
        + r'(?:\s*,?\s*\d{4}))\b',
        re.IGNORECASE,
    ),
    re.compile(
        r'\b(?:' + _MONTH_NAMES_PATTERN + r'\s+\d{1,2}(?:st|nd|rd|th)?(?:\s*,?\s*\d{4}))\b',
        re.IGNORECASE,
    ),
    # 5. Day + Month (no year)
    re.compile(
        r'\b(?:\d{1,2}(?:st|nd|rd|th)?\s+(?:of\s+)?' + _MONTH_NAMES_PATTERN + r')\b',
        re.IGNORECASE,
    ),
    re.compile(
        r'\b(?:' + _MONTH_NAMES_PATTERN + r'\s+\d{1,2}(?:st|nd|rd|th)?)\b',
        re.IGNORECASE,
    ),
    # 6. Month + Year
    re.compile(
        r'\b(?:(?:in\s+)?' + _MONTH_NAMES_PATTERN + r'\s+\d{4})\b',
        re.IGNORECASE,
    ),
    # 7. Last / this Month
    re.compile(
        r'\b(?:(?:last|this)\s+' + _MONTH_NAMES_PATTERN + r')\b',
        re.IGNORECASE,
    ),
    # 8. Bare month
    re.compile(
        r'\b(?:in\s+)?(january|jan|february|feb|march|mar|april|apr|june|jun|july|jul|august|aug|september|sep|sept|october|oct|november|nov|december|dec)\b',
        re.IGNORECASE,
    ),
    re.compile(
        r'\b(?:in\s+may|this\s+may|last\s+may)\b',
        re.IGNORECASE,
    ),
    # 9. Years only
    re.compile(
        r'\b(?:in\s+)?(19\d\d|20\d\d)\b',
        re.IGNORECASE,
    ),
    # 10. Relative periods
    re.compile(
        r'\b(?:last|this)\s+(?:week|month|year)\b',
        re.IGNORECASE,
    ),
    # 11. N units ago
    re.compile(
        r'\b(?:\w+|\d+)\s+(?:days?|weeks?|months?|years?)\s+ago\b',
        re.IGNORECASE,
    ),
    # 12. Relative days
    re.compile(
        r'\b(?:the\s+day\s+before\s+yesterday|yesterday|today|tonight|this\s+morning|this\s+afternoon|this\s+evening)\b',
        re.IGNORECASE,
    ),
    # 13. Ambiguous Row 13 phrases
    re.compile(
        r'\b(?:recently|a\s+while\s+ago|long\s+ago|just\s+now|lately|earlier|some\s+time\s+ago|spring|summer|autumn|fall|winter)\b',
        re.IGNORECASE,
    ),
)


class TemporalCueExtractor:
    """Extractor for temporal cues in retrieval queries per LLD Section 9.2."""

    def __init__(self, resolver: TimeResolver | None = None) -> None:
        self._resolver = resolver if resolver is not None else TimeResolver()

    def extract(
        self,
        question: str,
        anchor: datetime,
        tz: ZoneInfo,
    ) -> list[TimeCue]:
        """Scan question for temporal cues and resolve them against anchor."""
        _validate_utc_datetime(anchor, 'anchor')
        if not question or not question.strip():
            return []

        # Find all candidate spans
        matches: list[tuple[int, int, str]] = []
        for pattern in _CUE_PATTERNS:
            for m in pattern.finditer(question):
                start, end = m.span()
                matched_text = m.group(0).strip().strip('"\'')
                if matched_text:
                    matches.append((start, end, matched_text))

        if not matches:
            return []

        # Sort matches by span length descending so wider spans subsume inner spans
        matches.sort(key=lambda t: t[1] - t[0], reverse=True)

        accepted_spans: list[tuple[int, int, str]] = []
        for start, end, text in matches:
            # Check overlap with any already accepted wider span
            overlaps = any(
                not (end <= acc_start or start >= acc_end)
                for acc_start, acc_end, _ in accepted_spans
            )
            if not overlaps:
                accepted_spans.append((start, end, text))

        # Sort accepted spans by start position in original text
        accepted_spans.sort(key=lambda t: t[0])

        cues: list[TimeCue] = []
        for _, _, raw_text in accepted_spans:
            cue_text = raw_text.strip().rstrip('?.!,')
            cue = self._resolve_cue(cue_text, anchor, tz)
            if cue is not None:
                cues.append(cue)

        return cues

    def _resolve_cue(
        self,
        text: str,
        anchor: datetime,
        tz: ZoneInfo,
    ) -> TimeCue | None:
        lower = text.lower()

        # Check if text starts with range prefix
        is_range_prefix = any(lower.startswith(f'{w} ') for w in _RANGE_PREFIX_WORDS)

        # Check if text starts with point prefix
        is_point_prefix = lower.startswith('as of ') or lower.startswith('back in ')

        # Determine kind and resolve
        kind: CueKind
        if is_range_prefix:
            kind = CueKind.RANGE
        elif is_point_prefix:
            kind = CueKind.POINT
        else:
            pt = self._resolver.resolve_point(text, anchor, tz)
            if pt is not None and pt.precision in (TimePrecision.DAY, TimePrecision.EXACT):
                kind = CueKind.POINT
            else:
                kind = CueKind.RANGE

        # Resolve range
        interval = self._resolver.resolve_range(text, anchor, tz)
        if interval is None:
            # Try stripping leading preposition (in, on, as of, back in)
            cleaned = re.sub(r'^(?:in|on|as\s+of|back\s+in)\s+', '', text, flags=re.IGNORECASE)
            if cleaned != text:
                interval = self._resolver.resolve_range(cleaned, anchor, tz)

        return TimeCue(text=text, kind=kind, interval=interval)
