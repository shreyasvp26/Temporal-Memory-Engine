"""Intent classification for retrieval queries per LLD Section 9.2."""

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from memory_engine.domain.enums import Intent
from memory_engine.retrieve.planner.cues import CueKind, TimeCue

__all__ = [
    'IntentClassification',
    'IntentClassifier',
]


@dataclass(frozen=True, slots=True)
class IntentClassification:
    """Result of intent classification with confidence and matched rule families."""

    intent: Intent
    confidence: float
    matched_families: tuple[int, ...] = ()

    def __iter__(self) -> Any:
        return iter((self.intent, self.confidence))


_FAMILY_1_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r'\bhow\s+long\b', re.IGNORECASE),
    re.compile(r'\bhow\s+many\s+(?:days|weeks|months|years)\b', re.IGNORECASE),
    re.compile(r'\bbetween\b', re.IGNORECASE),
    re.compile(r'\bwhich\s+(?:came\s+)?first\b', re.IGNORECASE),
    re.compile(r'\bwhich\s+first\b', re.IGNORECASE),
    re.compile(r'\bhow\s+long\s+ago\b', re.IGNORECASE),
    re.compile(r'\bsince\s+when\b', re.IGNORECASE),
    re.compile(r'\bfirst\s+time\b', re.IGNORECASE),
    re.compile(r'\blast\s+time\b', re.IGNORECASE),
    re.compile(r'\bearlier\s+than\b', re.IGNORECASE),
    re.compile(r'\blater\s+than\b', re.IGNORECASE),
)

_FAMILY_2_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r'\bhow\s+many\b', re.IGNORECASE),
    re.compile(r'\bhow\s+much\b', re.IGNORECASE),
    re.compile(r'\blist\s+all\b', re.IGNORECASE),
    re.compile(r'\blist\s+every\b', re.IGNORECASE),
    re.compile(r'\btotal\b', re.IGNORECASE),
    re.compile(r'\bnumber\s+of\b', re.IGNORECASE),
    re.compile(r'\bcount\b', re.IGNORECASE),
)

_FAMILY_3_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r'\bbefore\b', re.IGNORECASE),
    re.compile(r'\bpreviously\b', re.IGNORECASE),
    re.compile(r'\bused?\s+to\b', re.IGNORECASE),
    re.compile(r'\bformerly\b', re.IGNORECASE),
    re.compile(r'\bprior\s+to\b', re.IGNORECASE),
    re.compile(r'\boriginally\b', re.IGNORECASE),
    re.compile(r'\bback\s+then\b', re.IGNORECASE),
    re.compile(r'\bat\s+first\b', re.IGNORECASE),
    re.compile(r'\bno\s+longer\b', re.IGNORECASE),
    re.compile(r'\bwhen\s+i\s+was\b', re.IGNORECASE),
    re.compile(r'\bin\s+the\s+past\b', re.IGNORECASE),
)

_FAMILY_4_AT_THAT_TIME = re.compile(r'\bat\s+that\s+time\b', re.IGNORECASE)
_FAMILY_4_PREFIX = re.compile(r'\b(?:as\s+of|back\s+in)\b', re.IGNORECASE)

_FAMILY_6_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r'\bwhere\s+do\s+i\b', re.IGNORECASE),
    re.compile(r'\bwhat\s+is\s+my\b', re.IGNORECASE),
    re.compile(r'\bam\s+i\b', re.IGNORECASE),
    re.compile(r'\bdo\s+i\s+currently\b', re.IGNORECASE),
    re.compile(r'\bcurrently\b', re.IGNORECASE),
    re.compile(r'\bright\s+now\b', re.IGNORECASE),
    re.compile(r'\bthese\s+days\b', re.IGNORECASE),
    re.compile(r'\bmy\s+current\b', re.IGNORECASE),
    re.compile(r'\bwho\s+is\s+my\b', re.IGNORECASE),
)


class IntentClassifier:
    """Query retrieval intent classifier implementing priority rules 1-7 per LLD Section 9.2."""

    def classify(
        self,
        question: str,
        cues: Sequence[TimeCue] = (),
    ) -> IntentClassification:
        """Classify retrieval query intent and confidence score."""
        q = question.strip()
        has_point_cue = any(c.kind == CueKind.POINT for c in cues)
        has_range_cue = any(c.kind == CueKind.RANGE for c in cues)

        matched_families: list[int] = []

        # Family 1: duration_or_order
        if any(p.search(q) for p in _FAMILY_1_PATTERNS):
            matched_families.append(1)

        # Family 2: aggregation
        if any(p.search(q) for p in _FAMILY_2_PATTERNS):
            matched_families.append(2)

        # Family 3: historical (with no explicit point cue)
        if (not has_point_cue) and any(p.search(q) for p in _FAMILY_3_PATTERNS):
            matched_families.append(3)

        # Family 4: point_in_time (a POINT cue, or as of, at that time, back in followed by a cue)
        f4_matches = (
            has_point_cue
            or bool(_FAMILY_4_AT_THAT_TIME.search(q))
            or (bool(cues) and bool(_FAMILY_4_PREFIX.search(q)))
        )
        if f4_matches:
            matched_families.append(4)

        # Family 5: interval (any RANGE cue)
        if has_range_cue:
            matched_families.append(5)

        # Family 6: current_state
        if any(p.search(q) for p in _FAMILY_6_PATTERNS):
            matched_families.append(6)

        # Determine Intent by strict priority order 1 to 7
        intent: Intent
        if 1 in matched_families:
            intent = Intent.DURATION_OR_ORDER
        elif 2 in matched_families:
            intent = Intent.AGGREGATION
        elif 3 in matched_families:
            intent = Intent.HISTORICAL
        elif 4 in matched_families:
            intent = Intent.POINT_IN_TIME
        elif 5 in matched_families:
            intent = Intent.INTERVAL
        elif 6 in matched_families:
            intent = Intent.CURRENT_STATE
        else:
            intent = Intent.OPEN

        # Compute confidence: 0.9 for 1 family, 0.6 for several, 0.4 for open >= 12 words
        confidence: float
        if len(matched_families) == 1:
            confidence = 0.9
        elif len(matched_families) > 1:
            confidence = 0.6
        else:
            # 0 families: Intent.OPEN
            word_count = len(q.split())
            confidence = 0.4 if word_count >= 12 else 0.9

        return IntentClassification(
            intent=intent,
            confidence=confidence,
            matched_families=tuple(matched_families),
        )
