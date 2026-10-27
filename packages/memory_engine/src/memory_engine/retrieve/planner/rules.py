"""RulesPlanner, QueryPlan, EntityHinter and LLM trigger checks per LLD Section 9.2."""

import re
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Literal
from zoneinfo import ZoneInfo

from memory_engine.domain.enums import Intent
from memory_engine.domain.errors import InvariantViolation
from memory_engine.domain.ids import EntityId, UserId
from memory_engine.domain.intervals import TimeInterval
from memory_engine.ingest.time_resolver import TimeResolver
from memory_engine.retrieve.planner.cues import TemporalCueExtractor, TimeCue
from memory_engine.retrieve.planner.intent import IntentClassifier

__all__ = [
    'PREDICATE_KEYWORDS',
    'EntityHinter',
    'QueryPlan',
    'RulesPlanner',
    'needs_llm_planner',
]


def _validate_utc_datetime(dt: datetime, name: str) -> None:
    if not isinstance(dt, datetime):
        raise InvariantViolation(f'{name} must be a datetime, got {type(dt)!r}')
    if dt.tzinfo is None or dt.utcoffset() != timedelta(0):
        raise InvariantViolation(f'{name} must be timezone-aware UTC, got {dt!r}')


@dataclass(frozen=True, slots=True)
class QueryPlan:
    """Retrieval query plan per LLD Section 9.2 [FROZEN]."""

    question: str
    question_ts: datetime
    tz: str
    intent: Intent
    as_of: datetime | None
    time_range: TimeInterval | None
    entity_ids: tuple[EntityId, ...]
    entity_names: tuple[str, ...]
    predicates: tuple[str, ...]
    multi_hop: bool
    expects_count: bool
    ambiguous_time: bool
    planner_path: Literal['rules', 'llm', 'fallback']
    confidence: float

    def __post_init__(self) -> None:
        _validate_utc_datetime(self.question_ts, 'question_ts')
        if self.as_of is not None:
            _validate_utc_datetime(self.as_of, 'as_of')


PREDICATE_KEYWORDS: dict[str, tuple[str, ...]] = {
    'live': ('lives_in',),
    'lives': ('lives_in',),
    'lived': ('lives_in',),
    'living': ('lives_in',),
    'moved': ('lives_in',),
    'moved_to': ('lives_in',),
    'reside': ('lives_in',),
    'resides': ('lives_in',),
    'resided': ('lives_in',),
    'residing': ('lives_in',),
    'resides_in': ('lives_in',),
    'home': ('lives_in',),
    'city': ('lives_in',),
    'work': ('works_at', 'job_title'),
    'works': ('works_at', 'job_title'),
    'worked': ('works_at', 'job_title'),
    'working': ('works_at', 'job_title'),
    'job': ('works_at', 'job_title'),
    'role': ('job_title',),
    'title': ('job_title',),
    'position': ('job_title',),
    'company': ('works_at', 'job_title'),
    'employer': ('works_at', 'job_title'),
    'employed': ('works_at', 'job_title'),
    'employed_at': ('works_at',),
    'works_for': ('works_at',),
    'read': ('is_reading',),
    'reading': ('is_reading',),
    'reads': ('is_reading',),
    'book': ('is_reading',),
    'books': ('is_reading',),
    'study': ('studies_at',),
    'studies': ('studies_at',),
    'studied': ('studies_at',),
    'studying': ('studies_at',),
    'school': ('studies_at',),
    'college': ('studies_at',),
    'university': ('studies_at',),
    'born': ('born_in',),
    'birthplace': ('born_in',),
    'birthday': ('birthday',),
    'own': ('owns',),
    'owns': ('owns',),
    'owned': ('owns',),
    'owning': ('owns',),
    'bought': ('bought', 'owns'),
    'buy': ('bought',),
    'purchased': ('bought', 'owns'),
    'purchase': ('bought',),
    'like': ('likes',),
    'likes': ('likes',),
    'liked': ('likes',),
    'liking': ('likes',),
    'love': ('likes',),
    'loves': ('likes',),
    'loved': ('likes',),
    'enjoy': ('likes',),
    'enjoys': ('likes',),
    'enjoyed': ('likes',),
    'dislike': ('dislikes',),
    'dislikes': ('dislikes',),
    'disliked': ('dislikes',),
    'hate': ('dislikes',),
    'hates': ('dislikes',),
    'prefer': ('prefers',),
    'prefers': ('prefers',),
    'preferred': ('prefers',),
    'skill': ('has_skill',),
    'skills': ('has_skill',),
    'friend': ('friend_of',),
    'friends': ('friend_of',),
    'family': ('family_member',),
    'brother': ('family_member',),
    'sister': ('family_member',),
    'mother': ('family_member',),
    'father': ('family_member',),
    'parent': ('family_member',),
    'parents': ('family_member',),
    'plan': ('plans_to',),
    'plans': ('plans_to',),
    'planned': ('plans_to',),
    'planning': ('plans_to',),
    'intend': ('plans_to',),
    'intends': ('plans_to',),
    'intended': ('plans_to',),
    'attend': ('attended',),
    'attended': ('attended',),
    'visit': ('visited',),
    'visited': ('visited',),
    'travel': ('traveled_to',),
    'traveled': ('traveled_to',),
    'travelled': ('traveled_to',),
    'trip': ('traveled_to',),
    'decide': ('decided',),
    'decided': ('decided',),
    'meet': ('met',),
    'met': ('met',),
}

_FIRST_PERSON_PATTERN = re.compile(
    r'\b(?:i|me|my|myself|user|the\s+user)\b',
    re.IGNORECASE,
)

_POSSESSIVE_CHAIN_PATTERN = re.compile(
    r"\bmy\s+[\w'-]+['’]s\b",
    re.IGNORECASE,
)

_COUNT_EXPECTATION_PATTERN = re.compile(
    r'\b(?:how\s+many|how\s+much|count|number\s+of|total)\b',
    re.IGNORECASE,
)


class EntityHinter:
    """Matches 1-4 word n-grams of normalized question against alias index per LLD Section 9.2."""

    def __init__(self) -> None:
        self._cache: dict[UserId, tuple[float, dict[str, tuple[EntityId, str]]]] = {}

    @staticmethod
    def normalize(name: str) -> str:
        """Normalize per LLD Section 8.6 (NFKC, casefold, drop leading article, drop punct/s)."""
        t = unicodedata.normalize('NFKC', name).casefold().strip()
        t = re.sub(r'\s+', ' ', t)
        t = re.sub(r'^(?:the|a|an)\s+', '', t)
        t = re.sub(r'[^\w\s]+$', '', t)
        t = re.sub(r"['’]s$", '', t)
        return t.strip()

    def hint(
        self,
        question: str,
        user_id: UserId | None = None,
        aliases: Mapping[str, tuple[EntityId, str] | EntityId] | None = None,
        self_entity_id: EntityId | None = None,
        now_ts: float | None = None,
    ) -> tuple[tuple[EntityId, ...], tuple[str, ...]]:
        """Extract matching entity hints (ids and canonical names) from question."""
        entity_ids: list[EntityId] = []
        entity_names: list[str] = []

        # 1. First-person check
        if _FIRST_PERSON_PATTERN.search(question) and self_entity_id is not None:
            entity_ids.append(self_entity_id)
            entity_names.append('User')

        # 2. Build or fetch alias index
        alias_map: dict[str, tuple[EntityId, str]] = {}
        if aliases is not None:
            for k, v in aliases.items():
                norm_k = self.normalize(k)
                if isinstance(v, tuple):
                    alias_map[norm_k] = v
                else:
                    alias_map[norm_k] = (v, k)
        elif user_id is not None and user_id in self._cache:
            ts, cached_map = self._cache[user_id]
            current_time = now_ts if now_ts is not None else datetime.now().timestamp()
            if current_time - ts <= 300.0:  # 5 minutes TTL
                alias_map = cached_map

        if not alias_map:
            return tuple(entity_ids), tuple(entity_names)

        # 3. 1-4 word n-grams
        words = re.findall(r"\b[\w'-]+\b", question)
        matched_keys: set[str] = set()

        for n in (4, 3, 2, 1):
            for i in range(len(words) - n + 1):
                phrase = ' '.join(words[i : i + n])
                norm_phrase = self.normalize(phrase)
                if norm_phrase in alias_map and norm_phrase not in matched_keys:
                    eid, ename = alias_map[norm_phrase]
                    if eid not in entity_ids:
                        entity_ids.append(eid)
                        entity_names.append(ename)
                    matched_keys.add(norm_phrase)

        return tuple(entity_ids), tuple(entity_names)

    def update_cache(
        self,
        user_id: UserId,
        aliases: Mapping[str, tuple[EntityId, str] | EntityId],
        now_ts: float | None = None,
    ) -> None:
        """Cache up to 5000 aliases per user for 5 minutes."""
        current_time = now_ts if now_ts is not None else datetime.now().timestamp()
        alias_map: dict[str, tuple[EntityId, str]] = {}
        for count, (k, v) in enumerate(aliases.items()):
            if count >= 5000:
                break
            norm_k = self.normalize(k)
            if isinstance(v, tuple):
                alias_map[norm_k] = v
            else:
                alias_map[norm_k] = (v, k)
        self._cache[user_id] = (current_time, alias_map)


class RulesPlanner:
    """Deterministic rule-based query planner per LLD Section 9.2."""

    def __init__(
        self,
        cues_extractor: TemporalCueExtractor | None = None,
        intent_classifier: IntentClassifier | None = None,
        entity_hinter: EntityHinter | None = None,
        time_resolver: TimeResolver | None = None,
    ) -> None:
        self._resolver = time_resolver if time_resolver is not None else TimeResolver()
        self._cues = (
            cues_extractor if cues_extractor is not None else TemporalCueExtractor(self._resolver)
        )
        self._intent = intent_classifier if intent_classifier is not None else IntentClassifier()
        self._hinter = entity_hinter if entity_hinter is not None else EntityHinter()

    def plan(
        self,
        question: str,
        question_ts: datetime,
        tz: str | ZoneInfo = 'UTC',
        user_id: UserId | None = None,
        aliases: Mapping[str, tuple[EntityId, str] | EntityId] | None = None,
        self_entity_id: EntityId | None = None,
    ) -> QueryPlan:
        """Create a QueryPlan deterministically using rule-based components."""
        _validate_utc_datetime(question_ts, 'question_ts')
        tz_str = str(tz)
        tz_zone = ZoneInfo(tz_str) if isinstance(tz, str) else tz

        # 1. Temporal cue extraction
        cues = self._cues.extract(question, question_ts, tz_zone)
        ambiguous_time = any(c.interval is None for c in cues)

        # 2. Intent classification
        classification = self._intent.classify(question, cues)
        intent = classification.intent
        confidence = classification.confidence

        # 3. Time fields by intent per Section 9.2:
        # POINT_IN_TIME: as_of = cue start. INTERVAL: time_range = cue interval.
        # CURRENT_STATE and OPEN: neither. HISTORICAL: none.
        # DURATION_OR_ORDER and AGGREGATION: time_range only if a cue exists.
        as_of: datetime | None = None
        time_range: TimeInterval | None = None

        primary_cue = cues[0] if cues else None

        if intent == Intent.POINT_IN_TIME:
            if primary_cue is not None and primary_cue.interval is not None:
                as_of = primary_cue.interval.start
        elif intent == Intent.INTERVAL:
            if primary_cue is not None:
                time_range = primary_cue.interval
        elif intent in (Intent.CURRENT_STATE, Intent.OPEN, Intent.HISTORICAL):
            as_of = None
            time_range = None
        elif intent in (Intent.DURATION_OR_ORDER, Intent.AGGREGATION) and primary_cue is not None:
            time_range = primary_cue.interval

        # 4. Entity hints
        entity_ids, entity_names = self._hinter.hint(
            question,
            user_id=user_id,
            aliases=aliases,
            self_entity_id=self_entity_id,
        )

        # 5. Multi-hop: two or more non-self entity hints, or possessive relation chains
        non_self_count = sum(
            1 for eid in entity_ids if (self_entity_id is None or eid != self_entity_id)
        )
        if self_entity_id is None:
            # Fall back to entity_names
            non_self_count = sum(1 for name in entity_names if name.lower() not in ('user', 'self'))
        has_possessive_chain = bool(_POSSESSIVE_CHAIN_PATTERN.search(question))
        multi_hop = (non_self_count >= 2) or has_possessive_chain

        # 6. Predicate hints
        predicates = self._extract_predicates(question)

        # 7. Expects count
        expects_count = bool(_COUNT_EXPECTATION_PATTERN.search(question))

        return QueryPlan(
            question=question,
            question_ts=question_ts,
            tz=tz_str,
            intent=intent,
            as_of=as_of,
            time_range=time_range,
            entity_ids=entity_ids,
            entity_names=entity_names,
            predicates=predicates,
            multi_hop=multi_hop,
            expects_count=expects_count,
            ambiguous_time=ambiguous_time,
            planner_path='rules',
            confidence=confidence,
        )

    def _extract_predicates(self, question: str) -> tuple[str, ...]:
        words = re.findall(r'\b[a-zA-Z_]+\b', question.lower())
        results: list[str] = []

        # Check bigrams first, then unigrams
        for i in range(len(words) - 1):
            bigram = f'{words[i]}_{words[i + 1]}'
            if bigram in PREDICATE_KEYWORDS:
                for p in PREDICATE_KEYWORDS[bigram]:
                    if p not in results:
                        results.append(p)

        for w in words:
            if w in PREDICATE_KEYWORDS:
                for p in PREDICATE_KEYWORDS[w]:
                    if p not in results:
                        results.append(p)

        return tuple(results)


def needs_llm_planner(
    plan: QueryPlan,
    cues: Sequence[TimeCue] | None = None,
) -> bool:
    """Check the 4 FROZEN LLM planner trigger conditions from LLD Section 9.2:

    1. rules confidence below 0.5
    2. two or more distinct time cues
    3. duration_or_order with two or more entities
    4. a question over 40 words
    """
    # 1. Rules confidence below 0.5
    if plan.confidence < 0.5:
        return True

    # 2. Two or more distinct time cues
    if cues is None:
        tz_zone = ZoneInfo(plan.tz)
        extracted = TemporalCueExtractor().extract(plan.question, plan.question_ts, tz_zone)
        cues = extracted

    # Count distinct cues (either distinct resolved intervals or distinct cue texts)
    distinct_intervals = {c.interval for c in cues if c.interval is not None}
    unresolved_count = sum(1 for c in cues if c.interval is None)
    total_distinct_cues = len(distinct_intervals) + unresolved_count
    if total_distinct_cues >= 2 or len(cues) >= 2:
        return True

    # 3. duration_or_order with two or more entities
    if plan.intent == Intent.DURATION_OR_ORDER and len(set(plan.entity_names)) >= 2:
        return True

    # 4. A question over 40 words
    word_count = len(plan.question.strip().split())
    return word_count > 40
