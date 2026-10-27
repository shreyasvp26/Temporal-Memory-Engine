"""Unit tests for retrieval QueryPlan, RulesPlanner, TemporalCueExtractor,

IntentClassifier, EntityHinter, and TemporalFilter.from_plan per LLD Section 9.2 and 9.3.
"""

from datetime import UTC, datetime, timedelta
from uuid import uuid4
from zoneinfo import ZoneInfo

import pytest
from memory_engine.domain.enums import Intent
from memory_engine.domain.errors import InvariantViolation
from memory_engine.domain.ids import EntityId, UserId
from memory_engine.retrieve import (
    CueKind,
    EntityHinter,
    FilterMode,
    IntentClassifier,
    RulesPlanner,
    TemporalCueExtractor,
    TemporalFilter,
    TimeCue,
    needs_llm_planner,
)


@pytest.fixture
def anchor() -> datetime:
    return datetime(2026, 10, 21, 12, 0, 0, tzinfo=UTC)


@pytest.fixture
def tz() -> ZoneInfo:
    return ZoneInfo('UTC')


@pytest.fixture
def planner() -> RulesPlanner:
    return RulesPlanner()


# ============================================================================
# 1. TemporalCueExtractor Tests
# ============================================================================


def test_cue_extractor_point_vs_range(anchor: datetime, tz: ZoneInfo) -> None:
    extractor = TemporalCueExtractor()

    # Day / exact precision -> CueKind.POINT
    cues_point = extractor.extract('Where was I on 2026-10-15?', anchor, tz)
    assert len(cues_point) == 1
    assert cues_point[0].kind == CueKind.POINT
    assert cues_point[0].interval is not None
    assert cues_point[0].interval.start == datetime(2026, 10, 15, 0, 0, tzinfo=UTC)

    # Yesterday -> CueKind.POINT
    cues_yesterday = extractor.extract('What did I do yesterday?', anchor, tz)
    assert len(cues_yesterday) == 1
    assert cues_yesterday[0].kind == CueKind.POINT
    assert cues_yesterday[0].interval is not None

    # Month-year -> CueKind.RANGE
    cues_range_month = extractor.extract('Where was I living in March 2025?', anchor, tz)
    assert len(cues_range_month) == 1
    assert cues_range_month[0].kind == CueKind.RANGE
    assert cues_range_month[0].interval is not None
    assert cues_range_month[0].interval.start == datetime(2025, 3, 1, 0, 0, tzinfo=UTC)

    # Year only -> CueKind.RANGE
    cues_range_year = extractor.extract('What company did I work for in 2024?', anchor, tz)
    assert len(cues_range_year) == 1
    assert cues_range_year[0].kind == CueKind.RANGE
    assert cues_range_year[0].interval is not None
    assert cues_range_year[0].interval.start == datetime(2024, 1, 1, 0, 0, tzinfo=UTC)


def test_cue_extractor_range_constructs(anchor: datetime, tz: ZoneInfo) -> None:
    extractor = TemporalCueExtractor()

    # between X and Y
    cues_between = extractor.extract('Where was I living between 2020 and 2022?', anchor, tz)
    assert len(cues_between) == 1
    assert cues_between[0].kind == CueKind.RANGE
    assert cues_between[0].interval is not None
    assert cues_between[0].interval.start == datetime(2020, 1, 1, 0, 0, tzinfo=UTC)
    assert cues_between[0].interval.end == datetime(2023, 1, 1, 0, 0, tzinfo=UTC)

    # since X
    cues_since = extractor.extract('Where have I worked since 2023?', anchor, tz)
    assert len(cues_since) == 1
    assert cues_since[0].kind == CueKind.RANGE
    assert cues_since[0].interval is not None
    assert cues_since[0].interval.start == datetime(2023, 1, 1, 0, 0, tzinfo=UTC)
    assert cues_since[0].interval.end is None

    # until X
    cues_until = extractor.extract('Where did I stay until yesterday?', anchor, tz)
    assert len(cues_until) == 1
    assert cues_until[0].kind == CueKind.RANGE
    assert cues_until[0].interval is not None
    assert cues_until[0].interval.start is None
    assert cues_until[0].interval.end is not None

    # after X
    cues_after = extractor.extract('Where did I live after 2023?', anchor, tz)
    assert len(cues_after) == 1
    assert cues_after[0].kind == CueKind.RANGE
    assert cues_after[0].interval is not None
    assert cues_after[0].interval.start == datetime(2024, 1, 1, 0, 0, tzinfo=UTC)
    assert cues_after[0].interval.end is None


def test_cue_extractor_unresolvable_ambiguous_cues(anchor: datetime, tz: ZoneInfo) -> None:
    extractor = TemporalCueExtractor()

    # "a while ago" -> Row 13 unresolvable cue
    cues_ambig = extractor.extract('What did I do a while ago?', anchor, tz)
    assert len(cues_ambig) == 1
    assert cues_ambig[0].interval is None

    # "recently" -> Row 13 unresolvable cue
    cues_rec = extractor.extract('Where did I travel recently?', anchor, tz)
    assert len(cues_rec) == 1
    assert cues_rec[0].interval is None


def test_cue_extractor_subsumption_and_multiple_cues(anchor: datetime, tz: ZoneInfo) -> None:
    extractor = TemporalCueExtractor()

    # "since March 2024" should subsume "March 2024"
    cues_sub = extractor.extract('Where have I lived since March 2024?', anchor, tz)
    assert len(cues_sub) == 1
    assert 'since' in cues_sub[0].text.lower()
    assert cues_sub[0].kind == CueKind.RANGE

    # Multiple distinct cues
    cues_multi = extractor.extract('Where was I in March 2024 and in October 2025?', anchor, tz)
    assert len(cues_multi) == 2
    assert cues_multi[0].interval != cues_multi[1].interval


def test_cue_extractor_empty_or_no_cues(anchor: datetime, tz: ZoneInfo) -> None:
    extractor = TemporalCueExtractor()
    assert extractor.extract('', anchor, tz) == []
    assert extractor.extract('   ', anchor, tz) == []
    assert extractor.extract('What is my favorite programming language?', anchor, tz) == []


# ============================================================================
# 2. IntentClassifier Tests (All 7 Priorities & Confidence)
# ============================================================================


def test_intent_priority_1_duration_or_order() -> None:
    classifier = IntentClassifier()

    triggers = [
        ('How long did I stay in Berlin?', ()),
        ('How many days did I spend in Tokyo?', ()),
        ('How many weeks was the project delayed?', ()),
        ('Between Alice and Bob who joined first?', ()),
        ('Which came first the egg or the chicken?', ()),
        ('Which first occurred?', ()),
        ('How long ago did I visit Paris?', ()),
        ('Since when have I been working here?', ()),
        ('When was the first time I went skiing?', ()),
        ('What was the last time I saw him?', ()),
        ('Was my trip earlier than June?', ()),
        ('Did I start later than September?', ()),
    ]

    for q, cues in triggers:
        res = classifier.classify(q, cues)
        assert res.intent == Intent.DURATION_OR_ORDER, f'Failed for question: {q}'
        assert 1 in res.matched_families


def test_intent_priority_2_aggregation() -> None:
    classifier = IntentClassifier()

    triggers = [
        'How many books did I read?',
        'How much money did I spend on groceries?',
        'List all cities I have visited',
        'List every movie I watched',
        'What was the total expenditure in 2025?',
        'What is the number of trips taken?',
        'Count all the certifications I earned',
    ]

    for q in triggers:
        res = classifier.classify(q)
        assert res.intent == Intent.AGGREGATION, f'Failed for question: {q}'
        assert 2 in res.matched_families


def test_intent_priority_3_historical() -> None:
    classifier = IntentClassifier()

    triggers = [
        'Where did I live before?',
        'What was previously my role?',
        'Where did I use to work?',
        'What was formerly my address?',
        'Where was I prior to moving?',
        'What was originally planned?',
        'What car did I drive back then?',
        'Where did I work at first?',
        'What do I no longer do?',
        'What did I study when I was in college?',
        'Where did I live in the past?',
    ]

    for q in triggers:
        res = classifier.classify(q, cues=())
        assert res.intent == Intent.HISTORICAL, f'Failed for question: {q}'
        assert 3 in res.matched_families

    # When an explicit POINT cue is present, priority 3 does NOT fire
    point_cue = TimeCue(text='2026-10-15', kind=CueKind.POINT)
    res_with_point = classifier.classify('Where did I live before 2026-10-15?', cues=[point_cue])
    # Point cue triggers family 4
    assert res_with_point.intent == Intent.POINT_IN_TIME
    assert 3 not in res_with_point.matched_families


def test_intent_priority_4_point_in_time() -> None:
    classifier = IntentClassifier()

    # Via POINT cue
    point_cue = TimeCue(text='yesterday', kind=CueKind.POINT)
    res_pt = classifier.classify('What did I do yesterday?', cues=[point_cue])
    assert res_pt.intent == Intent.POINT_IN_TIME
    assert 4 in res_pt.matched_families

    # Via 'at that time'
    res_att = classifier.classify('Where was I working at that time?')
    assert res_att.intent == Intent.POINT_IN_TIME
    assert 4 in res_att.matched_families

    # Via 'as of' followed by cue
    range_cue = TimeCue(text='March 2025', kind=CueKind.RANGE)
    res_as_of = classifier.classify('Where was I living as of March 2025?', cues=[range_cue])
    assert res_as_of.intent == Intent.POINT_IN_TIME
    assert 4 in res_as_of.matched_families

    # Via 'back in' followed by cue
    res_back_in = classifier.classify('Where was I back in 2024?', cues=[range_cue])
    assert res_back_in.intent == Intent.POINT_IN_TIME
    assert 4 in res_back_in.matched_families


def test_intent_priority_5_interval() -> None:
    classifier = IntentClassifier()

    range_cue = TimeCue(text='in 2024', kind=CueKind.RANGE)
    res = classifier.classify('Where was I living in 2024?', cues=[range_cue])
    assert res.intent == Intent.INTERVAL
    assert 5 in res.matched_families


def test_intent_priority_6_current_state() -> None:
    classifier = IntentClassifier()

    triggers = [
        'Where do I live?',
        'What is my phone number?',
        'Am I currently enrolled in school?',
        'Do I currently have any pet?',
        'What is currently my role?',
        'Where do I work right now?',
        'Where do I spend my time these days?',
        'What is my current address?',
        'Who is my manager?',
    ]

    for q in triggers:
        res = classifier.classify(q)
        assert res.intent == Intent.CURRENT_STATE, f'Failed for question: {q}'
        assert 6 in res.matched_families


def test_intent_priority_7_open() -> None:
    classifier = IntentClassifier()

    res = classifier.classify('Tell me a fun story')
    assert res.intent == Intent.OPEN
    assert len(res.matched_families) == 0


def test_intent_priority_order_resolution() -> None:
    classifier = IntentClassifier()

    # Priority 1 beats Priority 2: "how many days" matches family 1 and family 2
    res_1_2 = classifier.classify('How many days did I spend in London?')
    assert res_1_2.intent == Intent.DURATION_OR_ORDER
    assert 1 in res_1_2.matched_families
    assert 2 in res_1_2.matched_families

    # Priority 2 beats Priority 5: "how many books in 2024" matches family 2 and family 5
    range_cue = TimeCue(text='in 2024', kind=CueKind.RANGE)
    res_2_5 = classifier.classify('How many books did I read in 2024?', cues=[range_cue])
    assert res_2_5.intent == Intent.AGGREGATION
    assert 2 in res_2_5.matched_families
    assert 5 in res_2_5.matched_families

    # Priority 1 beats Priority 6: "how long" + "where do I"
    res_1_6 = classifier.classify('Where do I live and how long have I been there?')
    assert res_1_6.intent == Intent.DURATION_OR_ORDER
    assert 1 in res_1_6.matched_families
    assert 6 in res_1_6.matched_families


def test_confidence_scoring() -> None:
    classifier = IntentClassifier()

    # 1 family -> 0.9
    res_single = classifier.classify('Where do I live right now?')
    assert res_single.confidence == 0.9
    assert len(res_single.matched_families) == 1

    # Several families -> 0.6
    res_multi = classifier.classify(
        'How many books did I read in 2024?',
        cues=[TimeCue(text='in 2024', kind=CueKind.RANGE)],
    )
    assert res_multi.confidence == 0.6
    assert len(res_multi.matched_families) >= 2

    # Open intent < 12 words -> 0.9
    res_open_short = classifier.classify('What is that thing?')
    assert res_open_short.intent == Intent.OPEN
    assert res_open_short.confidence == 0.9

    # Open intent >= 12 words -> 0.4
    long_q = (
        'Can you give me detailed advice regarding general career progression '
        'in various emerging software engineering subfields?'
    )
    assert len(long_q.split()) >= 12
    res_open_long = classifier.classify(long_q)
    assert res_open_long.intent == Intent.OPEN
    assert res_open_long.confidence == 0.4


# ============================================================================
# 3. EntityHinter Tests
# ============================================================================


def test_entity_hinter_normalization() -> None:
    assert EntityHinter.normalize('  The Great Gatsby  ') == 'great gatsby'
    assert EntityHinter.normalize('Alice’s') == 'alice'
    assert EntityHinter.normalize("Bob's") == 'bob'
    assert EntityHinter.normalize('A Corporation,') == 'corporation'
    assert EntityHinter.normalize('An Apple.') == 'apple'


def test_entity_hinter_first_person_and_ngrams() -> None:
    hinter = EntityHinter()
    self_id = EntityId(uuid4())
    e1_id = EntityId(uuid4())
    e2_id = EntityId(uuid4())

    aliases = {
        'New York City': (e1_id, 'New York City'),
        'Acme Corp': (e2_id, 'Acme Corp'),
    }

    # First person pronoun matches self entity
    q1 = 'Where do I live?'
    ids1, names1 = hinter.hint(q1, aliases=aliases, self_entity_id=self_id)
    assert ids1 == (self_id,)
    assert names1 == ('User',)

    # 3-word n-gram "New York City"
    q2 = 'When did I visit New York City with my family?'
    ids2, names2 = hinter.hint(q2, aliases=aliases, self_entity_id=self_id)
    assert self_id in ids2
    assert e1_id in ids2
    assert 'New York City' in names2

    # 2-word n-gram "Acme Corp"
    q3 = 'Do I work for Acme Corp?'
    ids3, names3 = hinter.hint(q3, aliases=aliases, self_entity_id=self_id)
    assert e2_id in ids3
    assert 'Acme Corp' in names3


def test_entity_hinter_caching() -> None:
    hinter = EntityHinter()
    user_id = UserId(uuid4())
    e_id = EntityId(uuid4())

    hinter.update_cache(user_id, {'pune': (e_id, 'Pune')}, now_ts=1000.0)

    # Within 300s TTL (at 1200s -> 200s elapsed)
    ids, names = hinter.hint('I moved to Pune', user_id=user_id, now_ts=1200.0)
    assert ids == (e_id,)
    assert names == ('Pune',)

    # Expired cache at 1400s (400s elapsed > 300s)
    ids_exp, names_exp = hinter.hint('I moved to Pune', user_id=user_id, now_ts=1400.0)
    assert ids_exp == ()
    assert names_exp == ()


# ============================================================================
# 4. Multi-hop and Predicate Hints Tests
# ============================================================================


def test_multi_hop_detection(planner: RulesPlanner, anchor: datetime) -> None:
    self_id = EntityId(uuid4())
    e1_id = EntityId(uuid4())
    e2_id = EntityId(uuid4())

    aliases = {
        'Alice': (e1_id, 'Alice'),
        'Bob': (e2_id, 'Bob'),
    }

    # Two non-self entities -> multi_hop = True
    plan1 = planner.plan(
        'Did Alice and Bob work together?',
        anchor,
        aliases=aliases,
        self_entity_id=self_id,
    )
    assert plan1.multi_hop is True

    # One non-self entity + self entity -> multi_hop = False
    plan2 = planner.plan(
        'Did I meet Alice?',
        anchor,
        aliases=aliases,
        self_entity_id=self_id,
    )
    assert plan2.multi_hop is False

    # Possessive relation chain "my friend's" -> multi_hop = True
    plan3 = planner.plan(
        "What is my friend's job title?",
        anchor,
        self_entity_id=self_id,
    )
    assert plan3.multi_hop is True

    # Possessive relation chain "my colleague's" -> multi_hop = True
    plan4 = planner.plan(
        "Where does my colleague's company operate?",
        anchor,
        self_entity_id=self_id,
    )
    assert plan4.multi_hop is True


def test_predicate_hints(planner: RulesPlanner, anchor: datetime) -> None:
    # live/home -> lives_in
    p1 = planner.plan('Where do I live and where is my home?', anchor)
    assert 'lives_in' in p1.predicates

    # work/company/job -> works_at, job_title
    p2 = planner.plan('What company do I work for and what is my job?', anchor)
    assert 'works_at' in p2.predicates
    assert 'job_title' in p2.predicates

    # read/book -> is_reading
    p3 = planner.plan('What book am I reading?', anchor)
    assert 'is_reading' in p3.predicates

    # study/college -> studies_at
    p4 = planner.plan('Which college did I study at?', anchor)
    assert 'studies_at' in p4.predicates


# ============================================================================
# 5. RulesPlanner.plan and Time Fields by Intent
# ============================================================================


def test_rules_planner_time_fields_per_intent(planner: RulesPlanner, anchor: datetime) -> None:
    # POINT_IN_TIME: as_of = cue start, time_range = None
    p_point = planner.plan('Where was I on 2026-10-15?', anchor)
    assert p_point.intent == Intent.POINT_IN_TIME
    assert p_point.as_of == datetime(2026, 10, 15, 0, 0, tzinfo=UTC)
    assert p_point.time_range is None

    # INTERVAL: time_range = cue interval, as_of = None
    p_int = planner.plan('Where did I live in 2024?', anchor)
    assert p_int.intent == Intent.INTERVAL
    assert p_int.as_of is None
    assert p_int.time_range is not None
    assert p_int.time_range.start == datetime(2024, 1, 1, 0, 0, tzinfo=UTC)

    # CURRENT_STATE: neither
    p_cur = planner.plan('Where do I live right now?', anchor)
    assert p_cur.intent == Intent.CURRENT_STATE
    assert p_cur.as_of is None
    assert p_cur.time_range is None

    # OPEN: neither
    p_open = planner.plan('Tell me something interesting.', anchor)
    assert p_open.intent == Intent.OPEN
    assert p_open.as_of is None
    assert p_open.time_range is None

    # HISTORICAL: none
    p_hist = planner.plan('Where did I previously work?', anchor)
    assert p_hist.intent == Intent.HISTORICAL
    assert p_hist.as_of is None
    assert p_hist.time_range is None

    # DURATION_OR_ORDER: time_range only if cue exists
    p_dur_cue = planner.plan('How long did I stay in Rome in 2024?', anchor)
    assert p_dur_cue.intent == Intent.DURATION_OR_ORDER
    assert p_dur_cue.time_range is not None
    assert p_dur_cue.as_of is None

    p_dur_no_cue = planner.plan('How long have I worked here?', anchor)
    assert p_dur_no_cue.intent == Intent.DURATION_OR_ORDER
    assert p_dur_no_cue.time_range is None
    assert p_dur_no_cue.as_of is None

    # AGGREGATION: time_range only if cue exists
    p_agg_cue = planner.plan('How many books did I read in 2024?', anchor)
    assert p_agg_cue.intent == Intent.AGGREGATION
    assert p_agg_cue.time_range is not None

    p_agg_no_cue = planner.plan('How many books did I read?', anchor)
    assert p_agg_no_cue.intent == Intent.AGGREGATION
    assert p_agg_no_cue.time_range is None


def test_rules_planner_ambiguous_time_and_count(planner: RulesPlanner, anchor: datetime) -> None:
    # Ambiguous relative cue sets ambiguous_time = True
    p_ambig = planner.plan('Where did I travel recently?', anchor)
    assert p_ambig.ambiguous_time is True

    # Resolved date has ambiguous_time = False
    p_clear = planner.plan('Where was I on 2026-10-15?', anchor)
    assert p_clear.ambiguous_time is False

    # Expects count
    p_count = planner.plan('Count the number of books read', anchor)
    assert p_count.expects_count is True

    p_no_count = planner.plan('Where do I live?', anchor)
    assert p_no_count.expects_count is False


def test_rules_planner_utc_enforcement(planner: RulesPlanner) -> None:
    naive_dt = datetime(2026, 10, 21, 12, 0, 0)
    with pytest.raises(InvariantViolation, match='must be timezone-aware UTC'):
        planner.plan('Where do I live?', naive_dt)


# ============================================================================
# 6. LLM Planner Trigger Conditions Tests
# ============================================================================


def test_needs_llm_planner_triggers(planner: RulesPlanner, anchor: datetime) -> None:
    e1_id = EntityId(uuid4())
    e2_id = EntityId(uuid4())
    aliases = {'Alice': (e1_id, 'Alice'), 'Bob': (e2_id, 'Bob')}

    # Condition 1: confidence < 0.5 (open question with >= 12 words)
    long_open = (
        'Can you provide a very comprehensive overview of multiple unrelated topics '
        'without any specific temporal cues or entities mentioned?'
    )
    p_low_conf = planner.plan(long_open, anchor)
    assert p_low_conf.confidence < 0.5
    assert needs_llm_planner(p_low_conf) is True

    # Condition 2: two or more distinct time cues
    p_two_cues = planner.plan('Where was I in March 2024 and October 2025?', anchor)
    assert needs_llm_planner(p_two_cues) is True

    # Condition 3: duration_or_order with two or more entities
    p_dur_two_ent = planner.plan(
        'Between Alice and Bob who worked longer at the firm?',
        anchor,
        aliases=aliases,
    )
    assert p_dur_two_ent.intent == Intent.DURATION_OR_ORDER
    assert len(p_dur_two_ent.entity_names) >= 2
    assert needs_llm_planner(p_dur_two_ent) is True

    # Condition 4: question over 40 words
    long_question = (
        'I would like to ask a very long and detailed question about my past experiences '
        'and various activities that I may have participated in over the past several '
        'years of my career, including projects, meetings, team interactions, and other '
        'notable events that happened throughout that period.'
    )
    assert len(long_question.split()) > 40
    p_long = planner.plan(long_question, anchor)
    assert needs_llm_planner(p_long) is True

    # Standard clean rules query does NOT trigger LLM planner
    p_clean = planner.plan('Where do I live right now?', anchor)
    assert p_clean.confidence == 0.9
    assert needs_llm_planner(p_clean) is False


# ============================================================================
# 7. TemporalFilter.from_plan Tests with Real QueryPlan
# ============================================================================


def test_temporal_filter_from_real_plan(planner: RulesPlanner, anchor: datetime) -> None:
    # 1. CURRENT_STATE -> CURRENT
    p_cur = planner.plan('Where do I live right now?', anchor)
    f_cur = TemporalFilter.from_plan(p_cur)
    assert f_cur.mode == FilterMode.CURRENT

    # 2. POINT_IN_TIME -> AS_OF
    p_point = planner.plan('Where was I on 2026-10-15?', anchor)
    f_point = TemporalFilter.from_plan(p_point)
    assert f_point.mode == FilterMode.AS_OF
    assert f_point.as_of == datetime(2026, 10, 15, 0, 0, tzinfo=UTC)

    # 3. INTERVAL -> OVERLAP
    p_int = planner.plan('Where was I in 2024?', anchor)
    f_int = TemporalFilter.from_plan(p_int)
    assert f_int.mode == FilterMode.OVERLAP
    assert f_int.range is not None

    # 4. HISTORICAL -> ANY_HISTORY
    p_hist = planner.plan('Where did I previously work?', anchor)
    f_hist = TemporalFilter.from_plan(p_hist)
    assert f_hist.mode == FilterMode.ANY_HISTORY

    # 5. DURATION_OR_ORDER with range cue -> OVERLAP
    p_dur_cue = planner.plan('How long did I live in Rome in 2024?', anchor)
    f_dur_cue = TemporalFilter.from_plan(p_dur_cue)
    assert f_dur_cue.mode == FilterMode.OVERLAP

    # 6. DURATION_OR_ORDER without range -> ANY_HISTORY
    p_dur_no = planner.plan('How long have I worked here?', anchor)
    f_dur_no = TemporalFilter.from_plan(p_dur_no)
    assert f_dur_no.mode == FilterMode.ANY_HISTORY

    # 7. AGGREGATION without range -> ANY_HISTORY
    p_agg_no = planner.plan('How many books did I read?', anchor)
    f_agg_no = TemporalFilter.from_plan(p_agg_no)
    assert f_agg_no.mode == FilterMode.ANY_HISTORY

    # 8. OPEN -> OPEN (cutoff = anchor - 365 days)
    p_open = planner.plan('Tell me a story.', anchor)
    f_open = TemporalFilter.from_plan(p_open)
    assert f_open.mode == FilterMode.OPEN
    assert f_open.cutoff == anchor - timedelta(days=365)
