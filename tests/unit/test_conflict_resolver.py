"""Unit tests and property tests for ConflictResolver and resolution strategies (M2-03)."""

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from memory_engine.domain.enums import (
    Cardinality,
    FactOrigin,
    FactStatus,
    RelationType,
    TimePrecision,
)
from memory_engine.domain.errors import InvariantViolation
from memory_engine.domain.ids import EntityId, FactId, SourceId, UserId
from memory_engine.domain.models import Candidate, Fact, JudgeHint
from memory_engine.ingest.conflict import (
    AddEvidence,
    AddRelation,
    BumpConfidence,
    CloseFact,
    ConflictResolver,
    ExtendValidFrom,
    InsertFact,
    NewRef,
    Outcome,
    ResolutionContext,
    ResolutionPlan,
    SetStatus,
    StrategyRegistry,
    default_registry,
)


def utc(
    year: int,
    month: int,
    day: int,
    hour: int = 0,
    minute: int = 0,
    second: int = 0,
) -> datetime:
    """Helper creating timezone-aware UTC datetime."""
    return datetime(year, month, day, hour, minute, second, tzinfo=UTC)


def make_fact(
    *,
    fact_id: FactId | None = None,
    user_id: UserId | None = None,
    subject_id: EntityId | None = None,
    predicate: str = 'lives_in',
    cardinality: Cardinality = Cardinality.SINGLE,
    object_entity_id: EntityId | None = None,
    object_value: str | None = None,
    object_norm: str = 'pune',
    fact_text: str = 'User lives in Pune',
    valid_from: datetime | None = None,
    valid_to: datetime | None = None,
    time_precision: TimePrecision = TimePrecision.DAY,
    valid_from_explicit: bool = True,
    recorded_at: datetime | None = None,
    invalidated_at: datetime | None = None,
    confidence: float = 0.9,
    salience: float = 1.0,
    access_count: int = 0,
    last_accessed_at: datetime | None = None,
    status: FactStatus = FactStatus.ACTIVE,
    origin: FactOrigin = FactOrigin.EXTRACTED,
    event_hash: str | None = None,
) -> Fact:
    """Factory helper creating valid Fact instances."""
    val = object_value
    if object_entity_id is None and val is None:
        val = object_norm
    vf = valid_from if valid_from is not None else utc(2026, 1, 1)
    rec = recorded_at if recorded_at is not None else utc(2026, 1, 1)
    return Fact(
        id=fact_id or FactId(uuid4()),
        user_id=user_id or UserId(uuid4()),
        subject_id=subject_id or EntityId(uuid4()),
        predicate=predicate,
        cardinality=cardinality,
        object_entity_id=object_entity_id,
        object_value=val,
        object_norm=object_norm,
        fact_text=fact_text,
        valid_from=vf,
        valid_to=valid_to,
        time_precision=time_precision,
        valid_from_explicit=valid_from_explicit,
        recorded_at=rec,
        invalidated_at=invalidated_at,
        confidence=confidence,
        salience=salience,
        access_count=access_count,
        last_accessed_at=last_accessed_at,
        status=status,
        origin=origin,
        event_hash=event_hash,
    )


def make_candidate(
    *,
    user_id: UserId | None = None,
    subject_id: EntityId | None = None,
    predicate: str = 'lives_in',
    cardinality: Cardinality = Cardinality.SINGLE,
    object_entity_id: EntityId | None = None,
    object_value: str | None = None,
    object_norm: str = 'delhi',
    fact_text: str = 'User lives in Delhi',
    valid_from: datetime | None = None,
    valid_to: datetime | None = None,
    time_precision: TimePrecision = TimePrecision.DAY,
    valid_from_explicit: bool = True,
    confidence: float = 0.9,
    is_retraction: bool = False,
    source_id: SourceId | None = None,
    span: tuple[int, int] | None = None,
    source_ts: datetime | None = None,
    event_hash: str | None = None,
    embedding: list[float] | None = None,
    hint: JudgeHint | None = None,
) -> Candidate:
    """Factory helper creating valid Candidate instances."""
    val = object_value
    if object_entity_id is None and val is None:
        val = object_norm
    vf = valid_from if valid_from is not None else utc(2026, 1, 1)
    sts = source_ts if source_ts is not None else utc(2026, 1, 1)
    return Candidate(
        user_id=user_id or UserId(uuid4()),
        subject_id=subject_id or EntityId(uuid4()),
        predicate=predicate,
        cardinality=cardinality,
        object_entity_id=object_entity_id,
        object_value=val,
        object_norm=object_norm,
        fact_text=fact_text,
        valid_from=vf,
        valid_to=valid_to,
        time_precision=time_precision,
        valid_from_explicit=valid_from_explicit,
        confidence=confidence,
        is_retraction=is_retraction,
        source_id=source_id or SourceId(uuid4()),
        span=span,
        source_ts=sts,
        event_hash=event_hash,
        embedding=embedding or [0.1] * 768,
        hint=hint,
    )


def apply_plan_in_memory(
    facts: list[Fact],
    plan: ResolutionPlan,
    candidate: Candidate | None = None,
) -> list[Fact]:
    """Helper applying resolution plan ops to an in-memory fact timeline."""
    result: list[Fact] = list(facts)
    new_facts: list[Fact] = []

    for op in plan.ops:
        if isinstance(op, CloseFact):
            for i, f in enumerate(result):
                if f.id == op.fact_id:
                    result[i] = replace(f, valid_to=op.valid_to)
        elif isinstance(op, ExtendValidFrom):
            for i, f in enumerate(result):
                if f.id == op.fact_id:
                    result[i] = replace(f, valid_from=op.valid_from)
        elif isinstance(op, SetStatus):
            for i, f in enumerate(result):
                if f.id == op.fact_id:
                    result[i] = replace(f, status=op.status)
        elif isinstance(op, BumpConfidence):
            for i, f in enumerate(result):
                if f.id == op.fact_id:
                    result[i] = replace(f, confidence=op.confidence)
        elif isinstance(op, InsertFact):
            if candidate is not None:
                new_f = Fact(
                    id=FactId(uuid4()),
                    user_id=candidate.user_id,
                    subject_id=candidate.subject_id,
                    predicate=candidate.predicate,
                    cardinality=candidate.cardinality,
                    object_entity_id=candidate.object_entity_id,
                    object_value=candidate.object_value,
                    object_norm=candidate.object_norm,
                    fact_text=candidate.fact_text,
                    valid_from=op.valid_from,
                    valid_to=op.valid_to,
                    time_precision=candidate.time_precision,
                    valid_from_explicit=candidate.valid_from_explicit,
                    recorded_at=candidate.source_ts,
                    invalidated_at=None,
                    confidence=candidate.confidence,
                    salience=1.0,
                    access_count=0,
                    last_accessed_at=None,
                    status=op.status,
                    origin=FactOrigin.EXTRACTED,
                    event_hash=candidate.event_hash,
                )
            else:
                tmpl = facts[0] if facts else None
                new_f = Fact(
                    id=FactId(uuid4()),
                    user_id=tmpl.user_id if tmpl else UserId(uuid4()),
                    subject_id=tmpl.subject_id if tmpl else EntityId(uuid4()),
                    predicate=tmpl.predicate if tmpl else 'lives_in',
                    cardinality=tmpl.cardinality if tmpl else Cardinality.SINGLE,
                    object_entity_id=None,
                    object_value='delhi',
                    object_norm='delhi',
                    fact_text='User lives in Delhi',
                    valid_from=op.valid_from,
                    valid_to=op.valid_to,
                    time_precision=TimePrecision.DAY,
                    valid_from_explicit=True,
                    recorded_at=op.valid_from,
                    invalidated_at=None,
                    confidence=1.0,
                    salience=1.0,
                    access_count=0,
                    last_accessed_at=None,
                    status=op.status,
                    origin=FactOrigin.EXTRACTED,
                    event_hash=None,
                )
            new_facts.append(new_f)

    result.extend(new_facts)
    return result


def test_out_of_order_backfill_does_not_supersede_newer_truth() -> None:
    """Exact test from LLD Section 20.4 (T-CR-05)."""
    pune = make_fact(
        predicate='lives_in',
        cardinality=Cardinality.SINGLE,
        object_norm='pune',
        valid_from=utc(2026, 10, 1),
        valid_to=None,
        status=FactStatus.ACTIVE,
    )
    delhi = make_candidate(
        predicate='lives_in',
        cardinality=Cardinality.SINGLE,
        object_norm='delhi',
        valid_from=utc(2024, 1, 1),
        valid_to=None,
        valid_from_explicit=True,
        time_precision=TimePrecision.YEAR,
        source_ts=utc(2026, 10, 20),
    )
    plan = ConflictResolver(default_registry()).plan(
        delhi, [pune], ResolutionContext(now=utc(2026, 10, 20))
    )

    assert plan.outcome is Outcome.BACKFILLED
    inserts = [o for o in plan.ops if isinstance(o, InsertFact)]
    assert len(inserts) == 1
    assert inserts[0].valid_from == utc(2024, 1, 1)
    assert inserts[0].valid_to == utc(2026, 10, 1)
    assert inserts[0].status is FactStatus.ACTIVE
    assert not any(isinstance(o, CloseFact) for o in plan.ops)  # Pune is never closed
    assert (
        AddRelation(from_ref=pune.id, to_ref=NewRef(), relation=RelationType.SUPERSEDES) in plan.ops
    )

    after = apply_plan_in_memory([pune], plan)  # helper used by property tests
    assert [f.object_norm for f in after if f.valid_to is None] == ['pune']


def test_t_cr_01_first_fact_inserted() -> None:
    """T-CR-01: First fact on an empty timeline is inserted with status ACTIVE."""
    resolver = ConflictResolver(default_registry())
    cand = make_candidate(valid_from=utc(2024, 1, 1), valid_to=None, object_norm='delhi')
    plan = resolver.plan(cand, [], ResolutionContext(now=utc(2026, 1, 1)))

    assert plan.outcome is Outcome.INSERTED
    assert plan.reason == 'first_fact'
    assert len(plan.ops) == 2
    assert plan.ops[0] == InsertFact(utc(2024, 1, 1), None, FactStatus.ACTIVE)
    assert plan.ops[1] == AddEvidence(NewRef())


def test_t_cr_02_same_object_merges() -> None:
    """T-CR-02: Same object merges into existing open fact and updates bounds."""
    resolver = ConflictResolver(default_registry())
    fact = make_fact(
        valid_from=utc(2024, 6, 1),
        valid_to=None,
        object_norm='delhi',
        confidence=0.7,
    )
    cand_repeat = make_candidate(
        valid_from=utc(2024, 8, 1),
        valid_to=None,
        object_norm='delhi',
        confidence=0.9,
    )
    plan = resolver.plan(cand_repeat, [fact], ResolutionContext(now=utc(2026, 1, 1)))

    assert plan.outcome is Outcome.MERGED
    assert plan.reason == 'merged_same_fact'
    assert BumpConfidence(fact.id, 0.9) in plan.ops
    assert AddEvidence(fact.id) in plan.ops
    assert not any(isinstance(o, ExtendValidFrom) for o in plan.ops)

    # Sub-case: explicit earlier valid_from extends valid_from
    cand_earlier = make_candidate(
        valid_from=utc(2024, 1, 1),
        valid_to=utc(2024, 6, 1),
        object_norm='delhi',
        valid_from_explicit=True,
    )
    plan_ext = resolver.plan(cand_earlier, [fact], ResolutionContext(now=utc(2026, 1, 1)))
    assert plan_ext.outcome is Outcome.MERGED
    assert plan_ext.reason == 'merged_extended_valid_from'
    assert ExtendValidFrom(fact.id, utc(2024, 1, 1)) in plan_ext.ops

    # Sub-case: earlier explicit date does NOT extend if another active row overlaps
    older_fact = make_fact(
        valid_from=utc(2023, 6, 1),
        valid_to=utc(2024, 3, 1),
        object_norm='mumbai',
    )
    plan_blocked = resolver.plan(
        cand_earlier, [older_fact, fact], ResolutionContext(now=utc(2026, 1, 1))
    )
    assert plan_blocked.outcome is Outcome.MERGED
    assert not any(isinstance(o, ExtendValidFrom) for o in plan_blocked.ops)


def test_t_cr_03_different_object_supersedes() -> None:
    """T-CR-03: Different object with f > L.valid_from supersedes and closes the open fact."""
    resolver = ConflictResolver(default_registry())
    delhi = make_fact(valid_from=utc(2024, 1, 1), valid_to=None, object_norm='delhi')
    pune = make_candidate(valid_from=utc(2024, 6, 1), valid_to=None, object_norm='pune')
    plan = resolver.plan(pune, [delhi], ResolutionContext(now=utc(2026, 1, 1)))

    assert plan.outcome is Outcome.SUPERSEDED
    assert plan.reason == 'superseded_open_fact'
    assert CloseFact(delhi.id, utc(2024, 6, 1)) in plan.ops
    assert InsertFact(utc(2024, 6, 1), None, FactStatus.ACTIVE) in plan.ops
    assert AddEvidence(NewRef()) in plan.ops
    assert (
        AddRelation(from_ref=NewRef(), to_ref=delhi.id, relation=RelationType.SUPERSEDES)
        in plan.ops
    )


def test_t_cr_04_three_step_chain() -> None:
    """T-CR-04: Delhi -> Pune -> Mumbai creates a chain of three mutually disjoint active facts."""
    resolver = ConflictResolver(default_registry())
    ctx = ResolutionContext(now=utc(2026, 1, 1))

    c1 = make_candidate(valid_from=utc(2024, 1, 1), valid_to=None, object_norm='delhi')
    plan1 = resolver.plan(c1, [], ctx)
    t1 = apply_plan_in_memory([], plan1, candidate=c1)
    assert len(t1) == 1
    assert t1[0].valid_to is None

    c2 = make_candidate(valid_from=utc(2024, 6, 1), valid_to=None, object_norm='pune')
    plan2 = resolver.plan(c2, t1, ctx)
    t2 = apply_plan_in_memory(t1, plan2, candidate=c2)
    assert len(t2) == 2
    delhi_row = next(f for f in t2 if f.object_norm == 'delhi')
    pune_row = next(f for f in t2 if f.object_norm == 'pune')
    assert delhi_row.valid_to == utc(2024, 6, 1)
    assert pune_row.valid_to is None

    c3 = make_candidate(valid_from=utc(2025, 1, 1), valid_to=None, object_norm='mumbai')
    plan3 = resolver.plan(c3, t2, ctx)
    t3 = apply_plan_in_memory(t2, plan3, candidate=c3)
    assert len(t3) == 3
    delhi_final = next(f for f in t3 if f.object_norm == 'delhi')
    pune_final = next(f for f in t3 if f.object_norm == 'pune')
    mumbai_final = next(f for f in t3 if f.object_norm == 'mumbai')

    assert delhi_final.valid_from == utc(2024, 1, 1)
    assert delhi_final.valid_to == utc(2024, 6, 1)
    assert pune_final.valid_from == utc(2024, 6, 1)
    assert pune_final.valid_to == utc(2025, 1, 1)
    assert mumbai_final.valid_from == utc(2025, 1, 1)
    assert mumbai_final.valid_to is None


def test_t_cr_05_backfill_with_prior_gap_row() -> None:
    """T-CR-05 extension: Backfill between Q and N links both supersedes relations."""
    resolver = ConflictResolver(default_registry())
    mumbai = make_fact(
        valid_from=utc(2020, 1, 1),
        valid_to=utc(2022, 1, 1),
        object_norm='mumbai',
    )
    pune = make_fact(valid_from=utc(2026, 10, 1), valid_to=None, object_norm='pune')
    delhi = make_candidate(valid_from=utc(2024, 1, 1), valid_to=None, object_norm='delhi')

    plan = resolver.plan(delhi, [mumbai, pune], ResolutionContext(now=utc(2026, 10, 20)))
    assert plan.outcome is Outcome.BACKFILLED
    assert (
        AddRelation(from_ref=NewRef(), to_ref=mumbai.id, relation=RelationType.SUPERSEDES)
        in plan.ops
    )
    assert (
        AddRelation(from_ref=pune.id, to_ref=NewRef(), relation=RelationType.SUPERSEDES) in plan.ops
    )


def test_t_cr_06_backfill_inside_different_fact_goes_to_review() -> None:
    """T-CR-06: Backfill timestamp inside a closed different fact goes to REVIEW (S7/S9)."""
    resolver = ConflictResolver(default_registry())

    # Case S9: f < L.valid_from and older closed fact P contains f
    p = make_fact(
        valid_from=utc(2024, 1, 1),
        valid_to=utc(2025, 1, 1),
        object_norm='delhi',
    )
    l_row = make_fact(valid_from=utc(2025, 6, 1), valid_to=None, object_norm='pune')
    cand = make_candidate(valid_from=utc(2024, 6, 1), valid_to=None, object_norm='mumbai')

    plan_s9 = resolver.plan(cand, [p, l_row], ResolutionContext(now=utc(2026, 1, 1)))
    assert plan_s9.outcome is Outcome.REVIEW
    assert plan_s9.reason == 'overlap_conflict'
    assert InsertFact(utc(2024, 6, 1), None, FactStatus.NEEDS_REVIEW) in plan_s9.ops
    assert (
        AddRelation(from_ref=NewRef(), to_ref=p.id, relation=RelationType.CLARIFIES) in plan_s9.ops
    )

    # Case S7: f > L.valid_from and L is closed with L.valid_to > f
    closed_l = make_fact(
        valid_from=utc(2024, 1, 1),
        valid_to=utc(2025, 1, 1),
        object_norm='delhi',
    )
    cand_s7 = make_candidate(valid_from=utc(2024, 6, 1), valid_to=None, object_norm='mumbai')
    plan_s7 = resolver.plan(cand_s7, [closed_l], ResolutionContext(now=utc(2026, 1, 1)))
    assert plan_s7.outcome is Outcome.REVIEW
    assert plan_s7.reason == 'overlap_conflict'
    assert InsertFact(utc(2024, 6, 1), None, FactStatus.NEEDS_REVIEW) in plan_s7.ops
    assert (
        AddRelation(from_ref=NewRef(), to_ref=closed_l.id, relation=RelationType.CLARIFIES)
        in plan_s7.ops
    )


def test_t_cr_07_equal_timestamp_goes_to_review() -> None:
    """T-CR-07: Candidate with f == L.valid_from and different object triggers REVIEW (S8)."""
    resolver = ConflictResolver(default_registry())
    delhi = make_fact(valid_from=utc(2024, 1, 1), valid_to=None, object_norm='delhi')
    pune = make_candidate(valid_from=utc(2024, 1, 1), valid_to=None, object_norm='pune')

    plan = resolver.plan(pune, [delhi], ResolutionContext(now=utc(2026, 1, 1)))
    assert plan.outcome is Outcome.REVIEW
    assert plan.reason == 'same_timestamp_conflict'
    assert InsertFact(utc(2024, 1, 1), None, FactStatus.NEEDS_REVIEW) in plan.ops
    assert (
        AddRelation(from_ref=NewRef(), to_ref=delhi.id, relation=RelationType.CLARIFIES) in plan.ops
    )


def test_t_cr_08_retraction_closes_open_fact() -> None:
    """T-CR-08: Retraction closes matching open fact at candidate timestamp (S1)."""
    resolver = ConflictResolver(default_registry())
    delhi = make_fact(valid_from=utc(2024, 1, 1), valid_to=None, object_norm='delhi')
    retraction = make_candidate(
        valid_from=utc(2024, 6, 1),
        valid_to=None,
        object_norm='delhi',
        is_retraction=True,
    )

    plan = resolver.plan(retraction, [delhi], ResolutionContext(now=utc(2026, 1, 1)))
    assert plan.outcome is Outcome.RETRACTED
    assert plan.reason == 'retracted_open_fact'
    assert CloseFact(delhi.id, utc(2024, 6, 1)) in plan.ops
    assert AddEvidence(delhi.id) in plan.ops


def test_t_cr_09_retraction_without_target_is_noop() -> None:
    """T-CR-09: Retraction without open target or with f <= valid_from is a NOOP (S2)."""
    resolver = ConflictResolver(default_registry())

    # Case: empty timeline
    retraction = make_candidate(
        valid_from=utc(2024, 6, 1),
        valid_to=None,
        object_norm='delhi',
        is_retraction=True,
    )
    plan_no_target = resolver.plan(retraction, [], ResolutionContext(now=utc(2026, 1, 1)))
    assert plan_no_target.outcome is Outcome.NOOP
    assert plan_no_target.reason == 'retraction_no_target'
    assert plan_no_target.ops == ()

    # Case: f <= X.valid_from
    delhi = make_fact(valid_from=utc(2024, 6, 1), valid_to=None, object_norm='delhi')
    cand_early_retract = make_candidate(
        valid_from=utc(2024, 1, 1),
        valid_to=None,
        object_norm='delhi',
        is_retraction=True,
    )
    plan_before_start = resolver.plan(
        cand_early_retract, [delhi], ResolutionContext(now=utc(2026, 1, 1))
    )
    assert plan_before_start.outcome is Outcome.NOOP
    assert plan_before_start.reason == 'retraction_before_start'
    assert plan_before_start.ops == ()


def test_t_cr_10_delhi_pune_delhi_gives_three_rows() -> None:
    """T-CR-10: Candidate sequence Delhi -> Pune -> Delhi produces three active rows."""
    resolver = ConflictResolver(default_registry())
    ctx = ResolutionContext(now=utc(2026, 1, 1))

    c1 = make_candidate(valid_from=utc(2024, 1, 1), valid_to=None, object_norm='delhi')
    t = apply_plan_in_memory([], resolver.plan(c1, [], ctx), candidate=c1)

    c2 = make_candidate(valid_from=utc(2024, 6, 1), valid_to=None, object_norm='pune')
    t = apply_plan_in_memory(t, resolver.plan(c2, t, ctx), candidate=c2)

    c3 = make_candidate(valid_from=utc(2025, 1, 1), valid_to=None, object_norm='delhi')
    plan3 = resolver.plan(c3, t, ctx)
    assert plan3.outcome is Outcome.SUPERSEDED
    t = apply_plan_in_memory(t, plan3, candidate=c3)

    assert len(t) == 3
    assert [f.object_norm for f in t] == ['delhi', 'pune', 'delhi']
    assert [f.status for f in t] == [FactStatus.ACTIVE] * 3
    assert t[0].valid_from == utc(2024, 1, 1) and t[0].valid_to == utc(2024, 6, 1)
    assert t[1].valid_from == utc(2024, 6, 1) and t[1].valid_to == utc(2025, 1, 1)
    assert t[2].valid_from == utc(2025, 1, 1) and t[2].valid_to is None


def test_t_cr_11_multi_valued_merge_and_coexistence() -> None:
    """T-CR-11: Multi-valued values coexist or merge (M3/M4)."""
    resolver = ConflictResolver(default_registry())
    ctx = ResolutionContext(now=utc(2026, 1, 1))

    c_python = make_candidate(
        predicate='likes',
        cardinality=Cardinality.MULTI,
        object_norm='python',
        valid_from=utc(2024, 1, 1),
        confidence=0.8,
    )
    p1 = resolver.plan(c_python, [], ctx)
    assert p1.outcome is Outcome.INSERTED
    assert p1.reason == 'inserted_multi'
    t = apply_plan_in_memory([], p1, candidate=c_python)

    # Repeat python merges
    c_python_repeat = make_candidate(
        predicate='likes',
        cardinality=Cardinality.MULTI,
        object_norm='python',
        valid_from=utc(2024, 6, 1),
        confidence=0.95,
    )
    p2 = resolver.plan(c_python_repeat, t, ctx)
    assert p2.outcome is Outcome.MERGED
    assert p2.reason == 'merged_same_fact'
    assert BumpConfidence(t[0].id, 0.95) in p2.ops

    # Distinct object coexists
    c_rust = make_candidate(
        predicate='likes',
        cardinality=Cardinality.MULTI,
        object_norm='rust',
        valid_from=utc(2024, 6, 1),
    )
    p3 = resolver.plan(c_rust, t, ctx)
    assert p3.outcome is Outcome.INSERTED
    t = apply_plan_in_memory(t, p3, candidate=c_rust)
    assert len(t) == 2
    assert {f.object_norm for f in t} == {'python', 'rust'}

    # Multi extend valid_from
    c_python_earlier = make_candidate(
        predicate='likes',
        cardinality=Cardinality.MULTI,
        object_norm='python',
        valid_from=utc(2023, 1, 1),
        valid_from_explicit=True,
    )
    p_ext = resolver.plan(c_python_earlier, t, ctx)
    assert ExtendValidFrom(t[0].id, utc(2023, 1, 1)) in p_ext.ops


def test_t_cr_12_multi_valued_retraction() -> None:
    """T-CR-12: Multi-valued retraction closes open matching fact, or noops if no target (M1/M2)."""
    resolver = ConflictResolver(default_registry())
    ctx = ResolutionContext(now=utc(2026, 1, 1))
    fact = make_fact(
        predicate='likes',
        cardinality=Cardinality.MULTI,
        object_norm='python',
        valid_from=utc(2024, 1, 1),
    )

    retract_cand = make_candidate(
        predicate='likes',
        cardinality=Cardinality.MULTI,
        object_norm='python',
        valid_from=utc(2024, 6, 1),
        is_retraction=True,
    )
    plan = resolver.plan(retract_cand, [fact], ctx)
    assert plan.outcome is Outcome.RETRACTED
    assert CloseFact(fact.id, utc(2024, 6, 1)) in plan.ops

    # Retraction with no target
    retract_no_target = make_candidate(
        predicate='likes',
        cardinality=Cardinality.MULTI,
        object_norm='ruby',
        valid_from=utc(2024, 6, 1),
        is_retraction=True,
    )
    plan_noop = resolver.plan(retract_no_target, [fact], ctx)
    assert plan_noop.outcome is Outcome.NOOP
    assert plan_noop.reason == 'retraction_no_target'

    # Retraction before start
    retract_before = make_candidate(
        predicate='likes',
        cardinality=Cardinality.MULTI,
        object_norm='python',
        valid_from=utc(2023, 1, 1),
        is_retraction=True,
    )
    plan_before = resolver.plan(retract_before, [fact], ctx)
    assert plan_before.outcome is Outcome.NOOP
    assert plan_before.reason == 'retraction_before_start'


def test_t_cr_13_event_dedupe_by_hash() -> None:
    """T-CR-13: Event facts deduplicate identical event_hash and ignore retractions (E1-E3)."""
    resolver = ConflictResolver(default_registry())
    ctx = ResolutionContext(now=utc(2026, 1, 1))

    c1 = make_candidate(
        predicate='attended',
        cardinality=Cardinality.EVENT,
        object_norm='conf2024',
        valid_from=utc(2024, 5, 1),
        valid_to=utc(2024, 5, 3),
        event_hash='hash_abc123',
        confidence=0.8,
    )
    p1 = resolver.plan(c1, [], ctx)
    assert p1.outcome is Outcome.INSERTED
    assert p1.reason == 'inserted_event'
    t = apply_plan_in_memory([], p1, candidate=c1)

    # Identical hash merges (E2)
    c2 = make_candidate(
        predicate='attended',
        cardinality=Cardinality.EVENT,
        object_norm='conf2024',
        valid_from=utc(2024, 5, 1),
        valid_to=utc(2024, 5, 3),
        event_hash='hash_abc123',
        confidence=0.95,
    )
    p2 = resolver.plan(c2, t, ctx)
    assert p2.outcome is Outcome.MERGED
    assert p2.reason == 'deduped_by_event_hash'
    assert BumpConfidence(t[0].id, 0.95) in p2.ops
    assert AddEvidence(t[0].id) in p2.ops

    # Different hash inserts (E3)
    c3 = make_candidate(
        predicate='attended',
        cardinality=Cardinality.EVENT,
        object_norm='conf2025',
        valid_from=utc(2025, 5, 1),
        valid_to=utc(2025, 5, 3),
        event_hash='hash_xyz789',
    )
    p3 = resolver.plan(c3, t, ctx)
    assert p3.outcome is Outcome.INSERTED

    # Retraction is NOOP (E1)
    c_retract = make_candidate(
        predicate='attended',
        cardinality=Cardinality.EVENT,
        object_norm='conf2024',
        valid_from=utc(2024, 5, 1),
        valid_to=utc(2024, 5, 3),
        is_retraction=True,
    )
    p_retract = resolver.plan(c_retract, t, ctx)
    assert p_retract.outcome is Outcome.NOOP
    assert p_retract.reason == 'event_retraction_unsupported'


def test_t_cr_14_every_plan_puts_closes_before_inserts() -> None:
    """T-CR-14: ResolutionPlan enforces canonical op ordering."""
    fid = FactId(uuid4())
    unordered_ops = (
        AddRelation(NewRef(), fid, RelationType.SUPERSEDES),
        InsertFact(utc(2026, 1, 1), None, FactStatus.ACTIVE),
        AddEvidence(NewRef()),
        CloseFact(fid, utc(2026, 1, 1)),
        ExtendValidFrom(fid, utc(2025, 1, 1)),
        BumpConfidence(fid, 0.9),
    )
    plan = ResolutionPlan(ops=unordered_ops, outcome=Outcome.SUPERSEDED, reason='test')

    types = [type(op) for op in plan.ops]
    close_idx = types.index(CloseFact)
    extend_idx = types.index(ExtendValidFrom)
    insert_idx = types.index(InsertFact)
    evidence_idx = types.index(AddEvidence)
    relation_idx = types.index(AddRelation)

    assert close_idx < insert_idx
    assert extend_idx < insert_idx
    assert insert_idx < evidence_idx
    assert insert_idx < relation_idx


@given(
    st.lists(
        st.tuples(
            st.sampled_from(['delhi', 'pune', 'mumbai', 'bangalore']),
            st.integers(min_value=2020, max_value=2028),
            st.integers(min_value=1, max_value=12),
            st.booleans(),  # has valid_to
            st.booleans(),  # is_retraction
        ),
        min_size=1,
        max_size=8,
    )
)
@settings(max_examples=100)
def test_t_cr_15_property_g1(
    items: list[tuple[str, int, int, bool, bool]],
) -> None:
    """T-CR-15: Property test proving Guarantee G-1 across randomly ordered candidate sequences."""
    resolver = ConflictResolver(default_registry())
    ctx = ResolutionContext(now=utc(2029, 1, 1))
    timeline: list[Fact] = []

    for obj, year, month, has_valid_to, is_retraction in items:
        valid_from = utc(year, month, 1)
        valid_to = utc(year + 1, month, 1) if has_valid_to and not is_retraction else None
        cand = make_candidate(
            object_norm=obj,
            valid_from=valid_from,
            valid_to=valid_to,
            is_retraction=is_retraction,
            source_ts=utc(2029, 1, 1),
        )
        plan = resolver.plan(cand, timeline, ctx)
        timeline = apply_plan_in_memory(timeline, plan, candidate=cand)

    active = [f for f in timeline if f.status == FactStatus.ACTIVE]

    # Guarantee G-1 check 1: at most one open fact
    open_facts = [f for f in active if f.valid_to is None]
    assert len(open_facts) <= 1

    # Guarantee G-1 check 2: pairwise non-overlapping intervals
    for i, f1 in enumerate(active):
        assert f1.valid_to is None or f1.valid_to > f1.valid_from
        for f2 in active[i + 1 :]:
            s1, e1 = f1.valid_from, f1.valid_to
            s2, e2 = f2.valid_from, f2.valid_to
            # Half-open intervals [s1, e1) and [s2, e2) overlap if:
            # not (e1 <= s2 or e2 <= s1)
            is_disjoint = (e1 is not None and e1 <= s2) or (e2 is not None and e2 <= s1)
            assert is_disjoint, f'G-1 invariant broken: overlap between {f1} and {f2}'


def test_judge_hint_branches() -> None:
    """Test optional JudgeHint verdicts across single and multi valued strategies."""
    resolver = ConflictResolver(default_registry())
    ctx = ResolutionContext(now=utc(2026, 1, 1))

    # Single-valued judge hint
    fact_single = make_fact(
        valid_from=utc(2024, 1, 1),
        object_norm='delhi',
        cardinality=Cardinality.SINGLE,
    )
    cand_single_same = make_candidate(
        valid_from=utc(2024, 6, 1),
        object_norm='new_delhi',
        cardinality=Cardinality.SINGLE,
        hint=JudgeHint(target_fact_id=fact_single.id, verdict='same'),
    )
    p_same = resolver.plan(cand_single_same, [fact_single], ctx)
    assert p_same.outcome is Outcome.MERGED
    assert p_same.reason == 'judge_hint_same'

    cand_single_super = make_candidate(
        valid_from=utc(2024, 6, 1),
        object_norm='pune',
        cardinality=Cardinality.SINGLE,
        hint=JudgeHint(target_fact_id=fact_single.id, verdict='supersedes'),
    )
    p_super = resolver.plan(cand_single_super, [fact_single], ctx)
    assert p_super.outcome is Outcome.SUPERSEDED
    assert p_super.reason == 'judge_hint_supersedes'

    # Multi-valued judge hint
    fact_multi = make_fact(
        predicate='owns',
        valid_from=utc(2024, 1, 1),
        object_norm='car',
        cardinality=Cardinality.MULTI,
    )
    cand_multi_coexist = make_candidate(
        predicate='owns',
        valid_from=utc(2024, 6, 1),
        object_norm='bicycle',
        cardinality=Cardinality.MULTI,
        hint=JudgeHint(target_fact_id=fact_multi.id, verdict='coexists'),
    )
    p_coexist = resolver.plan(cand_multi_coexist, [fact_multi], ctx)
    assert p_coexist.outcome is Outcome.INSERTED
    assert p_coexist.reason == 'judge_hint_coexists'

    cand_multi_super = make_candidate(
        predicate='owns',
        valid_from=utc(2024, 6, 1),
        object_norm='supercar',
        cardinality=Cardinality.MULTI,
        hint=JudgeHint(target_fact_id=fact_multi.id, verdict='supersedes'),
    )
    p_multi_super = resolver.plan(cand_multi_super, [fact_multi], ctx)
    assert p_multi_super.outcome is Outcome.SUPERSEDED
    assert p_multi_super.reason == 'judge_hint_supersedes'


def test_strategy_registry_and_rule_s6() -> None:
    """Test StrategyRegistry errors and SingleValuedStrategy gap insertion (S6)."""
    reg = StrategyRegistry()
    with pytest.raises(InvariantViolation, match='No resolution strategy'):
        reg.for_cardinality(Cardinality.SINGLE)

    # S6: f > L.valid_from and L closed with L.valid_to <= f
    resolver = ConflictResolver(default_registry())
    closed_pune = make_fact(
        valid_from=utc(2024, 1, 1),
        valid_to=utc(2024, 6, 1),
        object_norm='pune',
    )
    delhi_after_gap = make_candidate(valid_from=utc(2025, 1, 1), valid_to=None, object_norm='delhi')
    plan_s6 = resolver.plan(delhi_after_gap, [closed_pune], ResolutionContext(now=utc(2026, 1, 1)))
    assert plan_s6.outcome is Outcome.INSERTED
    assert plan_s6.reason == 'inserted_after_gap'
    assert (
        AddRelation(
            from_ref=NewRef(),
            to_ref=closed_pune.id,
            relation=RelationType.SUPERSEDES,
        )
        in plan_s6.ops
    )


def test_command_dataclasses_invariants() -> None:
    """Test validation errors on command dataclasses."""
    fid = FactId(uuid4())

    # Naive datetime
    naive = datetime(2026, 1, 1)  # no tzinfo
    with pytest.raises(InvariantViolation, match='must be timezone-aware UTC'):
        InsertFact(naive, None, FactStatus.ACTIVE)

    with pytest.raises(InvariantViolation, match='must be timezone-aware UTC'):
        CloseFact(fid, naive)

    with pytest.raises(InvariantViolation, match='must be timezone-aware UTC'):
        ExtendValidFrom(fid, naive)

    # valid_to <= valid_from
    with pytest.raises(InvariantViolation, match='strictly greater than'):
        InsertFact(utc(2026, 1, 1), utc(2026, 1, 1), FactStatus.ACTIVE)

    # Confidence out of range
    with pytest.raises(InvariantViolation, match='between 0.0 and 1.0'):
        BumpConfidence(fid, 1.5)

    # ResolutionContext validation
    with pytest.raises(InvariantViolation, match='must be timezone-aware UTC'):
        ResolutionContext(now=naive)

    with pytest.raises(InvariantViolation, match='strictly positive'):
        ResolutionContext(now=utc(2026, 1, 1), merge_gap=timedelta(seconds=0))


def test_branch_coverage_comprehensive() -> None:
    """Ensure 100% coverage on remaining edge branches across strategies."""
    resolver = ConflictResolver(default_registry())
    ctx = ResolutionContext(now=utc(2026, 1, 1))

    # 1. SingleValuedStrategy S3 where candidate is earlier but valid_from_explicit is False
    fact_s3 = make_fact(
        valid_from=utc(2024, 6, 1),
        valid_to=None,
        object_norm='delhi',
    )
    cand_s3_implicit = make_candidate(
        valid_from=utc(2024, 1, 1),
        valid_to=utc(2024, 6, 1),
        object_norm='delhi',
        valid_from_explicit=False,
    )
    p_s3_implicit = resolver.plan(cand_s3_implicit, [fact_s3], ctx)
    assert p_s3_implicit.outcome is Outcome.MERGED
    assert p_s3_implicit.reason == 'merged_same_fact'
    assert not any(isinstance(o, ExtendValidFrom) for o in p_s3_implicit.ops)

    # 2. SingleValuedStrategy S10 with candidate having explicit finite valid_to < N.valid_from
    pune = make_fact(valid_from=utc(2026, 10, 1), valid_to=None, object_norm='pune')
    delhi_bounded = make_candidate(
        valid_from=utc(2024, 1, 1),
        valid_to=utc(2025, 1, 1),
        object_norm='delhi',
    )
    p_bounded = resolver.plan(delhi_bounded, [pune], ctx)
    assert p_bounded.outcome is Outcome.BACKFILLED
    ins = [o for o in p_bounded.ops if isinstance(o, InsertFact)][0]
    assert ins.valid_to == utc(2025, 1, 1)

    # 3. JudgeHint where target_fact_id does not exist in active_rows (single and multi)
    missing_id = FactId(uuid4())
    cand_missing_hint = make_candidate(
        valid_from=utc(2024, 6, 1),
        object_norm='delhi',
        hint=JudgeHint(target_fact_id=missing_id, verdict='same'),
    )
    p_missing = resolver.plan(cand_missing_hint, [fact_s3], ctx)
    assert p_missing.outcome is Outcome.MERGED
    assert p_missing.reason == 'merged_same_fact'

    cand_multi_missing = make_candidate(
        predicate='owns',
        cardinality=Cardinality.MULTI,
        valid_from=utc(2024, 6, 1),
        object_norm='boat',
        hint=JudgeHint(target_fact_id=missing_id, verdict='same'),
    )
    p_multi_missing = resolver.plan(cand_multi_missing, [], ctx)
    assert p_multi_missing.outcome is Outcome.INSERTED

    # 4. MultiValuedStrategy JudgeHint 'same' with earlier explicit valid_from
    fact_multi = make_fact(
        predicate='owns',
        cardinality=Cardinality.MULTI,
        object_norm='car',
        valid_from=utc(2024, 6, 1),
    )
    cand_multi_same_ext = make_candidate(
        predicate='owns',
        cardinality=Cardinality.MULTI,
        object_norm='car',
        valid_from=utc(2024, 1, 1),
        valid_from_explicit=True,
        hint=JudgeHint(target_fact_id=fact_multi.id, verdict='same'),
    )
    p_ext_multi = resolver.plan(cand_multi_same_ext, [fact_multi], ctx)
    assert p_ext_multi.outcome is Outcome.MERGED
    assert ExtendValidFrom(fact_multi.id, utc(2024, 1, 1)) in p_ext_multi.ops

    # 5. MultiValuedStrategy M3 with closed fact touching candidate
    closed_multi = make_fact(
        predicate='owns',
        cardinality=Cardinality.MULTI,
        object_norm='car',
        valid_from=utc(2024, 1, 1),
        valid_to=utc(2024, 6, 1),
    )
    cand_m3_touch = make_candidate(
        predicate='owns',
        cardinality=Cardinality.MULTI,
        object_norm='car',
        valid_from=utc(2024, 6, 1),
    )
    p_m3_touch = resolver.plan(cand_m3_touch, [closed_multi], ctx)
    assert p_m3_touch.outcome is Outcome.MERGED
    assert p_m3_touch.reason == 'merged_same_fact'

    # 6. EventStrategy with event_hash is None -> inserted_event
    cand_event_nohash = make_candidate(
        predicate='attended',
        cardinality=Cardinality.EVENT,
        object_norm='conf',
        valid_from=utc(2024, 1, 1),
        valid_to=utc(2024, 1, 2),
        event_hash=None,
    )
    p_event_nohash = resolver.plan(cand_event_nohash, [], ctx)
    assert p_event_nohash.outcome is Outcome.INSERTED
    assert p_event_nohash.reason == 'inserted_event'

    # 7. SingleValuedStrategy JudgeHint 'same' with earlier explicit valid_from
    cand_single_same_ext = make_candidate(
        valid_from=utc(2024, 1, 1),
        object_norm='new_delhi',
        cardinality=Cardinality.SINGLE,
        valid_from_explicit=True,
        hint=JudgeHint(target_fact_id=fact_s3.id, verdict='same'),
    )
    p_single_same_ext = resolver.plan(cand_single_same_ext, [fact_s3], ctx)
    assert p_single_same_ext.outcome is Outcome.MERGED
    assert ExtendValidFrom(fact_s3.id, utc(2024, 1, 1)) in p_single_same_ext.ops

    # 8. SingleValuedStrategy JudgeHint 'same' with earlier explicit but blocked by another row
    fact_older = make_fact(
        valid_from=utc(2023, 1, 1),
        valid_to=utc(2024, 3, 1),
        object_norm='mumbai',
    )
    p_single_blocked = resolver.plan(cand_single_same_ext, [fact_older, fact_s3], ctx)
    assert p_single_blocked.outcome is Outcome.MERGED
    assert not any(isinstance(o, ExtendValidFrom) for o in p_single_blocked.ops)
