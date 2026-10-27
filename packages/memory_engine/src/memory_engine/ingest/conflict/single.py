"""SingleValuedStrategy implementing conflict rules S1 through S10."""

from collections.abc import Sequence
from typing import ClassVar

from memory_engine.domain.enums import Cardinality, FactStatus, RelationType
from memory_engine.domain.models import Candidate, Fact, JudgeHint
from memory_engine.ingest.conflict.plan import (
    AddEvidence,
    AddRelation,
    BumpConfidence,
    CloseFact,
    ExtendValidFrom,
    InsertFact,
    NewRef,
    Op,
    Outcome,
    ResolutionContext,
    ResolutionPlan,
    ResolutionStrategy,
)


class SingleValuedStrategy(ResolutionStrategy):
    """Resolution strategy for single-valued predicates (rules S1-S10)."""

    cardinality: ClassVar[Cardinality] = Cardinality.SINGLE

    def plan(
        self,
        c: Candidate,
        timeline: Sequence[Fact],
        ctx: ResolutionContext,
        hint: JudgeHint | None = None,
    ) -> ResolutionPlan:
        """Evaluate candidate against timeline using rules S1 through S10."""
        # A = timeline rows with status ACTIVE (NEEDS_REVIEW rows are ignored)
        active_rows = [fact for fact in timeline if fact.status == FactStatus.ACTIVE]
        f = c.valid_from
        e = c.valid_to

        # Optional ambiguity judge hint handling
        if hint is not None:
            target = next((x for x in active_rows if x.id == hint.target_fact_id), None)
            if target is not None:
                if hint.verdict == 'same':
                    ops: list[Op] = [
                        AddEvidence(ref=target.id),
                        BumpConfidence(
                            fact_id=target.id,
                            confidence=max(target.confidence, c.confidence),
                        ),
                    ]
                    if c.valid_from_explicit and f < target.valid_from:
                        overlaps_other = any(
                            y.id != target.id
                            and y.valid_from < target.valid_from
                            and (y.valid_to is None or y.valid_to > f)
                            for y in active_rows
                        )
                        if not overlaps_other:
                            ops.append(ExtendValidFrom(fact_id=target.id, valid_from=f))
                    return ResolutionPlan(
                        ops=tuple(ops),
                        outcome=Outcome.MERGED,
                        reason='judge_hint_same',
                    )
                if hint.verdict == 'supersedes':
                    return ResolutionPlan(
                        ops=(
                            CloseFact(fact_id=target.id, valid_to=f),
                            InsertFact(valid_from=f, valid_to=e, status=FactStatus.ACTIVE),
                            AddEvidence(ref=NewRef()),
                            AddRelation(
                                from_ref=NewRef(),
                                to_ref=target.id,
                                relation=RelationType.SUPERSEDES,
                            ),
                        ),
                        outcome=Outcome.SUPERSEDED,
                        reason='judge_hint_supersedes',
                    )

        # S1: c.is_retraction and an open row X in A with same(X) and f > X.valid_from
        if c.is_retraction:
            open_matching = [
                x
                for x in active_rows
                if x.object_norm == c.object_norm and x.valid_to is None and f > x.valid_from
            ]
            if open_matching:
                x = open_matching[0]
                return ResolutionPlan(
                    ops=(
                        CloseFact(fact_id=x.id, valid_to=f),
                        AddEvidence(ref=x.id),
                    ),
                    outcome=Outcome.RETRACTED,
                    reason='retracted_open_fact',
                )

            # S2: c.is_retraction otherwise -> no ops
            has_open_same = any(
                x.object_norm == c.object_norm and x.valid_to is None for x in active_rows
            )
            reason = 'retraction_before_start' if has_open_same else 'retraction_no_target'
            return ResolutionPlan(ops=(), outcome=Outcome.NOOP, reason=reason)

        # S3: A has a row X with same(X) whose interval overlaps or touches [f, e or f + merge_gap)
        cand_end = e if e is not None else (f + ctx.merge_gap)
        matching_s3 = [
            x
            for x in active_rows
            if x.object_norm == c.object_norm
            and (x.valid_to is None or x.valid_to >= f)
            and (cand_end is None or cand_end >= x.valid_from)
        ]
        if matching_s3:
            # Pick the row whose valid_from is closest to f
            x = min(matching_s3, key=lambda row: abs((row.valid_from - f).total_seconds()))
            ops_s3: list[Op] = [
                AddEvidence(ref=x.id),
                BumpConfidence(fact_id=x.id, confidence=max(x.confidence, c.confidence)),
            ]
            reason_s3 = 'merged_same_fact'
            if c.valid_from_explicit and f < x.valid_from:
                overlaps_other = any(
                    y.id != x.id
                    and y.valid_from < x.valid_from
                    and (y.valid_to is None or y.valid_to > f)
                    for y in active_rows
                )
                if not overlaps_other:
                    ops_s3.append(ExtendValidFrom(fact_id=x.id, valid_from=f))
                    reason_s3 = 'merged_extended_valid_from'
            return ResolutionPlan(
                ops=tuple(ops_s3),
                outcome=Outcome.MERGED,
                reason=reason_s3,
            )

        # S4: A is empty
        if not active_rows:
            return ResolutionPlan(
                ops=(
                    InsertFact(valid_from=f, valid_to=e, status=FactStatus.ACTIVE),
                    AddEvidence(ref=NewRef()),
                ),
                outcome=Outcome.INSERTED,
                reason='first_fact',
            )

        # L = row of A with greatest valid_from
        latest = max(active_rows, key=lambda row: row.valid_from)

        # S5: f > L.valid_from and L is open
        if f > latest.valid_from and latest.valid_to is None:
            return ResolutionPlan(
                ops=(
                    CloseFact(fact_id=latest.id, valid_to=f),
                    InsertFact(valid_from=f, valid_to=e, status=FactStatus.ACTIVE),
                    AddEvidence(ref=NewRef()),
                    AddRelation(
                        from_ref=NewRef(),
                        to_ref=latest.id,
                        relation=RelationType.SUPERSEDES,
                    ),
                ),
                outcome=Outcome.SUPERSEDED,
                reason='superseded_open_fact',
            )

        # S6: f > L.valid_from and L closed with L.valid_to <= f (gap)
        if f > latest.valid_from and latest.valid_to is not None and latest.valid_to <= f:
            return ResolutionPlan(
                ops=(
                    InsertFact(valid_from=f, valid_to=e, status=FactStatus.ACTIVE),
                    AddEvidence(ref=NewRef()),
                    AddRelation(
                        from_ref=NewRef(),
                        to_ref=latest.id,
                        relation=RelationType.SUPERSEDES,
                    ),
                ),
                outcome=Outcome.INSERTED,
                reason='inserted_after_gap',
            )

        # S7: f > L.valid_from and L closed with L.valid_to > f (f inside a closed different fact)
        if f > latest.valid_from and latest.valid_to is not None and latest.valid_to > f:
            return ResolutionPlan(
                ops=(
                    InsertFact(valid_from=f, valid_to=e, status=FactStatus.NEEDS_REVIEW),
                    AddEvidence(ref=NewRef()),
                    AddRelation(
                        from_ref=NewRef(),
                        to_ref=latest.id,
                        relation=RelationType.CLARIFIES,
                    ),
                ),
                outcome=Outcome.REVIEW,
                reason='overlap_conflict',
            )

        # S8: f == L.valid_from (different object)
        if f == latest.valid_from:
            return ResolutionPlan(
                ops=(
                    InsertFact(valid_from=f, valid_to=e, status=FactStatus.NEEDS_REVIEW),
                    AddEvidence(ref=NewRef()),
                    AddRelation(
                        from_ref=NewRef(),
                        to_ref=latest.id,
                        relation=RelationType.CLARIFIES,
                    ),
                ),
                outcome=Outcome.REVIEW,
                reason='same_timestamp_conflict',
            )

        # f < L.valid_from: check if some P in A contains f
        containing_p = [
            row
            for row in active_rows
            if row.valid_from <= f and (row.valid_to is None or f < row.valid_to)
        ]

        # S9: f < L.valid_from and some P in A contains f with a different object
        if containing_p:
            p = containing_p[0]
            return ResolutionPlan(
                ops=(
                    InsertFact(valid_from=f, valid_to=e, status=FactStatus.NEEDS_REVIEW),
                    AddEvidence(ref=NewRef()),
                    AddRelation(
                        from_ref=NewRef(),
                        to_ref=p.id,
                        relation=RelationType.CLARIFIES,
                    ),
                ),
                outcome=Outcome.REVIEW,
                reason='overlap_conflict',
            )

        # S10: f < L.valid_from and no row contains f (backfill)
        # N = earliest row of A with valid_from > f
        # Q = latest row of A with valid_to <= f
        n = min((row for row in active_rows if row.valid_from > f), key=lambda row: row.valid_from)
        matching_q = [row for row in active_rows if row.valid_to is not None and row.valid_to <= f]
        q = max(matching_q, key=lambda row: (row.valid_to, row.valid_from)) if matching_q else None

        new_valid_to = min(e, n.valid_from) if e is not None else n.valid_from
        ops_s10: list[Op] = [
            InsertFact(valid_from=f, valid_to=new_valid_to, status=FactStatus.ACTIVE),
            AddEvidence(ref=NewRef()),
        ]
        if q is not None:
            ops_s10.append(
                AddRelation(from_ref=NewRef(), to_ref=q.id, relation=RelationType.SUPERSEDES)
            )
        ops_s10.append(
            AddRelation(from_ref=n.id, to_ref=NewRef(), relation=RelationType.SUPERSEDES)
        )

        return ResolutionPlan(
            ops=tuple(ops_s10),
            outcome=Outcome.BACKFILLED,
            reason='backfilled',
        )
