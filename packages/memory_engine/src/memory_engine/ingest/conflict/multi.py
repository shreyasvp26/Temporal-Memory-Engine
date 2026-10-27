"""MultiValuedStrategy implementing conflict rules M1 through M4."""

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


class MultiValuedStrategy(ResolutionStrategy):
    """Resolution strategy for multi-valued predicates (rules M1-M4)."""

    cardinality: ClassVar[Cardinality] = Cardinality.MULTI

    def plan(
        self,
        c: Candidate,
        timeline: Sequence[Fact],
        ctx: ResolutionContext,
        hint: JudgeHint | None = None,
    ) -> ResolutionPlan:
        """Evaluate candidate against timeline using rules M1 through M4."""
        # A = timeline rows with status ACTIVE
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
                if hint.verdict == 'coexists':
                    return ResolutionPlan(
                        ops=(
                            InsertFact(valid_from=f, valid_to=e, status=FactStatus.ACTIVE),
                            AddEvidence(ref=NewRef()),
                        ),
                        outcome=Outcome.INSERTED,
                        reason='judge_hint_coexists',
                    )

        # M1: retraction and an open row X in A with same(X) and f > X.valid_from
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

            # M2: retraction otherwise -> none
            has_open_same = any(
                x.object_norm == c.object_norm and x.valid_to is None for x in active_rows
            )
            reason = 'retraction_before_start' if has_open_same else 'retraction_no_target'
            return ResolutionPlan(ops=(), outcome=Outcome.NOOP, reason=reason)

        # M3: A has X with same(X) that is open or overlaps/touches [f, e or f + merge_gap)
        cand_end = e if e is not None else (f + ctx.merge_gap)
        matching_m3 = [
            x
            for x in active_rows
            if x.object_norm == c.object_norm
            and (
                x.valid_to is None
                or (x.valid_to >= f and (cand_end is None or cand_end >= x.valid_from))
            )
        ]
        if matching_m3:
            x = matching_m3[0]
            ops_m3: list[Op] = [
                AddEvidence(ref=x.id),
                BumpConfidence(fact_id=x.id, confidence=max(x.confidence, c.confidence)),
            ]
            if c.valid_from_explicit and f < x.valid_from:
                ops_m3.append(ExtendValidFrom(fact_id=x.id, valid_from=f))
            return ResolutionPlan(
                ops=tuple(ops_m3),
                outcome=Outcome.MERGED,
                reason='merged_same_fact',
            )

        # M4: otherwise
        return ResolutionPlan(
            ops=(
                InsertFact(valid_from=f, valid_to=e, status=FactStatus.ACTIVE),
                AddEvidence(ref=NewRef()),
            ),
            outcome=Outcome.INSERTED,
            reason='inserted_multi',
        )
