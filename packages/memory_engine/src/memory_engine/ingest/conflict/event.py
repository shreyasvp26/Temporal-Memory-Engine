"""EventStrategy implementing conflict rules E1 through E3."""

from collections.abc import Sequence
from typing import ClassVar

from memory_engine.domain.enums import Cardinality, FactStatus
from memory_engine.domain.models import Candidate, Fact, JudgeHint
from memory_engine.ingest.conflict.plan import (
    AddEvidence,
    BumpConfidence,
    InsertFact,
    NewRef,
    Outcome,
    ResolutionContext,
    ResolutionPlan,
    ResolutionStrategy,
)


class EventStrategy(ResolutionStrategy):
    """Resolution strategy for event predicates (rules E1-E3)."""

    cardinality: ClassVar[Cardinality] = Cardinality.EVENT

    def plan(
        self,
        c: Candidate,
        timeline: Sequence[Fact],
        ctx: ResolutionContext,
        hint: JudgeHint | None = None,
    ) -> ResolutionPlan:
        """Evaluate candidate against timeline using rules E1 through E3."""
        del ctx, hint  # Not used by event strategy

        # E1: retraction -> none
        if c.is_retraction:
            return ResolutionPlan(
                ops=(),
                outcome=Outcome.NOOP,
                reason='event_retraction_unsupported',
            )

        # E2: a row X in the timeline has X.event_hash == c.event_hash
        if c.event_hash is not None:
            matching = [row for row in timeline if row.event_hash == c.event_hash]
            if matching:
                x = matching[0]
                return ResolutionPlan(
                    ops=(
                        AddEvidence(ref=x.id),
                        BumpConfidence(fact_id=x.id, confidence=max(x.confidence, c.confidence)),
                    ),
                    outcome=Outcome.MERGED,
                    reason='deduped_by_event_hash',
                )

        # E3: otherwise
        return ResolutionPlan(
            ops=(
                InsertFact(valid_from=c.valid_from, valid_to=c.valid_to, status=FactStatus.ACTIVE),
                AddEvidence(ref=NewRef()),
            ),
            outcome=Outcome.INSERTED,
            reason='inserted_event',
        )
