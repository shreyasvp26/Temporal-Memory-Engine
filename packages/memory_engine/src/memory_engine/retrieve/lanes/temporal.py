"""Temporal / structured facts retrieval lane per LLD Section 9.4."""

from typing import Any

from memory_engine.domain.enums import Intent
from memory_engine.domain.ids import UserId
from memory_engine.ports.repositories import TemporalFilter
from memory_engine.ports.uow import UnitOfWork
from memory_engine.retrieve.fusion import Hit
from memory_engine.retrieve.planner.rules import QueryPlan

__all__ = ['TemporalLane']


class TemporalLane:
    """Retrieval lane executing structured temporal search over facts."""

    name: str = 'temporal_facts'
    weight: float = 1.5

    def __init__(self, limit: int = 40) -> None:
        self.limit = limit

    def applies(self, plan: QueryPlan, cfg: Any = None) -> bool:
        """Applies when intent is temporal or temporal anchors are present."""
        if plan.time_range is not None or plan.as_of is not None:
            return True

        intent_val = plan.intent.value if hasattr(plan.intent, 'value') else str(plan.intent)
        intent_upper = intent_val.upper()
        temporal_intents = {
            'TEMPORAL_RANGE',
            'INTERVAL',
            'AS_OF',
            'POINT_IN_TIME',
            'DURATION_OR_ORDER',
            'AGGREGATION',
        }
        return intent_upper in temporal_intents or plan.intent in (
            Intent.INTERVAL,
            Intent.POINT_IN_TIME,
            Intent.DURATION_OR_ORDER,
            Intent.AGGREGATION,
        )

    async def execute(
        self,
        uow: UnitOfWork,
        user_id: UserId,
        plan: QueryPlan,
        flt: TemporalFilter,
        limit: int | None = None,
    ) -> list[Hit]:
        """Execute temporal/structured facts search."""
        lim = limit or (
            100 if plan.intent == Intent.AGGREGATION or plan.expects_count else self.limit
        )
        subject_ids = list(plan.entity_ids) if plan.entity_ids else None
        predicates = list(plan.predicates) if plan.predicates else None

        search_method = getattr(uow.search, 'temporal_facts', None) or uow.search.structured_facts
        scored_facts = await search_method(
            user_id=user_id,
            subject_ids=subject_ids,
            predicates=predicates,
            flt=flt,
            limit=lim,
        )

        hits: list[Hit] = []
        for idx, sf in enumerate(scored_facts, start=1):
            hits.append(
                Hit(
                    kind='fact',
                    id=sf.fact.id,
                    lane='temporal_facts',
                    rank=idx,
                    raw_score=sf.score,
                    fact=sf.fact,
                )
            )
        return hits

    async def search(self, ctx: Any) -> list[Hit]:
        """Protocol compatibility wrapper accepting LaneContext."""
        async with ctx.uows.for_user(ctx.user_id) as uow:
            return await self.execute(
                uow=uow,
                user_id=ctx.user_id,
                plan=ctx.plan,
                flt=ctx.flt,
                limit=getattr(ctx, 'limit', None),
            )
