"""Full-text search (FTS) retrieval lane per LLD Section 9.4."""

from typing import Any

from memory_engine.domain.enums import Intent
from memory_engine.domain.ids import UserId
from memory_engine.ports.repositories import TemporalFilter
from memory_engine.ports.uow import UnitOfWork
from memory_engine.retrieve.fusion import Hit
from memory_engine.retrieve.planner.rules import QueryPlan

__all__ = ['FtsLane']


class FtsLane:
    """Retrieval lane executing full-text keyword search over facts."""

    name: str = 'fts_facts'
    weight: float = 0.8

    def __init__(self, limit: int = 40) -> None:
        self.limit = limit

    def applies(self, plan: QueryPlan, cfg: Any = None) -> bool:
        """FTS lane applies always."""
        return True

    async def execute(
        self,
        uow: UnitOfWork,
        user_id: UserId,
        plan: QueryPlan,
        flt: TemporalFilter,
        limit: int | None = None,
    ) -> list[Hit]:
        """Execute FTS keyword search over facts."""
        lim = limit or (
            100 if plan.intent == Intent.AGGREGATION or plan.expects_count else self.limit
        )
        if hasattr(uow.search, 'fts_facts'):
            scored_facts = await uow.search.fts_facts(
                user_id=user_id,
                query=plan.question,
                flt=flt,
                limit=lim,
            )
        else:
            scored_facts = await uow.search.keyword_facts(
                user_id=user_id,
                query=plan.question,
                flt=flt,
                limit=lim,
            )

        hits: list[Hit] = []
        for idx, sf in enumerate(scored_facts, start=1):
            hits.append(
                Hit(
                    kind='fact',
                    id=sf.fact.id,
                    lane='fts_facts',
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
