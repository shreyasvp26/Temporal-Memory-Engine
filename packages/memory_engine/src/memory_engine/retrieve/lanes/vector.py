"""Vector retrieval lane executing vector_facts and vector_sources per LLD Section 9.4."""

from collections.abc import Sequence
from typing import Any

from memory_engine.domain.enums import Intent
from memory_engine.domain.ids import UserId
from memory_engine.ports.repositories import TemporalFilter
from memory_engine.ports.uow import UnitOfWork
from memory_engine.retrieve.fusion import Hit
from memory_engine.retrieve.planner.rules import QueryPlan

__all__ = ['VectorLane']


class VectorLane:
    """Retrieval lane searching vector_facts and vector_sources using query embeddings."""

    name: str = 'vector_facts'
    weight: float = 1.0

    def __init__(
        self,
        fact_limit: int = 40,
        source_limit: int = 20,
        include_sources: bool = True,
    ) -> None:
        self.fact_limit = fact_limit
        self.source_limit = source_limit
        self.include_sources = include_sources

    def applies(self, plan: QueryPlan, cfg: Any = None) -> bool:
        """Vector lane applies always unless embedder is unavailable."""
        return True

    async def execute(
        self,
        uow: UnitOfWork,
        user_id: UserId,
        plan: QueryPlan,
        flt: TemporalFilter,
        embedding: Sequence[float] | None,
        fact_limit: int | None = None,
        source_limit: int | None = None,
    ) -> list[Hit]:
        """Execute vector search over facts and sources."""
        if embedding is None:
            return []

        flim = fact_limit or (
            100 if plan.intent == Intent.AGGREGATION or plan.expects_count else self.fact_limit
        )

        scored_facts = await uow.search.vector_facts(
            user_id=user_id,
            embedding=embedding,
            flt=flt,
            limit=flim,
        )

        hits: list[Hit] = []
        for idx, sf in enumerate(scored_facts, start=1):
            hits.append(
                Hit(
                    kind='fact',
                    id=sf.fact.id,
                    lane='vector_facts',
                    rank=idx,
                    raw_score=sf.score,
                    fact=sf.fact,
                )
            )

        if self.include_sources:
            slim = source_limit or self.source_limit
            scored_sources = await uow.search.vector_sources(
                user_id=user_id,
                embedding=embedding,
                limit=slim,
                before=plan.question_ts,
            )
            for idx, ss in enumerate(scored_sources, start=1):
                hits.append(
                    Hit(
                        kind='source',
                        id=ss.source.id,
                        lane='vector_sources',
                        rank=idx,
                        raw_score=ss.score,
                        source=ss.source,
                    )
                )

        return hits

    async def search(self, ctx: Any) -> list[Hit]:
        """Protocol compatibility wrapper accepting LaneContext."""
        embedding = None
        if hasattr(ctx, 'query_vector') and callable(ctx.query_vector):
            embedding = await ctx.query_vector()
        async with ctx.uows.for_user(ctx.user_id) as uow:
            return await self.execute(
                uow=uow,
                user_id=ctx.user_id,
                plan=ctx.plan,
                flt=ctx.flt,
                embedding=embedding,
                fact_limit=getattr(ctx, 'limit', None),
            )
