"""Graph expansion retrieval lane per LLD Section 9.4."""

from typing import Any
from uuid import UUID

from memory_engine.domain.ids import EntityId, UserId
from memory_engine.ports.repositories import TemporalFilter
from memory_engine.ports.uow import UnitOfWork
from memory_engine.retrieve.fusion import Hit
from memory_engine.retrieve.planner.rules import QueryPlan

__all__ = ['GraphLane']


class GraphLane:
    """Retrieval lane expanding the entity graph around recognized entity hints."""

    name: str = 'graph_expand'
    weight: float = 0.7

    def __init__(self, limit: int = 50, depth: int = 2) -> None:
        self.limit = limit
        self.depth = depth

    def applies(self, plan: QueryPlan, cfg: Any = None) -> bool:
        """Applies when multi_hop is true or entity hints/names/ids exist."""
        has_hints = bool(
            getattr(plan, 'entity_hints', None) or plan.entity_names or plan.entity_ids
        )
        return bool(plan.multi_hop or has_hints)

    async def execute(
        self,
        uow: UnitOfWork,
        user_id: UserId,
        plan: QueryPlan,
        flt: TemporalFilter,
        limit: int | None = None,
        depth: int | None = None,
    ) -> list[Hit]:
        """Resolve entity hints and perform graph expansion search."""
        lim = limit or self.limit
        d = depth or self.depth

        seed_ids: set[EntityId] = set(plan.entity_ids)
        raw_hints = getattr(plan, 'entity_hints', None) or plan.entity_names or ()

        for hint in raw_hints:
            if isinstance(hint, UUID):
                seed_ids.add(EntityId(hint))
            elif isinstance(hint, str):
                normalized = hint.strip().lower()
                if normalized:
                    try:
                        ent = await uow.entities.find_by_norm_name(user_id, normalized)
                        if ent is not None:
                            seed_ids.add(ent.id)
                    except Exception:
                        pass

        if not seed_ids:
            return []

        search_method = getattr(uow.search, 'graph_expand', None) or uow.search.expand_graph
        scored_facts = await search_method(
            user_id=user_id,
            seed_entity_ids=list(seed_ids),
            depth=d,
            flt=flt,
            limit=lim,
        )

        hits: list[Hit] = []
        for idx, sf in enumerate(scored_facts, start=1):
            hits.append(
                Hit(
                    kind='fact',
                    id=sf.fact.id,
                    lane='graph_expand',
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
