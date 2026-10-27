"""PostgreSQL Search repository implementation."""

from collections.abc import Sequence
from datetime import datetime
from typing import Any

from memory_engine.db.mappers import (
    map_evidence_snippet,
    map_fact,
    map_scored_fact,
    map_scored_source,
)
from memory_engine.db.pool import wrap_db_errors
from memory_engine.db.sql import (
    Q07_VECTOR_SEARCH_FACTS,
    Q08_FTS_SEARCH_FACTS,
    Q09_TEMPORAL_RANGE_FACTS,
    Q10_VECTOR_SEARCH_SOURCES,
    Q11_GRAPH_EXPANSION,
    Q14_EVIDENCE_FOR_FACTS,
    Q15_SUPERSESSION_NEIGHBORS,
    inject_temporal_filter,
)
from memory_engine.domain.ids import EntityId, FactId, UserId
from memory_engine.domain.models import Fact
from memory_engine.ports.repositories import (
    EvidenceSnippet,
    ScoredFact,
    ScoredSource,
    SearchRepository,
    TemporalFilter,
)


class PostgresSearchRepository(SearchRepository):
    """PostgreSQL implementation of SearchRepository port."""

    def __init__(self, conn_or_uow: Any) -> None:
        self._conn_source = conn_or_uow

    @property
    def conn(self) -> Any:
        """Resolve database connection from connection or UnitOfWork."""
        if hasattr(self._conn_source, 'conn'):
            return self._conn_source.conn
        return self._conn_source

    async def vector_facts(
        self,
        user_id: UserId,
        embedding: Sequence[float],
        flt: TemporalFilter,
        limit: int,
    ) -> list[ScoredFact]:
        """HNSW vector search over facts with iterative scan and temporal filter (Q-07)."""
        async with wrap_db_errors():
            await self.conn.execute("set local hnsw.iterative_scan = 'relaxed_order';")
            await self.conn.execute('set local hnsw.ef_search = 100;')
            sql, frag_params = inject_temporal_filter(Q07_VECTOR_SEARCH_FACTS, flt, alias='f')
            rows = await self.conn.fetch(
                sql,
                user_id,
                list(embedding),
                *frag_params,
                limit,
            )
        scored = [map_scored_fact(r) for r in rows]
        # Re-sort in Python because relaxed ordering may return slightly out-of-order rows
        scored.sort(key=lambda sf: sf.score, reverse=True)
        return scored[:limit]

    async def keyword_facts(
        self,
        user_id: UserId,
        query: str,
        flt: TemporalFilter,
        limit: int,
    ) -> list[ScoredFact]:
        """Perform full-text keyword search over facts with websearch tsquery (Q-08)."""
        async with wrap_db_errors():
            sql, frag_params = inject_temporal_filter(Q08_FTS_SEARCH_FACTS, flt, alias='f')
            rows = await self.conn.fetch(
                sql,
                user_id,
                query,
                *frag_params,
                limit,
            )
        return [map_scored_fact(r) for r in rows]

    async def fts_facts(
        self,
        user_id: UserId,
        query: str,
        flt: TemporalFilter,
        limit: int,
    ) -> list[ScoredFact]:
        """Convenience alias for keyword_facts (Q-08)."""
        return await self.keyword_facts(user_id, query, flt, limit)

    async def structured_facts(
        self,
        user_id: UserId,
        subject_ids: Sequence[EntityId] | None,
        predicates: Sequence[str] | None,
        flt: TemporalFilter,
        limit: int,
    ) -> list[ScoredFact]:
        """Perform structured lookup over subject/object entities and predicates (Q-09)."""
        subj_list = list(subject_ids) if subject_ids is not None else None
        pred_list = list(predicates) if predicates is not None else None
        async with wrap_db_errors():
            sql, frag_params = inject_temporal_filter(Q09_TEMPORAL_RANGE_FACTS, flt, alias='f')
            rows = await self.conn.fetch(
                sql,
                user_id,
                subj_list,
                pred_list,
                *frag_params,
                limit,
            )
        return [map_scored_fact(r) for r in rows]

    async def temporal_facts(
        self,
        user_id: UserId,
        subject_ids: Sequence[EntityId] | None,
        predicates: Sequence[str] | None,
        flt: TemporalFilter,
        limit: int,
    ) -> list[ScoredFact]:
        """Convenience alias for structured_facts (Q-09)."""
        return await self.structured_facts(user_id, subject_ids, predicates, flt, limit)

    async def vector_sources(
        self,
        user_id: UserId,
        embedding: Sequence[float],
        limit: int,
        before: datetime | None,
    ) -> list[ScoredSource]:
        """Perform HNSW vector search over conversation sources with iterative scan (Q-10)."""
        async with wrap_db_errors():
            await self.conn.execute("set local hnsw.iterative_scan = 'relaxed_order';")
            await self.conn.execute('set local hnsw.ef_search = 100;')
            rows = await self.conn.fetch(
                Q10_VECTOR_SEARCH_SOURCES,
                user_id,
                list(embedding),
                limit,
                before,
            )
        scored = [map_scored_source(r) for r in rows]
        scored.sort(key=lambda ss: ss.score, reverse=True)
        return scored[:limit]

    async def keyword_sources(
        self,
        user_id: UserId,
        query: str,
        limit: int,
        before: datetime | None,
    ) -> list[ScoredSource]:
        """Perform full-text search over sources with websearch query."""
        sql = """\
select s.*, ts_rank_cd(s.tsv, q) as score
from sources s, websearch_to_tsquery('english', $2) q
where s.user_id = $1 and s.tsv @@ q and s.deleted_at is null
  and ($4::timestamptz is null or s.source_ts <= $4)
order by score desc limit $3;"""
        async with wrap_db_errors():
            rows = await self.conn.fetch(sql, user_id, query, limit, before)
        return [map_scored_source(r) for r in rows]

    async def expand_graph(
        self,
        user_id: UserId,
        seed_entity_ids: Sequence[EntityId],
        depth: int,
        flt: TemporalFilter,
        limit: int,
    ) -> list[ScoredFact]:
        """Expand entity graph up to depth hops with degree cap 15 per node (Q-11)."""
        if not seed_entity_ids:
            return []
        async with wrap_db_errors():
            sql, frag_params = inject_temporal_filter(Q11_GRAPH_EXPANSION, flt, alias='f')
            rows = await self.conn.fetch(
                sql,
                user_id,
                list(seed_entity_ids),
                depth,
                *frag_params,
                limit,
            )
        return [map_scored_fact(r) for r in rows]

    async def graph_expand(
        self,
        user_id: UserId,
        seed_entity_ids: Sequence[EntityId],
        depth: int,
        flt: TemporalFilter,
        limit: int,
    ) -> list[ScoredFact]:
        """Convenience alias for expand_graph (Q-11)."""
        return await self.expand_graph(user_id, seed_entity_ids, depth, flt, limit)

    async def evidence_for(
        self,
        user_id: UserId,
        fact_ids: Sequence[FactId],
        per_fact: int = 5,
    ) -> dict[FactId, list[EvidenceSnippet]]:
        """Fetch source snippets supporting the specified facts (Q-14 / Q-13)."""
        if not fact_ids:
            return {}
        async with wrap_db_errors():
            rows = await self.conn.fetch(Q14_EVIDENCE_FOR_FACTS, user_id, list(fact_ids))
        result: dict[FactId, list[EvidenceSnippet]] = {fid: [] for fid in fact_ids}
        for r in rows:
            snippet = map_evidence_snippet(r)
            fid = snippet.fact_id
            if fid is not None and fid in result and len(result[fid]) < per_fact:
                result[fid].append(snippet)
        return result

    async def chain_neighbors(
        self,
        user_id: UserId,
        fact_ids: Sequence[FactId],
    ) -> list[Fact]:
        """Fetch 1-hop supersession neighbor facts (Q-15)."""
        if not fact_ids:
            return []
        async with wrap_db_errors():
            rows = await self.conn.fetch(Q15_SUPERSESSION_NEIGHBORS, user_id, list(fact_ids))
        return [map_fact(r) for r in rows]
