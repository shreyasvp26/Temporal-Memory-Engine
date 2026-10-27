"""PostgreSQL Fact repository implementation."""

import builtins
import re
from collections.abc import Sequence
from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

from memory_engine.db.mappers import (
    map_evidence_snippet,
    map_fact,
)
from memory_engine.db.pool import wrap_db_errors
from memory_engine.db.sql import (
    INSERT_FACT_EVIDENCE_SQL,
    INSERT_FACT_RELATION_SQL,
    INSERT_FACT_SQL,
    Q06_ACTIVE_CANDIDATES,
    Q06_CLOSE_FACT,
    Q12_SUPERSEDES_CHAIN,
    Q14_ACCESS_RECORDING,
    Q14_EVIDENCE_FOR_FACTS,
    Q15_FACTS_ONLY_SUPPORTED,
    Q15_SUPERSESSION_NEIGHBORS,
    Q18_HARD_DELETE_TOMBSTONES,
    UPDATE_FACT_CONFIDENCE_SQL,
    UPDATE_FACT_SALIENCE_SQL,
    UPDATE_FACT_STATUS_SQL,
    UPDATE_FACT_VALID_FROM_SQL,
)
from memory_engine.domain.enums import FactOrigin, FactStatus
from memory_engine.domain.ids import EntityId, FactId, SourceId, UserId
from memory_engine.domain.models import Candidate, Fact, FactEvidence, FactRelation
from memory_engine.ports.repositories import (
    EvidenceSnippet,
    FactListQuery,
    FactRepository,
    Page,
)


def _parse_affected_rows(res: Any) -> int:
    """Parse number of affected rows from database response."""
    if isinstance(res, int):
        return res
    if not isinstance(res, str):
        return 0
    parts = res.strip().split()
    if not parts:
        return 0
    try:
        return int(parts[-1])
    except ValueError:
        return 0


class PostgresFactRepository(FactRepository):
    """PostgreSQL implementation of FactRepository port."""

    def __init__(self, conn_or_uow: Any) -> None:
        self._conn_source = conn_or_uow

    @property
    def conn(self) -> Any:
        """Resolve database connection from connection or UnitOfWork."""
        if hasattr(self._conn_source, 'conn'):
            return self._conn_source.conn
        return self._conn_source

    async def load_timeline(
        self,
        user_id: UserId,
        subject_id: EntityId,
        predicate: str,
        *,
        lock: bool = False,
    ) -> list[Fact]:
        """Load ordered active candidates along a timeline for conflict evaluation (Q-05 / Q-06)."""
        async with wrap_db_errors():
            if lock:
                rows = await self.conn.fetch(
                    Q06_ACTIVE_CANDIDATES,
                    user_id,
                    subject_id,
                    predicate,
                )
            else:
                unlocked_sql = re.sub(
                    r'\s+for\s+update\s*;?',
                    ';',
                    Q06_ACTIVE_CANDIDATES,
                    flags=re.IGNORECASE,
                )
                rows = await self.conn.fetch(
                    unlocked_sql,
                    user_id,
                    subject_id,
                    predicate,
                )
        return [map_fact(r) for r in rows]

    async def find_for_resolution(
        self,
        user_id: UserId,
        subject_id: EntityId,
        predicate: str,
    ) -> list[Fact]:
        """Load candidate timeline with FOR UPDATE lock for conflict resolution (Q-06)."""
        return await self.load_timeline(user_id, subject_id, predicate, lock=True)

    async def insert(
        self,
        c: Candidate,
        *,
        valid_from: datetime,
        valid_to: datetime | None,
        status: FactStatus,
        origin: FactOrigin,
        model_id: str,
        now: datetime,
    ) -> Fact:
        """Insert a new fact row and return the persisted Fact model."""
        fact_id = FactId(uuid4())
        cardinality_val = (
            c.cardinality.value if hasattr(c.cardinality, 'value') else str(c.cardinality)
        )
        time_precision_val = (
            c.time_precision.value if hasattr(c.time_precision, 'value') else str(c.time_precision)
        )
        status_val = status.value if hasattr(status, 'value') else str(status)
        origin_val = origin.value if hasattr(origin, 'value') else str(origin)

        async with wrap_db_errors():
            await self.conn.fetchrow(
                INSERT_FACT_SQL,
                fact_id,
                c.user_id,
                c.subject_id,
                c.predicate,
                cardinality_val,
                c.object_entity_id,
                c.object_value,
                c.object_norm,
                c.fact_text,
                list(c.embedding),
                model_id,
                valid_from,
                valid_to,
                time_precision_val,
                c.valid_from_explicit,
                now,
                c.confidence,
                1.0,  # salience default
                status_val,
                origin_val,
                c.event_hash,
            )

        return Fact(
            id=fact_id,
            user_id=c.user_id,
            subject_id=c.subject_id,
            predicate=c.predicate,
            cardinality=c.cardinality,
            object_entity_id=c.object_entity_id,
            object_value=c.object_value,
            object_norm=c.object_norm,
            fact_text=c.fact_text,
            valid_from=valid_from,
            valid_to=valid_to,
            time_precision=c.time_precision,
            valid_from_explicit=c.valid_from_explicit,
            recorded_at=now,
            invalidated_at=None,
            confidence=c.confidence,
            salience=1.0,
            access_count=0,
            last_accessed_at=None,
            status=status,
            origin=origin,
            event_hash=c.event_hash,
        )

    async def close(
        self,
        user_id: UserId,
        fact_id: FactId,
        valid_to: datetime,
        now: datetime,
    ) -> None:
        """Close an open fact timeline bound at valid_to with invalidated_at = now (Q-06)."""
        async with wrap_db_errors():
            await self.conn.execute(Q06_CLOSE_FACT, user_id, fact_id, valid_to, now)

    async def extend_valid_from(
        self,
        user_id: UserId,
        fact_id: FactId,
        valid_from: datetime,
    ) -> None:
        """Extend a fact's valid_from bound backward in time."""
        async with wrap_db_errors():
            await self.conn.execute(UPDATE_FACT_VALID_FROM_SQL, user_id, fact_id, valid_from)

    async def set_status(
        self,
        user_id: UserId,
        fact_id: FactId,
        status: FactStatus,
        now: datetime,
    ) -> None:
        """Update fact status and record invalidation timestamp if non-active."""
        status_val = status.value if hasattr(status, 'value') else str(status)
        invalidated_at = now if status != FactStatus.ACTIVE else None
        async with wrap_db_errors():
            await self.conn.execute(
                UPDATE_FACT_STATUS_SQL,
                user_id,
                fact_id,
                status_val,
                invalidated_at,
            )

    async def bump_confidence(
        self,
        user_id: UserId,
        fact_id: FactId,
        confidence: float,
    ) -> None:
        """Update the confidence score for an existing fact."""
        async with wrap_db_errors():
            await self.conn.execute(UPDATE_FACT_CONFIDENCE_SQL, user_id, fact_id, confidence)

    async def add_evidence(self, ev: FactEvidence, user_id: UserId) -> None:
        """Record supporting source evidence for a fact."""
        async with wrap_db_errors():
            await self.conn.execute(
                INSERT_FACT_EVIDENCE_SQL,
                ev.fact_id,
                ev.source_id,
                user_id,
                ev.span_start,
                ev.span_end,
            )

    async def add_relation(
        self,
        user_id: UserId,
        rel: FactRelation,
        job_id: UUID | None = None,
    ) -> None:
        """Link two facts via a typed relationship (supersedes, clarifies, etc.)."""
        relation_val = rel.relation.value if hasattr(rel.relation, 'value') else str(rel.relation)
        async with wrap_db_errors():
            await self.conn.execute(
                INSERT_FACT_RELATION_SQL,
                user_id,
                rel.from_fact_id,
                rel.to_fact_id,
                relation_val,
                job_id,
            )

    async def get(self, user_id: UserId, fact_id: FactId) -> Fact | None:
        """Fetch a single non-deleted fact by ID."""
        async with wrap_db_errors():
            row = await self.conn.fetchrow(
                "select * from facts where user_id = $1 and id = $2 and status <> 'deleted';",
                user_id,
                fact_id,
            )
        return map_fact(row) if row is not None else None

    async def get_many(self, user_id: UserId, fact_ids: Sequence[FactId]) -> list[Fact]:
        """Fetch multiple non-deleted facts by ID."""
        if not fact_ids:
            return []
        async with wrap_db_errors():
            rows = await self.conn.fetch(
                'select * from facts where user_id = $1 and id = any($2::uuid[]) '
                "and status <> 'deleted';",
                user_id,
                list(fact_ids),
            )
        return [map_fact(r) for r in rows]

    async def list(self, user_id: UserId, q: FactListQuery) -> Page[Fact]:
        """List paginated facts matching query parameters."""
        conditions: list[str] = ['user_id = $1']
        params: list[Any] = [user_id]

        if q.status is not None:
            status_val = q.status.value if hasattr(q.status, 'value') else str(q.status)
            params.append(status_val)
            conditions.append(f'status = ${len(params)}')
        else:
            conditions.append("status <> 'deleted'")

        if q.predicate is not None:
            params.append(q.predicate)
            conditions.append(f'predicate = ${len(params)}')

        if q.entity_id is not None:
            params.append(q.entity_id)
            idx = len(params)
            conditions.append(f'(subject_id = ${idx} or object_entity_id = ${idx})')

        if q.current_only:
            conditions.append("status = 'active' and valid_to is null and invalidated_at is null")

        if q.as_of is not None:
            params.append(q.as_of)
            idx = len(params)
            conditions.append(f'valid_from <= ${idx} and (valid_to is null or ${idx} < valid_to)')

        if q.q is not None and q.q.strip():
            params.append(f'%{q.q.strip()}%')
            idx = len(params)
            conditions.append(f'(fact_text ilike ${idx} or object_norm ilike ${idx})')

        offset = 0
        if q.cursor is not None:
            try:
                offset = max(0, int(q.cursor))
            except ValueError:
                offset = 0

        order_by = 'valid_from desc'
        if q.sort == 'valid_from_asc':
            order_by = 'valid_from asc'
        elif q.sort == 'recorded_at':
            order_by = 'recorded_at desc'
        elif q.sort == 'confidence':
            order_by = 'confidence desc'

        page_size = max(1, min(q.page_size, 500))
        params.append(page_size + 1)
        limit_idx = len(params)
        params.append(offset)
        offset_idx = len(params)

        sql = (
            f'select * from facts where {" and ".join(conditions)} '
            f'order by {order_by} limit ${limit_idx} offset ${offset_idx};'
        )
        async with wrap_db_errors():
            rows = await self.conn.fetch(sql, *params)

        items = [map_fact(r) for r in rows[:page_size]]
        next_cursor = str(offset + page_size) if len(rows) > page_size else None
        return Page(items=items, next_cursor=next_cursor)

    async def list_page(self, user_id: UserId, q: FactListQuery) -> Page[Fact]:
        """Convenience alias for list()."""
        return await self.list(user_id, q)

    async def chain(self, user_id: UserId, fact_id: FactId) -> builtins.list[Fact]:
        """Walk the supersedes relationship chain bounded to 20 hops (Q-12)."""
        async with wrap_db_errors():
            rows = await self.conn.fetch(Q12_SUPERSEDES_CHAIN, user_id, fact_id)
        return [map_fact(r) for r in rows]

    async def only_supported_by(
        self,
        user_id: UserId,
        source_ids: Sequence[SourceId],
    ) -> builtins.list[FactId]:
        """Find facts that are exclusively supported by the specified sources (Q-15)."""
        if not source_ids:
            return []
        async with wrap_db_errors():
            rows = await self.conn.fetch(Q15_FACTS_ONLY_SUPPORTED, user_id, list(source_ids))
        return [FactId(r['id']) for r in rows]

    async def hard_delete(self, user_id: UserId, fact_ids: Sequence[FactId]) -> int:
        """Hard delete facts by ID for GDPR purge / cleanup."""
        if not fact_ids:
            return 0
        async with wrap_db_errors():
            res = await self.conn.execute(
                'delete from facts where user_id = $1 and id = any($2::uuid[]);',
                user_id,
                list(fact_ids),
            )
        return _parse_affected_rows(res)

    async def record_access(
        self,
        user_id: UserId,
        fact_ids: Sequence[FactId],
        at: datetime,
    ) -> None:
        """Bump access_count and update last_accessed_at for accessed facts (Q-14)."""
        if not fact_ids:
            return
        async with wrap_db_errors():
            await self.conn.execute(Q14_ACCESS_RECORDING, user_id, list(fact_ids), at)

    async def decay_candidates(self, user_id: UserId, now: datetime) -> builtins.list[Fact]:
        """Fetch candidates for decay calculation (up to 500 active/archived facts)."""
        async with wrap_db_errors():
            rows = await self.conn.fetch(
                "select * from facts where user_id = $1 and status in ('active', 'archived') "
                'order by valid_from asc limit 500;',
                user_id,
            )
        return [map_fact(r) for r in rows]

    async def update_salience(
        self,
        user_id: UserId,
        items: Sequence[tuple[FactId, float]],
    ) -> None:
        """Batch update salience values for facts."""
        if not items:
            return
        async with wrap_db_errors():
            for fact_id, salience in items:
                await self.conn.execute(UPDATE_FACT_SALIENCE_SQL, user_id, fact_id, salience)

    async def manual_update(
        self,
        user_id: UserId,
        fact_id: FactId,
        *,
        fact_text: str | None = None,
        object_value: str | None = None,
        object_norm: str | None = None,
        valid_from: datetime | None = None,
        valid_to: datetime | None = None,
        status: FactStatus | str | None = None,
        confidence: float | None = None,
        invalidated_at: datetime | None = None,
    ) -> None:
        """Perform a manual attribute update on a fact."""
        updates: list[str] = []
        params: list[Any] = [user_id, fact_id]

        if fact_text is not None:
            params.append(fact_text)
            updates.append(f'fact_text = ${len(params)}')
        if object_value is not None:
            params.append(object_value)
            updates.append(f'object_value = ${len(params)}')
        if object_norm is not None:
            params.append(object_norm)
            updates.append(f'object_norm = ${len(params)}')
        if valid_from is not None:
            params.append(valid_from)
            updates.append(f'valid_from = ${len(params)}')
        if valid_to is not None:
            params.append(valid_to)
            updates.append(f'valid_to = ${len(params)}')
        if status is not None:
            params.append(status.value if hasattr(status, 'value') else str(status))
            updates.append(f'status = ${len(params)}')
        if confidence is not None:
            params.append(confidence)
            updates.append(f'confidence = ${len(params)}')
        if invalidated_at is not None:
            params.append(invalidated_at)
            updates.append(f'invalidated_at = ${len(params)}')

        if updates:
            sql = f'update facts set {", ".join(updates)} where user_id = $1 and id = $2;'
            async with wrap_db_errors():
                await self.conn.execute(sql, *params)

    async def hard_delete_expired_tombstones(self, before: datetime) -> int:
        """Purge expired soft-deleted facts older than retention horizon (Q-18)."""
        async with wrap_db_errors():
            res = await self.conn.execute(Q18_HARD_DELETE_TOMBSTONES, before)
        return _parse_affected_rows(res)

    async def evidence_for(
        self,
        user_id: UserId,
        fact_ids: Sequence[FactId],
        per_fact: int = 5,
    ) -> dict[FactId, builtins.list[EvidenceSnippet]]:
        """Fetch evidence snippets for specified facts (Q-14 / Q-13)."""
        if not fact_ids:
            return {}
        async with wrap_db_errors():
            rows = await self.conn.fetch(Q14_EVIDENCE_FOR_FACTS, user_id, list(fact_ids))
        result: dict[FactId, builtins.list[EvidenceSnippet]] = {fid: [] for fid in fact_ids}
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
    ) -> builtins.list[Fact]:
        """Fetch immediate 1-hop supersession neighbors for facts (Q-15)."""
        if not fact_ids:
            return []
        async with wrap_db_errors():
            rows = await self.conn.fetch(Q15_SUPERSESSION_NEIGHBORS, user_id, list(fact_ids))
        return [map_fact(r) for r in rows]
