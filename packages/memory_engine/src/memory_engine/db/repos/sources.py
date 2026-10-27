"""PostgreSQL Source repository implementation."""

from collections.abc import Sequence
from datetime import datetime
from typing import Any
from uuid import uuid4

from memory_engine.db.mappers import map_source
from memory_engine.db.pool import wrap_db_errors
from memory_engine.db.sql import (
    INSERT_SOURCE_SQL,
    SET_SOURCE_EMBEDDING_SQL,
    SOFT_DELETE_SOURCE_SQL,
)
from memory_engine.domain.ids import SourceId, UserId
from memory_engine.domain.models import Source
from memory_engine.ports.repositories import (
    InsertedSource,
    NewSource,
    SourceRepository,
    SourceWindow,
)


class PostgresSourceRepository(SourceRepository):
    """PostgreSQL implementation of SourceRepository port."""

    def __init__(self, conn_or_uow: Any) -> None:
        self._conn_source = conn_or_uow

    @property
    def conn(self) -> Any:
        """Resolve database connection from connection or UnitOfWork."""
        if hasattr(self._conn_source, 'conn'):
            return self._conn_source.conn
        return self._conn_source

    async def insert_many(self, items: Sequence[NewSource]) -> list[InsertedSource]:
        """Insert sources with ON CONFLICT (user_id, content_hash) DO NOTHING.

        Returns InsertedSource(id, created) for every input source.
        """
        if not items:
            return []

        results: list[InsertedSource] = []
        async with wrap_db_errors():
            for item in items:
                source_id = item.id if item.id is not None else uuid4()
                kind_val = item.kind.value if hasattr(item.kind, 'value') else str(item.kind)
                role_val = item.role.value if hasattr(item.role, 'value') else str(item.role)
                row = await self.conn.fetchrow(
                    INSERT_SOURCE_SQL,
                    source_id,
                    item.user_id,
                    kind_val,
                    role_val,
                    item.session_id,
                    item.turn_index,
                    item.content,
                    item.source_ts,
                    item.content_hash,
                )
                if row is not None and row.get('id') is not None:
                    results.append(InsertedSource(id=SourceId(row['id']), created=True))
                else:
                    existing = await self.conn.fetchrow(
                        'select id from sources where user_id = $1 and content_hash = $2;',
                        item.user_id,
                        item.content_hash,
                    )
                    existing_id = existing['id'] if existing is not None else source_id
                    results.append(InsertedSource(id=SourceId(existing_id), created=False))
        return results

    async def get_many(self, user_id: UserId, ids: Sequence[SourceId]) -> list[Source]:
        """Fetch multiple sources by ID for a user."""
        if not ids:
            return []

        async with wrap_db_errors():
            rows = await self.conn.fetch(
                'select * from sources where user_id = $1 and id = any($2::uuid[]) '
                'and deleted_at is null order by source_ts;',
                user_id,
                list(ids),
            )
        return [map_source(r) for r in rows]

    async def window(
        self,
        user_id: UserId,
        session_id: str,
        target_ids: Sequence[SourceId],
        context_turns: int,
    ) -> SourceWindow:
        """Construct a window of target sources and up to context_turns earlier turns."""
        if not target_ids:
            return SourceWindow(targets=(), context=())

        async with wrap_db_errors():
            target_rows = await self.conn.fetch(
                'select * from sources where user_id = $1 and session_id = $2 '
                'and id = any($3::uuid[]) and deleted_at is null order by turn_index, source_ts;',
                user_id,
                session_id,
                list(target_ids),
            )
            targets = [map_source(r) for r in target_rows]
            if not targets or context_turns <= 0:
                return SourceWindow(targets=targets, context=())

            min_turn_index = min(t.turn_index for t in targets)
            context_rows = await self.conn.fetch(
                'select * from sources where user_id = $1 and session_id = $2 '
                'and turn_index < $3 and deleted_at is null '
                'order by turn_index desc, source_ts desc limit $4;',
                user_id,
                session_id,
                min_turn_index,
                context_turns,
            )
            # Reverse so context turns are in chronological ascending order
            context = [map_source(r) for r in reversed(context_rows)]
            return SourceWindow(targets=targets, context=context)

    async def get_window(
        self,
        user_id: UserId,
        session_id: str,
        target_ids: Sequence[SourceId],
        context_turns: int,
    ) -> SourceWindow:
        """Convenience alias for window."""
        return await self.window(user_id, session_id, target_ids, context_turns)

    async def set_embeddings(
        self,
        user_id: UserId,
        items: Sequence[tuple[SourceId, list[float]]],
        model_id: str,
    ) -> None:
        """Update dense vector embeddings for sources."""
        if not items:
            return

        async with wrap_db_errors():
            for source_id, embedding in items:
                await self.conn.execute(
                    SET_SOURCE_EMBEDDING_SQL,
                    user_id,
                    source_id,
                    embedding,
                    model_id,
                )

    async def soft_delete(self, user_id: UserId, ids: Sequence[SourceId], at: datetime) -> None:
        """Soft delete sources by setting deleted_at timestamp."""
        if not ids:
            return

        async with wrap_db_errors():
            await self.conn.execute(
                SOFT_DELETE_SOURCE_SQL,
                user_id,
                list(ids),
                at,
            )

    async def hard_delete(self, user_id: UserId, ids: Sequence[SourceId]) -> int:
        """Hard delete sources permanently and return count of deleted rows."""
        if not ids:
            return 0

        async with wrap_db_errors():
            status = await self.conn.execute(
                'delete from sources where user_id = $1 and id = any($2::uuid[]);',
                user_id,
                list(ids),
            )
        if isinstance(status, str) and status.startswith('DELETE '):
            return int(status.split(' ')[1])
        return len(ids)

    async def list_unprocessed_for_session(
        self,
        user_id: UserId,
        session_id: str | None = None,
        limit: int = 50,
    ) -> list[Source]:
        """List unprocessed sources with queued or running extract jobs via Q-13."""
        async with wrap_db_errors():
            if session_id is not None:
                query = """\
select s.* from sources s
join jobs j on j.user_id = s.user_id and j.type = 'extract'
  and j.status in ('queued','running')
  and j.payload->'source_ids' ? s.id::text
where s.user_id = $1 and s.session_id = $2 and s.deleted_at is null
order by s.turn_index desc limit $3;"""
                rows = await self.conn.fetch(query, user_id, session_id, limit)
            else:
                query = """\
select s.* from sources s
join jobs j on j.user_id = s.user_id and j.type = 'extract'
  and j.status in ('queued','running')
  and j.payload->'source_ids' ? s.id::text
where s.user_id = $1 and s.deleted_at is null
order by s.source_ts desc limit $2;"""
                rows = await self.conn.fetch(query, user_id, limit)
        return [map_source(r) for r in rows]
