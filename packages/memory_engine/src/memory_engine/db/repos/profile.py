"""PostgreSQL Profile repository implementation."""

from typing import Any

from memory_engine.db.pool import wrap_db_errors
from memory_engine.db.sql import Q16_PURGE_USER
from memory_engine.domain.ids import UserId
from memory_engine.ports.repositories import ProfileRepository


class PostgresProfileRepository(ProfileRepository):
    """PostgreSQL implementation of ProfileRepository port."""

    def __init__(self, conn_or_uow: Any) -> None:
        self._conn_source = conn_or_uow

    @property
    def conn(self) -> Any:
        """Resolve database connection from connection or UnitOfWork."""
        if hasattr(self._conn_source, 'conn'):
            return self._conn_source.conn
        return self._conn_source

    async def ensure(self, user_id: UserId) -> None:
        """Ensure auth stub and profile rows exist for the user."""
        async with wrap_db_errors():
            await self.conn.execute(
                'insert into auth.users (id) values ($1) on conflict do nothing;',
                user_id,
            )
            await self.conn.execute(
                "insert into profiles (user_id, timezone) values ($1, 'UTC') "
                'on conflict (user_id) do nothing;',
                user_id,
            )

    async def timezone(self, user_id: UserId) -> str:
        """Retrieve user's IANA timezone, defaulting to 'UTC' if not found."""
        async with wrap_db_errors():
            val = await self.conn.fetchval(
                'select timezone from profiles where user_id = $1;',
                user_id,
            )
        return str(val) if val is not None else 'UTC'

    async def set_timezone(self, user_id: UserId, tz: str) -> None:
        """Update or insert user timezone."""
        async with wrap_db_errors():
            await self.ensure(user_id)
            await self.conn.execute(
                """\
insert into profiles (user_id, timezone) values ($1, $2)
on conflict (user_id) do update set timezone = excluded.timezone;""",
                user_id,
                tz,
            )

    async def list_user_ids(self) -> list[UserId]:
        """List all user IDs (worker-only operation)."""
        async with wrap_db_errors():
            rows = await self.conn.fetch('select user_id from profiles order by created_at;')
        return [UserId(r['user_id']) for r in rows]

    async def purge_user(self, user_id: UserId) -> int:
        """Purge a user entirely (cascades to all sources, entities, facts, etc.) via Q-16."""
        async with wrap_db_errors():
            status = await self.conn.execute(Q16_PURGE_USER, user_id)
        if isinstance(status, str) and status.startswith('DELETE '):
            return int(status.split(' ')[1])
        return 1

    async def export_user(self, user_id: UserId) -> dict[str, Any]:
        """Export all user data for GDPR portability."""
        async with wrap_db_errors():
            tz = await self.timezone(user_id)
            fact_rows = await self.conn.fetch(
                "select * from facts where user_id = $1 and status <> 'deleted' "
                'order by valid_from;',
                user_id,
            )
            source_rows = await self.conn.fetch(
                'select * from sources where user_id = $1 and deleted_at is null '
                'order by source_ts;',
                user_id,
            )
            entity_rows = await self.conn.fetch(
                'select * from entities where user_id = $1 order by canonical_name;',
                user_id,
            )
            message_rows = await self.conn.fetch(
                'select * from messages where user_id = $1 order by created_at;',
                user_id,
            )

        return {
            'user_id': str(user_id),
            'timezone': tz,
            'facts': [dict(r) for r in fact_rows],
            'sources': [dict(r) for r in source_rows],
            'entities': [dict(r) for r in entity_rows],
            'messages': [dict(r) for r in message_rows],
        }
