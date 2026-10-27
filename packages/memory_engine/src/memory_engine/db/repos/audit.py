"""PostgreSQL Audit repository implementation."""

from typing import Any
from uuid import UUID

from memory_engine.db.pool import wrap_db_errors
from memory_engine.db.sql import INSERT_AUDIT_LOG_SQL
from memory_engine.domain.ids import UserId
from memory_engine.ports.repositories import AuditRepository


class PostgresAuditRepository(AuditRepository):
    """PostgreSQL implementation of AuditRepository port."""

    def __init__(self, conn_or_uow: Any) -> None:
        self._conn_source = conn_or_uow

    @property
    def conn(self) -> Any:
        """Resolve database connection from connection or UnitOfWork."""
        if hasattr(self._conn_source, 'conn'):
            return self._conn_source.conn
        return self._conn_source

    async def append(
        self,
        user_id: UserId,
        actor: str,
        action: str,
        target_type: str,
        target_id: UUID | None,
        before: dict[str, Any] | None,
        after: dict[str, Any] | None,
    ) -> None:
        """Append an audit log entry for fact modifications, purges, or reviews."""
        async with wrap_db_errors():
            await self.conn.execute(
                INSERT_AUDIT_LOG_SQL,
                user_id,
                actor,
                action,
                target_type,
                target_id,
                before,
                after,
            )

    async def record(
        self,
        user_id: UserId,
        actor: str,
        action: str,
        target_type: str,
        target_id: UUID | None = None,
        before: dict[str, Any] | None = None,
        after: dict[str, Any] | None = None,
    ) -> None:
        """Convenience alias for append."""
        await self.append(
            user_id=user_id,
            actor=actor,
            action=action,
            target_type=target_type,
            target_id=target_id,
            before=before,
            after=after,
        )

    async def list_for_user(
        self,
        user_id: UserId,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """List recent audit log entries for a user."""
        async with wrap_db_errors():
            rows = await self.conn.fetch(
                'select * from audit_log where user_id = $1 order by created_at desc limit $2;',
                user_id,
                limit,
            )
        return [dict(r) for r in rows]
