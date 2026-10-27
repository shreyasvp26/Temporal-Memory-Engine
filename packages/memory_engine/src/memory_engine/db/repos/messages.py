"""PostgreSQL Message repository implementation."""

from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

from memory_engine.db.mappers import map_message
from memory_engine.db.pool import wrap_db_errors
from memory_engine.db.sql import (
    INSERT_MESSAGE_SQL,
    Q12_RECENT_MESSAGES,
    Q13_UNPROCESSED_USER_SOURCES,
)
from memory_engine.domain.clock import Clock, SystemClock
from memory_engine.domain.ids import SourceId, UserId
from memory_engine.domain.models import Message
from memory_engine.ports.repositories import (
    InsertedMessage,
    MessageRepository,
    NewMessage,
)


class PostgresMessageRepository(MessageRepository):
    """PostgreSQL implementation of MessageRepository port."""

    def __init__(self, conn_or_uow: Any, clock: Clock | None = None) -> None:
        self._conn_source = conn_or_uow
        self._clock: Clock = clock if clock is not None else SystemClock()

    @property
    def conn(self) -> Any:
        """Resolve database connection from connection or UnitOfWork."""
        if hasattr(self._conn_source, 'conn'):
            return self._conn_source.conn
        return self._conn_source

    async def ensure_conversation(
        self,
        user_id: UserId,
        conversation_id: UUID | None,
        now: datetime,
    ) -> UUID:
        """Ensure a conversation exists, creating one if conversation_id is None."""
        cid = conversation_id if conversation_id is not None else uuid4()
        async with wrap_db_errors():
            await self.conn.execute(
                """\
insert into conversations (id, user_id, title, created_at, updated_at)
values ($1, $2, null, $3, $3)
on conflict (id) do update set updated_at = $3;""",
                cid,
                user_id,
                now,
            )
        return cid

    async def insert_message(self, m: NewMessage) -> InsertedMessage:
        """Insert a new message, idempotent on (conversation_id, client_message_id)."""
        msg_id = uuid4()
        role_val = m.role.value if hasattr(m.role, 'value') else str(m.role)
        now_ts = self._clock.now()

        async with wrap_db_errors():
            row = await self.conn.fetchrow(
                INSERT_MESSAGE_SQL,
                msg_id,
                m.conversation_id,
                m.user_id,
                role_val,
                m.content,
                m.client_message_id,
                m.source_id,
                m.partial,
                m.trace,
                now_ts,
            )
            if row is not None and row.get('id') is not None:
                return InsertedMessage(id=row['id'], created=True)

            # Idempotency hit: find existing message
            existing = await self.conn.fetchrow(
                """\
select id from messages
where conversation_id = $1 and client_message_id = $2;""",
                m.conversation_id,
                m.client_message_id,
            )
            existing_id = existing['id'] if existing is not None else msg_id
            return InsertedMessage(id=existing_id, created=False)

    async def insert(self, m: NewMessage) -> InsertedMessage:
        """Convenience alias for insert_message."""
        return await self.insert_message(m)

    async def recent_turns(
        self,
        user_id: UserId,
        conversation_id: UUID,
        n: int,
    ) -> list[Message]:
        """Fetch recent messages for a conversation ordered by created_at desc (Q-12)."""
        async with wrap_db_errors():
            rows = await self.conn.fetch(
                Q12_RECENT_MESSAGES,
                user_id,
                conversation_id,
                n,
            )
        return [map_message(r) for r in rows]

    async def list_recent(
        self,
        user_id: UserId,
        conversation_id: UUID,
        n: int,
    ) -> list[Message]:
        """Convenience alias for recent_turns."""
        return await self.recent_turns(user_id, conversation_id, n)

    async def pending_user_messages(self, user_id: UserId, limit: int) -> list[Message]:
        """Fetch pending user messages with queued/running extract jobs via Q-13 / Q-17."""
        async with wrap_db_errors():
            rows = await self.conn.fetch(
                Q13_UNPROCESSED_USER_SOURCES,
                user_id,
                limit,
            )
        return [map_message(r) for r in rows]

    async def finalize_assistant(
        self,
        user_id: UserId,
        message_id: UUID,
        content: str,
        partial: bool,
        trace: dict[str, Any] | None,
    ) -> None:
        """Finalize streamed assistant message content and metadata."""
        async with wrap_db_errors():
            await self.conn.execute(
                """\
update messages
set content = $3, partial = $4, trace = $5
where user_id = $1 and id = $2;""",
                user_id,
                message_id,
                content,
                partial,
                trace,
            )

    async def set_job(
        self,
        user_id: UserId,
        message_id: UUID,
        job_id: UUID | str | None = None,
        *,
        source_id: SourceId | None = None,
    ) -> None:
        """Link an extraction job or source id to a message."""
        async with wrap_db_errors():
            if job_id is not None and source_id is not None:
                await self.conn.execute(
                    """\
update messages
set trace = coalesce(trace, '{}'::jsonb) || jsonb_build_object('job_id', $3::text),
    source_id = $4
where user_id = $1 and id = $2;""",
                    user_id,
                    message_id,
                    str(job_id),
                    source_id,
                )
            elif job_id is not None:
                await self.conn.execute(
                    """\
update messages
set trace = coalesce(trace, '{}'::jsonb) || jsonb_build_object('job_id', $3::text)
where user_id = $1 and id = $2;""",
                    user_id,
                    message_id,
                    str(job_id),
                )
            elif source_id is not None:
                await self.conn.execute(
                    """\
update messages
set source_id = $3
where user_id = $1 and id = $2;""",
                    user_id,
                    message_id,
                    source_id,
                )
