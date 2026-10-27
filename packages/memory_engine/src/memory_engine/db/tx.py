"""Database transaction utilities, advisory locks, and tenant context isolation."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from memory_engine.db.pool import wrap_db_errors
from memory_engine.domain.ids import UserId

SET_APP_USER_ID_SQL: str = "SELECT set_config('app.user_id', $1, true)"
LOCK_USER_ADVISORY_SQL: str = "SELECT pg_advisory_xact_lock(hashtext('sb_fact:' || $1::text))"
SET_LOCK_TIMEOUT_SQL: str = 'SET LOCAL lock_timeout = $1'


async def set_transaction_user(conn: Any, user_id: UserId) -> None:
    """Set transaction-local tenant context (app.user_id) for Row Level Security."""
    await conn.execute(SET_APP_USER_ID_SQL, str(user_id))


async def acquire_user_advisory_lock(
    conn: Any,
    user_id: UserId,
    lock_timeout_s: float = 5.0,
) -> None:
    """Acquire transaction-level advisory lock serializing mutations for this user.

    Uses lock_timeout (default 5s) to guard against deadlocks or lock contention.
    """
    timeout_str = f'{int(lock_timeout_s * 1000)}ms' if lock_timeout_s != 5.0 else '5s'
    await conn.execute(f"SET LOCAL lock_timeout = '{timeout_str}'")
    await conn.execute(LOCK_USER_ADVISORY_SQL, str(user_id))


@asynccontextmanager
async def transaction_scope(
    conn: Any,
    *,
    user_id: UserId | None = None,
    readonly: bool = False,
    isolation: str = 'read_committed',
    lock_timeout_s: float = 5.0,
) -> AsyncIterator[Any]:
    """Execute within a single database transaction with automatic error translation."""
    async with wrap_db_errors():
        tx = conn.transaction(readonly=readonly, isolation=isolation)
        await tx.start()
        try:
            if user_id is not None:
                await set_transaction_user(conn, user_id)
            yield conn
        except Exception:
            await tx.rollback()
            raise
        else:
            await tx.commit()
