"""Database connection pooling, connection initialization, and error translation."""

import json
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Any

import asyncpg  # type: ignore[import-untyped]
from pgvector.asyncpg import register_vector

from memory_engine.db.migrate import EXPECTED_SCHEMA_VERSION, verify_startup_guard
from memory_engine.domain.errors import (
    ConstraintViolation,
    MemoryEngineError,
    StorageError,
    TransientStorageError,
)

logger = logging.getLogger(__name__)

TRANSIENT_PG_ERRORS = (
    asyncpg.SerializationError,
    asyncpg.DeadlockDetectedError,
    asyncpg.LockNotAvailableError,
    asyncpg.QueryCanceledError,
    asyncpg.CannotConnectNowError,
    asyncpg.AdminShutdownError,
    asyncpg.CrashShutdownError,
    asyncpg.PostgresConnectionError,
    asyncpg.ConnectionDoesNotExistError,
    asyncpg.ConnectionFailureError,
    asyncpg.ConnectionRejectionError,
    asyncpg.TooManyConnectionsError,
    asyncpg.InterfaceError,
    TimeoutError,
    ConnectionError,
)

CONSTRAINT_PG_ERRORS = (
    asyncpg.IntegrityConstraintViolationError,
    asyncpg.UniqueViolationError,
    asyncpg.ForeignKeyViolationError,
    asyncpg.CheckViolationError,
    asyncpg.NotNullViolationError,
    asyncpg.ExclusionViolationError,
)


def translate_pg_error(exc: BaseException) -> MemoryEngineError:
    """Translate asyncpg/Postgres exceptions to the domain StorageError hierarchy."""
    if isinstance(exc, MemoryEngineError):
        return exc

    details: dict[str, Any] = {}
    if hasattr(exc, 'sqlstate') and exc.sqlstate:
        details['sqlstate'] = str(exc.sqlstate)
    if hasattr(exc, 'detail') and exc.detail:
        details['detail'] = str(exc.detail)
    if hasattr(exc, 'schema_name') and exc.schema_name:
        details['schema'] = str(exc.schema_name)
    if hasattr(exc, 'table_name') and exc.table_name:
        details['table'] = str(exc.table_name)
    if hasattr(exc, 'constraint_name') and exc.constraint_name:
        details['constraint'] = str(exc.constraint_name)

    msg = str(exc)

    if isinstance(exc, TRANSIENT_PG_ERRORS):
        return TransientStorageError(msg, code='E-DB-01', details=details)

    if isinstance(exc, CONSTRAINT_PG_ERRORS):
        return ConstraintViolation(msg, code='E-DB-02', details=details)

    if isinstance(exc, asyncpg.PostgresError):
        return StorageError(msg, code='E-DB-01', details=details)

    return StorageError(f'Unexpected database error: {msg}', code='E-DB-01', details=details)


@asynccontextmanager
async def wrap_db_errors() -> AsyncIterator[None]:
    """Async context manager translating database exceptions to domain StorageErrors."""
    try:
        yield
    except Exception as exc:
        translated = translate_pg_error(exc)
        if translated is not exc:
            raise translated from exc
        raise


async def init_connection(conn: Any) -> None:
    """Initialize connection with pgvector codec, json/jsonb codecs, and session defaults."""
    # Register vector codec (from pgvector.asyncpg)
    try:
        await register_vector(conn)
    except Exception as err:
        logger.debug('Skipping or failed pgvector registration on connection: %s', err)

    # Register JSON and JSONB codecs
    if hasattr(conn, 'set_type_codec'):
        try:
            await conn.set_type_codec(
                'jsonb',
                encoder=json.dumps,
                decoder=json.loads,
                schema='pg_catalog',
            )
            await conn.set_type_codec(
                'json',
                encoder=json.dumps,
                decoder=json.loads,
                schema='pg_catalog',
            )
        except Exception as err:
            logger.debug('Skipping or failed JSON codec setup on connection: %s', err)


async def create_db_pool(
    dsn: str,
    *,
    min_size: int = 2,
    max_size: int = 10,
    command_timeout: float = 5.0,
    statement_timeout_s: float = 4.0,
    application_name: str = 'second_brain',
    server_settings: dict[str, str] | None = None,
    verify_schema: bool = True,
    expected_schema_version: int = EXPECTED_SCHEMA_VERSION,
    expected_embedding_model_id: str | None = None,
    init_hook: Callable[[Any], Awaitable[None]] | None = None,
    **kwargs: Any,
) -> asyncpg.Pool:
    """Create asyncpg connection pool configured per LLD Section 6.9.

    Sets jit=off, application_name, statement_timeout, registers vector & json codecs,
    and runs startup schema integrity checks via verify_startup_guard.
    """
    settings: dict[str, str] = {
        'jit': 'off',
        'application_name': application_name,
        'statement_timeout': f'{int(statement_timeout_s * 1000)}ms',
    }
    if server_settings:
        settings.update(server_settings)

    async def _init(conn: Any) -> None:
        await init_connection(conn)
        if init_hook is not None:
            await init_hook(conn)

    try:
        pool: asyncpg.Pool = await asyncpg.create_pool(
            dsn,
            min_size=min_size,
            max_size=max_size,
            command_timeout=command_timeout,
            server_settings=settings,
            init=_init,
            **kwargs,
        )
    except Exception as exc:
        raise translate_pg_error(exc) from exc

    if verify_schema:
        conn = await pool.acquire()
        try:
            await verify_startup_guard(
                conn,
                expected_version=expected_schema_version,
                expected_embedding_model_id=expected_embedding_model_id,
            )
        except Exception:
            await pool.release(conn)
            await pool.close()
            raise
        else:
            await pool.release(conn)

    return pool


create_pool = create_db_pool


async def create_api_pool(
    dsn: str,
    *,
    verify_schema: bool = True,
    expected_schema_version: int = EXPECTED_SCHEMA_VERSION,
    expected_embedding_model_id: str | None = None,
    **kwargs: Any,
) -> asyncpg.Pool:
    """Create API-role connection pool (2..10 conns, 5s cmd timeout, 4s stmt timeout)."""
    return await create_db_pool(
        dsn,
        min_size=2,
        max_size=10,
        command_timeout=5.0,
        statement_timeout_s=4.0,
        application_name='second_brain_api',
        verify_schema=verify_schema,
        expected_schema_version=expected_schema_version,
        expected_embedding_model_id=expected_embedding_model_id,
        **kwargs,
    )


async def create_worker_pool(
    dsn: str,
    *,
    verify_schema: bool = True,
    expected_schema_version: int = EXPECTED_SCHEMA_VERSION,
    expected_embedding_model_id: str | None = None,
    **kwargs: Any,
) -> asyncpg.Pool:
    """Create Worker-role connection pool (2..6 conns, 30s cmd timeout, 25s stmt timeout)."""
    return await create_db_pool(
        dsn,
        min_size=2,
        max_size=6,
        command_timeout=30.0,
        statement_timeout_s=25.0,
        application_name='second_brain_worker',
        verify_schema=verify_schema,
        expected_schema_version=expected_schema_version,
        expected_embedding_model_id=expected_embedding_model_id,
        **kwargs,
    )


async def create_eval_pool(
    dsn: str,
    *,
    verify_schema: bool = True,
    expected_schema_version: int = EXPECTED_SCHEMA_VERSION,
    expected_embedding_model_id: str | None = None,
    **kwargs: Any,
) -> asyncpg.Pool:
    """Create Eval-role connection pool (2..16 conns, 60s cmd timeout, 55s stmt timeout)."""
    return await create_db_pool(
        dsn,
        min_size=2,
        max_size=16,
        command_timeout=60.0,
        statement_timeout_s=55.0,
        application_name='second_brain_eval',
        verify_schema=verify_schema,
        expected_schema_version=expected_schema_version,
        expected_embedding_model_id=expected_embedding_model_id,
        **kwargs,
    )
