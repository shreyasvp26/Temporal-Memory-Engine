"""Unit tests for database connection pooling, Unit of Work, and error translation."""

from typing import Any
from unittest.mock import MagicMock, patch
from uuid import uuid4

import asyncpg  # type: ignore[import-untyped]
import pytest
from memory_engine.db.migrate import EXPECTED_SCHEMA_VERSION
from memory_engine.db.pool import (
    create_api_pool,
    create_db_pool,
    create_eval_pool,
    create_worker_pool,
    init_connection,
    translate_pg_error,
    wrap_db_errors,
)
from memory_engine.db.tx import (
    LOCK_USER_ADVISORY_SQL,
    SET_APP_USER_ID_SQL,
    transaction_scope,
)
from memory_engine.db.uow import (
    PostgresUnitOfWork,
    PostgresUnitOfWorkFactory,
    clear_repo_factories,
    get_active_uow,
    register_repo_factory,
)
from memory_engine.domain.errors import (
    ConfigError,
    ConstraintViolation,
    InvariantViolation,
    StorageError,
    TransientStorageError,
)
from memory_engine.domain.ids import UserId


class FakeTransaction:
    """Mock asyncpg transaction recording start, commit, and rollback calls."""

    def __init__(self, readonly: bool = False, isolation: str = 'read_committed') -> None:
        self.readonly = readonly
        self.isolation = isolation
        self.started = False
        self.committed = False
        self.rolled_back = False
        self.fail_on_commit: BaseException | None = None

    async def start(self) -> None:
        self.started = True

    async def commit(self) -> None:
        if self.fail_on_commit is not None:
            raise self.fail_on_commit
        self.committed = True

    async def rollback(self) -> None:
        self.rolled_back = True


class FakeConnection:
    """Mock asyncpg connection recording executed queries and managing transactions."""

    def __init__(self) -> None:
        self.executed: list[tuple[str, tuple[Any, ...]]] = []
        self.type_codecs: list[tuple[str, dict[str, Any]]] = []
        self.tx = FakeTransaction()
        self.fail_on_execute: BaseException | None = None
        self.fetchval_results: dict[str, Any] = {
            'SELECT max(version) FROM schema_migrations': EXPECTED_SCHEMA_VERSION,
            "SELECT value FROM app_meta WHERE key = 'embedding_model_id'": (
                'nomic-ai/nomic-embed-text-v1.5'
            ),
        }

    async def execute(self, query: str, *args: Any) -> str:
        self.executed.append((query, args))
        if self.fail_on_execute is not None:
            raise self.fail_on_execute
        return 'OK'

    async def fetchval(self, query: str, *args: Any) -> Any:
        self.executed.append((query, args))
        return self.fetchval_results.get(query)

    async def fetch(self, query: str, *args: Any) -> list[Any]:
        self.executed.append((query, args))
        return []

    def transaction(
        self, readonly: bool = False, isolation: str = 'read_committed'
    ) -> FakeTransaction:
        self.tx.readonly = readonly
        self.tx.isolation = isolation
        return self.tx

    async def set_type_codec(self, typename: str, **kwargs: Any) -> None:
        self.type_codecs.append((typename, kwargs))


class FakeAcquireContext:
    """Async context manager and awaitable for FakePool.acquire()."""

    def __init__(self, pool: 'FakePool') -> None:
        self.pool = pool

    def __await__(self) -> Any:
        return self._acquire().__await__()

    async def _acquire(self) -> FakeConnection:
        if self.pool.fail_on_acquire is not None:
            raise self.pool.fail_on_acquire
        self.pool.acquired_count += 1
        return self.pool.conn

    async def __aenter__(self) -> FakeConnection:
        return await self._acquire()

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.pool.release(self.pool.conn)


class FakePool:
    """Mock asyncpg pool tracking connection acquisition and release."""

    def __init__(self, conn: FakeConnection | None = None) -> None:
        self.conn = conn or FakeConnection()
        self.acquired_count = 0
        self.released_count = 0
        self.closed = False
        self.fail_on_acquire: BaseException | None = None

    def acquire(self) -> FakeAcquireContext:
        return FakeAcquireContext(self)

    async def release(self, conn: Any) -> None:
        self.released_count += 1

    async def close(self) -> None:
        self.closed = True


@pytest.fixture(autouse=True)
def _reset_global_factories() -> None:
    clear_repo_factories()


@pytest.mark.asyncio
async def test_uow_for_user_executes_set_config() -> None:
    user_id = UserId(uuid4())
    pool = FakePool()
    factory = PostgresUnitOfWorkFactory(pool)

    uow = factory.for_user(user_id)
    async with uow:
        assert uow.conn is pool.conn
        assert uow.user_id == user_id
        assert not uow.readonly
        assert pool.acquired_count == 1
        assert pool.conn.tx.started

    # Check set_config query execution
    assert len(pool.conn.executed) >= 1
    query, args = pool.conn.executed[0]
    assert query == SET_APP_USER_ID_SQL
    assert args == (str(user_id),)

    # Check commit and connection release
    assert pool.conn.tx.committed
    assert not pool.conn.tx.rolled_back
    assert pool.released_count == 1
    assert get_active_uow() is None


@pytest.mark.asyncio
async def test_uow_system_no_user_config() -> None:
    pool = FakePool()
    factory = PostgresUnitOfWorkFactory(pool)

    uow = factory.system()
    async with uow:
        assert uow.user_id is None
        assert pool.acquired_count == 1
        assert pool.conn.tx.started

    # No set_config executed for system UoW
    executed_queries = [q for q, _ in pool.conn.executed]
    assert SET_APP_USER_ID_SQL not in executed_queries
    assert pool.conn.tx.committed
    assert pool.released_count == 1


@pytest.mark.asyncio
async def test_uow_rolls_back_on_exception() -> None:
    user_id = UserId(uuid4())
    pool = FakePool()
    factory = PostgresUnitOfWorkFactory(pool)

    with pytest.raises(ValueError, match='something failed'):
        async with factory.for_user(user_id):
            raise ValueError('something failed')

    assert pool.conn.tx.rolled_back
    assert not pool.conn.tx.committed
    assert pool.released_count == 1
    assert get_active_uow() is None


@pytest.mark.asyncio
async def test_uow_lock_user_on_bound_user() -> None:
    user_id = UserId(uuid4())
    pool = FakePool()
    factory = PostgresUnitOfWorkFactory(pool)

    async with factory.for_user(user_id) as uow:
        await uow.lock_user()
        # Second call is idempotent
        await uow.lock_user()

    # Find lock query
    lock_queries = [(q, args) for q, args in pool.conn.executed if q == LOCK_USER_ADVISORY_SQL]
    assert len(lock_queries) == 1
    assert lock_queries[0][1] == (str(user_id),)

    # Ensure lock timeout was set before advisory lock
    timeout_queries = [q for q, _ in pool.conn.executed if 'SET LOCAL lock_timeout' in q]
    assert len(timeout_queries) == 1


@pytest.mark.asyncio
async def test_uow_lock_user_with_explicit_id() -> None:
    user_id = UserId(uuid4())
    pool = FakePool()
    factory = PostgresUnitOfWorkFactory(pool)

    async with factory.for_user(user_id) as uow:
        await uow.lock_user(user_id)

    lock_queries = [(q, args) for q, args in pool.conn.executed if q == LOCK_USER_ADVISORY_SQL]
    assert len(lock_queries) == 1
    assert lock_queries[0][1] == (str(user_id),)


@pytest.mark.asyncio
async def test_uow_lock_user_mismatched_user_raises() -> None:
    user1 = UserId(uuid4())
    user2 = UserId(uuid4())
    pool = FakePool()
    factory = PostgresUnitOfWorkFactory(pool)

    async with factory.for_user(user1) as uow:
        with pytest.raises(InvariantViolation, match='cannot lock different user'):
            await uow.lock_user(user2)


@pytest.mark.asyncio
async def test_uow_lock_user_system_without_id_raises() -> None:
    pool = FakePool()
    factory = PostgresUnitOfWorkFactory(pool)

    async with factory.system() as uow:
        with pytest.raises(InvariantViolation, match='without specifying user_id'):
            await uow.lock_user()


@pytest.mark.asyncio
async def test_uow_lock_user_system_with_explicit_id() -> None:
    user1 = UserId(uuid4())
    pool = FakePool()
    factory = PostgresUnitOfWorkFactory(pool)

    async with factory.system() as uow:
        await uow.lock_user(user1)

    lock_queries = [(q, args) for q, args in pool.conn.executed if q == LOCK_USER_ADVISORY_SQL]
    assert len(lock_queries) == 1
    assert lock_queries[0][1] == (str(user1),)


@pytest.mark.asyncio
async def test_uow_lock_user_not_open_raises() -> None:
    user_id = UserId(uuid4())
    pool = FakePool()
    uow = PostgresUnitOfWork(pool, user_id=user_id)

    with pytest.raises(InvariantViolation, match='UnitOfWork is not open'):
        await uow.lock_user()


@pytest.mark.asyncio
async def test_uow_nested_uow_forbidden() -> None:
    user1 = UserId(uuid4())
    user2 = UserId(uuid4())
    pool = FakePool()
    factory = PostgresUnitOfWorkFactory(pool)

    async with factory.for_user(user1):
        # Trying to begin or enter another UoW in the same context fails
        with pytest.raises(InvariantViolation, match='Nested UnitOfWork is forbidden'):
            factory.begin(user2)

        uow2 = PostgresUnitOfWork(pool, user_id=user2)
        with pytest.raises(InvariantViolation, match='Nested UnitOfWork is forbidden'):
            async with uow2:
                pass

    # Once the outer UoW exits, creating a new one works cleanly
    async with factory.for_user(user2):
        assert pool.acquired_count == 2


@pytest.mark.asyncio
async def test_uow_reenter_already_open_raises() -> None:
    user_id = UserId(uuid4())
    pool = FakePool()
    uow = PostgresUnitOfWork(pool, user_id=user_id)

    async with uow:
        with pytest.raises(InvariantViolation, match='already open'):
            await uow.__aenter__()


@pytest.mark.asyncio
async def test_uow_commit_failure_translates_error() -> None:
    user_id = UserId(uuid4())
    pool = FakePool()
    pool.conn.tx.fail_on_commit = asyncpg.SerializationError('could not serialize access')

    factory = PostgresUnitOfWorkFactory(pool)
    with pytest.raises(TransientStorageError) as exc_info:
        async with factory.for_user(user_id):
            pass

    assert exc_info.value.code == 'E-DB-01'
    assert 'could not serialize' in exc_info.value.message
    assert pool.released_count == 1
    assert get_active_uow() is None


@pytest.mark.asyncio
async def test_uow_enter_failure_releases_connection() -> None:
    user_id = UserId(uuid4())
    pool = FakePool()
    pool.conn.fail_on_execute = asyncpg.PostgresConnectionError('connection dropped')

    factory = PostgresUnitOfWorkFactory(pool)
    with pytest.raises(TransientStorageError):
        async with factory.for_user(user_id):
            pass

    assert pool.released_count == 1
    assert get_active_uow() is None


@pytest.mark.asyncio
async def test_uow_repository_stubs_and_wiring() -> None:
    user_id = UserId(uuid4())
    pool = FakePool()
    uow = PostgresUnitOfWork(pool, user_id=user_id)

    # By default, accessing unwired repos raises NotImplementedError
    with pytest.raises(NotImplementedError, match='Sources repository'):
        _ = uow.sources
    with pytest.raises(NotImplementedError, match='Facts repository'):
        _ = uow.facts

    # Passing mock repository instances in constructor
    mock_sources = MagicMock()
    mock_facts = MagicMock()
    wired_uow = PostgresUnitOfWork(
        pool,
        user_id=user_id,
        sources=mock_sources,
        facts=mock_facts,
    )
    assert wired_uow.sources is mock_sources
    assert wired_uow.facts is mock_facts

    # Using registered repo factories
    mock_entities = MagicMock()
    register_repo_factory('entities', lambda u: mock_entities)
    dynamic_uow = PostgresUnitOfWork(pool, user_id=user_id)
    assert dynamic_uow.entities is mock_entities


def test_postgres_error_translation_serialization_and_locks() -> None:
    err1 = asyncpg.SerializationError('serialization failure')
    translated1 = translate_pg_error(err1)
    assert isinstance(translated1, TransientStorageError)
    assert translated1.code == 'E-DB-01'
    assert translated1.retryable is True

    err2 = asyncpg.DeadlockDetectedError('deadlock detected')
    translated2 = translate_pg_error(err2)
    assert isinstance(translated2, TransientStorageError)

    err3 = asyncpg.LockNotAvailableError('lock not available')
    translated3 = translate_pg_error(err3)
    assert isinstance(translated3, TransientStorageError)

    err4 = asyncpg.QueryCanceledError('canceling statement due to statement timeout')
    translated4 = translate_pg_error(err4)
    assert isinstance(translated4, TransientStorageError)


def test_postgres_error_translation_connection_and_timeouts() -> None:
    err1 = asyncpg.PostgresConnectionError('connection lost')
    assert isinstance(translate_pg_error(err1), TransientStorageError)

    err2 = asyncpg.InterfaceError('connection was closed')
    assert isinstance(translate_pg_error(err2), TransientStorageError)

    err3 = TimeoutError('pool timeout')
    assert isinstance(translate_pg_error(err3), TransientStorageError)

    err4 = ConnectionResetError('socket reset')
    assert isinstance(translate_pg_error(err4), TransientStorageError)


def test_postgres_error_translation_constraints() -> None:
    err1 = asyncpg.UniqueViolationError('duplicate key value violates unique constraint')
    translated1 = translate_pg_error(err1)
    assert isinstance(translated1, ConstraintViolation)
    assert translated1.code == 'E-DB-02'
    assert translated1.retryable is False

    err2 = asyncpg.ForeignKeyViolationError('insert violates foreign key constraint')
    assert isinstance(translate_pg_error(err2), ConstraintViolation)

    err3 = asyncpg.CheckViolationError('check constraint violation')
    assert isinstance(translate_pg_error(err3), ConstraintViolation)

    err4 = asyncpg.NotNullViolationError('null value in column violates not-null constraint')
    assert isinstance(translate_pg_error(err4), ConstraintViolation)


def test_postgres_error_translation_generic() -> None:
    err = asyncpg.PostgresError('syntax error at or near SELECT')
    translated = translate_pg_error(err)
    assert isinstance(translated, StorageError)
    assert not isinstance(translated, (TransientStorageError, ConstraintViolation))


@pytest.mark.asyncio
async def test_wrap_db_errors_context_manager() -> None:
    with pytest.raises(TransientStorageError):
        async with wrap_db_errors():
            raise asyncpg.DeadlockDetectedError('deadlock detected')


@pytest.mark.asyncio
async def test_init_connection_registers_codecs() -> None:
    conn = FakeConnection()
    await init_connection(conn)

    codec_names = [name for name, _ in conn.type_codecs]
    assert 'vector' in codec_names
    assert 'jsonb' in codec_names
    assert 'json' in codec_names


@pytest.mark.asyncio
async def test_create_db_pool_with_schema_verification() -> None:
    fake_pool = FakePool()

    async def mock_create_pool(*args: Any, **kwargs: Any) -> FakePool:
        return fake_pool

    with patch('asyncpg.create_pool', new=mock_create_pool):
        pool = await create_db_pool('postgresql://localhost/test', verify_schema=True)
        assert pool is fake_pool
        assert fake_pool.acquired_count == 1
        assert fake_pool.released_count == 1
        assert not fake_pool.closed


@pytest.mark.asyncio
async def test_create_db_pool_schema_mismatch_closes_pool() -> None:
    fake_conn = FakeConnection()
    fake_conn.fetchval_results['SELECT max(version) FROM schema_migrations'] = 2  # old version
    fake_pool = FakePool(conn=fake_conn)

    async def mock_create_pool(*args: Any, **kwargs: Any) -> FakePool:
        return fake_pool

    with patch('asyncpg.create_pool', new=mock_create_pool):
        with pytest.raises(ConfigError, match='Database schema version 2 does not match expected'):
            await create_db_pool('postgresql://localhost/test', verify_schema=True)

        assert fake_pool.closed


@pytest.mark.asyncio
async def test_create_role_specific_pools() -> None:
    captured_kwargs: list[dict[str, Any]] = []

    async def mock_create_pool(dsn: str, **kwargs: Any) -> FakePool:
        captured_kwargs.append(kwargs)
        return FakePool()

    with patch('asyncpg.create_pool', new=mock_create_pool):
        _ = await create_api_pool('postgresql://localhost/test', verify_schema=False)
        assert captured_kwargs[-1]['min_size'] == 2
        assert captured_kwargs[-1]['max_size'] == 10
        assert captured_kwargs[-1]['command_timeout'] == 5.0
        assert captured_kwargs[-1]['server_settings']['application_name'] == 'second_brain_api'

        _ = await create_worker_pool('postgresql://localhost/test', verify_schema=False)
        assert captured_kwargs[-1]['min_size'] == 2
        assert captured_kwargs[-1]['max_size'] == 6
        assert captured_kwargs[-1]['command_timeout'] == 30.0
        assert captured_kwargs[-1]['server_settings']['application_name'] == 'second_brain_worker'

        _ = await create_eval_pool('postgresql://localhost/test', verify_schema=False)
        assert captured_kwargs[-1]['min_size'] == 2
        assert captured_kwargs[-1]['max_size'] == 16
        assert captured_kwargs[-1]['command_timeout'] == 60.0
        assert captured_kwargs[-1]['server_settings']['application_name'] == 'second_brain_eval'


@pytest.mark.asyncio
async def test_transaction_scope_helper() -> None:
    user_id = UserId(uuid4())
    conn = FakeConnection()

    async with transaction_scope(conn, user_id=user_id) as c:
        assert c is conn
        assert conn.tx.started

    assert conn.tx.committed
    assert len(conn.executed) >= 1
    assert conn.executed[0][0] == SET_APP_USER_ID_SQL
