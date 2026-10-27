"""Integration test fixtures and configuration for Postgres 16 + pgvector."""

import contextlib
import os
from collections.abc import AsyncIterator, Iterator
from typing import Any
from urllib.parse import urlparse, urlunparse
from uuid import UUID, uuid4

import asyncpg  # type: ignore[import-untyped]
import pytest
from memory_engine.db.migrate import apply_migrations, verify_startup_guard
from memory_engine.db.pool import create_api_pool, create_db_pool, create_worker_pool
from memory_engine.db.uow import PostgresUnitOfWorkFactory
from memory_engine.domain.ids import UserId

try:
    from testcontainers.community.postgres import PostgresContainer
except ImportError:
    from testcontainers.postgres import PostgresContainer


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Ensure all tests under tests/integration have the integration marker."""
    for item in items:
        if 'tests/integration' in str(item.fspath):
            item.add_marker(pytest.mark.integration)


@pytest.fixture(scope='session')
def pg_url() -> Iterator[str]:
    """Provide a PostgreSQL connection URL from SB_TEST_DATABASE_URL or testcontainers."""
    env_url = os.environ.get('SB_TEST_DATABASE_URL')
    if env_url:
        yield env_url
        return

    container: Any = None
    try:
        container = PostgresContainer('pgvector/pgvector:pg16')
        container.start()
        url = container.get_connection_url(driver=None)
    except Exception as exc:
        if container is not None:
            with contextlib.suppress(Exception):
                container.stop()
        pytest.skip(f'Docker or SB_TEST_DATABASE_URL not available: {exc}')
        return

    try:
        yield str(url)
    finally:
        if container is not None:
            with contextlib.suppress(Exception):
                container.stop()


@pytest.fixture(scope='session')
def migrated_db(pg_url: str) -> Iterator[str]:
    """Apply migrations 000-005, set role credentials, and verify startup guard."""
    import asyncio

    async def _setup() -> None:
        conn = await asyncpg.connect(pg_url)
        try:
            await apply_migrations(conn)
            await conn.execute("ALTER ROLE app_api WITH PASSWORD 'api_pass'")
            await conn.execute("ALTER ROLE app_worker WITH PASSWORD 'worker_pass'")
            await conn.execute('GRANT USAGE ON SCHEMA auth TO app_api, app_worker')
            await conn.execute('GRANT ALL ON TABLE auth.users TO app_api, app_worker')
            await verify_startup_guard(
                conn, expected_embedding_model_id='nomic-ai/nomic-embed-text-v1.5'
            )
        finally:
            await conn.close()

    asyncio.run(_setup())
    yield pg_url


def _build_role_url(base_url: str, username: str, password: str) -> str:
    parsed = urlparse(base_url)
    netloc = f'{username}:{password}@{parsed.hostname}:{parsed.port}'
    return urlunparse(parsed._replace(netloc=netloc))


@pytest.fixture
def api_db_url(migrated_db: str) -> str:
    """Connection URL for app_api role."""
    return _build_role_url(migrated_db, 'app_api', 'api_pass')


@pytest.fixture
def worker_db_url(migrated_db: str) -> str:
    """Connection URL for app_worker role."""
    return _build_role_url(migrated_db, 'app_worker', 'worker_pass')


@pytest.fixture
def admin_db_url(migrated_db: str) -> str:
    """Connection URL for superuser/admin role."""
    return migrated_db


@pytest.fixture
async def admin_pool(admin_db_url: str) -> AsyncIterator[asyncpg.Pool]:
    """Database connection pool running as superuser."""
    pool = await create_db_pool(
        admin_db_url,
        min_size=1,
        max_size=5,
        verify_schema=False,
    )
    try:
        yield pool
    finally:
        await pool.close()


@pytest.fixture
async def api_pool(api_db_url: str) -> AsyncIterator[asyncpg.Pool]:
    """Database connection pool running as app_api role."""
    pool = await create_api_pool(
        api_db_url,
        verify_schema=True,
        expected_embedding_model_id='nomic-ai/nomic-embed-text-v1.5',
    )
    try:
        yield pool
    finally:
        await pool.close()


@pytest.fixture
async def worker_pool(worker_db_url: str) -> AsyncIterator[asyncpg.Pool]:
    """Database connection pool running as app_worker role."""
    pool = await create_worker_pool(
        worker_db_url,
        verify_schema=True,
        expected_embedding_model_id='nomic-ai/nomic-embed-text-v1.5',
    )
    try:
        yield pool
    finally:
        await pool.close()


@pytest.fixture
async def uow_factory(api_pool: asyncpg.Pool) -> AsyncIterator[PostgresUnitOfWorkFactory]:
    """PostgresUnitOfWorkFactory configured for app_api role."""
    yield PostgresUnitOfWorkFactory(api_pool)


@pytest.fixture
async def worker_uow_factory(
    worker_pool: asyncpg.Pool,
) -> AsyncIterator[PostgresUnitOfWorkFactory]:
    """PostgresUnitOfWorkFactory configured for app_worker role."""
    yield PostgresUnitOfWorkFactory(worker_pool)


@pytest.fixture
def create_test_user(admin_pool: asyncpg.Pool) -> Any:
    """Helper to insert an auth.users record for foreign-key constraints."""

    async def _create(user_id: UserId | UUID | None = None) -> UserId:
        uid = UserId(user_id) if user_id is not None else UserId(uuid4())
        async with admin_pool.acquire() as conn:
            await conn.execute(
                'INSERT INTO auth.users (id) VALUES ($1) ON CONFLICT (id) DO NOTHING',
                uid,
            )
        return uid

    return _create
