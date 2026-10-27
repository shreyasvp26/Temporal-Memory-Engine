"""Static unit tests for SQL migrations, checksums, and startup integrity guard."""

from contextlib import AbstractAsyncContextManager
from types import TracebackType
from typing import Any

import pytest
from memory_engine.db.migrate import (
    EXPECTED_SCHEMA_VERSION,
    AsyncConnection,
    apply_migrations,
    compute_checksum,
    load_migration_files,
    verify_applied_checksums,
    verify_startup_guard,
)
from memory_engine.domain.errors import ConfigError

FROZEN_PREDICATES = {
    'lives_in',
    'works_at',
    'job_title',
    'studies_at',
    'has_name',
    'born_in',
    'birthday',
    'relationship_status',
    'owns',
    'likes',
    'dislikes',
    'prefers',
    'has_skill',
    'friend_of',
    'family_member',
    'plans_to',
    'is_reading',
    'is_working_on',
    'attended',
    'visited',
    'bought',
    'decided',
    'met',
    'started',
    'completed',
    'traveled_to',
    'said_assistant',
    'other',
}


class FakeTransaction(AbstractAsyncContextManager[None]):
    """Fake transaction async context manager."""

    async def __aenter__(self) -> None:
        return None

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool | None:
        return None


class FakeConnection:
    """In-memory connection implementing AsyncConnection for offline unit tests."""

    def __init__(
        self,
        schema_migrations: list[dict[str, Any]] | None = None,
        app_meta: dict[str, str] | None = None,
    ) -> None:
        self.schema_migrations: list[dict[str, Any]] = (
            [dict(r) for r in schema_migrations] if schema_migrations is not None else []
        )
        self.app_meta: dict[str, str] = dict(app_meta) if app_meta is not None else {}
        self.executed_queries: list[str] = []
        self.locked: bool = False
        self.transactions_count: int = 0
        self.raise_on_query: bool = False

    def transaction(self) -> AbstractAsyncContextManager[None]:
        self.transactions_count += 1
        return FakeTransaction()

    async def execute(self, query: str, *args: Any) -> str:
        if self.raise_on_query:
            raise RuntimeError('relation does not exist')
        self.executed_queries.append(query)
        if 'pg_advisory_lock' in query:
            self.locked = True
        elif 'pg_advisory_unlock' in query:
            self.locked = False
        elif 'INSERT INTO schema_migrations' in query:
            v = int(args[0])
            chk = str(args[1])
            self.schema_migrations.append({'version': v, 'checksum': chk})
        elif 'INSERT INTO app_meta' in query:
            val = str(args[0])
            if 'embedding_model_id' not in self.app_meta:
                self.app_meta['embedding_model_id'] = val
        return 'OK'

    async def fetch(self, query: str, *args: Any) -> list[Any]:
        if self.raise_on_query:
            raise RuntimeError('relation does not exist')
        self.executed_queries.append(query)
        if 'SELECT version, checksum FROM schema_migrations' in query:
            return sorted(self.schema_migrations, key=lambda r: int(r['version']))
        if 'SELECT version FROM schema_migrations' in query:
            return [{'version': r['version']} for r in self.schema_migrations]
        return []

    async def fetchrow(self, query: str, *args: Any) -> Any:
        rows = await self.fetch(query, *args)
        return rows[0] if rows else None

    async def fetchval(self, query: str, *args: Any) -> Any:
        if self.raise_on_query:
            raise RuntimeError('relation does not exist')
        self.executed_queries.append(query)
        if 'SELECT max(version) FROM schema_migrations' in query:
            if not self.schema_migrations:
                return None
            return max(int(r['version']) for r in self.schema_migrations)
        if "SELECT value FROM app_meta WHERE key = 'embedding_model_id'" in query:
            return self.app_meta.get('embedding_model_id')
        return None


def test_migration_files_exist_and_numbered() -> None:
    """Verify all 6 SQL files exist, are numbered 000..005, and have valid names."""
    migrations = load_migration_files()
    assert len(migrations) == 6
    assert EXPECTED_SCHEMA_VERSION == 5

    expected_files = [
        (0, '000_local_auth_stub.sql', 'local_auth_stub'),
        (1, '001_extensions_meta.sql', 'extensions_meta'),
        (2, '002_core_tables.sql', 'core_tables'),
        (3, '003_indexes.sql', 'indexes'),
        (4, '004_roles_rls.sql', 'roles_rls'),
        (5, '005_seed_predicates.sql', 'seed_predicates'),
    ]

    for m, (exp_v, exp_filename, exp_name) in zip(migrations, expected_files, strict=True):
        assert m.version == exp_v
        assert m.filename == exp_filename
        assert m.name == exp_name
        assert m.path.exists()
        assert len(m.content.strip()) > 0
        assert m.checksum == compute_checksum(m.content)


def test_seed_predicates_migration_contains_all_28() -> None:
    """Verify 005_seed_predicates.sql contains all 28 predicates and required synonyms."""
    migrations = load_migration_files()
    seed_migration = next(m for m in migrations if m.version == 5)
    content = seed_migration.content

    # Check all 28 frozen predicate names exist in the SQL content
    for name in FROZEN_PREDICATES:
        assert f"'{name}'" in content, f'Predicate {name} missing from 005_seed_predicates.sql'

    # Check key synonyms from Section 8.7
    assert "'moved_to'" in content and "'resides_in'" in content
    assert "'employed_at'" in content and "'works_for'" in content
    assert "'loves'" in content and "'enjoys'" in content
    assert "'job'" in content and "'role'" in content

    # Check on conflict do update clause is present
    assert 'on conflict (name) do update set' in content


def test_core_tables_and_indexes_ddl_sanity() -> None:
    """Verify core DDL tables and index definitions exist in 002 and 003."""
    migrations = load_migration_files()
    ddl = next(m for m in migrations if m.version == 2).content
    indexes = next(m for m in migrations if m.version == 3).content
    rls = next(m for m in migrations if m.version == 4).content

    for tbl in [
        'profiles',
        'predicates',
        'sources',
        'entities',
        'entity_aliases',
        'facts',
        'fact_sources',
        'fact_relations',
        'jobs',
        'conversations',
        'messages',
        'audit_log',
    ]:
        assert f'create table {tbl}' in ddl

    # Indexes
    assert 'create index facts_hnsw' in indexes
    assert 'create index sources_hnsw' in indexes
    assert 'create index entities_hnsw' in indexes
    assert 'create index facts_tsv' in indexes
    assert 'create unique index facts_single_open' in indexes

    # RLS & idempotent roles
    assert 'create role app_api' in rls
    assert 'create role app_worker' in rls
    assert 'enable row level security' in rls
    assert 'foreach t in array' in rls


def test_fake_connection_implements_protocol() -> None:
    """Verify FakeConnection satisfies AsyncConnection protocol."""
    conn = FakeConnection()
    assert isinstance(conn, AsyncConnection)


@pytest.mark.asyncio
async def test_apply_migrations_fresh_run() -> None:
    """Verify fresh run applies all 6 migrations under advisory lock."""
    conn = FakeConnection()
    applied = await apply_migrations(conn)

    assert applied == [0, 1, 2, 3, 4, 5]
    assert len(conn.schema_migrations) == 6
    assert conn.transactions_count == 6
    # Advisory lock was acquired and released
    assert any('pg_advisory_lock' in q for q in conn.executed_queries)
    assert not conn.locked


@pytest.mark.asyncio
async def test_apply_migrations_idempotent_when_already_applied() -> None:
    """Verify no migrations are applied when schema is already up to date."""
    migrations = load_migration_files()
    initial_migrations = [{'version': m.version, 'checksum': m.checksum} for m in migrations]
    conn = FakeConnection(schema_migrations=initial_migrations)

    applied = await apply_migrations(conn)
    assert applied == []
    assert conn.transactions_count == 0


@pytest.mark.asyncio
async def test_apply_migrations_partial_run() -> None:
    """Verify pending migrations after a partial previous run."""
    migrations = load_migration_files()
    initial_migrations = [{'version': m.version, 'checksum': m.checksum} for m in migrations[:3]]
    conn = FakeConnection(schema_migrations=initial_migrations)

    applied = await apply_migrations(conn)
    assert applied == [3, 4, 5]
    assert len(conn.schema_migrations) == 6


@pytest.mark.asyncio
async def test_verify_applied_checksums_detects_tampered_file() -> None:
    """Verify ConfigError E-CONFIG-01 is raised when checksum does not match."""
    migrations = load_migration_files()
    tampered_migrations = [
        {'version': m.version, 'checksum': 'tampered_hash' if m.version == 2 else m.checksum}
        for m in migrations
    ]
    conn = FakeConnection(schema_migrations=tampered_migrations)

    with pytest.raises(ConfigError) as exc_info:
        await verify_applied_checksums(conn, migrations)

    assert exc_info.value.code == 'E-CONFIG-01'
    assert 'checksum mismatch' in str(exc_info.value)


@pytest.mark.asyncio
async def test_verify_applied_checksums_detects_unknown_version() -> None:
    """Verify ConfigError E-CONFIG-01 when database contains unknown migration version."""
    migrations = load_migration_files()
    conn = FakeConnection(schema_migrations=[{'version': 99, 'checksum': 'abc'}])

    with pytest.raises(ConfigError) as exc_info:
        await verify_applied_checksums(conn, migrations)

    assert exc_info.value.code == 'E-CONFIG-01'
    assert 'version 99 not found' in str(exc_info.value)


@pytest.mark.asyncio
async def test_startup_guard_success() -> None:
    """Verify startup guard passes when schema version and embedding model match."""
    conn = FakeConnection(
        schema_migrations=[{'version': EXPECTED_SCHEMA_VERSION, 'checksum': 'any'}],
        app_meta={'embedding_model_id': 'nomic-ai/nomic-embed-text-v1.5'},
    )
    await verify_startup_guard(
        conn,
        expected_version=EXPECTED_SCHEMA_VERSION,
        expected_embedding_model_id='nomic-ai/nomic-embed-text-v1.5',
    )


@pytest.mark.asyncio
async def test_startup_guard_writes_embedding_model_on_first_start() -> None:
    """Verify startup guard writes embedding_model_id if absent."""
    conn = FakeConnection(
        schema_migrations=[{'version': EXPECTED_SCHEMA_VERSION, 'checksum': 'any'}],
        app_meta={},
    )
    await verify_startup_guard(
        conn,
        expected_version=EXPECTED_SCHEMA_VERSION,
        expected_embedding_model_id='nomic-ai/nomic-embed-text-v1.5',
    )
    assert conn.app_meta.get('embedding_model_id') == 'nomic-ai/nomic-embed-text-v1.5'


@pytest.mark.asyncio
async def test_startup_guard_fails_on_version_mismatch() -> None:
    """Verify startup guard raises ConfigError E-CONFIG-01 on schema version mismatch."""
    conn = FakeConnection(
        schema_migrations=[{'version': 3, 'checksum': 'any'}],
        app_meta={'embedding_model_id': 'nomic-ai/nomic-embed-text-v1.5'},
    )
    with pytest.raises(ConfigError) as exc_info:
        await verify_startup_guard(
            conn,
            expected_version=EXPECTED_SCHEMA_VERSION,
            expected_embedding_model_id='nomic-ai/nomic-embed-text-v1.5',
        )

    assert exc_info.value.code == 'E-CONFIG-01'
    assert 'does not match expected' in str(exc_info.value)


@pytest.mark.asyncio
async def test_startup_guard_fails_when_no_migrations() -> None:
    """Verify startup guard raises ConfigError E-CONFIG-01 when schema_migrations is empty."""
    conn = FakeConnection(
        schema_migrations=[],
        app_meta={'embedding_model_id': 'nomic-ai/nomic-embed-text-v1.5'},
    )
    with pytest.raises(ConfigError) as exc_info:
        await verify_startup_guard(
            conn,
            expected_version=EXPECTED_SCHEMA_VERSION,
            expected_embedding_model_id='nomic-ai/nomic-embed-text-v1.5',
        )

    assert exc_info.value.code == 'E-CONFIG-01'


@pytest.mark.asyncio
async def test_startup_guard_fails_on_embedding_model_mismatch() -> None:
    """Verify startup guard raises ConfigError E-CONFIG-01 on different embedding model."""
    conn = FakeConnection(
        schema_migrations=[{'version': EXPECTED_SCHEMA_VERSION, 'checksum': 'any'}],
        app_meta={'embedding_model_id': 'old-model-v1'},
    )
    with pytest.raises(ConfigError) as exc_info:
        await verify_startup_guard(
            conn,
            expected_version=EXPECTED_SCHEMA_VERSION,
            expected_embedding_model_id='nomic-ai/nomic-embed-text-v1.5',
        )

    assert exc_info.value.code == 'E-CONFIG-01'
    assert 'does not match configured' in str(exc_info.value)


@pytest.mark.asyncio
async def test_startup_guard_fails_on_db_query_error() -> None:
    """Verify startup guard raises ConfigError E-CONFIG-01 when query raises."""
    conn = FakeConnection()
    conn.raise_on_query = True

    with pytest.raises(ConfigError) as exc_info:
        await verify_startup_guard(
            conn,
            expected_version=EXPECTED_SCHEMA_VERSION,
            expected_embedding_model_id='nomic-ai/nomic-embed-text-v1.5',
        )

    assert exc_info.value.code == 'E-CONFIG-01'
