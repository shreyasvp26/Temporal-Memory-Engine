"""Database migration runner and startup integrity guards."""

import argparse
import asyncio
import hashlib
import os
import re
import sys
from collections.abc import Sequence
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from memory_engine.domain.errors import ConfigError

EXPECTED_SCHEMA_VERSION: int = 5
ADVISORY_LOCK_ID: int = 727001


@dataclass(frozen=True, slots=True)
class MigrationFile:
    """Representation of a numbered SQL migration file."""

    version: int
    name: str
    filename: str
    path: Path
    checksum: str
    content: str


@runtime_checkable
class AsyncConnection(Protocol):
    """Protocol for async connection compatible with asyncpg Connection."""

    async def execute(self, query: str, *args: Any) -> str: ...

    async def fetch(self, query: str, *args: Any) -> list[Any]: ...

    async def fetchrow(self, query: str, *args: Any) -> Any: ...

    async def fetchval(self, query: str, *args: Any) -> Any: ...

    def transaction(self) -> AbstractAsyncContextManager[None]: ...


def compute_checksum(content: str | bytes) -> str:
    """Compute hex SHA-256 digest of migration file content."""
    data = content.encode('utf-8') if isinstance(content, str) else content
    return hashlib.sha256(data).hexdigest()


def load_migration_files(migrations_dir: Path | None = None) -> list[MigrationFile]:
    """Load and parse numbered SQL migration files in ascending version order."""
    if migrations_dir is None:
        migrations_dir = Path(__file__).parent / 'migrations'

    if not migrations_dir.is_dir():
        raise ConfigError(
            f'Migrations directory not found: {migrations_dir}',
            code='E-CONFIG-01',
        )

    pattern = re.compile(r'^(\d{3})_(.+)\.sql$')
    migrations: list[MigrationFile] = []

    for entry in sorted(migrations_dir.iterdir(), key=lambda p: p.name):
        match = pattern.match(entry.name)
        if match:
            version = int(match.group(1))
            name = match.group(2)
            content = entry.read_text(encoding='utf-8')
            checksum = compute_checksum(content)
            migrations.append(
                MigrationFile(
                    version=version,
                    name=name,
                    filename=entry.name,
                    path=entry,
                    checksum=checksum,
                    content=content,
                )
            )

    versions = [m.version for m in migrations]
    if len(versions) != len(set(versions)):
        raise ConfigError('Duplicate migration versions detected', code='E-CONFIG-01')

    return sorted(migrations, key=lambda m: m.version)


async def verify_applied_checksums(
    conn: AsyncConnection,
    migrations: Sequence[MigrationFile],
) -> None:
    """Verify that applied migrations in the database match loaded migration checksums."""
    rows = await conn.fetch('SELECT version, checksum FROM schema_migrations ORDER BY version ASC')
    migration_by_version = {m.version: m for m in migrations}

    for row in rows:
        if isinstance(row, dict):
            v = int(row['version'])
            db_chk = str(row['checksum'])
        elif hasattr(row, 'get'):
            v = int(row.get('version'))
            db_chk = str(row.get('checksum'))
        else:
            v = int(row[0])
            db_chk = str(row[1])

        migration = migration_by_version.get(v)
        if migration is None:
            raise ConfigError(
                f'Database contains migration version {v} not found in migration files',
                code='E-CONFIG-01',
            )
        if migration.checksum != db_chk:
            raise ConfigError(
                f'Migration {v} checksum mismatch: database has {db_chk}, '
                f'file {migration.filename} has {migration.checksum}',
                code='E-CONFIG-01',
            )


async def apply_migrations(
    conn: AsyncConnection,
    migrations_dir: Path | None = None,
) -> list[int]:
    """Apply pending SQL migrations sequentially under advisory lock 727001."""
    migrations = load_migration_files(migrations_dir)
    await conn.execute('SELECT pg_advisory_lock($1)', ADVISORY_LOCK_ID)

    try:
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_migrations (
                version INT PRIMARY KEY,
                checksum TEXT NOT NULL,
                applied_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )
            """
        )

        await verify_applied_checksums(conn, migrations)

        applied_rows = await conn.fetch('SELECT version FROM schema_migrations')
        applied_versions: set[int] = set()
        for row in applied_rows:
            if isinstance(row, dict):
                applied_versions.add(int(row['version']))
            elif hasattr(row, 'get'):
                applied_versions.add(int(row.get('version')))
            else:
                applied_versions.add(int(row[0]))

        applied_this_run: list[int] = []
        for migration in migrations:
            if migration.version not in applied_versions:
                async with conn.transaction():
                    await conn.execute(migration.content)
                    await conn.execute(
                        'INSERT INTO schema_migrations (version, checksum) VALUES ($1, $2)',
                        migration.version,
                        migration.checksum,
                    )
                applied_this_run.append(migration.version)

        return applied_this_run
    finally:
        await conn.execute('SELECT pg_advisory_unlock($1)', ADVISORY_LOCK_ID)


async def verify_startup_guard(
    conn: AsyncConnection,
    expected_version: int = EXPECTED_SCHEMA_VERSION,
    expected_embedding_model_id: str | None = None,
) -> None:
    """Verify database schema version and embedding model match configuration.

    Refuses to start if schema_migrations max version != expected_version
    or app_meta.embedding_model_id != expected_embedding_model_id.
    Writes embedding_model_id on first start if absent.
    """
    try:
        max_version = await conn.fetchval('SELECT max(version) FROM schema_migrations')
    except Exception as err:
        raise ConfigError(
            f'Failed to check schema_migrations: {err}',
            code='E-CONFIG-01',
        ) from err

    if max_version is None or int(max_version) != expected_version:
        raise ConfigError(
            f'Database schema version {max_version} does not match expected {expected_version}',
            code='E-CONFIG-01',
        )

    if expected_embedding_model_id is not None:
        try:
            current_model = await conn.fetchval(
                "SELECT value FROM app_meta WHERE key = 'embedding_model_id'"
            )
        except Exception as err:
            raise ConfigError(
                f'Failed to check app_meta: {err}',
                code='E-CONFIG-01',
            ) from err

        if current_model is None:
            await conn.execute(
                "INSERT INTO app_meta (key, value) VALUES ('embedding_model_id', $1) "
                'ON CONFLICT (key) DO NOTHING',
                expected_embedding_model_id,
            )
            current_model = await conn.fetchval(
                "SELECT value FROM app_meta WHERE key = 'embedding_model_id'"
            )

        if current_model != expected_embedding_model_id:
            raise ConfigError(
                f"Database embedding model '{current_model}' does not match "
                f"configured '{expected_embedding_model_id}'",
                code='E-CONFIG-01',
            )


async def _async_main(db_url: str | None = None) -> None:
    import asyncpg  # type: ignore[import-untyped]

    if db_url is None:
        db_url = (
            os.environ.get('SB_DATABASE_URL_API')
            or os.environ.get('SB_DATABASE_URL')
            or os.environ.get('DATABASE_URL')
        )
    if not db_url:
        print(
            'Error: Database URL not provided. Set SB_DATABASE_URL_API or pass --db-url.',
            file=sys.stderr,
        )
        sys.exit(1)

    conn = await asyncpg.connect(db_url)
    try:
        applied = await apply_migrations(conn)
        if applied:
            print(f'Applied migrations: {applied}')
        else:
            print('Schema is up to date. No pending migrations.')
    finally:
        await conn.close()


def main() -> None:
    """CLI entrypoint for running migrations."""
    parser = argparse.ArgumentParser(description='Apply database migrations')
    parser.add_argument('--db-url', dest='db_url', default=None, help='Database connection URL')
    args = parser.parse_args()
    asyncio.run(_async_main(args.db_url))


if __name__ == '__main__':
    main()

__all__ = [
    'ADVISORY_LOCK_ID',
    'EXPECTED_SCHEMA_VERSION',
    'AsyncConnection',
    'MigrationFile',
    'apply_migrations',
    'compute_checksum',
    'load_migration_files',
    'main',
    'verify_applied_checksums',
    'verify_startup_guard',
]
