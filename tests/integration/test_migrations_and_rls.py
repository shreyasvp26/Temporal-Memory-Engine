"""Integration tests for migrations 000-005, startup guard, seed predicates, and RLS (T-SEC-01)."""

from collections.abc import Callable, Coroutine
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

import asyncpg  # type: ignore[import-untyped]
import pytest
from memory_engine.db.migrate import (
    apply_migrations,
    load_migration_files,
    verify_applied_checksums,
    verify_startup_guard,
)
from memory_engine.db.repos.predicates import SEED_PREDICATES, PostgresPredicateRegistry
from memory_engine.db.uow import PostgresUnitOfWorkFactory
from memory_engine.domain.enums import (
    Cardinality,
    EntityType,
    FactOrigin,
    FactStatus,
    JobType,
    Role,
    SourceKind,
    TimePrecision,
)
from memory_engine.domain.errors import ConfigError, StorageError
from memory_engine.domain.ids import EntityId, FactId, JobId, SourceId, UserId
from memory_engine.domain.models import Candidate
from memory_engine.ports.repositories import NewJob, NewSource

pytestmark = pytest.mark.integration


async def test_migrations_applied_and_idempotent(admin_pool: asyncpg.Pool) -> None:
    """Verify migrations 000-005 applied, checksums match, and re-run is idempotent."""
    async with admin_pool.acquire() as conn:
        # Check max version in schema_migrations
        max_v = await conn.fetchval('SELECT max(version) FROM schema_migrations')
        assert max_v == 5

        # Check all versions 0..5 exist
        rows = await conn.fetch('SELECT version FROM schema_migrations ORDER BY version')
        versions = [r['version'] for r in rows]
        assert versions == [0, 1, 2, 3, 4, 5]

        # Verify checksums match migration files on disk
        files = load_migration_files()
        await verify_applied_checksums(conn, files)

        # Applying migrations again is a no-op
        applied = await apply_migrations(conn)
        assert applied == []


async def test_startup_guard_verification(admin_pool: asyncpg.Pool) -> None:
    """Verify startup guard checks schema version and embedding model id."""
    async with admin_pool.acquire() as conn:
        # Expected version and model should pass
        await verify_startup_guard(
            conn,
            expected_version=5,
            expected_embedding_model_id='nomic-ai/nomic-embed-text-v1.5',
        )

        # Mismatched schema version raises ConfigError E-CONFIG-01
        with pytest.raises(ConfigError) as exc_v:
            await verify_startup_guard(
                conn,
                expected_version=99,
                expected_embedding_model_id='nomic-ai/nomic-embed-text-v1.5',
            )
        assert exc_v.value.code == 'E-CONFIG-01'

        # Mismatched embedding model raises ConfigError E-CONFIG-01
        with pytest.raises(ConfigError) as exc_m:
            await verify_startup_guard(
                conn,
                expected_version=5,
                expected_embedding_model_id='different-model-v2',
            )
        assert exc_m.value.code == 'E-CONFIG-01'


async def test_seed_predicates_present(
    admin_pool: asyncpg.Pool,
    uow_factory: PostgresUnitOfWorkFactory,
) -> None:
    """Verify exactly 28 seeded predicates exist and match SEED_PREDICATES catalog."""
    async with admin_pool.acquire() as conn:
        count = await conn.fetchval('SELECT count(*) FROM predicates')
        assert count == 28

    # Query through PostgresPredicateRegistry
    user_id = UserId(uuid4())
    async with uow_factory.for_user(user_id) as uow:
        registry = PostgresPredicateRegistry(uow)
        await registry.load()
        all_preds = registry.all()
        assert len(all_preds) == 28

        pred_map = {p.name: p for p in all_preds}
        for seed in SEED_PREDICATES:
            assert seed.name in pred_map
            db_p = pred_map[seed.name]
            assert db_p.cardinality == seed.cardinality
            assert db_p.decay_class == seed.decay_class
            assert db_p.object_kind == seed.object_kind


async def test_cross_tenant_rls_isolation_t_sec_01(
    create_test_user: Callable[[UserId | None], Coroutine[Any, Any, UserId]],
    uow_factory: PostgresUnitOfWorkFactory,
) -> None:
    """T-SEC-01: Verify cross-tenant isolation across sources, entities, facts, and jobs."""
    user_a = await create_test_user(None)
    user_b = await create_test_user(None)

    now = datetime(2026, 10, 28, 12, 0, 0, tzinfo=UTC)
    source_a_id: SourceId
    entity_a_id: EntityId
    fact_a_id: FactId
    job_a_id: JobId

    # 1. User A inserts source, entity, fact, and job
    async with uow_factory.for_user(user_a) as uow:
        # Source
        inserted_sources = await uow.sources.insert_many(
            [
                NewSource(
                    user_id=user_a,
                    kind=SourceKind.CHAT,
                    role=Role.USER,
                    session_id='sess_a',
                    turn_index=0,
                    content='I moved to Delhi',
                    source_ts=now,
                    content_hash=f'hash_turn_a_{uuid4()}',
                )
            ]
        )
        assert len(inserted_sources) == 1
        source_a_id = inserted_sources[0].id

        # Entity
        entity_a = await uow.entities.create(
            user_id=user_a,
            name='Delhi',
            type=EntityType.PLACE,
            embedding=[0.0] * 768,
        )
        entity_a_id = entity_a.id

        # Fact
        self_entity = await uow.entities.ensure_self(user_a)
        cand = Candidate(
            user_id=user_a,
            subject_id=self_entity.id,
            predicate='lives_in',
            cardinality=Cardinality.SINGLE,
            object_entity_id=entity_a_id,
            object_value=None,
            object_norm='delhi',
            fact_text='User lives in Delhi',
            valid_from=datetime(2025, 1, 1, tzinfo=UTC),
            valid_to=None,
            time_precision=TimePrecision.DAY,
            valid_from_explicit=True,
            confidence=1.0,
            is_retraction=False,
            source_id=source_a_id,
            span=(0, 16),
            source_ts=now,
            event_hash=None,
            embedding=[0.0] * 768,
        )
        fact_a = await uow.facts.insert(
            cand,
            valid_from=cand.valid_from,
            valid_to=cand.valid_to,
            status=FactStatus.ACTIVE,
            origin=FactOrigin.EXTRACTED,
            model_id='default',
            now=now,
        )
        fact_a_id = fact_a.id

        # Job
        job_a_id = await uow.jobs.enqueue(
            NewJob(
                user_id=user_a,
                type=JobType.EXTRACT,
                payload={'session_id': 'sess_a'},
                idempotency_key=f'job_user_a_{uuid4()}',
            )
        )

    # 2. User B queries under for_user(user_b): sees 0 rows for all tables
    async with uow_factory.for_user(user_b) as uow:
        # Sources
        sources_b = await uow.sources.get_many(user_b, [source_a_id])
        assert sources_b == []
        win_b = await uow.sources.window(user_b, 'sess_a', [source_a_id], context_turns=4)
        assert len(win_b.targets) == 0

        # Entities
        entities_b = await uow.entities.get_many(user_b, [entity_a_id])
        assert entities_b == {}
        named_b = await uow.entities.find_by_norm_name(user_b, 'Delhi')
        assert named_b is None

        # Facts
        fact_b = await uow.facts.get(user_b, fact_a_id)
        assert fact_b is None
        timeline_b = await uow.facts.load_timeline(user_b, entity_a_id, 'lives_in', lock=False)
        assert timeline_b == []

        # Jobs
        job_b = await uow.jobs.get(user_b, job_a_id)
        assert job_b is None

    # 3. User B cannot mutate User A's rows
    async with uow_factory.for_user(user_b) as uow:
        # Trying to close User A's fact
        await uow.facts.close(
            user_b,
            fact_a_id,
            valid_to=datetime(2026, 1, 1, tzinfo=UTC),
            now=now,
        )

        # Trying to insert row with user_id = user_a while bound to user_b violates RLS WITH CHECK
        with pytest.raises((StorageError, asyncpg.exceptions.InsufficientPrivilegeError)):
            await uow.conn.execute(
                """
                INSERT INTO sources (
                    id, user_id, kind, role, session_id, turn_index,
                    content, source_ts, content_hash
                )
                VALUES ($1, $2, 'chat', 'user', 'sess_hack', 0, 'Injected', now(), 'injected_hash')
                """,
                uuid4(),
                user_a,
            )

    # Verify User A's fact was NOT closed by User B's attempt
    async with uow_factory.for_user(user_a) as uow:
        fact_check = await uow.facts.get(user_a, fact_a_id)
        assert fact_check is not None
        assert fact_check.valid_to is None
        assert fact_check.status == FactStatus.ACTIVE


async def test_rls_missing_guc_returns_zero_rows(
    create_test_user: Callable[[UserId | None], Coroutine[Any, Any, UserId]],
    uow_factory: PostgresUnitOfWorkFactory,
) -> None:
    """Verify system connection without app.user_id sees 0 rows in app_api role."""
    user = await create_test_user(None)
    now = datetime(2026, 10, 28, 12, 0, 0, tzinfo=UTC)

    # Insert a source as user
    async with uow_factory.for_user(user) as uow:
        res = await uow.sources.insert_many(
            [
                NewSource(
                    user_id=user,
                    kind=SourceKind.CHAT,
                    role=Role.USER,
                    session_id='sess_guc',
                    turn_index=0,
                    content='Test GUC isolation',
                    source_ts=now,
                    content_hash=f'turn_guc_{uuid4()}',
                )
            ]
        )
        source_id = res[0].id

    # Query with system() (no user GUC set, role app_api)
    async with uow_factory.system() as uow:
        rows = await uow.conn.fetch('SELECT * FROM sources WHERE id = $1', source_id)
        assert len(rows) == 0

        # Query entities without user context
        ent_rows = await uow.conn.fetch('SELECT * FROM entities')
        assert len(ent_rows) == 0

        # Query facts without user context
        fact_rows = await uow.conn.fetch('SELECT * FROM facts')
        assert len(fact_rows) == 0
