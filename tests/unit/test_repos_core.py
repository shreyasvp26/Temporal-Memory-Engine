"""Unit tests for PostgreSQL core repository implementations and predicate registries."""

from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import uuid4

import pytest
from memory_engine.db.repos.audit import PostgresAuditRepository
from memory_engine.db.repos.entities import PostgresEntityRepository
from memory_engine.db.repos.messages import PostgresMessageRepository
from memory_engine.db.repos.predicates import (
    PostgresPredicateRegistry,
    StaticPredicateRegistry,
)
from memory_engine.db.repos.profile import PostgresProfileRepository
from memory_engine.db.repos.sources import PostgresSourceRepository
from memory_engine.db.uow import PostgresUnitOfWork
from memory_engine.domain.clock import FixedClock
from memory_engine.domain.enums import (
    Cardinality,
    DecayClass,
    EntityType,
    ObjectKind,
    Role,
    SourceKind,
)
from memory_engine.domain.ids import EntityId, SourceId, UserId
from memory_engine.ports.repositories import NewMessage, NewSource


class RecordingFakeConn:
    """Mock database connection recording calls and returning configured responses."""

    def __init__(self) -> None:
        self.executed: list[tuple[str, tuple[Any, ...]]] = []
        self.fetch_queue: list[list[dict[str, Any]]] = []
        self.fetchrow_queue: list[dict[str, Any] | None] = []
        self.fetchval_queue: list[Any] = []
        self.execute_status: str = 'OK'

    async def execute(self, query: str, *args: Any) -> str:
        self.executed.append((query, args))
        return self.execute_status

    async def fetch(self, query: str, *args: Any) -> list[dict[str, Any]]:
        self.executed.append((query, args))
        if self.fetch_queue:
            return self.fetch_queue.pop(0)
        return []

    async def fetchrow(self, query: str, *args: Any) -> dict[str, Any] | None:
        self.executed.append((query, args))
        if self.fetchrow_queue:
            return self.fetchrow_queue.pop(0)
        return None

    async def fetchval(self, query: str, *args: Any) -> Any:
        self.executed.append((query, args))
        if self.fetchval_queue:
            return self.fetchval_queue.pop(0)
        return None


# =============================================================================
# 1. StaticPredicateRegistry & PostgresPredicateRegistry Tests
# =============================================================================


class TestStaticPredicateRegistry:
    def test_contains_all_28_frozen_predicates(self) -> None:
        registry = StaticPredicateRegistry()
        all_preds = registry.all()
        assert len(all_preds) == 28

        pred_names = {p.name for p in all_preds}
        expected_names = {
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
        assert pred_names == expected_names

    def test_predicate_properties_match_spec(self) -> None:
        registry = StaticPredicateRegistry()

        lives_in = registry.get('lives_in')
        assert lives_in.cardinality == Cardinality.SINGLE
        assert lives_in.decay_class == DecayClass.SLOW
        assert lives_in.object_kind == ObjectKind.ENTITY
        assert lives_in.template == '{subject} lives in {object}'
        assert 'moved_to' in lives_in.synonyms
        assert 'resides_in' in lives_in.synonyms

        job_title = registry.get('job_title')
        assert job_title.cardinality == Cardinality.SINGLE
        assert job_title.decay_class == DecayClass.SLOW
        assert job_title.object_kind == ObjectKind.LITERAL
        assert job_title.synonyms == ('job', 'role')

        attended = registry.get('attended')
        assert attended.cardinality == Cardinality.EVENT
        assert attended.decay_class == DecayClass.NORMAL
        assert attended.object_kind == ObjectKind.ENTITY

        other = registry.get('other')
        assert other.cardinality == Cardinality.MULTI
        assert other.decay_class == DecayClass.NORMAL
        assert other.object_kind == ObjectKind.LITERAL

    @pytest.mark.parametrize(
        ('synonym', 'expected_predicate'),
        [
            ('moved_to', 'lives_in'),
            ('resides_in', 'lives_in'),
            ('employed_at', 'works_at'),
            ('works_for', 'works_at'),
            ('job', 'job_title'),
            ('role', 'job_title'),
            ('loves', 'likes'),
            ('enjoys', 'likes'),
        ],
    )
    def test_synonym_resolution(self, synonym: str, expected_predicate: str) -> None:
        registry = StaticPredicateRegistry()
        resolved = registry.resolve(synonym)
        assert resolved.name == expected_predicate

    def test_case_and_whitespace_insensitivity(self) -> None:
        registry = StaticPredicateRegistry()
        assert registry.resolve('  LIVES_IN  ').name == 'lives_in'
        assert registry.resolve('EMPLOYED_AT').name == 'works_at'
        assert registry.get('  Works_At  ').name == 'works_at'

    def test_unknown_predicate_resolves_to_other(self) -> None:
        registry = StaticPredicateRegistry()
        resolved = registry.resolve('completely_unknown_predicate')
        assert resolved.name == 'other'

    def test_get_raises_key_error_on_unknown(self) -> None:
        registry = StaticPredicateRegistry()
        with pytest.raises(KeyError, match='Unknown predicate'):
            registry.get('non_existent')


class TestPostgresPredicateRegistry:
    @pytest.mark.asyncio
    async def test_delegates_to_static_initially(self) -> None:
        conn = RecordingFakeConn()
        reg = PostgresPredicateRegistry(conn)
        assert reg.resolve('lives_in').name == 'lives_in'
        assert reg.resolve('loves').name == 'likes'
        assert len(reg.all()) == 28

    @pytest.mark.asyncio
    async def test_refresh_loads_from_database(self) -> None:
        conn = RecordingFakeConn()
        custom_row = {
            'name': 'custom_pred',
            'cardinality': 'single',
            'decay_class': 'slow',
            'object_kind': 'literal',
            'template': '{subject} custom {object}',
            'synonyms': ['custom_syn'],
        }
        conn.fetch_queue.append([custom_row])

        reg = PostgresPredicateRegistry(conn)
        await reg.refresh()

        assert len(conn.executed) == 1
        assert 'select * from predicates' in conn.executed[0][0]
        assert reg.resolve('custom_syn').name == 'custom_pred'


# =============================================================================
# 2. PostgresSourceRepository Tests
# =============================================================================


class TestPostgresSourceRepository:
    @pytest.mark.asyncio
    async def test_insert_many_new_and_conflict(self) -> None:
        conn = RecordingFakeConn()
        repo = PostgresSourceRepository(conn)
        user_id = UserId(uuid4())
        s1_id = uuid4()
        s2_id = uuid4()

        # s1 succeeds insert, s2 conflicts and queries existing
        conn.fetchrow_queue.append({'id': s1_id})
        conn.fetchrow_queue.append(None)  # conflict on insert
        conn.fetchrow_queue.append({'id': s2_id})  # select existing

        now = datetime.now(UTC)
        items = [
            NewSource(
                user_id=user_id,
                kind=SourceKind.CHAT,
                role=Role.USER,
                session_id='sess-1',
                turn_index=0,
                content='hello',
                source_ts=now,
                content_hash='hash1',
            ),
            NewSource(
                user_id=user_id,
                kind=SourceKind.CHAT,
                role=Role.ASSISTANT,
                session_id='sess-1',
                turn_index=1,
                content='hi there',
                source_ts=now + timedelta(seconds=1),
                content_hash='hash2',
            ),
        ]

        inserted = await repo.insert_many(items)
        assert len(inserted) == 2
        assert inserted[0].id == SourceId(s1_id)
        assert inserted[0].created is True
        assert inserted[1].id == SourceId(s2_id)
        assert inserted[1].created is False
        assert len(conn.executed) == 3

    @pytest.mark.asyncio
    async def test_insert_many_empty(self) -> None:
        conn = RecordingFakeConn()
        repo = PostgresSourceRepository(conn)
        assert await repo.insert_many([]) == []
        assert len(conn.executed) == 0

    @pytest.mark.asyncio
    async def test_get_many(self) -> None:
        conn = RecordingFakeConn()
        repo = PostgresSourceRepository(conn)
        user_id = UserId(uuid4())
        src_id = uuid4()
        now = datetime.now(UTC)

        conn.fetch_queue.append(
            [
                {
                    'id': src_id,
                    'user_id': user_id,
                    'kind': 'chat',
                    'role': 'user',
                    'session_id': 'sess-1',
                    'turn_index': 0,
                    'content': 'test content',
                    'source_ts': now,
                    'content_hash': 'h1',
                    'deleted_at': None,
                }
            ]
        )

        sources = await repo.get_many(user_id, [SourceId(src_id)])
        assert len(sources) == 1
        assert sources[0].id == SourceId(src_id)
        assert sources[0].content == 'test content'
        assert len(conn.executed) == 1

    @pytest.mark.asyncio
    async def test_window_and_get_window(self) -> None:
        conn = RecordingFakeConn()
        repo = PostgresSourceRepository(conn)
        user_id = UserId(uuid4())
        now = datetime.now(UTC)
        t1_id = uuid4()
        c1_id = uuid4()

        # Targets response
        conn.fetch_queue.append(
            [
                {
                    'id': t1_id,
                    'user_id': user_id,
                    'kind': 'chat',
                    'role': 'user',
                    'session_id': 'sess-1',
                    'turn_index': 5,
                    'content': 'target turn',
                    'source_ts': now,
                    'content_hash': 'ht',
                    'deleted_at': None,
                }
            ]
        )
        # Context response (ordered desc from DB, window should reverse to asc)
        conn.fetch_queue.append(
            [
                {
                    'id': c1_id,
                    'user_id': user_id,
                    'kind': 'chat',
                    'role': 'assistant',
                    'session_id': 'sess-1',
                    'turn_index': 4,
                    'content': 'earlier turn',
                    'source_ts': now - timedelta(seconds=10),
                    'content_hash': 'hc',
                    'deleted_at': None,
                }
            ]
        )

        window = await repo.get_window(
            user_id, 'sess-1', target_ids=[SourceId(t1_id)], context_turns=4
        )
        assert len(window.targets) == 1
        assert window.targets[0].id == SourceId(t1_id)
        assert len(window.context) == 1
        assert window.context[0].id == SourceId(c1_id)
        assert len(conn.executed) == 2

    @pytest.mark.asyncio
    async def test_set_embeddings(self) -> None:
        conn = RecordingFakeConn()
        repo = PostgresSourceRepository(conn)
        user_id = UserId(uuid4())
        sid = SourceId(uuid4())

        await repo.set_embeddings(user_id, [(sid, [0.1, 0.2, 0.3])], 'nomic-v1.5')
        assert len(conn.executed) == 1
        query, args = conn.executed[0]
        assert 'update sources set embedding' in query
        assert args == (user_id, sid, [0.1, 0.2, 0.3], 'nomic-v1.5')

    @pytest.mark.asyncio
    async def test_soft_and_hard_delete(self) -> None:
        conn = RecordingFakeConn()
        repo = PostgresSourceRepository(conn)
        user_id = UserId(uuid4())
        sid = SourceId(uuid4())
        now = datetime.now(UTC)

        await repo.soft_delete(user_id, [sid], now)
        assert len(conn.executed) == 1
        assert 'update sources set deleted_at' in conn.executed[0][0]

        conn.execute_status = 'DELETE 1'
        deleted_count = await repo.hard_delete(user_id, [sid])
        assert deleted_count == 1
        assert 'delete from sources' in conn.executed[1][0]

    @pytest.mark.asyncio
    async def test_list_unprocessed_for_session(self) -> None:
        conn = RecordingFakeConn()
        repo = PostgresSourceRepository(conn)
        user_id = UserId(uuid4())
        sid = uuid4()
        now = datetime.now(UTC)

        conn.fetch_queue.append(
            [
                {
                    'id': sid,
                    'user_id': user_id,
                    'kind': 'chat',
                    'role': 'user',
                    'session_id': 'sess-1',
                    'turn_index': 0,
                    'content': 'unprocessed query',
                    'source_ts': now,
                    'content_hash': 'hu',
                    'deleted_at': None,
                }
            ]
        )

        results = await repo.list_unprocessed_for_session(user_id, 'sess-1', limit=10)
        assert len(results) == 1
        assert results[0].id == SourceId(sid)
        assert len(conn.executed) == 1
        assert 'jobs j on j.user_id = s.user_id' in conn.executed[0][0]


# =============================================================================
# 3. PostgresEntityRepository Tests
# =============================================================================


class TestPostgresEntityRepository:
    @pytest.mark.asyncio
    async def test_ensure_self_existing(self) -> None:
        conn = RecordingFakeConn()
        repo = PostgresEntityRepository(conn)
        user_id = UserId(uuid4())
        eid = uuid4()

        conn.fetchrow_queue.append(
            {
                'id': eid,
                'user_id': user_id,
                'canonical_name': 'User',
                'type': 'person',
                'is_self': True,
            }
        )

        self_entity = await repo.get_or_create_self(user_id)
        assert self_entity.id == EntityId(eid)
        assert self_entity.is_self is True
        assert len(conn.executed) == 1

    @pytest.mark.asyncio
    async def test_ensure_self_creates_when_missing(self) -> None:
        conn = RecordingFakeConn()
        repo = PostgresEntityRepository(conn)
        user_id = UserId(uuid4())
        eid = uuid4()

        conn.fetchrow_queue.append(None)  # not found
        conn.fetchrow_queue.append(
            {  # insert returns created
                'id': eid,
                'user_id': user_id,
                'canonical_name': 'User',
                'type': 'person',
                'is_self': True,
            }
        )

        self_entity = await repo.ensure_self(user_id)
        assert self_entity.id == EntityId(eid)
        assert self_entity.is_self is True
        assert len(conn.executed) == 2

    @pytest.mark.asyncio
    async def test_find_by_alias_and_norm_name(self) -> None:
        conn = RecordingFakeConn()
        repo = PostgresEntityRepository(conn)
        user_id = UserId(uuid4())
        eid = uuid4()

        # First by alias
        conn.fetchrow_queue.append(
            {
                'id': eid,
                'user_id': user_id,
                'canonical_name': 'Paris',
                'type': 'place',
                'is_self': False,
            }
        )

        found = await repo.find_by_norm_name(user_id, EntityType.PLACE, 'city of light')
        assert found is not None
        assert found.canonical_name == 'Paris'
        assert len(conn.executed) == 1

        # Second test: alias miss, canonical name fallback hit
        conn.fetchrow_queue.append(None)  # alias miss
        conn.fetchrow_queue.append(
            {
                'id': eid,
                'user_id': user_id,
                'canonical_name': 'Paris',
                'type': 'place',
                'is_self': False,
            }
        )

        found2 = await repo.find_by_alias(user_id, EntityType.PLACE, 'paris')
        assert found2 is not None
        assert found2.canonical_name == 'Paris'
        assert len(conn.executed) == 3

    @pytest.mark.asyncio
    async def test_nearest_and_search_similar(self) -> None:
        conn = RecordingFakeConn()
        repo = PostgresEntityRepository(conn)
        user_id = UserId(uuid4())
        eid = uuid4()

        conn.fetch_queue.append(
            [
                {
                    'id': eid,
                    'user_id': user_id,
                    'canonical_name': 'Acme Corp',
                    'type': 'org',
                    'is_self': False,
                    'sim': 0.95,
                }
            ]
        )

        results = await repo.search_similar_by_embedding(user_id, EntityType.ORG, [0.1, 0.2], k=5)
        assert len(results) == 1
        entity, sim = results[0]
        assert entity.canonical_name == 'Acme Corp'
        assert sim == 0.95
        assert len(conn.executed) == 1

    @pytest.mark.asyncio
    async def test_create_and_add_alias(self) -> None:
        conn = RecordingFakeConn()
        repo = PostgresEntityRepository(conn)
        user_id = UserId(uuid4())
        eid = uuid4()

        conn.fetchrow_queue.append(
            {
                'id': eid,
                'user_id': user_id,
                'canonical_name': 'Bob',
                'type': 'person',
                'is_self': False,
            }
        )

        created = await repo.create(user_id, 'Bob', EntityType.PERSON, [0.1, 0.2])
        assert created.id == EntityId(eid)
        assert created.canonical_name == 'Bob'
        # 1 insert entity + 1 insert alias
        assert len(conn.executed) == 2
        assert 'insert into entities' in conn.executed[0][0]
        assert 'insert into entity_aliases' in conn.executed[1][0]

    @pytest.mark.asyncio
    async def test_get_many_and_alias_index_and_list_names(self) -> None:
        conn = RecordingFakeConn()
        repo = PostgresEntityRepository(conn)
        user_id = UserId(uuid4())
        eid = uuid4()

        # get_many
        conn.fetch_queue.append(
            [
                {
                    'id': eid,
                    'user_id': user_id,
                    'canonical_name': 'Alice',
                    'type': 'person',
                    'is_self': False,
                }
            ]
        )
        mapping = await repo.get_many(user_id, [EntityId(eid)])
        assert EntityId(eid) in mapping
        assert mapping[EntityId(eid)].canonical_name == 'Alice'

        # alias_index
        conn.fetch_queue.append([{'alias_norm': 'alice', 'entity_id': eid}])
        idx = await repo.alias_index(user_id)
        assert idx == {'alice': EntityId(eid)}

        # list_all_names
        conn.fetch_queue.append([{'canonical_name': 'Alice'}, {'canonical_name': 'Bob'}])
        names = await repo.list_all_names(user_id)
        assert names == ['Alice', 'Bob']


# =============================================================================
# 4. PostgresMessageRepository Tests
# =============================================================================


class TestPostgresMessageRepository:
    @pytest.mark.asyncio
    async def test_ensure_conversation(self) -> None:
        conn = RecordingFakeConn()
        repo = PostgresMessageRepository(conn)
        user_id = UserId(uuid4())
        cid = uuid4()
        now = datetime.now(UTC)

        res_cid = await repo.ensure_conversation(user_id, cid, now)
        assert res_cid == cid
        assert len(conn.executed) == 1
        assert 'insert into conversations' in conn.executed[0][0]

    @pytest.mark.asyncio
    async def test_insert_message_and_idempotency(self) -> None:
        clock = FixedClock(datetime(2026, 10, 27, 12, 0, tzinfo=UTC))
        conn = RecordingFakeConn()
        repo = PostgresMessageRepository(conn, clock=clock)
        user_id = UserId(uuid4())
        cid = uuid4()
        mid = uuid4()

        # 1. Successful insert
        conn.fetchrow_queue.append({'id': mid})
        msg = NewMessage(
            conversation_id=cid,
            user_id=user_id,
            role=Role.USER,
            content='hello world',
            client_message_id='cli-1',
        )
        res1 = await repo.insert(msg)
        assert res1.id == mid
        assert res1.created is True

        # 2. Idempotent hit
        conn.fetchrow_queue.append(None)  # conflict
        conn.fetchrow_queue.append({'id': mid})  # select existing
        res2 = await repo.insert(msg)
        assert res2.id == mid
        assert res2.created is False

    @pytest.mark.asyncio
    async def test_recent_turns_and_pending_user_messages(self) -> None:
        conn = RecordingFakeConn()
        repo = PostgresMessageRepository(conn)
        user_id = UserId(uuid4())
        cid = uuid4()
        mid = uuid4()
        now = datetime.now(UTC)

        conn.fetch_queue.append(
            [
                {
                    'id': mid,
                    'conversation_id': cid,
                    'user_id': user_id,
                    'role': 'user',
                    'content': 'recent text',
                    'client_message_id': 'c1',
                    'source_id': None,
                    'partial': False,
                    'created_at': now,
                }
            ]
        )

        messages = await repo.list_recent(user_id, cid, 5)
        assert len(messages) == 1
        assert messages[0].content == 'recent text'
        assert 'order by created_at desc limit $3' in conn.executed[0][0]

        # pending_user_messages
        conn.fetch_queue.append(
            [
                {
                    'id': mid,
                    'conversation_id': cid,
                    'user_id': user_id,
                    'role': 'user',
                    'content': 'pending text',
                    'client_message_id': 'c2',
                    'source_id': None,
                    'partial': False,
                    'created_at': now,
                }
            ]
        )
        pending = await repo.pending_user_messages(user_id, 10)
        assert len(pending) == 1
        assert pending[0].content == 'pending text'

    @pytest.mark.asyncio
    async def test_finalize_assistant_and_set_job(self) -> None:
        conn = RecordingFakeConn()
        repo = PostgresMessageRepository(conn)
        user_id = UserId(uuid4())
        mid = uuid4()
        jid = uuid4()
        sid = SourceId(uuid4())

        await repo.finalize_assistant(user_id, mid, 'final response', partial=False, trace={'t': 1})
        assert 'update messages' in conn.executed[0][0]

        await repo.set_job(user_id, mid, job_id=jid, source_id=sid)
        assert len(conn.executed) == 2
        assert 'trace = coalesce(trace' in conn.executed[1][0]


# =============================================================================
# 5. PostgresAuditRepository & PostgresProfileRepository Tests
# =============================================================================


class TestPostgresAuditRepository:
    @pytest.mark.asyncio
    async def test_append_and_record(self) -> None:
        conn = RecordingFakeConn()
        repo = PostgresAuditRepository(conn)
        user_id = UserId(uuid4())
        target_id = uuid4()

        await repo.record(
            user_id=user_id,
            actor='user',
            action='fact.edit',
            target_type='fact',
            target_id=target_id,
            before={'val': 'old'},
            after={'val': 'new'},
        )
        assert len(conn.executed) == 1
        assert 'insert into audit_log' in conn.executed[0][0]

    @pytest.mark.asyncio
    async def test_list_for_user(self) -> None:
        conn = RecordingFakeConn()
        repo = PostgresAuditRepository(conn)
        user_id = UserId(uuid4())

        conn.fetch_queue.append([{'id': 1, 'action': 'fact.edit'}])
        logs = await repo.list_for_user(user_id)
        assert len(logs) == 1
        assert logs[0]['action'] == 'fact.edit'


class TestPostgresProfileRepository:
    @pytest.mark.asyncio
    async def test_ensure_and_timezone(self) -> None:
        conn = RecordingFakeConn()
        repo = PostgresProfileRepository(conn)
        user_id = UserId(uuid4())

        await repo.ensure(user_id)
        # 1 insert auth.users + 1 insert profiles
        assert len(conn.executed) == 2
        assert 'insert into auth.users' in conn.executed[0][0]
        assert 'insert into profiles' in conn.executed[1][0]

        conn.fetchval_queue.append('America/New_York')
        tz = await repo.timezone(user_id)
        assert tz == 'America/New_York'

        await repo.set_timezone(user_id, 'Asia/Tokyo')
        assert 'update set timezone = excluded.timezone' in conn.executed[-1][0]

    @pytest.mark.asyncio
    async def test_list_user_ids_and_purge_user(self) -> None:
        conn = RecordingFakeConn()
        repo = PostgresProfileRepository(conn)
        u1 = uuid4()

        conn.fetch_queue.append([{'user_id': u1}])
        users = await repo.list_user_ids()
        assert users == [UserId(u1)]

        conn.execute_status = 'DELETE 1'
        purged = await repo.purge_user(UserId(u1))
        assert purged == 1
        assert 'delete from auth.users where id = $1' in conn.executed[-1][0]

    @pytest.mark.asyncio
    async def test_export_user(self) -> None:
        conn = RecordingFakeConn()
        repo = PostgresProfileRepository(conn)
        user_id = UserId(uuid4())

        conn.fetchval_queue.append('UTC')
        conn.fetch_queue.append([{'id': uuid4(), 'fact_text': 'fact1'}])  # facts
        conn.fetch_queue.append([{'id': uuid4(), 'content': 'source1'}])  # sources
        conn.fetch_queue.append([{'id': uuid4(), 'canonical_name': 'ent1'}])  # entities
        conn.fetch_queue.append([{'id': uuid4(), 'content': 'msg1'}])  # messages

        export = await repo.export_user(user_id)
        assert export['user_id'] == str(user_id)
        assert export['timezone'] == 'UTC'
        assert len(export['facts']) == 1
        assert len(export['sources']) == 1
        assert len(export['entities']) == 1
        assert len(export['messages']) == 1


# =============================================================================
# 6. PostgresUnitOfWork Wiring Tests
# =============================================================================


class TestUoWRepositoriesWiring:
    @pytest.mark.asyncio
    async def test_uow_instantiates_repos(self) -> None:
        fake_conn = RecordingFakeConn()

        class FakePool:
            async def acquire(self) -> Any:
                return fake_conn

            async def release(self, conn: Any) -> None:
                pass

        class FakeTxConn(RecordingFakeConn):
            def transaction(self, readonly: bool = False, isolation: str = 'read_committed') -> Any:
                class DummyTx:
                    async def start(self) -> None:
                        pass

                    async def commit(self) -> None:
                        pass

                    async def rollback(self) -> None:
                        pass

                return DummyTx()

        tx_conn = FakeTxConn()

        class PoolWithTx:
            async def acquire(self) -> Any:
                return tx_conn

            async def release(self, conn: Any) -> None:
                pass

        user_id = UserId(uuid4())
        uow = PostgresUnitOfWork(pool=PoolWithTx(), user_id=user_id)

        async with uow:
            assert isinstance(uow.sources, PostgresSourceRepository)
            assert isinstance(uow.entities, PostgresEntityRepository)
            assert isinstance(uow.messages, PostgresMessageRepository)
            assert isinstance(uow.audit, PostgresAuditRepository)
            assert isinstance(uow.profiles, PostgresProfileRepository)

            # Test that repo.conn accesses the active transaction connection
            assert uow.sources.conn is tx_conn
            assert uow.entities.conn is tx_conn
            assert uow.messages.conn is tx_conn
            assert uow.audit.conn is tx_conn
            assert uow.profiles.conn is tx_conn
