"""Unit tests for FactRepository, SearchRepository, and JobRepository (M3-04)."""

from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID, uuid4

import pytest
from memory_engine.db.repos.facts import PostgresFactRepository
from memory_engine.db.repos.jobs import PostgresJobRepository
from memory_engine.db.repos.search import PostgresSearchRepository
from memory_engine.db.uow import PostgresUnitOfWork
from memory_engine.domain.enums import (
    Cardinality,
    FactOrigin,
    FactStatus,
    JobType,
    RelationType,
    TimePrecision,
)
from memory_engine.domain.errors import LeaseLost
from memory_engine.domain.ids import EntityId, FactId, JobId, SourceId, UserId
from memory_engine.domain.models import Candidate, FactEvidence, FactRelation
from memory_engine.ports.repositories import (
    ErrorInfo,
    FactListQuery,
    FilterMode,
    NewExtractJob,
    NewJob,
    TemporalFilter,
)


class RecordingFakeConn:
    """Mock database connection recording queries and arguments."""

    def __init__(self) -> None:
        self.executed: list[tuple[str, tuple[Any, ...]]] = []
        self.fetch_queue: list[list[dict[str, Any]]] = []
        self.fetchrow_queue: list[dict[str, Any] | None] = []
        self.fetchval_queue: list[Any] = []
        self.execute_status_queue: list[str] = []
        self.execute_status: str = 'OK'

    async def execute(self, query: str, *args: Any) -> str:
        self.executed.append((query, args))
        if self.execute_status_queue:
            return self.execute_status_queue.pop(0)
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


def make_fact_row(
    fact_id: UUID,
    user_id: UserId,
    subject_id: UUID | None = None,
    predicate: str = 'lives_in',
    object_value: str = 'Pune',
    valid_from: datetime | None = None,
    status: str = 'active',
    score: float = 0.95,
) -> dict[str, Any]:
    now = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
    return {
        'id': fact_id,
        'user_id': user_id,
        'subject_id': subject_id or uuid4(),
        'predicate': predicate,
        'cardinality': 'single',
        'object_entity_id': None,
        'object_value': object_value,
        'object_norm': object_value.lower(),
        'fact_text': f'User {predicate} {object_value}',
        'valid_from': valid_from or now,
        'valid_to': None,
        'time_precision': 'day',
        'valid_from_explicit': True,
        'recorded_at': now,
        'invalidated_at': None,
        'confidence': 0.9,
        'salience': 1.0,
        'access_count': 0,
        'last_accessed_at': None,
        'status': status,
        'origin': 'extracted',
        'event_hash': None,
        'score': score,
    }


def make_job_row(
    job_id: UUID,
    user_id: UserId,
    status: str = 'queued',
    worker_id: UUID | None = None,
    attempts: int = 0,
) -> dict[str, Any]:
    now = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
    return {
        'id': job_id,
        'user_id': user_id,
        'type': 'extract',
        'payload': {'session_id': 's1', 'source_ids': [str(uuid4())]},
        'idempotency_key': f'extract:{job_id}',
        'coalesce_key': 'coalesce:1',
        'status': status,
        'attempts': attempts,
        'max_attempts': 3,
        'deferrals': 0,
        'run_after': now,
        'worker_id': worker_id,
        'lease_expires_at': now + timedelta(seconds=30) if worker_id else None,
        'last_error': None,
    }


def make_source_row(
    source_id: UUID,
    user_id: UserId,
    score: float = 0.88,
) -> dict[str, Any]:
    now = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
    return {
        'id': source_id,
        'user_id': user_id,
        'kind': 'chat',
        'role': 'user',
        'session_id': 'sess-1',
        'turn_index': 0,
        'content': 'I live in Pune',
        'source_ts': now,
        'content_hash': 'hash-1',
        'deleted_at': None,
        'score': score,
    }


# =============================================================================
# 1. FactRepository Tests
# =============================================================================


class TestPostgresFactRepository:
    @pytest.mark.asyncio
    async def test_load_timeline_and_find_for_resolution(self) -> None:
        conn = RecordingFakeConn()
        repo = PostgresFactRepository(conn)
        user_id = UserId(uuid4())
        subj_id = EntityId(uuid4())
        fid = uuid4()
        conn.fetch_queue.append([make_fact_row(fid, user_id, subject_id=subj_id)])
        conn.fetch_queue.append([make_fact_row(fid, user_id, subject_id=subj_id)])

        # Test unlocked load_timeline
        res_unlocked = await repo.load_timeline(user_id, subj_id, 'lives_in', lock=False)
        assert len(res_unlocked) == 1
        assert res_unlocked[0].id == FactId(fid)
        assert 'for update' not in conn.executed[0][0].lower()

        # Test find_for_resolution (locked load_timeline)
        res_locked = await repo.find_for_resolution(user_id, subj_id, 'lives_in')
        assert len(res_locked) == 1
        assert 'for update' in conn.executed[1][0].lower()

    @pytest.mark.asyncio
    async def test_insert_fact_mutation(self) -> None:
        conn = RecordingFakeConn()
        repo = PostgresFactRepository(conn)
        user_id = UserId(uuid4())
        subj_id = EntityId(uuid4())
        source_id = SourceId(uuid4())
        now = datetime(2026, 2, 1, 10, 0, tzinfo=UTC)
        valid_from = datetime(2026, 1, 1, 0, 0, tzinfo=UTC)

        cand = Candidate(
            user_id=user_id,
            subject_id=subj_id,
            predicate='works_at',
            cardinality=Cardinality.SINGLE,
            object_entity_id=None,
            object_value='Acme Corp',
            object_norm='acme corp',
            fact_text='User works at Acme Corp',
            valid_from=valid_from,
            valid_to=None,
            time_precision=TimePrecision.DAY,
            valid_from_explicit=True,
            confidence=0.92,
            is_retraction=False,
            source_id=source_id,
            span=(0, 23),
            source_ts=now,
            event_hash=None,
            embedding=[0.1, 0.2, 0.3],
        )

        inserted_fact = await repo.insert(
            cand,
            valid_from=valid_from,
            valid_to=None,
            status=FactStatus.ACTIVE,
            origin=FactOrigin.EXTRACTED,
            model_id='fastembed-bge',
            now=now,
        )

        assert inserted_fact.user_id == user_id
        assert inserted_fact.predicate == 'works_at'
        assert inserted_fact.status == FactStatus.ACTIVE
        assert inserted_fact.confidence == 0.92
        assert len(conn.executed) == 1
        query, args = conn.executed[0]
        assert 'insert into facts' in query.lower()
        assert args[1] == user_id
        assert args[3] == 'works_at'
        assert args[10] == 'fastembed-bge'

    @pytest.mark.asyncio
    async def test_close_extend_status_confidence_mutations(self) -> None:
        conn = RecordingFakeConn()
        repo = PostgresFactRepository(conn)
        user_id = UserId(uuid4())
        fact_id = FactId(uuid4())
        now = datetime(2026, 2, 1, 12, 0, tzinfo=UTC)
        valid_to = datetime(2026, 1, 15, 0, 0, tzinfo=UTC)
        valid_from = datetime(2025, 12, 1, 0, 0, tzinfo=UTC)

        # 1. Close fact
        await repo.close(user_id, fact_id, valid_to, now)
        assert 'update facts set valid_to' in conn.executed[0][0].lower()
        assert conn.executed[0][1] == (user_id, fact_id, valid_to, now)

        # 2. Extend valid from
        await repo.extend_valid_from(user_id, fact_id, valid_from)
        assert 'update facts set valid_from' in conn.executed[1][0].lower()
        assert conn.executed[1][1] == (user_id, fact_id, valid_from)

        # 3. Set status to deleted (sets invalidated_at)
        await repo.set_status(user_id, fact_id, FactStatus.DELETED, now)
        assert 'update facts set status' in conn.executed[2][0].lower()
        assert conn.executed[2][1] == (user_id, fact_id, 'deleted', now)

        # 4. Bump confidence
        await repo.bump_confidence(user_id, fact_id, 0.99)
        assert 'update facts set confidence' in conn.executed[3][0].lower()
        assert conn.executed[3][1] == (user_id, fact_id, 0.99)

    @pytest.mark.asyncio
    async def test_add_evidence_and_relation(self) -> None:
        conn = RecordingFakeConn()
        repo = PostgresFactRepository(conn)
        user_id = UserId(uuid4())
        f1_id = FactId(uuid4())
        f2_id = FactId(uuid4())
        source_id = SourceId(uuid4())

        # Evidence
        ev = FactEvidence(fact_id=f1_id, source_id=source_id, span_start=10, span_end=35)
        await repo.add_evidence(ev, user_id)
        assert 'insert into fact_sources' in conn.executed[0][0].lower()
        assert conn.executed[0][1] == (f1_id, source_id, user_id, 10, 35)

        # Relation
        rel = FactRelation(from_fact_id=f1_id, to_fact_id=f2_id, relation=RelationType.SUPERSEDES)
        job_id = uuid4()
        await repo.add_relation(user_id, rel, job_id)
        assert 'insert into fact_relations' in conn.executed[1][0].lower()
        assert conn.executed[1][1] == (user_id, f1_id, f2_id, 'supersedes', job_id)

    @pytest.mark.asyncio
    async def test_get_get_many_and_list_pagination(self) -> None:
        conn = RecordingFakeConn()
        repo = PostgresFactRepository(conn)
        user_id = UserId(uuid4())
        f1 = uuid4()
        f2 = uuid4()

        # get
        conn.fetchrow_queue.append(make_fact_row(f1, user_id))
        fact = await repo.get(user_id, FactId(f1))
        assert fact is not None
        assert fact.id == FactId(f1)

        # get_many
        conn.fetch_queue.append([make_fact_row(f1, user_id), make_fact_row(f2, user_id)])
        facts = await repo.get_many(user_id, [FactId(f1), FactId(f2)])
        assert len(facts) == 2

        # list with pagination
        conn.fetch_queue.append([make_fact_row(f1, user_id), make_fact_row(f2, user_id)])
        q = FactListQuery(status=FactStatus.ACTIVE, predicate='lives_in', page_size=1)
        page = await repo.list(user_id, q)
        assert len(page.items) == 1
        assert page.next_cursor == '1'
        assert 'limit $4 offset $5' in conn.executed[-1][0].lower()

    @pytest.mark.asyncio
    async def test_manual_update_and_tombstone_purge(self) -> None:
        conn = RecordingFakeConn()
        repo = PostgresFactRepository(conn)
        user_id = UserId(uuid4())
        fid = FactId(uuid4())
        now = datetime(2026, 1, 1, tzinfo=UTC)

        await repo.manual_update(
            user_id,
            fid,
            fact_text='New Fact Text',
            confidence=0.85,
            status=FactStatus.ACTIVE,
        )
        assert len(conn.executed) == 1
        assert 'update facts set' in conn.executed[0][0].lower()
        assert 'fact_text = $3' in conn.executed[0][0]
        assert 'status = $4' in conn.executed[0][0]
        assert 'confidence = $5' in conn.executed[0][0]

        conn.execute_status_queue.append('DELETE 4')
        deleted = await repo.hard_delete_expired_tombstones(now)
        assert deleted == 4
        assert 'delete from facts where status = ' in conn.executed[1][0].lower()


# =============================================================================
# 2. SearchRepository Tests
# =============================================================================


class TestPostgresSearchRepository:
    @pytest.mark.asyncio
    async def test_vector_facts_iterative_scan_and_temporal_splicing(self) -> None:
        conn = RecordingFakeConn()
        repo = PostgresSearchRepository(conn)
        user_id = UserId(uuid4())
        fid1 = uuid4()
        fid2 = uuid4()
        # Return slightly out-of-order rows to verify Python re-sort by score
        conn.fetch_queue.append(
            [
                make_fact_row(fid1, user_id, score=0.75),
                make_fact_row(fid2, user_id, score=0.92),
            ]
        )

        flt = TemporalFilter(mode=FilterMode.CURRENT)
        results = await repo.vector_facts(user_id, [0.1, 0.2, 0.3], flt, limit=10)

        # Verify HNSW settings executed first
        assert conn.executed[0][0] == "set local hnsw.iterative_scan = 'relaxed_order';"
        assert conn.executed[1][0] == 'set local hnsw.ef_search = 100;'

        # Verify Q-07 query and parameters
        query, args = conn.executed[2]
        assert 'select f.*, 1 - (f.embedding <=> $2) as score from facts f' in query
        assert "f.status = 'active' and f.valid_to is null and f.invalidated_at is null" in query
        assert 'order by f.embedding <=> $2 limit $3;' in query
        assert args == (user_id, [0.1, 0.2, 0.3], 10)

        # Verify Python re-sorting
        assert len(results) == 2
        assert results[0].score == 0.92
        assert results[0].fact.id == FactId(fid2)
        assert results[1].score == 0.75

    @pytest.mark.asyncio
    async def test_vector_facts_as_of_temporal_filter_shifting(self) -> None:
        conn = RecordingFakeConn()
        repo = PostgresSearchRepository(conn)
        user_id = UserId(uuid4())
        as_of_dt = datetime(2025, 6, 1, 12, 0, tzinfo=UTC)
        flt = TemporalFilter(mode=FilterMode.AS_OF, as_of=as_of_dt)

        await repo.vector_facts(user_id, [0.5, 0.5], flt, limit=5)

        query, args = conn.executed[2]
        # In Q-07, suffix $3 must have shifted to $4 because $3 is used for as_of!
        assert 'limit $4;' in query
        assert 'f.valid_from <= $3' in query
        assert args == (user_id, [0.5, 0.5], as_of_dt, 5)

    @pytest.mark.asyncio
    async def test_keyword_and_structured_facts(self) -> None:
        conn = RecordingFakeConn()
        repo = PostgresSearchRepository(conn)
        user_id = UserId(uuid4())
        fid = uuid4()
        conn.fetch_queue.append([make_fact_row(fid, user_id, score=0.8)])
        conn.fetch_queue.append([make_fact_row(fid, user_id, score=1.0)])

        flt = TemporalFilter(mode=FilterMode.CURRENT)

        # Keyword facts (Q-08)
        kw_res = await repo.keyword_facts(user_id, 'Pune', flt, limit=10)
        assert len(kw_res) == 1
        assert "websearch_to_tsquery('english', $2)" in conn.executed[0][0]
        assert conn.executed[0][1] == (user_id, 'Pune', 10)

        # Structured facts (Q-09)
        e_id = EntityId(uuid4())
        st_res = await repo.structured_facts(user_id, [e_id], ['lives_in'], flt, limit=10)
        assert len(st_res) == 1
        assert 'f.valid_from desc limit $4;' in conn.executed[1][0]
        assert conn.executed[1][1] == (user_id, [e_id], ['lives_in'], 10)

    @pytest.mark.asyncio
    async def test_vector_sources_and_graph_expansion(self) -> None:
        conn = RecordingFakeConn()
        repo = PostgresSearchRepository(conn)
        user_id = UserId(uuid4())
        s1 = uuid4()
        s2 = uuid4()
        conn.fetch_queue.append(
            [
                make_source_row(s1, user_id, score=0.6),
                make_source_row(s2, user_id, score=0.9),
            ]
        )

        # Vector sources (Q-10)
        before = datetime(2026, 1, 1, tzinfo=UTC)
        sources = await repo.vector_sources(user_id, [0.1, 0.2], limit=5, before=before)
        assert conn.executed[0][0] == "set local hnsw.iterative_scan = 'relaxed_order';"
        assert conn.executed[1][0] == 'set local hnsw.ef_search = 100;'
        assert len(sources) == 2
        assert sources[0].score == 0.9  # re-sorted
        assert conn.executed[2][1] == (user_id, [0.1, 0.2], 5, before)

        # Graph expansion (Q-11)
        fid = uuid4()
        e_seed = EntityId(uuid4())
        conn.fetch_queue.append([make_fact_row(fid, user_id, score=0.5)])
        flt = TemporalFilter(mode=FilterMode.CURRENT)
        graph_res = await repo.expand_graph(user_id, [e_seed], depth=2, flt=flt, limit=10)
        assert len(graph_res) == 1
        q_graph, args_graph = conn.executed[3]
        assert 'with recursive walk' in q_graph
        assert 'limit 15' in q_graph  # CR-03 degree cap
        assert args_graph == (user_id, [e_seed], 2, 10)


# =============================================================================
# 3. JobRepository Tests (Lease Fencing & Mutations)
# =============================================================================


class TestPostgresJobRepository:
    @pytest.mark.asyncio
    async def test_upsert_coalesced_and_enqueue(self) -> None:
        conn = RecordingFakeConn()
        repo = PostgresJobRepository(conn)
        user_id = UserId(uuid4())
        source_id = SourceId(uuid4())
        job_id = uuid4()

        # upsert_coalesced (Q-04)
        conn.fetchrow_queue.append({'id': job_id, 'inserted': True, 'sealed': False})
        j_ext = NewExtractJob(
            user_id=user_id,
            session_id='sess-1',
            source_id=source_id,
            idempotency_key='extract:src-1',
            coalesce_key='extract:sess-1:0',
        )
        res = await repo.upsert_coalesced(j_ext, debounce_s=2.5)
        assert res.job_id == JobId(job_id)
        assert res.created is True
        assert res.sealed is False
        assert 'insert into jobs' in conn.executed[0][0].lower()
        assert conn.executed[0][1] == (
            user_id,
            'sess-1',
            str(source_id),
            'extract:src-1',
            'extract:sess-1:0',
            2.5,
        )

        # enqueue (idempotent insert)
        conn.fetchrow_queue.append({'id': job_id})
        nj = NewJob(
            user_id=user_id,
            type=JobType.EXTRACT,
            payload={'k': 'v'},
            idempotency_key='idem-1',
        )
        enqueued_id = await repo.enqueue(nj)
        assert enqueued_id == JobId(job_id)

    @pytest.mark.asyncio
    async def test_claim_batch(self) -> None:
        conn = RecordingFakeConn()
        repo = PostgresJobRepository(conn)
        worker_id = uuid4()
        user_id = UserId(uuid4())
        j1 = uuid4()
        j2 = uuid4()
        conn.fetch_queue.append(
            [
                make_job_row(j1, user_id, status='running', worker_id=worker_id),
                make_job_row(j2, user_id, status='running', worker_id=worker_id),
            ]
        )

        claimed = await repo.claim_batch(worker_id, limit=2, lease_s=30)
        assert len(claimed) == 2
        assert claimed[0].id == JobId(j1)
        assert 'skip locked' in conn.executed[0][0].lower()
        assert conn.executed[0][1] == (worker_id, 2, 30)

    @pytest.mark.asyncio
    async def test_lease_fencing_heartbeat_raises_lease_lost(self) -> None:
        conn = RecordingFakeConn()
        repo = PostgresJobRepository(conn)
        job_id = JobId(uuid4())
        worker_id = uuid4()

        # Success: 1 row updated
        conn.execute_status_queue.append('UPDATE 1')
        assert await repo.heartbeat(job_id, worker_id, lease_s=30) is True

        # Fenced out: 0 rows updated (e.g. lease expired or reclaimed by another worker)
        conn.execute_status_queue.append('UPDATE 0')
        with pytest.raises(LeaseLost, match='Lease lost for job'):
            await repo.heartbeat(job_id, worker_id, lease_s=30)

    @pytest.mark.asyncio
    async def test_lease_fencing_complete_raises_lease_lost(self) -> None:
        conn = RecordingFakeConn()
        repo = PostgresJobRepository(conn)
        job_id = JobId(uuid4())
        worker_id = uuid4()
        now = datetime.now(UTC)

        # Success
        conn.execute_status_queue.append('UPDATE 1')
        assert await repo.complete(job_id, worker_id, now) is True

        # Fenced out on complete: raises LeaseLost
        conn.execute_status_queue.append('UPDATE 0')
        with pytest.raises(LeaseLost, match='Lease lost for job'):
            await repo.complete(job_id, worker_id, now)

    @pytest.mark.asyncio
    async def test_lease_token_parameter_support(self) -> None:
        conn = RecordingFakeConn()
        repo = PostgresJobRepository(conn)
        job_id = JobId(uuid4())
        stale_token = uuid4()
        now = datetime.now(UTC)

        # Passing lease_token to heartbeat
        conn.execute_status_queue.append('UPDATE 0')
        with pytest.raises(LeaseLost):
            await repo.heartbeat(job_id, worker_id=uuid4(), lease_s=30, lease_token=stale_token)
        assert conn.executed[0][1][1] == stale_token

        # Passing lease_token to complete
        conn.execute_status_queue.append('UPDATE 0')
        with pytest.raises(LeaseLost):
            await repo.complete(job_id, worker_id=uuid4(), now=now, lease_token=stale_token)
        assert conn.executed[1][1][1] == stale_token

    @pytest.mark.asyncio
    async def test_retry_fail_and_reap_expired(self) -> None:
        conn = RecordingFakeConn()
        repo = PostgresJobRepository(conn)
        job_id = JobId(uuid4())
        worker_id = uuid4()
        now = datetime.now(UTC)
        err = ErrorInfo(message='Rate limited', code='E-RATE-LIMIT')

        conn.execute_status_queue.append('UPDATE 1')
        retried = await repo.fail(job_id, worker_id, err, run_after=now + timedelta(seconds=10))
        assert retried is True
        assert 'update jobs set status = case when attempts >= max_attempts' in conn.executed[0][0]

        # Reap expired leases (Q-17)
        conn.execute_status_queue.append('UPDATE 3')
        reaped = await repo.reap_expired(now)
        assert reaped == 3
        assert "where status = 'running' and lease_expires_at < now()" in conn.executed[1][0]


# =============================================================================
# 4. PostgresUnitOfWork Wiring Verification
# =============================================================================


class FakePool:
    def __init__(self) -> None:
        self.conn = RecordingFakeConn()

    async def acquire(self) -> RecordingFakeConn:
        return self.conn

    async def release(self, conn: Any) -> None:
        pass


@pytest.mark.asyncio
async def test_uow_wires_all_repositories() -> None:
    user_id = UserId(uuid4())
    pool = FakePool()
    uow = PostgresUnitOfWork(pool, user_id=user_id)

    assert isinstance(uow.facts, PostgresFactRepository)
    assert isinstance(uow.search, PostgresSearchRepository)
    assert isinstance(uow.jobs, PostgresJobRepository)
