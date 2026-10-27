"""Integration tests for outbox, Q-04 coalescing/sealing, Q-01 claiming, and fencing."""

import asyncio
from collections.abc import Callable, Coroutine
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import uuid4

import pytest
from memory_engine.db.uow import PostgresUnitOfWorkFactory
from memory_engine.domain.enums import JobStatus, JobType
from memory_engine.domain.errors import LeaseLost
from memory_engine.domain.ids import JobId, SourceId, UserId
from memory_engine.ports.repositories import ErrorInfo, NewExtractJob, NewJob

pytestmark = pytest.mark.integration


async def test_q04_coalescing_and_sealing_at_12_sources(
    create_test_user: Callable[[UserId | None], Coroutine[Any, Any, UserId]],
    uow_factory: PostgresUnitOfWorkFactory,
) -> None:
    """CR-02 / Q-04: Coalesced extract job seals on 12th source and creates new job on 13th."""
    user_id = await create_test_user(None)
    coalesce_key = f'chat:{user_id}:sess_coalesce_{uuid4()}'

    initial_job_id: JobId

    async with uow_factory.for_user(user_id) as uow:
        # 1. Source 1: creates the job
        res1 = await uow.jobs.upsert_coalesced(
            NewExtractJob(
                user_id=user_id,
                session_id='sess_1',
                source_id=SourceId(uuid4()),
                idempotency_key=f'idem_1_{uuid4()}',
                coalesce_key=coalesce_key,
            ),
            debounce_s=15.0,
        )
        assert res1.created is True
        assert res1.sealed is False
        initial_job_id = res1.job_id

        # 2. Sources 2 through 11: coalesce into the same job
        for i in range(2, 12):
            res_mid = await uow.jobs.upsert_coalesced(
                NewExtractJob(
                    user_id=user_id,
                    session_id='sess_1',
                    source_id=SourceId(uuid4()),
                    idempotency_key=f'idem_{i}_{uuid4()}',
                    coalesce_key=coalesce_key,
                ),
                debounce_s=15.0,
            )
            assert res_mid.job_id == initial_job_id
            assert res_mid.created is False
            assert res_mid.sealed is False

        # 3. Source 12: reaches limit 12, seals the job (coalesce_key set to NULL)
        res12 = await uow.jobs.upsert_coalesced(
            NewExtractJob(
                user_id=user_id,
                session_id='sess_1',
                source_id=SourceId(uuid4()),
                idempotency_key=f'idem_12_{uuid4()}',
                coalesce_key=coalesce_key,
            ),
            debounce_s=15.0,
        )
        assert res12.job_id == initial_job_id
        assert res12.created is False

        # Verify job in DB has coalesce_key NULL and 12 sources
        row = await uow.conn.fetchrow('SELECT * FROM jobs WHERE id = $1', initial_job_id)
        assert row is not None
        assert row['coalesce_key'] is None
        assert len(row['payload']['source_ids']) == 12

        # 4. Source 13: with same coalesce_key, creates a brand new job because previous is sealed
        res13 = await uow.jobs.upsert_coalesced(
            NewExtractJob(
                user_id=user_id,
                session_id='sess_1',
                source_id=SourceId(uuid4()),
                idempotency_key=f'idem_13_{uuid4()}',
                coalesce_key=coalesce_key,
            ),
            debounce_s=15.0,
        )
        assert res13.job_id != initial_job_id
        assert res13.created is True
        assert res13.sealed is False


async def test_q01_concurrent_worker_claiming_skip_locked(
    create_test_user: Callable[[UserId | None], Coroutine[Any, Any, UserId]],
    worker_uow_factory: PostgresUnitOfWorkFactory,
) -> None:
    """Q-01: Concurrent workers claim disjoint batches of queued jobs via FOR UPDATE SKIP LOCKED."""
    user_id = await create_test_user(None)
    now = datetime.now(UTC)

    # Drain any existing queued jobs from prior tests
    async with worker_uow_factory.system() as uow:
        while await uow.jobs.claim(worker_id=uuid4(), limit=50, lease_s=1):
            pass

    # Enqueue 8 distinct queued jobs
    enqueued_job_ids: list[JobId] = []
    async with worker_uow_factory.for_user(user_id) as uow:
        for i in range(8):
            jid = await uow.jobs.enqueue(
                NewJob(
                    user_id=user_id,
                    type=JobType.EXTRACT,
                    payload={'index': i},
                    idempotency_key=f'claim_test_{i}_{uuid4()}',
                    run_after=now - timedelta(seconds=10),
                )
            )
            enqueued_job_ids.append(jid)
    assert len(enqueued_job_ids) == 8

    worker_a = uuid4()
    worker_b = uuid4()

    async def _claim(worker_id: Any) -> list[JobId]:
        async with worker_uow_factory.system() as uow:
            claimed = await uow.jobs.claim(worker_id=worker_id, limit=4, lease_s=60)
            return [j.id for j in claimed]

    # Two concurrent worker claims
    claimed_a, claimed_b = await asyncio.gather(_claim(worker_a), _claim(worker_b))

    assert len(claimed_a) == 4
    assert len(claimed_b) == 4

    # Verify completely disjoint claim sets
    set_a = set(claimed_a)
    set_b = set(claimed_b)
    assert len(set_a.intersection(set_b)) == 0
    assert set_a.union(set_b) == set(enqueued_job_ids)

    # Verify status in database
    async with worker_uow_factory.system() as uow:
        rows_a = await uow.conn.fetch(
            'SELECT * FROM jobs WHERE id = any($1::uuid[])',
            list(claimed_a),
        )
        for r in rows_a:
            assert r['status'] == 'running'
            assert r['worker_id'] == worker_a
            assert r['attempts'] == 1

        rows_b = await uow.conn.fetch(
            'SELECT * FROM jobs WHERE id = any($1::uuid[])',
            list(claimed_b),
        )
        for r in rows_b:
            assert r['status'] == 'running'
            assert r['worker_id'] == worker_b
            assert r['attempts'] == 1

        # Claiming again returns empty list since no queued jobs remain
        extra = await uow.jobs.claim(worker_id=uuid4(), limit=4, lease_s=60)
        assert extra == []


async def test_q02_q03_lease_token_fencing_and_lifecycle(
    create_test_user: Callable[[UserId | None], Coroutine[Any, Any, UserId]],
    worker_uow_factory: PostgresUnitOfWorkFactory,
) -> None:
    """Q-02 / Q-03: Lease heartbeat, complete, and fencing on stale worker tokens."""
    user_id = await create_test_user(None)
    now = datetime.now(UTC)

    # Enqueue a job
    async with worker_uow_factory.for_user(user_id) as uow:
        job_id = await uow.jobs.enqueue(
            NewJob(
                user_id=user_id,
                type=JobType.EXTRACT,
                payload={'task': 'fence_test'},
                idempotency_key=f'fence_test_{uuid4()}',
                run_after=now - timedelta(seconds=5),
            )
        )

    worker_1 = uuid4()
    worker_2 = uuid4()

    # Worker 1 claims job
    async with worker_uow_factory.system() as uow:
        claimed = await uow.jobs.claim(worker_id=worker_1, limit=1, lease_s=60)
        assert len(claimed) == 1
        assert claimed[0].id == job_id
        assert claimed[0].status == JobStatus.RUNNING

    # Worker 1 heartbeat succeeds
    async with worker_uow_factory.system() as uow:
        hb_ok = await uow.jobs.heartbeat(job_id, worker_1, lease_s=60)
        assert hb_ok is True

    # Worker 2 attempts heartbeat -> raises LeaseLost
    async with worker_uow_factory.system() as uow:
        with pytest.raises(LeaseLost):
            await uow.jobs.heartbeat(job_id, worker_2, lease_s=60)

    # Worker 2 attempts to complete -> raises LeaseLost
    async with worker_uow_factory.system() as uow:
        with pytest.raises(LeaseLost):
            await uow.jobs.complete(job_id, worker_2, now=now)

    # Worker 1 completes job -> succeeds
    async with worker_uow_factory.system() as uow:
        done_ok = await uow.jobs.complete(job_id, worker_1, now=now)
        assert done_ok is True

    # Check database state: status is 'done'
    async with worker_uow_factory.system() as uow:
        job_db = await uow.jobs.get(user_id, job_id)
        assert job_db is not None
        assert job_db.status == JobStatus.DONE
        assert job_db.worker_id is None
        assert job_db.lease_expires_at is None

    # After completion, Worker 1 heartbeat raises LeaseLost
    async with worker_uow_factory.system() as uow:
        with pytest.raises(LeaseLost):
            await uow.jobs.heartbeat(job_id, worker_1, lease_s=60)


async def test_q03_fail_and_retry_accounting(
    create_test_user: Callable[[UserId | None], Coroutine[Any, Any, UserId]],
    worker_uow_factory: PostgresUnitOfWorkFactory,
) -> None:
    """Q-03: Verify retry scheduling and transition to 'dead' when max_attempts exceeded."""
    user_id = await create_test_user(None)
    now = datetime.now(UTC)

    # Enqueue job with max_attempts = 2
    async with worker_uow_factory.for_user(user_id) as uow:
        job_id = await uow.jobs.enqueue(
            NewJob(
                user_id=user_id,
                type=JobType.EXTRACT,
                payload={'action': 'retry_test'},
                idempotency_key=f'retry_job_{uuid4()}',
                run_after=now - timedelta(seconds=1),
                max_attempts=2,
            )
        )

    worker_id = uuid4()

    # Attempt 1: claim and retry
    async with worker_uow_factory.system() as uow:
        claimed = await uow.jobs.claim(worker_id=worker_id, limit=1, lease_s=60)
        assert len(claimed) == 1
        assert claimed[0].attempts == 1

        retry_ok = await uow.jobs.retry(
            job_id,
            worker_id,
            run_after=now - timedelta(seconds=1),  # immediately retryable
            err=ErrorInfo(message='Transient network timeout', code='E-TRANS-01'),
            count_attempt=True,
        )
        assert retry_ok is True

    # Check job is back to queued with error info recorded
    async with worker_uow_factory.system() as uow:
        j1 = await uow.jobs.get(user_id, job_id)
        assert j1 is not None
        assert j1.status == JobStatus.QUEUED
        assert j1.attempts == 1
        assert j1.last_error == 'Transient network timeout'
        row1 = await uow.conn.fetchrow('SELECT error_code FROM jobs WHERE id = $1', job_id)
        assert row1 is not None and row1['error_code'] == 'E-TRANS-01'

    # Attempt 2: claim again and fail (attempts=2 >= max_attempts=2)
    async with worker_uow_factory.system() as uow:
        claimed2 = await uow.jobs.claim(worker_id=worker_id, limit=1, lease_s=60)
        assert len(claimed2) == 1
        assert claimed2[0].attempts == 2

        retry_ok2 = await uow.jobs.retry(
            job_id,
            worker_id,
            run_after=now + timedelta(seconds=60),
            err=ErrorInfo(message='Fatal parsing failure', code='E-FATAL-01'),
            count_attempt=True,
        )
        assert retry_ok2 is True

    # Check job transitioned to 'dead'
    async with worker_uow_factory.system() as uow:
        j2 = await uow.jobs.get(user_id, job_id)
        assert j2 is not None
        assert j2.status == JobStatus.DEAD
        assert j2.attempts == 2
        assert j2.last_error == 'Fatal parsing failure'
        row2 = await uow.conn.fetchrow(
            'SELECT error_code, finished_at FROM jobs WHERE id = $1', job_id
        )
        assert row2 is not None
        assert row2['error_code'] == 'E-FATAL-01'
        assert row2['finished_at'] is not None
