"""PostgreSQL Job repository implementation."""

import json
from datetime import datetime
from typing import Any
from uuid import UUID

from memory_engine.db.mappers import map_job
from memory_engine.db.pool import wrap_db_errors
from memory_engine.db.sql import (
    Q01_CLAIM_JOBS,
    Q02_HEARTBEAT_JOB,
    Q03_COMPLETE_JOB,
    Q03_FAIL_JOB,
    Q04_UPSERT_COALESCED_EXTRACT_JOB,
    Q17_REAP_EXPIRED_LEASES,
)
from memory_engine.domain.enums import JobStatus
from memory_engine.domain.errors import InvariantViolation, LeaseLost
from memory_engine.domain.ids import JobId, UserId
from memory_engine.domain.models import Job
from memory_engine.ports.repositories import (
    ErrorInfo,
    JobRepository,
    NewExtractJob,
    NewJob,
    UpsertResult,
)


def _parse_affected_rows(res: Any) -> int:
    """Parse number of affected rows from command tag or result."""
    if isinstance(res, int):
        return res
    if not isinstance(res, str):
        return 0
    parts = res.strip().split()
    if not parts:
        return 0
    try:
        return int(parts[-1])
    except ValueError:
        return 0


class PostgresJobRepository(JobRepository):
    """PostgreSQL implementation of JobRepository port."""

    def __init__(self, conn_or_uow: Any) -> None:
        self._conn_source = conn_or_uow

    @property
    def conn(self) -> Any:
        """Resolve database connection from connection or UnitOfWork."""
        if hasattr(self._conn_source, 'conn'):
            return self._conn_source.conn
        return self._conn_source

    async def upsert_coalesced(
        self,
        j: NewExtractJob,
        debounce_s: float,
    ) -> UpsertResult:
        """Upsert a coalesced extract job, sealing at 12 sources (Q-04)."""
        async with wrap_db_errors():
            row = await self.conn.fetchrow(
                Q04_UPSERT_COALESCED_EXTRACT_JOB,
                j.user_id,
                j.session_id,
                str(j.source_id),
                j.idempotency_key,
                j.coalesce_key,
                float(debounce_s),
            )
        if row is None:
            raise InvariantViolation('Q-04 upsert returned no row')
        job_id = JobId(row['id'])
        created = bool(row['inserted'])
        sealed = bool(row.get('sealed', False))
        return UpsertResult(job_id=job_id, created=created, sealed=sealed)

    async def enqueue(self, j: NewJob) -> JobId:
        """Enqueue a background job idempotently on idempotency_key."""
        sql = """\
insert into jobs (user_id, type, payload, idempotency_key, coalesce_key, run_after, max_attempts)
values ($1, $2, $3::jsonb, $4, $5, coalesce($6, now()), $7)
on conflict (idempotency_key) do nothing
returning id;"""
        type_val = j.type.value if hasattr(j.type, 'value') else str(j.type)
        payload_json = json.dumps(j.payload)
        async with wrap_db_errors():
            row = await self.conn.fetchrow(
                sql,
                j.user_id,
                type_val,
                payload_json,
                j.idempotency_key,
                j.coalesce_key,
                j.run_after,
                j.max_attempts,
            )
            if row is not None and row.get('id') is not None:
                return JobId(row['id'])

            existing = await self.conn.fetchrow(
                'select id from jobs where idempotency_key = $1;',
                j.idempotency_key,
            )
            if existing is not None:
                return JobId(existing['id'])
            raise InvariantViolation(
                f'Failed to enqueue or retrieve job for key {j.idempotency_key}'
            )

    async def claim(
        self,
        worker_id: UUID,
        limit: int,
        lease_s: int,
    ) -> list[Job]:
        """Claim a batch of queued jobs using SKIP LOCKED (Q-01)."""
        async with wrap_db_errors():
            rows = await self.conn.fetch(Q01_CLAIM_JOBS, worker_id, limit, lease_s)
        return [map_job(r) for r in rows]

    async def claim_batch(
        self,
        worker_id: UUID,
        limit: int,
        lease_s: int,
    ) -> list[Job]:
        """Convenience alias for claim (Q-01)."""
        return await self.claim(worker_id, limit, lease_s)

    async def heartbeat(
        self,
        job_id: JobId,
        worker_id: UUID,
        lease_s: int,
        *,
        lease_token: UUID | None = None,
    ) -> bool:
        """Extend lease expiration for a running job (Q-02). Raises LeaseLost if fenced out."""
        effective_worker = lease_token if lease_token is not None else worker_id
        async with wrap_db_errors():
            res = await self.conn.execute(Q02_HEARTBEAT_JOB, job_id, effective_worker, lease_s)
        count = _parse_affected_rows(res)
        if count == 0:
            raise LeaseLost(f'Lease lost for job {job_id} (worker {effective_worker})')
        return True

    async def complete(
        self,
        job_id: JobId,
        worker_id: UUID,
        now: datetime,
        *,
        lease_token: UUID | None = None,
    ) -> bool:
        """Mark a running job as done (Q-03). Raises LeaseLost if fenced out."""
        effective_worker = lease_token if lease_token is not None else worker_id
        async with wrap_db_errors():
            res = await self.conn.execute(Q03_COMPLETE_JOB, job_id, effective_worker, now)
        count = _parse_affected_rows(res)
        if count == 0:
            raise LeaseLost(f'Lease lost for job {job_id} (worker {effective_worker})')
        return True

    async def retry(
        self,
        job_id: JobId,
        worker_id: UUID,
        run_after: datetime,
        err: ErrorInfo,
        *,
        count_attempt: bool = True,
        lease_token: UUID | None = None,
    ) -> bool:
        """Schedule a job retry or transition to dead if attempts exceeded (Q-03)."""
        effective_worker = lease_token if lease_token is not None else worker_id
        async with wrap_db_errors():
            res = await self.conn.execute(
                Q03_FAIL_JOB,
                job_id,
                effective_worker,
                run_after,
                err.last_error,
                err.error_code,
            )
        return _parse_affected_rows(res) > 0

    async def fail(
        self,
        job_id: JobId,
        worker_id: UUID,
        err: ErrorInfo,
        run_after: datetime,
        *,
        lease_token: UUID | None = None,
    ) -> bool:
        """Fail/retry a running job (Q-03)."""
        return await self.retry(
            job_id,
            worker_id,
            run_after,
            err,
            lease_token=lease_token,
        )

    async def kill(
        self,
        job_id: JobId,
        worker_id: UUID,
        err: ErrorInfo,
        *,
        lease_token: UUID | None = None,
    ) -> bool:
        """Immediately transition a running job to dead."""
        effective_worker = lease_token if lease_token is not None else worker_id
        sql = """\
update jobs set status = 'dead', worker_id = null, lease_expires_at = null,
       last_error = $3, error_code = $4, finished_at = now()
where id = $1 and worker_id = $2 and status = 'running';"""
        async with wrap_db_errors():
            res = await self.conn.execute(
                sql,
                job_id,
                effective_worker,
                err.last_error,
                err.error_code,
            )
        return _parse_affected_rows(res) > 0

    async def reap_expired(self, now: datetime) -> int:
        """Reap running jobs with expired leases back to queued or dead (Q-17)."""
        async with wrap_db_errors():
            res = await self.conn.execute(Q17_REAP_EXPIRED_LEASES)
        return _parse_affected_rows(res)

    async def get(self, user_id: UserId, job_id: JobId) -> Job | None:
        """Fetch a job by ID belonging to a specific user."""
        async with wrap_db_errors():
            row = await self.conn.fetchrow(
                'select * from jobs where user_id = $1 and id = $2;',
                user_id,
                job_id,
            )
        return map_job(row) if row is not None else None

    async def get_by_id(self, job_id: JobId) -> Job | None:
        """Fetch a job by ID without user scope (worker/admin context)."""
        async with wrap_db_errors():
            row = await self.conn.fetchrow('select * from jobs where id = $1;', job_id)
        return map_job(row) if row is not None else None

    async def list_for_user(
        self,
        user_id: UserId,
        status: JobStatus | None = None,
        limit: int = 50,
    ) -> list[Job]:
        """List recent jobs for a user, optionally filtered by status."""
        status_val: str | None = None
        if status is not None:
            status_val = status.value if hasattr(status, 'value') else str(status)
        async with wrap_db_errors():
            rows = await self.conn.fetch(
                'select * from jobs where user_id = $1 and ($2::text is null or status = $2) '
                'order by created_at desc limit $3;',
                user_id,
                status_val,
                limit,
            )
        return [map_job(r) for r in rows]

    async def requeue_dead(self, user_id: UserId, job_id: JobId) -> bool:
        """Requeue a dead job for manual retry per Section 10.5."""
        sql = """\
update jobs set status = 'queued', attempts = 0, deferrals = 0, run_after = now(),
       worker_id = null, lease_expires_at = null, finished_at = null
where user_id = $1 and id = $2 and status = 'dead';"""
        async with wrap_db_errors():
            res = await self.conn.execute(sql, user_id, job_id)
        return _parse_affected_rows(res) > 0

    async def user_has_open_extract(self, user_id: UserId) -> bool:
        """Check if user has an active or queued extract job."""
        sql = """\
select 1 from jobs where user_id = $1 and type = 'extract'
  and status in ('queued', 'running') limit 1;"""
        async with wrap_db_errors():
            row = await self.conn.fetchrow(sql, user_id)
        return row is not None
