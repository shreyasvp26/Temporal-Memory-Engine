"""Worker runner and entrypoint for background jobs per docs/LLD.md Section 8.2 and 10."""

import asyncio
import contextlib
import logging
import random
from collections.abc import Mapping
from datetime import timedelta
from typing import Any
from uuid import UUID, uuid4

from memory_engine.config import WorkerSettings
from memory_engine.domain.clock import Clock, SystemClock
from memory_engine.domain.enums import JobType
from memory_engine.domain.errors import LeaseLost
from memory_engine.domain.models import Job
from memory_engine.ingest.service import IngestService
from memory_engine.ports.repositories import ErrorInfo
from memory_engine.ports.uow import UnitOfWorkFactory

_logger = logging.getLogger(__name__)


class WorkerRunner:
    """Worker job runner polling, heartbeating, executing, and cleaning up background jobs."""

    def __init__(
        self,
        uows: UnitOfWorkFactory,
        service: IngestService,
        clock: Clock | None = None,
        cfg: WorkerSettings | None = None,
        *,
        worker_id: UUID | None = None,
        concurrency: int = 2,
        lease_s: int = 120,
        heartbeat_interval_s: float = 30.0,
        max_attempts: int = 3,
        reaper_interval_s: float = 30.0,
        tombstone_retention_days: int = 30,
        handlers: Mapping[JobType, Any] | None = None,
    ) -> None:
        self.uows = uows
        self.service = service
        self.clock = clock or SystemClock()
        self.cfg = cfg or WorkerSettings()

        self.worker_id = worker_id or uuid4()
        self.concurrency = (
            getattr(self.cfg, 'concurrency', concurrency) if cfg is not None else concurrency
        )
        self.lease_s = getattr(self.cfg, 'lease_s', lease_s) if cfg is not None else lease_s
        self.heartbeat_interval_s = (
            float(getattr(self.cfg, 'heartbeat_s', heartbeat_interval_s))
            if cfg is not None
            else heartbeat_interval_s
        )
        self.max_attempts = (
            getattr(self.cfg, 'max_attempts', max_attempts) if cfg is not None else max_attempts
        )
        self.reaper_interval_s = reaper_interval_s
        self.tombstone_retention_days = tombstone_retention_days
        self.handlers = dict(handlers) if handlers is not None else {}

        self._last_maintenance = self.clock.now() - timedelta(seconds=self.reaper_interval_s + 1)

    def calculate_backoff(self, attempts: int) -> float:
        """Calculate exponential backoff: min(300, 5 * 2^attempts) + jitter."""
        base = min(300.0, 5.0 * (2.0 ** max(0, attempts)))
        jitter = random.uniform(0.0, 5.0)
        return base + jitter

    async def claim_batch(self, limit: int) -> list[Job]:
        """Claim a batch of queued jobs for this worker."""
        if limit <= 0:
            return []
        async with self.uows.begin(None) as uow:
            return await uow.jobs.claim(
                worker_id=self.worker_id,
                limit=limit,
                lease_s=self.lease_s,
            )

    async def _heartbeat_loop(self, job: Job, job_task: asyncio.Task[None]) -> None:
        """Heartbeat every heartbeat_interval_s; cancel job_task on lease loss."""
        try:
            while True:
                await asyncio.sleep(self.heartbeat_interval_s)
                alive = False
                try:
                    async with self.uows.begin(job.user_id) as uow:
                        alive = await uow.jobs.heartbeat(job.id, self.worker_id, self.lease_s)
                except LeaseLost:
                    alive = False
                except Exception as e:
                    _logger.warning('Heartbeat error for job %s: %s', job.id, e)
                    alive = False

                if not alive:
                    _logger.warning('Fenced out of job %s: cancelling handler', job.id)
                    job_task.cancel()
                    break
        except asyncio.CancelledError:
            pass

    async def _run_job_handler(self, job: Job) -> None:
        """Execute the appropriate handler for the claimed job."""
        if job.type == JobType.EXTRACT:
            await self.service.run_extract_job(job, worker_id=self.worker_id)
        elif job.type in self.handlers:
            handler = self.handlers[job.type]
            if hasattr(handler, 'handle'):
                await handler.handle(job, worker_id=self.worker_id)
            elif callable(handler):
                await handler(job, self.worker_id)
            else:
                raise ValueError(f'Unsupported handler for job type {job.type}')
        else:
            raise ValueError(f'No handler registered for job type {job.type}')

    async def _handle_failure(self, job: Job, exc: Exception) -> None:
        """Record job failure: schedule retry with backoff or kill if max attempts reached."""
        new_attempts = job.attempts + 1
        err_info = ErrorInfo(message=str(exc), code=exc.__class__.__name__)

        if new_attempts >= self.max_attempts:
            _logger.warning(
                'Job %s failed permanently after %d attempts; killing',
                job.id,
                new_attempts,
            )
            async with self.uows.begin(job.user_id) as uow:
                await uow.jobs.kill(job.id, self.worker_id, err_info)
        else:
            backoff_s = self.calculate_backoff(new_attempts)
            run_after = self.clock.now() + timedelta(seconds=backoff_s)
            _logger.info(
                'Job %s failed (attempt %d/%d); retrying at %s',
                job.id,
                new_attempts,
                self.max_attempts,
                run_after.isoformat(),
            )
            async with self.uows.begin(job.user_id) as uow:
                await uow.jobs.retry(
                    job.id,
                    self.worker_id,
                    run_after=run_after,
                    err=err_info,
                    count_attempt=True,
                )

    async def _process_job(self, job: Job) -> None:
        """Process a single claimed job with concurrent heartbeat fencing."""
        job_task: asyncio.Task[None] = asyncio.create_task(self._run_job_handler(job))
        heartbeat_task: asyncio.Task[None] = asyncio.create_task(
            self._heartbeat_loop(job, job_task)
        )

        try:
            await job_task
        except asyncio.CancelledError:
            _logger.warning('Job %s was cancelled (fenced out or worker stopping)', job.id)
        except LeaseLost:
            _logger.warning('Job %s lease lost during execution; fencing out', job.id)
        except Exception as exc:
            await self._handle_failure(job, exc)
        finally:
            heartbeat_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await heartbeat_task

    async def _run_periodic_maintenance(self) -> None:
        """Execute periodic reap_expired and hard_delete_expired_tombstones."""
        now = self.clock.now()
        if (now - self._last_maintenance).total_seconds() < self.reaper_interval_s:
            return

        self._last_maintenance = now
        try:
            async with self.uows.begin(None) as uow:
                if hasattr(uow.jobs, 'reap_expired'):
                    await uow.jobs.reap_expired(now)
                if hasattr(uow.facts, 'hard_delete_expired_tombstones'):
                    horizon = now - timedelta(days=self.tombstone_retention_days)
                    await uow.facts.hard_delete_expired_tombstones(horizon)
        except Exception as e:
            _logger.warning('Worker maintenance cycle error: %s', e)

    async def run_once(self) -> int:
        """Run a single iteration of maintenance and concurrent job claims.

        Returns the number of jobs claimed and processed.
        """
        await self._run_periodic_maintenance()
        jobs = await self.claim_batch(self.concurrency)
        if not jobs:
            return 0

        tasks = [self._process_job(j) for j in jobs]
        await asyncio.gather(*tasks, return_exceptions=True)
        return len(jobs)

    async def run_forever(self, stop_event: asyncio.Event | None = None) -> None:
        """Poll and execute background jobs until stop_event is set."""
        stop = stop_event or asyncio.Event()
        while not stop.is_set():
            try:
                processed = await self.run_once()
                if processed == 0:
                    with contextlib.suppress(asyncio.TimeoutError):
                        await asyncio.wait_for(stop.wait(), timeout=1.0)
            except asyncio.CancelledError:
                break
            except Exception as e:
                _logger.exception('Unexpected error in WorkerRunner poll loop: %s', e)
                await asyncio.sleep(1.0)


async def main() -> None:
    """Worker process entrypoint."""
    logging.basicConfig(level=logging.INFO)
    _logger.info('Starting Temporal Memory Engine worker runner...')
    from memory_engine.config import Settings
    from memory_engine.db.pool import create_worker_pool
    from memory_engine.db.uow import PostgresUnitOfWorkFactory
    from memory_engine.domain.clock import SystemClock
    from memory_engine.ingest.canonicalizer import Canonicalizer

    settings = Settings()
    db_url = settings.database_url_worker or settings.database_url_api
    if not db_url:
        _logger.warning('No worker database URL configured (SB_DATABASE_URL_WORKER). Exiting.')
        return

    _logger.info('Connecting worker to database...')
    pool = await create_worker_pool(db_url)
    try:
        clock = SystemClock()
        uows = PostgresUnitOfWorkFactory(pool)
        service = IngestService(
            uows=uows,
            clock=clock,
            canonicalizer=Canonicalizer(),
            cfg=settings.ingest,
        )
        runner = WorkerRunner(
            uows=uows,
            service=service,
            clock=clock,
            cfg=settings.worker,
        )
        _logger.info('Worker runner started with concurrency=%d', runner.concurrency)
        await runner.run_forever()
    finally:
        await pool.close()
        _logger.info('Worker runner stopped.')


if __name__ == '__main__':
    asyncio.run(main())
