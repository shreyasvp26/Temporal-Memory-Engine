"""Unit tests for IngestService, WorkerRunner, and end-to-end ingestion pipeline."""

import asyncio
import builtins
from collections.abc import AsyncIterator, Sequence
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from typing import Any, TypeVar
from uuid import UUID, uuid4

import pytest
from memory_engine.domain.clock import FixedClock
from memory_engine.domain.enums import (
    Cardinality,
    DecayClass,
    EntityType,
    FactOrigin,
    FactStatus,
    JobStatus,
    JobType,
    ObjectKind,
    RelationType,
    Role,
    SourceKind,
    TimePrecision,
)
from memory_engine.domain.ids import EntityId, FactId, JobId, SourceId, UserId
from memory_engine.domain.models import (
    Candidate,
    Entity,
    ExtractedFact,
    ExtractionResult,
    Fact,
    FactEvidence,
    FactRelation,
    Job,
    Predicate,
    Source,
)
from memory_engine.ingest.canonicalizer import Canonicalizer
from memory_engine.ingest.conflict import ConflictResolver, default_registry
from memory_engine.ingest.entity_linker import EntityLinker, norm_name
from memory_engine.ingest.extractor import FactExtractor
from memory_engine.ingest.plan_executor import PlanExecutor
from memory_engine.ingest.service import (
    IngestService,
)
from memory_engine.ingest.time_resolver import TimeResolver
from memory_engine.ports.embed import Embedder
from memory_engine.ports.llm import LLMGateway, LLMRequest, LLMResponse
from memory_engine.ports.repositories import (
    ErrorInfo,
    InsertedSource,
    NewExtractJob,
    NewSource,
    PredicateRegistry,
    SourceWindow,
    UpsertResult,
)

from apps.worker.main import WorkerRunner

T = TypeVar('T')

# -----------------------------------------------------------------------------
# In-Memory Shared State and Fake Repositories
# -----------------------------------------------------------------------------


class SharedTxTracker:
    """Tracks active write transactions across fake UoWs."""

    def __init__(self) -> None:
        self.active_write_txs: set[int] = set()

    def enter(self, uow_id: int, readonly: bool) -> None:
        if not readonly:
            self.active_write_txs.add(uow_id)

    def exit(self, uow_id: int) -> None:
        self.active_write_txs.discard(uow_id)

    def is_write_active(self) -> bool:
        return len(self.active_write_txs) > 0


class FakeSourceRepository:
    def __init__(self) -> None:
        self.sources: dict[SourceId, Source] = {}
        self.sources_by_hash: dict[tuple[UserId, str], Source] = {}
        self.embeddings: dict[SourceId, list[float]] = {}

    async def insert_many(self, items: Sequence[NewSource]) -> list[InsertedSource]:
        results: list[InsertedSource] = []
        for it in items:
            key = (it.user_id, it.content_hash)
            if key in self.sources_by_hash:
                results.append(InsertedSource(id=self.sources_by_hash[key].id, created=False))
            else:
                sid = it.id or SourceId(uuid4())
                s = Source(
                    id=sid,
                    user_id=it.user_id,
                    kind=it.kind,
                    role=it.role,
                    session_id=it.session_id,
                    turn_index=it.turn_index,
                    content=it.content,
                    source_ts=it.source_ts,
                    content_hash=it.content_hash,
                )
                self.sources[sid] = s
                self.sources_by_hash[key] = s
                results.append(InsertedSource(id=sid, created=True))
        return results

    async def get_many(self, user_id: UserId, ids: Sequence[SourceId]) -> list[Source]:
        return [self.sources[sid] for sid in ids if sid in self.sources]

    async def window(
        self,
        user_id: UserId,
        session_id: str,
        target_ids: Sequence[SourceId],
        context_turns: int,
    ) -> SourceWindow:
        targets: list[Source] = [
            self.sources[tid]
            for tid in target_ids
            if tid in self.sources and self.sources[tid].user_id == user_id
        ]
        min_target_turn = min((s.turn_index for s in targets), default=0)
        session_sources = [
            s
            for s in self.sources.values()
            if s.user_id == user_id
            and s.session_id == session_id
            and s.turn_index < min_target_turn
            and s.deleted_at is None
        ]
        context = sorted(session_sources, key=lambda s: s.turn_index)[-context_turns:]
        return SourceWindow(targets=targets, context=context)

    async def set_embeddings(
        self,
        user_id: UserId,
        items: Sequence[tuple[SourceId, list[float]]],
        model_id: str,
    ) -> None:
        for sid, emb in items:
            self.embeddings[sid] = list(emb)

    async def soft_delete(self, user_id: UserId, ids: Sequence[SourceId], at: datetime) -> None:
        for sid in ids:
            if sid in self.sources:
                self.sources[sid] = replace(self.sources[sid], deleted_at=at)

    async def hard_delete(self, user_id: UserId, ids: Sequence[SourceId]) -> int:
        count = 0
        for sid in ids:
            if sid in self.sources:
                del self.sources[sid]
                count += 1
        return count


class FakeJobRepository:
    def __init__(self, clock: FixedClock) -> None:
        self.clock = clock
        self.jobs: dict[JobId, Job] = {}
        self.coalesced: dict[str, JobId] = {}
        self.reaped_count: int = 0
        self.fence_worker: bool = False

    async def upsert_coalesced(self, j: NewExtractJob, debounce_s: float) -> UpsertResult:
        coalesce_key = j.coalesce_key
        if coalesce_key in self.coalesced:
            existing_id = self.coalesced[coalesce_key]
            existing_job = self.jobs[existing_id]
            if existing_job.status == JobStatus.QUEUED:
                source_ids = list(existing_job.payload.get('source_ids', []))
                source_ids.append(str(j.source_id))
                new_payload = dict(existing_job.payload)
                new_payload['source_ids'] = source_ids
                updated_job = replace(
                    existing_job,
                    payload=new_payload,
                    run_after=self.clock.now() + timedelta(seconds=debounce_s),
                )
                self.jobs[existing_id] = updated_job
                return UpsertResult(job_id=existing_id, created=False)

        new_id = JobId(uuid4())
        job = Job(
            id=new_id,
            user_id=j.user_id,
            type=JobType.EXTRACT,
            status=JobStatus.QUEUED,
            payload={'session_id': j.session_id, 'source_ids': [str(j.source_id)]},
            attempts=0,
            max_attempts=3,
            deferrals=0,
            run_after=self.clock.now() + timedelta(seconds=debounce_s),
            worker_id=None,
            lease_expires_at=None,
            coalesce_key=coalesce_key,
            idempotency_key=j.idempotency_key,
            last_error=None,
        )
        self.jobs[new_id] = job
        if coalesce_key:
            self.coalesced[coalesce_key] = new_id
        return UpsertResult(job_id=new_id, created=True)

    async def enqueue(self, j: Any) -> JobId:
        new_id = JobId(uuid4())
        return new_id

    async def claim(self, worker_id: UUID, limit: int, lease_s: int) -> list[Job]:
        claimed: list[Job] = []
        now = self.clock.now()
        for jid, job in list(self.jobs.items()):
            if len(claimed) >= limit:
                break
            if job.status == JobStatus.QUEUED and (job.run_after is None or job.run_after <= now):
                upd = replace(
                    job,
                    status=JobStatus.RUNNING,
                    worker_id=worker_id,
                    lease_expires_at=now + timedelta(seconds=lease_s),
                )
                self.jobs[jid] = upd
                claimed.append(upd)
        return claimed

    async def heartbeat(self, job_id: JobId, worker_id: UUID, lease_s: int) -> bool:
        if self.fence_worker:
            return False
        job = self.jobs.get(job_id)
        if job is None or job.status != JobStatus.RUNNING or job.worker_id != worker_id:
            return False
        self.jobs[job_id] = replace(
            job,
            lease_expires_at=self.clock.now() + timedelta(seconds=lease_s),
        )
        return True

    async def complete(self, job_id: JobId, worker_id: UUID, now: datetime) -> bool:
        job = self.jobs.get(job_id)
        if job is None or job.status != JobStatus.RUNNING or job.worker_id != worker_id:
            return False
        self.jobs[job_id] = replace(
            job,
            status=JobStatus.DONE,
        )
        return True

    async def retry(
        self,
        job_id: JobId,
        worker_id: UUID,
        run_after: datetime,
        err: ErrorInfo,
        *,
        count_attempt: bool,
    ) -> bool:
        job = self.jobs.get(job_id)
        if job is None:
            return False
        attempts = job.attempts + (1 if count_attempt else 0)
        self.jobs[job_id] = replace(
            job,
            status=JobStatus.QUEUED,
            attempts=attempts,
            run_after=run_after,
            last_error=err.message,
            worker_id=None,
            lease_expires_at=None,
        )
        return True

    async def kill(self, job_id: JobId, worker_id: UUID, err: ErrorInfo) -> bool:
        job = self.jobs.get(job_id)
        if job is None:
            return False
        self.jobs[job_id] = replace(
            job,
            status=JobStatus.DEAD,
            attempts=job.attempts + 1,
            last_error=err.message,
        )
        return True

    async def reap_expired(self, now: datetime) -> int:
        reaped = 0
        for jid, job in list(self.jobs.items()):
            if (
                job.status == JobStatus.RUNNING
                and job.lease_expires_at is not None
                and job.lease_expires_at < now
            ):
                if job.attempts >= job.max_attempts:
                    self.jobs[jid] = replace(job, status=JobStatus.DEAD)
                else:
                    self.jobs[jid] = replace(
                        job,
                        status=JobStatus.QUEUED,
                        worker_id=None,
                        lease_expires_at=None,
                    )
                reaped += 1
        self.reaped_count += reaped
        return reaped

    async def get(self, user_id: UserId, job_id: JobId) -> Job | None:
        return self.jobs.get(job_id)

    async def list_for_user(
        self, user_id: UserId, status: JobStatus | None, limit: int
    ) -> list[Job]:
        return [
            j
            for j in self.jobs.values()
            if j.user_id == user_id and (status is None or j.status == status)
        ][:limit]

    async def requeue_dead(self, user_id: UserId, job_id: JobId) -> bool:
        return True

    async def user_has_open_extract(self, user_id: UserId) -> bool:
        return False


class FakeEntityRepository:
    def __init__(self) -> None:
        self.entities: dict[EntityId, Entity] = {}
        self.self_entities: dict[UserId, Entity] = {}

    async def ensure_self(self, user_id: UserId) -> Entity:
        if user_id in self.self_entities:
            return self.self_entities[user_id]
        eid = EntityId(uuid4())
        ent = Entity(
            id=eid,
            user_id=user_id,
            canonical_name='user',
            type=EntityType.PERSON,
            is_self=True,
        )
        self.entities[eid] = ent
        self.self_entities[user_id] = ent
        return ent

    async def get_or_create_self(self, user_id: UserId) -> Entity:
        return await self.ensure_self(user_id)

    async def find_by_norm_name(
        self,
        user_id: UserId,
        name_or_type: str | EntityType,
        alias_or_type: str | EntityType | None = None,
        entity_type: EntityType | None = None,
    ) -> Entity | None:
        target_name = norm_name(str(name_or_type))
        for ent in self.entities.values():
            if ent.user_id == user_id and norm_name(ent.canonical_name) == target_name:
                return ent
        return None

    async def find_by_alias(
        self, user_id: UserId, type: EntityType, alias_norm: str
    ) -> Entity | None:
        return await self.find_by_norm_name(user_id, alias_norm)

    async def nearest(
        self,
        user_id: UserId,
        type: EntityType,
        embedding: list[float],
        k: int,
    ) -> list[tuple[Entity, float]]:
        return []

    async def search_similar_by_embedding(
        self,
        user_id: UserId,
        type_or_emb: Any,
        embedding: Any = None,
        k: Any = None,
        *,
        min_cosine: float = 0.92,
        limit: int = 3,
    ) -> list[tuple[Entity, float]]:
        return []

    async def create(
        self,
        user_id: UserId,
        name: str,
        type: EntityType,
        embedding: list[float],
    ) -> Entity:
        eid = EntityId(uuid4())
        ent = Entity(
            id=eid,
            user_id=user_id,
            canonical_name=name,
            type=type,
            is_self=False,
        )
        self.entities[eid] = ent
        return ent

    async def add_alias(
        self,
        user_id: UserId,
        entity_id: EntityId,
        type: EntityType,
        alias: str,
        alias_norm: str,
    ) -> None:
        pass

    async def get_many(self, user_id: UserId, ids: Sequence[EntityId]) -> dict[EntityId, Entity]:
        return {eid: self.entities[eid] for eid in ids if eid in self.entities}

    async def alias_index(self, user_id: UserId, limit: int = 5000) -> dict[str, EntityId]:
        return {}


class FakeFactRepository:
    def __init__(self) -> None:
        self.facts: dict[FactId, Fact] = {}
        self.relations: list[FactRelation] = []
        self.evidences: list[FactEvidence] = []
        self.tombstones_deleted: int = 0

    async def load_timeline(
        self,
        user_id: UserId,
        subject_id: EntityId,
        predicate: str,
        *,
        lock: bool = True,
    ) -> list[Fact]:
        return [
            f
            for f in self.facts.values()
            if f.user_id == user_id
            and f.subject_id == subject_id
            and f.predicate == predicate
            and f.status in (FactStatus.ACTIVE, FactStatus.NEEDS_REVIEW)
        ]

    async def find_for_resolution(
        self,
        user_id: UserId,
        subject_id: EntityId,
        predicate: str,
    ) -> list[Fact]:
        return await self.load_timeline(user_id, subject_id, predicate, lock=True)

    async def insert(
        self,
        c: Candidate,
        *,
        valid_from: datetime,
        valid_to: datetime | None,
        status: FactStatus,
        origin: FactOrigin,
        model_id: str,
        now: datetime,
    ) -> Fact:
        fid = FactId(uuid4())
        f = Fact(
            id=fid,
            user_id=c.user_id,
            subject_id=c.subject_id,
            predicate=c.predicate,
            cardinality=c.cardinality,
            object_entity_id=c.object_entity_id,
            object_value=c.object_value,
            object_norm=c.object_norm,
            fact_text=c.fact_text,
            valid_from=valid_from,
            valid_to=valid_to,
            time_precision=c.time_precision,
            valid_from_explicit=c.valid_from_explicit,
            recorded_at=now,
            invalidated_at=None,
            confidence=c.confidence,
            salience=1.0,
            access_count=0,
            last_accessed_at=None,
            status=status,
            origin=origin,
            event_hash=c.event_hash,
        )
        self.facts[fid] = f
        return f

    async def close(
        self, user_id: UserId, fact_id: FactId, valid_to: datetime, now: datetime
    ) -> None:
        if fact_id in self.facts:
            self.facts[fact_id] = replace(
                self.facts[fact_id],
                valid_to=valid_to,
                invalidated_at=now,
            )

    async def extend_valid_from(
        self, user_id: UserId, fact_id: FactId, valid_from: datetime
    ) -> None:
        if fact_id in self.facts:
            self.facts[fact_id] = replace(self.facts[fact_id], valid_from=valid_from)

    async def set_status(
        self,
        user_id: UserId,
        fact_id: FactId,
        status: FactStatus,
        now: datetime,
    ) -> None:
        if fact_id in self.facts:
            self.facts[fact_id] = replace(
                self.facts[fact_id],
                status=status,
                invalidated_at=now if status == FactStatus.DELETED else None,
            )

    async def bump_confidence(self, user_id: UserId, fact_id: FactId, confidence: float) -> None:
        if fact_id in self.facts:
            cur = self.facts[fact_id].confidence
            self.facts[fact_id] = replace(self.facts[fact_id], confidence=max(cur, confidence))

    async def add_evidence(self, ev: FactEvidence, user_id: UserId) -> None:
        self.evidences.append(ev)

    async def add_relation(self, user_id: UserId, rel: FactRelation, job_id: UUID | None) -> None:
        self.relations.append(rel)

    async def get(self, user_id: UserId, fact_id: FactId) -> Fact | None:
        return self.facts.get(fact_id)

    async def list(self, user_id: UserId, q: Any) -> Any:
        return None

    async def chain(self, user_id: UserId, fact_id: FactId) -> builtins.list[Fact]:
        return []

    async def only_supported_by(
        self, user_id: UserId, source_ids: Sequence[SourceId]
    ) -> builtins.list[FactId]:
        return []

    async def hard_delete(self, user_id: UserId, fact_ids: Sequence[FactId]) -> int:
        del_count = 0
        for fid in fact_ids:
            if fid in self.facts:
                del self.facts[fid]
                del_count += 1
        return del_count

    async def hard_delete_expired_tombstones(self, before: datetime) -> int:
        del_count = 0
        for fid, f in list(self.facts.items()):
            if (
                f.status == FactStatus.DELETED
                and f.invalidated_at is not None
                and f.invalidated_at < before
            ):
                del self.facts[fid]
                del_count += 1
        self.tombstones_deleted += del_count
        return del_count

    async def record_access(
        self, user_id: UserId, fact_ids: Sequence[FactId], at: datetime
    ) -> None:
        pass

    async def decay_candidates(self, user_id: UserId, now: datetime) -> builtins.list[Fact]:
        return []

    async def update_salience(self, user_id: UserId, items: Sequence[tuple[FactId, float]]) -> None:
        pass


class FakeProfileRepository:
    def __init__(self) -> None:
        self.timezones: dict[UserId, str] = {}

    async def ensure(self, user_id: UserId) -> None:
        if user_id not in self.timezones:
            self.timezones[user_id] = 'UTC'

    async def timezone(self, user_id: UserId) -> str:
        return self.timezones.get(user_id, 'UTC')

    async def set_timezone(self, user_id: UserId, tz: str) -> None:
        self.timezones[user_id] = tz

    async def list_user_ids(self) -> list[UserId]:
        return list(self.timezones.keys())


class FakeUnitOfWork:
    def __init__(
        self,
        sources: FakeSourceRepository,
        jobs: FakeJobRepository,
        entities: FakeEntityRepository,
        facts: FakeFactRepository,
        profiles: FakeProfileRepository,
        tx_tracker: SharedTxTracker,
        readonly: bool = False,
    ) -> None:
        self.sources = sources
        self.jobs = jobs
        self.entities = entities
        self.facts = facts
        self.profiles = profiles
        self.tx_tracker = tx_tracker
        self.readonly = readonly

        self.search: Any = None
        self.messages: Any = None
        self.audit: Any = None

        self.is_open: bool = False
        self.locked_user_called: bool = False

    async def lock_user(self, user_id: UserId | None = None) -> None:
        self.locked_user_called = True

    async def __aenter__(self) -> 'FakeUnitOfWork':
        self.is_open = True
        self.tx_tracker.enter(id(self), self.readonly)
        return self

    async def __aexit__(self, *args: Any) -> None:
        self.is_open = False
        self.tx_tracker.exit(id(self))


class FakeUnitOfWorkFactory:
    def __init__(
        self,
        sources: FakeSourceRepository,
        jobs: FakeJobRepository,
        entities: FakeEntityRepository,
        facts: FakeFactRepository,
        profiles: FakeProfileRepository,
        tx_tracker: SharedTxTracker,
    ) -> None:
        self.sources = sources
        self.jobs = jobs
        self.entities = entities
        self.facts = facts
        self.profiles = profiles
        self.tx_tracker = tx_tracker

    def begin(self, user_id: UserId | None, *, readonly: bool = False) -> FakeUnitOfWork:
        return FakeUnitOfWork(
            self.sources,
            self.jobs,
            self.entities,
            self.facts,
            self.profiles,
            self.tx_tracker,
            readonly=readonly,
        )


class DisciplinedFakeEmbedder(Embedder):
    """Embedder verifying ZERO write transactions are active when called."""

    def __init__(self, tx_tracker: SharedTxTracker) -> None:
        self.tx_tracker = tx_tracker
        self.call_count: int = 0
        self.model_id = 'test-embedder-v1'
        self.dim = 4

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        assert not self.tx_tracker.is_write_active(), (
            'VIOLATION: embedder.embed_documents called while a write transaction is active!'
        )
        self.call_count += 1
        return [[0.1, 0.2, 0.3, 0.4] for _ in texts]

    async def embed_queries(self, texts: Sequence[str]) -> list[list[float]]:
        return await self.embed_documents(texts)

    async def embed(self, text: str) -> list[float]:
        assert not self.tx_tracker.is_write_active(), (
            'VIOLATION: embedder.embed called while a write transaction is active!'
        )
        self.call_count += 1
        return [0.1, 0.2, 0.3, 0.4]


class DisciplinedFakeLLMGateway(LLMGateway):
    """LLMGateway verifying ZERO write transactions are active when called."""

    def __init__(
        self,
        tx_tracker: SharedTxTracker,
        canned_extractions: list[ExtractedFact] | None = None,
    ) -> None:
        self.tx_tracker = tx_tracker
        self.canned_extractions = canned_extractions or []
        self.call_count: int = 0

    async def complete_text(self, req: LLMRequest) -> LLMResponse:
        return LLMResponse(
            text='',
            model='test',
            provider='test',
            prompt_tokens=0,
            completion_tokens=0,
            latency_ms=0.0,
        )

    async def complete_json(
        self,
        req: LLMRequest,
        schema: type[T],
    ) -> T:
        assert not self.tx_tracker.is_write_active(), (
            'VIOLATION: LLMGateway.complete_json called while a write transaction is active!'
        )
        self.call_count += 1
        res = ExtractionResult(facts=self.canned_extractions)
        return res  # type: ignore[return-value]

    async def stream_text(self, req: LLMRequest) -> AsyncIterator[str]:
        if False:
            yield ''


class FakePredicateRegistry(PredicateRegistry):
    def __init__(self) -> None:
        self._preds: dict[str, Predicate] = {
            'lives_in': Predicate(
                name='lives_in',
                cardinality=Cardinality.SINGLE,
                decay_class=DecayClass.SLOW,
                object_kind=ObjectKind.LITERAL,
                template='{subject} lives in {object}',
                synonyms=('resides_in',),
            ),
            'plans_to': Predicate(
                name='plans_to',
                cardinality=Cardinality.SINGLE,
                decay_class=DecayClass.NORMAL,
                object_kind=ObjectKind.LITERAL,
                template='{subject} plans to {object}',
                synonyms=(),
            ),
            'related_to': Predicate(
                name='related_to',
                cardinality=Cardinality.MULTI,
                decay_class=DecayClass.NORMAL,
                object_kind=ObjectKind.LITERAL,
                template='{subject} is related to {object}',
                synonyms=(),
            ),
        }

    def lookup(self, name: str) -> Predicate:
        return self._preds.get(name, self._preds['related_to'])

    def resolve(self, name: str) -> Predicate:
        return self.lookup(name)

    def get(self, name: str) -> Predicate:
        return self.lookup(name)

    def all(self) -> Sequence[Predicate]:
        return list(self._preds.values())


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def test_clock() -> FixedClock:
    return FixedClock(datetime(2026, 10, 27, 12, 0, tzinfo=UTC))


@pytest.fixture
def tx_tracker() -> SharedTxTracker:
    return SharedTxTracker()


@pytest.fixture
def fake_repos(
    test_clock: FixedClock, tx_tracker: SharedTxTracker
) -> tuple[
    FakeUnitOfWorkFactory,
    FakeSourceRepository,
    FakeJobRepository,
    FakeEntityRepository,
    FakeFactRepository,
    FakeProfileRepository,
]:
    sources = FakeSourceRepository()
    jobs = FakeJobRepository(test_clock)
    entities = FakeEntityRepository()
    facts = FakeFactRepository()
    profiles = FakeProfileRepository()
    uows = FakeUnitOfWorkFactory(sources, jobs, entities, facts, profiles, tx_tracker)
    return uows, sources, jobs, entities, facts, profiles


# -----------------------------------------------------------------------------
# Tests: T-ING-01, T-ING-02, T-ING-03 (enqueue_turn & accept_note)
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_enqueue_turn_idempotency_t_ing_03(
    fake_repos: Any,
    test_clock: FixedClock,
) -> None:
    """T-ING-03: Repeated send of identical turn returns created=False and job_id=None."""
    uows, sources, jobs, entities, facts, profiles = fake_repos
    service = IngestService(clock=test_clock, debounce_s=15.0)

    user_id = UserId(uuid4())
    client_msg_id = uuid4()
    content = 'I work at DeepMind as a researcher.'

    # First send -> creates source and extract job
    async with uows.begin(user_id) as uow:
        res1 = await service.enqueue_turn(
            uow,
            user_id=user_id,
            session_id='sess-1',
            role=Role.USER,
            content=content,
            client_message_id=client_msg_id,
        )

    assert res1.created is True
    assert res1.job_id is not None
    assert len(sources.sources) == 1

    # Second send with same content and client_message_id -> deduplicated
    async with uows.begin(user_id) as uow:
        res2 = await service.enqueue_turn(
            uow,
            user_id=user_id,
            session_id='sess-1',
            role=Role.USER,
            content=content,
            client_message_id=client_msg_id,
        )

    assert res2.created is False
    assert res2.job_id is None
    assert res2.source_id == res1.source_id
    assert len(sources.sources) == 1
    assert len(jobs.jobs) == 1


@pytest.mark.asyncio
async def test_enqueue_turn_gate_skip_on_questions_t_ing_01(
    fake_repos: Any,
    test_clock: FixedClock,
) -> None:
    """T-ING-01: Interrogative questions are stored as source but skipped by IngestGate."""
    uows, sources, jobs, entities, facts, profiles = fake_repos
    service = IngestService(clock=test_clock)

    user_id = UserId(uuid4())
    question_text = 'Where do I live right now?'

    async with uows.begin(user_id) as uow:
        res = await service.enqueue_turn(
            uow,
            user_id=user_id,
            session_id='sess-2',
            role=Role.USER,
            content=question_text,
        )

    # Source is saved for context, but gate rejects extraction job
    assert res.created is True
    assert res.job_id is None
    assert res.gate_decision is not None
    assert res.gate_decision.ingest is False
    assert res.gate_decision.reason == 'question_only'
    assert len(sources.sources) == 1
    assert len(jobs.jobs) == 0


@pytest.mark.asyncio
async def test_enqueue_turn_coalescing_t_ing_02(
    fake_repos: Any,
    test_clock: FixedClock,
) -> None:
    """T-ING-02: Turns in the same session coalesce into one job within debounce window."""
    uows, sources, jobs, entities, facts, profiles = fake_repos
    service = IngestService(clock=test_clock, debounce_s=15.0)

    user_id = UserId(uuid4())
    session_id = 'sess-coalesce'

    # Turn 1
    async with uows.begin(user_id) as uow:
        res1 = await service.enqueue_turn(
            uow,
            user_id=user_id,
            session_id=session_id,
            role=Role.USER,
            content='I love hiking in the Bavarian Alps.',
        )

    assert res1.created is True
    assert res1.job_id is not None
    initial_job_id = res1.job_id

    # Advance clock within debounce window (5 seconds)
    test_clock.advance(timedelta(seconds=5.0))

    # Turn 2
    async with uows.begin(user_id) as uow:
        res2 = await service.enqueue_turn(
            uow,
            user_id=user_id,
            session_id=session_id,
            role=Role.USER,
            content='My favorite trail is near Garmisch.',
        )

    assert res2.created is True
    assert res2.job_id == initial_job_id
    assert len(jobs.jobs) == 1

    # Job payload has both source IDs coalesced
    coalesced_job = jobs.jobs[initial_job_id]
    assert len(coalesced_job.payload['source_ids']) == 2


@pytest.mark.asyncio
async def test_accept_note_chunking_and_immediate_extract(
    fake_repos: Any,
    test_clock: FixedClock,
) -> None:
    """accept_note chunks text, inserts sources with chunk_index, and enqueues immediate job."""
    uows, sources, jobs, entities, facts, profiles = fake_repos
    service = IngestService(clock=test_clock)

    user_id = UserId(uuid4())
    p1 = 'This is the first paragraph with important design notes. ' * 20
    p2 = 'This is the second paragraph with architectural constraints. ' * 20
    note_content = f'{p1}\n\n{p2}'

    async with uows.begin(user_id) as uow:
        res = await service.accept_note(
            uow,
            user_id=user_id,
            text=note_content,
            title='Architecture Sync',
        )

    assert res.created is True
    assert len(res.source_ids) == 2
    assert res.job_id is not None

    enqueued_job = jobs.jobs[res.job_id]
    # Debounce is 0 for notes -> run_after is immediately now
    assert enqueued_job.run_after == test_clock.now()


# -----------------------------------------------------------------------------
# Tests: Full 4-Phase run_extract_job End-to-End
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_extract_job_four_phases_and_sequential_supersession(
    fake_repos: Any,
    test_clock: FixedClock,
    tx_tracker: SharedTxTracker,
) -> None:
    """Full 4-phase extract job verifying zero DB tx during Phase B, entity linking,

    and sequential in-memory supersession for two facts touching the same (subject, predicate).
    """
    uows, sources, jobs, entities, facts, profiles = fake_repos

    # Two extracted facts touching the same single-valued predicate ('lives_in'):
    # Fact 1: Seattle in 2020
    # Fact 2: Portland in 2023
    canned_facts = [
        ExtractedFact(
            subject='I',
            subject_type=EntityType.PERSON,
            predicate='lives_in',
            object='Seattle',
            object_type='literal',
            valid_from_text='in 2020',
            valid_to_text=None,
            confidence=0.95,
            is_retraction=False,
            evidence_span='I moved to Seattle back in 2020',
            source_turn=1,
        ),
        ExtractedFact(
            subject='I',
            subject_type=EntityType.PERSON,
            predicate='lives_in',
            object='Portland',
            object_type='literal',
            valid_from_text='in 2023',
            valid_to_text=None,
            confidence=0.98,
            is_retraction=False,
            evidence_span='I moved to Portland in 2023',
            source_turn=1,
        ),
    ]

    llm = DisciplinedFakeLLMGateway(tx_tracker, canned_extractions=canned_facts)
    embedder = DisciplinedFakeEmbedder(tx_tracker)
    registry = FakePredicateRegistry()
    canonicalizer = Canonicalizer(registry)
    time_resolver = TimeResolver()
    entity_linker = EntityLinker(embedder=embedder)
    conflict_resolver = ConflictResolver(default_registry())
    plan_executor = PlanExecutor(clock=test_clock)

    extractor = FactExtractor(llm=llm, registry=registry)

    service = IngestService(
        uows=uows,
        clock=test_clock,
        extractor=extractor,
        canonicalizer=canonicalizer,
        time_resolver=time_resolver,
        entity_linker=entity_linker,
        conflict_resolver=conflict_resolver,
        plan_executor=plan_executor,
        embedder=embedder,
    )

    user_id = UserId(uuid4())
    session_id = 'sess-e2e'

    # Pre-populate source in FakeSourceRepository
    turn_content = 'I moved to Seattle back in 2020. Then I moved to Portland in 2023.'
    new_src = NewSource(
        user_id=user_id,
        kind=SourceKind.CHAT,
        role=Role.USER,
        session_id=session_id,
        turn_index=1,
        content=turn_content,
        source_ts=test_clock.now(),
        content_hash='test-hash-e2e',
    )
    async with uows.begin(user_id) as uow:
        ins = await uow.sources.insert_many([new_src])
        source_id = ins[0].id

    # Create extract job
    job_id = JobId(uuid4())
    job = Job(
        id=job_id,
        user_id=user_id,
        type=JobType.EXTRACT,
        status=JobStatus.RUNNING,
        payload={'session_id': session_id, 'source_ids': [str(source_id)]},
        attempts=0,
        max_attempts=3,
        deferrals=0,
        run_after=test_clock.now(),
        worker_id=uuid4(),
        lease_expires_at=test_clock.now() + timedelta(seconds=120),
        coalesce_key=None,
        idempotency_key='extract:test',
        last_error=None,
    )
    jobs.jobs[job_id] = job

    # Execute extract job with worker fencing check
    worker_uuid = job.worker_id
    touched_fact_ids = await service.run_extract_job(job, worker_id=worker_uuid)

    assert len(touched_fact_ids) == 2
    # Verify LLM & Embedder calls occurred (without write transactions!)
    assert llm.call_count == 1
    assert embedder.call_count > 0

    # Verify facts in repository:
    all_facts = list(facts.facts.values())
    assert len(all_facts) == 2

    # Fact 1 (Seattle) should be CLOSED at 2023 (superseded by Fact 2)
    seattle_facts = [f for f in all_facts if f.object_value == 'Seattle']
    assert len(seattle_facts) == 1
    seattle_fact = seattle_facts[0]
    assert seattle_fact.status == FactStatus.ACTIVE
    assert seattle_fact.valid_from == datetime(2020, 1, 1, tzinfo=UTC)
    assert seattle_fact.valid_to == datetime(2023, 1, 1, tzinfo=UTC)

    # Fact 2 (Portland) should be OPEN
    portland_facts = [f for f in all_facts if f.object_value == 'Portland']
    assert len(portland_facts) == 1
    portland_fact = portland_facts[0]
    assert portland_fact.status == FactStatus.ACTIVE
    assert portland_fact.valid_from == datetime(2023, 1, 1, tzinfo=UTC)
    assert portland_fact.valid_to is None

    # Verify relation: Portland supersedes Seattle
    relations = facts.relations
    supersedes_rels = [r for r in relations if r.relation == RelationType.SUPERSEDES]
    assert len(supersedes_rels) == 1
    assert supersedes_rels[0].from_fact_id == portland_fact.id
    assert supersedes_rels[0].to_fact_id == seattle_fact.id

    # Verify job completed in repository
    assert jobs.jobs[job_id].status == JobStatus.DONE


# -----------------------------------------------------------------------------
# Tests: WorkerRunner (Polling, Claiming, Retries, Heartbeat Fencing, Reaper)
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_worker_runner_claim_complete_and_exponential_backoff(
    fake_repos: Any,
    test_clock: FixedClock,
    tx_tracker: SharedTxTracker,
) -> None:
    """WorkerRunner claims jobs, completes them, and applies exponential backoff on retry."""
    uows, sources, jobs, entities, facts, profiles = fake_repos

    # Backoff calculation test: min(300, 5 * 2^attempts) + jitter
    llm = DisciplinedFakeLLMGateway(tx_tracker)
    extractor = FactExtractor(llm=llm)
    service = IngestService(uows=uows, clock=test_clock, extractor=extractor)
    worker = WorkerRunner(uows=uows, service=service, clock=test_clock, max_attempts=3)

    backoff_0 = worker.calculate_backoff(0)
    assert 5.0 <= backoff_0 <= 10.0

    backoff_2 = worker.calculate_backoff(2)
    assert 20.0 <= backoff_2 <= 25.0

    backoff_large = worker.calculate_backoff(10)
    assert 300.0 <= backoff_large <= 305.0

    user_id = UserId(uuid4())

    # Create a dummy queued extract job
    job_id = JobId(uuid4())
    job = Job(
        id=job_id,
        user_id=user_id,
        type=JobType.EXTRACT,
        status=JobStatus.QUEUED,
        payload={'session_id': 'sess-worker', 'source_ids': []},
        attempts=0,
        max_attempts=3,
        deferrals=0,
        run_after=test_clock.now(),
        worker_id=None,
        lease_expires_at=None,
        coalesce_key=None,
        idempotency_key='k1',
        last_error=None,
    )
    jobs.jobs[job_id] = job

    # Process via run_once: empty target turns -> completes noop
    claimed = await worker.run_once()
    assert claimed == 1
    assert jobs.jobs[job_id].status == JobStatus.DONE


@pytest.mark.asyncio
async def test_worker_runner_transient_failure_and_dead_killing(
    fake_repos: Any,
    test_clock: FixedClock,
) -> None:
    """WorkerRunner schedules retry on failure, and kills job when max attempts reached."""
    uows, sources, jobs, entities, facts, profiles = fake_repos

    class FailingService(IngestService):
        async def run_extract_job(self, job: Job, worker_id: UUID | None = None) -> list[FactId]:
            raise RuntimeError('Temporary network timeout to LLM')

    service = FailingService(clock=test_clock)
    worker = WorkerRunner(uows=uows, service=service, clock=test_clock, max_attempts=2)

    user_id = UserId(uuid4())
    job_id = JobId(uuid4())
    job = Job(
        id=job_id,
        user_id=user_id,
        type=JobType.EXTRACT,
        status=JobStatus.QUEUED,
        payload={'session_id': 'sess-fail', 'source_ids': []},
        attempts=0,
        max_attempts=2,
        deferrals=0,
        run_after=test_clock.now(),
        worker_id=None,
        lease_expires_at=None,
        coalesce_key=None,
        idempotency_key='k-fail',
        last_error=None,
    )
    jobs.jobs[job_id] = job

    # Attempt 1 -> fails and gets rescheduled with retry
    await worker.run_once()
    assert jobs.jobs[job_id].status == JobStatus.QUEUED
    assert jobs.jobs[job_id].attempts == 1
    assert jobs.jobs[job_id].run_after > test_clock.now()

    # Fast forward clock past run_after
    test_clock.advance(timedelta(seconds=350.0))

    # Attempt 2 -> hits max_attempts (2) -> killed
    await worker.run_once()
    assert jobs.jobs[job_id].status == JobStatus.DEAD
    assert jobs.jobs[job_id].attempts == 2
    assert jobs.jobs[job_id].last_error == 'Temporary network timeout to LLM'


@pytest.mark.asyncio
async def test_worker_runner_heartbeat_lease_lost_cancellation(
    fake_repos: Any,
    test_clock: FixedClock,
) -> None:
    """WorkerRunner cancels job task when heartbeat reports lease lost / fencing."""
    uows, sources, jobs, entities, facts, profiles = fake_repos

    cancelled_event = asyncio.Event()

    class LongRunningService(IngestService):
        async def run_extract_job(self, job: Job, worker_id: UUID | None = None) -> list[FactId]:
            try:
                # Hang until cancelled
                await asyncio.sleep(10.0)
            except asyncio.CancelledError:
                cancelled_event.set()
                raise
            return []

    service = LongRunningService(clock=test_clock)
    worker = WorkerRunner(
        uows=uows,
        service=service,
        clock=test_clock,
        heartbeat_interval_s=0.01,
    )

    # Trigger fence in FakeJobRepository
    jobs.fence_worker = True

    user_id = UserId(uuid4())
    job_id = JobId(uuid4())
    job = Job(
        id=job_id,
        user_id=user_id,
        type=JobType.EXTRACT,
        status=JobStatus.QUEUED,
        payload={'session_id': 'sess-fence', 'source_ids': []},
        attempts=0,
        max_attempts=3,
        deferrals=0,
        run_after=test_clock.now(),
        worker_id=None,
        lease_expires_at=None,
        coalesce_key=None,
        idempotency_key='k-fence',
        last_error=None,
    )
    jobs.jobs[job_id] = job

    # Run once; heartbeat should detect fence_worker immediately and cancel job_task
    await worker.run_once()

    assert cancelled_event.is_set(), 'Long-running handler was not cancelled on LeaseLost'


@pytest.mark.asyncio
async def test_worker_runner_maintenance_reap_and_tombstone_purge(
    fake_repos: Any,
    test_clock: FixedClock,
) -> None:
    """WorkerRunner executes reap_expired and hard_delete_expired_tombstones during maintenance."""
    uows, sources, jobs, entities, facts, profiles = fake_repos
    service = IngestService(clock=test_clock)
    worker = WorkerRunner(
        uows=uows,
        service=service,
        clock=test_clock,
        reaper_interval_s=10.0,
        tombstone_retention_days=30,
    )

    # 1. Add expired running job
    expired_job_id = JobId(uuid4())
    jobs.jobs[expired_job_id] = Job(
        id=expired_job_id,
        user_id=UserId(uuid4()),
        type=JobType.EXTRACT,
        status=JobStatus.RUNNING,
        payload={},
        attempts=1,
        max_attempts=3,
        deferrals=0,
        run_after=test_clock.now() - timedelta(minutes=10),
        worker_id=uuid4(),
        lease_expires_at=test_clock.now() - timedelta(minutes=5),
        coalesce_key=None,
        idempotency_key='k-exp',
        last_error=None,
    )

    # 2. Add expired soft-deleted fact (tombstone)
    old_date = test_clock.now() - timedelta(days=45)
    tombstone_fid = FactId(uuid4())
    facts.facts[tombstone_fid] = Fact(
        id=tombstone_fid,
        user_id=UserId(uuid4()),
        subject_id=EntityId(uuid4()),
        predicate='related_to',
        cardinality=Cardinality.MULTI,
        object_entity_id=None,
        object_value='old_val',
        object_norm='old_val',
        fact_text='User related to old_val',
        valid_from=old_date,
        valid_to=None,
        time_precision=TimePrecision.EXACT,
        valid_from_explicit=True,
        confidence=1.0,
        salience=1.0,
        status=FactStatus.DELETED,
        origin=FactOrigin.USER_EDIT,
        recorded_at=old_date,
        invalidated_at=old_date,
        event_hash=None,
        access_count=0,
        last_accessed_at=None,
    )

    # Trigger maintenance cycle via run_once
    test_clock.advance(timedelta(seconds=15.0))
    await worker.run_once()

    # Verify expired job was reaped back to QUEUED
    assert jobs.jobs[expired_job_id].status == JobStatus.QUEUED
    assert jobs.reaped_count >= 1

    # Verify expired tombstone was purged
    assert tombstone_fid not in facts.facts
    assert facts.tombstones_deleted == 1
