"""Unit tests for Workbench API helper endpoints and telemetry per Milestone M7."""

import builtins
import json
from collections.abc import AsyncIterator, Sequence
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID, uuid4

import httpx
import jwt
import pytest
from memory_engine.config import Settings
from memory_engine.domain.clock import FixedClock
from memory_engine.domain.enums import (
    EntityType,
    FactOrigin,
    FactStatus,
    JobStatus,
    JobType,
    RelationType,
    Role,
)
from memory_engine.domain.ids import EntityId, FactId, JobId, SourceId, UserId
from memory_engine.domain.models import (
    Candidate,
    Entity,
    Fact,
    FactEvidence,
    FactRelation,
    Job,
)
from memory_engine.ingest.canonicalizer import Canonicalizer
from memory_engine.lifecycle.service import MemoryLifecycleService
from memory_engine.ports.embed import Embedder
from memory_engine.ports.repositories import (
    EvidenceSnippet,
    FactListQuery,
    InsertedSource,
    NewSource,
    Page,
)
from memory_engine.retrieve.chat_service import (
    ChatRequest as ServiceChatRequest,
)
from memory_engine.retrieve.chat_service import (
    ChatStreamEvent,
    ChatTurnResult,
)

from apps.api import AppContainer, JwtVerifier, RateLimiter, create_app

JWT_TEST_SECRET = 'super-secret-test-key-32-bytes-long!'


def utc_now() -> datetime:
    return datetime(2026, 10, 29, 12, 0, 0, tzinfo=UTC)


def make_jwt(user_id: UUID | UserId, secret: str = JWT_TEST_SECRET) -> str:
    payload = {
        'sub': str(user_id),
        'aud': 'authenticated',
        'exp': int((datetime.now(UTC) + timedelta(hours=1)).timestamp()),
    }
    return jwt.encode(payload, secret, algorithm='HS256')


# ---------------------------------------------------------------------------
# In-Memory Fake Repositories for Workbench Testing
# ---------------------------------------------------------------------------


class WorkbenchFakeFactsRepo:
    def __init__(self) -> None:
        self.facts: dict[FactId, Fact] = {}
        self.evidence: dict[FactId, list[FactEvidence]] = {}
        self.relations: list[FactRelation] = []

    async def get(self, user_id: UserId, fact_id: FactId) -> Fact | None:
        f = self.facts.get(fact_id)
        if f is not None and f.user_id == user_id:
            return f
        return None

    async def list(self, user_id: UserId, q: FactListQuery) -> Page[Fact]:
        matched = [f for f in self.facts.values() if f.user_id == user_id]
        if q.status is not None:
            matched = [f for f in matched if f.status == q.status]
        return Page(items=matched, next_cursor=None)

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
        fact = Fact(
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
            salience=0.5,
            access_count=0,
            last_accessed_at=None,
            status=status,
            origin=origin,
            event_hash=c.event_hash,
        )
        self.facts[fid] = fact
        return fact

    async def add_evidence(self, ev: FactEvidence, user_id: UserId) -> None:
        self.evidence.setdefault(ev.fact_id, []).append(ev)

    async def add_relation(
        self, user_id: UserId, rel: FactRelation, job_id: UUID | None = None
    ) -> None:
        self.relations.append(rel)

    async def evidence_for(
        self, user_id: UserId, fact_ids: Sequence[FactId], per_fact: int
    ) -> dict[FactId, builtins.list[EvidenceSnippet]]:
        res: dict[FactId, builtins.list[EvidenceSnippet]] = {}
        for fid in fact_ids:
            evs = self.evidence.get(fid, [])
            res[fid] = [
                EvidenceSnippet(
                    source_id=e.source_id,
                    source_ts=datetime.now(UTC),
                    role=Role.USER,
                    snippet='snippet',
                )
                for e in evs[:per_fact]
            ]
        return res

    async def chain(self, user_id: UserId, fact_id: FactId) -> builtins.list[Fact]:
        return []


class WorkbenchFakeEntitiesRepo:
    def __init__(self) -> None:
        self.entities: dict[EntityId, Entity] = {}

    async def ensure_self(self, user_id: UserId) -> Entity:
        for ent in self.entities.values():
            if ent.is_self and ent.user_id == user_id:
                return ent
        eid = EntityId(uuid4())
        ent = Entity(
            id=eid,
            user_id=user_id,
            canonical_name='User',
            type=EntityType.PERSON,
            is_self=True,
        )
        self.entities[eid] = ent
        return ent

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


class WorkbenchFakeSourcesRepo:
    def __init__(self) -> None:
        self.sources: list[NewSource] = []
        self.embeddings: dict[SourceId, list[float]] = {}

    async def insert_many(self, items: Sequence[NewSource]) -> list[InsertedSource]:
        res: list[InsertedSource] = []
        for it in items:
            self.sources.append(it)
            res.append(InsertedSource(id=SourceId(uuid4()), created=True))
        return res

    async def set_embeddings(
        self,
        user_id: UserId,
        items: Sequence[tuple[SourceId, list[float]]],
        model_id: str,
    ) -> None:
        for sid, emb in items:
            self.embeddings[sid] = list(emb)


class WorkbenchFakeJobsRepo:
    def __init__(self) -> None:
        self.jobs: dict[JobId, Job] = {}

    async def get(self, user_id: UserId, job_id: JobId) -> Job | None:
        j = self.jobs.get(job_id)
        if j is not None and j.user_id == user_id:
            return j
        return None

    async def list_for_user(
        self, user_id: UserId, status: JobStatus | None = None, limit: int = 50
    ) -> list[Job]:
        matched = [j for j in self.jobs.values() if j.user_id == user_id]
        if status is not None:
            matched = [j for j in matched if j.status == status]
        return matched[:limit]


class WorkbenchFakeProfilesRepo:
    def __init__(self) -> None:
        self.purged_users: list[UserId] = []

    async def purge_user(self, user_id: UserId) -> int:
        self.purged_users.append(user_id)
        return 5

    async def export_user(self, user_id: UserId) -> dict[str, Any]:
        return {'user_id': str(user_id), 'facts': [], 'sources': []}


class WorkbenchFakeAuditRepo:
    def __init__(self) -> None:
        self.entries: list[dict[str, Any]] = []

    async def append(self, **kwargs: Any) -> None:
        self.entries.append(kwargs)

    async def record(self, **kwargs: Any) -> None:
        self.entries.append(kwargs)


class WorkbenchFakeUow:
    def __init__(self) -> None:
        self.facts = WorkbenchFakeFactsRepo()
        self.entities = WorkbenchFakeEntitiesRepo()
        self.sources = WorkbenchFakeSourcesRepo()
        self.jobs = WorkbenchFakeJobsRepo()
        self.profiles = WorkbenchFakeProfilesRepo()
        self.audit = WorkbenchFakeAuditRepo()

    async def lock_user(self, user_id: UserId | None = None) -> None:
        pass

    async def __aenter__(self) -> 'WorkbenchFakeUow':
        return self

    async def __aexit__(self, *args: Any) -> None:
        pass


class WorkbenchFakeUowFactory:
    def __init__(self, uow: WorkbenchFakeUow | None = None) -> None:
        self.uow = uow or WorkbenchFakeUow()

    def for_user(self, user_id: UserId, *, readonly: bool = False) -> Any:
        return self.uow

    def begin(self, user_id: UserId | None, *, readonly: bool = False) -> Any:
        return self.uow

    def system(self, *, readonly: bool = False) -> Any:
        return self.uow


class WorkbenchFakeChatService:
    def __init__(self) -> None:
        self.telemetry_payload = {
            'intent': 'current_state',
            'filter_mode': 'current',
            'as_of': None,
            'time_range': None,
            'entity_hints': ['User'],
            'predicate_hints': ['lives_in'],
            'lane_counts': {'vector_facts': 2, 'fts_facts': 1},
            'degraded_lanes': [],
            'abstained': False,
            'abstention_reason': None,
            'context_text': '[F1] User lives in Pune | valid 2026-10-01 to present',
            'token_count': 18,
            'latency_ms': 42.5,
        }

    async def respond(self, user_id: UserId, req: ServiceChatRequest) -> ChatTurnResult:
        return ChatTurnResult(
            conversation_id=req.conversation_id or uuid4(),
            user_message_id=uuid4(),
            assistant_message_id=uuid4(),
            content='You currently live in Pune and work at Stripe.',
            cited_fact_ids=[FactId(uuid4())],
            cited_source_ids=[SourceId(uuid4())],
            statement_only=False,
            abstained=False,
            latency_ms=42.5,
            telemetry=self.telemetry_payload,
            trace=self.telemetry_payload,
        )

    async def stream_respond(
        self, user_id: UserId, req: ServiceChatRequest
    ) -> AsyncIterator[ChatStreamEvent]:
        asst_id = uuid4()
        f_id = FactId(uuid4())
        s_id = SourceId(uuid4())
        yield ChatStreamEvent.plan({'intent': 'current_state'})
        yield ChatStreamEvent.trace([{'label': 'F1', 'id': str(f_id)}])
        yield ChatStreamEvent.token('You live in ')
        yield ChatStreamEvent.token('Pune.')
        yield ChatStreamEvent.citations([f_id], [s_id])
        yield ChatStreamEvent.done(asst_id, partial=False, telemetry=self.telemetry_payload)


class WorkbenchFakeIngestService:
    def __init__(self) -> None:
        self.executed_jobs: list[Job] = []

    async def run_extract_job(self, job: Job, worker_id: UUID | None = None) -> list[FactId]:
        self.executed_jobs.append(job)
        return [FactId(uuid4())]


class WorkbenchFakeEmbedder(Embedder):
    def __init__(self) -> None:
        self.dim = 768
        self.model_id = 'test-embedder'

    async def embed_query(self, text: str) -> list[float]:
        return [0.1] * self.dim

    async def embed_queries(self, texts: Sequence[str]) -> list[list[float]]:
        return [[0.1] * self.dim for _ in texts]

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [[0.1] * self.dim for _ in texts]


# ---------------------------------------------------------------------------
# Test Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def workbench_clock() -> FixedClock:
    return FixedClock(utc_now())


@pytest.fixture
def workbench_uow() -> WorkbenchFakeUow:
    return WorkbenchFakeUow()


@pytest.fixture
def workbench_uow_factory(workbench_uow: WorkbenchFakeUow) -> WorkbenchFakeUowFactory:
    return WorkbenchFakeUowFactory(workbench_uow)


@pytest.fixture
def workbench_container(
    workbench_clock: FixedClock,
    workbench_uow_factory: WorkbenchFakeUowFactory,
) -> AppContainer:
    settings = Settings(
        env='test',
        jwt_secret=JWT_TEST_SECRET,
        jwt_audience='authenticated',
    )
    lifecycle = MemoryLifecycleService(
        uows=workbench_uow_factory,
        clock=workbench_clock,
        canonicalizer=Canonicalizer(),
    )
    chat_svc = WorkbenchFakeChatService()
    ingest_svc = WorkbenchFakeIngestService()
    embedder = WorkbenchFakeEmbedder()
    limiter = RateLimiter(
        clock=workbench_clock,
        bucket_configs={
            'chat': (30.0, 10.0),
            'ingest': (30.0, 10.0),
            'default': (120.0, 30.0),
        },
    )
    jwt_verifier = JwtVerifier(settings.jwt, env='test')

    return AppContainer(
        settings=settings,
        clock=workbench_clock,
        uows=workbench_uow_factory,
        ingest=ingest_svc,  # type: ignore[arg-type]
        retrieval=None,  # type: ignore[arg-type]
        chat=chat_svc,  # type: ignore[arg-type]
        lifecycle=lifecycle,
        rate_limiter=limiter,
        jwt_verifier=jwt_verifier,
        embedder=embedder,
    )


@pytest.fixture
async def client(workbench_container: AppContainer) -> AsyncIterator[httpx.AsyncClient]:
    app = create_app(container=workbench_container)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url='http://testserver') as ac:
        yield ac


# ---------------------------------------------------------------------------
# Tests: GET /v1/dev/token
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_dev_token_default_and_custom_user(
    client: httpx.AsyncClient,
    workbench_container: AppContainer,
) -> None:
    """GET /v1/dev/token mints valid token for default and custom user_id."""
    # 1. Default user_id
    resp = await client.get('/v1/dev/token')
    assert resp.status_code == 200
    data = resp.json()
    assert 'token' in data
    assert data['user_id'] == '00000000-0000-0000-0000-000000000001'
    assert data['env'] == 'test'

    # Verify token payload
    decoded = jwt.decode(
        data['token'],
        JWT_TEST_SECRET,
        algorithms=['HS256'],
        audience='authenticated',
        options={'verify_iat': False},
    )
    assert decoded['sub'] == '00000000-0000-0000-0000-000000000001'

    # Verify through JwtVerifier
    auth_user = await workbench_container.jwt_verifier.verify(data['token'])  # type: ignore[union-attr]
    assert str(auth_user.user_id) == '00000000-0000-0000-0000-000000000001'

    # 2. Custom user_id query param
    custom_uid = UUID('22222222-2222-2222-2222-222222222222')
    resp2 = await client.get(f'/v1/dev/token?user_id={custom_uid}')
    assert resp2.status_code == 200
    data2 = resp2.json()
    assert data2['user_id'] == str(custom_uid)
    decoded2 = jwt.decode(
        data2['token'],
        JWT_TEST_SECRET,
        algorithms=['HS256'],
        audience='authenticated',
        options={'verify_iat': False},
    )
    assert decoded2['sub'] == str(custom_uid)
    auth_user2 = await workbench_container.jwt_verifier.verify(data2['token'])  # type: ignore[union-attr]
    assert auth_user2.user_id == UserId(custom_uid)


@pytest.mark.asyncio
async def test_get_dev_token_disabled_in_prod(
    workbench_clock: FixedClock,
    workbench_uow_factory: WorkbenchFakeUowFactory,
) -> None:
    """GET /v1/dev/token returns 403 Forbidden when env == 'prod'."""
    prod_settings = Settings(
        env='prod',
        jwt_secret=JWT_TEST_SECRET,
    )
    prod_container = AppContainer(
        settings=prod_settings,
        clock=workbench_clock,
        uows=workbench_uow_factory,
        ingest=WorkbenchFakeIngestService(),  # type: ignore[arg-type]
        retrieval=None,  # type: ignore[arg-type]
        chat=WorkbenchFakeChatService(),  # type: ignore[arg-type]
        lifecycle=MemoryLifecycleService(
            uows=workbench_uow_factory, clock=workbench_clock, canonicalizer=Canonicalizer()
        ),
        jwt_verifier=JwtVerifier(prod_settings.jwt, env='prod'),
    )
    app = create_app(container=prod_container)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url='http://testserver') as ac:
        resp = await ac.get('/v1/dev/token')
        assert resp.status_code == 403
        err = resp.json()
        assert err['error']['code'] == 'E-AUTH-02'


# ---------------------------------------------------------------------------
# Tests: POST /v1/jobs/run-once
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_jobs_run_once_auth_and_execution(
    client: httpx.AsyncClient,
    workbench_uow: WorkbenchFakeUow,
) -> None:
    """POST /v1/jobs/run-once enforces auth and executes pending jobs."""
    # 1. Unauthenticated rejects with 401
    unauth = await client.post('/v1/jobs/run-once')
    assert unauth.status_code == 401

    # 2. Authenticated with 0 pending jobs
    user_id = uuid4()
    headers = {'Authorization': f'Bearer {make_jwt(user_id)}'}
    resp = await client.post('/v1/jobs/run-once', headers=headers)
    assert resp.status_code == 200
    assert resp.json() == {'processed': 0, 'job_ids': []}

    # 3. Authenticated with 2 queued extract jobs
    j1_id = JobId(uuid4())
    j2_id = JobId(uuid4())
    other_user_id = uuid4()
    j_other_id = JobId(uuid4())

    workbench_uow.jobs.jobs[j1_id] = Job(
        id=j1_id,
        user_id=UserId(user_id),
        type=JobType.EXTRACT,
        payload={'session_id': 'sess1'},
        idempotency_key='k1',
        coalesce_key=None,
        status=JobStatus.QUEUED,
        attempts=0,
        max_attempts=3,
        deferrals=0,
        run_after=utc_now(),
    )
    workbench_uow.jobs.jobs[j2_id] = Job(
        id=j2_id,
        user_id=UserId(user_id),
        type=JobType.EXTRACT,
        payload={'session_id': 'sess2'},
        idempotency_key='k2',
        coalesce_key=None,
        status=JobStatus.QUEUED,
        attempts=0,
        max_attempts=3,
        deferrals=0,
        run_after=utc_now(),
    )
    workbench_uow.jobs.jobs[j_other_id] = Job(
        id=j_other_id,
        user_id=UserId(other_user_id),
        type=JobType.EXTRACT,
        payload={'session_id': 'other'},
        idempotency_key='k3',
        coalesce_key=None,
        status=JobStatus.QUEUED,
        attempts=0,
        max_attempts=3,
        deferrals=0,
        run_after=utc_now(),
    )

    resp2 = await client.post('/v1/jobs/run-once', headers=headers)
    assert resp2.status_code == 200
    data = resp2.json()
    assert data['processed'] == 2
    assert str(j1_id) in data['job_ids']
    assert str(j2_id) in data['job_ids']
    assert str(j_other_id) not in data['job_ids']

    # Jobs marked done
    assert workbench_uow.jobs.jobs[j1_id].status == JobStatus.DONE
    assert workbench_uow.jobs.jobs[j2_id].status == JobStatus.DONE
    # Other user's job untouched
    assert workbench_uow.jobs.jobs[j_other_id].status == JobStatus.QUEUED


# ---------------------------------------------------------------------------
# Tests: POST /v1/dev/seed-scenario
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dev_seed_scenario_deterministic(
    client: httpx.AsyncClient,
    workbench_uow: WorkbenchFakeUow,
) -> None:
    """POST /v1/dev/seed-scenario purges and seeds the 3-stage bitemporal scenario."""
    user_id = uuid4()
    headers = {'Authorization': f'Bearer {make_jwt(user_id)}'}

    # 1. Unauthenticated rejects with 401
    unauth = await client.post('/v1/dev/seed-scenario')
    assert unauth.status_code == 401

    # 2. Seed scenario
    resp = await client.post('/v1/dev/seed-scenario', headers=headers)
    assert resp.status_code == 201
    data = resp.json()
    assert data['seeded_facts'] == 8
    assert data['seeded_sources'] == 3
    assert data['user_id'] == str(user_id)

    # Verify purge was called for tenant
    assert UserId(user_id) in workbench_uow.profiles.purged_users

    # Verify facts were created
    facts = list(workbench_uow.facts.facts.values())
    assert len(facts) == 8

    # Verify Bangalore historical backfill
    blr_f = next(f for f in facts if f.object_norm == 'bangalore')
    assert blr_f.valid_from == datetime(2024, 1, 1, tzinfo=UTC)
    assert blr_f.valid_to == datetime(2025, 1, 1, tzinfo=UTC)

    # Verify Delhi prior state
    delhi_f = next(f for f in facts if f.object_norm == 'delhi')
    assert delhi_f.valid_from == datetime(2025, 1, 1, tzinfo=UTC)
    assert delhi_f.valid_to == datetime(2026, 10, 1, tzinfo=UTC)

    # Verify Pune current active state
    pune_f = next(f for f in facts if f.object_norm == 'pune')
    assert pune_f.valid_from == datetime(2026, 10, 1, tzinfo=UTC)
    assert pune_f.valid_to is None

    # Verify supersession relations
    assert len(workbench_uow.facts.relations) == 3
    supersedes = [r for r in workbench_uow.facts.relations if r.relation == RelationType.SUPERSEDES]
    assert len(supersedes) == 3

    # 3. Idempotent re-run
    resp2 = await client.post('/v1/dev/seed-scenario', headers=headers)
    assert resp2.status_code == 201
    data2 = resp2.json()
    assert data2['seeded_facts'] == 8


# ---------------------------------------------------------------------------
# Tests: Enriched Chat & SSE Telemetry Fields
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_chat_non_streaming_enriched_telemetry(
    client: httpx.AsyncClient,
) -> None:
    """POST /v1/chat returns rich trace and telemetry objects with all 13 fields."""
    user_id = uuid4()
    headers = {'Authorization': f'Bearer {make_jwt(user_id)}'}
    payload = {'message': 'Where do I live?'}

    resp = await client.post('/v1/chat', json=payload, headers=headers)
    assert resp.status_code == 200
    data = resp.json()

    assert 'content' in data
    assert 'conversation_id' in data
    assert 'assistant_message_id' in data

    # Verify both telemetry and trace objects are populated
    for key in ('telemetry', 'trace'):
        assert key in data
        telem = data[key]
        assert telem is not None
        assert telem['intent'] == 'current_state'
        assert telem['filter_mode'] == 'current'
        assert telem['as_of'] is None
        assert telem['time_range'] is None
        assert telem['entity_hints'] == ['User']
        assert telem['predicate_hints'] == ['lives_in']
        assert telem['lane_counts'] == {'vector_facts': 2, 'fts_facts': 1}
        assert telem['degraded_lanes'] == []
        assert telem['abstained'] is False
        assert telem['abstention_reason'] is None
        assert telem['context_text'] == '[F1] User lives in Pune | valid 2026-10-01 to present'
        assert telem['token_count'] == 18
        assert telem['latency_ms'] == 42.5


@pytest.mark.asyncio
async def test_chat_stream_done_event_enriched_telemetry(
    client: httpx.AsyncClient,
) -> None:
    """POST /v1/chat/stream includes telemetry and trace in the final done SSE frame."""
    user_id = uuid4()
    headers = {'Authorization': f'Bearer {make_jwt(user_id)}'}
    payload = {'message': 'Where do I live?'}

    resp = await client.post('/v1/chat/stream', json=payload, headers=headers)
    assert resp.status_code == 200
    assert 'text/event-stream' in resp.headers['content-type']

    body = resp.text
    assert 'event: plan\n' in body
    assert 'event: token\n' in body
    assert 'event: done\n' in body

    # Extract done event payload
    done_line = [
        line
        for line in body.splitlines()
        if line.startswith('data:') and 'assistant_message_id' in line
    ][-1]
    done_payload = json.loads(done_line[len('data:') :].strip())

    assert 'assistant_message_id' in done_payload
    assert done_payload['partial'] is False

    # Check telemetry and trace in done event
    assert 'telemetry' in done_payload
    assert 'trace' in done_payload

    telem = done_payload['telemetry']
    assert telem['intent'] == 'current_state'
    assert telem['filter_mode'] == 'current'
    assert telem['context_text'] == '[F1] User lives in Pune | valid 2026-10-01 to present'
    assert telem['token_count'] == 18
    assert telem['latency_ms'] == 42.5
