"""Unit and contract tests for FastAPI REST and SSE API routes (M5-03).

Runs 100% socket-free with httpx.ASGITransport.
Verifies:
  1. Auth rejection (401 E-AUTH-01) on missing/invalid JWT and valid extraction (T-API-01).
  2. Rate limiting (429 E-RATE-01) with Retry-After header (T-API-02).
  3. POST /v1/chat and POST /v1/chat/stream SSE event formatting (T-API-03).
  4. POST /v1/notes chunking and outbox job creation (T-API-04).
  5. GET /v1/facts, GET /v1/facts/{id}, PATCH /v1/facts/{id}, POST /v1/facts/{id}/retract,
     DELETE /v1/facts/{id}, GET /v1/jobs/{id}, GET /v1/profile/export, DELETE /v1/profile,
     /healthz, /readyz.
  6. PayloadTooLarge (413 E-REQ-01) and RequestId echo (X-Request-Id).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

if TYPE_CHECKING:
    import builtins
    from collections.abc import AsyncIterator, Sequence

import httpx
import jwt
import pytest
from memory_engine.config import Settings
from memory_engine.domain.clock import FixedClock
from memory_engine.domain.enums import (
    Cardinality,
    FactOrigin,
    FactStatus,
    JobStatus,
    JobType,
    TimePrecision,
)
from memory_engine.domain.ids import EntityId, FactId, JobId, SourceId, UserId
from memory_engine.domain.models import Entity, Fact, Job, Message
from memory_engine.ingest.canonicalizer import Canonicalizer
from memory_engine.ingest.service import NoteIngestResult
from memory_engine.lifecycle.service import MemoryLifecycleService
from memory_engine.ports.repositories import (
    EvidenceSnippet,
    FactListQuery,
    InsertedMessage,
    InsertedSource,
    NewExtractJob,
    NewMessage,
    Page,
    UpsertResult,
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
    return datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)


def make_jwt(
    user_id: UUID | UserId,
    secret: str = JWT_TEST_SECRET,
    expired: bool = False,
    aud: str = 'authenticated',
) -> str:
    """Helper to generate JWT bearer tokens for tests."""
    base_time = datetime.now(UTC)
    exp_time = base_time - timedelta(minutes=5) if expired else base_time + timedelta(hours=1)
    payload = {
        'sub': str(user_id),
        'aud': aud,
        'exp': int(exp_time.timestamp()),
    }
    return jwt.encode(payload, secret, algorithm='HS256')


# ---------------------------------------------------------------------------
# Fake Repositories & UoW for API Testing
# ---------------------------------------------------------------------------


class FakeFactsRepo:
    def __init__(self) -> None:
        self.facts: dict[FactId, Fact] = {}
        self.evidence: dict[FactId, builtins.list[EvidenceSnippet]] = {}
        self.chains: dict[FactId, builtins.list[Fact]] = {}

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

    async def evidence_for(
        self,
        user_id: UserId,
        fact_ids: Sequence[FactId],
        per_fact: int,
    ) -> dict[FactId, builtins.list[EvidenceSnippet]]:
        return {fid: self.evidence.get(fid, []) for fid in fact_ids}

    async def chain(self, user_id: UserId, fact_id: FactId) -> builtins.list[Fact]:
        return self.chains.get(fact_id, [])

    async def manual_update(
        self,
        user_id: UserId,
        fact_id: FactId,
        *,
        fact_text: str | None = None,
        object_value: str | None = None,
        object_norm: str | None = None,
        valid_from: datetime | None = None,
        valid_to: datetime | None = None,
        status: FactStatus | str | None = None,
        confidence: float | None = None,
        invalidated_at: datetime | None = None,
    ) -> None:
        if fact_id in self.facts:
            f = self.facts[fact_id]
            new_status: FactStatus = f.status
            if status is not None:
                if isinstance(status, FactStatus):
                    new_status = status
                elif status == 'deleted':
                    new_status = FactStatus.DELETED
                elif status == 'retracted':
                    new_status = FactStatus.ARCHIVED
            self.facts[fact_id] = Fact(
                id=f.id,
                user_id=f.user_id,
                subject_id=f.subject_id,
                predicate=f.predicate,
                cardinality=f.cardinality,
                object_entity_id=f.object_entity_id,
                object_value=object_value if object_value is not None else f.object_value,
                object_norm=object_norm if object_norm is not None else f.object_norm,
                fact_text=fact_text if fact_text is not None else f.fact_text,
                valid_from=valid_from if valid_from is not None else f.valid_from,
                valid_to=valid_to if valid_to is not None else f.valid_to,
                time_precision=f.time_precision,
                valid_from_explicit=f.valid_from_explicit,
                recorded_at=f.recorded_at,
                invalidated_at=invalidated_at if invalidated_at is not None else f.invalidated_at,
                confidence=confidence if confidence is not None else f.confidence,
                salience=f.salience,
                access_count=f.access_count,
                last_accessed_at=f.last_accessed_at,
                status=new_status,
                origin=f.origin,
                event_hash=f.event_hash,
            )


class FakeEntitiesRepo:
    def __init__(self) -> None:
        self.entities: dict[EntityId, Entity] = {}

    async def get_many(self, user_id: UserId, ids: Sequence[EntityId]) -> dict[EntityId, Entity]:
        return {eid: self.entities[eid] for eid in ids if eid in self.entities}


class FakeAuditRepo:
    def __init__(self) -> None:
        self.entries: list[dict[str, Any]] = []

    async def append(self, **kwargs: Any) -> None:
        self.entries.append(kwargs)

    async def record(self, **kwargs: Any) -> None:
        self.entries.append(kwargs)


class FakeJobsRepo:
    def __init__(self) -> None:
        self.jobs: dict[JobId, Job] = {}

    async def get(self, user_id: UserId, job_id: JobId) -> Job | None:
        j = self.jobs.get(job_id)
        if j is not None and j.user_id == user_id:
            return j
        return None

    async def upsert_coalesced(self, j: NewExtractJob, debounce_s: float) -> UpsertResult:
        jid = JobId(uuid4())
        return UpsertResult(job_id=jid, created=True)


class FakeProfilesRepo:
    def __init__(self) -> None:
        self.purged_count = 10
        self.exported_data: dict[str, Any] = {'facts': [], 'sources': []}

    async def purge_user(self, user_id: UserId) -> int:
        return self.purged_count

    async def export_user(self, user_id: UserId) -> dict[str, Any]:
        return {**self.exported_data, 'user_id': str(user_id)}


class FakeMessagesRepo:
    def __init__(self) -> None:
        self.messages: list[NewMessage] = []

    async def ensure_conversation(
        self, user_id: UserId, conv_id: UUID | None, now: datetime
    ) -> UUID:
        return conv_id or uuid4()

    async def insert(self, m: NewMessage) -> InsertedMessage:
        self.messages.append(m)
        return InsertedMessage(id=uuid4(), created=True)

    async def recent_turns(self, user_id: UserId, conv_id: UUID, n: int) -> list[Message]:
        return []

    async def pending_user_messages(self, user_id: UserId, limit: int) -> list[Message]:
        return []


class FakeSourcesRepo:
    def __init__(self) -> None:
        self.sources: list[Any] = []

    async def insert_many(self, items: Sequence[Any]) -> list[InsertedSource]:
        res: list[InsertedSource] = []
        for _ in items:
            res.append(InsertedSource(id=SourceId(uuid4()), created=True))
        return res


class FakeUnitOfWork:
    def __init__(self) -> None:
        self.facts: Any = FakeFactsRepo()
        self.entities: Any = FakeEntitiesRepo()
        self.audit: Any = FakeAuditRepo()
        self.jobs: Any = FakeJobsRepo()
        self.profiles: Any = FakeProfilesRepo()
        self.messages: Any = FakeMessagesRepo()
        self.sources: Any = FakeSourcesRepo()
        self.search: Any = None
        self.locked_users: list[UserId | None] = []

    async def lock_user(self, user_id: UserId | None = None) -> None:
        self.locked_users.append(user_id)

    async def __aenter__(self) -> FakeUnitOfWork:
        return self

    async def __aexit__(self, *args: Any) -> None:
        pass


class FakeUnitOfWorkFactory:
    def __init__(self, shared_uow: FakeUnitOfWork | None = None) -> None:
        self.shared_uow = shared_uow or FakeUnitOfWork()

    def begin(self, user_id: UserId | None, *, readonly: bool = False) -> Any:
        return self.shared_uow

    def for_user(self, user_id: UserId, *, readonly: bool = False) -> Any:
        return self.shared_uow

    def system(self, *, readonly: bool = False) -> Any:
        return self.shared_uow


class FakeChatService:
    def __init__(self) -> None:
        self.last_message: str | None = None

    async def respond(
        self,
        user_id: UserId,
        req: ServiceChatRequest,
    ) -> ChatTurnResult:
        self.last_message = req.message
        telem = {
            'intent': 'current_state',
            'filter_mode': 'current',
            'as_of': None,
            'time_range': None,
            'entity_hints': ['User'],
            'predicate_hints': ['lives_in'],
            'lane_counts': {'vector_facts': 1},
            'degraded_lanes': [],
            'abstained': False,
            'abstention_reason': None,
            'context_text': '[F1] User lives in Delhi',
            'token_count': 12,
            'latency_ms': 15.0,
        }
        return ChatTurnResult(
            conversation_id=req.conversation_id or uuid4(),
            user_message_id=uuid4(),
            assistant_message_id=uuid4(),
            content='This is a helpful memory assistant reply.',
            cited_fact_ids=[FactId(uuid4())],
            cited_source_ids=[SourceId(uuid4())],
            statement_only=False,
            abstained=False,
            telemetry=telem,
            trace=telem,
        )

    async def stream_respond(
        self,
        user_id: UserId,
        req: ServiceChatRequest,
    ) -> AsyncIterator[ChatStreamEvent]:
        self.last_message = req.message
        asst_id = uuid4()
        f_id = FactId(uuid4())
        s_id = SourceId(uuid4())
        telem = {
            'intent': 'current_state',
            'filter_mode': 'current',
            'as_of': None,
            'time_range': None,
            'entity_hints': ['User'],
            'predicate_hints': ['lives_in'],
            'lane_counts': {'vector_facts': 1},
            'degraded_lanes': [],
            'abstained': False,
            'abstention_reason': None,
            'context_text': '[F1] User lives in Delhi',
            'token_count': 12,
            'latency_ms': 15.0,
        }
        yield ChatStreamEvent.plan({'intent': 'QUESTION'})
        yield ChatStreamEvent.trace([{'label': 'F1', 'id': str(f_id)}])
        yield ChatStreamEvent.token('Hello ')
        yield ChatStreamEvent.token('world!')
        yield ChatStreamEvent.citations([f_id], [s_id])
        yield ChatStreamEvent.done(asst_id, partial=False, telemetry=telem)


class FakeIngestService:
    async def accept_note(
        self,
        uow: Any,
        *,
        user_id: UserId,
        content: str | None = None,
        title: str | None = None,
        source_ts: datetime | None = None,
        **kwargs: Any,
    ) -> NoteIngestResult:
        s1 = SourceId(uuid4())
        s2 = SourceId(uuid4())
        jid = JobId(uuid4())
        return NoteIngestResult(source_ids=[s1, s2], job_id=jid, created=True)


# ---------------------------------------------------------------------------
# Test Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def test_clock() -> FixedClock:
    return FixedClock(utc_now())


@pytest.fixture
def uow_factory() -> FakeUnitOfWorkFactory:
    return FakeUnitOfWorkFactory()


@pytest.fixture
def app_container(test_clock: FixedClock, uow_factory: FakeUnitOfWorkFactory) -> AppContainer:
    settings = Settings(
        env='test',
        jwt_secret=JWT_TEST_SECRET,
        jwt_audience='authenticated',
    )
    lifecycle = MemoryLifecycleService(
        uows=uow_factory,
        clock=test_clock,
        canonicalizer=Canonicalizer(),
    )
    chat_svc = FakeChatService()
    ingest_svc = FakeIngestService()
    limiter = RateLimiter(
        clock=test_clock,
        bucket_configs={
            'chat': (30.0, 10.0),
            'ingest': (30.0, 10.0),
            'default': (120.0, 30.0),
        },
    )
    jwt_verifier = JwtVerifier(settings.jwt, env='test')

    return AppContainer(
        settings=settings,
        clock=test_clock,
        uows=uow_factory,
        ingest=ingest_svc,  # type: ignore[arg-type]
        retrieval=None,  # type: ignore[arg-type]
        chat=chat_svc,  # type: ignore[arg-type]
        lifecycle=lifecycle,
        rate_limiter=limiter,
        jwt_verifier=jwt_verifier,
    )


@pytest.fixture
def app(app_container: AppContainer) -> Any:
    return create_app(settings=app_container.settings, container=app_container)


@pytest.fixture
async def client(app: Any) -> AsyncIterator[httpx.AsyncClient]:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url='http://test') as c:
        yield c


# ---------------------------------------------------------------------------
# Tests: T-API-01 Auth Rejection & Valid Extraction
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_auth_rejection_missing_header(client: httpx.AsyncClient) -> None:
    """Missing Authorization header gives 401 E-AUTH-01."""
    resp = await client.get('/v1/facts')
    assert resp.status_code == 401
    data = resp.json()
    assert 'error' in data
    assert data['error']['code'] == 'E-AUTH-01'
    assert data['error']['retryable'] is False
    assert resp.headers.get('x-request-id') is not None


@pytest.mark.asyncio
async def test_auth_rejection_invalid_token(client: httpx.AsyncClient) -> None:
    """Invalid token signature/payload gives 401 E-AUTH-01."""
    resp = await client.get(
        '/v1/facts',
        headers={'Authorization': 'Bearer this-is-not-a-valid-jwt'},
    )
    assert resp.status_code == 401
    data = resp.json()
    assert data['error']['code'] == 'E-AUTH-01'


@pytest.mark.asyncio
async def test_auth_rejection_expired_token(client: httpx.AsyncClient) -> None:
    """Expired token gives 401 E-AUTH-01."""
    token = make_jwt(uuid4(), expired=True)
    resp = await client.get(
        '/v1/facts',
        headers={'Authorization': f'Bearer {token}'},
    )
    assert resp.status_code == 401
    assert resp.json()['error']['code'] == 'E-AUTH-01'


@pytest.mark.asyncio
async def test_auth_valid_jwt_extraction(client: httpx.AsyncClient) -> None:
    """Valid JWT token extracts authenticated user_id and succeeds (200)."""
    user_id = uuid4()
    token = make_jwt(user_id)
    resp = await client.get(
        '/v1/facts',
        headers={'Authorization': f'Bearer {token}'},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert 'items' in data
    assert isinstance(data['items'], list)


@pytest.mark.asyncio
async def test_auth_dev_mode_debug_user_header(client: httpx.AsyncClient) -> None:
    """X-Debug-User-Id succeeds in dev/test environment without JWT."""
    user_id = uuid4()
    resp = await client.get(
        '/v1/facts',
        headers={'X-Debug-User-Id': str(user_id)},
    )
    assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Tests: T-API-02 Rate Limiting (429 E-RATE-01 with Retry-After)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rate_limiting_exceeded_with_retry_after(
    client: httpx.AsyncClient,
    app_container: AppContainer,
) -> None:
    """Exceeding burst capacity raises RateLimited (429 E-RATE-01) with Retry-After header."""
    user_id = uuid4()
    # Configure tiny burst limit for chat
    app_container.rate_limiter = RateLimiter(
        clock=app_container.clock,
        bucket_configs={'chat': (60.0, 2.0)},  # burst of 2 tokens
    )

    headers = {'X-Debug-User-Id': str(user_id)}
    payload = {'message': 'Hello memory assistant'}

    # First 2 requests succeed
    r1 = await client.post('/v1/chat', json=payload, headers=headers)
    assert r1.status_code == 200
    r2 = await client.post('/v1/chat', json=payload, headers=headers)
    assert r2.status_code == 200

    # 3rd request exceeds burst capacity
    r3 = await client.post('/v1/chat', json=payload, headers=headers)
    assert r3.status_code == 429
    err = r3.json()['error']
    assert err['code'] == 'E-RATE-01'
    assert err['retryable'] is True
    assert 'Retry-After' in r3.headers
    retry_after = int(r3.headers['Retry-After'])
    assert retry_after >= 1


# ---------------------------------------------------------------------------
# Tests: T-API-03 POST /v1/chat and POST /v1/chat/stream SSE Framing
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_chat_stream_sse_formatting(client: httpx.AsyncClient) -> None:
    """POST /v1/chat/stream returns text/event-stream with well-formed SSE frames."""
    user_id = uuid4()
    headers = {'X-Debug-User-Id': str(user_id)}
    payload = {'message': 'Tell me about my travels'}

    resp = await client.post('/v1/chat/stream', json=payload, headers=headers)
    assert resp.status_code == 200
    assert 'text/event-stream' in resp.headers['content-type']

    body_text = resp.text
    # SSE events must be formatted as event: <type>\ndata: <json>\n\n
    assert 'event: plan\n' in body_text
    assert 'event: trace\n' in body_text
    assert 'event: token\n' in body_text
    assert 'event: citations\n' in body_text
    assert 'event: done\n' in body_text


@pytest.mark.asyncio
async def test_chat_non_streaming_json(client: httpx.AsyncClient) -> None:
    """POST /v1/chat without stream request returns standard JSON ChatResponse."""
    user_id = uuid4()
    headers = {'X-Debug-User-Id': str(user_id)}
    payload = {'message': 'What is my favorite color?'}

    resp = await client.post('/v1/chat', json=payload, headers=headers)
    assert resp.status_code == 200
    data = resp.json()
    assert 'conversation_id' in data
    assert 'assistant_message_id' in data
    assert 'content' in data
    assert len(data['content']) > 0


@pytest.mark.asyncio
async def test_chat_sse_via_accept_header(client: httpx.AsyncClient) -> None:
    """POST /v1/chat with Accept: text/event-stream returns SSE stream."""
    user_id = uuid4()
    headers = {
        'X-Debug-User-Id': str(user_id),
        'Accept': 'text/event-stream',
    }
    payload = {'message': 'Stream me an answer'}

    resp = await client.post('/v1/chat', json=payload, headers=headers)
    assert resp.status_code == 200
    assert 'text/event-stream' in resp.headers['content-type']
    assert 'event: token\n' in resp.text


# ---------------------------------------------------------------------------
# Tests: T-API-04 POST /v1/notes Chunking and Ingestion
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_post_notes_creation(client: httpx.AsyncClient) -> None:
    """POST /v1/notes accepts content, chunks, and schedules outbox job (202 Accepted)."""
    user_id = uuid4()
    headers = {'X-Debug-User-Id': str(user_id)}
    payload = {
        'title': 'Meeting Notes',
        'content': 'Discussed quarterly roadmap with Alice and Bob.',
    }

    resp = await client.post('/v1/notes', json=payload, headers=headers)
    assert resp.status_code == 202
    data = resp.json()
    assert data['source_count'] == 2
    assert 'job_id' in data
    assert len(data['source_ids']) == 2


# ---------------------------------------------------------------------------
# Tests: Memory Fact Lifecycle & CRUD Endpoints
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_facts_crud_and_lifecycle(
    client: httpx.AsyncClient,
    uow_factory: FakeUnitOfWorkFactory,
) -> None:
    """Test GET /v1/facts, GET /v1/facts/{id}, PATCH, RETRACT, DELETE."""
    user_id = UserId(uuid4())
    fact_id = FactId(uuid4())
    headers = {'X-Debug-User-Id': str(user_id)}

    # Seed fact in fake repository
    fact = Fact(
        id=fact_id,
        user_id=user_id,
        subject_id=EntityId(uuid4()),
        predicate='lives_in',
        cardinality=Cardinality.SINGLE,
        object_entity_id=None,
        object_value='Paris',
        object_norm='paris',
        fact_text='User lives_in Paris',
        valid_from=utc_now() - timedelta(days=30),
        valid_to=None,
        time_precision=TimePrecision.DAY,
        valid_from_explicit=True,
        recorded_at=utc_now() - timedelta(days=30),
        invalidated_at=None,
        confidence=0.95,
        salience=1.0,
        access_count=2,
        last_accessed_at=None,
        status=FactStatus.ACTIVE,
        origin=FactOrigin.EXTRACTED,
        event_hash=None,
    )
    uow_factory.shared_uow.facts.facts[fact_id] = fact

    # 1. GET /v1/facts
    r_list = await client.get('/v1/facts', headers=headers)
    assert r_list.status_code == 200
    items = r_list.json()['items']
    assert len(items) == 1
    assert items[0]['id'] == str(fact_id)
    assert items[0]['object']['value'] == 'Paris'
    assert items[0]['state'] == 'current'

    # 2. GET /v1/facts/{id}
    r_get = await client.get(f'/v1/facts/{fact_id}', headers=headers)
    assert r_get.status_code == 200
    detail = r_get.json()
    assert detail['id'] == str(fact_id)
    assert 'evidence' in detail
    assert 'chain' in detail

    # 3. PATCH /v1/facts/{id}
    r_patch = await client.patch(
        f'/v1/facts/{fact_id}',
        json={'object_value': 'London'},
        headers=headers,
    )
    assert r_patch.status_code == 200
    assert r_patch.json()['object']['value'] == 'London'

    # 4. POST /v1/facts/{id}/retract
    r_retract = await client.post(
        f'/v1/facts/{fact_id}/retract',
        json={'reason': 'moved away'},
        headers=headers,
    )
    assert r_retract.status_code == 200
    assert r_retract.json()['status'] == 'retracted'

    # 5. DELETE /v1/facts/{id}
    r_del = await client.delete(f'/v1/facts/{fact_id}', headers=headers)
    assert r_del.status_code == 200
    assert r_del.json()['status'] == 'deleted'


@pytest.mark.asyncio
async def test_fact_not_found(client: httpx.AsyncClient) -> None:
    """GET non-existent fact gives 404 E-NF-01."""
    user_id = uuid4()
    missing_id = uuid4()
    headers = {'X-Debug-User-Id': str(user_id)}
    resp = await client.get(f'/v1/facts/{missing_id}', headers=headers)
    assert resp.status_code == 404
    assert resp.json()['error']['code'] == 'E-NF-01'


# ---------------------------------------------------------------------------
# Tests: Outbox Jobs, Profile, and Health Probes
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_jobs_get(
    client: httpx.AsyncClient,
    uow_factory: FakeUnitOfWorkFactory,
) -> None:
    """GET /v1/jobs/{job_id} retrieves job details or 404."""
    user_id = UserId(uuid4())
    job_id = JobId(uuid4())
    headers = {'X-Debug-User-Id': str(user_id)}

    # Missing job -> 404
    r_missing = await client.get(f'/v1/jobs/{job_id}', headers=headers)
    assert r_missing.status_code == 404
    assert r_missing.json()['error']['code'] == 'E-NF-01'

    # Seed job
    job = Job(
        id=job_id,
        user_id=user_id,
        type=JobType.EXTRACT,
        payload={'source_ids': [str(uuid4())]},
        idempotency_key='test-key',
        coalesce_key=None,
        status=JobStatus.QUEUED,
        attempts=0,
        max_attempts=3,
        deferrals=0,
        run_after=utc_now(),
        worker_id=None,
        lease_expires_at=None,
        last_error=None,
    )
    uow_factory.shared_uow.jobs.jobs[job_id] = job

    # Existing job -> 200 JobView
    r_job = await client.get(f'/v1/jobs/{job_id}', headers=headers)
    assert r_job.status_code == 200
    assert r_job.json()['id'] == str(job_id)
    assert r_job.json()['status'] == 'queued'


@pytest.mark.asyncio
async def test_profile_export_and_purge(client: httpx.AsyncClient) -> None:
    """GET /v1/profile/export and DELETE /v1/profile."""
    user_id = uuid4()
    headers = {'X-Debug-User-Id': str(user_id)}

    # Export
    r_exp = await client.get('/v1/profile/export', headers=headers)
    assert r_exp.status_code == 200
    assert r_exp.json()['user_id'] == str(user_id)

    # Purge
    r_purge = await client.delete('/v1/profile', headers=headers)
    assert r_purge.status_code == 200
    assert r_purge.json()['status'] == 'purged'
    assert r_purge.json()['purged_rows'] == 10


@pytest.mark.asyncio
async def test_health_and_readiness_no_auth(client: httpx.AsyncClient) -> None:
    """Healthz and Readyz respond without authentication."""
    r_health = await client.get('/healthz')
    assert r_health.status_code == 200
    assert r_health.json()['status'] == 'ok'

    r_ready = await client.get('/readyz')
    assert r_ready.status_code == 200
    assert r_ready.json()['status'] == 'ready'


# ---------------------------------------------------------------------------
# Tests: Middleware (PayloadSizeLimit & RequestId)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_payload_size_limit_rejection(client: httpx.AsyncClient) -> None:
    """Payload over size limit returns 413 E-REQ-01."""
    user_id = uuid4()
    headers = {'X-Debug-User-Id': str(user_id)}

    # Max body limit for /v1/chat is 256 KB. Send 300 KB string
    huge_message = 'A' * (300 * 1024)
    resp = await client.post(
        '/v1/chat',
        json={'message': huge_message},
        headers=headers,
    )
    assert resp.status_code == 413
    assert resp.json()['error']['code'] == 'E-REQ-01'


@pytest.mark.asyncio
async def test_request_id_echo(client: httpx.AsyncClient) -> None:
    """X-Request-Id header is echoed back on every response."""
    custom_req_id = 'req-trace-12345'
    resp = await client.get(
        '/healthz',
        headers={'X-Request-Id': custom_req_id},
    )
    assert resp.headers.get('x-request-id') == custom_req_id
