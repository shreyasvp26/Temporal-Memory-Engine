"""Unit tests for ChatService and MemoryLifecycleService (M5-02).

Verifies:
  1. Citation parsing ([F1], [S2] mapping to FactId and SourceId).
  2. ChatService statement-only bypass (skips retrieval/LLM, returns ack, enqueues turn).
  3. ChatService abstention short-circuit (canned reply, no LLM call).
  4. ChatService full retrieval + synthesis with citation extraction and turn enqueueing.
  5. ChatService streaming SSE event sequences (plan, trace, token, citations, done, error).
  6. MemoryLifecycleService list_facts, get_fact_with_evidence (and NotFound).
  7. MemoryLifecycleService update_fact (re-rendering, re-embedding, audit log).
  8. MemoryLifecycleService retract_fact (closing timeline, status retracted, audit log).
  9. MemoryLifecycleService delete_fact (soft delete tombstone, audit log).
  10. MemoryLifecycleService purge_user and export_user (Q-16 hard-purge and GDPR export).
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

if TYPE_CHECKING:
    import builtins
    from collections.abc import AsyncIterator, Sequence
    from types import TracebackType

import pytest
from memory_engine.domain.clock import FixedClock
from memory_engine.domain.enums import (
    Cardinality,
    EntityType,
    FactOrigin,
    FactStatus,
    Intent,
    Role,
    TimePrecision,
)
from memory_engine.domain.errors import InvariantViolation, NotFound
from memory_engine.domain.ids import EntityId, FactId, JobId, SourceId, UserId
from memory_engine.domain.models import Entity, Fact, Message
from memory_engine.ingest.service import IngestService
from memory_engine.lifecycle.service import MemoryLifecycleService
from memory_engine.ports.embed import Embedder
from memory_engine.ports.llm import LLMRequest, LLMResponse
from memory_engine.ports.repositories import (
    AuditRepository,
    EvidenceSnippet,
    FactListQuery,
    InsertedMessage,
    InsertedSource,
    NewMessage,
    Page,
    UpsertResult,
)
from memory_engine.retrieve.abstain import AbstentionDecision
from memory_engine.retrieve.assembler import AssembledContext, TraceItem
from memory_engine.retrieve.chat_service import (
    ChatRequest,
    ChatService,
    ChatStreamEvent,
    parse_citations,
)
from memory_engine.retrieve.orchestrator import RetrievalResult
from memory_engine.retrieve.planner.rules import QueryPlan


def utc(
    year: int,
    month: int = 1,
    day: int = 1,
    hour: int = 0,
    minute: int = 0,
) -> datetime:
    """Helper constructing UTC datetime."""
    return datetime(year, month, day, hour, minute, tzinfo=UTC)


def make_fact(
    user_id: UserId,
    fact_id: FactId | None = None,
    predicate: str = 'lives_in',
    object_value: str = 'Pune',
    valid_from: datetime | None = None,
    status: FactStatus = FactStatus.ACTIVE,
) -> Fact:
    """Helper constructing test Fact."""
    fid = fact_id or FactId(uuid4())
    v_from = valid_from or utc(2026, 1, 1)
    return Fact(
        id=fid,
        user_id=user_id,
        subject_id=EntityId(uuid4()),
        predicate=predicate,
        cardinality=Cardinality.SINGLE,
        object_entity_id=None,
        object_value=object_value,
        object_norm=object_value.strip().lower(),
        fact_text=f'User {predicate} {object_value}',
        valid_from=v_from,
        valid_to=None,
        time_precision=TimePrecision.DAY,
        valid_from_explicit=True,
        recorded_at=utc(2026, 1, 1),
        invalidated_at=None,
        confidence=0.9,
        salience=1.0,
        access_count=0,
        last_accessed_at=None,
        status=status,
        origin=FactOrigin.EXTRACTED,
        event_hash=None,
    )


class FakeFactsRepo:
    """In-memory fake FactRepository for lifecycle and chat tests."""

    def __init__(self) -> None:
        self.facts: dict[FactId, Fact] = {}
        self.evidence: dict[FactId, list[EvidenceSnippet]] = {}
        self.chains: dict[FactId, list[Fact]] = {}
        self.manual_updates: list[dict[str, Any]] = []

    async def get(self, user_id: UserId, fact_id: FactId) -> Fact | None:
        return self.facts.get(fact_id)

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
        call_record = {
            'user_id': user_id,
            'fact_id': fact_id,
            'fact_text': fact_text,
            'object_value': object_value,
            'object_norm': object_norm,
            'valid_from': valid_from,
            'valid_to': valid_to,
            'status': status,
            'confidence': confidence,
            'invalidated_at': invalidated_at,
        }
        self.manual_updates.append(call_record)

        if fact_id in self.facts:
            f = self.facts[fact_id]
            new_status: FactStatus = f.status
            if status is not None:
                if isinstance(status, FactStatus):
                    new_status = status
                elif status == 'deleted':
                    new_status = FactStatus.DELETED
                elif status in ('archived', 'retracted'):
                    new_status = FactStatus.ARCHIVED

            updated_fact = Fact(
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
            self.facts[fact_id] = updated_fact


class FakeMessagesRepo:
    """In-memory fake MessageRepository for chat service tests."""

    def __init__(self) -> None:
        self.messages: list[NewMessage] = []
        self.recent: list[Message] = []
        self.pending: list[Message] = []

    async def ensure_conversation(
        self,
        user_id: UserId,
        conversation_id: UUID | None,
        now: datetime,
    ) -> UUID:
        return conversation_id or uuid4()

    async def insert(self, m: NewMessage) -> InsertedMessage:
        self.messages.append(m)
        return InsertedMessage(id=uuid4(), created=True)

    async def insert_message(self, m: NewMessage) -> InsertedMessage:
        return await self.insert(m)

    async def recent_turns(
        self,
        user_id: UserId,
        conversation_id: UUID,
        n: int,
    ) -> list[Message]:
        return list(self.recent[:n])

    async def pending_user_messages(
        self,
        user_id: UserId,
        limit: int,
    ) -> list[Message]:
        return list(self.pending[:limit])

    async def finalize_assistant(
        self,
        user_id: UserId,
        message_id: UUID,
        content: str,
        partial: bool,
        trace: dict[str, Any] | None,
    ) -> None:
        pass


class FakeAuditRepo(AuditRepository):
    """In-memory fake AuditRepository."""

    def __init__(self) -> None:
        self.logs: list[dict[str, Any]] = []

    async def append(
        self,
        user_id: UserId,
        actor: str,
        action: str,
        target_type: str,
        target_id: UUID | None,
        before: dict[str, Any] | None,
        after: dict[str, Any] | None,
    ) -> None:
        self.logs.append(
            {
                'user_id': user_id,
                'actor': actor,
                'action': action,
                'target_type': target_type,
                'target_id': target_id,
                'before': before,
                'after': after,
            }
        )

    async def record(
        self,
        user_id: UserId,
        actor: str,
        action: str,
        target_type: str,
        target_id: UUID | None = None,
        before: dict[str, Any] | None = None,
        after: dict[str, Any] | None = None,
    ) -> None:
        await self.append(user_id, actor, action, target_type, target_id, before, after)


class FakeEntitiesRepo:
    """In-memory fake EntityRepository."""

    def __init__(self) -> None:
        self.entities: dict[EntityId, Entity] = {}

    async def ensure_self(self, user_id: UserId) -> Entity:
        self_e = Entity(
            id=EntityId(user_id),
            user_id=user_id,
            canonical_name='User',
            type=EntityType.PERSON,
            is_self=True,
        )
        self.entities[self_e.id] = self_e
        return self_e

    async def get_many(
        self,
        user_id: UserId,
        ids: Sequence[EntityId],
    ) -> dict[EntityId, Entity]:
        return {eid: self.entities[eid] for eid in ids if eid in self.entities}


class FakeProfilesRepo:
    """In-memory fake ProfileRepository."""

    def __init__(self) -> None:
        self.purged: list[UserId] = []
        self.exported: dict[str, Any] = {'status': 'ok'}

    async def ensure(self, user_id: UserId) -> None:
        pass

    async def timezone(self, user_id: UserId) -> str:
        return 'UTC'

    async def set_timezone(self, user_id: UserId, tz: str) -> None:
        pass

    async def list_user_ids(self) -> list[UserId]:
        return []

    async def purge_user(self, user_id: UserId) -> int:
        self.purged.append(user_id)
        return 5

    async def export_user(self, user_id: UserId) -> dict[str, Any]:
        return {**self.exported, 'user_id': str(user_id)}


class FakeSourcesRepo:
    def __init__(self) -> None:
        self.sources: list[Any] = []

    async def insert_many(self, items: Sequence[Any]) -> list[InsertedSource]:
        res: list[InsertedSource] = []
        for it in items:
            sid = getattr(it, 'id', None) or SourceId(uuid4())
            res.append(InsertedSource(id=sid, created=True))
        return res


class FakeJobsRepo:
    def __init__(self) -> None:
        self.jobs: list[Any] = []

    async def upsert_coalesced(self, j: Any, debounce_s: float) -> UpsertResult:
        jid = JobId(uuid4())
        self.jobs.append(j)
        return UpsertResult(job_id=jid, created=True)


class FakeUnitOfWork:
    """Fake UoW coordinating fake repos."""

    def __init__(self) -> None:
        self.facts: Any = FakeFactsRepo()
        self.messages: Any = FakeMessagesRepo()
        self.audit: Any = FakeAuditRepo()
        self.entities: Any = FakeEntitiesRepo()
        self.profiles: Any = FakeProfilesRepo()
        self.sources: Any = FakeSourcesRepo()
        self.jobs: Any = FakeJobsRepo()
        self.search: Any = None
        self.locked_users: list[UserId | None] = []

    async def lock_user(self, user_id: UserId | None = None) -> None:
        self.locked_users.append(user_id)

    async def __aenter__(self) -> FakeUnitOfWork:
        return self

    async def __aexit__(
        self,
        et: type[BaseException] | None,
        ev: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        pass


class FakeUnitOfWorkFactory:
    """UoW factory yielding shared or fresh FakeUnitOfWork."""

    def __init__(self, shared_uow: FakeUnitOfWork | None = None) -> None:
        self.shared_uow = shared_uow or FakeUnitOfWork()

    def begin(self, user_id: UserId | None, *, readonly: bool = False) -> Any:
        return self.shared_uow

    def for_user(self, user_id: UserId, *, readonly: bool = False) -> Any:
        return self.shared_uow

    def system(self, *, readonly: bool = False) -> Any:
        return self.shared_uow


class FakeLLM:
    """Configurable fake LLMGateway."""

    def __init__(
        self,
        response_text: str = 'Assistant response [F1].',
        stream_chunks: Sequence[str] | None = None,
        should_fail: bool = False,
    ) -> None:
        self.response_text = response_text
        self.stream_chunks = (
            list(stream_chunks) if stream_chunks is not None else ['Chunk 1 ', 'Chunk 2 [F1]']
        )
        self.should_fail = should_fail
        self.received_requests: list[LLMRequest] = []

    async def complete_text(self, req: LLMRequest) -> LLMResponse:
        self.received_requests.append(req)
        if self.should_fail:
            raise RuntimeError('LLM synthesis failed')
        return LLMResponse(
            text=self.response_text,
            model='test-model',
            provider='fake',
            prompt_tokens=100,
            completion_tokens=20,
            latency_ms=45.0,
        )

    async def complete_json(self, req: LLMRequest, schema: type[Any]) -> Any:
        raise NotImplementedError

    async def stream_text(self, req: LLMRequest) -> AsyncIterator[str]:
        self.received_requests.append(req)
        if self.should_fail:
            raise RuntimeError('LLM stream interrupted')
        for chunk in self.stream_chunks:
            yield chunk


class FakeEmbedder(Embedder):
    """In-memory fake embedder."""

    model_id: str = 'test-embedder'
    dim: int = 768

    def __init__(self) -> None:
        self.embedded: list[str] = []

    async def embed_query(self, text: str) -> list[float]:
        self.embedded.append(text)
        return [0.1] * 768

    async def embed_queries(self, texts: Sequence[str]) -> list[list[float]]:
        self.embedded.extend(texts)
        return [[0.1] * 768 for _ in texts]

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        self.embedded.extend(texts)
        return [[0.1] * 768 for _ in texts]


class FakeRetrievalOrchestrator:
    """Configurable fake RetrievalOrchestrator for ChatService tests."""

    def __init__(
        self,
        canned_result: RetrievalResult | None = None,
        should_abstain: bool = False,
    ) -> None:
        self.canned_result = canned_result
        self.should_abstain = should_abstain
        self.called_with: list[dict[str, Any]] = []

    async def retrieve(
        self,
        user_id: UserId,
        query: str | QueryPlan,
        question_ts: datetime | None = None,
        tz: Any = None,
        known_entities: Sequence[Any] = (),
        recent: Sequence[Any] = (),
        pending: Sequence[Any] = (),
        statement_only: bool = False,
    ) -> RetrievalResult:
        self.called_with.append(
            {
                'user_id': user_id,
                'query': query,
                'statement_only': statement_only,
            }
        )
        if self.canned_result is not None:
            return self.canned_result

        plan = QueryPlan(
            question=str(query),
            question_ts=question_ts or utc(2026, 1, 1),
            tz='UTC',
            intent=Intent.CURRENT_STATE,
            as_of=None,
            time_range=None,
            entity_ids=(),
            entity_names=(),
            predicates=(),
            multi_hop=False,
            expects_count=False,
            ambiguous_time=False,
            planner_path='rules',
            confidence=0.9,
        )

        f1_id = FactId(uuid4())
        s1_id = SourceId(uuid4())
        ctx = AssembledContext(
            text='Today: 2026-10-21\nMEMORY:\n[F1] User lives in Pune\n[S1] raw snippet',
            fact_labels={'F1': f1_id},
            source_labels={'S1': s1_id},
            traces=(
                TraceItem(label='F1', kind='fact', id=f1_id),
                TraceItem(label='S1', kind='source', id=s1_id),
            ),
            used_fact_ids=(f1_id,),
        )

        abstention = (
            AbstentionDecision.abstain(reason='no_evidence')
            if self.should_abstain
            else AbstentionDecision.proceed()
        )

        return RetrievalResult(
            plan=plan,
            ranked_items=[],
            context=ctx,
            abstention=abstention,
            lane_counts={'vector_facts': 1},
            degraded_lanes=[],
        )


# =============================================================================
# 1. Citation Parsing Tests
# =============================================================================


def test_parse_citations_extracts_and_maps_facts_and_sources() -> None:
    f1 = FactId(uuid4())
    f2 = FactId(uuid4())
    s1 = SourceId(uuid4())
    ctx = AssembledContext(
        text='MEMORY:\n[F1] Fact 1\n[F2] Fact 2\n[S1] Source 1',
        fact_labels={'F1': f1, 'F2': f2},
        source_labels={'S1': s1},
        traces=(
            TraceItem(label='F1', kind='fact', id=f1),
            TraceItem(label='F2', kind='fact', id=f2),
            TraceItem(label='S1', kind='source', id=s1),
        ),
        used_fact_ids=(f1, f2),
    )

    text = (
        'You moved to Pune according to [F1] and confirmed in [S1]. '
        'Later verified in [F2] and [F1].'
    )
    cited_facts, cited_sources = parse_citations(text, ctx)

    assert cited_facts == [f1, f2]
    assert cited_sources == [s1]


def test_parse_citations_handles_empty_and_unmapped_tags() -> None:
    assert parse_citations('') == ([], [])
    assert parse_citations('No citations here.') == ([], [])

    ctx = AssembledContext(
        text='',
        fact_labels={},
        source_labels={},
        traces=(),
        used_fact_ids=(),
    )
    # [F99] not present in context traces
    facts, sources = parse_citations('Check [F99] and [S42]', ctx)
    assert facts == []
    assert sources == []


# =============================================================================
# 2. ChatService Tests
# =============================================================================


@pytest.mark.asyncio
async def test_chat_service_statement_only_bypass() -> None:
    clock = FixedClock(utc(2026, 10, 21))
    uow = FakeUnitOfWork()
    uow_factory = FakeUnitOfWorkFactory(uow)
    llm = FakeLLM()
    retrieval = FakeRetrievalOrchestrator()
    ingest = IngestService(clock=clock, uows=uow_factory)

    chat = ChatService(
        uows=uow_factory,
        retrieval=retrieval,  # type: ignore[arg-type]
        ingestion=ingest,
        llm=llm,
        clock=clock,
    )

    user_id = UserId(uuid4())
    # Pure statement containing no interrogative or question marks
    req = ChatRequest(message='I just moved from Delhi to Pune last week.')

    result = await chat.respond(user_id, req)

    assert result.statement_only is True
    assert 'noted that down' in result.content.lower()
    # Retrieval and LLM should have been bypassed
    assert len(retrieval.called_with) == 0
    assert len(llm.received_requests) == 0
    # Both user message and assistant acknowledgment were inserted
    assert len(uow.messages.messages) == 2
    assert uow.messages.messages[0].role == Role.USER
    assert uow.messages.messages[1].role == Role.ASSISTANT


@pytest.mark.asyncio
async def test_chat_service_abstention_short_circuit() -> None:
    clock = FixedClock(utc(2026, 10, 21))
    uow = FakeUnitOfWork()
    uow_factory = FakeUnitOfWorkFactory(uow)
    llm = FakeLLM()
    retrieval = FakeRetrievalOrchestrator(should_abstain=True)
    ingest = IngestService(clock=clock, uows=uow_factory)

    chat = ChatService(
        uows=uow_factory,
        retrieval=retrieval,  # type: ignore[arg-type]
        ingestion=ingest,
        llm=llm,
        clock=clock,
    )

    user_id = UserId(uuid4())
    req = ChatRequest(message='Where do I live right now?')

    result = await chat.respond(user_id, req)

    assert result.abstained is True
    assert "don't have anything in your memory" in result.content.lower()
    # Retrieval was called, but LLM synthesis was skipped
    assert len(retrieval.called_with) == 1
    assert len(llm.received_requests) == 0
    assert len(uow.messages.messages) == 2


@pytest.mark.asyncio
async def test_chat_service_full_retrieval_and_llm_response() -> None:
    clock = FixedClock(utc(2026, 10, 21))
    uow = FakeUnitOfWork()
    uow_factory = FakeUnitOfWorkFactory(uow)
    llm = FakeLLM(response_text='You live in Pune based on [F1].')
    retrieval = FakeRetrievalOrchestrator(should_abstain=False)
    ingest = IngestService(clock=clock, uows=uow_factory)

    chat = ChatService(
        uows=uow_factory,
        retrieval=retrieval,  # type: ignore[arg-type]
        ingestion=ingest,
        llm=llm,
        clock=clock,
    )

    user_id = UserId(uuid4())
    req = ChatRequest(message='Where do I currently live?')

    result = await chat.respond(user_id, req)

    assert result.statement_only is False
    assert result.abstained is False
    assert 'You live in Pune' in result.content
    assert len(result.cited_fact_ids) == 1
    assert len(retrieval.called_with) == 1
    assert len(llm.received_requests) == 1
    assert llm.received_requests[0].purpose == 'synth'
    assert len(uow.messages.messages) == 2


@pytest.mark.asyncio
async def test_chat_service_streaming_event_sequence_statement_only() -> None:
    clock = FixedClock(utc(2026, 10, 21))
    uow = FakeUnitOfWork()
    uow_factory = FakeUnitOfWorkFactory(uow)
    llm = FakeLLM()
    retrieval = FakeRetrievalOrchestrator()
    ingest = IngestService(clock=clock, uows=uow_factory)

    chat = ChatService(
        uows=uow_factory,
        retrieval=retrieval,  # type: ignore[arg-type]
        ingestion=ingest,
        llm=llm,
        clock=clock,
    )

    user_id = UserId(uuid4())
    req = ChatRequest(message='I just started reading Neuromancer.')

    events: list[ChatStreamEvent] = []
    async for ev in chat.stream_respond(user_id, req):
        events.append(ev)

    event_types = [e.event for e in events]
    assert event_types == ['plan', 'token', 'done']
    assert isinstance(events[0].data, dict)
    assert events[0].data.get('statement_only') is True
    assert isinstance(events[1].data, dict)
    assert 'noted that down' in events[1].data['text'].lower()
    assert isinstance(events[2].data, dict)
    assert events[2].data['partial'] is False


@pytest.mark.asyncio
async def test_chat_service_streaming_event_sequence_full_with_citations() -> None:
    clock = FixedClock(utc(2026, 10, 21))
    uow = FakeUnitOfWork()
    uow_factory = FakeUnitOfWorkFactory(uow)
    llm = FakeLLM(stream_chunks=['You live in Pune ', 'as recorded in [F1].'])
    retrieval = FakeRetrievalOrchestrator(should_abstain=False)
    ingest = IngestService(clock=clock, uows=uow_factory)

    chat = ChatService(
        uows=uow_factory,
        retrieval=retrieval,  # type: ignore[arg-type]
        ingestion=ingest,
        llm=llm,
        clock=clock,
    )

    user_id = UserId(uuid4())
    req = ChatRequest(message='What city do I live in?')

    events: list[ChatStreamEvent] = []
    async for ev in chat.stream_respond(user_id, req):
        events.append(ev)

    event_types = [e.event for e in events]
    assert event_types == ['plan', 'trace', 'token', 'token', 'citations', 'done']
    assert isinstance(events[4].data, dict)
    assert len(events[4].data['cited_fact_ids']) == 1
    assert isinstance(events[5].data, dict)
    assert events[5].data['partial'] is False


@pytest.mark.asyncio
async def test_chat_service_streaming_error_handling() -> None:
    clock = FixedClock(utc(2026, 10, 21))
    uow = FakeUnitOfWork()
    uow_factory = FakeUnitOfWorkFactory(uow)
    llm = FakeLLM(should_fail=True)
    retrieval = FakeRetrievalOrchestrator(should_abstain=False)
    ingest = IngestService(clock=clock, uows=uow_factory)

    chat = ChatService(
        uows=uow_factory,
        retrieval=retrieval,  # type: ignore[arg-type]
        ingestion=ingest,
        llm=llm,
        clock=clock,
    )

    user_id = UserId(uuid4())
    req = ChatRequest(message='Where do I live?')

    events: list[ChatStreamEvent] = []
    async for ev in chat.stream_respond(user_id, req):
        events.append(ev)

    event_types = [e.event for e in events]
    assert 'error' in event_types
    assert events[-1].event == 'done'
    assert isinstance(events[-1].data, dict)
    assert events[-1].data['partial'] is True


# =============================================================================
# 3. MemoryLifecycleService Tests
# =============================================================================


@pytest.mark.asyncio
async def test_lifecycle_list_facts() -> None:
    clock = FixedClock(utc(2026, 10, 21))
    uow = FakeUnitOfWork()
    uow_factory = FakeUnitOfWorkFactory(uow)
    service = MemoryLifecycleService(uows=uow_factory, clock=clock)

    user_id = UserId(uuid4())
    f1 = make_fact(user_id, object_value='Delhi')
    f2 = make_fact(user_id, object_value='Pune')
    uow.facts.facts[f1.id] = f1
    uow.facts.facts[f2.id] = f2

    page = await service.list_facts(user_id, FactListQuery(page_size=10))
    assert len(page.items) == 2


@pytest.mark.asyncio
async def test_lifecycle_get_fact_with_evidence() -> None:
    clock = FixedClock(utc(2026, 10, 21))
    uow = FakeUnitOfWork()
    uow_factory = FakeUnitOfWorkFactory(uow)
    service = MemoryLifecycleService(uows=uow_factory, clock=clock)

    user_id = UserId(uuid4())
    f1 = make_fact(user_id, object_value='Delhi')
    uow.facts.facts[f1.id] = f1

    ev = EvidenceSnippet(
        source_id=SourceId(uuid4()),
        source_ts=utc(2026, 1, 1),
        role=Role.USER,
        snippet='I live in Delhi',
        fact_id=f1.id,
    )
    uow.facts.evidence[f1.id] = [ev]
    uow.facts.chains[f1.id] = [f1]

    fact, snippets, chain = await service.get_fact_with_evidence(user_id, f1.id)
    assert fact.id == f1.id
    assert len(snippets) == 1
    assert snippets[0].snippet == 'I live in Delhi'
    assert len(chain) == 1

    # Non-existent fact raises NotFound
    missing_id = FactId(uuid4())
    with pytest.raises(NotFound):
        await service.get_fact_with_evidence(user_id, missing_id)


@pytest.mark.asyncio
async def test_lifecycle_update_fact_rerenders_and_creates_audit_log() -> None:
    clock = FixedClock(utc(2026, 10, 21))
    uow = FakeUnitOfWork()
    uow_factory = FakeUnitOfWorkFactory(uow)
    embedder = FakeEmbedder()
    service = MemoryLifecycleService(uows=uow_factory, clock=clock, embedder=embedder)

    user_id = UserId(uuid4())
    f1 = make_fact(user_id, object_value='Delhi', valid_from=utc(2025, 1, 1))
    uow.facts.facts[f1.id] = f1

    updated = await service.update_fact(
        user_id=user_id,
        fact_id=f1.id,
        object_value='Bangalore',
        valid_from=utc(2025, 6, 1),
        valid_to=utc(2026, 1, 1),
    )

    assert updated.object_value == 'Bangalore'
    assert updated.valid_from == utc(2025, 6, 1)
    assert updated.valid_to == utc(2026, 1, 1)

    # Verify embedder was invoked outside tx
    assert len(embedder.embedded) == 1

    # Verify audit log was recorded
    assert len(uow.audit.logs) == 1
    log = uow.audit.logs[0]
    assert log['action'] == 'fact.manual_update'
    assert log['target_id'] == f1.id
    assert log['before']['object_value'] == 'Delhi'
    assert log['after']['object_value'] == 'Bangalore'

    # InvariantViolation if valid_to <= valid_from
    with pytest.raises(InvariantViolation):
        await service.update_fact(
            user_id=user_id,
            fact_id=f1.id,
            object_value='Mumbai',
            valid_from=utc(2026, 1, 1),
            valid_to=utc(2025, 1, 1),
        )


@pytest.mark.asyncio
async def test_lifecycle_retract_fact_sets_status_and_audit_log() -> None:
    clock = FixedClock(utc(2026, 10, 21))
    uow = FakeUnitOfWork()
    uow_factory = FakeUnitOfWorkFactory(uow)
    service = MemoryLifecycleService(uows=uow_factory, clock=clock)

    user_id = UserId(uuid4())
    f1 = make_fact(user_id, object_value='Delhi')
    uow.facts.facts[f1.id] = f1

    await service.retract_fact(user_id, f1.id, reason='User left Delhi')

    # Fact timeline closed
    assert len(uow.facts.manual_updates) == 1
    update_call = uow.facts.manual_updates[0]
    assert update_call['status'] == 'retracted'
    assert update_call['valid_to'] == clock.now()
    assert update_call['invalidated_at'] == clock.now()

    # Audit log recorded
    assert len(uow.audit.logs) == 1
    log = uow.audit.logs[0]
    assert log['action'] == 'fact.retract'
    assert log['target_id'] == f1.id
    assert log['after']['reason'] == 'User left Delhi'

    # NotFound when missing
    with pytest.raises(NotFound):
        await service.retract_fact(user_id, FactId(uuid4()), reason='test')


@pytest.mark.asyncio
async def test_lifecycle_delete_fact_soft_deletes_with_tombstone() -> None:
    clock = FixedClock(utc(2026, 10, 21))
    uow = FakeUnitOfWork()
    uow_factory = FakeUnitOfWorkFactory(uow)
    service = MemoryLifecycleService(uows=uow_factory, clock=clock)

    user_id = UserId(uuid4())
    f1 = make_fact(user_id, object_value='Delhi')
    uow.facts.facts[f1.id] = f1

    await service.delete_fact(user_id, f1.id)

    assert len(uow.facts.manual_updates) == 1
    update_call = uow.facts.manual_updates[0]
    assert update_call['status'] == FactStatus.DELETED
    assert update_call['invalidated_at'] == clock.now()

    # Audit log
    assert len(uow.audit.logs) == 1
    log = uow.audit.logs[0]
    assert log['action'] == 'fact.delete'
    assert log['target_id'] == f1.id

    # NotFound when missing
    with pytest.raises(NotFound):
        await service.delete_fact(user_id, FactId(uuid4()))


@pytest.mark.asyncio
async def test_lifecycle_purge_and_export_user() -> None:
    clock = FixedClock(utc(2026, 10, 21))
    uow = FakeUnitOfWork()
    uow_factory = FakeUnitOfWorkFactory(uow)
    service = MemoryLifecycleService(uows=uow_factory, clock=clock)

    user_id = UserId(uuid4())

    # Export user data
    export_data = await service.export_user(user_id)
    assert export_data['status'] == 'ok'
    assert export_data['user_id'] == str(user_id)

    # Purge user data
    purged_count = await service.purge_user(user_id)
    assert purged_count == 5
    assert user_id in uow.profiles.purged

    # Audit log
    assert len(uow.audit.logs) == 1
    log = uow.audit.logs[0]
    assert log['action'] == 'user.purge'
    assert log['target_id'] == user_id
    assert log['after']['purged_rows'] == 5
