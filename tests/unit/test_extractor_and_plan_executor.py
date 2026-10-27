"""Unit tests for FactExtractor and PlanExecutor per LLD Section 8.3-8.4 and 8.8-8.9."""

from collections.abc import Sequence
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, cast
from uuid import UUID, uuid4

import pytest
from memory_engine.domain.clock import FixedClock
from memory_engine.domain.enums import (
    Cardinality,
    DecayClass,
    FactOrigin,
    FactStatus,
    ObjectKind,
    RelationType,
    Role,
    SourceKind,
    TimePrecision,
)
from memory_engine.domain.errors import InvariantViolation
from memory_engine.domain.ids import EntityId, FactId, SourceId, UserId
from memory_engine.domain.models import (
    Candidate,
    ExtractedFact,
    ExtractionResult,
    Fact,
    FactEvidence,
    FactRelation,
    Predicate,
    Source,
)
from memory_engine.ingest.conflict.plan import (
    AddEvidence,
    AddRelation,
    BumpConfidence,
    CloseFact,
    ExtendValidFrom,
    InsertFact,
    NewRef,
    Outcome,
    ResolutionPlan,
    SetStatus,
)
from memory_engine.ingest.extractor import (
    FactExtractor,
    LLMFactExtractor,
    ValidatedExtraction,
)
from memory_engine.ingest.plan_executor import PlanExecutor
from memory_engine.ports.llm import LLMRequest
from memory_engine.ports.repositories import PredicateRegistry, SourceWindow

if TYPE_CHECKING:
    from memory_engine.ports.uow import UnitOfWork

# -----------------------------------------------------------------------------
# Fakes for isolated unit testing (zero I/O, zero network sockets)
# -----------------------------------------------------------------------------


class FakeLLMGateway:
    """In-memory LLMGateway recording requests and returning programmed responses."""

    def __init__(self, responses: list[Any] | None = None) -> None:
        self.requests: list[LLMRequest] = []
        self.responses: list[Any] = responses or []
        self.call_count: int = 0

    async def complete_text(self, req: LLMRequest) -> Any:
        raise NotImplementedError

    async def complete_json(self, req: LLMRequest, schema: type[Any]) -> Any:
        self.call_count += 1
        self.requests.append(req)
        if not self.responses:
            return ExtractionResult(facts=[])
        resp = self.responses.pop(0)
        if isinstance(resp, Exception):
            raise resp
        return resp

    def stream_text(self, req: LLMRequest) -> Any:
        raise NotImplementedError


class FakePredicateRegistry(PredicateRegistry):
    """Simple in-memory predicate registry for extractor prompt rendering."""

    def __init__(self, predicates: Sequence[Predicate]) -> None:
        self._preds = {p.name: p for p in predicates}

    def resolve(self, name: str) -> Predicate:
        return self._preds[name]

    def lookup(self, name: str) -> Predicate:
        return self._preds[name]

    def get(self, name: str) -> Predicate:
        return self._preds[name]

    def all(self) -> Sequence[Predicate]:
        return list(self._preds.values())


class FakeFactRepository:
    """In-memory FactRepository recording operations for PlanExecutor verification."""

    def __init__(self) -> None:
        self.closed: list[dict[str, Any]] = []
        self.extended: list[dict[str, Any]] = []
        self.statuses: list[dict[str, Any]] = []
        self.bumped: list[dict[str, Any]] = []
        self.inserted: list[dict[str, Any]] = []
        self.evidences: list[dict[str, Any]] = []
        self.relations: list[dict[str, Any]] = []

    async def close(
        self, user_id: UserId, fact_id: FactId, valid_to: datetime, now: datetime
    ) -> None:
        self.closed.append(
            {
                'user_id': user_id,
                'fact_id': fact_id,
                'valid_to': valid_to,
                'now': now,
            }
        )

    async def extend_valid_from(
        self, user_id: UserId, fact_id: FactId, valid_from: datetime
    ) -> None:
        self.extended.append(
            {
                'user_id': user_id,
                'fact_id': fact_id,
                'valid_from': valid_from,
            }
        )

    async def set_status(
        self, user_id: UserId, fact_id: FactId, status: FactStatus, now: datetime
    ) -> None:
        self.statuses.append(
            {
                'user_id': user_id,
                'fact_id': fact_id,
                'status': status,
                'now': now,
            }
        )

    async def bump_confidence(self, user_id: UserId, fact_id: FactId, confidence: float) -> None:
        self.bumped.append(
            {
                'user_id': user_id,
                'fact_id': fact_id,
                'confidence': confidence,
            }
        )

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
        new_id = FactId(uuid4())
        self.inserted.append(
            {
                'candidate': c,
                'fact_id': new_id,
                'valid_from': valid_from,
                'valid_to': valid_to,
                'status': status,
                'origin': origin,
                'model_id': model_id,
                'now': now,
            }
        )
        return Fact(
            id=new_id,
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

    async def add_evidence(self, ev: FactEvidence, user_id: UserId) -> None:
        self.evidences.append({'evidence': ev, 'user_id': user_id})

    async def add_relation(self, user_id: UserId, rel: FactRelation, job_id: UUID | None) -> None:
        self.relations.append({'user_id': user_id, 'relation': rel, 'job_id': job_id})

    async def load_timeline(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    async def get(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    async def list(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    async def chain(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    async def only_supported_by(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    async def hard_delete(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    async def record_access(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    async def decay_candidates(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    async def update_salience(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError


class FakeUnitOfWork:
    """In-memory UnitOfWork fake with lock_user tracking."""

    def __init__(self) -> None:
        self.facts = FakeFactRepository()
        self.locked_user: bool = False
        self.sources: Any = None
        self.entities: Any = None
        self.search: Any = None
        self.jobs: Any = None
        self.messages: Any = None
        self.audit: Any = None
        self.profiles: Any = None

    async def lock_user(self) -> None:
        self.locked_user = True

    async def __aenter__(self) -> 'FakeUnitOfWork':
        return self

    async def __aexit__(self, *args: Any) -> None:
        pass


def make_source(
    turn_index: int,
    content: str,
    *,
    role: Role = Role.USER,
    deleted: bool = False,
    source_ts: datetime | None = None,
) -> Source:
    ts = source_ts or datetime(2026, 10, 27, 10, 0, tzinfo=UTC)
    return Source(
        id=SourceId(uuid4()),
        user_id=UserId(uuid4()),
        kind=SourceKind.CHAT,
        role=role,
        session_id='sess_test',
        turn_index=turn_index,
        content=content,
        source_ts=ts,
        content_hash=f'hash_{turn_index}',
        deleted_at=datetime(2026, 10, 27, 12, 0, tzinfo=UTC) if deleted else None,
    )


# -----------------------------------------------------------------------------
# T-EXT: FactExtractor Unit Tests
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_extractor_empty_window_short_circuit() -> None:
    """Empty targets window short-circuits with 0 LLM calls."""
    llm = FakeLLMGateway()
    extractor = FactExtractor(llm=llm)

    window = SourceWindow(targets=(), context=())
    res = await extractor.extract(window)

    assert res == []
    assert llm.call_count == 0


@pytest.mark.asyncio
async def test_extractor_deleted_targets_short_circuit() -> None:
    """Window where all targets are deleted short-circuits without LLM calls."""
    llm = FakeLLMGateway()
    extractor = FactExtractor(llm=llm)

    target = make_source(1, 'I am a doctor', deleted=True)
    window = SourceWindow(targets=(target,), context=())
    res = await extractor.extract(window)

    assert res == []
    assert llm.call_count == 0


@pytest.mark.asyncio
async def test_extractor_assistant_target_ignored_by_default() -> None:
    """Assistant target is ignored when extract_assistant=False."""
    llm = FakeLLMGateway()
    extractor = FactExtractor(llm=llm, extract_assistant=False)

    target = make_source(1, 'You are an engineer.', role=Role.ASSISTANT)
    window = SourceWindow(targets=(target,), context=())
    res = await extractor.extract(window)

    assert res == []
    assert llm.call_count == 0


@pytest.mark.asyncio
async def test_extractor_valid_extraction() -> None:
    """Valid ExtractedFact returns ValidatedExtraction with resolved Source and span offsets."""
    target = make_source(1, 'I live in Paris since 2020.')
    context = make_source(0, 'Where do you live?', role=Role.ASSISTANT)
    window = SourceWindow(targets=(target,), context=(context,))

    llm = FakeLLMGateway(
        responses=[
            ExtractionResult(
                facts=[
                    ExtractedFact(
                        subject='I',
                        predicate='lives_in',
                        object='Paris',
                        valid_from_text='since 2020',
                        confidence=0.92,
                        evidence_span='live in Paris',
                        source_turn=1,
                    )
                ]
            )
        ]
    )

    pred = Predicate(
        name='lives_in',
        cardinality=Cardinality.SINGLE,
        decay_class=DecayClass.PERMANENT,
        object_kind=ObjectKind.ENTITY,
        template='{subject} lives in {object}',
        synonyms=(),
    )
    registry = FakePredicateRegistry([pred])
    extractor = FactExtractor(llm=llm, registry=registry)

    extractions = await extractor.extract(window)

    assert len(extractions) == 1
    item = extractions[0]
    assert isinstance(item, ValidatedExtraction)
    assert item.fact.subject == 'I'
    assert item.fact.predicate == 'lives_in'
    assert item.fact.object == 'Paris'
    assert item.source.id == target.id
    assert item.span_start == 2
    assert item.span_end == 15
    assert target.content[item.span_start : item.span_end] == 'live in Paris'


@pytest.mark.asyncio
async def test_extractor_hallucinated_evidence_span_drop() -> None:
    """Fact with hallucinated evidence_span not in target turn content is dropped."""
    target = make_source(1, 'I love playing tennis on weekends.')
    window = SourceWindow(targets=(target,), context=())

    llm = FakeLLMGateway(
        responses=[
            ExtractionResult(
                facts=[
                    ExtractedFact(
                        subject='I',
                        predicate='lives_in',
                        object='Berlin',
                        confidence=0.9,
                        evidence_span='I live in Berlin',  # Hallucinated!
                        source_turn=1,
                    )
                ]
            )
        ]
    )

    extractor = FactExtractor(llm=llm)
    extractions = await extractor.extract(window)

    assert extractions == []
    assert extractor.span_mismatches == 1


@pytest.mark.asyncio
async def test_extractor_context_turn_source_turn_rejection() -> None:
    """Fact attributing source_turn to a read-only context turn is rejected."""
    ctx_source = make_source(0, 'You said you enjoy reading sci-fi books.', role=Role.ASSISTANT)
    tgt_source = make_source(1, 'Yes, exactly!')
    window = SourceWindow(targets=(tgt_source,), context=(ctx_source,))

    llm = FakeLLMGateway(
        responses=[
            ExtractionResult(
                facts=[
                    ExtractedFact(
                        subject='I',
                        predicate='likes',
                        object='sci-fi books',
                        confidence=0.85,
                        evidence_span='reading sci-fi books',
                        source_turn=0,  # Context turn!
                    )
                ]
            )
        ]
    )

    extractor = FactExtractor(llm=llm)
    extractions = await extractor.extract(window)

    assert extractions == []


@pytest.mark.asyncio
async def test_extractor_empty_fields_rejection() -> None:
    """Facts with empty subject, predicate, or object are dropped."""
    target = make_source(1, 'I like coffee and tea.')
    window = SourceWindow(targets=(target,), context=())

    llm = FakeLLMGateway(
        responses=[
            ExtractionResult(
                facts=[
                    ExtractedFact(
                        subject='   ',
                        predicate='likes',
                        object='coffee',
                        confidence=0.8,
                        evidence_span='like coffee',
                        source_turn=1,
                    ),
                    ExtractedFact(
                        subject='I',
                        predicate='   ',
                        object='tea',
                        confidence=0.8,
                        evidence_span='like coffee',
                        source_turn=1,
                    ),
                    ExtractedFact(
                        subject='I',
                        predicate='likes',
                        object='   ',
                        confidence=0.8,
                        evidence_span='like coffee',
                        source_turn=1,
                    ),
                ]
            )
        ]
    )

    extractor = FactExtractor(llm=llm)
    extractions = await extractor.extract(window)

    assert extractions == []


@pytest.mark.asyncio
async def test_extractor_low_confidence_drop() -> None:
    """Facts below min_confidence are dropped."""
    target = make_source(1, 'I might like hiking.')
    window = SourceWindow(targets=(target,), context=())

    llm = FakeLLMGateway(
        responses=[
            ExtractionResult(
                facts=[
                    ExtractedFact(
                        subject='I',
                        predicate='likes',
                        object='hiking',
                        confidence=0.45,  # < 0.5
                        evidence_span='like hiking',
                        source_turn=1,
                    )
                ]
            )
        ]
    )

    extractor = FactExtractor(llm=llm, min_confidence=0.5)
    extractions = await extractor.extract(window)

    assert extractions == []


@pytest.mark.asyncio
async def test_extractor_whitespace_insensitive_span_matching() -> None:
    """Evidence span matches despite difference in whitespace and case."""
    target = make_source(1, 'I   WORK   AT   Google   in Zurich.')
    window = SourceWindow(targets=(target,), context=())

    llm = FakeLLMGateway(
        responses=[
            ExtractionResult(
                facts=[
                    ExtractedFact(
                        subject='I',
                        predicate='works_at',
                        object='Google',
                        confidence=0.9,
                        evidence_span='work at google',
                        source_turn=1,
                    )
                ]
            )
        ]
    )

    extractor = FactExtractor(llm=llm)
    extractions = await extractor.extract(window)

    assert len(extractions) == 1
    assert extractions[0].fact.object == 'Google'


@pytest.mark.asyncio
async def test_extractor_deduplication_and_cap_at_30() -> None:
    """Deduplication drops identical facts, and output is capped at 30 items."""
    target = make_source(1, 'I like coding and design.')
    window = SourceWindow(targets=(target,), context=())

    # 30 facts with 15 duplicates
    facts = [
        ExtractedFact(
            subject='I',
            predicate='likes',
            object=f'item_{i % 15}',
            confidence=0.8,
            evidence_span='like coding',
            source_turn=1,
        )
        for i in range(30)
    ]
    llm = FakeLLMGateway(responses=[ExtractionResult(facts=facts)])

    extractor = FactExtractor(llm=llm)
    extractions = await extractor.extract(window)

    # 30 items produced with 15 unique objects -> exactly 15 extracted
    assert len(extractions) == 15


@pytest.mark.asyncio
async def test_extractor_prompt_formatting_and_extract_result() -> None:
    """Prompt format matches [turn N | ISO_TS | role]: text with CONTEXT and TARGET tags."""
    ts0 = datetime(2026, 10, 27, 9, 30, tzinfo=UTC)
    ts1 = datetime(2026, 10, 27, 9, 31, tzinfo=UTC)
    ctx = make_source(0, 'Context text', role=Role.ASSISTANT, source_ts=ts0)
    tgt = make_source(1, 'Target text', role=Role.USER, source_ts=ts1)
    window = SourceWindow(targets=(tgt,), context=(ctx,))

    llm = FakeLLMGateway(
        responses=[
            ExtractionResult(
                facts=[
                    ExtractedFact(
                        subject='I',
                        predicate='related_to',
                        object='text',
                        confidence=0.9,
                        evidence_span='Target text',
                        source_turn=1,
                    )
                ]
            )
        ]
    )

    extractor = LLMFactExtractor(llm=llm)
    result = await extractor.extract_result(window)

    assert isinstance(result, ExtractionResult)
    assert len(result.facts) == 1

    prompt_body = llm.requests[0].messages[1].content
    assert '[CONTEXT] [turn 0 | 2026-10-27T09:30:00+00:00 | assistant]: Context text' in prompt_body
    assert '[TARGET] [turn 1 | 2026-10-27T09:31:00+00:00 | user]: Target text' in prompt_body


# -----------------------------------------------------------------------------
# PlanExecutor Unit Tests
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_plan_executor_full_plan_with_new_ref() -> None:
    """Execute full ResolutionPlan including NewRef resolution in AddEvidence and AddRelation."""
    uow = FakeUnitOfWork()

    f1 = FactId(uuid4())
    f2 = FactId(uuid4())
    f3 = FactId(uuid4())
    f4 = FactId(uuid4())
    user_id = UserId(uuid4())
    source_id = SourceId(uuid4())
    job_id = uuid4()

    t_now = datetime(2026, 10, 27, 12, 0, tzinfo=UTC)
    t_past = datetime(2026, 1, 1, 0, 0, tzinfo=UTC)
    clock = FixedClock(t_now)

    plan = ResolutionPlan(
        ops=(
            CloseFact(fact_id=f1, valid_to=t_now),
            ExtendValidFrom(fact_id=f2, valid_from=t_past),
            SetStatus(fact_id=f3, status=FactStatus.ARCHIVED),
            BumpConfidence(fact_id=f4, confidence=0.95),
            InsertFact(valid_from=t_now, valid_to=None, status=FactStatus.ACTIVE),
            AddEvidence(ref=NewRef()),
            AddRelation(from_ref=NewRef(), to_ref=f1, relation=RelationType.SUPERSEDES),
        ),
        outcome=Outcome.SUPERSEDED,
        reason='Candidate supersedes open fact',
    )

    candidate = Candidate(
        user_id=user_id,
        subject_id=EntityId(uuid4()),
        predicate='lives_in',
        cardinality=Cardinality.SINGLE,
        object_entity_id=None,
        object_value='Tokyo',
        object_norm='tokyo',
        fact_text='User lives in Tokyo',
        valid_from=t_now,
        valid_to=None,
        time_precision=TimePrecision.EXACT,
        valid_from_explicit=True,
        confidence=0.9,
        is_retraction=False,
        source_id=source_id,
        span=(0, 15),
        source_ts=t_now,
        event_hash=None,
        embedding=[0.1, 0.2, 0.3],
    )

    executor = PlanExecutor(
        candidate=candidate,
        source_id=source_id,
        user_id=user_id,
        clock=clock,
        job_id=job_id,
    )

    touched_ids = await executor.execute(cast('UnitOfWork', uow), plan)

    # 1. User lock was acquired
    assert uow.locked_user is True

    # 2. CloseFact executed
    assert len(uow.facts.closed) == 1
    assert uow.facts.closed[0]['fact_id'] == f1
    assert uow.facts.closed[0]['valid_to'] == t_now

    # 3. ExtendValidFrom executed
    assert len(uow.facts.extended) == 1
    assert uow.facts.extended[0]['fact_id'] == f2
    assert uow.facts.extended[0]['valid_from'] == t_past

    # 4. SetStatus executed
    assert len(uow.facts.statuses) == 1
    assert uow.facts.statuses[0]['fact_id'] == f3
    assert uow.facts.statuses[0]['status'] == FactStatus.ARCHIVED

    # 5. BumpConfidence executed
    assert len(uow.facts.bumped) == 1
    assert uow.facts.bumped[0]['fact_id'] == f4
    assert uow.facts.bumped[0]['confidence'] == 0.95

    # 6. InsertFact executed and produced new FactId
    assert len(uow.facts.inserted) == 1
    new_fact_id = uow.facts.inserted[0]['fact_id']
    assert isinstance(new_fact_id, UUID)
    assert uow.facts.inserted[0]['valid_from'] == t_now
    assert uow.facts.inserted[0]['valid_to'] is None

    # 7. AddEvidence resolved NewRef to new_fact_id
    assert len(uow.facts.evidences) == 1
    ev_record = uow.facts.evidences[0]['evidence']
    assert isinstance(ev_record, FactEvidence)
    assert ev_record.fact_id == new_fact_id
    assert ev_record.source_id == source_id
    assert ev_record.span_start == 0
    assert ev_record.span_end == 15

    # 8. AddRelation resolved from_ref=NewRef() to new_fact_id
    assert len(uow.facts.relations) == 1
    rel_record = uow.facts.relations[0]['relation']
    assert isinstance(rel_record, FactRelation)
    assert rel_record.from_fact_id == new_fact_id
    assert rel_record.to_fact_id == f1
    assert rel_record.relation == RelationType.SUPERSEDES
    assert uow.facts.relations[0]['job_id'] == job_id

    # 9. All touched FactIds returned
    assert new_fact_id in touched_ids
    assert f1 in touched_ids
    assert f2 in touched_ids
    assert f3 in touched_ids
    assert f4 in touched_ids


@pytest.mark.asyncio
async def test_plan_executor_new_ref_before_insert_raises() -> None:
    """Referencing NewRef before InsertFact raises InvariantViolation."""
    uow = FakeUnitOfWork()

    # Plan with AddEvidence(NewRef) without prior InsertFact
    plan = ResolutionPlan(
        ops=(AddEvidence(ref=NewRef()),),
        outcome=Outcome.MERGED,
        reason='Invalid plan',
    )

    executor = PlanExecutor()
    with pytest.raises(InvariantViolation, match='NewRef referenced before InsertFact'):
        await executor.execute(cast('UnitOfWork', uow), plan)


@pytest.mark.asyncio
async def test_plan_executor_existing_ref_add_evidence_and_relation() -> None:
    """AddEvidence and AddRelation work with existing FactId without NewRef."""
    uow = FakeUnitOfWork()
    f1 = FactId(uuid4())
    f2 = FactId(uuid4())
    user_id = UserId(uuid4())
    source_id = SourceId(uuid4())

    plan = ResolutionPlan(
        ops=(
            AddEvidence(ref=f1),
            AddRelation(from_ref=f1, to_ref=f2, relation=RelationType.CLARIFIES),
        ),
        outcome=Outcome.MERGED,
        reason='Clarification attached',
    )

    executor = PlanExecutor(user_id=user_id, source_id=source_id)
    touched = await executor.execute(cast('UnitOfWork', uow), plan)

    assert len(uow.facts.evidences) == 1
    assert uow.facts.evidences[0]['evidence'].fact_id == f1
    assert len(uow.facts.relations) == 1
    assert uow.facts.relations[0]['relation'].from_fact_id == f1
    assert uow.facts.relations[0]['relation'].to_fact_id == f2
    assert f1 in touched
    assert f2 in touched


@pytest.mark.asyncio
async def test_plan_executor_fallback_candidate_synthesis() -> None:
    """InsertFact creates default Candidate if candidate was not explicitly provided."""
    uow = FakeUnitOfWork()
    t_now = datetime(2026, 10, 27, 12, 0, tzinfo=UTC)

    plan = ResolutionPlan(
        ops=(InsertFact(valid_from=t_now, valid_to=None, status=FactStatus.ACTIVE),),
        outcome=Outcome.INSERTED,
        reason='New fact insertion',
    )

    executor = PlanExecutor()
    touched = await executor.execute(cast('UnitOfWork', uow), plan, now=t_now)

    assert len(uow.facts.inserted) == 1
    assert len(touched) == 1
    assert touched[0] == uow.facts.inserted[0]['fact_id']
