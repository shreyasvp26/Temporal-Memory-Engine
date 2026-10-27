"""Boundary Pydantic schemas and serialization helpers per docs/LLD.md Section 11.4."""

from datetime import datetime
from typing import Literal
from uuid import UUID

from memory_engine.domain.models import Fact, Job
from memory_engine.ports.repositories import EvidenceSnippet
from pydantic import BaseModel, ConfigDict, Field


class ChatRequest(BaseModel):
    """Chat turn request boundary schema per LLD Section 11.4 [FROZEN]."""

    model_config = ConfigDict(extra='forbid', frozen=True)

    message: str = Field(min_length=1, max_length=20000)
    conversation_id: UUID | None = None
    client_ts: datetime | None = None
    client_message_id: str | None = Field(default=None, max_length=64)


class ChatResponse(BaseModel):
    """Non-streaming chat turn response."""

    model_config = ConfigDict(extra='forbid', frozen=True)

    conversation_id: UUID
    user_message_id: UUID
    assistant_message_id: UUID
    content: str
    cited_fact_ids: list[UUID] = Field(default_factory=list)
    cited_source_ids: list[UUID] = Field(default_factory=list)
    statement_only: bool = False
    abstained: bool = False


class NoteCreateRequest(BaseModel):
    """Note/document ingestion request schema."""

    model_config = ConfigDict(extra='forbid', frozen=True)

    content: str = Field(min_length=1, max_length=200000)
    title: str | None = Field(default=None, max_length=200)
    source_ts: datetime | None = None


class NoteCreateResponse(BaseModel):
    """Note ingestion response returning created sources and scheduled job."""

    model_config = ConfigDict(extra='forbid', frozen=True)

    source_count: int
    job_id: UUID | None = None
    source_ids: list[UUID] = Field(default_factory=list)


class EntityRef(BaseModel):
    """Subject/Object entity reference."""

    model_config = ConfigDict(extra='forbid', frozen=True)

    id: UUID
    name: str


class ObjectView(BaseModel):
    """Object view for facts."""

    model_config = ConfigDict(extra='forbid', frozen=True)

    kind: Literal['entity', 'literal']
    entity_id: UUID | None = None
    value: str


FactState = Literal['current', 'superseded', 'past_event', 'archived', 'needs_review', 'deleted']


class FactView(BaseModel):
    """Standard view representation of a memory fact [FROZEN]."""

    model_config = ConfigDict(extra='forbid', frozen=True)

    id: UUID
    text: str
    subject: EntityRef
    predicate: str
    object: ObjectView
    valid_from: datetime
    valid_to: datetime | None
    time_precision: str
    state: FactState
    status: str
    confidence: float
    origin: str
    recorded_at: datetime
    evidence_count: int = 0
    access_count: int = 0


class EvidenceView(BaseModel):
    """Supporting evidence excerpt view."""

    model_config = ConfigDict(extra='forbid', frozen=True)

    source_id: UUID
    source_ts: datetime
    role: str
    snippet: str


class FactDetail(FactView):
    """Detailed fact view with supporting evidence and supersession chain."""

    evidence: list[EvidenceView] = Field(default_factory=list)
    chain: list[FactView] = Field(default_factory=list)


class FactEditRequest(BaseModel):
    """Request payload for manual fact editing."""

    model_config = ConfigDict(extra='forbid', frozen=True)

    object_value: str = Field(min_length=1)
    valid_from: datetime | None = None
    valid_to: datetime | None = None


class FactRetractRequest(BaseModel):
    """Request payload for retracting a fact."""

    model_config = ConfigDict(extra='forbid', frozen=True)

    reason: str = Field(default='user_retracted', max_length=500)


class JobView(BaseModel):
    """Background outbox job view [FROZEN]."""

    model_config = ConfigDict(extra='forbid', frozen=True)

    id: UUID
    type: str
    status: str
    attempts: int
    last_error: str | None = None
    error_code: str | None = None
    created_at: datetime
    preview: str | None = None


class HealthResponse(BaseModel):
    """Liveness probe response."""

    model_config = ConfigDict(extra='forbid', frozen=True)

    status: str = 'ok'
    version: str = '0.1.0'


class ReadyResponse(BaseModel):
    """Readiness probe response."""

    model_config = ConfigDict(extra='forbid', frozen=True)

    status: str = 'ready'
    db: str = 'ok'


def compute_fact_state(fact: Fact) -> FactState:
    """Compute state server-side per LLD Section 11.4."""
    status_str = fact.status.value if hasattr(fact.status, 'value') else str(fact.status)
    if status_str == 'deleted':
        return 'deleted'
    if status_str == 'needs_review':
        return 'needs_review'
    if status_str == 'archived':
        return 'archived'
    if status_str in ('superseded', 'retracted') or fact.invalidated_at is not None:
        return 'superseded'
    card_str = (
        fact.cardinality.value if hasattr(fact.cardinality, 'value') else str(fact.cardinality)
    )
    if card_str == 'event':
        return 'past_event'
    return 'current'


def fact_to_view(
    fact: Fact,
    subject_name: str = 'User',
    evidence_count: int = 0,
) -> FactView:
    """Hydrate FactView model from domain Fact entity."""
    state = compute_fact_state(fact)
    obj_kind: Literal['entity', 'literal'] = (
        'entity' if fact.object_entity_id is not None else 'literal'
    )
    obj_view = ObjectView(
        kind=obj_kind,
        entity_id=fact.object_entity_id,
        value=fact.object_value or '',
    )
    subj_ref = EntityRef(
        id=fact.subject_id,
        name=subject_name,
    )
    time_prec = (
        fact.time_precision.value
        if hasattr(fact.time_precision, 'value')
        else str(fact.time_precision)
    )
    stat = fact.status.value if hasattr(fact.status, 'value') else str(fact.status)
    orig = fact.origin.value if hasattr(fact.origin, 'value') else str(fact.origin)

    return FactView(
        id=fact.id,
        text=fact.fact_text,
        subject=subj_ref,
        predicate=fact.predicate,
        object=obj_view,
        valid_from=fact.valid_from,
        valid_to=fact.valid_to,
        time_precision=time_prec,
        state=state,
        status=stat,
        confidence=fact.confidence,
        origin=orig,
        recorded_at=fact.recorded_at,
        evidence_count=evidence_count,
        access_count=fact.access_count,
    )


def fact_to_detail(
    fact: Fact,
    evidence: list[EvidenceSnippet],
    chain: list[Fact],
    subject_name: str = 'User',
) -> FactDetail:
    """Hydrate FactDetail model with supporting evidence snippets and chain."""
    base_view = fact_to_view(fact, subject_name=subject_name, evidence_count=len(evidence))
    ev_views = [
        EvidenceView(
            source_id=e.source_id,
            source_ts=e.source_ts,
            role=e.role.value if hasattr(e.role, 'value') else str(e.role),
            snippet=e.snippet,
        )
        for e in evidence
    ]
    chain_views = [fact_to_view(c) for c in chain]
    return FactDetail(
        id=base_view.id,
        text=base_view.text,
        subject=base_view.subject,
        predicate=base_view.predicate,
        object=base_view.object,
        valid_from=base_view.valid_from,
        valid_to=base_view.valid_to,
        time_precision=base_view.time_precision,
        state=base_view.state,
        status=base_view.status,
        confidence=base_view.confidence,
        origin=base_view.origin,
        recorded_at=base_view.recorded_at,
        evidence_count=len(evidence),
        access_count=base_view.access_count,
        evidence=ev_views,
        chain=chain_views,
    )


def job_to_view(job: Job) -> JobView:
    """Hydrate JobView model from domain Job entity."""
    job_type = job.type.value if hasattr(job.type, 'value') else str(job.type)
    job_status = job.status.value if hasattr(job.status, 'value') else str(job.status)
    err_code: str | None = None
    preview: str | None = None
    if isinstance(job.payload, dict):
        raw_code = job.payload.get('error_code')
        if raw_code is not None:
            err_code = str(raw_code)
        raw_preview = job.payload.get('preview')
        if raw_preview is not None:
            preview = str(raw_preview)

    return JobView(
        id=job.id,
        type=job_type,
        status=job_status,
        attempts=job.attempts,
        last_error=job.last_error,
        error_code=err_code,
        created_at=job.run_after,
        preview=preview,
    )
