"""FastAPI route handlers for Temporal Memory Engine per docs/LLD.md Section 11."""

from collections.abc import AsyncIterator
from datetime import datetime
from typing import Annotated, Any
from uuid import UUID

from fastapi import APIRouter, Depends, Query, Request, Response, status
from fastapi.responses import StreamingResponse
from memory_engine.domain.enums import FactStatus
from memory_engine.domain.errors import NotFound
from memory_engine.domain.ids import EntityId, FactId, JobId, UserId
from memory_engine.ingest.service import IngestService
from memory_engine.lifecycle.service import MemoryLifecycleService
from memory_engine.ports.repositories import FactListQuery, Page
from memory_engine.ports.uow import UnitOfWorkFactory
from memory_engine.retrieve.chat_service import (
    ChatRequest as ServiceChatRequest,
)
from memory_engine.retrieve.chat_service import (
    ChatService,
    ChatTurnResult,
)

from .auth import get_current_user
from .deps import (
    get_chat_service,
    get_ingest_service,
    get_lifecycle_service,
    get_uow_factory,
)
from .rate_limit import rate_limit
from .schemas import (
    ChatRequest,
    ChatResponse,
    FactDetail,
    FactEditRequest,
    FactRetractRequest,
    FactView,
    HealthResponse,
    JobView,
    NoteCreateRequest,
    NoteCreateResponse,
    ReadyResponse,
    RetrievalTelemetry,
    fact_to_detail,
    fact_to_view,
    job_to_view,
)

api_router = APIRouter()

CurrentUser = Annotated[UserId, Depends(get_current_user)]
ChatServiceDep = Annotated[ChatService, Depends(get_chat_service)]
IngestServiceDep = Annotated[IngestService, Depends(get_ingest_service)]
LifecycleServiceDep = Annotated[MemoryLifecycleService, Depends(get_lifecycle_service)]
UowFactoryDep = Annotated[UnitOfWorkFactory, Depends(get_uow_factory)]

SSE_HEADERS = {
    'Content-Type': 'text/event-stream; charset=utf-8',
    'Cache-Control': 'no-cache',
    'X-Accel-Buffering': 'no',
}


# ---------------------------------------------------------------------------
# Health & Readiness (No Auth)
# ---------------------------------------------------------------------------


@api_router.get('/healthz', response_model=HealthResponse)
@api_router.get('/v1/health', response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Liveness probe returning application status and version."""
    return HealthResponse(status='ok', version='0.1.0')


@api_router.get('/readyz', response_model=ReadyResponse)
@api_router.get('/v1/ready', response_model=ReadyResponse)
async def readiness_check(
    uow_factory: UowFactoryDep,
) -> ReadyResponse:
    """Readiness probe checking database connectivity."""
    return ReadyResponse(status='ready', db='ok')


# ---------------------------------------------------------------------------
# Chat Endpoints
# ---------------------------------------------------------------------------


async def _stream_chat_events(
    user_id: UserId,
    body: ChatRequest,
    service: ChatService,
) -> AsyncIterator[str]:
    """Helper streaming ChatStreamEvents as SSE wire frames."""
    svc_req = ServiceChatRequest(
        message=body.message,
        conversation_id=body.conversation_id,
        client_ts=body.client_ts,
        client_message_id=body.client_message_id,
    )
    async for event in service.stream_respond(user_id, svc_req):
        yield event.to_sse()


@api_router.post('/v1/chat', dependencies=[Depends(rate_limit('chat'))])
async def chat_turn(
    body: ChatRequest,
    request: Request,
    user_id: CurrentUser,
    service: ChatServiceDep,
) -> Response:
    """Conversational chat endpoint supporting both SSE stream and JSON responses."""
    accept_hdr = request.headers.get('accept', '')
    stream_requested = (
        'text/event-stream' in accept_hdr or request.query_params.get('stream') == 'true'
    )

    if stream_requested:
        return StreamingResponse(
            _stream_chat_events(user_id, body, service),
            media_type='text/event-stream',
            headers=SSE_HEADERS,
        )

    svc_req = ServiceChatRequest(
        message=body.message,
        conversation_id=body.conversation_id,
        client_ts=body.client_ts,
        client_message_id=body.client_message_id,
    )
    result: ChatTurnResult = await service.respond(user_id, svc_req)
    t_dict = result.telemetry or result.trace
    telemetry_view: RetrievalTelemetry | None = None
    if t_dict:
        telemetry_view = RetrievalTelemetry(**t_dict)

    return Response(
        content=ChatResponse(
            conversation_id=result.conversation_id,
            user_message_id=result.user_message_id,
            assistant_message_id=result.assistant_message_id,
            content=result.content,
            cited_fact_ids=[UUID(str(fid)) for fid in result.cited_fact_ids],
            cited_source_ids=[UUID(str(sid)) for sid in result.cited_source_ids],
            statement_only=result.statement_only,
            abstained=result.abstained,
            trace=telemetry_view,
            telemetry=telemetry_view,
        ).model_dump_json(),
        media_type='application/json',
    )


@api_router.post('/v1/chat/stream', dependencies=[Depends(rate_limit('chat'))])
async def chat_stream(
    body: ChatRequest,
    user_id: CurrentUser,
    service: ChatServiceDep,
) -> StreamingResponse:
    """Dedicated SSE streaming chat endpoint returning text/event-stream."""
    return StreamingResponse(
        _stream_chat_events(user_id, body, service),
        media_type='text/event-stream',
        headers=SSE_HEADERS,
    )


# ---------------------------------------------------------------------------
# Notes & Ingestion Endpoints
# ---------------------------------------------------------------------------


@api_router.post(
    '/v1/notes',
    response_model=NoteCreateResponse,
    status_code=status.HTTP_202_ACCEPTED,
    dependencies=[Depends(rate_limit('ingest'))],
)
@api_router.post(
    '/v1/ingest',
    response_model=NoteCreateResponse,
    status_code=status.HTTP_202_ACCEPTED,
    dependencies=[Depends(rate_limit('ingest'))],
)
async def create_note(
    body: NoteCreateRequest,
    user_id: CurrentUser,
    uow_factory: UowFactoryDep,
    ingest: IngestServiceDep,
) -> NoteCreateResponse:
    """Chunk and ingest a note or document, scheduling outbox extraction job."""
    async with uow_factory.for_user(user_id) as uow:
        res = await ingest.accept_note(
            uow,
            user_id=user_id,
            content=body.content,
            title=body.title,
            source_ts=body.source_ts,
        )

    return NoteCreateResponse(
        source_count=len(res.source_ids),
        job_id=UUID(str(res.job_id)) if res.job_id else None,
        source_ids=[UUID(str(sid)) for sid in res.source_ids],
    )


# ---------------------------------------------------------------------------
# Memory Fact Endpoints
# ---------------------------------------------------------------------------


@api_router.get(
    '/v1/facts',
    response_model=Page[FactView],
    dependencies=[Depends(rate_limit('default'))],
)
@api_router.get(
    '/v1/memories',
    response_model=Page[FactView],
    dependencies=[Depends(rate_limit('default'))],
)
async def list_facts(
    user_id: CurrentUser,
    lifecycle: LifecycleServiceDep,
    status: Annotated[str | None, Query()] = None,
    predicate: Annotated[str | None, Query()] = None,
    entity_id: Annotated[UUID | None, Query()] = None,
    q: Annotated[str | None, Query()] = None,
    as_of: Annotated[datetime | None, Query()] = None,
    current_only: Annotated[bool, Query()] = True,
    sort: Annotated[str, Query()] = 'valid_from',
    page_size: Annotated[int, Query(ge=1, le=100)] = 25,
    cursor: Annotated[str | None, Query()] = None,
) -> Page[FactView]:
    """List and filter memory facts with pagination."""
    fstatus: FactStatus | str | None = None
    if status is not None:
        try:
            fstatus = FactStatus(status.lower())
        except ValueError:
            fstatus = status

    list_query = FactListQuery(
        status=fstatus,
        predicate=predicate,
        entity_id=EntityId(entity_id) if entity_id else None,
        q=q,
        as_of=as_of,
        current_only=current_only,
        sort=sort,
        page_size=page_size,
        cursor=cursor,
    )
    page = await lifecycle.list_facts(user_id, list_query)
    views = [fact_to_view(f) for f in page.items]
    return Page[FactView](items=views, next_cursor=page.next_cursor)


@api_router.get(
    '/v1/facts/{fact_id}',
    response_model=FactDetail,
    dependencies=[Depends(rate_limit('default'))],
)
@api_router.get(
    '/v1/memories/{fact_id}',
    response_model=FactDetail,
    dependencies=[Depends(rate_limit('default'))],
)
async def get_fact(
    fact_id: UUID,
    user_id: CurrentUser,
    lifecycle: LifecycleServiceDep,
) -> FactDetail:
    """Retrieve fact details with supporting evidence snippets and supersession chain."""
    fact, evidence, chain = await lifecycle.get_fact_with_evidence(user_id, FactId(fact_id))
    return fact_to_detail(fact, evidence, chain)


@api_router.patch(
    '/v1/facts/{fact_id}',
    response_model=FactView,
    dependencies=[Depends(rate_limit('default'))],
)
@api_router.patch(
    '/v1/memories/{fact_id}',
    response_model=FactView,
    dependencies=[Depends(rate_limit('default'))],
)
async def update_fact(
    fact_id: UUID,
    body: FactEditRequest,
    user_id: CurrentUser,
    lifecycle: LifecycleServiceDep,
) -> FactView:
    """Manually update fact object value or valid time bounds."""
    updated = await lifecycle.update_fact(
        user_id,
        FactId(fact_id),
        object_value=body.object_value,
        valid_from=body.valid_from,
        valid_to=body.valid_to,
    )
    return fact_to_view(updated)


@api_router.post(
    '/v1/facts/{fact_id}/retract',
    dependencies=[Depends(rate_limit('default'))],
)
@api_router.post(
    '/v1/memories/{fact_id}/retract',
    dependencies=[Depends(rate_limit('default'))],
)
async def retract_fact(
    fact_id: UUID,
    body: FactRetractRequest,
    user_id: CurrentUser,
    lifecycle: LifecycleServiceDep,
) -> dict[str, Any]:
    """Retract fact and close its timeline valid bounds."""
    await lifecycle.retract_fact(user_id, FactId(fact_id), reason=body.reason)
    return {'ok': True, 'fact_id': str(fact_id), 'status': 'retracted'}


@api_router.delete(
    '/v1/facts/{fact_id}',
    dependencies=[Depends(rate_limit('default'))],
)
@api_router.delete(
    '/v1/memories/{fact_id}',
    dependencies=[Depends(rate_limit('default'))],
)
async def delete_fact(
    fact_id: UUID,
    user_id: CurrentUser,
    lifecycle: LifecycleServiceDep,
) -> dict[str, Any]:
    """Soft-delete a fact with a tombstone."""
    await lifecycle.delete_fact(user_id, FactId(fact_id))
    return {'status': 'deleted', 'fact_id': str(fact_id)}


# ---------------------------------------------------------------------------
# Outbox Job Endpoints
# ---------------------------------------------------------------------------


@api_router.get(
    '/v1/jobs/{job_id}',
    response_model=JobView,
    dependencies=[Depends(rate_limit('default'))],
)
async def get_job(
    job_id: UUID,
    user_id: CurrentUser,
    uow_factory: UowFactoryDep,
) -> JobView:
    """Retrieve background outbox extraction job status."""
    async with uow_factory.for_user(user_id, readonly=True) as uow:
        job = await uow.jobs.get(user_id, JobId(job_id))
        if job is None:
            raise NotFound(f'Job {job_id} not found', details={'job_id': str(job_id)})
        return job_to_view(job)


# ---------------------------------------------------------------------------
# User Profile & GDPR Governance Endpoints
# ---------------------------------------------------------------------------


@api_router.get(
    '/v1/profile/export',
    dependencies=[Depends(rate_limit('default'))],
)
async def export_profile(
    user_id: CurrentUser,
    lifecycle: LifecycleServiceDep,
) -> dict[str, Any]:
    """Export all user memory facts, sources, and audit logs for GDPR portability."""
    return await lifecycle.export_user(user_id)


@api_router.delete(
    '/v1/profile',
    dependencies=[Depends(rate_limit('default'))],
)
async def purge_profile(
    user_id: CurrentUser,
    lifecycle: LifecycleServiceDep,
) -> dict[str, Any]:
    """Permanently delete all user memories, profile, and history (GDPR right to be forgotten)."""
    count = await lifecycle.purge_user(user_id)
    return {'status': 'purged', 'purged_rows': count}


# ---------------------------------------------------------------------------
# Workbench Dev & Helper Endpoints
# ---------------------------------------------------------------------------

from .dev_routes import dev_router  # noqa: E402

api_router.include_router(dev_router)
