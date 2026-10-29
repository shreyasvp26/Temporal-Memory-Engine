"""FastAPI entrypoint and composition root per docs/LLD.md Section 11.1."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from uuid import uuid4

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from memory_engine.config import Settings
from memory_engine.domain.errors import PayloadTooLarge
from memory_engine.obs.logging import bind_context
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from .deps import AppContainer
from .errors import build_error_response, register_exception_handlers
from .routes import api_router


class RequestIdMiddleware:
    """ASGI middleware extracting or generating X-Request-Id and echoing it in response headers."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope['type'] != 'http':
            await self.app(scope, receive, send)
            return

        req_id: str | None = None
        for key, val in scope.get('headers', []):
            if key.lower() == b'x-request-id':
                try:
                    val_str = val.decode('utf-8').strip()
                    if val_str and len(val_str) <= 64:
                        req_id = val_str
                except Exception:
                    pass
                break

        if not req_id:
            req_id = str(uuid4())

        if 'state' not in scope:
            scope['state'] = {}
        scope['state']['request_id'] = req_id
        bind_context(request_id=req_id)

        async def send_wrapper(message: Message) -> None:
            if message['type'] == 'http.response.start':
                headers = list(message.get('headers', []))
                headers.append((b'x-request-id', req_id.encode('utf-8')))
                message['headers'] = headers
            await send(message)

        await self.app(scope, receive, send_wrapper)


class PayloadSizeLimitMiddleware:
    """ASGI middleware enforcing body limits:

    - 256 KB for /v1/chat
    - 2 MB for /v1/notes and /v1/ingest
    - 64 KB elsewhere
    Rejection returns 413 E-REQ-01.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope['type'] != 'http':
            await self.app(scope, receive, send)
            return

        path = scope.get('path', '')
        if path.startswith('/v1/chat'):
            limit = 256 * 1024
        elif path.startswith('/v1/notes') or path.startswith('/v1/ingest'):
            limit = 2 * 1024 * 1024
        else:
            limit = 64 * 1024

        req_id = scope.get('state', {}).get('request_id') or str(uuid4())

        # 1. Content-Length header guard
        for key, val in scope.get('headers', []):
            if key.lower() == b'content-length':
                try:
                    content_length = int(val.decode('utf-8'))
                    if content_length > limit:
                        resp = build_error_response(
                            status_code=413,
                            code='E-REQ-01',
                            message=f'Payload size exceeds limit of {limit} bytes',
                            request_id=req_id,
                            retryable=False,
                            details={'max_bytes': limit, 'received_bytes': content_length},
                        )
                        await resp(scope, receive, send)
                        return
                except ValueError:
                    pass
                break

        # 2. Chunked stream body accumulation guard
        bytes_received = 0

        async def receive_wrapper() -> Message:
            nonlocal bytes_received
            message = await receive()
            if message['type'] == 'http.request':
                body = message.get('body', b'')
                bytes_received += len(body)
                if bytes_received > limit:
                    raise PayloadTooLarge(
                        f'Request payload exceeds maximum allowed size of {limit} bytes',
                        code='E-REQ-01',
                        details={'max_bytes': limit, 'received_bytes': bytes_received},
                    )
            return message

        try:
            await self.app(scope, receive_wrapper, send)
        except PayloadTooLarge as exc:
            resp = build_error_response(
                status_code=413,
                code='E-REQ-01',
                message=exc.message,
                request_id=req_id,
                retryable=False,
                details=exc.details,
            )
            await resp(scope, receive, send)


def create_app(
    settings: Settings | None = None,
    container: AppContainer | None = None,
) -> FastAPI:
    """Application factory for Temporal Memory Engine REST & SSE API."""
    effective_settings = settings or (container.settings if container else Settings())

    @asynccontextmanager
    async def app_lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
        pool = None
        if container is not None:
            app.state.container = container
        elif effective_settings.database_url_api:
            from memory_engine.db.pool import create_api_pool
            from memory_engine.db.uow import PostgresUnitOfWorkFactory
            from memory_engine.domain.clock import SystemClock
            from memory_engine.ingest.canonicalizer import Canonicalizer
            from memory_engine.ingest.service import IngestService
            from memory_engine.lifecycle.service import MemoryLifecycleService
            from memory_engine.retrieve.chat_service import ChatService
            from memory_engine.retrieve.orchestrator import RetrievalOrchestrator

            from .auth import JwtVerifier
            from .rate_limit import RateLimiter

            clock = SystemClock()
            pool = await create_api_pool(effective_settings.database_url_api)
            uows = PostgresUnitOfWorkFactory(pool)
            canonicalizer = Canonicalizer()
            lifecycle = MemoryLifecycleService(uows=uows, clock=clock, canonicalizer=canonicalizer)
            embedder = None
            try:
                from memory_engine.embed.fastembed_adapter import FastEmbedAdapter

                embedder = FastEmbedAdapter()
            except Exception:
                embedder = None

            ingest = IngestService(
                uows=uows, clock=clock, canonicalizer=canonicalizer, embedder=embedder
            )
            retrieval = RetrievalOrchestrator(
                uows=uows, clock=clock, cfg=effective_settings.retrieval, embedder=embedder
            )
            chat = ChatService(orchestrator=retrieval, clock=clock, uows=uows)
            rate_limiter = RateLimiter(
                clock=clock,
                bucket_configs={
                    'chat': (float(effective_settings.rate_chat_per_min), 10.0),
                    'ingest': (float(effective_settings.rate_ingest_per_min), 10.0),
                    'search': (float(effective_settings.rate_search_per_min), 20.0),
                    'default': (float(effective_settings.rate_default_per_min), 30.0),
                },
            )
            jwt_verifier = JwtVerifier(effective_settings.jwt, env=effective_settings.env)

            app.state.container = AppContainer(
                settings=effective_settings,
                clock=clock,
                uows=uows,
                ingest=ingest,
                retrieval=retrieval,
                chat=chat,
                lifecycle=lifecycle,
                rate_limiter=rate_limiter,
                jwt_verifier=jwt_verifier,
                embedder=embedder,
            )

        yield

        if pool is not None:
            await pool.close()

    app = FastAPI(
        title='Temporal Memory Engine',
        version='0.1.0',
        lifespan=app_lifespan,
    )

    if container is not None:
        app.state.container = container

    # CORS configuration
    web_origin = effective_settings.web_origin
    origins = [web_origin] if web_origin else ['*']
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=['*'],
        allow_headers=['Authorization', 'Content-Type', 'X-Request-Id', 'X-Debug-User-Id'],
        expose_headers=['X-Request-Id', 'Retry-After'],
    )

    # Payload size limit and Request ID middleware
    app.add_middleware(PayloadSizeLimitMiddleware)
    app.add_middleware(RequestIdMiddleware)

    # Register error handlers
    register_exception_handlers(app)

    # Mount API routes
    app.include_router(api_router)

    return app


app = create_app()
