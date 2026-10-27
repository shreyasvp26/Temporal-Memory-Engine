"""Exception handlers mapping domain errors to HTTP statuses and standard error envelopes."""

import logging
import math
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from memory_engine.domain.errors import (
    AllLanesFailed,
    ConfigError,
    ConstraintViolation,
    DomainError,
    EmbeddingError,
    ExtractionError,
    Forbidden,
    IngestionError,
    InvariantViolation,
    LeaseLost,
    LLMContentBlocked,
    LLMError,
    LLMRateLimited,
    LLMTimeout,
    LLMUnavailable,
    MemoryEngineError,
    NotFound,
    PayloadTooLarge,
    PlanningError,
    RateLimited,
    RetrievalError,
    SchemaViolation,
    StateConflict,
    StorageError,
    StreamInterrupted,
    TransientStorageError,
    Unauthorized,
    ValidationFailed,
)
from memory_engine.obs.logging import request_id_var
from pydantic import BaseModel, ConfigDict, Field
from starlette.exceptions import HTTPException as StarletteHTTPException

_logger = logging.getLogger(__name__)


class ErrorDetail(BaseModel):
    """Inner error payload per LLD Section 11.3 and Section 17.1 [FROZEN]."""

    model_config = ConfigDict(extra='forbid', frozen=True)

    code: str
    message: str
    request_id: str
    retryable: bool = False
    details: dict[str, Any] = Field(default_factory=dict)


class ErrorEnvelope(BaseModel):
    """Standard frozen JSON error envelope."""

    model_config = ConfigDict(extra='forbid', frozen=True)

    error: ErrorDetail


ERROR_STATUS_MAP: dict[type[MemoryEngineError], tuple[int, str, bool]] = {
    Unauthorized: (401, 'E-AUTH-01', False),
    Forbidden: (403, 'E-AUTH-02', False),
    ValidationFailed: (422, 'E-VAL-01', False),
    NotFound: (404, 'E-NF-01', False),
    StateConflict: (409, 'E-CONF-01', False),
    PayloadTooLarge: (413, 'E-REQ-01', False),
    RateLimited: (429, 'E-RATE-01', True),
    LLMUnavailable: (503, 'E-LLM-UNAVAILABLE', True),
    LLMTimeout: (504, 'E-LLM-TIMEOUT', True),
    LLMRateLimited: (503, 'E-LLM-RATELIMIT', True),
    SchemaViolation: (502, 'E-LLM-SCHEMA', True),
    LLMContentBlocked: (422, 'E-LLM-BLOCKED', False),
    StreamInterrupted: (500, 'E-LLM-STREAM', True),
    TransientStorageError: (503, 'E-DB-01', True),
    ConstraintViolation: (409, 'E-DB-02', False),
    ExtractionError: (502, 'E-EXT-SCHEMA', True),
    PlanningError: (500, 'E-RET-02', False),
    AllLanesFailed: (503, 'E-RET-01', False),
    LeaseLost: (500, 'E-JOB-LEASE', False),
    InvariantViolation: (500, 'E-INT-01', False),
    ConfigError: (500, 'E-CONFIG-01', False),
    DomainError: (400, 'E-VAL-01', False),
    StorageError: (503, 'E-DB-01', True),
    IngestionError: (502, 'E-EXT-SCHEMA', False),
    RetrievalError: (500, 'E-RET-01', False),
    LLMError: (503, 'E-LLM-UNAVAILABLE', True),
}


def _resolve_request_id(request: Request) -> str:
    """Extract request_id from request state, contextvar, headers, or generate new."""
    if hasattr(request.state, 'request_id') and request.state.request_id:
        return str(request.state.request_id)
    ctx_id = request_id_var.get()
    if ctx_id:
        return ctx_id
    hdr_id = request.headers.get('x-request-id')
    if hdr_id and len(hdr_id) <= 64:
        return hdr_id
    return str(uuid4())


def build_error_response(
    status_code: int,
    code: str,
    message: str,
    request_id: str,
    *,
    retryable: bool = False,
    details: dict[str, Any] | None = None,
    retry_after: float | int | None = None,
) -> JSONResponse:
    """Build standard JSON error envelope response with optional Retry-After header."""
    headers: dict[str, str] = {'X-Request-Id': request_id}
    if retry_after is not None:
        headers['Retry-After'] = str(max(1, math.ceil(retry_after)))

    content = {
        'error': {
            'code': code,
            'message': message,
            'request_id': request_id,
            'retryable': retryable,
            'details': details or {},
        }
    }
    return JSONResponse(status_code=status_code, content=content, headers=headers)


def classify_memory_engine_error(
    exc: MemoryEngineError,
) -> tuple[int, str, bool, float | None]:
    """Map MemoryEngineError to status_code, error_code, retryable, retry_after."""
    retry_after: float | None = getattr(exc, 'retry_after', None)

    if isinstance(exc, EmbeddingError):
        code = getattr(exc, 'code', 'E-EMB-01')
        if code == 'E-EMB-02':
            return 503, 'E-EMB-02', True, retry_after
        return 503, 'E-EMB-01', False, retry_after

    for exc_cls, (status, code, retryable) in ERROR_STATUS_MAP.items():
        if isinstance(exc, exc_cls):
            actual_code = getattr(exc, 'code', code) or code
            actual_retryable = getattr(exc, 'retryable', retryable)
            return status, actual_code, actual_retryable, retry_after

    code = getattr(exc, 'code', 'E-INT-01') or 'E-INT-01'
    retryable = getattr(exc, 'retryable', False)
    return 500, code, retryable, retry_after


async def memory_engine_error_handler(
    request: Request,
    exc: MemoryEngineError,
) -> JSONResponse:
    """Handle all typed MemoryEngineError subclasses."""
    req_id = _resolve_request_id(request)
    status_code, code, retryable, retry_after = classify_memory_engine_error(exc)

    details = dict(getattr(exc, 'details', {}) or {})
    _logger.warning(
        'MemoryEngineError: %s (status=%d, code=%s, req_id=%s)',
        exc.message,
        status_code,
        code,
        req_id,
    )
    return build_error_response(
        status_code=status_code,
        code=code,
        message=getattr(exc, 'message', str(exc)) or str(exc),
        request_id=req_id,
        retryable=retryable,
        details=details,
        retry_after=retry_after,
    )


async def validation_error_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    """Handle Pydantic request validation errors (422 E-VAL-01)."""
    req_id = _resolve_request_id(request)
    errors = exc.errors()
    # Sanitize errors for JSON
    sanitized_errors = [
        {
            'type': e.get('type'),
            'loc': list(e.get('loc', [])),
            'msg': e.get('msg'),
        }
        for e in errors
    ]
    return build_error_response(
        status_code=422,
        code='E-VAL-01',
        message='Request validation failed',
        request_id=req_id,
        retryable=False,
        details={'errors': sanitized_errors},
    )


async def starlette_http_exception_handler(
    request: Request,
    exc: StarletteHTTPException,
) -> JSONResponse:
    """Handle Starlette/FastAPI HTTPExceptions with standard envelope."""
    req_id = _resolve_request_id(request)
    status = exc.status_code

    code_map = {
        400: 'E-VAL-01',
        401: 'E-AUTH-01',
        403: 'E-AUTH-02',
        404: 'E-NF-01',
        409: 'E-CONF-01',
        413: 'E-REQ-01',
        422: 'E-VAL-01',
        429: 'E-RATE-01',
        502: 'E-LLM-SCHEMA',
        503: 'E-LLM-UNAVAILABLE',
        504: 'E-LLM-TIMEOUT',
    }
    code = code_map.get(status, 'E-INT-01')
    retryable = status in (429, 502, 503, 504)

    retry_after: float | None = None
    if exc.headers and 'Retry-After' in exc.headers:
        try:
            retry_after = float(exc.headers['Retry-After'])
        except ValueError:
            retry_after = 1.0

    return build_error_response(
        status_code=status,
        code=code,
        message=exc.detail if exc.detail else 'HTTP error',
        request_id=req_id,
        retryable=retryable,
        details={},
        retry_after=retry_after,
    )


async def unhandled_exception_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    """Catch-all unhandled exceptions giving 500 E-INT-01 without leaking internals."""
    req_id = _resolve_request_id(request)
    _logger.exception(
        'Unhandled exception on %s %s (req_id=%s)',
        request.method,
        request.url,
        req_id,
    )
    return build_error_response(
        status_code=500,
        code='E-INT-01',
        message='An unexpected internal error occurred',
        request_id=req_id,
        retryable=False,
        details={},
    )


def register_exception_handlers(app: FastAPI) -> None:
    """Register all standard exception handlers on the FastAPI application."""
    app.add_exception_handler(MemoryEngineError, memory_engine_error_handler)  # type: ignore[arg-type]
    app.add_exception_handler(RequestValidationError, validation_error_handler)  # type: ignore[arg-type]
    app.add_exception_handler(StarletteHTTPException, starlette_http_exception_handler)  # type: ignore[arg-type]
    app.add_exception_handler(Exception, unhandled_exception_handler)
