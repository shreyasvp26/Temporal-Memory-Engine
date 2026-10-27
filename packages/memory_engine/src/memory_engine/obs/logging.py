"""Structured logging and context propagation for Temporal Memory Engine."""

import hashlib
import logging
import sys
from collections.abc import MutableMapping
from contextvars import ContextVar
from typing import Any, TextIO
from uuid import UUID

import structlog
from structlog.typing import EventDict, WrappedLogger

request_id_var: ContextVar[str | None] = ContextVar('request_id', default=None)
user_id_var: ContextVar[str | None] = ContextVar('user_id', default=None)
job_id_var: ContextVar[str | None] = ContextVar('job_id', default=None)
conversation_id_var: ContextVar[str | None] = ContextVar('conversation_id', default=None)

MEMORY_CONTENT_KEYS: frozenset[str] = frozenset(
    {
        'text',
        'content',
        'fact_text',
        'memory_text',
        'raw_text',
        'evidence',
        'evidence_span',
        'snippet',
        'turn_text',
        'message_text',
        'prompt_text',
    }
)


def user_hash(user_id: UUID | str) -> str:
    """Return the first 12 hex characters of sha256(str(user_id))."""
    digest = hashlib.sha256(str(user_id).encode('utf-8')).hexdigest()
    return digest[:12]


def bind_context(
    *,
    request_id: UUID | str | None = None,
    user_id: UUID | str | None = None,
    job_id: UUID | str | None = None,
    conversation_id: UUID | str | None = None,
) -> None:
    """Bind contextual identifiers to the current task/coroutine context."""
    if request_id is not None:
        request_id_var.set(str(request_id))
    if user_id is not None:
        user_id_var.set(str(user_id))
    if job_id is not None:
        job_id_var.set(str(job_id))
    if conversation_id is not None:
        conversation_id_var.set(str(conversation_id))


def clear_context() -> None:
    """Clear all contextual identifiers from the current context."""
    request_id_var.set(None)
    user_id_var.set(None)
    job_id_var.set(None)
    conversation_id_var.set(None)


def get_context() -> dict[str, str | None]:
    """Return current context variables as a dict."""
    return {
        'request_id': request_id_var.get(),
        'user_id': user_id_var.get(),
        'job_id': job_id_var.get(),
        'conversation_id': conversation_id_var.get(),
    }


def redact_content(
    event_dict: MutableMapping[str, Any],
    *,
    allow_content: bool = False,
) -> MutableMapping[str, Any]:
    """Redact raw memory text fields from event dictionary unless allowed."""
    if allow_content:
        return event_dict

    for key in list(event_dict.keys()):
        if key in MEMORY_CONTENT_KEYS:
            val = event_dict[key]
            if isinstance(val, str):
                event_dict[f'{key}_length'] = len(val)
                del event_dict[key]
    return event_dict


def configure_logging(
    service: str,
    level: str = 'INFO',
    *,
    log_content: bool = False,
    file: TextIO | None = None,
) -> None:
    """Configure structlog JSON logging with mandatory fields and redaction."""
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    output_file = file if file is not None else sys.stdout

    def add_service_and_context(
        logger: WrappedLogger,
        method_name: str,
        event_dict: EventDict,
    ) -> EventDict:
        event_dict['service'] = service

        req_id = request_id_var.get()
        if req_id is not None and 'request_id' not in event_dict:
            event_dict['request_id'] = req_id

        jb_id = job_id_var.get()
        if jb_id is not None and 'job_id' not in event_dict:
            event_dict['job_id'] = jb_id

        conv_id = conversation_id_var.get()
        if conv_id is not None and 'conversation_id' not in event_dict:
            event_dict['conversation_id'] = conv_id

        if 'user_id' in event_dict:
            raw_uid = event_dict.pop('user_id')
            if 'user_hash' not in event_dict and raw_uid is not None:
                event_dict['user_hash'] = user_hash(raw_uid)
        else:
            u_id = user_id_var.get()
            if u_id is not None and 'user_hash' not in event_dict:
                event_dict['user_hash'] = user_hash(u_id)

        return event_dict

    def extract_error_fields(
        logger: WrappedLogger,
        method_name: str,
        event_dict: EventDict,
    ) -> EventDict:
        exc = event_dict.get('exc_info')
        if isinstance(exc, tuple) and len(exc) >= 2:
            exc = exc[1]
        elif exc is True:
            exc = sys.exc_info()[1]
        if exc is None:
            exc = event_dict.get('exception')

        if isinstance(exc, BaseException):
            event_dict.setdefault('exception_class', exc.__class__.__name__)
            code = getattr(exc, 'code', getattr(exc, 'error_code', None))
            if code is not None:
                event_dict.setdefault('error_code', str(code))
        return event_dict

    def redact_content_processor(
        logger: WrappedLogger,
        method_name: str,
        event_dict: EventDict,
    ) -> EventDict:
        allow = (method_name.lower() == 'debug') and log_content
        return redact_content(event_dict, allow_content=allow)

    processors: list[structlog.typing.Processor] = [
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt='iso', key='ts'),
        add_service_and_context,
        extract_error_fields,
        redact_content_processor,
        structlog.processors.format_exc_info,
        structlog.processors.JSONRenderer(),
    ]

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(numeric_level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=output_file),
        cache_logger_on_first_use=False,
    )


get_logger = structlog.get_logger
