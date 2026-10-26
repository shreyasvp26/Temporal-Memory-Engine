"""Exception hierarchy and error codes for Temporal Memory Engine."""

from typing import Any


class MemoryEngineError(Exception):
    """Base exception for all Temporal Memory Engine errors."""

    message: str
    code: str
    details: dict[str, Any]
    retryable: bool

    def __init__(
        self,
        message: str = 'Memory engine error',
        *,
        code: str = 'E-INT-01',
        details: dict[str, Any] | None = None,
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.details = dict(details) if details is not None else {}
        self.retryable = retryable


class DomainError(MemoryEngineError):
    """Base error for domain-level rule violations."""

    def __init__(
        self,
        message: str = 'Domain error',
        *,
        code: str = 'E-INT-01',
        details: dict[str, Any] | None = None,
        retryable: bool = False,
    ) -> None:
        super().__init__(message, code=code, details=details, retryable=retryable)


class ValidationFailed(DomainError):
    """Request or model validation failed (E-VAL-01)."""

    def __init__(
        self,
        message: str = 'Validation failed',
        *,
        code: str = 'E-VAL-01',
        details: dict[str, Any] | None = None,
        retryable: bool = False,
    ) -> None:
        super().__init__(message, code=code, details=details, retryable=retryable)


class InvariantViolation(DomainError):
    """Domain invariant violated due to programming error or corrupt state (E-INT-01)."""

    def __init__(
        self,
        message: str = 'Domain invariant violated',
        *,
        code: str = 'E-INT-01',
        details: dict[str, Any] | None = None,
        retryable: bool = False,
    ) -> None:
        super().__init__(message, code=code, details=details, retryable=retryable)


class NotFound(DomainError):
    """Resource missing or not owned by caller (E-NF-01)."""

    def __init__(
        self,
        message: str = 'Resource not found',
        *,
        code: str = 'E-NF-01',
        details: dict[str, Any] | None = None,
        retryable: bool = False,
    ) -> None:
        super().__init__(message, code=code, details=details, retryable=retryable)


class StateConflict(DomainError):
    """Operation not allowed in the resource's current state (E-CONF-01)."""

    def __init__(
        self,
        message: str = 'Resource state conflict',
        *,
        code: str = 'E-CONF-01',
        details: dict[str, Any] | None = None,
        retryable: bool = False,
    ) -> None:
        super().__init__(message, code=code, details=details, retryable=retryable)


class Unauthorized(DomainError):
    """Missing, invalid, or expired authentication token (E-AUTH-01)."""

    def __init__(
        self,
        message: str = 'Unauthorized',
        *,
        code: str = 'E-AUTH-01',
        details: dict[str, Any] | None = None,
        retryable: bool = False,
    ) -> None:
        super().__init__(message, code=code, details=details, retryable=retryable)


class Forbidden(DomainError):
    """Access forbidden by permissions or role checks (E-AUTH-02)."""

    def __init__(
        self,
        message: str = 'Forbidden',
        *,
        code: str = 'E-AUTH-02',
        details: dict[str, Any] | None = None,
        retryable: bool = False,
    ) -> None:
        super().__init__(message, code=code, details=details, retryable=retryable)


class RateLimited(DomainError):
    """Per-user rate limit exceeded, carries retry_after (E-RATE-01)."""

    retry_after: float | None

    def __init__(
        self,
        message: str = 'Rate limit exceeded',
        *,
        retry_after: float | None = None,
        code: str = 'E-RATE-01',
        details: dict[str, Any] | None = None,
        retryable: bool = True,
    ) -> None:
        merged_details = dict(details) if details is not None else {}
        if retry_after is not None:
            merged_details.setdefault('retry_after', retry_after)
        super().__init__(message, code=code, details=merged_details, retryable=retryable)
        self.retry_after = retry_after


class PayloadTooLarge(DomainError):
    """Request payload exceeds maximum allowed size (E-REQ-01)."""

    def __init__(
        self,
        message: str = 'Payload too large',
        *,
        code: str = 'E-REQ-01',
        details: dict[str, Any] | None = None,
        retryable: bool = False,
    ) -> None:
        super().__init__(message, code=code, details=details, retryable=retryable)


class LLMError(MemoryEngineError):
    """Base error for LLM provider operations."""

    def __init__(
        self,
        message: str = 'LLM error',
        *,
        code: str = 'E-LLM-UNAVAILABLE',
        details: dict[str, Any] | None = None,
        retryable: bool = True,
    ) -> None:
        super().__init__(message, code=code, details=details, retryable=retryable)


class LLMTimeout(LLMError):
    """Provider timeout after fallbacks (E-LLM-TIMEOUT)."""

    def __init__(
        self,
        message: str = 'LLM provider timeout',
        *,
        code: str = 'E-LLM-TIMEOUT',
        details: dict[str, Any] | None = None,
        retryable: bool = True,
    ) -> None:
        super().__init__(message, code=code, details=details, retryable=retryable)


class LLMRateLimited(LLMError):
    """LLM provider rate limited, carries retry_after (E-LLM-RATELIMIT)."""

    retry_after: float | None

    def __init__(
        self,
        message: str = 'LLM provider rate limited',
        *,
        retry_after: float | None = None,
        code: str = 'E-LLM-RATELIMIT',
        details: dict[str, Any] | None = None,
        retryable: bool = True,
    ) -> None:
        merged_details = dict(details) if details is not None else {}
        if retry_after is not None:
            merged_details.setdefault('retry_after', retry_after)
        super().__init__(message, code=code, details=merged_details, retryable=retryable)
        self.retry_after = retry_after


class LLMUnavailable(LLMError):
    """Every provider in the chain failed, carries retry_after (E-LLM-UNAVAILABLE)."""

    retry_after: float | None

    def __init__(
        self,
        message: str = 'LLM provider unavailable',
        *,
        retry_after: float | None = None,
        code: str = 'E-LLM-UNAVAILABLE',
        details: dict[str, Any] | None = None,
        retryable: bool = True,
    ) -> None:
        merged_details = dict(details) if details is not None else {}
        if retry_after is not None:
            merged_details.setdefault('retry_after', retry_after)
        super().__init__(message, code=code, details=merged_details, retryable=retryable)
        self.retry_after = retry_after


class SchemaViolation(LLMError):
    """Model output failed validation after repair (E-LLM-SCHEMA)."""

    def __init__(
        self,
        message: str = 'LLM output violated schema',
        *,
        code: str = 'E-LLM-SCHEMA',
        details: dict[str, Any] | None = None,
        retryable: bool = True,
    ) -> None:
        super().__init__(message, code=code, details=details, retryable=retryable)


class LLMContentBlocked(LLMError):
    """Provider refused content (E-LLM-BLOCKED)."""

    def __init__(
        self,
        message: str = 'LLM content blocked',
        *,
        code: str = 'E-LLM-BLOCKED',
        details: dict[str, Any] | None = None,
        retryable: bool = False,
    ) -> None:
        super().__init__(message, code=code, details=details, retryable=retryable)


class StreamInterrupted(LLMError):
    """Stream interrupted before completion (E-LLM-STREAM)."""

    def __init__(
        self,
        message: str = 'LLM stream interrupted',
        *,
        code: str = 'E-LLM-STREAM',
        details: dict[str, Any] | None = None,
        retryable: bool = True,
    ) -> None:
        super().__init__(message, code=code, details=details, retryable=retryable)


class EmbeddingError(MemoryEngineError):
    """Embedding generation failed or queue saturated (E-EMB-01 or E-EMB-02)."""

    def __init__(
        self,
        message: str = 'Embedding error',
        *,
        code: str = 'E-EMB-01',
        details: dict[str, Any] | None = None,
        retryable: bool | None = None,
    ) -> None:
        is_retryable = retryable if retryable is not None else (code == 'E-EMB-02')
        super().__init__(message, code=code, details=details, retryable=is_retryable)


class StorageError(MemoryEngineError):
    """Base error for database and storage failures."""

    def __init__(
        self,
        message: str = 'Storage error',
        *,
        code: str = 'E-DB-01',
        details: dict[str, Any] | None = None,
        retryable: bool = True,
    ) -> None:
        super().__init__(message, code=code, details=details, retryable=retryable)


class TransientStorageError(StorageError):
    """Transient database failure: pool wait, lock timeout, connection loss (E-DB-01)."""

    def __init__(
        self,
        message: str = 'Transient storage error',
        *,
        code: str = 'E-DB-01',
        details: dict[str, Any] | None = None,
        retryable: bool = True,
    ) -> None:
        super().__init__(message, code=code, details=details, retryable=retryable)


class ConstraintViolation(StorageError):
    """Database constraint violation (E-DB-02)."""

    def __init__(
        self,
        message: str = 'Database constraint violation',
        *,
        code: str = 'E-DB-02',
        details: dict[str, Any] | None = None,
        retryable: bool = False,
    ) -> None:
        super().__init__(message, code=code, details=details, retryable=retryable)


class IngestionError(MemoryEngineError):
    """Base error for ingestion pipeline failures."""

    def __init__(
        self,
        message: str = 'Ingestion error',
        *,
        code: str = 'E-EXT-SCHEMA',
        details: dict[str, Any] | None = None,
        retryable: bool = False,
    ) -> None:
        super().__init__(message, code=code, details=details, retryable=retryable)


class ExtractionError(IngestionError):
    """Extraction output invalid after retries (E-EXT-SCHEMA)."""

    def __init__(
        self,
        message: str = 'Extraction error',
        *,
        code: str = 'E-EXT-SCHEMA',
        details: dict[str, Any] | None = None,
        retryable: bool = True,
    ) -> None:
        super().__init__(message, code=code, details=details, retryable=retryable)


class LeaseLost(IngestionError):
    """Job lock or lease lost / fenced out (E-JOB-LEASE)."""

    def __init__(
        self,
        message: str = 'Job lease lost',
        *,
        code: str = 'E-JOB-LEASE',
        details: dict[str, Any] | None = None,
        retryable: bool = False,
    ) -> None:
        super().__init__(message, code=code, details=details, retryable=retryable)


class RetrievalError(MemoryEngineError):
    """Base error for retrieval pipeline failures."""

    def __init__(
        self,
        message: str = 'Retrieval error',
        *,
        code: str = 'E-RET-01',
        details: dict[str, Any] | None = None,
        retryable: bool = False,
    ) -> None:
        super().__init__(message, code=code, details=details, retryable=retryable)


class PlanningError(RetrievalError):
    """Query planner failed to produce a valid query plan (E-RET-02)."""

    def __init__(
        self,
        message: str = 'Query planning error',
        *,
        code: str = 'E-RET-02',
        details: dict[str, Any] | None = None,
        retryable: bool = False,
    ) -> None:
        super().__init__(message, code=code, details=details, retryable=retryable)


class AllLanesFailed(RetrievalError):
    """All retrieval lanes failed to return candidates (E-RET-01)."""

    def __init__(
        self,
        message: str = 'All retrieval lanes failed',
        *,
        code: str = 'E-RET-01',
        details: dict[str, Any] | None = None,
        retryable: bool = False,
    ) -> None:
        super().__init__(message, code=code, details=details, retryable=retryable)


class ConfigError(MemoryEngineError):
    """Startup configuration guard failure (E-CONFIG-01)."""

    def __init__(
        self,
        message: str = 'Configuration error',
        *,
        code: str = 'E-CONFIG-01',
        details: dict[str, Any] | None = None,
        retryable: bool = False,
    ) -> None:
        super().__init__(message, code=code, details=details, retryable=retryable)


__all__ = [
    'AllLanesFailed',
    'ConfigError',
    'ConstraintViolation',
    'DomainError',
    'EmbeddingError',
    'ExtractionError',
    'Forbidden',
    'IngestionError',
    'InvariantViolation',
    'LLMContentBlocked',
    'LLMError',
    'LLMRateLimited',
    'LLMTimeout',
    'LLMUnavailable',
    'LeaseLost',
    'MemoryEngineError',
    'NotFound',
    'PayloadTooLarge',
    'PlanningError',
    'RateLimited',
    'RetrievalError',
    'SchemaViolation',
    'StateConflict',
    'StorageError',
    'StreamInterrupted',
    'TransientStorageError',
    'Unauthorized',
    'ValidationFailed',
]
