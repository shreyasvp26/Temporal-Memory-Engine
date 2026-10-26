"""Unit tests for domain primitives: IDs, Enums, and Errors."""

import uuid

from memory_engine.domain.enums import (
    Cardinality,
    DecayClass,
    EntityType,
    FactOrigin,
    FactStatus,
    Intent,
    JobStatus,
    JobType,
    ObjectKind,
    RelationType,
    Role,
    SourceKind,
    TimePrecision,
)
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
from memory_engine.domain.ids import (
    EntityId,
    FactId,
    JobId,
    SourceId,
    UserId,
)


def test_domain_ids() -> None:
    """Verify UUID NewType aliases instantiate and preserve UUID semantics."""
    raw_uuid = uuid.uuid4()

    fact_id = FactId(raw_uuid)
    source_id = SourceId(raw_uuid)
    entity_id = EntityId(raw_uuid)
    job_id = JobId(raw_uuid)
    user_id = UserId(raw_uuid)

    assert fact_id == raw_uuid
    assert source_id == raw_uuid
    assert entity_id == raw_uuid
    assert job_id == raw_uuid
    assert user_id == raw_uuid

    assert str(fact_id) == str(raw_uuid)


def test_cardinality_enum_values() -> None:
    """Verify Cardinality enum members and string values."""
    assert Cardinality.SINGLE.value == 'single'
    assert Cardinality.MULTI.value == 'multi'
    assert Cardinality.EVENT.value == 'event'
    assert len(Cardinality) == 3
    assert isinstance(Cardinality.SINGLE, str)
    assert Cardinality('single') is Cardinality.SINGLE


def test_decay_class_enum_values() -> None:
    """Verify DecayClass enum members and string values."""
    assert DecayClass.PERMANENT.value == 'permanent'
    assert DecayClass.SLOW.value == 'slow'
    assert DecayClass.NORMAL.value == 'normal'
    assert DecayClass.FAST.value == 'fast'
    assert len(DecayClass) == 4
    assert DecayClass('permanent') is DecayClass.PERMANENT


def test_object_kind_enum_values() -> None:
    """Verify ObjectKind enum members and string values."""
    assert ObjectKind.ENTITY.value == 'entity'
    assert ObjectKind.LITERAL.value == 'literal'
    assert len(ObjectKind) == 2
    assert ObjectKind('entity') is ObjectKind.ENTITY


def test_entity_type_enum_values() -> None:
    """Verify EntityType enum members and string values."""
    assert EntityType.PERSON.value == 'person'
    assert EntityType.PLACE.value == 'place'
    assert EntityType.ORG.value == 'org'
    assert EntityType.THING.value == 'thing'
    assert EntityType.TOPIC.value == 'topic'
    assert len(EntityType) == 5
    assert EntityType('person') is EntityType.PERSON


def test_time_precision_enum_values() -> None:
    """Verify TimePrecision enum members and string values."""
    assert TimePrecision.YEAR.value == 'year'
    assert TimePrecision.MONTH.value == 'month'
    assert TimePrecision.DAY.value == 'day'
    assert TimePrecision.EXACT.value == 'exact'
    assert len(TimePrecision) == 4
    assert TimePrecision('year') is TimePrecision.YEAR


def test_fact_status_enum_values() -> None:
    """Verify FactStatus enum members and string values."""
    assert FactStatus.ACTIVE.value == 'active'
    assert FactStatus.ARCHIVED.value == 'archived'
    assert FactStatus.DELETED.value == 'deleted'
    assert FactStatus.NEEDS_REVIEW.value == 'needs_review'
    assert len(FactStatus) == 4
    assert FactStatus('active') is FactStatus.ACTIVE


def test_fact_origin_enum_values() -> None:
    """Verify FactOrigin enum members and string values."""
    assert FactOrigin.EXTRACTED.value == 'extracted'
    assert FactOrigin.USER_EDIT.value == 'user_edit'
    assert len(FactOrigin) == 2
    assert FactOrigin('extracted') is FactOrigin.EXTRACTED


def test_source_kind_enum_values() -> None:
    """Verify SourceKind enum members and string values."""
    assert SourceKind.CHAT.value == 'chat'
    assert SourceKind.NOTE.value == 'note'
    assert SourceKind.DOC.value == 'doc'
    assert len(SourceKind) == 3
    assert SourceKind('chat') is SourceKind.CHAT


def test_role_enum_values() -> None:
    """Verify Role enum members and string values."""
    assert Role.USER.value == 'user'
    assert Role.ASSISTANT.value == 'assistant'
    assert Role.SYSTEM.value == 'system'
    assert len(Role) == 3
    assert Role('user') is Role.USER


def test_job_type_enum_values() -> None:
    """Verify JobType enum members and string values."""
    assert JobType.EXTRACT.value == 'extract'
    assert JobType.PURGE.value == 'purge'
    assert JobType.DECAY.value == 'decay'
    assert JobType.REINDEX.value == 'reindex'
    assert len(JobType) == 4
    assert JobType('extract') is JobType.EXTRACT


def test_job_status_enum_values() -> None:
    """Verify JobStatus enum members and string values."""
    assert JobStatus.QUEUED.value == 'queued'
    assert JobStatus.RUNNING.value == 'running'
    assert JobStatus.DONE.value == 'done'
    assert JobStatus.DEAD.value == 'dead'
    assert len(JobStatus) == 4
    assert JobStatus('queued') is JobStatus.QUEUED


def test_relation_type_enum_values() -> None:
    """Verify RelationType enum members and string values."""
    assert RelationType.SUPERSEDES.value == 'supersedes'
    assert RelationType.SUPPORTS.value == 'supports'
    assert RelationType.CLARIFIES.value == 'clarifies'
    assert RelationType.DERIVED_FROM.value == 'derived_from'
    assert len(RelationType) == 4
    assert RelationType('supersedes') is RelationType.SUPERSEDES


def test_intent_enum_values() -> None:
    """Verify Intent enum members and string values."""
    assert Intent.CURRENT_STATE.value == 'current_state'
    assert Intent.HISTORICAL.value == 'historical'
    assert Intent.POINT_IN_TIME.value == 'point_in_time'
    assert Intent.INTERVAL.value == 'interval'
    assert Intent.AGGREGATION.value == 'aggregation'
    assert Intent.DURATION_OR_ORDER.value == 'duration_or_order'
    assert Intent.OPEN.value == 'open'
    assert len(Intent) == 7
    assert Intent('current_state') is Intent.CURRENT_STATE


def test_domain_error_hierarchy_and_defaults() -> None:
    """Verify DomainError subclasses and error codes."""
    err_val = ValidationFailed('Field missing', details={'field': 'content'})
    assert isinstance(err_val, DomainError)
    assert isinstance(err_val, MemoryEngineError)
    assert err_val.code == 'E-VAL-01'
    assert not err_val.retryable
    assert err_val.details == {'field': 'content'}
    assert str(err_val) == 'Field missing'

    err_inv = InvariantViolation('Time must be UTC')
    assert err_inv.code == 'E-INT-01'
    assert not err_inv.retryable

    err_nf = NotFound('Fact not found')
    assert err_nf.code == 'E-NF-01'
    assert not err_nf.retryable

    err_conf = StateConflict('Cannot archive active fact')
    assert err_conf.code == 'E-CONF-01'
    assert not err_conf.retryable

    err_auth = Unauthorized('Missing token')
    assert err_auth.code == 'E-AUTH-01'
    assert not err_auth.retryable

    err_forbid = Forbidden('No permission')
    assert err_forbid.code == 'E-AUTH-02'
    assert not err_forbid.retryable

    err_rate = RateLimited('Rate exceeded', retry_after=30.0)
    assert err_rate.code == 'E-RATE-01'
    assert err_rate.retryable
    assert err_rate.retry_after == 30.0
    assert err_rate.details['retry_after'] == 30.0

    err_req = PayloadTooLarge('Body exceeds limit')
    assert err_req.code == 'E-REQ-01'
    assert not err_req.retryable


def test_llm_error_hierarchy_and_defaults() -> None:
    """Verify LLMError subclasses and error codes."""
    err_timeout = LLMTimeout()
    assert isinstance(err_timeout, LLMError)
    assert isinstance(err_timeout, MemoryEngineError)
    assert err_timeout.code == 'E-LLM-TIMEOUT'
    assert err_timeout.retryable

    err_rate = LLMRateLimited(retry_after=12.5)
    assert err_rate.code == 'E-LLM-RATELIMIT'
    assert err_rate.retryable
    assert err_rate.retry_after == 12.5
    assert err_rate.details['retry_after'] == 12.5

    err_unavail = LLMUnavailable(retry_after=5.0)
    assert err_unavail.code == 'E-LLM-UNAVAILABLE'
    assert err_unavail.retryable
    assert err_unavail.retry_after == 5.0

    err_schema = SchemaViolation()
    assert err_schema.code == 'E-LLM-SCHEMA'
    assert err_schema.retryable

    err_blocked = LLMContentBlocked()
    assert err_blocked.code == 'E-LLM-BLOCKED'
    assert not err_blocked.retryable

    err_stream = StreamInterrupted()
    assert err_stream.code == 'E-LLM-STREAM'
    assert err_stream.retryable


def test_embedding_error_codes() -> None:
    """Verify EmbeddingError error codes and retryable flags."""
    err1 = EmbeddingError('Bad dimension', code='E-EMB-01')
    assert err1.code == 'E-EMB-01'
    assert not err1.retryable

    err2 = EmbeddingError('Queue saturated', code='E-EMB-02')
    assert err2.code == 'E-EMB-02'
    assert err2.retryable

    # Explicit override takes precedence
    err3 = EmbeddingError('Bad output with override', code='E-EMB-01', retryable=True)
    assert err3.retryable


def test_storage_error_hierarchy_and_defaults() -> None:
    """Verify StorageError subclasses and error codes."""
    err_trans = TransientStorageError('Connection lost')
    assert isinstance(err_trans, StorageError)
    assert err_trans.code == 'E-DB-01'
    assert err_trans.retryable

    err_const = ConstraintViolation('Unique violation')
    assert isinstance(err_const, StorageError)
    assert err_const.code == 'E-DB-02'
    assert not err_const.retryable


def test_ingestion_error_hierarchy_and_defaults() -> None:
    """Verify IngestionError subclasses and error codes."""
    err_ext = ExtractionError('Output invalid')
    assert isinstance(err_ext, IngestionError)
    assert err_ext.code == 'E-EXT-SCHEMA'
    assert err_ext.retryable

    err_lease = LeaseLost('Worker lost fence')
    assert isinstance(err_lease, IngestionError)
    assert err_lease.code == 'E-JOB-LEASE'
    assert not err_lease.retryable


def test_retrieval_and_config_error_hierarchy() -> None:
    """Verify RetrievalError subclasses and ConfigError."""
    err_plan = PlanningError('Invalid grammar')
    assert isinstance(err_plan, RetrievalError)
    assert err_plan.code == 'E-RET-02'
    assert not err_plan.retryable

    err_lanes = AllLanesFailed('All lanes failed')
    assert isinstance(err_lanes, RetrievalError)
    assert err_lanes.code == 'E-RET-01'
    assert not err_lanes.retryable

    err_cfg = ConfigError('Missing setting')
    assert isinstance(err_cfg, MemoryEngineError)
    assert err_cfg.code == 'E-CONFIG-01'
    assert not err_cfg.retryable


def test_error_details_isolation() -> None:
    """Verify details dict is safely copied and cannot be mutated externally."""
    details = {'k': 'v'}
    err = MemoryEngineError('msg', code='E-INT-01', details=details)
    details['k'] = 'mutated'
    assert err.details['k'] == 'v'
