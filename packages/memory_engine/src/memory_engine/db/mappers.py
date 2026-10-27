"""Pure row-mapping functions converting database records to frozen domain models."""

import json
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID

from memory_engine.domain.enums import (
    Cardinality,
    DecayClass,
    EntityType,
    FactOrigin,
    FactStatus,
    JobStatus,
    JobType,
    ObjectKind,
    RelationType,
    Role,
    SourceKind,
    TimePrecision,
)
from memory_engine.domain.errors import InvariantViolation
from memory_engine.domain.ids import (
    EntityId,
    FactId,
    JobId,
    SourceId,
    UserId,
)
from memory_engine.domain.models import (
    Entity,
    Fact,
    FactEvidence,
    FactRelation,
    Job,
    Message,
    Predicate,
    Source,
)
from memory_engine.ports.repositories import (
    EvidenceSnippet,
    ScoredFact,
    ScoredSource,
)


def _to_uuid(val: Any, field_name: str = 'id') -> UUID:
    if isinstance(val, UUID):
        return val
    try:
        return UUID(str(val))
    except (ValueError, AttributeError) as exc:
        raise InvariantViolation(f'{field_name} must be a valid UUID, got {val!r}') from exc


def _to_utc_datetime(val: Any, field_name: str) -> datetime:
    if val is None:
        raise InvariantViolation(f'{field_name} cannot be None')
    dt: datetime
    if isinstance(val, str):
        dt = datetime.fromisoformat(val)
    elif isinstance(val, datetime):
        dt = val
    else:
        raise InvariantViolation(
            f'{field_name} must be a datetime or ISO string, got {type(val)!r}'
        )
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    elif dt.utcoffset() != timedelta(0):
        dt = dt.astimezone(UTC)
    return dt


def _to_optional_utc_datetime(val: Any, field_name: str) -> datetime | None:
    if val is None:
        return None
    return _to_utc_datetime(val, field_name)


def _parse_payload(val: Any) -> dict[str, Any]:
    if isinstance(val, str):
        parsed = json.loads(val)
        if isinstance(parsed, dict):
            return dict(parsed)
        raise InvariantViolation(f'payload must be a JSON object, got {type(parsed)!r}')
    if isinstance(val, dict):
        return dict(val)
    raise InvariantViolation(f'payload must be a dict or JSON string, got {type(val)!r}')


def map_source(row: Mapping[str, Any]) -> Source:
    """Map a database row to a frozen Source domain entity."""
    return Source(
        id=SourceId(_to_uuid(row['id'], 'id')),
        user_id=UserId(_to_uuid(row['user_id'], 'user_id')),
        kind=SourceKind(row['kind']),
        role=Role(row['role']),
        session_id=str(row['session_id']),
        turn_index=int(row['turn_index']),
        content=str(row['content']),
        source_ts=_to_utc_datetime(row['source_ts'], 'source_ts'),
        content_hash=str(row['content_hash']),
        deleted_at=_to_optional_utc_datetime(row.get('deleted_at'), 'deleted_at'),
    )


def map_entity(row: Mapping[str, Any]) -> Entity:
    """Map a database row to a frozen Entity domain model."""
    return Entity(
        id=EntityId(_to_uuid(row['id'], 'id')),
        user_id=UserId(_to_uuid(row['user_id'], 'user_id')),
        canonical_name=str(row['canonical_name']),
        type=EntityType(row['type']),
        is_self=bool(row['is_self']),
    )


def map_predicate(row: Mapping[str, Any]) -> Predicate:
    """Map a database row to a frozen Predicate domain model."""
    synonyms_raw = row.get('synonyms')
    synonyms: tuple[str, ...] = (
        tuple(str(s) for s in synonyms_raw) if synonyms_raw is not None else ()
    )
    return Predicate(
        name=str(row['name']),
        cardinality=Cardinality(row['cardinality']),
        decay_class=DecayClass(row['decay_class']),
        object_kind=ObjectKind(row['object_kind']),
        template=str(row['template']),
        synonyms=synonyms,
    )


def map_fact(row: Mapping[str, Any]) -> Fact:
    """Map a database row to a frozen Fact domain entity."""
    object_entity_id = (
        EntityId(_to_uuid(row['object_entity_id'], 'object_entity_id'))
        if row.get('object_entity_id') is not None
        else None
    )
    object_value = str(row['object_value']) if row.get('object_value') is not None else None

    return Fact(
        id=FactId(_to_uuid(row['id'], 'id')),
        user_id=UserId(_to_uuid(row['user_id'], 'user_id')),
        subject_id=EntityId(_to_uuid(row['subject_id'], 'subject_id')),
        predicate=str(row['predicate']),
        cardinality=Cardinality(row['cardinality']),
        object_entity_id=object_entity_id,
        object_value=object_value,
        object_norm=str(row['object_norm']),
        fact_text=str(row['fact_text']),
        valid_from=_to_utc_datetime(row['valid_from'], 'valid_from'),
        valid_to=_to_optional_utc_datetime(row.get('valid_to'), 'valid_to'),
        time_precision=TimePrecision(row['time_precision']),
        valid_from_explicit=bool(row['valid_from_explicit']),
        recorded_at=_to_utc_datetime(row['recorded_at'], 'recorded_at'),
        invalidated_at=_to_optional_utc_datetime(row.get('invalidated_at'), 'invalidated_at'),
        confidence=float(row['confidence']),
        salience=float(row['salience']),
        access_count=int(row['access_count']),
        last_accessed_at=_to_optional_utc_datetime(row.get('last_accessed_at'), 'last_accessed_at'),
        status=FactStatus(row['status']),
        origin=FactOrigin(row['origin']),
        event_hash=str(row['event_hash']) if row.get('event_hash') is not None else None,
    )


def map_job(row: Mapping[str, Any]) -> Job:
    """Map a database row to a frozen Job domain model."""
    worker_id = (
        _to_uuid(row['worker_id'], 'worker_id') if row.get('worker_id') is not None else None
    )
    return Job(
        id=JobId(_to_uuid(row['id'], 'id')),
        user_id=UserId(_to_uuid(row['user_id'], 'user_id')),
        type=JobType(row['type']),
        payload=_parse_payload(row['payload']),
        idempotency_key=str(row['idempotency_key']),
        coalesce_key=str(row['coalesce_key']) if row.get('coalesce_key') is not None else None,
        status=JobStatus(row['status']),
        attempts=int(row['attempts']),
        max_attempts=int(row['max_attempts']),
        deferrals=int(row['deferrals']),
        run_after=_to_utc_datetime(row['run_after'], 'run_after'),
        worker_id=worker_id,
        lease_expires_at=_to_optional_utc_datetime(row.get('lease_expires_at'), 'lease_expires_at'),
        last_error=str(row['last_error']) if row.get('last_error') is not None else None,
    )


def map_message(row: Mapping[str, Any]) -> Message:
    """Map a database row to a frozen Message domain entity."""
    source_id = (
        SourceId(_to_uuid(row['source_id'], 'source_id'))
        if row.get('source_id') is not None
        else None
    )
    return Message(
        id=_to_uuid(row['id'], 'id'),
        conversation_id=_to_uuid(row['conversation_id'], 'conversation_id'),
        user_id=UserId(_to_uuid(row['user_id'], 'user_id')),
        role=Role(row['role']),
        content=str(row['content']),
        client_message_id=(
            str(row['client_message_id']) if row.get('client_message_id') is not None else None
        ),
        source_id=source_id,
        partial=bool(row['partial']),
        created_at=_to_utc_datetime(row['created_at'], 'created_at'),
    )


def map_scored_fact(row: Mapping[str, Any], score_key: str = 'score') -> ScoredFact:
    """Map a database row with a score column to a ScoredFact DTO."""
    fact = map_fact(row)
    score = float(row[score_key])
    return ScoredFact(fact=fact, score=score)


def map_scored_source(row: Mapping[str, Any], score_key: str = 'score') -> ScoredSource:
    """Map a database row with a score column to a ScoredSource DTO."""
    source = map_source(row)
    score = float(row[score_key])
    return ScoredSource(source=source, score=score)


def map_evidence_snippet(row: Mapping[str, Any]) -> EvidenceSnippet:
    """Map a database row to an EvidenceSnippet DTO."""
    fact_id = (
        FactId(_to_uuid(row['fact_id'], 'fact_id')) if row.get('fact_id') is not None else None
    )
    return EvidenceSnippet(
        source_id=SourceId(_to_uuid(row['source_id'], 'source_id')),
        source_ts=_to_utc_datetime(row['source_ts'], 'source_ts'),
        role=Role(row['role']),
        snippet=str(row['snippet']),
        fact_id=fact_id,
    )


def map_fact_evidence(row: Mapping[str, Any]) -> FactEvidence:
    """Map a database row to a FactEvidence domain model."""
    span_start = int(row['span_start']) if row.get('span_start') is not None else None
    span_end = int(row['span_end']) if row.get('span_end') is not None else None
    return FactEvidence(
        fact_id=FactId(_to_uuid(row['fact_id'], 'fact_id')),
        source_id=SourceId(_to_uuid(row['source_id'], 'source_id')),
        span_start=span_start,
        span_end=span_end,
    )


def map_fact_relation(row: Mapping[str, Any]) -> FactRelation:
    """Map a database row to a FactRelation domain model."""
    return FactRelation(
        from_fact_id=FactId(_to_uuid(row['from_fact_id'], 'from_fact_id')),
        to_fact_id=FactId(_to_uuid(row['to_fact_id'], 'to_fact_id')),
        relation=RelationType(row['relation']),
    )


__all__ = [
    'map_entity',
    'map_evidence_snippet',
    'map_fact',
    'map_fact_evidence',
    'map_fact_relation',
    'map_job',
    'map_message',
    'map_predicate',
    'map_scored_fact',
    'map_scored_source',
    'map_source',
]
