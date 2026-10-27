"""Repository port protocols and supporting data transfer objects."""

import builtins
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum
from typing import Any, Generic, Protocol, TypeVar
from uuid import UUID

from pydantic import BaseModel, ConfigDict

from memory_engine.domain.enums import (
    EntityType,
    FactOrigin,
    FactStatus,
    JobStatus,
    JobType,
    Role,
    SourceKind,
)
from memory_engine.domain.errors import InvariantViolation
from memory_engine.domain.ids import (
    EntityId,
    FactId,
    JobId,
    SourceId,
    UserId,
)
from memory_engine.domain.intervals import TimeInterval
from memory_engine.domain.models import (
    Candidate,
    Entity,
    Fact,
    FactEvidence,
    FactRelation,
    Job,
    Message,
    Predicate,
    Source,
)


def _validate_utc_datetime(dt: datetime, name: str) -> None:
    if not isinstance(dt, datetime):
        raise InvariantViolation(f'{name} must be a datetime, got {type(dt)!r}')
    if dt.tzinfo is None or dt.utcoffset() != timedelta(0):
        raise InvariantViolation(f'{name} must be timezone-aware UTC, got {dt!r}')


@dataclass(frozen=True, slots=True)
class NewSource:
    user_id: UserId
    kind: SourceKind
    role: Role
    session_id: str
    turn_index: int
    content: str
    source_ts: datetime
    content_hash: str
    id: SourceId | None = None

    def __post_init__(self) -> None:
        _validate_utc_datetime(self.source_ts, 'source_ts')


@dataclass(frozen=True, slots=True)
class InsertedSource:
    id: SourceId
    created: bool


@dataclass(frozen=True, slots=True)
class SourceWindow:
    targets: Sequence[Source]
    context: Sequence[Source]


@dataclass(frozen=True, slots=True)
class FactListQuery:
    status: FactStatus | str | None = FactStatus.ACTIVE
    predicate: str | None = None
    entity_id: EntityId | None = None
    q: str | None = None
    as_of: datetime | None = None
    current_only: bool = False
    sort: str = 'valid_from'
    page_size: int = 25
    cursor: str | None = None

    def __post_init__(self) -> None:
        if self.as_of is not None:
            _validate_utc_datetime(self.as_of, 'as_of')


T = TypeVar('T')


class Page(BaseModel, Generic[T]):
    """Paginated result wrapper."""

    model_config = ConfigDict(extra='forbid', frozen=True)

    items: list[T]
    next_cursor: str | None = None


@dataclass(frozen=True, slots=True)
class ScoredFact:
    fact: Fact
    score: float


@dataclass(frozen=True, slots=True)
class ScoredSource:
    source: Source
    score: float


@dataclass(frozen=True, slots=True)
class EvidenceSnippet:
    source_id: SourceId
    source_ts: datetime
    role: Role
    snippet: str
    fact_id: FactId | None = None

    def __post_init__(self) -> None:
        _validate_utc_datetime(self.source_ts, 'source_ts')


@dataclass(frozen=True, slots=True)
class NewExtractJob:
    user_id: UserId
    session_id: str
    source_id: SourceId
    idempotency_key: str
    coalesce_key: str


@dataclass(frozen=True, slots=True)
class UpsertResult:
    job_id: JobId
    created: bool
    sealed: bool = False


@dataclass(frozen=True, slots=True)
class NewJob:
    user_id: UserId
    type: JobType
    payload: dict[str, Any]
    idempotency_key: str
    coalesce_key: str | None = None
    run_after: datetime | None = None
    max_attempts: int = 3

    def __post_init__(self) -> None:
        if self.run_after is not None:
            _validate_utc_datetime(self.run_after, 'run_after')


@dataclass(frozen=True, slots=True)
class ErrorInfo:
    message: str
    code: str | None = None
    detail: str | None = None

    @property
    def last_error(self) -> str:
        return self.message

    @property
    def error_code(self) -> str | None:
        return self.code


@dataclass(frozen=True, slots=True)
class NewMessage:
    conversation_id: UUID
    user_id: UserId
    role: Role
    content: str
    client_message_id: str | None = None
    source_id: SourceId | None = None
    partial: bool = False
    trace: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class InsertedMessage:
    id: UUID
    created: bool


class FilterMode(StrEnum):
    CURRENT = 'current'
    AS_OF = 'as_of'
    OVERLAP = 'overlap'
    ANY_HISTORY = 'any_history'
    OPEN = 'open'

    @classmethod
    def _missing_(cls, value: object) -> 'FilterMode | None':
        if isinstance(value, str):
            v = value.lower()
            for member in cls:
                if member.value == v:
                    return member
        return None


@dataclass(frozen=True, slots=True)
class SqlFragment:
    sql: str
    params: tuple[Any, ...] = ()

    @property
    def text(self) -> str:
        return self.sql


@dataclass(frozen=True, slots=True)
class TemporalFilter:
    mode: FilterMode
    as_of: datetime | None = None
    range: TimeInterval | None = None
    cutoff: datetime | None = None

    def __post_init__(self) -> None:
        if self.as_of is not None:
            _validate_utc_datetime(self.as_of, 'as_of')
        if self.cutoff is not None:
            _validate_utc_datetime(self.cutoff, 'cutoff')

    @staticmethod
    def from_plan(plan: Any) -> 'TemporalFilter':
        time_range = getattr(plan, 'time_range', None)
        as_of = getattr(plan, 'as_of', None)
        intent = getattr(plan, 'intent', None)
        question_ts = getattr(plan, 'question_ts', None)

        if time_range is not None:
            return TemporalFilter(mode=FilterMode.OVERLAP, range=time_range)

        if as_of is not None:
            return TemporalFilter(mode=FilterMode.AS_OF, as_of=as_of)

        intent_str = str(intent).lower() if intent is not None else ''
        if 'current_state' in intent_str or intent_str == 'current':
            return TemporalFilter(mode=FilterMode.CURRENT)
        if 'point_in_time' in intent_str:
            return TemporalFilter(mode=FilterMode.AS_OF, as_of=as_of)
        if 'interval' in intent_str:
            return TemporalFilter(mode=FilterMode.OVERLAP, range=time_range)
        if any(k in intent_str for k in ('historical', 'duration_or_order', 'aggregation')):
            return TemporalFilter(mode=FilterMode.ANY_HISTORY)
        if 'open' in intent_str:
            cutoff = (question_ts - timedelta(days=365)) if question_ts is not None else None
            return TemporalFilter(mode=FilterMode.OPEN, cutoff=cutoff)

        return TemporalFilter(mode=FilterMode.CURRENT)

    def to_sql(
        self,
        alias: str | int = 'f',
        first_param: int | str = 1,
    ) -> SqlFragment:
        if isinstance(alias, int) and isinstance(first_param, str):
            alias, first_param = first_param, alias
        first_param_int = int(first_param)
        alias_str = str(alias)

        if self.mode == FilterMode.CURRENT:
            return SqlFragment(
                sql=(
                    f"{alias_str}.status = 'active' and {alias_str}.valid_to is null "
                    f'and {alias_str}.invalidated_at is null'
                ),
                params=(),
            )

        if self.mode == FilterMode.AS_OF:
            if self.as_of is None:
                raise InvariantViolation('as_of must be provided for AS_OF filter mode')
            return SqlFragment(
                sql=(
                    f"{alias_str}.status in ('active','archived') "
                    f'and {alias_str}.valid_from <= ${first_param_int} '
                    f'and ({alias_str}.valid_to is null or '
                    f'${first_param_int} < {alias_str}.valid_to)'
                ),
                params=(self.as_of,),
            )

        if self.mode == FilterMode.OVERLAP:
            clauses = [f"{alias_str}.status in ('active','archived')"]
            params_list: list[Any] = []
            curr_param = first_param_int

            if self.range is not None and self.range.end is not None:
                clauses.append(f'{alias_str}.valid_from < ${curr_param}')
                params_list.append(self.range.end)
                curr_param += 1

            if self.range is not None and self.range.start is not None:
                clauses.append(
                    f'({alias_str}.valid_to is null or {alias_str}.valid_to > ${curr_param})'
                )
                params_list.append(self.range.start)
                curr_param += 1

            return SqlFragment(
                sql=' and '.join(clauses),
                params=tuple(params_list),
            )

        if self.mode == FilterMode.ANY_HISTORY:
            return SqlFragment(
                sql=f"{alias_str}.status in ('active','archived')",
                params=(),
            )

        if self.mode == FilterMode.OPEN:
            if self.cutoff is not None:
                return SqlFragment(
                    sql=(
                        f"{alias_str}.status = 'active' and ({alias_str}.valid_to is null "
                        f'or {alias_str}.valid_to > ${first_param_int})'
                    ),
                    params=(self.cutoff,),
                )
            return SqlFragment(
                sql=f"{alias_str}.status = 'active' and {alias_str}.valid_to is null",
                params=(),
            )

        raise InvariantViolation(f'Unknown filter mode: {self.mode}')


class SourceRepository(Protocol):
    async def insert_many(self, items: Sequence[NewSource]) -> list[InsertedSource]: ...

    async def get_many(self, user_id: UserId, ids: Sequence[SourceId]) -> list[Source]: ...

    async def window(
        self,
        user_id: UserId,
        session_id: str,
        target_ids: Sequence[SourceId],
        context_turns: int,
    ) -> SourceWindow: ...

    async def set_embeddings(
        self,
        user_id: UserId,
        items: Sequence[tuple[SourceId, list[float]]],
        model_id: str,
    ) -> None: ...

    async def soft_delete(self, user_id: UserId, ids: Sequence[SourceId], at: datetime) -> None: ...

    async def hard_delete(self, user_id: UserId, ids: Sequence[SourceId]) -> int: ...


class EntityRepository(Protocol):
    async def ensure_self(self, user_id: UserId) -> Entity: ...

    async def get_or_create_self(self, user_id: UserId) -> Entity: ...

    async def find_by_alias(
        self, user_id: UserId, type: EntityType, alias_norm: str
    ) -> Entity | None: ...

    async def find_by_norm_name(
        self,
        user_id: UserId,
        name_or_type: str | EntityType,
        alias_or_type: str | EntityType | None = None,
        entity_type: EntityType | None = None,
    ) -> Entity | None: ...

    async def nearest(
        self,
        user_id: UserId,
        type: EntityType,
        embedding: list[float],
        k: int,
    ) -> list[tuple[Entity, float]]: ...

    async def search_similar_by_embedding(
        self,
        user_id: UserId,
        type_or_emb: EntityType | Sequence[float],
        embedding: Sequence[float] | None = None,
        k: int | None = None,
        *,
        min_cosine: float = 0.92,
        limit: int = 3,
    ) -> list[tuple[Entity, float]]: ...

    async def create(
        self,
        user_id: UserId,
        name: str,
        type: EntityType,
        embedding: list[float],
    ) -> Entity: ...

    async def add_alias(
        self,
        user_id: UserId,
        entity_id: EntityId,
        type: EntityType,
        alias: str,
        alias_norm: str,
    ) -> None: ...

    async def get_many(
        self, user_id: UserId, ids: Sequence[EntityId]
    ) -> dict[EntityId, Entity]: ...

    async def alias_index(self, user_id: UserId, limit: int = 5000) -> dict[str, EntityId]: ...


class FactRepository(Protocol):
    async def load_timeline(
        self,
        user_id: UserId,
        subject_id: EntityId,
        predicate: str,
        *,
        lock: bool,
    ) -> list[Fact]: ...

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
    ) -> Fact: ...

    async def close(
        self, user_id: UserId, fact_id: FactId, valid_to: datetime, now: datetime
    ) -> None: ...

    async def extend_valid_from(
        self, user_id: UserId, fact_id: FactId, valid_from: datetime
    ) -> None: ...

    async def set_status(
        self,
        user_id: UserId,
        fact_id: FactId,
        status: FactStatus,
        now: datetime,
    ) -> None: ...

    async def bump_confidence(
        self, user_id: UserId, fact_id: FactId, confidence: float
    ) -> None: ...

    async def add_evidence(self, ev: FactEvidence, user_id: UserId) -> None: ...

    async def add_relation(
        self, user_id: UserId, rel: FactRelation, job_id: UUID | None
    ) -> None: ...

    async def get(self, user_id: UserId, fact_id: FactId) -> Fact | None: ...

    async def list(self, user_id: UserId, q: FactListQuery) -> Page[Fact]: ...

    async def chain(self, user_id: UserId, fact_id: FactId) -> builtins.list[Fact]: ...

    async def only_supported_by(
        self, user_id: UserId, source_ids: Sequence[SourceId]
    ) -> builtins.list[FactId]: ...

    async def hard_delete(self, user_id: UserId, fact_ids: Sequence[FactId]) -> int: ...

    async def record_access(
        self, user_id: UserId, fact_ids: Sequence[FactId], at: datetime
    ) -> None: ...

    async def decay_candidates(self, user_id: UserId, now: datetime) -> builtins.list[Fact]: ...

    async def update_salience(
        self, user_id: UserId, items: Sequence[tuple[FactId, float]]
    ) -> None: ...


class SearchRepository(Protocol):
    async def vector_facts(
        self,
        user_id: UserId,
        embedding: Sequence[float],
        flt: TemporalFilter,
        limit: int,
    ) -> list[ScoredFact]: ...

    async def keyword_facts(
        self,
        user_id: UserId,
        query: str,
        flt: TemporalFilter,
        limit: int,
    ) -> list[ScoredFact]: ...

    async def structured_facts(
        self,
        user_id: UserId,
        subject_ids: Sequence[EntityId] | None,
        predicates: Sequence[str] | None,
        flt: TemporalFilter,
        limit: int,
    ) -> list[ScoredFact]: ...

    async def vector_sources(
        self,
        user_id: UserId,
        embedding: Sequence[float],
        limit: int,
        before: datetime | None,
    ) -> list[ScoredSource]: ...

    async def keyword_sources(
        self,
        user_id: UserId,
        query: str,
        limit: int,
        before: datetime | None,
    ) -> list[ScoredSource]: ...

    async def expand_graph(
        self,
        user_id: UserId,
        seed_entity_ids: Sequence[EntityId],
        depth: int,
        flt: TemporalFilter,
        limit: int,
    ) -> list[ScoredFact]: ...

    async def evidence_for(
        self,
        user_id: UserId,
        fact_ids: Sequence[FactId],
        per_fact: int,
    ) -> dict[FactId, list[EvidenceSnippet]]: ...

    async def chain_neighbors(self, user_id: UserId, fact_ids: Sequence[FactId]) -> list[Fact]: ...


class JobRepository(Protocol):
    async def upsert_coalesced(self, j: NewExtractJob, debounce_s: float) -> UpsertResult: ...

    async def enqueue(self, j: NewJob) -> JobId: ...

    async def claim(self, worker_id: UUID, limit: int, lease_s: int) -> list[Job]: ...

    async def heartbeat(self, job_id: JobId, worker_id: UUID, lease_s: int) -> bool: ...

    async def complete(self, job_id: JobId, worker_id: UUID, now: datetime) -> bool: ...

    async def retry(
        self,
        job_id: JobId,
        worker_id: UUID,
        run_after: datetime,
        err: ErrorInfo,
        *,
        count_attempt: bool,
    ) -> bool: ...

    async def kill(self, job_id: JobId, worker_id: UUID, err: ErrorInfo) -> bool: ...

    async def reap_expired(self, now: datetime) -> int: ...

    async def get(self, user_id: UserId, job_id: JobId) -> Job | None: ...

    async def list_for_user(
        self, user_id: UserId, status: JobStatus | None, limit: int
    ) -> list[Job]: ...

    async def requeue_dead(self, user_id: UserId, job_id: JobId) -> bool: ...

    async def user_has_open_extract(self, user_id: UserId) -> bool: ...


class MessageRepository(Protocol):
    async def ensure_conversation(
        self,
        user_id: UserId,
        conversation_id: UUID | None,
        now: datetime,
    ) -> UUID: ...

    async def insert_message(self, m: NewMessage) -> InsertedMessage: ...

    async def recent_turns(
        self, user_id: UserId, conversation_id: UUID, n: int
    ) -> list[Message]: ...

    async def pending_user_messages(self, user_id: UserId, limit: int) -> list[Message]: ...

    async def finalize_assistant(
        self,
        user_id: UserId,
        message_id: UUID,
        content: str,
        partial: bool,
        trace: dict[str, Any] | None,
    ) -> None: ...


class PredicateRegistry(Protocol):
    def resolve(self, name: str) -> Predicate: ...

    def lookup(self, name: str) -> Predicate: ...

    def get(self, name: str) -> Predicate: ...

    def all(self) -> Sequence[Predicate]: ...


class AuditRepository(Protocol):
    async def append(
        self,
        user_id: UserId,
        actor: str,
        action: str,
        target_type: str,
        target_id: UUID | None,
        before: dict[str, Any] | None,
        after: dict[str, Any] | None,
    ) -> None: ...


class ProfileRepository(Protocol):
    async def ensure(self, user_id: UserId) -> None: ...

    async def timezone(self, user_id: UserId) -> str: ...

    async def set_timezone(self, user_id: UserId, tz: str) -> None: ...

    async def list_user_ids(self) -> list[UserId]: ...
