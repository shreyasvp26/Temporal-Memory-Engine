"""Domain enumeration types."""

from enum import StrEnum


class Cardinality(StrEnum):
    """Predicate cardinality semantics."""

    SINGLE = 'single'
    MULTI = 'multi'
    EVENT = 'event'


class DecayClass(StrEnum):
    """Fact decay rates."""

    PERMANENT = 'permanent'
    SLOW = 'slow'
    NORMAL = 'normal'
    FAST = 'fast'


class ObjectKind(StrEnum):
    """Kind of object in fact triple."""

    ENTITY = 'entity'
    LITERAL = 'literal'


class EntityType(StrEnum):
    """Named entity categories."""

    PERSON = 'person'
    PLACE = 'place'
    ORG = 'org'
    THING = 'thing'
    TOPIC = 'topic'


class TimePrecision(StrEnum):
    """Temporal anchor precisions."""

    YEAR = 'year'
    MONTH = 'month'
    DAY = 'day'
    EXACT = 'exact'


class FactStatus(StrEnum):
    """Fact lifecycle states."""

    ACTIVE = 'active'
    ARCHIVED = 'archived'
    DELETED = 'deleted'
    NEEDS_REVIEW = 'needs_review'


class FactOrigin(StrEnum):
    """Source of fact creation."""

    EXTRACTED = 'extracted'
    USER_EDIT = 'user_edit'


class SourceKind(StrEnum):
    """Raw source input types."""

    CHAT = 'chat'
    NOTE = 'note'
    DOC = 'doc'


class Role(StrEnum):
    """Conversation participant roles."""

    USER = 'user'
    ASSISTANT = 'assistant'
    SYSTEM = 'system'


class JobType(StrEnum):
    """Background worker job types."""

    EXTRACT = 'extract'
    PURGE = 'purge'
    DECAY = 'decay'
    REINDEX = 'reindex'


class JobStatus(StrEnum):
    """Background job processing states."""

    QUEUED = 'queued'
    RUNNING = 'running'
    DONE = 'done'
    DEAD = 'dead'


class RelationType(StrEnum):
    """Graph edge relationship types between facts."""

    SUPERSEDES = 'supersedes'
    SUPPORTS = 'supports'
    CLARIFIES = 'clarifies'
    DERIVED_FROM = 'derived_from'


class Intent(StrEnum):
    """Query retrieval intents."""

    CURRENT_STATE = 'current_state'
    HISTORICAL = 'historical'
    POINT_IN_TIME = 'point_in_time'
    INTERVAL = 'interval'
    AGGREGATION = 'aggregation'
    DURATION_OR_ORDER = 'duration_or_order'
    OPEN = 'open'


__all__ = [
    'Cardinality',
    'DecayClass',
    'EntityType',
    'FactOrigin',
    'FactStatus',
    'Intent',
    'JobStatus',
    'JobType',
    'ObjectKind',
    'RelationType',
    'Role',
    'SourceKind',
    'TimePrecision',
]
