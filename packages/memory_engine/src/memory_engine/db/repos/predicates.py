"""Predicate registry implementations (in-memory snapshot and PostgreSQL adapter)."""

from collections.abc import Sequence
from typing import Any

from memory_engine.db.mappers import map_predicate
from memory_engine.db.pool import wrap_db_errors
from memory_engine.domain.enums import Cardinality, DecayClass, ObjectKind
from memory_engine.domain.models import Predicate
from memory_engine.ports.repositories import PredicateRegistry

# -----------------------------------------------------------------------------
# 28 Frozen Predicates from docs/LLD.md Section 5.5
# -----------------------------------------------------------------------------

SEED_PREDICATES: tuple[Predicate, ...] = (
    Predicate(
        name='lives_in',
        cardinality=Cardinality.SINGLE,
        decay_class=DecayClass.SLOW,
        object_kind=ObjectKind.ENTITY,
        template='{subject} lives in {object}',
        synonyms=('moved_to', 'resides_in'),
    ),
    Predicate(
        name='works_at',
        cardinality=Cardinality.SINGLE,
        decay_class=DecayClass.SLOW,
        object_kind=ObjectKind.ENTITY,
        template='{subject} works at {object}',
        synonyms=('employed_at', 'works_for'),
    ),
    Predicate(
        name='job_title',
        cardinality=Cardinality.SINGLE,
        decay_class=DecayClass.SLOW,
        object_kind=ObjectKind.LITERAL,
        template='{subject} works as {object}',
        synonyms=('job', 'role'),
    ),
    Predicate(
        name='studies_at',
        cardinality=Cardinality.SINGLE,
        decay_class=DecayClass.SLOW,
        object_kind=ObjectKind.ENTITY,
        template='{subject} studies at {object}',
        synonyms=(),
    ),
    Predicate(
        name='has_name',
        cardinality=Cardinality.SINGLE,
        decay_class=DecayClass.PERMANENT,
        object_kind=ObjectKind.LITERAL,
        template='{subject} is named {object}',
        synonyms=(),
    ),
    Predicate(
        name='born_in',
        cardinality=Cardinality.SINGLE,
        decay_class=DecayClass.PERMANENT,
        object_kind=ObjectKind.ENTITY,
        template='{subject} was born in {object}',
        synonyms=(),
    ),
    Predicate(
        name='birthday',
        cardinality=Cardinality.SINGLE,
        decay_class=DecayClass.PERMANENT,
        object_kind=ObjectKind.LITERAL,
        template="{subject}'s birthday is {object}",
        synonyms=(),
    ),
    Predicate(
        name='relationship_status',
        cardinality=Cardinality.SINGLE,
        decay_class=DecayClass.SLOW,
        object_kind=ObjectKind.LITERAL,
        template="{subject}'s relationship status is {object}",
        synonyms=(),
    ),
    Predicate(
        name='owns',
        cardinality=Cardinality.MULTI,
        decay_class=DecayClass.NORMAL,
        object_kind=ObjectKind.ENTITY,
        template='{subject} owns {object}',
        synonyms=(),
    ),
    Predicate(
        name='likes',
        cardinality=Cardinality.MULTI,
        decay_class=DecayClass.NORMAL,
        object_kind=ObjectKind.ENTITY,
        template='{subject} likes {object}',
        synonyms=('loves', 'enjoys'),
    ),
    Predicate(
        name='dislikes',
        cardinality=Cardinality.MULTI,
        decay_class=DecayClass.NORMAL,
        object_kind=ObjectKind.ENTITY,
        template='{subject} dislikes {object}',
        synonyms=(),
    ),
    Predicate(
        name='prefers',
        cardinality=Cardinality.MULTI,
        decay_class=DecayClass.NORMAL,
        object_kind=ObjectKind.LITERAL,
        template='{subject} prefers {object}',
        synonyms=(),
    ),
    Predicate(
        name='has_skill',
        cardinality=Cardinality.MULTI,
        decay_class=DecayClass.SLOW,
        object_kind=ObjectKind.ENTITY,
        template='{subject} has skill in {object}',
        synonyms=(),
    ),
    Predicate(
        name='friend_of',
        cardinality=Cardinality.MULTI,
        decay_class=DecayClass.SLOW,
        object_kind=ObjectKind.ENTITY,
        template='{subject} is a friend of {object}',
        synonyms=(),
    ),
    Predicate(
        name='family_member',
        cardinality=Cardinality.MULTI,
        decay_class=DecayClass.SLOW,
        object_kind=ObjectKind.ENTITY,
        template='{subject} has family member {object}',
        synonyms=(),
    ),
    Predicate(
        name='plans_to',
        cardinality=Cardinality.MULTI,
        decay_class=DecayClass.FAST,
        object_kind=ObjectKind.LITERAL,
        template='{subject} plans to {object}',
        synonyms=(),
    ),
    Predicate(
        name='is_reading',
        cardinality=Cardinality.MULTI,
        decay_class=DecayClass.FAST,
        object_kind=ObjectKind.ENTITY,
        template='{subject} is reading {object}',
        synonyms=(),
    ),
    Predicate(
        name='is_working_on',
        cardinality=Cardinality.MULTI,
        decay_class=DecayClass.FAST,
        object_kind=ObjectKind.ENTITY,
        template='{subject} is working on {object}',
        synonyms=(),
    ),
    Predicate(
        name='attended',
        cardinality=Cardinality.EVENT,
        decay_class=DecayClass.NORMAL,
        object_kind=ObjectKind.ENTITY,
        template='{subject} attended {object}',
        synonyms=(),
    ),
    Predicate(
        name='visited',
        cardinality=Cardinality.EVENT,
        decay_class=DecayClass.NORMAL,
        object_kind=ObjectKind.ENTITY,
        template='{subject} visited {object}',
        synonyms=(),
    ),
    Predicate(
        name='bought',
        cardinality=Cardinality.EVENT,
        decay_class=DecayClass.NORMAL,
        object_kind=ObjectKind.ENTITY,
        template='{subject} bought {object}',
        synonyms=(),
    ),
    Predicate(
        name='decided',
        cardinality=Cardinality.EVENT,
        decay_class=DecayClass.NORMAL,
        object_kind=ObjectKind.LITERAL,
        template='{subject} decided {object}',
        synonyms=(),
    ),
    Predicate(
        name='met',
        cardinality=Cardinality.EVENT,
        decay_class=DecayClass.NORMAL,
        object_kind=ObjectKind.ENTITY,
        template='{subject} met {object}',
        synonyms=(),
    ),
    Predicate(
        name='started',
        cardinality=Cardinality.EVENT,
        decay_class=DecayClass.NORMAL,
        object_kind=ObjectKind.ENTITY,
        template='{subject} started {object}',
        synonyms=(),
    ),
    Predicate(
        name='completed',
        cardinality=Cardinality.EVENT,
        decay_class=DecayClass.NORMAL,
        object_kind=ObjectKind.ENTITY,
        template='{subject} completed {object}',
        synonyms=(),
    ),
    Predicate(
        name='traveled_to',
        cardinality=Cardinality.EVENT,
        decay_class=DecayClass.NORMAL,
        object_kind=ObjectKind.ENTITY,
        template='{subject} traveled to {object}',
        synonyms=(),
    ),
    Predicate(
        name='said_assistant',
        cardinality=Cardinality.EVENT,
        decay_class=DecayClass.NORMAL,
        object_kind=ObjectKind.LITERAL,
        template='The assistant told {subject}: {object}',
        synonyms=(),
    ),
    Predicate(
        name='other',
        cardinality=Cardinality.MULTI,
        decay_class=DecayClass.NORMAL,
        object_kind=ObjectKind.LITERAL,
        template='{subject}: {object}',
        synonyms=(),
    ),
)


class StaticPredicateRegistry(PredicateRegistry):
    """In-memory predicate registry loaded from frozen seed data or custom predicate list."""

    def __init__(self, predicates: Sequence[Predicate] = SEED_PREDICATES) -> None:
        self._predicates = tuple(predicates)
        self._by_name: dict[str, Predicate] = {}
        self._by_synonym: dict[str, Predicate] = {}

        for p in self._predicates:
            norm_name = p.name.strip().lower()
            self._by_name[norm_name] = p
            for s in p.synonyms:
                self._by_synonym[s.strip().lower()] = p

        self._other = self._by_name.get(
            'other',
            Predicate(
                name='other',
                cardinality=Cardinality.MULTI,
                decay_class=DecayClass.NORMAL,
                object_kind=ObjectKind.LITERAL,
                template='{subject}: {object}',
                synonyms=(),
            ),
        )
        self._related_to = self._by_name.get(
            'related_to',
            Predicate(
                name='related_to',
                cardinality=Cardinality.MULTI,
                decay_class=DecayClass.NORMAL,
                object_kind=ObjectKind.LITERAL,
                template='{subject} is related to {object}',
                synonyms=(),
            ),
        )

    def resolve(self, name: str) -> Predicate:
        """Resolve a predicate name: exact match, then synonym, else fallback to 'other'."""
        norm = name.strip().lower()
        if norm in self._by_name:
            return self._by_name[norm]
        if norm in self._by_synonym:
            return self._by_synonym[norm]
        return self._other

    def lookup(self, name: str) -> Predicate:
        """Resolve predicate by exact name or synonym, else fallback to 'related_to'."""
        norm = name.strip().lower().replace(' ', '_').replace('-', '_')
        if norm in self._by_name:
            return self._by_name[norm]
        if norm in self._by_synonym:
            return self._by_synonym[norm]
        return self._related_to

    def get(self, name: str) -> Predicate:
        """Get a predicate by exact name; raises KeyError if not found."""
        norm = name.strip().lower()
        if norm in self._by_name:
            return self._by_name[norm]
        raise KeyError(f'Unknown predicate: {name}')

    def all(self) -> Sequence[Predicate]:
        """Return all registered predicates."""
        return self._predicates


class PostgresPredicateRegistry(PredicateRegistry):
    """PostgreSQL-backed predicate registry with in-memory caching and periodic refresh."""

    def __init__(self, conn_or_uow: Any = None) -> None:
        self._conn_source = conn_or_uow
        self._static = StaticPredicateRegistry()

    @property
    def conn(self) -> Any:
        """Resolve database connection from connection or UnitOfWork."""
        if self._conn_source is None:
            return None
        if hasattr(self._conn_source, 'conn'):
            return self._conn_source.conn
        return self._conn_source

    async def load(self) -> None:
        """Load all predicates from PostgreSQL predicates table and update memory cache."""
        if self.conn is None:
            return

        async with wrap_db_errors():
            rows = await self.conn.fetch('select * from predicates order by name;')
        if rows:
            loaded = [map_predicate(r) for r in rows]
            self._static = StaticPredicateRegistry(loaded)

    async def refresh(self) -> None:
        """Refresh in-memory snapshot from database."""
        await self.load()

    def resolve(self, name: str) -> Predicate:
        """Resolve predicate by exact match, synonym, or 'other' fallback."""
        return self._static.resolve(name)

    def lookup(self, name: str) -> Predicate:
        """Resolve predicate by exact name or synonym, else fallback to 'related_to'."""
        return self._static.lookup(name)

    def get(self, name: str) -> Predicate:
        """Get predicate by exact name."""
        return self._static.get(name)

    def all(self) -> Sequence[Predicate]:
        """Return all registered predicates."""
        return self._static.all()
