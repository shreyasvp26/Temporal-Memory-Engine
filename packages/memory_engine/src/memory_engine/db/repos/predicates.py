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

EXPANDED_SYNONYMS: dict[str, tuple[str, ...]] = {
    'lives_in': ('relocated_to', 'staying_in'),
    'job_title': (
        'has_occupation',
        'works_as',
        'role_is',
        'has_role',
        'occupation',
        'profession',
        'title',
        'position',
    ),
    'studies_at': ('studying_at', 'student_at'),
    'has_name': ('named', 'name'),
    'born_in': ('birthplace',),
    'birthday': ('born_on', 'date_of_birth'),
    'relationship_status': (
        'married',
        'married_to',
        'spouse_of',
        'dating',
        'partner_of',
        'partner',
        'in_relationship_with',
        'single',
        'divorced',
    ),
    'owns': ('has_pet', 'pet_name', 'pet', 'cat_name', 'dog_name', 'owns_pet', 'possesses'),
    'likes': ('fan_of',),
    'dislikes': ('hates', 'detests'),
    'prefers': (
        'diet',
        'eats',
        'food_preference',
        'dietary_preference',
        'dietary',
        'preference',
        'preferences',
        'favorite_food',
        'favourite_food',
    ),
    'has_skill': (
        'uses',
        'uses_tool',
        'codes_in',
        'migrating_to',
        'switched_to',
        'uses_tech',
        'tech_stack',
        'programming_language',
        'skilled_in',
        'proficient_in',
    ),
    'friend_of': (
        'works_with',
        'reports_to',
        'manages',
        'colleague_of',
        'colleague',
        'coworker_of',
        'coworker',
        'teammate_of',
    ),
    'family_member': (),
    'plans_to': (
        'looking_for',
        'planning_to',
        'wants_to',
        'signed_up_for',
        'preparing_for',
        'plans',
        'intends_to',
        'aims_to',
        'hoping_to',
        'going_to',
    ),
    'is_reading': ('reading', 'reads'),
    'is_working_on': ('building', 'working_on', 'worked_on', 'developing', 'working'),
    'attended': ('graduated_from', 'graduated', 'went_to', 'attended_event', 'attending'),
    'visited': ('visited_place', 'visited_city'),
    'bought': ('purchased', 'acquired'),
    'decided': ('decided_to', 'chose'),
    'met': ('met_with',),
    'started': ('began', 'started_project', 'founded', 'created', 'initiated'),
    'completed': (
        'launched',
        'finished',
        'submitted',
        'submitted_a_paper_to',
        'submitted a paper to',
        'completed_project',
        'done_with',
        'published',
    ),
    'traveled_to': ('flew_to', 'trip_to'),
    'said_assistant': (),
    'other': ('related_to', 'general'),
}


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
                clean_s = s.strip().lower()
                self._by_synonym[clean_s] = p
                self._by_synonym[clean_s.replace(' ', '_').replace('-', '_')] = p
            for s in EXPANDED_SYNONYMS.get(norm_name, ()):
                clean_s = s.strip().lower()
                self._by_synonym[clean_s] = p
                self._by_synonym[clean_s.replace(' ', '_').replace('-', '_')] = p

        self._other = self._by_name.get(
            'other',
            Predicate(
                name='other',
                cardinality=Cardinality.MULTI,
                decay_class=DecayClass.NORMAL,
                object_kind=ObjectKind.LITERAL,
                template='{subject}: {object}',
                synonyms=('related_to',),
            ),
        )
        self._related_to = self._other

    def resolve(self, name: str) -> Predicate:
        """Resolve a predicate name: exact match, then synonym, else fallback to 'other'."""
        norm = name.strip().lower()
        if norm in self._by_name:
            return self._by_name[norm]
        norm_underscore = norm.replace(' ', '_').replace('-', '_')
        if norm_underscore in self._by_name:
            return self._by_name[norm_underscore]
        if norm in self._by_synonym:
            return self._by_synonym[norm]
        if norm_underscore in self._by_synonym:
            return self._by_synonym[norm_underscore]
        return self._other

    def lookup(self, name: str) -> Predicate:
        """Resolve predicate by exact name or synonym, else fallback to 'other'."""
        return self.resolve(name)

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
