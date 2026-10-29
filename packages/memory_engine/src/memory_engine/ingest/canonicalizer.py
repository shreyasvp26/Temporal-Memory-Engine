"""Canonicalizer for predicate resolution, object validation, and canonical text rendering."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from memory_engine.domain.enums import Cardinality, DecayClass, EntityType, ObjectKind
from memory_engine.domain.ids import EntityId
from memory_engine.domain.models import Entity, Predicate
from memory_engine.ingest.entity_linker import SELF_PRONOUNS, norm_name
from memory_engine.ports.repositories import PredicateRegistry

FALLBACK_PREDICATE: Predicate = Predicate(
    name='other',
    cardinality=Cardinality.MULTI,
    decay_class=DecayClass.NORMAL,
    object_kind=ObjectKind.LITERAL,
    template='{subject}: {object}',
    synonyms=('related_to',),
)

SEEDED_PREDICATE_NAMES: frozenset[str] = frozenset(
    {
        'attended',
        'birthday',
        'born_in',
        'bought',
        'completed',
        'decided',
        'dislikes',
        'family_member',
        'friend_of',
        'has_name',
        'has_skill',
        'is_reading',
        'is_working_on',
        'job_title',
        'likes',
        'lives_in',
        'met',
        'other',
        'owns',
        'plans_to',
        'prefers',
        'relationship_status',
        'said_assistant',
        'started',
        'studies_at',
        'traveled_to',
        'visited',
        'works_at',
    }
)

SYNONYM_MAP: dict[str, str] = {
    # graduated_from, studied_at, went_to -> attended
    'graduated_from': 'attended',
    'graduated': 'attended',
    'went_to': 'attended',
    'attended_event': 'attended',
    'attending': 'attended',
    'alumnus_of': 'attended',
    'alumna_of': 'attended',
    # diet, eats, food_preference -> prefers
    'diet': 'prefers',
    'eats': 'prefers',
    'food_preference': 'prefers',
    'dietary_preference': 'prefers',
    'dietary': 'prefers',
    'food_allergies': 'prefers',
    'favorite_food': 'prefers',
    'favourite_food': 'prefers',
    'preference': 'prefers',
    'preferences': 'prefers',
    # looking_for, planning_to, wants_to, signed_up_for, preparing_for -> plans_to
    'looking_for': 'plans_to',
    'planning_to': 'plans_to',
    'wants_to': 'plans_to',
    'signed_up_for': 'plans_to',
    'preparing_for': 'plans_to',
    'plans': 'plans_to',
    'intends_to': 'plans_to',
    'aims_to': 'plans_to',
    'hoping_to': 'plans_to',
    'going_to': 'plans_to',
    'will_visit': 'plans_to',
    'will_move': 'plans_to',
    # uses, uses_tool, codes_in, migrating_to, switched_to -> has_skill
    'uses': 'has_skill',
    'uses_tool': 'has_skill',
    'codes_in': 'has_skill',
    'migrating_to': 'has_skill',
    'switched_to': 'has_skill',
    'uses_tech': 'has_skill',
    'tech_stack': 'has_skill',
    'programming_language': 'has_skill',
    'tool_used': 'has_skill',
    'skilled_in': 'has_skill',
    'proficient_in': 'has_skill',
    'knows_language': 'has_skill',
    # married, married_to, spouse_of, dating -> relationship_status
    'married': 'relationship_status',
    'married_to': 'relationship_status',
    'spouse_of': 'relationship_status',
    'dating': 'relationship_status',
    'partner_of': 'relationship_status',
    'partner': 'relationship_status',
    'in_relationship_with': 'relationship_status',
    'engaged_to': 'relationship_status',
    'fiance_of': 'relationship_status',
    'husband_of': 'relationship_status',
    'wife_of': 'relationship_status',
    'single': 'relationship_status',
    'divorced': 'relationship_status',
    # works_with, reports_to, manages -> friend_of
    'works_with': 'friend_of',
    'reports_to': 'friend_of',
    'manages': 'friend_of',
    'colleague_of': 'friend_of',
    'colleague': 'friend_of',
    'coworker_of': 'friend_of',
    'coworker': 'friend_of',
    'collaborates_with': 'friend_of',
    'teammate_of': 'friend_of',
    # has_occupation, job_title, works_as, role_is -> job_title
    'has_occupation': 'job_title',
    'works_as': 'job_title',
    'role_is': 'job_title',
    'has_role': 'job_title',
    'job': 'job_title',
    'role': 'job_title',
    'occupation': 'job_title',
    'profession': 'job_title',
    'position': 'job_title',
    # launched, finished, submitted, submitted a paper to, completed_project -> completed
    'launched': 'completed',
    'finished': 'completed',
    'submitted': 'completed',
    'submitted_a_paper_to': 'completed',
    'submitted a paper to': 'completed',
    'completed_project': 'completed',
    'done_with': 'completed',
    'published': 'completed',
    'graduated_project': 'completed',
    # started, building, working_on -> worked_on / is_working_on / started
    'building': 'is_working_on',
    'working_on': 'is_working_on',
    'worked_on': 'is_working_on',
    'developing': 'is_working_on',
    'working': 'is_working_on',
    'started': 'started',
    'began': 'started',
    'founded': 'started',
    'created': 'started',
    'initiated': 'started',
    # owns, bought, has_pet, pet_name -> owns / prefers / bought
    'owns': 'owns',
    'has_pet': 'owns',
    'pet_name': 'owns',
    'pet': 'owns',
    'dog_name': 'owns',
    'cat_name': 'owns',
    'owns_pet': 'owns',
    'possesses': 'owns',
    'bought': 'bought',
    'purchased': 'bought',
    'acquired': 'bought',
    # others
    'loves': 'likes',
    'enjoys': 'likes',
    'fan_of': 'likes',
    'hates': 'dislikes',
    'detests': 'dislikes',
    'moved_to': 'lives_in',
    'resides_in': 'lives_in',
    'relocated_to': 'lives_in',
    'employed_at': 'works_at',
    'works_for': 'works_at',
    'studying_at': 'studies_at',
    'student_at': 'studies_at',
    'reading': 'is_reading',
    'reads': 'is_reading',
    'flew_to': 'traveled_to',
    'decided_to': 'decided',
    'chose': 'decided',
    'met_with': 'met',
    'birthplace': 'born_in',
    'born_on': 'birthday',
    'date_of_birth': 'birthday',
    'named': 'has_name',
    'related_to': 'other',
}

_DEFAULT_PREDICATES: dict[str, Predicate] = {
    'lives_in': Predicate(
        'lives_in',
        Cardinality.SINGLE,
        DecayClass.SLOW,
        ObjectKind.ENTITY,
        '{subject} lives in {object}',
        ('moved_to', 'resides_in'),
    ),
    'works_at': Predicate(
        'works_at',
        Cardinality.SINGLE,
        DecayClass.SLOW,
        ObjectKind.ENTITY,
        '{subject} works at {object}',
        ('employed_at', 'works_for'),
    ),
    'job_title': Predicate(
        'job_title',
        Cardinality.SINGLE,
        DecayClass.SLOW,
        ObjectKind.LITERAL,
        '{subject} works as {object}',
        ('job', 'role', 'has_occupation', 'works_as', 'role_is', 'has_role'),
    ),
    'studies_at': Predicate(
        'studies_at',
        Cardinality.SINGLE,
        DecayClass.SLOW,
        ObjectKind.ENTITY,
        '{subject} studies at {object}',
        ('studying_at', 'student_at'),
    ),
    'has_name': Predicate(
        'has_name',
        Cardinality.SINGLE,
        DecayClass.PERMANENT,
        ObjectKind.LITERAL,
        '{subject} is named {object}',
        ('named', 'name'),
    ),
    'born_in': Predicate(
        'born_in',
        Cardinality.SINGLE,
        DecayClass.PERMANENT,
        ObjectKind.ENTITY,
        '{subject} was born in {object}',
        ('birthplace',),
    ),
    'birthday': Predicate(
        'birthday',
        Cardinality.SINGLE,
        DecayClass.PERMANENT,
        ObjectKind.LITERAL,
        "{subject}'s birthday is {object}",
        ('born_on', 'date_of_birth'),
    ),
    'relationship_status': Predicate(
        'relationship_status',
        Cardinality.SINGLE,
        DecayClass.SLOW,
        ObjectKind.LITERAL,
        "{subject}'s relationship status is {object}",
        ('married', 'married_to', 'spouse_of', 'dating', 'partner_of'),
    ),
    'owns': Predicate(
        'owns',
        Cardinality.MULTI,
        DecayClass.NORMAL,
        ObjectKind.ENTITY,
        '{subject} owns {object}',
        ('has_pet', 'pet_name', 'pet'),
    ),
    'likes': Predicate(
        'likes',
        Cardinality.MULTI,
        DecayClass.NORMAL,
        ObjectKind.ENTITY,
        '{subject} likes {object}',
        ('loves', 'enjoys', 'fan_of'),
    ),
    'dislikes': Predicate(
        'dislikes',
        Cardinality.MULTI,
        DecayClass.NORMAL,
        ObjectKind.ENTITY,
        '{subject} dislikes {object}',
        ('hates', 'detests'),
    ),
    'prefers': Predicate(
        'prefers',
        Cardinality.MULTI,
        DecayClass.NORMAL,
        ObjectKind.LITERAL,
        '{subject} prefers {object}',
        ('diet', 'eats', 'food_preference', 'dietary_preference', 'preference'),
    ),
    'has_skill': Predicate(
        'has_skill',
        Cardinality.MULTI,
        DecayClass.SLOW,
        ObjectKind.ENTITY,
        '{subject} has skill in {object}',
        ('uses', 'uses_tool', 'codes_in', 'migrating_to', 'switched_to', 'uses_tech'),
    ),
    'friend_of': Predicate(
        'friend_of',
        Cardinality.MULTI,
        DecayClass.SLOW,
        ObjectKind.ENTITY,
        '{subject} is a friend of {object}',
        ('works_with', 'reports_to', 'manages', 'colleague_of'),
    ),
    'family_member': Predicate(
        'family_member',
        Cardinality.MULTI,
        DecayClass.SLOW,
        ObjectKind.ENTITY,
        '{subject} has family member {object}',
        (),
    ),
    'plans_to': Predicate(
        'plans_to',
        Cardinality.MULTI,
        DecayClass.FAST,
        ObjectKind.LITERAL,
        '{subject} plans to {object}',
        ('looking_for', 'planning_to', 'wants_to', 'signed_up_for', 'preparing_for'),
    ),
    'is_reading': Predicate(
        'is_reading',
        Cardinality.MULTI,
        DecayClass.FAST,
        ObjectKind.ENTITY,
        '{subject} is reading {object}',
        ('reading',),
    ),
    'is_working_on': Predicate(
        'is_working_on',
        Cardinality.MULTI,
        DecayClass.FAST,
        ObjectKind.ENTITY,
        '{subject} is working on {object}',
        ('building', 'working_on', 'worked_on'),
    ),
    'attended': Predicate(
        'attended',
        Cardinality.EVENT,
        DecayClass.NORMAL,
        ObjectKind.ENTITY,
        '{subject} attended {object}',
        ('graduated_from', 'went_to'),
    ),
    'visited': Predicate(
        'visited',
        Cardinality.EVENT,
        DecayClass.NORMAL,
        ObjectKind.ENTITY,
        '{subject} visited {object}',
        (),
    ),
    'bought': Predicate(
        'bought',
        Cardinality.EVENT,
        DecayClass.NORMAL,
        ObjectKind.ENTITY,
        '{subject} bought {object}',
        ('purchased',),
    ),
    'decided': Predicate(
        'decided',
        Cardinality.EVENT,
        DecayClass.NORMAL,
        ObjectKind.LITERAL,
        '{subject} decided {object}',
        (),
    ),
    'met': Predicate(
        'met',
        Cardinality.EVENT,
        DecayClass.NORMAL,
        ObjectKind.ENTITY,
        '{subject} met {object}',
        ('met_with',),
    ),
    'started': Predicate(
        'started',
        Cardinality.EVENT,
        DecayClass.NORMAL,
        ObjectKind.ENTITY,
        '{subject} started {object}',
        ('began',),
    ),
    'completed': Predicate(
        'completed',
        Cardinality.EVENT,
        DecayClass.NORMAL,
        ObjectKind.ENTITY,
        '{subject} completed {object}',
        (
            'launched',
            'finished',
            'submitted',
            'submitted a paper to',
            'completed_project',
        ),
    ),
    'traveled_to': Predicate(
        'traveled_to',
        Cardinality.EVENT,
        DecayClass.NORMAL,
        ObjectKind.ENTITY,
        '{subject} traveled to {object}',
        ('flew_to',),
    ),
    'said_assistant': Predicate(
        'said_assistant',
        Cardinality.EVENT,
        DecayClass.NORMAL,
        ObjectKind.LITERAL,
        'The assistant told {subject}: {object}',
        (),
    ),
    'other': FALLBACK_PREDICATE,
}


@dataclass(frozen=True, slots=True)
class CanonicalObject:
    """Validated object component of a fact triple."""

    kind: ObjectKind
    entity_id: EntityId | None
    value: str | None
    entity_type: EntityType | None
    text: str


class DefaultPredicateRegistry(PredicateRegistry):
    """Minimal in-memory fallback predicate registry seeded with all 28 frozen predicates."""

    def __init__(self) -> None:
        self._predicates: dict[str, Predicate] = dict(_DEFAULT_PREDICATES)

    def resolve(self, name: str) -> Predicate:
        clean = name.strip().lower().replace(' ', '_').replace('-', '_')
        return self._predicates.get(clean, FALLBACK_PREDICATE)

    def lookup(self, name: str) -> Predicate:
        return self.resolve(name)

    def get(self, name: str) -> Predicate:
        clean = name.strip().lower().replace(' ', '_').replace('-', '_')
        if clean in self._predicates:
            return self._predicates[clean]
        raise KeyError(f'Unknown predicate: {name}')

    def all(self) -> Sequence[Predicate]:
        return list(self._predicates.values())


class Canonicalizer:
    """Predicate and text canonicalizer per docs/LLD.md Section 8.7."""

    def __init__(self, registry: PredicateRegistry | None = None) -> None:
        self.registry: PredicateRegistry = registry or DefaultPredicateRegistry()

    def _get_predicate(self, name: str) -> Predicate:
        if hasattr(self.registry, 'get'):
            try:
                return self.registry.get(name)
            except (KeyError, Exception):
                pass
        if hasattr(self.registry, 'resolve'):
            p = self.registry.resolve(name)
            if p.name in SEEDED_PREDICATE_NAMES:
                return p
        return _DEFAULT_PREDICATES.get(name, FALLBACK_PREDICATE)

    @staticmethod
    def _heuristic_match(norm: str) -> str | None:
        tokens = set(norm.split('_'))

        # Ownership / pets (check before actions)
        if any(t in tokens for t in ('own', 'owns', 'pet', 'pets', 'possess', 'possesses')):
            return 'owns'

        # Action completion / past events
        if any(
            t in tokens
            for t in (
                'finish',
                'finished',
                'finishes',
                'launch',
                'launched',
                'launches',
                'submit',
                'submitted',
                'submits',
                'completed',
                'complete',
                'completes',
                'publish',
                'published',
            )
        ):
            return 'completed'
        if any(t in tokens for t in ('graduated', 'graduate', 'graduating')):
            return 'attended'

        # Future intentions
        if (
            any(
                t in tokens
                for t in (
                    'plan',
                    'plans',
                    'planning',
                    'want',
                    'wants',
                    'intend',
                    'intends',
                    'intending',
                    'hope',
                    'hopes',
                    'aim',
                    'aims',
                )
            )
            or 'look_for' in norm
            or 'sign_up' in norm
            or 'prepar' in norm
            or 'going_to' in norm
        ):
            return 'plans_to'

        # Current projects / tasks
        if ('work' in tokens and ('on' in tokens or 'upon' in tokens)) or any(
            t in tokens for t in ('build', 'building', 'develop', 'developing')
        ):
            return 'is_working_on'

        # Activity start
        if any(
            t in tokens
            for t in (
                'start',
                'started',
                'starts',
                'began',
                'begin',
                'found',
                'founded',
                'initiate',
            )
        ):
            return 'started'

        # Reading
        if any(t in tokens for t in ('read', 'reading', 'reads')):
            return 'is_reading'

        # Education
        if any(
            t in tokens
            for t in (
                'attend',
                'attended',
                'attending',
                'school',
                'college',
                'university',
            )
        ):
            return 'attended'
        if any(t in tokens for t in ('study', 'studies', 'studying')):
            return 'studies_at'

        # Workplace
        if ('work' in tokens and ('at' in tokens or 'for' in tokens)) or 'employ' in norm:
            return 'works_at'

        # Job / occupation
        if any(
            t in tokens
            for t in (
                'occupation',
                'role',
                'job',
                'profession',
                'title',
                'position',
            )
        ) or ('work' in tokens and 'as' in tokens):
            return 'job_title'

        # Residence
        if any(
            t in tokens
            for t in (
                'live',
                'lives',
                'living',
                'reside',
                'resides',
                'relocate',
                'relocated',
            )
        ) or ('move' in tokens and 'to' in tokens):
            return 'lives_in'

        # Travel / trips
        if any(t in tokens for t in ('travel', 'traveled', 'travels', 'flight', 'trip', 'flew')):
            return 'traveled_to'
        if any(t in tokens for t in ('visit', 'visited', 'visits')):
            return 'visited'

        # Purchases
        if any(
            t in tokens
            for t in (
                'buy',
                'bought',
                'purchase',
                'purchased',
                'acquire',
                'acquired',
            )
        ):
            return 'bought'

        # Relationships
        if any(t in tokens for t in ('married', 'marry', 'spouse', 'dating', 'partner', 'fiance')):
            return 'relationship_status'

        # Colleagues
        if (
            any(
                t in tokens
                for t in ('colleague', 'colleagues', 'coworker', 'coworkers', 'teammate')
            )
            or ('work' in tokens and 'with' in tokens)
            or 'report_to' in norm
        ):
            return 'friend_of'
        if any(t in tokens for t in ('friend', 'friends')):
            return 'friend_of'

        # Family
        if any(
            t in tokens
            for t in (
                'family',
                'mother',
                'father',
                'sister',
                'brother',
                'parent',
                'child',
                'son',
                'daughter',
                'wife',
                'husband',
            )
        ):
            return 'family_member'

        # Tech skills / tools
        if any(
            t in tokens
            for t in (
                'skill',
                'skills',
                'tool',
                'tools',
                'code',
                'codes',
                'coding',
                'program',
                'programming',
                'tech',
                'technology',
                'stack',
                'language',
                'use',
                'uses',
            )
        ):
            return 'has_skill'

        # Dietary and general preferences
        if any(
            t in tokens
            for t in (
                'diet',
                'dietary',
                'eat',
                'eats',
                'food',
                'prefer',
                'prefers',
                'preference',
            )
        ):
            return 'prefers'

        # Likes & Dislikes
        if any(t in tokens for t in ('love', 'loves', 'like', 'likes', 'enjoy', 'enjoys')):
            return 'likes'
        if any(t in tokens for t in ('hate', 'hates', 'dislike', 'dislikes', 'detest')):
            return 'dislikes'

        # Decisions
        if any(t in tokens for t in ('decide', 'decided', 'choose', 'chose', 'chosen')):
            return 'decided'

        # Meetings
        if any(t in tokens for t in ('meet', 'met', 'meeting')):
            return 'met'

        # Identity
        if 'birth' in tokens or 'birthplace' in tokens:
            return 'born_in'
        if 'name' in tokens or 'named' in tokens:
            return 'has_name'

        return None

    def resolve_predicate(self, raw: str) -> Predicate:
        """Resolve a raw predicate string to one of the 28 seeded Predicates.

        Exact name, synonym mapping, or keyword substring matching.
        Guaranteed to return one of the 28 seeded predicates in SEEDED_PREDICATE_NAMES.
        """
        clean = raw.strip().lower()
        clean_norm = clean.replace(' ', '_').replace('-', '_')
        while '__' in clean_norm:
            clean_norm = clean_norm.replace('__', '_')

        # 1. Exact match against 28 seeded predicate names
        if clean_norm in SEEDED_PREDICATE_NAMES:
            return self._get_predicate(clean_norm)

        # 2. Known synonym mappings
        mapped = SYNONYM_MAP.get(clean_norm) or SYNONYM_MAP.get(clean)
        if mapped and mapped in SEEDED_PREDICATE_NAMES:
            return self._get_predicate(mapped)

        # 3. Registry lookup / resolve
        if hasattr(self.registry, 'lookup'):
            pred = self.registry.lookup(raw)
        else:
            pred = self.registry.resolve(clean_norm)

        if pred.name in SEEDED_PREDICATE_NAMES and pred.name != 'other':
            return pred

        # 4. Token/substring keyword heuristic matching
        heuristic = self._heuristic_match(clean_norm)
        if heuristic and heuristic in SEEDED_PREDICATE_NAMES:
            return self._get_predicate(heuristic)

        # 5. Fallback safety net: never emit unseeded predicate
        if pred.name in SEEDED_PREDICATE_NAMES:
            return pred
        return FALLBACK_PREDICATE

    canonicalize = resolve_predicate
    normalize_predicate = resolve_predicate

    def canonicalize_object(
        self,
        predicate: Predicate,
        raw_object: str,
        object_type: EntityType | Literal['literal'] | str | None = None,
        resolved_entity: Entity | None = None,
    ) -> CanonicalObject:
        """Enforce ObjectKind rules (ENTITY vs VALUE/LITERAL).

        - If predicate.object_kind == ObjectKind.LITERAL:
          The object is treated as a literal value (even if entity type or entity was supplied).
          entity_id is None, value is resolved_entity.canonical_name or raw_object.
        - If predicate.object_kind == ObjectKind.ENTITY:
          The object must be an entity. If literal was supplied or no type, promote to THING.
          entity_id is resolved_entity.id (if resolved), value is None.
        """
        raw_clean = raw_object.strip()

        if predicate.object_kind == ObjectKind.LITERAL:
            val = resolved_entity.canonical_name if resolved_entity is not None else raw_clean
            return CanonicalObject(
                kind=ObjectKind.LITERAL,
                entity_id=None,
                value=val,
                entity_type=None,
                text=val,
            )

        # ObjectKind.ENTITY
        type_str = str(object_type).lower() if object_type is not None else ''
        target_type: EntityType
        if type_str == 'literal' or not type_str:
            target_type = EntityType.THING
        elif isinstance(object_type, EntityType):
            target_type = object_type
        else:
            try:
                target_type = EntityType(type_str)
            except ValueError:
                target_type = EntityType.THING

        eid = resolved_entity.id if resolved_entity is not None else None
        txt = resolved_entity.canonical_name if resolved_entity is not None else raw_clean
        return CanonicalObject(
            kind=ObjectKind.ENTITY,
            entity_id=eid,
            value=None,
            entity_type=target_type,
            text=txt,
        )

    enforce_object_kind = canonicalize_object

    def render_canonical_text(
        self,
        predicate: Predicate | str,
        subject: Entity | str,
        object: Entity | str | CanonicalObject,
        *,
        is_self: bool = False,
    ) -> str:
        """Render canonical_text from Predicate.template with 'I' for self entity."""
        pred = self.resolve_predicate(predicate) if isinstance(predicate, str) else predicate

        # Subject text resolution
        if isinstance(subject, Entity):
            subj_is_self = subject.is_self or is_self
            subj_text = 'I' if subj_is_self else subject.canonical_name
        else:
            subj_is_self = is_self or norm_name(subject) in SELF_PRONOUNS
            subj_text = 'I' if subj_is_self else subject.strip()

        # Object text resolution
        if isinstance(object, CanonicalObject):
            obj_text = object.text
        elif isinstance(object, Entity):
            obj_text = 'I' if object.is_self else object.canonical_name
        else:
            obj_text = object.strip()

        return pred.template.format(subject=subj_text, object=obj_text)

    render = render_canonical_text
