"""Canonicalizer for predicate resolution, object validation, and canonical text rendering."""

from dataclasses import dataclass
from typing import Literal

from memory_engine.domain.enums import Cardinality, DecayClass, EntityType, ObjectKind
from memory_engine.domain.ids import EntityId
from memory_engine.domain.models import Entity, Predicate
from memory_engine.ingest.entity_linker import SELF_PRONOUNS, norm_name
from memory_engine.ports.repositories import PredicateRegistry

FALLBACK_PREDICATE: Predicate = Predicate(
    name='related_to',
    cardinality=Cardinality.MULTI,
    decay_class=DecayClass.NORMAL,
    object_kind=ObjectKind.LITERAL,
    template='{subject} is related to {object}',
    synonyms=(),
)


@dataclass(frozen=True, slots=True)
class CanonicalObject:
    """Validated object component of a fact triple."""

    kind: ObjectKind
    entity_id: EntityId | None
    value: str | None
    entity_type: EntityType | None
    text: str


class Canonicalizer:
    """Predicate and text canonicalizer per docs/LLD.md Section 8.7."""

    def __init__(self, registry: PredicateRegistry) -> None:
        self.registry = registry

    def resolve_predicate(self, raw: str) -> Predicate:
        """Resolve a raw predicate string via PredicateRegistry.lookup.

        Exact name or synonym matches return the registered Predicate.
        Unknown predicates fallback to 'related_to'.
        """
        clean = raw.strip().lower().replace(' ', '_').replace('-', '_')
        if hasattr(self.registry, 'lookup'):
            pred = self.registry.lookup(raw)
        else:
            pred = self.registry.resolve(clean)

        if pred.name == 'other' and clean != 'other':
            return FALLBACK_PREDICATE
        return pred

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
            obj_text = str(object).strip()

        return pred.template.format(subject=subj_text, object=obj_text)

    render = render_canonical_text
