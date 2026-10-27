"""Unit tests for EntityLinker and Canonicalizer per LLD Sections 8.6 and 8.7."""

from collections.abc import Sequence
from typing import Any
from uuid import uuid4

import pytest
from memory_engine.db.repos.predicates import StaticPredicateRegistry
from memory_engine.domain.enums import Cardinality, EntityType, ObjectKind
from memory_engine.domain.ids import EntityId, UserId
from memory_engine.domain.models import Entity
from memory_engine.ingest.canonicalizer import Canonicalizer
from memory_engine.ingest.entity_linker import (
    EntityLinker,
    is_compatible_type,
    is_self_mention,
    norm_name,
    precompute_mention_embeddings,
)

# -----------------------------------------------------------------------------
# Fakes for isolated unit testing (zero IO, zero network sockets)
# -----------------------------------------------------------------------------


class FakeEntityRepository:
    """In-memory EntityRepository for unit testing EntityLinker."""

    def __init__(self) -> None:
        self.self_entity: Entity | None = None
        self.entities: dict[EntityId, Entity] = {}
        self.aliases: list[dict[str, Any]] = []
        self.similar_queue: list[list[tuple[Entity, float]]] = []
        self.created_calls: list[dict[str, Any]] = []

    async def ensure_self(self, user_id: UserId) -> Entity:
        if self.self_entity is None:
            self.self_entity = Entity(
                id=EntityId(uuid4()),
                user_id=user_id,
                canonical_name='User',
                type=EntityType.PERSON,
                is_self=True,
            )
            self.entities[self.self_entity.id] = self.self_entity
        return self.self_entity

    async def get_or_create_self(self, user_id: UserId) -> Entity:
        return await self.ensure_self(user_id)

    async def find_by_alias(
        self, user_id: UserId, type: EntityType, alias_norm: str
    ) -> Entity | None:
        for a in self.aliases:
            if a['user_id'] == user_id and a['alias_norm'] == alias_norm and a['type'] == type:
                return self.entities.get(a['entity_id'])
        for e in self.entities.values():
            if (
                e.user_id == user_id
                and e.type == type
                and norm_name(e.canonical_name) == alias_norm
            ):
                return e
        return None

    async def find_by_norm_name(
        self,
        user_id: UserId,
        name_or_type: str | EntityType,
        alias_or_type: str | EntityType | None = None,
        entity_type: EntityType | None = None,
    ) -> Entity | None:
        if isinstance(name_or_type, EntityType):
            return await self.find_by_alias(user_id, name_or_type, str(alias_or_type))

        norm = norm_name(name_or_type)
        pref_type = alias_or_type if isinstance(alias_or_type, EntityType) else entity_type

        # Find matching candidates
        matching: list[Entity] = []
        for a in self.aliases:
            if a['user_id'] == user_id and a['alias_norm'] == norm:
                ent = self.entities.get(a['entity_id'])
                if ent and ent not in matching:
                    matching.append(ent)
        for e in self.entities.values():
            if e.user_id == user_id and norm_name(e.canonical_name) == norm and e not in matching:
                matching.append(e)

        if not matching:
            return None

        # Prefer matching entity_type if specified
        if pref_type is not None:
            for ent in matching:
                if ent.type == pref_type:
                    return ent

        return matching[0]

    async def nearest(
        self,
        user_id: UserId,
        type: EntityType,
        embedding: list[float],
        k: int,
    ) -> list[tuple[Entity, float]]:
        if self.similar_queue:
            return self.similar_queue.pop(0)
        return []

    async def search_similar_by_embedding(
        self,
        user_id: UserId,
        type_or_emb: EntityType | Sequence[float],
        embedding: Sequence[float] | None = None,
        k: int | None = None,
        *,
        min_cosine: float = 0.92,
        limit: int = 3,
    ) -> list[tuple[Entity, float]]:
        if self.similar_queue:
            return self.similar_queue.pop(0)
        return []

    async def create(
        self,
        user_id: UserId,
        name: str,
        type: EntityType,
        embedding: list[float],
    ) -> Entity:
        ent = Entity(
            id=EntityId(uuid4()),
            user_id=user_id,
            canonical_name=name,
            type=type,
            is_self=False,
        )
        self.entities[ent.id] = ent
        self.created_calls.append(
            {
                'user_id': user_id,
                'name': name,
                'type': type,
                'embedding': embedding,
            }
        )
        await self.add_alias(user_id, ent.id, type, name, norm_name(name))
        return ent

    async def add_alias(
        self,
        user_id: UserId,
        entity_id: EntityId,
        type: EntityType,
        alias: str,
        alias_norm: str,
    ) -> None:
        self.aliases.append(
            {
                'user_id': user_id,
                'entity_id': entity_id,
                'type': type,
                'alias': alias,
                'alias_norm': alias_norm,
            }
        )

    async def get_many(self, user_id: UserId, ids: Sequence[EntityId]) -> dict[EntityId, Entity]:
        return {eid: self.entities[eid] for eid in ids if eid in self.entities}

    async def alias_index(self, user_id: UserId, limit: int = 5000) -> dict[str, EntityId]:
        return {a['alias_norm']: a['entity_id'] for a in self.aliases if a['user_id'] == user_id}


class FakeUnitOfWork:
    def __init__(self, entities: FakeEntityRepository | None = None) -> None:
        self.entities = entities or FakeEntityRepository()
        self.locked = False

    async def lock_user(self) -> None:
        self.locked = True

    async def __aenter__(self) -> 'FakeUnitOfWork':
        return self

    async def __aexit__(self, et: Any, ev: Any, tb: Any) -> None:
        pass


class FakeEmbedder:
    def __init__(self, dim: int = 4) -> None:
        self.dim = dim
        self.model_id = 'fake-nomic-embed'
        self.embedded_texts: list[str] = []

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        self.embedded_texts.extend(texts)
        return [[0.1 * (i + 1)] * self.dim for i, _ in enumerate(texts)]

    async def embed_queries(self, texts: Sequence[str]) -> list[list[float]]:
        return await self.embed_documents(texts)


# -----------------------------------------------------------------------------
# Unit Tests: norm_name and is_compatible_type
# -----------------------------------------------------------------------------


class TestNormName:
    def test_lowercase_and_whitespace(self) -> None:
        assert norm_name('  Alice   Wonderland  ') == 'alice wonderland'
        assert norm_name('\t\nNEW   YORK\t') == 'new york'

    def test_strip_leading_articles(self) -> None:
        assert norm_name('The New York Times') == 'new york times'
        assert norm_name('a beautiful mind') == 'beautiful mind'
        assert norm_name('an exciting adventure') == 'exciting adventure'
        assert norm_name('the   dog') == 'dog'

    def test_strip_surrounding_punctuation(self) -> None:
        assert norm_name('"Paris"') == 'paris'
        assert norm_name("'Tokyo'") == 'tokyo'
        assert norm_name('...London!...') == 'london'
        assert norm_name('- The Beatles -') == 'beatles'
        assert norm_name('«Rome»') == 'rome'

    def test_empty_and_punctuation_only(self) -> None:
        assert norm_name('') == ''
        assert norm_name('   ') == ''
        assert norm_name('???!!!') == ''

    def test_self_pronouns(self) -> None:
        for p in ('i', 'me', 'my', 'myself', 'user', 'I', 'ME', 'The User', '  the user  '):
            assert is_self_mention(p) is True

        assert is_self_mention('Bob') is False
        assert is_self_mention('Alice') is False


class TestCompatibility:
    def test_same_type_is_compatible(self) -> None:
        assert is_compatible_type(EntityType.PERSON, EntityType.PERSON) is True
        assert is_compatible_type(EntityType.PLACE, EntityType.PLACE) is True
        assert is_compatible_type(EntityType.ORG, EntityType.ORG) is True

    def test_other_is_compatible_with_any(self) -> None:
        assert is_compatible_type('other', EntityType.PERSON) is True
        assert is_compatible_type(EntityType.PLACE, 'other') is True
        assert is_compatible_type('other', 'other') is True

    def test_different_concrete_types_incompatible(self) -> None:
        assert is_compatible_type(EntityType.PERSON, EntityType.PLACE) is False
        assert is_compatible_type(EntityType.PERSON, EntityType.ORG) is False
        assert is_compatible_type(EntityType.PLACE, EntityType.ORG) is False


# -----------------------------------------------------------------------------
# Unit Tests: EntityLinker (T-ENT)
# -----------------------------------------------------------------------------


class TestEntityLinker:
    @pytest.fixture
    def user_id(self) -> UserId:
        return UserId(uuid4())

    @pytest.fixture
    def repo(self) -> FakeEntityRepository:
        return FakeEntityRepository()

    @pytest.fixture
    def uow(self, repo: FakeEntityRepository) -> Any:
        return FakeUnitOfWork(repo)

    @pytest.mark.asyncio
    async def test_self_pronoun_collapse(
        self, uow: Any, repo: FakeEntityRepository, user_id: UserId
    ) -> None:
        linker = EntityLinker()

        for mention in ('I', 'me', 'my', 'myself', 'user', 'the user', 'The User!'):
            resolved = await linker.resolve(uow, user_id, mention)
            assert resolved.is_self is True
            assert resolved.canonical_name == 'User'

        # Only 1 self entity should exist
        assert len(repo.entities) == 1
        assert len(repo.created_calls) == 0

    @pytest.mark.asyncio
    async def test_exact_norm_name_match(
        self, uow: Any, repo: FakeEntityRepository, user_id: UserId
    ) -> None:
        # Pre-populate an entity
        existing = await repo.create(
            user_id=user_id,
            name='new york times',
            type=EntityType.ORG,
            embedding=[0.1, 0.2, 0.3, 0.4],
        )

        linker = EntityLinker()
        resolved = await linker.resolve(
            uow, user_id, 'The New York Times', entity_type=EntityType.ORG
        )
        assert resolved.id == existing.id
        assert resolved.canonical_name == 'new york times'
        # No new entity created
        assert len(repo.created_calls) == 1  # Only the pre-populated one

    @pytest.mark.asyncio
    async def test_exact_match_prefers_matching_entity_type(
        self, uow: Any, repo: FakeEntityRepository, user_id: UserId
    ) -> None:
        # Pre-populate two entities with same norm_name but different types
        await repo.create(
            user_id=user_id,
            name='apple',
            type=EntityType.THING,
            embedding=[0.1, 0.1, 0.1, 0.1],
        )
        org_ent = await repo.create(
            user_id=user_id,
            name='apple',
            type=EntityType.ORG,
            embedding=[0.2, 0.2, 0.2, 0.2],
        )

        linker = EntityLinker()
        # When ORG requested, prefers org_ent
        resolved = await linker.resolve(uow, user_id, 'Apple', entity_type=EntityType.ORG)
        assert resolved.id == org_ent.id

    @pytest.mark.asyncio
    async def test_cosine_high_similarity_links_existing_entity(
        self, uow: Any, repo: FakeEntityRepository, user_id: UserId
    ) -> None:
        target = Entity(
            id=EntityId(uuid4()),
            user_id=user_id,
            canonical_name='Paris',
            type=EntityType.PLACE,
            is_self=False,
        )
        repo.entities[target.id] = target

        # Queue candidate with cosine 0.95 (>= 0.92)
        repo.similar_queue.append([(target, 0.95)])

        linker = EntityLinker(min_cosine=0.92)
        resolved = await linker.resolve(
            uow,
            user_id,
            'City of Lights',
            entity_type=EntityType.PLACE,
            precomputed_embedding=[0.51, 0.49, 0.50, 0.52],
        )

        assert resolved.id == target.id
        # Verify alias was added
        assert any(
            a['alias'] == 'City of Lights' and a['entity_id'] == target.id for a in repo.aliases
        )
        # Create was not called
        assert len(repo.created_calls) == 0

    @pytest.mark.asyncio
    async def test_cosine_low_similarity_creates_new_entity(
        self, uow: Any, repo: FakeEntityRepository, user_id: UserId
    ) -> None:
        candidate = Entity(
            id=EntityId(uuid4()),
            user_id=user_id,
            canonical_name='London',
            type=EntityType.PLACE,
            is_self=False,
        )
        repo.entities[candidate.id] = candidate

        # Queue candidate with cosine 0.88 (< 0.92)
        repo.similar_queue.append([(candidate, 0.88)])

        linker = EntityLinker(min_cosine=0.92)
        resolved = await linker.resolve(
            uow,
            user_id,
            'Paris',
            entity_type=EntityType.PLACE,
            precomputed_embedding=[0.5, 0.5, 0.5, 0.5],
        )

        # Candidate rejected, new entity created
        assert resolved.id != candidate.id
        assert resolved.canonical_name == 'paris'
        assert resolved.type == EntityType.PLACE
        assert len(repo.created_calls) == 1

    @pytest.mark.asyncio
    async def test_incompatible_entity_type_rejection(
        self, uow: Any, repo: FakeEntityRepository, user_id: UserId
    ) -> None:
        # Candidate has cosine 0.96 (high), but is a PLACE
        candidate = Entity(
            id=EntityId(uuid4()),
            user_id=user_id,
            canonical_name='Washington',
            type=EntityType.PLACE,
            is_self=False,
        )
        repo.entities[candidate.id] = candidate
        repo.similar_queue.append([(candidate, 0.96)])

        linker = EntityLinker(min_cosine=0.92)
        # Mention is a PERSON ('George Washington')
        resolved = await linker.resolve(
            uow,
            user_id,
            'George Washington',
            entity_type=EntityType.PERSON,
            precomputed_embedding=[0.71, 0.69, 0.70, 0.72],
        )

        # Must reject candidate because PERSON != PLACE, creating a new PERSON entity
        assert resolved.id != candidate.id
        assert resolved.canonical_name == 'george washington'
        assert resolved.type == EntityType.PERSON
        assert len(repo.created_calls) == 1

    @pytest.mark.asyncio
    async def test_compatible_entity_type_other_accepted(
        self, uow: Any, repo: FakeEntityRepository, user_id: UserId
    ) -> None:
        # Candidate has type ORG, mention requested type 'other' -> should accept
        candidate = Entity(
            id=EntityId(uuid4()),
            user_id=user_id,
            canonical_name='Acme Global',
            type=EntityType.ORG,
            is_self=False,
        )
        repo.entities[candidate.id] = candidate
        repo.similar_queue.append([(candidate, 0.94)])

        linker = EntityLinker(min_cosine=0.92)
        resolved = await linker.resolve(
            uow,
            user_id,
            'Acme Corporation',
            entity_type='other',
            precomputed_embedding=[0.31, 0.29, 0.30, 0.32],
        )

        assert resolved.id == candidate.id
        assert len(repo.created_calls) == 0

    @pytest.mark.asyncio
    async def test_empty_mention_raises_error(self, uow: Any, user_id: UserId) -> None:
        linker = EntityLinker()
        with pytest.raises(ValueError, match='Invalid entity mention'):
            await linker.resolve(uow, user_id, '   ')

    @pytest.mark.asyncio
    async def test_precompute_embeddings_skips_self_pronouns(self) -> None:
        embedder = FakeEmbedder(dim=4)
        mentions = ['I', 'me', 'The User', 'Paris', 'New York', 'Paris']

        result = await precompute_mention_embeddings(embedder, mentions)

        # Self pronouns skipped
        assert 'i' not in embedder.embedded_texts
        assert 'me' not in embedder.embedded_texts
        assert 'The User' not in embedder.embedded_texts

        # Unique mentions embedded
        assert 'Paris' in embedder.embedded_texts
        assert 'New York' in embedder.embedded_texts
        assert len(embedder.embedded_texts) == 2

        # Mapping contains both raw and normalized keys
        assert 'Paris' in result
        assert 'paris' in result
        assert 'New York' in result
        assert 'new york' in result


# -----------------------------------------------------------------------------
# Unit Tests: Canonicalizer
# -----------------------------------------------------------------------------


class TestCanonicalizer:
    @pytest.fixture
    def canonicalizer(self) -> Canonicalizer:
        return Canonicalizer(StaticPredicateRegistry())

    def test_synonym_resolution(self, canonicalizer: Canonicalizer) -> None:
        # lives_in synonyms
        assert canonicalizer.resolve_predicate('resides_in').name == 'lives_in'
        assert canonicalizer.resolve_predicate('moved_to').name == 'lives_in'
        assert canonicalizer.resolve_predicate('Resides-In').name == 'lives_in'

        # works_at synonyms
        assert canonicalizer.resolve_predicate('employed_at').name == 'works_at'
        assert canonicalizer.resolve_predicate('works_for').name == 'works_at'
        assert canonicalizer.resolve_predicate('Works For').name == 'works_at'

        # likes synonyms
        assert canonicalizer.resolve_predicate('loves').name == 'likes'
        assert canonicalizer.resolve_predicate('enjoys').name == 'likes'

        # job_title synonyms
        assert canonicalizer.resolve_predicate('job').name == 'job_title'
        assert canonicalizer.resolve_predicate('role').name == 'job_title'

    def test_fallback_to_related_to(self, canonicalizer: Canonicalizer) -> None:
        resolved = canonicalizer.resolve_predicate('some_random_unknown_pred')
        assert resolved.name == 'related_to'
        assert resolved.template == '{subject} is related to {object}'
        assert resolved.cardinality == Cardinality.MULTI

    def test_object_kind_rules_literal(self, canonicalizer: Canonicalizer) -> None:
        # job_title has ObjectKind.LITERAL
        job_pred = canonicalizer.resolve_predicate('job_title')
        assert job_pred.object_kind == ObjectKind.LITERAL

        # Even if entity type OR Entity object provided, enforces LITERAL value
        ent = Entity(
            id=EntityId(uuid4()),
            user_id=UserId(uuid4()),
            canonical_name='Senior Architect',
            type=EntityType.PERSON,
            is_self=False,
        )

        canon_obj = canonicalizer.canonicalize_object(
            predicate=job_pred,
            raw_object='Software Engineer',
            object_type=EntityType.PERSON,
            resolved_entity=ent,
        )

        assert canon_obj.kind == ObjectKind.LITERAL
        assert canon_obj.entity_id is None
        assert canon_obj.value == 'Senior Architect'
        assert canon_obj.text == 'Senior Architect'

    def test_object_kind_rules_entity(self, canonicalizer: Canonicalizer) -> None:
        # lives_in has ObjectKind.ENTITY
        lives_pred = canonicalizer.resolve_predicate('lives_in')
        assert lives_pred.object_kind == ObjectKind.ENTITY

        # When object_type is 'literal' or not specified, promotes to EntityType.THING
        canon_obj_promoted = canonicalizer.canonicalize_object(
            predicate=lives_pred,
            raw_object='Paris',
            object_type='literal',
        )
        assert canon_obj_promoted.kind == ObjectKind.ENTITY
        assert canon_obj_promoted.entity_type == EntityType.THING
        assert canon_obj_promoted.text == 'Paris'

        # When resolved entity provided, preserves entity_id
        eid = EntityId(uuid4())
        ent = Entity(
            id=eid,
            user_id=UserId(uuid4()),
            canonical_name='Paris',
            type=EntityType.PLACE,
            is_self=False,
        )
        canon_obj_ent = canonicalizer.canonicalize_object(
            predicate=lives_pred,
            raw_object='Paris',
            object_type=EntityType.PLACE,
            resolved_entity=ent,
        )
        assert canon_obj_ent.kind == ObjectKind.ENTITY
        assert canon_obj_ent.entity_id == eid
        assert canon_obj_ent.entity_type == EntityType.PLACE
        assert canon_obj_ent.text == 'Paris'

    def test_template_rendering_self_entity(self, canonicalizer: Canonicalizer) -> None:
        self_entity = Entity(
            id=EntityId(uuid4()),
            user_id=UserId(uuid4()),
            canonical_name='User',
            type=EntityType.PERSON,
            is_self=True,
        )

        # Self entity rendered as 'I'
        text1 = canonicalizer.render_canonical_text(
            predicate='lives_in',
            subject=self_entity,
            object='Paris',
        )
        assert text1 == 'I lives in Paris'

        # Self entity with likes predicate
        text2 = canonicalizer.render_canonical_text(
            predicate='likes',
            subject=self_entity,
            object='Jazz',
        )
        assert text2 == 'I likes Jazz'

        # With fallback predicate
        text3 = canonicalizer.render_canonical_text(
            predicate='custom_pred',
            subject='I',
            object='Quantum Computing',
        )
        assert text3 == 'I is related to Quantum Computing'

    def test_template_rendering_non_self_entity(self, canonicalizer: Canonicalizer) -> None:
        other_entity = Entity(
            id=EntityId(uuid4()),
            user_id=UserId(uuid4()),
            canonical_name='Alice',
            type=EntityType.PERSON,
            is_self=False,
        )

        text = canonicalizer.render_canonical_text(
            predicate='works_at',
            subject=other_entity,
            object='Google',
        )
        assert text == 'Alice works at Google'

        text_synonym = canonicalizer.render_canonical_text(
            predicate='works_for',
            subject='Bob',
            object='Acme Corp',
        )
        assert text_synonym == 'Bob works at Acme Corp'
