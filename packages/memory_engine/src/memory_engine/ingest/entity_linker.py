"""Entity linker implementing 5-step cascade entity resolution."""

import contextlib
import string
import unicodedata
from collections.abc import Sequence
from typing import Any

from memory_engine.domain.enums import EntityType
from memory_engine.domain.ids import UserId
from memory_engine.domain.models import Entity
from memory_engine.ports.embed import Embedder
from memory_engine.ports.uow import UnitOfWork

SELF_PRONOUNS: frozenset[str] = frozenset({'i', 'me', 'my', 'myself', 'user'})

_LEADING_ARTICLES: tuple[str, ...] = ('the ', 'a ', 'an ')
_PUNCTUATION_CHARS: str = string.punctuation + ' \t\n\r"\'`«»“”‘’…'


def norm_name(s: str) -> str:
    """Normalize an entity mention string.

    1. NFKC unicode normalization and lowercase.
    2. Strip surrounding whitespace and punctuation.
    3. Strip leading articles ('the ', 'a ', 'an ').
    4. Collapse internal whitespace.
    5. Strip surrounding punctuation again.
    """
    if not s:
        return ''
    norm = unicodedata.normalize('NFKC', s).strip().lower()
    norm = norm.strip(_PUNCTUATION_CHARS).strip()

    for article in _LEADING_ARTICLES:
        if norm.startswith(article):
            norm = norm[len(article) :].strip()
            break

    norm = norm.strip(_PUNCTUATION_CHARS).strip()
    norm = ' '.join(norm.split())
    return norm.strip(_PUNCTUATION_CHARS).strip()


def is_compatible_type(
    type_a: EntityType | str,
    type_b: EntityType | str,
) -> bool:
    """Check if two entity types are compatible.

    'other' is compatible with any entity type.
    Otherwise, types must match.
    """
    a = type_a.value if isinstance(type_a, EntityType) else str(type_a).lower()
    b = type_b.value if isinstance(type_b, EntityType) else str(type_b).lower()
    if a == b:
        return True
    return a == 'other' or b == 'other'


def is_self_mention(mention: str) -> bool:
    """Return True if mention normalizes to a self pronoun."""
    return norm_name(mention) in SELF_PRONOUNS


async def _embed_documents(embedder: Embedder, texts: Sequence[str]) -> list[list[float]]:
    raw_embedder: Any = embedder
    if hasattr(raw_embedder, 'embed_documents'):
        results = await raw_embedder.embed_documents(texts)
        return [[float(x) for x in vec] for vec in results]
    if hasattr(raw_embedder, 'embed_many'):
        results = await raw_embedder.embed_many(texts)
        return [[float(x) for x in vec] for vec in results]
    if hasattr(raw_embedder, 'embed'):
        results_list: list[list[float]] = []
        for t in texts:
            vec = await raw_embedder.embed(t)
            results_list.append([float(x) for x in vec])
        return results_list
    raise AttributeError('Embedder has no embed_documents or embed method')


async def _embed_single(embedder: Embedder, text: str) -> list[float]:
    raw_embedder: Any = embedder
    if hasattr(raw_embedder, 'embed_documents'):
        results = await raw_embedder.embed_documents([text])
        return [float(x) for x in results[0]]
    if hasattr(raw_embedder, 'embed'):
        vec = await raw_embedder.embed(text)
        return [float(x) for x in vec]
    if hasattr(raw_embedder, 'embed_many'):
        results = await raw_embedder.embed_many([text])
        return [float(x) for x in results[0]]
    raise AttributeError('Embedder has no embed_documents or embed method')


async def precompute_mention_embeddings(
    embedder: Embedder,
    mentions: Sequence[str],
) -> dict[str, list[float]]:
    """Precompute embeddings for unresolved entity mentions outside the DB transaction.

    Self pronouns are skipped since they resolve in Step 1 without embeddings.
    Returns mapping from mention (both raw and normalized) to embedding vector.
    """
    to_embed: list[str] = []
    seen: set[str] = set()
    for m in mentions:
        norm = norm_name(m)
        if norm and norm not in SELF_PRONOUNS and norm not in seen:
            seen.add(norm)
            to_embed.append(m)

    if not to_embed:
        return {}

    embeddings = await _embed_documents(embedder, to_embed)
    mapping: dict[str, list[float]] = {}
    for mention, emb in zip(to_embed, embeddings, strict=True):
        emb_list = list(emb)
        mapping[mention] = emb_list
        mapping[norm_name(mention)] = emb_list
    return mapping


class EntityLinker:
    """5-step cascade entity linker per docs/LLD.md Section 8.6."""

    def __init__(
        self,
        embedder: Embedder | None = None,
        min_cosine: float = 0.92,
    ) -> None:
        self.embedder = embedder
        self.min_cosine = min_cosine

    async def precompute_embeddings(
        self,
        mentions: Sequence[str],
    ) -> dict[str, list[float]]:
        """Precompute embeddings for mentions using injected embedder."""
        if self.embedder is None:
            raise ValueError('Cannot precompute embeddings: no embedder configured')
        return await precompute_mention_embeddings(self.embedder, mentions)

    async def resolve(
        self,
        uow: UnitOfWork,
        user_id: UserId,
        mention: str,
        entity_type: EntityType | str = EntityType.THING,
        precomputed_embedding: Sequence[float] | None = None,
    ) -> Entity:
        """Resolve an entity mention to an Entity using 5-step cascade.

        1. Self pronouns ({'i', 'me', 'my', 'myself', 'user'}) -> get_or_create_self
        2. Exact normalized match -> find_by_norm_name
        3. Fuzzy/embedding match -> search_similar_by_embedding (cosine >= 0.92 & compatible type)
        4. Create new entity -> create with norm_name and embedding
        """
        normalized = norm_name(mention)
        if not normalized:
            raise ValueError(f'Invalid entity mention: {mention!r}')

        requested_type_str: str = (
            entity_type.value if isinstance(entity_type, EntityType) else str(entity_type).lower()
        )
        target_type: EntityType
        if isinstance(entity_type, EntityType):
            target_type = entity_type
        else:
            try:
                target_type = EntityType(requested_type_str)
            except ValueError:
                target_type = EntityType.THING

        # Step 1: Self pronouns
        if normalized in SELF_PRONOUNS:
            if hasattr(uow.entities, 'get_or_create_self'):
                return await uow.entities.get_or_create_self(user_id)
            return await uow.entities.ensure_self(user_id)

        # Step 2: Exact normalized match
        exact_match: Entity | None = None
        if hasattr(uow.entities, 'find_by_norm_name'):
            try:
                exact_match = await uow.entities.find_by_norm_name(
                    user_id, normalized, entity_type=target_type
                )
            except TypeError:
                exact_match = await uow.entities.find_by_norm_name(user_id, normalized)

        if exact_match is None and hasattr(uow.entities, 'find_by_alias'):
            exact_match = await uow.entities.find_by_alias(user_id, target_type, normalized)

        if exact_match is not None:
            return exact_match

        # Resolve embedding for steps 3 and 4
        emb: list[float] | None = (
            list(precomputed_embedding) if precomputed_embedding is not None else None
        )
        if emb is None and self.embedder is not None:
            emb = list(await _embed_single(self.embedder, mention))

        # Step 3: Fuzzy / embedding match
        if emb is not None:
            candidates: list[tuple[Entity, float]] = []
            if hasattr(uow.entities, 'search_similar_by_embedding'):
                try:
                    candidates = await uow.entities.search_similar_by_embedding(
                        user_id, emb, min_cosine=self.min_cosine, limit=3
                    )
                except TypeError:
                    candidates = await uow.entities.search_similar_by_embedding(
                        user_id, target_type, emb, 3
                    )
            elif hasattr(uow.entities, 'nearest'):
                candidates = await uow.entities.nearest(user_id, target_type, emb, 3)

            if candidates:
                top_entity, cosine = candidates[0]
                if cosine >= self.min_cosine and is_compatible_type(
                    top_entity.type, requested_type_str
                ):
                    if hasattr(uow.entities, 'add_alias'):
                        with contextlib.suppress(Exception):
                            await uow.entities.add_alias(
                                user_id=user_id,
                                entity_id=top_entity.id,
                                type=top_entity.type,
                                alias=mention.strip(),
                                alias_norm=normalized,
                            )
                    return top_entity

        # Step 4: Create new entity
        new_entity = await uow.entities.create(
            user_id=user_id,
            name=normalized,
            type=target_type,
            embedding=emb if emb is not None else [],
        )
        return new_entity
