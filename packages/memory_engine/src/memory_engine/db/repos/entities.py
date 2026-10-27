"""PostgreSQL Entity repository implementation."""

from collections.abc import Sequence
from typing import Any
from uuid import uuid4

from memory_engine.db.mappers import map_entity
from memory_engine.db.pool import wrap_db_errors
from memory_engine.db.sql import (
    INSERT_ENTITY_ALIAS_SQL,
    Q16_ENTITY_NEAREST,
)
from memory_engine.domain.enums import EntityType
from memory_engine.domain.ids import EntityId, UserId
from memory_engine.domain.models import Entity
from memory_engine.ports.repositories import EntityRepository


class PostgresEntityRepository(EntityRepository):
    """PostgreSQL implementation of EntityRepository port."""

    def __init__(self, conn_or_uow: Any) -> None:
        self._conn_source = conn_or_uow

    @property
    def conn(self) -> Any:
        """Resolve database connection from connection or UnitOfWork."""
        if hasattr(self._conn_source, 'conn'):
            return self._conn_source.conn
        return self._conn_source

    async def ensure_self(self, user_id: UserId) -> Entity:
        """Ensure the self entity exists for the given user, creating it if missing."""
        async with wrap_db_errors():
            row = await self.conn.fetchrow(
                'select * from entities where user_id = $1 and is_self limit 1;',
                user_id,
            )
            if row is not None:
                return map_entity(row)

            new_id = uuid4()
            insert_row = await self.conn.fetchrow(
                """\
insert into entities (id, user_id, canonical_name, type, is_self)
values ($1, $2, 'User', 'person', true)
on conflict (user_id) where is_self do update set canonical_name = entities.canonical_name
returning *;""",
                new_id,
                user_id,
            )
            if insert_row is not None:
                return map_entity(insert_row)

            # Fallback if conflict returned nothing
            row = await self.conn.fetchrow(
                'select * from entities where user_id = $1 and is_self limit 1;',
                user_id,
            )
            return map_entity(row)

    async def get_or_create_self(self, user_id: UserId) -> Entity:
        """Convenience alias for ensure_self."""
        return await self.ensure_self(user_id)

    async def find_by_alias(
        self,
        user_id: UserId,
        type: EntityType,
        alias_norm: str,
    ) -> Entity | None:
        """Find an entity by exact normalized alias or fallback canonical name."""
        type_val = type.value if hasattr(type, 'value') else str(type)
        norm = alias_norm.strip().lower()

        async with wrap_db_errors():
            row = await self.conn.fetchrow(
                """\
select e.* from entities e
join entity_aliases ea on ea.entity_id = e.id and ea.user_id = e.user_id
where ea.user_id = $1 and ea.type = $2 and ea.alias_norm = $3;""",
                user_id,
                type_val,
                norm,
            )
            if row is not None:
                return map_entity(row)

            # Fallback to direct match on lower(canonical_name)
            fallback_row = await self.conn.fetchrow(
                """\
select * from entities
where user_id = $1 and type = $2 and lower(trim(canonical_name)) = $3;""",
                user_id,
                type_val,
                norm,
            )
            if fallback_row is not None:
                return map_entity(fallback_row)
            return None

    async def find_by_norm_name(
        self,
        user_id: UserId,
        type: EntityType,
        alias_norm: str,
    ) -> Entity | None:
        """Convenience alias for find_by_alias."""
        return await self.find_by_alias(user_id, type, alias_norm)

    async def nearest(
        self,
        user_id: UserId,
        type: EntityType,
        embedding: list[float],
        k: int,
    ) -> list[tuple[Entity, float]]:
        """Find k nearest non-self entities by cosine similarity over embeddings."""
        type_val = type.value if hasattr(type, 'value') else str(type)
        async with wrap_db_errors():
            rows = await self.conn.fetch(
                Q16_ENTITY_NEAREST,
                user_id,
                type_val,
                embedding,
                k,
            )
        return [(map_entity(r), float(r['sim'])) for r in rows]

    async def search_similar_by_embedding(
        self,
        user_id: UserId,
        type: EntityType,
        embedding: list[float],
        k: int,
    ) -> list[tuple[Entity, float]]:
        """Convenience alias for nearest."""
        return await self.nearest(user_id, type, embedding, k)

    async def create(
        self,
        user_id: UserId,
        name: str,
        type: EntityType,
        embedding: list[float],
    ) -> Entity:
        """Create a new entity with optional embedding and insert primary alias."""
        type_val = type.value if hasattr(type, 'value') else str(type)
        new_id = uuid4()
        async with wrap_db_errors():
            row = await self.conn.fetchrow(
                """\
insert into entities (id, user_id, canonical_name, type, is_self, embedding)
values ($1, $2, $3, $4, false, $5)
on conflict (user_id, canonical_name, type)
do update set canonical_name = excluded.canonical_name
returning *;""",
                new_id,
                user_id,
                name,
                type_val,
                embedding,
            )
            entity = map_entity(row)
            await self.add_alias(
                user_id=user_id,
                entity_id=entity.id,
                type=type,
                alias=name,
                alias_norm=name.strip().lower(),
            )
            return entity

    async def add_alias(
        self,
        user_id: UserId,
        entity_id: EntityId,
        type: EntityType,
        alias: str,
        alias_norm: str,
    ) -> None:
        """Register an alias for an entity."""
        type_val = type.value if hasattr(type, 'value') else str(type)
        async with wrap_db_errors():
            await self.conn.execute(
                INSERT_ENTITY_ALIAS_SQL,
                user_id,
                type_val,
                alias_norm,
                alias,
                entity_id,
            )

    async def get_many(
        self,
        user_id: UserId,
        ids: Sequence[EntityId],
    ) -> dict[EntityId, Entity]:
        """Fetch multiple entities by ID and return mapping by EntityId."""
        if not ids:
            return {}

        async with wrap_db_errors():
            rows = await self.conn.fetch(
                'select * from entities where user_id = $1 and id = any($2::uuid[]);',
                user_id,
                list(ids),
            )
        entities = [map_entity(r) for r in rows]
        return {e.id: e for e in entities}

    async def alias_index(self, user_id: UserId, limit: int = 5000) -> dict[str, EntityId]:
        """Fetch alias-to-entity index for query planning hints."""
        async with wrap_db_errors():
            rows = await self.conn.fetch(
                'select alias_norm, entity_id from entity_aliases where user_id = $1 limit $2;',
                user_id,
                limit,
            )
        return {str(r['alias_norm']): EntityId(r['entity_id']) for r in rows}

    async def list_all_names(self, user_id: UserId, limit: int = 5000) -> list[str]:
        """Fetch all canonical entity names for a user."""
        async with wrap_db_errors():
            rows = await self.conn.fetch(
                'select canonical_name from entities where user_id = $1 '
                'order by canonical_name limit $2;',
                user_id,
                limit,
            )
        return [str(r['canonical_name']) for r in rows]
