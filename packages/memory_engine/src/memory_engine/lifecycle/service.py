"""Memory lifecycle service implementing list, inspect, edit, retract,
delete, purge, and export per LLD Section 10/12.
"""

import logging
from datetime import datetime
from typing import Any

from memory_engine.domain.clock import Clock, SystemClock
from memory_engine.domain.enums import FactStatus
from memory_engine.domain.errors import InvariantViolation, NotFound
from memory_engine.domain.ids import FactId, UserId
from memory_engine.domain.models import Fact
from memory_engine.ingest.canonicalizer import Canonicalizer
from memory_engine.ports.embed import Embedder
from memory_engine.ports.repositories import (
    EvidenceSnippet,
    FactListQuery,
    Page,
)
from memory_engine.ports.uow import UnitOfWorkFactory

__all__ = [
    'MemoryLifecycleService',
]

_logger = logging.getLogger(__name__)


class MemoryLifecycleService:
    """Manages memory fact lifecycle, edits, retractions, tombstones, and user data governance."""

    def __init__(
        self,
        uows: UnitOfWorkFactory | None = None,
        clock: Clock | None = None,
        embedder: Embedder | None = None,
        canonicalizer: Canonicalizer | None = None,
        cfg: Any = None,
        *,
        uow_factory: UnitOfWorkFactory | None = None,
    ) -> None:
        resolved_uows = uows or uow_factory
        if resolved_uows is None:
            raise InvariantViolation('uows is required for MemoryLifecycleService')
        self.uows: UnitOfWorkFactory = resolved_uows
        self.clock: Clock = clock or SystemClock()
        self.embedder: Embedder | None = embedder
        self.canonicalizer: Canonicalizer = canonicalizer or Canonicalizer()
        self.cfg: Any = cfg

    async def list_facts(
        self,
        user_id: UserId,
        query: FactListQuery,
    ) -> Page[Fact]:
        """List and filter facts for a user with pagination."""
        async with self.uows.for_user(user_id, readonly=True) as uow:
            page: Page[Fact] = await uow.facts.list(user_id, query)
            return page

    async def get_fact_with_evidence(
        self,
        user_id: UserId,
        fact_id: FactId,
    ) -> tuple[Fact, list[EvidenceSnippet], list[Fact]]:
        """Retrieve a fact with its supporting evidence snippets and supersession chain."""
        async with self.uows.for_user(user_id, readonly=True) as uow:
            fact = await uow.facts.get(user_id, fact_id)
            if fact is None:
                raise NotFound(
                    f'Fact {fact_id} not found',
                    details={'user_id': str(user_id), 'fact_id': str(fact_id)},
                )

            evidence_snippets: list[EvidenceSnippet] = []
            if hasattr(uow.facts, 'evidence_for'):
                ev_map = await uow.facts.evidence_for(user_id, [fact_id], per_fact=10)
                evidence_snippets = ev_map.get(fact_id, [])
            elif hasattr(uow.search, 'evidence_for'):
                ev_map = await uow.search.evidence_for(user_id, [fact_id], per_fact=10)
                evidence_snippets = ev_map.get(fact_id, [])

            chain: list[Fact] = []
            if hasattr(uow.facts, 'chain'):
                chain = await uow.facts.chain(user_id, fact_id)

            return fact, evidence_snippets, chain

    async def update_fact(
        self,
        user_id: UserId,
        fact_id: FactId,
        object_value: str,
        valid_from: datetime | None = None,
        valid_to: datetime | None = None,
    ) -> Fact:
        """Update a fact's value or bounds, re-rendering canonical text and re-embedding."""
        # 1. Fetch current fact (read transaction)
        async with self.uows.for_user(user_id, readonly=True) as uow:
            existing = await uow.facts.get(user_id, fact_id)
            if existing is None:
                raise NotFound(
                    f'Fact {fact_id} not found',
                    details={'user_id': str(user_id), 'fact_id': str(fact_id)},
                )
            entities = await uow.entities.get_many(user_id, [existing.subject_id])
            subject_entity = entities.get(existing.subject_id)

        # 2. Re-render canonical text
        subject_name = subject_entity.canonical_name if subject_entity else 'User'
        is_self = subject_entity.is_self if subject_entity else False
        canonical_text = self.canonicalizer.render_canonical_text(
            existing.predicate,
            subject_entity or subject_name,
            object_value,
            is_self=is_self,
        )

        # 3. Re-embed outside the write transaction (IO discipline)
        new_embedding: list[float] | None = None
        if self.embedder is not None:
            try:
                if hasattr(self.embedder, 'embed_query'):
                    emb = await self.embedder.embed_query(canonical_text)
                elif hasattr(self.embedder, 'embed_queries'):
                    embs = await self.embedder.embed_queries([canonical_text])
                    emb = embs[0]
                elif hasattr(self.embedder, 'embed'):
                    emb = await self.embedder.embed(canonical_text)
                else:
                    emb = None
                new_embedding = list(emb) if emb is not None else None
            except Exception as exc:
                _logger.warning('Failed to generate embedding for updated fact: %s', exc)
                new_embedding = None

        eff_valid_from = valid_from if valid_from is not None else existing.valid_from
        eff_valid_to = valid_to if valid_to is not None else existing.valid_to

        if eff_valid_to is not None and eff_valid_to <= eff_valid_from:
            raise InvariantViolation(
                f'valid_to ({eff_valid_to}) must be greater than valid_from ({eff_valid_from})'
            )

        # 4. Write transaction under user lock
        async with self.uows.for_user(user_id) as uow:
            if hasattr(uow, 'lock_user'):
                await uow.lock_user(user_id)

            curr = await uow.facts.get(user_id, fact_id)
            if curr is None:
                raise NotFound(
                    f'Fact {fact_id} not found',
                    details={'user_id': str(user_id), 'fact_id': str(fact_id)},
                )

            before_snapshot: dict[str, Any] = {
                'fact_text': curr.fact_text,
                'object_value': curr.object_value,
                'valid_from': curr.valid_from.isoformat() if curr.valid_from else None,
                'valid_to': curr.valid_to.isoformat() if curr.valid_to else None,
                'status': str(curr.status),
            }

            await uow.facts.manual_update(
                user_id,
                fact_id,
                fact_text=canonical_text,
                object_value=object_value,
                object_norm=object_value.strip().lower(),
                valid_from=eff_valid_from,
                valid_to=eff_valid_to,
            )

            after_snapshot: dict[str, Any] = {
                'fact_text': canonical_text,
                'object_value': object_value,
                'valid_from': eff_valid_from.isoformat(),
                'valid_to': eff_valid_to.isoformat() if eff_valid_to else None,
                'status': str(curr.status),
                'embedded': new_embedding is not None,
            }

            await uow.audit.append(
                user_id=user_id,
                actor='user',
                action='fact.manual_update',
                target_type='fact',
                target_id=fact_id,
                before=before_snapshot,
                after=after_snapshot,
            )

            updated = await uow.facts.get(user_id, fact_id)
            if updated is not None:
                return updated
            return curr

    async def retract_fact(
        self,
        user_id: UserId,
        fact_id: FactId,
        reason: str,
    ) -> None:
        """Retract a fact by closing its timeline bounds and setting status to retracted."""
        now = self.clock.now()

        async with self.uows.for_user(user_id) as uow:
            if hasattr(uow, 'lock_user'):
                await uow.lock_user(user_id)

            fact = await uow.facts.get(user_id, fact_id)
            if fact is None:
                raise NotFound(
                    f'Fact {fact_id} not found',
                    details={'user_id': str(user_id), 'fact_id': str(fact_id)},
                )

            eff_valid_to = fact.valid_to or now

            before_snapshot: dict[str, Any] = {
                'status': str(fact.status),
                'valid_to': fact.valid_to.isoformat() if fact.valid_to else None,
                'invalidated_at': (
                    fact.invalidated_at.isoformat() if fact.invalidated_at else None
                ),
            }

            await uow.facts.manual_update(
                user_id,
                fact_id,
                valid_to=eff_valid_to,
                invalidated_at=now,
                status='retracted',
            )

            after_snapshot: dict[str, Any] = {
                'status': 'retracted',
                'valid_to': eff_valid_to.isoformat(),
                'invalidated_at': now.isoformat(),
                'reason': reason,
            }

            await uow.audit.append(
                user_id=user_id,
                actor='user',
                action='fact.retract',
                target_type='fact',
                target_id=fact_id,
                before=before_snapshot,
                after=after_snapshot,
            )

    async def delete_fact(
        self,
        user_id: UserId,
        fact_id: FactId,
    ) -> None:
        """Soft-delete a fact with tombstone timestamp and audit entry."""
        now = self.clock.now()

        async with self.uows.for_user(user_id) as uow:
            if hasattr(uow, 'lock_user'):
                await uow.lock_user(user_id)

            fact = await uow.facts.get(user_id, fact_id)
            if fact is None:
                raise NotFound(
                    f'Fact {fact_id} not found',
                    details={'user_id': str(user_id), 'fact_id': str(fact_id)},
                )

            before_snapshot: dict[str, Any] = {
                'status': str(fact.status),
                'invalidated_at': (
                    fact.invalidated_at.isoformat() if fact.invalidated_at else None
                ),
            }

            await uow.facts.manual_update(
                user_id,
                fact_id,
                status=FactStatus.DELETED,
                invalidated_at=now,
            )

            after_snapshot: dict[str, Any] = {
                'status': FactStatus.DELETED.value,
                'invalidated_at': now.isoformat(),
            }

            await uow.audit.append(
                user_id=user_id,
                actor='user',
                action='fact.delete',
                target_type='fact',
                target_id=fact_id,
                before=before_snapshot,
                after=after_snapshot,
            )

    async def purge_user(self, user_id: UserId) -> int:
        """Perform a GDPR hard-purge of all user data via Q-16."""
        async with self.uows.for_user(user_id) as uow:
            if hasattr(uow, 'lock_user'):
                await uow.lock_user(user_id)

            count = await uow.profiles.purge_user(user_id)

            await uow.audit.append(
                user_id=user_id,
                actor='user',
                action='user.purge',
                target_type='user',
                target_id=user_id,
                before=None,
                after={'purged_rows': count},
            )
            return count

    async def export_user(self, user_id: UserId) -> dict[str, Any]:
        """Export full user memory and profile state for GDPR portability."""
        async with self.uows.for_user(user_id, readonly=True) as uow:
            return await uow.profiles.export_user(user_id)
