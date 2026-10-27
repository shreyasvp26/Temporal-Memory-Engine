"""End-to-end integration test for bitemporal ingestion, out-of-order backfill, and retrieval.

Validates LLD Section 20.2 & 20.4:
1. Ingest initial fact: "I live in Delhi" in Jan 2025 (valid [2025-01-01, None))
2. Supersede: "I moved to Pune" in Oct 2026 (Delhi becomes [2025-01-01, 2026-10-01),
   Pune becomes [2026-10-01, None))
3. Backfill out-of-order: "In 2024 I lived in Bangalore"
   (Bangalore becomes [2024-01-01, 2025-01-01))
4. Verify CURRENT, AS_OF, and OVERLAP queries via RetrievalOrchestrator against
   real Postgres + pgvector.
"""

import math
from collections.abc import Callable, Coroutine, Sequence
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

import pytest
from memory_engine.db.uow import PostgresUnitOfWorkFactory
from memory_engine.domain.clock import FixedClock
from memory_engine.domain.enums import (
    Cardinality,
    EntityType,
    Intent,
    Role,
    SourceKind,
    TimePrecision,
)
from memory_engine.domain.ids import UserId
from memory_engine.domain.intervals import TimeInterval
from memory_engine.domain.models import Candidate
from memory_engine.ingest.conflict import ConflictResolver, ResolutionContext, default_registry
from memory_engine.ingest.plan_executor import PlanExecutor
from memory_engine.ports.repositories import NewSource
from memory_engine.retrieve.orchestrator import RetrievalOrchestrator
from memory_engine.retrieve.planner.rules import QueryPlan

pytestmark = pytest.mark.integration


class HashEmbedder:
    """Deterministic 768-d token hash embedder per LLD Section 20.2."""

    model_id: str = 'nomic-ai/nomic-embed-text-v1.5'
    dim: int = 768

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for word in text.lower().split():
            h = abs(hash(word)) % self.dim
            vec[h] += 1.0
        sq_sum = sum(x * x for x in vec)
        if sq_sum == 0.0:
            vec[0] = 1.0
            sq_sum = 1.0
        norm = math.sqrt(sq_sum)
        return [x / norm for x in vec]

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    async def embed_queries(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    async def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


async def test_bitemporal_ingest_supersede_backfill_and_retrieval(
    create_test_user: Callable[[UserId | None], Coroutine[Any, Any, UserId]],
    uow_factory: PostgresUnitOfWorkFactory,
) -> None:
    """End-to-end test of bitemporal timeline and RetrievalOrchestrator over Postgres + pgvector."""
    user_id = await create_test_user(None)
    embedder = HashEmbedder()
    resolver = ConflictResolver(default_registry())

    # Step 0: Ensure self entity and place entities exist with real pgvector embeddings
    async with uow_factory.for_user(user_id) as uow:
        self_ent = await uow.entities.ensure_self(user_id)
        delhi_ent = await uow.entities.create(
            user_id, 'Delhi', EntityType.PLACE, await embedder.embed_query('Delhi')
        )
        pune_ent = await uow.entities.create(
            user_id, 'Pune', EntityType.PLACE, await embedder.embed_query('Pune')
        )
        blr_ent = await uow.entities.create(
            user_id, 'Bangalore', EntityType.PLACE, await embedder.embed_query('Bangalore')
        )

    # Step 1: Ingest initial fact — "I live in Delhi" in Jan 2025
    now_delhi = datetime(2025, 1, 15, 12, 0, 0, tzinfo=UTC)
    async with uow_factory.for_user(user_id) as uow:
        src_delhi = (
            await uow.sources.insert_many(
                [
                    NewSource(
                        user_id=user_id,
                        kind=SourceKind.CHAT,
                        role=Role.USER,
                        session_id='sess_delhi',
                        turn_index=0,
                        content='I live in Delhi',
                        source_ts=now_delhi,
                        content_hash=f'hash_delhi_{uuid4()}',
                    )
                ]
            )
        )[0]

        cand_delhi = Candidate(
            user_id=user_id,
            subject_id=self_ent.id,
            predicate='lives_in',
            cardinality=Cardinality.SINGLE,
            object_entity_id=delhi_ent.id,
            object_value=None,
            object_norm='delhi',
            fact_text='User lives in Delhi',
            valid_from=datetime(2025, 1, 1, tzinfo=UTC),
            valid_to=None,
            time_precision=TimePrecision.DAY,
            valid_from_explicit=True,
            confidence=1.0,
            is_retraction=False,
            source_id=src_delhi.id,
            span=(0, 16),
            source_ts=now_delhi,
            event_hash=None,
            embedding=await embedder.embed_query('User lives in Delhi'),
        )
        timeline_1 = await uow.facts.find_for_resolution(user_id, self_ent.id, 'lives_in')
        plan_1 = resolver.plan(cand_delhi, timeline_1, ResolutionContext(now=now_delhi))
        await PlanExecutor(cand_delhi, lock_user=False).execute(uow, plan_1, now=now_delhi)

    # Verify Step 1: Delhi is active with valid_to = None
    async with uow_factory.for_user(user_id) as uow:
        facts_1 = await uow.facts.load_timeline(user_id, self_ent.id, 'lives_in', lock=False)
        assert len(facts_1) == 1
        assert facts_1[0].object_norm == 'delhi'
        assert facts_1[0].valid_from == datetime(2025, 1, 1, tzinfo=UTC)
        assert facts_1[0].valid_to is None

    # Step 2: Supersede — "I moved to Pune" in Oct 2026
    now_pune = datetime(2026, 10, 5, 12, 0, 0, tzinfo=UTC)
    async with uow_factory.for_user(user_id) as uow:
        src_pune = (
            await uow.sources.insert_many(
                [
                    NewSource(
                        user_id=user_id,
                        kind=SourceKind.CHAT,
                        role=Role.USER,
                        session_id='sess_pune',
                        turn_index=1,
                        content='I moved to Pune',
                        source_ts=now_pune,
                        content_hash=f'hash_pune_{uuid4()}',
                    )
                ]
            )
        )[0]

        cand_pune = Candidate(
            user_id=user_id,
            subject_id=self_ent.id,
            predicate='lives_in',
            cardinality=Cardinality.SINGLE,
            object_entity_id=pune_ent.id,
            object_value=None,
            object_norm='pune',
            fact_text='User lives in Pune',
            valid_from=datetime(2026, 10, 1, tzinfo=UTC),
            valid_to=None,
            time_precision=TimePrecision.DAY,
            valid_from_explicit=True,
            confidence=1.0,
            is_retraction=False,
            source_id=src_pune.id,
            span=(0, 15),
            source_ts=now_pune,
            event_hash=None,
            embedding=await embedder.embed_query('User lives in Pune'),
        )
        timeline_2 = await uow.facts.find_for_resolution(user_id, self_ent.id, 'lives_in')
        plan_2 = resolver.plan(cand_pune, timeline_2, ResolutionContext(now=now_pune))
        await PlanExecutor(cand_pune, lock_user=False).execute(uow, plan_2, now=now_pune)

    # Verify Step 2: Delhi closed at 2026-10-01, Pune active from 2026-10-01 to None
    async with uow_factory.for_user(user_id) as uow:
        facts_2 = await uow.facts.load_timeline(user_id, self_ent.id, 'lives_in', lock=False)
        assert len(facts_2) == 2
        delhi_f = next(f for f in facts_2 if f.object_norm == 'delhi')
        pune_f = next(f for f in facts_2 if f.object_norm == 'pune')
        assert delhi_f.valid_to == datetime(2026, 10, 1, tzinfo=UTC)
        assert pune_f.valid_from == datetime(2026, 10, 1, tzinfo=UTC)
        assert pune_f.valid_to is None

    # Step 3: Out-of-order backfill (T-CR-05) — "In 2024 I lived in Bangalore"
    now_blr = datetime(2026, 10, 20, 12, 0, 0, tzinfo=UTC)
    async with uow_factory.for_user(user_id) as uow:
        src_blr = (
            await uow.sources.insert_many(
                [
                    NewSource(
                        user_id=user_id,
                        kind=SourceKind.NOTE,
                        role=Role.USER,
                        session_id='note_backfill',
                        turn_index=0,
                        content='In 2024 I lived in Bangalore',
                        source_ts=now_blr,
                        content_hash=f'hash_blr_{uuid4()}',
                    )
                ]
            )
        )[0]

        cand_blr = Candidate(
            user_id=user_id,
            subject_id=self_ent.id,
            predicate='lives_in',
            cardinality=Cardinality.SINGLE,
            object_entity_id=blr_ent.id,
            object_value=None,
            object_norm='bangalore',
            fact_text='User lives in Bangalore',
            valid_from=datetime(2024, 1, 1, tzinfo=UTC),
            valid_to=None,
            time_precision=TimePrecision.YEAR,
            valid_from_explicit=True,
            confidence=1.0,
            is_retraction=False,
            source_id=src_blr.id,
            span=(0, 28),
            source_ts=now_blr,
            event_hash=None,
            embedding=await embedder.embed_query('User lives in Bangalore'),
        )
        timeline_3 = await uow.facts.find_for_resolution(user_id, self_ent.id, 'lives_in')
        plan_3 = resolver.plan(cand_blr, timeline_3, ResolutionContext(now=now_blr))
        await PlanExecutor(cand_blr, lock_user=False).execute(uow, plan_3, now=now_blr)

    # Verify Step 3: Bangalore is clipped to Delhi's start [2024-01-01, 2025-01-01)
    async with uow_factory.for_user(user_id) as uow:
        facts_3 = await uow.facts.load_timeline(user_id, self_ent.id, 'lives_in', lock=False)
        assert len(facts_3) == 3
        blr_f = next(f for f in facts_3 if f.object_norm == 'bangalore')
        delhi_f = next(f for f in facts_3 if f.object_norm == 'delhi')
        pune_f = next(f for f in facts_3 if f.object_norm == 'pune')

        assert blr_f.valid_from == datetime(2024, 1, 1, tzinfo=UTC)
        assert blr_f.valid_to == datetime(2025, 1, 1, tzinfo=UTC)
        assert delhi_f.valid_from == datetime(2025, 1, 1, tzinfo=UTC)
        assert delhi_f.valid_to == datetime(2026, 10, 1, tzinfo=UTC)
        assert pune_f.valid_from == datetime(2026, 10, 1, tzinfo=UTC)
        assert pune_f.valid_to is None

    # Step 4: Verify queries via RetrievalOrchestrator against Postgres + pgvector
    eval_now = datetime(2026, 10, 28, 12, 0, 0, tzinfo=UTC)
    orchestrator = RetrievalOrchestrator(
        uow_factory=uow_factory,
        embedder=embedder,
        clock=FixedClock(eval_now),
    )

    # 4a. CURRENT query: "Where do I live currently?"
    res_current = await orchestrator.retrieve(
        user_id=user_id,
        query='Where do I live currently?',
        question_ts=eval_now,
    )
    current_fact_texts = [
        item.fact.fact_text for item in res_current.items if item.fact is not None
    ]
    assert any('Pune' in t for t in current_fact_texts)
    assert not any('Delhi' in t for t in current_fact_texts)
    assert not any('Bangalore' in t for t in current_fact_texts)

    # 4b. AS_OF query: "Where did I live as of June 2025?"
    as_of_2025 = datetime(2025, 6, 1, 12, 0, 0, tzinfo=UTC)
    plan_asof_2025 = QueryPlan(
        question='Where did I live as of June 2025?',
        question_ts=eval_now,
        tz='UTC',
        intent=Intent.POINT_IN_TIME,
        as_of=as_of_2025,
        time_range=None,
        entity_ids=(self_ent.id,),
        entity_names=('User',),
        predicates=('lives_in',),
        multi_hop=False,
        expects_count=False,
        ambiguous_time=False,
        planner_path='rules',
        confidence=1.0,
    )
    res_asof = await orchestrator.retrieve(user_id=user_id, query=plan_asof_2025)
    asof_fact_texts = [item.fact.fact_text for item in res_asof.items if item.fact is not None]
    assert any('Delhi' in t for t in asof_fact_texts)
    assert not any('Pune' in t for t in asof_fact_texts)
    assert not any('Bangalore' in t for t in asof_fact_texts)

    # 4c. AS_OF query: "Where did I live as of June 2024?"
    as_of_2024 = datetime(2024, 6, 1, 12, 0, 0, tzinfo=UTC)
    plan_asof_2024 = QueryPlan(
        question='Where did I live as of June 2024?',
        question_ts=eval_now,
        tz='UTC',
        intent=Intent.POINT_IN_TIME,
        as_of=as_of_2024,
        time_range=None,
        entity_ids=(self_ent.id,),
        entity_names=('User',),
        predicates=('lives_in',),
        multi_hop=False,
        expects_count=False,
        ambiguous_time=False,
        planner_path='rules',
        confidence=1.0,
    )
    res_asof_2024 = await orchestrator.retrieve(user_id=user_id, query=plan_asof_2024)
    asof_2024_fact_texts = [
        item.fact.fact_text for item in res_asof_2024.items if item.fact is not None
    ]
    assert any('Bangalore' in t for t in asof_2024_fact_texts)
    assert not any('Delhi' in t for t in asof_2024_fact_texts)
    assert not any('Pune' in t for t in asof_2024_fact_texts)

    # 4d. OVERLAP query: year 2024 [2024-01-01, 2025-01-01)
    plan_overlap_2024 = QueryPlan(
        question='Where did I live in 2024?',
        question_ts=eval_now,
        tz='UTC',
        intent=Intent.INTERVAL,
        as_of=None,
        time_range=TimeInterval(
            start=datetime(2024, 1, 1, tzinfo=UTC),
            end=datetime(2025, 1, 1, tzinfo=UTC),
        ),
        entity_ids=(self_ent.id,),
        entity_names=('User',),
        predicates=('lives_in',),
        multi_hop=False,
        expects_count=False,
        ambiguous_time=False,
        planner_path='rules',
        confidence=1.0,
    )
    res_overlap_2024 = await orchestrator.retrieve(user_id=user_id, query=plan_overlap_2024)
    overlap_2024_facts = [
        item.fact.fact_text for item in res_overlap_2024.items if item.fact is not None
    ]
    assert any('Bangalore' in t for t in overlap_2024_facts)
    assert not any('Delhi' in t for t in overlap_2024_facts)
    assert not any('Pune' in t for t in overlap_2024_facts)

    # 4e. OVERLAP query: multi-year interval 2024 to 2027 [2024-01-01, 2027-01-01)
    plan_multi = QueryPlan(
        question='Where have I lived from 2024 to 2027?',
        question_ts=eval_now,
        tz='UTC',
        intent=Intent.INTERVAL,
        as_of=None,
        time_range=TimeInterval(
            start=datetime(2024, 1, 1, tzinfo=UTC),
            end=datetime(2027, 1, 1, tzinfo=UTC),
        ),
        entity_ids=(self_ent.id,),
        entity_names=('User',),
        predicates=('lives_in',),
        multi_hop=False,
        expects_count=False,
        ambiguous_time=False,
        planner_path='rules',
        confidence=1.0,
    )
    res_multi = await orchestrator.retrieve(user_id=user_id, query=plan_multi)
    multi_facts = [item.fact.fact_text for item in res_multi.items if item.fact is not None]
    assert any('Bangalore' in t for t in multi_facts)
    assert any('Delhi' in t for t in multi_facts)
    assert any('Pune' in t for t in multi_facts)
