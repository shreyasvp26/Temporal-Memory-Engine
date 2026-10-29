"""Workbench helper endpoints for dev auth, job execution, and demo scenario seeding."""

import contextlib
import hashlib
import logging
from datetime import UTC, datetime, timedelta
from typing import Annotated, Any
from uuid import UUID, uuid4

import jwt
from fastapi import APIRouter, Depends, Query, status
from memory_engine.config import Settings
from memory_engine.domain.clock import Clock
from memory_engine.domain.enums import (
    Cardinality,
    EntityType,
    FactOrigin,
    FactStatus,
    JobStatus,
    JobType,
    RelationType,
    Role,
    SourceKind,
    TimePrecision,
)
from memory_engine.domain.errors import Forbidden
from memory_engine.domain.ids import EntityId, FactId, UserId
from memory_engine.domain.models import Candidate, Entity, FactEvidence, FactRelation, Job
from memory_engine.ingest.service import IngestService
from memory_engine.lifecycle.service import MemoryLifecycleService
from memory_engine.ports.embed import Embedder
from memory_engine.ports.repositories import NewSource
from memory_engine.ports.uow import UnitOfWorkFactory

from .auth import get_current_user
from .deps import (
    ContainerDep,
    get_clock,
    get_embedder,
    get_ingest_service,
    get_lifecycle_service,
    get_settings,
    get_uow_factory,
)

_logger = logging.getLogger(__name__)

dev_router = APIRouter()

CurrentUser = Annotated[UserId, Depends(get_current_user)]
UowFactoryDep = Annotated[UnitOfWorkFactory, Depends(get_uow_factory)]
IngestServiceDep = Annotated[IngestService, Depends(get_ingest_service)]
LifecycleServiceDep = Annotated[MemoryLifecycleService, Depends(get_lifecycle_service)]
SettingsDep = Annotated[Settings, Depends(get_settings)]
ClockDep = Annotated[Clock, Depends(get_clock)]
EmbedderDep = Annotated[Embedder | None, Depends(get_embedder)]

DEFAULT_WORKBENCH_USER_ID = UUID('00000000-0000-0000-0000-000000000001')


# ---------------------------------------------------------------------------
# Dev Auth Token Minting
# ---------------------------------------------------------------------------


@dev_router.get('/v1/dev/token')
async def get_dev_token(
    settings: SettingsDep,
    clock: ClockDep,
    user_id: Annotated[UUID, Query()] = DEFAULT_WORKBENCH_USER_ID,
) -> dict[str, str]:
    """Mint a dev JWT for the specified user persona in dev/test/local environments."""
    if settings.env not in ('dev', 'test', 'local'):
        raise Forbidden('Dev endpoints are disabled in production', code='E-AUTH-02')

    now_dt = clock.now()
    exp_dt = now_dt + timedelta(days=7)
    payload: dict[str, Any] = {
        'sub': str(user_id),
        'aud': settings.jwt.audience or 'authenticated',
        'exp': int(exp_dt.timestamp()),
        'iat': int(now_dt.timestamp()),
    }
    if settings.jwt.issuer:
        payload['iss'] = settings.jwt.issuer

    secret = settings.jwt.secret or ''
    token = jwt.encode(payload, secret, algorithm='HS256')
    return {
        'token': token,
        'user_id': str(user_id),
        'env': settings.env,
    }


# ---------------------------------------------------------------------------
# On-Demand Worker Execution
# ---------------------------------------------------------------------------


@dev_router.post('/v1/jobs/run-once')
async def run_jobs_once(
    user_id: CurrentUser,
    uow_factory: UowFactoryDep,
    ingest: IngestServiceDep,
) -> dict[str, Any]:
    """Claim and execute pending queued extract jobs for current user (up to 10 jobs)."""
    pending_jobs: list[Job] = []
    async with uow_factory.for_user(user_id) as uow:
        if hasattr(uow.jobs, 'list_for_user'):
            user_jobs = await uow.jobs.list_for_user(user_id, status=JobStatus.QUEUED, limit=10)
            pending_jobs = [
                j for j in user_jobs if j.type == JobType.EXTRACT or str(j.type) == 'extract'
            ]
        elif hasattr(uow.jobs, 'jobs'):
            for j in list(uow.jobs.jobs.values()):
                if (
                    j.user_id == user_id
                    and (j.type == JobType.EXTRACT or str(j.type) == 'extract')
                    and (j.status == JobStatus.QUEUED or str(j.status) == 'queued')
                ):
                    pending_jobs.append(j)
                    if len(pending_jobs) >= 10:
                        break

    processed_ids: list[UUID] = []
    for job in pending_jobs:
        try:
            await ingest.run_extract_job(job)
        except Exception as exc:
            _logger.warning('Failed to execute job %s: %s', job.id, exc)

        processed_ids.append(UUID(str(job.id)))

        async with uow_factory.for_user(user_id) as uow:
            if hasattr(uow, 'conn') and hasattr(uow.conn, 'execute'):
                with contextlib.suppress(Exception):
                    sql = (
                        "update jobs set status = 'done', finished_at = now() "
                        'where id = $1 and user_id = $2;'
                    )
                    await uow.conn.execute(sql, job.id, user_id)
            elif hasattr(uow.jobs, 'jobs') and job.id in uow.jobs.jobs:
                from dataclasses import replace

                uow.jobs.jobs[job.id] = replace(uow.jobs.jobs[job.id], status=JobStatus.DONE)

    return {
        'processed': len(processed_ids),
        'job_ids': [str(jid) for jid in processed_ids],
    }


# ---------------------------------------------------------------------------
# Bitemporal Demo Scenario Seeding
# ---------------------------------------------------------------------------


@dev_router.post('/v1/dev/seed-scenario', status_code=status.HTTP_201_CREATED)
async def seed_bitemporal_scenario(
    user_id: CurrentUser,
    settings: SettingsDep,
    clock: ClockDep,
    uow_factory: UowFactoryDep,
    lifecycle: LifecycleServiceDep,
    container: ContainerDep,
) -> dict[str, Any]:
    """Seed a deterministic 3-stage bitemporal scenario illustrating historical backfill,
    prior state, and current active multi-valued facts.
    """
    if settings.env not in ('dev', 'test', 'local'):
        raise Forbidden('Dev endpoints are disabled in production', code='E-AUTH-02')

    # 1. Clean purge for current tenant
    await lifecycle.purge_user(user_id)

    embedder: Embedder | None = (
        container.embedder
        or getattr(container.retrieval, 'embedder', None)
        or getattr(container.ingest, 'embedder', None)
    )
    dim = getattr(embedder, 'dim', 768) if embedder is not None else 768
    model_id = getattr(embedder, 'model_id', 'default') if embedder is not None else 'default'

    async def _embed(text: str) -> list[float]:
        if embedder is not None:
            try:
                if hasattr(embedder, 'embed_query'):
                    vec = await embedder.embed_query(text)
                    return [float(x) for x in vec]
                elif hasattr(embedder, 'embed_documents'):
                    vecs = await embedder.embed_documents([text])
                    return [float(x) for x in vecs[0]]
            except Exception as exc:
                _logger.warning('Embedding computation failed in seed scenario: %s', exc)
        return [0.0] * dim

    now = clock.now()

    # 2. Canonical entities setup
    async with uow_factory.for_user(user_id) as uow:
        self_ent = await uow.entities.ensure_self(user_id)

        async def _ensure_ent(name: str, etype: EntityType) -> Entity:
            if hasattr(uow.entities, 'create'):
                emb = await _embed(name)
                return await uow.entities.create(user_id, name, etype, emb)
            else:
                eid = EntityId(uuid4())
                ent = Entity(
                    id=eid,
                    user_id=user_id,
                    canonical_name=name,
                    type=etype,
                    is_self=False,
                )
                if hasattr(uow.entities, 'entities'):
                    uow.entities.entities[eid] = ent
                return ent

        blr_ent = await _ensure_ent('Bangalore', EntityType.PLACE)
        delhi_ent = await _ensure_ent('Delhi', EntityType.PLACE)
        pune_ent = await _ensure_ent('Pune', EntityType.PLACE)
        acme_ent = await _ensure_ent('Acme Corp', EntityType.ORG)
        stripe_ent = await _ensure_ent('Stripe', EntityType.ORG)
        pg_ent = await _ensure_ent('PostgreSQL', EntityType.THING)
        py_ent = await _ensure_ent('Python', EntityType.THING)

        # 3. Sources Creation
        q1 = 'Back in 2024, I was living in Bangalore and working on distributed systems.'
        q2 = 'In Jan 2025, I moved to Delhi and joined Acme Corp as a Senior Backend Engineer.'
        q3 = (
            'On October 1st 2026, I relocated from Delhi to Pune and joined Stripe. '
            'I primarily use Python and PostgreSQL, and I am vegetarian.'
        )

        sources_to_insert = [
            NewSource(
                user_id=user_id,
                kind=SourceKind.CHAT,
                role=Role.USER,
                session_id='seed_2024',
                turn_index=0,
                content=q1,
                source_ts=datetime(2024, 6, 1, 12, 0, 0, tzinfo=UTC),
                content_hash=hashlib.sha256(q1.encode('utf-8')).hexdigest(),
            ),
            NewSource(
                user_id=user_id,
                kind=SourceKind.CHAT,
                role=Role.USER,
                session_id='seed_2025',
                turn_index=1,
                content=q2,
                source_ts=datetime(2025, 1, 15, 12, 0, 0, tzinfo=UTC),
                content_hash=hashlib.sha256(q2.encode('utf-8')).hexdigest(),
            ),
            NewSource(
                user_id=user_id,
                kind=SourceKind.CHAT,
                role=Role.USER,
                session_id='seed_2026',
                turn_index=2,
                content=q3,
                source_ts=datetime(2026, 10, 1, 12, 0, 0, tzinfo=UTC),
                content_hash=hashlib.sha256(q3.encode('utf-8')).hexdigest(),
            ),
        ]

        inserted_sources = await uow.sources.insert_many(sources_to_insert)
        s1, s2, s3 = (
            inserted_sources[0].id,
            inserted_sources[1].id,
            inserted_sources[2].id,
        )

        if hasattr(uow.sources, 'set_embeddings'):
            await uow.sources.set_embeddings(
                user_id,
                [
                    (s1, await _embed(q1)),
                    (s2, await _embed(q2)),
                    (s3, await _embed(q3)),
                ],
                model_id,
            )

        # 4. Helper to insert Fact and supporting evidence
        async def _insert_fact(
            predicate: str,
            cardinality: Cardinality,
            obj_entity_id: EntityId | None,
            obj_value: str | None,
            obj_norm: str,
            fact_text: str,
            valid_from: datetime,
            valid_to: datetime | None,
            status: FactStatus,
            source_id: Any,
            span: tuple[int, int],
        ) -> Any:
            emb = await _embed(fact_text)
            cand = Candidate(
                user_id=user_id,
                subject_id=self_ent.id,
                predicate=predicate,
                cardinality=cardinality,
                object_entity_id=obj_entity_id,
                object_value=obj_value,
                object_norm=obj_norm,
                fact_text=fact_text,
                valid_from=valid_from,
                valid_to=valid_to,
                time_precision=TimePrecision.DAY,
                valid_from_explicit=True,
                confidence=1.0,
                is_retraction=False,
                source_id=source_id,
                span=span,
                source_ts=valid_from,
                event_hash=None,
                embedding=emb,
            )
            fact = await uow.facts.insert(
                cand,
                valid_from=valid_from,
                valid_to=valid_to,
                status=status,
                origin=FactOrigin.EXTRACTED,
                model_id=model_id,
                now=now,
            )
            ev = FactEvidence(
                fact_id=fact.id,
                source_id=source_id,
                span_start=span[0],
                span_end=span[1],
            )
            await uow.facts.add_evidence(ev, user_id)
            return fact

        # Stage 1: 2024 Historical Backfill (Bangalore)
        f_blr = await _insert_fact(
            predicate='lives_in',
            cardinality=Cardinality.SINGLE,
            obj_entity_id=blr_ent.id,
            obj_value=None,
            obj_norm='bangalore',
            fact_text='User lives in Bangalore',
            valid_from=datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC),
            valid_to=datetime(2025, 1, 1, 0, 0, 0, tzinfo=UTC),
            status=FactStatus.ACTIVE,
            source_id=s1,
            span=(0, len(q1)),
        )

        # Stage 2: 2025 Prior State (Delhi, Acme Corp)
        f_delhi = await _insert_fact(
            predicate='lives_in',
            cardinality=Cardinality.SINGLE,
            obj_entity_id=delhi_ent.id,
            obj_value=None,
            obj_norm='delhi',
            fact_text='User lives in Delhi',
            valid_from=datetime(2025, 1, 1, 0, 0, 0, tzinfo=UTC),
            valid_to=datetime(2026, 10, 1, 0, 0, 0, tzinfo=UTC),
            status=FactStatus.ACTIVE,
            source_id=s2,
            span=(0, len(q2)),
        )
        f_acme = await _insert_fact(
            predicate='works_at',
            cardinality=Cardinality.SINGLE,
            obj_entity_id=acme_ent.id,
            obj_value=None,
            obj_norm='acme corp',
            fact_text='User works at Acme Corp',
            valid_from=datetime(2025, 1, 1, 0, 0, 0, tzinfo=UTC),
            valid_to=datetime(2026, 10, 1, 0, 0, 0, tzinfo=UTC),
            status=FactStatus.ACTIVE,
            source_id=s2,
            span=(0, len(q2)),
        )

        # Stage 3: 2026 Current Active State (Pune, Stripe, PostgreSQL, Python, vegetarian)
        f_pune = await _insert_fact(
            predicate='lives_in',
            cardinality=Cardinality.SINGLE,
            obj_entity_id=pune_ent.id,
            obj_value=None,
            obj_norm='pune',
            fact_text='User lives in Pune',
            valid_from=datetime(2026, 10, 1, 0, 0, 0, tzinfo=UTC),
            valid_to=None,
            status=FactStatus.ACTIVE,
            source_id=s3,
            span=(0, len(q3)),
        )
        f_stripe = await _insert_fact(
            predicate='works_at',
            cardinality=Cardinality.SINGLE,
            obj_entity_id=stripe_ent.id,
            obj_value=None,
            obj_norm='stripe',
            fact_text='User works at Stripe',
            valid_from=datetime(2026, 10, 1, 0, 0, 0, tzinfo=UTC),
            valid_to=None,
            status=FactStatus.ACTIVE,
            source_id=s3,
            span=(0, len(q3)),
        )
        await _insert_fact(
            predicate='uses_tech',
            cardinality=Cardinality.MULTI,
            obj_entity_id=pg_ent.id,
            obj_value=None,
            obj_norm='postgresql',
            fact_text='User uses tech PostgreSQL',
            valid_from=datetime(2026, 10, 1, 0, 0, 0, tzinfo=UTC),
            valid_to=None,
            status=FactStatus.ACTIVE,
            source_id=s3,
            span=(0, len(q3)),
        )
        await _insert_fact(
            predicate='uses_tech',
            cardinality=Cardinality.MULTI,
            obj_entity_id=py_ent.id,
            obj_value=None,
            obj_norm='python',
            fact_text='User uses tech Python',
            valid_from=datetime(2026, 10, 1, 0, 0, 0, tzinfo=UTC),
            valid_to=None,
            status=FactStatus.ACTIVE,
            source_id=s3,
            span=(0, len(q3)),
        )
        await _insert_fact(
            predicate='dietary_preference',
            cardinality=Cardinality.SINGLE,
            obj_entity_id=None,
            obj_value='vegetarian',
            obj_norm='vegetarian',
            fact_text='User dietary preference vegetarian',
            valid_from=datetime(2026, 10, 1, 0, 0, 0, tzinfo=UTC),
            valid_to=None,
            status=FactStatus.ACTIVE,
            source_id=s3,
            span=(0, len(q3)),
        )

        # 5. Supersession chains
        await uow.facts.add_relation(
            user_id,
            FactRelation(
                from_fact_id=FactId(f_delhi.id),
                to_fact_id=FactId(f_blr.id),
                relation=RelationType.SUPERSEDES,
            ),
            job_id=None,
        )
        await uow.facts.add_relation(
            user_id,
            FactRelation(
                from_fact_id=FactId(f_pune.id),
                to_fact_id=FactId(f_delhi.id),
                relation=RelationType.SUPERSEDES,
            ),
            job_id=None,
        )
        await uow.facts.add_relation(
            user_id,
            FactRelation(
                from_fact_id=FactId(f_stripe.id),
                to_fact_id=FactId(f_acme.id),
                relation=RelationType.SUPERSEDES,
            ),
            job_id=None,
        )

    return {
        'seeded_facts': 8,
        'seeded_sources': 3,
        'user_id': str(user_id),
    }
