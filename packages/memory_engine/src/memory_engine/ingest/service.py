"""Ingestion service implementing enqueue_turn, accept_note, and 4-phase extract job."""

import hashlib
import logging
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Any
from uuid import UUID, uuid4
from zoneinfo import ZoneInfo

from memory_engine.config import IngestSettings
from memory_engine.domain.clock import Clock, SystemClock
from memory_engine.domain.enums import (
    Cardinality,
    DecayClass,
    EntityType,
    FactOrigin,
    ObjectKind,
    Role,
    SourceKind,
)
from memory_engine.domain.errors import InvariantViolation, LeaseLost
from memory_engine.domain.ids import EntityId, FactId, JobId, SourceId, UserId
from memory_engine.domain.models import Candidate, Fact, Job, JudgeHint, Predicate
from memory_engine.ingest.canonicalizer import (
    Canonicalizer,
    CanonicalObject,
)
from memory_engine.ingest.conflict import (
    CloseFact,
    ConflictResolver,
    ExtendValidFrom,
    InsertFact,
    ResolutionContext,
    ResolutionPlan,
    SetStatus,
    default_registry,
)
from memory_engine.ingest.conflict.plan import BumpConfidence
from memory_engine.ingest.entity_linker import (
    EntityLinker,
    is_self_mention,
    precompute_mention_embeddings,
)
from memory_engine.ingest.extractor import (
    FactExtractor,
    ValidatedExtraction,
)
from memory_engine.ingest.gate import GateDecision, IngestGate
from memory_engine.ingest.plan_executor import PlanExecutor
from memory_engine.ingest.time_resolver import (
    ResolvedCandidateTime,
    TimeResolver,
)
from memory_engine.ingest.windowing import NoteChunker
from memory_engine.ports.embed import Embedder
from memory_engine.ports.llm import LLMGateway
from memory_engine.ports.repositories import (
    InsertedSource,
    NewExtractJob,
    NewSource,
)
from memory_engine.ports.uow import UnitOfWork, UnitOfWorkFactory

_logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class EnqueueResult:
    """Result of enqueue_turn operation."""

    source_id: SourceId
    created: bool
    job_id: JobId | None = None
    gate_decision: GateDecision | None = None


TurnAccepted = EnqueueResult


@dataclass(frozen=True, slots=True)
class NoteIngestResult:
    """Result of accept_note operation."""

    source_ids: list[SourceId]
    job_id: JobId | None = None
    created: bool = True


NoteAccepted = NoteIngestResult


def _apply_plan_to_in_memory_timeline(
    timeline: list[Fact],
    plan: ResolutionPlan,
    candidate: Candidate,
    touched_ids: list[FactId],
    now: datetime,
) -> None:
    """Reflect plan operations in-memory for sequential same-window candidate handling."""
    fact_map = {f.id: f for f in timeline}
    new_fact_id = touched_ids[-1] if touched_ids else None

    for op in plan.ops:
        if isinstance(op, CloseFact):
            if op.fact_id in fact_map:
                old = fact_map[op.fact_id]
                fact_map[op.fact_id] = replace(old, valid_to=op.valid_to, invalidated_at=now)
        elif isinstance(op, ExtendValidFrom):
            if op.fact_id in fact_map:
                old = fact_map[op.fact_id]
                fact_map[op.fact_id] = replace(old, valid_from=op.valid_from)
        elif isinstance(op, SetStatus):
            if op.fact_id in fact_map:
                old = fact_map[op.fact_id]
                fact_map[op.fact_id] = replace(old, status=op.status)
        elif isinstance(op, BumpConfidence):
            if op.fact_id in fact_map:
                old = fact_map[op.fact_id]
                fact_map[op.fact_id] = replace(old, confidence=op.confidence)
        elif isinstance(op, InsertFact):
            fid = new_fact_id or FactId(uuid4())
            new_fact = Fact(
                id=fid,
                user_id=candidate.user_id,
                subject_id=candidate.subject_id,
                predicate=candidate.predicate,
                cardinality=candidate.cardinality,
                object_entity_id=candidate.object_entity_id,
                object_value=candidate.object_value,
                object_norm=candidate.object_norm,
                fact_text=candidate.fact_text,
                valid_from=op.valid_from,
                valid_to=op.valid_to,
                time_precision=candidate.time_precision,
                valid_from_explicit=candidate.valid_from_explicit,
                recorded_at=now,
                invalidated_at=None,
                confidence=candidate.confidence,
                salience=1.0,
                access_count=0,
                last_accessed_at=None,
                status=op.status,
                origin=FactOrigin.EXTRACTED,
                event_hash=candidate.event_hash,
            )
            fact_map[fid] = new_fact

    timeline.clear()
    sorted_facts = sorted(fact_map.values(), key=lambda f: (f.valid_from, f.recorded_at))
    timeline.extend(sorted_facts)


class IngestService:
    """Ingestion service coordinating writes, gates, chunking, and 4-phase extract jobs."""

    def __init__(
        self,
        gate: IngestGate | None = None,
        chunker: NoteChunker | None = None,
        clock: Clock | None = None,
        cfg: IngestSettings | None = None,
        *,
        uows: UnitOfWorkFactory | None = None,
        extractor: FactExtractor | None = None,
        canonicalizer: Canonicalizer | None = None,
        time_resolver: TimeResolver | None = None,
        entity_linker: EntityLinker | None = None,
        conflict_resolver: ConflictResolver | None = None,
        plan_executor: PlanExecutor | None = None,
        embedder: Embedder | None = None,
        llm: LLMGateway | None = None,
        debounce_s: float | None = None,
        notes_debounce_s: float | None = None,
        context_turns: int = 4,
        extract_assistant: bool | None = None,
        enable_conflict_judge: bool = False,
    ) -> None:
        self.gate = gate or IngestGate()
        self.chunker = chunker or NoteChunker()
        self.clock = clock or SystemClock()
        self.cfg = cfg or IngestSettings()

        self.uows = uows
        self.extractor = extractor
        self.canonicalizer = canonicalizer or Canonicalizer()
        self.time_resolver = time_resolver or TimeResolver()
        self.embedder = embedder
        self.entity_linker = entity_linker or EntityLinker(embedder=embedder)
        self.conflict_resolver = conflict_resolver or ConflictResolver(default_registry())
        self.plan_executor = plan_executor or PlanExecutor(clock=self.clock)
        self.llm = llm

        self.debounce_s = (
            debounce_s if debounce_s is not None else float(getattr(self.cfg, 'debounce_s', 15.0))
        )
        self.notes_debounce_s = (
            notes_debounce_s
            if notes_debounce_s is not None
            else float(getattr(self.cfg, 'notes_debounce_s', 2.0))
        )
        self.context_turns = context_turns
        self.extract_assistant = (
            extract_assistant
            if extract_assistant is not None
            else bool(getattr(self.cfg, 'extract_assistant', False))
        )
        self.enable_conflict_judge = enable_conflict_judge

    def _is_extractable(self, role: Role | str) -> bool:
        r = role.value if hasattr(role, 'value') else str(role)
        if r == Role.USER.value or r == 'user':
            return True
        return bool((r == Role.ASSISTANT.value or r == 'assistant') and self.extract_assistant)

    async def ensure_user_bootstrap(self, uow: UnitOfWork, user_id: UserId) -> None:
        """Ensure profile row and self entity exist for user."""
        if hasattr(uow, 'profiles') and hasattr(uow.profiles, 'ensure'):
            await uow.profiles.ensure(user_id)
        if hasattr(uow, 'entities') and hasattr(uow.entities, 'ensure_self'):
            await uow.entities.ensure_self(user_id)

    async def enqueue_turn(
        self,
        uow: UnitOfWork,
        *,
        user_id: UserId,
        session_id: str | None = None,
        conversation_id: UUID | None = None,
        role: Role,
        text: str | None = None,
        content: str | None = None,
        source_ts: datetime | None = None,
        ts: datetime | None = None,
        client_message_id: str | UUID | None = None,
        message_id: UUID | str | None = None,
        kind: SourceKind = SourceKind.CHAT,
    ) -> EnqueueResult:
        """Enqueue a chat turn idempotently and schedule coalesced extract job if gate passes."""
        raw_content = text if content is None else content
        content_str = '' if raw_content is None else raw_content

        effective_ts = ts if source_ts is None else source_ts
        if effective_ts is None:
            effective_ts = self.clock.now()

        effective_session_id = str(conversation_id) if session_id is None else session_id
        if not effective_session_id:
            effective_session_id = str(uuid4())

        effective_msg_id = (
            str(uuid4())
            if (client_message_id is None and message_id is None)
            else str(message_id if client_message_id is None else client_message_id)
        )

        kind_val = kind.value if hasattr(kind, 'value') else str(kind)
        role_val = role.value if hasattr(role, 'value') else str(role)

        raw_hash = f'{kind_val}|{effective_session_id}|{effective_msg_id}|{role_val}|{content_str}'
        content_hash = hashlib.sha256(raw_hash.encode('utf-8')).hexdigest()

        new_src = NewSource(
            user_id=user_id,
            kind=kind,
            role=role,
            session_id=effective_session_id,
            turn_index=0,
            content=content_str,
            source_ts=effective_ts,
            content_hash=content_hash,
        )

        inserted_list: list[InsertedSource] = await uow.sources.insert_many([new_src])
        if not inserted_list:
            raise InvariantViolation('insert_many returned empty results')
        inserted = inserted_list[0]

        if not inserted.created:
            # Client retry / idempotent deduplication (CR-01)
            return EnqueueResult(
                source_id=inserted.id,
                created=False,
                job_id=None,
                gate_decision=None,
            )

        gate_decision: GateDecision | None = None
        is_extractable = self._is_extractable(role)
        if is_extractable:
            gate_decision = self.gate.decide(content_str)
            if not gate_decision.ingest:
                return EnqueueResult(
                    source_id=inserted.id,
                    created=True,
                    job_id=None,
                    gate_decision=gate_decision,
                )

            # Ingestible turn -> upsert coalesced extract job (CR-02)
            coalesce_key = f'extract:{user_id}:{effective_session_id}'
            idempotency_key = f'extract:{inserted.id}'
            job_req = NewExtractJob(
                user_id=user_id,
                session_id=effective_session_id,
                source_id=inserted.id,
                idempotency_key=idempotency_key,
                coalesce_key=coalesce_key,
            )
            upsert_res = await uow.jobs.upsert_coalesced(job_req, self.debounce_s)
            return EnqueueResult(
                source_id=inserted.id,
                created=True,
                job_id=upsert_res.job_id,
                gate_decision=gate_decision,
            )

        return EnqueueResult(
            source_id=inserted.id,
            created=True,
            job_id=None,
            gate_decision=None,
        )

    async def accept_note(
        self,
        uow: UnitOfWork,
        *,
        user_id: UserId,
        text: str | None = None,
        content: str | None = None,
        source_ts: datetime | None = None,
        ts: datetime | None = None,
        kind: SourceKind = SourceKind.NOTE,
        title: str | None = None,
        note_id: UUID | None = None,
    ) -> NoteIngestResult:
        """Chunk note, insert sources, and schedule immediate extract job."""
        effective_note_id = note_id or uuid4()
        session_id = f'note:{effective_note_id}'

        raw_content = text if content is None else content
        note_text = '' if raw_content is None else raw_content
        if title:
            note_text = f'{title}\n\n{note_text}'.strip()

        effective_ts = ts if source_ts is None else source_ts
        if effective_ts is None:
            effective_ts = self.clock.now()

        chunks = self.chunker.split(note_text)
        if not chunks:
            return NoteIngestResult(source_ids=[], job_id=None, created=False)

        kind_val = kind.value if hasattr(kind, 'value') else str(kind)
        role_val = Role.USER.value if hasattr(Role.USER, 'value') else str(Role.USER)

        new_sources: list[NewSource] = []
        for idx, chunk in enumerate(chunks):
            chunk_hash_raw = f'{kind_val}|{session_id}|{idx}|{role_val}|{chunk}'
            content_hash = hashlib.sha256(chunk_hash_raw.encode('utf-8')).hexdigest()
            new_sources.append(
                NewSource(
                    user_id=user_id,
                    kind=kind,
                    role=Role.USER,
                    session_id=session_id,
                    turn_index=idx,
                    content=chunk,
                    source_ts=effective_ts,
                    content_hash=content_hash,
                )
            )

        inserted_list = await uow.sources.insert_many(new_sources)
        source_ids = [s.id for s in inserted_list]
        any_created = any(s.created for s in inserted_list)

        if not any_created:
            return NoteIngestResult(source_ids=source_ids, job_id=None, created=False)

        # Enqueue immediate extract job (debounce_s=0.0)
        coalesce_key = f'extract:{user_id}:{session_id}'
        last_job_id: JobId | None = None
        for s in inserted_list:
            if s.created:
                job_req = NewExtractJob(
                    user_id=user_id,
                    session_id=session_id,
                    source_id=s.id,
                    idempotency_key=f'extract:{s.id}',
                    coalesce_key=coalesce_key,
                )
                upsert_res = await uow.jobs.upsert_coalesced(job_req, 0.0)
                last_job_id = upsert_res.job_id

        return NoteIngestResult(
            source_ids=source_ids,
            job_id=last_job_id,
            created=True,
        )

    async def run_extract_job(
        self,
        job: Job,
        worker_id: UUID | None = None,
    ) -> list[FactId]:
        """Execute 4-phase extract job per docs/LLD.md Section 8.2 and 8.3."""
        if self.uows is None:
            raise InvariantViolation('uows (UnitOfWorkFactory) must be configured on IngestService')
        if self.extractor is None:
            raise InvariantViolation('extractor must be configured on IngestService')

        session_id = str(job.payload.get('session_id', ''))
        raw_source_ids: list[Any] = job.payload.get('source_ids', [])
        target_ids = [SourceId(UUID(str(sid))) for sid in raw_source_ids]

        # ---------------------------------------------------------------------
        # Phase A: Read-only transaction (fetch window and timezone)
        # ---------------------------------------------------------------------
        async with self.uows.begin(job.user_id, readonly=True) as uow:
            window = await uow.sources.window(
                job.user_id,
                session_id,
                target_ids,
                self.context_turns,
            )
            tz_str = 'UTC'
            if hasattr(uow, 'profiles') and hasattr(uow.profiles, 'timezone'):
                try:
                    tz_str = await uow.profiles.timezone(job.user_id)
                except Exception:
                    tz_str = 'UTC'

        try:
            user_tz = ZoneInfo(tz_str)
        except Exception:
            user_tz = ZoneInfo('UTC')

        extractable_targets = [
            s for s in window.targets if s.deleted_at is None and self._is_extractable(s.role)
        ]
        if not extractable_targets:
            if worker_id is not None:
                async with self.uows.begin(job.user_id) as uow:
                    if hasattr(uow.jobs, 'complete'):
                        ok = await uow.jobs.complete(job.id, worker_id, self.clock.now())
                        if not ok:
                            raise LeaseLost(f'Lease lost for job {job.id}')
            return []

        # ---------------------------------------------------------------------
        # Phase B: Zero DB tx (LLM extraction, embedding, normalization, time resolution)
        # ---------------------------------------------------------------------
        source_embeddings: list[tuple[SourceId, list[float]]] = []
        if self.embedder is not None and extractable_targets:
            embs = await self.embedder.embed_documents([s.content for s in extractable_targets])
            source_embeddings = [
                (s.id, [float(x) for x in emb])
                for s, emb in zip(extractable_targets, embs, strict=True)
            ]

        extractions: list[ValidatedExtraction] = await self.extractor.extract(window)

        if not extractions:
            if worker_id is not None or source_embeddings:
                async with self.uows.begin(job.user_id) as uow:
                    if source_embeddings and hasattr(uow.sources, 'set_embeddings'):
                        model_id = getattr(self.embedder, 'model_id', 'default')
                        await uow.sources.set_embeddings(job.user_id, source_embeddings, model_id)
                    if worker_id is not None and hasattr(uow.jobs, 'complete'):
                        ok = await uow.jobs.complete(job.id, worker_id, self.clock.now())
                        if not ok:
                            raise LeaseLost(f'Lease lost for job {job.id}')
            return []

        # Canonicalize predicates, objects, times, and render provisional text
        prepared_candidates: list[dict[str, Any]] = []
        mentions_to_embed: list[str] = []
        provisional_texts_to_embed: list[str] = []

        for ext in extractions:
            if self.canonicalizer is not None:
                pred = self.canonicalizer.resolve_predicate(ext.fact.predicate)
                canon_obj: CanonicalObject = self.canonicalizer.canonicalize_object(
                    pred,
                    ext.fact.object,
                    object_type=ext.fact.object_type,
                )
            else:
                pred = Predicate(
                    name=ext.fact.predicate,
                    cardinality=Cardinality.SINGLE,
                    decay_class=DecayClass.NORMAL,
                    object_kind=ObjectKind.LITERAL,
                    template='{subject} ' + ext.fact.predicate + ' {object}',
                    synonyms=(),
                )
                canon_obj = CanonicalObject(
                    kind=ObjectKind.LITERAL,
                    entity_id=None,
                    value=ext.fact.object.strip(),
                    entity_type=None,
                    text=ext.fact.object.strip(),
                )

            time_bounds: ResolvedCandidateTime = self.time_resolver.resolve_fact_bounds(
                valid_from_text=ext.fact.valid_from_text,
                valid_to_text=ext.fact.valid_to_text,
                source_ts=ext.source.source_ts,
                tz=user_tz,
                predicate=pred.name,
                cardinality=pred.cardinality,
            )

            if self.canonicalizer is not None:
                prov_text = self.canonicalizer.render_canonical_text(
                    pred,
                    subject=ext.fact.subject,
                    object=canon_obj,
                )
            else:
                prov_text = f'{ext.fact.subject} {pred.name} {canon_obj.text}'

            prepared_candidates.append(
                {
                    'ext': ext,
                    'pred': pred,
                    'canon_obj': canon_obj,
                    'time_bounds': time_bounds,
                    'prov_text': prov_text,
                }
            )

            if not is_self_mention(ext.fact.subject):
                mentions_to_embed.append(ext.fact.subject)
            if pred.object_kind == ObjectKind.ENTITY and not is_self_mention(ext.fact.object):
                mentions_to_embed.append(ext.fact.object)

            provisional_texts_to_embed.append(prov_text)

        # Batch-embed unique mentions and provisional fact texts
        mention_embeddings: dict[str, list[float]] = {}
        fact_text_embeddings: list[list[float]] = []

        if self.embedder is not None:
            if mentions_to_embed:
                mention_embeddings = await precompute_mention_embeddings(
                    self.embedder, mentions_to_embed
                )
            if provisional_texts_to_embed:
                raw_text_embs = await self.embedder.embed_documents(provisional_texts_to_embed)
                fact_text_embeddings = [[float(x) for x in emb] for emb in raw_text_embs]

        # ---------------------------------------------------------------------
        # Phase C: Optional ambiguity judge
        # ---------------------------------------------------------------------
        hints: list[JudgeHint | None] = [None] * len(prepared_candidates)
        if self.enable_conflict_judge and self.llm is not None:
            # Ambiguity judge hook when enabled (defaults off per LLD 8.8)
            pass

        # ---------------------------------------------------------------------
        # Phase D: Serialized write transaction under uow.lock_user
        # ---------------------------------------------------------------------
        model_id = getattr(self.embedder, 'model_id', 'default')
        touched_fact_ids: list[FactId] = []

        async with self.uows.begin(job.user_id) as uow:
            await uow.lock_user(job.user_id)

            if source_embeddings and hasattr(uow.sources, 'set_embeddings'):
                await uow.sources.set_embeddings(job.user_id, source_embeddings, model_id)

            timeline_cache: dict[tuple[EntityId, str], list[Fact]] = {}

            for idx, item in enumerate(prepared_candidates):
                ext = item['ext']
                pred = item['pred']
                canon_obj = item['canon_obj']
                time_bounds = item['time_bounds']
                prov_text = item['prov_text']
                hint = hints[idx]
                fact_emb = fact_text_embeddings[idx] if idx < len(fact_text_embeddings) else []

                # Link subject entity
                subj_emb = mention_embeddings.get(ext.fact.subject)
                subject_entity = await self.entity_linker.resolve(
                    uow,
                    job.user_id,
                    ext.fact.subject,
                    entity_type=ext.fact.subject_type or EntityType.THING,
                    precomputed_embedding=subj_emb,
                )

                object_entity_id: EntityId | None = None
                object_value: str | None = None
                object_norm: str = ''

                if pred.object_kind == ObjectKind.ENTITY:
                    obj_emb = mention_embeddings.get(ext.fact.object)
                    object_entity = await self.entity_linker.resolve(
                        uow,
                        job.user_id,
                        ext.fact.object,
                        entity_type=ext.fact.object_type or EntityType.THING,
                        precomputed_embedding=obj_emb,
                    )
                    object_entity_id = object_entity.id
                    object_norm = object_entity.canonical_name
                    if self.canonicalizer is not None:
                        final_fact_text = self.canonicalizer.render_canonical_text(
                            pred,
                            subject=subject_entity,
                            object=object_entity,
                        )
                    else:
                        s_name = subject_entity.canonical_name
                        o_name = object_entity.canonical_name
                        final_fact_text = f'{s_name} {pred.name} {o_name}'
                else:
                    object_value = (
                        canon_obj.value if canon_obj.value is not None else ext.fact.object.strip()
                    )
                    object_norm = object_value.strip().lower()
                    if self.canonicalizer is not None:
                        final_fact_text = self.canonicalizer.render_canonical_text(
                            pred,
                            subject=subject_entity,
                            object=object_value,
                        )
                    else:
                        s_name = subject_entity.canonical_name
                        final_fact_text = f'{s_name} {pred.name} {object_value}'

                # Event hash calculation
                event_hash: str | None = None
                if pred.cardinality == Cardinality.EVENT:
                    to_iso = time_bounds.valid_to.isoformat() if time_bounds.valid_to else ''
                    hash_str = (
                        f'{job.user_id}|{subject_entity.id}|{pred.name}|{object_norm}|'
                        f'{time_bounds.valid_from.isoformat()}|{to_iso}'
                    )
                    event_hash = hashlib.sha256(hash_str.encode('utf-8')).hexdigest()

                effective_conf = min(
                    1.0,
                    max(0.0, ext.fact.confidence * time_bounds.confidence_multiplier),
                )

                candidate = Candidate(
                    user_id=job.user_id,
                    subject_id=subject_entity.id,
                    predicate=pred.name,
                    cardinality=pred.cardinality,
                    object_entity_id=object_entity_id,
                    object_value=object_value,
                    object_norm=object_norm,
                    fact_text=final_fact_text,
                    valid_from=time_bounds.valid_from,
                    valid_to=time_bounds.valid_to,
                    time_precision=time_bounds.time_precision,
                    valid_from_explicit=time_bounds.valid_from_explicit,
                    confidence=effective_conf,
                    is_retraction=ext.fact.is_retraction,
                    source_id=ext.source.id,
                    span=(
                        (ext.span_start, ext.span_end)
                        if ext.span_start is not None and ext.span_end is not None
                        else None
                    ),
                    source_ts=ext.source.source_ts,
                    event_hash=event_hash,
                    embedding=fact_emb,
                    hint=hint,
                )

                timeline_key = (subject_entity.id, pred.name)
                if timeline_key not in timeline_cache:
                    if hasattr(uow.facts, 'find_for_resolution'):
                        loaded = await uow.facts.find_for_resolution(
                            job.user_id, subject_entity.id, pred.name
                        )
                    else:
                        loaded = await uow.facts.load_timeline(
                            job.user_id, subject_entity.id, pred.name, lock=True
                        )
                    timeline_cache[timeline_key] = list(loaded)

                current_timeline = timeline_cache[timeline_key]

                resolution_ctx = ResolutionContext(
                    now=self.clock.now(),
                )

                plan = self.conflict_resolver.resolve(
                    candidate,
                    current_timeline,
                    resolution_ctx,
                    hint=candidate.hint,
                )

                exec_ids = await self.plan_executor.execute(
                    uow,
                    plan,
                    candidate=candidate,
                    source_id=ext.source.id,
                    user_id=job.user_id,
                    now=self.clock.now(),
                    job_id=job.id,
                    span_start=ext.span_start,
                    span_end=ext.span_end,
                    origin=FactOrigin.EXTRACTED,
                    model_id=model_id,
                )

                for fid in exec_ids:
                    if fid not in touched_fact_ids:
                        touched_fact_ids.append(fid)

                # Update candidate list in-memory for subsequent facts touching (subj, pred)
                _apply_plan_to_in_memory_timeline(
                    current_timeline,
                    plan,
                    candidate,
                    exec_ids,
                    self.clock.now(),
                )
                timeline_cache[timeline_key] = current_timeline

            if worker_id is not None and hasattr(uow.jobs, 'complete'):
                completed = await uow.jobs.complete(job.id, worker_id, self.clock.now())
                if not completed:
                    raise LeaseLost(f'Lease lost for job {job.id}')

        return touched_fact_ids


IngestionService = IngestService
