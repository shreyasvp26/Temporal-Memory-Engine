"""Resolution plan executor applying plan operations against UnitOfWork."""

from datetime import UTC, datetime
from uuid import UUID, uuid4

from memory_engine.domain.clock import Clock
from memory_engine.domain.enums import (
    Cardinality,
    FactOrigin,
    TimePrecision,
)
from memory_engine.domain.errors import InvariantViolation
from memory_engine.domain.ids import EntityId, FactId, SourceId, UserId
from memory_engine.domain.models import (
    Candidate,
    FactEvidence,
    FactRelation,
)
from memory_engine.ingest.conflict.plan import (
    AddEvidence,
    AddRelation,
    BumpConfidence,
    CloseFact,
    ExtendValidFrom,
    FactRef,
    InsertFact,
    NewRef,
    ResolutionPlan,
    SetStatus,
)
from memory_engine.ports.uow import UnitOfWork


class PlanExecutor:
    """Executes a ResolutionPlan against UnitOfWork.facts under uow.lock_user."""

    def __init__(
        self,
        candidate: Candidate | None = None,
        *,
        source_id: SourceId | None = None,
        user_id: UserId | None = None,
        clock: Clock | None = None,
        now: datetime | None = None,
        model_id: str = 'default',
        origin: FactOrigin = FactOrigin.EXTRACTED,
        job_id: UUID | None = None,
        span_start: int | None = None,
        span_end: int | None = None,
        lock_user: bool = True,
    ) -> None:
        self.candidate = candidate
        self.source_id = source_id
        self.user_id = user_id
        self.clock = clock
        self._now = now
        self.model_id = model_id
        self.origin = origin
        self.job_id = job_id
        self.span_start = span_start
        self.span_end = span_end
        self.lock_user = lock_user

    async def execute(
        self,
        uow: UnitOfWork,
        plan: ResolutionPlan,
        *,
        candidate: Candidate | None = None,
        source_id: SourceId | None = None,
        user_id: UserId | None = None,
        now: datetime | None = None,
        job_id: UUID | None = None,
        span_start: int | None = None,
        span_end: int | None = None,
        origin: FactOrigin | None = None,
        model_id: str | None = None,
    ) -> list[FactId]:
        """Execute all operations in plan.ops against uow.facts in order."""
        cand = candidate or self.candidate
        uid = user_id or self.user_id or (cand.user_id if cand else UserId(uuid4()))
        src_id = source_id or self.source_id or (cand.source_id if cand else SourceId(uuid4()))
        jid = job_id if job_id is not None else self.job_id
        orig = origin or self.origin
        mid = model_id or self.model_id

        # Determine effective span bounds
        s_start = span_start if span_start is not None else self.span_start
        s_end = span_end if span_end is not None else self.span_end
        if cand and cand.span:
            s_start = cand.span[0] if s_start is None else s_start
            s_end = cand.span[1] if s_end is None else s_end

        # Determine effective current UTC timestamp
        if now is not None:
            exec_now = now
        elif self._now is not None:
            exec_now = self._now
        elif self.clock is not None:
            exec_now = self.clock.now()
        else:
            exec_now = datetime.now(UTC)

        # Acquire user lock when requested and supported
        if self.lock_user and hasattr(uow, 'lock_user') and callable(uow.lock_user):
            await uow.lock_user()

        new_fact_id: FactId | None = None
        touched_fact_ids: list[FactId] = []

        def _record_fact(fid: FactId) -> None:
            if fid not in touched_fact_ids:
                touched_fact_ids.append(fid)

        def _resolve_ref(ref: FactRef) -> FactId:
            if isinstance(ref, NewRef):
                if new_fact_id is None:
                    raise InvariantViolation('NewRef referenced before InsertFact execution')
                return new_fact_id
            return ref

        for op in plan.ops:
            if isinstance(op, CloseFact):
                await uow.facts.close(uid, op.fact_id, op.valid_to, exec_now)
                _record_fact(op.fact_id)
            elif isinstance(op, ExtendValidFrom):
                await uow.facts.extend_valid_from(uid, op.fact_id, op.valid_from)
                _record_fact(op.fact_id)
            elif isinstance(op, SetStatus):
                await uow.facts.set_status(uid, op.fact_id, op.status, exec_now)
                _record_fact(op.fact_id)
            elif isinstance(op, BumpConfidence):
                await uow.facts.bump_confidence(uid, op.fact_id, op.confidence)
                _record_fact(op.fact_id)
            elif isinstance(op, InsertFact):
                if cand is None:
                    # Provide default candidate if none was explicitly configured
                    cand = Candidate(
                        user_id=uid,
                        subject_id=EntityId(uuid4()),
                        predicate='related_to',
                        cardinality=Cardinality.SINGLE,
                        object_entity_id=None,
                        object_value='value',
                        object_norm='value',
                        fact_text='User is related to value',
                        valid_from=op.valid_from,
                        valid_to=op.valid_to,
                        time_precision=TimePrecision.EXACT,
                        valid_from_explicit=True,
                        confidence=1.0,
                        is_retraction=False,
                        source_id=src_id,
                        span=(
                            (s_start, s_end) if s_start is not None and s_end is not None else None
                        ),
                        source_ts=exec_now,
                        event_hash=None,
                        embedding=[],
                    )
                inserted = await uow.facts.insert(
                    cand,
                    valid_from=op.valid_from,
                    valid_to=op.valid_to,
                    status=op.status,
                    origin=orig,
                    model_id=mid,
                    now=exec_now,
                )
                new_fact_id = inserted.id
                _record_fact(inserted.id)
            elif isinstance(op, AddEvidence):
                target_fact_id = _resolve_ref(op.ref)
                ev = FactEvidence(
                    fact_id=target_fact_id,
                    source_id=src_id,
                    span_start=s_start,
                    span_end=s_end,
                )
                await uow.facts.add_evidence(ev, uid)
                _record_fact(target_fact_id)
            elif isinstance(op, AddRelation):
                from_id = _resolve_ref(op.from_ref)
                to_id = _resolve_ref(op.to_ref)
                rel = FactRelation(
                    from_fact_id=from_id,
                    to_fact_id=to_id,
                    relation=op.relation,
                )
                await uow.facts.add_relation(uid, rel, jid)
                _record_fact(from_id)
                _record_fact(to_id)
            else:
                raise InvariantViolation(f'Unknown operation in ResolutionPlan: {type(op)}')

        return touched_fact_ids
