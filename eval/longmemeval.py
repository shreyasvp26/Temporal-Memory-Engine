"""LongMemEval dataset loader, chronological session replay harness, QA judge, and report generator.

Governed strictly by docs/LLD.md Section 13.2–13.6 and Section 19 [FROZEN].

Provides:
- EvalTurn, EvalSession, EvalQuestion: Immutable domain entities for LongMemEval benchmarks.
- EvalItemResult, CategoryMetricSummary, EvalReport: Outcome and metric aggregation containers.
- load_dataset: Parses JSON and JSONL datasets into validated EvalQuestion sequences.
- strip_citation_labels: Cleans square-bracket citation labels ([F1], [S2]) before judging.
- LongMemEvalJudge: QA judge with category-specific prompts (knowledge-update, temporal-reasoning,
  abstention, single/multi-session) and deterministic heuristic fallback for tests.
- LongMemEvalRunner: Replays sessions chronologically with FixedClock anchored to session.timestamp,
  synchronously executes IngestService.run_extract_job, evaluates via ChatService.respond
  (or comparative baselines), scores with QA judge + token_f1, and computes paired bootstrap CIs.
- render_markdown_report: Renders publication-ready Markdown benchmark tables.
"""

import json
import logging
import re
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import uuid5

from memory_engine.domain.clock import FixedClock
from memory_engine.domain.enums import Role
from memory_engine.domain.errors import InvariantViolation
from memory_engine.domain.ids import UserId
from memory_engine.ingest.service import IngestService
from memory_engine.ports.embed import Embedder
from memory_engine.ports.llm import ChatMessage, LLMGateway, LLMRequest
from memory_engine.ports.uow import UnitOfWorkFactory
from memory_engine.retrieve.chat_service import ChatRequest, ChatService

from eval.baselines import (
    EVAL_NS,
    EvalCase,
    EvalSession,
    EvalTurn,
    FullBitemporalBaseline,
    JudgeResult,
    MemorySystem,
    Reader,
    SimpleReader,
    create_baseline,
)
from eval.metrics import (
    LatencyStats,
    compute_latency_stats,
    exact_match,
    is_abstention_text,
    normalize_answer,
    paired_bootstrap_ci,
    percentile_bootstrap_ci,
    token_f1,
)

__all__ = [
    'CategoryMetricSummary',
    'EvalItemResult',
    'EvalQuestion',
    'EvalReport',
    'EvalSession',
    'EvalTurn',
    'LongMemEvalJudge',
    'LongMemEvalRunner',
    'load_dataset',
    'normalize_category',
    'render_markdown_report',
    'strip_citation_labels',
]

_logger = logging.getLogger(__name__)

_CITATION_PATTERN = re.compile(r'\[(?:F|S|T|P)\d+(?:,\s*[FSTP]\d+)*\]', re.IGNORECASE)

LONGMEMEVAL_JUDGE_SYSTEM_PROMPT = """You are an expert impartial judge evaluating an \
AI assistant's memory retrieval answer against the ground truth answer.

Instructions per question category:
1. Standard / Multi-session: Compare the predicted response with the ground truth. \
The answer is CORRECT if it contains the factual answer asked for.
2. Knowledge Update: The user's information changed over time. The prediction is CORRECT \
only if it reflects the most updated answer, not the outdated previous state.
3. Temporal Reasoning: For date/time or duration questions, verify dates and counts. \
Tolerate minor off-by-one errors in day, week, or month calculations.
4. Abstention: The question asks about something never mentioned in the history. \
The prediction is CORRECT only if it states that the information is not available, \
unknown, or cannot be answered. It is INCORRECT if it invents an answer.

Your final output must start with:
DECISION: CORRECT
or
DECISION: INCORRECT
Followed by a brief explanation."""


def strip_citation_labels(text: str) -> str:
    """Strip citation labels like [F1], [S2], [T3], [P1] from answers before judging."""
    cleaned = _CITATION_PATTERN.sub('', text)
    return re.sub(r'[ \t]+', ' ', cleaned).strip()


def normalize_category(cat: str) -> str:
    """Normalize question category name to canonical hyphenated lowercase string."""
    c = cat.strip().lower().replace('_', '-')
    if c in ('knowledge-update', 'knowledge-updates'):
        return 'knowledge-update'
    if c in ('temporal-reasoning', 'temporal'):
        return 'temporal-reasoning'
    if c in ('abstention', 'abstain'):
        return 'abstention'
    if c in ('single-session-user', 'single-session', 'single-user'):
        return 'single-session-user'
    if c in ('single-session-assistant', 'single-assistant'):
        return 'single-session-assistant'
    if c in ('multi-session', 'multi-session-reasoning'):
        return 'multi-session'
    return c


@dataclass(frozen=True, slots=True)
class EvalQuestion:
    """Benchmark evaluation question per LongMemEval specification."""

    question_id: str
    question_type: str
    question: str
    gold_answer: str
    question_date: datetime
    sessions: tuple[EvalSession, ...]
    evidence_session_ids: frozenset[str] = frozenset()
    is_abstention: bool = False

    def __init__(
        self,
        question_id: str,
        question_type: str,
        question: str,
        gold_answer: str,
        question_date: datetime | None = None,
        sessions: Sequence[EvalSession] = (),
        evidence_session_ids: Iterable[str] | None = None,
        is_abstention: bool | None = None,
        *,
        id: str | None = None,
        case_id: str | None = None,
        qtype: str | None = None,
        category: str | None = None,
        gold: str | None = None,
        question_ts: datetime | None = None,
        timestamp: datetime | None = None,
    ) -> None:
        effective_id = question_id or id or case_id or ''
        effective_type = question_type or qtype or category or 'general'
        effective_gold = gold_answer or gold or ''
        effective_dt = question_date or question_ts or timestamp
        if effective_dt is None:
            raise InvariantViolation('EvalQuestion requires question_date or question_ts')
        if effective_dt.tzinfo is None:
            effective_dt = effective_dt.replace(tzinfo=UTC)

        effective_type = normalize_category(effective_type)
        if is_abstention is None:
            is_abs = effective_type == 'abstention' or effective_id.endswith(
                ('_abs', '_abstention')
            )
        else:
            is_abs = is_abstention

        object.__setattr__(self, 'question_id', str(effective_id))
        object.__setattr__(self, 'question_type', effective_type)
        object.__setattr__(self, 'question', str(question))
        object.__setattr__(self, 'gold_answer', str(effective_gold))
        object.__setattr__(self, 'question_date', effective_dt)
        object.__setattr__(self, 'sessions', tuple(sessions))
        object.__setattr__(
            self,
            'evidence_session_ids',
            frozenset(evidence_session_ids) if evidence_session_ids is not None else frozenset(),
        )
        object.__setattr__(self, 'is_abstention', bool(is_abs))

    @property
    def id(self) -> str:
        return self.question_id

    @property
    def case_id(self) -> str:
        return self.question_id

    @property
    def qtype(self) -> str:
        return self.question_type

    @property
    def category(self) -> str:
        return self.question_type

    @property
    def gold(self) -> str:
        return self.gold_answer

    @property
    def question_ts(self) -> datetime:
        return self.question_date

    @property
    def timestamp(self) -> datetime:
        return self.question_date

    def to_eval_case(self) -> EvalCase:
        """Convert to baseline EvalCase domain entity."""
        return EvalCase(
            case_id=self.question_id,
            qtype=self.question_type,
            question=self.question,
            gold=self.gold_answer,
            question_ts=self.question_date,
            sessions=self.sessions,
            evidence_session_ids=self.evidence_session_ids,
            is_abstention=self.is_abstention,
        )


@dataclass(frozen=True, slots=True)
class EvalItemResult:
    """Outcome of evaluating an individual question against system and baseline."""

    question_id: str
    question_type: str
    question: str
    gold_answer: str
    system_answer: str
    judge_correct: bool
    token_f1: float
    exact_match: bool
    latency_ms: float
    is_abstention: bool = False
    baseline_answer: str | None = None
    baseline_judge_correct: bool | None = None
    baseline_token_f1: float | None = None
    baseline_exact_match: bool | None = None
    baseline_latency_ms: float | None = None
    error: str | None = None
    extras: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class CategoryMetricSummary:
    """Aggregated metrics and bootstrap confidence intervals for a single category."""

    category: str
    count: int
    accuracy: float
    token_f1: float
    exact_match: float
    ci_low: float
    ci_high: float
    baseline_accuracy: float | None = None
    baseline_token_f1: float | None = None
    baseline_exact_match: float | None = None
    paired_ci_diff: float | None = None
    paired_ci_low: float | None = None
    paired_ci_high: float | None = None


@dataclass(frozen=True, slots=True)
class EvalReport:
    """Complete LongMemEval benchmark report across categories and ablations."""

    benchmark_name: str
    system_name: str
    baseline_name: str | None
    total_questions: int
    overall_accuracy: float
    overall_token_f1: float
    overall_exact_match: float
    abstention_accuracy: float
    category_metrics: Mapping[str, CategoryMetricSummary]
    paired_ci_accuracy: tuple[float, float, float] | None = None
    paired_ci_f1: tuple[float, float, float] | None = None
    latency_stats: LatencyStats | None = None
    baseline_latency_stats: LatencyStats | None = None
    item_results: tuple[EvalItemResult, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert report to JSON-serializable dictionary."""
        cats: dict[str, Any] = {}
        for k, v in self.category_metrics.items():
            cats[k] = {
                'category': v.category,
                'count': v.count,
                'accuracy': v.accuracy,
                'token_f1': v.token_f1,
                'exact_match': v.exact_match,
                'ci_low': v.ci_low,
                'ci_high': v.ci_high,
                'baseline_accuracy': v.baseline_accuracy,
                'baseline_token_f1': v.baseline_token_f1,
                'baseline_exact_match': v.baseline_exact_match,
                'paired_ci_diff': v.paired_ci_diff,
                'paired_ci_low': v.paired_ci_low,
                'paired_ci_high': v.paired_ci_high,
            }

        items: list[dict[str, Any]] = []
        for r in self.item_results:
            items.append(
                {
                    'question_id': r.question_id,
                    'question_type': r.question_type,
                    'question': r.question,
                    'gold_answer': r.gold_answer,
                    'system_answer': r.system_answer,
                    'judge_correct': r.judge_correct,
                    'token_f1': r.token_f1,
                    'exact_match': r.exact_match,
                    'latency_ms': r.latency_ms,
                    'is_abstention': r.is_abstention,
                    'baseline_answer': r.baseline_answer,
                    'baseline_judge_correct': r.baseline_judge_correct,
                    'baseline_token_f1': r.baseline_token_f1,
                    'baseline_exact_match': r.baseline_exact_match,
                    'baseline_latency_ms': r.baseline_latency_ms,
                    'error': r.error,
                }
            )

        lat = (
            {
                'p50': self.latency_stats.p50,
                'p95': self.latency_stats.p95,
                'p99': self.latency_stats.p99,
                'mean': self.latency_stats.mean,
            }
            if self.latency_stats
            else None
        )

        base_lat = (
            {
                'p50': self.baseline_latency_stats.p50,
                'p95': self.baseline_latency_stats.p95,
                'p99': self.baseline_latency_stats.p99,
                'mean': self.baseline_latency_stats.mean,
            }
            if self.baseline_latency_stats
            else None
        )

        return {
            'benchmark_name': self.benchmark_name,
            'system_name': self.system_name,
            'baseline_name': self.baseline_name,
            'total_questions': self.total_questions,
            'overall_accuracy': self.overall_accuracy,
            'overall_token_f1': self.overall_token_f1,
            'overall_exact_match': self.overall_exact_match,
            'abstention_accuracy': self.abstention_accuracy,
            'category_metrics': cats,
            'paired_ci_accuracy': list(self.paired_ci_accuracy)
            if self.paired_ci_accuracy
            else None,
            'paired_ci_f1': list(self.paired_ci_f1) if self.paired_ci_f1 else None,
            'latency_stats': lat,
            'baseline_latency_stats': base_lat,
            'item_results': items,
            'metadata': dict(self.metadata),
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert report to formatted JSON string."""
        return json.dumps(self.to_dict(), indent=indent)


def load_dataset(path: str | Path) -> list[EvalQuestion]:
    """Load LongMemEval questions from a JSON or JSONL file."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f'Dataset file not found: {p}')

    raw_text = p.read_text(encoding='utf-8').strip()
    if not raw_text:
        return []

    items: list[dict[str, Any]] = []
    if p.suffix.lower() == '.jsonl':
        for line in raw_text.splitlines():
            line_str = line.strip()
            if line_str:
                items.append(json.loads(line_str))
    else:
        try:
            parsed = json.loads(raw_text)
            if isinstance(parsed, list):
                items = parsed
            elif isinstance(parsed, dict):
                dict_list = parsed.get('questions') or parsed.get('data') or parsed.get('cases')
                items = dict_list if isinstance(dict_list, list) else []
            else:
                items = []
        except json.JSONDecodeError:
            for line in raw_text.splitlines():
                line_str = line.strip()
                if line_str:
                    items.append(json.loads(line_str))

    questions: list[EvalQuestion] = []
    for item in items:
        qid = str(
            item.get('question_id')
            or item.get('id')
            or item.get('case_id')
            or f'q_{len(questions)}'
        )
        qtype = str(
            item.get('question_type') or item.get('qtype') or item.get('category') or 'general'
        )
        question_text = str(item.get('question') or item.get('prompt') or '')
        gold = str(item.get('gold_answer') or item.get('answer') or item.get('gold') or '')

        raw_qdate = item.get('question_date') or item.get('question_ts') or item.get('date')
        if isinstance(raw_qdate, datetime):
            qdate = raw_qdate if raw_qdate.tzinfo is not None else raw_qdate.replace(tzinfo=UTC)
        elif raw_qdate:
            qdate = datetime.fromisoformat(str(raw_qdate).replace('Z', '+00:00'))
            if qdate.tzinfo is None:
                qdate = qdate.replace(tzinfo=UTC)
        else:
            qdate = datetime(2024, 1, 1, tzinfo=UTC)

        raw_sessions = item.get('sessions') or item.get('haystack_sessions') or ()
        eval_sessions: list[EvalSession] = []
        for s_idx, sess in enumerate(raw_sessions):
            sid = str(sess.get('session_id') or sess.get('id') or f'sess_{s_idx}')
            raw_sdate = sess.get('date') or sess.get('timestamp') or sess.get('ts')
            if isinstance(raw_sdate, datetime):
                sdate = raw_sdate if raw_sdate.tzinfo is not None else raw_sdate.replace(tzinfo=UTC)
            elif raw_sdate:
                sdate = datetime.fromisoformat(str(raw_sdate).replace('Z', '+00:00'))
                if sdate.tzinfo is None:
                    sdate = sdate.replace(tzinfo=UTC)
            else:
                sdate = qdate - timedelta(days=len(raw_sessions) - s_idx)

            raw_turns = sess.get('turns') or ()
            turns: list[EvalTurn] = []
            for t in raw_turns:
                turns.append(
                    EvalTurn(
                        role=str(t.get('role', 'user')),
                        content=str(t.get('content') or t.get('text') or ''),
                        has_answer=bool(t.get('has_answer', False)),
                    )
                )
            eval_sessions.append(EvalSession(session_id=sid, ts=sdate, turns=tuple(turns)))

        ev_ids = item.get('evidence_session_ids') or item.get('answer_session_ids')
        ev_set = frozenset(str(x) for x in ev_ids) if isinstance(ev_ids, Iterable) else frozenset()
        is_abs = item.get('is_abstention')

        questions.append(
            EvalQuestion(
                question_id=qid,
                question_type=qtype,
                question=question_text,
                gold_answer=gold,
                question_date=qdate,
                sessions=eval_sessions,
                evidence_session_ids=ev_set,
                is_abstention=is_abs,
            )
        )

    return questions


class LongMemEvalJudge:
    """Judge evaluating predictions against ground truth per category criteria."""

    def __init__(
        self,
        llm: LLMGateway | None = None,
        model_id: str | None = None,
    ) -> None:
        self.llm = llm
        self.model_id = model_id or 'judge-model'

    async def judge(
        self,
        question: EvalQuestion,
        predicted_answer: str,
    ) -> JudgeResult:
        """Evaluate predicted answer using QA LLM judge prompt or fallback heuristics."""
        cleaned_answer = strip_citation_labels(predicted_answer)

        if self.llm is not None:
            user_prompt = (
                f'Question Type: {question.question_type}\n'
                f'Question: {question.question}\n'
                f'Ground Truth: {question.gold_answer}\n'
                f'Predicted Answer: {cleaned_answer}\n\n'
                'Judge the predicted answer:'
            )
            req = LLMRequest(
                purpose='judge',
                tier='fast',
                messages=[
                    ChatMessage(role='system', content=LONGMEMEVAL_JUDGE_SYSTEM_PROMPT),
                    ChatMessage(role='user', content=user_prompt),
                ],
                prompt_version='longmemeval_v1',
                temperature=0.0,
                max_tokens=256,
            )
            try:
                resp = await self.llm.complete_text(req)
                raw_output = resp.text.strip()
                upper = raw_output.upper()
                is_correct = 'DECISION: CORRECT' in upper or (
                    'CORRECT' in upper and 'INCORRECT' not in upper
                )
                return JudgeResult(correct=is_correct, raw=raw_output, judge_model=self.model_id)
            except Exception as err:
                _logger.warning('LLM judge failed (%s); falling back to heuristic judge', err)

        return self._heuristic_judge(question, cleaned_answer)

    def _heuristic_judge(
        self,
        question: EvalQuestion,
        cleaned_answer: str,
    ) -> JudgeResult:
        """Deterministic heuristic judge for zero-network testing."""
        if question.is_abstention or question.question_type == 'abstention':
            is_correct = is_abstention_text(cleaned_answer)
            return JudgeResult(
                correct=is_correct,
                raw=f'Heuristic abstention judge: {is_correct}',
                judge_model='heuristic',
            )

        gold_norm = normalize_answer(question.gold_answer)
        pred_norm = normalize_answer(cleaned_answer)

        if exact_match(cleaned_answer, question.gold_answer):
            return JudgeResult(correct=True, raw='Exact match', judge_model='heuristic')

        if question.question_type == 'knowledge-update':
            is_correct = (
                gold_norm in pred_norm or token_f1(cleaned_answer, question.gold_answer) >= 0.5
            )
            return JudgeResult(
                correct=is_correct,
                raw=f'Knowledge-update heuristic: {is_correct}',
                judge_model='heuristic',
            )

        if question.question_type == 'temporal-reasoning':
            is_correct = (
                gold_norm in pred_norm or token_f1(cleaned_answer, question.gold_answer) >= 0.4
            )
            return JudgeResult(
                correct=is_correct,
                raw=f'Temporal-reasoning heuristic: {is_correct}',
                judge_model='heuristic',
            )

        f1 = token_f1(cleaned_answer, question.gold_answer)
        is_correct = f1 >= 0.6 or gold_norm in pred_norm
        return JudgeResult(
            correct=is_correct,
            raw=f'General heuristic F1={f1:.2f}',
            judge_model='heuristic',
        )


class LongMemEvalRunner:
    """LongMemEval execution harness per LLD Section 13.2-13.6 and 19.1-19.5."""

    def __init__(
        self,
        *,
        system_name: str = 'tmem',
        chat_service: ChatService | None = None,
        ingest_service: IngestService | None = None,
        uow_factory: UnitOfWorkFactory | None = None,
        clock: FixedClock | None = None,
        baseline: MemorySystem | str | None = None,
        judge: LongMemEvalJudge | None = None,
        llm: LLMGateway | None = None,
        embedder: Embedder | None = None,
        reader: Reader | None = None,
        seed: int = 42,
        system: MemorySystem | None = None,
    ) -> None:
        self.system_name = system_name
        self.chat_service = chat_service
        self.ingest_service = ingest_service
        self.uow_factory = uow_factory
        self.clock = clock
        self.seed = seed
        self.embedder = embedder
        self.reader = reader or SimpleReader()
        self.judge = judge or LongMemEvalJudge(llm=llm)
        self.system = system
        if self.system is None and self.chat_service is None:
            try:
                self.system = create_baseline(
                    self.system_name,
                    reader=self.reader,
                    uow_factory=self.uow_factory,
                    embedder=self.embedder,
                )
            except Exception as err:
                _logger.debug('Could not auto-instantiate system %s: %s', self.system_name, err)

        if isinstance(baseline, str):
            self.baseline: MemorySystem | None = create_baseline(
                baseline,
                reader=self.reader,
                uow_factory=self.uow_factory,
                embedder=self.embedder,
            )
            self.baseline_name: str | None = baseline
        else:
            self.baseline = baseline
            self.baseline_name = baseline.name if baseline is not None else None

    async def run(
        self,
        dataset: str | Path | Sequence[EvalQuestion],
        limit: int | None = None,
    ) -> EvalReport:
        """Execute evaluation over questions and return comprehensive EvalReport."""
        questions = load_dataset(dataset) if isinstance(dataset, (str, Path)) else list(dataset)

        if limit is not None and limit > 0:
            questions = questions[:limit]

        results: list[EvalItemResult] = []
        for q in questions:
            item_res = await self._evaluate_question(q)
            results.append(item_res)

        return self._build_report(results)

    async def _evaluate_question(self, question: EvalQuestion) -> EvalItemResult:
        user_id = UserId(uuid5(EVAL_NS, f'longmemeval_{question.question_id}'))

        # 1. Evaluate primary system
        sys_answer = ''
        sys_latency_ms = 0.0
        err_msg: str | None = None

        try:
            if (
                self.chat_service is not None
                and self.ingest_service is not None
                and self.uow_factory is not None
            ):
                # Replay each session chronologically using FixedClock anchored to session.timestamp
                sorted_sessions = sorted(question.sessions, key=lambda s: s.ts)
                for session in sorted_sessions:
                    if self.clock is not None:
                        self.clock.set(session.ts)

                    for turn_idx, turn in enumerate(session.turns):
                        turn_ts = session.ts + timedelta(seconds=turn_idx)
                        if self.clock is not None:
                            self.clock.set(turn_ts)

                        role = Role.USER if turn.role.lower() == 'user' else Role.ASSISTANT
                        async with self.uow_factory.for_user(user_id) as uow:
                            await self.ingest_service.ensure_user_bootstrap(uow, user_id)
                            res = await self.ingest_service.enqueue_turn(
                                uow,
                                user_id=user_id,
                                session_id=session.session_id,
                                role=role,
                                text=turn.content,
                                source_ts=turn_ts,
                            )

                        # Synchronously execute extraction jobs so all facts are committed
                        if res.job_id is not None:
                            async with self.uow_factory.for_user(user_id, readonly=True) as uow:
                                job = await uow.jobs.get(user_id, res.job_id)
                            if job is not None:
                                await self.ingest_service.run_extract_job(job)

                # Set clock to question timestamp
                if self.clock is not None:
                    self.clock.set(question.question_date)

                # Evaluate question via ChatService.respond
                t0 = time.perf_counter()
                chat_req = ChatRequest(
                    user_id=user_id,
                    message=question.question,
                    client_ts=question.question_date,
                )
                resp = await self.chat_service.respond(user_id, chat_req)
                sys_latency_ms = (time.perf_counter() - t0) * 1000.0
                sys_answer = resp.content

            elif self.system is not None:
                eval_case = question.to_eval_case()
                await self.system.prepare(eval_case)
                ans = await self.system.answer(eval_case)
                sys_latency_ms = ans.latency_ms
                sys_answer = ans.text
                await self.system.cleanup(eval_case)

            elif self.uow_factory is not None:
                temp_sys = FullBitemporalBaseline(
                    reader=self.reader,
                    uow_factory=self.uow_factory,
                    embedder=self.embedder,
                )
                eval_case = question.to_eval_case()
                await temp_sys.prepare(eval_case)
                ans = await temp_sys.answer(eval_case)
                sys_latency_ms = ans.latency_ms
                sys_answer = ans.text
                await temp_sys.cleanup(eval_case)
            else:
                raise InvariantViolation('No system or services configured on LongMemEvalRunner')

        except Exception as exc:
            _logger.exception('System evaluation failed for question %s', question.question_id)
            err_msg = str(exc)
            sys_answer = f'ERROR: {exc}'

        # 2. Evaluate comparative baseline if configured
        base_answer: str | None = None
        base_latency_ms: float | None = None
        if self.baseline is not None:
            try:
                eval_case = question.to_eval_case()
                await self.baseline.prepare(eval_case)
                b_ans = await self.baseline.answer(eval_case)
                base_answer = b_ans.text
                base_latency_ms = b_ans.latency_ms
                await self.baseline.cleanup(eval_case)
            except Exception as exc:
                _logger.exception(
                    'Baseline evaluation failed for question %s', question.question_id
                )
                base_answer = f'ERROR: {exc}'

        # 3. Clean citation labels and score
        clean_sys = strip_citation_labels(sys_answer)
        sys_judge = await self.judge.judge(question, clean_sys)
        sys_f1 = token_f1(clean_sys, question.gold_answer)
        sys_em = exact_match(clean_sys, question.gold_answer)

        base_judge_correct: bool | None = None
        base_f1: float | None = None
        base_em: bool | None = None

        if base_answer is not None:
            clean_base = strip_citation_labels(base_answer)
            b_judge = await self.judge.judge(question, clean_base)
            base_judge_correct = b_judge.correct
            base_f1 = token_f1(clean_base, question.gold_answer)
            base_em = exact_match(clean_base, question.gold_answer)

        return EvalItemResult(
            question_id=question.question_id,
            question_type=question.question_type,
            question=question.question,
            gold_answer=question.gold_answer,
            system_answer=sys_answer,
            judge_correct=sys_judge.correct,
            token_f1=sys_f1,
            exact_match=sys_em,
            latency_ms=sys_latency_ms,
            is_abstention=question.is_abstention,
            baseline_answer=base_answer,
            baseline_judge_correct=base_judge_correct,
            baseline_token_f1=base_f1,
            baseline_exact_match=base_em,
            baseline_latency_ms=base_latency_ms,
            error=err_msg,
        )

    def _build_report(self, results: list[EvalItemResult]) -> EvalReport:
        n = len(results)
        if n == 0:
            return EvalReport(
                benchmark_name='LongMemEval',
                system_name=self.system_name,
                baseline_name=self.baseline_name,
                total_questions=0,
                overall_accuracy=0.0,
                overall_token_f1=0.0,
                overall_exact_match=0.0,
                abstention_accuracy=0.0,
                category_metrics={},
            )

        overall_acc = sum(1.0 for r in results if r.judge_correct) / n
        overall_f1 = sum(r.token_f1 for r in results) / n
        overall_em = sum(1.0 for r in results if r.exact_match) / n

        abs_cases = [r for r in results if r.is_abstention or r.question_type == 'abstention']
        abs_acc = (
            sum(1.0 for r in abs_cases if r.judge_correct) / len(abs_cases) if abs_cases else 1.0
        )

        has_baseline = any(r.baseline_judge_correct is not None for r in results)
        paired_ci_acc: tuple[float, float, float] | None = None
        paired_ci_f1: tuple[float, float, float] | None = None

        if has_baseline:
            sys_scores = [1.0 if r.judge_correct else 0.0 for r in results]
            base_scores = [1.0 if r.baseline_judge_correct else 0.0 for r in results]
            paired_ci_acc = paired_bootstrap_ci(sys_scores, base_scores, seed=self.seed)

            sys_f1s = [r.token_f1 for r in results]
            base_f1s = [r.baseline_token_f1 or 0.0 for r in results]
            paired_ci_f1 = paired_bootstrap_ci(sys_f1s, base_f1s, seed=self.seed)

        by_cat: dict[str, list[EvalItemResult]] = {}
        for r in results:
            cat = normalize_category(r.question_type)
            by_cat.setdefault(cat, []).append(r)

        cat_metrics: dict[str, CategoryMetricSummary] = {}
        for cat, items in by_cat.items():
            k = len(items)
            c_acc = sum(1.0 for it in items if it.judge_correct) / k
            c_f1 = sum(it.token_f1 for it in items) / k
            c_em = sum(1.0 for it in items if it.exact_match) / k

            acc_scores = [1.0 if it.judge_correct else 0.0 for it in items]
            _, ci_low, ci_high = percentile_bootstrap_ci(acc_scores, seed=self.seed)

            b_acc: float | None = None
            b_f1: float | None = None
            b_em: float | None = None
            p_diff: float | None = None
            p_low: float | None = None
            p_high: float | None = None

            if has_baseline:
                b_scores = [1.0 if it.baseline_judge_correct else 0.0 for it in items]
                b_acc = sum(b_scores) / k
                b_f1 = sum(it.baseline_token_f1 or 0.0 for it in items) / k
                b_em = sum(1.0 for it in items if it.baseline_exact_match) / k
                p_diff, p_low, p_high = paired_bootstrap_ci(acc_scores, b_scores, seed=self.seed)

            cat_metrics[cat] = CategoryMetricSummary(
                category=cat,
                count=k,
                accuracy=c_acc,
                token_f1=c_f1,
                exact_match=c_em,
                ci_low=ci_low,
                ci_high=ci_high,
                baseline_accuracy=b_acc,
                baseline_token_f1=b_f1,
                baseline_exact_match=b_em,
                paired_ci_diff=p_diff,
                paired_ci_low=p_low,
                paired_ci_high=p_high,
            )

        lat_stats = compute_latency_stats([r.latency_ms for r in results])
        base_lat_stats = (
            compute_latency_stats(
                [r.baseline_latency_ms for r in results if r.baseline_latency_ms is not None]
            )
            if has_baseline
            else None
        )

        return EvalReport(
            benchmark_name='LongMemEval',
            system_name=self.system_name,
            baseline_name=self.baseline_name,
            total_questions=n,
            overall_accuracy=overall_acc,
            overall_token_f1=overall_f1,
            overall_exact_match=overall_em,
            abstention_accuracy=abs_acc,
            category_metrics=cat_metrics,
            paired_ci_accuracy=paired_ci_acc,
            paired_ci_f1=paired_ci_f1,
            latency_stats=lat_stats,
            baseline_latency_stats=base_lat_stats,
            item_results=tuple(results),
        )


def render_markdown_report(report: EvalReport) -> str:
    """Generate publication-ready Markdown benchmark table and report."""
    lines: list[str] = [
        '# LongMemEval Benchmark Report',
        '',
        f'- **System:** `{report.system_name}`',
        f'- **Baseline:** `{report.baseline_name or "None"}`',
        f'- **Total Questions:** `{report.total_questions}`',
        f'- **Overall Accuracy:** `{report.overall_accuracy * 100:.1f}%`',
        f'- **Overall Token F1:** `{report.overall_token_f1 * 100:.1f}%`',
        f'- **Abstention Accuracy:** `{report.abstention_accuracy * 100:.1f}%`',
    ]

    if report.paired_ci_accuracy is not None:
        diff, low, high = report.paired_ci_accuracy
        sign = '+' if diff >= 0 else ''
        lines.append(
            f'- **Paired Accuracy 95% CI (A - B):** '
            f'`{sign}{diff * 100:.1f}% [{low * 100:.1f}%, {high * 100:.1f}%]`'
        )

    if report.paired_ci_f1 is not None:
        diff_f1, low_f1, high_f1 = report.paired_ci_f1
        sign_f1 = '+' if diff_f1 >= 0 else ''
        lines.append(
            f'- **Paired Token F1 95% CI (A - B):** '
            f'`{sign_f1}{diff_f1 * 100:.1f}% [{low_f1 * 100:.1f}%, {high_f1 * 100:.1f}%]`'
        )

    lines.extend(['', '## Category Breakdown', ''])

    has_baseline = report.baseline_name is not None and any(
        m.baseline_accuracy is not None for m in report.category_metrics.values()
    )

    if has_baseline:
        lines.append(
            '| Category | N | System Acc | 95% CI | Baseline Acc | '
            'Δ (Paired 95% CI) | System F1 | Baseline F1 |'
        )
        lines.append('| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |')

        for cat, m in sorted(report.category_metrics.items()):
            b_acc_str = (
                f'{m.baseline_accuracy * 100:.1f}%' if m.baseline_accuracy is not None else '-'
            )
            b_f1_str = (
                f'{m.baseline_token_f1 * 100:.1f}%' if m.baseline_token_f1 is not None else '-'
            )

            if (
                m.paired_ci_diff is not None
                and m.paired_ci_low is not None
                and m.paired_ci_high is not None
            ):
                p_sign = '+' if m.paired_ci_diff >= 0 else ''
                delta_str = (
                    f'{p_sign}{m.paired_ci_diff * 100:.1f}% '
                    f'[{m.paired_ci_low * 100:.1f}%, {m.paired_ci_high * 100:.1f}%]'
                )
            else:
                delta_str = '-'

            lines.append(
                f'| {cat} | {m.count} | {m.accuracy * 100:.1f}% | '
                f'[{m.ci_low * 100:.1f}%, {m.ci_high * 100:.1f}%] | '
                f'{b_acc_str} | {delta_str} | {m.token_f1 * 100:.1f}% | {b_f1_str} |'
            )

        b_overall_acc = '-'
        b_overall_f1 = '-'
        if report.item_results:
            b_accs = [
                1.0
                for r in report.item_results
                if r.baseline_judge_correct is not None and r.baseline_judge_correct
            ]
            total_b = sum(1 for r in report.item_results if r.baseline_judge_correct is not None)
            if total_b > 0:
                b_overall_acc = f'{(len(b_accs) / total_b) * 100:.1f}%'

            b_f1s = [
                r.baseline_token_f1 for r in report.item_results if r.baseline_token_f1 is not None
            ]
            if b_f1s:
                b_overall_f1 = f'{(sum(b_f1s) / len(b_f1s)) * 100:.1f}%'

        delta_overall_str = '-'
        if report.paired_ci_accuracy is not None:
            p_diff, p_low, p_high = report.paired_ci_accuracy
            p_sign = '+' if p_diff >= 0 else ''
            delta_overall_str = (
                f'{p_sign}{p_diff * 100:.1f}% [{p_low * 100:.1f}%, {p_high * 100:.1f}%]'
            )

        sys_acc_scores = [1.0 if r.judge_correct else 0.0 for r in report.item_results]
        _, o_ci_low, o_ci_high = (
            percentile_bootstrap_ci(sys_acc_scores) if sys_acc_scores else (0.0, 0.0, 0.0)
        )

        lines.append(
            f'| **Overall** | **{report.total_questions}** | '
            f'**{report.overall_accuracy * 100:.1f}%** | '
            f'**[{o_ci_low * 100:.1f}%, {o_ci_high * 100:.1f}%]** | **{b_overall_acc}** | '
            f'**{delta_overall_str}** | **{report.overall_token_f1 * 100:.1f}%** | '
            f'**{b_overall_f1}** |'
        )
    else:
        lines.append('| Category | N | Accuracy | 95% CI | Token F1 | Exact Match |')
        lines.append('| :--- | :---: | :---: | :---: | :---: | :---: |')

        for cat, m in sorted(report.category_metrics.items()):
            lines.append(
                f'| {cat} | {m.count} | {m.accuracy * 100:.1f}% | '
                f'[{m.ci_low * 100:.1f}%, {m.ci_high * 100:.1f}%] | '
                f'{m.token_f1 * 100:.1f}% | {m.exact_match * 100:.1f}% |'
            )

        sys_acc_scores = [1.0 if r.judge_correct else 0.0 for r in report.item_results]
        _, o_ci_low, o_ci_high = (
            percentile_bootstrap_ci(sys_acc_scores) if sys_acc_scores else (0.0, 0.0, 0.0)
        )
        lines.append(
            f'| **Overall** | **{report.total_questions}** | '
            f'**{report.overall_accuracy * 100:.1f}%** | '
            f'**[{o_ci_low * 100:.1f}%, {o_ci_high * 100:.1f}%]** | '
            f'**{report.overall_token_f1 * 100:.1f}%** | '
            f'**{report.overall_exact_match * 100:.1f}%** |'
        )

    if report.latency_stats is not None:
        lines.extend(['', '## Latency Profile', ''])
        if has_baseline and report.baseline_latency_stats is not None:
            lines.append('| Metric | System (ms) | Baseline (ms) |')
            lines.append('| :--- | :---: | :---: |')
            p50_s = f'{report.latency_stats.p50:.1f}'
            p50_b = f'{report.baseline_latency_stats.p50:.1f}'
            p95_s = f'{report.latency_stats.p95:.1f}'
            p95_b = f'{report.baseline_latency_stats.p95:.1f}'
            p99_s = f'{report.latency_stats.p99:.1f}'
            p99_b = f'{report.baseline_latency_stats.p99:.1f}'
            mean_s = f'{report.latency_stats.mean:.1f}'
            mean_b = f'{report.baseline_latency_stats.mean:.1f}'
            lines.append(f'| p50 | {p50_s} | {p50_b} |')
            lines.append(f'| p95 | {p95_s} | {p95_b} |')
            lines.append(f'| p99 | {p99_s} | {p99_b} |')
            lines.append(f'| Mean | {mean_s} | {mean_b} |')
        else:
            lines.append('| Metric | Latency (ms) |')
            lines.append('| :--- | :---: |')
            lines.append(f'| p50 | {report.latency_stats.p50:.1f} |')
            lines.append(f'| p95 | {report.latency_stats.p95:.1f} |')
            lines.append(f'| p99 | {report.latency_stats.p99:.1f} |')
            lines.append(f'| Mean | {report.latency_stats.mean:.1f} |')

    return '\n'.join(lines) + '\n'
