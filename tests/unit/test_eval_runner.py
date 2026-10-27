"""Unit tests for LongMemEval runner, dataset loader, QA judge, CLI, and report generator (M6-02).

Tests:
1. Dataset loading from JSON and JSONL with field normalization and UTC validation.
2. Citation label stripping ([F1], [S2], [T3], [P1]).
3. LongMemEvalJudge: QA prompt handling and heuristic fallbacks across question categories.
4. LongMemEvalRunner chronological session replay with FixedClock, synchronous
   extract job execution, ChatService evaluation, scoring, and paired bootstrap CIs.
5. Markdown report rendering with publication-ready comparative tables and latency profiles.
6. JSON report serialization and dictionary conversion.
7. CLI argument parsing and full command-line execution writing report.md and report.json.
All tests run with zero network sockets.
"""

import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import cast
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from memory_engine.domain.clock import FixedClock
from memory_engine.domain.enums import JobStatus, JobType
from memory_engine.domain.ids import FactId, JobId, SourceId, UserId
from memory_engine.domain.models import Job
from memory_engine.ingest.service import EnqueueResult
from memory_engine.ports.llm import LLMResponse
from memory_engine.ports.uow import UnitOfWork
from memory_engine.retrieve.chat_service import ChatTurnResult

from eval.baselines import (
    EvalCase,
    SystemAnswer,
)
from eval.cli import main, parse_args
from eval.longmemeval import (
    CategoryMetricSummary,
    EvalItemResult,
    EvalQuestion,
    EvalReport,
    EvalSession,
    EvalTurn,
    LongMemEvalJudge,
    LongMemEvalRunner,
    load_dataset,
    normalize_category,
    render_markdown_report,
    strip_citation_labels,
)
from eval.metrics import LatencyStats


def utc(
    year: int,
    month: int = 1,
    day: int = 1,
    hour: int = 0,
    minute: int = 0,
    second: int = 0,
) -> datetime:
    """Helper creating timezone-aware UTC datetime."""
    return datetime(year, month, day, hour, minute, second, tzinfo=UTC)


# ============================================================================
# Synthetic 3-Question LongMemEval Fixture
# ============================================================================


def make_synthetic_3question_dataset() -> list[EvalQuestion]:
    """Create synthetic 3-question dataset covering KU, TR, and abstention."""
    # Question 1: Knowledge Update
    q1 = EvalQuestion(
        question_id='ku_001',
        question_type='knowledge-update',
        question='Where do I currently live?',
        gold_answer='Amsterdam',
        question_date=utc(2024, 3, 1),
        sessions=[
            EvalSession(
                session_id='sess_ku_1',
                ts=utc(2023, 5, 10, 10, 0, 0),
                turns=(
                    EvalTurn(role='user', content='I just moved into my apartment in Berlin.'),
                    EvalTurn(role='assistant', content='Berlin is a great city with rich culture!'),
                ),
            ),
            EvalSession(
                session_id='sess_ku_2',
                ts=utc(2024, 1, 15, 14, 30, 0),
                turns=(
                    EvalTurn(
                        role='user',
                        content='I got a new job offer and relocated to Amsterdam last week.',
                    ),
                    EvalTurn(role='assistant', content='Congratulations on the move to Amsterdam!'),
                ),
            ),
        ],
        evidence_session_ids=frozenset(['sess_ku_2']),
        is_abstention=False,
    )

    # Question 2: Temporal Reasoning
    q2 = EvalQuestion(
        question_id='tr_002',
        question_type='temporal-reasoning',
        question='When was I visiting Paris and how long was the trip?',
        gold_answer='March 5 to March 12 for 7 days',
        question_date=utc(2024, 3, 20),
        sessions=[
            EvalSession(
                session_id='sess_tr_1',
                ts=utc(2024, 2, 1, 9, 0, 0),
                turns=(
                    EvalTurn(
                        role='user',
                        content='I booked a vacation to Paris from March 5 to March 12.',
                    ),
                ),
            ),
            EvalSession(
                session_id='sess_tr_2',
                ts=utc(2024, 3, 15, 11, 0, 0),
                turns=(
                    EvalTurn(
                        role='user',
                        content='I am back from Paris. That 7-day trip was fantastic.',
                    ),
                ),
            ),
        ],
        evidence_session_ids=frozenset(['sess_tr_1', 'sess_tr_2']),
        is_abstention=False,
    )

    # Question 3: Abstention
    q3 = EvalQuestion(
        question_id='abs_003',
        question_type='abstention',
        question="What is my mother's maiden name?",
        gold_answer='The information is not available.',
        question_date=utc(2024, 4, 1),
        sessions=[
            EvalSession(
                session_id='sess_abs_1',
                ts=utc(2024, 3, 10, 16, 0, 0),
                turns=(
                    EvalTurn(
                        role='user', content='I like drinking Japanese green tea every morning.'
                    ),
                ),
            ),
        ],
        evidence_session_ids=frozenset(),
        is_abstention=True,
    )

    return [q1, q2, q3]


# ============================================================================
# Fakes for LongMemEvalRunner testing
# ============================================================================


class TrackingClock(FixedClock):
    """FixedClock that tracks all timestamps it is set to."""

    def __init__(self, fixed_now: datetime) -> None:
        super().__init__(fixed_now)
        self.history: list[datetime] = []

    def set(self, new_now: datetime) -> None:
        self.history.append(new_now)
        super().set(new_now)


class FakeUow:
    """Fake UnitOfWork tracking calls and simulating repositories."""

    def __init__(self) -> None:
        self.jobs = MagicMock()
        self.sources = MagicMock()
        self.profiles = MagicMock()
        self.entities = MagicMock()
        self.facts = MagicMock()
        self.messages = MagicMock()
        self.audit = MagicMock()
        self.search = MagicMock()
        self.locked_users: list[UserId | None] = []

        fake_job = Job(
            id=JobId(uuid4()),
            user_id=UserId(uuid4()),
            type=JobType.EXTRACT,
            payload={'session_id': 'sess_1', 'source_ids': []},
            idempotency_key='test-key',
            coalesce_key=None,
            status=JobStatus.QUEUED,
            attempts=0,
            max_attempts=3,
            deferrals=0,
            run_after=utc(2024, 1, 1),
            worker_id=None,
            lease_expires_at=None,
            last_error=None,
        )
        self.jobs.get = AsyncMock(return_value=fake_job)

    async def lock_user(self, user_id: UserId | None = None) -> None:
        self.locked_users.append(user_id)

    async def __aenter__(self) -> 'FakeUow':
        return self

    async def __aexit__(
        self,
        et: type[BaseException] | None,
        ev: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        pass


class FakeUowFactory:
    """Fake UnitOfWorkFactory."""

    def __init__(self, uow: FakeUow | None = None) -> None:
        self.uow = uow or FakeUow()

    def begin(self, user_id: UserId | None, *, readonly: bool = False) -> UnitOfWork:
        return cast('UnitOfWork', self.uow)

    def for_user(self, user_id: UserId, *, readonly: bool = False) -> UnitOfWork:
        return cast('UnitOfWork', self.uow)

    def system(self, *, readonly: bool = False) -> UnitOfWork:
        return cast('UnitOfWork', self.uow)


class MockMemorySystem:
    """Mock MemorySystem implementing baseline protocol."""

    def __init__(self, name: str, answers: Sequence[str] | None = None) -> None:
        self.name = name
        self._answers = list(answers or [])
        self._idx = 0
        self.prepared_cases: list[str] = []
        self.cleaned_cases: list[str] = []

    async def prepare(self, case: EvalCase) -> None:
        self.prepared_cases.append(case.case_id)

    async def answer(self, case: EvalCase) -> SystemAnswer:
        if self._idx < len(self._answers):
            ans_text = self._answers[self._idx]
            self._idx += 1
        else:
            ans_text = f'Answer from {self.name} for {case.case_id}'

        return SystemAnswer(
            text=ans_text,
            retrieved_session_ids=tuple(case.evidence_session_ids),
            context_tokens=150,
            reader_tokens=30,
            latency_ms=25.0,
            extras={},
        )

    async def cleanup(self, case: EvalCase) -> None:
        self.cleaned_cases.append(case.case_id)


# ============================================================================
# Unit Tests
# ============================================================================


def test_strip_citation_labels() -> None:
    """Verify stripping citation labels like [F1], [S2], [T3], [P4] and collapsing whitespace."""
    raw = (
        'User moved to Amsterdam [F1]. This was mentioned in [S2, S3] '
        'and verified in [T1] and [P4].'
    )
    cleaned = strip_citation_labels(raw)
    assert '[F1]' not in cleaned
    assert '[S2, S3]' not in cleaned
    assert '[T1]' not in cleaned
    assert '[P4]' not in cleaned
    assert cleaned == 'User moved to Amsterdam . This was mentioned in and verified in and .'


def test_normalize_category() -> None:
    """Verify question category normalization to canonical hyphenated lowercase strings."""
    assert normalize_category('knowledge_update') == 'knowledge-update'
    assert normalize_category('knowledge-updates') == 'knowledge-update'
    assert normalize_category('temporal_reasoning') == 'temporal-reasoning'
    assert normalize_category('abstain') == 'abstention'
    assert normalize_category('single_session_user') == 'single-session-user'
    assert normalize_category('multi_session') == 'multi-session'
    assert normalize_category('CUSTOM_TYPE') == 'custom-type'


def test_load_dataset_json_and_jsonl(tmp_path: Path) -> None:
    """Verify loading dataset from JSON array and JSONL lines with timezone handling."""
    data = [
        {
            'question_id': 'q_json_1',
            'question_type': 'knowledge-update',
            'question': 'Where do I live?',
            'answer': 'Seattle',
            'question_date': '2024-03-01T12:00:00Z',
            'sessions': [
                {
                    'session_id': 's1',
                    'date': '2023-01-01T10:00:00Z',
                    'turns': [{'role': 'user', 'content': 'I live in Seattle.'}],
                }
            ],
            'evidence_session_ids': ['s1'],
        }
    ]

    # 1. JSON file
    json_path = tmp_path / 'dataset.json'
    json_path.write_text(json.dumps(data), encoding='utf-8')
    loaded_json = load_dataset(json_path)
    assert len(loaded_json) == 1
    q = loaded_json[0]
    assert q.question_id == 'q_json_1'
    assert q.question_type == 'knowledge-update'
    assert q.gold_answer == 'Seattle'
    assert q.question_date.tzinfo is not None
    assert len(q.sessions) == 1
    assert q.sessions[0].ts.tzinfo is not None
    assert q.sessions[0].timestamp == q.sessions[0].ts
    assert len(q.sessions[0].turns) == 1

    # 2. JSONL file
    jsonl_path = tmp_path / 'dataset.jsonl'
    jsonl_path.write_text(json.dumps(data[0]) + '\n', encoding='utf-8')
    loaded_jsonl = load_dataset(jsonl_path)
    assert len(loaded_jsonl) == 1
    assert loaded_jsonl[0].question_id == 'q_json_1'


@pytest.mark.asyncio
async def test_longmemeval_judge_heuristics() -> None:
    """Verify heuristic judging across knowledge-update, temporal-reasoning, abstention."""
    judge = LongMemEvalJudge(llm=None)

    # Abstention test
    q_abs = EvalQuestion(
        question_id='q_abs',
        question_type='abstention',
        question='What is my secret code?',
        gold_answer='The information is not available.',
        question_date=utc(2024, 1, 1),
        is_abstention=True,
    )
    res_abs_ok = await judge.judge(
        q_abs, 'I do not have that information available in your memory [F1].'
    )
    assert res_abs_ok.correct is True

    res_abs_bad = await judge.judge(q_abs, 'Your secret code is 12345.')
    assert res_abs_bad.correct is False

    # Knowledge Update test
    q_ku = EvalQuestion(
        question_id='q_ku',
        question_type='knowledge-update',
        question='Where do I live?',
        gold_answer='Amsterdam',
        question_date=utc(2024, 1, 1),
    )
    res_ku_ok = await judge.judge(q_ku, 'You currently reside in Amsterdam.')
    assert res_ku_ok.correct is True

    res_ku_bad = await judge.judge(q_ku, 'You currently reside in Berlin.')
    assert res_ku_bad.correct is False

    # Temporal Reasoning test
    q_tr = EvalQuestion(
        question_id='q_tr',
        question_type='temporal-reasoning',
        question='How long was my vacation?',
        gold_answer='7 days',
        question_date=utc(2024, 1, 1),
    )
    res_tr_ok = await judge.judge(q_tr, 'Your vacation was 7 days long.')
    assert res_tr_ok.correct is True


@pytest.mark.asyncio
async def test_longmemeval_judge_with_mock_llm() -> None:
    """Verify LLM-based judge parses DECISION: CORRECT and DECISION: INCORRECT properly."""
    fake_llm = MagicMock()
    fake_llm.complete_text = AsyncMock(
        return_value=LLMResponse(
            text='DECISION: CORRECT\nIdentified Amsterdam as current residence.',
            model='judge-v1',
            provider='fake',
            prompt_tokens=120,
            completion_tokens=25,
            latency_ms=50.0,
        )
    )

    judge = LongMemEvalJudge(llm=fake_llm, model_id='judge-v1')
    q = EvalQuestion(
        question_id='q_test',
        question_type='knowledge-update',
        question='Where do I live?',
        gold_answer='Amsterdam',
        question_date=utc(2024, 1, 1),
    )
    res = await judge.judge(q, 'You live in Amsterdam [F1].')
    assert res.correct is True
    assert res.judge_model == 'judge-v1'

    # Now verify INCORRECT decision
    fake_llm.complete_text = AsyncMock(
        return_value=LLMResponse(
            text='DECISION: INCORRECT\nThe prediction failed to mention Amsterdam.',
            model='judge-v1',
            provider='fake',
            prompt_tokens=120,
            completion_tokens=25,
            latency_ms=50.0,
        )
    )
    res_incorrect = await judge.judge(q, 'You live in Berlin.')
    assert res_incorrect.correct is False


@pytest.mark.asyncio
async def test_eval_runner_chronological_replay_and_scoring() -> None:
    """Test full session replay with FixedClock, extract jobs, scoring, and paired CIs."""
    dataset = make_synthetic_3question_dataset()

    # Create TrackingClock
    clock = TrackingClock(utc(2023, 1, 1))

    # Fake UoW factory
    fake_uow = FakeUow()
    uow_factory = FakeUowFactory(fake_uow)

    # Fake IngestService
    ingest_service = MagicMock()
    job_id = JobId(uuid4())
    ingest_service.ensure_user_bootstrap = AsyncMock()
    ingest_service.enqueue_turn = AsyncMock(
        return_value=EnqueueResult(
            source_id=SourceId(uuid4()),
            created=True,
            job_id=job_id,
        )
    )
    ingest_service.run_extract_job = AsyncMock(return_value=[FactId(uuid4())])

    # Fake ChatService
    chat_service = MagicMock()
    # Provide system answers corresponding to gold answers
    chat_service.respond = AsyncMock(
        side_effect=[
            ChatTurnResult(
                conversation_id=uuid4(),
                user_message_id=uuid4(),
                assistant_message_id=uuid4(),
                content='You currently live in Amsterdam [F2].',
            ),
            ChatTurnResult(
                conversation_id=uuid4(),
                user_message_id=uuid4(),
                assistant_message_id=uuid4(),
                content='You were visiting Paris from March 5 to March 12 for 7 days [F1].',
            ),
            ChatTurnResult(
                conversation_id=uuid4(),
                user_message_id=uuid4(),
                assistant_message_id=uuid4(),
                content="I do not have information about your mother's maiden name.",
                abstained=True,
            ),
        ]
    )

    # Comparative baseline (outdated / naive answers)
    baseline_system = MockMemorySystem(
        name='naive_baseline',
        answers=[
            'You live in Berlin.',  # wrong on knowledge-update
            'March 5 to March 12 for 7 days.',  # correct on temporal-reasoning
            'Your mother is Sarah.',  # hallucinates on abstention
        ],
    )

    runner = LongMemEvalRunner(
        system_name='tmem',
        chat_service=chat_service,
        ingest_service=ingest_service,
        uow_factory=uow_factory,
        clock=clock,
        baseline=baseline_system,
        seed=1337,
    )

    report = await runner.run(dataset)

    # 1. Verify chronological replay using FixedClock anchored to session.timestamp
    assert len(clock.history) > 0
    # Last set for question 1 should be its question_date (2024-03-01)
    assert any(dt == utc(2024, 3, 1) for dt in clock.history)
    assert any(dt == utc(2024, 3, 20) for dt in clock.history)
    assert any(dt == utc(2024, 4, 1) for dt in clock.history)

    # 2. Verify synchronous extract jobs were run
    assert ingest_service.run_extract_job.call_count > 0

    # 3. Verify chat_service.respond was called for each question
    assert chat_service.respond.call_count == 3

    # 4. Verify evaluation report metrics
    assert report.total_questions == 3
    assert report.system_name == 'tmem'
    assert report.baseline_name == 'naive_baseline'
    assert report.overall_accuracy == 1.0  # System got 3/3 correct
    assert report.abstention_accuracy == 1.0
    assert report.overall_token_f1 > 0.4

    # 5. Verify category metrics
    cats = report.category_metrics
    assert 'knowledge-update' in cats
    assert 'temporal-reasoning' in cats
    assert 'abstention' in cats

    ku = cats['knowledge-update']
    assert ku.count == 1
    assert ku.accuracy == 1.0
    assert ku.baseline_accuracy == 0.0

    tr = cats['temporal-reasoning']
    assert tr.count == 1
    assert tr.accuracy == 1.0
    assert tr.baseline_accuracy == 1.0

    abs_cat = cats['abstention']
    assert abs_cat.count == 1
    assert abs_cat.accuracy == 1.0
    assert abs_cat.baseline_accuracy == 0.0

    # 6. Verify paired bootstrap CI against baseline exists
    assert report.paired_ci_accuracy is not None
    diff, low, high = report.paired_ci_accuracy
    # System scored 1.0, baseline scored 1/3 ~ 0.333, diff is positive
    assert diff > 0.0


def test_render_markdown_report_with_and_without_baseline() -> None:
    """Verify Markdown report generation format and tables."""
    # 1. Report with baseline
    cat_summary = {
        'knowledge-update': CategoryMetricSummary(
            category='knowledge-update',
            count=1,
            accuracy=1.0,
            token_f1=0.88,
            exact_match=1.0,
            ci_low=1.0,
            ci_high=1.0,
            baseline_accuracy=0.0,
            baseline_token_f1=0.15,
            baseline_exact_match=0.0,
            paired_ci_diff=1.0,
            paired_ci_low=1.0,
            paired_ci_high=1.0,
        ),
        'temporal-reasoning': CategoryMetricSummary(
            category='temporal-reasoning',
            count=1,
            accuracy=1.0,
            token_f1=0.92,
            exact_match=1.0,
            ci_low=1.0,
            ci_high=1.0,
            baseline_accuracy=1.0,
            baseline_token_f1=0.92,
            baseline_exact_match=1.0,
            paired_ci_diff=0.0,
            paired_ci_low=0.0,
            paired_ci_high=0.0,
        ),
    }

    report = EvalReport(
        benchmark_name='LongMemEval',
        system_name='tmem',
        baseline_name='full_context',
        total_questions=2,
        overall_accuracy=1.0,
        overall_token_f1=0.90,
        overall_exact_match=1.0,
        abstention_accuracy=1.0,
        category_metrics=cat_summary,
        paired_ci_accuracy=(0.5, 0.0, 1.0),
        paired_ci_f1=(0.35, 0.0, 0.70),
        latency_stats=LatencyStats(p50=45.0, p95=50.0, p99=52.0, mean=46.0),
        baseline_latency_stats=LatencyStats(p50=35.0, p95=40.0, p99=42.0, mean=36.0),
    )

    md = render_markdown_report(report)
    assert '# LongMemEval Benchmark Report' in md
    assert '**System:** `tmem`' in md
    assert '**Baseline:** `full_context`' in md
    assert '| Category | N | System Acc | 95% CI | Baseline Acc |' in md
    assert 'Δ (Paired 95% CI) | System F1 | Baseline F1 |' in md
    assert 'knowledge-update' in md
    assert 'temporal-reasoning' in md
    assert '## Latency Profile' in md
    assert 'knowledge-update' in md
    assert 'temporal-reasoning' in md
    assert '## Latency Profile' in md

    # 2. Report without baseline
    report_no_base = EvalReport(
        benchmark_name='LongMemEval',
        system_name='tmem',
        baseline_name=None,
        total_questions=2,
        overall_accuracy=1.0,
        overall_token_f1=0.90,
        overall_exact_match=1.0,
        abstention_accuracy=1.0,
        category_metrics=cat_summary,
        paired_ci_accuracy=None,
        paired_ci_f1=None,
        latency_stats=LatencyStats(p50=45.0, p95=50.0, p99=52.0, mean=46.0),
    )
    md_no_base = render_markdown_report(report_no_base)
    assert '| Category | N | Accuracy | 95% CI | Token F1 | Exact Match |' in md_no_base


def test_eval_report_json_serialization() -> None:
    """Verify EvalReport converts to dictionary and valid JSON string."""
    item = EvalItemResult(
        question_id='q1',
        question_type='knowledge-update',
        question='Where do I live?',
        gold_answer='Amsterdam',
        system_answer='Amsterdam',
        judge_correct=True,
        token_f1=1.0,
        exact_match=True,
        latency_ms=30.0,
    )
    report = EvalReport(
        benchmark_name='LongMemEval',
        system_name='tmem',
        baseline_name=None,
        total_questions=1,
        overall_accuracy=1.0,
        overall_token_f1=1.0,
        overall_exact_match=1.0,
        abstention_accuracy=1.0,
        category_metrics={},
        item_results=(item,),
    )
    raw_json = report.to_json()
    parsed = json.loads(raw_json)
    assert parsed['benchmark_name'] == 'LongMemEval'
    assert parsed['total_questions'] == 1
    assert len(parsed['item_results']) == 1
    assert parsed['item_results'][0]['question_id'] == 'q1'


def test_cli_parsing_and_run(tmp_path: Path) -> None:
    """Verify eval/cli.py argument parsing and file generation."""
    # 1. Argument parsing with direct flags
    args = parse_args(['--dataset', 'test.json', '--limit', '10', '--baseline', 'full_context'])
    assert args.dataset == 'test.json'
    assert args.limit == 10
    assert args.baseline == 'full_context'
    assert args.seed == 42
    assert args.output_dir == 'results'

    # 2. Argument parsing with 'run' prefix
    args_run = parse_args(['run', '--dataset', 'test.json', '--output-dir', 'out_dir'])
    assert args_run.dataset == 'test.json'
    assert args_run.output_dir == 'out_dir'

    # 3. CLI execution writing files
    ds_file = tmp_path / 'cli_dataset.json'
    ds_content = [
        {
            'question_id': 'cli_1',
            'question_type': 'knowledge-update',
            'question': 'Where do I live?',
            'answer': 'Tokyo',
            'question_date': '2024-03-01T12:00:00Z',
            'sessions': [
                {
                    'session_id': 's1',
                    'date': '2024-01-01T10:00:00Z',
                    'turns': [{'role': 'user', 'content': 'I moved to Tokyo.'}],
                }
            ],
        }
    ]
    ds_file.write_text(json.dumps(ds_content), encoding='utf-8')
    out_dir = tmp_path / 'out'

    ret = main(
        [
            '--dataset',
            str(ds_file),
            '--output-dir',
            str(out_dir),
            '--system',
            'full_context',  # uses built-in baseline
        ]
    )
    assert ret == 0
    assert (out_dir / 'report.md').exists()
    assert (out_dir / 'report.json').exists()
    md_content = (out_dir / 'report.md').read_text(encoding='utf-8')
    assert 'LongMemEval Benchmark Report' in md_content
