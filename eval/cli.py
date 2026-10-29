"""CLI entrypoint for LongMemEval and memory engine ablations per LLD Section 13.10 and 19.

Usage:
    python -m eval.cli --dataset path/to/dataset.json --baseline full_context --output-dir results
    python -m eval.cli run --dataset path/to/dataset.json --limit 50 --seed 42
    python -m eval.cli --prepare-sample
    python -m eval.cli --live --baselines vector_only,bm25_only,full_bitemporal --delay-s 4.0
"""

import argparse
import asyncio
import json
import logging
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

from eval.longmemeval import LongMemEvalRunner, load_dataset, render_markdown_report

if TYPE_CHECKING:
    from memory_engine.ports.uow import UnitOfWorkFactory

_logger = logging.getLogger(__name__)

T = TypeVar('T')


def _strip_code_fences(text: str) -> str:
    """Strip markdown code block fences and whitespace from model response."""
    text = text.strip()
    if text.startswith('```'):
        lines = text.splitlines()
        if lines and lines[0].startswith('```'):
            lines = lines[1:]
        if lines and lines[-1].strip() == '```':
            lines = lines[:-1]
        text = '\n'.join(lines).strip()
    return text


def build_parser() -> argparse.ArgumentParser:
    """Build argument parser supporting both top-level and run subcommand flags."""
    parser = argparse.ArgumentParser(
        prog='eval.cli',
        description='Temporal Memory Engine Evaluation CLI (LongMemEval & Ablations)',
    )

    parser.add_argument(
        '--dataset',
        type=str,
        required=False,
        default=None,
        help='Path to dataset JSON or JSONL file',
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit number of questions to evaluate',
    )
    parser.add_argument(
        '--baseline',
        type=str,
        default=None,
        help='Baseline system name (e.g. full_context, vector_only, bm25_only)',
    )
    parser.add_argument(
        '--baselines',
        type=str,
        default=None,
        help=(
            'Comma-separated list of baselines to run in --live mode. '
            'Default: vector_only,bm25_only,hybrid_no_bitemporal,'
            'bitemporal_no_decay,full_bitemporal'
        ),
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results',
        help='Directory to output report.md and report.json',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for bootstrap resampling and reproducibility',
    )
    parser.add_argument(
        '--system',
        type=str,
        default='tmem',
        help='System under test name (default: tmem)',
    )
    parser.add_argument(
        '--reader',
        type=str,
        default=None,
        help='Reader LLM model ID',
    )
    parser.add_argument(
        '--judge',
        type=str,
        default=None,
        help='Judge LLM model ID',
    )
    parser.add_argument(
        '--live',
        action='store_true',
        default=False,
        help=(
            'Enable live evaluation using GeminiClient (SB_GEMINI_API_KEY from .env) '
            'as reader and judge, with CachedEmbedder in eval/.cache/'
        ),
    )
    parser.add_argument(
        '--delay-s',
        type=float,
        default=4.0,
        dest='delay_s',
        help=(
            'Sleep delay in seconds between LLM judge/reader calls to respect '
            'free-tier Gemini 15 RPM rate limit (default: 4.0)'
        ),
    )
    parser.add_argument(
        '--cache-dir',
        type=str,
        default='eval/.cache',
        dest='cache_dir',
        help='Directory for embedding SQLite cache (default: eval/.cache)',
    )
    parser.add_argument(
        '--question-id',
        type=str,
        default=None,
        dest='question_id',
        help='Evaluate only the question matching this question_id (e.g. ssu_006 or ku_001)',
    )
    parser.add_argument(
        '--prepare-sample',
        action='store_true',
        default=False,
        dest='prepare_sample',
        help=(
            'Write the bundled 30-item stratified LongMemEval sample dataset to '
            'data/longmemeval_stratified_30.json (or --dataset path) and exit.'
        ),
    )

    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments, tolerating optional run subcommand."""
    raw_args = list(argv if argv is not None else sys.argv[1:])

    # If first token is 'run', strip it to support `python -m eval.cli run --dataset ...`
    if raw_args and raw_args[0] == 'run':
        raw_args = raw_args[1:]

    parser = build_parser()
    return parser.parse_args(raw_args)


def _resolve_env_value(key: str, default: str = '') -> str:
    """Read a key=value pair from .env or os.environ, return default if missing."""
    import os

    for candidate in (Path('.env'), Path(__file__).parent.parent / '.env'):
        if candidate.exists():
            for line in candidate.read_text(encoding='utf-8').splitlines():
                stripped = line.strip()
                if stripped.startswith(f'{key}='):
                    return stripped[len(f'{key}=') :].strip()
            break  # found the file, key just was not in it

    return os.environ.get(key, default)


def _sanitize_facts_data(data: Any) -> Any:
    """Sanitize extraction facts data to handle LLM quirks like speaker vs subject."""
    if isinstance(data, list):
        data = {'facts': data}
    if not isinstance(data, dict) or 'facts' not in data or not isinstance(data['facts'], list):
        return data

    from memory_engine.domain.models import ExtractedFact

    allowed_fields = set(ExtractedFact.model_fields.keys())
    cleaned_facts: list[dict[str, Any]] = []

    for item in data['facts']:
        if not isinstance(item, dict):
            continue

        item_dict = dict(item)

        # If subject is missing or empty and speaker is present, rename speaker to subject
        subj = item_dict.get('subject')
        if (subj is None or not str(subj).strip()) and 'speaker' in item_dict:
            item_dict['subject'] = str(item_dict.pop('speaker')).strip()
        else:
            item_dict.pop('speaker', None)

        # Pop any unknown keys not in ExtractedFact
        for k in list(item_dict.keys()):
            if k not in allowed_fields:
                item_dict.pop(k, None)

        # Filter out if subject, predicate, object, or evidence_span is None or empty after
        # stripping
        is_valid = True
        for field in ('subject', 'predicate', 'object', 'evidence_span'):
            val = item_dict.get(field)
            if val is None or not str(val).strip():
                is_valid = False
                break
        if not is_valid:
            continue

        # Standard string clean and defaults
        item_dict['subject'] = str(item_dict['subject']).strip()
        item_dict['predicate'] = str(item_dict['predicate']).strip()
        item_dict['object'] = str(item_dict['object']).strip()
        item_dict['evidence_span'] = str(item_dict['evidence_span']).strip()

        item_dict.setdefault('confidence', 1.0)
        item_dict.setdefault('subject_type', 'person')
        item_dict.setdefault('object_type', 'literal')
        item_dict.setdefault('is_retraction', False)
        item_dict.setdefault('valid_from_text', None)
        item_dict.setdefault('valid_to_text', None)
        item_dict.setdefault('source_turn', 0)

        cleaned_facts.append(item_dict)

    data['facts'] = cleaned_facts
    return data


class _PacedGateway:
    """LLMGateway shim with pacing, retry/backoff on 503/429/timeout, and JSON completion."""

    _MAX_ATTEMPTS: int = 4

    def __init__(
        self,
        client: object,
        delay_s: float,
        fallback_client: object | None = None,
    ) -> None:
        self._client = client
        self._delay_s = delay_s
        self._fallback_client = fallback_client

    async def complete_text(self, req: object) -> object:
        """Pace then delegate; retry with exp-backoff on 503/429/timeout transients."""
        from memory_engine.domain.errors import (
            LLMRateLimited,
            LLMTimeout,
            LLMUnavailable,
        )

        if self._delay_s > 0.0:
            await asyncio.sleep(self._delay_s)

        last_exc: Exception | None = None
        for attempt in range(self._MAX_ATTEMPTS):
            try:
                # GeminiClient exposes .complete(), not .complete_text()
                return await self._client.complete(req)  # type: ignore[attr-defined]
            except (LLMUnavailable, LLMRateLimited, LLMTimeout) as exc:
                last_exc = exc
                wait = self._delay_s * (2**attempt)
                _logger.warning(
                    'Gemini transient error on attempt %d/%d (%s); retrying in %.1fs',
                    attempt + 1,
                    self._MAX_ATTEMPTS,
                    exc,
                    wait,
                )
                await asyncio.sleep(wait)

        # All primary retries exhausted — try fallback model once
        if self._fallback_client is not None:
            _logger.warning(
                'All %d attempts on primary model failed; trying fallback model',
                self._MAX_ATTEMPTS,
            )
            try:
                return await self._fallback_client.complete(req)  # type: ignore[attr-defined]
            except Exception as fb_exc:
                _logger.error('Fallback model also failed: %s', fb_exc)

        raise last_exc  # type: ignore[misc]

    async def complete_json(self, req: Any, schema: type[T]) -> T:
        """Pace then delegate JSON completion with backoff and schema validation."""
        from memory_engine.domain.errors import (
            LLMRateLimited,
            LLMTimeout,
            LLMUnavailable,
        )
        from memory_engine.ports.llm import LLMRequest

        if not getattr(req, 'json_mode', False):
            req = LLMRequest(
                purpose=req.purpose,
                tier=req.tier,
                messages=req.messages,
                prompt_version=req.prompt_version,
                temperature=req.temperature,
                max_tokens=req.max_tokens,
                json_mode=True,
                priority=getattr(req, 'priority', 'interactive'),
                timeout_s=getattr(req, 'timeout_s', 30.0),
            )

        schema_any: Any = schema
        last_exc: Exception | None = None
        for attempt in range(self._MAX_ATTEMPTS):
            try:
                resp = await self.complete_text(req)
                raw_text = getattr(resp, 'text', '')
                clean_text = _strip_code_fences(raw_text)
                data = json.loads(clean_text)
                data = _sanitize_facts_data(data)
                if hasattr(schema_any, 'model_fields') and isinstance(data, dict):
                    allowed_fields = set(schema_any.model_fields.keys())
                    data = {k: v for k, v in data.items() if k in allowed_fields}
                if hasattr(schema_any, 'model_validate'):
                    val_res: T = schema_any.model_validate(data)
                    return val_res
                fb_inst: T = schema(**data)
                return fb_inst
            except (LLMUnavailable, LLMRateLimited, LLMTimeout) as exc:
                last_exc = exc
                wait = self._delay_s * (2**attempt)
                _logger.warning(
                    'Gemini transient error on attempt %d/%d (%s); retrying in %.1fs',
                    attempt + 1,
                    self._MAX_ATTEMPTS,
                    exc,
                    wait,
                )
                await asyncio.sleep(wait)
            except Exception as parse_exc:
                last_exc = parse_exc
                _logger.warning('JSON parse attempt %d failed: %s', attempt + 1, parse_exc)
                if self._delay_s > 0.0:
                    await asyncio.sleep(self._delay_s)

        if self._fallback_client is not None:
            _logger.warning(
                'All %d attempts on primary model failed; trying fallback model for complete_json',
                self._MAX_ATTEMPTS,
            )
            try:
                fb_resp = await self._fallback_client.complete(req)  # type: ignore[attr-defined]
                raw_text = getattr(fb_resp, 'text', '')
                clean_text = _strip_code_fences(raw_text)
                data = json.loads(clean_text)
                data = _sanitize_facts_data(data)
                if hasattr(schema_any, 'model_fields') and isinstance(data, dict):
                    allowed_fields = set(schema_any.model_fields.keys())
                    data = {k: v for k, v in data.items() if k in allowed_fields}
                if hasattr(schema_any, 'model_validate'):
                    val_res_fb: T = schema_any.model_validate(data)
                    return val_res_fb
                fb_inst2: T = schema(**data)
                return fb_inst2
            except Exception as fb_exc:
                _logger.error('Fallback model also failed for complete_json: %s', fb_exc)

        raise last_exc  # type: ignore[misc]


async def _run_live_eval(args: argparse.Namespace) -> int:
    """Run live multi-baseline evaluation using GeminiClient and CachedEmbedder."""
    from memory_engine.domain.clock import FixedClock
    from memory_engine.embed.cached import CachedEmbedder
    from memory_engine.embed.fastembed_adapter import FastEmbedAdapter
    from memory_engine.ingest.extractor import FactExtractor
    from memory_engine.ingest.service import IngestService
    from memory_engine.llm.gemini import GeminiClient

    from eval.baselines import (
        LLMReader,
        build_baseline_orchestrator,
        create_baseline,
        get_baseline_config,
    )
    from eval.longmemeval import EvalReport, LongMemEvalJudge

    api_key = _resolve_env_value('SB_GEMINI_API_KEY')
    if not api_key:
        sys.stderr.write(
            'Error: SB_GEMINI_API_KEY not found in .env or environment. '
            'Set it before using --live.\n'
        )
        return 2

    model = _resolve_env_value('SB_GEMINI_MODEL', 'gemini-3.8-flash')

    # Embedding cache
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_db = cache_dir / 'embeddings.sqlite'

    inner_embedder = FastEmbedAdapter()
    embedder = CachedEmbedder(inner=inner_embedder, cache_db_path=cache_db)

    # Primary Gemini client + optional fallback to lighter model on sustained 503
    primary_client: GeminiClient = GeminiClient(
        api_key=api_key,
        model_fast=model,
        model_synth=model,
    )
    fallback_model = 'gemini-3.5-flash-lite'
    fallback_client: GeminiClient = GeminiClient(
        api_key=api_key,
        model_fast=fallback_model,
        model_synth=fallback_model,
    )
    paced_gw = _PacedGateway(
        client=primary_client,
        delay_s=args.delay_s,
        fallback_client=fallback_client,
    )

    # Connect to PostgreSQL via SB_DATABASE_URL if available
    db_url = _resolve_env_value('SB_DATABASE_URL')
    pool: Any = None
    uow_factory: UnitOfWorkFactory | None = None
    if db_url:
        try:
            from memory_engine.db.pool import create_db_pool
            from memory_engine.db.uow import PostgresUnitOfWorkFactory

            pool = await create_db_pool(db_url, verify_schema=False)
            uow_factory = PostgresUnitOfWorkFactory(pool)
            _logger.info('Connected live Postgres pool at %s', db_url)
        except Exception as err:
            _logger.warning('Could not connect to Postgres at %s: %s', db_url, err)

    from memory_engine.ingest.canonicalizer import Canonicalizer

    canonicalizer = Canonicalizer()
    clock = FixedClock(datetime(2024, 1, 1, tzinfo=UTC))
    extractor = FactExtractor(llm=paced_gw)  # type: ignore[arg-type]
    ingest_service = (
        IngestService(
            uows=uow_factory,
            extractor=extractor,
            canonicalizer=canonicalizer,
            embedder=embedder,
            clock=clock,
            debounce_s=0.0,
        )
        if uow_factory is not None
        else None
    )

    # Reader and judge backed by real Gemini
    reader = LLMReader(llm_gateway=paced_gw)  # type: ignore[arg-type]
    judge = LongMemEvalJudge(llm=paced_gw, model_id=model)  # type: ignore[arg-type]

    # Resolve baselines list
    if args.baselines:
        baseline_names = [b.strip() for b in args.baselines.split(',') if b.strip()]
    elif args.baseline:
        baseline_names = [args.baseline]
    else:
        baseline_names = [
            'vector_only',
            'bm25_only',
            'hybrid_no_bitemporal',
            'bitemporal_no_decay',
            'full_bitemporal',
        ]

    # Load dataset
    dataset_path = (
        Path(args.dataset) if args.dataset else Path('data/longmemeval_stratified_30.json')
    )
    if not dataset_path.exists():
        sys.stderr.write(f'Error: dataset file not found: {dataset_path}\n')
        if pool is not None:
            await pool.close()
        return 2

    questions = load_dataset(dataset_path)
    if not questions:
        sys.stderr.write(f'Error: no questions loaded from {dataset_path}\n')
        if pool is not None:
            await pool.close()
        return 2

    if args.question_id:
        questions = [q for q in questions if q.question_id == args.question_id]
        if not questions:
            sys.stderr.write(
                f'Error: question_id {args.question_id!r} not found in {dataset_path}\n'
            )
            if pool is not None:
                await pool.close()
            return 2

    if args.limit:
        questions = questions[: args.limit]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    overall_exit = 0

    all_reports: list[tuple[str, EvalReport]] = []

    try:
        for bl_name in baseline_names:
            _logger.info('Running live baseline: %s (%d questions)', bl_name, len(questions))
            sys.stdout.write(f'\n=== Baseline: {bl_name} ===\n')
            sys.stdout.flush()

            bl_cfg = get_baseline_config(bl_name)
            orch = None
            if uow_factory is not None and bl_cfg.enabled_lanes:
                orch = build_baseline_orchestrator(
                    config=bl_cfg,
                    uow_factory=uow_factory,
                    embedder=embedder,
                    clock=clock,
                    llm_gateway=paced_gw,  # type: ignore[arg-type]
                )

            bl_system = create_baseline(
                bl_name,
                reader=reader,
                uow_factory=uow_factory,
                embedder=embedder,
                orchestrator=orch,
                ingest_service=ingest_service,
                clock=clock,
            )

            runner = LongMemEvalRunner(
                system_name=bl_name,
                system=bl_system,
                baseline=None,
                judge=judge,
                reader=reader,
                embedder=embedder,
                seed=args.seed,
                uow_factory=uow_factory,
                ingest_service=ingest_service,
                clock=clock,
            )

            report = await runner.run(questions)
            all_reports.append((bl_name, report))

            md_content = render_markdown_report(report)
            json_content = report.to_json(indent=2)

            md_file = out_dir / f'report_{bl_name}.md'
            json_file = out_dir / f'report_{bl_name}.json'

            md_file.write_text(md_content, encoding='utf-8')
            json_file.write_text(json_content, encoding='utf-8')

            sys.stdout.write(md_content)
            _logger.info('Saved %s reports to %s and %s', bl_name, md_file, json_file)

            if any(r.error is not None for r in report.item_results):
                overall_exit = 3
    finally:
        if pool is not None:
            await pool.close()

    # Write combined summary table across all baselines
    if all_reports:
        summary_lines = [
            '# LongMemEval Live Evaluation — Baseline Comparison',
            '',
            f'- **Dataset:** `{dataset_path}`',
            f'- **Model:** `{model}`',
            f'- **Questions:** `{len(questions)}`',
            '',
            '| Baseline | Accuracy | Token F1 | Abstention Acc | p50 Latency (ms) |',
            '| :--- | :---: | :---: | :---: | :---: |',
        ]
        for bl_name, rpt in all_reports:
            lat = f'{rpt.latency_stats.p50:.0f}' if rpt.latency_stats else '-'
            summary_lines.append(
                f'| {bl_name} '
                f'| {rpt.overall_accuracy * 100:.1f}% '
                f'| {rpt.overall_token_f1 * 100:.1f}% '
                f'| {rpt.abstention_accuracy * 100:.1f}% '
                f'| {lat} |'
            )
        summary_md = '\n'.join(summary_lines) + '\n'
        summary_file = out_dir / 'report_summary.md'
        summary_file.write_text(summary_md, encoding='utf-8')
        sys.stdout.write(f'\n{summary_md}')
        _logger.info('Saved combined summary to %s', summary_file)

    sys.stdout.write(f'\nEmbedding cache — hits: {embedder.hits}, misses: {embedder.misses}\n')
    return overall_exit


async def run_eval_cli(args: argparse.Namespace) -> int:
    """Execute evaluation and output markdown and JSON reports."""
    if args.live:
        return await _run_live_eval(args)

    if args.dataset is None:
        sys.stderr.write('Error: --dataset is required unless --live or --prepare-sample is set\n')
        return 2

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        _logger.error('Dataset path does not exist: %s', dataset_path)
        sys.stderr.write(f'Error: dataset file not found: {dataset_path}\n')
        return 2

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    _logger.info('Loading questions from %s', dataset_path)
    questions = load_dataset(dataset_path)
    if not questions:
        sys.stderr.write(f'Error: no questions loaded from {dataset_path}\n')
        return 2

    if args.question_id:
        questions = [q for q in questions if q.question_id == args.question_id]
        if not questions:
            sys.stderr.write(
                f'Error: question_id {args.question_id!r} not found in {dataset_path}\n'
            )
            return 2

    runner = LongMemEvalRunner(
        system_name=args.system,
        baseline=args.baseline,
        seed=args.seed,
    )

    report = await runner.run(questions, limit=args.limit)

    # Render and save reports
    md_content = render_markdown_report(report)
    json_content = report.to_json(indent=2)

    md_file = out_dir / 'report.md'
    json_file = out_dir / 'report.json'

    md_file.write_text(md_content, encoding='utf-8')
    json_file.write_text(json_content, encoding='utf-8')

    sys.stdout.write(md_content)
    _logger.info('Saved reports to %s and %s', md_file, json_file)

    has_errors = any(r.error is not None for r in report.item_results)
    return 3 if has_errors else 0


def _write_sample_dataset(dest: Path) -> None:
    """Copy the bundled 30-item stratified dataset to dest."""
    import shutil

    dest.parent.mkdir(parents=True, exist_ok=True)
    bundled = Path(__file__).parent.parent / 'data' / 'longmemeval_stratified_30.json'
    if not bundled.exists():
        sys.stderr.write(
            f'Bundled dataset not found at {bundled}. '
            'Ensure data/longmemeval_stratified_30.json exists in the project root.\n'
        )
        return
    if bundled.resolve() == dest.resolve():
        sys.stdout.write(f'Sample dataset already at: {dest}\n')
        return
    shutil.copy2(bundled, dest)
    sys.stdout.write(f'Sample dataset written to: {dest}\n')


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for evaluation CLI."""
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(name)s %(message)s')

    try:
        args = parse_args(argv)
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else 2

    if args.prepare_sample:
        dest = Path(args.dataset) if args.dataset else Path('data/longmemeval_stratified_30.json')
        _write_sample_dataset(dest)
        return 0

    return asyncio.run(run_eval_cli(args))


if __name__ == '__main__':
    sys.exit(main())
