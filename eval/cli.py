"""CLI entrypoint for LongMemEval and memory engine ablations per LLD Section 13.10 and 19.

Usage:
    python -m eval.cli --dataset path/to/dataset.json --baseline full_context --output-dir results
    python -m eval.cli run --dataset path/to/dataset.json --limit 50 --seed 42
"""

import argparse
import asyncio
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from eval.longmemeval import LongMemEvalRunner, load_dataset, render_markdown_report

_logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Build argument parser supporting both top-level and 'run' subcommand flags."""
    parser = argparse.ArgumentParser(
        prog='eval.cli',
        description='Temporal Memory Engine Evaluation CLI (LongMemEval & Ablations)',
    )

    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
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

    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments, tolerating optional 'run' subcommand."""
    raw_args = list(argv if argv is not None else sys.argv[1:])

    # If first token is 'run', strip it to support `python -m eval.cli run --dataset ...`
    if raw_args and raw_args[0] == 'run':
        raw_args = raw_args[1:]

    parser = build_parser()
    return parser.parse_args(raw_args)


async def run_eval_cli(args: argparse.Namespace) -> int:
    """Execute evaluation and output markdown and JSON reports."""
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


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for evaluation CLI."""
    try:
        args = parse_args(argv)
    except SystemExit as exc:
        return int(exc.code) if isinstance(exc.code, int) else 2

    return asyncio.run(run_eval_cli(args))


if __name__ == '__main__':
    sys.exit(main())
