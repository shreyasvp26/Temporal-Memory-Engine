"""Evaluation metrics for LongMemEval and memory engine ablations per LLD Section 13.7.

Provides:
- normalize_answer: lowercase, remove punctuation, remove articles, collapse whitespace.
- exact_match: exact string match on normalized answers.
- token_f1: token-level precision, recall, and harmonic mean F1.
- paired_bootstrap_ci: paired bootstrap 95% confidence interval for score differences (A - B).
- percentile_bootstrap_ci: percentile bootstrap confidence interval for single metric.
- compute_latency_stats: p50, p95, p99 percentiles and mean latency.
- abstention_accuracy: classification accuracy for abstention cases.
"""

import math
import random
import re
import string
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

__all__ = [
    'LatencyStats',
    'abstention_accuracy',
    'compute_latency_stats',
    'exact_match',
    'is_abstention_text',
    'normalize_answer',
    'paired_bootstrap_ci',
    'percentile_bootstrap_ci',
    'token_f1',
]

_ARTICLES = re.compile(r'\b(a|an|the)\b', re.IGNORECASE)
_PUNCTUATION_SET = set(string.punctuation)

_ABSTENTION_PHRASES = (
    'not available',
    'information is not available',
    'i do not know',
    "i don't know",
    'do not have information',
    "don't have information",
    'do not have that information',
    "don't have that information",
    'not have that information',
    'unknown',
    'cannot answer',
    'not mentioned',
    'insufficient information',
    'no information',
)


@dataclass(frozen=True, slots=True)
class LatencyStats:
    """Summary statistics for latency measurements in milliseconds."""

    p50: float
    p95: float
    p99: float
    mean: float


def normalize_answer(text: str) -> str:
    """Normalize text: lowercase, remove punctuation/articles, collapse whitespace."""
    lower_text = text.lower()
    no_punc = ''.join(ch for ch in lower_text if ch not in _PUNCTUATION_SET)
    no_articles = _ARTICLES.sub(' ', no_punc)
    return ' '.join(no_articles.split())


def exact_match(prediction: str, ground_truth: str) -> bool:
    """Evaluate exact match between prediction and ground truth after normalization."""
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def token_f1(prediction: str, ground_truth: str) -> float:
    """Calculate token-level F1 score between prediction and ground truth."""
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()

    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0

    precision = float(num_same) / len(pred_tokens)
    recall = float(num_same) / len(gold_tokens)
    return (2.0 * precision * recall) / (precision + recall)


def _percentile(sorted_vals: Sequence[float], p: float) -> float:
    """Calculate p-th percentile (0 <= p <= 100) using linear interpolation."""
    if not sorted_vals:
        return 0.0
    n = len(sorted_vals)
    if n == 1:
        return float(sorted_vals[0])
    k = (n - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return float(sorted_vals[f])
    d0 = sorted_vals[f] * (c - k)
    d1 = sorted_vals[c] * (k - f)
    return float(d0 + d1)


def compute_latency_stats(latencies_ms: Sequence[float]) -> LatencyStats:
    """Compute p50, p95, p99 percentiles and mean latency."""
    if not latencies_ms:
        return LatencyStats(p50=0.0, p95=0.0, p99=0.0, mean=0.0)

    sorted_lat = sorted(float(x) for x in latencies_ms)
    mean_val = sum(sorted_lat) / len(sorted_lat)
    p50_val = _percentile(sorted_lat, 50.0)
    p95_val = _percentile(sorted_lat, 95.0)
    p99_val = _percentile(sorted_lat, 99.0)

    return LatencyStats(
        p50=p50_val,
        p95=p95_val,
        p99=p99_val,
        mean=mean_val,
    )


def paired_bootstrap_ci(
    scores_a: Sequence[float],
    scores_b: Sequence[float],
    n_resamples: int = 2000,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Calculate paired bootstrap 95% confidence interval for score differences (A - B).

    Returns:
        (mean_diff, ci_low, ci_high) representing the observed difference in means
        and the 2.5% and 97.5% percentiles of the paired bootstrap resamples.
    """
    if len(scores_a) != len(scores_b):
        msg = (
            f'scores_a (len={len(scores_a)}) and scores_b (len={len(scores_b)}) '
            'must have the same length'
        )
        raise ValueError(msg)

    n = len(scores_a)
    if n == 0:
        return (0.0, 0.0, 0.0)

    deltas = [float(a - b) for a, b in zip(scores_a, scores_b, strict=True)]
    mean_diff = sum(deltas) / n

    if n_resamples <= 0:
        return (mean_diff, mean_diff, mean_diff)

    rng = random.Random(seed)
    bootstrap_diffs: list[float] = []
    for _ in range(n_resamples):
        sample = rng.choices(deltas, k=n)
        bootstrap_diffs.append(sum(sample) / n)

    bootstrap_diffs.sort()
    ci_low = _percentile(bootstrap_diffs, 2.5)
    ci_high = _percentile(bootstrap_diffs, 97.5)

    return (mean_diff, ci_low, ci_high)


def percentile_bootstrap_ci(
    scores: Sequence[float],
    n_resamples: int = 2000,
    seed: int = 42,
    ci: float = 0.95,
) -> tuple[float, float, float]:
    """Calculate percentile bootstrap confidence interval for a single distribution.

    Returns:
        (mean, ci_low, ci_high)
    """
    n = len(scores)
    if n == 0:
        return (0.0, 0.0, 0.0)

    vals = [float(x) for x in scores]
    mean_val = sum(vals) / n

    if n_resamples <= 0:
        return (mean_val, mean_val, mean_val)

    alpha = (1.0 - ci) / 2.0
    low_pct = alpha * 100.0
    high_pct = (1.0 - alpha) * 100.0

    rng = random.Random(seed)
    bootstrap_means: list[float] = []
    for _ in range(n_resamples):
        sample = rng.choices(vals, k=n)
        bootstrap_means.append(sum(sample) / n)

    bootstrap_means.sort()
    ci_low = _percentile(bootstrap_means, low_pct)
    ci_high = _percentile(bootstrap_means, high_pct)

    return (mean_val, ci_low, ci_high)


def is_abstention_text(text: str) -> bool:
    """Determine whether text conveys an abstention per LLD Section 13.6."""
    norm = text.lower()
    return any(phrase in norm for phrase in _ABSTENTION_PHRASES)


def abstention_accuracy(
    predictions: Sequence[str | bool],
    ground_truths: Sequence[bool],
) -> float:
    """Calculate classification accuracy for abstention cases.

    A prediction is considered an abstention if it is True (boolean) or contains
    standard abstention phrases (string). Ground truth is True if case was an abstention.
    """
    if len(predictions) != len(ground_truths):
        raise ValueError(
            f'predictions (len={len(predictions)}) and ground_truths (len={len(ground_truths)}) '
            'must have the same length'
        )
    if not predictions:
        return 0.0

    correct = 0
    for pred, gold in zip(predictions, ground_truths, strict=True):
        is_abs = pred if isinstance(pred, bool) else is_abstention_text(pred)
        if is_abs == gold:
            correct += 1

    return float(correct) / len(predictions)
