"""Deterministic retrieval-quality metric math (RE-01).

Pure functions over plain lists/sets so they are directly unit-testable.
All functions are tie-safe (position-based, no score comparisons) and
deterministic. Metrics that cannot be computed return ``None`` so callers can
mark them explicitly unavailable instead of reporting zero.
"""

from __future__ import annotations

import math
import random
import re
from collections.abc import Callable

_TOKEN_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _tokens(text: str) -> set[str]:
    """Extract the lowercase identifier tokens of a text chunk.

    Args:
        text: Chunk content.

    Returns:
        Set of identifier tokens (mirrors ``repository_context`` tokenization).
    """
    return {term.lower() for term in _TOKEN_PATTERN.findall(text)}


def recall_at_k(expected: set[str], ranked: list[str], k: int) -> float:
    """Fraction of expected files found within the top-k ranked positions.

    Args:
        expected: Expected file paths for the case.
        ranked: Retrieved file paths in final rank order.
        k: Rank cutoff (positions 1..k count).

    Returns:
        ``|expected ∩ ranked[:k]| / |expected|``; 0.0 when nothing matches or
        the expected set is empty (callers decide unavailability separately).
    """
    if not expected:
        return 0.0
    window = ranked[:k] if k > 0 else []
    hits = len(expected.intersection(window))
    return hits / len(expected)


def reciprocal_rank(relevant: set[str], ranked: list[str]) -> float:
    """Reciprocal rank of the first relevant item in a ranked list.

    Args:
        relevant: Paths counted as relevant (grade >= 1).
        ranked: Retrieved paths in rank order.

    Returns:
        ``1 / rank`` of the first relevant item, or 0.0 when none appears.
    """
    for rank, item in enumerate(ranked, 1):
        if item in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(grades: list[int], k: int) -> float:
    """Normalized discounted cumulative gain at rank cutoff k.

    Uses the exponential gain ``2^grade - 1`` and the standard log2 discount.
    The ideal ranking (IDCG) is the graded list sorted descending and
    truncated to k, so tied grades are handled naturally by position.

    Args:
        grades: Relevance grade per ranked position (0 = not relevant).
        k: Rank cutoff.

    Returns:
        nDCG in ``[0.0, 1.0]``; 0.0 when no grade is positive.
    """
    if k <= 0 or not grades:
        return 0.0

    def dcg(values: list[int]) -> float:
        total = 0.0
        for rank, grade in enumerate(values[:k], 1):
            if grade > 0:
                total += (2**grade - 1) / math.log2(rank + 1)
        return total

    ideal = sorted(grades, reverse=True)
    idcg = dcg(ideal)
    if idcg <= 0.0:
        return 0.0
    return dcg(grades) / idcg


def evidence_precision_at_k(relevant: set[str], selected: list[str], k: int) -> float:
    """Precision of the top-k selected context citations at file granularity.

    The denominator is the constant ``k`` (not the number of selected items),
    so an under-filled context is measured honestly as wasted budget.

    Args:
        relevant: File paths counted as relevant (grade >= 1).
        selected: Selected context file paths in selection order.
        k: Cutoff (fixed at 5 for the shipped metric).

    Returns:
        Relevant citations in the top-k divided by k.
    """
    if k <= 0:
        return 0.0
    window = selected[:k]
    hits = sum(1 for item in window if item in relevant)
    return hits / k


def percentile(values: list[float], pct: float) -> float | None:
    """Linear-interpolated percentile of a list of measurements.

    Args:
        values: Measurements (empty means unavailable).
        pct: Percentile in ``[0, 100]``.

    Returns:
        The interpolated percentile, or ``None`` when no values exist.
    """
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * (pct / 100.0)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[int(position)]
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def mean(values: list[float]) -> float | None:
    """Arithmetic mean of a list of measurements.

    Args:
        values: Measurements (empty means unavailable).

    Returns:
        The mean, or ``None`` when no values exist.
    """
    if not values:
        return None
    return sum(values) / len(values)


def duplicate_token_ratio(contents: list[str]) -> float | None:
    """Share of duplicated identifier tokens across the selected context.

    Definition: for each token ``t``, let ``c_t`` be the number of selected
    chunks containing it. The ratio is
    ``(sum_t c_t - |{t : c_t >= 1}|) / max(sum_t c_t, 1)`` — the fraction of
    cross-chunk token occurrences that repeat a token already present in an
    earlier chunk. Within-chunk repetition is ignored; a single chunk has
    ratio 0.0. With no non-empty selected content the metric cannot be
    computed and ``None`` is returned.

    Args:
        contents: Content strings of the selected context in selection order.

    Returns:
        Duplication ratio in ``[0.0, 1.0]``, or ``None`` when unavailable.
    """
    chunk_tokens = [_tokens(content) for content in contents if _tokens(content)]
    if not chunk_tokens:
        return None
    total = sum(len(tokens) for tokens in chunk_tokens)
    if total == 0:
        return None
    unique: set[str] = set()
    for tokens in chunk_tokens:
        unique |= tokens
    return (total - len(unique)) / total


def bootstrap_ci(
    values: list[float],
    *,
    samples: int,
    seed: int,
    level: float = 0.95,
    statistic: Callable[[list[float]], float | None] = mean,
) -> tuple[float, float] | None:
    """Deterministic percentile bootstrap confidence interval for the mean.

    Resampling uses ``random.Random(seed)`` so identical inputs and seed
    always produce identical intervals. With ``samples <= 0`` or no values
    the interval is unavailable (``None``).

    Args:
        values: Per-case metric values.
        samples: Number of bootstrap resamples (0 disables).
        seed: Deterministic RNG seed.
        level: Confidence level in ``(0, 1)``.
        statistic: Statistic applied to each resample (default: mean).

    Returns:
        ``(low, high)`` bounds, or ``None`` when unavailable.
    """
    if not values or samples <= 0:
        return None
    center = (1.0 - level) / 2.0
    rng = random.Random(seed)
    stats: list[float] = []
    for _ in range(samples):
        resample = [values[rng.randrange(len(values))] for _ in range(len(values))]
        value = statistic(resample)
        if value is not None:
            stats.append(value)
    if not stats:
        return None
    low = percentile(stats, center * 100.0)
    high = percentile(stats, (1.0 - center) * 100.0)
    if low is None or high is None:
        return None
    return (low, high)
