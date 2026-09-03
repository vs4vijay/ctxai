"""Cost estimation for recorded LLM usage (HH-04).

A small, checked-in price table (``PriceTable``) turns provider-reported token
usage into an estimated USD cost for documented models. Unknown models yield
``None`` — surfaced as "cost unknown", never as a fabricated zero. Prices are
indicative public list prices and can be extended by editing
:data:`PRICES_PER_1M_TOKENS` (see ``docs/RUN_TRANSCRIPTS.md``).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

# USD per 1M tokens as (prompt, completion). Indicative public list prices,
# last reviewed 2026-09. Extend by adding the model id exactly as the
# provider reports it in usage records.
PRICES_PER_1M_TOKENS: dict[str, tuple[float, float]] = {
    # Anthropic (direct model ids)
    "claude-3-5-sonnet-20241022": (3.0, 15.0),
    "claude-3-5-sonnet": (3.0, 15.0),
    "claude-3-opus-20240229": (15.0, 75.0),
    "claude-3-opus": (15.0, 75.0),
    "claude-3-haiku-20240307": (0.25, 1.25),
    "claude-3-haiku": (0.25, 1.25),
    # OpenAI (direct model ids)
    "gpt-4o": (2.5, 10.0),
    "gpt-4o-mini": (0.15, 0.6),
    "gpt-4-turbo": (10.0, 30.0),
    "o1": (15.0, 60.0),
    "o1-preview": (15.0, 60.0),
    "o1-mini": (1.1, 4.4),
    # OpenRouter ids documented in the chat provider catalog
    "anthropic/claude-3.5-sonnet": (3.0, 15.0),
    "anthropic/claude-3-opus": (15.0, 75.0),
    "openai/gpt-4o": (2.5, 10.0),
    "openai/o1-mini": (1.1, 4.4),
    "openai/o1": (15.0, 60.0),
    "deepseek/deepseek-r1": (0.55, 2.19),
    "deepseek/deepseek-chat": (0.14, 0.28),
    "google/gemini-pro-1.5": (1.25, 5.0),
    "meta-llama/llama-3-70b-instruct": (0.59, 0.79),
}

# Vendor prefixes tried stripped when a "vendor/model" id does not match a
# table entry directly (e.g. "openai/gpt-4o-mini" -> "gpt-4o-mini").
_VENDOR_PREFIXES = ("anthropic/", "openai/", "deepseek/", "google/", "meta-llama/")


@dataclass
class RunCostEstimate:
    """Aggregated cost estimate for one run's usage records.

    Attributes:
        total_cost: Estimated USD cost across all records, or ``None`` when
            any record used a model without a price entry.
        unknown_model: The first model id lacking a price entry, if any.
    """

    total_cost: float | None
    unknown_model: str | None


class PriceTable:
    """Static price data and the single cost-estimation entry point."""

    @staticmethod
    def _resolve(model: str) -> str | None:
        """Resolve a model id to a price-table key.

        Exact matches win; otherwise a documented vendor prefix is stripped
        once and retried so ``openai/gpt-4o-mini`` prices like ``gpt-4o-mini``.

        Args:
            model: Model identifier as reported on the usage record.

        Returns:
            The matching price-table key, or ``None`` when unknown.
        """
        key = (model or "").strip()
        if not key:
            return None
        if key in PRICES_PER_1M_TOKENS:
            return key
        for prefix in _VENDOR_PREFIXES:
            if key.startswith(prefix):
                stripped = key[len(prefix) :]
                if stripped in PRICES_PER_1M_TOKENS:
                    return stripped
                break
        return None

    @staticmethod
    def estimate_cost(model: str, usage: dict[str, Any]) -> float | None:
        """Estimate the USD cost of one usage payload against the price table.

        Args:
            model: Model identifier the usage was reported for.
            usage: Provider-reported usage dict (``prompt_tokens`` /
                ``completion_tokens``; other keys are ignored).

        Returns:
            The estimated cost in USD rounded to 6 decimals, ``0.0`` for a
            known model with no tokens, or ``None`` when the model has no
            price entry — unknown cost is never reported as zero.
        """
        key = PriceTable._resolve(model)
        if key is None:
            return None
        prompt_price, completion_price = PRICES_PER_1M_TOKENS[key]
        prompt_tokens = int(usage.get("prompt_tokens") or 0)
        completion_tokens = int(usage.get("completion_tokens") or 0)
        cost = prompt_tokens / 1_000_000 * prompt_price + completion_tokens / 1_000_000 * completion_price
        return round(cost, 6)


def estimate_run_cost(records: Iterable[Any]) -> RunCostEstimate:
    """Estimate the total cost of a run's usage records.

    Records are aggregated per model (token sums), then each model is priced.
    A single model without a price entry makes the whole estimate unknown —
    the result names the first such model rather than fabricating a total.

    Args:
        records: Usage records exposing ``model``, ``prompt_tokens``, and
            ``completion_tokens`` (``workflow.UsageRecord`` qualifies).

    Returns:
        A :class:`RunCostEstimate` with the total or the unknown-model name.
    """
    tokens_by_model: dict[str, list[int]] = {}
    for record in records:
        totals = tokens_by_model.setdefault(str(record.model), [0, 0])
        totals[0] += int(record.prompt_tokens)
        totals[1] += int(record.completion_tokens)

    total = 0.0
    for model, (prompt_tokens, completion_tokens) in tokens_by_model.items():
        cost = PriceTable.estimate_cost(model, {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens})
        if cost is None:
            return RunCostEstimate(total_cost=None, unknown_model=model)
        total += cost
    return RunCostEstimate(total_cost=round(total, 6), unknown_model=None)


def format_unknown_cost(model: str | None) -> str:
    """Render the display wording for an unknown model cost.

    Args:
        model: The model id without a price entry (may be ``None``).

    Returns:
        Text shaped like ``unknown (no price entry for mock-model-v1)``.
    """
    if model:
        return f"unknown (no price entry for {model})"
    return "unknown (model has no price entry)"
