"""Unit tests for the HH-04 cost ledger (agent/costing.py).

Covers price-table math for known models, the unknown-model None contract,
OpenRouter-style alias resolution, zero-token honesty, run-level aggregation
over usage ledger records, and the display wording for the usage/cost line.
"""

from __future__ import annotations

import pytest

from ctxai.agent.costing import (
    PRICES_PER_1M_TOKENS,
    PriceTable,
    RunCostEstimate,
    estimate_run_cost,
    format_unknown_cost,
)
from ctxai.agent.workflow import UsageRecord


class TestPriceTable:
    def test_known_model_cost_math(self):
        """Cost equals prompt/1M * prompt_price + completion/1M * completion_price."""
        cost = PriceTable.estimate_cost(
            "gpt-4o",
            {"prompt_tokens": 1_000_000, "completion_tokens": 1_000_000},
        )

        assert cost == pytest.approx(2.5 + 10.0)

    def test_small_usage_stays_nonzero_and_rounded(self):
        """Tiny token counts produce a small non-zero estimate, not a fabricated zero."""
        cost = PriceTable.estimate_cost(
            "claude-3-5-sonnet-20241022",
            {"prompt_tokens": 1000, "completion_tokens": 100},
        )

        assert cost is not None
        assert cost > 0
        assert cost == pytest.approx(1000 / 1_000_000 * 3.0 + 100 / 1_000_000 * 15.0, abs=1e-9)

    def test_unknown_model_returns_none_never_zero(self):
        """An unknown model yields None — never a fabricated zero."""
        assert PriceTable.estimate_cost("totally-unknown-model", {"prompt_tokens": 10, "completion_tokens": 10}) is None
        assert PriceTable.estimate_cost("mock-model-v1", {"prompt_tokens": 10, "completion_tokens": 10}) is None
        assert PriceTable.estimate_cost("mock-model-v1", {}) is None

    def test_zero_tokens_on_known_model_is_an_honest_zero(self):
        """Zero usage on a known model computes a real zero — distinct from unknown."""
        cost = PriceTable.estimate_cost("gpt-4o", {"prompt_tokens": 0, "completion_tokens": 0})

        assert cost == 0.0

    def test_empty_usage_dict_on_known_model_is_zero(self):
        """An empty usage payload on a known model computes zero."""
        assert PriceTable.estimate_cost("gpt-4o", {}) == 0.0

    def test_documented_models_are_covered(self):
        """Every model id from the documented provider catalogs has a price entry."""
        documented = {
            "claude-3-5-sonnet-20241022",
            "claude-3-opus-20240229",
            "claude-3-haiku-20240307",
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-4-turbo",
            "o1-mini",
            "o1-preview",
            "anthropic/claude-3.5-sonnet",
            "anthropic/claude-3-opus",
            "openai/gpt-4o",
            "openai/o1-mini",
            "openai/o1",
            "deepseek/deepseek-r1",
            "deepseek/deepseek-chat",
            "google/gemini-pro-1.5",
            "meta-llama/llama-3-70b-instruct",
        }

        missing = documented - set(PRICES_PER_1M_TOKENS)
        assert not missing, f"documented models missing from the price table: {sorted(missing)}"

    def test_vendor_prefixed_ids_resolve_to_bare_entries(self):
        """An id like openai/gpt-4o-mini resolves through the documented alias rule."""
        direct = PriceTable.estimate_cost("gpt-4o-mini", {"prompt_tokens": 1_000_000, "completion_tokens": 0})
        aliased = PriceTable.estimate_cost("openai/gpt-4o-mini", {"prompt_tokens": 1_000_000, "completion_tokens": 0})

        assert direct is not None
        assert aliased == direct


class TestRunCostEstimate:
    def test_estimate_run_cost_single_known_model(self):
        """A single known model aggregates tokens and returns a known total."""
        records = [
            UsageRecord(
                provider="P", model="gpt-4o", prompt_tokens=500_000, completion_tokens=100_000, total_tokens=600_000
            ),
            UsageRecord(
                provider="P", model="gpt-4o", prompt_tokens=500_000, completion_tokens=100_000, total_tokens=600_000
            ),
        ]

        estimate = estimate_run_cost(records)

        assert isinstance(estimate, RunCostEstimate)
        assert estimate.unknown_model is None
        assert estimate.total_cost == pytest.approx(2.5 + 2.0)

    def test_estimate_run_cost_unknown_model_surfaces_name(self):
        """Any unknown model makes the run estimate explicitly unknown and names the model."""
        records = [
            UsageRecord(provider="P", model="gpt-4o", prompt_tokens=10, completion_tokens=0, total_tokens=10),
            UsageRecord(provider="P", model="mystery-model", prompt_tokens=10, completion_tokens=0, total_tokens=10),
        ]

        estimate = estimate_run_cost(records)

        assert estimate.total_cost is None
        assert estimate.unknown_model == "mystery-model"

    def test_estimate_run_cost_empty_ledger(self):
        """An empty ledger yields a known zero — no tokens were billed."""
        estimate = estimate_run_cost([])

        assert estimate.total_cost == 0.0
        assert estimate.unknown_model is None


class TestDisplayWording:
    def test_format_unknown_cost_wording(self):
        """The unknown-cost wording names the model and never shows a dollar figure."""
        text = format_unknown_cost("mock-model-v1")

        assert "unknown" in text
        assert "mock-model-v1" in text
        assert "$" not in text
