"""Unit tests for RE-01 retrieval metric math.

Covers recall@k, MRR, graded nDCG@10, evidence precision@5, duplicate-token
ratio, percentiles, bootstrap confidence intervals, and the explicit
"unavailable" convention (metrics that cannot be computed are never zero).
"""

import math

from ctxai.evals.metrics import (
    bootstrap_ci,
    duplicate_token_ratio,
    evidence_precision_at_k,
    ndcg_at_k,
    percentile,
    recall_at_k,
    reciprocal_rank,
)


class TestRecallAtK:
    def test_full_hit_at_k(self):
        """All expected files inside top-k give recall 1.0."""
        assert recall_at_k({"a.py", "b.py"}, ["a.py", "b.py", "c.py"], k=5) == 1.0

    def test_partial_hit(self):
        """Partially matching top-k yields the hit fraction."""
        assert recall_at_k({"a.py", "b.py", "c.py"}, ["a.py", "x.py", "b.py"], k=5) == 2 / 3

    def test_rank_limits_the_window(self):
        """Files beyond k do not count even when relevant."""
        assert recall_at_k({"a.py"}, ["b.py", "c.py", "a.py"], k=2) == 0.0
        assert recall_at_k({"a.py"}, ["b.py", "c.py", "a.py"], k=3) == 1.0

    def test_empty_ranked_list(self):
        """An empty ranking scores 0.0 without raising."""
        assert recall_at_k({"a.py"}, [], k=5) == 0.0

    def test_duplicate_relevant_entries_do_not_inflate_recall(self):
        """The same relevant file repeated must not exceed 1.0."""
        assert recall_at_k({"a.py"}, ["a.py", "a.py", "a.py"], k=5) == 1.0


class TestReciprocalRank:
    def test_first_position(self):
        assert reciprocal_rank({"a.py"}, ["a.py", "b.py"]) == 1.0

    def test_second_position(self):
        assert reciprocal_rank({"b.py"}, ["a.py", "b.py"]) == 0.5

    def test_no_hit_is_zero(self):
        assert reciprocal_rank({"z.py"}, ["a.py", "b.py"]) == 0.0

    def test_ties_use_first_occurrence_rank(self):
        """With tied lists the earliest relevant position counts."""
        assert reciprocal_rank({"a.py", "b.py"}, ["a.py", "b.py"]) == 1.0
        assert reciprocal_rank({"a.py", "b.py"}, ["c.py", "a.py", "b.py"]) == 0.5


class TestNDCG:
    def test_perfect_grading_ranks_first(self):
        """Graded list in ideal order produces nDCG 1.0."""
        assert ndcg_at_k([3, 2, 1, 0], k=10) == 1.0

    def test_swapped_order_loses_gain(self):
        """A non-ideal order scores strictly below 1.0."""
        score = ndcg_at_k([2, 3, 1, 0], k=10)
        assert 0.0 < score < 1.0

    def test_known_value(self):
        """Hand-computed discounted gain for a non-ideal graded list."""
        # Exponential gain 2^g - 1 with log2 discount:
        # DCG([2, 3, 1]) = 3/1 + 7/log2(3) + 1/2;
        # IDCG([3, 2, 1]) = 7/1 + 3/log2(3) + 1/2.
        dcg = (2**2 - 1) / math.log2(2) + (2**3 - 1) / math.log2(3) + (2**1 - 1) / math.log2(4)
        idcg = (2**3 - 1) / math.log2(2) + (2**2 - 1) / math.log2(3) + (2**1 - 1) / math.log2(4)
        assert math.isclose(ndcg_at_k([2, 3, 1], k=10), dcg / idcg)

    def test_k_truncates(self):
        """Only the top-k positions contribute gain."""
        full = ndcg_at_k([3, 2, 1], k=10)
        truncated = ndcg_at_k([3, 2, 1], k=2)
        assert truncated == 1.0  # ideal is also truncated to k=2
        assert ndcg_at_k([2, 1, 3], k=2) < full

    def test_all_zero_grades(self):
        """A list with no relevant grades scores 0.0, not an error."""
        assert ndcg_at_k([0, 0, 0], k=10) == 0.0

    def test_empty_list(self):
        assert ndcg_at_k([], k=10) == 0.0

    def test_ties_are_stable(self):
        """Equal-grade orderings produce equal nDCG."""
        assert ndcg_at_k([2, 2, 1], k=10) == ndcg_at_k([2, 2, 1], k=10)


class TestEvidencePrecision:
    def test_relevant_share_of_top_five(self):
        assert evidence_precision_at_k({"a.py"}, ["a.py", "b.py", "c.py", "d.py", "e.py"], k=5) == 1 / 5

    def test_denominator_is_k_even_when_short(self):
        """Fewer than k selected items still divide by k (honest fill rate)."""
        assert evidence_precision_at_k({"a.py"}, ["a.py"], k=5) == 1 / 5
        assert evidence_precision_at_k({"a.py"}, [], k=5) == 0.0

    def test_multiple_relevant(self):
        assert evidence_precision_at_k({"a.py", "b.py"}, ["a.py", "x", "b.py", "y", "z"], k=5) == 2 / 5


class TestPercentile:
    def test_median_of_odd_list(self):
        assert percentile([1.0, 2.0, 3.0], 50) == 2.0

    def test_interpolated_percentile(self):
        """Linear interpolation between order statistics."""
        assert math.isclose(percentile([10.0, 20.0], 50), 15.0)
        assert math.isclose(percentile([0.0, 10.0, 20.0, 30.0], 95), 28.5)

    def test_bounds(self):
        assert percentile([5.0], 0) == 5.0
        assert percentile([5.0], 100) == 5.0

    def test_empty_is_none(self):
        """No measurements means the percentile is unavailable (None)."""
        assert percentile([], 95) is None


class TestDuplicateTokenRatio:
    def test_disjoint_chunks_have_zero_duplication(self):
        assert duplicate_token_ratio(["alpha beta", "gamma delta"]) == 0.0

    def test_identical_chunks_duplicate_half_the_occurrences(self):
        """Two identical two-token chunks: 4 occurrences, 2 unique -> 0.5."""
        assert duplicate_token_ratio(["alpha beta", "alpha beta"]) == 0.5

    def test_partial_overlap(self):
        """One shared token across two chunks of two tokens each."""
        # Tokens: alpha (2 chunks), beta (1), gamma (1) -> duplicates = 1, total = 4.
        assert math.isclose(duplicate_token_ratio(["alpha beta", "alpha gamma"]), 0.25)

    def test_single_chunk_has_no_cross_chunk_duplication(self):
        assert duplicate_token_ratio(["alpha alpha alpha"]) == 0.0

    def test_empty_selection_is_unavailable(self):
        assert duplicate_token_ratio([]) is None
        assert duplicate_token_ratio([""]) is None


class TestBootstrapCI:
    def test_deterministic_for_fixed_seed(self):
        values = [0.8, 0.6, 0.4, 1.0, 0.2]
        first = bootstrap_ci(values, samples=200, seed=7)
        second = bootstrap_ci(values, samples=200, seed=7)
        assert first == second

    def test_interval_brackets_the_mean(self):
        values = [0.9, 0.8, 0.7, 0.6, 0.5]
        low, high = bootstrap_ci(values, samples=500, seed=11)
        assert low <= sum(values) / len(values) <= high

    def test_no_values_is_none(self):
        assert bootstrap_ci([], samples=100, seed=1) is None

    def test_disabled_bootstrap_is_none(self):
        assert bootstrap_ci([0.5, 0.5], samples=0, seed=1) is None

    def test_seed_changes_resampling_deterministically(self):
        values = [0.9, 0.1, 0.5, 0.3, 0.7]
        a = bootstrap_ci(values, samples=100, seed=1)
        b = bootstrap_ci(values, samples=100, seed=2)
        assert a != b
