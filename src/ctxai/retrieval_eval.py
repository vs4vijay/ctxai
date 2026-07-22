"""Small, deterministic retrieval-quality benchmark primitives."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RetrievalMetrics:
    queries: int
    recall_at_5: float
    mrr: float


def evaluate_retrieval(cases: list[dict]) -> RetrievalMetrics:
    """Calculate query hit-rate at five and mean reciprocal rank."""
    if not cases:
        return RetrievalMetrics(queries=0, recall_at_5=0.0, mrr=0.0)
    hits = 0
    reciprocal_ranks = 0.0
    for case in cases:
        expected = set(case["expected_locations"])
        retrieved = case["retrieved_locations"]
        ranks = [rank for rank, location in enumerate(retrieved, 1) if location in expected]
        if any(rank <= 5 for rank in ranks):
            hits += 1
        if ranks:
            reciprocal_ranks += 1 / min(ranks)
    return RetrievalMetrics(
        queries=len(cases),
        recall_at_5=hits / len(cases),
        mrr=reciprocal_ranks / len(cases),
    )
