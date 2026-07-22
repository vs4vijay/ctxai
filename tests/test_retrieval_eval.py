import json
from pathlib import Path

from ctxai.retrieval_eval import evaluate_retrieval


def test_recorded_retrieval_baseline():
    fixture = Path(__file__).parent / "fixtures" / "retrieval_benchmark.json"
    cases = json.loads(fixture.read_text(encoding="utf-8"))
    metrics = evaluate_retrieval(cases)
    assert metrics.queries == 20
    assert metrics.recall_at_5 == 1.0
    assert metrics.mrr >= 0.75
