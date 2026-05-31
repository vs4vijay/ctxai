"""
Lightweight in-process metrics collection for ctxai.

Captures counters, gauges, and histograms without external dependencies.
A Prometheus exporter (text format) is available for the service layer.
Thread-safe via a single internal lock.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field


def _labels_key(labels: dict[str, str] | None) -> tuple[tuple[str, str], ...]:
    if not labels:
        return ()
    return tuple(sorted(labels.items()))


@dataclass
class _Histogram:
    values: list[float] = field(default_factory=list)
    count: int = 0
    total: float = 0.0

    def observe(self, value: float) -> None:
        self.values.append(value)
        self.count += 1
        self.total += value
        # Cap memory at 5000 samples per histogram.
        if len(self.values) > 5000:
            self.values = self.values[-5000:]

    def snapshot(self) -> dict[str, float]:
        if not self.values:
            return {"count": 0, "sum": 0.0, "avg": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0}
        sorted_vals = sorted(self.values)
        n = len(sorted_vals)

        def pct(p: float) -> float:
            idx = max(0, min(n - 1, int(p * n)))
            return sorted_vals[idx]

        return {
            "count": self.count,
            "sum": self.total,
            "avg": self.total / self.count if self.count else 0.0,
            "p50": pct(0.50),
            "p95": pct(0.95),
            "p99": pct(0.99),
        }


class MetricsRegistry:
    """In-memory registry for counters, gauges, and histograms."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: dict[tuple[str, tuple], float] = defaultdict(float)
        self._gauges: dict[tuple[str, tuple], float] = {}
        self._histograms: dict[tuple[str, tuple], _Histogram] = defaultdict(_Histogram)
        self._start_time = time.monotonic()

    # ----- Counters -----

    def increment(self, name: str, value: float = 1.0, labels: dict[str, str] | None = None) -> None:
        key = (name, _labels_key(labels))
        with self._lock:
            self._counters[key] += value

    def counter(self, name: str, labels: dict[str, str] | None = None) -> float:
        key = (name, _labels_key(labels))
        with self._lock:
            return self._counters.get(key, 0.0)

    # ----- Gauges -----

    def set_gauge(self, name: str, value: float, labels: dict[str, str] | None = None) -> None:
        key = (name, _labels_key(labels))
        with self._lock:
            self._gauges[key] = value

    def gauge(self, name: str, labels: dict[str, str] | None = None) -> float:
        key = (name, _labels_key(labels))
        with self._lock:
            return self._gauges.get(key, 0.0)

    # ----- Histograms -----

    def observe(self, name: str, value: float, labels: dict[str, str] | None = None) -> None:
        key = (name, _labels_key(labels))
        with self._lock:
            self._histograms[key].observe(value)

    @contextmanager
    def timer(self, name: str, labels: dict[str, str] | None = None) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            self.observe(name, time.perf_counter() - start, labels)

    def histogram_snapshot(self, name: str, labels: dict[str, str] | None = None) -> dict[str, float]:
        key = (name, _labels_key(labels))
        with self._lock:
            hist = self._histograms.get(key)
            return hist.snapshot() if hist else _Histogram().snapshot()

    # ----- Exporters -----

    def uptime_seconds(self) -> float:
        return time.monotonic() - self._start_time

    def snapshot(self) -> dict[str, dict]:
        with self._lock:
            return {
                "counters": {
                    f"{name}{self._format_labels(labels)}": value
                    for (name, labels), value in self._counters.items()
                },
                "gauges": {
                    f"{name}{self._format_labels(labels)}": value
                    for (name, labels), value in self._gauges.items()
                },
                "histograms": {
                    f"{name}{self._format_labels(labels)}": hist.snapshot()
                    for (name, labels), hist in self._histograms.items()
                },
                "uptime_seconds": self.uptime_seconds(),
            }

    def reset(self) -> None:
        with self._lock:
            self._counters.clear()
            self._gauges.clear()
            self._histograms.clear()

    @staticmethod
    def _format_labels(labels: tuple[tuple[str, str], ...]) -> str:
        if not labels:
            return ""
        return "{" + ",".join(f'{k}="{v}"' for k, v in labels) + "}"

    def to_prometheus(self) -> str:
        """Render registered metrics in Prometheus text exposition format."""
        lines: list[str] = []
        with self._lock:
            for (name, labels), value in self._counters.items():
                lines.append(f"# TYPE {name} counter")
                lines.append(f"{name}{self._format_labels(labels)} {value}")
            for (name, labels), value in self._gauges.items():
                lines.append(f"# TYPE {name} gauge")
                lines.append(f"{name}{self._format_labels(labels)} {value}")
            for (name, labels), hist in self._histograms.items():
                snap = hist.snapshot()
                base = f"{name}{self._format_labels(labels)}"
                lines.append(f"# TYPE {name} summary")
                lines.append(f"{name}_count{self._format_labels(labels)} {snap['count']}")
                lines.append(f"{name}_sum{self._format_labels(labels)} {snap['sum']}")
                for quantile in ("p50", "p95", "p99"):
                    q = {"p50": "0.5", "p95": "0.95", "p99": "0.99"}[quantile]
                    quantile_labels = dict(labels) if labels else {}
                    quantile_labels["quantile"] = q
                    label_str = self._format_labels(tuple(sorted(quantile_labels.items())))
                    lines.append(f"{name}{label_str} {snap[quantile]}")
        lines.append("# TYPE ctxai_uptime_seconds gauge")
        lines.append(f"ctxai_uptime_seconds {self.uptime_seconds()}")
        return "\n".join(lines) + "\n"


_registry = MetricsRegistry()


def get_metrics() -> MetricsRegistry:
    return _registry


# Common metric names — keep them centralized so providers and tools agree.
LLM_LATENCY = "ctxai_llm_latency_seconds"
LLM_TOKENS = "ctxai_tokens_total"
LLM_ERRORS = "ctxai_llm_errors_total"
TOOL_DURATION = "ctxai_tool_execution_seconds"
TOOL_CALLS = "ctxai_tool_calls_total"
AGENT_ITERATIONS = "ctxai_agent_iterations_total"
ACTIVE_SESSIONS = "ctxai_active_sessions"
REQUESTS_TOTAL = "ctxai_requests_total"
REQUEST_DURATION = "ctxai_request_duration_seconds"
