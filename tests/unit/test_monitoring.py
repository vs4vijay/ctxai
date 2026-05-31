"""Tests for ctxai.monitoring metrics registry."""

import time

from ctxai.monitoring import MetricsRegistry, get_metrics


def test_counter_increments():
    registry = MetricsRegistry()
    registry.increment("calls")
    registry.increment("calls", 4)
    assert registry.counter("calls") == 5


def test_counter_with_labels_isolated():
    registry = MetricsRegistry()
    registry.increment("requests", labels={"endpoint": "/a"})
    registry.increment("requests", labels={"endpoint": "/b"})
    assert registry.counter("requests", labels={"endpoint": "/a"}) == 1
    assert registry.counter("requests", labels={"endpoint": "/b"}) == 1


def test_gauges_overwrite_previous_value():
    registry = MetricsRegistry()
    registry.set_gauge("queue", 5)
    registry.set_gauge("queue", 7)
    assert registry.gauge("queue") == 7


def test_histogram_observations_and_snapshot():
    registry = MetricsRegistry()
    for v in [0.1, 0.2, 0.3, 0.4, 0.5]:
        registry.observe("latency", v)
    snap = registry.histogram_snapshot("latency")
    assert snap["count"] == 5
    assert snap["sum"] == pytest_approx_close(1.5)
    assert snap["p50"] >= 0.2
    assert snap["p95"] >= 0.4


def test_timer_context_records_duration():
    registry = MetricsRegistry()
    with registry.timer("op"):
        time.sleep(0.01)
    snap = registry.histogram_snapshot("op")
    assert snap["count"] == 1
    assert snap["sum"] > 0


def test_prometheus_export_contains_metrics():
    registry = MetricsRegistry()
    registry.increment("x")
    registry.set_gauge("g", 9)
    registry.observe("h", 1.0)
    text = registry.to_prometheus()
    assert "# TYPE x counter" in text
    assert "# TYPE g gauge" in text
    assert "h_count" in text
    assert "ctxai_uptime_seconds" in text


def test_reset_clears_state():
    registry = MetricsRegistry()
    registry.increment("a")
    registry.set_gauge("b", 1)
    registry.observe("c", 1.0)
    registry.reset()
    assert registry.counter("a") == 0
    assert registry.gauge("b") == 0
    assert registry.histogram_snapshot("c")["count"] == 0


def test_global_get_metrics_returns_singleton():
    a = get_metrics()
    b = get_metrics()
    assert a is b


def pytest_approx_close(value, tol=1e-6):
    class _Close:
        def __eq__(self, other):
            return abs(other - value) < tol
    return _Close()
