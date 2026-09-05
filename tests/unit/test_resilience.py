"""Tests for ctxai.resilience."""

import time

import pytest

from ctxai.resilience import (
    CircuitBreaker,
    CircuitBreakerOpen,
    RetryConfig,
    compute_delay,
    retry_async,
    retry_sync,
)


def test_compute_delay_grows_exponentially():
    cfg = RetryConfig(initial_delay=1.0, multiplier=2.0, jitter=0.0, max_delay=100.0)
    assert compute_delay(0, cfg) == 1.0
    assert compute_delay(1, cfg) == 2.0
    assert compute_delay(2, cfg) == 4.0


def test_compute_delay_respects_max():
    cfg = RetryConfig(initial_delay=10.0, multiplier=10.0, jitter=0.0, max_delay=5.0)
    assert compute_delay(5, cfg) == 5.0


def test_retry_sync_succeeds_after_failure():
    counter = {"calls": 0}

    def flaky():
        counter["calls"] += 1
        if counter["calls"] < 3:
            raise RuntimeError("nope")
        return "ok"

    out = retry_sync(flaky, config=RetryConfig(max_attempts=3, initial_delay=0.0, jitter=0.0))
    assert out == "ok"
    assert counter["calls"] == 3


def test_retry_sync_gives_up_after_max():
    def always_fail():
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        retry_sync(always_fail, config=RetryConfig(max_attempts=2, initial_delay=0.0, jitter=0.0))


@pytest.mark.asyncio
async def test_retry_async_succeeds_after_failure():
    counter = {"calls": 0}

    async def flaky():
        counter["calls"] += 1
        if counter["calls"] < 2:
            raise RuntimeError("nope")
        return "ok"

    out = await retry_async(flaky, config=RetryConfig(max_attempts=3, initial_delay=0.0, jitter=0.0))
    assert out == "ok"


def test_circuit_breaker_opens_after_threshold():
    cb = CircuitBreaker(failure_threshold=2, reset_after=60.0)
    for _ in range(2):
        try:
            cb.call(lambda: (_ for _ in ()).throw(RuntimeError("x")))
        except RuntimeError:
            pass
    assert cb.state == "OPEN"
    with pytest.raises(CircuitBreakerOpen):
        cb.call(lambda: "no")


def test_circuit_breaker_half_open_after_reset():
    cb = CircuitBreaker(failure_threshold=1, reset_after=0.01)
    try:
        cb.call(lambda: (_ for _ in ()).throw(RuntimeError("x")))
    except RuntimeError:
        pass
    assert cb.state == "OPEN"
    time.sleep(0.02)
    assert cb.allow() is True
    cb.record_success()
    assert cb.state == "CLOSED"
