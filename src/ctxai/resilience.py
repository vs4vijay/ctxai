"""
Resilience helpers: exponential backoff and a simple circuit breaker.

Used by LLM providers and the agent loop to gracefully handle transient
failures without spamming external APIs.
"""

from __future__ import annotations

import asyncio
import random
import threading
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, TypeVar

T = TypeVar("T")


@dataclass
class RetryConfig:
    max_attempts: int = 3
    initial_delay: float = 1.0
    max_delay: float = 30.0
    multiplier: float = 2.0
    jitter: float = 0.25
    retryable_exceptions: tuple[type[BaseException], ...] = (Exception,)


def compute_delay(attempt: int, config: RetryConfig) -> float:
    """Exponential backoff with full jitter, capped at config.max_delay."""
    base = min(config.max_delay, config.initial_delay * (config.multiplier**attempt))
    if config.jitter > 0:
        base = base * (1.0 + random.uniform(-config.jitter, config.jitter))
    return max(0.0, base)


def retry_sync(fn: Callable[..., T], *args: Any, config: RetryConfig | None = None, **kwargs: Any) -> T:
    """Synchronously retry `fn` with exponential backoff."""
    cfg = config or RetryConfig()
    last_exc: BaseException | None = None
    for attempt in range(cfg.max_attempts):
        try:
            return fn(*args, **kwargs)
        except cfg.retryable_exceptions as exc:
            last_exc = exc
            if attempt == cfg.max_attempts - 1:
                break
            time.sleep(compute_delay(attempt, cfg))
    assert last_exc is not None
    raise last_exc


async def retry_async(
    fn: Callable[..., Awaitable[T]],
    *args: Any,
    config: RetryConfig | None = None,
    **kwargs: Any,
) -> T:
    """Asynchronously retry `fn` with exponential backoff."""
    cfg = config or RetryConfig()
    last_exc: BaseException | None = None
    for attempt in range(cfg.max_attempts):
        try:
            return await fn(*args, **kwargs)
        except cfg.retryable_exceptions as exc:
            last_exc = exc
            if attempt == cfg.max_attempts - 1:
                break
            await asyncio.sleep(compute_delay(attempt, cfg))
    assert last_exc is not None
    raise last_exc


class CircuitBreakerOpen(Exception):
    """Raised when a call is attempted while the breaker is open."""


@dataclass
class CircuitBreaker:
    """
    Minimal circuit breaker.

    - CLOSED: calls flow through; failures increment a counter.
    - OPEN:   reject calls until `reset_after` elapses.
    - HALF_OPEN: allow one trial call; success closes, failure re-opens.
    """

    failure_threshold: int = 5
    reset_after: float = 30.0
    _failures: int = 0
    _state: str = "CLOSED"  # CLOSED | OPEN | HALF_OPEN
    _opened_at: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def allow(self) -> bool:
        with self._lock:
            if self._state == "OPEN":
                if time.monotonic() - self._opened_at >= self.reset_after:
                    self._state = "HALF_OPEN"
                    return True
                return False
            return True

    def record_success(self) -> None:
        with self._lock:
            self._failures = 0
            self._state = "CLOSED"

    def record_failure(self) -> None:
        with self._lock:
            self._failures += 1
            if self._state == "HALF_OPEN" or self._failures >= self.failure_threshold:
                self._state = "OPEN"
                self._opened_at = time.monotonic()

    @property
    def state(self) -> str:
        with self._lock:
            return self._state

    def call(self, fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        if not self.allow():
            raise CircuitBreakerOpen(f"Circuit breaker open (failures={self._failures})")
        try:
            result = fn(*args, **kwargs)
        except Exception:
            self.record_failure()
            raise
        else:
            self.record_success()
            return result

    async def call_async(self, fn: Callable[..., Awaitable[T]], *args: Any, **kwargs: Any) -> T:
        if not self.allow():
            raise CircuitBreakerOpen(f"Circuit breaker open (failures={self._failures})")
        try:
            result = await fn(*args, **kwargs)
        except Exception:
            self.record_failure()
            raise
        else:
            self.record_success()
            return result
