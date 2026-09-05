"""
Rate limiting for the service layer.

Default backend is in-process (sliding window). Designed to be swapped
for Redis without changing callers.
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from dataclasses import dataclass
from typing import Protocol


class RateLimitBackend(Protocol):
    async def increment(self, key: str, window_seconds: int) -> int: ...
    async def reset(self, key: str) -> None: ...


class InMemoryBackend:
    """Sliding-window counter in-memory backend."""

    def __init__(self) -> None:
        self._windows: dict[str, deque[float]] = {}
        self._lock = asyncio.Lock()

    async def increment(self, key: str, window_seconds: int) -> int:
        now = time.monotonic()
        async with self._lock:
            window = self._windows.setdefault(key, deque())
            cutoff = now - window_seconds
            while window and window[0] < cutoff:
                window.popleft()
            window.append(now)
            return len(window)

    async def reset(self, key: str) -> None:
        async with self._lock:
            self._windows.pop(key, None)


@dataclass
class RateLimitResult:
    allowed: bool
    limit: int
    remaining: int
    reset_at: float


class RateLimiter:
    """High-level rate limiter; pairs a backend with a quota."""

    def __init__(self, backend: RateLimitBackend | None = None):
        self.backend = backend or InMemoryBackend()

    async def check(self, key: str, max_requests: int, window_seconds: int) -> RateLimitResult:
        count = await self.backend.increment(key, window_seconds)
        return RateLimitResult(
            allowed=count <= max_requests,
            limit=max_requests,
            remaining=max(0, max_requests - count),
            reset_at=time.time() + window_seconds,
        )

    async def reset(self, key: str) -> None:
        await self.backend.reset(key)
