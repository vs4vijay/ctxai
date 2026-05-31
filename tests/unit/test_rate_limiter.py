"""Tests for ctxai.service.rate_limiter."""

import asyncio
import time

import pytest

from ctxai.service.rate_limiter import InMemoryBackend, RateLimiter


@pytest.mark.asyncio
async def test_under_limit_allowed():
    limiter = RateLimiter()
    for _ in range(3):
        r = await limiter.check("key", max_requests=5, window_seconds=10)
        assert r.allowed is True
        assert r.remaining >= 0


@pytest.mark.asyncio
async def test_over_limit_blocked():
    limiter = RateLimiter()
    for _ in range(3):
        await limiter.check("k", max_requests=3, window_seconds=10)
    r = await limiter.check("k", max_requests=3, window_seconds=10)
    assert r.allowed is False
    assert r.remaining == 0


@pytest.mark.asyncio
async def test_keys_isolated():
    limiter = RateLimiter()
    for _ in range(5):
        await limiter.check("a", max_requests=1, window_seconds=10)
    r = await limiter.check("b", max_requests=1, window_seconds=10)
    assert r.allowed is True


@pytest.mark.asyncio
async def test_reset_clears():
    limiter = RateLimiter()
    for _ in range(5):
        await limiter.check("k", max_requests=1, window_seconds=10)
    await limiter.reset("k")
    r = await limiter.check("k", max_requests=1, window_seconds=10)
    assert r.allowed is True


@pytest.mark.asyncio
async def test_window_slides():
    limiter = RateLimiter()
    r = await limiter.check("k", max_requests=1, window_seconds=1)
    assert r.allowed is True
    r = await limiter.check("k", max_requests=1, window_seconds=1)
    assert r.allowed is False
    await asyncio.sleep(1.05)
    r = await limiter.check("k", max_requests=1, window_seconds=1)
    assert r.allowed is True
