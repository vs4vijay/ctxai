"""Unit tests for agent resilience primitives (HH-02).

Covers RetryPolicy defaults and serialization, the call_with_retry backoff
sequence with injected sleep/rng, retryability gating per ProviderErrorKind,
cancellation between attempts, the retry-notice UI contract, and the
provider-kind mapping into the FailureKind taxonomy and MCP envelope codes.
"""

from __future__ import annotations

import asyncio
import random

import pytest

from ctxai.agent.config import AgentBehaviorConfig
from ctxai.agent.llm.base import ProviderError, ProviderErrorKind
from ctxai.agent.resilience import (
    RetryNotice,
    RetryPolicy,
    backoff_delay,
    call_with_retry,
    format_retry_notice,
)
from ctxai.agent.workflow import FailureKind, classify_provider_failure
from ctxai.commands.server_command import _provider_error_code
from ctxai.mcp_protocol import MCPErrorCode


class SleepRecorder:
    """Async sleep stand-in that records requested delays without waiting."""

    def __init__(self) -> None:
        self.delays: list[float] = []

    async def __call__(self, delay: float) -> None:
        self.delays.append(delay)


class MaxJitterRandom(random.Random):
    """Deterministic rng whose uniform(a, b) always returns the ceiling."""

    def uniform(self, a: float, b: float) -> float:  # noqa: A003 - matches stdlib signature
        return b


def make_policy(**overrides) -> RetryPolicy:
    """Build a RetryPolicy with test-sized defaults.

    Args:
        overrides: RetryPolicy field overrides.

    Returns:
        The configured RetryPolicy.
    """
    values = {"max_retries": 3, "base_delay_s": 1.0, "max_delay_s": 30.0}
    values.update(overrides)
    return RetryPolicy(**values)


async def failing_fn(errors: list[Exception | str], calls: list[int]):
    """Return an async callable that raises each error once, then succeeds.

    Args:
        errors: Exceptions to raise in order; a string raises a ValueError.
        calls: List mutated to record invocation counts.

    Returns:
        An async zero-argument callable returning ``"ok"`` on success.
    """

    async def _fn():
        calls.append(1)
        if errors:
            item = errors.pop(0)
            if isinstance(item, Exception):
                raise item
            raise ValueError(item)
        return "ok"

    return _fn


def rate_limit(message: str = "429 slow down") -> ProviderError:
    """Build a rate-limit ProviderError.

    Args:
        message: Error text.

    Returns:
        A ProviderError with kind RATE_LIMIT.
    """
    return ProviderError(ProviderErrorKind.RATE_LIMIT, message, provider="FakeProvider")


def retry_all(error: Exception) -> bool:
    """Retry every exception.

    Args:
        error: The raised exception.

    Returns:
        Always True.
    """
    return True


def test_retry_policy_defaults_match_spec():
    """RetryPolicy defaults must match the HH-02 contract."""
    policy = RetryPolicy()
    assert policy.max_retries == 3
    assert policy.base_delay_s == 1.0
    assert policy.max_delay_s == 30.0
    assert policy.retry_kinds == {ProviderErrorKind.RATE_LIMIT, ProviderErrorKind.TIMEOUT, ProviderErrorKind.TRANSPORT}


def test_retry_policy_round_trip_serialization():
    """RetryPolicy serializes to a dict and restores identical values."""
    policy = RetryPolicy(max_retries=5, base_delay_s=0.5, max_delay_s=10.0)
    data = policy.to_dict()
    assert data["max_retries"] == 5
    assert data["retry_kinds"] == ["rate_limit", "timeout", "transport"]
    restored = RetryPolicy.from_dict(data)
    assert restored == policy


def test_retry_policy_is_retryable():
    """Only the declared retry kinds are retryable."""
    policy = RetryPolicy()
    assert policy.is_retryable(ProviderErrorKind.RATE_LIMIT)
    assert policy.is_retryable(ProviderErrorKind.TIMEOUT)
    assert policy.is_retryable(ProviderErrorKind.TRANSPORT)
    assert not policy.is_retryable(ProviderErrorKind.AUTHENTICATION)
    assert not policy.is_retryable(ProviderErrorKind.UNSUPPORTED)
    assert not policy.is_retryable(ProviderErrorKind.INVALID_RESPONSE)
    assert not policy.is_retryable(ProviderErrorKind.CANCELLED)


async def test_call_with_retry_succeeds_after_transient_failures():
    """Transient retryable failures are retried with bounded exponential backoff."""
    calls: list[int] = []
    sleep = SleepRecorder()
    fn = await failing_fn([rate_limit("first"), rate_limit("second")], calls)

    result = await call_with_retry(
        fn,
        policy=make_policy(),
        should_retry=retry_all,
        sleep=sleep,
        rng=MaxJitterRandom(),
    )

    assert result == "ok"
    assert len(calls) == 3, "two failed attempts plus the successful one"
    assert sleep.delays == [1.0, 2.0], "backoff doubles per attempt with the ceiling returned by the fake rng"


async def test_call_with_retry_backoff_stays_within_declared_bounds():
    """With full jitter, every delay lies in [0, min(max_delay, base * 2**attempt)]."""
    calls: list[int] = []
    sleep = SleepRecorder()
    policy = make_policy(base_delay_s=2.0, max_delay_s=5.0)
    fn = await failing_fn([rate_limit()] * 3, calls)

    await call_with_retry(fn, policy=policy, should_retry=retry_all, sleep=sleep, rng=random.Random(7))

    ceilings = [2.0, 4.0, 5.0]  # capped by max_delay_s = 5.0
    assert len(sleep.delays) == 3
    for delay, ceiling in zip(sleep.delays, ceilings):
        assert 0 <= delay <= ceiling


async def test_call_with_retry_exhaustion_raises_last_error():
    """When retries are exhausted the last exception is raised."""
    calls: list[int] = []
    sleep = SleepRecorder()
    errors = [ProviderError(ProviderErrorKind.TIMEOUT, f"timeout {i}") for i in range(3)]
    fn = await failing_fn(list(errors), calls)

    with pytest.raises(ProviderError) as excinfo:
        await call_with_retry(
            fn,
            policy=make_policy(max_retries=2),
            should_retry=retry_all,
            sleep=sleep,
            rng=MaxJitterRandom(),
        )

    assert excinfo.value is errors[-1], "the last raised error is surfaced"
    assert len(calls) == 3, "max_retries=2 means one initial attempt plus two retries"
    assert len(sleep.delays) == 2


async def test_call_with_retry_non_retryable_error_raises_immediately():
    """A non-retryable error propagates on the first attempt with no waits."""
    calls: list[int] = []
    sleep = SleepRecorder()
    auth_error = ProviderError(ProviderErrorKind.AUTHENTICATION, "invalid api key")
    fn = await failing_fn([auth_error], calls)

    with pytest.raises(ProviderError) as excinfo:
        await call_with_retry(
            fn,
            policy=make_policy(),
            should_retry=lambda error: error.kind in RetryPolicy().retry_kinds,
            sleep=sleep,
            rng=MaxJitterRandom(),
        )

    assert excinfo.value is auth_error
    assert len(calls) == 1
    assert sleep.delays == []


async def test_call_with_retry_cancel_event_before_start():
    """A pre-set cancel event raises CancelledError before any attempt."""
    calls: list[int] = []
    sleep = SleepRecorder()
    fn = await failing_fn([], calls)
    cancel_event = asyncio.Event()
    cancel_event.set()

    with pytest.raises(asyncio.CancelledError):
        await call_with_retry(
            fn,
            policy=make_policy(),
            should_retry=retry_all,
            sleep=sleep,
            rng=MaxJitterRandom(),
            cancel_event=cancel_event,
        )

    assert calls == []
    assert sleep.delays == []


async def test_call_with_retry_cancel_between_attempts():
    """Setting the cancel event between attempts cancels cleanly instead of retrying."""
    calls: list[int] = []
    sleep = SleepRecorder()
    cancel_event = asyncio.Event()
    error = rate_limit("transient")

    async def fn():
        calls.append(1)
        cancel_event.set()  # cancelled while "waiting" between attempts
        raise error

    with pytest.raises(asyncio.CancelledError):
        await call_with_retry(
            fn,
            policy=make_policy(),
            should_retry=retry_all,
            sleep=sleep,
            rng=MaxJitterRandom(),
            cancel_event=cancel_event,
        )

    assert len(calls) == 1, "no second attempt after cancellation"
    assert sleep.delays == [], "no backoff sleep runs after cancellation"


async def test_call_with_retry_fires_retry_notice_with_kind():
    """Each retry wait emits a structured notice for the CLI surfaces."""
    calls: list[int] = []
    sleep = SleepRecorder()
    notices: list[RetryNotice] = []
    fn = await failing_fn([rate_limit(), ProviderError(ProviderErrorKind.TIMEOUT, "timed out")], calls)

    await call_with_retry(
        fn,
        policy=make_policy(max_retries=3),
        should_retry=retry_all,
        sleep=sleep,
        rng=MaxJitterRandom(),
        on_retry=notices.append,
    )

    assert [notice.attempt for notice in notices] == [1, 2]
    assert [notice.kind for notice in notices] == ["rate_limit", "timeout"]
    assert all(notice.max_retries == 3 for notice in notices)
    assert all(0 <= notice.delay_s <= 30.0 for notice in notices)


def test_format_retry_notice_matches_ui_contract():
    """The rendered retry line matches the documented CLI contract."""
    notice = RetryNotice(attempt=2, max_retries=3, delay_s=2.1, kind="rate_limit")
    assert format_retry_notice(notice) == "retry 2/3 after 2.1s (rate_limit)"


def test_backoff_delay_is_exponential_and_capped():
    """backoff_delay doubles per attempt and is capped at max_delay_s."""
    policy = make_policy(base_delay_s=1.0, max_delay_s=4.0)
    rng = MaxJitterRandom()
    assert backoff_delay(policy, 0, rng) == 1.0
    assert backoff_delay(policy, 1, rng) == 2.0
    assert backoff_delay(policy, 2, rng) == 4.0
    assert backoff_delay(policy, 3, rng) == 4.0, "capped at max_delay_s"


def test_behavior_config_round_trips_loop_break_threshold():
    """AgentBehaviorConfig serializes the new loop_break_threshold field."""
    config = AgentBehaviorConfig(loop_break_threshold=5)
    data = config.to_dict()
    assert data["loop_break_threshold"] == 5
    restored = AgentBehaviorConfig.from_dict(data)
    assert restored.loop_break_threshold == 5
    assert AgentBehaviorConfig.from_dict({}).loop_break_threshold == 3, "default is 3"


def test_normalize_error_maps_provider_failures():
    """normalize_error classifies common provider failure shapes."""
    from ctxai.agent.config import AgentLLMConfig
    from tests.mocks.mock_llm import MockLLMProvider

    provider = MockLLMProvider(config=AgentLLMConfig(provider="mock", model="m", api_key="k"))

    assert provider.normalize_error(RuntimeError("authentication failed: bad api key")).kind is (
        ProviderErrorKind.AUTHENTICATION
    )
    assert provider.normalize_error(RuntimeError("401 unauthorized")).kind is ProviderErrorKind.AUTHENTICATION
    assert provider.normalize_error(RuntimeError("Error code: 429 too many requests")).kind is (
        ProviderErrorKind.RATE_LIMIT
    )
    assert provider.normalize_error(RuntimeError("request timed out after 60s")).kind is ProviderErrorKind.TIMEOUT
    assert provider.normalize_error(ValueError("invalid json at position 0")).kind is (
        ProviderErrorKind.INVALID_RESPONSE
    )
    transport = provider.normalize_error(ConnectionError("connection reset"))
    assert transport.kind is ProviderErrorKind.TRANSPORT
    assert transport.provider == "MockLLMProvider"


def test_classify_provider_failure_uses_failure_kind_taxonomy():
    """Every ProviderErrorKind maps into the shared FailureKind taxonomy."""
    for kind in ProviderErrorKind:
        assert classify_provider_failure(kind) is FailureKind.INFRASTRUCTURE_FAILURE


def test_mcp_maps_provider_cancelled_to_cancelled_envelope():
    """MCP maps ProviderErrorKind values onto the stable envelope codes."""
    assert _provider_error_code(ProviderError(ProviderErrorKind.CANCELLED, "cancelled")) == MCPErrorCode.CANCELLED
    assert _provider_error_code(ProviderError(ProviderErrorKind.TIMEOUT, "timed out")) == MCPErrorCode.TIMEOUT
    assert (
        _provider_error_code(ProviderError(ProviderErrorKind.AUTHENTICATION, "bad key")) == MCPErrorCode.INTERNAL_ERROR
    )
    assert _provider_error_code(ValueError("not a provider error")) is None
