"""Resilience primitives for the agent loop (HH-02).

Provides the in-memory :class:`RetryPolicy`, exponential backoff with full
jitter, and :func:`call_with_retry` — the single retry entry point used by the
agent loop for provider calls. Only the LLM call is retried; tools are never
re-executed by this module.
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TypeVar

from .llm.base import ProviderError, ProviderErrorKind

T = TypeVar("T")

RETRYABLE_ERROR_KINDS: frozenset[ProviderErrorKind] = frozenset(
    {ProviderErrorKind.RATE_LIMIT, ProviderErrorKind.TIMEOUT, ProviderErrorKind.TRANSPORT}
)


@dataclass
class RetryPolicy:
    """Retry configuration for provider calls.

    Attributes:
        max_retries: Maximum number of retries after the initial attempt.
        base_delay_s: Delay ceiling for the first retry wait.
        max_delay_s: Upper bound for any single backoff delay.
        retry_kinds: Provider error kinds that are eligible for retry.
    """

    max_retries: int = 3
    base_delay_s: float = 1.0
    max_delay_s: float = 30.0
    retry_kinds: set[ProviderErrorKind] = field(
        default_factory=lambda: {
            ProviderErrorKind.RATE_LIMIT,
            ProviderErrorKind.TIMEOUT,
            ProviderErrorKind.TRANSPORT,
        }
    )

    def is_retryable(self, kind: ProviderErrorKind) -> bool:
        """Check whether an error kind is retried under this policy.

        Args:
            kind: The normalized provider error kind.

        Returns:
            True when the kind belongs to ``retry_kinds``.
        """
        return kind in self.retry_kinds

    def to_dict(self) -> dict:
        """Convert to a dictionary for serialization.

        Returns:
            Dictionary representation with retry kinds as sorted value strings.
        """
        return {
            "max_retries": self.max_retries,
            "base_delay_s": self.base_delay_s,
            "max_delay_s": self.max_delay_s,
            "retry_kinds": sorted(kind.value for kind in self.retry_kinds),
        }

    @classmethod
    def from_dict(cls, data: dict) -> RetryPolicy:
        """Create a policy from a dictionary.

        Unknown retry kind values are ignored so older serialized
        configurations keep loading.

        Args:
            data: Dictionary produced by ``to_dict`` or an older version.

        Returns:
            A ``RetryPolicy`` instance.
        """
        known_kinds = {kind.value: kind for kind in ProviderErrorKind}
        retry_kinds = {known_kinds[value] for value in data.get("retry_kinds", []) if value in known_kinds}
        return cls(
            max_retries=data.get("max_retries", 3),
            base_delay_s=data.get("base_delay_s", 1.0),
            max_delay_s=data.get("max_delay_s", 30.0),
            retry_kinds=retry_kinds or set(RETRYABLE_ERROR_KINDS),
        )


@dataclass(frozen=True)
class RetryNotice:
    """Structured description of one retry wait, rendered on CLI surfaces.

    Attributes:
        attempt: 1-based index of the retry that is about to happen.
        max_retries: Maximum retries configured on the policy.
        delay_s: The computed backoff delay in seconds.
        kind: Provider error kind value (or ``"error"`` when unclassified).
    """

    attempt: int
    max_retries: int
    delay_s: float
    kind: str


def format_retry_notice(notice: RetryNotice) -> str:
    """Render a retry notice in the documented CLI format.

    Args:
        notice: The retry notice to render.

    Returns:
        A string shaped like ``retry 2/3 after 2.1s (rate_limit)``.
    """
    return f"retry {notice.attempt}/{notice.max_retries} after {notice.delay_s:.1f}s ({notice.kind})"


def backoff_delay(policy: RetryPolicy, attempt: int, rng: random.Random) -> float:
    """Compute one backoff delay: exponential growth with full jitter.

    The ceiling is ``min(max_delay_s, base_delay_s * 2 ** attempt)`` and the
    returned delay is uniformly distributed in ``[0, ceiling]`` (full jitter).

    Args:
        policy: The active retry policy.
        attempt: Zero-based index of the attempt that just failed.
        rng: Random source used for jitter.

    Returns:
        The delay in seconds.
    """
    ceiling = min(policy.max_delay_s, policy.base_delay_s * (2**attempt))
    return rng.uniform(0.0, ceiling)


async def call_with_retry(
    fn: Callable[[], Awaitable[T]],
    *,
    policy: RetryPolicy,
    should_retry: Callable[[Exception], bool],
    sleep: Callable[[float], Awaitable[None]] | None = None,
    rng: random.Random | None = None,
    cancel_event: asyncio.Event | None = None,
    on_retry: Callable[[RetryNotice], None] | None = None,
) -> T:
    """Await ``fn()`` retrying only errors the policy declares retryable.

    Only the awaited callable is retried — callers must ensure retries never
    re-execute side effects such as tool runs. The cancel event is honored
    before the first attempt and between attempts; when set, the call raises
    :class:`asyncio.CancelledError` instead of retrying.

    Args:
        fn: Zero-argument async callable to invoke.
        policy: Retry policy bounding attempts and delays.
        should_retry: Predicate deciding whether an exception is retryable.
        sleep: Async sleep used for backoff waits (defaults to ``asyncio.sleep``).
        rng: Random source for jitter (defaults to a fresh ``random.Random``).
        cancel_event: Optional event checked before and between attempts.
        on_retry: Optional callback invoked with a :class:`RetryNotice` per wait.

    Returns:
        The successful result of ``fn()``.

    Raises:
        asyncio.CancelledError: When the cancel event is observed.
        Exception: The last exception when retries are exhausted or the error
            is not retryable.
    """
    sleep_fn = sleep or asyncio.sleep
    rng = rng or random.Random()  # nosec B311 - jitter only, not security-sensitive
    last_error: Exception | None = None

    for attempt in range(policy.max_retries + 1):
        if cancel_event is not None and cancel_event.is_set():
            raise asyncio.CancelledError()
        try:
            return await fn()
        except Exception as error:
            last_error = error
            if attempt >= policy.max_retries or not should_retry(error):
                raise
            if cancel_event is not None and cancel_event.is_set():
                raise asyncio.CancelledError() from error
            delay = backoff_delay(policy, attempt, rng)
            if on_retry is not None:
                kind = error.kind.value if isinstance(error, ProviderError) else "error"
                on_retry(RetryNotice(attempt=attempt + 1, max_retries=policy.max_retries, delay_s=delay, kind=kind))
            await sleep_fn(delay)

    # Defensive: the loop body always returns or raises, so last_error is set
    # if control ever reaches here (e.g. max_retries were negative).
    if last_error is None:  # pragma: no cover
        raise RuntimeError("call_with_retry loop exited without recording an error")
    raise last_error  # pragma: no cover
