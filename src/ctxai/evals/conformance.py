"""Executable provider conformance suite derived from PROVIDER_SPECS (HH-09).

For one provider instance the suite verifies, without any benchmark or
repository: auth presence, declared-vs-observed capabilities (drift = named
failure), simple chat, a tool round trip, event streaming, and error
normalization. The CI path runs the suite against the scripted
``MockLLMProvider`` with a checked-in mock spec — no network, no credentials.
Live providers run only on demand (``--provider P``): the checks then make
real API calls and the CLI prints a cost warning first.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..agent.config import AgentLLMConfig
from ..agent.events import StreamEvent
from ..agent.llm.base import (
    BaseLLMProvider,
    Message,
    MessageRole,
    ProviderCapabilities,
    ProviderError,
    ProviderErrorKind,
)
from ..agent.llm.contract import ProviderSpec
from ..agent.llm.mock_provider import MOCK_PROVIDER_MODEL, MockLLMProvider, create_mock_response

# The checked-in spec the CI (mock) conformance run verifies against. It is
# not part of PROVIDER_SPECS: the mock provider is a testing construct, not
# a shippable provider boundary.
MOCK_PROVIDER_SPEC = ProviderSpec(
    name="mock",
    transport="Scripted mock (ctxai.agent.llm.mock_provider)",
    local=True,
    capabilities=ProviderCapabilities(tools=True, streaming=False),
    models=MOCK_PROVIDER_MODEL,
)

# Sample (exception, expected kind) pairs for the error-normalization check.
_ERROR_SAMPLES: tuple[tuple[Exception, ProviderErrorKind], ...] = (
    (RuntimeError("401 unauthorized: bad api key"), ProviderErrorKind.AUTHENTICATION),
    (RuntimeError("429 too many requests"), ProviderErrorKind.RATE_LIMIT),
    (TimeoutError("request timed out after 30s"), ProviderErrorKind.TIMEOUT),
    (ValueError("invalid json: expecting value"), ProviderErrorKind.INVALID_RESPONSE),
    (RuntimeError("connection reset by peer"), ProviderErrorKind.TRANSPORT),
)

ECHO_TOOL_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "echo",
        "description": "Echo the provided text back to the caller.",
        "parameters": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    },
}


@dataclass(frozen=True)
class ConformanceCheck:
    """One executed conformance check for a provider.

    Attributes:
        name: Check name (``auth_presence``, ``capabilities_declared_vs_observed``,
            ``simple_chat``, ``tool_round_trip``, ``streaming``,
            ``error_normalization``).
        passed: Whether the check passed.
        drift: True when the failure is a declared-vs-observed mismatch.
        detail: Human-readable result detail (error kind on failures).
    """

    name: str
    passed: bool
    drift: bool = False
    detail: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to the JSON representation.

        Returns:
            Dictionary matching the conformance report schema.
        """
        return {"name": self.name, "passed": self.passed, "drift": self.drift, "detail": self.detail}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ConformanceCheck:
        """Rebuild from the JSON representation.

        Args:
            data: Dictionary produced by :meth:`to_dict`.

        Returns:
            The reconstructed ConformanceCheck.
        """
        return cls(
            name=str(data["name"]),
            passed=bool(data["passed"]),
            drift=bool(data.get("drift", False)),
            detail=data.get("detail"),
        )


@dataclass(frozen=True)
class ProviderConformanceReport:
    """Conformance results for one provider against its declared spec.

    Attributes:
        provider: Provider name (spec name or provider class name for mock).
        declared: Declared capability dict from the spec.
        observed: Capability dict reported by the provider instance.
        checks: Executed checks in suite order.
        status: ``pass`` when every check passed, ``fail`` otherwise.
    """

    provider: str
    declared: dict[str, Any]
    observed: dict[str, Any]
    checks: list[ConformanceCheck]
    status: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to the JSON representation.

        Returns:
            Dictionary matching the conformance report schema.
        """
        return {
            "provider": self.provider,
            "declared": dict(self.declared),
            "observed": dict(self.observed),
            "checks": [check.to_dict() for check in self.checks],
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProviderConformanceReport:
        """Rebuild from the JSON representation.

        Args:
            data: Dictionary produced by :meth:`to_dict`.

        Returns:
            The reconstructed ProviderConformanceReport.
        """
        return cls(
            provider=str(data["provider"]),
            declared=dict(data.get("declared") or {}),
            observed=dict(data.get("observed") or {}),
            checks=[ConformanceCheck.from_dict(check) for check in data.get("checks", [])],
            status=str(data["status"]),
        )


def run_provider_conformance(provider: BaseLLMProvider, spec: ProviderSpec) -> ProviderConformanceReport:
    """Run the conformance suite for one provider against its declared spec.

    Every check is isolated: a raised exception becomes a failed check with
    the normalized provider error kind in its detail, never an abort of the
    suite. Live providers make real API calls in the chat/tool/streaming
    checks; the mock provider consumes its script instead.

    Args:
        provider: The provider instance under test.
        spec: The declared provider spec (``MOCK_PROVIDER_SPEC`` for the
            mock conformance path, an entry of ``PROVIDER_SPECS`` for live).

    Returns:
        The ProviderConformanceReport with per-check results.
    """
    checks: list[ConformanceCheck] = []
    observed = provider.get_capabilities()
    checks.append(_check_auth(provider))
    checks.append(_check_capabilities(spec, observed))
    checks.append(_check_simple_chat(provider))
    checks.append(_check_tool_round_trip(provider, bool(observed.tools)))
    checks.append(_check_streaming(provider, bool(observed.streaming)))
    checks.append(_check_error_normalization(provider))
    status = "pass" if all(check.passed for check in checks) else "fail"
    return ProviderConformanceReport(
        provider=spec.name,
        declared={
            "tools": bool(spec.capabilities.tools),
            "streaming": bool(spec.capabilities.streaming),
            "context_size": int(spec.capabilities.context_size),
            "transport": spec.transport,
            "local": spec.local,
        },
        observed={
            "tools": bool(observed.tools),
            "streaming": bool(observed.streaming),
            "context_size": observed.context_size,
        },
        checks=checks,
        status=status,
    )


def run_mock_conformance() -> ProviderConformanceReport:
    """Run the CI-path conformance suite against the scripted mock provider.

    Builds a fresh MockLLMProvider with a deterministic script (one plain
    completion, one tool call, one completion for the buffered fallback) and
    verifies every suite check — no network, no credentials.

    Returns:
        The ProviderConformanceReport for the mock provider.
    """
    provider = MockLLMProvider(
        config=AgentLLMConfig(provider="mock", model=MOCK_PROVIDER_MODEL, api_key="mock-key"),
        responses=[
            create_mock_response(content="ok", usage={"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11}),
            create_mock_response(
                tool_calls=[{"name": "echo", "parameters": {"text": "hi"}}],
                usage={"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
            ),
            create_mock_response(
                content="streamed", usage={"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13}
            ),
        ],
    )
    return run_provider_conformance(provider, MOCK_PROVIDER_SPEC)


def _check_auth(provider: BaseLLMProvider) -> ConformanceCheck:
    """Check auth presence: providers that require a key must have one.

    Args:
        provider: The provider instance under test.

    Returns:
        The auth_presence check result.
    """
    if not provider.requires_api_key():
        return ConformanceCheck("auth_presence", True, detail="provider does not require an API key")
    if provider.api_key:
        return ConformanceCheck("auth_presence", True, detail="API key configured")
    return ConformanceCheck(
        "auth_presence",
        False,
        detail="provider requires an API key but none is configured (run 'ctxai login <provider>')",
    )


def _check_capabilities(spec: ProviderSpec, observed: Any) -> ConformanceCheck:
    """Compare declared spec capabilities against the observed ones.

    Args:
        spec: The declared provider spec.
        observed: The capabilities reported by the provider instance.

    Returns:
        The capabilities_declared_vs_observed check result; any mismatch is
        drift and therefore a failure.
    """
    declared = spec.capabilities
    mismatches = []
    if bool(declared.tools) != bool(observed.tools):
        mismatches.append(f"tools declared={bool(declared.tools)} observed={bool(observed.tools)}")
    if bool(declared.streaming) != bool(observed.streaming):
        mismatches.append(f"streaming declared={bool(declared.streaming)} observed={bool(observed.streaming)}")
    if mismatches:
        return ConformanceCheck("capabilities_declared_vs_observed", False, drift=True, detail="; ".join(mismatches))
    return ConformanceCheck(
        "capabilities_declared_vs_observed", True, detail="declared capabilities match observations"
    )


def _check_simple_chat(provider: BaseLLMProvider) -> ConformanceCheck:
    """Check a minimal buffered chat round trip.

    Args:
        provider: The provider instance under test.

    Returns:
        The simple_chat check result.
    """
    try:
        response = provider.chat([Message(role=MessageRole.USER, content="Reply with the single word: ok")])
    except Exception as error:  # noqa: BLE001 - every failure is a named check result
        return ConformanceCheck("simple_chat", False, detail=_failure_detail(provider, error))
    if response.finish_reason == "error":
        return ConformanceCheck("simple_chat", False, detail=str(response.content or "provider returned an error"))
    if not response.content and not response.has_tool_calls:
        return ConformanceCheck("simple_chat", False, detail="empty response")
    return ConformanceCheck("simple_chat", True, detail=f"finish_reason={response.finish_reason}")


def _check_tool_round_trip(provider: BaseLLMProvider, tools_supported: bool) -> ConformanceCheck:
    """Check the tool round trip: a call with a tool schema returns a
    well-formed response (parsed tool calls or a completion).

    Args:
        provider: The provider instance under test.
        tools_supported: Whether the provider reports tool support.

    Returns:
        The tool_round_trip check result.
    """
    messages = [Message(role=MessageRole.USER, content="Use the echo tool with text 'hi'.")]
    if not tools_supported:
        # Declared no-tools: validate_request must reject the request up
        # front with UNSUPPORTED instead of letting it through.
        try:
            provider.validate_request(messages, tools=[ECHO_TOOL_SCHEMA])
        except ProviderError as error:
            if error.kind is ProviderErrorKind.UNSUPPORTED:
                return ConformanceCheck("tool_round_trip", True, detail="tool requests rejected as unsupported")
            return ConformanceCheck("tool_round_trip", False, detail=f"unexpected rejection kind: {error.kind.value}")
        except Exception as error:  # noqa: BLE001 - every failure is a named check result
            return ConformanceCheck("tool_round_trip", False, detail=_failure_detail(provider, error))
        return ConformanceCheck(
            "tool_round_trip",
            False,
            drift=True,
            detail="provider declares no tool support but accepted a tool request",
        )
    try:
        response = provider.chat(messages, tools=[ECHO_TOOL_SCHEMA])
    except Exception as error:  # noqa: BLE001 - every failure is a named check result
        return ConformanceCheck("tool_round_trip", False, detail=_failure_detail(provider, error))
    if response.finish_reason == "error":
        return ConformanceCheck("tool_round_trip", False, detail=str(response.content or "provider returned an error"))
    for call in response.tool_calls:
        if not call.name or not isinstance(call.parameters, dict):
            return ConformanceCheck("tool_round_trip", False, detail="malformed tool call payload")
    detail = f"{len(response.tool_calls)} tool call(s)" if response.tool_calls else "well-formed completion"
    return ConformanceCheck("tool_round_trip", True, detail=detail)


def _check_streaming(provider: BaseLLMProvider, streaming_supported: bool) -> ConformanceCheck:
    """Check event streaming (or the documented buffered fallback).

    Args:
        provider: The provider instance under test.
        streaming_supported: Whether the provider reports event streaming.

    Returns:
        The streaming check result.
    """
    messages = [Message(role=MessageRole.USER, content="Stream the word ok.")]
    try:
        generator = provider.stream_chat_events(messages)
        events: list[StreamEvent] = []
        while True:
            try:
                events.append(next(generator))
            except StopIteration as stop:
                response = stop.value
                break
    except Exception as error:  # noqa: BLE001 - every failure is a named check result
        return ConformanceCheck("streaming", False, detail=_failure_detail(provider, error))
    if response is None:
        return ConformanceCheck("streaming", False, detail="stream ended without a complete response")
    if streaming_supported and not events:
        return ConformanceCheck("streaming", False, detail="declared streaming produced no events")
    if not streaming_supported and len(events) > 1:
        return ConformanceCheck(
            "streaming",
            False,
            drift=True,
            detail="provider declares buffered mode but emitted multiple stream events",
        )
    detail = f"{len(events)} event(s), finish_reason={response.finish_reason}"
    return ConformanceCheck("streaming", True, detail=detail)


def _check_error_normalization(provider: BaseLLMProvider) -> ConformanceCheck:
    """Check that sample errors map onto the declared stable error kinds.

    Args:
        provider: The provider instance under test.

    Returns:
        The error_normalization check result.
    """
    mismatches = []
    for sample, expected in _ERROR_SAMPLES:
        kind = provider.normalize_error(sample).kind
        if kind is not expected:
            mismatches.append(f"{sample!r} -> {kind.value} (expected {expected.value})")
    if mismatches:
        return ConformanceCheck("error_normalization", False, detail="; ".join(mismatches))
    return ConformanceCheck("error_normalization", True, detail=f"{len(_ERROR_SAMPLES)} sample errors normalized")


def _failure_detail(provider: BaseLLMProvider, error: Exception) -> str:
    """Render one check failure with its normalized provider error kind.

    Args:
        provider: The provider instance under test.
        error: The raised exception.

    Returns:
        A ``ProviderErrorKind: message`` string.
    """
    if isinstance(error, ProviderError):
        return f"{error.kind.value}: {error}"
    normalized = provider.normalize_error(error)
    return f"{normalized.kind.value}: {normalized}"
