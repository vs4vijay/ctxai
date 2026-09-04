"""Unit tests for the HH-05 streaming event protocol.

Covers the ``AgentEvent`` vocabulary shared by the loop and its UI surfaces,
the ``StreamEvent`` shapes providers emit, the default (fallback)
``stream_chat_events`` implementation on ``BaseLLMProvider``, and the
``MockLLMProvider`` streaming mode (scripted token deltas).
"""

from __future__ import annotations

import dataclasses

import pytest

from ctxai.agent.config import AgentLLMConfig
from ctxai.agent.events import AgentEvent, AgentEventKind, StreamEvent
from ctxai.agent.llm.base import BaseLLMProvider, LLMResponse, Message, MessageRole, ProviderCapabilities
from tests.mocks.mock_llm import MockLLMProvider

# Part II contract: the closed event vocabulary of the agent loop.
CONTRACT_KINDS = {
    "token",
    "tool_call_started",
    "tool_result",
    "approval_required",
    "approval_decided",
    "status",
    "usage",
    "final_report",
}


def test_agent_event_kinds_match_the_part_ii_contract():
    """The event vocabulary is exactly the Part II contract set."""
    assert {kind.value for kind in AgentEventKind} == CONTRACT_KINDS


def test_agent_event_defaults_and_equality():
    """AgentEvent carries kind, text, and data with sensible defaults."""
    event = AgentEvent(kind=AgentEventKind.TOKEN, text="hello ")
    assert event.kind is AgentEventKind.TOKEN
    assert event.text == "hello "
    assert event.data == {}

    same = AgentEvent(kind=AgentEventKind.TOKEN, text="hello ")
    assert event == same, "equal payloads compare equal"
    assert AgentEvent(kind=AgentEventKind.TOKEN) != AgentEvent(kind=AgentEventKind.STATUS)

    data_event = AgentEvent(kind=AgentEventKind.USAGE, data={"total_tokens": 5})
    assert data_event.data == {"total_tokens": 5}


def test_agent_event_is_frozen_and_data_default_is_not_shared():
    """Events are immutable value objects and each gets its own data dict."""
    event = AgentEvent(kind=AgentEventKind.STATUS)
    with pytest.raises(dataclasses.FrozenInstanceError):
        event.text = "mutated"  # type: ignore[misc]

    first = AgentEvent(kind=AgentEventKind.STATUS)
    second = AgentEvent(kind=AgentEventKind.STATUS)
    first.data["x"] = 1
    assert second.data == {}, "data dicts must not be shared between events"


class MinimalProvider(BaseLLMProvider):
    """A provider that does not override stream_chat_events (fallback class)."""

    def get_default_model(self) -> str:
        """Return the default model name.

        Returns:
            The model identifier.
        """
        return "minimal-model"

    def chat(self, messages: list[Message], tools=None, **kwargs) -> LLMResponse:
        """Return a fixed buffered response.

        Args:
            messages: Conversation messages.
            tools: Optional tool schemas.
            **kwargs: Additional provider arguments.

        Returns:
            A fixed LLMResponse.
        """
        return LLMResponse(content="buffered answer", finish_reason="stop", usage={"total_tokens": 7})

    def stream_chat(self, messages: list[Message], tools=None, **kwargs):
        """Yield the buffered answer as one chunk (text-only legacy streaming).

        Args:
            messages: Conversation messages.
            tools: Optional tool schemas.
            **kwargs: Additional provider arguments.

        Yields:
            The full response content as a single chunk.
        """
        yield "buffered answer"

    def supports_function_calling(self) -> bool:
        """Report tool-calling support.

        Returns:
            False; the minimal provider exposes no tools.
        """
        return False

    def requires_api_key(self) -> bool:
        """Report API-key requirements.

        Returns:
            False; the minimal provider needs no key.
        """
        return False


def test_default_stream_chat_events_falls_back_to_one_text_event():
    """Providers without real streaming degrade to one buffered text event."""
    provider = MinimalProvider(AgentLLMConfig(provider="mock", model="minimal-model"))
    assert provider.get_capabilities().streaming is False, "capabilities must reflect the fallback"

    generator = provider.stream_chat_events([Message(MessageRole.USER, "hi")])
    events: list[StreamEvent] = []
    while True:
        try:
            events.append(next(generator))
        except StopIteration as stop:
            response = stop.value
            break

    assert events == [("text", "buffered answer")], "fallback emits exactly one text StreamEvent"
    assert isinstance(response, LLMResponse)
    assert response.content == "buffered answer"
    assert response.usage == {"total_tokens": 7}


def test_default_stream_chat_events_silent_on_empty_content():
    """A buffered response without content emits no text event but still returns."""
    provider = MinimalProvider(AgentLLMConfig(provider="mock", model="minimal-model"))
    generator = provider.stream_chat_events([Message(MessageRole.USER, "hi")])
    # Replace the scripted response with an empty one.
    provider.chat = lambda messages, tools=None, **kwargs: LLMResponse(content="", finish_reason="stop")  # type: ignore[method-assign]
    events: list[StreamEvent] = []
    while True:
        try:
            events.append(next(generator))
        except StopIteration as stop:
            response = stop.value
            break
    assert events == []
    assert isinstance(response, LLMResponse)


def test_mock_streaming_mode_yields_scripted_token_deltas():
    """MockLLMProvider with supports_streaming=True streams word chunks and returns the response."""
    provider = MockLLMProvider(
        responses=[{"content": "Reading the note now", "usage": {"prompt_tokens": 10, "completion_tokens": 4}}],
    )
    assert provider.get_capabilities().streaming is False, "streaming mode defaults to off"

    streaming = MockLLMProvider(
        supports_streaming=True,
        responses=[{"content": "Reading the note now", "usage": {"prompt_tokens": 10, "completion_tokens": 4}}],
    )
    assert streaming.get_capabilities().streaming is True

    generator = streaming.stream_chat_events([Message(MessageRole.USER, "hi")])
    chunks: list[str] = []
    while True:
        try:
            kind, payload = next(generator)
        except StopIteration as stop:
            response = stop.value
            break
        assert kind == "text"
        chunks.append(payload)

    assert len(chunks) > 1, "content is streamed as multiple deltas"
    assert "".join(chunks) == "Reading the note now", "deltas concatenate to the full content"
    assert isinstance(response, LLMResponse)
    assert response.content == "Reading the note now"
    assert response.usage == {"prompt_tokens": 10, "completion_tokens": 4}
    assert streaming.call_count == 1, "one streaming pass is exactly one provider call"


def test_mock_streaming_mode_routes_through_chat_overrides():
    """The streaming mode shares chat() bookkeeping so scripted subclasses behave identically."""
    provider = MockLLMProvider(supports_streaming=True, responses=[{"content": "answer"}])
    generator = provider.stream_chat_events([Message(MessageRole.USER, "hi")])
    while True:
        try:
            next(generator)
        except StopIteration:
            break
    assert provider.call_count == 1
    assert len(provider.call_history) == 1


def test_mock_streaming_mode_disabled_delegates_to_default_fallback():
    """With streaming off, the mock's stream_chat_events degrades like the base default."""
    provider = MockLLMProvider(supports_streaming=False, responses=[{"content": "buffered"}])
    generator = provider.stream_chat_events([Message(MessageRole.USER, "hi")])
    events: list[StreamEvent] = []
    while True:
        try:
            events.append(next(generator))
        except StopIteration as stop:
            response = stop.value
            break
    assert events == [("text", "buffered")]
    assert isinstance(response, LLMResponse)


def test_stream_event_alias_documents_the_three_shapes():
    """StreamEvent is a (kind, payload) tuple alias with three documented kinds."""
    text_event: StreamEvent = ("text", "delta")
    tool_event: StreamEvent = ("tool_call_delta", {"index": 0})
    usage_event: StreamEvent = ("usage", {"total_tokens": 3})
    for event in (text_event, tool_event, usage_event):
        assert isinstance(event, tuple) and len(event) == 2
    assert ProviderCapabilities().streaming is True, "capabilities default is unchanged"
