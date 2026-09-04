"""Unit tests for real-provider ``stream_chat_events`` implementations (HH-05).

Exercises the Anthropic, OpenAI, and OpenRouter event-streaming code paths
against fake SDK objects (no network): text deltas stream as
``("text", ...)`` StreamEvents, tool-call fragments accumulate into complete
tool calls, usage is captured, and the returned ``LLMResponse`` matches what
``chat()`` would have produced.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from ctxai.agent.config import AgentLLMConfig
from ctxai.agent.llm.base import LLMResponse, Message, MessageRole


def make_message(text: str = "hello") -> list[Message]:
    """Build a one-turn conversation.

    Args:
        text: The user text.

    Returns:
        The message list.
    """
    return [Message(MessageRole.USER, text)]


# ---------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------


class _FakeAnthropicStream:
    """Context-manager stream exposing text_stream and get_final_message."""

    def __init__(self, texts: list[str], final: object):
        """Initialize the fake stream.

        Args:
            texts: Text deltas yielded by ``text_stream``.
            final: The object returned by ``get_final_message``.
        """
        self._texts = texts
        self._final = final

    def __enter__(self) -> _FakeAnthropicStream:
        """Enter the streaming context.

        Returns:
            The stream itself.
        """
        return self

    def __exit__(self, *exc: object) -> bool:
        """Exit the streaming context.

        Args:
            *exc: Exception information (ignored).

        Returns:
            False; exceptions propagate.
        """
        return False

    @property
    def text_stream(self):
        """Yield the configured text deltas.

        Yields:
            Text chunks.
        """
        return iter(self._texts)

    def get_final_message(self) -> object:
        """Return the final accumulated message.

        Returns:
            The configured final message.
        """
        return self._final


class _FakeAnthropicMessages:
    """Fake ``client.messages`` namespace capturing stream request params."""

    def __init__(self, stream: _FakeAnthropicStream):
        """Initialize with the stream to serve.

        Args:
            stream: The fake stream returned per request.
        """
        self._stream = stream
        self.kwargs: dict | None = None

    def stream(self, **kwargs) -> _FakeAnthropicStream:
        """Return the fake stream, capturing the request parameters.

        Args:
            **kwargs: Request parameters.

        Returns:
            The fake stream.
        """
        self.kwargs = kwargs
        return self._stream


def _final_anthropic_message():
    """Build an Anthropic-shaped final message with text and one tool_use block."""
    return SimpleNamespace(
        content=[
            SimpleNamespace(type="text", text="Reading the file"),
            SimpleNamespace(type="tool_use", id="toolu_1", name="read_file", input={"path": "note.txt"}),
        ],
        stop_reason="tool_use",
        usage=SimpleNamespace(input_tokens=120, output_tokens=30),
    )


def test_anthropic_stream_chat_events_streams_deltas_and_returns_response():
    """Anthropic streaming emits text deltas and returns the parsed final message."""
    from ctxai.agent.llm.anthropic_provider import AnthropicProvider

    provider = AnthropicProvider(AgentLLMConfig(provider="anthropic", model="claude-test", api_key="k"))
    stream = _FakeAnthropicStream(["Reading ", "the ", "file"], _final_anthropic_message())
    provider.client = SimpleNamespace(messages=_FakeAnthropicMessages(stream))

    assert provider.get_capabilities().streaming is True

    generator = provider.stream_chat_events(make_message())
    events: list[tuple] = []
    while True:
        try:
            events.append(next(generator))
        except StopIteration as stop:
            response = stop.value
            break

    assert [payload for kind, payload in events] == ["Reading ", "the ", "file"]
    assert all(kind == "text" for kind, _ in events)
    assert isinstance(response, LLMResponse)
    assert response.content == "Reading the file"
    assert [(call.id, call.name, call.parameters) for call in response.tool_calls] == [
        ("toolu_1", "read_file", {"path": "note.txt"})
    ]
    assert response.finish_reason == "tool_calls"
    assert response.usage == {"prompt_tokens": 120, "completion_tokens": 30, "total_tokens": 150}


def test_anthropic_stream_chat_events_maps_errors_to_error_response():
    """A failing Anthropic stream yields an error-shaped response, mirroring chat()."""
    from ctxai.agent.llm.anthropic_provider import AnthropicProvider

    provider = AnthropicProvider(AgentLLMConfig(provider="anthropic", model="claude-test", api_key="k"))

    class _ExplodingStream(_FakeAnthropicStream):
        def __enter__(self):
            raise RuntimeError("stream exploded")

    provider.client = SimpleNamespace(messages=_FakeAnthropicMessages(_ExplodingStream([], None)))
    generator = provider.stream_chat_events(make_message())
    while True:
        try:
            next(generator)
        except StopIteration as stop:
            response = stop.value
            break
    assert isinstance(response, LLMResponse)
    assert response.finish_reason == "error"
    assert "stream exploded" in response.content


# ---------------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------------


def _openai_chunk(content=None, tool_calls=None, finish_reason=None, usage=None, choices=True):
    """Build an OpenAI-shaped streaming chunk.

    Args:
        content: Text delta content.
        tool_calls: Tool-call delta fragments.
        finish_reason: The chunk finish reason.
        usage: Usage payload (final chunk only).
        choices: Whether the chunk carries choices.

    Returns:
        A namespace shaped like an openai ChatCompletionChunk.
    """
    delta = SimpleNamespace(content=content, tool_calls=tool_calls)
    return SimpleNamespace(
        choices=[SimpleNamespace(delta=delta, finish_reason=finish_reason)] if choices else [],
        usage=usage,
    )


class _FakeCompletions:
    """Fake ``client.chat.completions`` returning scripted chunks."""

    def __init__(self, chunks: list):
        """Initialize with the chunks to serve.

        Args:
            chunks: The chunk sequence returned by ``create``.
        """
        self._chunks = chunks
        self.kwargs: dict | None = None

    def create(self, **kwargs):
        """Return the scripted chunk iterator, capturing request parameters.

        Args:
            **kwargs: Request parameters.

        Returns:
            Iterator over the scripted chunks.
        """
        self.kwargs = kwargs
        return iter(self._chunks)


def test_openai_stream_chat_events_accumulates_tool_calls_and_usage():
    """OpenAI streaming streams deltas and returns accumulated tool calls plus usage."""
    from ctxai.agent.llm.openai_provider import OpenAIProvider

    provider = OpenAIProvider(AgentLLMConfig(provider="openai", model="gpt-test", api_key="k"))
    chunks = [
        _openai_chunk(content="Writing"),
        _openai_chunk(
            tool_calls=[
                SimpleNamespace(index=0, id="call_1", function=SimpleNamespace(name="write_file", arguments='{"pat'))
            ]
        ),
        _openai_chunk(
            tool_calls=[
                SimpleNamespace(index=0, id=None, function=SimpleNamespace(name=None, arguments='h": "out.txt"}'))
            ]
        ),
        _openai_chunk(finish_reason="tool_calls"),
        _openai_chunk(
            choices=False,
            usage=SimpleNamespace(prompt_tokens=90, completion_tokens=11, total_tokens=101),
        ),
    ]
    provider.client = SimpleNamespace(chat=SimpleNamespace(completions=_FakeCompletions(chunks)))

    assert provider.get_capabilities().streaming is True

    generator = provider.stream_chat_events(make_message())
    events: list[tuple] = []
    while True:
        try:
            events.append(next(generator))
        except StopIteration as stop:
            response = stop.value
            break

    assert [payload for kind, payload in events] == ["Writing"]
    assert isinstance(response, LLMResponse)
    assert response.content == "Writing"
    assert [(call.id, call.name, call.parameters) for call in response.tool_calls] == [
        ("call_1", "write_file", {"path": "out.txt"})
    ]
    assert response.finish_reason == "tool_calls"
    assert response.usage == {"prompt_tokens": 90, "completion_tokens": 11, "total_tokens": 101}
    assert provider.client.chat.completions.kwargs["stream_options"] == {"include_usage": True}


# ---------------------------------------------------------------------------
# OpenRouter
# ---------------------------------------------------------------------------


class _FakeSSEResponse:
    """Fake requests response streaming SSE lines."""

    def __init__(self, lines: list[bytes]):
        """Initialize with the raw SSE lines.

        Args:
            lines: Byte lines returned by ``iter_lines``.
        """
        self._lines = lines
        self.status_code = 200
        self.text = ""

    def iter_lines(self):
        """Yield the configured SSE lines.

        Yields:
            Byte lines.
        """
        return iter(self._lines)


def test_openrouter_stream_chat_events_accumulates_tool_calls_and_usage(monkeypatch):
    """OpenRouter SSE streaming streams deltas and returns complete tool calls plus usage."""
    from ctxai.agent.llm import openrouter_provider as module
    from ctxai.agent.llm.openrouter_provider import OpenRouterProvider

    provider = OpenRouterProvider(AgentLLMConfig(provider="openrouter", model="or-test", api_key="k"))

    def sse_data(payload: dict) -> bytes:
        """Encode one SSE data line.

        Args:
            payload: The JSON payload.

        Returns:
            The encoded ``data: `` line.
        """
        return b"data: " + json.dumps(payload).encode("utf-8")

    lines = [
        sse_data({"choices": [{"delta": {"content": "Writing"}, "finish_reason": None}]}),
        sse_data(
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {"index": 0, "id": "call_9", "function": {"name": "write_file", "arguments": '{"path"'}}
                            ]
                        },
                        "finish_reason": None,
                    }
                ]
            }
        ),
        sse_data(
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {"index": 0, "function": {"arguments": ': "out.txt"}'}},
                                {"index": 1, "id": "call_10", "function": {"name": "read_file", "arguments": "{}"}},
                            ]
                        },
                        "finish_reason": None,
                    }
                ]
            }
        ),
        sse_data({"choices": [{"delta": {}, "finish_reason": "tool_calls"}]}),
        sse_data({"choices": [], "usage": {"prompt_tokens": 70, "completion_tokens": 9, "total_tokens": 79}}),
        b"data: [DONE]",
    ]
    captured: dict = {}

    def fake_post(url, headers=None, json=None, stream=False, timeout=None):
        """Capture the request and return the fake SSE response.

        Args:
            url: Request URL.
            headers: Request headers.
            json: Request body.
            stream: Streaming flag.
            timeout: Request timeout.

        Returns:
            The fake SSE response.
        """
        captured["body"] = json
        return _FakeSSEResponse(lines)

    monkeypatch.setattr(module.requests, "post", fake_post)

    assert provider.get_capabilities().streaming is True

    generator = provider.stream_chat_events(make_message())
    events: list[tuple] = []
    while True:
        try:
            events.append(next(generator))
        except StopIteration as stop:
            response = stop.value
            break

    assert [payload for kind, payload in events] == ["Writing"]
    assert isinstance(response, LLMResponse)
    assert response.content == "Writing"
    assert [(call.id, call.name, call.parameters) for call in response.tool_calls] == [
        ("call_9", "write_file", {"path": "out.txt"}),
        ("call_10", "read_file", {}),
    ]
    assert response.finish_reason == "tool_calls"
    assert response.usage == {"prompt_tokens": 70, "completion_tokens": 9, "total_tokens": 79}
    assert captured["body"]["usage"] == {"include": True}


def test_openrouter_stream_chat_events_raises_on_http_error(monkeypatch):
    """A non-200 response raises so the loop can normalize it."""
    from ctxai.agent.llm.openrouter_provider import OpenRouterProvider

    provider = OpenRouterProvider(AgentLLMConfig(provider="openrouter", model="or-test", api_key="k"))

    def fake_post(url, headers=None, json=None, stream=False, timeout=None):
        """Return a failing response.

        Args:
            url: Request URL.
            headers: Request headers.
            json: Request body.
            stream: Streaming flag.
            timeout: Request timeout.

        Returns:
            A response with a non-200 status.
        """
        return SimpleNamespace(status_code=429, text="slow down", iter_lines=lambda: iter([]))

    monkeypatch.setattr("requests.post", fake_post)

    with pytest.raises(Exception, match="429"):
        generator = provider.stream_chat_events(make_message())
        while True:
            try:
                next(generator)
            except StopIteration:
                break
