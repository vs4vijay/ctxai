"""Tests for ctxai.agent.llm.base error classes and helpers."""

from ctxai.agent.llm.base import (
    AuthenticationError,
    ContextLengthError,
    Message,
    MessageRole,
    ProviderError,
    ProviderTimeoutError,
    RateLimitError,
    ToolCall,
)


def test_provider_error_carries_metadata():
    err = ProviderError("boom", provider="openrouter", status_code=500)
    assert err.provider == "openrouter"
    assert err.status_code == 500
    assert str(err) == "boom"


def test_rate_limit_error_has_retry_after():
    err = RateLimitError("slow down", retry_after=10.0)
    assert err.retry_after == 10.0
    assert isinstance(err, ProviderError)


def test_context_length_error_is_provider_error():
    assert isinstance(ContextLengthError("too big"), ProviderError)


def test_authentication_error_is_provider_error():
    assert isinstance(AuthenticationError("bad key"), ProviderError)


def test_provider_timeout_error_is_provider_error():
    assert isinstance(ProviderTimeoutError("slow"), ProviderError)


def test_message_to_dict_basic():
    m = Message(role=MessageRole.USER, content="hi")
    d = m.to_dict()
    assert d["role"] == "user"
    assert d["content"] == "hi"


def test_message_to_dict_with_tool_calls_openai():
    tc = ToolCall(id="1", name="read", parameters={"path": "a.py"})
    m = Message(role=MessageRole.ASSISTANT, content="", tool_calls=[tc])
    d = m.to_dict(format="openai")
    assert d["tool_calls"][0]["function"]["name"] == "read"


def test_tool_result_message_role_becomes_tool():
    m = Message(role=MessageRole.USER, content="result", tool_call_id="1", name="read")
    d = m.to_dict()
    assert d["role"] == "tool"
    assert d["tool_call_id"] == "1"
