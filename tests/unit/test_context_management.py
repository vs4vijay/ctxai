"""Tests for ctxai.agent.context."""

from ctxai.agent.context import ConversationContext
from ctxai.agent.llm.base import MessageRole


def test_add_messages_and_count():
    ctx = ConversationContext()
    ctx.add_system_message("you are helpful")
    ctx.add_user_message("hi")
    ctx.add_assistant_message("hello!")
    assert ctx.get_message_count() == 3


def test_clear_preserves_system_messages():
    ctx = ConversationContext()
    ctx.add_system_message("sys")
    ctx.add_user_message("u")
    ctx.add_assistant_message("a")
    ctx.clear()
    assert ctx.get_message_count() == 1
    assert ctx.messages[0].role == MessageRole.SYSTEM


def test_truncate_old_messages_keeps_system():
    ctx = ConversationContext()
    ctx.add_system_message("sys")
    big = "x" * 8000  # ~2000 tokens
    for _ in range(20):
        ctx.add_user_message(big)
    ctx.truncate_old_messages(max_tokens=1000)
    assert any(m.role == MessageRole.SYSTEM for m in ctx.messages)
    assert ctx.get_token_count_estimate() <= 1000 + 250


def test_truncate_drops_orphan_tool_results():
    ctx = ConversationContext()
    ctx.add_system_message("sys")
    ctx.add_user_message("u" * 1000)
    ctx.add_assistant_message("a")
    ctx.add_tool_result("call_x", "tool_x", "result " + "y" * 2000)
    ctx.truncate_old_messages(max_tokens=200)
    # Either everything fits or the tool result was dropped to avoid an orphan.
    if ctx.messages:
        non_system = [m for m in ctx.messages if m.role != MessageRole.SYSTEM]
        if non_system:
            assert non_system[0].tool_call_id is None


def test_token_estimate_grows_with_content():
    ctx = ConversationContext()
    base = ctx.get_token_count_estimate()
    ctx.add_user_message("a" * 400)
    assert ctx.get_token_count_estimate() > base + 90
