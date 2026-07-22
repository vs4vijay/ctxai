"""VS-05 acceptance tests for durable, provider-independent chat sessions."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ctxai.agent.config import AgentConfig
from ctxai.agent.context import ConversationContext
from ctxai.agent.core import Agent, AgentLoopConfig
from ctxai.agent.llm.base import MessageRole, ProviderCapabilities, ToolCall
from ctxai.agent.sessions import SessionRecord, SessionStore
from ctxai.agent.tools.registry import ToolRegistry
from tests.mocks.mock_llm import MockLLMProvider


def _record(root: Path, context: ConversationContext, name: str = "work") -> SessionRecord:
    return SessionRecord(name, context, "mock", "mock-model", str(root.resolve()))


def test_session_save_resume_clear_export_and_secret_redaction(tmp_path: Path) -> None:
    context = ConversationContext(metadata={"token": "metadata-secret-value"})
    context.add_system_message("Repository assistant")
    context.add_user_message("Use api_key=super-secret-value and Bearer abcdefghijklmnop")
    context.add_assistant_message("Decision: keep the public interface stable")
    store = SessionStore(tmp_path)

    saved = store.save(_record(tmp_path, context))
    raw = saved.read_text(encoding="utf-8")
    assert "super-secret-value" not in raw
    assert "abcdefghijklmnop" not in raw
    assert "metadata-secret-value" not in raw
    assert raw.count("[REDACTED]") >= 3

    resumed = store.load("work")
    assert resumed.context.messages[-1].content.endswith("public interface stable")
    exported = store.export(resumed, tmp_path / "session.md")
    assert "[REDACTED]" in exported.read_text(encoding="utf-8")
    with pytest.raises(ValueError, match="inside the repository"):
        store.export(resumed, tmp_path.parent / "escaped-session.md")

    store.clear("work")
    assert not saved.exists()


def test_context_compaction_preserves_decisions_failures_and_valid_tool_messages() -> None:
    context = ConversationContext()
    context.add_system_message("system")
    context.add_user_message("Please change parser behavior " + "x" * 500)
    context.add_assistant_message(
        "I will inspect it",
        [ToolCall(id="read-1", name="read_file", parameters={"path": "parser.py"})],
    )
    context.add_tool_result("read-1", "read_file", "Test failure: parser regression " + "y" * 500)
    context.add_assistant_message("Decision: preserve compatibility and add tests " + "z" * 500)
    context.add_user_message("What remains open?")
    context.add_assistant_message("The documentation task remains open.")

    context.truncate_old_messages(max_tokens=120)

    assert context.messages[0].content == "system"
    summary = context.messages[1].content
    assert summary.startswith("Conversation summary")
    assert "Test failure" in summary
    assert "Decision" in summary
    assert context.messages[-1].content == "The documentation task remains open."
    restored = ConversationContext.from_dict(context.to_dict())
    assert [message.to_dict() for message in restored.messages] == [
        message.to_dict() for message in context.messages
    ]


@pytest.mark.asyncio
async def test_multi_turn_model_switch_preserves_context_and_provider_capabilities(tmp_path: Path) -> None:
    first = MockLLMProvider(responses=[{"content": "first answer"}])
    tools = ToolRegistry()
    agent = Agent(AgentLoopConfig(
        llm_provider=first,
        tool_registry=tools,
        agent_config=AgentConfig(),
        working_directory=tmp_path,
        available_indexes=[],
    ))
    await agent.process_message("first question")

    second = MockLLMProvider(responses=[{"content": "second answer"}])
    agent.llm = second
    await agent.process_message("follow-up question")

    sent = second.call_history[0]
    contents = [message["content"] for message in sent["messages"]]
    assert "first question" in contents
    assert "first answer" in contents
    assert contents[-1] == "follow-up question"
    capabilities = second.get_capabilities()
    assert capabilities == ProviderCapabilities(tools=True, streaming=True)


def test_session_schema_rejects_cross_repository_and_unsafe_names(tmp_path: Path) -> None:
    source = tmp_path / "source"
    other = tmp_path / "other"
    source.mkdir()
    other.mkdir()
    context = ConversationContext()
    context.add_system_message("system")
    store = SessionStore(source)
    saved = store.save(_record(source, context))

    other_store = SessionStore(other, storage_dir=saved.parent)
    with pytest.raises(ValueError, match="different repository"):
        other_store.load("work")
    with pytest.raises(ValueError, match="Session name"):
        store.save(_record(source, context, "../escape"))

    payload = json.loads(saved.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
