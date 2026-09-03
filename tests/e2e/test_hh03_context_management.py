"""HH-03 acceptance tests: context window management.

Runs the real agent loop, tool registry, and tools against scripted
MockLLMProvider responses to prove: a long tool-heavy run whose cumulative
context crosses the soft limit compacts before the next call and completes
with a correct final report, tool-call pairing survives compaction for both
provider message formatters, the system prompt survives every compaction,
per-run usage totals equal the sum of provider-reported per-call usage, and a
``length`` finish-reason response surfaces as INVALID_RESPONSE-class handling
instead of a crash.
"""

from __future__ import annotations

import re

import pytest

from ctxai.agent.config import AgentConfig, AgentLLMConfig
from ctxai.agent.context import ConversationContext
from ctxai.agent.core import Agent, AgentLoopConfig
from ctxai.agent.llm.anthropic_provider import AnthropicProvider
from ctxai.agent.llm.base import Message, MessageRole, ToolCall
from ctxai.agent.tools.execution import ToolExecutionContext
from ctxai.agent.tools.file_ops import ReadFileTool
from ctxai.agent.tools.registry import ToolRegistry
from tests.mocks.mock_llm import MockLLMProvider, create_mock_response

CONTEXT_SIZE = 600  # Injected model context window; soft limit = 480 tokens


def make_agent(temp_dir, mock_llm_config, provider: MockLLMProvider) -> Agent:
    """Build a real agent (loop + registry + read tool) over the given provider.

    Args:
        temp_dir: Project root for the run.
        mock_llm_config: LLM configuration for the provider.
        provider: Scripted LLM provider instance.

    Returns:
        The configured Agent.
    """
    context = ToolExecutionContext.for_project(temp_dir)
    registry = ToolRegistry()
    registry.register(ReadFileTool(context=context, max_output_chars=20_000))
    loop_config = AgentLoopConfig(
        llm_provider=provider,
        tool_registry=registry,
        agent_config=AgentConfig(),
        working_directory=temp_dir,
        available_indexes=[],
        max_iterations=12,
        require_user_approval=True,
        approval_callback=lambda call: True,
    )
    return Agent(loop_config)


def assert_no_orphan_tool_results(messages: list[Message]) -> None:
    """Assert every tool result is preceded by an assistant carrying its id.

    Validates pairing on the Message level and through both the OpenAI wire
    format and the Anthropic message formatter (acceptance criterion 2).

    Args:
        messages: Conversation messages after compaction.
    """
    seen_ids: set[str] = set()
    for message in messages:
        if message.role == MessageRole.ASSISTANT and message.tool_calls:
            seen_ids.update(call.id for call in message.tool_calls)
        if message.tool_call_id:
            assert message.tool_call_id in seen_ids, f"orphan tool result {message.tool_call_id}"

    openai_payload = [message.to_dict(format="openai") for message in messages]
    openai_ids: set[str] = set()
    for entry in openai_payload:
        for call in entry.get("tool_calls") or []:
            openai_ids.add(call["id"])
        if entry.get("role") == "tool":
            assert entry["tool_call_id"] in openai_ids, "openai formatter sees an orphan tool result"

    # The Anthropic formatter never touches provider state; call it unbound.
    anthropic_payload = AnthropicProvider._format_messages_for_anthropic(None, messages)
    anthropic_ids: set[str] = set()
    for entry in anthropic_payload:
        for block in entry.get("content") or []:
            if isinstance(block, dict) and block.get("type") == "tool_use":
                anthropic_ids.add(block["id"])
            if isinstance(block, dict) and block.get("type") == "tool_result":
                assert block["tool_use_id"] in anthropic_ids, "anthropic formatter sees an orphan tool result"


class LongToolSessionProvider(MockLLMProvider):
    """Mock provider scripting a tool-heavy run with heavy reported usage.

    Every response reports ``prompt_tokens`` equal to the injected context
    size, so the measured context estimate crosses the soft limit from the
    second call onward — exactly how a real tool-heavy run would drift toward
    the model's window.
    """

    def __init__(self, mock_llm_config: AgentLLMConfig, file_count: int):
        """Initialize the provider.

        Args:
            mock_llm_config: LLM configuration for the provider.
            file_count: Number of scripted note-reading tool turns.
        """
        usage = {"prompt_tokens": CONTEXT_SIZE, "completion_tokens": 8, "total_tokens": CONTEXT_SIZE + 8}
        responses = [
            create_mock_response(
                tool_calls=[{"name": "read_file", "parameters": {"path": f"note-{index}.txt"}}],
                usage=usage,
            )
            for index in range(file_count)
        ]
        responses.append(
            create_mock_response(
                content="All seven notes have been read and summarized.",
                usage={"prompt_tokens": CONTEXT_SIZE, "completion_tokens": 4, "total_tokens": CONTEXT_SIZE + 4},
            )
        )
        super().__init__(config=mock_llm_config, responses=responses, context_size=CONTEXT_SIZE)


@pytest.mark.e2e
@pytest.mark.agent
@pytest.mark.asyncio
async def test_long_tool_session_compacts_mid_run_and_completes(temp_dir, mock_llm_config, patch_embeddings_factory):
    """A run above the soft limit compacts before the next call and completes correctly."""
    for index in range(7):
        (temp_dir / f"note-{index}.txt").write_text("x" * 400, encoding="utf-8")
    provider = LongToolSessionProvider(mock_llm_config, file_count=7)
    agent = make_agent(temp_dir, mock_llm_config, provider)
    system_prompt = agent.context.messages[0].content

    report = await agent.process_message("Read each note file in order")

    # Criterion 1: the run crosses the soft limit, compacts, and still completes
    # with a correct final report.
    assert provider.call_count == 8, "seven tool turns plus the final response"
    assert agent.context.compaction_count >= 1, "compaction triggered above the soft limit"
    assert agent.context.elided_message_count >= 1, "old tool results were elided"
    assert "Status: succeeded" in report
    assert "All seven notes have been read and summarized." in report

    # Criterion 2: pairing survives for both openai and anthropic formatting.
    assert_no_orphan_tool_results(agent.context.messages)

    # Criterion 5: the system prompt survives every compaction unchanged.
    system_messages = [message for message in agent.context.messages if message.role == MessageRole.SYSTEM]
    assert [message.content for message in system_messages] == [system_prompt]

    # Criterion 3: usage totals equal the sum of per-call provider-reported usage.
    assert agent.last_run is not None
    totals = agent.last_run.usage.totals()
    assert totals["calls"] == 8
    assert totals["prompt_tokens"] == 8 * CONTEXT_SIZE
    assert totals["completion_tokens"] == 7 * 8 + 4
    assert totals["total_tokens"] == totals["prompt_tokens"] + totals["completion_tokens"]

    # Elision markers are honest about what was removed: the exact formatted
    # result length (file content plus tool formatting, so > the 400-char body)
    # for the tool that produced it.
    elided = [
        message for message in agent.context.messages if message.tool_call_id and message.content.startswith("[elided ")
    ]
    assert elided, "elided tool results carry explicit markers"
    for message in elided:
        match = re.fullmatch(r"\[elided (\d+) chars of tool result for read_file\]", message.content)
        assert match is not None, f"marker is not honest: {message.content}"
        assert int(match.group(1)) > 400


@pytest.mark.e2e
@pytest.mark.agent
@pytest.mark.asyncio
async def test_length_finish_reason_recovers_once_and_completes(temp_dir, mock_llm_config, patch_embeddings_factory):
    """A length-truncated response takes the INVALID_RESPONSE recovery path, not a crash."""
    provider = MockLLMProvider(
        config=mock_llm_config,
        responses=[
            create_mock_response(content="Partial ans", finish_reason="length"),
            create_mock_response(content="The complete final answer."),
        ],
    )
    agent = make_agent(temp_dir, mock_llm_config, provider)

    report = await agent.process_message("Answer the question")

    assert provider.call_count == 2, "exactly one recovery attempt after the truncation"
    recovery_prompts = [
        message.content
        for message in agent.context.messages
        if message.role == MessageRole.USER and "failed with the following error" in message.content
    ]
    assert len(recovery_prompts) == 1, "INVALID_RESPONSE-class handling injects one recovery prompt"
    assert "Status: succeeded" in report
    assert "The complete final answer." in report


@pytest.mark.e2e
@pytest.mark.agent
@pytest.mark.asyncio
async def test_persistent_length_finish_reason_fails_without_crashing(
    temp_dir, mock_llm_config, patch_embeddings_factory
):
    """Persistent truncation fails the run with invalid_response-class handling."""
    provider = MockLLMProvider(
        config=mock_llm_config,
        responses=[
            create_mock_response(content="Partial ans", finish_reason="length"),
            create_mock_response(content="Partial ans", finish_reason="length"),
        ],
    )
    agent = make_agent(temp_dir, mock_llm_config, provider)

    report = await agent.process_message("Answer the question")

    assert report.startswith("Status: failed")
    assert "truncated" in report, "the failure names the truncation cause"
    assert agent.last_run is not None
    assert agent.last_run.failure_kind is not None


@pytest.mark.e2e
@pytest.mark.agent
def test_compaction_is_deterministic_for_identical_history(mock_llm_config):
    """Criterion 4: identical histories compact to identical message lists."""

    def build() -> ConversationContext:
        context = ConversationContext()
        context.add_system_message("system prompt")
        context.add_user_message("Inspect everything")
        for index in range(9):
            context.add_assistant_message(
                f"step {index}",
                [ToolCall(id=f"call-{index}", name="read_file", parameters={"path": f"f{index}.txt"})],
            )
            context.add_tool_result(f"call-{index}", "read_file", "x" * 500)
        context.add_assistant_message("Done inspecting.")
        return context

    first = build()
    second = build()
    first.compact(target_tokens=100, keep_recent=6, max_output_chars=20_000)
    second.compact(target_tokens=100, keep_recent=6, max_output_chars=20_000)

    assert [message.to_dict() for message in first.messages] == [message.to_dict() for message in second.messages]
    assert any(message.content.startswith("[elided ") for message in first.messages)
    assert_no_orphan_tool_results(first.messages)


@pytest.mark.e2e
@pytest.mark.agent
@pytest.mark.asyncio
async def test_usage_capture_stores_tokens_only(temp_dir, mock_llm_config, patch_embeddings_factory):
    """The usage ledger records token counts and never message content."""
    (temp_dir / "secret-note.txt").write_text("super-secret-content", encoding="utf-8")
    provider = MockLLMProvider(
        config=mock_llm_config,
        responses=[
            create_mock_response(
                tool_calls=[{"name": "read_file", "parameters": {"path": "secret-note.txt"}}],
                usage={"prompt_tokens": 12, "completion_tokens": 2, "total_tokens": 14},
            ),
            create_mock_response(
                content="Read it.", usage={"prompt_tokens": 15, "completion_tokens": 3, "total_tokens": 18}
            ),
        ],
    )
    agent = make_agent(temp_dir, mock_llm_config, provider)

    await agent.process_message("Read the note")

    assert agent.last_run is not None
    serialized = [record.__dict__ for record in agent.last_run.usage.records]
    assert serialized
    for record in serialized:
        assert set(record) == {"provider", "model", "prompt_tokens", "completion_tokens", "total_tokens"}
        assert all(isinstance(value, int) for key, value in record.items() if key not in ("provider", "model"))
        assert "super-secret-content" not in str(record)
    assert agent.last_run.usage.totals()["total_tokens"] == 32
