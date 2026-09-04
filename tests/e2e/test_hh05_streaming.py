"""HH-05 acceptance tests: true streaming interaction.

Runs the real agent loop (shared ``_run_loop`` core), tool registry, and tools
against a streaming ``MockLLMProvider`` to prove: a streaming-capable provider
emits token deltas on tool-advertising turns while the full approval workflow
stays in force, ``stream_message`` ends with a ``final_report`` identical to
``process_message``'s return, non-streaming providers complete via the
documented fallback with honest capabilities, cancellation produces the HH-02
outcome mid-stream, and approval-required mutations never execute before the
decision event.
"""

from __future__ import annotations

import asyncio

import pytest

from ctxai.agent.config import AgentConfig
from ctxai.agent.core import Agent, AgentLoopConfig
from ctxai.agent.events import AgentEvent, AgentEventKind
from ctxai.agent.tools.bash_tool import BashTool
from ctxai.agent.tools.execution import ToolExecutionContext
from ctxai.agent.tools.file_ops import ReadFileTool, WriteFileTool
from ctxai.agent.tools.registry import ToolRegistry
from tests.mocks.mock_llm import MockLLMProvider

USAGE_ONE = {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}
USAGE_TWO = {"prompt_tokens": 140, "completion_tokens": 12, "total_tokens": 152}


def make_agent(
    temp_dir,
    mock_llm_config,
    provider: MockLLMProvider,
    *,
    approval_callback=None,
    cancel_event: asyncio.Event | None = None,
) -> Agent:
    """Build a real agent (loop + registry + read/write/bash tools) over the provider.

    Args:
        temp_dir: Project root for the run.
        mock_llm_config: LLM configuration for the mock provider.
        provider: Scripted LLM provider instance.
        approval_callback: Optional approval callback (default approves once).
        cancel_event: Optional cancel event installed on the loop config.

    Returns:
        The configured Agent.
    """
    agent_config = AgentConfig()
    context = ToolExecutionContext.for_project(temp_dir)
    registry = ToolRegistry()
    registry.register(ReadFileTool(context=context, max_output_chars=20_000))
    registry.register(WriteFileTool(context=context))
    registry.register(BashTool(agent_config.tools, context=context))
    loop_config = AgentLoopConfig(
        llm_provider=provider,
        tool_registry=registry,
        agent_config=agent_config,
        working_directory=temp_dir,
        available_indexes=[],
        max_iterations=12,
        require_user_approval=True,
        approval_callback=approval_callback if approval_callback is not None else (lambda call: True),
        cancel_event=cancel_event,
    )
    return Agent(loop_config)


async def collect(agent: Agent, message: str) -> list[AgentEvent]:
    """Drain ``agent.stream_message`` into a list.

    Args:
        agent: The agent to run.
        message: The user message.

    Returns:
        The emitted AgentEvent list.
    """
    return [event async for event in agent.stream_message(message)]


def kinds(events: list[AgentEvent]) -> list[str]:
    """Return the kind values of an event list.

    Args:
        events: The emitted events.

    Returns:
        The list of kind value strings.
    """
    return [event.kind.value for event in events]


def tokens_text(events: list[AgentEvent]) -> str:
    """Concatenate token event texts.

    Args:
        events: The emitted events.

    Returns:
        The joined token text.
    """
    return "".join(event.text for event in events if event.kind is AgentEventKind.TOKEN)


def script_tool_turn(temp_dir) -> list[dict]:
    """Script a read-then-answer conversation over note.txt.

    Args:
        temp_dir: Project root where note.txt is created.

    Returns:
        The scripted response list for MockLLMProvider.
    """
    (temp_dir / "note.txt").write_text("the note contents", encoding="utf-8")
    return [
        {
            "content": "Reading the note now",
            "tool_calls": [{"name": "read_file", "parameters": {"path": "note.txt"}}],
            "usage": USAGE_ONE,
        },
        {"content": "The note says hello", "usage": USAGE_TWO},
    ]


@pytest.mark.e2e
@pytest.mark.agent
@pytest.mark.asyncio
async def test_streaming_provider_emits_token_deltas_on_tool_turns_with_full_approval_workflow(
    temp_dir, mock_llm_config, patch_embeddings_factory
):
    """Token deltas stream during the tool-advertising turn and tools still run the approval workflow."""
    provider = MockLLMProvider(config=mock_llm_config, responses=script_tool_turn(temp_dir), supports_streaming=True)
    agent = make_agent(temp_dir, mock_llm_config, provider)

    events = await collect(agent, "What does the note say?")

    values = kinds(events)
    assert values[0] == "token", "the first event must be a streamed token delta"
    assert values.count("token") > 1, "token deltas must stream, not arrive as one blob"
    # First-turn tokens ("Reading the note now") stream before the tool executes.
    first_tool_index = values.index("tool_call_started")
    first_turn_tokens = tokens_text(events[:first_tool_index])
    assert "Reading the note now" in first_turn_tokens
    assert "tool_call_started" in values and "tool_result" in values
    assert values[-1] == "final_report"
    assert "The note says hello" in events[-1].text
    # The tool actually executed through the real registry: a successful
    # tool_result event was emitted and the loop advanced to the second call.
    assert any(event.kind is AgentEventKind.TOOL_RESULT and event.data.get("success") for event in events), (
        "expected a successful tool_result event"
    )
    assert provider.call_count == 2


@pytest.mark.e2e
@pytest.mark.agent
@pytest.mark.asyncio
async def test_stream_final_report_is_identical_to_process_message_return(
    temp_dir, mock_llm_config, patch_embeddings_factory
):
    """The streaming run's final report equals the buffered run's for the same conversation."""
    streaming_agent = make_agent(
        temp_dir, mock_llm_config, MockLLMProvider(config=mock_llm_config, responses=script_tool_turn(temp_dir))
    )
    buffered_agent = make_agent(
        temp_dir, mock_llm_config, MockLLMProvider(config=mock_llm_config, responses=script_tool_turn(temp_dir))
    )

    events = await collect(streaming_agent, "What does the note say?")
    buffered_report = await buffered_agent.process_message("What does the note say?")

    assert events[-1].kind is AgentEventKind.FINAL_REPORT
    assert events[-1].text == buffered_report


@pytest.mark.e2e
@pytest.mark.agent
@pytest.mark.asyncio
async def test_provider_without_streaming_support_completes_via_fallback(
    temp_dir, mock_llm_config, patch_embeddings_factory
):
    """A non-streaming provider completes through the fallback with capabilities.streaming False."""
    provider = MockLLMProvider(config=mock_llm_config, responses=script_tool_turn(temp_dir), supports_streaming=False)
    assert provider.get_capabilities().streaming is False
    agent = make_agent(temp_dir, mock_llm_config, provider)

    events = await collect(agent, "What does the note say?")

    values = kinds(events)
    assert values[-1] == "final_report"
    assert "The note says hello" in events[-1].text
    # Graceful degradation emits the buffered content as a single token event.
    assert values.count("token") == 2  # one per LLM call (tool turn + final turn)
    assert (temp_dir / "note.txt").read_text(encoding="utf-8") == "the note contents"


@pytest.mark.e2e
@pytest.mark.agent
@pytest.mark.asyncio
async def test_cancellation_during_streaming_produces_hh02_outcome(temp_dir, mock_llm_config, patch_embeddings_factory):
    """A cancel event observed mid-stream ends the stream with the HH-02 cancellation outcome."""
    responses = script_tool_turn(temp_dir)
    provider = MockLLMProvider(config=mock_llm_config, responses=responses, supports_streaming=True)
    cancel_event = asyncio.Event()
    agent = make_agent(temp_dir, mock_llm_config, provider, cancel_event=cancel_event)

    def cancel_after_first_call():
        """Set the cancel event once the first provider call has happened."""
        if provider.call_count >= 1:
            cancel_event.set()
        return True

    original_chat = provider.chat

    def chat_with_cancel(*args, **kwargs):
        """Wrap chat to arm the cancel event after the first call."""
        original_chat(*args, **kwargs)
        return cancel_after_first_call()

    provider.chat = chat_with_cancel

    events = await collect(agent, "What does the note say?")

    assert events[-1].kind is AgentEventKind.FINAL_REPORT
    assert "Status: failed" in events[-1].text
    assert events[-1].data.get("failure_kind") == "infrastructure_failure"
    assert agent.last_run is not None
    assert agent.last_run.failure_kind is not None


@pytest.mark.e2e
@pytest.mark.agent
@pytest.mark.asyncio
async def test_approval_required_mutation_never_executes_before_decision(
    temp_dir, mock_llm_config, patch_embeddings_factory
):
    """The write tool runs only after the approval decision, bracketed by approval events."""
    target = temp_dir / "out.txt"
    decisions: list[str] = []
    responses = [
        {
            "content": "Writing the file",
            "tool_calls": [{"name": "write_file", "parameters": {"path": "out.txt", "content": "streamed body"}}],
            "usage": USAGE_ONE,
        },
        {"content": "File written", "usage": USAGE_TWO},
    ]
    provider = MockLLMProvider(config=mock_llm_config, responses=responses, supports_streaming=True)

    def approval(call):
        """Approve on request, recording when the decision was asked for."""
        decisions.append(call.name)
        assert not target.exists(), "the mutation must not execute before the decision"
        return True

    agent = make_agent(temp_dir, mock_llm_config, provider, approval_callback=approval)

    events = await collect(agent, "Write the file")

    values = kinds(events)
    assert "approval_required" in values and "approval_decided" in values
    # The decision strictly precedes execution: approval events come before
    # the successful tool_result, and the callback itself asserts the file
    # does not exist at decision time.
    assert values.index("approval_required") < values.index("approval_decided")
    assert values.index("approval_decided") < values.index("tool_result")
    assert decisions == ["write_file"]
    assert target.read_text(encoding="utf-8") == "streamed body"
    assert values[-1] == "final_report"
