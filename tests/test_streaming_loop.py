"""Unit tests for the HH-05 streaming agent loop.

Drives the real agent loop (shared ``_run_loop`` core, real tool registry, real
tools) against a scripted streaming ``MockLLMProvider`` to prove: tool turns
emit token/tool/approval events in order, ``stream_message``'s final report is
identical to ``process_message``'s return, non-streaming providers degrade
through the documented fallback, cancellation and compaction surface as
events, approvals gate execution, and HH-04 transcripts are unchanged by
streaming mode.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from ctxai.agent.config import AgentConfig, AgentLLMConfig
from ctxai.agent.core import Agent, AgentLoopConfig
from ctxai.agent.events import AgentEvent, AgentEventKind
from ctxai.agent.run_recorder import RunEvent, runs_dir_for
from ctxai.agent.tools.bash_tool import BashTool
from ctxai.agent.tools.execution import ToolExecutionContext
from ctxai.agent.tools.file_ops import ReadFileTool, WriteFileTool
from ctxai.agent.tools.registry import ToolRegistry
from ctxai.agent.workflow import FailureKind
from tests.mocks.mock_llm import MockLLMProvider

USAGE_ONE = {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}
USAGE_TWO = {"prompt_tokens": 140, "completion_tokens": 12, "total_tokens": 152}


@pytest.fixture
def mock_llm_config() -> AgentLLMConfig:
    """LLM configuration for the mock provider.

    Returns:
        An AgentLLMConfig shaped like the e2e fixture.
    """
    return AgentLLMConfig(
        provider="mock", model="mock-model-v1", api_key="mock-key", temperature=0.7, max_tokens=4096, timeout=30
    )


def make_agent(
    temp_dir,
    provider: MockLLMProvider,
    *,
    behavior: dict[str, Any] | None = None,
    approval_callback=None,
    cancel_event: asyncio.Event | None = None,
    run_id: str | None = None,
    session_store=None,
    max_output_chars: int = 100_000,
    tools_max_output_chars: int | None = None,
) -> Agent:
    """Build a real agent (loop + registry + read/write tools) over the provider.

    Args:
        temp_dir: Project root for the run.
        provider: Scripted LLM provider instance.
        behavior: Optional overrides for ``AgentBehaviorConfig`` fields.
        approval_callback: Optional approval callback (default approves once).
        cancel_event: Optional cancel event installed on the loop config.
        run_id: Optional pinned transcript run id.
        session_store: Optional session store for cancellation persistence.
        max_output_chars: Read-tool output cap (raised so compaction tests
            control truncation through the budget, not the tool).
        tools_max_output_chars: Optional ``AgentToolsConfig.max_output_chars``
            override (the cap compaction uses; defaults to the config default).

    Returns:
        The configured Agent.
    """
    agent_config = AgentConfig()
    if behavior:
        for key, value in behavior.items():
            setattr(agent_config.behavior, key, value)
    if tools_max_output_chars is not None:
        agent_config.tools.max_output_chars = tools_max_output_chars
    context = ToolExecutionContext.for_project(temp_dir)
    registry = ToolRegistry()
    registry.register(ReadFileTool(context=context, max_output_chars=max_output_chars))
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
        run_id=run_id,
        session_store=session_store,
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


def script_tool_turn(temp_dir) -> list[dict[str, Any]]:
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


@pytest.mark.asyncio
async def test_streaming_tool_turn_emits_scripted_event_sequence(temp_dir, mock_llm_config):
    """A scripted tool turn streams token deltas, tool events, usage, and the final report."""
    provider = MockLLMProvider(supports_streaming=True, responses=script_tool_turn(temp_dir))
    agent = make_agent(temp_dir, provider)

    events = await collect(agent, "Read the note")

    assert kinds(events)[-1] == "final_report", "the stream ends with the final report"
    # First LLM call: token deltas then the usage event.
    first_tokens: list[str] = []
    index = 0
    while events[index].kind is AgentEventKind.TOKEN:
        first_tokens.append(events[index].text)
        index += 1
    assert first_tokens == ["Reading ", "the ", "note ", "now"], "deltas arrive incrementally"
    assert events[index].kind is AgentEventKind.USAGE
    assert events[index].data == USAGE_ONE
    index += 1
    # Tool execution events.
    assert events[index].kind is AgentEventKind.TOOL_CALL_STARTED
    assert events[index].text == "read_file"
    assert events[index].data["tool"] == "read_file"
    assert events[index].data["parameters"] == {"path": "note.txt"}
    index += 1
    assert events[index].kind is AgentEventKind.TOOL_RESULT
    assert events[index].data["success"] is True
    index += 1
    # Second LLM call: token deltas then usage.
    second_tokens: list[str] = []
    while events[index].kind is AgentEventKind.TOKEN:
        second_tokens.append(events[index].text)
        index += 1
    assert "".join(second_tokens) == "The note says hello"
    assert events[index].kind is AgentEventKind.USAGE
    assert events[index].data == USAGE_TWO
    index += 1
    # Final report closes the stream.
    assert index == len(events) - 1
    report_event = events[-1]
    assert report_event.kind is AgentEventKind.FINAL_REPORT
    assert "The note says hello" in report_event.text
    assert report_event.data["status"] == "succeeded"

    # The token deltas were streamed during the same turns where tools were
    # advertised: the first token group precedes the tool events.
    assert kinds(events).index("tool_call_started") > 0
    assert events[0].kind is AgentEventKind.TOKEN


@pytest.mark.asyncio
async def test_stream_final_report_equals_process_message_return(temp_dir, mock_llm_config):
    """Criterion 2: the streamed final report is identical to the buffered return."""
    streaming_agent = make_agent(
        temp_dir, MockLLMProvider(supports_streaming=True, responses=script_tool_turn(temp_dir))
    )
    buffered_agent = make_agent(temp_dir, MockLLMProvider(responses=script_tool_turn(temp_dir)))

    events = await collect(streaming_agent, "Read the note")
    buffered_report = await buffered_agent.process_message("Read the note")

    final_events = [event for event in events if event.kind is AgentEventKind.FINAL_REPORT]
    assert len(final_events) == 1, "exactly one final report event is emitted"
    assert final_events[0].text == buffered_report, "streamed and buffered finals diverge"
    assert buffered_report.startswith("Status: succeeded")


@pytest.mark.asyncio
async def test_provider_without_streaming_support_falls_back_with_diagnostic(temp_dir, mock_llm_config):
    """Criterion 3: a fallback provider completes the run with a documented diagnostic."""
    provider = MockLLMProvider(supports_streaming=False, responses=script_tool_turn(temp_dir))
    assert provider.get_capabilities().streaming is False
    agent = make_agent(temp_dir, provider)

    events = await collect(agent, "Read the note")

    diagnostics = [event for event in events if event.kind is AgentEventKind.STATUS]
    assert any("does not support token streaming" in event.text for event in diagnostics), (
        "the fallback carries a documented diagnostic"
    )
    assert diagnostics[0].data.get("streaming") is False
    # Graceful degradation: one buffered text event per LLM call, not deltas.
    token_events = [event for event in events if event.kind is AgentEventKind.TOKEN]
    assert [event.text for event in token_events] == ["Reading the note now", "The note says hello"]
    assert kinds(events)[-1] == "final_report"
    assert "Status: succeeded" in events[-1].text


@pytest.mark.asyncio
async def test_stream_responses_false_forces_the_buffered_path(temp_dir, mock_llm_config):
    """``stream_responses: false`` forces the fallback even for streaming-capable providers."""
    provider = MockLLMProvider(supports_streaming=True, responses=script_tool_turn(temp_dir))
    agent = make_agent(temp_dir, provider, behavior={"stream_responses": False})

    events = await collect(agent, "Read the note")

    token_events = [event for event in events if event.kind is AgentEventKind.TOKEN]
    assert [event.text for event in token_events] == ["Reading the note now", "The note says hello"], (
        "the provider call uses chat(), not stream_chat_events"
    )
    assert not [event for event in events if event.kind is AgentEventKind.STATUS], (
        "a configured fallback is silent (the user opted out)"
    )
    assert kinds(events)[-1] == "final_report"


class CancelOnSecondCallProvider(MockLLMProvider):
    """Mock provider that sets the cancel event while its second call runs."""

    def __init__(self, config: AgentLLMConfig, cancel_event: asyncio.Event, responses: list[dict[str, Any]]):
        """Initialize the provider.

        Args:
            config: LLM configuration.
            cancel_event: Event set when the second chat call starts.
            responses: Scripted tool-call responses.
        """
        super().__init__(config=config, responses=responses, supports_streaming=True)
        self.cancel_event = cancel_event

    def chat(self, messages: list[Any], tools: list[dict] | None = None, **kwargs) -> Any:
        """Cancel the run when the second call starts.

        Args:
            messages: Conversation messages.
            tools: Optional tool schemas.
            **kwargs: Additional provider arguments.

        Returns:
            The scripted LLMResponse.
        """
        if self.call_count >= 1:
            self.cancel_event.set()
        return super().chat(messages, tools=tools, **kwargs)


@pytest.mark.asyncio
async def test_cancellation_during_streaming_produces_hh02_outcome(temp_dir, mock_llm_config):
    """Criterion 4: cancellation mid-stream yields the HH-02 cancellation outcome."""
    cancel_event = asyncio.Event()
    responses = [
        {
            "content": "Reading",
            "tool_calls": [{"name": "read_file", "parameters": {"path": "note.txt"}}],
            "usage": USAGE_ONE,
        }
    ] * 10
    (temp_dir / "note.txt").write_text("contents", encoding="utf-8")
    provider = CancelOnSecondCallProvider(mock_llm_config, cancel_event, responses)
    agent = make_agent(temp_dir, provider, cancel_event=cancel_event)

    events = await collect(agent, "Keep reading the note")

    assert kinds(events)[-1] == "final_report", "the generator ends cleanly with the final report"
    assert events[-1].text.startswith("Status: failed")
    assert agent.last_run is not None
    assert agent.last_run.failure_kind is FailureKind.INFRASTRUCTURE_FAILURE
    assert provider.call_count == 2, "cancellation stops the loop at the next iteration boundary"
    recovery = [
        message.content
        for message in agent.context.messages
        if message.role.value == "user" and "failed with the following error" in message.content
    ]
    assert recovery == [], "cancellation never injects a recovery prompt"


@pytest.mark.asyncio
async def test_approval_required_pauses_stream_until_decision(temp_dir, mock_llm_config):
    """Criterion 5: the mutation executes only after the decision event, in stream order."""
    decisions: list[str] = []

    def approve(call) -> bool:
        if call.name == "write_file":
            assert not (temp_dir / "out.txt").exists(), "the write must not run before the decision"
        decisions.append("decided")
        return True

    provider = MockLLMProvider(
        supports_streaming=True,
        responses=[
            {
                "content": "Writing the file",
                "tool_calls": [{"name": "write_file", "parameters": {"path": "out.txt", "content": "written"}}],
                "usage": USAGE_ONE,
            },
            {"content": "Verifying", "tool_calls": [{"name": "bash", "parameters": {"command": "cat out.txt"}}]},
            {"content": "Wrote out.txt and verified it", "usage": USAGE_TWO},
        ],
    )
    agent = make_agent(temp_dir, provider, approval_callback=approve)

    events = await collect(agent, "Create out.txt")

    order = kinds(events)
    assert order.index("tool_call_started") < order.index("approval_required") < order.index("approval_decided")
    assert order.index("approval_decided") < order.index("tool_result"), "the tool runs only after the decision"
    assert len(decisions) == 2, "write and verify each paused on a decision"
    approval_required = next(event for event in events if event.kind is AgentEventKind.APPROVAL_REQUIRED)
    assert "write_file" in approval_required.text
    assert approval_required.data["tool"] == "write_file"
    approval_decided = next(event for event in events if event.kind is AgentEventKind.APPROVAL_DECIDED)
    assert approval_decided.data["approved"] is True
    write_result = next(event for event in events if event.kind is AgentEventKind.TOOL_RESULT)
    assert write_result.data["success"] is True
    assert (temp_dir / "out.txt").read_text(encoding="utf-8") == "written"
    assert events[-1].text.startswith("Status: succeeded")


@pytest.mark.asyncio
async def test_approval_denial_blocks_mutation_in_streaming(temp_dir, mock_llm_config):
    """A denied approval never executes the mutation on the streaming path."""

    def deny(call) -> bool:
        return False

    provider = MockLLMProvider(
        supports_streaming=True,
        responses=[
            {
                "content": "Writing the file",
                "tool_calls": [{"name": "write_file", "parameters": {"path": "out.txt", "content": "written"}}],
                "usage": USAGE_ONE,
            },
            {"content": "The write was denied", "usage": USAGE_TWO},
        ],
    )
    agent = make_agent(temp_dir, provider, approval_callback=deny)

    events = await collect(agent, "Create out.txt")

    approval_decided = next(event for event in events if event.kind is AgentEventKind.APPROVAL_DECIDED)
    assert approval_decided.data["approved"] is False
    tool_result = next(event for event in events if event.kind is AgentEventKind.TOOL_RESULT)
    assert tool_result.data["success"] is False
    assert not (temp_dir / "out.txt").exists(), "a denied approval never executes"
    assert events[-1].text.startswith("Status: failed")
    assert agent.last_run is not None
    assert agent.last_run.failure_kind is FailureKind.APPROVAL_DENIAL


class SmallWindowProvider(MockLLMProvider):
    """Mock provider reporting a tiny context window so every call overflows."""

    def __init__(self, config: AgentLLMConfig, responses: list[dict[str, Any]], context_size: int):
        """Initialize the provider.

        Args:
            config: LLM configuration.
            responses: Scripted responses.
            context_size: The injected context window size.
        """
        super().__init__(config=config, responses=responses, context_size=context_size, supports_streaming=True)


@pytest.mark.asyncio
async def test_compaction_mid_stream_emits_status_event(temp_dir, mock_llm_config):
    """A compaction triggered mid-stream emits a status event through the shared sink."""
    (temp_dir / "big.txt").write_text("x" * 30_000, encoding="utf-8")
    usage = {"prompt_tokens": 200, "completion_tokens": 8, "total_tokens": 208}
    provider = SmallWindowProvider(
        mock_llm_config,
        responses=[
            {
                "content": "Reading the big file",
                "tool_calls": [{"name": "read_file", "parameters": {"path": "big.txt"}}],
                "usage": usage,
            },
            {"content": "Summarized the big file", "usage": usage},
        ],
        context_size=200,
    )
    agent = make_agent(temp_dir, provider)

    events = await collect(agent, "Summarize big.txt")

    compactions = [
        event for event in events if event.kind is AgentEventKind.STATUS and event.text.startswith("context compacted:")
    ]
    assert len(compactions) == 1, "exactly one compaction status event"
    assert compactions[0].data["target_tokens"] == 160, "budget = context_size * soft limit ratio"
    assert compactions[0].data["tokens_after"] < compactions[0].data["tokens_before"]
    assert agent.context.compaction_count == 1
    assert events[-1].text.startswith("Status: succeeded")


def read_transcript(temp_dir, run_id: str) -> list[RunEvent]:
    """Parse a run transcript from disk.

    Args:
        temp_dir: Project root.
        run_id: The pinned run id.

    Returns:
        The parsed RunEvent list.
    """
    path = runs_dir_for(temp_dir) / f"{run_id}.jsonl"
    return [RunEvent.from_dict(json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines() if line]


@pytest.mark.asyncio
async def test_streaming_transcript_matches_buffered_transcript(temp_dir, mock_llm_config):
    """HH-04 transcripts are unchanged by streaming mode: same kinds, same usage."""
    streaming_agent = make_agent(
        temp_dir,
        MockLLMProvider(supports_streaming=True, responses=script_tool_turn(temp_dir)),
        run_id="stream-run",
    )
    buffered_agent = make_agent(
        temp_dir,
        MockLLMProvider(responses=script_tool_turn(temp_dir)),
        run_id="buffered-run",
    )

    await collect(streaming_agent, "Read the note")
    await buffered_agent.process_message("Read the note")

    streamed = read_transcript(temp_dir, "stream-run")
    buffered = read_transcript(temp_dir, "buffered-run")
    assert [event.kind for event in streamed] == [event.kind for event in buffered], (
        "streaming does not add, remove, or reorder transcript events"
    )
    streamed_usage = [event.usage for event in streamed if event.usage is not None]
    buffered_usage = [event.usage for event in buffered if event.usage is not None]
    assert streamed_usage == buffered_usage, "per-call usage records are identical"
    assert not any("Reading " == (event.payload or {}).get("content") for event in streamed), (
        "token deltas are never persisted"
    )
