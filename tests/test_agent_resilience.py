"""Unit tests for the resilient agent loop (HH-02).

Runs the real agent loop over scripted providers to prove: provider error
kinds map to the declared outcomes (retry, fail fast, one recovery prompt),
cancellation is clean with session persistence, the LLM call does not block
the event loop, the hash-window loop detection breaks at the configured
threshold, and terminal exits return status-bearing final reports.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import pytest

from ctxai.agent.config import AgentBehaviorConfig, AgentConfig, AgentLLMConfig
from ctxai.agent.core import Agent, AgentLoopConfig
from ctxai.agent.llm.base import LLMResponse, ProviderError, ProviderErrorKind
from ctxai.agent.resilience import RetryNotice, RetryPolicy
from ctxai.agent.sessions import SessionStore
from ctxai.agent.tools.execution import ToolExecutionContext
from ctxai.agent.tools.file_ops import ReadFileTool
from ctxai.agent.tools.registry import ToolRegistry
from ctxai.agent.workflow import FailureKind, TaskState
from tests.mocks.mock_llm import MockLLMProvider, create_mock_response

FAST_POLICY = RetryPolicy(max_retries=3, base_delay_s=0.001, max_delay_s=0.004)

RECOVERY_PROMPT_MARKER = "failed with the following error"


@pytest.fixture
def mock_llm_config() -> AgentLLMConfig:
    """Provide an LLM configuration for the scripted mock provider.

    Returns:
        AgentLLMConfig for the mock provider.
    """
    return AgentLLMConfig(provider="mock", model="mock-model", api_key="mock-key")


def make_agent(
    temp_dir,
    config: AgentLLMConfig,
    provider: MockLLMProvider,
    *,
    retry_policy: RetryPolicy | None = None,
    cancel_event: asyncio.Event | None = None,
    session_store: SessionStore | None = None,
    session_name: str = "default",
    on_retry=None,
    max_iterations: int = 10,
    behavior_config: AgentConfig | None = None,
) -> Agent:
    """Build a real agent (loop + registry + read tool) over the given provider.

    Args:
        temp_dir: Project root for the run.
        config: LLM configuration used by the provider.
        provider: Scripted LLM provider instance.
        retry_policy: Retry policy for the loop (defaults to a fast test policy).
        cancel_event: Optional cancellation event installed on the loop config.
        session_store: Optional session store for cancellation persistence.
        session_name: Session name used when persisting state.
        on_retry: Optional retry-notification hook.
        max_iterations: Loop iteration budget.
        behavior_config: Optional full agent config (controls loop_break_threshold).

    Returns:
        The configured Agent.
    """
    context = ToolExecutionContext.for_project(temp_dir)
    registry = ToolRegistry()
    registry.register(ReadFileTool(context=context, max_output_chars=20_000))
    agent_config = behavior_config or AgentConfig()
    loop_config = AgentLoopConfig(
        llm_provider=provider,
        tool_registry=registry,
        agent_config=agent_config,
        working_directory=temp_dir,
        available_indexes=[],
        max_iterations=max_iterations,
        verbose=False,
        retry_policy=retry_policy or FAST_POLICY,
        cancel_event=cancel_event,
        on_retry=on_retry,
        session_store=session_store,
        session_name=session_name,
    )
    return Agent(loop_config)


class ScriptedErrorProvider(MockLLMProvider):
    """Mock provider that raises scripted provider errors before its responses."""

    def __init__(self, config=None, responses=None, errors: list[ProviderError] | None = None):
        """Initialize the scripted provider.

        Args:
            config: LLM configuration.
            responses: Scripted responses used once errors are exhausted.
            errors: Errors raised in order before the scripted responses.
        """
        super().__init__(config=config, responses=responses or [])
        self.errors = list(errors or [])

    def chat(self, messages: list[Any], tools: list[dict] | None = None, **kwargs) -> LLMResponse:
        """Raise the next scripted error, otherwise fall back to the mock responses.

        Args:
            messages: Conversation messages.
            tools: Optional tool schemas.
            **kwargs: Additional provider arguments.

        Returns:
            The scripted LLMResponse.
        """
        if self.errors:
            self.call_count += 1  # base chat() only counts attempts that return
            raise self.errors.pop(0)
        return super().chat(messages, tools=tools, **kwargs)


def recovery_prompts(agent: Agent) -> list[str]:
    """Return user-role messages that are injected recovery prompts.

    Args:
        agent: The agent whose context is inspected.

    Returns:
        Recovery prompt contents.
    """
    return [
        message.content
        for message in agent.context.messages
        if message.role.value == "user" and RECOVERY_PROMPT_MARKER in message.content
    ]


@pytest.mark.asyncio
async def test_rate_limit_errors_are_retried_and_run_completes(temp_dir, mock_llm_config):
    """Two rate-limit errors then a final response complete the run with exactly two waits."""
    provider = ScriptedErrorProvider(
        config=mock_llm_config,
        responses=[create_mock_response("All done.")],
        errors=[
            ProviderError(ProviderErrorKind.RATE_LIMIT, "429 slow down", provider="ScriptedErrorProvider"),
            ProviderError(ProviderErrorKind.RATE_LIMIT, "429 still busy", provider="ScriptedErrorProvider"),
        ],
    )
    notices: list[RetryNotice] = []
    agent = make_agent(temp_dir, mock_llm_config, provider, on_retry=notices.append)

    report = await agent.process_message("Finish the task")

    assert provider.call_count == 3, "two failed attempts plus the successful one"
    assert len(notices) == 2, "exactly two retry waits occurred"
    assert [notice.attempt for notice in notices] == [1, 2]
    assert [notice.kind for notice in notices] == ["rate_limit", "rate_limit"]
    ceilings = [FAST_POLICY.base_delay_s, FAST_POLICY.base_delay_s * 2]
    for notice, ceiling in zip(notices, ceilings):
        assert 0 <= notice.delay_s <= ceiling
    assert "Status: succeeded" in report
    assert agent.last_run is not None and agent.last_run.state is TaskState.SUMMARIZE


@pytest.mark.asyncio
async def test_authentication_error_fails_fast_with_provider_name(temp_dir, mock_llm_config):
    """An authentication error ends the run in one iteration without a recovery prompt."""
    provider = ScriptedErrorProvider(
        config=mock_llm_config,
        errors=[ProviderError(ProviderErrorKind.AUTHENTICATION, "invalid api key", provider="ScriptedErrorProvider")],
    )
    notices: list[RetryNotice] = []
    agent = make_agent(temp_dir, mock_llm_config, provider, on_retry=notices.append)

    report = await agent.process_message("Finish the task")

    assert provider.call_count == 1, "authentication errors never retry"
    assert notices == []
    assert "ScriptedErrorProvider" in report
    assert "authentication" in report
    assert report.startswith("Status: failed")
    assert agent.last_run is not None
    assert agent.last_run.failure_kind is FailureKind.INFRASTRUCTURE_FAILURE
    assert agent.last_run.state is TaskState.FAILED
    assert recovery_prompts(agent) == [], "no recovery prompt is injected"


@pytest.mark.asyncio
async def test_unsupported_error_fails_fast_with_provider_name(temp_dir, mock_llm_config):
    """An unsupported-capability error fails fast with a provider-qualified message."""
    provider = ScriptedErrorProvider(
        config=mock_llm_config,
        errors=[
            ProviderError(ProviderErrorKind.UNSUPPORTED, "tool calling disabled", provider="ScriptedErrorProvider")
        ],
    )
    agent = make_agent(temp_dir, mock_llm_config, provider)

    report = await agent.process_message("Finish the task")

    assert provider.call_count == 1
    assert "ScriptedErrorProvider" in report and "unsupported" in report
    assert report.startswith("Status: failed")
    assert recovery_prompts(agent) == []


@pytest.mark.asyncio
async def test_invalid_response_gets_one_recovery_prompt_then_fails(temp_dir, mock_llm_config):
    """INVALID_RESPONSE receives exactly one recovery prompt, then the run fails."""
    provider = ScriptedErrorProvider(
        config=mock_llm_config,
        errors=[ProviderError(ProviderErrorKind.INVALID_RESPONSE, "malformed payload")] * 2,
    )
    agent = make_agent(temp_dir, mock_llm_config, provider)

    report = await agent.process_message("Finish the task")

    assert provider.call_count == 2, "one recovery attempt, then the failure"
    prompts = recovery_prompts(agent)
    assert len(prompts) == 1, "exactly one recovery prompt for the malformed response"
    assert report.startswith("Status: failed")
    assert agent.last_run is not None
    assert agent.last_run.failure_kind is FailureKind.INFRASTRUCTURE_FAILURE


@pytest.mark.asyncio
async def test_retry_exhaustion_fails_with_precise_reason(temp_dir, mock_llm_config):
    """Persistent rate limiting exhausts retries and fails with a counted reason."""
    provider = ScriptedErrorProvider(
        config=mock_llm_config,
        errors=[ProviderError(ProviderErrorKind.RATE_LIMIT, "429 busy")] * 5,
    )
    agent = make_agent(temp_dir, mock_llm_config, provider)

    report = await agent.process_message("Finish the task")

    assert provider.call_count == 4, "initial attempt plus max_retries=3"
    assert "rate_limit" in report
    assert "3 retries" in report
    assert report.startswith("Status: failed")
    assert recovery_prompts(agent) == [], "retry exhaustion never injects a recovery prompt"
    assert agent.last_run is not None
    assert agent.last_run.failure_kind is FailureKind.INFRASTRUCTURE_FAILURE


@pytest.mark.asyncio
async def test_cancel_event_between_iterations_cancels_cleanly(temp_dir, mock_llm_config):
    """A set cancel event ends the run with a failed TaskRun and a saved session."""

    class CancelOnSecondCallProvider(MockLLMProvider):
        def chat(self, messages, tools=None, **kwargs):
            if self.call_count >= 1:  # deterministic: cancel while the second call runs
                cancel_event.set()
            return super().chat(messages, tools=tools, **kwargs)

    provider = CancelOnSecondCallProvider(
        config=mock_llm_config,
        responses=[
            create_mock_response(tool_calls=[{"name": "read_file", "parameters": {"path": "note.txt"}}]),
            create_mock_response(tool_calls=[{"name": "read_file", "parameters": {"path": "note.txt"}}]),
        ],
    )
    (temp_dir / "note.txt").write_text("hello", encoding="utf-8")
    cancel_event = asyncio.Event()
    store = SessionStore(temp_dir)
    agent = make_agent(
        temp_dir,
        mock_llm_config,
        provider,
        cancel_event=cancel_event,
        session_store=store,
        session_name="hh02",
    )

    report = await agent.process_message("Keep reading the note")

    assert provider.call_count == 2, "cancellation must not burn further iterations"
    assert report.startswith("Status: failed")
    assert agent.last_run is not None
    assert agent.last_run.failure_kind is FailureKind.INFRASTRUCTURE_FAILURE
    assert agent.last_run.state is TaskState.FAILED
    assert recovery_prompts(agent) == []
    record = store.load("hh02")
    assert any(message.content == "Keep reading the note" for message in record.context.messages)


@pytest.mark.asyncio
async def test_task_cancellation_marks_run_failed_and_saves_session(temp_dir, mock_llm_config):
    """asyncio task cancellation produces a failed run, a saved session, and no recovery prompt."""
    running = asyncio.Event()

    class BlockingProvider(MockLLMProvider):
        def chat(self, messages, tools=None, **kwargs):
            running.set()
            time.sleep(0.05)
            return super().chat(messages, tools=tools, **kwargs)

    provider = BlockingProvider(
        config=mock_llm_config,
        responses=[create_mock_response(tool_calls=[{"name": "read_file", "parameters": {"path": "note.txt"}}])] * 10,
    )
    (temp_dir / "note.txt").write_text("hello", encoding="utf-8")
    store = SessionStore(temp_dir)
    agent = make_agent(temp_dir, mock_llm_config, provider, session_store=store, session_name="hh02", max_iterations=50)

    task = asyncio.create_task(agent.process_message("Long running task"))
    await running.wait()
    await asyncio.sleep(0.01)
    task.cancel()
    report = await asyncio.wait_for(task, timeout=5)

    assert report.startswith("Status: failed")
    assert agent.last_run is not None
    assert agent.last_run.failure_kind is FailureKind.INFRASTRUCTURE_FAILURE
    assert agent.last_run.state is TaskState.FAILED
    assert recovery_prompts(agent) == []
    record = store.load("hh02")
    assert any(message.content == "Long running task" for message in record.context.messages)


@pytest.mark.asyncio
async def test_provider_cancelled_error_is_treated_as_cancellation(temp_dir, mock_llm_config):
    """A CANCELLED provider error follows the cancellation path, not the failure path."""
    provider = ScriptedErrorProvider(
        config=mock_llm_config,
        errors=[ProviderError(ProviderErrorKind.CANCELLED, "request cancelled", provider="ScriptedErrorProvider")],
    )
    store = SessionStore(temp_dir)
    agent = make_agent(temp_dir, mock_llm_config, provider, session_store=store, session_name="hh02")

    report = await agent.process_message("Finish the task")

    assert provider.call_count == 1
    assert report.startswith("Status: failed")
    assert agent.last_run is not None
    assert agent.last_run.failure_kind is FailureKind.INFRASTRUCTURE_FAILURE
    assert recovery_prompts(agent) == []
    assert store.load("hh02") is not None


@pytest.mark.asyncio
async def test_identical_tool_results_break_loop_at_threshold(temp_dir, mock_llm_config):
    """Three identical consecutive tool results end the run with a status-bearing report."""
    (temp_dir / "note.txt").write_text("same content", encoding="utf-8")
    provider = MockLLMProvider(
        config=mock_llm_config,
        responses=[create_mock_response(tool_calls=[{"name": "read_file", "parameters": {"path": "note.txt"}}])] * 10,
    )
    agent = make_agent(temp_dir, mock_llm_config, provider, max_iterations=20)

    report = await agent.process_message("Read the note repeatedly")

    assert provider.call_count == 3, "the third identical result breaks the loop"
    assert report.startswith("Status:")
    assert "failed" in report
    assert "identical consecutive tool results" in report
    assert agent.last_run is not None
    assert agent.last_run.failure_kind is not None


@pytest.mark.asyncio
async def test_loop_break_threshold_is_configurable(temp_dir, mock_llm_config):
    """A custom AgentBehaviorConfig.loop_break_threshold changes the break point."""
    (temp_dir / "note.txt").write_text("same content", encoding="utf-8")
    provider = MockLLMProvider(
        config=mock_llm_config,
        responses=[create_mock_response(tool_calls=[{"name": "read_file", "parameters": {"path": "note.txt"}}])] * 10,
    )
    agent = make_agent(
        temp_dir,
        mock_llm_config,
        provider,
        max_iterations=20,
        behavior_config=AgentConfig(behavior=AgentBehaviorConfig(loop_break_threshold=2)),
    )

    await agent.process_message("Read the note repeatedly")

    assert provider.call_count == 2


@pytest.mark.asyncio
async def test_max_iterations_returns_status_bearing_report(temp_dir, mock_llm_config):
    """The max-iterations exit returns run.final_report instead of a bare string."""
    (temp_dir / "a.txt").write_text("alpha", encoding="utf-8")
    (temp_dir / "b.txt").write_text("beta", encoding="utf-8")
    provider = MockLLMProvider(
        config=mock_llm_config,
        responses=[
            create_mock_response(tool_calls=[{"name": "read_file", "parameters": {"path": "a.txt"}}]),
            create_mock_response(tool_calls=[{"name": "read_file", "parameters": {"path": "b.txt"}}]),
        ],
    )
    agent = make_agent(temp_dir, mock_llm_config, provider, max_iterations=2)

    report = await agent.process_message("Read both files forever")

    assert provider.call_count == 2
    assert report.startswith("Status:")
    assert "max iterations (2)" in report.lower()
    assert "Changed files:" in report
    assert agent.last_run is not None
    assert agent.last_run.state is TaskState.FAILED


@pytest.mark.asyncio
async def test_llm_call_does_not_block_event_loop(temp_dir, mock_llm_config):
    """A concurrent task progresses while the sync provider call sleeps in a thread."""

    class BlockingProvider(MockLLMProvider):
        def chat(self, messages, tools=None, **kwargs):
            time.sleep(0.3)
            return super().chat(messages, tools=tools, **kwargs)

    provider = BlockingProvider(config=mock_llm_config, responses=[create_mock_response("done")])
    agent = make_agent(temp_dir, mock_llm_config, provider)

    ticks = 0

    async def ticker() -> None:
        nonlocal ticks
        while True:
            await asyncio.sleep(0.05)
            ticks += 1

    tick_task = asyncio.create_task(ticker())
    try:
        await asyncio.wait_for(agent.process_message("Say something"), timeout=10)
    finally:
        tick_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await tick_task

    assert ticks >= 3, f"expected the event loop to stay responsive, saw {ticks} ticks"
