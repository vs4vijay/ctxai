"""HH-02 acceptance tests: resilient agent loop.

Runs the real agent loop, tool registry, and tools against scripted
MockLLMProvider subclasses to prove: transient rate-limit failures are retried
transparently with bounded exponential backoff, authentication errors fail fast
with a provider-qualified message and no recovery prompt, cancellation mid-loop
marks the run failed and preserves a reloadable session, and identical
tool-result loops end with a status-bearing final report.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import pytest

from ctxai.agent.config import AgentConfig, AgentLLMConfig
from ctxai.agent.core import Agent, AgentLoopConfig
from ctxai.agent.llm.base import LLMResponse, ProviderError, ProviderErrorKind
from ctxai.agent.resilience import RetryNotice, RetryPolicy, format_retry_notice
from ctxai.agent.sessions import SessionStore
from ctxai.agent.tools.execution import ToolExecutionContext
from ctxai.agent.tools.file_ops import ReadFileTool
from ctxai.agent.tools.registry import ToolRegistry
from ctxai.agent.workflow import FailureKind, TaskState
from tests.mocks.mock_llm import MockLLMProvider, create_mock_response

FAST_POLICY = RetryPolicy(max_retries=3, base_delay_s=0.01, max_delay_s=0.04)
RECOVERY_PROMPT_MARKER = "failed with the following error"


class ScriptedFailureProvider(MockLLMProvider):
    """Mock provider that raises configured provider errors before its responses."""

    def __init__(
        self,
        config: AgentLLMConfig | None = None,
        responses: list[dict[str, Any]] | None = None,
        errors: list[ProviderError] | None = None,
    ):
        """Initialize the scripted provider.

        Args:
            config: LLM configuration.
            responses: Scripted responses served once the errors are exhausted.
            errors: Provider errors raised in order before the responses.
        """
        super().__init__(config=config, responses=responses or [])
        self.errors = list(errors or [])

    def chat(self, messages: list[Any], tools: list[dict] | None = None, **kwargs) -> LLMResponse:
        """Raise the next scripted error, otherwise defer to the scripted responses.

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


class CancelMidLoopProvider(MockLLMProvider):
    """Mock provider that sets a cancel event after its first chat call completes."""

    def __init__(
        self,
        config: AgentLLMConfig,
        cancel_event: asyncio.Event,
        responses: list[dict[str, Any]] | None = None,
    ):
        """Initialize the provider.

        Args:
            config: LLM configuration.
            cancel_event: Event set when the second chat call starts.
            responses: Scripted responses served until cancellation.
        """
        super().__init__(config=config, responses=responses or [])
        self.cancel_event = cancel_event

    def chat(self, messages: list[Any], tools: list[dict] | None = None, **kwargs) -> LLMResponse:
        """Cancel the run after the second call starts, then serve a scripted response.

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


def make_agent(
    temp_dir,
    mock_llm_config,
    provider,
    *,
    cancel_event: asyncio.Event | None = None,
    session_store: SessionStore | None = None,
    on_retry=None,
    max_iterations: int = 12,
) -> Agent:
    """Build a real agent (loop + registry + read tool) over the given provider.

    Args:
        temp_dir: Project root for the run.
        mock_llm_config: LLM configuration for the provider.
        provider: Scripted LLM provider instance.
        cancel_event: Optional cancel event installed on the loop config.
        session_store: Optional session store for cancellation persistence.
        on_retry: Optional retry-notification hook.
        max_iterations: Loop iteration budget.

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
        max_iterations=max_iterations,
        require_user_approval=True,
        approval_callback=lambda call: True,
        retry_policy=FAST_POLICY,
        cancel_event=cancel_event,
        on_retry=on_retry,
        session_store=session_store,
    )
    return Agent(loop_config)


def recovery_prompts(agent: Agent) -> list[str]:
    """Return injected recovery prompts from the conversation context.

    Args:
        agent: The agent whose context is inspected.

    Returns:
        Recovery prompt message contents.
    """
    return [
        message.content
        for message in agent.context.messages
        if message.role.value == "user" and RECOVERY_PROMPT_MARKER in message.content
    ]


@pytest.mark.e2e
@pytest.mark.agent
@pytest.mark.asyncio
async def test_rate_limit_failures_are_retried_and_run_completes(temp_dir, mock_llm_config, patch_embeddings_factory):
    """Two rate-limit errors then success completes the task with exactly two bounded waits."""
    provider = ScriptedFailureProvider(
        config=mock_llm_config,
        responses=[create_mock_response("The task completed.")],
        errors=[
            ProviderError(ProviderErrorKind.RATE_LIMIT, "429 too many requests", provider="ScriptedFailureProvider"),
            ProviderError(ProviderErrorKind.RATE_LIMIT, "429 slow down", provider="ScriptedFailureProvider"),
        ],
    )
    notices: list[RetryNotice] = []
    agent = make_agent(temp_dir, mock_llm_config, provider, on_retry=notices.append)

    report = await agent.process_message("Finish the task")

    assert provider.call_count == 3, "two failed attempts plus the successful one"
    assert len(notices) == 2, "exactly two retry waits occurred"
    for notice in notices:
        assert notice.kind == "rate_limit"
        # Backoff ceilings double per attempt and never exceed max_delay_s.
        ceiling = min(FAST_POLICY.max_delay_s, FAST_POLICY.base_delay_s * (2 ** (notice.attempt - 1)))
        assert 0 <= notice.delay_s <= ceiling
    assert notices[0].attempt == 1 and notices[1].attempt == 2
    assert format_retry_notice(notices[0]).startswith("retry 1/3 after ")
    assert "Status: succeeded" in report
    assert agent.last_run is not None and agent.last_run.state is TaskState.SUMMARIZE


@pytest.mark.e2e
@pytest.mark.agent
@pytest.mark.asyncio
async def test_authentication_error_fails_fast_with_precise_message(
    temp_dir, mock_llm_config, patch_embeddings_factory
):
    """An authentication error ends the run in one iteration, naming the provider."""
    provider = ScriptedFailureProvider(
        config=mock_llm_config,
        errors=[ProviderError(ProviderErrorKind.AUTHENTICATION, "invalid api key", provider="ScriptedFailureProvider")],
    )
    notices: list[RetryNotice] = []
    agent = make_agent(temp_dir, mock_llm_config, provider, on_retry=notices.append)

    report = await agent.process_message("Finish the task")

    assert provider.call_count == 1, "no retries and no iteration burn"
    assert notices == []
    assert "ScriptedFailureProvider" in report
    assert "authentication" in report
    assert report.startswith("Status: failed")
    assert agent.last_run is not None
    assert agent.last_run.failure_kind is FailureKind.INFRASTRUCTURE_FAILURE
    assert agent.last_run.state is TaskState.FAILED
    assert recovery_prompts(agent) == [], "no recovery prompt is added to the conversation"


@pytest.mark.e2e
@pytest.mark.agent
@pytest.mark.asyncio
async def test_cancellation_mid_loop_preserves_reloadable_session(temp_dir, mock_llm_config, patch_embeddings_factory):
    """Cancellation mid-loop produces a failed TaskRun and a session the store can reload."""
    (temp_dir / "note.txt").write_text("hello", encoding="utf-8")
    cancel_event = asyncio.Event()
    provider = CancelMidLoopProvider(
        mock_llm_config,
        cancel_event,
        responses=[create_mock_response(tool_calls=[{"name": "read_file", "parameters": {"path": "note.txt"}}])] * 10,
    )
    store = SessionStore(temp_dir)
    agent = make_agent(temp_dir, mock_llm_config, provider, cancel_event=cancel_event, session_store=store)

    report = await agent.process_message("Keep reading the note")

    assert provider.call_count == 2, "cancellation stops the loop at the next iteration boundary"
    assert report.startswith("Status: failed")
    assert agent.last_run is not None
    assert agent.last_run.failure_kind is FailureKind.INFRASTRUCTURE_FAILURE
    assert agent.last_run.state is TaskState.FAILED
    assert recovery_prompts(agent) == [], "cancellation never injects a recovery prompt"
    record = store.load("default")
    assert any(message.content == "Keep reading the note" for message in record.context.messages)


@pytest.mark.e2e
@pytest.mark.agent
@pytest.mark.asyncio
async def test_task_cancellation_marks_run_failed_and_saves_session(
    temp_dir, mock_llm_config, patch_embeddings_factory
):
    """asyncio task cancellation unwinds cleanly with a failed TaskRun and saved session."""

    running = asyncio.Event()

    class BlockingProvider(MockLLMProvider):
        def chat(self, messages, tools=None, **kwargs):
            running.set()
            time.sleep(0.05)
            return super().chat(messages, tools=tools, **kwargs)

    provider = BlockingProvider(
        config=mock_llm_config,
        responses=[create_mock_response("still working")] * 10,
    )
    store = SessionStore(temp_dir)
    agent = make_agent(temp_dir, mock_llm_config, provider, session_store=store, max_iterations=50)

    task = asyncio.create_task(agent.process_message("Long running task"))
    await running.wait()
    await asyncio.sleep(0.01)
    task.cancel()
    report = await asyncio.wait_for(task, timeout=5)

    assert report.startswith("Status: failed")
    assert agent.last_run is not None
    assert agent.last_run.failure_kind is FailureKind.INFRASTRUCTURE_FAILURE
    assert recovery_prompts(agent) == []
    assert store.load("default").project_root == str(temp_dir.resolve())


@pytest.mark.e2e
@pytest.mark.agent
@pytest.mark.asyncio
async def test_identical_tool_results_end_with_status_bearing_report(
    temp_dir, mock_llm_config, patch_embeddings_factory
):
    """Three identical consecutive tool results break the loop with evidence intact."""
    (temp_dir / "note.txt").write_text("same content", encoding="utf-8")
    provider = MockLLMProvider(
        config=mock_llm_config,
        responses=[create_mock_response(tool_calls=[{"name": "read_file", "parameters": {"path": "note.txt"}}])] * 10,
    )
    agent = make_agent(temp_dir, mock_llm_config, provider, max_iterations=20)

    report = await agent.process_message("Read the note repeatedly")

    assert provider.call_count == 3, "the third identical result tuple breaks the loop"
    assert report.startswith("Status:")
    assert "identical consecutive tool results" in report
    assert "Read the note repeatedly" not in report or "Outcome:" in report
    assert agent.last_run is not None
    assert agent.last_run.failure_kind is not None
