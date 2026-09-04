"""HH-07 acceptance tests: approval ergonomics and planner control.

Runs the real agent loop, tool registry, and tools against a ``MockLLMProvider``
to prove: a session-scope approval suppresses re-prompting for exactly the same
(tool, target) key within the session, a denial keeps the existing failure
report, a file that changed between approval and execution re-prompts with a
fresh diff (the stale approval never executes), and ``--plan force`` /
``--plan off`` override keyword planning classification. Transcripts record the
actual decision scope alongside the historical ``approved`` boolean.
"""

from __future__ import annotations

import sys
from collections.abc import Callable

import pytest

from ctxai.agent.approvals import APPROVAL_MEMORY_KEY, ApprovalDecision, ApprovalMemory
from ctxai.agent.config import AgentConfig
from ctxai.agent.core import Agent, AgentLoopConfig
from ctxai.agent.events import AgentEvent, AgentEventKind
from ctxai.agent.llm.base import ToolCall
from ctxai.agent.tools.bash_tool import BashTool
from ctxai.agent.tools.execution import ToolExecutionContext
from ctxai.agent.tools.file_ops import EditFileTool, ReadFileTool, WriteFileTool
from ctxai.agent.tools.registry import ToolRegistry
from ctxai.agent.workflow import TaskRun
from ctxai.commands.runs_command import read_run_events
from tests.mocks.mock_llm import MockLLMProvider, create_mock_response

USAGE = {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}


def make_agent(
    temp_dir,
    mock_llm_config,
    responses,
    decision_callback: Callable[[ToolCall], ApprovalDecision],
    *,
    plan_mode: str = "auto",
) -> Agent:
    """Build a real agent (loop + registry + file/bash tools) with a decision callback.

    Args:
        temp_dir: Project root for the run.
        mock_llm_config: LLM configuration for the mock provider.
        responses: Scripted MockLLMProvider responses.
        decision_callback: The HH-07 decision callback (returns ApprovalDecision).
        plan_mode: Planning override for the loop config.

    Returns:
        The configured Agent.
    """
    context = ToolExecutionContext.for_project(temp_dir)
    registry = ToolRegistry()
    registry.register(ReadFileTool(context=context))
    registry.register(WriteFileTool(context=context))
    registry.register(EditFileTool(context=context))
    registry.register(BashTool(AgentConfig().tools, context=context))
    llm = MockLLMProvider(config=mock_llm_config, responses=responses)
    loop_config = AgentLoopConfig(
        llm_provider=llm,
        tool_registry=registry,
        agent_config=AgentConfig(),
        working_directory=temp_dir,
        available_indexes=[],
        max_iterations=12,
        require_user_approval=True,
        plan_mode=plan_mode,
        approval_callback=decision_callback,
    )
    return Agent(loop_config)


def scripted_decisions(script: list[ApprovalDecision], asks: list[ToolCall]) -> Callable[[ToolCall], ApprovalDecision]:
    """Build a decision callback that records asks and pops a scripted queue.

    Args:
        script: The decision for each ask, in order.
        asks: List receiving every approval-shaped call the loop asked about.

    Returns:
        The decision callback for the loop configuration.
    """

    def decide(call: ToolCall) -> ApprovalDecision:
        """Record the ask and return the next scripted decision.

        Args:
            call: The approval-shaped tool call.

        Returns:
            The scripted decision.
        """
        asks.append(call)
        return script.pop(0)

    return decide


def approval_events(events: list[AgentEvent]) -> list[AgentEvent]:
    """Return the approval_decided events of a run.

    Args:
        events: The emitted AgentEvent list.

    Returns:
        The approval_decided events in emission order.
    """
    return [event for event in events if event.kind is AgentEventKind.APPROVAL_DECIDED]


def transcript_approvals(temp_dir, agent: Agent) -> list[dict]:
    """Read the approval event payloads of the agent's last run transcript.

    Args:
        temp_dir: Project root the transcript was recorded under.
        agent: The agent that completed the run.

    Returns:
        The approval payload dictionaries in on-disk order.
    """
    assert agent.last_run is not None and agent.last_run.run_id
    events = read_run_events(agent.last_run.run_id, temp_dir)
    return [event.payload for event in events if event.kind == "approval"]


@pytest.mark.e2e
@pytest.mark.agent
@pytest.mark.asyncio
async def test_session_scope_approval_suppresses_reprompting_for_same_tool_and_path(
    temp_dir, mock_llm_config, patch_embeddings_factory
):
    """Approve-session covers exactly the granted (tool, path) key for the session."""
    (temp_dir / "app.py").write_text("VALUE = 0\n", encoding="utf-8")
    responses = [
        create_mock_response(tool_calls=[{"name": "read_file", "parameters": {"path": "app.py"}}], usage=USAGE),
        create_mock_response(
            tool_calls=[{"name": "write_file", "parameters": {"path": "app.py", "content": "VALUE = 1\n"}}],
            usage=USAGE,
        ),
        create_mock_response(
            tool_calls=[{"name": "write_file", "parameters": {"path": "app.py", "content": "VALUE = 2\n"}}],
            usage=USAGE,
        ),
        create_mock_response(
            tool_calls=[{"name": "bash", "parameters": {"command": f"{sys.executable} -m py_compile app.py"}}],
            usage=USAGE,
        ),
        create_mock_response("Both writes verified.", usage=USAGE),
    ]
    asks: list[ToolCall] = []
    script = [ApprovalDecision.APPROVE_SESSION, ApprovalDecision.APPROVE_ONCE]
    agent = make_agent(temp_dir, mock_llm_config, responses, scripted_decisions(script, asks))

    events = [event async for event in agent.stream_message("Update app.py twice")]

    assert "Status: succeeded" in events[-1].text
    # The human was asked for the first write (session grant) and for the
    # ungranted bash verification — never again for the same write path.
    assert [call.name for call in asks] == ["write_file", "bash"]
    decided = approval_events(events)
    assert [event.data.get("decision") for event in decided] == ["session", "session", "once"]
    assert [event.data.get("source") for event in decided] == ["callback", "session_memory", "callback"]
    assert (temp_dir / "app.py").read_text(encoding="utf-8") == "VALUE = 2\n"
    # The grant lives in the conversation metadata (persisted with sessions).
    memory = ApprovalMemory.from_dict(agent.context.metadata[APPROVAL_MEMORY_KEY])
    assert memory.check("write_file", "app.py") is ApprovalDecision.APPROVE_SESSION
    # It suppresses exactly that key and nothing else.
    assert memory.check("write_file", "other.py") is None
    assert memory.check("edit_file", "app.py") is None
    assert memory.check("bash", "app.py") is None
    # The transcript reflects the actual decision and scope (criterion 5).
    approvals = transcript_approvals(temp_dir, agent)
    assert [payload["decision"] for payload in approvals] == ["session", "session", "once"]
    assert [payload["approved"] for payload in approvals] == [True, True, True]


@pytest.mark.e2e
@pytest.mark.agent
@pytest.mark.asyncio
async def test_denied_approval_keeps_the_existing_failure_report(temp_dir, mock_llm_config, patch_embeddings_factory):
    """A denial produces the APPROVAL_DENIAL failure report and writes nothing."""
    responses = [
        create_mock_response(
            tool_calls=[{"name": "write_file", "parameters": {"path": "out.txt", "content": "body"}}],
            usage=USAGE,
        ),
        create_mock_response("Task ended.", usage=USAGE),
    ]
    asks: list[ToolCall] = []
    agent = make_agent(temp_dir, mock_llm_config, responses, scripted_decisions([ApprovalDecision.DENY], asks))

    events = [event async for event in agent.stream_message("Create out.txt")]

    assert [call.name for call in asks] == ["write_file"]
    assert events[-1].kind is AgentEventKind.FINAL_REPORT
    assert "Status: failed" in events[-1].text
    assert events[-1].data.get("failure_kind") == "approval_denial"
    assert agent.last_run is not None and agent.last_run.failure_kind is not None
    assert agent.last_run.failure_kind.value == "approval_denial"
    assert not (temp_dir / "out.txt").exists()
    assert [event.data.get("decision") for event in approval_events(events)] == ["deny"]
    approvals = transcript_approvals(temp_dir, agent)
    assert len(approvals) == 1
    assert approvals[0]["approved"] is False
    assert approvals[0]["decision"] == "deny"


@pytest.mark.e2e
@pytest.mark.agent
@pytest.mark.asyncio
async def test_file_changed_between_approval_and_execution_reprompts(
    temp_dir, mock_llm_config, patch_embeddings_factory
):
    """A stale approval re-prompts with a fresh diff and the stale diff never executes."""
    (temp_dir / "app.py").write_text("VALUE = 1\n", encoding="utf-8")
    responses = [
        create_mock_response(tool_calls=[{"name": "read_file", "parameters": {"path": "app.py"}}], usage=USAGE),
        create_mock_response(
            tool_calls=[
                {
                    "name": "edit_file",
                    "parameters": {"path": "app.py", "old_text": "VALUE = 1", "new_text": "VALUE = 2"},
                }
            ],
            usage=USAGE,
        ),
        create_mock_response(
            tool_calls=[{"name": "bash", "parameters": {"command": f"{sys.executable} -m py_compile app.py"}}],
            usage=USAGE,
        ),
        create_mock_response("Edit verified.", usage=USAGE),
    ]
    asks: list[ToolCall] = []
    script = [ApprovalDecision.APPROVE_ONCE, ApprovalDecision.APPROVE_ONCE, ApprovalDecision.APPROVE_ONCE]

    def decide(call: ToolCall) -> ApprovalDecision:
        """Approve each ask, moving the file on exactly once at the first edit ask.

        The first edit ask touches the file after the human saw the diff but
        before the approval returns; the loop must detect the stale binding and
        re-prompt with a fresh diff instead of executing the stale one.

        Args:
            call: The approval-shaped tool call.

        Returns:
            The approval decision for this ask.
        """
        asks.append(call)
        if len(asks) == 1:
            (temp_dir / "app.py").write_text("VALUE = 1\nextra = 0\n", encoding="utf-8")
        return script.pop(0)

    agent = make_agent(temp_dir, mock_llm_config, responses, decide)

    events = [event async for event in agent.stream_message("Update the value")]

    edit_asks = [call for call in asks if call.name == "edit_file"]
    assert len(edit_asks) == 2, "the stale approval must re-prompt instead of executing"
    first_diff = edit_asks[0].parameters["proposed_diff"]
    second_diff = edit_asks[1].parameters["proposed_diff"]
    assert first_diff != second_diff, "the re-prompt must present a fresh diff"
    assert "extra = 0" in second_diff
    assert "Status: succeeded" in events[-1].text
    # The applied result is based on the fresh approval round, not the stale one.
    assert (temp_dir / "app.py").read_text(encoding="utf-8") == "VALUE = 2\nextra = 0\n"
    approvals = transcript_approvals(temp_dir, agent)
    assert [payload["decision"] for payload in approvals] == ["once", "once", "once"]
    assert len(approvals) == 3


@pytest.mark.e2e
@pytest.mark.agent
@pytest.mark.asyncio
async def test_plan_force_routes_a_simple_task_through_submit_plan(temp_dir, mock_llm_config, patch_embeddings_factory):
    """--plan force requires submit_plan for a task the keyword classifier would not flag."""
    goal = "Add a trailing marker to the module"
    assert TaskRun.resolve_plan_required(goal, "auto") is False, "the keyword classifier must not flag this task"
    (temp_dir / "app.py").write_text("VALUE = 1\n", encoding="utf-8")
    command = f"{sys.executable} -m py_compile app.py"
    responses = [
        create_mock_response(tool_calls=[{"name": "read_file", "parameters": {"path": "app.py"}}], usage=USAGE),
        create_mock_response(
            tool_calls=[
                {
                    "name": "submit_plan",
                    "parameters": {
                        "goal": goal,
                        "reasoning": "The inspected constant is the smallest change point.",
                        "actions": [
                            {
                                "action_id": "edit-value",
                                "description": "Append the trailing marker to the constant",
                                "tool": "edit_file",
                                "parameters": {
                                    "path": "app.py",
                                    "old_text": "VALUE = 1",
                                    "new_text": "VALUE = 2  # updated",
                                },
                                "evidence": ["app.py:1-1"],
                                "completion_criteria": "app.py contains the trailing marker",
                            },
                            {
                                "action_id": "compile",
                                "description": "Compile the changed module",
                                "tool": "bash",
                                "parameters": {"command": command},
                                "evidence": ["app.py:1-1"],
                                "completion_criteria": "The compiler exits successfully",
                            },
                        ],
                    },
                }
            ],
            usage=USAGE,
        ),
        create_mock_response(
            tool_calls=[
                {
                    "name": "edit_file",
                    "parameters": {"path": "app.py", "old_text": "VALUE = 1", "new_text": "VALUE = 2  # updated"},
                }
            ],
            usage=USAGE,
        ),
        create_mock_response(tool_calls=[{"name": "bash", "parameters": {"command": command}}], usage=USAGE),
        create_mock_response("Forced plan executed and verified.", usage=USAGE),
    ]
    asks: list[ToolCall] = []
    script = [ApprovalDecision.APPROVE_ONCE, ApprovalDecision.APPROVE_ONCE]
    agent = make_agent(temp_dir, mock_llm_config, responses, scripted_decisions(script, asks), plan_mode="force")

    report = await agent.process_message(goal)

    assert "Status: succeeded" in report
    assert "Plan progress: 2/2 completed, 0 failed" in report
    assert agent.last_run is not None and agent.last_run.plan is not None
    assert (temp_dir / "app.py").read_text(encoding="utf-8") == "VALUE = 2  # updated\n"
    assert [call.name for call in asks] == ["edit_file", "bash"]


@pytest.mark.e2e
@pytest.mark.agent
@pytest.mark.asyncio
async def test_plan_off_skips_planning_for_a_keyword_flagged_task(temp_dir, mock_llm_config, patch_embeddings_factory):
    """--plan off lets a flagged task mutate without submit_plan; tools stay policy-gated."""
    goal = "Refactor the module across files"
    assert TaskRun.resolve_plan_required(goal, "auto") is True, "the keyword classifier must flag this task"
    (temp_dir / "app.py").write_text("VALUE = 1\n", encoding="utf-8")
    command = f"{sys.executable} -m py_compile app.py"
    responses = [
        create_mock_response(tool_calls=[{"name": "read_file", "parameters": {"path": "app.py"}}], usage=USAGE),
        create_mock_response(
            tool_calls=[
                {
                    "name": "edit_file",
                    "parameters": {"path": "app.py", "old_text": "VALUE = 1", "new_text": "VALUE = 2"},
                }
            ],
            usage=USAGE,
        ),
        create_mock_response(tool_calls=[{"name": "bash", "parameters": {"command": command}}], usage=USAGE),
        create_mock_response("Unplanned edit completed and verified.", usage=USAGE),
    ]
    asks: list[ToolCall] = []
    script = [ApprovalDecision.APPROVE_ONCE, ApprovalDecision.APPROVE_ONCE]
    agent = make_agent(temp_dir, mock_llm_config, responses, scripted_decisions(script, asks), plan_mode="off")

    report = await agent.process_message(goal)

    assert "Status: succeeded" in report
    assert agent.last_run is not None and agent.last_run.plan is None
    assert agent.last_run.plan_required is False
    assert (temp_dir / "app.py").read_text(encoding="utf-8") == "VALUE = 2\n"
    # The approval gate remains in force: the human still approved the exact edit.
    assert [call.name for call in asks] == ["edit_file", "bash"]
