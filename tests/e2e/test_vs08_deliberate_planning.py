"""VS-08 acceptance tests for deliberate, evidence-backed execution plans."""

from __future__ import annotations

import sys

import pytest

from ctxai.agent.config import AgentConfig
from ctxai.agent.core import Agent, AgentLoopConfig
from ctxai.agent.tools.bash_tool import BashTool
from ctxai.agent.tools.execution import ToolExecutionContext
from ctxai.agent.tools.file_ops import EditFileTool, ReadFileTool
from ctxai.agent.tools.registry import ToolRegistry
from ctxai.agent.workflow import FailureKind, format_approval_prompt
from tests.mocks.mock_llm import MockLLMProvider, create_mock_response


def make_agent(temp_dir, mock_llm_config, responses, approvals):
    context = ToolExecutionContext.for_project(temp_dir)
    registry = ToolRegistry()
    registry.register(ReadFileTool(context=context))
    registry.register(EditFileTool(context=context))
    registry.register(BashTool(AgentConfig().tools, context=context))
    llm = MockLLMProvider(config=mock_llm_config, responses=responses)
    return Agent(AgentLoopConfig(
        llm_provider=llm,
        tool_registry=registry,
        agent_config=AgentConfig(),
        working_directory=temp_dir,
        available_indexes=[],
        max_iterations=12,
        planning_enabled=True,
        require_user_approval=True,
        approval_callback=lambda call: approvals.append(call) or True,
    ))


def plan_call(command):
    return {"name": "submit_plan", "parameters": {
        "goal": "Refactor the configured value and verify the module",
        "reasoning": "The inspected constant is the smallest safe change point.",
        "actions": [
            {
                "action_id": "edit-value",
                "description": "Update the value",
                "tool": "edit_file",
                "parameters": {"path": "app.py", "old_text": "VALUE = 1", "new_text": "VALUE = 2"},
                "evidence": ["app.py:1-1"],
                "completion_criteria": "app.py contains VALUE = 2",
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
    }}


@pytest.mark.e2e
@pytest.mark.agent
@pytest.mark.asyncio
async def test_complex_task_plans_approves_exact_actions_and_tracks_progress(
    temp_dir, mock_llm_config
):
    (temp_dir / "app.py").write_text("VALUE = 1\n")
    command = f"{sys.executable} -m py_compile app.py"
    approvals = []
    agent = make_agent(temp_dir, mock_llm_config, [
        create_mock_response(tool_calls=[{"name": "read_file", "parameters": {"path": "app.py"}}]),
        create_mock_response(tool_calls=[plan_call(command)]),
        create_mock_response(tool_calls=[{"name": "edit_file", "parameters": {
            "path": "app.py", "old_text": "VALUE = 1", "new_text": "VALUE = 2",
        }}]),
        create_mock_response(tool_calls=[{"name": "bash", "parameters": {"command": command}}]),
        create_mock_response("Refactor and focused verification completed."),
    ], approvals)

    report = await agent.process_message("Refactor VALUE end to end")

    assert "Status: succeeded" in report
    assert "Plan progress: 2/2 completed, 0 failed" in report
    assert (temp_dir / "app.py").read_text() == "VALUE = 2\n"
    assert len(approvals) == 2
    assert approvals[0].parameters["approval_target"] == "app.py"
    assert "-VALUE = 1" in approvals[0].parameters["proposed_diff"]
    assert "+VALUE = 2" in approvals[0].parameters["proposed_diff"]
    assert approvals[1].parameters["approval_target"] == command
    assert command in format_approval_prompt(approvals[1])
    assert "Proposed diff:" in format_approval_prompt(approvals[0])
    assert agent.last_run and all(
        action.status == "completed" for action in agent.last_run.plan.actions
    )


@pytest.mark.e2e
@pytest.mark.agent
@pytest.mark.asyncio
async def test_complex_mutation_without_plan_is_denied(temp_dir, mock_llm_config):
    target = temp_dir / "app.py"
    target.write_text("VALUE = 1\n")
    approvals = []
    agent = make_agent(temp_dir, mock_llm_config, [
        create_mock_response(tool_calls=[{"name": "read_file", "parameters": {"path": "app.py"}}]),
        create_mock_response(tool_calls=[{"name": "edit_file", "parameters": {
            "path": "app.py", "old_text": "VALUE = 1", "new_text": "VALUE = 2",
        }}]),
        create_mock_response("Done."),
    ], approvals)

    report = await agent.process_message("Refactor this across the application")

    assert "Status: failed" in report
    assert "requires submit_plan" in report
    assert target.read_text() == "VALUE = 1\n"
    assert not approvals


@pytest.mark.e2e
@pytest.mark.agent
@pytest.mark.asyncio
async def test_plan_rejects_evidence_that_was_not_inspected(temp_dir, mock_llm_config):
    (temp_dir / "app.py").write_text("VALUE = 1\n")
    approvals = []
    agent = make_agent(temp_dir, mock_llm_config, [
        create_mock_response(tool_calls=[{"name": "submit_plan", "parameters": {
            "goal": "Refactor value",
            "reasoning": "Use the value definition.",
            "actions": [{
                "description": "Edit value", "tool": "edit_file",
                "parameters": {"path": "app.py", "old_text": "1", "new_text": "2"},
                "evidence": ["app.py:1-1"], "completion_criteria": "Value is updated",
            }],
        }}]),
        create_mock_response("Cannot proceed."),
    ], approvals)

    report = await agent.process_message("Refactor VALUE across the application")

    assert "Status: failed" in report
    assert "Plan evidence was not inspected" in report
    assert agent.last_run and agent.last_run.failure_kind == FailureKind.INCOMPLETE_WORKFLOW
