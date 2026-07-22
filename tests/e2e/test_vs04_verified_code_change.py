"""VS-04 acceptance tests for evidence-backed one-shot agent workflows."""

from __future__ import annotations

import sys

import pytest

from ctxai.agent.config import AgentConfig
from ctxai.agent.core import Agent, AgentLoopConfig
from ctxai.agent.tools.bash_tool import BashTool
from ctxai.agent.tools.execution import ToolExecutionContext
from ctxai.agent.tools.file_ops import EditFileTool, ReadFileTool, WriteFileTool
from ctxai.agent.tools.registry import ToolRegistry
from ctxai.agent.workflow import FailureKind, TaskState
from tests.mocks.mock_llm import MockLLMProvider, create_mock_response


def make_agent(temp_dir, mock_llm_config, responses, *, approval=lambda call: True):
    context = ToolExecutionContext.for_project(temp_dir)
    registry = ToolRegistry()
    registry.register(ReadFileTool(context=context))
    registry.register(WriteFileTool(context=context))
    registry.register(EditFileTool(context=context))
    registry.register(BashTool(AgentConfig().tools, context=context))
    llm = MockLLMProvider(config=mock_llm_config, responses=responses)
    agent = Agent(
        AgentLoopConfig(
            llm_provider=llm,
            tool_registry=registry,
            agent_config=AgentConfig(),
            working_directory=temp_dir,
            available_indexes=[],
            max_iterations=12,
            require_user_approval=True,
            approval_callback=approval,
        )
    )
    return agent, context


@pytest.mark.e2e
@pytest.mark.agent
@pytest.mark.asyncio
async def test_read_only_answer_has_no_mutation(temp_dir, mock_llm_config):
    (temp_dir / "app.py").write_text("VALUE = 1\n")
    agent, context = make_agent(
        temp_dir,
        mock_llm_config,
        [
            create_mock_response(tool_calls=[{"name": "read_file", "parameters": {"path": "app.py"}}]),
            create_mock_response("VALUE is 1 (app.py:1-1)."),
        ],
    )

    report = await agent.process_message("What is VALUE?")

    assert "Status: succeeded" in report
    assert "Changed files: None" in report
    assert not context.audit_log


@pytest.mark.e2e
@pytest.mark.agent
@pytest.mark.asyncio
async def test_one_file_edit_requires_inspection_and_reports_verification(temp_dir, mock_llm_config):
    (temp_dir / "app.py").write_text("VALUE = 1\n")
    command = f"{sys.executable} -m py_compile app.py"
    agent, context = make_agent(
        temp_dir,
        mock_llm_config,
        [
            create_mock_response(tool_calls=[{"name": "read_file", "parameters": {"path": "app.py"}}]),
            create_mock_response(tool_calls=[{"name": "edit_file", "parameters": {
                "path": "app.py", "old_text": "VALUE = 1", "new_text": "VALUE = 2"
            }}]),
            create_mock_response(tool_calls=[{"name": "bash", "parameters": {"command": command}}]),
            create_mock_response("Updated VALUE and compiled the module."),
        ],
    )

    report = await agent.process_message("Change VALUE to 2")

    assert (temp_dir / "app.py").read_text() == "VALUE = 2\n"
    assert "Status: succeeded" in report
    assert "app.py" in report and command in report
    assert agent.last_run and agent.last_run.diff_reviewed
    assert TaskState.APPROVE in agent.last_run.transitions
    assert any(record.action == "edit" for record in context.audit_log)


@pytest.mark.e2e
@pytest.mark.agent
@pytest.mark.asyncio
async def test_multi_file_edit_reports_actual_files(temp_dir, mock_llm_config):
    command = f"{sys.executable} -m py_compile one.py two.py"
    agent, _ = make_agent(
        temp_dir,
        mock_llm_config,
        [
            create_mock_response(tool_calls=[{"name": "write_file", "parameters": {
                "path": "one.py", "content": "ONE = 1\n"
            }}]),
            create_mock_response(tool_calls=[{"name": "write_file", "parameters": {
                "path": "two.py", "content": "TWO = 2\n"
            }}]),
            create_mock_response(tool_calls=[{"name": "bash", "parameters": {"command": command}}]),
            create_mock_response("Created both modules."),
        ],
    )

    report = await agent.process_message("Create two modules")

    assert "Status: succeeded" in report
    assert "one.py" in report and "two.py" in report
    assert agent.last_run and len(agent.last_run.changed_files) == 2


@pytest.mark.e2e
@pytest.mark.agent
@pytest.mark.asyncio
async def test_failed_required_check_cannot_report_success(temp_dir, mock_llm_config):
    command = f"{sys.executable} -m py_compile broken.py"
    agent, _ = make_agent(
        temp_dir,
        mock_llm_config,
        [
            create_mock_response(tool_calls=[{"name": "write_file", "parameters": {
                "path": "broken.py", "content": "def broken(:\n"
            }}]),
            create_mock_response(tool_calls=[{"name": "bash", "parameters": {"command": command}}]),
            create_mock_response("Everything passed successfully."),
        ],
    )

    report = await agent.process_message("Create broken.py")

    assert "Status: failed" in report
    assert f"{command} (failed)" in report
    assert "Everything passed successfully" in report
    assert agent.last_run and agent.last_run.failure_kind == FailureKind.TEST_FAILURE


@pytest.mark.e2e
@pytest.mark.agent
@pytest.mark.asyncio
async def test_approval_denial_prevents_mutation(temp_dir, mock_llm_config):
    target = temp_dir / "app.py"
    target.write_text("VALUE = 1\n")
    agent, context = make_agent(
        temp_dir,
        mock_llm_config,
        [
            create_mock_response(tool_calls=[{"name": "read_file", "parameters": {"path": "app.py"}}]),
            create_mock_response(tool_calls=[{"name": "edit_file", "parameters": {
                "path": "app.py", "old_text": "1", "new_text": "2"
            }}]),
            create_mock_response("The change is complete."),
        ],
        approval=lambda call: False,
    )

    report = await agent.process_message("Change VALUE")

    assert target.read_text() == "VALUE = 1\n"
    assert "Status: failed" in report
    assert "Approval denied" in report
    assert not context.audit_log
    assert agent.last_run and agent.last_run.failure_kind == FailureKind.APPROVAL_DENIAL
