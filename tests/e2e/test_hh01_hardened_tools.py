"""HH-01 acceptance tests: hardened tool execution.

Runs the real agent loop, tool registry, and tools against a scripted
MockLLMProvider to prove: no environment secrets reach subprocesses or audit
records, unbounded output is truncated with an explicit marker, ambiguous edits
fail closed with count-bearing errors, and the approval preview diff is
byte-identical to the applied change for regex edits.
"""

from __future__ import annotations

import json

import pytest

from ctxai.agent.config import AgentConfig, AgentToolsConfig
from ctxai.agent.core import Agent, AgentLoopConfig
from ctxai.agent.tools.bash_tool import BashTool
from ctxai.agent.tools.execution import ToolExecutionContext
from ctxai.agent.tools.file_ops import EditFileTool, ReadFileTool, WriteFileTool
from ctxai.agent.tools.registry import ToolRegistry
from tests.mocks.mock_llm import MockLLMProvider, create_mock_response

SEEDED_API_KEY = "sk-fake-anthropic-key-hh01"
SEEDED_SECRET = "ctxai-hh01-secret-value"


def make_agent(
    temp_dir,
    mock_llm_config,
    responses,
    *,
    tools_config: AgentToolsConfig | None = None,
    approval=lambda call: True,
):
    """Build a real agent (loop + registry + tools) over a scripted mock provider.

    Args:
        temp_dir: Project root for the run.
        mock_llm_config: LLM configuration for the mock provider.
        responses: Scripted mock provider responses.
        tools_config: Optional tools configuration (defaults to ``AgentConfig().tools``).
        approval: Approval callback for mutation and verification tools.

    Returns:
        Tuple of the ``Agent`` and its shared ``ToolExecutionContext``.
    """
    tools_config = tools_config or AgentConfig().tools
    context = ToolExecutionContext.for_project(
        temp_dir,
        allow_outside_project=tools_config.allow_outside_project,
        timeout=tools_config.bash_timeout,
        env_passthrough=tools_config.env_passthrough,
    )
    registry = ToolRegistry()
    registry.register(ReadFileTool(context=context, max_output_chars=tools_config.max_output_chars))
    registry.register(WriteFileTool(context=context))
    registry.register(EditFileTool(context=context))
    registry.register(BashTool(tools_config, context=context))
    llm = MockLLMProvider(config=mock_llm_config, responses=responses)
    agent = Agent(
        AgentLoopConfig(
            llm_provider=llm,
            tool_registry=registry,
            agent_config=AgentConfig(tools=tools_config),
            working_directory=temp_dir,
            available_indexes=[],
            max_iterations=12,
            require_user_approval=True,
            approval_callback=approval,
        )
    )
    return agent, context


def transcript(agent) -> str:
    """Serialize every conversation message for assertions.

    Args:
        agent: The agent whose context is inspected.

    Returns:
        Newline-joined message contents.
    """
    return "\n".join(message.content for message in agent.context.messages)


@pytest.mark.e2e
@pytest.mark.agent
@pytest.mark.asyncio
async def test_subprocess_never_sees_seeded_environment_secrets(
    temp_dir, mock_llm_config, patch_embeddings_factory, monkeypatch
):
    monkeypatch.setenv("ANTHROPIC_API_KEY", SEEDED_API_KEY)
    monkeypatch.setenv("CTXAI_TEST_SECRET", SEEDED_SECRET)
    agent, context = make_agent(
        temp_dir,
        mock_llm_config,
        [
            create_mock_response(tool_calls=[{"name": "bash", "parameters": {"command": "env"}}]),
            create_mock_response("Environment inspection complete."),
        ],
    )

    await agent.process_message("Show the subprocess environment")

    seen = transcript(agent)
    assert "PATH=" in seen  # the allowlist still exposes what subprocesses need
    assert SEEDED_API_KEY not in seen
    assert SEEDED_SECRET not in seen
    assert "ANTHROPIC_API_KEY" not in seen
    dumped_audit = json.dumps([record.__dict__ for record in context.audit_log])
    assert SEEDED_API_KEY not in dumped_audit
    assert SEEDED_SECRET not in dumped_audit
    assert any(record.action == "command" and record.success for record in context.audit_log)


@pytest.mark.e2e
@pytest.mark.agent
@pytest.mark.asyncio
async def test_env_passthrough_opt_in_reaches_subprocess(
    temp_dir, mock_llm_config, patch_embeddings_factory, monkeypatch
):
    monkeypatch.setenv("CTXAI_OPTED_IN", "visible-value")
    agent, _ = make_agent(
        temp_dir,
        mock_llm_config,
        [
            create_mock_response(tool_calls=[{"name": "bash", "parameters": {"command": "env"}}]),
            create_mock_response("Done."),
        ],
        tools_config=AgentToolsConfig(env_passthrough=["CTXAI_OPTED_IN"]),
    )

    await agent.process_message("Check the opted-in variable")

    assert "CTXAI_OPTED_IN=visible-value" in transcript(agent)


@pytest.mark.e2e
@pytest.mark.agent
@pytest.mark.asyncio
async def test_huge_command_output_enters_context_truncated(temp_dir, mock_llm_config, patch_embeddings_factory):
    agent, context = make_agent(
        temp_dir,
        mock_llm_config,
        [
            create_mock_response(
                tool_calls=[{"name": "write_file", "parameters": {"path": "big.txt", "content": "x" * 40000}}]
            ),
            create_mock_response(tool_calls=[{"name": "bash", "parameters": {"command": "head -c 40000 big.txt"}}]),
            create_mock_response("Truncation verified."),
        ],
    )

    await agent.process_message("Echo a huge file into the context")

    assert "...[truncated 20000 of 40000 chars]" in transcript(agent)
    command_records = [record for record in context.audit_log if record.action == "command" and record.success]
    assert command_records, "expected a successful command audit record"
    assert command_records[-1].details["stdout_chars"] == 40000
    assert command_records[-1].details["truncated"] is True


@pytest.mark.e2e
@pytest.mark.agent
@pytest.mark.asyncio
async def test_huge_file_read_enters_context_truncated(temp_dir, mock_llm_config, patch_embeddings_factory):
    agent, _ = make_agent(
        temp_dir,
        mock_llm_config,
        [
            create_mock_response(
                tool_calls=[{"name": "write_file", "parameters": {"path": "wide.py", "content": "x" * 30000}}]
            ),
            create_mock_response(tool_calls=[{"name": "read_file", "parameters": {"path": "wide.py"}}]),
            create_mock_response("Read truncation verified."),
        ],
    )

    await agent.process_message("Read the wide file")

    seen = transcript(agent)
    # "   1 | " prefixes the single 30000-character line.
    assert "...[truncated 10007 of 30007 chars]" in seen
    assert "original_chars" in seen


@pytest.mark.e2e
@pytest.mark.agent
@pytest.mark.asyncio
async def test_multi_match_edit_fails_closed_with_count_bearing_error(
    temp_dir, mock_llm_config, patch_embeddings_factory
):
    target = temp_dir / "values.py"
    target.write_text("first = 1\nsecond = 1\n", encoding="utf-8")
    agent, context = make_agent(
        temp_dir,
        mock_llm_config,
        [
            create_mock_response(tool_calls=[{"name": "read_file", "parameters": {"path": "values.py"}}]),
            create_mock_response(
                tool_calls=[
                    {"name": "edit_file", "parameters": {"path": "values.py", "old_text": "1", "new_text": "2"}}
                ]
            ),
            create_mock_response("I could not apply an ambiguous edit."),
        ],
    )

    await agent.process_message("Change every 1 to 2")

    assert target.read_text(encoding="utf-8") == "first = 1\nsecond = 1\n"
    assert "Edit failed: pattern matched 2 occurrence(s)" in transcript(agent)
    failed_edits = [record for record in context.audit_log if record.action == "edit" and not record.success]
    assert failed_edits and "2" in failed_edits[-1].details["error"]
    assert not [record for record in context.audit_log if record.action == "edit" and record.success]
    assert agent.last_run is not None and not agent.last_run.mutated


@pytest.mark.e2e
@pytest.mark.agent
@pytest.mark.asyncio
async def test_regex_edit_approval_diff_is_byte_identical_to_applied_diff(
    temp_dir, mock_llm_config, patch_embeddings_factory
):
    (temp_dir / "app.py").write_text("v = 1\nw = v1 times\n", encoding="utf-8")
    approvals: list = []
    agent, context = make_agent(
        temp_dir,
        mock_llm_config,
        [
            create_mock_response(tool_calls=[{"name": "read_file", "parameters": {"path": "app.py"}}]),
            create_mock_response(
                tool_calls=[
                    {
                        "name": "edit_file",
                        "parameters": {
                            "path": "app.py",
                            "old_text": r"v\d",
                            "new_text": "vX",
                            "use_regex": True,
                        },
                    }
                ]
            ),
            create_mock_response("Regex edit applied."),
        ],
        approval=lambda call: approvals.append(call) or True,
    )

    await agent.process_message("Apply the regex edit")

    applied_diffs = [
        record.details.get("diff") for record in context.audit_log if record.action == "edit" and record.success
    ]
    assert applied_diffs, "expected a successful edit audit record with a diff"
    assert approvals, "expected the edit to be presented for approval"
    proposed_diff = approvals[0].parameters["proposed_diff"]
    assert proposed_diff == applied_diffs[-1]
    assert (temp_dir / "app.py").read_text(encoding="utf-8") == "v = 1\nw = vX times\n"
