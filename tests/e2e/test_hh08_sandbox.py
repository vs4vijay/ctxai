"""HH-08 acceptance tests: OS-sandboxed command execution.

Runs the real agent loop, tool registry, and BashTool against a scripted
MockLLMProvider under an enforcing OS sandbox backend (macOS seatbelt when
``sandbox-exec`` exists, Linux bubblewrap when ``bwrap`` exists). Proves: a
plain build command succeeds with byte-identical stdout capture under wrap,
a network-touching command fails under deny-network, and sandbox wrapping is
recorded in the audit trail. Tests skip gracefully on hosts without a
sandbox backend (``pytest.importorskip``-style guard).
"""

from __future__ import annotations

import pytest

from ctxai.agent.config import AgentConfig, AgentToolsConfig
from ctxai.agent.core import Agent, AgentLoopConfig
from ctxai.agent.tools.bash_tool import BashTool
from ctxai.agent.tools.execution import ToolExecutionContext
from ctxai.agent.tools.file_ops import EditFileTool, ReadFileTool, WriteFileTool
from ctxai.agent.tools.registry import ToolRegistry
from ctxai.agent.tools.sandbox import select_backend
from tests.mocks.mock_llm import MockLLMProvider, create_mock_response

_SELECTED_BACKEND, _BACKEND_DIAGNOSTIC = select_backend("required")
HAS_BACKEND = _SELECTED_BACKEND.enforces

requires_backend = pytest.mark.skipif(
    not HAS_BACKEND, reason=f"no OS sandbox backend on this host: {_BACKEND_DIAGNOSTIC}"
)


def backend_name() -> str:
    """Return the enforcing backend name for the current host.

    Returns:
        The sandbox backend name (for example ``seatbelt`` or ``bwrap``).
    """
    backend, _ = select_backend("required")
    assert backend.enforces, "test requires an enforcing backend (guarded by skipif)"
    return backend.name


def make_agent(
    temp_dir,
    mock_llm_config,
    responses,
    *,
    tools_config: AgentToolsConfig,
    approval=lambda call: True,
):
    """Build a real agent (loop + registry + tools) over a scripted mock provider.

    Args:
        temp_dir: Project root for the run.
        mock_llm_config: LLM configuration for the mock provider.
        responses: Scripted mock provider responses.
        tools_config: Tools configuration carrying the sandbox settings.
        approval: Approval callback for mutation and verification tools.

    Returns:
        Tuple of the ``Agent`` and its shared ``ToolExecutionContext``.
    """
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
@requires_backend
async def test_agent_loop_runs_build_command_under_sandbox(temp_dir, mock_llm_config, patch_embeddings_factory):
    (temp_dir / "hello.py").write_text("print('ok')\n", encoding="utf-8")
    agent, context = make_agent(
        temp_dir,
        mock_llm_config,
        [
            create_mock_response(
                tool_calls=[{"name": "bash", "parameters": {"command": "python3 -m py_compile hello.py"}}]
            ),
            create_mock_response(tool_calls=[{"name": "bash", "parameters": {"command": "python3 hello.py"}}]),
            create_mock_response("Build and run verified under the sandbox."),
        ],
        tools_config=AgentToolsConfig(sandbox="required"),
    )

    await agent.process_message("Compile and run hello.py")

    assert "ok" in transcript(agent)
    command_records = [record for record in context.audit_log if record.action == "command" and record.success]
    assert len(command_records) == 2
    for record in command_records:
        assert record.details["sandbox"] == backend_name()
        assert record.details["sandbox_network"] is False
    assert (temp_dir / "__pycache__").exists(), "compile output should land inside the writable project root"


@pytest.mark.e2e
@pytest.mark.agent
@requires_backend
async def test_network_touching_command_fails_under_deny_network(temp_dir, mock_llm_config, patch_embeddings_factory):
    # The probe binds an ephemeral port, closes it, then tries to reconnect:
    # unsandboxed this fails fast with ConnectionRefused; under a
    # deny-network sandbox the socket operation is refused by the OS.
    probe = temp_dir / "net_probe.py"
    probe.write_text(
        "import socket\n"
        "s = socket.socket()\n"
        "s.bind(('127.0.0.1', 0))\n"
        "port = s.getsockname()[1]\n"
        "s.close()\n"
        "s2 = socket.socket()\n"
        "s2.connect(('127.0.0.1', port))\n"
        "print('connected')\n",
        encoding="utf-8",
    )
    agent, context = make_agent(
        temp_dir,
        mock_llm_config,
        [
            create_mock_response(tool_calls=[{"name": "bash", "parameters": {"command": "python3 net_probe.py"}}]),
            create_mock_response("The network probe was blocked, as expected."),
        ],
        tools_config=AgentToolsConfig(sandbox="required"),
    )

    await agent.process_message("Try to open a local network connection")

    assert "connected" not in transcript(agent)
    failed_commands = [record for record in context.audit_log if record.action == "command" and not record.success]
    assert failed_commands, "expected the network probe to fail"
    assert "Operation not permitted" in transcript(agent), "seatbelt must deny the socket by OS policy"


@pytest.mark.e2e
@pytest.mark.agent
@requires_backend
async def test_sandboxed_plain_command_captures_stdout_identically_to_off_mode(
    temp_dir, mock_llm_config, patch_embeddings_factory
):
    (temp_dir / "hello.py").write_text("print('ok')\n", encoding="utf-8")
    config = ToolExecutionContext.for_project(temp_dir)
    sandboxed = BashTool(AgentToolsConfig(sandbox="required"), context=config)
    unsandboxed = BashTool(AgentToolsConfig(sandbox="off"), context=config)

    wrapped = await sandboxed.execute("python3 hello.py")
    plain = await unsandboxed.execute("python3 hello.py")

    assert wrapped["success"] is True and plain["success"] is True
    assert wrapped["result"] == plain["result"] == "ok\n"
    assert wrapped["metadata"]["exit_code"] == plain["metadata"]["exit_code"] == 0
    assert wrapped["metadata"]["sandbox"] == backend_name()
    assert plain["metadata"]["sandbox"] is None


@pytest.mark.e2e
@pytest.mark.agent
@requires_backend
async def test_sandboxed_compile_command_matches_unsandboxed_behavior(
    temp_dir, mock_llm_config, patch_embeddings_factory
):
    (temp_dir / "hello.py").write_text("print('ok')\n", encoding="utf-8")
    config = ToolExecutionContext.for_project(temp_dir)
    sandboxed = BashTool(AgentToolsConfig(sandbox="required"), context=config)
    unsandboxed = BashTool(AgentToolsConfig(sandbox="off"), context=config)

    wrapped = await sandboxed.execute("python3 -m py_compile hello.py")
    plain = await unsandboxed.execute("python3 -m py_compile hello.py")

    assert wrapped["success"] is True and plain["success"] is True
    assert wrapped["result"] == plain["result"]
    assert wrapped["metadata"]["exit_code"] == plain["metadata"]["exit_code"] == 0
