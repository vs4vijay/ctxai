"""Unit tests for HH-08 OS-sandboxed command execution.

Covers the sandbox backend contract with a fake backend (wrap composition,
mode matrix, fail-closed semantics), configuration round-trips, audit
integration, and the real seatbelt backend's profile/cleanup behavior on
macOS hosts that provide ``sandbox-exec``.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path
from typing import Any

import pytest

from ctxai.agent.config import AgentToolsConfig
from ctxai.agent.tools.bash_tool import BashTool
from ctxai.agent.tools.execution import Capability, ToolExecutionContext
from ctxai.agent.tools.sandbox import (
    SANDBOX_MODES,
    BubblewrapBackend,
    MacOSSeatbeltBackend,
    NoopBackend,
    SandboxError,
    SandboxUnavailableError,
    default_backends,
    describe_sandbox,
    select_backend,
)


class FakeBackend:
    """Scripted SandboxBackend for unit tests.

    Records wrap calls and can be made unavailable or fail wrap() to prove
    fail-closed semantics. ``wrap`` composes ``/usr/bin/env`` in front of the
    argv so the wrapped command actually executes when a test runs it.
    """

    name = "fake"
    enforces = True

    def __init__(self, *, available: bool = True, fail_wrap: bool = False):
        self.available = available
        self.fail_wrap = fail_wrap
        self.wrapped: list[tuple[list[str], Path, bool]] = []
        self.cleanup_count = 0
        self.env_adjusted = False

    def is_available(self) -> bool:
        """Report scripted availability."""
        return self.available

    def wrap(self, argv: list[str], cwd: Path, *, network: bool) -> list[str]:
        """Record the wrap request and return an executable composed argv."""
        if self.fail_wrap:
            raise SandboxError("simulated wrap failure")
        self.wrapped.append((list(argv), Path(cwd), network))
        return ["/usr/bin/env", *argv]

    def adjust_environment(self, env: dict[str, str]) -> dict[str, str]:
        """Flag that environment adjustment ran; add a marker variable."""
        self.env_adjusted = True
        return {**env, "FAKE_SANDBOX_ENV": "1"}

    def cleanup(self) -> None:
        """Count cleanup invocations."""
        self.cleanup_count += 1


def make_tool(
    tmp_path: Path,
    tools_config: AgentToolsConfig | None = None,
    *,
    sandbox_backends: list[Any] | None = None,
    capabilities: set[Capability] | None = None,
) -> BashTool:
    """Build a BashTool over a temp project with an optional injected backend.

    Args:
        tmp_path: Project root for the execution context.
        tools_config: Tools configuration (defaults to a fresh AgentToolsConfig).
        sandbox_backends: Optional backend list injected into the tool.
        capabilities: Optional explicit capability set for the context.

    Returns:
        A BashTool bound to the temp project.
    """
    context = ToolExecutionContext.for_project(tmp_path)
    if capabilities is not None:
        context.capabilities = set(capabilities)
    config = tools_config or AgentToolsConfig()
    kwargs: dict[str, Any] = {}
    if sandbox_backends is not None:
        kwargs["sandbox_backends"] = sandbox_backends
    return BashTool(config, context=context, **kwargs)


# ============================================================================
# Configuration
# ============================================================================


def test_sandbox_config_defaults_preserve_current_behavior():
    config = AgentToolsConfig()
    assert config.sandbox == "off"
    assert config.sandbox_network is False
    assert set(SANDBOX_MODES) == {"off", "auto", "required"}


def test_sandbox_config_round_trips_new_fields():
    config = AgentToolsConfig(sandbox="required", sandbox_network=True)
    data = config.to_dict()
    assert data["sandbox"] == "required"
    assert data["sandbox_network"] is True
    restored = AgentToolsConfig.from_dict(data)
    assert restored.sandbox == "required"
    assert restored.sandbox_network is True


def test_sandbox_config_from_dict_defaults_when_absent():
    config = AgentToolsConfig.from_dict({})
    assert config.sandbox == "off"
    assert config.sandbox_network is False


def test_sandbox_config_rejects_unknown_mode():
    with pytest.raises(ValueError, match="sandbox"):
        AgentToolsConfig(sandbox="banana")


# ============================================================================
# Backend selection and descriptions
# ============================================================================


def test_noop_backend_is_a_passthrough(tmp_path):
    backend = NoopBackend()
    argv = ["echo", "hi"]
    assert backend.enforces is False
    assert backend.is_available() is True
    assert backend.wrap(argv, tmp_path, network=False) == argv
    env = {"PATH": "/usr/bin"}
    assert backend.adjust_environment(env) == {"PATH": "/usr/bin"}
    backend.cleanup()


def test_select_backend_off_always_returns_noop():
    backend, diagnostic = select_backend("off", backends=[FakeBackend()])
    assert isinstance(backend, NoopBackend)
    assert diagnostic is None


def test_select_backend_prefers_first_available_backend():
    unavailable = FakeBackend(available=False)
    available = FakeBackend()
    backend, diagnostic = select_backend("auto", backends=[unavailable, available])
    assert backend is available
    assert diagnostic is None


def test_select_backend_auto_without_backend_returns_noop_with_diagnostic():
    backend, diagnostic = select_backend("auto", backends=[FakeBackend(available=False)])
    assert isinstance(backend, NoopBackend)
    assert diagnostic is not None
    assert "no sandbox backend" in diagnostic


def test_select_backend_required_without_backend_reports_diagnostic():
    backend, diagnostic = select_backend("required", backends=[FakeBackend(available=False)])
    assert backend.enforces is False
    assert diagnostic is not None


def test_default_backends_include_seatbelt_and_bubblewrap():
    names = [type(backend).__name__ for backend in default_backends()]
    assert names == ["MacOSSeatbeltBackend", "BubblewrapBackend"]


def test_describe_sandbox_off_is_none():
    assert describe_sandbox("off", False, backends=[FakeBackend()]) is None


def test_describe_sandbox_with_backend_names_network_state():
    assert describe_sandbox("auto", False, backends=[FakeBackend()]) == "sandbox: fake (network denied)"
    assert describe_sandbox("auto", True, backends=[FakeBackend()]) == "sandbox: fake (network allowed)"


def test_describe_sandbox_unavailable_auto_diagnoses_fallback():
    text = describe_sandbox("auto", False, backends=[FakeBackend(available=False)])
    assert "sandbox unavailable" in text


def test_describe_sandbox_unavailable_required_warns_commands_fail():
    text = describe_sandbox("required", False, backends=[FakeBackend(available=False)])
    assert "sandbox unavailable" in text
    assert "required" in text


# ============================================================================
# BashTool mode matrix (fake backend)
# ============================================================================


async def test_off_mode_runs_unsandboxed_and_ignores_injected_backend(tmp_path):
    backend = FakeBackend()
    tool = make_tool(tmp_path, AgentToolsConfig(sandbox="off"), sandbox_backends=[backend])
    result = await tool.execute("echo off-mode")
    assert result["success"] is True
    assert result["result"] == "off-mode\n"
    assert backend.wrapped == []
    assert backend.cleanup_count == 0
    assert result["metadata"]["sandbox"] is None
    assert tool.context.audit_log[-1].details["sandbox"] is None


async def test_auto_mode_wraps_command_with_available_backend(tmp_path):
    backend = FakeBackend()
    tool = make_tool(tmp_path, AgentToolsConfig(sandbox="auto"), sandbox_backends=[backend])
    result = await tool.execute("echo wrapped-run")
    assert result["success"] is True
    assert result["result"] == "wrapped-run\n"
    assert backend.wrapped, "expected the backend to be asked to wrap argv"
    argv, cwd, network = backend.wrapped[0]
    assert argv == ["echo", "wrapped-run"]
    assert cwd == tmp_path.resolve()
    assert network is False
    assert backend.env_adjusted is True
    assert backend.cleanup_count >= 1
    assert result["metadata"]["sandbox"] == "fake"
    assert tool.context.audit_log[-1].details["sandbox"] == "fake"


async def test_auto_mode_without_backend_runs_unsandboxed_with_diagnostic(tmp_path):
    backend = FakeBackend(available=False)
    tool = make_tool(tmp_path, AgentToolsConfig(sandbox="auto"), sandbox_backends=[backend])
    result = await tool.execute("echo fallback-run")
    assert result["success"] is True
    assert result["result"] == "fallback-run\n"
    assert backend.wrapped == []
    record = tool.context.audit_log[-1]
    assert record.details["sandbox"] is None
    assert "no sandbox backend" in record.details["sandbox_diagnostic"]
    assert result["metadata"]["sandbox"] is None


async def test_required_mode_without_backend_fails_closed_and_executes_nothing(tmp_path):
    backend = FakeBackend(available=False)
    tool = make_tool(tmp_path, AgentToolsConfig(sandbox="required"), sandbox_backends=[backend])
    result = await tool.execute("echo must-not-run")
    assert result["success"] is False
    assert result["result"] is None
    assert "required" in result["error"]
    assert backend.wrapped == []
    record = tool.context.audit_log[-1]
    assert record.success is False
    assert "no sandbox backend" in record.details["error"]


async def test_required_mode_with_backend_wraps_command(tmp_path):
    backend = FakeBackend()
    tool = make_tool(tmp_path, AgentToolsConfig(sandbox="required"), sandbox_backends=[backend])
    result = await tool.execute("echo required-run")
    assert result["success"] is True
    assert result["result"] == "required-run\n"
    assert backend.wrapped[0][0] == ["echo", "required-run"]
    assert tool.context.audit_log[-1].details["sandbox"] == "fake"


async def test_wrap_failure_fails_closed_in_required_mode(tmp_path):
    backend = FakeBackend(fail_wrap=True)
    tool = make_tool(tmp_path, AgentToolsConfig(sandbox="required"), sandbox_backends=[backend])
    result = await tool.execute("echo must-not-run")
    assert result["success"] is False
    assert "simulated wrap failure" in result["error"]
    record = tool.context.audit_log[-1]
    assert record.success is False
    assert "simulated wrap failure" in record.details["error"]


async def test_wrap_failure_fails_closed_even_in_auto_mode(tmp_path):
    """A failing wrap never falls back to an unsandboxed run (fail closed)."""
    backend = FakeBackend(fail_wrap=True)
    tool = make_tool(tmp_path, AgentToolsConfig(sandbox="auto"), sandbox_backends=[backend])
    result = await tool.execute("echo must-not-run")
    assert result["success"] is False
    assert "simulated wrap failure" in result["error"]


# ============================================================================
# Policy composition: classification and allowlist run before wrap
# ============================================================================


async def test_classification_denies_shell_operators_before_wrap(tmp_path):
    backend = FakeBackend()
    tool = make_tool(tmp_path, AgentToolsConfig(sandbox="required"), sandbox_backends=[backend])
    result = await tool.execute("echo hi && echo there")
    assert result["success"] is False
    assert "Shell operators" in result["error"]
    assert backend.wrapped == []


async def test_allowlist_denies_unlisted_executable_before_wrap(tmp_path):
    backend = FakeBackend()
    config = AgentToolsConfig(sandbox="required", bash_allowed_commands=["git"])
    tool = make_tool(tmp_path, config, sandbox_backends=[backend])
    result = await tool.execute("echo not-allowed")
    assert result["success"] is False
    assert "allowlisted" in result["error"]
    assert backend.wrapped == []


async def test_output_caps_apply_under_wrap(tmp_path):
    backend = FakeBackend()
    config = AgentToolsConfig(sandbox="required", max_output_chars=50)
    tool = make_tool(tmp_path, config, sandbox_backends=[backend])
    result = await tool.execute("echo " + "x" * 200)
    assert result["success"] is True
    assert "...[truncated" in result["result"]
    assert result["metadata"]["sandbox"] == "fake"


async def test_timeout_still_enforced_under_wrap(tmp_path):
    backend = FakeBackend()
    config = AgentToolsConfig(sandbox="required", bash_timeout=1)
    tool = make_tool(tmp_path, config, sandbox_backends=[backend])
    result = await tool.execute("sleep 5")
    assert result["success"] is False
    assert "timed out" in result["error"]


# ============================================================================
# Network policy composition
# ============================================================================


async def test_network_denied_by_default_passes_network_false_to_backend(tmp_path):
    backend = FakeBackend()
    tool = make_tool(tmp_path, AgentToolsConfig(sandbox="required"), sandbox_backends=[backend])
    await tool.execute("echo hi")
    assert backend.wrapped[0][2] is False


async def test_sandbox_network_true_satisfies_network_capability(tmp_path):
    """sandbox_network=True lets classified network commands through and wraps with network allowed."""
    backend = FakeBackend()
    config = AgentToolsConfig(sandbox="required", sandbox_network=True)
    tool = make_tool(tmp_path, config, sandbox_backends=[backend])
    result = await tool.execute("curl http://example.invalid")
    assert result["success"] is False  # curl ran and failed to reach the invalid host
    assert backend.wrapped[0][2] is True
    assert Capability.NETWORK in tool.context.capabilities


async def test_granted_network_capability_wraps_with_network_allowed(tmp_path):
    """Vice versa: a context granted Capability.NETWORK runs the sandbox with network allowed."""
    backend = FakeBackend()
    capabilities = {Capability.READ, Capability.WORKSPACE_WRITE, Capability.COMMAND, Capability.NETWORK}
    tool = make_tool(
        tmp_path,
        AgentToolsConfig(sandbox="required"),
        sandbox_backends=[backend],
        capabilities=capabilities,
    )
    result = await tool.execute("curl http://example.invalid")
    assert result["success"] is False
    assert backend.wrapped[0][2] is True


async def test_sandbox_network_true_without_backend_does_not_grant_network(tmp_path):
    """Without an enforcing backend, network commands still require the capability."""
    backend = FakeBackend(available=False)
    config = AgentToolsConfig(sandbox="auto", sandbox_network=True)
    tool = make_tool(tmp_path, config, sandbox_backends=[backend])
    result = await tool.execute("curl http://example.invalid")
    assert result["success"] is False
    assert Capability.NETWORK not in tool.context.capabilities


async def test_approval_granted_network_wraps_with_network_allowed_for_that_command(tmp_path):
    """An approval that satisfies the NETWORK check allows network in the sandbox — once."""
    backend = FakeBackend()
    tool = make_tool(
        tmp_path,
        AgentToolsConfig(sandbox="required"),
        sandbox_backends=[backend],
        capabilities={Capability.READ, Capability.WORKSPACE_WRITE, Capability.COMMAND},
    )
    tool.context.approval_callback = lambda capability, action, target: capability is Capability.NETWORK

    approved = await tool.execute("curl http://example.invalid")
    assert approved["success"] is False  # curl ran and failed to reach the invalid host
    assert backend.wrapped[0][2] is True
    assert Capability.NETWORK in tool.context.approved_capabilities

    # The grant is per-command: the next command without a network
    # requirement wraps with network denied again.
    plain = await tool.execute("echo next")
    assert plain["success"] is True
    assert backend.wrapped[1][2] is False


async def test_denied_network_approval_never_wraps(tmp_path):
    backend = FakeBackend()
    tool = make_tool(tmp_path, AgentToolsConfig(sandbox="required"), sandbox_backends=[backend])
    tool.context.approval_callback = lambda capability, action, target: False
    result = await tool.execute("curl http://example.invalid")
    assert result["success"] is False
    assert "Capability denied" in result["error"]
    assert backend.wrapped == []


# ============================================================================
# Real seatbelt backend (macOS hosts with sandbox-exec)
# ============================================================================

HAS_SEATBELT = sys.platform == "darwin" and shutil.which("sandbox-exec") is not None

requires_seatbelt = pytest.mark.skipif(not HAS_SEATBELT, reason="macOS seatbelt backend (sandbox-exec) not available")


@requires_seatbelt
def test_seatbelt_is_available_on_macos_hosts():
    assert MacOSSeatbeltBackend().is_available() is True


@requires_seatbelt
def test_seatbelt_profile_denies_network_by_default_and_cleans_up(tmp_path):
    backend = MacOSSeatbeltBackend()
    argv = backend.wrap(["echo", "hi"], tmp_path, network=False)
    assert argv[0].endswith("sandbox-exec")
    assert argv[1] == "-f"
    profile_path = Path(argv[2])
    assert argv[3] == "--"
    assert argv[4:] == ["echo", "hi"]
    try:
        profile_text = profile_path.read_text(encoding="utf-8")
        assert "(deny network*)" in profile_text
        assert "(deny file-write*)" in profile_text
        assert str(tmp_path.resolve()) in profile_text
    finally:
        backend.cleanup()
    assert not profile_path.exists()
    assert backend.pending_profiles() == []


@requires_seatbelt
def test_seatbelt_profile_allows_network_when_requested(tmp_path):
    backend = MacOSSeatbeltBackend()
    argv = backend.wrap(["echo", "hi"], tmp_path, network=True)
    profile_path = Path(argv[2])
    try:
        profile_text = profile_path.read_text(encoding="utf-8")
        assert "(deny network*)" not in profile_text
        assert "(deny file-write*)" in profile_text
    finally:
        backend.cleanup()
    assert not profile_path.exists()


@requires_seatbelt
def test_seatbelt_profile_escapes_quotes_in_paths(tmp_path):
    weird = tmp_path / 'we ird"dir'
    weird.mkdir()
    backend = MacOSSeatbeltBackend()
    argv = backend.wrap(["echo", "hi"], weird, network=False)
    profile_path = Path(argv[2])
    try:
        assert 'we ird\\"dir' in profile_path.read_text(encoding="utf-8")
    finally:
        backend.cleanup()


@requires_seatbelt
async def test_seatbelt_wrapped_command_runs_and_cleans_up(tmp_path):
    backend = MacOSSeatbeltBackend()
    tool = make_tool(tmp_path, AgentToolsConfig(sandbox="required"), sandbox_backends=[backend])
    result = await tool.execute("echo seatbelt-run")
    assert result["success"] is True
    assert result["result"] == "seatbelt-run\n"
    assert result["metadata"]["sandbox"] == "seatbelt"
    assert backend.pending_profiles() == []
    record = tool.context.audit_log[-1]
    assert record.details["sandbox"] == "seatbelt"
    assert record.details["sandbox_network"] is False


@requires_seatbelt
async def test_seatbelt_denies_outbound_connection(tmp_path):
    backend = MacOSSeatbeltBackend()
    tool = make_tool(tmp_path, AgentToolsConfig(sandbox="required"), sandbox_backends=[backend])
    probe = tmp_path / "net_probe.py"
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
    result = await tool.execute(f"python3 {probe.name}")
    assert result["success"] is False
    assert "Operation not permitted" in (result["error"] or "")


@requires_seatbelt
async def test_seatbelt_allows_plain_command_with_identical_stdout(tmp_path):
    sandboxed = make_tool(tmp_path, AgentToolsConfig(sandbox="required"), sandbox_backends=[MacOSSeatbeltBackend()])
    plain = make_tool(tmp_path, AgentToolsConfig(sandbox="off"))
    probe = tmp_path / "hello.py"
    probe.write_text("print('ok')\n", encoding="utf-8")
    wrapped = await sandboxed.execute(f"python3 {probe.name}")
    unwrapped = await plain.execute(f"python3 {probe.name}")
    assert wrapped["success"] is True and unwrapped["success"] is True
    assert wrapped["result"] == unwrapped["result"] == "ok\n"


def test_bubblewrap_backend_availability_follows_bwrap_binary(monkeypatch):
    backend = BubblewrapBackend()
    monkeypatch.setattr("ctxai.agent.tools.sandbox.shutil.which", lambda name: None)
    assert backend.is_available() is False


def test_bubblewrap_wrap_composition(tmp_path):
    backend = BubblewrapBackend(executable="/usr/bin/bwrap")
    assert backend.is_available() is True
    argv = backend.wrap(["echo", "hi"], tmp_path, network=False)
    assert argv[0] == "/usr/bin/bwrap"
    assert "--unshare-net" in argv
    assert "--ro-bind" in argv
    assert "--" in argv
    assert argv[-2:] == ["echo", "hi"]


def test_bubblewrap_wrap_keeps_network_when_allowed(tmp_path):
    backend = BubblewrapBackend(executable="/usr/bin/bwrap")
    argv = backend.wrap(["echo", "hi"], tmp_path, network=True)
    assert "--unshare-net" not in argv


def test_bubblewrap_adjusts_tmpdir_for_tmpfs():
    backend = BubblewrapBackend(executable="/usr/bin/bwrap")
    env = backend.adjust_environment({"PATH": "/usr/bin", "TMPDIR": "/var/folders/x/T"})
    assert env["TMPDIR"] == "/tmp"


def test_bubblewrap_without_executable_fails_closed(tmp_path, monkeypatch):
    backend = BubblewrapBackend()
    monkeypatch.setattr("ctxai.agent.tools.sandbox.shutil.which", lambda name: None)
    assert backend.is_available() is False
    with pytest.raises(SandboxError):
        backend.wrap(["echo", "hi"], tmp_path, network=False)


def test_sandbox_unavailable_error_is_sandbox_error():
    assert issubclass(SandboxUnavailableError, SandboxError)
