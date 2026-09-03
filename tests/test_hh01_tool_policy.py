"""Unit tests for the consolidated command policy, environment allowlist, and output caps (HH-01)."""

import asyncio
import io
import json
import os

import pytest
from rich.console import Console

from ctxai.agent.config import AgentConfig, AgentToolsConfig
from ctxai.agent.core import Agent, AgentLoopConfig
from ctxai.agent.tools.bash_tool import BashTool
from ctxai.agent.tools.execution import ALLOWED_ENVIRONMENT_KEYS, ToolExecutionContext
from ctxai.agent.tools.file_ops import ReadFileTool
from ctxai.agent.tools.registry import ToolRegistry
from tests.mocks.mock_llm import MockLLMProvider


class _FakeProcess:
    def __init__(self, stdout: bytes = b"", stderr: bytes = b"", returncode: int = 0):
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode

    async def communicate(self):
        return (self.stdout, self.stderr)

    def kill(self):
        pass


# ---------------------------------------------------------------------------
# Consolidated command policy
# ---------------------------------------------------------------------------


def test_substring_command_matcher_is_removed():
    assert not hasattr(AgentToolsConfig, "is_bash_command_allowed")


def test_substring_blocklist_config_is_removed():
    config = AgentToolsConfig()
    assert not hasattr(config, "bash_blocked_commands")


def test_agent_tools_config_round_trips_new_fields():
    config = AgentToolsConfig(env_passthrough=["A", "B"], max_output_chars=500)
    restored = AgentToolsConfig.from_dict(config.to_dict())
    assert restored == config


def test_agent_tools_config_defaults_preserve_current_behavior():
    config = AgentToolsConfig()
    assert config.env_passthrough == []
    assert config.max_output_chars == 20_000


def test_agent_tools_config_from_dict_ignores_removed_keys():
    restored = AgentToolsConfig.from_dict({"bash_blocked_commands": ["rm -rf /"], "max_output_chars": 100})
    assert restored.max_output_chars == 100
    assert not hasattr(restored, "bash_blocked_commands")


@pytest.mark.asyncio
async def test_bash_exact_name_allowlist_remains_the_policy(tmp_path):
    config = AgentToolsConfig(bash_allowed_commands=["echo"])
    context = ToolExecutionContext.for_project(tmp_path)
    bash = BashTool(config, context=context)
    allowed = await bash.execute("echo hello")
    denied = await bash.execute("ls .")
    assert allowed["success"]
    assert not denied["success"]
    assert "allowlist" in str(denied["error"])


# ---------------------------------------------------------------------------
# Environment allowlist
# ---------------------------------------------------------------------------


def _strip_non_allowlisted_env(monkeypatch):
    for name in list(os.environ):
        if name not in ALLOWED_ENVIRONMENT_KEYS:
            monkeypatch.delenv(name, raising=False)


def test_command_environment_returns_only_allowlisted_variables(monkeypatch, tmp_path):
    for name in list(os.environ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-secret")
    monkeypatch.setenv("CTXAI_TEST_SECRET", "leak-me")
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    monkeypatch.setenv("HOME", "/tmp/fake-home")
    monkeypatch.setenv("LANG", "C.UTF-8")

    env = ToolExecutionContext.for_project(tmp_path).command_environment()

    assert env == {"PATH": "/usr/bin:/bin", "HOME": "/tmp/fake-home", "LANG": "C.UTF-8"}


def test_command_environment_covers_the_documented_allowlist(monkeypatch, tmp_path):
    _strip_non_allowlisted_env(monkeypatch)
    seeded = {
        "PATH": "/usr/bin",
        "HOME": "/tmp/h",
        "LANG": "C",
        "LC_ALL": "C",
        "TMPDIR": "/tmp",
        "SHELL": "/bin/sh",
        "TERM": "dumb",
        "USER": "agent",
        "LOGNAME": "agent",
    }
    for name, value in seeded.items():
        monkeypatch.setenv(name, value)

    env = ToolExecutionContext.for_project(tmp_path).command_environment()

    assert env == seeded


def test_env_passthrough_opt_in_copies_named_variables(monkeypatch, tmp_path):
    _strip_non_allowlisted_env(monkeypatch)
    monkeypatch.setenv("CTXAI_EXTRA", "opted-in")

    context = ToolExecutionContext.for_project(tmp_path, env_passthrough=["CTXAI_EXTRA", "CTXAI_MISSING"])
    env = context.command_environment()

    assert env["CTXAI_EXTRA"] == "opted-in"
    assert "CTXAI_MISSING" not in env


def test_explicit_environment_wins_over_inherited_values(monkeypatch, tmp_path):
    _strip_non_allowlisted_env(monkeypatch)
    monkeypatch.setenv("PATH", "/usr/bin:/bin")

    context = ToolExecutionContext(tmp_path, environment={"PATH": "/custom", "EXTRA_VAR": "1"})
    env = context.command_environment()

    assert env["PATH"] == "/custom"
    assert env["EXTRA_VAR"] == "1"


@pytest.mark.asyncio
async def test_bash_tool_passes_allowlisted_environment_to_subprocess(monkeypatch, tmp_path):
    _strip_non_allowlisted_env(monkeypatch)
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-secret")
    monkeypatch.setenv("CTXAI_OPTED_IN", "visible")

    config = AgentToolsConfig(env_passthrough=["CTXAI_OPTED_IN"])
    captured: dict = {}

    async def fake_exec(*argv, **kwargs):
        captured["argv"] = argv
        captured["env"] = kwargs.get("env")
        return _FakeProcess(stdout=b"ok")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)

    bash = BashTool(config, context=ToolExecutionContext.for_project(tmp_path))
    result = await bash.execute("true")

    assert result["success"]
    env = captured["env"]
    assert "ANTHROPIC_API_KEY" not in env
    assert env["CTXAI_OPTED_IN"] == "visible"
    assert set(env) <= set(ALLOWED_ENVIRONMENT_KEYS) | {"CTXAI_OPTED_IN"}


# ---------------------------------------------------------------------------
# Output limits: bash stdout/stderr
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_bash_stdout_is_truncated_with_marker_and_audit_records_original_size(tmp_path):
    config = AgentToolsConfig(max_output_chars=100)
    context = ToolExecutionContext.for_project(tmp_path)
    bash = BashTool(config, context=context)

    result = await bash.execute(f"echo {'a' * 300}")

    assert result["success"]
    assert "...[truncated 201 of 301 chars]" in result["result"]
    assert result["metadata"]["stdout_truncated"] is True
    assert result["metadata"]["original_stdout_chars"] == 301
    audit = context.audit_log[-1]
    assert audit.details["stdout_chars"] == 301
    assert audit.details["truncated"] is True


@pytest.mark.asyncio
async def test_bash_stderr_is_truncated_in_error_field(monkeypatch, tmp_path):
    config = AgentToolsConfig(max_output_chars=50)
    captured: dict = {}

    async def fake_exec(*argv, **kwargs):
        captured["env"] = kwargs.get("env")
        return _FakeProcess(stderr=b"e" * 500, returncode=1)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    bash = BashTool(config, context=ToolExecutionContext.for_project(tmp_path))

    result = await bash.execute("false")

    assert not result["success"]
    assert "...[truncated 450 of 500 chars]" in str(result["error"])
    assert result["metadata"]["stderr_truncated"] is True
    assert result["metadata"]["original_stderr_chars"] == 500


@pytest.mark.asyncio
async def test_bash_audit_records_never_contain_secrets(monkeypatch, tmp_path):
    _strip_non_allowlisted_env(monkeypatch)
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-secret")

    bash = BashTool(AgentToolsConfig(), context=ToolExecutionContext.for_project(tmp_path))
    result = await bash.execute("echo hello")

    assert result["success"]
    dumped = json.dumps([record.__dict__ for record in bash.context.audit_log])
    assert "sk-test-secret" not in dumped


# ---------------------------------------------------------------------------
# Output limits: read_file
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_read_file_truncates_content_entering_the_context(tmp_path):
    content = "x" * 300
    (tmp_path / "big.txt").write_text(content, encoding="utf-8")
    tool = ReadFileTool(context=ToolExecutionContext.for_project(tmp_path), max_output_chars=100)

    result = await tool.execute(file_path="big.txt")

    assert result["success"]
    expected_original = len(f"   1 | {content}")
    assert result["metadata"]["truncated"] is True
    assert result["metadata"]["original_chars"] == expected_original
    assert result["result"].endswith(f"...[truncated {expected_original - 100} of {expected_original} chars]")


@pytest.mark.asyncio
async def test_read_file_within_limit_is_not_truncated(tmp_path):
    (tmp_path / "small.txt").write_text("tiny\n", encoding="utf-8")
    tool = ReadFileTool(context=ToolExecutionContext.for_project(tmp_path), max_output_chars=100)

    result = await tool.execute(file_path="small.txt")

    assert result["success"]
    assert result["metadata"]["truncated"] is False
    assert "truncated" not in result["result"]


# ---------------------------------------------------------------------------
# Verbose diagnostics
# ---------------------------------------------------------------------------


def test_verbose_diagnostics_report_truncation_and_replacement_counts(tmp_path):
    agent = Agent(
        AgentLoopConfig(
            llm_provider=MockLLMProvider(),
            tool_registry=ToolRegistry(),
            agent_config=AgentConfig(),
            working_directory=tmp_path,
            available_indexes=[],
        )
    )
    buffer = io.StringIO()
    agent.console = Console(file=buffer, legacy_windows=False, width=200)

    agent._print_result_diagnostics(
        {
            "metadata": {
                "stdout_truncated": True,
                "original_stdout_chars": 40000,
                "replacements": 2,
                "strategy": "replace_all",
            }
        }
    )

    output = buffer.getvalue()
    assert "Truncated stdout" in output
    assert "40000" in output
    assert "20000" in output
    assert "2 replacement(s)" in output
    assert "replace_all" in output
