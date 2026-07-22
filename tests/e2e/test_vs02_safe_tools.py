"""Acceptance tests for VS-02 safe repository tools."""

from __future__ import annotations

import subprocess
import sys

import pytest

from ctxai.agent.config import AgentToolsConfig
from ctxai.agent.tools.bash_tool import BashTool
from ctxai.agent.tools.execution import Capability, ToolExecutionContext
from ctxai.agent.tools.file_ops import EditFileTool, GlobTool, GrepTool, ListFilesTool, ReadFileTool, WriteFileTool
from ctxai.agent.tools.git_tools import GitDiffTool, GitLogTool, GitStatusTool


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_all_file_tools_reject_traversal_and_symlink_escape(tmp_path):
    project = tmp_path / "project"
    outside = tmp_path / "outside"
    project.mkdir()
    outside.mkdir()
    (outside / "secret.txt").write_text("secret")
    (project / "escape").symlink_to(outside, target_is_directory=True)
    context = ToolExecutionContext.for_project(project)

    results = [
        await ReadFileTool(context=context).execute(file_path="../outside/secret.txt"),
        await ReadFileTool(context=context).execute(file_path="escape/secret.txt"),
        await WriteFileTool(context=context).execute(file_path="../outside/new.txt", content="no"),
        await WriteFileTool(context=context).execute(file_path="escape/new.txt", content="no"),
        await EditFileTool(context=context).execute(file_path="escape/secret.txt", old_text="secret", new_text="leak"),
        await ListFilesTool(context=context).execute(directory_path="escape"),
        await GlobTool(context=context).execute(pattern="*", base_path="escape"),
        await GrepTool(context=context).execute(pattern="secret", file_pattern="*", base_path="escape"),
    ]
    assert all(not result["success"] for result in results)
    assert all("outside_project" in result["error"] for result in results)
    assert not (outside / "new.txt").exists()
    assert (outside / "secret.txt").read_text() == "secret"


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_mutations_return_diff_and_share_audit_log(tmp_path):
    context = ToolExecutionContext.for_project(tmp_path)
    write = await WriteFileTool(context=context).execute(file_path="example.py", content="value = 1\n")
    edit = await EditFileTool(context=context).execute(file_path="example.py", old_text="1", new_text="2")

    assert write["success"] and "diff" in write and "+value = 1" in write["diff"]
    assert edit["success"] and "-value = 1" in edit["diff"] and "+value = 2" in edit["diff"]
    assert [record.action for record in context.audit_log] == ["write", "edit"]
    assert all(record.request_id == context.request_id for record in context.audit_log)
    assert tmp_path.joinpath("example.py").read_text() == "value = 2\n"


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_capabilities_and_command_policy_are_enforced(tmp_path):
    read_only = ToolExecutionContext(tmp_path, capabilities={Capability.READ})
    denied_write = await WriteFileTool(context=read_only).execute(file_path="blocked.txt", content="no")
    denied_command = await BashTool(AgentToolsConfig(), context=read_only).execute("python -V")
    assert not denied_write["success"] and "workspace_write" in denied_write["error"]
    assert not denied_command["success"] and "command" in denied_command["error"]

    context = ToolExecutionContext.for_project(tmp_path)
    bash = BashTool(AgentToolsConfig(), context=context)
    dangerous = await bash.execute("rm -f harmless.txt")
    network = await bash.execute("curl https://example.com")
    chained = await bash.execute("python -V && rm -f harmless.txt")
    escaped_cwd = await bash.execute("python -V", working_directory="..")
    inline_code = await bash.execute("python -c 'print(42)'")
    git_mutation = await bash.execute("git reset --hard")
    safe = await bash.execute(f"{sys.executable} -V")
    assert all(not item["success"] for item in (dangerous, network, chained, escaped_cwd))
    assert dangerous["error_type"] == "PolicyDenied"
    assert network["error_type"] == "PolicyDenied"
    assert not inline_code["success"] and inline_code["error_type"] == "PolicyDenied"
    assert not git_mutation["success"] and git_mutation["error_type"] == "PolicyDenied"
    assert safe["success"]
    assert context.audit_log[-1].action == "command"


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_outside_access_requires_explicit_capability(tmp_path):
    project = tmp_path / "project"
    outside = tmp_path / "outside.txt"
    project.mkdir()
    outside.write_text("allowed")
    context = ToolExecutionContext.for_project(project, allow_outside_project=True)
    result = await ReadFileTool(context=context).execute(file_path=str(outside))
    assert result["success"] and "allowed" in result["result"]


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_read_only_git_tools_are_rooted_and_functional(tmp_path):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, check=True)
    (tmp_path / "tracked.txt").write_text("one\n")
    subprocess.run(["git", "add", "tracked.txt"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "initial"], cwd=tmp_path, check=True)
    (tmp_path / "tracked.txt").write_text("two\n")
    context = ToolExecutionContext.for_project(tmp_path)

    status = await GitStatusTool(context=context).execute()
    diff = await GitDiffTool(context=context).execute(path="tracked.txt")
    log = await GitLogTool(context=context).execute(limit=1)
    escaped = await GitStatusTool(context=context).execute(path="..")
    assert status["success"] and "tracked.txt" in status["result"]
    assert diff["success"] and "+two" in diff["result"]
    assert log["success"] and "initial" in log["result"]
    assert not escaped["success"] and "outside_project" in escaped["error"]
