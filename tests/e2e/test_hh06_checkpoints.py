"""HH-06 acceptance tests: checkpoints and rollback.

Runs the real agent loop, tool registry, and tools against scripted
MockLLMProvider responses to prove: a run whose verification fails is
reversible with one command — ``ctxai checkpoints restore`` returns every
touched file byte-identically to its pre-run state, removes files the run
created, and recreates files the run captured and later deleted (here via a
policy-approved bash ``rm``); every mutation is preceded by a checkpoint
capture (fault injected between capture and write); a post-run manual edit
blocks restore with a per-file reason unless ``--force``; and the restore is
recorded as a ``rollback`` event on the run's HH-04 transcript.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from ctxai.agent.checkpoints import CaptureKind, CheckpointManager
from ctxai.agent.config import AgentConfig
from ctxai.agent.core import Agent, AgentLoopConfig
from ctxai.agent.run_recorder import RunEvent, RunEventKind, runs_dir_for
from ctxai.agent.tools.bash_tool import BashTool
from ctxai.agent.tools.execution import Capability, ToolExecutionContext
from ctxai.agent.tools.file_ops import ReadFileTool, WriteFileTool
from ctxai.agent.tools.registry import ToolRegistry
from ctxai.app import checkpoints_app
from tests.mocks.mock_llm import MockLLMProvider, create_mock_response

USAGE = {"prompt_tokens": 200, "completion_tokens": 10, "total_tokens": 210}


class FaultyWriteFileTool(WriteFileTool):
    """Write tool that always fails after the loop captured the pre-state."""

    def __init__(self, **kwargs) -> None:
        """Initialize under the real write_file identity.

        Args:
            **kwargs: Forwarded to WriteFileTool.
        """
        super().__init__(**kwargs)
        self.name = "write_file"  # replace the real tool in the registry

    async def execute(self, content: str, file_path: str | None = None, path: str | None = None) -> dict:
        """Fail without writing (fault injected between capture and write).

        Returns the failure shape the real tool layer produces (error_type
        preserved) so the workflow classifies the fault like a real I/O error.

        Args:
            content: Unused scripted content.
            file_path: Unused target path.
            path: Unused legacy alias.

        Returns:
            A failure result with ``error_type`` set.
        """
        return {
            "success": False,
            "result": None,
            "error": "injected fault between capture and write",
            "error_type": "OSError",
        }


def make_agent(
    temp_dir,
    mock_llm_config,
    provider,
    *,
    agent_config: AgentConfig | None = None,
    run_id: str | None = None,
    destructive: bool = False,
) -> Agent:
    """Build a real agent (loop + registry + file/bash tools) over the provider.

    Args:
        temp_dir: Project root for the run.
        mock_llm_config: LLM configuration for the provider.
        provider: Scripted LLM provider instance.
        agent_config: Optional agent config override (behavior flags).
        run_id: Optional pinned checkpoint/transcript run id.
        destructive: When True, grant the bash tool the destructive capability
            with an approving capability callback (models the user approving
            the exact destructive command, e.g. ``rm``).

    Returns:
        The configured Agent.
    """
    agent_config = agent_config or AgentConfig()
    if destructive:
        context = ToolExecutionContext(
            temp_dir,
            capabilities={Capability.READ, Capability.WORKSPACE_WRITE, Capability.COMMAND, Capability.DESTRUCTIVE},
            approval_callback=lambda capability, action, target: True,
        )
    else:
        context = ToolExecutionContext.for_project(temp_dir)
    registry = ToolRegistry()
    registry.register(ReadFileTool(context=context, max_output_chars=20_000))
    registry.register(WriteFileTool(context=context))
    registry.register(BashTool(agent_config.tools, context=context))
    loop_config = AgentLoopConfig(
        llm_provider=provider,
        tool_registry=registry,
        agent_config=agent_config,
        working_directory=temp_dir,
        available_indexes=[],
        max_iterations=12,
        require_user_approval=True,
        approval_callback=lambda call: True,
        run_id=run_id,
        checkpoint_manager=CheckpointManager.for_project(
            temp_dir,
            retention=agent_config.behavior.checkpoint_retention,
            max_bytes=agent_config.behavior.checkpoint_max_bytes,
        ),
    )
    return Agent(loop_config)


def read_transcript(temp_dir: Path, run_id: str) -> list[RunEvent]:
    """Parse a transcript from disk into events.

    Args:
        temp_dir: Project root.
        run_id: The run identifier.

    Returns:
        The parsed RunEvent list.
    """
    path = runs_dir_for(temp_dir) / f"{run_id}.jsonl"
    return [RunEvent.from_dict(json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines() if line]


@pytest.mark.e2e
@pytest.mark.agent
@pytest.mark.asyncio
async def test_failed_run_rolls_back_created_modified_and_deleted_files(
    temp_dir, mock_llm_config, patch_embeddings_factory
):
    """Criterion 1: after a failed run, one restore returns every touched file
    to byte-identical pre-run state across create/modify/delete."""
    (temp_dir / "mod.txt").write_bytes(b"original mod")
    (temp_dir / "del.txt").write_bytes(b"original del")
    run_id = "hh06-rollback-run"
    provider = MockLLMProvider(
        config=mock_llm_config,
        responses=[
            create_mock_response(tool_calls=[{"name": "read_file", "parameters": {"path": "del.txt"}}], usage=USAGE),
            create_mock_response(
                tool_calls=[{"name": "write_file", "parameters": {"path": "del.txt", "content": "temp change"}}],
                usage=USAGE,
            ),
            create_mock_response(tool_calls=[{"name": "bash", "parameters": {"command": "rm del.txt"}}], usage=USAGE),
            create_mock_response(tool_calls=[{"name": "read_file", "parameters": {"path": "mod.txt"}}], usage=USAGE),
            create_mock_response(
                tool_calls=[{"name": "write_file", "parameters": {"path": "mod.txt", "content": "changed"}}],
                usage=USAGE,
            ),
            create_mock_response(
                tool_calls=[{"name": "write_file", "parameters": {"path": "new.txt", "content": "brand new"}}],
                usage=USAGE,
            ),
            create_mock_response(tool_calls=[{"name": "bash", "parameters": {"command": "false"}}], usage=USAGE),
            create_mock_response(content="I applied the scripted changes.", usage=USAGE),
        ],
    )
    agent = make_agent(temp_dir, mock_llm_config, provider, run_id=run_id, destructive=True)

    report = await agent.process_message("Apply the scripted edits and add the new file.")

    assert "Status: failed" in report, "the failing verification command fails the run"

    # The checkpoint exists, is finalized (not retained — the run failed), and
    # captured exactly the three touched files with the right capture kinds.
    manager = CheckpointManager.for_project(temp_dir)
    checkpoint = manager.load(run_id)
    assert checkpoint.status == "finalized"
    assert checkpoint.retained is False
    kinds = {entry.path: entry.kind for entry in checkpoint.files}
    assert kinds == {"mod.txt": CaptureKind.FILE, "del.txt": CaptureKind.FILE, "new.txt": CaptureKind.CREATED}

    # The CLI restore asks for confirmation (decline first, then confirm).
    runner = CliRunner()
    declined = runner.invoke(
        checkpoints_app, ["restore", run_id, "--project-path", str(temp_dir)], input="n\n", env={"COLUMNS": "200"}
    )
    assert declined.exit_code != 0, "declining the confirmation aborts without touching files"
    assert (temp_dir / "mod.txt").read_bytes() == b"changed"

    restored = runner.invoke(
        checkpoints_app, ["restore", run_id, "--project-path", str(temp_dir)], input="y\n", env={"COLUMNS": "200"}
    )
    assert restored.exit_code == 0, restored.output
    assert "mod.txt" in restored.output and "del.txt" in restored.output and "new.txt" in restored.output

    # Byte-identical pre-run state across create/modify/delete.
    assert (temp_dir / "mod.txt").read_bytes() == b"original mod"
    assert (temp_dir / "del.txt").read_bytes() == b"original del", "the deleted file is recreated"
    assert not (temp_dir / "new.txt").exists(), "the created file is removed"

    # The restore is recorded as a rollback event continuing the transcript.
    events = read_transcript(temp_dir, run_id)
    seqs = [event.seq for event in events]
    assert seqs == list(range(1, len(events) + 1))
    rollback = events[-1]
    assert rollback.kind == RunEventKind.ROLLBACK.value
    assert rollback.payload["status"] == "restored"
    assert rollback.payload["forced"] is False
    assert {item["path"] for item in rollback.payload["files"]} == {"mod.txt", "del.txt", "new.txt"}


@pytest.mark.e2e
@pytest.mark.agent
@pytest.mark.asyncio
async def test_post_run_manual_edit_blocks_restore_without_force(temp_dir, mock_llm_config, patch_embeddings_factory):
    """Criterion 3: a moved-on working tree refuses restore per file unless forced."""
    (temp_dir / "mod.txt").write_bytes(b"original")
    run_id = "hh06-stale-run"
    provider = MockLLMProvider(
        config=mock_llm_config,
        responses=[
            create_mock_response(tool_calls=[{"name": "read_file", "parameters": {"path": "mod.txt"}}], usage=USAGE),
            create_mock_response(
                tool_calls=[{"name": "write_file", "parameters": {"path": "mod.txt", "content": "run version"}}],
                usage=USAGE,
            ),
            create_mock_response(tool_calls=[{"name": "bash", "parameters": {"command": "true"}}], usage=USAGE),
            create_mock_response(content="All done.", usage=USAGE),
        ],
    )
    agent = make_agent(temp_dir, mock_llm_config, provider, run_id=run_id)
    report = await agent.process_message("Update the note file and verify.")
    assert "Status: succeeded" in report

    manager = CheckpointManager.for_project(temp_dir)
    checkpoint = manager.load(run_id)
    assert checkpoint.retained is True, "a succeeded run keeps its checkpoint for audit"

    # The user edits the file after the run: the tree moved on.
    (temp_dir / "mod.txt").write_bytes(b"user touched this after the run")

    runner = CliRunner()
    refused = runner.invoke(
        checkpoints_app,
        ["restore", run_id, "--project-path", str(temp_dir)],
        input="y\n",
        env={"COLUMNS": "200"},
    )
    assert refused.exit_code == 1, refused.output
    assert "hash mismatch" in refused.output, "the refusal names the per-file reason"
    assert "moved on" in refused.output
    assert "--force" in refused.output
    assert (temp_dir / "mod.txt").read_bytes() == b"user touched this after the run", "nothing was modified"

    forced = runner.invoke(
        checkpoints_app,
        ["restore", run_id, "--force", "--project-path", str(temp_dir)],
        input="y\n",
        env={"COLUMNS": "200"},
    )
    assert forced.exit_code == 0, forced.output
    assert (temp_dir / "mod.txt").read_bytes() == b"original"


@pytest.mark.e2e
@pytest.mark.agent
@pytest.mark.asyncio
async def test_capture_precedes_write_under_fault_injection(temp_dir, mock_llm_config, patch_embeddings_factory):
    """Criterion 2: a fault between capture and write still leaves the
    pre-mutation bytes checkpointed and restorable."""
    (temp_dir / "note.txt").write_bytes(b"pre-run note")
    run_id = "hh06-fault-run"
    provider = MockLLMProvider(
        config=mock_llm_config,
        responses=[
            create_mock_response(tool_calls=[{"name": "read_file", "parameters": {"path": "note.txt"}}], usage=USAGE),
            create_mock_response(
                tool_calls=[{"name": "write_file", "parameters": {"path": "note.txt", "content": "never written"}}],
                usage=USAGE,
            ),
            create_mock_response(content="The write faulted.", usage=USAGE),
        ],
    )
    agent = make_agent(temp_dir, mock_llm_config, provider, run_id=run_id)

    # Swap the registered write tool for the fault-injecting one.
    registered = agent.tools.get_tool("write_file")
    assert registered is not None
    agent.tools.register(FaultyWriteFileTool(context=registered.context))

    report = await agent.process_message("Update the note file.")
    assert "Status: failed" in report
    assert (temp_dir / "note.txt").read_bytes() == b"pre-run note", "the faulted write changed nothing"

    manager = CheckpointManager.for_project(temp_dir)
    checkpoint = manager.load(run_id)
    entry = checkpoint.files[0]
    assert entry.path == "note.txt" and entry.kind is CaptureKind.FILE
    blob = manager.run_dir(run_id) / "files" / f"{entry.sha256}.blob"
    assert blob.read_bytes() == b"pre-run note", "the capture happened before the write faulted"

    result = CliRunner().invoke(
        checkpoints_app, ["restore", run_id, "--project-path", str(temp_dir)], input="y\n", env={"COLUMNS": "200"}
    )
    assert result.exit_code == 0, result.output
    assert (temp_dir / "note.txt").read_bytes() == b"pre-run note"


@pytest.mark.e2e
@pytest.mark.agent
@pytest.mark.asyncio
async def test_checkpoints_cli_list_and_delete(temp_dir, mock_llm_config, patch_embeddings_factory):
    """``checkpoints list`` (table, --json, --run filter) and delete manage the store."""
    (temp_dir / "mod.txt").write_bytes(b"original")
    run_id = "hh06-cli-run"
    provider = MockLLMProvider(
        config=mock_llm_config,
        responses=[
            create_mock_response(tool_calls=[{"name": "read_file", "parameters": {"path": "mod.txt"}}], usage=USAGE),
            create_mock_response(
                tool_calls=[{"name": "write_file", "parameters": {"path": "mod.txt", "content": "cli run"}}],
                usage=USAGE,
            ),
            create_mock_response(tool_calls=[{"name": "bash", "parameters": {"command": "true"}}], usage=USAGE),
            create_mock_response(content="Done.", usage=USAGE),
        ],
    )
    agent = make_agent(temp_dir, mock_llm_config, provider, run_id=run_id)
    await agent.process_message("Update the note file and verify.")

    runner = CliRunner()
    listed = runner.invoke(checkpoints_app, ["list", "--json", "--project-path", str(temp_dir)], env={"COLUMNS": "200"})
    assert listed.exit_code == 0, listed.output
    envelope = json.loads(listed.output)
    assert envelope["schema_version"] == 1
    assert [checkpoint["checkpoint_id"] for checkpoint in envelope["checkpoints"]] == [run_id]
    checkpoint = envelope["checkpoints"][0]
    assert checkpoint["run_id"] == run_id
    assert checkpoint["retained"] is True
    assert [entry["path"] for entry in checkpoint["files"]] == ["mod.txt"]

    filtered = runner.invoke(
        checkpoints_app, ["list", "--run", "no-such-run", "--json", "--project-path", str(temp_dir)]
    )
    assert filtered.exit_code == 0
    assert json.loads(filtered.output)["checkpoints"] == []

    table = runner.invoke(checkpoints_app, ["list", "--project-path", str(temp_dir)], env={"COLUMNS": "200"})
    assert table.exit_code == 0
    assert run_id in table.output
    assert "retained" in table.output

    deleted = runner.invoke(checkpoints_app, ["delete", run_id, "--project-path", str(temp_dir)])
    assert deleted.exit_code == 0, deleted.output
    assert not CheckpointManager.for_project(temp_dir).run_dir(run_id).exists()

    again = runner.invoke(checkpoints_app, ["delete", run_id, "--project-path", str(temp_dir)])
    assert again.exit_code == 1
    assert "No checkpoint" in again.output
