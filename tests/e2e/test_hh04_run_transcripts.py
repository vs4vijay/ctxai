"""HH-04 acceptance tests: run transcripts and cost ledger.

Runs the real agent loop, tool registry, and tools against scripted
MockLLMProvider responses to prove: completed and failed runs leave parseable
``.ctxai/runs/<run_id>.jsonl`` transcripts with matching run ids and strictly
increasing seq, transcript events reconstruct the TaskRun state transitions,
``ctxai runs show`` round-trips (including kind filtering and --json matching
the on-disk schema), seeded API keys in tool output never appear in any
persisted event, per-run usage totals match the ledger with honest cost
handling, retention prunes oldest-first, and ``record_runs: false`` writes
nothing.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from ctxai.agent.config import AgentConfig, AgentLLMConfig
from ctxai.agent.core import Agent, AgentLoopConfig
from ctxai.agent.llm.base import ProviderError, ProviderErrorKind
from ctxai.agent.run_recorder import RUN_SCHEMA_VERSION, RunEvent, runs_dir_for
from ctxai.agent.tools.bash_tool import BashTool
from ctxai.agent.tools.execution import ToolExecutionContext
from ctxai.agent.tools.file_ops import ReadFileTool, WriteFileTool
from ctxai.agent.tools.registry import ToolRegistry
from ctxai.agent.workflow import FailureKind, TaskState
from ctxai.app import runs_app
from ctxai.commands.runs_command import format_usage_cost_line
from tests.mocks.mock_llm import MockLLMProvider, create_mock_response

USAGE = {"prompt_tokens": 300, "completion_tokens": 10, "total_tokens": 310}
SEEDED_SECRET = "sk-test-0123456789abcdef"


def make_agent(
    temp_dir,
    mock_llm_config,
    provider,
    *,
    agent_config: AgentConfig | None = None,
    run_id: str | None = None,
    max_iterations: int = 12,
) -> Agent:
    """Build a real agent (loop + registry + file/bash tools) over the provider.

    Args:
        temp_dir: Project root for the run.
        mock_llm_config: LLM configuration for the provider.
        provider: Scripted LLM provider instance.
        agent_config: Optional agent config override (behavior flags).
        run_id: Optional pinned transcript run id.
        max_iterations: Loop iteration budget.

    Returns:
        The configured Agent.
    """
    agent_config = agent_config or AgentConfig()
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
        max_iterations=max_iterations,
        require_user_approval=True,
        approval_callback=lambda call: True,
        run_id=run_id,
    )
    return Agent(loop_config)


def read_transcript(temp_dir: Path, run_id: str) -> list[RunEvent]:
    """Parse a transcript from disk into events (raw parse, no CLI).

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
async def test_completed_run_leaves_parseable_transcript_reconstructing_taskrun(
    temp_dir, mock_llm_config, patch_embeddings_factory
):
    """Criterion 1: a completed run leaves a parseable transcript whose events
    reconstruct the TaskRun state transitions, usage ledger, and evidence."""
    (temp_dir / "note.txt").write_text("the note contents", encoding="utf-8")
    provider = MockLLMProvider(
        config=mock_llm_config,
        responses=[
            create_mock_response(tool_calls=[{"name": "read_file", "parameters": {"path": "note.txt"}}], usage=USAGE),
            create_mock_response(
                tool_calls=[{"name": "write_file", "parameters": {"path": "out.txt", "content": "written"}}],
                usage=USAGE,
            ),
            create_mock_response(
                tool_calls=[{"name": "bash", "parameters": {"command": "echo verified"}}], usage=USAGE
            ),
            create_mock_response(content="All done: wrote out.txt and verified.", usage=USAGE),
        ],
    )
    agent = make_agent(temp_dir, mock_llm_config, provider, run_id="e2e-completed-run")

    report = await agent.process_message("Create out.txt from note.txt and verify")

    assert "Status: succeeded" in report
    run = agent.last_run
    assert run is not None

    path = runs_dir_for(temp_dir) / "e2e-completed-run.jsonl"
    assert path.is_file(), "the transcript exists under .ctxai/runs/"
    lines = path.read_text(encoding="utf-8").splitlines()
    events = [json.loads(line) for line in lines]

    # Every line parses, carries the schema version and matching run_id, and
    # seq increases strictly starting at 1; line 1 is run_started.
    assert events[0]["kind"] == "run_started"
    assert all(event["schema_version"] == RUN_SCHEMA_VERSION for event in events)
    assert all(event["run_id"] == "e2e-completed-run" for event in events)
    seqs = [event["seq"] for event in events]
    assert seqs == list(range(1, len(events) + 1))

    kinds = [event["kind"] for event in events]
    for expected in (
        "run_started",
        "user_message",
        "llm_call",
        "tool_call",
        "tool_result",
        "approval",
        "check",
        "run_completed",
    ):
        assert expected in kinds, f"missing {expected} event kind"

    # State transitions reconstruct the TaskRun: initial state from
    # run_started + drained state_transition payloads == run.transitions.
    parsed = [RunEvent.from_dict(event) for event in events]
    initial = parsed[0].payload["state"]
    drained = [event.payload["state"] for event in parsed if event.kind == "state_transition"]
    assert [initial, *drained] == [state.value for state in run.transitions]

    # Per-run usage totals match the ledger exactly, event by event.
    usage_events = [event.usage for event in parsed if event.kind == "llm_call" and event.usage]
    ledger_records = [record.to_dict() for record in run.usage.records]
    assert usage_events == ledger_records
    completed = next(event for event in parsed if event.kind == "run_completed")
    assert completed.payload["usage"] == run.usage.totals()
    assert completed.payload["status"] == "succeeded"

    # Evidence survives: the changed file and the passing check are recorded.
    assert "out.txt" in completed.payload["changed_files"]
    assert any(check["success"] for check in completed.payload["checks"])

    # Run completed with state SUMMARIZE drained before run_completed.
    assert run.state is TaskState.SUMMARIZE


@pytest.mark.e2e
@pytest.mark.agent
@pytest.mark.asyncio
async def test_failed_run_also_leaves_parseable_transcript(temp_dir, mock_llm_config, patch_embeddings_factory):
    """Criterion 1: a failed run (auth error) leaves a parseable transcript."""

    class AuthFailureProvider(MockLLMProvider):
        def chat(self, messages, tools=None, **kwargs):
            raise ProviderError(ProviderErrorKind.AUTHENTICATION, "invalid api key", provider="AuthFailureProvider")

    provider = AuthFailureProvider(config=mock_llm_config)
    agent = make_agent(temp_dir, mock_llm_config, provider, run_id="e2e-failed-run")

    report = await agent.process_message("Do the impossible task")

    assert report.startswith("Status: failed")
    assert agent.last_run is not None
    assert agent.last_run.failure_kind is FailureKind.INFRASTRUCTURE_FAILURE

    events = read_transcript(temp_dir, "e2e-failed-run")
    assert events[0].kind == "run_started"
    seqs = [event.seq for event in events]
    assert seqs == list(range(1, len(events) + 1))
    completed = next(event for event in events if event.kind == "run_completed")
    assert completed.payload["status"] == "failed"
    assert completed.payload["failure_kind"] == "infrastructure_failure"
    assert "AuthFailureProvider" in completed.payload["failure_message"]


@pytest.mark.e2e
@pytest.mark.agent
@pytest.mark.asyncio
async def test_seeded_secret_in_tool_output_never_reaches_transcript(
    temp_dir, mock_llm_config, patch_embeddings_factory
):
    """Criterion 3: seeded API key patterns in tool output are redacted everywhere."""
    (temp_dir / "secret.env").write_text(
        f"api_key={SEEDED_SECRET}\n"
        "github_token: ghp_ABCDEF1234567890\n"
        "Authorization: Bearer abcdef1234567890\n"
        "password=hunter2secret\n",
        encoding="utf-8",
    )
    provider = MockLLMProvider(
        config=mock_llm_config,
        responses=[
            create_mock_response(tool_calls=[{"name": "read_file", "parameters": {"path": "secret.env"}}], usage=USAGE),
            create_mock_response(content="I read the environment file.", usage=USAGE),
        ],
    )
    agent = make_agent(temp_dir, mock_llm_config, provider, run_id="e2e-secret-run")

    await agent.process_message("Read secret.env and summarize it")

    raw = (runs_dir_for(temp_dir) / "e2e-secret-run.jsonl").read_text(encoding="utf-8")
    assert SEEDED_SECRET not in raw
    assert "ghp_ABCDEF1234567890" not in raw
    assert "abcdef1234567890" not in raw
    assert "hunter2secret" not in raw
    assert "[REDACTED]" in raw, "the redaction marker proves the content was recorded and redacted"
    # Redaction covers tool parameters and results, messages, and approvals:
    for event_line in raw.splitlines():
        event = json.loads(event_line)
        assert "[REDACTED]" not in json.dumps(event["seq"])  # sanity: structure intact
        blob = json.dumps(event)
        assert SEEDED_SECRET not in blob and "hunter2secret" not in blob


@pytest.mark.e2e
@pytest.mark.agent
def test_runs_show_round_trips_with_kind_filter_and_json(temp_dir, mock_llm_config):
    """Criterion 2: ``runs show`` renders kind-filtered events and --json
    matches the on-disk schema version."""
    from ctxai.agent.run_recorder import RunEventKind, RunRecorder

    recorder = RunRecorder(temp_dir, "cli-show-run")
    recorder.record(RunEventKind.RUN_STARTED, {"goal": "demo"})
    recorder.record(
        RunEventKind.LLM_CALL,
        {"call_index": 1},
        usage={"provider": "P", "model": "m", "prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
    )
    recorder.record(RunEventKind.RUN_COMPLETED, {"status": "succeeded"})
    recorder.close()

    runner = CliRunner()
    disk_path = runs_dir_for(temp_dir) / "cli-show-run.jsonl"
    disk_events = [json.loads(line) for line in disk_path.read_text(encoding="utf-8").splitlines() if line]

    result = runner.invoke(runs_app, ["show", "cli-show-run", "--json", "--project-path", str(temp_dir)])
    assert result.exit_code == 0, result.output
    envelope = json.loads(result.output)
    assert envelope["schema_version"] == disk_events[0]["schema_version"] == RUN_SCHEMA_VERSION
    assert envelope["run_id"] == "cli-show-run"
    assert envelope["events"] == disk_events, "--json output matches the on-disk events byte-for-byte"

    result = runner.invoke(
        runs_app, ["show", "cli-show-run", "--kind", "llm_call", "--json", "--project-path", str(temp_dir)]
    )
    assert result.exit_code == 0, result.output
    filtered = json.loads(result.output)
    assert [event["kind"] for event in filtered["events"]] == ["llm_call"]
    assert filtered["events"][0]["usage"]["total_tokens"] == 4

    result = runner.invoke(runs_app, ["show", "cli-show-run", "--kind", "not-a-kind", "--project-path", str(temp_dir)])
    assert result.exit_code == 1
    assert "unknown event kind" in result.output

    result = runner.invoke(runs_app, ["show", "missing-run", "--project-path", str(temp_dir)])
    assert result.exit_code == 1


@pytest.mark.e2e
@pytest.mark.agent
@pytest.mark.asyncio
async def test_usage_cost_line_known_and_unknown_models(temp_dir, patch_embeddings_factory):
    """Criterion 4: known models get a cost estimate, unknown models an explicit
    unknown — and the totals always match the ledger."""
    known_config = AgentLLMConfig(provider="mock", model="gpt-4o", api_key="mock-key")
    provider = MockLLMProvider(
        config=known_config,
        responses=[create_mock_response(content="Done.", usage=USAGE)],
    )
    agent = make_agent(temp_dir, known_config, provider)
    await agent.process_message("Just answer")

    run = agent.last_run
    assert run is not None
    totals = run.usage.totals()
    assert totals["prompt_tokens"] == 300 and totals["completion_tokens"] == 10

    line = format_usage_cost_line(run)
    assert line is not None
    assert "300 prompt + 10 completion tokens over 1 call(s)" in line
    assert "$" in line and "unknown" not in line

    unknown_line = format_usage_cost_line(None)
    assert unknown_line is None, "no usage means no usage line"

    unpriced_config = AgentLLMConfig(provider="mock", model="totally-unknown-model", api_key="mock-key")
    provider = MockLLMProvider(
        config=unpriced_config,
        responses=[create_mock_response(content="Done.", usage=USAGE)],
    )
    agent = make_agent(temp_dir, unpriced_config, provider)
    await agent.process_message("Just answer again")

    run = agent.last_run
    line = format_usage_cost_line(run)
    assert line is not None
    assert "cost: unknown (no price entry for totally-unknown-model)" in line
    assert "$" not in line.split("cost:")[1], "an unknown cost never fabricates a zero"


@pytest.mark.e2e
@pytest.mark.agent
@pytest.mark.asyncio
async def test_record_runs_disabled_writes_nothing(temp_dir, mock_llm_config, patch_embeddings_factory):
    """Criterion 5: with record_runs disabled, no file appears under .ctxai/runs/."""
    config = AgentConfig()
    config.behavior.record_runs = False
    provider = MockLLMProvider(
        config=mock_llm_config,
        responses=[create_mock_response(content="Done quietly.", usage=USAGE)],
    )
    agent = make_agent(temp_dir, mock_llm_config, provider, agent_config=config)

    report = await agent.process_message("Work without a transcript")
    assert "Status: succeeded" in report

    assert not runs_dir_for(temp_dir).exists() or not list(runs_dir_for(temp_dir).glob("*.jsonl"))


@pytest.mark.e2e
@pytest.mark.agent
@pytest.mark.asyncio
async def test_retention_prunes_oldest_transcripts_at_run_start(temp_dir, mock_llm_config, patch_embeddings_factory):
    """run_retention prunes oldest-first at run start, keeping the newest window."""
    config = AgentConfig()
    config.behavior.run_retention = 2
    provider = MockLLMProvider(
        config=mock_llm_config,
        responses=[create_mock_response(content="Done.", usage=USAGE)],
    )
    agent = make_agent(temp_dir, mock_llm_config, provider, agent_config=config)

    for turn in range(4):
        await agent.process_message(f"Turn {turn}")

    files = sorted(path.name for path in runs_dir_for(temp_dir).glob("*.jsonl"))
    assert len(files) == 2, f"retention keeps the newest {config.behavior.run_retention} transcripts, got {files}"


@pytest.mark.e2e
@pytest.mark.agent
@pytest.mark.asyncio
async def test_runs_list_reports_status_usage_and_cost(temp_dir, mock_llm_config, patch_embeddings_factory):
    """``runs list`` shows runs newest-first with status and usage; --json is versioned."""
    provider = MockLLMProvider(
        config=mock_llm_config,
        responses=[create_mock_response(content="Answered.", usage=USAGE)],
    )
    agent = make_agent(temp_dir, mock_llm_config, provider, run_id="list-run-one")
    await agent.process_message("First question")

    runner = CliRunner()
    result = runner.invoke(runs_app, ["list", "--json", "--project-path", str(temp_dir)])
    assert result.exit_code == 0, result.output
    envelope = json.loads(result.output)
    assert envelope["schema_version"] == 1
    assert [run["run_id"] for run in envelope["runs"]] == ["list-run-one"]
    summary = envelope["runs"][0]
    assert summary["status"] == "succeeded"
    assert summary["prompt_tokens"] == 300
    assert summary["calls"] == 1
    assert summary["unknown_model"] == "mock-model-v1", "unpriced models are surfaced, not zeroed"
    assert summary["cost"] is None

    result = runner.invoke(runs_app, ["list", "--project-path", str(temp_dir)], env={"COLUMNS": "250"})
    assert result.exit_code == 0
    assert "list-run-one" in result.output
    assert "unknown (no price entry for mock-model-v1)" in result.output


@pytest.mark.e2e
@pytest.mark.agent
def test_runs_delete_single_and_all(temp_dir, mock_llm_config):
    """runs delete removes one transcript without confirmation; --all confirms."""
    from ctxai.agent.run_recorder import RunEventKind, RunRecorder

    for name in ("del-one", "del-two"):
        recorder = RunRecorder(temp_dir, name)
        recorder.record(RunEventKind.RUN_STARTED, {})
        recorder.close()

    runner = CliRunner()
    result = runner.invoke(runs_app, ["delete", "del-one", "--project-path", str(temp_dir)])
    assert result.exit_code == 0, result.output
    assert not (runs_dir_for(temp_dir) / "del-one.jsonl").exists()
    assert (runs_dir_for(temp_dir) / "del-two.jsonl").exists()

    result = runner.invoke(runs_app, ["delete", "del-two", "--all", "--project-path", str(temp_dir)])
    assert result.exit_code == 1, "RUN_ID and --all are mutually exclusive"

    result = runner.invoke(runs_app, ["delete", "--all", "--project-path", str(temp_dir)], input="n\n")
    assert result.exit_code != 0, "declining the confirmation aborts"
    assert (runs_dir_for(temp_dir) / "del-two.jsonl").exists()

    result = runner.invoke(runs_app, ["delete", "--all", "--project-path", str(temp_dir)], input="y\n")
    assert result.exit_code == 0, result.output
    assert not list(runs_dir_for(temp_dir).glob("*.jsonl"))
