"""Execute a versioned agent task benchmark through the real agent loop (HH-09).

The runner is the only agent-benchmark execution path: for each case it
materializes the setup fixture into a fresh per-case project directory, drives
the production ``Agent`` loop (real tools, real workflow policy, HH-01..HH-07
behavior) with the selected provider, runs the case's expected checks through
the bash policy, evaluates forbidden paths, and scores the outcome with
:mod:`ctxai.evals.scoring`. Results aggregate into an immutable
:class:`ctxai.evals.agent_artifacts.AgentEvalArtifact` with per-case HH-04
transcript evidence.

Mock-provider runs (the CI path) are fully deterministic: scripted provider
responses plus fixed usage payloads make every metric byte-stable modulo the
documented volatile fields. Configured-provider runs are explicit maintainer
actions: the caller supplies the provider factory and the CLI prints a cost
warning; this module never touches the network on its own.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..agent.config import AgentConfig, AgentLLMConfig
from ..agent.core import Agent, AgentLoopConfig
from ..agent.costing import PriceTable
from ..agent.llm.base import BaseLLMProvider
from ..agent.llm.mock_provider import MockLLMProvider
from ..agent.tools.bash_tool import BashTool
from ..agent.tools.execution import ToolExecutionContext
from ..agent.tools.file_ops import EditFileTool, ListFilesTool, ReadFileTool, WriteFileTool
from ..agent.tools.registry import ToolRegistry
from ..agent.workflow import TaskState
from .agent_artifacts import (
    EVALUATION_KIND_AGENT,
    AgentCaseRunRecord,
    AgentEvalArtifact,
    CaseJudgment,
    agent_workspaces_dir,
    configuration_fingerprint,
)
from .common import EvalError, MetricValue
from .scoring import (
    aggregate_case_records,
    case_is_passed,
    judge_approvals,
    judge_budget,
    judge_checks,
    judge_forbidden_paths,
    judge_plan,
)
from .task_benchmark import AgentTaskBenchmark, AgentTaskCase

# Workspace retention: how many recent per-run workspace directories (each
# holding the per-case fixture projects and their run transcripts) are kept
# under .ctxai/evaluations/agent/workspaces.
MIN_KEEP_WORKSPACES = 1
MAX_KEEP_WORKSPACES = 50
DEFAULT_KEEP_WORKSPACES = 5

MIN_PER_CASE_TIMEOUT_S = 1.0
MAX_PER_CASE_TIMEOUT_S = 3600.0
DEFAULT_PER_CASE_TIMEOUT_S = 300.0

MUTATION_TOOL_NAMES = frozenset({"write_file", "edit_file"})

ProviderFactory = Callable[[AgentTaskCase], BaseLLMProvider]

MOCK_PROVIDER_MODE = "mock"
MOCK_PROVIDER_NAME = "MockLLMProvider"
MOCK_PROVIDER_MODEL = "mock-model-v1"


def mock_provider_identity() -> dict[str, Any]:
    """Provider identity dict for mock-provider runs (fingerprint input).

    Returns:
        The deterministic mock identity shared by the CLI and the runner.
    """
    return {"mode": MOCK_PROVIDER_MODE, "name": MOCK_PROVIDER_NAME, "model": MOCK_PROVIDER_MODEL}


def mock_provider_factory(llm_config: AgentLLMConfig | None = None) -> ProviderFactory:
    """Build a provider factory that scripts each case's ``mock_script``.

    The scripted provider makes mock runs fully deterministic: same script,
    same fixed usage payloads, same tool calls, byte-stable metrics.

    Args:
        llm_config: Optional LLM configuration for the mock provider
            (defaults to the standard deterministic mock identity).

    Returns:
        A factory mapping a benchmark case to a scripted MockLLMProvider.
    """
    config = llm_config or AgentLLMConfig(
        provider=MOCK_PROVIDER_MODE,
        model=MOCK_PROVIDER_MODEL,
        api_key="mock-key",
        temperature=0.7,
        max_tokens=4096,
        timeout=30,
    )

    def _factory(case: AgentTaskCase) -> BaseLLMProvider:
        return MockLLMProvider(config=config, responses=list(case.mock_script))

    return _factory


def configured_provider_factory(project_root: Path) -> tuple[ProviderFactory, dict[str, Any]]:
    """Build a provider factory from the project's configured default provider.

    Args:
        project_root: Resolved repository root whose ``.ctxai/config.toml``
            (merged with the global config) selects the provider.

    Returns:
        A ``(factory, identity)`` pair; the identity dict feeds the
        configuration fingerprint.

    Raises:
        EvalError: When the provider configuration is missing or the provider
            cannot be constructed.
    """
    from ..agent.llm.factory import LLMProviderFactory
    from ..config import ConfigManager

    try:
        agent_config = ConfigManager(project_root).load().agent
    except Exception as exc:
        raise EvalError(f"cannot load agent configuration for {project_root}: {exc}") from exc
    llm_config = agent_config.llm
    if not llm_config.provider:
        raise EvalError("no default provider configured; run 'ctxai login <provider>' or set the provider in config")

    def _factory(case: AgentTaskCase) -> BaseLLMProvider:
        try:
            return LLMProviderFactory.create_provider(llm_config)
        except Exception as exc:
            raise EvalError(f"cannot create the configured provider: {exc}") from exc

    identity = {
        "mode": "configured",
        "name": str(llm_config.provider),
        "model": str(llm_config.model or "provider-default"),
    }
    return _factory, identity


@dataclass
class AgentEvalConfig:
    """Runner configuration; result-affecting fields feed the config fingerprint.

    Attributes:
        per_case_timeout_s: Hard per-case wall-clock timeout in seconds.
        keep_workspaces: How many recent run workspace directories are
            retained (oldest pruned at run start).
    """

    per_case_timeout_s: float = DEFAULT_PER_CASE_TIMEOUT_S
    keep_workspaces: int = DEFAULT_KEEP_WORKSPACES

    def __post_init__(self) -> None:
        """Reject out-of-bounds configuration before any work begins.

        Raises:
            ValueError: If any bound is violated.
        """
        if not MIN_PER_CASE_TIMEOUT_S <= self.per_case_timeout_s <= MAX_PER_CASE_TIMEOUT_S:
            raise ValueError(
                f"per_case_timeout_s must be between {MIN_PER_CASE_TIMEOUT_S} and {MAX_PER_CASE_TIMEOUT_S}"
            )
        if not MIN_KEEP_WORKSPACES <= self.keep_workspaces <= MAX_KEEP_WORKSPACES:
            raise ValueError(f"keep_workspaces must be between {MIN_KEEP_WORKSPACES} and {MAX_KEEP_WORKSPACES}")

    def result_affecting_settings(self) -> dict[str, Any]:
        """Settings that can change scored outcomes (used for fingerprinting).

        Operational-only settings (timeout, workspace retention) are excluded
        so they do not invalidate baselines.

        Returns:
            Dictionary of result-affecting runner settings.
        """
        return {
            "require_user_approval": True,
            "approval_policy": "approve_all",
            "plan_mode_policy": "force-for-plan-required-cases",
        }


@dataclass
class _CaseExecution:
    """Raw evidence collected while executing one benchmark case.

    Attributes:
        project: The per-case fixture project directory.
        case_run_id: Transcript run id for the case.
        succeeded: Whether the run reached its succeeded state.
        iterations: Successful LLM calls made by the loop.
        totals: Usage ledger totals.
        changed_files: Canonical paths the run mutated.
        approvals: Approval records captured by the TaskRun.
        plan: Plan evidence dict (required/submitted/actions).
        check_results: Per-check execution evidence.
        transcript_events: Number of events in the case transcript.
    """

    project: Path
    case_run_id: str
    succeeded: bool
    iterations: int
    totals: dict[str, int]
    changed_files: set[Path]
    approvals: list[dict[str, Any]]
    plan: dict[str, Any] | None
    check_results: list[dict[str, Any]]
    transcript_events: int


class AgentBenchmarkRunner:
    """Run an agent task benchmark through the production agent loop.

    Accepts its dependencies explicitly (provider factory, clock, run id) so
    tests need no network, credentials, global configuration, or wall-clock
    timing.
    """

    def __init__(
        self,
        *,
        project_root: Path,
        benchmark: AgentTaskBenchmark,
        provider_factory: ProviderFactory,
        provider_identity: dict[str, Any],
        config: AgentEvalConfig | None = None,
        clock: Callable[[], float] = time.perf_counter,
        run_id: str | None = None,
        cases: list[AgentTaskCase] | None = None,
    ) -> None:
        """Prepare the runner.

        Args:
            project_root: Resolved repository root receiving artifacts and
                workspaces.
            benchmark: The validated benchmark to execute.
            provider_factory: Maps a case to its LLM provider (mock scripts
                or a configured provider).
            provider_identity: Provider identity dict for the configuration
                fingerprint (mode/name/model).
            config: Runner configuration; defaults apply when omitted.
            clock: Monotonic clock for the run duration.
            run_id: Optional pinned run id (defaults to a fresh uuid4 hex).
            cases: Optional explicit case subset (defaults to all cases).
        """
        import uuid as uuid_module

        self.project_root = project_root.resolve()
        self.benchmark = benchmark
        self.provider_factory = provider_factory
        self.provider_identity = dict(provider_identity)
        self.config = config or AgentEvalConfig()
        self.clock = clock
        self.run_id = run_id or uuid_module.uuid4().hex
        self.cases = cases if cases is not None else list(benchmark.cases)

    # ------------------------------------------------------------------
    # Configuration identity
    # ------------------------------------------------------------------

    def configuration_fingerprint(self) -> str:
        """Content-derived fingerprint of the evaluation configuration.

        Returns:
            Hex digest over the artifact schema version, evaluation kind,
            provider identity, and result-affecting runner settings.
        """
        return configuration_fingerprint(self.provider_identity, self.config.result_affecting_settings())

    # ------------------------------------------------------------------
    # Case execution
    # ------------------------------------------------------------------

    def _case_project_dir(self, case: AgentTaskCase) -> Path:
        """Resolve (and create) the fixture project directory for one case.

        Args:
            case: The benchmark case.

        Returns:
            The fresh project directory for this run's case.
        """
        project = agent_workspaces_dir(self.project_root) / self.run_id / case.id / "project"
        if project.exists():
            shutil.rmtree(project)
        project.mkdir(parents=True, exist_ok=True)
        for relative, content in case.setup.files.items():
            target = project / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
        return project

    def _build_agent(self, case: AgentTaskCase, project: Path) -> tuple[Agent, BashTool]:
        """Build the real agent (loop + registry + tools) for one case.

        Approvals are required and auto-approved by the callback; scoring
        separately proves every mutation still carries an approval record
        (the workflow must not have been bypassed).

        Args:
            case: The benchmark case.
            project: The per-case fixture project directory.

        Returns:
            The configured Agent and the bash tool used for expected checks.
        """
        context = ToolExecutionContext.for_project(project)
        bash = BashTool(AgentConfig().tools, context=context)
        registry = ToolRegistry()
        registry.register(ReadFileTool(context=context, max_output_chars=20_000))
        registry.register(WriteFileTool(context=context))
        registry.register(EditFileTool(context=context))
        registry.register(ListFilesTool(context=context))
        registry.register(bash)
        loop_config = AgentLoopConfig(
            llm_provider=self.provider_factory(case),
            tool_registry=registry,
            agent_config=AgentConfig(),
            working_directory=project,
            available_indexes=[],
            planning_enabled=True,
            require_user_approval=True,
            max_iterations=case.max_iterations,
            approval_callback=lambda call: True,
            run_id=f"eval-{case.id}",
            plan_mode="force" if case.plan_required else "auto",
        )
        return Agent(loop_config), bash

    async def _execute_case_async(self, case: AgentTaskCase, project: Path) -> _CaseExecution:
        """Run one case end to end (agent loop, then expected checks).

        Args:
            case: The benchmark case.
            project: The per-case fixture project directory.

        Returns:
            The raw execution evidence.
        """
        agent, bash = self._build_agent(case, project)
        await agent.process_message(case.instruction)

        check_results: list[dict[str, Any]] = []
        for check in case.expected_checks:
            result = await bash.execute(check.command)
            metadata = result.get("metadata") or {}
            stdout = str(result.get("result") or "")
            output_matched = True if check.expect_output is None else check.expect_output in stdout
            passed = bool(result.get("success")) and output_matched
            check_results.append(
                {
                    "command": check.command,
                    "description": check.description,
                    "passed": passed,
                    "exit_code": metadata.get("exit_code"),
                    "expect_output": check.expect_output,
                    "output_matched": output_matched if check.expect_output is not None else None,
                }
            )

        run = agent.last_run
        assert run is not None, "the agent loop always leaves a TaskRun"
        transcript_path = project / ".ctxai" / "runs" / f"eval-{case.id}.jsonl"
        transcript_events = 0
        if transcript_path.is_file():
            with open(transcript_path, encoding="utf-8") as handle:
                transcript_events = sum(1 for line in handle if line.strip())

        plan = run.plan
        plan_evidence = {
            "required": case.plan_required,
            "submitted": plan is not None,
            "actions": len(plan.actions) if plan is not None else 0,
            "actions_completed": (
                sum(action.status == "completed" for action in plan.actions) if plan is not None else 0
            ),
        }
        return _CaseExecution(
            project=project,
            case_run_id=f"eval-{case.id}",
            succeeded=run.state is TaskState.SUMMARIZE,
            iterations=len(run.usage.records),
            totals=run.usage.totals(),
            changed_files=set(run.changed_files),
            approvals=list(run.approvals),
            plan=plan_evidence,
            check_results=check_results,
            transcript_events=transcript_events,
        )

    def _execute_case(self, case: AgentTaskCase, project: Path) -> _CaseExecution:
        """Run one case under the configured per-case timeout.

        Args:
            case: The benchmark case.
            project: The per-case fixture project directory.

        Returns:
            The raw execution evidence.

        Raises:
            EvalError: When the case exceeds ``per_case_timeout_s`` or the
                loop raises unexpectedly.
        """
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ctxai-agent-eval-case")
        try:
            future = executor.submit(lambda: asyncio.run(self._execute_case_async(case, project)))
            try:
                return future.result(timeout=self.config.per_case_timeout_s)
            except FutureTimeoutError as exc:
                raise EvalError(f"case timed out after {self.config.per_case_timeout_s}s") from exc
        except EvalError:
            raise
        except Exception as exc:
            raise EvalError(f"case failed with an unexpected error: {exc}") from exc
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _forbidden_findings(self, case: AgentTaskCase, project: Path) -> list[dict[str, Any]]:
        """Evaluate every forbidden path against the post-run project state.

        A forbidden path that is part of the setup must be byte-identical to
        its setup content; one that is not must not exist at all.

        Args:
            case: The benchmark case.
            project: The per-case fixture project directory.

        Returns:
            One finding dict per forbidden path.
        """
        findings: list[dict[str, Any]] = []
        for relative in case.forbidden_paths:
            target = project / relative
            setup_content = case.setup.files.get(relative)
            if setup_content is not None:
                untouched = target.is_file() and target.read_bytes() == setup_content.encode("utf-8")
                detail = "byte-identical to setup content" if untouched else "content differs from the setup bytes"
            else:
                untouched = not target.exists()
                detail = "does not exist" if untouched else "exists but was not part of the setup"
            findings.append({"path": relative, "untouched": untouched, "detail": detail})
        return findings

    def _unapproved_mutations(self, execution: _CaseExecution) -> list[str]:
        """Find mutated files lacking a matching approved mutation record.

        Args:
            execution: The raw execution evidence.

        Returns:
            Repository-relative paths of unapproved mutations.
        """
        approved_targets: set[Path] = set()
        for approval in execution.approvals:
            if not approval.get("approved") or approval.get("tool") not in MUTATION_TOOL_NAMES:
                continue
            parameters = approval.get("parameters") or {}
            target_value = parameters.get("path") or parameters.get("file_path")
            if not target_value:
                continue
            target = Path(str(target_value)).expanduser()
            if not target.is_absolute():
                target = execution.project / target
            approved_targets.add(Path(os.path.realpath(target)))
        unapproved: list[str] = []
        for changed in execution.changed_files:
            if changed in approved_targets:
                continue
            unapproved.append(_relative_to(changed, execution.project))
        return unapproved

    def _case_cost(self, execution: _CaseExecution) -> MetricValue:
        """Estimate one case's cost from its usage totals.

        Args:
            execution: The raw execution evidence.

        Returns:
            An available MetricValue when the model is priced, otherwise an
            explicitly unavailable one (never a fabricated zero).
        """
        if not execution.totals.get("total_tokens"):
            return MetricValue.unavailable("no usage recorded")
        model = str(self.provider_identity.get("model") or "unknown")
        cost = PriceTable.estimate_cost(
            model,
            {
                "prompt_tokens": execution.totals.get("prompt_tokens", 0),
                "completion_tokens": execution.totals.get("completion_tokens", 0),
            },
        )
        if cost is None:
            return MetricValue.unavailable(f"no price entry for {model}")
        return MetricValue.available(cost)

    def _case_record(
        self,
        case: AgentTaskCase,
        execution: _CaseExecution | None,
        error: str | None,
        timestamp: str,
    ) -> AgentCaseRunRecord:
        """Score one case and build its artifact record.

        Args:
            case: The benchmark case.
            execution: Raw execution evidence, or ``None`` when the case
                could not execute.
            error: Error message when the case errored, else ``None``.
            timestamp: ISO-8601 execution timestamp.

        Returns:
            The scored AgentCaseRunRecord.
        """
        if execution is None:
            return AgentCaseRunRecord(
                case_id=case.id,
                run_id=self.run_id,
                timestamp=timestamp,
                cohort=case.cohort,
                split=case.split,
                status="error",
                error=error or "case failed",
                iterations=0,
                max_iterations=case.max_iterations,
                tokens={},
                cost=MetricValue.unavailable("case did not execute"),
                judgments=[CaseJudgment("checks", False, reason=error or "case failed")],
                checks=[],
                forbidden_paths=[
                    {"path": path, "untouched": False, "detail": "case did not execute"}
                    for path in case.forbidden_paths
                ],
                plan={"required": case.plan_required, "submitted": False, "actions": 0, "actions_completed": 0},
                approvals={"mutations": 0, "approved_records": 0, "unapproved": []},
                changed_files=[],
                transcript={
                    "path": f"{case.id}/project/.ctxai/runs/eval-{case.id}.jsonl",
                    "run_id": f"eval-{case.id}",
                    "events": 0,
                },
            )

        findings = self._forbidden_findings(case, execution.project)
        unapproved = self._unapproved_mutations(execution)
        approved_records = len(
            [
                approval
                for approval in execution.approvals
                if approval.get("approved") and approval.get("tool") in MUTATION_TOOL_NAMES
            ]
        )

        judgments = [
            judge_checks(execution.check_results),
            judge_forbidden_paths(findings),
            judge_budget(
                succeeded=execution.succeeded,
                iterations=execution.iterations,
                max_iterations=case.max_iterations,
            ),
            judge_plan(case.plan_required, execution.plan if case.plan_required else None),
            judge_approvals(unapproved_mutations=unapproved),
        ]
        status = "passed" if case_is_passed(judgments) else "failed"

        changed_relative = sorted(_relative_to(path, execution.project) for path in execution.changed_files)
        return AgentCaseRunRecord(
            case_id=case.id,
            run_id=self.run_id,
            timestamp=timestamp,
            cohort=case.cohort,
            split=case.split,
            status=status,
            error=None,
            iterations=execution.iterations,
            max_iterations=case.max_iterations,
            tokens={
                "prompt_tokens": execution.totals.get("prompt_tokens", 0),
                "completion_tokens": execution.totals.get("completion_tokens", 0),
                "total_tokens": execution.totals.get("total_tokens", 0),
                "calls": execution.totals.get("calls", 0),
            },
            cost=self._case_cost(execution),
            judgments=judgments,
            checks=execution.check_results,
            forbidden_paths=findings,
            plan=execution.plan or {},
            approvals={
                "mutations": len(execution.changed_files),
                "approved_records": approved_records,
                "unapproved": unapproved,
            },
            changed_files=changed_relative,
            transcript={
                "path": f"{case.id}/project/.ctxai/runs/eval-{case.id}.jsonl",
                "run_id": execution.case_run_id,
                "events": execution.transcript_events,
            },
        )

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def run(self) -> AgentEvalArtifact:
        """Execute the whole benchmark and build the evaluation artifact.

        Cases that cannot execute (timeout, unexpected loop error) are
        recorded as case errors; the run is marked ``partial`` and never
        looks like a successful benchmark.

        Returns:
            The completed (or partial) AgentEvalArtifact.
        """
        started = self.clock()
        self._prune_workspaces()
        records: list[AgentCaseRunRecord] = []
        run_errors: list[str] = []
        for case in self.cases:
            timestamp = _utc_now()
            execution: _CaseExecution | None = None
            error: str | None = None
            try:
                project = self._case_project_dir(case)
                execution = self._execute_case(case, project)
            except EvalError as exc:
                error = str(exc)
            if error is not None:
                run_errors.append(f"{case.id}: {error}")
            records.append(self._case_record(case, execution, error, timestamp))
        duration_ms = (self.clock() - started) * 1000.0

        artifact = AgentEvalArtifact(
            schema_version=1,
            kind=EVALUATION_KIND_AGENT,
            run_id=self.run_id,
            created_at=_utc_now(),
            duration_ms=duration_ms,
            status="complete" if not run_errors else "partial",
            benchmark={
                "name": self.benchmark.name,
                "fingerprint": self.benchmark.fingerprint,
                "schema_version": self.benchmark.schema_version,
                "case_count": len(self.cases),
                "selected_cases": [case.id for case in self.cases],
            },
            configuration={
                "fingerprint": self.configuration_fingerprint(),
                "provider": self.provider_identity,
                "runner": self.config.result_affecting_settings(),
            },
            environment=self._environment(),
            runs=records,
            aggregates={
                "overall": aggregate_case_records(records),
                "by_cohort": self._aggregate_by(records, lambda record: record.cohort),
                "by_split": self._aggregate_by(records, lambda record: record.split),
            },
            comparison=None,
            errors=run_errors,
        )
        return artifact

    def _aggregate_by(
        self, records: list[AgentCaseRunRecord], key: Callable[[AgentCaseRunRecord], str]
    ) -> dict[str, Any]:
        """Aggregate case records grouped by a label function.

        Args:
            records: Scored case records.
            key: Label extractor (cohort or split).

        Returns:
            Mapping of label to aggregate metrics block.
        """
        groups: dict[str, list[AgentCaseRunRecord]] = {}
        for record in records:
            groups.setdefault(key(record), []).append(record)
        return {label: aggregate_case_records(group) for label, group in sorted(groups.items())}

    def _prune_workspaces(self) -> None:
        """Prune old run workspaces beyond the retention window.

        The current run's workspace is never pruned. Failures are
        diagnostics and never fail the evaluation.
        """
        workspaces = agent_workspaces_dir(self.project_root)
        if not workspaces.is_dir():
            return
        runs = sorted(
            (path for path in workspaces.iterdir() if path.is_dir() and path.name != self.run_id),
            key=lambda path: (path.stat().st_mtime_ns, path.name),
        )
        excess = len(runs) - (self.config.keep_workspaces - 1)
        for path in runs[: max(0, excess)]:
            shutil.rmtree(path, ignore_errors=True)

    def _environment(self) -> dict[str, Any]:
        """Environment metadata describing how the benchmark ran.

        Returns:
            Dictionary with python/platform/version, the evaluator identity,
            and an honest network-access statement.
        """
        import platform
        import sys

        try:
            from importlib import metadata

            ctxai_version = metadata.version("ctxai")
        except Exception:
            ctxai_version = "unknown"
        network = "none" if self.provider_identity.get("mode") == "mock" else "provider-api (explicit maintainer run)"
        return {
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
            "ctxai_version": ctxai_version,
            "evaluator": "local-agent-loop",
            "network_access": network,
        }


def _relative_to(path: Path, root: Path) -> str:
    """Render a path relative to the root, POSIX-style, when possible.

    Args:
        path: The path to render.
        root: The project root.

    Returns:
        The repository-relative POSIX path, or the original string when the
        path lies outside the root.
    """
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _utc_now() -> str:
    """Current UTC time as ISO-8601.

    Returns:
        Timestamp string.
    """
    return datetime.now(timezone.utc).isoformat()
