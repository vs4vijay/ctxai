"""Versioned agent task benchmark schema and validation (HH-09).

A task benchmark is a checked-in, immutable JSON document whose cases carry a
stable ID, a natural-language instruction, setup files written into a fresh
per-case fixture project, ``expected_checks`` commands that must exit 0 at
run time, forbidden paths that must remain untouched, a planning requirement,
an iteration budget, and cohort/split labels. Scoring evidence is produced at
runtime by the real agent loop — never embedded in the benchmark.

Cases may additionally carry a ``mock_script``: the exact
:class:`~tests.mocks.mock_llm.MockLLMProvider` response sequence that makes a
mock-provider run fully deterministic. It is benchmark identity (it is
fingerprinted) but it is *provider scripting*, not expectations: configured
providers ignore it and attempt the task for real.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

from .common import content_fingerprint

AGENT_BENCHMARK_SCHEMA_VERSION = 1

VALID_SPLITS = ("train", "dev", "test")

# Hard bounds: out-of-bounds benchmarks are rejected before any work begins.
MAX_BENCHMARK_BYTES = 1_000_000
MAX_CASES = 1000
MAX_INSTRUCTION_CHARS = 2000
MAX_PATH_CHARS = 512
MAX_TAGS = 32
MAX_SETUP_FILES = 100
MAX_SETUP_FILE_CHARS = 100_000
MAX_CHECKS_PER_CASE = 20
MAX_EXPECT_OUTPUT_CHARS = 1000
MAX_MOCK_SCRIPT_ENTRIES = 100

MIN_MAX_ITERATIONS = 1
MAX_MAX_ITERATIONS = 100
DEFAULT_MAX_ITERATIONS = 10

# Case ids become transcript run ids (``eval-<case_id>``) and artifact keys,
# so they are restricted to safe single-path-component characters.
CASE_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")


class BenchmarkValidationError(ValueError):
    """Raised when an agent task benchmark document fails validation.

    Attributes:
        errors: Every validation problem found (not just the first).
    """

    def __init__(self, errors: list[str]) -> None:
        """Create the error from the collected validation problems.

        Args:
            errors: Human-readable validation problem strings.
        """
        self.errors = errors
        super().__init__("Invalid agent benchmark: " + "; ".join(errors))


@dataclass(frozen=True)
class ExpectedCheck:
    """One post-run verification command for a benchmark case.

    Attributes:
        command: Single (no shell operators) command executed through the
            agent's bash policy after the run; must exit 0.
        description: Human-readable description of what the check proves.
        expect_output: Optional substring that must appear in stdout.
    """

    command: str
    description: str
    expect_output: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to the JSON representation.

        Returns:
            Dictionary matching the benchmark schema for one check.
        """
        payload: dict[str, Any] = {"command": self.command, "description": self.description}
        if self.expect_output is not None:
            payload["expect_output"] = self.expect_output
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExpectedCheck:
        """Rebuild from the JSON representation.

        Args:
            data: Dictionary produced by :meth:`to_dict`.

        Returns:
            The reconstructed ExpectedCheck.
        """
        return cls(
            command=str(data["command"]),
            description=str(data.get("description", "")),
            expect_output=data.get("expect_output"),
        )


@dataclass(frozen=True)
class CaseSetup:
    """Fixture files written into a fresh per-case project directory.

    Attributes:
        files: Repository-relative POSIX path -> full file content.
    """

    files: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to the JSON representation.

        Returns:
            Dictionary with the ``files`` mapping.
        """
        return {"files": dict(self.files)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CaseSetup:
        """Rebuild from the JSON representation.

        Args:
            data: Dictionary produced by :meth:`to_dict`.

        Returns:
            The reconstructed CaseSetup.
        """
        files = data.get("files") or {}
        return cls(files={str(path): str(content) for path, content in files.items()})


@dataclass(frozen=True)
class AgentTaskCase:
    """One agent task with its fixture, expectations, and labels.

    Attributes:
        id: Stable unique case identifier (safe path component).
        instruction: Natural-language instruction given to the agent.
        cohort: Cohort label used for gate aggregation.
        split: ``train``, ``dev``, or ``test``.
        setup: Fixture files written into the per-case project directory.
        expected_checks: Commands that must exit 0 (and optionally echo an
            expected marker) after the run.
        forbidden_paths: Repository-relative paths that must not be created
            and, when present in ``setup``, must stay byte-identical.
        plan_required: When True the run must go through ``submit_plan``.
        max_iterations: Iteration budget for the agent loop.
        tags: Free-form labels for filtering and reporting.
        mock_script: Optional provider scripting (mock-provider runs only;
            ignored by configured providers and fingerprinted as identity).
    """

    id: str
    instruction: str
    cohort: str
    split: str
    setup: CaseSetup = field(default_factory=CaseSetup)
    expected_checks: list[ExpectedCheck] = field(default_factory=list)
    forbidden_paths: list[str] = field(default_factory=list)
    plan_required: bool = False
    max_iterations: int = DEFAULT_MAX_ITERATIONS
    tags: list[str] = field(default_factory=list)
    mock_script: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to the JSON representation.

        Returns:
            Dictionary matching the benchmark schema for one case.
        """
        payload: dict[str, Any] = {
            "id": self.id,
            "instruction": self.instruction,
            "tags": list(self.tags),
            "cohort": self.cohort,
            "split": self.split,
            "setup": self.setup.to_dict(),
            "expected_checks": [check.to_dict() for check in self.expected_checks],
            "forbidden_paths": list(self.forbidden_paths),
            "plan_required": self.plan_required,
            "max_iterations": self.max_iterations,
        }
        if self.mock_script:
            payload["mock_script"] = [dict(entry) for entry in self.mock_script]
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AgentTaskCase:
        """Rebuild from the JSON representation.

        Args:
            data: Dictionary produced by :meth:`to_dict`.

        Returns:
            The reconstructed AgentTaskCase.
        """
        return cls(
            id=data["id"],
            instruction=data["instruction"],
            cohort=data["cohort"],
            split=data["split"],
            setup=CaseSetup.from_dict(data.get("setup") or {}),
            expected_checks=[ExpectedCheck.from_dict(check) for check in data.get("expected_checks", [])],
            forbidden_paths=list(data.get("forbidden_paths", [])),
            plan_required=bool(data.get("plan_required", False)),
            max_iterations=int(data.get("max_iterations", DEFAULT_MAX_ITERATIONS)),
            tags=list(data.get("tags", [])),
            mock_script=[dict(entry) for entry in data.get("mock_script", [])],
        )


@dataclass(frozen=True)
class AgentTaskBenchmark:
    """A versioned, immutable agent task benchmark document.

    Attributes:
        schema_version: Benchmark schema version (currently 1).
        name: Stable benchmark name used in artifact identity.
        cases: Validated benchmark cases.
        description: Optional human-readable description.
    """

    schema_version: int
    name: str
    cases: list[AgentTaskCase]
    description: str = ""

    @property
    def fingerprint(self) -> str:
        """Content-derived fingerprint of the full benchmark document.

        Includes ``mock_script`` entries: they are part of the benchmark
        identity because the deterministic mock-provider CI path depends on
        them byte-for-byte.

        Returns:
            Hex digest over the canonical JSON of the whole document.
        """
        return content_fingerprint(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        """Convert to the JSON representation.

        Returns:
            Dictionary matching the benchmark file schema.
        """
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "description": self.description,
            "cases": [case.to_dict() for case in self.cases],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AgentTaskBenchmark:
        """Rebuild from the JSON representation (validation not re-run).

        Args:
            data: Dictionary produced by :meth:`to_dict`.

        Returns:
            The reconstructed AgentTaskBenchmark.
        """
        return cls(
            schema_version=int(data["schema_version"]),
            name=data["name"],
            description=data.get("description", ""),
            cases=[AgentTaskCase.from_dict(case) for case in data["cases"]],
        )

    def select_cases(self, case_ids: list[str] | None) -> list[AgentTaskCase]:
        """Return the cases selected for a run, preserving document order.

        Args:
            case_ids: Optional explicit case-id subset; ``None`` selects all.

        Returns:
            The selected cases in benchmark order.

        Raises:
            BenchmarkValidationError: When any requested id is unknown.
        """
        if not case_ids:
            return list(self.cases)
        known = {case.id for case in self.cases}
        unknown = [case_id for case_id in case_ids if case_id not in known]
        if unknown:
            raise BenchmarkValidationError([f"unknown case id(s): {', '.join(sorted(unknown))}"])
        wanted = set(case_ids)
        return [case for case in self.cases if case.id in wanted]


def _validate_case_path(path: str, errors: list[str], label: str) -> None:
    """Validate one repository-relative fixture path.

    Args:
        path: Candidate path string.
        errors: Error list to append to.
        label: Description of the field being validated (for messages).
    """
    if not path or len(path) > MAX_PATH_CHARS:
        errors.append(f"{label}: path must be a non-empty string of at most {MAX_PATH_CHARS} characters")
        return
    if "\\" in path:
        errors.append(f"{label}: '{path}' must use POSIX '/' separators")
        return
    candidate = PurePosixPath(path)
    if candidate.is_absolute() or path.startswith("/") or path.startswith("~"):
        errors.append(f"{label}: '{path}' must be repository-relative, not absolute")
        return
    if ".." in candidate.parts:
        errors.append(f"{label}: '{path}' must not contain '..' segments")
        return
    if not candidate.name:
        errors.append(f"{label}: '{path}' must name a file")


def validate_agent_benchmark_payload(payload: Any) -> list[str]:
    """Validate an agent task benchmark document and return every problem.

    Checks schema version, name, case count bounds, unique safe IDs,
    non-empty instructions, valid splits/cohorts/tags, repository-relative
    setup/forbidden paths, check shapes, iteration budget bounds, and
    ``mock_script`` shape.

    Args:
        payload: Parsed benchmark JSON (any type).

    Returns:
        List of human-readable validation errors; empty when valid.
    """
    errors: list[str] = []
    if not isinstance(payload, dict):
        return ["benchmark must be a JSON object"]
    if payload.get("schema_version") != AGENT_BENCHMARK_SCHEMA_VERSION:
        errors.append(f"schema_version must be {AGENT_BENCHMARK_SCHEMA_VERSION}")
    name = payload.get("name")
    if not isinstance(name, str) or not name.strip():
        errors.append("name must be a non-empty string")
    cases = payload.get("cases")
    if not isinstance(cases, list) or not cases:
        errors.append("cases must be a non-empty list")
        return errors
    if len(cases) > MAX_CASES:
        errors.append(f"cases exceeds the maximum of {MAX_CASES}")
        return errors

    seen_ids: set[str] = set()
    for position, case in enumerate(cases):
        label = f"case #{position}"
        if not isinstance(case, dict):
            errors.append(f"{label}: must be an object")
            continue
        case_id = case.get("id")
        if not isinstance(case_id, str) or not case_id.strip():
            errors.append(f"{label}: id must be a non-empty string")
        elif not CASE_ID_PATTERN.fullmatch(case_id):
            errors.append(
                f"{label}: id '{case_id}' must match {CASE_ID_PATTERN.pattern} (it becomes the transcript run id)"
            )
        elif case_id in seen_ids:
            errors.append(f"{label}: duplicate case id '{case_id}'")
        else:
            seen_ids.add(case_id)
            label = f"case '{case_id}'"

        instruction = case.get("instruction")
        if not isinstance(instruction, str) or not instruction.strip():
            errors.append(f"{label}: instruction must be a non-empty string")
        elif len(instruction) > MAX_INSTRUCTION_CHARS:
            errors.append(f"{label}: instruction exceeds {MAX_INSTRUCTION_CHARS} characters")

        cohort = case.get("cohort")
        if not isinstance(cohort, str) or not cohort.strip():
            errors.append(f"{label}: cohort must be a non-empty string")

        split = case.get("split")
        if split not in VALID_SPLITS:
            errors.append(f"{label}: split must be one of {', '.join(VALID_SPLITS)}")

        tags = case.get("tags", [])
        if not isinstance(tags, list) or len(tags) > MAX_TAGS or any(not isinstance(tag, str) for tag in tags):
            errors.append(f"{label}: tags must be a list of at most {MAX_TAGS} strings")

        max_iterations = case.get("max_iterations", DEFAULT_MAX_ITERATIONS)
        if isinstance(max_iterations, bool) or not isinstance(max_iterations, int):
            errors.append(f"{label}: max_iterations must be an integer")
        elif not MIN_MAX_ITERATIONS <= max_iterations <= MAX_MAX_ITERATIONS:
            errors.append(f"{label}: max_iterations must be between {MIN_MAX_ITERATIONS} and {MAX_MAX_ITERATIONS}")

        plan_required = case.get("plan_required", False)
        if not isinstance(plan_required, bool):
            errors.append(f"{label}: plan_required must be a boolean")

        setup = case.get("setup", {})
        if not isinstance(setup, dict) or "files" not in setup or not isinstance(setup.get("files"), dict):
            errors.append(f"{label}: setup must be an object with a 'files' mapping of path -> content")
        else:
            setup_files: dict[str, Any] = setup["files"]
            if len(setup_files) > MAX_SETUP_FILES:
                errors.append(f"{label}: setup.files exceeds the maximum of {MAX_SETUP_FILES} files")
            for path, content in setup_files.items():
                _validate_case_path(path, errors, f"{label} setup.files")
                if not isinstance(content, str):
                    errors.append(f"{label} setup.files: '{path}' content must be a string")
                elif len(content) > MAX_SETUP_FILE_CHARS:
                    errors.append(f"{label} setup.files: '{path}' exceeds {MAX_SETUP_FILE_CHARS} characters")

        forbidden = case.get("forbidden_paths", [])
        if not isinstance(forbidden, list) or any(not isinstance(path, str) for path in forbidden):
            errors.append(f"{label}: forbidden_paths must be a list of repository-relative paths")
        else:
            for path in forbidden:
                _validate_case_path(path, errors, f"{label} forbidden_paths")

        checks = case.get("expected_checks")
        if not isinstance(checks, list) or not checks:
            errors.append(f"{label}: expected_checks must be a non-empty list of check objects")
        elif len(checks) > MAX_CHECKS_PER_CASE:
            errors.append(f"{label}: expected_checks exceeds the maximum of {MAX_CHECKS_PER_CASE}")
        else:
            for index, check in enumerate(checks):
                check_label = f"{label} expected_checks[{index}]"
                if not isinstance(check, dict):
                    errors.append(f"{check_label}: must be an object")
                    continue
                command = check.get("command")
                if not isinstance(command, str) or not command.strip():
                    errors.append(f"{check_label}: command must be a non-empty string")
                elif len(command) > MAX_PATH_CHARS:
                    errors.append(f"{check_label}: command exceeds {MAX_PATH_CHARS} characters")
                if not isinstance(check.get("description", ""), str):
                    errors.append(f"{check_label}: description must be a string")
                expect_output = check.get("expect_output")
                if expect_output is not None and (
                    not isinstance(expect_output, str) or len(expect_output) > MAX_EXPECT_OUTPUT_CHARS
                ):
                    errors.append(
                        f"{check_label}: expect_output must be a string of at most {MAX_EXPECT_OUTPUT_CHARS} characters"
                    )

        mock_script = case.get("mock_script", [])
        if not isinstance(mock_script, list):
            errors.append(f"{label}: mock_script must be a list of response objects")
        elif len(mock_script) > MAX_MOCK_SCRIPT_ENTRIES:
            errors.append(f"{label}: mock_script exceeds the maximum of {MAX_MOCK_SCRIPT_ENTRIES} responses")
        else:
            for index, entry in enumerate(mock_script):
                if not isinstance(entry, dict):
                    errors.append(f"{label} mock_script[{index}]: must be an object")
    return errors


def agent_benchmark_from_payload(payload: Any) -> AgentTaskBenchmark:
    """Validate an agent benchmark payload and build the immutable object.

    Args:
        payload: Parsed benchmark JSON (any type).

    Returns:
        The validated AgentTaskBenchmark.

    Raises:
        BenchmarkValidationError: If any validation problem exists.
    """
    errors = validate_agent_benchmark_payload(payload)
    if errors:
        raise BenchmarkValidationError(errors)
    return AgentTaskBenchmark.from_dict(payload)


def load_agent_benchmark(path: Path) -> AgentTaskBenchmark:
    """Load, size-check, parse, and validate an agent benchmark file.

    The benchmark file is never modified; it is only read. Oversized files
    are rejected before parsing.

    Args:
        path: Path to the benchmark JSON document.

    Returns:
        The validated AgentTaskBenchmark.

    Raises:
        BenchmarkValidationError: On unreadable, oversized, malformed, or
            invalid benchmark content.
    """
    path = Path(path)
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise BenchmarkValidationError([f"cannot read benchmark file {path}: {exc}"]) from exc
    if size > MAX_BENCHMARK_BYTES:
        raise BenchmarkValidationError([f"benchmark file exceeds the maximum of {MAX_BENCHMARK_BYTES} bytes"])
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise BenchmarkValidationError([f"cannot parse benchmark file {path}: {exc}"]) from exc
    return agent_benchmark_from_payload(payload)
