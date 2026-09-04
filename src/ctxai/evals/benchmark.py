"""Versioned retrieval benchmark schema and validation (RE-01).

A benchmark is a checked-in, immutable JSON document whose cases carry a
stable ID, natural-language query, tags/cohort, expected evidence (files,
symbols, optional line ranges), relevance grades, and a train/dev/test split.
Retrieved results are never embedded in the benchmark; they are produced at
runtime by the production retrieval service.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

from .common import content_fingerprint

BENCHMARK_SCHEMA_VERSION = 1

# Relevance grades: 0 not relevant, 1 marginal, 2 relevant, 3 core answer.
MIN_RELEVANCE_GRADE = 0
MAX_RELEVANCE_GRADE = 3
# Expected files without an explicit grade default to "relevant".
DEFAULT_RELEVANCE_GRADE = 2

VALID_SPLITS = ("train", "dev", "test")

# Hard bounds: out-of-bounds benchmarks are rejected before any work begins.
MAX_BENCHMARK_BYTES = 1_000_000
MAX_CASES = 1000
MAX_QUERY_CHARS = 2000
MAX_PATH_CHARS = 512
MAX_TAGS = 32


class BenchmarkValidationError(ValueError):
    """Raised when a benchmark document fails validation.

    Attributes:
        errors: Every validation problem found (not just the first).
    """

    def __init__(self, errors: list[str]) -> None:
        """Create the error from the collected validation problems.

        Args:
            errors: Human-readable validation problem strings.
        """
        self.errors = errors
        super().__init__("Invalid benchmark: " + "; ".join(errors))


@dataclass(frozen=True)
class ExpectedEvidence:
    """Expected evidence for one benchmark case.

    Attributes:
        files: Repository-relative file paths that should be retrieved.
        symbols: Symbol names expected in the retrieved evidence.
        line_ranges: Optional repository-relative path -> inclusive
            ``[start_line, end_line]`` evidence ranges.
    """

    files: list[str] = field(default_factory=list)
    symbols: list[str] = field(default_factory=list)
    line_ranges: dict[str, list[int]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to the JSON representation.

        Returns:
            Dictionary with ``files``, ``symbols``, and ``line_ranges``.
        """
        return {
            "files": list(self.files),
            "symbols": list(self.symbols),
            "line_ranges": {path: list(value) for path, value in self.line_ranges.items()},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExpectedEvidence:
        """Rebuild from the JSON representation.

        Args:
            data: Dictionary produced by :meth:`to_dict`.

        Returns:
            The reconstructed ExpectedEvidence.
        """
        return cls(
            files=list(data.get("files", [])),
            symbols=list(data.get("symbols", [])),
            line_ranges={path: list(value) for path, value in (data.get("line_ranges") or {}).items()},
        )


@dataclass(frozen=True)
class BenchmarkCase:
    """One benchmark question with its expectations and labels.

    Attributes:
        id: Stable unique case identifier.
        query: Natural-language query executed at runtime.
        cohort: Cohort label used for gate aggregation (e.g. ``agent``).
        split: ``train``, ``dev``, or ``test``.
        tags: Free-form labels for filtering and reporting.
        expected: Expected evidence for the case.
        relevance: Explicit per-path relevance grades (0-3); expected files
            without an explicit grade default to ``DEFAULT_RELEVANCE_GRADE``.
    """

    id: str
    query: str
    cohort: str
    split: str
    tags: list[str] = field(default_factory=list)
    expected: ExpectedEvidence = field(default_factory=ExpectedEvidence)
    relevance: dict[str, int] = field(default_factory=dict)

    def effective_relevance(self) -> dict[str, int]:
        """Relevance grades used for scoring, with defaults applied.

        Expected files without an explicit grade receive
        ``DEFAULT_RELEVANCE_GRADE``; explicit grades (for expected or
        additional marginally relevant paths) override the default.

        Returns:
            Mapping of repository-relative path to grade.
        """
        effective = {path: DEFAULT_RELEVANCE_GRADE for path in self.expected.files}
        effective.update(self.relevance)
        return effective

    def relevant_paths(self) -> set[str]:
        """Paths counted as relevant (grade >= 1) for rank-based metrics.

        Returns:
            Set of repository-relative paths with grade >= 1.
        """
        return {path for path, grade in self.effective_relevance().items() if grade >= 1}

    def to_dict(self) -> dict[str, Any]:
        """Convert to the JSON representation.

        Returns:
            Dictionary matching the benchmark schema for one case.
        """
        return {
            "id": self.id,
            "query": self.query,
            "tags": list(self.tags),
            "cohort": self.cohort,
            "split": self.split,
            "expected": self.expected.to_dict(),
            "relevance": dict(self.relevance),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BenchmarkCase:
        """Rebuild from the JSON representation.

        Args:
            data: Dictionary produced by :meth:`to_dict`.

        Returns:
            The reconstructed BenchmarkCase.
        """
        return cls(
            id=data["id"],
            query=data["query"],
            cohort=data["cohort"],
            split=data["split"],
            tags=list(data.get("tags", [])),
            expected=ExpectedEvidence.from_dict(data.get("expected") or {}),
            relevance={path: int(grade) for path, grade in (data.get("relevance") or {}).items()},
        )


@dataclass(frozen=True)
class RetrievalBenchmark:
    """A versioned, immutable retrieval benchmark document.

    Attributes:
        schema_version: Benchmark schema version (currently 1).
        name: Stable benchmark name used in artifact identity.
        cases: Validated benchmark cases.
        description: Optional human-readable description.
    """

    schema_version: int
    name: str
    cases: list[BenchmarkCase]
    description: str = ""

    @property
    def fingerprint(self) -> str:
        """Content-derived fingerprint of the full benchmark document.

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
    def from_dict(cls, data: dict[str, Any]) -> RetrievalBenchmark:
        """Rebuild from the JSON representation (validation not re-run).

        Args:
            data: Dictionary produced by :meth:`to_dict`.

        Returns:
            The reconstructed RetrievalBenchmark.
        """
        return cls(
            schema_version=int(data["schema_version"]),
            name=data["name"],
            description=data.get("description", ""),
            cases=[BenchmarkCase.from_dict(case) for case in data["cases"]],
        )


def _validate_case_path(path: str, errors: list[str], label: str) -> None:
    """Validate one repository-relative evidence path.

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


def validate_benchmark_payload(payload: Any) -> list[str]:
    """Validate a benchmark document and return every problem found.

    Checks schema version, name, case count bounds, unique IDs, non-empty
    queries, valid splits/cohorts/tags, repository-relative evidence paths,
    line-range sanity, and relevance grade bounds.

    Args:
        payload: Parsed benchmark JSON (any type).

    Returns:
        List of human-readable validation errors; empty when valid.
    """
    errors: list[str] = []
    if not isinstance(payload, dict):
        return ["benchmark must be a JSON object"]
    if payload.get("schema_version") != BENCHMARK_SCHEMA_VERSION:
        errors.append(f"schema_version must be {BENCHMARK_SCHEMA_VERSION}")
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
        elif case_id in seen_ids:
            errors.append(f"{label}: duplicate case id '{case_id}'")
        else:
            seen_ids.add(case_id)
            label = f"case '{case_id}'"

        query = case.get("query")
        if not isinstance(query, str) or not query.strip():
            errors.append(f"{label}: query must be a non-empty string")
        elif len(query) > MAX_QUERY_CHARS:
            errors.append(f"{label}: query exceeds {MAX_QUERY_CHARS} characters")

        cohort = case.get("cohort")
        if not isinstance(cohort, str) or not cohort.strip():
            errors.append(f"{label}: cohort must be a non-empty string")

        split = case.get("split")
        if split not in VALID_SPLITS:
            errors.append(f"{label}: split must be one of {', '.join(VALID_SPLITS)}")

        tags = case.get("tags", [])
        if not isinstance(tags, list) or len(tags) > MAX_TAGS or any(not isinstance(tag, str) for tag in tags):
            errors.append(f"{label}: tags must be a list of at most {MAX_TAGS} strings")

        expected = case.get("expected")
        if not isinstance(expected, dict):
            errors.append(f"{label}: expected must be an object")
            continue
        files = expected.get("files")
        if not isinstance(files, list) or not files or any(not isinstance(item, str) for item in files):
            errors.append(f"{label}: expected.files must be a non-empty list of paths")
        else:
            for path in files:
                _validate_case_path(path, errors, f"{label} expected.files")
        symbols = expected.get("symbols", [])
        if not isinstance(symbols, list) or any(not isinstance(item, str) for item in symbols):
            errors.append(f"{label}: expected.symbols must be a list of strings")
        line_ranges = expected.get("line_ranges", {})
        if not isinstance(line_ranges, dict):
            errors.append(f"{label}: expected.line_ranges must be an object")
        else:
            for path, value in line_ranges.items():
                _validate_case_path(path, errors, f"{label} expected.line_ranges")
                if (
                    not isinstance(value, list)
                    or len(value) != 2
                    or any(isinstance(bound, bool) or not isinstance(bound, int) for bound in value)
                ):
                    errors.append(f"{label}: line_ranges['{path}'] must be [start_line, end_line] integers")
                    continue
                start, end = value
                if start < 1:
                    errors.append(f"{label}: line_ranges['{path}'] start must be >= 1")
                if end < start:
                    errors.append(f"{label}: line_ranges['{path}'] end must be >= start")

        relevance = case.get("relevance", {})
        if not isinstance(relevance, dict):
            errors.append(f"{label}: relevance must be an object")
        else:
            for path, grade in relevance.items():
                _validate_case_path(path, errors, f"{label} relevance")
                if isinstance(grade, bool) or not isinstance(grade, int):
                    errors.append(f"{label}: relevance['{path}'] must be an integer grade")
                elif not MIN_RELEVANCE_GRADE <= grade <= MAX_RELEVANCE_GRADE:
                    errors.append(
                        f"{label}: relevance['{path}'] must be between {MIN_RELEVANCE_GRADE} and {MAX_RELEVANCE_GRADE}"
                    )
                elif isinstance(files, list) and path in files and grade == MIN_RELEVANCE_GRADE:
                    errors.append(f"{label}: expected file '{path}' cannot have relevance grade 0")
    return errors


def benchmark_from_payload(payload: Any) -> RetrievalBenchmark:
    """Validate a benchmark payload and build the immutable benchmark object.

    Args:
        payload: Parsed benchmark JSON (any type).

    Returns:
        The validated RetrievalBenchmark.

    Raises:
        BenchmarkValidationError: If any validation problem exists.
    """
    errors = validate_benchmark_payload(payload)
    if errors:
        raise BenchmarkValidationError(errors)
    return RetrievalBenchmark.from_dict(payload)


def load_benchmark(path: Path) -> RetrievalBenchmark:
    """Load, size-check, parse, and validate a benchmark file.

    The benchmark file is never modified; it is only read. Oversized files
    are rejected before parsing.

    Args:
        path: Path to the benchmark JSON document.

    Returns:
        The validated RetrievalBenchmark.

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
    return benchmark_from_payload(payload)
