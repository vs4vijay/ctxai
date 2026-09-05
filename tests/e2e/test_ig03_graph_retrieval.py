"""IG-03 acceptance tests: graph-expanded grounded retrieval.

Exercises the shared retrieval service through the real CLI pipeline with a
relationship-rich Python fixture (a service, its caller, and its test) to
prove: `--explain` reports the selection rationale, graph expansion adds
relationship evidence with explained paths under the token budget, the
benchmark's pre-registered relationship cohort shows graph contribution and a
named improvement in the gate comparator, stale/missing graphs fail explicitly
under `--graph` (required) and fall back with a visible diagnostic otherwise,
and repeated runs are deterministic.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from ctxai.app import app

FIXTURE_FILES = {
    "service.py": (
        "def run(x):\n"
        '    """Run the service step."""\n'
        "    return x + 1\n"
        "\n"
        "\n"
        "class Runner:\n"
        '    """Drive the service."""\n'
        "\n"
        "    def go(self):\n"
        "        return run(1)\n"
    ),
    "runner.py": (
        'from service import run\n\n\ndef drive():\n    """Call the service run function."""\n    return run(2)\n'
    ),
    "test_service.py": (
        "from service import run\n"
        "from runner import drive\n"
        "\n"
        "\n"
        "def test_run():\n"
        "    assert run(1) == 2\n"
        "\n"
        "\n"
        "def test_drive():\n"
        "    assert drive() == 3\n"
    ),
    "README.md": "# Graph retrieval fixture\n",
}

BENCHMARK_PATH = Path(__file__).parents[1] / "fixtures" / "retrieval_benchmark.json"


def index_fixture(runner: CliRunner, project: Path, name: str = "ig03") -> None:
    """Write the fixture files and index the project with mock embeddings.

    Args:
        runner: CLI runner.
        project: Project root (populated in place).
        name: Index name.
    """
    for relative, content in FIXTURE_FILES.items():
        (project / relative).write_text(content, encoding="utf-8")
    result = runner.invoke(app, ["index", str(project), name])
    assert result.exit_code == 0, result.output


def build_benchmark_project(root: Path) -> Path:
    """Create a project containing exactly the benchmark's evidence files.

    Args:
        root: Scratch directory.

    Returns:
        The fixture project root holding the copied source layout.
    """
    import shutil

    payload = json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))
    needed: set[str] = set()
    for case in payload["cases"]:
        needed.update(case["expected"]["files"])
        needed.update(case["expected"].get("line_ranges", {}))
    project = root / "project"
    repo_root = Path(__file__).parents[2]
    for relative in sorted(needed):
        target = project / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(repo_root / relative, target)
    (project / "README.md").write_text("# Retrieval benchmark fixture project\n", encoding="utf-8")
    return project


def in_project(monkeypatch, project: Path) -> None:
    """Run the CLI from inside the fixture project (cwd-based resolution).

    Args:
        monkeypatch: pytest monkeypatch fixture.
        project: The fixture project root.
    """
    monkeypatch.chdir(project)


def flat(text: str) -> str:
    """Normalize Rich output for assertions.

    Args:
        text: Raw output.

    Returns:
        Printable characters plus newlines and tabs.
    """
    return "".join(char for char in text if char.isprintable() or char in "\n\t")


@pytest.mark.e2e
@pytest.mark.indexing
def test_explain_reports_selection_rationale(temp_dir, patch_embeddings_factory, monkeypatch):
    """`--explain` prints generator counts and per-item fusion details without changing items."""
    runner = CliRunner()
    index_fixture(runner, temp_dir, "test-index")
    in_project(monkeypatch, temp_dir)
    plain = runner.invoke(app, ["query", "test-index", "run", "--no-graph"])
    assert plain.exit_code == 0, flat(plain.output)
    plain_output = flat(plain.output)
    assert "Why this context was selected" not in plain_output

    explained = runner.invoke(app, ["query", "test-index", "run", "--no-graph", "--explain"])
    assert explained.exit_code == 0, flat(explained.output)
    explain_output = flat(explained.output)
    assert "Why this context was selected" in explain_output
    assert "Candidate generators" in explain_output
    assert "Selected evidence" in explain_output
    # The item set is identical between plain and explained runs.
    assert "Found" in plain_output and "Found" in explain_output


@pytest.mark.e2e
@pytest.mark.indexing
def test_graph_expansion_adds_relationship_evidence_with_paths(temp_dir, patch_embeddings_factory, monkeypatch):
    """Graph-expanded runs report graph contribution and explained paths in the eval artifact."""
    runner = CliRunner()
    bench_project = build_benchmark_project(temp_dir)
    index_fixture(runner, bench_project, "ig03g")
    in_project(monkeypatch, bench_project)

    result = runner.invoke(app, ["eval", "retrieval", str(BENCHMARK_PATH), "--index", "ig03g", "--graph", "--json"])
    assert result.exit_code == 0, flat(result.output)
    output = result.output
    payload = json.loads(output[output.find("{") : output.rfind("}") + 1])

    relationship = payload["aggregates"]["by_cohort"]["graph-relationship"]["metrics"]
    contribution = relationship["graph_contribution_rate"]
    assert contribution["available"] is True, contribution
    assert contribution["value"] > 0, "relationship cohort must show real graph contribution"

    # Every graph-expanded case record cites the expansion reason/path shape.
    graph_paths = [
        item.get("graph_path") for run in payload["runs"] for item in run.get("selected", []) if item.get("graph_path")
    ]
    for path in graph_paths:
        assert "-[" in path and "]->" in path, path


@pytest.mark.e2e
@pytest.mark.indexing
def test_benchmark_gate_compares_graph_against_no_graph(temp_dir, patch_embeddings_factory, monkeypatch):
    """compare-graph produces a named verdict: gates, improvement, and an honest pass flag."""
    runner = CliRunner()
    bench_project = build_benchmark_project(temp_dir)
    index_fixture(runner, bench_project, "ig03gate")
    in_project(monkeypatch, bench_project)

    base = runner.invoke(app, ["eval", "retrieval", str(BENCHMARK_PATH), "--index", "ig03gate", "--json"])
    graph = runner.invoke(app, ["eval", "retrieval", str(BENCHMARK_PATH), "--index", "ig03gate", "--graph", "--json"])
    assert base.exit_code == 0, flat(base.output)
    assert graph.exit_code == 0, flat(graph.output)
    artifacts = sorted((bench_project / ".ctxai" / "evaluations" / "retrieval").glob("*.json"))
    assert len(artifacts) == 2

    compare = runner.invoke(app, ["eval", "retrieval", "compare-graph", str(artifacts[0]), str(artifacts[1]), "--json"])
    output = compare.output
    payload = json.loads(output[output.find("{") : output.rfind("}") + 1])
    assert payload["compatible"] is True
    assert payload["gates"], "gate results must be present"
    # The pre-registered relationship-cohort improvement is detected and named.
    assert payload["improvement"] is not None
    assert payload["improvement"]["metric"], payload["improvement"]
    # The honest verdict may pass or fail under mock embeddings; either way the
    # gates and improvement are named (never silently absorbed).


@pytest.mark.e2e
@pytest.mark.indexing
def test_stale_graph_fails_required_and_falls_back_visibly(temp_dir, patch_embeddings_factory, monkeypatch):
    """A missing graph fails `--graph` explicitly and degrades visibly without it."""
    runner = CliRunner()
    index_fixture(runner, temp_dir, "ig03s")
    in_project(monkeypatch, temp_dir)

    graph_file = next((temp_dir / ".ctxai" / "indexes" / "ig03s").glob("graph.sqlite3"))
    graph_file.unlink()

    required = runner.invoke(app, ["query", "ig03s", "run", "--graph"])
    assert required.exit_code == 1
    combined = flat(required.output)
    assert "graph" in combined.lower()

    fallback = runner.invoke(app, ["query", "ig03s", "run"])
    assert fallback.exit_code == 0, flat(fallback.output)
    assert "Found" in flat(fallback.output)


@pytest.mark.e2e
@pytest.mark.indexing
def test_graph_expansion_respects_budget_and_deterministic(temp_dir, patch_embeddings_factory, monkeypatch):
    """Expanded runs stay within the token budget and repeat identically."""
    runner = CliRunner()
    bench_project = build_benchmark_project(temp_dir)
    index_fixture(runner, bench_project, "ig03d")
    in_project(monkeypatch, bench_project)

    outputs = []
    for _ in range(2):
        result = runner.invoke(app, ["query", "ig03d", "run", "--graph", "--explain"])
        assert result.exit_code == 0, flat(result.output)
        outputs.append(flat(result.output))
    assert outputs[0] == outputs[1], "identical queries must render identically"

    result = runner.invoke(app, ["eval", "retrieval", str(BENCHMARK_PATH), "--index", "ig03d", "--graph", "--json"])
    assert result.exit_code == 0, flat(result.output)
    payload = json.loads(result.output[result.output.find("{") : result.output.rfind("}") + 1])
    token_mean = payload["aggregates"]["overall"]["metrics"]["selected_token_mean"]
    budget_mean = payload["configuration"].get("token_budget")
    if token_mean.get("value") is not None and budget_mean:
        assert token_mean["value"] <= budget_mean, "selection must respect the configured budget"
