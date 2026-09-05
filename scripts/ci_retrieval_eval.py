#!/usr/bin/env python
"""Deterministic retrieval-quality CI gate and baseline refresh tool (RE-03).

Runs the checked-in retrieval benchmark against a freshly built fixture
project using the registered deterministic ``mock`` embedding provider (no
network, no credentials, no model downloads), then compares the fresh
artifact against the checked-in baseline through the CLI compare path
(``ctxai eval retrieval compare``) with ``--fail-on-regression`` semantics.

The mock embedding provider is registered under the name ``mock`` and pinned
in the fixture project's ``.ctxai/config.toml``, so the whole pipeline —
traversal, chunking, indexing, and evaluation — runs through the production
code path with byte-stable MD5-seeded vectors. CTXAI_HOME is pointed at the
fixture project's ``.ctxai`` directory so a developer's global configuration
can never influence the gate.

Usage::

    # CI gate (default): run the benchmark, compare against the checked-in
    # baseline, exit 0 (pass) / 1 (regression) / 2 (incompatible).
    uv run python scripts/ci_retrieval_eval.py --artifacts-dir retrieval-eval-artifacts

    # Deliberate baseline refresh (maintainer action): regenerate the
    # checked-in baseline and print the reviewable artifact diff.
    uv run python scripts/ci_retrieval_eval.py \\
        --update-baseline tests/fixtures/retrieval_baseline.json
"""

from __future__ import annotations

import argparse
import difflib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ctxai.commands.eval_command import compare_retrieval_runs, run_retrieval_eval  # noqa: E402
from ctxai.commands.index_command import index_codebase  # noqa: E402
from ctxai.embeddings import EmbeddingsFactory  # noqa: E402
from ctxai.evals.common import atomic_write_json, canonical_json, strip_volatile  # noqa: E402
from tests.mocks.mock_embeddings import MockEmbeddingProvider  # noqa: E402

BENCHMARK_PATH = REPO_ROOT / "tests" / "fixtures" / "retrieval_benchmark.json"
DEFAULT_BASELINE_PATH = REPO_ROOT / "tests" / "fixtures" / "retrieval_baseline.json"
INDEX_NAME = "retrieval-ci"

FIXTURE_CONFIG_TOML = """\
# Pinned deterministic embedding identity for the retrieval CI gate (RE-03).
# The "mock" provider is registered by scripts/ci_retrieval_eval.py.
[embedding]
provider = "mock"
model = "mock-model"
"""

MAX_DIFF_LINES_PER_KEY = 30


def build_fixture_project(project: Path) -> None:
    """Copy the benchmark's expected evidence files into a clean fixture project.

    Mirrors the RE-01 e2e ``build_fixture_project`` pattern: the benchmark
    expects repository-relative paths from the real ctxai tree, so exactly
    those files (plus a README) are copied, keeping their relative layout.

    Args:
        project: The fixture project root to populate.
    """
    payload = json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))
    needed: set[str] = set()
    for case in payload["cases"]:
        needed.update(case["expected"]["files"])
        needed.update(case["expected"].get("line_ranges", {}))
    for relative in sorted(needed):
        source = REPO_ROOT / relative
        target = project / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(source, target)
    (project / "README.md").write_text("# Retrieval benchmark fixture project\n", encoding="utf-8")
    config_dir = project / ".ctxai"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "config.toml").write_text(FIXTURE_CONFIG_TOML, encoding="utf-8")


def _print_baseline_diff(old: dict, new: dict) -> None:
    """Print the reviewable (volatile-stripped) diff between two baselines.

    Args:
        old: Previous baseline payload (may be the empty dict for a fresh one).
        new: Newly generated baseline payload.
    """
    old_stripped = strip_volatile(old)
    new_stripped = strip_volatile(new)
    if old_stripped == new_stripped:
        print("[OK] No content changes versus the previous baseline (volatile fields only).")
        return
    for key in sorted(set(old_stripped) | set(new_stripped)):
        old_lines = canonical_json(old_stripped.get(key)).splitlines()
        new_lines = canonical_json(new_stripped.get(key)).splitlines()
        if old_lines == new_lines:
            continue
        diff = list(
            difflib.unified_diff(
                old_lines,
                new_lines,
                fromfile=f"previous/{key}",
                tofile=f"new/{key}",
                lineterm="",
            )
        )
        print(f"\n--- diff: {key} ---")
        if len(diff) > MAX_DIFF_LINES_PER_KEY:
            print("\n".join(diff[:MAX_DIFF_LINES_PER_KEY]))
            print(f"... ({len(diff) - MAX_DIFF_LINES_PER_KEY} more diff lines truncated)")
        else:
            print("\n".join(diff))


def main() -> int:
    """Run the deterministic retrieval benchmark and the baseline gate.

    Returns:
        Process exit code: 0 pass (or successful baseline refresh), 1
        regression or run failure, 2 incompatible artifacts.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else "")
    parser.add_argument(
        "--baseline",
        type=Path,
        default=DEFAULT_BASELINE_PATH,
        help=f"Checked-in baseline artifact (default: {DEFAULT_BASELINE_PATH.relative_to(REPO_ROOT)})",
    )
    parser.add_argument(
        "--update-baseline",
        type=Path,
        default=None,
        metavar="PATH",
        help="Deliberate baseline refresh: write the fresh artifact to PATH and print a reviewable diff",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help="Copy the candidate artifact and comparison envelope here for CI upload",
    )
    args = parser.parse_args()

    EmbeddingsFactory.register_provider("mock", MockEmbeddingProvider)

    with tempfile.TemporaryDirectory(prefix="ctxai-retrieval-ci-") as scratch:
        project = Path(scratch) / "project"
        project.mkdir(parents=True)
        build_fixture_project(project)
        # Isolate the global configuration layer (developer machines) while
        # keeping everything project-contained: env home == project .ctxai.
        os.environ["CTXAI_HOME"] = str(project / ".ctxai")

        print("[*] Indexing the retrieval-benchmark fixture project (mock embeddings, no network)")
        index_result = index_codebase(path=project, index_name=INDEX_NAME)
        if not index_result.chunks:
            print("[X] Indexing produced no chunks", file=sys.stderr)
            return 1
        candidate_path = project / "candidate.json"
        print("[*] Running the retrieval benchmark")
        run_exit = run_retrieval_eval(
            benchmark_path=BENCHMARK_PATH,
            index_name=INDEX_NAME,
            project_path=project,
            output=candidate_path,
        )
        if run_exit != 0:
            print(f"[X] Benchmark run failed with exit code {run_exit}", file=sys.stderr)
            return 1

        candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
        print(
            "[*] candidate fingerprints: "
            f"benchmark={candidate['benchmark']['fingerprint'][:12]} "
            f"configuration={candidate['configuration']['fingerprint'][:12]}"
        )

        if args.artifacts_dir is not None:
            artifacts_dir = Path(args.artifacts_dir)
            artifacts_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy(candidate_path, artifacts_dir / "candidate.json")

        if args.update_baseline is not None:
            target = args.update_baseline.resolve()
            previous = json.loads(target.read_text(encoding="utf-8")) if target.is_file() else None
            atomic_write_json(target, candidate)
            if previous is not None:
                _print_baseline_diff(previous, candidate)
            else:
                print(f"[OK] Wrote a fresh baseline to {target}")
            print(
                "[*] Review evidence: benchmark fingerprint "
                f"{candidate['benchmark']['fingerprint']} / configuration fingerprint "
                f"{candidate['configuration']['fingerprint']}"
            )
            print(
                "[!] A baseline update must never be bundled invisibly with the retrieval change it "
                "excuses; land it as a separate reviewed change (docs/RETRIEVAL_BENCHMARK.md)."
            )
            return 0

        baseline_path = args.baseline.resolve()
        if not baseline_path.is_file():
            print(f"[X] Baseline artifact not found at {baseline_path}", file=sys.stderr)
            return 1
        print(f"[*] Comparing against the checked-in baseline ({baseline_path.relative_to(REPO_ROOT)})")
        exit_code = compare_retrieval_runs(baseline_path=baseline_path, candidate_path=candidate_path)

        if args.artifacts_dir is not None:
            from ctxai.evals.operations import compare_retrieval_payloads

            baseline_payload = json.loads(baseline_path.read_text(encoding="utf-8"))
            comparison = compare_retrieval_payloads(baseline_payload, candidate)
            atomic_write_json(
                Path(args.artifacts_dir) / "comparison.json",
                {"schema_version": 1, "kind": "retrieval-run-comparison", **comparison.to_dict()},
            )
        return exit_code


if __name__ == "__main__":
    sys.exit(main())
