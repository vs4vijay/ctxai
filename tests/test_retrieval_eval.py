"""RE-01 acceptance tests for the executable retrieval benchmark (unit level).

The historical hard asserts (20 pre-populated queries, recall@5 == 1.0) are
replaced by the versioned benchmark schema and runtime retrieval: the shipped
benchmark must validate against the real repository, and the runner must
produce honest per-case records, aggregates, unavailability markers, and
byte-stable artifacts apart from documented volatile fields.
"""

import hashlib
import json
from pathlib import Path

import pytest

from ctxai.chunking import CodeChunker
from ctxai.config import EmbeddingConfig
from ctxai.evals.artifacts import evaluations_dir
from ctxai.evals.benchmark import BenchmarkCase, ExpectedEvidence, RetrievalBenchmark, load_benchmark
from ctxai.evals.common import content_fingerprint, strip_volatile
from ctxai.evals.retrieval_runner import EvalError, RetrievalBenchmarkRunner, RetrievalEvalConfig
from ctxai.index_manifest import IndexedFile, IndexManifest
from ctxai.vector_store import VectorStore
from tests.mocks.mock_embeddings import MockEmbeddingProvider

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_BENCHMARK = Path(__file__).parent / "fixtures" / "retrieval_benchmark.json"

SYNTHETIC_FILES = {
    "src/agent_loop.py": (
        "def run_agent_loop(iterations):\n"
        '    """Run the agent iteration loop until the model stops."""\n'
        "    for _ in range(iterations):\n"
        "        pass\n"
    ),
    "src/context_window.py": (
        "def truncate_conversation(messages):\n"
        '    """Truncate the conversation context window."""\n'
        "    return messages[-10:]\n"
    ),
    "src/size_limits.py": (
        "def validate_project_size(limit_mb):\n"
        '    """Check the project size limits before indexing."""\n'
        "    return limit_mb > 0\n"
    ),
}


def build_synthetic_project(project: Path) -> None:
    """Create a small deterministic project and index it with mock embeddings.

    Args:
        project: Directory to create the project in.
    """
    for relative, content in SYNTHETIC_FILES.items():
        target = project / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")

    provider = MockEmbeddingProvider()
    chunker = CodeChunker()
    chunks = []
    for relative in sorted(SYNTHETIC_FILES):
        chunks.extend(chunker.chunk_file(project / relative))
    index_path = project / ".ctxai" / "indexes" / "eval-index"
    store = VectorStore(index_path, "eval-index")
    store.add_chunks(chunks, provider.generate_embeddings([chunk.content for chunk in chunks]))
    manifest = IndexManifest.create(
        index_name="eval-index",
        repository_root=project,
        embedding_provider="mock",
        embedding_model="mock-model",
        embedding_dimension=provider.get_dimension(),
    )
    chunks_by_file: dict[str, int] = {}
    for chunk in chunks:
        chunks_by_file[str(chunk.file_path.resolve())] = chunks_by_file.get(str(chunk.file_path.resolve()), 0) + 1
    manifest.files = {
        path: IndexedFile(sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(), chunks=count)
        for path, count in chunks_by_file.items()
    }
    manifest.file_count = len(manifest.files)
    manifest.chunk_count = len(chunks)
    manifest.save(index_path)


def synthetic_benchmark(tmp_path: Path) -> RetrievalBenchmark:
    """Build a three-case benchmark over the synthetic project.

    Args:
        tmp_path: Unused scratch directory (kept for fixture symmetry).

    Returns:
        The benchmark covering all three synthetic files.
    """
    return RetrievalBenchmark(
        schema_version=1,
        name="synthetic-eval",
        cases=[
            BenchmarkCase(
                id="s1",
                query="agent iteration loop",
                cohort="loop",
                split="test",
                tags=["loop"],
                expected=ExpectedEvidence(files=["src/agent_loop.py"]),
                relevance={"src/agent_loop.py": 3},
            ),
            BenchmarkCase(
                id="s2",
                query="conversation truncation",
                cohort="loop",
                split="test",
                tags=["context"],
                expected=ExpectedEvidence(files=["src/context_window.py"]),
                relevance={"src/context_window.py": 3},
            ),
            BenchmarkCase(
                id="s3",
                query="project size limits",
                cohort="sizes",
                split="train",
                tags=["limits"],
                expected=ExpectedEvidence(files=["src/size_limits.py"]),
                relevance={"src/size_limits.py": 3},
            ),
        ],
    )


def make_runner(project: Path, benchmark: RetrievalBenchmark, **config_overrides) -> RetrievalBenchmarkRunner:
    """Create a runner over the synthetic project with mock embeddings.

    Args:
        project: The synthetic project root.
        benchmark: Benchmark to run.
        **config_overrides: RetrievalEvalConfig field overrides.

    Returns:
        The configured runner.
    """
    return RetrievalBenchmarkRunner(
        project_root=project,
        benchmark=benchmark,
        index_name="eval-index",
        embedding_provider=MockEmbeddingProvider(),
        embedding_config=EmbeddingConfig(provider="mock", model="mock-model"),
        config=RetrievalEvalConfig(**config_overrides),
    )


class TestMigratedBenchmark:
    def test_twenty_questions_migrated_with_runtime_retrieval(self):
        """Criterion 1: the 20 questions live in the versioned schema without
        embedded retrieved results, and every expected file exists."""
        benchmark = load_benchmark(FIXTURE_BENCHMARK)
        assert benchmark.schema_version == 1
        assert benchmark.name == "ctxai-retrieval-core"
        assert len(benchmark.cases) == 20
        ids = [case.id for case in benchmark.cases]
        assert len(set(ids)) == 20
        for case in benchmark.cases:
            assert case.query.strip()
            assert case.split in ("train", "dev", "test")
            assert case.cohort in ("agent", "pipeline", "interface")
            assert case.tags, "migrated cases carry domain tags"
            assert case.expected.files, "every case expects at least one file"
            for expected_file in case.expected.files:
                assert (REPO_ROOT / expected_file).is_file(), f"{expected_file} missing from the repository"
            # Retrieved results are produced at runtime; the schema has no
            # field for them.
            assert "retrieved" not in json.dumps(case.to_dict())

    def test_migrated_benchmark_fingerprint_is_stable(self):
        benchmark = load_benchmark(FIXTURE_BENCHMARK)
        assert benchmark.fingerprint == load_benchmark(FIXTURE_BENCHMARK).fingerprint

    def test_validate_command_accepts_shipped_benchmark(self):
        """The shipped benchmark passes full payload validation."""
        from ctxai.evals.benchmark import benchmark_from_payload

        payload = json.loads(FIXTURE_BENCHMARK.read_text(encoding="utf-8"))
        assert benchmark_from_payload(payload).fingerprint


class TestRunnerExecution:
    def test_runtime_retrieval_produces_records_metrics_and_identity(self, tmp_path):
        """Criterion 2: runtime retrieval yields per-case ranks, citations,
        tokens, timings, configuration identity, and required aggregates."""
        project = tmp_path / "project"
        project.mkdir()
        build_synthetic_project(project)
        artifact = make_runner(project, synthetic_benchmark(tmp_path)).run()

        assert artifact.status == "complete"
        assert artifact.kind == "retrieval"
        assert [run.case_id for run in artifact.runs] == ["s1", "s2", "s3"]

        first = artifact.runs[0]
        assert first.status == "ok"
        assert first.candidate_count >= 1
        assert first.first_relevant_rank == 1
        assert first.metrics["recall@5"].value == 1.0
        assert first.metrics["mrr"].value == 1.0
        assert first.estimated_tokens > 0
        assert first.latency["values_ms"], "per-case latency is measured"
        assert first.timings["retrieve_ms"] >= 0.0
        selected = [candidate for candidate in first.candidates if candidate.decision == "selected"]
        assert selected, "selected context is recorded"
        assert all(candidate.citation.startswith("src/") for candidate in first.candidates)
        assert all(candidate.truncated is None for candidate in first.candidates if candidate.decision != "selected")

        overall = artifact.aggregates["overall"]
        assert overall.metrics["successful_query_rate"].value == 1.0
        for metric in ("recall@1", "recall@5", "recall@10", "mrr", "ndcg@10", "evidence_precision@5"):
            assert overall.metrics[metric].is_available
        assert overall.metrics["latency_p50_ms"].is_available
        assert overall.metrics["selected_token_mean"].is_available
        assert overall.metrics["duplicate_token_ratio"].is_available
        graph = overall.metrics["graph_contribution_rate"]
        assert not graph.is_available and "graph expansion not enabled" in graph.reason
        assert set(overall.confidence_intervals) == {"recall@5", "mrr"}
        assert sorted(artifact.aggregates["by_cohort"]) == ["loop", "sizes"]
        assert sorted(artifact.aggregates["by_split"]) == ["test", "train"]

        assert artifact.configuration["fingerprint"]
        assert artifact.configuration["embedding"] == {"provider": "mock", "model": "mock-model", "dimension": 384}
        assert artifact.index["name"] == "eval-index"
        assert artifact.environment["network_access"] == "none"

    def test_repeated_deterministic_runs_are_byte_stable_modulo_volatile_fields(self, tmp_path):
        """Criterion 3: two runs differ only in documented volatile fields."""
        project = tmp_path / "project"
        project.mkdir()
        build_synthetic_project(project)
        benchmark = synthetic_benchmark(tmp_path)
        first = make_runner(project, benchmark, repeats=3).run().to_dict()
        second = make_runner(project, benchmark, repeats=3).run().to_dict()

        assert strip_volatile(first) == strip_volatile(second)
        assert first["created_at"] != second["created_at"]
        assert first["run_id"] != second["run_id"]
        # Warm-up handling: 3 repeats -> first is warm-up, 2 latency samples.
        assert first["runs"][0]["latency"]["warmup_excluded"] == 1
        assert len(first["runs"][0]["latency"]["values_ms"]) == 2

    def test_query_recording_disabled_stores_hash_only(self, tmp_path):
        project = tmp_path / "project"
        project.mkdir()
        build_synthetic_project(project)
        artifact = make_runner(project, synthetic_benchmark(tmp_path), record_queries=False).run()
        encoded = json.dumps(artifact.to_dict())
        for case in synthetic_benchmark(tmp_path).cases:
            assert case.query not in encoded
        first = artifact.runs[0]
        assert first.query is None
        assert first.query_hash == content_fingerprint("agent iteration loop")


class TestFailureRepresentation:
    def test_invalid_line_range_makes_run_partial(self, tmp_path):
        """Criterion 5: invalid expectations cannot look successful."""
        project = tmp_path / "project"
        project.mkdir()
        build_synthetic_project(project)
        benchmark = synthetic_benchmark(tmp_path)
        broken_case = BenchmarkCase(
            id="s4",
            query="agent iteration loop",
            cohort="sizes",
            split="test",
            tags=[],
            expected=ExpectedEvidence(files=["src/agent_loop.py"], line_ranges={"src/agent_loop.py": [100, 200]}),
        )
        artifact = make_runner(
            project, RetrievalBenchmark(schema_version=1, name="partial", cases=[*benchmark.cases, broken_case])
        ).run()

        assert artifact.status == "partial"
        errored = [run for run in artifact.runs if run.status == "error"]
        assert [run.case_id for run in errored] == ["s4"]
        assert "beyond file length" in errored[0].error
        assert artifact.errors
        overall = artifact.aggregates["overall"]
        assert overall.metrics["successful_query_rate"].value == 3 / 4

    def test_missing_expected_file_is_a_case_error(self, tmp_path):
        project = tmp_path / "project"
        project.mkdir()
        build_synthetic_project(project)
        case = BenchmarkCase(
            id="ghost",
            query="anything",
            cohort="ghosts",
            split="test",
            expected=ExpectedEvidence(files=["src/does_not_exist.py"]),
        )
        artifact = make_runner(project, RetrievalBenchmark(schema_version=1, name="ghost", cases=[case])).run()
        assert artifact.status == "partial"
        assert "missing from repository" in artifact.runs[0].error
        cohort = artifact.aggregates["by_cohort"]["ghosts"]
        assert cohort.metrics["successful_query_rate"].value == 0.0
        assert not cohort.metrics["recall@5"].is_available
        assert "no successful cases" in cohort.metrics["recall@5"].reason

    def test_missing_index_fails_clearly(self, tmp_path):
        project = tmp_path / "project"
        project.mkdir()
        build_synthetic_project(project)
        with pytest.raises(EvalError, match="cannot be inspected"):
            RetrievalBenchmarkRunner(
                project_root=project,
                benchmark=synthetic_benchmark(tmp_path),
                index_name="no-such-index",
                embedding_provider=MockEmbeddingProvider(),
                embedding_config=EmbeddingConfig(provider="mock", model="mock-model"),
            )

    def test_embedding_identity_mismatch_fails_clearly(self, tmp_path):
        """Criterion 5: embedding mismatch is explicit, never a silent pass."""
        project = tmp_path / "project"
        project.mkdir()
        build_synthetic_project(project)
        with pytest.raises(EvalError, match="embedding identity"):
            RetrievalBenchmarkRunner(
                project_root=project,
                benchmark=synthetic_benchmark(tmp_path),
                index_name="eval-index",
                embedding_provider=MockEmbeddingProvider(),
                embedding_config=EmbeddingConfig(provider="mock", model="other-model"),
            )

    def test_unhealthy_index_fails_clearly(self, tmp_path):
        project = tmp_path / "project"
        project.mkdir()
        build_synthetic_project(project)
        (project / ".ctxai" / "indexes" / "eval-index" / "manifest.json").write_text("{broken", encoding="utf-8")
        with pytest.raises(EvalError, match="unhealthy"):
            make_runner(project, synthetic_benchmark(tmp_path))

    def test_stale_index_fails_clearly(self, tmp_path, monkeypatch):
        project = tmp_path / "project"
        project.mkdir()
        build_synthetic_project(project)
        manifest_path = project / ".ctxai" / "indexes" / "eval-index" / "manifest.json"
        manifest = IndexManifest.load(manifest_path.parent)
        manifest.repository_revision = "stale-revision"
        manifest.save(manifest_path.parent)
        monkeypatch.setattr("ctxai.index_operations.get_repository_revision", lambda _root: "fresh-revision")
        with pytest.raises(EvalError, match="stale"):
            make_runner(project, synthetic_benchmark(tmp_path))


class TestBounds:
    def test_repeat_bounds_rejected(self):
        with pytest.raises(ValueError):
            RetrievalEvalConfig(repeats=0)
        with pytest.raises(ValueError):
            RetrievalEvalConfig(repeats=11)

    def test_candidate_limit_bounds_rejected(self):
        with pytest.raises(ValueError):
            RetrievalEvalConfig(candidate_limit=0)
        with pytest.raises(ValueError):
            RetrievalEvalConfig(candidate_limit=101)

    def test_timeout_and_bootstrap_bounds_rejected(self):
        with pytest.raises(ValueError):
            RetrievalEvalConfig(per_case_timeout_s=0)
        with pytest.raises(ValueError):
            RetrievalEvalConfig(bootstrap_samples=-1)

    def test_artifact_written_atomically_under_evaluations_dir(self, tmp_path):
        """Artifacts belong under .ctxai/evaluations/retrieval by default."""
        project = tmp_path / "project"
        project.mkdir()
        build_synthetic_project(project)
        artifact = make_runner(project, synthetic_benchmark(tmp_path)).run()
        assert evaluations_dir(project).name == "retrieval"
        payload = strip_volatile(artifact.to_dict())
        assert payload["status"] == "complete"
