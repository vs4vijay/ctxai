"""IG-03 unit tests: graph-expanded grounded retrieval.

Deterministic fixtures prove useful one-hop expansion with an explained path,
cycle handling, confidence/depth decay, deduplication, cap enforcement, token
budget enforcement, stale-graph fallback semantics, and fusion determinism.
"""

import asyncio
import json
from pathlib import Path

import pytest

from ctxai.chunking import CodeChunker
from ctxai.config import ConfigManager, RetrievalConfig
from ctxai.evals.artifacts import RELATIONSHIP_COHORT, compare_graph_gate
from ctxai.evals.benchmark import BenchmarkCase, ExpectedEvidence, RetrievalBenchmark
from ctxai.evals.retrieval_runner import EvalError, RetrievalBenchmarkRunner, RetrievalEvalConfig
from ctxai.graph.builder import GraphBuilder
from ctxai.index_manifest import IndexedFile, IndexManifest
from ctxai.repository_context import (
    CONFIDENCE_FACTORS,
    FUSION_RANK_OFFSET,
    ContextAssembler,
    GraphExpansionSettings,
    HybridRetriever,
    retrieve_evidence,
)
from ctxai.vector_store import VectorStore
from tests.mocks.mock_embeddings import MockEmbeddingProvider

REPO_ROOT = Path(__file__).resolve().parents[1]

SERVICE_PY = '''def fetch_data():
    """Load rows for the report."""
    return [1, 2, 3]


def run_pipeline():
    """Call the loader and summarize the rows."""
    return sum(fetch_data())
'''

TEST_SERVICE_PY = '''def test_fetch_data():
    """Verify the loader returns rows."""
    assert fetch_data() == [1, 2, 3]
'''

NOISE_COUNT = 6


def build_graph_project(project: Path) -> None:
    """Create a deterministic project with caller/test relationships and index it.

    ``test_service.py`` calls (and therefore tests) ``fetch_data`` in
    ``service.py`` — a genuine cross-file relationship edge; the noise files
    push unrelated chunks into every ranking so expansion must earn its place.

    Args:
        project: Directory to create the project in.
    """
    files = {"service.py": SERVICE_PY, "test_service.py": TEST_SERVICE_PY}
    for number in range(NOISE_COUNT):
        files[f"noise{number}.py"] = (
            f"def noise_helper_{number}(payload):\n"
            f'    """Background module {number} with unrelated archive words."""\n'
            "    return sorted(filter(None, payload))\n"
        )
    for name, content in files.items():
        (project / name).write_text(content, encoding="utf-8")

    provider = MockEmbeddingProvider()
    chunker = CodeChunker()
    chunks = []
    for name in sorted(files):
        chunks.extend(chunker.chunk_file(project / name))
    index_path = project / ".ctxai" / "indexes" / "graph-index"
    store = VectorStore(index_path, "graph-index")
    store.add_chunks(chunks, provider.generate_embeddings([chunk.content for chunk in chunks]))
    GraphBuilder(repository_root=project).update(
        index_path=index_path,
        files=set(files),
        changed=set(files),
        deleted=set(),
        force_full=True,
    )
    manifest = IndexManifest.create(
        index_name="graph-index",
        repository_root=project,
        embedding_provider="mock",
        embedding_model="mock-model",
        embedding_dimension=provider.get_dimension(),
    )
    manifest.files = {
        str(project / name): IndexedFile(sha256=hashlib_sha256(project / name), chunks=1) for name in files
    }
    manifest.file_count = len(files)
    manifest.chunk_count = len(chunks)
    manifest.save(index_path)


def hashlib_sha256(path: Path) -> str:
    """Hash a file's bytes (helper kept local to avoid importing hashlib twice).

    Args:
        path: File to hash.

    Returns:
        Hex digest of the file content.
    """
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_retriever(project: Path, **settings) -> HybridRetriever:
    """Create a graph-enabled retriever over the fixture project.

    Args:
        project: Fixture project root.
        **settings: GraphExpansionSettings field overrides.

    Returns:
        The retriever.
    """
    provider = MockEmbeddingProvider()
    return HybridRetriever(
        project,
        provider,
        index_name="graph-index",
        graph=GraphExpansionSettings(enabled=True, required=False, **settings),
    )


def graph_benchmark() -> RetrievalBenchmark:
    """A three-case benchmark over the graph fixture project.

    Returns:
        The benchmark; the pipeline case targets the implementation symbol.
    """
    return RetrievalBenchmark(
        schema_version=1,
        name="graph-expansion-eval",
        cases=[
            BenchmarkCase(
                id="g1",
                query="run_pipeline summarize rows",
                cohort="pipeline",
                split="test",
                tags=["relationship"],
                expected=ExpectedEvidence(files=["service.py"]),
                relevance={"service.py": 3},
            ),
            BenchmarkCase(
                id="g2",
                query="loader verification",
                cohort="pipeline",
                split="train",
                tags=["relationship"],
                expected=ExpectedEvidence(files=["test_service.py"]),
                relevance={"test_service.py": 3},
            ),
            BenchmarkCase(
                id="g3",
                query="archive words",
                cohort="noise",
                split="test",
                tags=["noise"],
                expected=ExpectedEvidence(files=["noise1.py"]),
                relevance={"noise1.py": 3},
            ),
        ],
    )


def make_eval_runner(project: Path, **config_overrides) -> RetrievalBenchmarkRunner:
    """Create an eval runner over the fixture project with mock embeddings.

    Args:
        project: Fixture project root.
        **config_overrides: RetrievalEvalConfig overrides.

    Returns:
        The configured runner.
    """
    from ctxai.config import EmbeddingConfig

    return RetrievalBenchmarkRunner(
        project_root=project,
        benchmark=graph_benchmark(),
        index_name="graph-index",
        embedding_provider=MockEmbeddingProvider(),
        embedding_config=EmbeddingConfig(provider="mock", model="mock-model"),
        config=RetrievalEvalConfig(**config_overrides),
    )


# ----------------------------------------------------------------------
# RetrievalConfig bounds and round trip
# ----------------------------------------------------------------------


class TestRetrievalConfig:
    def test_defaults_are_disabled_with_bounded_policy(self):
        config = RetrievalConfig()
        assert config.graph_enabled is False
        assert config.graph_seed_count == 3
        assert config.graph_expansion_cap == 24
        assert config.graph_max_neighbors_per_seed == 8
        assert config.graph_depth == 1
        assert set(config.graph_edge_weights) == {"calls", "imports", "inherits", "tests", "references"}

    def test_to_dict_from_dict_round_trip(self):
        config = RetrievalConfig(
            token_budget=1500,
            graph_enabled=True,
            graph_edge_weights={"calls": 0.8, "tests": 0.6},
            graph_seed_count=2,
            graph_expansion_cap=10,
            graph_max_neighbors_per_seed=4,
            graph_depth=2,
        )
        assert RetrievalConfig.from_dict(config.to_dict()) == config

    def test_from_dict_ignores_unknown_keys_and_partial_payloads(self):
        config = RetrievalConfig.from_dict({"graph_enabled": True, "unknown_key": 1})
        assert config.graph_enabled is True
        assert config.token_budget == 2000

    def test_out_of_bounds_values_rejected(self):
        with pytest.raises(ValueError):
            RetrievalConfig(graph_seed_count=0)
        with pytest.raises(ValueError):
            RetrievalConfig(graph_seed_count=11)
        with pytest.raises(ValueError):
            RetrievalConfig(graph_expansion_cap=0)
        with pytest.raises(ValueError):
            RetrievalConfig(graph_expansion_cap=101)
        with pytest.raises(ValueError):
            RetrievalConfig(graph_max_neighbors_per_seed=0)
        with pytest.raises(ValueError):
            RetrievalConfig(graph_depth=3)
        with pytest.raises(ValueError):
            RetrievalConfig(token_budget=0)
        with pytest.raises(ValueError):
            RetrievalConfig(graph_edge_weights={"teleports": 1.0})
        with pytest.raises(ValueError):
            RetrievalConfig(graph_edge_weights={"calls": 1.5})
        with pytest.raises(ValueError):
            RetrievalConfig(graph_edge_weights={"calls": 0.0})

    def test_project_layer_overrides_global_layer(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CTXAI_HOME", str(tmp_path / "global"))
        global_manager = ConfigManager(use_global=True)
        global_config = global_manager.load()
        global_config.retrieval.graph_seed_count = 5
        global_manager.save(global_config)

        project = tmp_path / "project"
        (project / ".ctxai").mkdir(parents=True)
        project_manager = ConfigManager(project)
        project_config = project_manager.load()
        project_config.retrieval.graph_enabled = True
        project_config.retrieval.graph_seed_count = 2
        project_manager.save(project_config)

        merged = ConfigManager(project).load()
        assert merged.retrieval.graph_enabled is True  # project override
        assert merged.retrieval.graph_seed_count == 2  # project wins key by key


# ----------------------------------------------------------------------
# GraphExpansionSettings
# ----------------------------------------------------------------------


class TestGraphExpansionSettings:
    def test_default_is_disabled(self):
        assert GraphExpansionSettings().enabled is False

    def test_from_config_respects_override_and_requirement(self):
        config = RetrievalConfig(graph_enabled=True)
        assert GraphExpansionSettings.from_config(config).enabled is True
        assert GraphExpansionSettings.from_config(config, enabled=False).enabled is False
        assert GraphExpansionSettings.from_config(RetrievalConfig(), enabled=True).enabled is True
        assert GraphExpansionSettings.from_config(config, required=True).required is True

    def test_invalid_settings_rejected(self):
        with pytest.raises(ValueError):
            GraphExpansionSettings(seed_count=99)
        with pytest.raises(ValueError):
            GraphExpansionSettings(depth=5)


# ----------------------------------------------------------------------
# Expansion behavior
# ----------------------------------------------------------------------


class TestGraphExpansion:
    @pytest.fixture(autouse=True)
    def _project(self, tmp_path):
        self.project = tmp_path
        build_graph_project(self.project)

    def test_expansion_adds_caller_with_explained_path(self):
        """Acceptance 1+2: a seeded implementation symbol pulls in its caller
        and the item cites both source and relationship evidence."""
        retriever = make_retriever(self.project, seed_count=1)
        result = retriever.retrieve_detailed("run_pipeline summarize rows", limit=10, explain=True)

        assert result.explain.seeds, "a seed was resolved"
        assert result.explain.seeds[0]["symbol"] == "service.run_pipeline"
        graph_items = [item for item in result.items if item.graph_evidence is not None]
        assert graph_items, "graph expansion contributed candidates"
        by_symbol = {item.graph_evidence.expanded_symbol: item for item in graph_items}
        assert "service.fetch_data" in by_symbol
        evidence = by_symbol["service.fetch_data"].graph_evidence
        assert evidence.path == "service.run_pipeline -[calls]-> service.fetch_data"
        assert evidence.confidence == "exact"
        assert evidence.depth == 1
        assert evidence.seed_citation.endswith("service.py:6-8")
        reason = evidence.reason_text()
        assert "graph expansion from" in reason and "calls" in reason

    def test_depth_two_reaches_the_test_symbol(self):
        """Default depth is one; depth two is an explicit option that reaches
        the transitive test relationship through the same visited set."""
        one_hop = make_retriever(self.project, seed_count=1)
        two_hop = make_retriever(self.project, seed_count=1, depth=2)
        query = "run_pipeline summarize rows"
        one_symbols = {
            item.graph_evidence.expanded_symbol
            for item in one_hop.retrieve_detailed(query, limit=20).items
            if item.graph_evidence is not None
        }
        two_symbols = {
            item.graph_evidence.expanded_symbol
            for item in two_hop.retrieve_detailed(query, limit=20).items
            if item.graph_evidence is not None
        }
        assert "test_service.test_fetch_data" not in one_symbols
        assert "test_service.test_fetch_data" in two_symbols

    def test_weight_and_depth_decay(self):
        """Contribution follows seed score * weight**depth * confidence."""
        retriever = make_retriever(self.project, seed_count=1, edge_weights={"calls": 0.5})
        result = retriever.retrieve_detailed("run_pipeline summarize rows", limit=20, explain=True)
        seed_score = result.explain.seeds[0]["base_score"]
        contributions = [
            (candidate["contribution"], CONFIDENCE_FACTORS[candidate["confidence"]])
            for candidate in result.explain.graph_candidates
            if candidate["path"].endswith("service.fetch_data")
        ]
        assert contributions, "the one-hop caller was expanded"
        for contribution, confidence_factor in contributions:
            assert contribution == pytest.approx(seed_score * 0.5 * confidence_factor)

        two_hop = make_retriever(self.project, seed_count=1, depth=2, edge_weights={"calls": 0.5, "tests": 0.5})
        deep = two_hop.retrieve_detailed("run_pipeline summarize rows", limit=20, explain=True)
        seed_score = deep.explain.seeds[0]["base_score"]
        deep_contributions = [
            (candidate["contribution"], CONFIDENCE_FACTORS[candidate["confidence"]])
            for candidate in deep.explain.graph_candidates
            if candidate["path"].endswith("service.test_fetch_data")
        ]
        # Decay is weight**depth (0.5 ** 2) times the edge's confidence factor.
        for contribution, confidence_factor in deep_contributions:
            assert contribution == pytest.approx(seed_score * 0.5**2 * confidence_factor)

    def test_probable_confidence_is_decayed_below_exact(self):
        from ctxai.repository_context import HybridRetriever as _  # noqa: F401 - context import

        class FakeEdges:
            def __init__(self, edges):
                self.edges = edges

        class FakeEdge:
            def __init__(self, edge_id, kind, confidence):
                self.id = edge_id
                self.kind = kind
                self.confidence = confidence
                self.source_id = "a"
                self.target_id = "b"

        exact = FakeEdge("e1", "calls", "exact")
        probable = FakeEdge("e2", "calls", "probable")
        allowed = {"calls": 1.0}
        best = HybridRetriever._best_allowlisted_edge(FakeEdges([probable, exact]), "a", "b", allowed)
        assert best is exact
        assert CONFIDENCE_FACTORS["probable"] < CONFIDENCE_FACTORS["exact"]

    def test_cycles_never_duplicate_evidence(self):
        """A cyclic call graph (a -> b -> a) cannot duplicate evidence: the
        visited set stops re-accepting symbols and the assembler sees each
        chunk identity once."""
        cyclic = self.project / "cyclic.py"
        cyclic.write_text(
            "def ping():\n    return pong()\n\n\ndef pong():\n    return ping()\n",
            encoding="utf-8",
        )
        self._reindex_with(cyclic)
        retriever = make_retriever(self.project, seed_count=1, depth=2)
        result = retriever.retrieve_detailed("ping pong cycle", limit=20, explain=True)
        chunk_ids = [item.id for item in result.items]
        assert len(chunk_ids) == len(set(chunk_ids)), "no chunk enters the ranking twice"
        identities = [(item.file_path, item.start_line, item.end_line) for item in result.items]
        assert len(identities) == len(set(identities))
        expanded = [item.graph_evidence.expanded_symbol for item in result.items if item.graph_evidence]
        assert len(expanded) == len(set(expanded)), "a symbol is never accepted twice"

    def _reindex_with(self, new_file: Path) -> None:
        """Add one file to the fixture index and graph (deterministic rebuild).

        Args:
            new_file: The file to include.
        """
        provider = MockEmbeddingProvider()
        chunker = CodeChunker()
        files = sorted(p for p in self.project.glob("*.py"))
        chunks = []
        for file_path in files:
            chunks.extend(chunker.chunk_file(file_path))
        index_path = self.project / ".ctxai" / "indexes" / "graph-index"
        store = VectorStore(index_path, "graph-index")
        store.clear()
        store.add_chunks(chunks, provider.generate_embeddings([chunk.content for chunk in chunks]))
        GraphBuilder(repository_root=self.project).update(
            index_path=index_path,
            files={file_path.name for file_path in files},
            changed={new_file.name},
            deleted=set(),
            force_full=False,
        )

    def test_high_degree_nodes_respect_neighbor_caps(self):
        """A hub symbol with many neighbors is truncated at the configured cap."""

        class FakeNode:
            def __init__(self, node_id, name):
                self.id = node_id
                self.qualified_name = name
                self.display_name = name
                self.file_path = "service.py"
                self.start_line = 1
                self.end_line = 2

        class FakeEdge:
            def __init__(self, edge_id, source_id, target_id):
                self.id = edge_id
                self.kind = "calls"
                self.confidence = "exact"
                self.source_id = source_id
                self.target_id = target_id

        class FakeResult:
            def __init__(self, nodes, edges, truncated):
                self.nodes = nodes
                self.edges = edges
                self.truncated = truncated

        class FakeOperations:
            """Hub with 12 neighbors; both neighbors and total expansion cap."""

            def __init__(self, hub, neighbors):
                self.hub = hub
                self.neighbors_list = neighbors

            def find_symbols(self, index_name, query, kind=None, language=None, limit=20):
                return [self.hub]

            def neighbors(self, index_name, symbol_id, edge_kind=None, direction="both", depth=1, limit=50):
                if symbol_id == self.hub.id:
                    return FakeResult(
                        [self.hub, *self.neighbors_list],
                        [FakeEdge(f"e{n:02d}", self.hub.id, node.id) for n, node in enumerate(self.neighbors_list)],
                        truncated=True,
                    )
                return FakeResult([self.hub], [], False)

        retriever = make_retriever(self.project, seed_count=1, max_neighbors_per_seed=4, expansion_cap=3)
        fake_hub = FakeNode("hub-node-0001", "service.hub")
        retriever._graph_operations = FakeOperations(
            fake_hub, [FakeNode(f"neighbor-{i:04d}", f"service.n{i}") for i in range(12)]
        )

        result = retriever.retrieve_detailed("run_pipeline summarize rows", limit=20, explain=True)

        expanded = [item for item in result.items if item.graph_evidence is not None]
        assert len(expanded) <= 3, "total expansion cap respected"
        assert any("neighbors truncated at cap 4" in line for line in result.explain.diagnostics)
        assert any("expansion cap 3 reached" in line for line in result.explain.diagnostics)

    def test_repeated_runs_are_identical(self):
        """Acceptance 3: identical index + configuration + query produce the
        identical ordering, scores, and evidence."""
        first = make_retriever(self.project).retrieve_detailed("run_pipeline summarize rows", limit=20, explain=True)
        second = make_retriever(self.project).retrieve_detailed("run_pipeline summarize rows", limit=20, explain=True)
        assert [(i.id, round(i.score, 9), list(i.reasons)) for i in first.items] == [
            (i.id, round(i.score, 9), list(i.reasons)) for i in second.items
        ]
        assert first.explain.seeds == second.explain.seeds
        assert first.explain.graph_candidates == second.explain.graph_candidates

    def test_disabled_retriever_matches_legacy_behavior(self):
        """With expansion disabled, results equal an explicit-disabled run."""
        provider = MockEmbeddingProvider()
        default_retriever = HybridRetriever(self.project, provider, index_name="graph-index")
        disabled_retriever = HybridRetriever(
            self.project, provider, index_name="graph-index", graph=GraphExpansionSettings(enabled=False)
        )
        query = "run_pipeline summarize rows"
        assert [item.citation for item in default_retriever.retrieve(query)] == [
            item.citation for item in disabled_retriever.retrieve(query)
        ]
        assert default_retriever.graph_diagnostic is None


# ----------------------------------------------------------------------
# Stale/missing/corrupt graph behavior
# ----------------------------------------------------------------------


class TestGraphAvailability:
    @pytest.fixture(autouse=True)
    def _project(self, tmp_path):
        self.project = tmp_path
        build_graph_project(self.project)
        self.index_path = self.project / ".ctxai" / "indexes" / "graph-index"

    def test_required_but_missing_graph_fails_explicitly(self):
        (self.index_path / "graph.sqlite3").unlink()
        provider = MockEmbeddingProvider()
        with pytest.raises(LookupError, match="graph data has not been built"):
            HybridRetriever(
                self.project,
                provider,
                index_name="graph-index",
                graph=GraphExpansionSettings(enabled=True, required=True),
            )

    def test_config_enabled_but_missing_graph_falls_back_with_diagnostic(self):
        (self.index_path / "graph.sqlite3").unlink()
        provider = MockEmbeddingProvider()
        retriever = HybridRetriever(
            self.project,
            provider,
            index_name="graph-index",
            graph=GraphExpansionSettings(enabled=True, required=False),
        )
        assert retriever.graph_diagnostic is not None
        assert "falling back" in retriever.graph_diagnostic
        fallback = retriever.retrieve("run_pipeline summarize rows", limit=10)
        base = HybridRetriever(self.project, MockEmbeddingProvider(), index_name="graph-index").retrieve(
            "run_pipeline summarize rows", limit=10
        )
        assert [item.citation for item in fallback] == [item.citation for item in base]

    def test_stale_generation_fails_when_required(self):
        manifest = IndexManifest.load(self.index_path)
        manifest.graph_schema_version = 2
        manifest.graph_extractor_version = "python/1"
        manifest.graph_generation = 99
        manifest.graph_node_count = 7
        manifest.graph_edge_count = 7
        manifest.save(self.index_path)
        provider = MockEmbeddingProvider()
        with pytest.raises(LookupError, match="does not match manifest generation"):
            HybridRetriever(
                self.project,
                provider,
                index_name="graph-index",
                graph=GraphExpansionSettings(enabled=True, required=True),
            )

    def test_corrupt_graph_falls_back_with_diagnostic(self):
        (self.index_path / "graph.sqlite3").write_bytes(b"this is not a sqlite database")
        provider = MockEmbeddingProvider()
        retriever = HybridRetriever(
            self.project,
            provider,
            index_name="graph-index",
            graph=GraphExpansionSettings(enabled=True, required=False),
        )
        assert retriever.graph_diagnostic is not None
        items = retriever.retrieve("run_pipeline summarize rows", limit=10)
        assert items


# ----------------------------------------------------------------------
# Budget enforcement and assembler exclusions
# ----------------------------------------------------------------------


class TestBudget:
    @pytest.fixture(autouse=True)
    def _project(self, tmp_path):
        self.project = tmp_path
        build_graph_project(self.project)

    def test_assembler_records_exclusions(self):
        provider = MockEmbeddingProvider()
        retriever = HybridRetriever(self.project, provider, index_name="graph-index")
        ranked = retriever.retrieve("run_pipeline summarize rows", limit=20)

        # A duplicate identity is recorded, never re-selected.
        once = ContextAssembler(token_budget=10_000).assemble("graph-index", [ranked[0], ranked[0]])
        assert len(once.items) == 1
        assert once.excluded == ((ranked[0].citation, "duplicate"),)

        # A token budget of one excludes the first examined item explicitly.
        starved = ContextAssembler(token_budget=1).assemble("graph-index", list(ranked[:2]))
        assert starved.items == []
        assert starved.excluded[0] == (ranked[0].citation, "budget")

    def test_expansion_cannot_exceed_the_token_budget(self):
        evidence = retrieve_evidence(
            self.project,
            "run_pipeline summarize rows",
            embedding_provider=MockEmbeddingProvider(),
            index_name="graph-index",
            limit=20,
            token_budget=90,
            graph=GraphExpansionSettings(enabled=True, seed_count=1),
        )
        assert evidence.context.estimated_tokens <= 90

    def test_shared_service_returns_explain_and_diagnostic(self):
        evidence = retrieve_evidence(
            self.project,
            "run_pipeline summarize rows",
            embedding_provider=MockEmbeddingProvider(),
            index_name="graph-index",
            graph=GraphExpansionSettings(enabled=True, seed_count=1),
            explain=True,
        )
        assert evidence.explain is not None
        assert evidence.graph_diagnostic is None
        assert evidence.semantic_distances, "semantic distances are carried for adapters"


# ----------------------------------------------------------------------
# Evaluation integration
# ----------------------------------------------------------------------


class TestEvalIntegration:
    @pytest.fixture(autouse=True)
    def _project(self, tmp_path):
        self.project = tmp_path
        build_graph_project(self.project)

    def test_no_graph_run_keeps_contribution_unavailable(self):
        artifact = make_eval_runner(self.project).run()
        graph_metric = artifact.aggregates["overall"].metrics["graph_contribution_rate"]
        assert not graph_metric.is_available
        assert "graph expansion not enabled" in graph_metric.reason
        assert artifact.configuration["graph_expansion"]["enabled"] is False
        assert artifact.configuration["graph_expansion"]["relationship_cohort_cases"]

    def test_graph_run_records_contribution_paths_and_cohort(self):
        artifact = make_eval_runner(self.project, graph_enabled=True).run()
        assert artifact.status == "complete"
        overall = artifact.aggregates["overall"]
        contribution = overall.metrics["graph_contribution_rate"]
        assert contribution.is_available and contribution.value > 0.0
        assert RELATIONSHIP_COHORT in artifact.aggregates["by_cohort"]
        paths = [
            candidate.graph_path
            for run in artifact.runs
            for candidate in run.candidates
            if candidate.graph_path is not None
        ]
        assert paths, "graph-enabled runs record per-candidate expansion paths"
        assert any("calls" in path or "tests" in path for path in paths)
        cohort_cases = artifact.configuration["graph_expansion"]["relationship_cohort_cases"]
        assert set(cohort_cases) == {"g1", "g2"}

    def test_graph_run_requires_healthy_graph(self):
        (self.project / ".ctxai" / "indexes" / "graph-index" / "graph.sqlite3").unlink()
        with pytest.raises(EvalError, match="no graph data"):
            make_eval_runner(self.project, graph_enabled=True)

    def test_graph_and_no_graph_runs_derive_the_same_cohort(self):
        plain = make_eval_runner(self.project).run()
        expanded = make_eval_runner(self.project, graph_enabled=True).run()
        assert (
            plain.configuration["graph_expansion"]["relationship_cohort_cases"]
            == expanded.configuration["graph_expansion"]["relationship_cohort_cases"]
        )

    def test_compare_graph_gate_reports_verdict(self):
        baseline = make_eval_runner(self.project).run().to_dict()
        expanded = make_eval_runner(self.project, graph_enabled=True).run().to_dict()
        verdict = compare_graph_gate(baseline, expanded)
        assert verdict.compatible
        metric_names = {(gate.cohort, gate.metric) for gate in verdict.gates}
        assert ("overall", "recall@5") in metric_names
        assert ("overall", "mrr") in metric_names
        # The verdict is exactly the gate rule: pass = no regression plus a
        # pre-registered relationship improvement.
        has_regression = any(gate.status == "regression" for gate in verdict.gates)
        has_improvement = verdict.improvement is not None and verdict.improvement.status == "pass"
        assert verdict.passed == (not has_regression and has_improvement)

    def test_compare_graph_gate_rejects_wrong_modes(self):
        expanded = make_eval_runner(self.project, graph_enabled=True).run().to_dict()
        verdict = compare_graph_gate(expanded, expanded)
        assert not verdict.compatible
        assert any("no-graph run" in item for item in verdict.incompatibilities)

    def test_no_graph_artifact_json_round_trips(self, tmp_path):
        artifact = make_eval_runner(self.project).run()
        payload = json.loads(json.dumps(artifact.to_dict()))
        assert payload["aggregates"]["overall"]["metrics"]["graph_contribution_rate"]["value"] is None


# ----------------------------------------------------------------------
# Cross-interface consistency
# ----------------------------------------------------------------------


class TestInterfaceConsistency:
    @pytest.fixture(autouse=True)
    def _project(self, tmp_path):
        self.project = tmp_path
        build_graph_project(self.project)

    def _enable_graph_in_project_config(self) -> None:
        """Persist graph-enabled expansion in the project config layer.

        ``seed_count=1`` keeps only the top base hit as a seed so the fixture's
        related symbols are genuinely *expanded* rather than already seeded.
        """
        manager = ConfigManager(self.project)
        config = manager.load()
        config.retrieval.graph_enabled = True
        config.retrieval.graph_seed_count = 1
        manager.save(config)

    def test_agent_tool_uses_the_shared_service(self):
        from ctxai.agent.tools.code_search import SemanticSearchTool

        self._enable_graph_in_project_config()
        tool = SemanticSearchTool(project_path=self.project, embedding_provider=MockEmbeddingProvider())
        result = asyncio.run(tool.execute("run_pipeline summarize rows", n_results=10, token_budget=500, debug=True))
        assert result["success"] is True
        assert result["metadata"]["graph_expanded"] >= 1
        assert result["metadata"]["estimated_tokens"] <= 500
        assert result["metadata"]["explanation"]

    def test_agent_tool_reports_fallback_diagnostic(self):
        from ctxai.agent.tools.code_search import SemanticSearchTool

        self._enable_graph_in_project_config()
        (self.project / ".ctxai" / "indexes" / "graph-index" / "graph.sqlite3").unlink()
        tool = SemanticSearchTool(project_path=self.project, embedding_provider=MockEmbeddingProvider())
        result = asyncio.run(tool.execute("run_pipeline summarize rows", n_results=5))
        assert result["success"] is True
        assert "falling back" in result["metadata"]["graph_diagnostic"]

    def test_fusion_constants_unchanged(self):
        """The fusion policy keeps its RE-01-era shape: 1/(60 + rank)."""
        assert FUSION_RANK_OFFSET == 60
