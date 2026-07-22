from __future__ import annotations

from pathlib import Path

import pytest

from ctxai.agent.tools.code_search import SemanticSearchTool
from ctxai.chunking import CodeChunk
from ctxai.index_manifest import IndexManifest
from ctxai.repository_context import ContextAssembler, HybridRetriever, discover_repository_indexes
from ctxai.retrieval_eval import evaluate_retrieval
from ctxai.vector_store import VectorStore


class FixedEmbeddings:
    model = "fixed"

    def generate_embedding(self, text: str) -> list[float]:
        return [1.0, 0.0]

    def get_dimension(self) -> int:
        return 2


def build_index(project: Path, name: str = "repo-index") -> None:
    index_path = project / ".ctxai" / "indexes" / name
    manifest = IndexManifest.create(
        index_name=name,
        repository_root=project,
        embedding_provider="fixed",
        embedding_model="fixed",
        embedding_dimension=2,
    )
    chunks = []
    vectors = []
    for number in range(7):
        is_target = number == 6
        content = (
            "def authorize_payment(request):\n    return policy.check(request)"
            if is_target else f"def unrelated_{number}():\n    return 'background code {number}'"
        )
        chunks.append(CodeChunk(
            content=content,
            file_path=(project / ("payments.py" if is_target else f"module_{number}.py")).resolve(),
            start_line=10 + number,
            end_line=11 + number,
            chunk_type="function_definition",
            language="python",
            metadata={"name": "authorize_payment" if is_target else f"unrelated_{number}"},
        ))
        vectors.append([0.0, 1.0] if is_target else [1.0, number / 100])
    store = VectorStore(index_path, name)
    store.add_chunks(chunks, vectors)
    manifest.file_count = 7
    manifest.chunk_count = 7
    manifest.save(index_path)


@pytest.mark.asyncio
async def test_chat_discovers_matching_index_and_returns_bounded_cited_evidence(tmp_path):
    project = tmp_path / "project"
    other = tmp_path / "other"
    project.mkdir()
    other.mkdir()
    build_index(project)
    build_index(other, "other-index")

    assert discover_repository_indexes(project) == ["repo-index"]
    tool = SemanticSearchTool(project_path=project, embedding_provider=FixedEmbeddings())
    result = await tool.execute("authorize_payment policy", n_results=5, token_budget=80, debug=True)

    assert result["success"] is True
    assert result["metadata"]["index_name"] == "repo-index"
    assert result["metadata"]["estimated_tokens"] <= 80
    assert any("payments.py:16-17" in citation for citation in result["metadata"]["citations"])
    assert "Selected because:" in result["result"]
    assert "payments.py:16-17" in result["result"]


def test_hybrid_retrieval_beats_vector_only_and_assembler_deduplicates(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    build_index(project)
    retriever = HybridRetriever(project, FixedEmbeddings())

    vector_only = retriever.store.search([1.0, 0.0], n_results=7)
    hybrid = retriever.retrieve("authorize_payment policy", limit=7)
    expected = str((project / "payments.py").resolve())
    vector_metrics = evaluate_retrieval([{
        "expected_locations": [expected],
        "retrieved_locations": [item["metadata"]["file_path"] for item in vector_only],
    }])
    hybrid_metrics = evaluate_retrieval([{
        "expected_locations": [expected],
        "retrieved_locations": [item.file_path for item in hybrid],
    }])

    assert hybrid_metrics.mrr > vector_metrics.mrr
    assembled = ContextAssembler(token_budget=100).assemble("repo-index", [hybrid[0], hybrid[0]])
    assert len(assembled.items) == 1
    assert assembled.estimated_tokens <= 100
