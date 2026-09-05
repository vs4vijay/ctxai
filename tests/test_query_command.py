"""
Tests for query command functionality.

The command is a thin wrapper over the shared retrieval service
(``ctxai.repository_context.retrieve_evidence``); these tests pin that
routing, the flag handling (--explain/--graph/--no-graph), and the output
rendering boundaries.
"""

from contextlib import ExitStack
from unittest.mock import MagicMock, patch

from ctxai.commands.query_command import query_codebase
from ctxai.config import Config, EmbeddingConfig, IndexConfig, RetrievalConfig
from ctxai.repository_context import (
    ContextAssembler,
    ContextItem,
    EvidenceResult,
    GraphExpansionEvidence,
)


def _item(chunk_id: str, path: str, start: int, end: int, content: str) -> ContextItem:
    return ContextItem(
        id=chunk_id,
        content=content,
        file_path=path,
        start_line=start,
        end_line=end,
        chunk_type="function_definition",
        score=0.05,
    )


def _evidence(items: list[ContextItem], *, explain=None) -> EvidenceResult:
    context = ContextAssembler(token_budget=2000, debug=False).assemble("test-index", items)
    return EvidenceResult(
        index_name="test-index",
        items=items,
        context=context,
        explain=explain,
        graph_diagnostic=None,
        semantic_distances={item.id: 0.1 for item in items},
    )


def _query_patches(tmp_path, items=None, evidence=None):
    """Patch the command's boundaries: config, provider, index dir, service."""
    evidence = evidence or _evidence(items or [])
    indexes_dir = tmp_path / "indexes"
    indexes_dir.mkdir(exist_ok=True)
    (indexes_dir / "test-index").mkdir(exist_ok=True)
    manifest_mock = MagicMock()
    manifest_mock.load_optional.return_value = None  # legacy index: no identity check
    manifest_patch = patch("ctxai.commands.query_command.IndexManifest", manifest_mock)
    return (
        patch("ctxai.commands.query_command.ConfigManager"),
        patch("ctxai.commands.query_command.EmbeddingsFactory"),
        patch("ctxai.commands.query_command.get_indexes_dir", return_value=indexes_dir),
        manifest_patch,
        patch("ctxai.commands.query_command.retrieve_evidence", return_value=evidence),
    )


def test_query_command_routes_through_shared_service(tmp_path, capsys):
    """The command delegates retrieval to retrieve_evidence with config values."""
    items = [_item("c1", str(tmp_path / "app.py"), 10, 20, "def test_function():\n    pass")]
    patches = _query_patches(tmp_path, items)
    with ExitStack() as stack:
        mock_config_manager = stack.enter_context(patches[0])
        mock_retrieve = stack.enter_context(patches[4])
        for extra in patches[1:4]:
            stack.enter_context(extra)
        manager = mock_config_manager.return_value
        manager.load.return_value = Config(
            embedding=EmbeddingConfig(provider="local"),
            indexing=IndexConfig(),
            retrieval=RetrievalConfig(token_budget=1234),
        )
        query_codebase(index_name="test-index", query="test query", n_results=5, show_content=True)

    assert mock_retrieve.call_count == 1
    kwargs = mock_retrieve.call_args.kwargs
    assert kwargs["index_name"] == "test-index"
    assert kwargs["limit"] == 5
    assert kwargs["token_budget"] == 1234
    assert kwargs["explain"] is False
    output = capsys.readouterr().out
    assert "app.py" in output
    assert "test_function" in output


DEFAULT_TEST_CONFIG = Config(
    embedding=EmbeddingConfig(provider="local"),
    indexing=IndexConfig(),
    retrieval=RetrievalConfig(token_budget=2000),
)


def test_query_command_no_results(tmp_path, capsys):
    patches = _query_patches(tmp_path, [])
    with ExitStack() as stack:
        mock_config_manager = stack.enter_context(patches[0])
        mock_config_manager.return_value.load.return_value = DEFAULT_TEST_CONFIG
        mock_retrieve = stack.enter_context(patches[4])
        for extra in patches[1:4]:
            stack.enter_context(extra)
        query_codebase(index_name="test-index", query="test query", n_results=5, show_content=True)
    mock_retrieve.assert_called_once()
    assert "No results found" in capsys.readouterr().out


def test_query_command_index_not_found(tmp_path, capsys):
    indexes_dir = tmp_path / "indexes"
    indexes_dir.mkdir()
    with (
        patch("ctxai.commands.query_command.ConfigManager"),
        patch("ctxai.commands.query_command.EmbeddingsFactory"),
        patch("ctxai.commands.query_command.get_indexes_dir", return_value=indexes_dir),
        patch("ctxai.commands.query_command.IndexManifest"),
        patch("ctxai.commands.query_command.retrieve_evidence") as mock_retrieve,
    ):
        query_codebase(index_name="nonexistent-index", query="test query", n_results=5, show_content=True)
    mock_retrieve.assert_not_called()
    assert "not found" in capsys.readouterr().out


def test_query_command_no_content(tmp_path, capsys):
    items = [_item("c1", str(tmp_path / "app.py"), 10, 20, "def test_function():\n    pass")]
    patches = _query_patches(tmp_path, items)
    with ExitStack() as stack:
        mock_config_manager = stack.enter_context(patches[0])
        mock_config_manager.return_value.load.return_value = DEFAULT_TEST_CONFIG
        mock_retrieve = stack.enter_context(patches[4])
        for extra in patches[1:4]:
            stack.enter_context(extra)
        query_codebase(index_name="test-index", query="test query", n_results=5, show_content=False)
    mock_retrieve.assert_called_once()
    assert "app.py" in capsys.readouterr().out


def test_query_command_explain_flag_is_forwarded(tmp_path):
    items = [_item("c1", str(tmp_path / "app.py"), 10, 20, "def test_function():\n    pass")]
    patches = _query_patches(tmp_path, items)
    with ExitStack() as stack:
        mock_config_manager = stack.enter_context(patches[0])
        mock_config_manager.return_value.load.return_value = DEFAULT_TEST_CONFIG
        mock_retrieve = stack.enter_context(patches[4])
        for extra in patches[1:4]:
            stack.enter_context(extra)
        query_codebase(index_name="test-index", query="test query", n_results=5, show_content=True, explain=True)
    assert mock_retrieve.call_args.kwargs["explain"] is True


def test_query_command_graph_flags_configure_expansion(tmp_path):
    """--graph requires the graph, --no-graph disables it, None defers to config."""
    items = [_item("c1", str(tmp_path / "app.py"), 10, 20, "def test_function():\n    pass")]
    patches = _query_patches(tmp_path, items)
    with ExitStack() as stack:
        mock_config_manager = stack.enter_context(patches[0])
        mock_retrieve = stack.enter_context(patches[4])
        for extra in patches[1:4]:
            stack.enter_context(extra)
        manager = mock_config_manager.return_value
        manager.load.return_value = Config(
            embedding=EmbeddingConfig(provider="local"),
            indexing=IndexConfig(),
            retrieval=RetrievalConfig(graph_enabled=True, graph_seed_count=2),
        )
        query_codebase(index_name="test-index", query="q", n_results=5, show_content=True)
        explicit = mock_retrieve.call_args.kwargs["graph"]
        assert explicit.enabled is True and explicit.required is False  # config-driven

        query_codebase(index_name="test-index", query="q", n_results=5, show_content=True, graph=True)
        required = mock_retrieve.call_args.kwargs["graph"]
        assert required.enabled is True and required.required is True  # --graph requires

        query_codebase(index_name="test-index", query="q", n_results=5, show_content=True, graph=False)
        disabled = mock_retrieve.call_args.kwargs["graph"]
        assert disabled.enabled is False  # --no-graph wins over config


def test_query_command_graph_evidence_is_displayed(tmp_path, capsys):
    items = [
        _item("c1", str(tmp_path / "app.py"), 10, 20, "def test_function():\n    pass"),
        _item("c2", str(tmp_path / "caller.py"), 1, 3, "def caller():\n    test_function()"),
    ]
    items[1].graph_evidence = GraphExpansionEvidence(
        seed_chunk_id="c1",
        seed_citation=f"{tmp_path / 'app.py'}:10-20",
        seed_symbol="app.test_function",
        expanded_symbol="caller.caller",
        edge_kind="calls",
        confidence="exact",
        depth=1,
        path="app.test_function -[calls]-> caller.caller",
        contribution=0.04,
    )
    patches = _query_patches(tmp_path, items)
    with ExitStack() as stack:
        mock_config_manager = stack.enter_context(patches[0])
        mock_config_manager.return_value.load.return_value = DEFAULT_TEST_CONFIG
        for extra in patches[1:]:
            stack.enter_context(extra)
        query_codebase(index_name="test-index", query="test query", n_results=5, show_content=False)
    output = capsys.readouterr().out
    assert "app.test_function -[calls]-> caller.caller" in output
