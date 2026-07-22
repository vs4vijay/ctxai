"""VS-09 browser acceptance tests for shared index operations."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from starlette.testclient import TestClient

from ctxai.commands.dashboard_command import create_dashboard_app, start_dashboard
from ctxai.commands.index_command import index_codebase
from ctxai.commands.indexes_command import doctor_index, list_indexes
from ctxai.index_manifest import IndexManifestError
from ctxai.index_operations import IndexOperations


@pytest.fixture
def dashboard_index(sample_python_code, temp_dir, patch_embeddings_factory, monkeypatch):
    indexes_dir = temp_dir / ".ctxai" / "indexes"
    monkeypatch.setattr("ctxai.commands.index_command.get_indexes_dir", lambda _path: indexes_dir)
    monkeypatch.setattr("ctxai.index_operations.get_indexes_dir", lambda _path: indexes_dir)
    index_codebase(sample_python_code, "dashboard-test", ["*.py"], follow_gitignore=False)
    return temp_dir, indexes_dir, patch_embeddings_factory


@pytest.mark.e2e
def test_browser_list_inspect_query_and_shared_cli_results(dashboard_index, monkeypatch):
    project_path, _, embeddings = dashboard_index
    monkeypatch.setattr("ctxai.index_operations.EmbeddingsFactory.create", lambda _config: embeddings)
    service = IndexOperations(project_path)
    expected = service.inspect("dashboard-test")

    with TestClient(create_dashboard_app(project_path)) as client:
        home = client.get("/")
        assert home.status_code == 200
        assert "dashboard-test" in home.text
        assert "healthy &amp; current" in home.text
        assert f"schema {expected.manifest.schema_version}" in home.text
        assert expected.manifest.embedding_model in home.text

        details = client.get("/index/dashboard-test")
        assert details.status_code == 200
        assert expected.manifest.repository_root in details.text
        assert str(expected.manifest.chunk_count) in details.text

        query_page = client.get("/query?index=dashboard-test")
        assert query_page.status_code == 200
        assert "value='dashboard-test' selected" in query_page.text
        result = client.post(
            "/query/search",
            data={"index": "dashboard-test", "query": "calculator functions", "n_results": "3"},
        )
        assert result.status_code == 200
        assert "Evidence for" in result.text
        assert ".py:" in result.text

    assert [item.index_name for item in list_indexes(project_path)] == ["dashboard-test"]
    assert doctor_index("dashboard-test", project_path).healthy


@pytest.mark.e2e
def test_browser_delete_and_index_name_boundary(dashboard_index):
    project_path, indexes_dir, _ = dashboard_index
    operations = IndexOperations(project_path)
    with pytest.raises(IndexManifestError, match="Invalid index name"):
        operations.inspect("../outside")

    with TestClient(create_dashboard_app(project_path)) as client:
        traversal = client.get("/index/%2E%2E%2Foutside")
        # The ASGI router rejects an encoded slash before it reaches domain validation.
        assert traversal.status_code == 404
        deleted = client.post("/index/dashboard-test/delete")
        assert deleted.status_code == 200
        assert "Index deleted" in deleted.text
    assert not (indexes_dir / "dashboard-test").exists()


@pytest.mark.e2e
def test_dashboard_binds_to_localhost_by_default(monkeypatch):
    captured = {}
    monkeypatch.setattr("ctxai.commands.dashboard_command.create_dashboard_app", lambda _path: object())
    monkeypatch.setattr("ctxai.commands.dashboard_command.serve", lambda **kwargs: captured.update(kwargs))
    start_dashboard(port=9876)
    assert captured["host"] == "127.0.0.1"
    assert captured["port"] == 9876

    captured.clear()
    with pytest.raises(ValueError, match="--allow-remote"):
        start_dashboard(port=9876, host="0.0.0.0")
    assert captured == {}
    with patch("ctxai.commands.dashboard_command.console.print") as output:
        start_dashboard(port=9876, host="0.0.0.0", allow_remote=True)
    assert captured["host"] == "0.0.0.0"
    assert "no authentication" in " ".join(str(call) for call in output.call_args_list)
