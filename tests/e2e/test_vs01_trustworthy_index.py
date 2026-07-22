"""Acceptance coverage for VS-01: trustworthy local index and query."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from ctxai.app import app
from ctxai.commands.index_command import index_codebase
from ctxai.commands.indexes_command import doctor_index, get_index_info, list_indexes
from ctxai.index_manifest import MANIFEST_FILENAME
from ctxai.vector_store import VectorStore, VectorStoreWriteError


@pytest.mark.e2e
@pytest.mark.indexing
def test_index_survives_real_process_restart(sample_multi_language_code, temp_dir):
    index_path = temp_dir / "indexes" / "restart-index"
    helper = Path(__file__).with_name("index_process_helper.py")
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(Path(__file__).parents[2] / "src")

    build = subprocess.run(
        [sys.executable, str(helper), str(sample_multi_language_code), str(index_path), "build"],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert int(build.stdout.strip().splitlines()[-1]) > 0

    python_query = subprocess.run(
        [sys.executable, str(helper), str(sample_multi_language_code), str(index_path), "query", "python greet"],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    javascript_query = subprocess.run(
        [
            sys.executable,
            str(helper),
            str(sample_multi_language_code),
            str(index_path),
            "query",
            "javascript calculator",
        ],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert python_query.stdout.strip().endswith(".py")
    assert javascript_query.stdout.strip().endswith(".js")


@pytest.mark.e2e
@pytest.mark.indexing
def test_manifest_noop_change_and_delete_updates(
    sample_python_code, temp_dir, patch_embeddings_factory, monkeypatch
):
    indexes_dir = temp_dir / ".ctxai" / "indexes"
    monkeypatch.setattr("ctxai.commands.index_command.get_indexes_dir", lambda _path: indexes_dir)
    provider = patch_embeddings_factory
    generated: list[list[str]] = []
    original_generate = provider.generate_embeddings

    def recording_generate(texts):
        generated.append(list(texts))
        return original_generate(texts)

    monkeypatch.setattr(provider, "generate_embeddings", recording_generate)
    first = index_codebase(sample_python_code, "incremental", ["*.py"], follow_gitignore=False)
    assert first.embedded_chunks > 0
    manifest = get_index_info("incremental", temp_dir)
    assert manifest.repository_root == str(sample_python_code.resolve())
    assert manifest.file_count == first.files
    assert manifest.chunk_count == first.chunks
    assert manifest.embedding_provider == "local"
    assert json.loads((indexes_dir / "incremental" / MANIFEST_FILENAME).read_text())["schema_version"] == 1

    generated.clear()
    unchanged = index_codebase(sample_python_code, "incremental", ["*.py"], follow_gitignore=False)
    assert unchanged.embedded_chunks == 0
    assert generated == []

    removed = sample_python_code / "utils.py"
    removed_path = str(removed.resolve())
    removed.unlink()
    changed_file = sample_python_code / "main.py"
    changed_file.write_text(changed_file.read_text() + "\n\ndef newly_added():\n    return 42\n")
    updated = index_codebase(sample_python_code, "incremental", ["*.py"], follow_gitignore=False)
    assert updated.changed_files == 1
    assert updated.deleted_files == 1
    records = VectorStore(indexes_dir, "incremental").collection.get(include=["metadatas", "documents"])
    assert removed_path not in {metadata["file_path"] for metadata in records["metadatas"]}
    assert "newly_added" in "\n".join(records["documents"])
    assert doctor_index("incremental", temp_dir).healthy
    assert [item.index_name for item in list_indexes(temp_dir)] == ["incremental"]


@pytest.mark.e2e
@pytest.mark.indexing
def test_failed_write_is_fatal_and_manifest_is_not_published(
    sample_python_code, temp_dir, patch_embeddings_factory, monkeypatch
):
    indexes_dir = temp_dir / ".ctxai" / "indexes"
    monkeypatch.setattr("ctxai.commands.index_command.get_indexes_dir", lambda _path: indexes_dir)
    monkeypatch.setattr(
        VectorStore,
        "add_chunks",
        lambda *args, **kwargs: (_ for _ in ()).throw(VectorStoreWriteError("disk unavailable")),
    )
    with pytest.raises(VectorStoreWriteError, match="disk unavailable"):
        index_codebase(sample_python_code, "broken", ["*.py"], follow_gitignore=False)
    assert not (indexes_dir / "broken" / MANIFEST_FILENAME).exists()


@pytest.mark.e2e
@pytest.mark.indexing
def test_indexes_cli_lifecycle(sample_python_code, temp_dir, patch_embeddings_factory, monkeypatch):
    indexes_dir = temp_dir / ".ctxai" / "indexes"
    monkeypatch.setattr("ctxai.commands.index_command.get_indexes_dir", lambda _path: indexes_dir)
    index_codebase(sample_python_code, "managed", ["*.py"], follow_gitignore=False)
    runner = CliRunner()

    listed = runner.invoke(app, ["indexes", "list", "--project-path", str(temp_dir)])
    assert listed.exit_code == 0
    assert "managed" in listed.stdout
    info = runner.invoke(app, ["indexes", "info", "managed", "--project-path", str(temp_dir)])
    assert info.exit_code == 0
    assert '"schema_version": 1' in info.stdout
    doctor = runner.invoke(app, ["indexes", "doctor", "managed", "--project-path", str(temp_dir)])
    assert doctor.exit_code == 0
    assert "healthy" in doctor.stdout
    deleted = runner.invoke(
        app, ["indexes", "delete", "managed", "--yes", "--project-path", str(temp_dir)]
    )
    assert deleted.exit_code == 0
    assert not (indexes_dir / "managed").exists()
