"""Durable, incremental code indexing workflow."""

from __future__ import annotations

import hashlib
import threading
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn

from ..chunking import CodeChunk, CodeChunker
from ..config import ConfigManager
from ..embeddings import EmbeddingsFactory
from ..index_manifest import SCHEMA_VERSION, IndexedFile, IndexManifest, get_repository_revision
from ..size_validator import ProjectSizeLimitError, ProjectSizeValidator
from ..traversal import CodeTraversal
from ..utils import get_ctxai_home, get_indexes_dir, is_using_global_home
from ..vector_store import VectorStore

console = Console(legacy_windows=False)


class IndexingError(RuntimeError):
    """Raised when an index cannot be completed reliably."""


class IndexingCancelled(IndexingError):
    """Raised when an indexing caller requests cooperative cancellation."""


ProgressCallback = Callable[[int, int, str], None]


@dataclass(frozen=True)
class IndexingResult:
    index_name: str
    storage_path: Path
    files: int
    chunks: int
    embedded_chunks: int
    unchanged_files: int
    changed_files: int
    deleted_files: int


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _embedding_identity(config, provider) -> tuple[str, str, int]:
    model = config.model or getattr(provider, "model", None) or "default"
    return config.provider, str(model), provider.get_dimension()


def _validate_embeddings(embeddings: list[list[float]], expected: int, dimension: int) -> None:
    if len(embeddings) != expected:
        raise IndexingError(f"Embedding provider returned {len(embeddings)} vectors for {expected} chunks")
    if any(len(vector) != dimension for vector in embeddings):
        raise IndexingError(f"Embedding provider returned a vector with a dimension other than {dimension}")
    if expected and any(not any(value != 0 for value in vector) for vector in embeddings):
        raise IndexingError("Embedding provider returned an empty/zero vector; refusing to publish the index")


def index_codebase(
    path: Path,
    index_name: str | None = None,
    include_patterns: list[str] | None = None,
    exclude_patterns: list[str] | None = None,
    follow_gitignore: bool = True,
    progress_callback: ProgressCallback | None = None,
    cancel_event: threading.Event | None = None,
) -> IndexingResult:
    """Build or deterministically update a persistent repository index."""
    path = path.resolve()
    console.print("\n[bold blue][*] Starting codebase indexing...[/bold blue]\n")
    ctxai_home = get_ctxai_home(path)
    label = "global CTXAI_HOME" if is_using_global_home() else "project .ctxai"
    console.print(f"[dim]Using {label}: {ctxai_home}[/dim]")

    config_manager = ConfigManager(path)
    config = config_manager.load()
    index_name = index_name or config.index_name or f"{path.name}-index"
    config_manager.update_index_metadata(index_name=index_name, status="indexing")

    def report(completed: int, total: int, message: str, *, cancellation_boundary: bool = True) -> None:
        if cancellation_boundary and cancel_event is not None and cancel_event.is_set():
            raise IndexingCancelled("Indexing cancelled by client")
        if progress_callback is not None:
            progress_callback(completed, total, message)

    try:
        report(0, 5, "Loading index configuration")
        embedding_provider = EmbeddingsFactory.create(config.embedding)
        provider_name, model_name, dimension = _embedding_identity(config.embedding, embedding_provider)
        indexes_dir = get_indexes_dir(path)
        storage_path = indexes_dir / index_name
        vector_store = VectorStore(storage_path, index_name)
        manifest = IndexManifest.load_optional(storage_path)

        if manifest and (
            Path(manifest.repository_root) != path
            or manifest.embedding_provider != provider_name
            or manifest.embedding_model != model_name
            or manifest.embedding_dimension != dimension
            or manifest.schema_version != SCHEMA_VERSION
        ):
            raise IndexingError(
                "Existing index identity does not match this repository or embedding model; delete it before rebuilding"
            )

        traversal = CodeTraversal(
            root_path=path,
            include_patterns=include_patterns,
            exclude_patterns=exclude_patterns,
            follow_gitignore=follow_gitignore,
        )
        files = sorted((file.resolve() for file in traversal.traverse()), key=str)
        if not files:
            raise IndexingError("No files found to index; check include and exclude patterns")

        validator = ProjectSizeValidator(config.indexing)
        project_stats = validator.analyze_files(files)
        valid, messages = validator.validate(project_stats)
        if not valid:
            raise ProjectSizeLimitError(messages)
        oversized = {item[0].resolve() for item in project_stats.oversized_files}
        files = [file for file in files if file not in oversized]
        report(1, 5, f"Discovered {len(files)} files")

        hashes = {str(file): _file_hash(file) for file in files}
        old_files = manifest.files if manifest else {}
        changed = [file for file in files if hashes[str(file)] != getattr(old_files.get(str(file)), "sha256", None)]
        unchanged_count = len(files) - len(changed)
        deleted = sorted(set(old_files) - set(hashes))

        # A legacy database has no trustworthy file state, so rebuild it once.
        if manifest is None and vector_store.get_stats()["total_chunks"]:
            vector_store.clear()

        chunker = CodeChunker(
            max_chunk_size=config.indexing.chunk_size,
            overlap=config.indexing.chunk_overlap,
        )
        chunks_by_file: dict[str, list[CodeChunk]] = {}
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Chunking changed files...", total=len(changed))
            for file in changed:
                report(1, 5, f"Chunking {file.name}")
                try:
                    chunks_by_file[str(file)] = chunker.chunk_file(file)
                except Exception as exc:
                    raise IndexingError(f"Failed to chunk {file}: {exc}") from exc
                progress.update(task, advance=1)

        changed_chunks = [chunk for file in changed for chunk in chunks_by_file[str(file)]]
        embeddings: list[list[float]] = []
        batch_size = max(1, config.embedding.batch_size)
        for offset in range(0, len(changed_chunks), batch_size):
            report(2, 5, f"Embedding chunks {offset + 1}-{min(offset + batch_size, len(changed_chunks))}")
            batch = changed_chunks[offset : offset + batch_size]
            try:
                embeddings.extend(embedding_provider.generate_embeddings([chunk.content for chunk in batch]))
            except Exception as exc:
                raise IndexingError(f"Embedding generation failed: {exc}") from exc
        _validate_embeddings(embeddings, len(changed_chunks), dimension)
        report(3, 5, "Writing index storage")

        # Mutate storage only after all changed content has valid embeddings.
        vector_store.delete_files(deleted + [str(file) for file in changed])
        vector_store.add_chunks(changed_chunks, embeddings)
        stats = vector_store.get_stats()

        changed_paths = {str(file) for file in changed}
        file_state = {key: value for key, value in old_files.items() if key in hashes and key not in changed_paths}
        for file in changed:
            file_state[str(file)] = IndexedFile(sha256=hashes[str(file)], chunks=len(chunks_by_file[str(file)]))

        expected_chunks = sum(item.chunks for item in file_state.values())
        expected_stored_files = sum(item.chunks > 0 for item in file_state.values())
        if stats["total_chunks"] != expected_chunks or stats["unique_files"] != expected_stored_files:
            raise IndexingError(
                "Index integrity check failed: stored counts do not match the files prepared for publication"
            )

        # Once storage mutation starts, complete manifest publication so storage
        # and metadata cannot be left at different revisions.
        report(4, 5, "Publishing index manifest", cancellation_boundary=False)

        if manifest is None:
            manifest = IndexManifest.create(
                index_name=index_name,
                repository_root=path,
                embedding_provider=provider_name,
                embedding_model=model_name,
                embedding_dimension=dimension,
            )
        manifest.repository_revision = get_repository_revision(path)
        manifest.updated_at = datetime.now(timezone.utc).isoformat()
        manifest.files = file_state
        manifest.file_count = len(file_state)
        manifest.chunk_count = expected_chunks
        manifest.save(storage_path)

        config_manager.update_index_metadata(
            index_name=index_name,
            status="completed",
            files_count=len(file_state),
            size_mb=project_stats.total_size_mb,
            chunks_count=expected_chunks,
        )
        result = IndexingResult(
            index_name=index_name,
            storage_path=storage_path,
            files=len(file_state),
            chunks=expected_chunks,
            embedded_chunks=len(changed_chunks),
            unchanged_files=unchanged_count,
            changed_files=len(changed),
            deleted_files=len(deleted),
        )
        console.print("[bold green][OK] Indexing complete![/bold green]")
        console.print(
            f"[dim]{result.files} files, {result.chunks} chunks; "
            f"embedded {result.embedded_chunks} changed chunks, removed {result.deleted_files} stale files[/dim]\n"
        )
        report(5, 5, "Indexing complete", cancellation_boundary=False)
        return result
    except Exception:
        config_manager.update_index_metadata(index_name=index_name, status="failed")
        raise
