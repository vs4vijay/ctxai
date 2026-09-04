"""Repository-aware hybrid retrieval and bounded evidence assembly."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path

from .index_manifest import IndexManifest, IndexManifestError
from .utils import get_indexes_dir
from .vector_store import VectorStore

TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


@dataclass
class ContextItem:
    """A ranked code chunk with inspectable retrieval evidence."""

    id: str
    content: str
    file_path: str
    start_line: int
    end_line: int
    chunk_type: str
    score: float = 0.0
    reasons: list[str] = field(default_factory=list)

    @property
    def citation(self) -> str:
        return f"{self.file_path}:{self.start_line}-{self.end_line}"


@dataclass(frozen=True)
class AssembledContext:
    index_name: str
    items: list[ContextItem]
    text: str
    estimated_tokens: int
    truncated: tuple[bool, ...] = ()


def discover_repository_indexes(project_path: Path) -> list[str]:
    """Find healthy indexes whose manifest identifies the current repository."""
    root = project_path.resolve()
    indexes_dir = get_indexes_dir(root)
    if not indexes_dir.exists():
        return []
    matches: list[tuple[str, str]] = []
    for index_path in indexes_dir.iterdir():
        if not index_path.is_dir():
            continue
        try:
            manifest = IndexManifest.load(index_path)
        except IndexManifestError:
            continue
        if Path(manifest.repository_root).resolve() == root:
            matches.append((manifest.updated_at, manifest.index_name))
    return [name for _, name in sorted(matches, reverse=True)]


def _terms(text: str) -> list[str]:
    return [term.lower() for term in TOKEN_RE.findall(text) if len(term) > 1]


def _item(record: dict) -> ContextItem:
    metadata = record.get("metadata") or {}
    return ContextItem(
        id=record["id"],
        content=record.get("content", ""),
        file_path=metadata.get("file_path", "unknown"),
        start_line=int(metadata.get("start_line", 1)),
        end_line=int(metadata.get("end_line", 1)),
        chunk_type=metadata.get("chunk_type", "unknown"),
    )


class HybridRetriever:
    """Fuse semantic, lexical, symbol, and repository-structure signals."""

    def __init__(self, project_path: Path, embedding_provider, index_name: str | None = None):
        self.project_path = project_path.resolve()
        matches = discover_repository_indexes(self.project_path)
        self.index_name = index_name or (matches[0] if matches else None)
        if not self.index_name:
            raise LookupError("No index matches the current repository. Run 'ctxai index' first.")
        index_path = get_indexes_dir(self.project_path) / self.index_name
        manifest = IndexManifest.load(index_path)
        if Path(manifest.repository_root).resolve() != self.project_path:
            raise LookupError(f"Index '{self.index_name}' belongs to a different repository")
        self.store = VectorStore(index_path, self.index_name)
        self.embedding_provider = embedding_provider

    def retrieve(self, query: str, limit: int = 20, debug: bool = False) -> list[ContextItem]:
        records = self.store.get_chunks()
        if not records:
            return []
        query_terms = set(_terms(query))
        # Fuse over the complete local corpus (bounded by index size), then
        # apply the caller's result limit. Early truncation can hide an exact
        # symbol match merely because its vector rank is low.
        semantic = self.store.search(self.embedding_provider.generate_embedding(query), n_results=len(records))
        rankings: list[tuple[str, list[dict]]] = [("semantic", semantic)]

        lexical = sorted(
            records,
            key=lambda record: sum(_terms(record.get("content", "")).count(term) for term in query_terms),
            reverse=True,
        )
        lexical = [r for r in lexical if query_terms & set(_terms(r.get("content", "")))]
        rankings.append(("lexical", lexical))

        symbol = [
            record
            for record in records
            if query_terms & set(_terms((record.get("metadata") or {}).get("meta_name", "")))
        ]
        rankings.append(("symbol", symbol))

        # Repository-map signal: filenames and important definition types provide
        # useful structure even when a query's wording differs from code prose.
        structure = sorted(
            records,
            key=lambda record: (
                bool(query_terms & set(_terms((record.get("metadata") or {}).get("file_path", "")))),
                (record.get("metadata") or {}).get("chunk_type", "")
                in {"class_definition", "class_declaration", "function_definition", "function_declaration"},
            ),
            reverse=True,
        )
        rankings.append(("repository-map", structure))

        by_id = {record["id"]: _item(record) for record in records}
        for source, ranked in rankings:
            for rank, record in enumerate(ranked, 1):
                item = by_id[record["id"]]
                item.score += 1.0 / (60 + rank)
                if debug or source != "repository-map":
                    item.reasons.append(f"{source} rank {rank}")
        return sorted(by_id.values(), key=lambda item: (-item.score, item.file_path, item.start_line))[:limit]


class ContextAssembler:
    """Deduplicate and format ranked evidence within a hard approximate token budget."""

    def __init__(self, token_budget: int = 2000, debug: bool = False):
        if token_budget < 1:
            raise ValueError("token_budget must be positive")
        self.token_budget = token_budget
        self.debug = debug

    @staticmethod
    def estimate_tokens(text: str) -> int:
        return math.ceil(len(text) / 4)

    def assemble(self, index_name: str, items: list[ContextItem]) -> AssembledContext:
        selected: list[ContextItem] = []
        blocks: list[str] = []
        truncated_flags: list[bool] = []
        seen: set[tuple[str, int, int]] = set()
        used = 0
        for item in items:
            identity = (item.file_path, item.start_line, item.end_line)
            if identity in seen:
                continue
            reason = f"\nSelected because: {', '.join(item.reasons)}" if self.debug else ""
            header = f"[{item.citation}] ({item.chunk_type}){reason}\n"
            remaining_chars = (self.token_budget - used) * 4 - len(header)
            if remaining_chars <= 0:
                break
            content = item.content[:remaining_chars]
            block = header + content
            cost = self.estimate_tokens(block)
            if used + cost > self.token_budget:
                break
            blocks.append(block)
            selected.append(item)
            truncated_flags.append(len(content) < len(item.content))
            seen.add(identity)
            used += cost
        return AssembledContext(index_name, selected, "\n\n".join(blocks), used, tuple(truncated_flags))
