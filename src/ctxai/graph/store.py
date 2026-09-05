"""Transactional SQLite persistence for the symbol graph (IG-01).

One ``graph.sqlite3`` file lives inside the canonical index directory next to
the ChromaDB store and ``manifest.json``. Full publications build a temporary
database and atomically replace the live file; incremental publications run in
a single transaction so a failure always leaves the prior healthy graph
visible. All queries are parameterized and bounded.
"""

from __future__ import annotations

import json
import os
import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .model import (
    CONFIDENCE_UNRESOLVED,
    GRAPH_FILENAME,
    GRAPH_SCHEMA_VERSION,
    GraphEdge,
    GraphMetadata,
    GraphNode,
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS graph_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS nodes (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    qualified_name TEXT NOT NULL,
    display_name TEXT NOT NULL,
    language TEXT NOT NULL,
    file_path TEXT NOT NULL,
    start_line INTEGER NOT NULL,
    end_line INTEGER NOT NULL,
    parent_id TEXT REFERENCES nodes(id) ON DELETE CASCADE,
    visibility TEXT NOT NULL DEFAULT 'public',
    source_hash TEXT,
    adapter_version TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS edges (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    source_id TEXT NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
    target_id TEXT REFERENCES nodes(id) ON DELETE CASCADE,
    target_text TEXT,
    evidence_file TEXT NOT NULL,
    evidence_line INTEGER NOT NULL,
    confidence TEXT NOT NULL,
    resolver_version TEXT NOT NULL
);

-- Statically exported names per module (IG-02): lets import bindings resolve
-- through re-exports/default exports even in incremental rebuilds where the
-- exporting file was not re-extracted. One row per (module, exported name).
CREATE TABLE IF NOT EXISTS module_exports (
    module_name TEXT NOT NULL,
    name TEXT NOT NULL,
    target TEXT NOT NULL,
    file_path TEXT NOT NULL,
    language TEXT NOT NULL,
    adapter_version TEXT NOT NULL DEFAULT '',
    complete INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (module_name, name)
);

CREATE INDEX IF NOT EXISTS idx_nodes_qualified_name ON nodes(qualified_name);
CREATE INDEX IF NOT EXISTS idx_nodes_display_name ON nodes(display_name);
CREATE INDEX IF NOT EXISTS idx_nodes_file_path ON nodes(file_path);
CREATE INDEX IF NOT EXISTS idx_nodes_kind ON nodes(kind);
CREATE INDEX IF NOT EXISTS idx_nodes_parent ON nodes(parent_id);
CREATE INDEX IF NOT EXISTS idx_nodes_language ON nodes(language);
CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source_id);
CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target_id);
CREATE INDEX IF NOT EXISTS idx_edges_kind ON edges(kind);
CREATE INDEX IF NOT EXISTS idx_edges_kind_source ON edges(kind, source_id);
CREATE INDEX IF NOT EXISTS idx_edges_kind_target ON edges(kind, target_id);
CREATE INDEX IF NOT EXISTS idx_edges_target_text ON edges(target_text);
CREATE INDEX IF NOT EXISTS idx_module_exports_file ON module_exports(file_path);
"""

_META_SCHEMA_VERSION = "schema_version"
_META_EXTRACTOR = "extractor_version"
_META_RESOLVER = "resolver_version"
_META_LANGUAGES = "supported_languages"
_META_BUILT_AT = "built_at"
_META_GENERATION = "generation"
_META_NODE_COUNTS = "node_counts"
_META_EDGE_COUNTS = "edge_counts"
_META_UNRESOLVED = "unresolved_edges"
_META_ADAPTER_VERSIONS = "adapter_versions"


class GraphStoreError(RuntimeError):
    """Raised when the graph store cannot be read or updated reliably."""


class GraphSchemaError(GraphStoreError):
    """Raised when a store declares a different schema version than this build.

    Migrations are forward-only: a v1 (IG-01) store is never silently
    reinterpreted — callers rebuild it from source on the next index run.
    """


@dataclass
class NeighborResult:
    """Bounded traversal result around one start node.

    Attributes:
        start: The start node (when it exists).
        nodes: Reached nodes including the start node, deterministically
            ordered by id and bounded by ``limit``.
        edges: Traversed edges, deterministically ordered by id.
        truncated: Whether more nodes existed beyond the limit.
    """

    start: GraphNode | None
    nodes: list[GraphNode]
    edges: list[GraphEdge]
    truncated: bool


@dataclass(frozen=True)
class ModuleExport:
    """One statically exported module name (IG-02).

    Attributes:
        module_name: Qualified module name that exposes the export.
        name: Exported name (``"default"`` for ES default exports, ``"*"``
            for ``export *``).
        target: Qualified-name candidate the export points at.
        file_path: Repository-relative file that declares the export.
        language: Language of the declaring file.
        adapter_version: Language adapter version at extraction time.
        complete: Whether the module's export surface is statically complete.
    """

    module_name: str
    name: str
    target: str
    file_path: str
    language: str
    adapter_version: str
    complete: bool

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary.

        Returns:
            Dictionary of all fields.
        """
        return {
            "module_name": self.module_name,
            "name": self.name,
            "target": self.target,
            "file_path": self.file_path,
            "language": self.language,
            "adapter_version": self.adapter_version,
            "complete": self.complete,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ModuleExport:
        """Rebuild an export record from :meth:`to_dict` output.

        Args:
            payload: Dictionary produced by :meth:`to_dict`.

        Returns:
            The rebuilt ModuleExport.
        """
        return cls(
            module_name=payload["module_name"],
            name=payload["name"],
            target=payload["target"],
            file_path=payload["file_path"],
            language=payload["language"],
            adapter_version=payload.get("adapter_version", ""),
            complete=bool(payload.get("complete", False)),
        )


@dataclass(frozen=True)
class FileGraphIdentity:
    """Per-file graph identity: language and adapter version of its nodes."""

    file_path: str
    language: str
    adapter_version: str


def _fsync_directory(path: Path) -> None:
    fd = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _escape_like(text: str) -> str:
    return text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _node_row(node: GraphNode) -> tuple[Any, ...]:
    return (
        node.id,
        node.kind,
        node.qualified_name,
        node.display_name,
        node.language,
        node.file_path,
        node.start_line,
        node.end_line,
        node.parent_id,
        node.visibility,
        node.source_hash,
        node.adapter_version,
    )


def _edge_row(edge: GraphEdge) -> tuple[Any, ...]:
    return (
        edge.id,
        edge.kind,
        edge.source_id,
        edge.target_id,
        edge.target_text,
        edge.evidence_file,
        edge.evidence_line,
        edge.confidence,
        edge.resolver_version,
    )


def _node_from_row(row: sqlite3.Row) -> GraphNode:
    return GraphNode(
        id=row["id"],
        kind=row["kind"],
        qualified_name=row["qualified_name"],
        display_name=row["display_name"],
        language=row["language"],
        file_path=row["file_path"],
        start_line=row["start_line"],
        end_line=row["end_line"],
        parent_id=row["parent_id"],
        visibility=row["visibility"],
        source_hash=row["source_hash"],
        adapter_version=row["adapter_version"],
    )


def _edge_from_row(row: sqlite3.Row) -> GraphEdge:
    return GraphEdge(
        id=row["id"],
        kind=row["kind"],
        source_id=row["source_id"],
        target_id=row["target_id"],
        target_text=row["target_text"],
        evidence_file=row["evidence_file"],
        evidence_line=row["evidence_line"],
        confidence=row["confidence"],
        resolver_version=row["resolver_version"],
    )


class GraphStore:
    """Transactional SQLite-backed symbol graph store for one index."""

    def __init__(self, index_path: Path) -> None:
        """Point the store at one canonical index directory.

        Args:
            index_path: Directory holding ``manifest.json`` and ChromaDB.
        """
        self.index_path = Path(index_path)
        self.path = self.index_path / GRAPH_FILENAME

    def exists(self) -> bool:
        """Return whether the graph database file exists.

        Returns:
            True when ``graph.sqlite3`` is present.
        """
        return self.path.exists()

    # -- connections -------------------------------------------------------

    def _connect(self, target: Path | None = None) -> sqlite3.Connection:
        connection = sqlite3.connect(str(target or self.path), isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _wrap(self, exc: sqlite3.Error, action: str) -> GraphStoreError:
        return GraphStoreError(f"Graph store {action} failed for {self.path}: {exc}")

    # -- publication -------------------------------------------------------

    def replace_all(
        self,
        nodes: list[GraphNode],
        edges: list[GraphEdge],
        metadata: GraphMetadata,
        exports: list[ModuleExport] | tuple[ModuleExport, ...] = (),
    ) -> GraphMetadata:
        """Atomically replace the whole graph with the supplied content.

        Builds a temporary database, fsyncs it, then renames it over the live
        file; a failure at any point leaves the prior graph untouched.

        Args:
            nodes: Full node set to publish.
            edges: Full edge set to publish.
            metadata: Metadata to store (counts are recomputed from rows).
            exports: Full module export set to publish (IG-02).

        Returns:
            The published metadata with authoritative counts filled in.

        Raises:
            GraphStoreError: If the publication fails; the prior graph stays
                visible and no temporary files remain.
        """
        temporary = self.path.with_name(GRAPH_FILENAME + ".tmp")
        if temporary.exists():
            temporary.unlink()
        try:
            connection = self._connect(temporary)
            try:
                connection.executescript(_SCHEMA)
                connection.execute("BEGIN IMMEDIATE")
                self._insert_rows(connection, nodes, edges, exports)
                published = self._write_meta(connection, metadata)
                connection.execute("COMMIT")
            except sqlite3.Error as exc:
                connection.close()
                temporary.unlink(missing_ok=True)
                raise self._wrap(exc, "publication") from exc
            except Exception:
                connection.close()
                temporary.unlink(missing_ok=True)
                raise
            connection.close()
            handle = os.open(str(temporary), os.O_RDONLY)
            try:
                os.fsync(handle)
            finally:
                os.close(handle)
            os.replace(temporary, self.path)
            _fsync_directory(self.index_path)
            return published
        except sqlite3.Error as exc:
            temporary.unlink(missing_ok=True)
            raise self._wrap(exc, "publication") from exc

    def update_files(
        self,
        changed: dict[str, tuple[list[GraphNode], list[GraphEdge]]],
        deleted: list[str],
        metadata: GraphMetadata,
        exports: list[ModuleExport] | tuple[ModuleExport, ...] = (),
    ) -> GraphMetadata:
        """Replace the owned rows of changed files inside one transaction.

        Args:
            changed: Mapping of repository-relative file path to the new
                ``(nodes, edges)`` owned by that file.
            deleted: Repository-relative paths whose nodes must be removed.
            metadata: Metadata to store (counts are recomputed from rows).
            exports: Fresh module export rows for the changed files; rows of
                deleted and changed files are replaced (IG-02).

        Returns:
            The published metadata with authoritative counts filled in.

        Raises:
            GraphStoreError: If the transaction fails; it is rolled back and
                the prior graph and generation stay visible.
        """
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            for file_path in [*deleted, *changed.keys()]:
                connection.execute("DELETE FROM nodes WHERE file_path = ?", (file_path,))
                connection.execute("DELETE FROM module_exports WHERE file_path = ?", (file_path,))
            # Insert every node before any edge so cross-file edges never
            # reference a node that is not yet present in the transaction.
            all_nodes = [node for nodes, _ in changed.values() for node in nodes]
            all_edges = [edge for _, edges in changed.values() for edge in edges]
            self._insert_rows(connection, all_nodes, all_edges, exports)
            published = self._write_meta(connection, metadata)
            connection.execute("COMMIT")
            return published
        except sqlite3.Error as exc:
            self._safe_rollback(connection)
            raise self._wrap(exc, "update") from exc
        except Exception:
            self._safe_rollback(connection)
            raise
        finally:
            connection.close()

    @staticmethod
    def _insert_rows(
        connection: sqlite3.Connection,
        nodes: list[GraphNode],
        edges: list[GraphEdge],
        exports: list[ModuleExport] | tuple[ModuleExport, ...] = (),
    ) -> None:
        for node in sorted(nodes, key=lambda item: (item.file_path, item.qualified_name, item.id)):
            connection.execute(
                "INSERT INTO nodes (id, kind, qualified_name, display_name, language, file_path, start_line,"
                " end_line, parent_id, visibility, source_hash, adapter_version)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                _node_row(node),
            )
        for edge in sorted(edges, key=lambda item: item.id):
            connection.execute(
                "INSERT INTO edges (id, kind, source_id, target_id, target_text, evidence_file, evidence_line,"
                " confidence, resolver_version) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                _edge_row(edge),
            )
        for export in sorted(exports, key=lambda item: (item.module_name, item.name)):
            connection.execute(
                "INSERT INTO module_exports (module_name, name, target, file_path, language, adapter_version,"
                " complete) VALUES (?, ?, ?, ?, ?, ?, ?)"
                " ON CONFLICT(module_name, name) DO UPDATE SET target = excluded.target,"
                " file_path = excluded.file_path, language = excluded.language,"
                " adapter_version = excluded.adapter_version, complete = excluded.complete",
                (
                    export.module_name,
                    export.name,
                    export.target,
                    export.file_path,
                    export.language,
                    export.adapter_version,
                    1 if export.complete else 0,
                ),
            )

    @staticmethod
    def _safe_rollback(connection: sqlite3.Connection) -> None:
        try:
            connection.execute("ROLLBACK")
        except sqlite3.Error:
            pass

    def _write_meta(self, connection: sqlite3.Connection, metadata: GraphMetadata) -> GraphMetadata:
        node_counts = {
            row["kind"]: row["count"]
            for row in connection.execute("SELECT kind, COUNT(*) AS count FROM nodes GROUP BY kind")
        }
        edge_counts = {
            row["kind"]: row["count"]
            for row in connection.execute("SELECT kind, COUNT(*) AS count FROM edges GROUP BY kind")
        }
        unresolved = connection.execute(
            "SELECT COUNT(*) AS count FROM edges WHERE confidence = ?", (CONFIDENCE_UNRESOLVED,)
        ).fetchone()["count"]
        published = GraphMetadata(
            schema_version=metadata.schema_version,
            extractor_version=metadata.extractor_version,
            resolver_version=metadata.resolver_version,
            supported_languages=list(metadata.supported_languages),
            built_at=metadata.built_at,
            generation=metadata.generation,
            node_counts=node_counts,
            edge_counts=edge_counts,
            unresolved_edges=int(unresolved),
            adapter_versions=dict(metadata.adapter_versions),
        )
        values: dict[str, str] = {
            _META_SCHEMA_VERSION: str(published.schema_version),
            _META_EXTRACTOR: published.extractor_version,
            _META_RESOLVER: published.resolver_version,
            _META_LANGUAGES: ",".join(published.supported_languages),
            _META_BUILT_AT: published.built_at,
            _META_GENERATION: str(published.generation),
            _META_NODE_COUNTS: json.dumps(published.node_counts, sort_keys=True),
            _META_EDGE_COUNTS: json.dumps(published.edge_counts, sort_keys=True),
            _META_UNRESOLVED: str(published.unresolved_edges),
            _META_ADAPTER_VERSIONS: json.dumps(published.adapter_versions, sort_keys=True),
        }
        for key, value in values.items():
            connection.execute(
                "INSERT INTO graph_meta (key, value) VALUES (?, ?)"
                " ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, value),
            )
        return published

    # -- metadata ----------------------------------------------------------

    def read_metadata(self) -> GraphMetadata:
        """Read the stored graph metadata.

        Returns:
            The stored :class:`GraphMetadata`.

        Raises:
            GraphStoreError: If the file is missing, corrupt, or declares an
                unsupported schema version.
        """
        if not self.exists():
            raise GraphStoreError(f"Graph store is missing at {self.path}")
        try:
            connection = self._connect()
            try:
                rows = {row["key"]: row["value"] for row in connection.execute("SELECT key, value FROM graph_meta")}
                integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
            finally:
                connection.close()
        except sqlite3.Error as exc:
            raise GraphStoreError(f"Graph store at {self.path} is corrupt or unreadable: {exc}") from exc
        if integrity != "ok":
            raise GraphStoreError(f"Graph store at {self.path} is corrupt: integrity_check returned {integrity}")
        schema_version = rows.get(_META_SCHEMA_VERSION)
        if schema_version is None:
            raise GraphStoreError(f"Graph store at {self.path} is corrupt: schema version missing")
        if int(schema_version) != GRAPH_SCHEMA_VERSION:
            raise GraphSchemaError(
                f"Unsupported graph schema {schema_version}; expected {GRAPH_SCHEMA_VERSION}."
                " Rebuild the index (a v1 store is rebuilt on the next index run)."
            )
        try:
            node_counts = json.loads(rows.get(_META_NODE_COUNTS, "{}"))
            edge_counts = json.loads(rows.get(_META_EDGE_COUNTS, "{}"))
            adapter_versions = json.loads(rows.get(_META_ADAPTER_VERSIONS, "{}"))
            if (
                not isinstance(node_counts, dict)
                or not isinstance(edge_counts, dict)
                or not isinstance(adapter_versions, dict)
            ):
                raise ValueError("counts are not objects")
            return GraphMetadata(
                schema_version=int(schema_version),
                extractor_version=rows.get(_META_EXTRACTOR, ""),
                resolver_version=rows.get(_META_RESOLVER, ""),
                supported_languages=[item for item in rows.get(_META_LANGUAGES, "").split(",") if item],
                built_at=rows.get(_META_BUILT_AT, ""),
                generation=int(rows.get(_META_GENERATION, "0")),
                node_counts={str(key): int(value) for key, value in node_counts.items()},
                edge_counts={str(key): int(value) for key, value in edge_counts.items()},
                unresolved_edges=int(rows.get(_META_UNRESOLVED, "0")),
                adapter_versions={str(key): str(value) for key, value in adapter_versions.items()},
            )
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            raise GraphStoreError(f"Graph store at {self.path} is corrupt: metadata unreadable ({exc})") from exc

    def integrity_check(self) -> str:
        """Run SQLite's integrity check.

        Returns:
            The pragma result string (``ok`` for a healthy store).

        Raises:
            GraphStoreError: If the store is missing or unreadable.
        """
        if not self.exists():
            raise GraphStoreError(f"Graph store is missing at {self.path}")
        try:
            connection = self._connect()
            try:
                return str(connection.execute("PRAGMA integrity_check").fetchone()[0])
            finally:
                connection.close()
        except sqlite3.Error as exc:
            raise GraphStoreError(f"Graph store at {self.path} is corrupt: {exc}") from exc

    # -- counts and lookups ------------------------------------------------

    def count_nodes(self) -> int:
        """Count all stored nodes.

        Returns:
            Total node count.

        Raises:
            GraphStoreError: If the store is unreadable.
        """
        return self._scalar("SELECT COUNT(*) AS count FROM nodes")

    def count_edges(self) -> int:
        """Count all stored edges.

        Returns:
            Total edge count.

        Raises:
            GraphStoreError: If the store is unreadable.
        """
        return self._scalar("SELECT COUNT(*) AS count FROM edges")

    def _scalar(self, sql: str, parameters: tuple = ()) -> int:
        if not self.exists():
            raise GraphStoreError(f"Graph store is missing at {self.path}")
        try:
            connection = self._connect()
            try:
                return int(connection.execute(sql, parameters).fetchone()[0])
            finally:
                connection.close()
        except sqlite3.Error as exc:
            raise GraphStoreError(f"Graph store query failed at {self.path}: {exc}") from exc

    def get_node(self, node_id: str) -> GraphNode | None:
        """Fetch one node by id.

        Args:
            node_id: Stable node id.

        Returns:
            The node, or ``None`` when absent.

        Raises:
            GraphStoreError: If the store is unreadable.
        """
        return self._node_query("SELECT * FROM nodes WHERE id = ?", (node_id,))

    def _node_query(self, sql: str, parameters: tuple) -> GraphNode | None:
        if not self.exists():
            raise GraphStoreError(f"Graph store is missing at {self.path}")
        try:
            connection = self._connect()
            try:
                row = connection.execute(sql, parameters).fetchone()
            finally:
                connection.close()
        except sqlite3.Error as exc:
            raise GraphStoreError(f"Graph store query failed at {self.path}: {exc}") from exc
        return _node_from_row(row) if row is not None else None

    def find_nodes(
        self,
        text: str,
        kind: str | None = None,
        language: str | None = None,
        limit: int = 20,
    ) -> list[GraphNode]:
        """Find nodes whose qualified or display name contains ``text``.

        Args:
            text: Case-insensitive substring to search for.
            kind: Optional exact node kind filter.
            language: Optional exact language filter.
            limit: Maximum number of nodes to return.

        Returns:
            Nodes ordered deterministically by qualified name.

        Raises:
            GraphStoreError: If the store is unreadable.
        """
        if not self.exists():
            raise GraphStoreError(f"Graph store is missing at {self.path}")
        clauses = ["(qualified_name LIKE ? ESCAPE '\\' OR display_name LIKE ? ESCAPE '\\')"]
        parameters: list[Any] = [f"%{_escape_like(text)}%", f"%{_escape_like(text)}%"]
        if kind is not None:
            clauses.append("kind = ?")
            parameters.append(kind)
        if language is not None:
            clauses.append("language = ?")
            parameters.append(language)
        parameters.append(limit)
        sql = "SELECT * FROM nodes WHERE " + " AND ".join(clauses) + " ORDER BY qualified_name, file_path LIMIT ?"
        try:
            connection = self._connect()
            try:
                rows = connection.execute(sql, parameters).fetchall()
            finally:
                connection.close()
        except sqlite3.Error as exc:
            raise GraphStoreError(f"Graph store query failed at {self.path}: {exc}") from exc
        return [_node_from_row(row) for row in rows]

    def find_node_ids_by_prefix(self, prefix: str) -> list[str]:
        """Find node ids starting with the given prefix.

        Args:
            prefix: Stable-id prefix.

        Returns:
            Matching ids in deterministic order.

        Raises:
            GraphStoreError: If the store is unreadable.
        """
        if not self.exists():
            raise GraphStoreError(f"Graph store is missing at {self.path}")
        sql = "SELECT id FROM nodes WHERE id LIKE ? ESCAPE '\\' ORDER BY id LIMIT 10"
        try:
            connection = self._connect()
            try:
                rows = connection.execute(sql, (f"{_escape_like(prefix)}%",)).fetchall()
            finally:
                connection.close()
        except sqlite3.Error as exc:
            raise GraphStoreError(f"Graph store query failed at {self.path}: {exc}") from exc
        return [row["id"] for row in rows]

    def iter_nodes(self) -> list[GraphNode]:
        """Return every stored node (used for incremental re-resolution).

        Returns:
            All nodes ordered by id.

        Raises:
            GraphStoreError: If the store is unreadable.
        """
        if not self.exists():
            raise GraphStoreError(f"Graph store is missing at {self.path}")
        try:
            connection = self._connect()
            try:
                rows = connection.execute("SELECT * FROM nodes ORDER BY id").fetchall()
            finally:
                connection.close()
        except sqlite3.Error as exc:
            raise GraphStoreError(f"Graph store query failed at {self.path}: {exc}") from exc
        return [_node_from_row(row) for row in rows]

    def file_identities(self) -> dict[str, FileGraphIdentity]:
        """Report the language and adapter version of every file's nodes (IG-02).

        Used to detect files whose stored adapter version no longer matches
        the current adapter so only affected files become stale.

        Returns:
            Mapping of repository-relative file path to its graph identity.

        Raises:
            GraphStoreError: If the store is unreadable.
        """
        if not self.exists():
            raise GraphStoreError(f"Graph store is missing at {self.path}")
        try:
            connection = self._connect()
            try:
                rows = connection.execute(
                    "SELECT DISTINCT file_path, language, adapter_version FROM nodes ORDER BY file_path"
                ).fetchall()
            finally:
                connection.close()
        except sqlite3.Error as exc:
            raise GraphStoreError(f"Graph store query failed at {self.path}: {exc}") from exc
        return {
            row["file_path"]: FileGraphIdentity(
                file_path=row["file_path"], language=row["language"], adapter_version=row["adapter_version"]
            )
            for row in rows
        }

    def language_node_counts(self) -> dict[str, int]:
        """Count stored nodes per language (IG-02).

        Returns:
            Mapping of language to node count, ordered by language.

        Raises:
            GraphStoreError: If the store is unreadable.
        """
        return {
            str(row["language"]): int(row["count"])
            for row in self._rows("SELECT language, COUNT(*) AS count FROM nodes GROUP BY language ORDER BY language")
        }

    def unresolved_counts_by_kind(self) -> dict[str, int]:
        """Count unresolved edges per edge kind (IG-02).

        Returns:
            Mapping of edge kind to unresolved edge count (kinds with zero
            unresolved edges are omitted), ordered by kind.

        Raises:
            GraphStoreError: If the store is unreadable.
        """
        return {
            str(row["kind"]): int(row["count"])
            for row in self._rows(
                "SELECT kind, COUNT(*) AS count FROM edges WHERE confidence = ? GROUP BY kind ORDER BY kind",
                (CONFIDENCE_UNRESOLVED,),
            )
        }

    def export_maps(self) -> dict[str, tuple[dict[str, str], set[str], bool]]:
        """Read the persisted module export surface (IG-02).

        Returns:
            Mapping of module name to ``(bindings, files, complete)`` where
            ``bindings`` maps exported names to qualified-name candidates,
            ``files`` is the set of repository-relative declaring files, and
            ``complete`` says whether the export surface is statically
            complete.

        Raises:
            GraphStoreError: If the store is unreadable.
        """
        if not self.exists():
            raise GraphStoreError(f"Graph store is missing at {self.path}")
        try:
            connection = self._connect()
            try:
                rows = connection.execute(
                    "SELECT module_name, name, target, file_path, complete FROM module_exports"
                    " ORDER BY module_name, name"
                ).fetchall()
            finally:
                connection.close()
        except sqlite3.Error as exc:
            raise GraphStoreError(f"Graph store query failed at {self.path}: {exc}") from exc
        maps: dict[str, tuple[dict[str, str], set[str], bool]] = {}
        for row in rows:
            bindings, files, complete = maps.setdefault(row["module_name"], ({}, set(), True))
            bindings[row["name"]] = row["target"]
            files.add(row["file_path"])
            maps[row["module_name"]] = (bindings, files, complete and bool(row["complete"]))
        return maps

    def _rows(self, sql: str, parameters: tuple = ()) -> list[sqlite3.Row]:
        if not self.exists():
            raise GraphStoreError(f"Graph store is missing at {self.path}")
        try:
            connection = self._connect()
            try:
                return list(connection.execute(sql, parameters).fetchall())
            finally:
                connection.close()
        except sqlite3.Error as exc:
            raise GraphStoreError(f"Graph store query failed at {self.path}: {exc}") from exc

    def files_with_edges_into(self, file_paths: Iterable[str]) -> set[str]:
        """Find files owning edges that point into the supplied files' nodes.

        Used to discover the dependent files that must be re-extracted when
        another file's nodes are replaced.

        Args:
            file_paths: Repository-relative paths whose owned nodes changed.

        Returns:
            Set of repository-relative paths owning dependent source nodes.

        Raises:
            GraphStoreError: If the store is unreadable.
        """
        paths = sorted(set(file_paths))
        if not paths or not self.exists():
            return set()
        placeholders = ", ".join("?" for _ in paths)
        sql = (
            "SELECT DISTINCT source.file_path AS owner FROM edges"
            " JOIN nodes AS target ON edges.target_id = target.id"
            " JOIN nodes AS source ON edges.source_id = source.id"
            f" WHERE target.file_path IN ({placeholders})"
        )
        try:
            connection = self._connect()
            try:
                rows = connection.execute(sql, paths).fetchall()
            finally:
                connection.close()
        except sqlite3.Error as exc:
            raise GraphStoreError(f"Graph store query failed at {self.path}: {exc}") from exc
        return {row["owner"] for row in rows}

    def cross_file_relationship_files(
        self,
        file_paths: Iterable[str],
        edge_kinds: Iterable[str],
    ) -> set[str]:
        """Report which supplied files participate in cross-file relationships.

        A file qualifies when it owns (or hosts the target of) a resolved edge
        of one of ``edge_kinds`` whose other endpoint lives in a different
        file. Used by the retrieval benchmark to derive the pre-registered
        relationship-oriented case cohort (IG-03).

        Args:
            file_paths: Repository-relative candidate file paths.
            edge_kinds: Edge kinds counted as relationships (validated by the
                caller; every query is parameterized).

        Returns:
            The subset of ``file_paths`` with at least one cross-file
            relationship edge.

        Raises:
            GraphStoreError: If the store is unreadable.
        """
        paths = sorted(set(file_paths))
        kinds = sorted(set(edge_kinds))
        if not paths or not kinds or not self.exists():
            return set()
        kind_placeholders = ", ".join("?" for _ in kinds)
        path_placeholders = ", ".join("?" for _ in paths)
        # Two UNION branches, each parameterized as kinds + paths.
        parameters = [*kinds, *paths, *kinds, *paths]
        sql = (
            "SELECT DISTINCT source.file_path AS related FROM edges"
            " JOIN nodes AS source ON source.id = edges.source_id"
            " JOIN nodes AS target ON target.id = edges.target_id"
            f" WHERE edges.kind IN ({kind_placeholders})"
            " AND target.file_path != source.file_path"
            f" AND source.file_path IN ({path_placeholders})"
            " UNION"
            " SELECT DISTINCT target.file_path FROM edges"
            " JOIN nodes AS source ON source.id = edges.source_id"
            " JOIN nodes AS target ON target.id = edges.target_id"
            f" WHERE edges.kind IN ({kind_placeholders})"
            " AND target.file_path != source.file_path"
            f" AND target.file_path IN ({path_placeholders})"
        )
        try:
            connection = self._connect()
            try:
                rows = connection.execute(sql, parameters).fetchall()
            finally:
                connection.close()
        except sqlite3.Error as exc:
            raise GraphStoreError(f"Graph store query failed at {self.path}: {exc}") from exc
        return {row["related"] for row in rows}

    def neighbors(
        self,
        node_id: str,
        edge_kind: str | None = None,
        direction: str = "both",
        depth: int = 1,
        limit: int = 50,
    ) -> NeighborResult:
        """Traverse up to ``depth`` hops around a node, bounded and parameterized.

        Args:
            node_id: Start node id.
            edge_kind: Optional edge kind filter.
            direction: ``"in"``, ``"out"``, or ``"both"``.
            depth: Number of hops (1-based, bounded by the caller).
            limit: Maximum number of returned nodes (including the start).

        Returns:
            A :class:`NeighborResult`; an unknown start node yields an empty
            result.

        Raises:
            GraphStoreError: If the store is unreadable.
        """
        if not self.exists():
            raise GraphStoreError(f"Graph store is missing at {self.path}")
        try:
            connection = self._connect()
            try:
                start_row = connection.execute("SELECT * FROM nodes WHERE id = ?", (node_id,)).fetchone()
                if start_row is None:
                    return NeighborResult(start=None, nodes=[], edges=[], truncated=False)
                start = _node_from_row(start_row)
                visited: dict[str, GraphNode] = {start.id: start}
                edge_map: dict[str, GraphEdge] = {}
                frontier = [start.id]
                truncated = False
                for _ in range(max(0, depth)):
                    next_frontier: list[str] = []
                    placeholders = ", ".join("?" for _ in frontier)
                    if direction in ("out", "both"):
                        sql = f"SELECT * FROM edges WHERE source_id IN ({placeholders})"
                        parameters: list[Any] = list(frontier)
                        if edge_kind is not None:
                            sql += " AND kind = ?"
                            parameters.append(edge_kind)
                        for row in connection.execute(sql, parameters):
                            edge = _edge_from_row(row)
                            edge_map.setdefault(edge.id, edge)
                            if edge.target_id and edge.target_id not in visited:
                                next_frontier.append(edge.target_id)
                    if direction in ("in", "both"):
                        sql = f"SELECT * FROM edges WHERE target_id IN ({placeholders})"
                        parameters = list(frontier)
                        if edge_kind is not None:
                            sql += " AND kind = ?"
                            parameters.append(edge_kind)
                        for row in connection.execute(sql, parameters):
                            edge = _edge_from_row(row)
                            edge_map.setdefault(edge.id, edge)
                            if edge.source_id not in visited:
                                next_frontier.append(edge.source_id)
                    for reached in sorted(set(next_frontier)):
                        row = connection.execute("SELECT * FROM nodes WHERE id = ?", (reached,)).fetchone()
                        if row is not None:
                            visited[reached] = _node_from_row(row)
                    if not next_frontier:
                        break
                    frontier = sorted(set(next_frontier))
            finally:
                connection.close()
        except sqlite3.Error as exc:
            raise GraphStoreError(f"Graph store query failed at {self.path}: {exc}") from exc
        ordered_nodes = sorted(visited.values(), key=lambda item: item.id)
        truncated = len(ordered_nodes) > limit
        result_nodes = ordered_nodes[:limit]
        keep_ids = {node.id for node in result_nodes}
        result_edges = sorted(
            (
                edge
                for edge in edge_map.values()
                if edge.source_id in keep_ids and (edge.target_id is None or edge.target_id in keep_ids)
            ),
            key=lambda item: item.id,
        )
        return NeighborResult(start=start, nodes=result_nodes, edges=result_edges, truncated=truncated)
