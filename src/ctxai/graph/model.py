"""Symbol graph data model and stable identity derivation (IG-01).

Part II contract: :class:`GraphNode`, :class:`GraphEdge`, and
:class:`GraphMetadata` are schema-versioned records with round-trip
serialization. Stable IDs derive from repository identity plus canonical
source identity (a sha256 over the repository root, the repository-relative
file path, the record kind, and the qualified name) — never from a database
sequence — so identical inputs always produce identical identities.
"""

from __future__ import annotations

import hashlib
from dataclasses import MISSING, dataclass, field, fields
from typing import Any

GRAPH_SCHEMA_VERSION = 2
GRAPH_FILENAME = "graph.sqlite3"

NODE_KINDS = ("module", "class", "function", "method", "interface", "test")
EDGE_KINDS = ("contains", "imports", "calls", "inherits", "references", "tests")

# Node kinds each language adapter can emit (IG-02 capability matrix).
GRAPH_NODE_KINDS_BY_LANGUAGE = {
    "python": ("module", "class", "function", "method", "test"),
    "javascript": ("module", "class", "function", "method", "test"),
    "typescript": NODE_KINDS,
}

CONFIDENCE_EXACT = "exact"
CONFIDENCE_PROBABLE = "probable"
CONFIDENCE_UNRESOLVED = "unresolved"
CONFIDENCES = (CONFIDENCE_EXACT, CONFIDENCE_PROBABLE, CONFIDENCE_UNRESOLVED)

PYTHON_EXTRACTOR_VERSION = "python/1"
PYTHON_RESOLVER_VERSION = "python/1"
JAVASCRIPT_EXTRACTOR_VERSION = "javascript/1"
JAVASCRIPT_RESOLVER_VERSION = "javascript/1"
TYPESCRIPT_EXTRACTOR_VERSION = "typescript/1"
TYPESCRIPT_RESOLVER_VERSION = "typescript/1"
SUPPORTED_LANGUAGES = ("python", "javascript", "typescript")

# Safety bounds for graph queries (enforced by GraphOperations, documented in
# docs/SYMBOL_GRAPH.md).
MAX_SYMBOL_QUERY_LENGTH = 200
MAX_TRAVERSAL_DEPTH = 3
MAX_RESULT_LIMIT = 500


def _sha256(identity: str) -> str:
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def derive_node_id(repository_root: str, file_path: str, kind: str, qualified_name: str) -> str:
    """Derive a stable node id from repository and source identity.

    Args:
        repository_root: Canonical repository root (repository identity).
        file_path: Repository-relative file path of the definition.
        kind: Node kind (one of NODE_KINDS).
        qualified_name: Fully qualified name of the definition.

    Returns:
        Hex sha256 digest over the canonical identity tuple.
    """
    identity = "\0".join(["node", repository_root, file_path, kind, qualified_name])
    return _sha256(identity)


def derive_edge_id(
    repository_root: str,
    kind: str,
    source_id: str,
    target_id: str | None,
    target_text: str | None,
    evidence_file: str,
    evidence_line: int,
) -> str:
    """Derive a stable edge id from its endpoints and evidence.

    Args:
        repository_root: Canonical repository root (repository identity).
        kind: Edge kind (one of EDGE_KINDS).
        source_id: Source node id.
        target_id: Target node id when resolved, otherwise ``None``.
        target_text: Unresolved target text when unresolved, otherwise ``None``.
        evidence_file: Repository-relative evidence file.
        evidence_line: 1-based evidence line.

    Returns:
        Hex sha256 digest over the canonical identity tuple.
    """
    identity = "\0".join(
        [
            "edge",
            repository_root,
            kind,
            source_id,
            target_id or "",
            target_text or "",
            evidence_file,
            str(evidence_line),
        ]
    )
    return _sha256(identity)


def _validated(value: str, allowed: tuple[str, ...], label: str) -> str:
    if value not in allowed:
        raise ValueError(f"Invalid {label} {value!r}; expected one of {', '.join(allowed)}")
    return value


@dataclass
class GraphNode:
    """One symbol definition with repository-relative source evidence."""

    id: str
    kind: str
    qualified_name: str
    display_name: str
    language: str
    file_path: str
    start_line: int
    end_line: int
    parent_id: str | None = None
    visibility: str = "public"
    source_hash: str | None = None
    adapter_version: str = ""

    def evidence(self) -> str:
        """Return the ``file:start-end`` evidence citation.

        Returns:
            Repository-relative ``file:start-end`` string.
        """
        return f"{self.file_path}:{self.start_line}-{self.end_line}"

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary.

        Returns:
            Dictionary of all dataclass fields.
        """
        return {
            "id": self.id,
            "kind": self.kind,
            "qualified_name": self.qualified_name,
            "display_name": self.display_name,
            "language": self.language,
            "file_path": self.file_path,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "parent_id": self.parent_id,
            "visibility": self.visibility,
            "source_hash": self.source_hash,
            "adapter_version": self.adapter_version,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> GraphNode:
        """Rebuild a node from :meth:`to_dict` output.

        Fields that carry defaults may be absent from the payload (IG-01
        payloads without ``adapter_version`` still load).

        Args:
            payload: Dictionary produced by :meth:`to_dict`.

        Returns:
            The rebuilt GraphNode.

        Raises:
            ValueError: If the kind is unknown or required fields are absent.
        """
        field_names = {item.name for item in fields(cls)}
        optional = {
            item.name for item in fields(cls) if item.default is not MISSING or item.default_factory is not MISSING
        }
        missing = field_names - set(payload) - optional
        if missing:
            raise ValueError(f"GraphNode payload is missing fields: {sorted(missing)}")
        return cls(
            **{
                **payload,
                "kind": _validated(payload["kind"], NODE_KINDS, "kind"),
            }
        )


@dataclass
class GraphEdge:
    """One structural relationship with evidence and resolution confidence."""

    id: str
    kind: str
    source_id: str
    target_id: str | None
    target_text: str | None
    evidence_file: str
    evidence_line: int
    confidence: str
    resolver_version: str

    def evidence(self) -> str:
        """Return the ``file:line`` evidence citation.

        Returns:
            Repository-relative ``file:line`` string.
        """
        return f"{self.evidence_file}:{self.evidence_line}"

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary.

        Returns:
            Dictionary of all dataclass fields.
        """
        return {
            "id": self.id,
            "kind": self.kind,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "target_text": self.target_text,
            "evidence_file": self.evidence_file,
            "evidence_line": self.evidence_line,
            "confidence": self.confidence,
            "resolver_version": self.resolver_version,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> GraphEdge:
        """Rebuild an edge from :meth:`to_dict` output.

        Args:
            payload: Dictionary produced by :meth:`to_dict`.

        Returns:
            The rebuilt GraphEdge.

        Raises:
            ValueError: If the kind or confidence is unknown or fields are absent.
        """
        field_names = {item.name for item in fields(cls)}
        missing = field_names - set(payload)
        if missing:
            raise ValueError(f"GraphEdge payload is missing fields: {sorted(missing)}")
        return cls(
            **{
                **payload,
                "kind": _validated(payload["kind"], EDGE_KINDS, "kind"),
                "confidence": _validated(payload["confidence"], CONFIDENCES, "confidence"),
            }
        )


@dataclass
class GraphMetadata:
    """Versioned build identity and health summary for one graph generation.

    ``extractor_version``/``resolver_version`` are deterministic comma-joined
    descriptors of every language adapter that contributed (sorted by
    language, e.g. ``"javascript/1,python/1,typescript/1"``);
    ``adapter_versions`` records the per-language versions so an adapter
    upgrade marks only the files of that language stale.
    """

    schema_version: int
    extractor_version: str
    resolver_version: str
    supported_languages: list[str]
    built_at: str
    generation: int
    node_counts: dict[str, int] = field(default_factory=dict)
    edge_counts: dict[str, int] = field(default_factory=dict)
    unresolved_edges: int = 0
    adapter_versions: dict[str, str] = field(default_factory=dict)

    @property
    def total_nodes(self) -> int:
        """Total node count across kinds.

        Returns:
            Sum of per-kind node counts.
        """
        return sum(self.node_counts.values())

    @property
    def total_edges(self) -> int:
        """Total edge count across kinds.

        Returns:
            Sum of per-kind edge counts.
        """
        return sum(self.edge_counts.values())

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary.

        Returns:
            Dictionary of all dataclass fields.
        """
        return {
            "schema_version": self.schema_version,
            "extractor_version": self.extractor_version,
            "resolver_version": self.resolver_version,
            "supported_languages": list(self.supported_languages),
            "built_at": self.built_at,
            "generation": self.generation,
            "node_counts": dict(self.node_counts),
            "edge_counts": dict(self.edge_counts),
            "unresolved_edges": self.unresolved_edges,
            "adapter_versions": dict(self.adapter_versions),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> GraphMetadata:
        """Rebuild metadata from :meth:`to_dict` output.

        Args:
            payload: Dictionary produced by :meth:`to_dict`.

        Returns:
            The rebuilt GraphMetadata.

        Raises:
            ValueError: If required fields are absent.
        """
        field_names = {item.name for item in fields(cls)}
        optional = {
            item.name for item in fields(cls) if item.default is not MISSING or item.default_factory is not MISSING
        }
        missing = field_names - set(payload) - optional
        if missing:
            raise ValueError(f"GraphMetadata payload is missing fields: {sorted(missing)}")
        payload = dict(payload)
        payload["supported_languages"] = list(payload["supported_languages"])
        payload["node_counts"] = dict(payload["node_counts"])
        payload["edge_counts"] = dict(payload["edge_counts"])
        payload.setdefault("adapter_versions", {})
        payload["adapter_versions"] = {str(key): str(value) for key, value in payload["adapter_versions"].items()}
        return cls(**payload)


def combined_adapter_version(versions: dict[str, str]) -> str:
    """Join per-language adapter versions into one deterministic descriptor.

    Args:
        versions: Mapping of language to adapter version (e.g.
            ``{"python": "python/1"}``).

    Returns:
        Comma-joined versions sorted by language (empty string when empty).
    """
    return ",".join(versions[language] for language in sorted(versions))
