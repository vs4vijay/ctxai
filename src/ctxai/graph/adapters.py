"""Language adapter registry and capability reporting (IG-02).

The registry is the single place that maps languages and file extensions to
language adapters. Adapters never query graph storage: they extract structural
records from source text and resolve them against shared symbol indexes. All
user-facing surfaces (CLI, MCP, dashboard) read the graph only through
:class:`ctxai.graph.operations.GraphOperations`.

The capability constants in this module drive the versioned capabilities
report (CLI ``ctxai graph capabilities``, MCP ``graph_stats`` payload, and the
dashboard) and the generated support matrix published in
``docs/SYMBOL_GRAPH.md`` (kept in sync by ``tests/test_graph_docs_sync.py``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from .js_adapter import JavaScriptAdapter, TypeScriptAdapter
from .model import (
    EDGE_KINDS,
    GRAPH_NODE_KINDS_BY_LANGUAGE,
    NODE_KINDS,
    SUPPORTED_LANGUAGES,
    GraphEdge,
)
from .python_adapter import PythonAdapter
from .resolution import FileExtraction, SymbolIndexes

CAPABILITIES_SCHEMA_VERSION = 1


@runtime_checkable
class LanguageAdapter(Protocol):
    """Structural contract every language adapter satisfies."""

    language: str
    extractor_version: str
    resolver_version: str

    def extract_file(self, relative_path: str, source: bytes, repository_root: str) -> FileExtraction:
        """Parse one file into structural graph records."""
        ...

    @classmethod
    def resolve_edges(
        cls, extractions: list[FileExtraction], indexes: SymbolIndexes, repository_root: str
    ) -> list[GraphEdge]:
        """Resolve extraction records into final graph edges."""
        ...

    @staticmethod
    def supports_file(relative_path: str) -> bool:
        """Return whether the adapter handles this repository-relative path."""
        ...


@dataclass(frozen=True)
class LanguageCapability:
    """Static support matrix entry for one language (versioned DTO).

    Attributes:
        language: Language name (one of SUPPORTED_LANGUAGES when supported).
        supported: Whether this build ships an adapter for the language.
        adapter_version: Adapter version when supported, otherwise ``None``.
        file_extensions: File extensions the adapter consumes.
        node_kinds: Node kinds the adapter can emit.
        edge_kinds: Edge kinds the adapter can emit.
        supported_constructs: Constructs resolved into nodes/edges.
        unsupported_constructs: Constructs that explicitly stay unresolved
            (or unrecorded) rather than guessed.
    """

    language: str
    supported: bool
    adapter_version: str | None
    file_extensions: tuple[str, ...]
    node_kinds: tuple[str, ...]
    edge_kinds: tuple[str, ...]
    supported_constructs: tuple[str, ...]
    unsupported_constructs: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary.

        Returns:
            Dictionary of all fields with tuple values as lists.
        """
        return {
            "language": self.language,
            "supported": self.supported,
            "adapter_version": self.adapter_version,
            "file_extensions": list(self.file_extensions),
            "node_kinds": list(self.node_kinds),
            "edge_kinds": list(self.edge_kinds),
            "supported_constructs": list(self.supported_constructs),
            "unsupported_constructs": list(self.unsupported_constructs),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> LanguageCapability:
        """Rebuild a capability entry from :meth:`to_dict` output.

        Args:
            payload: Dictionary produced by :meth:`to_dict`.

        Returns:
            The rebuilt LanguageCapability.
        """
        return cls(
            language=payload["language"],
            supported=bool(payload["supported"]),
            adapter_version=payload.get("adapter_version"),
            file_extensions=tuple(payload["file_extensions"]),
            node_kinds=tuple(payload["node_kinds"]),
            edge_kinds=tuple(payload["edge_kinds"]),
            supported_constructs=tuple(payload["supported_constructs"]),
            unsupported_constructs=tuple(payload["unsupported_constructs"]),
        )


@dataclass(frozen=True)
class GraphCapabilities:
    """Versioned capability report for the graph package (versioned DTO).

    Attributes:
        schema_version: Capabilities payload schema version.
        languages: Per-language support matrix entries.
        node_kinds: The closed node-kind vocabulary.
        edge_kinds: The closed edge-kind vocabulary.
    """

    schema_version: int
    languages: tuple[LanguageCapability, ...]
    node_kinds: tuple[str, ...] = NODE_KINDS
    edge_kinds: tuple[str, ...] = EDGE_KINDS

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary.

        Returns:
            Dictionary of all fields with tuple values as lists.
        """
        return {
            "schema_version": self.schema_version,
            "languages": [item.to_dict() for item in self.languages],
            "node_kinds": list(self.node_kinds),
            "edge_kinds": list(self.edge_kinds),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> GraphCapabilities:
        """Rebuild a capabilities report from :meth:`to_dict` output.

        Args:
            payload: Dictionary produced by :meth:`to_dict`.

        Returns:
            The rebuilt GraphCapabilities.
        """
        return cls(
            schema_version=int(payload["schema_version"]),
            languages=tuple(LanguageCapability.from_dict(item) for item in payload["languages"]),
            node_kinds=tuple(payload.get("node_kinds", NODE_KINDS)),
            edge_kinds=tuple(payload.get("edge_kinds", EDGE_KINDS)),
        )


_COMMON_SUPPORTED_CONSTRUCTS = (
    "modules",
    "classes",
    "functions",
    "methods",
    "nested definitions",
    "inheritance",
    "statically named calls",
    "statically named references",
    "test detection (name-based)",
    "file:start-end evidence with confidence",
)

_PYTHON_SUPPORTED_CONSTRUCTS = (
    *_COMMON_SUPPORTED_CONSTRUCTS,
    "imports (absolute, relative, aliased, from-imports)",
    "decorators",
)

_JAVASCRIPT_SUPPORTED_CONSTRUCTS = (
    *_COMMON_SUPPORTED_CONSTRUCTS,
    "ES imports (default, named, namespace, side-effect)",
    "CommonJS require (literal and destructured)",
    "re-exports (named and star)",
    "default exports (named declarations and identifiers)",
    "extends/implements inheritance",
    "this-calls within the enclosing class",
    "test detection (test file locations)",
)

_TYPESCRIPT_SUPPORTED_CONSTRUCTS = (
    *_JAVASCRIPT_SUPPORTED_CONSTRUCTS,
    "interfaces and type aliases",
    "function overloads (implementation node wins)",
    "abstract classes",
)

_PYTHON_UNSUPPORTED_CONSTRUCTS = (
    "dynamic imports (__import__, importlib)",
    "wildcard import * targets",
    "calls through call results (factory()())",
    "calls through locals/attributes of unknown objects",
    "monkey-patching, reflection, and generated code",
    "cross-language call resolution",
)

_JAVASCRIPT_UNSUPPORTED_CONSTRUCTS = (
    "require/import with non-literal or template-path arguments",
    "computed member access (a[name]())",
    "anonymous default exports",
    "calls through call results (factory()())",
    "calls through locals/attributes of unknown objects",
    "cross-language call resolution",
)

_TYPESCRIPT_UNSUPPORTED_CONSTRUCTS = (
    *_JAVASCRIPT_UNSUPPORTED_CONSTRUCTS,
    "type-level and namespace-qualified resolution",
    "enum members (enums emit no nodes)",
)

_CAPABILITY_CONSTRUCTS = {
    "python": (_PYTHON_SUPPORTED_CONSTRUCTS, _PYTHON_UNSUPPORTED_CONSTRUCTS),
    "javascript": (_JAVASCRIPT_SUPPORTED_CONSTRUCTS, _JAVASCRIPT_UNSUPPORTED_CONSTRUCTS),
    "typescript": (_TYPESCRIPT_SUPPORTED_CONSTRUCTS, _TYPESCRIPT_UNSUPPORTED_CONSTRUCTS),
}

_LANGUAGES_WITHOUT_ADAPTERS = ("java", "ruby", "go", "rust", "csharp", "php", "kotlin", "swift")


def all_adapters() -> tuple[type, ...]:
    """Return every adapter class in the registry.

    Returns:
        Adapter classes ordered by language name.
    """
    return (JavaScriptAdapter, PythonAdapter, TypeScriptAdapter)


def get_adapter(language: str) -> Any | None:
    """Return an adapter instance for a language name.

    Args:
        language: Language name (e.g. ``"python"``, ``"javascript"``).

    Returns:
        A shared adapter instance, or ``None`` when the language has no
        adapter.
    """
    for adapter_class in all_adapters():
        if adapter_class.language == language:
            return adapter_class()
    return None


def language_for_file(relative_path: str) -> str | None:
    """Detect the graph language for a repository-relative path.

    Args:
        relative_path: Repository-relative path with forward slashes.

    Returns:
        The language name when an adapter consumes the extension, otherwise
        ``None`` (the file stays indexable as ordinary chunks).
    """
    lowered = relative_path.lower()
    for adapter_class in all_adapters():
        if lowered.endswith(adapter_class.extensions):
            return adapter_class.language
    return None


def adapter_for_file(relative_path: str) -> Any | None:
    """Return an adapter instance for a repository-relative path.

    Args:
        relative_path: Repository-relative path with forward slashes.

    Returns:
        A shared adapter instance, or ``None`` when no adapter consumes the
        extension.
    """
    language = language_for_file(relative_path)
    return get_adapter(language) if language is not None else None


def resolve_extraction_edges(
    extractions: list[FileExtraction],
    indexes: SymbolIndexes,
    repository_root: str,
) -> list[GraphEdge]:
    """Resolve mixed-language extractions with each language's own resolver.

    Resolution never crosses languages: a JavaScript call resolves only
    against JavaScript (and TS) definitions reachable through the shared
    symbol indexes, matching the no-cross-language non-goal.

    Args:
        extractions: Fresh extractions of any supported language.
        indexes: Repository-wide symbol indexes over all nodes.
        repository_root: Canonical repository root used for stable ids.

    Returns:
        Deterministically sorted edge list over all extractions.
    """
    by_language: dict[str, list[FileExtraction]] = {}
    for extraction in extractions:
        by_language.setdefault(extraction.language, []).append(extraction)
    edges: list[GraphEdge] = []
    for language in sorted(by_language):
        adapter = get_adapter(language)
        if adapter is None:  # pragma: no cover - extraction implies an adapter
            continue
        edges.extend(adapter.resolve_edges(by_language[language], indexes, repository_root))
    unique: dict[str, GraphEdge] = {}
    for edge in edges:
        unique.setdefault(edge.id, edge)
    return sorted(unique.values(), key=lambda item: item.id)


def capabilities_payload() -> dict[str, Any]:
    """Build the versioned capabilities report (shared by all surfaces).

    Returns:
        The versioned :class:`GraphCapabilities` payload as a dictionary.
    """
    languages: list[LanguageCapability] = []
    for language in sorted(SUPPORTED_LANGUAGES):
        adapter = get_adapter(language)
        supported_constructs, unsupported_constructs = _CAPABILITY_CONSTRUCTS.get(
            language, ((), ("constructs for this language are not catalogued",))
        )
        if adapter is not None:
            languages.append(
                LanguageCapability(
                    language=language,
                    supported=True,
                    adapter_version=adapter.extractor_version,
                    file_extensions=tuple(adapter.extensions),
                    node_kinds=_language_node_kinds(language),
                    edge_kinds=tuple(EDGE_KINDS),
                    supported_constructs=tuple(supported_constructs),
                    unsupported_constructs=tuple(unsupported_constructs),
                )
            )
        else:
            languages.append(
                LanguageCapability(
                    language=language,
                    supported=False,
                    adapter_version=None,
                    file_extensions=(),
                    node_kinds=(),
                    edge_kinds=(),
                    supported_constructs=(),
                    unsupported_constructs=("no adapter in this build; files stay indexable as chunks",),
                )
            )
    for language in _LANGUAGES_WITHOUT_ADAPTERS:
        languages.append(
            LanguageCapability(
                language=language,
                supported=False,
                adapter_version=None,
                file_extensions=(),
                node_kinds=(),
                edge_kinds=(),
                supported_constructs=(),
                unsupported_constructs=("no adapter in this build; files stay indexable as chunks",),
            )
        )
    capabilities = GraphCapabilities(
        schema_version=CAPABILITIES_SCHEMA_VERSION,
        languages=tuple(languages),
    )
    return capabilities.to_dict()


def _language_node_kinds(language: str) -> tuple[str, ...]:
    return GRAPH_NODE_KINDS_BY_LANGUAGE.get(language, NODE_KINDS)


def capability_matrix_markdown() -> str:
    """Render the generated support matrix published in docs/SYMBOL_GRAPH.md.

    Returns:
        Markdown lines describing per-language node kinds, edge kinds,
        adapter versions, supported constructs, and unsupported constructs.
    """
    payload = capabilities_payload()
    lines = [
        "<!-- CAPABILITY-MATRIX:BEGIN (generated from ctxai.graph.adapters; do not edit by hand) -->",
        "| Language | Supported | Adapter version | File extensions | Node kinds | Edge kinds |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for entry in payload["languages"]:
        extensions = ", ".join(entry["file_extensions"]) if entry["file_extensions"] else "—"
        node_kinds = ", ".join(entry["node_kinds"]) if entry["node_kinds"] else "—"
        edge_kinds = ", ".join(entry["edge_kinds"]) if entry["edge_kinds"] else "—"
        version = entry["adapter_version"] or "—"
        supported = "yes" if entry["supported"] else "no"
        lines.append(f"| {entry['language']} | {supported} | {version} | {extensions} | {node_kinds} | {edge_kinds} |")
    lines.append("")
    for entry in payload["languages"]:
        constructs = "; ".join(entry["supported_constructs"])
        lines.append(f"**{entry['language']}** — resolved constructs: {constructs}.")
        lines.append("")
        lines.append(
            f"*{entry['language']}* — never resolved (kept honest, no fabricated edges): "
            + "; ".join(entry["unsupported_constructs"])
            + "."
        )
        lines.append("")
    lines.append("<!-- CAPABILITY-MATRIX:END -->")
    return "\n".join(lines)
