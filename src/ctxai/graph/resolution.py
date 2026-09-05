"""Language-neutral symbol graph records and the conservative resolution ladder (IG-01/IG-02).

Every language adapter (Python, JavaScript, TypeScript) produces the same
structural records from :mod:`ctxai.graph.resolution`:

1. ``extract_file`` parses one file with tree-sitter and produces a
   :class:`FileExtraction`: definition nodes, containment edges, import
   records, base-class records, module export records, and call/reference
   usages. It never imports or executes indexed code.
2. ``resolve_*`` turns the per-file records into final graph edges using a
   repository-wide :class:`SymbolIndexes`. The resolution ladder is
   deliberately conservative: a target is connected only when statically
   unambiguous (import binding, lexical/module scope, or a unique
   repository-wide display name); everything else stays an *unresolved* edge
   (imports, calls, inheritance) or is simply not recorded (references),
   never guessed.

The shared ladder (``resolve_dotted``) is parameterized per language by the
module's import binding map, its ``self``/``this`` names, and its runtime
builtin names.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .model import (
    CONFIDENCE_EXACT,
    CONFIDENCE_PROBABLE,
    CONFIDENCE_UNRESOLVED,
    GraphEdge,
    GraphNode,
    derive_edge_id,
    derive_node_id,
)


@dataclass(frozen=True)
class ImportRecord:
    """One import binding produced by a single import statement.

    Attributes:
        binding: Local name the import binds (``"*"`` for star imports).
        module_target: Absolute module target of the statement (relative
            imports/specifiers are pre-resolved against the file's location);
            this is what the ``imports`` edge points at.
        binds_module: Module name the *binding* refers to (the root package
            for plain ``import a.b``, the full target for aliases and JS
            namespace/default imports).
        symbol: Name imported from ``module_target``, or ``None`` for plain
            module imports.
        line: 1-based evidence line of the import statement.
    """

    binding: str
    module_target: str
    binds_module: str
    symbol: str | None
    line: int


@dataclass(frozen=True)
class UsageRecord:
    """One call or non-call name usage inside a definition.

    Attributes:
        kind: ``"call"`` or ``"reference"``.
        source_node_id: Enclosing definition node id (or module node id).
        scope_qn: Qualified name of the enclosing lexical scope.
        class_qn: Qualified name of the enclosing class, when any.
        root: Leftmost identifier of the dotted expression (``None`` when the
            expression is not a plain dotted chain).
        attrs: Remaining dotted components after ``root``.
        text: Raw source text of the expression.
        line: 1-based evidence line.
    """

    kind: str
    source_node_id: str
    scope_qn: str
    class_qn: str | None
    root: str | None
    attrs: tuple[str, ...]
    text: str
    line: int


@dataclass(frozen=True)
class BaseRecord:
    """One base-class/interface expression of a class or interface definition.

    Attributes:
        class_node_id: Node id of the inheriting class or interface.
        root: Leftmost identifier of the base expression (``None`` when not a
            plain dotted chain).
        attrs: Remaining dotted components.
        text: Raw source text of the base expression.
        line: 1-based evidence line of the definition.
    """

    class_node_id: str
    root: str | None
    attrs: tuple[str, ...]
    text: str
    line: int


@dataclass(frozen=True)
class ExportRecord:
    """One name a module exposes to importers (IG-02).

    Attributes:
        name: Exported name as importers reference it (``"default"`` for ES
            default exports, ``"*"`` for ``export *``).
        target: Qualified-name candidate the export points at: a symbol of
            the same module for local exports, or ``"<module>.<symbol>"`` /
            ``"<module>.*"`` for re-exports.
    """

    name: str
    target: str


@dataclass
class FileExtraction:
    """Structural extraction result for a single source file."""

    file_path: str
    language: str
    module_name: str
    module_node: GraphNode
    nodes: list[GraphNode]
    contains: list[GraphEdge]
    imports: list[ImportRecord]
    bases: list[BaseRecord]
    usages: list[UsageRecord]
    exports: list[ExportRecord] = field(default_factory=list)
    exports_complete: bool = False
    syntax_errors: int = 0


@dataclass
class SymbolIndexes:
    """Repository-wide lookup tables used to resolve usage records.

    Attributes:
        symbol: Qualified name to node id for non-module definitions.
        module: Module name to module node id.
        packages: Ancestor prefixes of known modules (namespace packages have
            no node but make dotted chains walkable).
        display: Display name to the (sorted) node ids sharing it.
        qualified_by_id: Node id to qualified name.
        kind_by_id: Node id to node kind.
        exports: Module name to exported name to qualified-name candidate
            (from fresh extractions and/or the persisted export table).
        exports_complete: Module names whose export surface is statically
            complete (ES ``export`` statements); absent modules resolve
            imports leniently by definition existence.
    """

    symbol: dict[str, str] = field(default_factory=dict)
    module: dict[str, str] = field(default_factory=dict)
    packages: set[str] = field(default_factory=set)
    display: dict[str, list[str]] = field(default_factory=dict)
    qualified_by_id: dict[str, str] = field(default_factory=dict)
    kind_by_id: dict[str, str] = field(default_factory=dict)
    exports: dict[str, dict[str, str]] = field(default_factory=dict)
    exports_complete: set[str] = field(default_factory=set)


def build_symbol_indexes(
    nodes: list[GraphNode],
    exports: dict[str, dict[str, str]] | None = None,
    exports_complete: set[str] | None = None,
) -> SymbolIndexes:
    """Build resolution indexes from a repository-wide node set.

    Args:
        nodes: All candidate nodes (existing store nodes plus fresh extractions).
        exports: Optional module export maps (module name to exported name to
            qualified-name candidate) to merge into the indexes.
        exports_complete: Optional set of module names with complete export
            surfaces.

    Returns:
        Deterministically built :class:`SymbolIndexes`; on duplicate qualified
        names the first node in ``(file_path, qualified_name, id)`` order wins.
    """
    indexes = SymbolIndexes()
    for node in sorted(nodes, key=lambda item: (item.file_path, item.qualified_name, item.id)):
        indexes.qualified_by_id[node.id] = node.qualified_name
        indexes.kind_by_id[node.id] = node.kind
        if node.kind == "module":
            indexes.module.setdefault(node.qualified_name, node.id)
            parts = node.qualified_name.split(".")
            for depth in range(1, len(parts)):
                indexes.packages.add(".".join(parts[:depth]))
            continue
        indexes.symbol.setdefault(node.qualified_name, node.id)
        indexes.display.setdefault(node.display_name, []).append(node.id)
    if exports:
        for module_name, bindings in exports.items():
            indexes.exports.setdefault(module_name, {}).update(bindings)
    if exports_complete:
        indexes.exports_complete |= exports_complete
    return indexes


def walkable(indexes: SymbolIndexes, qualified: str) -> bool:
    """Return whether a dotted name can be walked through the indexes.

    Args:
        indexes: Repository-wide symbol indexes.
        qualified: Dotted qualified name.

    Returns:
        True when the name is a known symbol, module, or ancestor package.
    """
    return qualified in indexes.symbol or qualified in indexes.module or qualified in indexes.packages


def chain_from(indexes: SymbolIndexes, base_qualified: str, attrs: tuple[str, ...]) -> str | None:
    """Walk ``attrs`` below a qualified name through the symbol indexes.

    A hop that is not itself a known symbol may still resolve through the
    module export surface (a statically recorded re-export binding), so
    ``ns.calculate`` resolves across ``export { calculate } from './calc'``
    hubs without ever guessing.

    Args:
        indexes: Repository-wide symbol indexes.
        base_qualified: Qualified name the chain starts from (root, binding
            referent, or enclosing scope).
        attrs: Remaining dotted components.

    Returns:
        Node id when the full chain resolves to a known symbol or module,
        otherwise ``None``.
    """
    current = base_qualified
    if not walkable(indexes, current):
        return None
    node_id = indexes.symbol.get(current) or indexes.module.get(current)
    for attr in attrs:
        current = f"{current}.{attr}"
        next_id = indexes.symbol.get(current) or indexes.module.get(current)
        if next_id is None:
            next_id = chase_export_binding(indexes, current)
            if next_id is None:
                return None
            current = indexes.qualified_by_id.get(next_id, current)
        node_id = next_id
    return node_id


def chase_export_binding(indexes: SymbolIndexes, dotted: str) -> str | None:
    """Follow one statically recorded re-export binding (``module.symbol``).

    Args:
        indexes: Repository-wide symbol indexes (carrying export maps).
        dotted: ``<module>.<exported name>`` candidate.

    Returns:
        The node id the binding resolves to, or ``None`` when the module has
        no such recorded export.
    """
    module_qn, _, symbol = dotted.rpartition(".")
    if not module_qn:
        return None
    exports = indexes.exports.get(module_qn)
    if not exports or symbol not in exports:
        return None
    target = exports[symbol]
    if target.endswith(".*"):
        return indexes.module.get(target[:-2])
    return indexes.symbol.get(target) or indexes.module.get(target)


def make_resolved_edge(
    repository_root: str,
    resolver_version: str,
    file_path: str,
    kind: str,
    source_id: str,
    target_id: str | None,
    target_text: str | None,
    line: int,
    confidence: str | None = None,
) -> GraphEdge:
    """Build one evidence-backed edge with derived id and confidence.

    Args:
        repository_root: Canonical repository root used for stable ids.
        resolver_version: Adapter resolver version stamped on the edge.
        file_path: Repository-relative evidence file.
        kind: Edge kind (one of EDGE_KINDS).
        source_id: Source node id.
        target_id: Target node id when resolved, otherwise ``None``.
        target_text: Unresolved target text when unresolved, otherwise ``None``.
        line: 1-based evidence line.
        confidence: Optional confidence override for resolved edges.

    Returns:
        The :class:`GraphEdge`.
    """
    resolved = target_id is not None
    return GraphEdge(
        id=derive_edge_id(repository_root, kind, source_id, target_id, target_text, file_path, line),
        kind=kind,
        source_id=source_id,
        target_id=target_id,
        target_text=target_text,
        evidence_file=file_path,
        evidence_line=line,
        confidence=confidence
        if resolved and confidence
        else CONFIDENCE_UNRESOLVED
        if not resolved
        else CONFIDENCE_EXACT,
        resolver_version=resolver_version,
    )


def resolve_dotted(
    indexes: SymbolIndexes,
    bindings: dict[str, tuple[str, Any]],
    module_name: str,
    root: str | None,
    attrs: tuple[str, ...],
    scope_qn: str,
    class_qn: str | None,
    self_names: frozenset[str],
    builtins: frozenset[str],
    preserve_unresolved: bool,
) -> tuple[str | None, str | None]:
    """Resolve a dotted usage through the conservative ladder.

    The ladder, in order — first hit wins: import bindings; ``self``/``this``
    attributes within the enclosing class; lexical scope; module scope;
    unique repository-wide display name (``probable``); runtime builtins
    (no edge at all).

    Args:
        indexes: Repository-wide symbol indexes.
        bindings: The module's import binding map (binding name to
            ``("node", id)``, ``("module", qn)``, or ``("unresolved", text)``).
        module_name: Qualified module name of the file the usage belongs to.
        root: Leftmost identifier (``None`` for dynamic expressions).
        attrs: Remaining dotted components.
        scope_qn: Enclosing lexical scope qualified name.
        class_qn: Enclosing class qualified name when any.
        self_names: Names that refer to the enclosing instance (``self``,
            ``this``).
        builtins: Runtime builtin names that create no edge.
        preserve_unresolved: Whether an unresolved edge must be preserved
            (calls, inheritance) as opposed to silently skipped (references).

    Returns:
        ``(node_id, confidence)`` where confidence is ``exact``/``probable``
        and ``node_id`` is set; ``(None, None)`` when nothing should be
        recorded (builtins, unrecorded references); or ``(None, "unresolved")``
        when an unresolved edge must be preserved.
    """
    unresolved = CONFIDENCE_UNRESOLVED if preserve_unresolved else None
    if root is None:
        return None, unresolved

    binding = bindings.get(root)
    if binding is not None:
        kind, value = binding
        if kind == "node":
            if not attrs:
                return value, CONFIDENCE_EXACT
            base_qualified = indexes.qualified_by_id.get(value)
            resolved = chain_from(indexes, base_qualified, attrs) if base_qualified else None
            return (resolved, CONFIDENCE_EXACT) if resolved else (None, unresolved)
        if kind == "module":
            if not attrs:
                return indexes.module.get(value), CONFIDENCE_EXACT
            resolved = chain_from(indexes, value, attrs)
            return (resolved, CONFIDENCE_EXACT) if resolved else (None, unresolved)
        return None, unresolved

    if root in self_names and class_qn is not None and attrs:
        resolved = chain_from(indexes, class_qn, attrs)
        if resolved:
            return resolved, CONFIDENCE_EXACT
        return None, unresolved

    if scope_qn:
        resolved = chain_from(indexes, f"{scope_qn}.{root}", attrs)
        if resolved:
            return resolved, CONFIDENCE_EXACT

    module_resolved = chain_from(indexes, f"{module_name}.{root}", attrs)
    if module_resolved:
        return module_resolved, CONFIDENCE_EXACT

    display_ids = indexes.display.get(root, [])
    if len(display_ids) == 1 and not attrs:
        return display_ids[0], CONFIDENCE_PROBABLE

    if root in builtins:
        return None, None

    return None, unresolved


def resolve_bases(
    extraction: FileExtraction,
    bindings: dict[str, tuple[str, Any]],
    indexes: SymbolIndexes,
    repository_root: str,
    resolver_version: str,
    self_names: frozenset[str],
    builtins: frozenset[str],
) -> list[GraphEdge]:
    """Resolve base-class/interface records into ``inherits`` edges.

    Args:
        extraction: The extraction whose bases to resolve.
        bindings: The module's import binding map.
        indexes: Repository-wide symbol indexes.
        repository_root: Canonical repository root used for stable ids.
        resolver_version: Adapter resolver version.
        self_names: Names that refer to the enclosing instance.
        builtins: Runtime builtin names that create no edge.

    Returns:
        The resolved ``inherits`` edges for the extraction.
    """
    edges: list[GraphEdge] = []
    for record in extraction.bases:
        node_id, confidence = resolve_dotted(
            indexes,
            bindings,
            extraction.module_name,
            record.root,
            record.attrs,
            extraction.module_name,
            None,
            self_names,
            builtins,
            preserve_unresolved=True,
        )
        if node_id is not None:
            edges.append(
                make_resolved_edge(
                    repository_root,
                    resolver_version,
                    extraction.file_path,
                    "inherits",
                    record.class_node_id,
                    node_id,
                    None,
                    record.line,
                    confidence=confidence,
                )
            )
        elif confidence == CONFIDENCE_UNRESOLVED:
            edges.append(
                make_resolved_edge(
                    repository_root,
                    resolver_version,
                    extraction.file_path,
                    "inherits",
                    record.class_node_id,
                    None,
                    record.text,
                    record.line,
                )
            )
    return edges


def resolve_usages(
    extraction: FileExtraction,
    bindings: dict[str, tuple[str, Any]],
    indexes: SymbolIndexes,
    repository_root: str,
    resolver_version: str,
    self_names: frozenset[str],
    builtins: frozenset[str],
) -> list[GraphEdge]:
    """Resolve call/reference usage records into ``calls``/``references``/``tests`` edges.

    Args:
        extraction: The extraction whose usages to resolve.
        bindings: The module's import binding map.
        indexes: Repository-wide symbol indexes.
        repository_root: Canonical repository root used for stable ids.
        resolver_version: Adapter resolver version.
        self_names: Names that refer to the enclosing instance.
        builtins: Runtime builtin names that create no edge.

    Returns:
        The resolved edges for the extraction's usages.
    """
    edges: list[GraphEdge] = []
    for usage in extraction.usages:
        node_id, confidence = resolve_dotted(
            indexes,
            bindings,
            extraction.module_name,
            usage.root,
            usage.attrs,
            usage.scope_qn,
            usage.class_qn,
            self_names,
            builtins,
            preserve_unresolved=usage.kind == "call",
        )
        if usage.kind == "reference":
            if node_id is not None and confidence is not None:
                edges.append(
                    make_resolved_edge(
                        repository_root,
                        resolver_version,
                        extraction.file_path,
                        "references",
                        usage.source_node_id,
                        node_id,
                        None,
                        usage.line,
                        confidence=confidence,
                    )
                )
            continue
        # Calls always preserve an edge unless the target is a runtime builtin.
        if node_id is None and confidence is None:
            continue
        edges.append(
            make_resolved_edge(
                repository_root,
                resolver_version,
                extraction.file_path,
                "calls",
                usage.source_node_id,
                node_id,
                usage.text if node_id is None else None,
                usage.line,
                confidence=confidence,
            )
        )
        if node_id is not None and indexes.kind_by_id.get(usage.source_node_id) == "test":
            target_kind = indexes.kind_by_id.get(node_id)
            if target_kind is not None and target_kind != "test":
                edges.append(
                    make_resolved_edge(
                        repository_root,
                        resolver_version,
                        extraction.file_path,
                        "tests",
                        usage.source_node_id,
                        node_id,
                        None,
                        usage.line,
                        confidence=confidence,
                    )
                )
    return edges


def derive_definition_node(
    repository_root: str,
    file_path: str,
    kind: str,
    qualified_name: str,
    display_name: str,
    language: str,
    start_line: int,
    end_line: int,
    parent_id: str | None,
    visibility: str,
    source_hash: str,
    adapter_version: str,
) -> GraphNode:
    """Build one definition node with a derived stable id.

    Args:
        repository_root: Canonical repository root used for stable ids.
        file_path: Repository-relative file path.
        kind: Node kind (one of NODE_KINDS).
        qualified_name: Fully qualified name of the definition.
        display_name: Bare definition name.
        language: Language of the source file.
        start_line: 1-based start line.
        end_line: 1-based end line.
        parent_id: Parent node id for containment, when any.
        visibility: ``public`` or ``private``.
        source_hash: sha256 of the definition's source text.
        adapter_version: Language adapter version to stamp on the node.

    Returns:
        The :class:`GraphNode`.
    """
    return GraphNode(
        id=derive_node_id(repository_root, file_path, kind, qualified_name),
        kind=kind,
        qualified_name=qualified_name,
        display_name=display_name,
        language=language,
        file_path=file_path,
        start_line=start_line,
        end_line=end_line,
        parent_id=parent_id,
        visibility=visibility,
        source_hash=source_hash,
        adapter_version=adapter_version,
    )
