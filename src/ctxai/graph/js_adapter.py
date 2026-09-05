"""Deterministic JavaScript/TypeScript symbol extraction and resolution (IG-02).

The adapter mirrors the Python adapter's two-phase shape:

1. ``extract_file`` parses one file with tree-sitter (the same grammar family
   the chunker uses) and produces the shared structural records from
   :mod:`ctxai.graph.resolution`: definition nodes, containment edges, import
   records (ES ``import`` and CommonJS ``require``), base-class records
   (``extends``/``implements``), module export records, and call/reference
   usages. It never imports or executes indexed code.
2. ``resolve_edges`` turns the per-file records into final graph edges using
   the shared conservative ladder plus the module export surface. A target is
   connected only when statically unambiguous; dynamic requires/imports,
   template paths, computed member access, and call results stay unresolved
   (imports/calls) or are not recorded (references) — never guessed.
"""

from __future__ import annotations

import hashlib
from typing import Any

from tree_sitter_language_pack import get_parser

from .model import (
    CONFIDENCE_UNRESOLVED,
    JAVASCRIPT_EXTRACTOR_VERSION,
    JAVASCRIPT_RESOLVER_VERSION,
    TYPESCRIPT_EXTRACTOR_VERSION,
    TYPESCRIPT_RESOLVER_VERSION,
    GraphEdge,
    GraphNode,
    derive_edge_id,
    derive_node_id,
)
from .resolution import (
    BaseRecord,
    ExportRecord,
    FileExtraction,
    ImportRecord,
    SymbolIndexes,
    UsageRecord,
    make_resolved_edge,
    resolve_bases,
    resolve_usages,
)

JAVASCRIPT_EXTENSIONS = (".js", ".jsx", ".mjs", ".cjs")
TYPESCRIPT_EXTENSIONS = (".ts", ".tsx", ".mts", ".cts")

_INDEX_BASENAMES = ("index",)

# Names that are lexically bound and never repository symbols (the JS `this`).
SELF_NAMES = frozenset({"this"})

# Runtime builtins/environment globals: calls to these names create no edge
# when nothing in the repository defines the same name.
BUILTIN_NAMES = frozenset(
    {
        "console",
        "Math",
        "JSON",
        "Object",
        "Array",
        "String",
        "Number",
        "Boolean",
        "Promise",
        "Date",
        "RegExp",
        "Error",
        "TypeError",
        "RangeError",
        "SyntaxError",
        "EvalError",
        "URIError",
        "AggregateError",
        "Map",
        "Set",
        "WeakMap",
        "WeakSet",
        "WeakRef",
        "Symbol",
        "BigInt",
        "Proxy",
        "Reflect",
        "Intl",
        "ArrayBuffer",
        "SharedArrayBuffer",
        "DataView",
        "Atomics",
        "Uint8Array",
        "Int8Array",
        "Uint8ClampedArray",
        "Int16Array",
        "Uint16Array",
        "Int32Array",
        "Uint32Array",
        "Float32Array",
        "Float64Array",
        "BigInt64Array",
        "BigUint64Array",
        "parseInt",
        "parseFloat",
        "isNaN",
        "isFinite",
        "encodeURIComponent",
        "decodeURIComponent",
        "encodeURI",
        "decodeURI",
        "setTimeout",
        "setInterval",
        "clearTimeout",
        "clearInterval",
        "setImmediate",
        "clearImmediate",
        "queueMicrotask",
        "requestAnimationFrame",
        "cancelAnimationFrame",
        "fetch",
        "URL",
        "URLSearchParams",
        "TextEncoder",
        "TextDecoder",
        "Blob",
        "File",
        "FormData",
        "Headers",
        "Request",
        "Response",
        "AbortController",
        "Event",
        "EventTarget",
        "CustomEvent",
        "process",
        "Buffer",
        "global",
        "globalThis",
        "window",
        "document",
        "navigator",
        "localStorage",
        "sessionStorage",
        "crypto",
        "performance",
        "undefined",
        "NaN",
        "Infinity",
        "arguments",
        "super",
        "this",
        "require",
        "module",
        "exports",
        "import",
        "eval",
        "atob",
        "btoa",
        "structuredClone",
    }
)

# tree-sitter node types that open a nested scope owning their own names.
_SCOPE_NODE_TYPES = frozenset(
    {
        "function_declaration",
        "generator_function_declaration",
        "function_signature",
        "function_expression",
        "arrow_function",
        "class_declaration",
        "abstract_class_declaration",
        "class",
        "interface_declaration",
        "type_alias_declaration",
        "enum_declaration",
        "method_definition",
        "statement_block",
    }
)

# Identifier parents whose identifier children are part of the declaration
# structure, not name usages.
_STRUCTURAL_PARENT_TYPES = frozenset(
    {
        "function_declaration",
        "generator_function_declaration",
        "function_signature",
        "function_expression",
        "class_declaration",
        "abstract_class_declaration",
        "interface_declaration",
        "type_alias_declaration",
        "enum_declaration",
        "method_definition",
        "method_signature",
        "abstract_method_signature",
        "property_signature",
        "public_field_definition",
        "variable_declarator",
        "import_specifier",
        "export_specifier",
        "namespace_import",
        "namespace_export",
        "import_clause",
        "named_imports",
        "export_clause",
        "labeled_statement",
        "enum_body",
        "pair_pattern",
        "import_equals_declaration",
        "function_type",
        "ambient_declaration",
    }
)

# Value node types that make a variable declarator a function definition.
_FUNCTION_VALUE_TYPES = frozenset({"arrow_function", "function_expression", "generator_function_expression"})


def module_name_for_path(relative_path: str) -> str:
    """Derive the dotted module name from a repository-relative JS/TS path.

    ``index`` files name their directory (``src/utils/index.js`` is
    ``src.utils``), mirroring ``__init__.py`` semantics.

    Args:
        relative_path: Repository-relative path using forward slashes.

    Returns:
        Dotted module name.
    """
    stem = relative_path
    for extension in (*JAVASCRIPT_EXTENSIONS, *TYPESCRIPT_EXTENSIONS):
        if stem.endswith(extension):
            stem = stem[: -len(extension)]
            break
    parts = [part for part in stem.split("/") if part]
    if len(parts) > 1 and parts[-1] in _INDEX_BASENAMES:
        parts = parts[:-1]
    return ".".join(parts) if parts else "__module__"


def _is_test_file(relative_path: str) -> bool:
    """Return whether the path marks the file as a test module.

    Args:
        relative_path: Repository-relative path using forward slashes.

    Returns:
        True for files under ``tests``/``test``/``__tests__`` directories or
        named ``*.test.*``/``*.spec.*``.
    """
    stem = relative_path
    for extension in (*JAVASCRIPT_EXTENSIONS, *TYPESCRIPT_EXTENSIONS):
        if stem.endswith(extension):
            stem = stem[: -len(extension)]
            break
    name = stem.rsplit("/", 1)[-1]
    if ".test." in f"{name}." or ".spec." in f"{name}.":
        return True
    return any(part in ("tests", "test", "__tests__", "spec", "specs") for part in stem.split("/")[:-1])


def _is_test_name(name: str) -> bool:
    return name == "test" or name.startswith("test_")


def _count_error_nodes(node: Any) -> int:
    """Count ERROR and MISSING nodes anywhere in a tree.

    Args:
        node: Root tree-sitter node to sweep.

    Returns:
        Number of error nodes found (0 for a clean parse).
    """
    total = 1 if (node.is_error or node.is_missing) else 0
    for child in node.children:
        total += _count_error_nodes(child)
    return total


class JavaScriptAdapter:
    """LanguageAdapter for JavaScript source files (tree-sitter based)."""

    language = "javascript"
    extractor_version = JAVASCRIPT_EXTRACTOR_VERSION
    resolver_version = JAVASCRIPT_RESOLVER_VERSION
    extensions = JAVASCRIPT_EXTENSIONS

    def __init__(self) -> None:
        """Create the adapter and its shared tree-sitter parser.

        Raises:
            RuntimeError: If the JavaScript grammar is unavailable.
        """
        self._parser = get_parser("javascript")

    @staticmethod
    def supports_file(relative_path: str) -> bool:
        """Return whether the adapter handles this repository-relative path.

        Args:
            relative_path: Repository-relative file path.

        Returns:
            True for JavaScript extensions (``.js``, ``.jsx``, ``.mjs``, ``.cjs``).
        """
        return relative_path.endswith(JAVASCRIPT_EXTENSIONS)

    def _parser_for(self, relative_path: str) -> Any:
        return self._parser

    def extract_file(self, relative_path: str, source: bytes, repository_root: str) -> FileExtraction:
        """Parse one JavaScript file into structural graph records.

        Args:
            relative_path: Repository-relative path with forward slashes.
            source: Raw file bytes (never imported or executed).
            repository_root: Canonical repository root used for stable ids.

        Returns:
            The :class:`FileExtraction` for the file; syntax errors are
            tolerated and counted, they never raise.
        """
        walker = _JsWalker(
            self._parser_for(relative_path),
            relative_path,
            source,
            repository_root,
            self.language,
            self.extractor_version,
            self.resolver_version,
        )
        return walker.run()

    @classmethod
    def resolve_edges(
        cls,
        extractions: list[FileExtraction],
        indexes: SymbolIndexes,
        repository_root: str,
    ) -> list[GraphEdge]:
        """Resolve per-file extraction records into final graph edges.

        Args:
            extractions: Fresh JS/TS extractions to resolve.
            indexes: Repository-wide symbol indexes (fresh nodes included).
            repository_root: Canonical repository root used for stable ids.

        Returns:
            Deterministically sorted edge list, including containment edges.
        """
        return resolve_edges(extractions, indexes, repository_root)


class TypeScriptAdapter(JavaScriptAdapter):
    """LanguageAdapter for TypeScript source files (tree-sitter based)."""

    language = "typescript"
    extractor_version = TYPESCRIPT_EXTRACTOR_VERSION
    resolver_version = TYPESCRIPT_RESOLVER_VERSION
    extensions = TYPESCRIPT_EXTENSIONS

    def __init__(self) -> None:
        """Create the adapter and its shared tree-sitter parsers.

        Raises:
            RuntimeError: If the TypeScript/TSX grammars are unavailable.
        """
        self._parsers = {"typescript": get_parser("typescript"), "tsx": get_parser("tsx")}

    @staticmethod
    def supports_file(relative_path: str) -> bool:
        """Return whether the adapter handles this repository-relative path.

        Args:
            relative_path: Repository-relative file path.

        Returns:
            True for TypeScript extensions (``.ts``, ``.tsx``, ``.mts``, ``.cts``).
        """
        return relative_path.endswith(TYPESCRIPT_EXTENSIONS)

    def _parser_for(self, relative_path: str) -> Any:
        if relative_path.endswith(".tsx"):
            return self._parsers["tsx"]
        return self._parsers["typescript"]


def resolve_edges(
    extractions: list[FileExtraction],
    indexes: SymbolIndexes,
    repository_root: str,
) -> list[GraphEdge]:
    """Resolve per-file extraction records into final graph edges.

    Args:
        extractions: Fresh per-file extractions to resolve.
        indexes: Repository-wide symbol indexes (fresh nodes included).
        repository_root: Canonical repository root used for stable ids.

    Returns:
        Deterministically sorted edge list, including containment edges.
    """
    edges: list[GraphEdge] = []
    for extraction in extractions:
        resolver_version = (
            TYPESCRIPT_RESOLVER_VERSION if extraction.language == "typescript" else JAVASCRIPT_RESOLVER_VERSION
        )
        edges.extend(extraction.contains)
        import_edges, bindings = _resolve_imports(extraction, indexes, repository_root, resolver_version)
        edges.extend(import_edges)
        edges.extend(
            resolve_bases(extraction, bindings, indexes, repository_root, resolver_version, SELF_NAMES, BUILTIN_NAMES)
        )
        edges.extend(
            resolve_usages(extraction, bindings, indexes, repository_root, resolver_version, SELF_NAMES, BUILTIN_NAMES)
        )
    unique: dict[str, GraphEdge] = {}
    for edge in edges:
        unique.setdefault(edge.id, edge)
    return sorted(unique.values(), key=lambda item: item.id)


def _resolve_exported_symbol(
    module_qn: str,
    symbol: str,
    indexes: SymbolIndexes,
    depth: int = 0,
) -> tuple[str | None, str]:
    """Resolve one imported symbol through a module's export surface.

    Follows re-export bindings and one bounded chain of ``export *`` hubs;
    when a module's export surface is statically complete and does not list
    the symbol, the import stays unresolved instead of guessing.

    Args:
        module_qn: Qualified module name imported from.
        symbol: Exported symbol name.
        indexes: Repository-wide symbol indexes (carrying export maps).
        depth: Current star-reexport chase depth.

    Returns:
        ``(node_id, candidate_text)`` with ``node_id`` set when resolved.
    """
    candidate = f"{module_qn}.{symbol}"
    exports = indexes.exports.get(module_qn)
    if exports is not None:
        complete = module_qn in indexes.exports_complete
        if symbol in exports:
            return _resolve_export_target(exports[symbol], indexes, depth)
        star = exports.get("*")
        if star is not None and star.endswith(".*") and depth < 3:
            return _resolve_exported_symbol(star[:-2], symbol, indexes, depth + 1)
        if complete:
            return None, candidate
    node_id = indexes.symbol.get(candidate) or indexes.module.get(candidate)
    return node_id, candidate


def _resolve_export_target(
    target: str,
    indexes: SymbolIndexes,
    depth: int,
) -> tuple[str | None, str]:
    """Resolve one export-map target (a qualified name or ``<module>.*``)."""
    if target.endswith(".*"):
        module_qn = target[:-2]
        return indexes.module.get(module_qn), target
    node_id = indexes.symbol.get(target) or indexes.module.get(target)
    if node_id is not None:
        return node_id, target
    if depth < 3:
        module_qn, _, symbol = target.rpartition(".")
        if module_qn:
            exports = indexes.exports.get(module_qn)
            if exports is not None and symbol in exports:
                return _resolve_export_target(exports[symbol], indexes, depth + 1)
    return None, target


def _resolve_imports(
    extraction: FileExtraction,
    indexes: SymbolIndexes,
    repository_root: str,
    resolver_version: str,
) -> tuple[list[GraphEdge], dict[str, tuple[str, Any]]]:
    """Emit ``imports`` edges and return the module's import binding map.

    Args:
        extraction: The extraction whose imports to resolve.
        indexes: Repository-wide symbol indexes.
        repository_root: Canonical repository root.
        resolver_version: Adapter resolver version.

    Returns:
        Tuple of (edges, binding map). The binding map maps binding names to
        ``("node", id)``, ``("module", qn)``, or ``("unresolved", text)``.
    """
    bindings: dict[str, tuple[str, Any]] = {}
    edges: list[GraphEdge] = []
    module_id = extraction.module_node.id
    for record in extraction.imports:
        if record.symbol is None:
            target_id = indexes.module.get(record.module_target)
            edges.append(
                make_resolved_edge(
                    repository_root,
                    resolver_version,
                    extraction.file_path,
                    "imports",
                    module_id,
                    target_id,
                    None if target_id is not None else record.module_target,
                    record.line,
                )
            )
            if record.binding:
                bindings[record.binding] = ("module", record.binds_module)
        elif record.symbol == "*":
            text = f"{record.module_target}.*"
            edges.append(
                make_resolved_edge(
                    repository_root,
                    resolver_version,
                    extraction.file_path,
                    "imports",
                    module_id,
                    None,
                    text,
                    record.line,
                )
            )
            if record.binding:
                bindings.setdefault(record.binding, ("unresolved", text))
        else:
            target_id, candidate = _resolve_exported_symbol(record.module_target, record.symbol, indexes)
            if target_id is not None:
                bindings[record.binding] = ("node", target_id)
                edges.append(
                    make_resolved_edge(
                        repository_root,
                        resolver_version,
                        extraction.file_path,
                        "imports",
                        module_id,
                        target_id,
                        None,
                        record.line,
                    )
                )
            else:
                bindings[record.binding] = ("unresolved", candidate)
                edges.append(
                    make_resolved_edge(
                        repository_root,
                        resolver_version,
                        extraction.file_path,
                        "imports",
                        module_id,
                        None,
                        candidate,
                        record.line,
                    )
                )
    return edges, bindings


class _JsWalker:
    """Single-file tree-sitter walker producing a :class:`FileExtraction`."""

    def __init__(
        self,
        parser: Any,
        relative_path: str,
        source: bytes,
        repository_root: str,
        language: str,
        adapter_version: str,
        resolver_version: str,
    ) -> None:
        self._parser = parser
        self._repository_root = repository_root
        self._relative_path = relative_path
        self._source = source
        self._language = language
        self._adapter_version = adapter_version
        self._resolver_version = resolver_version
        self._module_name = module_name_for_path(relative_path)
        self._is_test_file = _is_test_file(relative_path)
        self._nodes: list[GraphNode] = []
        self._seen_ids: set[str] = set()
        self._contains: list[GraphEdge] = []
        self._imports: list[ImportRecord] = []
        self._bases: list[BaseRecord] = []
        self._usages: list[UsageRecord] = []
        self._exports: list[ExportRecord] = []
        # (exported name, candidate, resolved): candidate is a qualified-name
        # candidate when resolved, otherwise a local name resolved at the end
        # of the walk (forward references need the full node set).
        self._cjs_exports: list[tuple[str, str, bool]] = []
        self._cjs_dynamic = False
        self._exports_complete = False
        self._syntax_errors = 0
        self._bound_stack: list[set[str]] = []

    # -- small helpers -----------------------------------------------------

    def _text(self, node: Any) -> str:
        return self._source[node.start_byte : node.end_byte].decode("utf-8", errors="ignore")

    def _line(self, node: Any) -> int:
        return node.start_point[0] + 1

    def _node_hash(self, node: Any) -> str:
        return hashlib.sha256(self._source[node.start_byte : node.end_byte]).hexdigest()

    def _bound(self, name: str) -> bool:
        return any(name in scope for scope in self._bound_stack)

    def _dotted_parts(self, node: Any) -> tuple[str, tuple[str, ...]] | None:
        """Return ``(root, attrs)`` for identifier/``this``/member chains."""
        if node.type == "identifier":
            return self._text(node), ()
        if node.type == "this":
            return "this", ()
        if node.type != "member_expression":
            return None
        parts: list[str] = []
        current = node
        while current is not None and current.type == "member_expression":
            prop = current.child_by_field_name("property")
            if prop is None or prop.type not in ("property_identifier", "identifier"):
                return None
            parts.append(self._text(prop))
            current = current.child_by_field_name("object")
        if current is None:
            return None
        if current.type == "identifier":
            parts.append(self._text(current))
        elif current.type == "this":
            parts.append("this")
        else:
            return None
        parts.reverse()
        return parts[0], tuple(parts[1:])

    def _string_text(self, node: Any) -> str | None:
        """Return the literal text of a string node, when static."""
        if node is None or node.type != "string":
            return None
        for child in node.children:
            if child.type == "string_fragment":
                return self._text(child)
        return None

    def _first_argument(self, arguments: Any) -> Any | None:
        """Return the first named argument of a call (punctuation excluded)."""
        if arguments is None:
            return None
        for child in arguments.children:
            if child.is_named:
                return child
        return None

    # -- definitions -------------------------------------------------------

    def _definition_kind(self, name: str, base_kind: str) -> str:
        if base_kind in ("function", "method", "class", "test") and (self._is_test_file or _is_test_name(name)):
            return "test"
        return base_kind

    def _add_definition(self, node: Any, kind: str, name: str, parent_id: str | None, prefix: str) -> GraphNode | None:
        qualified = f"{prefix}.{name}" if prefix else name
        graph_node = GraphNode(
            id=derive_node_id(self._repository_root, self._relative_path, kind, qualified),
            kind=kind,
            qualified_name=qualified,
            display_name=name,
            language=self._language,
            file_path=self._relative_path,
            start_line=self._line(node),
            end_line=node.end_point[0] + 1,
            parent_id=parent_id,
            visibility="public",
            source_hash=self._node_hash(node),
            adapter_version=self._adapter_version,
        )
        if graph_node.id in self._seen_ids:
            return None  # redeclaration/overload/accessor pair: first wins
        self._seen_ids.add(graph_node.id)
        self._nodes.append(graph_node)
        if parent_id is not None:
            self._contains.append(
                self._edge(
                    kind="contains",
                    source_id=parent_id,
                    target_id=graph_node.id,
                    target_text=None,
                    evidence_line=graph_node.start_line,
                )
            )
        return graph_node

    def _edge(
        self,
        kind: str,
        source_id: str,
        target_id: str | None,
        target_text: str | None,
        evidence_line: int,
    ) -> GraphEdge:
        return GraphEdge(
            id=derive_edge_id(
                self._repository_root, kind, source_id, target_id, target_text, self._relative_path, evidence_line
            ),
            kind=kind,
            source_id=source_id,
            target_id=target_id,
            target_text=target_text,
            evidence_file=self._relative_path,
            evidence_line=evidence_line,
            confidence=CONFIDENCE_UNRESOLVED if target_id is None else "exact",
            resolver_version=self._resolver_version,
        )

    def _record_reference(
        self,
        root: str,
        attrs: tuple[str, ...],
        node: Any,
        source_id: str,
        scope_qn: str,
        class_qn: str | None,
    ) -> None:
        if root in SELF_NAMES or root in BUILTIN_NAMES or self._bound(root):
            return
        self._usages.append(
            UsageRecord(
                kind="reference",
                source_node_id=source_id,
                scope_qn=scope_qn,
                class_qn=class_qn,
                root=root,
                attrs=attrs,
                text=self._text(node),
                line=self._line(node),
            )
        )

    def _record_call(
        self,
        parts: tuple[str, tuple[str, ...]] | None,
        node: Any,
        source_id: str,
        scope_qn: str,
        class_qn: str | None,
        text: str,
    ) -> None:
        root = parts[0] if parts else None
        # `this` is handled by the resolution ladder (this-attribute within the
        # enclosing class), so it must not be dropped as a builtin here.
        if root is not None and root in BUILTIN_NAMES and root != "this":
            return
        self._usages.append(
            UsageRecord(
                kind="call",
                source_node_id=source_id,
                scope_qn=scope_qn,
                class_qn=class_qn,
                root=root,
                attrs=parts[1] if parts else (),
                text=text,
                line=self._line(node),
            )
        )

    # -- bound-name collection --------------------------------------------

    def _collect_bound_names(self, body: Any) -> set[str]:
        """Collect names bound anywhere in a scope body (pre-pass).

        Import statements and ``require`` declarators are skipped: imported
        names resolve through the import map, not as locals.

        Args:
            body: tree-sitter node of the scope body.

        Returns:
            The set of locally bound names in this scope.
        """
        bound: set[str] = set()
        if body is None:
            return bound
        self._visit_bound(body, bound)
        return bound

    def _visit_bound(self, node: Any, bound: set[str]) -> None:
        if node.type in _SCOPE_NODE_TYPES and node.type != "statement_block":
            return  # nested scopes own their names
        if node.type in ("lexical_declaration", "variable_declaration"):
            for declarator in node.children:
                if declarator.type != "variable_declarator":
                    continue
                if self._declarator_is_require(declarator):
                    continue  # import bindings resolve through the import map
                value = declarator.child_by_field_name("value")
                if value is not None and value.type in _FUNCTION_VALUE_TYPES:
                    continue  # function definitions are nodes, not locals
                self._collect_pattern_identifiers(declarator.child_by_field_name("name"), bound)
        elif node.type == "assignment_expression":
            left = node.child_by_field_name("left")
            if left is not None and left.type == "identifier":
                bound.add(self._text(left))
        elif node.type in ("for_in_statement", "for_statement"):
            left = node.child_by_field_name("left")
            if left is not None:
                if left.type in ("lexical_declaration", "variable_declaration", "identifier"):
                    self._collect_pattern_identifiers(left, bound)
        elif node.type == "catch_clause":
            name = node.child_by_field_name("parameter")
            if name is not None:
                self._collect_pattern_identifiers(name, bound)
        elif node.type == "export_statement":
            # exported declarations bind like their bare counterparts
            declaration = node.child_by_field_name("declaration")
            if declaration is not None and declaration.type in ("lexical_declaration", "variable_declaration"):
                for declarator in declaration.children:
                    if declarator.type != "variable_declarator":
                        continue
                    value = declarator.child_by_field_name("value")
                    if value is not None and value.type in _FUNCTION_VALUE_TYPES:
                        continue
                    self._collect_pattern_identifiers(declarator.child_by_field_name("name"), bound)
        for child in node.children:
            self._visit_bound(child, bound)

    def _declarator_is_require(self, declarator: Any) -> bool:
        value = declarator.child_by_field_name("value")
        if value is None or value.type != "call_expression":
            return False
        func = value.child_by_field_name("function")
        return func is not None and func.type == "identifier" and self._text(func) == "require"

    def _collect_pattern_identifiers(self, pattern: Any, bound: set[str]) -> None:
        """Collect destructuring/declaration target identifiers into ``bound``.

        Args:
            pattern: tree-sitter pattern node (identifier, object/array
                pattern, or declaration).
            bound: The set to extend.
        """
        if pattern is None:
            return
        if pattern.type in ("identifier", "shorthand_property_identifier_pattern"):
            bound.add(self._text(pattern))
            return
        if pattern.type in ("lexical_declaration", "variable_declaration"):
            for child in pattern.children:
                if child.type == "variable_declarator":
                    self._collect_pattern_identifiers(child.child_by_field_name("name"), bound)
            return
        if pattern.type == "rest_pattern":
            for child in pattern.children:
                self._collect_pattern_identifiers(child, bound)
            return
        if pattern.type == "assignment_pattern":
            self._collect_pattern_identifiers(pattern.child_by_field_name("left"), bound)
            return
        if pattern.type == "pair_pattern":
            self._collect_pattern_identifiers(pattern.child_by_field_name("value"), bound)
            return
        for child in pattern.children:
            self._collect_pattern_identifiers(child, bound)

    def _parameter_names(self, parameters: Any) -> set[str]:
        names: set[str] = set()
        if parameters is None:
            return names
        self._collect_pattern_identifiers(parameters, names)
        return names

    # -- walkers -----------------------------------------------------------

    def run(self) -> FileExtraction:
        tree = self._parser.parse(self._source)
        root = tree.root_node
        self._syntax_errors = _count_error_nodes(root)
        module_node = GraphNode(
            id=derive_node_id(self._repository_root, self._relative_path, "module", self._module_name),
            kind="module",
            qualified_name=self._module_name,
            display_name=self._module_name,
            language=self._language,
            file_path=self._relative_path,
            start_line=1,
            end_line=root.end_point[0] + 1,
            parent_id=None,
            visibility="public",
            source_hash=hashlib.sha256(self._source).hexdigest(),
            adapter_version=self._adapter_version,
        )
        self._nodes.append(module_node)
        self._bound_stack.append(self._collect_bound_names(root))
        for child in root.children:
            self._walk(child, source_id=module_node.id, scope_qn=self._module_name, class_qn=None)
        self._bound_stack.pop()
        exports = list(self._exports)
        for exported, candidate, resolved in self._cjs_exports:
            target = candidate if resolved else self._cjs_export_target(candidate)
            if target is not None:
                exports.append(ExportRecord(name=exported, target=target))
        complete = self._exports_complete or (bool(self._cjs_exports) and not self._cjs_dynamic)
        return FileExtraction(
            file_path=self._relative_path,
            language=self._language,
            module_name=self._module_name,
            module_node=module_node,
            nodes=self._nodes,
            contains=self._contains,
            imports=self._imports,
            bases=self._bases,
            usages=self._usages,
            exports=exports,
            exports_complete=complete,
            syntax_errors=self._syntax_errors,
        )

    # -- CommonJS exports ---------------------------------------------------

    def _handle_assignment(self, node: Any, source_id: str, scope_qn: str, class_qn: str | None) -> None:
        """Walk an assignment; ``module.exports``/``exports`` targets become export records."""
        left = node.child_by_field_name("left")
        right = node.child_by_field_name("right")
        if right is not None:
            self._walk_assignment_value(right, source_id, scope_qn, class_qn)
        if left is None:
            return
        surface, exported = self._cjs_export_path(left)
        if surface is None:
            self._walk(left, source_id, scope_qn, class_qn)
            return
        self._record_cjs_export(surface, exported, right)

    def _walk_assignment_value(self, right: Any, source_id: str, scope_qn: str, class_qn: str | None) -> None:
        """Walk an assignment value, defining statically named functions/classes."""
        name_node = right.child_by_field_name("name") if right is not None else None
        if right is not None and name_node is not None:
            if right.type in _FUNCTION_VALUE_TYPES:
                self._define_function(right, self._text(name_node), source_id, scope_qn, class_qn)
                return
            if right.type == "class_expression":
                self._handle_class(right, source_id, scope_qn, class_qn)
                return
        if right is not None:
            self._walk(right, source_id, scope_qn, class_qn)

    def _cjs_export_path(self, left: Any) -> tuple[str | None, str | None]:
        """Classify an assignment target as a CommonJS export surface.

        Args:
            left: The assignment target node.

        Returns:
            ``(surface, exported_name)`` where surface is ``"module.exports"``
            or ``"exports"`` and ``exported_name`` is the statically named
            property (``None`` for a whole-surface assignment), or
            ``(None, None)`` when the target is not a static export.
        """
        parts = self._dotted_parts(left)
        if parts is None:
            if left.type == "subscript_expression":
                self._cjs_dynamic = True  # module.exports[expr] = ...: surface not complete
            return None, None
        root, attrs = parts
        if root == "exports":
            return ("exports", attrs[0]) if len(attrs) == 1 else (None, None)
        if root == "module" and attrs == ("exports",):
            return "module.exports", None
        if root == "module" and len(attrs) == 2 and attrs[0] == "exports":
            return "module.exports", attrs[1]
        return None, None

    def _record_cjs_export(self, surface: str, exported: str | None, right: Any) -> None:
        """Record the statically named exports produced by one CJS assignment."""
        if exported is not None:
            candidate = self._cjs_value_candidate(right)
            if candidate is not None:
                self._cjs_exports.append((exported, *candidate))
            return
        if right is None:
            return
        # A whole-surface assignment replaces previously recorded exports.
        self._cjs_exports = []
        if right.type == "identifier":
            self._cjs_exports.append(("default", self._text(right)))
            return
        if right.type in _FUNCTION_VALUE_TYPES:
            name_node = right.child_by_field_name("name")
            if name_node is not None:
                self._cjs_exports.append(("default", self._text(name_node)))
            return
        if right.type in ("object", "object_pattern"):
            for pair in right.children:
                if pair.type == "pair":
                    key = pair.child_by_field_name("key")
                    value = pair.child_by_field_name("value")
                    if key is None or key.type not in (
                        "property_identifier",
                        "shorthand_property_identifier",
                        "string",
                    ):
                        self._cjs_dynamic = True
                        continue
                    name = self._string_text(key) if key.type == "string" else self._text(key)
                    candidate = self._cjs_value_candidate(value)
                    if candidate is not None:
                        self._cjs_exports.append((name, *candidate))
                elif pair.type == "shorthand_property_identifier":
                    self._cjs_exports.append((self._text(pair), self._text(pair), False))
                elif pair.type in ("method_definition", "pair_pattern"):
                    self._cjs_dynamic = True

    def _cjs_value_candidate(self, value: Any) -> tuple[str, bool] | None:
        """Map an exported value expression to a resolvable export candidate.

        Args:
            value: The exported value expression node.

        Returns:
            ``(candidate, resolved)`` where ``candidate`` is a qualified-name
            candidate when ``resolved``, otherwise a local name that must be
            resolved once the whole file has been walked.
        """
        if value is None:
            return None
        if value.type == "identifier":
            return self._text(value), False
        if value.type in _FUNCTION_VALUE_TYPES:
            name_node = value.child_by_field_name("name")
            if name_node is not None:
                return f"{self._module_name}.{self._text(name_node)}", True
            return None
        if value.type == "call_expression" and self._is_require_call(value):
            specifier = self._string_text(self._first_argument(value.child_by_field_name("arguments")))
            if specifier is not None:
                return self._module_target(specifier), True
        return None

    def _cjs_export_target(self, local: str) -> str | None:
        """Resolve a CommonJS exported name to its qualified-name candidate."""
        prefix = f"{self._module_name}."
        for node in self._nodes:
            if node.kind != "module" and node.qualified_name == f"{prefix}{local}":
                return node.qualified_name
        for record in self._imports:
            if record.binding == local:
                if record.symbol and record.symbol != "*":
                    return f"{record.module_target}.{record.symbol}"
                return record.binds_module or record.module_target
        return None

    def _walk(self, node: Any, source_id: str, scope_qn: str, class_qn: str | None) -> None:
        node_type = node.type
        if node_type == "import_statement":
            self._handle_import(node)
            return
        if node_type == "export_statement":
            self._handle_export(node, source_id, scope_qn, class_qn)
            return
        if node_type in ("function_declaration", "generator_function_declaration"):
            name_node = node.child_by_field_name("name")
            if name_node is not None:
                self._define_function(
                    node,
                    self._text(name_node),
                    source_id,
                    scope_qn,
                    class_qn,
                )
            return
        if node_type == "function_signature":
            return  # TS overload signature: the implementation carries the node
        if node_type in ("class_declaration", "abstract_class_declaration"):
            self._handle_class(node, source_id, scope_qn, class_qn)
            return
        if node_type == "interface_declaration":
            self._handle_interface(node, source_id, scope_qn)
            return
        if node_type == "type_alias_declaration":
            name_node = node.child_by_field_name("name")
            if name_node is not None:
                self._add_definition(node, "interface", self._text(name_node), source_id, scope_qn)
            return
        if node_type == "enum_declaration":
            return  # enums are out of the node vocabulary; they stay chunks
        if node_type in ("lexical_declaration", "variable_declaration"):
            self._handle_variables(node, source_id, scope_qn, class_qn)
            return
        if node_type in ("method_definition", "public_field_definition"):
            self._handle_class_member(node, source_id, scope_qn, class_qn)
            return
        if node_type == "call_expression":
            self._handle_call(node, source_id, scope_qn, class_qn)
            return
        if node_type == "assignment_expression":
            self._handle_assignment(node, source_id, scope_qn, class_qn)
            return
        if node_type == "new_expression":
            self._handle_new(node, source_id, scope_qn, class_qn)
            return
        if node_type == "member_expression":
            parts = self._dotted_parts(node)
            if parts is not None:
                self._record_reference(parts[0], parts[1], node, source_id, scope_qn, class_qn)
                return
            obj = node.child_by_field_name("object")
            if obj is not None:
                self._walk(obj, source_id, scope_qn, class_qn)
            return
        if node.type in ("identifier", "type_identifier", "shorthand_property_identifier"):
            if self._is_structural_identifier(node):
                return
            self._record_reference(self._text(node), (), node, source_id, scope_qn, class_qn)
            return
        if node_type in ("string", "comment", "regex"):
            return
        if node_type in ("class_heritage", "extends_clause", "implements_clause", "extends_type_clause"):
            return  # bases are captured as BaseRecords by the class/interface handlers
        for child in node.children:
            self._walk(child, source_id, scope_qn, class_qn)

    def _is_structural_identifier(self, node: Any) -> bool:
        parent = node.parent
        if parent is None:
            return False
        if parent.type == "variable_declarator":
            return parent.child_by_field_name("name") is node
        if parent.type == "new_expression":
            return parent.child_by_field_name("constructor") is node
        if parent.type in ("public_field_definition", "property_signature", "method_definition"):
            return parent.child_by_field_name("name") is node
        if parent.type == "pair":
            return parent.child_by_field_name("key") is node
        return parent.type in _STRUCTURAL_PARENT_TYPES

    # -- imports and exports ------------------------------------------------

    def _handle_import(self, node: Any) -> None:
        line = self._line(node)
        source = node.child_by_field_name("source")
        specifier = self._string_text(source)
        if specifier is None:
            return
        module_target = self._module_target(specifier)
        # ``import_clause`` is a child type in the grammar, not a field name.
        clause = next((child for child in node.children if child.type == "import_clause"), None)
        if clause is None:
            self._imports.append(
                ImportRecord(
                    binding="", module_target=module_target, binds_module=module_target, symbol=None, line=line
                )
            )
            return
        for child in clause.children:
            if child.type == "identifier":
                self._imports.append(
                    ImportRecord(
                        binding=self._text(child),
                        module_target=module_target,
                        binds_module=module_target,
                        symbol="default",
                        line=line,
                    )
                )
            elif child.type == "namespace_import":
                name_node = next((sub for sub in child.children if sub.type == "identifier"), None)
                if name_node is not None:
                    self._imports.append(
                        ImportRecord(
                            binding=self._text(name_node),
                            module_target=module_target,
                            binds_module=module_target,
                            symbol=None,
                            line=line,
                        )
                    )
            elif child.type == "named_imports":
                for specifier_node in child.children:
                    if specifier_node.type != "import_specifier":
                        continue
                    name_node = specifier_node.child_by_field_name("name")
                    alias_node = specifier_node.child_by_field_name("alias")
                    if name_node is None:
                        continue
                    symbol = self._text(name_node)
                    binding = self._text(alias_node) if alias_node is not None else symbol
                    self._imports.append(
                        ImportRecord(
                            binding=binding,
                            module_target=module_target,
                            binds_module=module_target,
                            symbol=symbol,
                            line=line,
                        )
                    )

    def _handle_export(self, node: Any, source_id: str, scope_qn: str, class_qn: str | None) -> None:
        line = self._line(node)
        children = node.children
        is_default = any(child.type == "default" for child in children)
        source = node.child_by_field_name("source")
        specifier = self._string_text(source)
        declaration = node.child_by_field_name("declaration")
        export_clause = next((child for child in children if child.type == "export_clause"), None)
        namespace_export = next((child for child in children if child.type == "namespace_export"), None)

        if is_default:
            self._exports_complete = True
            if declaration is not None:
                self._export_declaration(declaration, source_id, scope_qn, class_qn, default=True)
            else:
                value = node.child_by_field_name("value")
                if value is not None and value.type == "identifier":
                    name = self._text(value)
                    self._exports.append(ExportRecord(name="default", target=f"{self._module_name}.{name}"))
                    self._record_reference(name, (), value, source_id, scope_qn, class_qn)
                elif value is not None:
                    self._walk(value, source_id, scope_qn, class_qn)
            return

        if namespace_export is not None:
            # `export * as ns from './m'`
            self._exports_complete = True
            if specifier is None and source is not None:
                text = self._string_text(source)
                if text is not None:
                    module_target = self._module_target(text)
                    name_node = next((sub for sub in namespace_export.children if sub.type == "identifier"), None)
                    binding = self._text(name_node) if name_node is not None else ""
                    self._imports.append(
                        ImportRecord(
                            binding=binding,
                            module_target=module_target,
                            binds_module=module_target,
                            symbol=None,
                            line=line,
                        )
                    )
                    if binding:
                        self._exports.append(ExportRecord(name=binding, target=module_target))
            return

        if specifier is not None:
            # Re-export: `export { a as b } from './m'`, `export * from './m'`.
            self._exports_complete = True
            module_target = self._module_target(specifier)
            if export_clause is not None:
                for specifier_node in export_clause.children:
                    if specifier_node.type != "export_specifier":
                        continue
                    name_node = specifier_node.child_by_field_name("name")
                    alias_node = specifier_node.child_by_field_name("alias")
                    if name_node is None:
                        continue
                    symbol = self._text(name_node)
                    binding = self._text(alias_node) if alias_node is not None else symbol
                    self._imports.append(
                        ImportRecord(
                            binding=binding,
                            module_target=module_target,
                            binds_module=module_target,
                            symbol=symbol,
                            line=line,
                        )
                    )
                    self._exports.append(ExportRecord(name=binding, target=f"{module_target}.{symbol}"))
            else:
                self._imports.append(
                    ImportRecord(
                        binding="*",
                        module_target=module_target,
                        binds_module=module_target,
                        symbol="*",
                        line=line,
                    )
                )
                self._exports.append(ExportRecord(name="*", target=f"{module_target}.*"))
            return

        self._exports_complete = True
        if export_clause is not None:
            # Local re-export of own bindings: `export { a, b as c }`.
            for specifier_node in export_clause.children:
                if specifier_node.type != "export_specifier":
                    continue
                name_node = specifier_node.child_by_field_name("name")
                alias_node = specifier_node.child_by_field_name("alias")
                if name_node is None:
                    continue
                symbol = self._text(name_node)
                exported = self._text(alias_node) if alias_node is not None else symbol
                self._exports.append(ExportRecord(name=exported, target=f"{self._module_name}.{symbol}"))
            return
        if declaration is not None:
            self._export_declaration(declaration, source_id, scope_qn, class_qn, default=False)
            return
        for child in children:
            if child.is_named:
                self._walk(child, source_id, scope_qn, class_qn)

    def _export_declaration(
        self, declaration: Any, source_id: str, scope_qn: str, class_qn: str | None, default: bool
    ) -> str | None:
        """Walk an exported declaration and record its export names.

        Named declarations (``export function f``, ``export const x = ...``,
        ``export class C``) export every statically named top-level binding;
        ``export default`` exports the primary name as ``"default"``.

        Args:
            declaration: The exported declaration node.
            source_id: Containing node id (module or class).
            scope_qn: Enclosing scope qualified name.
            class_qn: Enclosing class qualified name when any.
            default: Whether this is a default export.

        Returns:
            The primary defined display name, when statically named.
        """
        before = len(self._nodes)
        self._walk(declaration, source_id, scope_qn, class_qn)
        names = [
            node.display_name for node in self._nodes[before:] if node.kind != "module" and node.parent_id == source_id
        ]
        for name in self._static_declaration_names(declaration):
            if name not in names:
                names.append(name)
        if default:
            if names:
                self._exports.append(ExportRecord(name="default", target=f"{self._module_name}.{names[0]}"))
        else:
            for name in names:
                self._exports.append(ExportRecord(name=name, target=f"{self._module_name}.{name}"))
        return names[0] if names else None

    def _static_declaration_names(self, declaration: Any) -> list[str]:
        """Collect statically named declarators of a variable declaration."""
        if declaration.type not in ("lexical_declaration", "variable_declaration"):
            return []
        names: list[str] = []
        for declarator in declaration.children:
            if declarator.type != "variable_declarator":
                continue
            name_node = declarator.child_by_field_name("name")
            if name_node is not None and name_node.type == "identifier":
                names.append(self._text(name_node))
        return names

    # -- definitions ---------------------------------------------------------

    def _define_function(
        self,
        node: Any,
        name: str,
        parent_id: str,
        scope_qn: str,
        class_qn: str | None,
    ) -> str | None:
        kind = self._definition_kind(name, "method" if class_qn is not None else "function")
        graph_node = self._add_definition(node, kind, name, parent_id, scope_qn)
        if graph_node is None:
            return None
        body = node.child_by_field_name("body")
        bound = self._parameter_names(node.child_by_field_name("parameters"))
        bound |= self._collect_bound_names(body)
        self._bound_stack.append(bound)
        if body is not None:
            if body.type == "statement_block":
                for child in body.children:
                    self._walk(child, source_id=graph_node.id, scope_qn=graph_node.qualified_name, class_qn=class_qn)
            else:
                self._walk(body, source_id=graph_node.id, scope_qn=graph_node.qualified_name, class_qn=class_qn)
        self._bound_stack.pop()
        return graph_node.id

    def _handle_variables(self, node: Any, source_id: str, scope_qn: str, class_qn: str | None) -> None:
        for declarator in node.children:
            if declarator.type != "variable_declarator":
                continue
            self._handle_declarator(declarator, source_id, scope_qn, class_qn)

    def _handle_declarator(self, declarator: Any, source_id: str, scope_qn: str, class_qn: str | None) -> None:
        name_node = declarator.child_by_field_name("name")
        value = declarator.child_by_field_name("value")
        if value is None:
            return
        if value.type == "call_expression" and self._is_require_call(value):
            self._handle_require(declarator, value, scope_qn)
            return
        if value.type in _FUNCTION_VALUE_TYPES and name_node is not None and name_node.type == "identifier":
            self._define_function(value, self._text(name_node), source_id, scope_qn, class_qn)
            return
        if value.type in _FUNCTION_VALUE_TYPES:
            # Anonymous function value with destructured/absent name: walk its
            # body for usages without creating a node.
            self._walk_anonymous_function(value, source_id, scope_qn, class_qn)
            return
        self._walk(value, source_id, scope_qn, class_qn)

    def _walk_anonymous_function(self, node: Any, source_id: str, scope_qn: str, class_qn: str | None) -> None:
        bound = self._parameter_names(node.child_by_field_name("parameters"))
        bound |= self._collect_bound_names(node.child_by_field_name("body"))
        self._bound_stack.append(bound)
        body = node.child_by_field_name("body")
        if body is not None:
            if body.type == "statement_block":
                for child in body.children:
                    self._walk(child, source_id=source_id, scope_qn=scope_qn, class_qn=class_qn)
            else:
                self._walk(body, source_id=source_id, scope_qn=scope_qn, class_qn=class_qn)
        self._bound_stack.pop()

    def _is_require_call(self, node: Any) -> bool:
        func = node.child_by_field_name("function")
        return func is not None and func.type == "identifier" and self._text(func) == "require"

    def _handle_require(self, declarator: Any, call: Any, scope_qn: str) -> None:
        """Record a CommonJS ``require`` as an import record (or unresolved)."""
        line = self._line(call)
        arguments = call.child_by_field_name("arguments")
        specifier = None
        raw = ""
        first = self._first_argument(arguments)
        if first is not None:
            raw = self._text(first)
            specifier = self._string_text(first)
        if specifier is not None:
            module_target = self._module_target(specifier)
        else:
            module_target = raw
        name_node = declarator.child_by_field_name("name")
        if specifier is None:
            # Dynamic require: preserve an unresolved import edge.
            self._imports.append(
                ImportRecord(
                    binding="",
                    module_target=module_target,
                    binds_module=module_target,
                    symbol=None,
                    line=line,
                )
            )
            return
        if name_node is None:
            self._imports.append(
                ImportRecord(
                    binding="", module_target=module_target, binds_module=module_target, symbol=None, line=line
                )
            )
            return
        if name_node.type == "identifier":
            self._imports.append(
                ImportRecord(
                    binding=self._text(name_node),
                    module_target=module_target,
                    binds_module=module_target,
                    symbol=None,
                    line=line,
                )
            )
            return
        # Destructured require: `const { a, b: c } = require('./m')`.
        names: list[tuple[str, str]] = []

        def collect(pattern: Any) -> None:
            if pattern is None:
                return
            if pattern.type == "shorthand_property_identifier_pattern":
                names.append((self._text(pattern), self._text(pattern)))
            elif pattern.type == "pair_pattern":
                key = pattern.child_by_field_name("key")
                value = pattern.child_by_field_name("value")
                if key is not None and value is not None:
                    names.append((self._text(value), self._text(key)))
            elif pattern.type == "rest_pattern":
                for child in pattern.children:
                    collect(child)
            elif pattern.type == "assignment_pattern":
                collect(pattern.child_by_field_name("left"))
            else:
                for child in pattern.children:
                    collect(child)

        collect(name_node)
        for binding, symbol in names:
            self._imports.append(
                ImportRecord(
                    binding=binding,
                    module_target=module_target,
                    binds_module=module_target,
                    symbol=symbol,
                    line=line,
                )
            )

    def _module_target(self, specifier: str) -> str:
        """Map an import specifier to a dotted module candidate.

        Relative specifiers resolve against the importing file's directory;
        package specifiers stay as-is and resolve only if a repository module
        happens to share the name (they normally stay unresolved).

        Args:
            specifier: Raw import/require specifier text.

        Returns:
            Dotted module candidate (never empty).
        """
        stem = specifier
        for extension in (*JAVASCRIPT_EXTENSIONS, *TYPESCRIPT_EXTENSIONS):
            if stem.endswith(extension):
                stem = stem[: -len(extension)]
                break
        if not stem.startswith("."):
            parts = [part for part in stem.split("/") if part]
            return ".".join(parts) if parts else specifier
        importer_parts = self._module_name.split(".")
        if len(importer_parts) > 1 and importer_parts[-1] in _INDEX_BASENAMES:
            importer_parts = importer_parts[:-1]
        directory = importer_parts[:-1]
        target_parts = list(directory)
        for segment in stem.split("/"):
            if segment in ("", "."):
                continue
            if segment == "..":
                if target_parts:
                    target_parts.pop()
                continue
            target_parts.append(segment)
        while len(target_parts) > 1 and target_parts[-1] in _INDEX_BASENAMES:
            target_parts = target_parts[:-1]
        return ".".join(target_parts) if target_parts else self._module_name

    # -- classes and interfaces ----------------------------------------------

    def _handle_class(self, node: Any, source_id: str, scope_qn: str, class_qn: str | None) -> None:
        name_node = node.child_by_field_name("name")
        if name_node is None:
            for child in node.children:
                if child.is_named:
                    self._walk(child, source_id, scope_qn, class_qn)
            return
        name = self._text(name_node)
        kind = self._definition_kind(name, "class")
        graph_node = self._add_definition(node, kind, name, source_id, scope_qn)
        if graph_node is None:
            return
        line = self._line(node)
        heritage = next((child for child in node.children if child.type == "class_heritage"), None)
        if heritage is not None:
            for child in heritage.children:
                if child.type == "extends_clause":
                    # TypeScript grammar: the base expression sits in `value`.
                    value = child.child_by_field_name("value")
                    if value is not None:
                        parts = self._dotted_parts(value)
                        root, attrs = parts if parts is not None else (None, ())
                        self._bases.append(
                            BaseRecord(
                                class_node_id=graph_node.id,
                                root=root,
                                attrs=attrs,
                                text=self._text(value),
                                line=line,
                            )
                        )
                elif child.type == "implements_clause":
                    for sub in child.children:
                        if sub.type in ("type_identifier", "identifier"):
                            self._bases.append(
                                BaseRecord(
                                    class_node_id=graph_node.id,
                                    root=self._text(sub),
                                    attrs=(),
                                    text=self._text(sub),
                                    line=line,
                                )
                            )
                elif child.type in ("identifier", "member_expression"):
                    # JavaScript grammar: `class Dog extends Animal` puts the
                    # base expression directly in the heritage.
                    parts = self._dotted_parts(child)
                    root, attrs = parts if parts is not None else (None, ())
                    self._bases.append(
                        BaseRecord(
                            class_node_id=graph_node.id,
                            root=root,
                            attrs=attrs,
                            text=self._text(child),
                            line=line,
                        )
                    )
        body = next((child for child in node.children if child.type == "class_body"), None)
        self._bound_stack.append(self._collect_bound_names(body))
        if body is not None:
            for child in body.children:
                if child.type in ("decorator",):
                    for sub in child.children:
                        self._walk(sub, source_id=graph_node.id, scope_qn=graph_node.qualified_name, class_qn=None)
                    continue
                if not child.is_named:
                    continue
                if child.type in ("method_definition", "public_field_definition"):
                    self._handle_class_member(
                        child,
                        source_id=graph_node.id,
                        scope_qn=graph_node.qualified_name,
                        class_qn=graph_node.qualified_name,
                    )
                    continue
                if child.type in (
                    "method_signature",
                    "abstract_method_signature",
                    "property_signature",
                    "index_signature",
                ):
                    continue  # signatures carry no runtime definition
                if child.type == "static_block":
                    for sub in child.children:
                        self._walk(
                            sub,
                            source_id=graph_node.id,
                            scope_qn=graph_node.qualified_name,
                            class_qn=graph_node.qualified_name,
                        )
                    continue
                self._walk(
                    child,
                    source_id=graph_node.id,
                    scope_qn=graph_node.qualified_name,
                    class_qn=graph_node.qualified_name,
                )
        self._bound_stack.pop()

    def _handle_class_member(self, node: Any, source_id: str, scope_qn: str, class_qn: str | None) -> None:
        name_node = node.child_by_field_name("name")
        if name_node is None or name_node.type not in ("property_identifier", "identifier"):
            if name_node is not None:
                return  # computed/symbol names stay unresolved
            for child in node.children:
                if child.is_named:
                    self._walk(child, source_id, scope_qn, class_qn)
            return
        name = self._text(name_node)
        value = node.child_by_field_name("value")
        if node.type == "public_field_definition" and (value is None or value.type not in _FUNCTION_VALUE_TYPES):
            return  # plain data field: owned by the class, not a definition node
        if node.type == "method_definition" and node.child_by_field_name("body") is None:
            return  # signature without a body (abstract/overload)
        for child in node.children:
            if child.type == "decorator":
                for sub in child.children:
                    self._walk(sub, source_id=source_id, scope_qn=scope_qn, class_qn=class_qn)
        self._define_function(
            node if node.type == "method_definition" else value,
            name,
            source_id,
            scope_qn,
            class_qn,
        )

    def _handle_interface(self, node: Any, source_id: str, scope_qn: str) -> None:
        name_node = node.child_by_field_name("name")
        if name_node is None:
            return
        name = self._text(name_node)
        graph_node = self._add_definition(node, "interface", name, source_id, scope_qn)
        if graph_node is None:
            return
        line = self._line(node)
        extends = next((child for child in node.children if child.type == "extends_type_clause"), None)
        if extends is not None:
            for child in extends.children:
                if child.type in ("type_identifier", "identifier"):
                    self._bases.append(
                        BaseRecord(
                            class_node_id=graph_node.id,
                            root=self._text(child),
                            attrs=(),
                            text=self._text(child),
                            line=line,
                        )
                    )

    # -- calls ---------------------------------------------------------------

    def _handle_call(self, node: Any, source_id: str, scope_qn: str, class_qn: str | None) -> None:
        func = node.child_by_field_name("function")
        if func is not None:
            if func.type == "identifier" and self._text(func) == "require":
                self._handle_bare_require(node)
                return
            if func.type == "import":
                self._handle_dynamic_import(node)
                return
            if func.type == "call":
                # A call result being called: record the inner call too.
                self._handle_call(func, source_id, scope_qn, class_qn)
                self._record_call(None, node, source_id, scope_qn, class_qn, self._text(func))
            else:
                parts = self._dotted_parts(func)
                self._record_call(parts, node, source_id, scope_qn, class_qn, self._text(func))
        arguments = node.child_by_field_name("arguments")
        if arguments is not None:
            for child in arguments.children:
                self._walk(child, source_id, scope_qn, class_qn)

    def _handle_new(self, node: Any, source_id: str, scope_qn: str, class_qn: str | None) -> None:
        constructor = node.child_by_field_name("constructor")
        if constructor is not None:
            parts = self._dotted_parts(constructor)
            self._record_call(parts, node, source_id, scope_qn, class_qn, self._text(constructor))
        arguments = node.child_by_field_name("arguments")
        if arguments is not None:
            for child in arguments.children:
                self._walk(child, source_id, scope_qn, class_qn)

    def _handle_bare_require(self, node: Any) -> None:
        """A ``require(...)`` outside a declarator: imports edge only."""
        arguments = node.child_by_field_name("arguments")
        specifier = None
        raw = ""
        first = self._first_argument(arguments)
        if first is not None:
            raw = self._text(first)
            specifier = self._string_text(first)
        line = self._line(node)
        module_target = self._module_target(specifier) if specifier is not None else raw
        self._imports.append(
            ImportRecord(
                binding="",
                module_target=module_target,
                binds_module=module_target,
                symbol=None,
                line=line,
            )
        )

    def _handle_dynamic_import(self, node: Any) -> None:
        """A dynamic ``import(...)``: static strings resolve, expressions do not."""
        arguments = node.child_by_field_name("arguments")
        specifier = None
        raw = ""
        first = self._first_argument(arguments)
        if first is not None:
            raw = self._text(first)
            specifier = self._string_text(first)
        line = self._line(node)
        module_target = self._module_target(specifier) if specifier is not None else raw
        self._imports.append(
            ImportRecord(
                binding="",
                module_target=module_target,
                binds_module=module_target,
                symbol=None,
                line=line,
            )
        )
