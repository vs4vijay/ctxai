"""Deterministic Python symbol extraction and resolution for the symbol graph (IG-01).

The adapter runs in two phases:

1. ``extract_file`` parses one file with tree-sitter (the same parser the
   chunker uses) and produces structural records: definition nodes,
   containment edges, import records, base-class records, and call/reference
   usages. It never imports or executes indexed code.
2. ``resolve_edges`` turns the per-file records into final graph edges using a
   repository-wide symbol index. The resolution ladder is deliberately
   conservative: a target is connected only when statically unambiguous
   (import binding, lexical/module scope, or a unique repository-wide display
   name); everything else stays an *unresolved* edge (imports, calls,
   inheritance) or is simply not recorded (references), never guessed.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

from tree_sitter_language_pack import get_parser

from .model import (
    CONFIDENCE_EXACT,
    CONFIDENCE_PROBABLE,
    CONFIDENCE_UNRESOLVED,
    PYTHON_EXTRACTOR_VERSION,
    PYTHON_RESOLVER_VERSION,
    GraphEdge,
    GraphNode,
    derive_edge_id,
    derive_node_id,
)

PYTHON_EXTENSION = ".py"

# Calls to these names create no edge when nothing in the repository defines
# the same name; they are runtime builtins, not repository symbols.
BUILTIN_CALLABLES = frozenset(
    {
        "abs",
        "all",
        "any",
        "ascii",
        "bin",
        "bool",
        "bytearray",
        "bytes",
        "callable",
        "chr",
        "classmethod",
        "compile",
        "complex",
        "delattr",
        "dict",
        "dir",
        "divmod",
        "enumerate",
        "eval",
        "exec",
        "exit",
        "filter",
        "float",
        "format",
        "frozenset",
        "getattr",
        "globals",
        "hasattr",
        "hash",
        "hex",
        "id",
        "input",
        "int",
        "isinstance",
        "issubclass",
        "iter",
        "len",
        "list",
        "locals",
        "map",
        "max",
        "memoryview",
        "min",
        "next",
        "object",
        "oct",
        "open",
        "ord",
        "pow",
        "print",
        "property",
        "quit",
        "range",
        "repr",
        "reversed",
        "round",
        "set",
        "setattr",
        "slice",
        "sorted",
        "staticmethod",
        "str",
        "sum",
        "super",
        "tuple",
        "type",
        "vars",
        "zip",
        "__build_class__",
        "__import__",
    }
)

# Names that are lexically bound and never repository symbols.
SELF_NAMES = frozenset({"self", "cls", "mcs"})


def module_name_for_path(relative_path: str) -> str:
    """Derive the dotted module name from a repository-relative Python path.

    Args:
        relative_path: Repository-relative path using forward slashes.

    Returns:
        Dotted module name; ``__init__.py`` files name their package.
    """
    without_extension = (
        relative_path[: -len(PYTHON_EXTENSION)] if relative_path.endswith(PYTHON_EXTENSION) else relative_path
    )
    parts = [part for part in without_extension.split("/") if part]
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts) if parts else "__module__"


def _visibility(name: str) -> str:
    if name.startswith("__") and name.endswith("__") and len(name) > 4:
        return "public"
    return "private" if name.startswith("_") else "public"


@dataclass(frozen=True)
class ImportRecord:
    """One import binding produced by a single import statement.

    Attributes:
        binding: Local name the import binds (``"*"`` for star imports).
        module_target: Absolute dotted module target of the statement
            (relative imports are pre-resolved against the file's package);
            this is what the ``imports`` edge points at.
        binds_module: Dotted module name the *binding* refers to (the root
            package for plain ``import a.b``, the full target for aliases).
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
    """One base-class expression of a class definition.

    Attributes:
        class_node_id: Node id of the inheriting class.
        root: Leftmost identifier of the base expression (``None`` when not a
            plain dotted chain).
        attrs: Remaining dotted components.
        text: Raw source text of the base expression.
        line: 1-based evidence line of the class definition.
    """

    class_node_id: str
    root: str | None
    attrs: tuple[str, ...]
    text: str
    line: int


@dataclass
class FileExtraction:
    """Structural extraction result for a single Python file."""

    file_path: str
    language: str
    module_name: str
    module_node: GraphNode
    nodes: list[GraphNode]
    contains: list[GraphEdge]
    imports: list[ImportRecord]
    bases: list[BaseRecord]
    usages: list[UsageRecord]
    syntax_errors: int = 0


@dataclass
class SymbolIndexes:
    """Repository-wide lookup tables used to resolve usage records.

    Attributes:
        symbol: Qualified name to node id for non-module definitions.
        module: Dotted module name to module node id.
        packages: Ancestor prefixes of known modules (namespace packages have
            no node but make dotted chains walkable).
        display: Display name to the (sorted) node ids sharing it.
        qualified_by_id: Node id to qualified name.
        kind_by_id: Node id to node kind.
    """

    symbol: dict[str, str] = field(default_factory=dict)
    module: dict[str, str] = field(default_factory=dict)
    packages: set[str] = field(default_factory=set)
    display: dict[str, list[str]] = field(default_factory=dict)
    qualified_by_id: dict[str, str] = field(default_factory=dict)
    kind_by_id: dict[str, str] = field(default_factory=dict)


def build_symbol_indexes(nodes: list[GraphNode]) -> SymbolIndexes:
    """Build resolution indexes from a repository-wide node set.

    Args:
        nodes: All candidate nodes (existing store nodes plus fresh extractions).

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
    return indexes


def _walkable(indexes: SymbolIndexes, qualified: str) -> bool:
    return qualified in indexes.symbol or qualified in indexes.module or qualified in indexes.packages


def _chain_from(indexes: SymbolIndexes, base_qualified: str, attrs: tuple[str, ...]) -> str | None:
    """Walk ``attrs`` below a qualified name through the symbol indexes.

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
    if not _walkable(indexes, current):
        return None
    node_id = indexes.symbol.get(current) or indexes.module.get(current)
    for attr in attrs:
        current = f"{current}.{attr}"
        if not _walkable(indexes, current):
            return None
        node_id = indexes.symbol.get(current) or indexes.module.get(current)
    return node_id


class PythonAdapter:
    """LanguageAdapter for Python source files (tree-sitter based)."""

    language = "python"
    extractor_version = PYTHON_EXTRACTOR_VERSION
    resolver_version = PYTHON_RESOLVER_VERSION

    def __init__(self) -> None:
        """Create the adapter and its shared tree-sitter parser.

        Raises:
            RuntimeError: If the Python grammar is unavailable.
        """
        self._parser = get_parser("python")

    @staticmethod
    def supports_file(relative_path: str) -> bool:
        """Return whether the adapter handles this repository-relative path.

        Args:
            relative_path: Repository-relative file path.

        Returns:
            True for ``*.py`` files.
        """
        return relative_path.endswith(PYTHON_EXTENSION)

    def extract_file(self, relative_path: str, source: bytes, repository_root: str) -> FileExtraction:
        """Parse one Python file into structural graph records.

        Args:
            relative_path: Repository-relative path with forward slashes.
            source: Raw file bytes (never imported or executed).
            repository_root: Canonical repository root used for stable ids.

        Returns:
            The :class:`FileExtraction` for the file; syntax errors are
            tolerated and counted, they never raise.
        """
        walker = _FileWalker(self._parser, relative_path, source, repository_root, self.language, self.resolver_version)
        return walker.run()


class _FileWalker:
    """Single-file tree-sitter walker producing a :class:`FileExtraction`."""

    def __init__(
        self,
        parser: Any,
        relative_path: str,
        source: bytes,
        repository_root: str,
        language: str,
        resolver_version: str,
    ) -> None:
        self._parser = parser
        self._repository_root = repository_root
        self._relative_path = relative_path
        self._source = source
        self._language = language
        self._resolver_version = resolver_version
        self._module_name = module_name_for_path(relative_path)
        self._nodes: list[GraphNode] = []
        self._contains: list[GraphEdge] = []
        self._imports: list[ImportRecord] = []
        self._bases: list[BaseRecord] = []
        self._usages: list[UsageRecord] = []
        self._syntax_errors = 0
        # Lexically bound names per active scope (module, function, class).
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
        """Return ``(root, attrs)`` for identifier/attribute chains, else None."""
        if node.type == "identifier":
            return self._text(node), ()
        if node.type != "attribute":
            return None
        parts: list[str] = []
        current = node
        while current is not None and current.type == "attribute":
            attr = current.child_by_field_name("attribute") or current.child_by_field_name("attr")
            if attr is None or attr.type != "identifier":
                return None
            parts.append(self._text(attr))
            current = current.child_by_field_name("object")
        if current is None or current.type != "identifier":
            return None
        parts.append(self._text(current))
        parts.reverse()
        return parts[0], tuple(parts[1:])

    # -- record helpers ----------------------------------------------------

    def _add_definition(self, node: Any, kind: str, name: str, parent_id: str | None, prefix: str) -> GraphNode:
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
            visibility=_visibility(name),
            source_hash=self._node_hash(node),
        )
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
            confidence=CONFIDENCE_UNRESOLVED if target_id is None else CONFIDENCE_EXACT,
            resolver_version=self._resolver_version,
        )

    # -- bound-name collection --------------------------------------------

    def _collect_bound_names(self, body: Any) -> set[str]:
        """Collect names bound anywhere in a scope body (pre-pass).

        Import statements are skipped: imported names resolve through the
        import map, not as locals.

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
        if node.type in ("function_definition", "class_definition", "lambda"):
            return  # nested scopes own their names
        if node.type in ("assignment", "augmented_assignment", "for_statement", "for_in_clause"):
            left = node.child_by_field_name("left")
            if left is not None:
                _collect_target_identifiers(left, bound)
        elif node.type == "as_pattern":
            for child in node.children:
                if child.type in ("identifier", "as_pattern_target"):
                    _collect_target_identifiers(child, bound)
        elif node.type == "named_expression":
            name = node.child_by_field_name("name")
            if name is not None and name.type == "identifier":
                bound.add(self._text(name))
        elif node.type in ("global_statement", "nonlocal_statement"):
            for child in node.children:
                if child.type == "identifier":
                    bound.add(self._text(child))
        for child in node.children:
            self._visit_bound(child, bound)

    def _parameter_names(self, parameters: Any) -> set[str]:
        names: set[str] = set()
        if parameters is None:
            return names
        for child in parameters.children:
            if child.type == "identifier":
                names.add(self._text(child))
            elif child.type in (
                "default_parameter",
                "typed_default_parameter",
                "typed_parameter",
                "list_splat_pattern",
                "dictionary_splat_pattern",
            ):
                name_node = child.child_by_field_name("name")
                if name_node is not None and name_node.type == "identifier":
                    names.add(self._text(name_node))
                    continue
                for sub in child.children:
                    if sub.type == "identifier":
                        names.add(self._text(sub))
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
        )
        self._nodes.append(module_node)
        self._bound_stack.append(self._collect_bound_names(root))
        for child in root.children:
            self._walk(child, source_id=module_node.id, scope_qn=self._module_name, class_qn=None)
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
            syntax_errors=self._syntax_errors,
        )

    def _walk(self, node: Any, source_id: str, scope_qn: str, class_qn: str | None) -> None:
        node_type = node.type
        if node_type in ("import_statement", "import_from_statement"):
            self._handle_import(node)
            return
        if node_type == "function_definition":
            self._handle_function(node, source_id, scope_qn, class_qn)
            return
        if node_type == "class_definition":
            self._handle_class(node, source_id, scope_qn, class_qn)
            return
        if node_type == "decorated_definition":
            self._handle_decorated(node, source_id, scope_qn, class_qn)
            return
        if node_type == "call":
            self._handle_call(node, source_id, scope_qn, class_qn)
            return
        if node_type == "attribute":
            parts = self._dotted_parts(node)
            if parts is not None:
                self._record_reference(parts[0], parts[1], node, source_id, scope_qn, class_qn)
                return
            obj = node.child_by_field_name("object")
            if obj is not None:
                self._walk(obj, source_id, scope_qn, class_qn)
            return
        if node_type == "identifier":
            if self._is_structural_identifier(node):
                return
            self._record_reference(self._text(node), (), node, source_id, scope_qn, class_qn)
            return
        if node_type in ("string", "comment"):
            return
        for child in node.children:
            self._walk(child, source_id, scope_qn, class_qn)

    def _is_structural_identifier(self, node: Any) -> bool:
        parent = node.parent
        if parent is None:
            return False
        if parent.type == "keyword_argument" and parent.child_by_field_name("name") is node:
            return True
        return parent.type in ("import_statement", "import_from_statement", "aliased_import", "dotted_name")

    def _record_reference(
        self,
        root: str,
        attrs: tuple[str, ...],
        node: Any,
        source_id: str,
        scope_qn: str,
        class_qn: str | None,
    ) -> None:
        if root in SELF_NAMES or self._bound(root):
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

    def _handle_call(self, node: Any, source_id: str, scope_qn: str, class_qn: str | None) -> None:
        func = node.child_by_field_name("function")
        if func is not None:
            attrs: tuple[str, ...]
            if func.type == "call":
                # A call result being called: record the inner call too.
                self._handle_call(func, source_id, scope_qn, class_qn)
                root, attrs, text = None, (), self._text(func)
            else:
                parts = self._dotted_parts(func)
                root = parts[0] if parts else None
                attrs = parts[1] if parts else ()
                text = self._text(func)
            self._usages.append(
                UsageRecord(
                    kind="call",
                    source_node_id=source_id,
                    scope_qn=scope_qn,
                    class_qn=class_qn,
                    root=root,
                    attrs=attrs,
                    text=text,
                    line=self._line(node),
                )
            )
        arguments = node.child_by_field_name("arguments")
        if arguments is not None:
            for child in arguments.children:
                self._walk(child, source_id, scope_qn, class_qn)

    def _handle_import(self, node: Any) -> None:
        line = self._line(node)
        if node.type == "import_statement":
            for child in node.children:
                if child.type == "dotted_name":
                    target = self._text(child)
                    self._imports.append(
                        ImportRecord(
                            binding=target.split(".")[0],
                            module_target=target,
                            binds_module=target.split(".")[0],
                            symbol=None,
                            line=line,
                        )
                    )
                elif child.type == "aliased_import":
                    name_node = child.child_by_field_name("name")
                    alias = child.child_by_field_name("alias")
                    if name_node is None:
                        continue
                    target = self._text(name_node)
                    binding = self._text(alias) if alias is not None else target.split(".")[0]
                    self._imports.append(
                        ImportRecord(
                            binding=binding,
                            module_target=target,
                            binds_module=target if alias is not None else target.split(".")[0],
                            symbol=None,
                            line=line,
                        )
                    )
            return

        module_field = node.child_by_field_name("module_name")
        if module_field is not None and module_field.type == "relative_import":
            level, base = self._parse_relative_import(module_field)
        elif module_field is not None:
            level, base = 0, self._text(module_field)
        else:
            level, base = 0, ""
        target_module = self._absolute_module(base, level)
        for child in node.children:
            if child is module_field or child.type in ("from", "import", ",", "(", ")", "relative_import"):
                continue
            if child.type == "dotted_name":
                symbol = self._text(child)
                self._imports.append(
                    ImportRecord(
                        binding=symbol,
                        module_target=target_module,
                        binds_module=target_module,
                        symbol=symbol,
                        line=line,
                    )
                )
            elif child.type == "aliased_import":
                name_node = child.child_by_field_name("name")
                alias = child.child_by_field_name("alias")
                if name_node is None:
                    continue
                symbol = self._text(name_node)
                binding = self._text(alias) if alias is not None else symbol
                self._imports.append(
                    ImportRecord(
                        binding=binding,
                        module_target=target_module,
                        binds_module=target_module,
                        symbol=symbol,
                        line=line,
                    )
                )
            elif child.type == "wildcard_import":
                self._imports.append(
                    ImportRecord(
                        binding="*", module_target=target_module, binds_module=target_module, symbol="*", line=line
                    )
                )

    def _parse_relative_import(self, node: Any) -> tuple[int, str]:
        level = 0
        base_parts: list[str] = []
        for child in node.children:
            if child.type == "import_prefix":
                level += self._text(child).count(".")
            elif child.type == "dotted_name":
                base_parts.append(self._text(child))
        return level, ".".join(base_parts)

    def _absolute_module(self, base: str, level: int) -> str:
        if level == 0:
            return base
        package_parts = self._module_name.split(".")[:-1]
        for _ in range(level - 1):
            if package_parts:
                package_parts.pop()
        package = ".".join(package_parts)
        if base and package:
            return f"{package}.{base}"
        return base or package

    def _handle_decorated(self, node: Any, source_id: str, scope_qn: str, class_qn: str | None) -> None:
        inner = next(
            (child for child in node.children if child.type in ("function_definition", "class_definition")), None
        )
        decorators = [child for child in node.children if child.type == "decorator"]
        if inner is None:
            for child in node.children:
                self._walk(child, source_id, scope_qn, class_qn)
            return
        defined_id = (
            self._handle_function(inner, source_id, scope_qn, class_qn)
            if inner.type == "function_definition"
            else self._handle_class(inner, source_id, scope_qn, class_qn)
        )
        for decorator in decorators:
            for child in decorator.children:
                if child.type == "@":
                    continue
                self._walk(child, source_id=defined_id, scope_qn=scope_qn, class_qn=class_qn)

    def _handle_function(self, node: Any, parent_id: str, scope_qn: str, class_qn: str | None) -> str:
        name = self._text(node.child_by_field_name("name"))
        if _is_test_name(name):
            kind = "test"
        elif class_qn is not None:
            kind = "method"
        else:
            kind = "function"
        graph_node = self._add_definition(node, kind, name, parent_id, scope_qn)
        body = node.child_by_field_name("body")
        bound = self._parameter_names(node.child_by_field_name("parameters"))
        bound |= self._collect_bound_names(body)
        self._bound_stack.append(bound)
        if body is not None:
            for child in body.children:
                self._walk(child, source_id=graph_node.id, scope_qn=graph_node.qualified_name, class_qn=class_qn)
        self._bound_stack.pop()
        return graph_node.id

    def _handle_class(self, node: Any, parent_id: str, scope_qn: str, class_qn: str | None) -> str:
        name = self._text(node.child_by_field_name("name"))
        kind = "test" if name.startswith("Test") else "class"
        graph_node = self._add_definition(node, kind, name, parent_id, scope_qn)
        line = self._line(node)
        superclasses = node.child_by_field_name("superclasses")
        if superclasses is not None:
            for child in superclasses.children:
                if child.type in ("(", ")", ","):
                    continue
                if child.type == "keyword_argument":
                    value = child.child_by_field_name("value")
                    if value is not None:
                        self._walk(value, source_id=graph_node.id, scope_qn=graph_node.qualified_name, class_qn=None)
                    continue
                parts = self._dotted_parts(child)
                root, attrs = parts if parts is not None else (None, ())
                self._bases.append(
                    BaseRecord(class_node_id=graph_node.id, root=root, attrs=attrs, text=self._text(child), line=line)
                )
        body = node.child_by_field_name("body")
        self._bound_stack.append(self._collect_bound_names(body))
        if body is not None:
            for child in body.children:
                self._walk(
                    child,
                    source_id=graph_node.id,
                    scope_qn=graph_node.qualified_name,
                    class_qn=graph_node.qualified_name,
                )
        self._bound_stack.pop()
        return graph_node.id


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


def _collect_target_identifiers(target: Any, bound: set[str]) -> None:
    """Collect assignment/loop target identifiers into ``bound``.

    Args:
        target: tree-sitter node on the left-hand side of an assignment.
        bound: The set to extend.
    """
    if target.type == "identifier":
        bound.add(target.text.decode("utf-8", errors="ignore"))
        return
    if target.type in ("pattern_list", "tuple_pattern", "list_pattern", "as_pattern_target"):
        for child in target.children:
            _collect_target_identifiers(child, bound)


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
        edges.extend(extraction.contains)
        import_edges, bindings = _resolve_imports(extraction, indexes, repository_root)
        edges.extend(import_edges)
        edges.extend(_resolve_bases(extraction, bindings, indexes, repository_root))
        edges.extend(_resolve_usages(extraction, bindings, indexes, repository_root))
    unique: dict[str, GraphEdge] = {}
    for edge in edges:
        unique.setdefault(edge.id, edge)
    return sorted(unique.values(), key=lambda item: item.id)


def _make_resolved_edge(
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


def _resolve_imports(
    extraction: FileExtraction,
    indexes: SymbolIndexes,
    repository_root: str,
) -> tuple[list[GraphEdge], dict[str, tuple[str, Any]]]:
    """Emit ``imports`` edges and return the module's import binding map.

    Args:
        extraction: The extraction whose imports to resolve.
        indexes: Repository-wide symbol indexes.
        repository_root: Canonical repository root.

    Returns:
        Tuple of (edges, binding map). The binding map maps binding names to
        ``("node", id)``, ``("module", qn)``, or ``("unresolved", text)`` and
        is used by later usage resolution.
    """
    resolver_version = PYTHON_RESOLVER_VERSION
    bindings: dict[str, tuple[str, Any]] = {}
    edges: list[GraphEdge] = []
    module_id = extraction.module_node.id
    for record in extraction.imports:
        if record.symbol is None:
            target_id = indexes.module.get(record.module_target)
            edges.append(
                _make_resolved_edge(
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
            bindings[record.binding] = ("module", record.binds_module)
        elif record.symbol == "*":
            text = f"{record.module_target}.*"
            edges.append(
                _make_resolved_edge(
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
            bindings.setdefault(record.binding, ("unresolved", text))
        else:
            candidate = f"{record.module_target}.{record.symbol}"
            target_id = indexes.symbol.get(candidate) or indexes.module.get(candidate)
            if target_id is not None:
                bindings[record.binding] = ("node", target_id)
                edges.append(
                    _make_resolved_edge(
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
                    _make_resolved_edge(
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


def _resolve_dotted(
    extraction: FileExtraction,
    bindings: dict[str, tuple[str, Any]],
    indexes: SymbolIndexes,
    root: str | None,
    attrs: tuple[str, ...],
    scope_qn: str,
    class_qn: str | None,
    preserve_unresolved: bool,
) -> tuple[str | None, str | None]:
    """Resolve a dotted usage through the conservative ladder.

    Args:
        extraction: Extraction the usage belongs to.
        bindings: The module's import binding map.
        indexes: Repository-wide symbol indexes.
        root: Leftmost identifier (``None`` for dynamic expressions).
        attrs: Remaining dotted components.
        scope_qn: Enclosing lexical scope qualified name.
        class_qn: Enclosing class qualified name when any.
        preserve_unresolved: Whether an unresolved edge must be preserved (calls,
            inheritance) as opposed to silently skipped (references).

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
            resolved = _chain_from(indexes, base_qualified, attrs) if base_qualified else None
            return (resolved, CONFIDENCE_EXACT) if resolved else (None, unresolved)
        if kind == "module":
            if not attrs:
                return indexes.module.get(value), CONFIDENCE_EXACT
            resolved = _chain_from(indexes, value, attrs)
            return (resolved, CONFIDENCE_EXACT) if resolved else (None, unresolved)
        return None, unresolved

    if root in SELF_NAMES and class_qn is not None and attrs:
        resolved = _chain_from(indexes, class_qn, attrs)
        if resolved:
            return resolved, CONFIDENCE_EXACT
        return None, unresolved

    if scope_qn:
        resolved = _chain_from(indexes, f"{scope_qn}.{root}", attrs)
        if resolved:
            return resolved, CONFIDENCE_EXACT

    module_resolved = _chain_from(indexes, f"{extraction.module_name}.{root}", attrs)
    if module_resolved:
        return module_resolved, CONFIDENCE_EXACT

    display_ids = indexes.display.get(root, [])
    if len(display_ids) == 1 and not attrs:
        return display_ids[0], CONFIDENCE_PROBABLE

    if root in BUILTIN_CALLABLES:
        return None, None

    return None, unresolved


def _resolve_bases(
    extraction: FileExtraction,
    bindings: dict[str, tuple[str, Any]],
    indexes: SymbolIndexes,
    repository_root: str,
) -> list[GraphEdge]:
    edges: list[GraphEdge] = []
    for record in extraction.bases:
        node_id, confidence = _resolve_dotted(
            extraction,
            bindings,
            indexes,
            record.root,
            record.attrs,
            extraction.module_name,
            None,
            preserve_unresolved=True,
        )
        if node_id is not None:
            edges.append(
                _make_resolved_edge(
                    repository_root,
                    PYTHON_RESOLVER_VERSION,
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
                _make_resolved_edge(
                    repository_root,
                    PYTHON_RESOLVER_VERSION,
                    extraction.file_path,
                    "inherits",
                    record.class_node_id,
                    None,
                    record.text,
                    record.line,
                )
            )
    return edges


def _resolve_usages(
    extraction: FileExtraction,
    bindings: dict[str, tuple[str, Any]],
    indexes: SymbolIndexes,
    repository_root: str,
) -> list[GraphEdge]:
    edges: list[GraphEdge] = []
    for usage in extraction.usages:
        node_id, confidence = _resolve_dotted(
            extraction,
            bindings,
            indexes,
            usage.root,
            usage.attrs,
            usage.scope_qn,
            usage.class_qn,
            preserve_unresolved=usage.kind == "call",
        )
        if usage.kind == "reference":
            if node_id is not None and confidence is not None:
                edges.append(
                    _make_resolved_edge(
                        repository_root,
                        PYTHON_RESOLVER_VERSION,
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
            _make_resolved_edge(
                repository_root,
                PYTHON_RESOLVER_VERSION,
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
                    _make_resolved_edge(
                        repository_root,
                        PYTHON_RESOLVER_VERSION,
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
