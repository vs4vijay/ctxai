"""Shared adapter contract tests run against every language adapter (IG-02).

Equivalent Python/JavaScript/TypeScript fixtures must expose consistent
node/edge semantics and evidence: the same kinds of definitions, the same
relationship edges with ``file:line`` evidence and confidence on every
non-``contains`` edge, deterministic stable ids, and honest unresolved edges
for dynamic imports. No adapter may fabricate a resolution another keeps
unresolved for the same construct shape.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pytest

from ctxai.graph.adapters import all_adapters, get_adapter, language_for_file
from ctxai.graph.builder import GraphBuilder
from ctxai.graph.model import CONFIDENCE_UNRESOLVED, EDGE_KINDS, GRAPH_NODE_KINDS_BY_LANGUAGE, NODE_KINDS
from ctxai.graph.store import GraphStore

# -- equivalent fixture sources per language ----------------------------------

PYTHON_FILES = {
    "pkg/calc.py": "def calculate(a, b):\n    return a + b\n",
    "pkg/models.py": (
        "class Animal:\n"
        "    def speak(self):\n"
        '        return "..."\n'
        "\n"
        "\n"
        "class Dog(Animal):\n"
        "    def speak(self):\n"
        '        return "woof"\n'
    ),
    "pkg/service.py": (
        "from pkg.calc import calculate\n"
        "from pkg.calc import calculate as calc\n"
        "from pkg.models import Dog\n"
        "\n"
        "\n"
        "def run(a, b):\n"
        "    total = calculate(a, b)\n"
        "    aliased = calc(a, b)\n"
        "    return total + aliased\n"
        "\n"
        "\n"
        "def pick():\n"
        "    return Dog\n"
        "\n"
        "\n"
        "def test_run():\n"
        "    return run(1, 2)\n"
    ),
    "pkg/dynamic.py": ("import importlib\n\n\ndef load(name):\n    return importlib.import_module(name)\n"),
}

JAVASCRIPT_FILES = {
    "pkg/calc.js": "export function calculate(a, b) {\n  return a + b;\n}\n",
    "pkg/models.js": (
        "export class Animal {\n"
        "  speak() {\n"
        '    return "...";\n'
        "  }\n"
        "}\n"
        "\n"
        "export class Dog extends Animal {\n"
        "  speak() {\n"
        '    return "woof";\n'
        "  }\n"
        "}\n"
    ),
    "pkg/service.js": (
        "import { calculate } from './calc';\n"
        "import { calculate as calc } from './calc';\n"
        "import { Dog } from './models';\n"
        "\n"
        "export function run(a, b) {\n"
        "  const total = calculate(a, b);\n"
        "  const aliased = calc(a, b);\n"
        "  return total + aliased;\n"
        "}\n"
        "\n"
        "export function pick() {\n"
        "  return Dog;\n"
        "}\n"
        "\n"
        "export function test_run() {\n"
        "  return run(1, 2);\n"
        "}\n"
    ),
    "pkg/legacy.cjs": (
        "const { calculate } = require('./calc');\n"
        "\n"
        "module.exports.legacy = function legacy(a, b) {\n"
        "  return calculate(a, b);\n"
        "};\n"
        "const mystery = require(notALiteral);\n"
    ),
    "pkg/dynamic.js": "export function load(path) {\n  return import(path);\n}\n",
}

TYPESCRIPT_FILES = {
    "pkg/calc.ts": (
        "export function calculate(a: number, b: number): number {\n  return a + b;\n}\n"
        "\n"
        "export function format(value: string): string;\n"
        "export function format(value: number): string;\n"
        "export function format(value: unknown): string {\n  return String(value);\n}\n"
    ),
    "pkg/models.ts": (
        "export interface Pet {\n  speak(): string;\n}\n"
        "\n"
        "export class Animal implements Pet {\n"
        "  speak(): string {\n"
        '    return "...";\n'
        "  }\n"
        "}\n"
        "\n"
        "export class Dog extends Animal {\n"
        "  speak(): string {\n"
        '    return "woof";\n'
        "  }\n"
        "}\n"
    ),
    "pkg/service.ts": (
        "import { calculate } from './calc';\n"
        "import { calculate as calc } from './calc';\n"
        "import { Dog } from './models';\n"
        "\n"
        "export function run(a: number, b: number): number {\n"
        "  const total = calculate(a, b);\n"
        "  const aliased = calc(a, b);\n"
        "  return total + aliased;\n"
        "}\n"
        "\n"
        "export function pick(): typeof Dog {\n"
        "  return Dog;\n"
        "}\n"
        "\n"
        "export function test_run(): number {\n"
        "  return run(1, 2);\n"
        "}\n"
    ),
    "pkg/dynamic.ts": "export function load(path: string): Promise<unknown> {\n  return import(path);\n}\n",
}


@dataclass
class LanguageCase:
    """Per-language contract fixture: sources plus language-specific facts.

    Attributes:
        language: Adapter language name.
        files: Repository-relative source files.
        interface_language: Whether the language emits ``interface`` nodes.
        requires_parser: Grammar needed by the adapter.
    """

    language: str
    files: dict[str, str]
    interface_language: bool = False
    notes: tuple[str, ...] = field(default_factory=tuple)


LANGUAGE_CASES = [
    LanguageCase("python", PYTHON_FILES),
    LanguageCase("javascript", JAVASCRIPT_FILES),
    LanguageCase("typescript", TYPESCRIPT_FILES, interface_language=True),
]


def build_contract_repo(root: Path, case: LanguageCase) -> GraphStore:
    """Write the fixture files and build the graph for one language.

    Args:
        root: Temporary repository root.
        case: The language case under test.

    Returns:
        The published graph store.
    """
    for relative_path, source in case.files.items():
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source)
    index_path = root / ".ctxai" / "indexes" / "contract"
    index_path.mkdir(parents=True, exist_ok=True)
    builder = GraphBuilder(repository_root=root)
    builder.update(index_path, set(case.files), set(case.files), set(), force_full=True)
    return GraphStore(index_path)


def ids_by_qualified_name(store: GraphStore) -> dict[str, str]:
    return {node.qualified_name: node.id for node in store.iter_nodes()}


def nodes_by_qualified_name(store: GraphStore) -> dict:
    return {node.qualified_name: node for node in store.iter_nodes()}


def edges_of(store: GraphStore, kind: str) -> list:
    return sorted(
        (
            edge
            for edge in (store.neighbors(node.id, depth=3, limit=500).edges for node in store.iter_nodes())
            for edge in edge
        ),
        key=lambda edge: edge.id,
    )


def all_edges(store: GraphStore) -> list:
    seen: dict[str, object] = {}
    for node in store.iter_nodes():
        result = store.neighbors(node.id, direction="both", depth=1, limit=500)
        for edge in result.edges:
            seen.setdefault(edge.id, edge)
    return sorted(seen.values(), key=lambda edge: edge.id)  # type: ignore[arg-type,return-value]


# -- registry contract --------------------------------------------------------


class TestAdapterRegistryContract:
    def test_every_language_has_a_unique_adapter(self):
        adapters = all_adapters()
        languages = [adapter.language for adapter in adapters]
        assert sorted(languages) == ["javascript", "python", "typescript"]
        extensions = [set(adapter.extensions) for adapter in adapters]
        for first in extensions:
            for other in extensions:
                if first is not other:
                    assert not first & other, "extensions must be disjoint across adapters"

    def test_extension_detection_maps_to_adapters(self):
        assert language_for_file("a.py") == "python"
        assert language_for_file("b.JS".lower()) == "javascript"
        assert language_for_file("src/x.mjs") == "javascript"
        assert language_for_file("src/x.cjs") == "javascript"
        assert language_for_file("src/x.ts") == "typescript"
        assert language_for_file("src/x.tsx") == "typescript"
        assert language_for_file("src/x.mts") == "typescript"
        assert language_for_file("src/x.go") is None

    def test_node_kind_vocabulary_stays_closed(self):
        for language, kinds in GRAPH_NODE_KINDS_BY_LANGUAGE.items():
            assert set(kinds) <= set(NODE_KINDS), language
            assert get_adapter(language) is not None

    def test_edge_kind_vocabulary_is_shared(self):
        assert set(EDGE_KINDS) == {"contains", "imports", "calls", "inherits", "references", "tests"}


# -- shared cross-language contract -------------------------------------------


@pytest.mark.parametrize("case", LANGUAGE_CASES, ids=[case.language for case in LANGUAGE_CASES])
class TestSharedAdapterContract:
    def test_module_function_class_method_nodes_with_evidence(self, tmp_path, case):
        store = build_contract_repo(tmp_path, case)
        nodes = nodes_by_qualified_name(store)
        module = nodes["pkg.service"]
        assert module.kind == "module"
        assert module.file_path == f"pkg/service.{'py' if case.language == 'python' else 'js'}".replace(
            "js", "js"
        ) or module.file_path.startswith("pkg/service.")
        assert module.start_line == 1
        assert module.language == case.language
        run = nodes["pkg.service.run"]
        assert run.kind == "function"
        assert run.evidence().startswith("pkg/service.")
        assert run.evidence() == f"{run.file_path}:{run.start_line}-{run.end_line}"
        animal = nodes["pkg.models.Animal"]
        assert animal.kind == "class"
        speak = nodes["pkg.models.Animal.speak"]
        assert speak.kind == "method"
        dog = nodes["pkg.models.Dog"]
        assert dog.kind == "class"

    def test_interface_nodes_only_where_supported(self, tmp_path, case):
        store = build_contract_repo(tmp_path, case)
        qualified = set(ids_by_qualified_name(store))
        if case.interface_language:
            assert "pkg.models.Pet" in qualified
            assert nodes_by_qualified_name(store)["pkg.models.Pet"].kind == "interface"
        else:
            assert "pkg.models.Pet" not in qualified

    def test_contains_edges_from_module_and_class(self, tmp_path, case):
        store = build_contract_repo(tmp_path, case)
        nodes = nodes_by_qualified_name(store)
        module = nodes["pkg.service"]
        run = nodes["pkg.service.run"]
        contained = store.neighbors(module.id, edge_kind="contains", direction="out", depth=1)
        assert run.id in {node.id for node in contained.nodes}
        animal = nodes["pkg.models.Animal"]
        speak = nodes["pkg.models.Animal.speak"]
        owned = store.neighbors(animal.id, edge_kind="contains", direction="out", depth=1)
        assert speak.id in {node.id for node in owned.nodes}

    def test_inheritance_resolves_between_definitions(self, tmp_path, case):
        store = build_contract_repo(tmp_path, case)
        nodes = nodes_by_qualified_name(store)
        dog = nodes["pkg.models.Dog"]
        animal = nodes["pkg.models.Animal"]
        bases = store.neighbors(dog.id, edge_kind="inherits", direction="out", depth=1)
        assert animal.id in {node.id for node in bases.nodes}
        assert all(edge.confidence == "exact" for edge in bases.edges)

    def test_imports_and_calls_resolve_exactly(self, tmp_path, case):
        store = build_contract_repo(tmp_path, case)
        nodes = nodes_by_qualified_name(store)
        service_module = nodes["pkg.service"]
        calculate = nodes["pkg.calc.calculate"]
        imported = store.neighbors(service_module.id, edge_kind="imports", direction="out", depth=1)
        assert calculate.id in {node.id for node in imported.nodes}
        assert any(edge.confidence == "exact" for edge in imported.edges)
        run = nodes["pkg.service.run"]
        calls = store.neighbors(run.id, edge_kind="calls", direction="out", depth=1)
        assert calculate.id in {node.id for node in calls.nodes}
        assert all(edge.confidence == "exact" for edge in calls.edges if edge.target_id == calculate.id)

    def test_aliased_import_resolves_to_same_target(self, tmp_path, case):
        store = build_contract_repo(tmp_path, case)
        nodes = nodes_by_qualified_name(store)
        run = nodes["pkg.service.run"]
        calculate = nodes["pkg.calc.calculate"]
        calls = store.neighbors(run.id, edge_kind="calls", direction="out", depth=1)
        targets = {edge.target_id for edge in calls.edges}
        assert calculate.id in targets

    def test_references_are_recorded_with_evidence(self, tmp_path, case):
        store = build_contract_repo(tmp_path, case)
        nodes = nodes_by_qualified_name(store)
        pick = nodes["pkg.service.pick"]
        dog = nodes["pkg.models.Dog"]
        refs = store.neighbors(pick.id, edge_kind="references", direction="out", depth=1)
        assert dog.id in {node.id for node in refs.nodes}
        assert all(edge.evidence_file.startswith("pkg/service.") for edge in refs.edges)

    def test_test_nodes_carry_tests_edges(self, tmp_path, case):
        store = build_contract_repo(tmp_path, case)
        nodes = nodes_by_qualified_name(store)
        test_run = nodes["pkg.service.test_run"]
        assert test_run.kind == "test"
        run = nodes["pkg.service.run"]
        tested = store.neighbors(test_run.id, edge_kind="tests", direction="out", depth=1)
        assert run.id in {node.id for node in tested.nodes}
        assert all(edge.confidence == "exact" for edge in tested.edges)

    def test_dynamic_imports_stay_unresolved(self, tmp_path, case):
        store = build_contract_repo(tmp_path, case)
        nodes = nodes_by_qualified_name(store)
        loader = nodes["pkg.dynamic.load"]
        module = nodes["pkg.dynamic"]
        edges = store.neighbors(loader.id, direction="out", depth=1).edges
        edges += store.neighbors(module.id, direction="out", depth=1).edges
        unresolved = [edge for edge in edges if edge.confidence == CONFIDENCE_UNRESOLVED]
        assert unresolved, "the dynamic import must be preserved as an unresolved edge"
        assert all(edge.target_id is None and edge.target_text for edge in unresolved)

    def test_every_non_contains_edge_has_evidence_and_confidence(self, tmp_path, case):
        store = build_contract_repo(tmp_path, case)
        for edge in all_edges(store):
            if edge.kind == "contains":
                continue
            assert edge.evidence_file and edge.evidence_file in case.files, edge.id
            assert edge.evidence_line >= 1
            assert edge.confidence in ("exact", "probable", "unresolved")
            assert edge.resolver_version.startswith(case.language)

    def test_unresolved_edges_never_fabricate_targets(self, tmp_path, case):
        store = build_contract_repo(tmp_path, case)
        for edge in all_edges(store):
            if edge.confidence == CONFIDENCE_UNRESOLVED:
                assert edge.target_id is None
                assert edge.target_text
            else:
                assert edge.target_id is not None

    def test_ids_are_stable_across_rebuilds_and_repo_roots(self, tmp_path, case):
        # Rebuilding the identical source in the same repository keeps ids.
        root = tmp_path / "one"
        first = build_contract_repo(root, case)
        before = ids_by_qualified_name(first)
        (root / ".ctxai" / "indexes" / "contract" / "graph.sqlite3").unlink()
        second = build_contract_repo(root, case)
        assert ids_by_qualified_name(second) == before

        # A different repository root changes repository identity, not structure.
        other = build_contract_repo(tmp_path / "two", case)
        assert ids_by_qualified_name(other) != before
        assert set(ids_by_qualified_name(other)) == set(before)


# -- language-specific contract details ----------------------------------------


class TestJavaScriptCommonJsContract:
    def test_destructured_require_resolves_calls(self, tmp_path):
        store = build_contract_repo(tmp_path, LanguageCase("javascript", JAVASCRIPT_FILES))
        nodes = nodes_by_qualified_name(store)
        legacy = nodes["pkg.legacy.legacy"]
        calculate = nodes["pkg.calc.calculate"]
        calls = store.neighbors(legacy.id, edge_kind="calls", direction="out", depth=1)
        assert calculate.id in {node.id for node in calls.nodes}

    def test_dynamic_require_stays_unresolved(self, tmp_path):
        store = build_contract_repo(tmp_path, LanguageCase("javascript", JAVASCRIPT_FILES))
        nodes = nodes_by_qualified_name(store)
        module = nodes["pkg.legacy"]
        imports = store.neighbors(module.id, edge_kind="imports", direction="out", depth=1)
        unresolved = [edge for edge in imports.edges if edge.target_id is None]
        assert unresolved, "the non-literal require must stay unresolved"
        assert all("notALiteral" in (edge.target_text or "") for edge in unresolved)


class TestTypeScriptContract:
    def test_overloads_produce_one_definition(self, tmp_path):
        store = build_contract_repo(tmp_path, LanguageCase("typescript", TYPESCRIPT_FILES))
        nodes = nodes_by_qualified_name(store)
        assert "pkg.calc.format" in nodes
        format_nodes = [name for name in nodes if name.startswith("pkg.calc.format")]
        assert format_nodes == ["pkg.calc.format"]

    def test_implementation_resolves_against_interface(self, tmp_path):
        store = build_contract_repo(tmp_path, LanguageCase("typescript", TYPESCRIPT_FILES))
        nodes = nodes_by_qualified_name(store)
        animal = nodes["pkg.models.Animal"]
        pet = nodes["pkg.models.Pet"]
        bases = store.neighbors(animal.id, edge_kind="inherits", direction="out", depth=1)
        assert pet.id in {node.id for node in bases.nodes}
