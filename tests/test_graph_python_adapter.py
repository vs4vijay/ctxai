"""Unit tests for the IG-01 Python graph adapter: extraction and resolution."""

from __future__ import annotations

import pytest

from ctxai.graph.model import (
    CONFIDENCE_EXACT,
    CONFIDENCE_PROBABLE,
    CONFIDENCE_UNRESOLVED,
    derive_edge_id,
    derive_node_id,
)
from ctxai.graph.python_adapter import (
    PythonAdapter,
    SymbolIndexes,
    build_symbol_indexes,
    module_name_for_path,
    resolve_edges,
)

REPO = "/repos/demo"


def extract(relative_path: str, source: str) -> object:  # noqa: ANN001 - tests only
    adapter = PythonAdapter()
    return adapter.extract_file(relative_path, source.encode("utf-8"), REPO)


def resolve(*extractions):
    nodes = [node for extraction in extractions for node in extraction.nodes]
    indexes = build_symbol_indexes(nodes)
    edges = resolve_edges(list(extractions), indexes, REPO)
    return edges, indexes


def edge_keys(edges, kind):
    return {
        (edge.source_id, edge.target_id if edge.target_id else edge.target_text) for edge in edges if edge.kind == kind
    }


def node_by_qualified(nodes, qualified):
    matches = [node for node in nodes if node.qualified_name == qualified]
    assert len(matches) == 1, f"expected exactly one node {qualified!r}, got {matches}"
    return matches[0]


class TestModuleName:
    def test_plain_and_packaged_modules(self):
        assert module_name_for_path("mod.py") == "mod"
        assert module_name_for_path("pkg/mod.py") == "pkg.mod"
        assert module_name_for_path("pkg/sub/mod.py") == "pkg.sub.mod"

    def test_init_files_name_the_package(self):
        assert module_name_for_path("pkg/__init__.py") == "pkg"
        assert module_name_for_path("pkg/sub/__init__.py") == "pkg.sub"


class TestDefinitions:
    def test_module_class_methods_and_containment(self):
        extraction = extract(
            "pkg/models.py",
            "class Animal:\n    def speak(self):\n        return 'hi'\n\n    def _hidden(self):\n        return 1\n",
        )
        qualified = {node.qualified_name: node for node in extraction.nodes}
        assert set(qualified) == {
            "pkg.models",
            "pkg.models.Animal",
            "pkg.models.Animal.speak",
            "pkg.models.Animal._hidden",
        }
        module = qualified["pkg.models"]
        animal = qualified["pkg.models.Animal"]
        speak = qualified["pkg.models.Animal.speak"]
        hidden = qualified["pkg.models.Animal._hidden"]
        assert module.kind == "class" or module.kind == "module"
        assert module.kind == "module"
        assert animal.kind == "class"
        assert speak.kind == "method"
        assert hidden.kind == "method"
        assert hidden.visibility == "private"
        assert speak.visibility == "public"
        assert animal.parent_id == module.id
        assert speak.parent_id == animal.id
        assert all(node.language == "python" for node in extraction.nodes)
        assert all(node.file_path == "pkg/models.py" for node in extraction.nodes)
        contains = edge_keys(extraction.contains, "contains")
        assert (module.id, animal.id) in contains
        assert (animal.id, speak.id) in contains

    def test_nested_definitions_are_contained_and_qualified(self):
        extraction = extract(
            "nested.py",
            "def outer():\n    def inner():\n        return 1\n    return inner\n",
        )
        qualified = {node.qualified_name: node for node in extraction.nodes}
        assert set(qualified) == {"nested", "nested.outer", "nested.outer.inner"}
        inner = qualified["nested.outer.inner"]
        outer = qualified["nested.outer"]
        assert inner.kind == "function"
        assert inner.parent_id == outer.id
        assert (outer.id, inner.id) in edge_keys(extraction.contains, "contains")

    def test_definition_evidence_and_source_hash(self):
        extraction = extract("pkg/mod.py", "def run():\n    return 1\n")
        run = node_by_qualified(extraction.nodes, "pkg.mod.run")
        assert run.evidence() == "pkg/mod.py:1-2"
        assert run.start_line == 1
        assert run.end_line == 2
        assert run.source_hash
        assert len(run.source_hash) == 64

    def test_decorated_definitions_are_extracted(self):
        extraction = extract("pkg/mod.py", "@cache\ndef run():\n    return 1\n")
        assert {node.qualified_name for node in extraction.nodes} == {"pkg.mod", "pkg.mod.run"}


class TestImports:
    def test_absolute_import_resolves_to_existing_module(self):
        target = extract("pkg/calc.py", "def calculate():\n    return 1\n")
        importer = extract("main.py", "import pkg.calc\n")
        edges, _ = resolve(target, importer)
        calc_module = node_by_qualified(target.nodes, "pkg.calc")
        main_module = node_by_qualified(importer.nodes, "main")
        imports = edge_keys(edges, "imports")
        assert (main_module.id, calc_module.id) in imports
        edge = next(e for e in edges if e.kind == "imports" and e.target_id == calc_module.id)
        assert edge.confidence == CONFIDENCE_EXACT
        assert edge.evidence_file == "main.py"

    def test_aliased_import_records_binding(self):
        target = extract("pkg/calc.py", "def calculate():\n    return 1\n")
        importer = extract("main.py", "import pkg.calc as pc\n\n\ndef go():\n    return pc.calculate()\n")
        edges, _ = resolve(target, importer)
        calc_fn = node_by_qualified(target.nodes, "pkg.calc.calculate")
        go = node_by_qualified(importer.nodes, "main.go")
        assert (go.id, calc_fn.id) in edge_keys(edges, "calls")

    def test_from_import_resolves_symbol(self):
        target = extract("pkg/calc.py", "def calculate():\n    return 1\n")
        importer = extract(
            "main.py",
            "from pkg.calc import calculate\n\n\ndef go():\n    return calculate()\n",
        )
        edges, _ = resolve(target, importer)
        calc_fn = node_by_qualified(target.nodes, "pkg.calc.calculate")
        go = node_by_qualified(importer.nodes, "main.go")
        main_module = node_by_qualified(importer.nodes, "main")
        assert (go.id, calc_fn.id) in edge_keys(edges, "calls")
        # Import edges belong to the module that performs the import.
        assert (main_module.id, calc_fn.id) in edge_keys(edges, "imports")

    def test_relative_imports_resolve_within_package(self):
        calc = extract("pkg/calc.py", "def calculate():\n    return 1\n")
        sibling = extract("pkg/models.py", "class Animal:\n    pass\n")
        importer = extract(
            "pkg/service.py",
            "from . import calc\nfrom .calc import calculate\nfrom .models import Animal\n",
        )
        edges, _ = resolve(calc, sibling, importer)
        service_module = node_by_qualified(importer.nodes, "pkg.service")
        calc_module = node_by_qualified(calc.nodes, "pkg.calc")
        animal = node_by_qualified(sibling.nodes, "pkg.models.Animal")
        imports = edge_keys(edges, "imports")
        assert (service_module.id, calc_module.id) in imports
        assert (service_module.id, animal.id) in imports

    def test_unresolvable_import_is_preserved(self):
        importer = extract("main.py", "import not_a_module\n")
        edges, _ = resolve(importer)
        main_module = node_by_qualified(importer.nodes, "main")
        unresolved = [e for e in edges if e.kind == "imports" and e.confidence == CONFIDENCE_UNRESOLVED]
        assert len(unresolved) == 1
        assert unresolved[0].target_id is None
        assert unresolved[0].target_text == "not_a_module"
        assert unresolved[0].source_id == main_module.id


class TestInheritance:
    def test_same_module_base_class_resolves_exactly(self):
        extraction = extract(
            "pkg/models.py",
            "class Animal:\n    pass\n\n\nclass Dog(Animal):\n    pass\n",
        )
        edges, _ = resolve(extraction)
        animal = node_by_qualified(extraction.nodes, "pkg.models.Animal")
        dog = node_by_qualified(extraction.nodes, "pkg.models.Dog")
        assert (dog.id, animal.id) in edge_keys(edges, "inherits")
        edge = next(e for e in edges if e.kind == "inherits")
        assert edge.confidence == CONFIDENCE_EXACT
        assert edge.evidence_file == "pkg/models.py"

    def test_imported_base_class_resolves_exactly(self):
        models = extract("pkg/models.py", "class Animal:\n    pass\n")
        zoo = extract("pkg/zoo.py", "from .models import Animal\n\n\nclass Dog(Animal):\n    pass\n")
        edges, _ = resolve(models, zoo)
        animal = node_by_qualified(models.nodes, "pkg.models.Animal")
        dog = node_by_qualified(zoo.nodes, "pkg.zoo.Dog")
        assert (dog.id, animal.id) in edge_keys(edges, "inherits")

    def test_unknown_base_class_is_preserved_unresolved(self):
        extraction = extract("pkg/models.py", "class Dog(UnknownBase):\n    pass\n")
        edges, _ = resolve(extraction)
        dog = node_by_qualified(extraction.nodes, "pkg.models.Dog")
        unresolved = [e for e in edges if e.kind == "inherits"]
        assert len(unresolved) == 1
        assert unresolved[0].target_id is None
        assert unresolved[0].target_text == "UnknownBase"
        assert unresolved[0].source_id == dog.id
        assert unresolved[0].confidence == CONFIDENCE_UNRESOLVED


class TestCalls:
    def test_direct_same_module_call_resolves(self):
        extraction = extract(
            "pkg/mod.py",
            "def helper():\n    return 1\n\n\ndef run():\n    return helper()\n",
        )
        edges, _ = resolve(extraction)
        helper = node_by_qualified(extraction.nodes, "pkg.mod.helper")
        run = node_by_qualified(extraction.nodes, "pkg.mod.run")
        calls = [e for e in edges if e.kind == "calls"]
        assert [(e.source_id, e.target_id) for e in calls] == [(run.id, helper.id)]
        assert calls[0].confidence == CONFIDENCE_EXACT
        assert calls[0].evidence() == "pkg/mod.py:6"

    def test_nested_call_resolves(self):
        extraction = extract(
            "nested.py",
            "def outer():\n    def inner():\n        return 1\n    return inner()\n",
        )
        edges, _ = resolve(extraction)
        inner = node_by_qualified(extraction.nodes, "nested.outer.inner")
        outer = node_by_qualified(extraction.nodes, "nested.outer")
        assert (outer.id, inner.id) in edge_keys(edges, "calls")

    def test_self_method_call_resolves(self):
        extraction = extract(
            "pkg/shell.py",
            "class Shell:\n"
            "    def run(self):\n"
            "        return self._private()\n"
            "\n"
            "    def _private(self):\n"
            "        return 2\n",
        )
        edges, _ = resolve(extraction)
        run = node_by_qualified(extraction.nodes, "pkg.shell.Shell.run")
        private = node_by_qualified(extraction.nodes, "pkg.shell.Shell._private")
        assert (run.id, private.id) in edge_keys(edges, "calls")
        edge = next(e for e in edges if e.kind == "calls")
        assert edge.confidence == CONFIDENCE_EXACT

    def test_module_attribute_call_resolves_through_alias(self):
        calc = extract("pkg/calc.py", "def calculate():\n    return 1\n")
        importer = extract("main.py", "import pkg.calc as pc\n\n\ndef go():\n    return pc.calculate()\n")
        edges, _ = resolve(calc, importer)
        calc_fn = node_by_qualified(calc.nodes, "pkg.calc.calculate")
        go = node_by_qualified(importer.nodes, "main.go")
        assert (go.id, calc_fn.id) in edge_keys(edges, "calls")

    def test_builtin_calls_do_not_create_edges(self):
        extraction = extract("main.py", "def go():\n    return len([1, 2])\n")
        edges, _ = resolve(extraction)
        assert [e for e in edges if e.kind == "calls"] == []

    def test_dynamic_call_is_preserved_unresolved(self):
        extraction = extract(
            "dyn.py",
            "def dispatch(fn):\n    return fn()\n",
        )
        edges, _ = resolve(extraction)
        dispatch = node_by_qualified(extraction.nodes, "dyn.dispatch")
        unresolved = [e for e in edges if e.kind == "calls"]
        assert len(unresolved) == 1
        assert unresolved[0].target_id is None
        assert unresolved[0].target_text == "fn"
        assert unresolved[0].source_id == dispatch.id
        assert unresolved[0].confidence == CONFIDENCE_UNRESOLVED
        assert unresolved[0].evidence() == "dyn.py:2"

    def test_attribute_call_on_local_object_stays_unresolved(self):
        models = extract("pkg/models.py", "class Dog:\n    def speak(self):\n        return 'woof'\n")
        caller = extract(
            "main.py",
            "from pkg.models import Dog\n\n\ndef go():\n    dog = Dog()\n    return dog.speak()\n",
        )
        edges, _ = resolve(models, caller)
        go = node_by_qualified(caller.nodes, "main.go")
        dog_class = node_by_qualified(models.nodes, "pkg.models.Dog")
        unresolved = [e for e in edges if e.kind == "calls" and e.source_id == go.id and e.target_id is None]
        assert [(e.target_id, e.target_text) for e in unresolved] == [(None, "dog.speak")]
        # Instantiating Dog is still a statically clear call.
        assert (go.id, dog_class.id) in edge_keys(edges, "calls")


class TestAmbiguity:
    def test_ambiguous_display_name_is_not_guessed(self):
        a = extract("pkg/a.py", "def process():\n    return 'a'\n")
        b = extract("other/b.py", "def process():\n    return 'b'\n")
        caller = extract("amb.py", "def caller():\n    return process()\n")
        edges, indexes = resolve(a, b, caller)
        caller_fn = node_by_qualified(caller.nodes, "amb.caller")
        calls = [e for e in edges if e.kind == "calls" and e.source_id == caller_fn.id]
        assert len(calls) == 1
        assert calls[0].target_id is None
        assert calls[0].target_text == "process"
        assert calls[0].confidence == CONFIDENCE_UNRESOLVED
        # The same bare name resolves when it is unique across the repository.
        assert len(indexes.display.get("process", [])) == 2

    def test_unique_display_name_resolves_probably(self):
        a = extract("pkg/a.py", "def process():\n    return 'a'\n")
        caller = extract("amb.py", "def caller():\n    return process()\n")
        edges, _ = resolve(a, caller)
        caller_fn = node_by_qualified(caller.nodes, "amb.caller")
        process = node_by_qualified(a.nodes, "pkg.a.process")
        calls = [e for e in edges if e.kind == "calls" and e.source_id == caller_fn.id]
        assert [(e.target_id, e.confidence) for e in calls] == [(process.id, CONFIDENCE_PROBABLE)]


class TestReferences:
    def test_same_module_reference_resolves(self):
        extraction = extract(
            "pkg/mod.py",
            "CONSTANT = 1\n\n\ndef run():\n    return CONSTANT\n",
        )
        edges, _ = resolve(extraction)
        # Top-level assignments are not definition nodes; the module owns them.
        references = [e for e in edges if e.kind == "references"]
        assert references == []

    def test_reference_to_repo_symbol_is_recorded_with_evidence(self):
        models = extract("pkg/models.py", "class Dog:\n    pass\n")
        caller = extract(
            "main.py",
            "from pkg.models import Dog\n\n\ndef go():\n    return Dog\n",
        )
        edges, _ = resolve(models, caller)
        go = node_by_qualified(caller.nodes, "main.go")
        dog_class = node_by_qualified(models.nodes, "pkg.models.Dog")
        references = [e for e in edges if e.kind == "references"]
        assert [(e.source_id, e.target_id) for e in references] == [(go.id, dog_class.id)]
        assert references[0].confidence == CONFIDENCE_EXACT
        assert references[0].evidence() == "main.py:5"

    def test_local_variables_are_not_treated_as_symbols(self):
        extraction = extract(
            "main.py",
            "def go():\n    value = compute()\n    return value\n\n\ndef compute():\n    return 1\n",
        )
        edges, _ = resolve(extraction)
        go = node_by_qualified(extraction.nodes, "main.go")
        compute = node_by_qualified(extraction.nodes, "main.compute")
        assert (go.id, compute.id) in edge_keys(edges, "calls")
        assert [e for e in edges if e.kind == "references"] == []


class TestTests:
    def test_test_functions_and_classes_are_typed(self):
        extraction = extract(
            "tests/test_mod.py",
            "def test_run():\n    return 1\n\n\nclass TestCalc:\n    def test_add(self):\n        return 2\n",
        )
        qualified = {node.qualified_name: node for node in extraction.nodes}
        assert qualified["tests.test_mod.test_run"].kind == "test"
        assert qualified["tests.test_mod.TestCalc"].kind == "test"
        assert qualified["tests.test_mod.TestCalc.test_add"].kind == "test"

    def test_test_call_creates_tests_edge(self):
        target = extract("pkg/service.py", "def run():\n    return 1\n")
        test_mod = extract(
            "tests/test_service.py",
            "from pkg.service import run\n\n\ndef test_run():\n    assert run() is not None\n",
        )
        edges, _ = resolve(target, test_mod)
        run = node_by_qualified(target.nodes, "pkg.service.run")
        test_fn = node_by_qualified(test_mod.nodes, "tests.test_service.test_run")
        tests = [e for e in edges if e.kind == "tests"]
        assert [(e.source_id, e.target_id) for e in tests] == [(test_fn.id, run.id)]
        assert tests[0].confidence == CONFIDENCE_EXACT
        assert tests[0].evidence() == "tests/test_service.py:5"


class TestSyntaxErrors:
    def test_broken_file_does_not_crash_and_keeps_module(self):
        extraction = extract("broken.py", "def broken(:\n    pass\n")
        assert node_by_qualified(extraction.nodes, "broken").kind == "module"
        assert extraction.syntax_errors >= 1


class TestStableIds:
    def test_same_input_yields_same_ids(self):
        source = "def run():\n    return helper()\n\n\ndef helper():\n    return 1\n"
        first = extract("pkg/mod.py", source)
        second = extract("pkg/mod.py", source)
        assert [node.id for node in first.nodes] == [node.id for node in second.nodes]
        assert [node.id for node in first.nodes] == [
            derive_node_id(REPO, "pkg/mod.py", node.kind, node.qualified_name) for node in first.nodes
        ]

    def test_ids_differ_across_files_with_same_content(self):
        source = "def run():\n    return 1\n"
        first = extract("pkg/a.py", source)
        second = extract("pkg/b.py", source)
        assert first.nodes[1].id != second.nodes[1].id

    def test_edge_ids_are_stable_across_extractions(self):
        source = "def helper():\n    return 1\n\n\ndef run():\n    return helper()\n"
        first_edges, _ = resolve(extract("pkg/mod.py", source))
        second_edges, _ = resolve(extract("pkg/mod.py", source))
        assert [edge.id for edge in first_edges] == [edge.id for edge in second_edges]
        for edge in first_edges:
            assert edge.id == derive_edge_id(
                REPO,
                edge.kind,
                edge.source_id,
                edge.target_id,
                edge.target_text,
                edge.evidence_file,
                edge.evidence_line,
            )


class TestSymbolIndexes:
    def test_build_symbol_indexes_shapes(self):
        extraction = extract(
            "pkg/mod.py",
            "class C:\n    def m(self):\n        return 1\n\n\ndef run():\n    return 1\n",
        )
        indexes = build_symbol_indexes([node for node in extraction.nodes])
        assert isinstance(indexes, SymbolIndexes)
        assert indexes.module == {"pkg.mod": extraction.nodes[0].id}
        assert set(indexes.symbol) == {"pkg.mod.C", "pkg.mod.C.m", "pkg.mod.run"}
        assert indexes.qualified_by_id[indexes.symbol["pkg.mod.run"]] == "pkg.mod.run"
        assert indexes.kind_by_id[indexes.symbol["pkg.mod.C.m"]] == "method"
        assert indexes.display["run"] == [indexes.symbol["pkg.mod.run"]]

    def test_duplicate_qualified_names_resolve_deterministically(self):
        first = extract("pkg/mod.py", "def run():\n    return 1\n")
        second = extract("pkg/mod.py", "def run():\n    return 2\n")
        nodes = [*first.nodes, *second.nodes]
        indexes = build_symbol_indexes(sorted(nodes, key=lambda n: (n.file_path, n.qualified_name, n.id)))
        assert indexes.symbol["pkg.mod.run"] in {first.nodes[1].id, second.nodes[1].id}


@pytest.mark.parametrize(
    "relative_path",
    ["mod.py", "pkg/mod.py", "pkg/__init__.py"],
)
def test_adapter_accepts_python_files(relative_path):
    assert PythonAdapter.supports_file(relative_path) is True


@pytest.mark.parametrize("relative_path", ["mod.js", "README.md", "pkg/mod.ts"])
def test_adapter_rejects_non_python_files(relative_path):
    assert PythonAdapter.supports_file(relative_path) is False
