"""Unit tests for the IG-01 graph data model: records, identity, evidence."""

from __future__ import annotations

import pytest

from ctxai.graph.model import (
    CONFIDENCE_EXACT,
    CONFIDENCE_PROBABLE,
    CONFIDENCE_UNRESOLVED,
    EDGE_KINDS,
    NODE_KINDS,
    GraphEdge,
    GraphMetadata,
    GraphNode,
    derive_edge_id,
    derive_node_id,
)

REPO = "/repos/demo"
NODE_ID = "a" * 64
EDGE_ID = "b" * 64
PARENT_ID = "c" * 64


def make_node() -> GraphNode:
    return GraphNode(
        id=NODE_ID,
        kind="function",
        qualified_name="pkg.mod.run",
        display_name="run",
        language="python",
        file_path="pkg/mod.py",
        start_line=10,
        end_line=20,
        parent_id=PARENT_ID,
        visibility="public",
        source_hash="d" * 64,
    )


def make_edge() -> GraphEdge:
    return GraphEdge(
        id=EDGE_ID,
        kind="calls",
        source_id=NODE_ID,
        target_id=PARENT_ID,
        target_text=None,
        evidence_file="pkg/mod.py",
        evidence_line=12,
        confidence=CONFIDENCE_EXACT,
        resolver_version="python/1",
    )


def make_metadata() -> GraphMetadata:
    return GraphMetadata(
        schema_version=1,
        extractor_version="python/1",
        resolver_version="python/1",
        supported_languages=["python"],
        built_at="2026-09-05T00:00:00+00:00",
        generation=3,
        node_counts={"module": 2, "function": 5},
        edge_counts={"contains": 5, "calls": 3},
        unresolved_edges=1,
    )


class TestGraphNode:
    def test_to_dict_from_dict_round_trip(self):
        node = make_node()
        restored = GraphNode.from_dict(node.to_dict())
        assert restored == node
        assert node.to_dict() == GraphNode.from_dict(node.to_dict()).to_dict()

    def test_evidence_is_repository_relative_file_range(self):
        node = make_node()
        assert node.evidence() == "pkg/mod.py:10-20"

    def test_optional_fields_default(self):
        node = GraphNode(
            id=NODE_ID,
            kind="module",
            qualified_name="pkg",
            display_name="pkg",
            language="python",
            file_path="pkg/__init__.py",
            start_line=1,
            end_line=1,
        )
        assert node.parent_id is None
        assert node.visibility == "public"
        assert node.source_hash is None

    def test_unknown_kind_is_rejected(self):
        payload = make_node().to_dict()
        payload["kind"] = "galaxy"
        with pytest.raises(ValueError, match="kind"):
            GraphNode.from_dict(payload)


class TestGraphEdge:
    def test_to_dict_from_dict_round_trip(self):
        edge = make_edge()
        restored = GraphEdge.from_dict(edge.to_dict())
        assert restored == edge

    def test_unresolved_edge_carries_target_text(self):
        edge = GraphEdge(
            id=EDGE_ID,
            kind="calls",
            source_id=NODE_ID,
            target_id=None,
            target_text="dynamic_fn",
            evidence_file="dyn.py",
            evidence_line=4,
            confidence=CONFIDENCE_UNRESOLVED,
            resolver_version="python/1",
        )
        restored = GraphEdge.from_dict(edge.to_dict())
        assert restored == edge
        assert restored.target_id is None
        assert restored.target_text == "dynamic_fn"
        assert restored.confidence == CONFIDENCE_UNRESOLVED

    def test_evidence_points_at_call_site(self):
        edge = make_edge()
        assert edge.evidence() == "pkg/mod.py:12"

    def test_unknown_confidence_is_rejected(self):
        payload = make_edge().to_dict()
        payload["confidence"] = "guessed"
        with pytest.raises(ValueError, match="confidence"):
            GraphEdge.from_dict(payload)

    def test_unknown_edge_kind_is_rejected(self):
        payload = make_edge().to_dict()
        payload["kind"] = "teleports"
        with pytest.raises(ValueError, match="kind"):
            GraphEdge.from_dict(payload)


class TestGraphMetadata:
    def test_to_dict_from_dict_round_trip(self):
        metadata = make_metadata()
        restored = GraphMetadata.from_dict(metadata.to_dict())
        assert restored == metadata

    def test_totals_sum_kind_counts(self):
        metadata = make_metadata()
        assert metadata.total_nodes == 7
        assert metadata.total_edges == 8


class TestStableIdentity:
    def test_node_id_is_deterministic(self):
        first = derive_node_id(REPO, "pkg/mod.py", "function", "pkg.mod.run")
        second = derive_node_id(REPO, "pkg/mod.py", "function", "pkg.mod.run")
        assert first == second
        assert len(first) == 64

    def test_node_id_varies_with_source_identity(self):
        base = derive_node_id(REPO, "pkg/mod.py", "function", "pkg.mod.run")
        assert base != derive_node_id(REPO, "pkg/other.py", "function", "pkg.mod.run")
        assert base != derive_node_id(REPO, "pkg/mod.py", "method", "pkg.mod.run")
        assert base != derive_node_id(REPO, "pkg/mod.py", "function", "pkg.mod.other")
        assert base != derive_node_id("/repos/other", "pkg/mod.py", "function", "pkg.mod.run")

    def test_edge_id_is_deterministic_and_distinguishes_targets(self):
        source = "a" * 64
        target = "b" * 64
        resolved = derive_edge_id(REPO, "calls", source, target, None, "m.py", 7)
        unresolved = derive_edge_id(REPO, "calls", source, None, "dynamic_fn", "m.py", 7)
        assert resolved == derive_edge_id(REPO, "calls", source, target, None, "m.py", 7)
        assert resolved != unresolved
        assert resolved != derive_edge_id(REPO, "calls", source, target, None, "m.py", 8)
        assert resolved != derive_edge_id(REPO, "references", source, target, None, "m.py", 7)


class TestContractConstants:
    def test_node_and_edge_kinds_match_part_ii_contract(self):
        assert set(NODE_KINDS) == {"module", "class", "function", "method", "interface", "test"}
        assert set(EDGE_KINDS) == {"contains", "imports", "calls", "inherits", "references", "tests"}

    def test_confidence_ladder(self):
        from ctxai.graph.model import CONFIDENCES

        assert set(CONFIDENCES) == {CONFIDENCE_EXACT, CONFIDENCE_PROBABLE, CONFIDENCE_UNRESOLVED}
        assert CONFIDENCE_UNRESOLVED not in (CONFIDENCE_EXACT, CONFIDENCE_PROBABLE)
