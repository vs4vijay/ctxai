"""Inspectable symbol graph for one repository (IG-01).

Public surface: the data model (:mod:`ctxai.graph.model`), the Python
language adapter (:mod:`ctxai.graph.python_adapter`), the transactional
SQLite store (:mod:`ctxai.graph.store`), the build stage
(:mod:`ctxai.graph.builder`), and the shared read service
(:mod:`ctxai.graph.operations`).
"""

from .builder import GraphBuilder, GraphBuildError, GraphBuildResult
from .model import (
    CONFIDENCES,
    EDGE_KINDS,
    GRAPH_FILENAME,
    GRAPH_SCHEMA_VERSION,
    MAX_RESULT_LIMIT,
    MAX_SYMBOL_QUERY_LENGTH,
    MAX_TRAVERSAL_DEPTH,
    NODE_KINDS,
    GraphEdge,
    GraphMetadata,
    GraphNode,
)
from .operations import DIRECTIONS, GraphError, GraphHealth, GraphOperations, GraphStats, graph_health
from .store import GraphStore, GraphStoreError, NeighborResult

__all__ = [
    "CONFIDENCES",
    "DIRECTIONS",
    "EDGE_KINDS",
    "GRAPH_FILENAME",
    "GRAPH_SCHEMA_VERSION",
    "MAX_RESULT_LIMIT",
    "MAX_SYMBOL_QUERY_LENGTH",
    "MAX_TRAVERSAL_DEPTH",
    "NODE_KINDS",
    "GraphBuildError",
    "GraphBuildResult",
    "GraphBuilder",
    "GraphEdge",
    "GraphError",
    "GraphHealth",
    "GraphMetadata",
    "GraphOperations",
    "GraphStats",
    "GraphStore",
    "GraphStoreError",
    "GraphNode",
    "NeighborResult",
    "graph_health",
]
