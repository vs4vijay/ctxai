"""Inspectable multi-language symbol graph (IG-01/IG-02).

Public surface: the data model (:mod:`ctxai.graph.model`), the language
adapters (:mod:`ctxai.graph.adapters`, registry and capability reporting),
the transactional SQLite store (:mod:`ctxai.graph.store`), the build stage
(:mod:`ctxai.graph.builder`), the shared read service
(:mod:`ctxai.graph.operations`), and the versioned result DTOs
(:mod:`ctxai.graph.dto`) used by the CLI, MCP, and dashboard.
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
from .operations import (
    DIRECTIONS,
    GraphError,
    GraphHealth,
    GraphIndexNotFoundError,
    GraphNotBuiltError,
    GraphOperations,
    GraphStats,
    graph_health,
)
from .store import GraphSchemaError, GraphStore, GraphStoreError, NeighborResult

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
    "GraphIndexNotFoundError",
    "GraphNotBuiltError",
    "GraphMetadata",
    "GraphOperations",
    "GraphStats",
    "GraphSchemaError",
    "GraphStore",
    "GraphStoreError",
    "GraphNode",
    "NeighborResult",
    "graph_health",
]
