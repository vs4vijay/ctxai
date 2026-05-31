"""
ctxai_core — reusable agent harness extracted from ctxai.

This package exposes a clean, stable API surface that downstream projects
can depend on without pulling in the ctxai CLI. The implementation
delegates to the canonical modules under `ctxai.*` to avoid code
duplication; future releases may invert this so `ctxai` depends on
`ctxai_core` instead.
"""

from ctxai_core.api import (
    Agent,
    AgentConfig,
    AgentLoopConfig,
    BaseTool,
    ConversationContext,
    LLMResponse,
    Message,
    MessageRole,
    Plan,
    PlanExecutor,
    PlanStep,
    ToolCall,
    ToolParameter,
    ToolParameterType,
    ToolRegistry,
    ToolSchema,
    create_agent,
    create_plan,
    create_provider,
    create_tool,
)
from ctxai_core.plugins import PluginInterface, PluginManager, get_plugin_manager

__version__ = "1.0.0"

__all__ = [
    "Agent",
    "AgentConfig",
    "AgentLoopConfig",
    "BaseTool",
    "ConversationContext",
    "LLMResponse",
    "Message",
    "MessageRole",
    "Plan",
    "PlanExecutor",
    "PlanStep",
    "PluginInterface",
    "PluginManager",
    "ToolCall",
    "ToolParameter",
    "ToolParameterType",
    "ToolRegistry",
    "ToolSchema",
    "create_agent",
    "create_plan",
    "create_provider",
    "create_tool",
    "get_plugin_manager",
    "__version__",
]
