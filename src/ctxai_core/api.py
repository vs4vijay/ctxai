"""
Public API surface for ctxai_core.

This module is the only entry point downstream consumers should import
from. It re-exports a curated set of classes and provides factory
functions for building agents, tools, and providers without leaking
implementation details.
"""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from ctxai.agent.config import (
    AgentBehaviorConfig,
    AgentConfig,
    AgentLLMConfig,
    AgentToolsConfig,
)
from ctxai.agent.context import ConversationContext
from ctxai.agent.core import Agent, AgentLoopConfig
from ctxai.agent.llm.base import (
    AuthenticationError,
    BaseLLMProvider,
    ContextLengthError,
    LLMResponse,
    Message,
    MessageRole,
    ProviderError,
    ProviderTimeoutError,
    RateLimitError,
    ToolCall,
)
from ctxai.agent.llm.factory import LLMProviderFactory
from ctxai.agent.planning import (
    Plan,
    PlanExecutor,
    PlanStatus,
    PlanStep,
    StepStatus,
    create_plan,
)
from ctxai.agent.tools.base import (
    BaseTool,
    ToolParameter,
    ToolParameterType,
    ToolSchema,
)
from ctxai.agent.tools.registry import ToolRegistry

LLMProvider = BaseLLMProvider  # alias for naming consistency


def create_provider(
    provider: str,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    **kwargs: Any,
) -> BaseLLMProvider:
    """
    Create an LLM provider by name.

    Examples:
        >>> create_provider("openrouter", model="anthropic/claude-3.5-sonnet")
        >>> create_provider("ollama", model="codellama:13b")
    """
    cfg = AgentLLMConfig(
        provider=provider,
        model=model,
        api_key=api_key,
        base_url=base_url,
        **{k: v for k, v in kwargs.items() if k in AgentLLMConfig.__annotations__},
    )
    return LLMProviderFactory.create_provider(cfg)


def _resolve_tools(tool_specs: list[Any] | None) -> ToolRegistry:
    """Build a ToolRegistry from a list of either tool names or instances."""
    registry = ToolRegistry()
    if not tool_specs:
        return registry

    from ctxai.agent.tools.bash_tool import BashTool
    from ctxai.agent.tools.code_search import SemanticSearchTool
    from ctxai.agent.tools.file_ops import (
        EditFileTool,
        GlobTool,
        GrepTool,
        ListFilesTool,
        ReadFileTool,
        WriteFileTool,
    )
    from ctxai.agent.tools.git_tools import (
        GitAddTool,
        GitBranchTool,
        GitCommitTool,
        GitDiffTool,
        GitLogTool,
        GitStatusTool,
    )

    builtin = {
        "read_file": ReadFileTool,
        "write_file": WriteFileTool,
        "edit_file": EditFileTool,
        "list_files": ListFilesTool,
        "glob": GlobTool,
        "grep": GrepTool,
        "bash": BashTool,
        "git_status": GitStatusTool,
        "git_diff": GitDiffTool,
        "git_log": GitLogTool,
        "semantic_search": SemanticSearchTool,
        # Aliases for the "category" names from the plan
        "file_ops": [ReadFileTool, WriteFileTool, EditFileTool, ListFilesTool, GlobTool, GrepTool],
        "git": [GitStatusTool, GitDiffTool, GitLogTool, GitAddTool, GitCommitTool, GitBranchTool],
        "search": [SemanticSearchTool],
    }

    for spec in tool_specs:
        if isinstance(spec, BaseTool):
            registry.register(spec)
            continue
        if isinstance(spec, str):
            cls_or_list = builtin.get(spec)
            if cls_or_list is None:
                raise ValueError(f"Unknown built-in tool: {spec}")
            if isinstance(cls_or_list, list):
                for cls in cls_or_list:
                    registry.register(cls())
            else:
                registry.register(cls_or_list())
            continue
        # Class reference
        if inspect.isclass(spec) and issubclass(spec, BaseTool):
            registry.register(spec())
            continue
        raise TypeError(f"Cannot register tool spec of type {type(spec)}")

    return registry


def create_agent(
    provider: str = "openrouter",
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    tools: list[Any] | None = None,
    working_directory: Path | str | None = None,
    available_indexes: list[str] | None = None,
    preset: str | None = None,
    max_iterations: int = 10,
    verbose: bool = False,
    **provider_kwargs: Any,
) -> Agent:
    """
    Build a ready-to-use Agent.

    Args:
        provider: LLM provider name (openrouter, anthropic, openai, ollama, ...).
        model: Specific model identifier; provider default used if None.
        api_key: Optional API key override (otherwise pulled from env/keystore).
        base_url: Optional base URL override (used by ollama, custom, nvidia).
        tools: List of built-in tool names, tool classes, or tool instances.
        working_directory: Workspace path; defaults to CWD.
        available_indexes: Optional list of vector index names exposed to the agent.
        preset: Optional name for an architect/editor preset (overrides provider/model).
        max_iterations: Hard cap on agent loop iterations per message.
        verbose: Enable verbose logging.
        **provider_kwargs: Additional kwargs forwarded to provider config.

    Returns:
        Initialized Agent ready to receive `process_message()` calls.
    """
    if preset:
        arch_cfg, edit_cfg = LLMProviderFactory.get_architect_editor_pair(preset)
        llm_cfg = edit_cfg  # editor handles the bulk of interaction
    else:
        llm_cfg = AgentLLMConfig(
            provider=provider,
            model=model,
            api_key=api_key,
            base_url=base_url,
            **{k: v for k, v in provider_kwargs.items() if k in AgentLLMConfig.__annotations__},
        )

    llm_provider = LLMProviderFactory.create_provider(llm_cfg)
    registry = _resolve_tools(tools)

    behavior = AgentBehaviorConfig(max_iterations=max_iterations, verbose=verbose)
    agent_cfg = AgentConfig(llm=llm_cfg, behavior=behavior)

    loop_cfg = AgentLoopConfig(
        llm_provider=llm_provider,
        tool_registry=registry,
        agent_config=agent_cfg,
        working_directory=Path(working_directory) if working_directory else Path.cwd(),
        available_indexes=available_indexes or [],
        max_iterations=max_iterations,
        verbose=verbose,
    )
    return Agent(loop_cfg)


def create_tool(
    name: str,
    description: str,
    parameters: list[ToolParameter] | None = None,
) -> Callable[[Callable[..., Awaitable[Any]]], BaseTool]:
    """
    Decorator that wraps an async function as a BaseTool subclass.

    Example:
        @create_tool(
            name="upper",
            description="Uppercase a string",
            parameters=[
                ToolParameter(
                    name="text",
                    type=ToolParameterType.STRING,
                    description="Input",
                ),
            ],
        )
        async def upper(text: str) -> dict:
            return {"success": True, "result": text.upper()}
    """
    parameters = parameters or []

    def decorator(fn: Callable[..., Awaitable[Any]]) -> BaseTool:
        if not inspect.iscoroutinefunction(fn):
            raise TypeError(f"@create_tool requires an async function, got {fn}")

        class _FunctionTool(BaseTool):
            def __init__(self) -> None:
                super().__init__()
                self.name = name  # override class-derived name

            def get_schema(self) -> ToolSchema:
                return ToolSchema(name=name, description=description, parameters=list(parameters))

            async def execute(self, **kwargs):
                try:
                    result = await fn(**kwargs)
                    if isinstance(result, dict) and "success" in result:
                        return result
                    return {"success": True, "result": result}
                except Exception as exc:
                    return {"success": False, "result": None, "error": str(exc)}

        _FunctionTool.__name__ = f"{name.title().replace('_', '')}Tool"
        return _FunctionTool()

    return decorator


__all__ = [
    "Agent",
    "AgentBehaviorConfig",
    "AgentConfig",
    "AgentLLMConfig",
    "AgentLoopConfig",
    "AgentToolsConfig",
    "AuthenticationError",
    "BaseLLMProvider",
    "BaseTool",
    "ContextLengthError",
    "ConversationContext",
    "LLMProvider",
    "LLMProviderFactory",
    "LLMResponse",
    "Message",
    "MessageRole",
    "Plan",
    "PlanExecutor",
    "PlanStatus",
    "PlanStep",
    "ProviderError",
    "ProviderTimeoutError",
    "RateLimitError",
    "StepStatus",
    "ToolCall",
    "ToolParameter",
    "ToolParameterType",
    "ToolRegistry",
    "ToolSchema",
    "create_agent",
    "create_plan",
    "create_provider",
    "create_tool",
]
