"""
Plugin base interface for ctxai_core.

A plugin is any object that implements one or more lifecycle hooks.
Plugins are discovered by the PluginManager from entry points or from
`~/.ctxai/plugins/`.
"""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field
from typing import Any


@dataclass
class PluginMetadata:
    """Metadata advertised by every plugin."""

    name: str
    version: str = "0.0.1"
    description: str = ""
    author: str = ""
    dependencies: list[str] = field(default_factory=list)


@dataclass
class PluginContext:
    """
    Container passed to every hook. Mutable so hooks can attach state.

    Attributes:
        agent: The Agent instance (None outside agent lifecycle).
        data: Arbitrary key/value scratch space shared across hooks.
    """

    agent: Any = None
    data: dict[str, Any] = field(default_factory=dict)


class PluginInterface(ABC):
    """
    Abstract base for plugins.

    Override only the hooks you need; the default implementations are no-ops.
    Tool/provider/planning plugins typically use a subset.
    """

    metadata: PluginMetadata = PluginMetadata(name="unnamed")

    # ----- Lifecycle hooks -----

    def on_register(self, manager: PluginManager) -> None:  # noqa: F821
        """Called once when the plugin is registered."""

    def on_unregister(self, manager: PluginManager) -> None:  # noqa: F821
        """Called when the plugin is unregistered."""

    # ----- Agent hooks -----

    def on_agent_init(self, context: PluginContext) -> None:
        """Called once after an agent is constructed."""

    def on_message_start(self, message: str, context: PluginContext) -> str:
        """
        Called before the agent processes a user message.

        Return the (possibly transformed) message to pass through.
        """
        return message

    def on_message_end(self, response: str, context: PluginContext) -> str:
        """
        Called after the agent has produced a response.

        Return the (possibly transformed) response to deliver to the caller.
        """
        return response

    # ----- Tool hooks -----

    def on_tool_call(
        self, tool_name: str, args: dict[str, Any], context: PluginContext
    ) -> dict[str, Any]:
        """Called before a tool is executed. Return possibly-modified args."""
        return args

    def on_tool_result(
        self, tool_name: str, result: dict[str, Any], context: PluginContext
    ) -> dict[str, Any]:
        """Called after a tool finishes. Return possibly-modified result."""
        return result

    # ----- Provider hooks -----

    def on_llm_request(
        self, messages: list[Any], context: PluginContext
    ) -> list[Any]:
        """Called before sending a request to the LLM."""
        return messages

    def on_llm_response(self, response: Any, context: PluginContext) -> Any:
        """Called after receiving a response from the LLM."""
        return response


class Plugin(PluginInterface):
    """Convenience non-abstract subclass for users who just want to subclass."""
