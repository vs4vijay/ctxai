"""
Tool registry for managing and executing tools.
"""

import asyncio
from typing import Any

from rich.console import Console

from .base import BaseTool


class ToolRegistry:
    """
    Central registry for managing agent tools.

    The registry handles tool registration, schema generation, and execution.
    """

    def __init__(self, verbose: bool = False):
        """
        Initialize the tool registry.

        Args:
            verbose: If True, print debug information
        """
        self._tools: dict[str, BaseTool] = {}
        self.verbose = verbose
        self.console = Console(legacy_windows=False) if verbose else None

    def register(self, tool: BaseTool) -> None:
        """
        Register a tool.

        Args:
            tool: Tool instance to register

        Raises:
            ValueError: If tool with same name already registered
        """
        if tool.name in self._tools:
            if self.verbose:
                self.console.print(f"[yellow]Warning: Tool '{tool.name}' already registered, replacing[/yellow]")

        self._tools[tool.name] = tool

        if self.verbose:
            self.console.print(f"[green][OK] Registered tool:[/green] {tool.name}")

    def register_multiple(self, tools: list[BaseTool]) -> None:
        """
        Register multiple tools at once.

        Args:
            tools: List of tool instances
        """
        for tool in tools:
            self.register(tool)

    def unregister(self, tool_name: str) -> bool:
        """
        Unregister a tool.

        Args:
            tool_name: Name of tool to unregister

        Returns:
            True if tool was unregistered, False if not found
        """
        if tool_name in self._tools:
            del self._tools[tool_name]
            if self.verbose:
                self.console.print(f"[yellow]Unregistered tool:[/yellow] {tool_name}")
            return True
        return False

    def get_tool(self, name: str) -> BaseTool | None:
        """
        Get a tool by name.

        Args:
            name: Tool name

        Returns:
            Tool instance or None if not found
        """
        return self._tools.get(name)

    def has_tool(self, name: str) -> bool:
        """
        Check if a tool is registered.

        Args:
            name: Tool name

        Returns:
            True if tool is registered
        """
        return name in self._tools

    def list_tools(self) -> list[str]:
        """
        Get list of registered tool names.

        Returns:
            List of tool names
        """
        return list(self._tools.keys())

    def get_all_schemas(self, format: str = "anthropic") -> list[dict[str, Any]]:
        """
        Get schemas for all registered tools.

        Args:
            format: Schema format - "anthropic", "openai", or "generic"

        Returns:
            List of tool schemas

        Raises:
            ValueError: If format is invalid
        """
        if format not in ["anthropic", "openai", "generic"]:
            raise ValueError(f"Invalid format: {format}. Must be anthropic, openai, or generic")

        schemas = []
        for tool in self._tools.values():
            schema = tool.get_schema()

            if format == "anthropic":
                schemas.append(schema.to_anthropic_format())
            elif format == "openai":
                schemas.append(schema.to_openai_format())
            else:  # generic
                schemas.append(schema.to_dict())

        return schemas

    def get_schema(self, tool_name: str, format: str = "anthropic") -> dict[str, Any] | None:
        """
        Get schema for a specific tool.

        Args:
            tool_name: Name of the tool
            format: Schema format - "anthropic", "openai", or "generic"

        Returns:
            Tool schema or None if tool not found
        """
        tool = self.get_tool(tool_name)
        if not tool:
            return None

        schema = tool.get_schema()

        if format == "anthropic":
            return schema.to_anthropic_format()
        elif format == "openai":
            return schema.to_openai_format()
        else:  # generic
            return schema.to_dict()

    async def execute_tool(self, name: str, **kwargs) -> dict[str, Any]:
        """
        Execute a tool by name.

        Args:
            name: Tool name
            **kwargs: Tool parameters

        Returns:
            Tool execution result dictionary

        Raises:
            ValueError: If tool not found
        """
        tool = self.get_tool(name)
        if not tool:
            return {
                "success": False,
                "result": None,
                "error": f"Tool not found: {name}",
            }

        if self.verbose:
            self.console.print(f"[cyan]Executing tool:[/cyan] {name}")
            self.console.print(f"[dim]Parameters: {kwargs}[/dim]")

        # Execute tool with validation
        result = await tool.safe_execute(**kwargs)

        if self.verbose:
            if result.get("success"):
                self.console.print(f"[green][OK] Tool {name} succeeded[/green]")
            else:
                self.console.print(f"[red][X] Tool {name} failed:[/red] {result.get('error')}")

        return result

    async def execute_multiple(self, tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Execute multiple tool calls concurrently.

        Args:
            tool_calls: List of tool call dictionaries with 'name' and 'parameters'

        Returns:
            List of execution results in same order as input
        """
        # Create tasks for concurrent execution
        tasks = []
        for call in tool_calls:
            tool_name = call.get("name")
            parameters = call.get("parameters", {})
            tasks.append(self.execute_tool(tool_name, **parameters))

        # Execute concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append(
                    {
                        "success": False,
                        "result": None,
                        "error": f"Exception during execution: {str(result)}",
                    }
                )
            else:
                processed_results.append(result)

        return processed_results

    def get_tool_descriptions(self) -> str:
        """
        Get human-readable descriptions of all tools.

        Returns:
            Formatted string with tool descriptions
        """
        if not self._tools:
            return "No tools registered."

        lines = ["Available tools:"]
        for tool_name, tool in self._tools.items():
            schema = tool.get_schema()
            lines.append(f"\n• {tool_name}")
            lines.append(f"  {schema.description}")

            if schema.parameters:
                lines.append("  Parameters:")
                for param in schema.parameters:
                    required = "required" if param.required else "optional"
                    lines.append(f"    - {param.name} ({param.type.value}, {required}): {param.description}")

        return "\n".join(lines)

    def clear(self) -> None:
        """Clear all registered tools."""
        self._tools.clear()
        if self.verbose:
            self.console.print("[yellow]Cleared all tools[/yellow]")

    def __len__(self) -> int:
        """Get number of registered tools."""
        return len(self._tools)

    def __contains__(self, tool_name: str) -> bool:
        """Check if tool is registered."""
        return tool_name in self._tools

    def __repr__(self) -> str:
        """String representation."""
        return f"ToolRegistry(tools={len(self._tools)})"
