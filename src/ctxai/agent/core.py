"""
Core agent implementation with tool calling and planning.
"""

from dataclasses import dataclass
from typing import Optional
from pathlib import Path
import uuid

from rich.console import Console

from .llm.base import BaseLLMProvider, MessageRole, ToolCall
from .tools.registry import ToolRegistry
from .context import ConversationContext
from .config import AgentConfig
from .prompts import get_system_prompt, get_tool_error_recovery_prompt


@dataclass
class AgentLoopConfig:
    """Configuration for agent execution."""
    llm_provider: BaseLLMProvider
    tool_registry: ToolRegistry
    agent_config: AgentConfig
    working_directory: Path
    available_indexes: list[str]
    planning_enabled: bool = True
    require_user_approval: bool = True
    max_iterations: int = 10
    verbose: bool = False


class Agent:
    """
    Autonomous coding agent with tool use and planning capabilities.
    """

    def __init__(self, config: AgentLoopConfig):
        """
        Initialize agent.

        Args:
            config: Agent loop configuration
        """
        self.config = config
        self.llm = config.llm_provider
        self.tools = config.tool_registry
        self.context = ConversationContext()
        self.console = Console(legacy_windows=False)

        # Initialize system message
        tool_descriptions = self.tools.get_tool_descriptions()
        system_prompt = get_system_prompt(
            working_directory=config.working_directory,
            available_indexes=config.available_indexes,
            tool_descriptions=tool_descriptions
        )
        self.context.add_system_message(system_prompt)

    async def process_message(self, user_message: str) -> str:
        """
        Process a user message through the agent loop.

        Args:
            user_message: User's input message

        Returns:
            Agent's response
        """
        # Add user message to context
        self.context.add_user_message(user_message)

        if self.config.verbose:
            self.console.print(f"[dim]Processing: {user_message}[/dim]")

        # Agent loop with tool calling
        iteration = 0
        while iteration < self.config.max_iterations:
            if self.config.verbose:
                self.console.print(f"[dim]Iteration {iteration + 1}/{self.config.max_iterations}[/dim]")

            # Get messages for LLM
            messages = self.context.get_messages_for_llm()

            # Get tool schemas
            tool_format = self._get_tool_format()
            tools = self.tools.get_all_schemas(format=tool_format)

            # Call LLM
            try:
                response = self.llm.chat(messages, tools=tools)

                if self.config.verbose:
                    self.console.print(f"[dim]Response: {response.content[:100]}...[/dim]")
                    self.console.print(f"[dim]Tool calls: {len(response.tool_calls)}[/dim]")

                # Check if LLM wants to use tools
                if response.has_tool_calls:
                    # Add assistant message with tool calls
                    self.context.add_assistant_message(
                        response.content,
                        tool_calls=response.tool_calls
                    )

                    # Execute tools
                    tool_results = await self._execute_tools(response.tool_calls)

                    # Add tool results to context
                    for tool_call, result in zip(response.tool_calls, tool_results):
                        result_text = self._format_tool_result(result)
                        self.context.add_tool_result(
                            tool_call_id=tool_call.id,
                            tool_name=tool_call.name,
                            result=result_text
                        )

                    iteration += 1
                    continue

                else:
                    # No tool calls - this is the final response
                    self.context.add_assistant_message(response.content)

                    # Truncate context if needed
                    self.context.truncate_old_messages()

                    return response.content

            except Exception as e:
                error_msg = f"Error during agent loop: {str(e)}"
                if self.config.verbose:
                    self.console.print(f"[red]{error_msg}[/red]")

                # Try to recover
                recovery_prompt = get_tool_error_recovery_prompt(
                    tool_name="LLM",
                    error=str(e),
                    original_goal=user_message
                )
                self.context.add_user_message(recovery_prompt)
                iteration += 1
                continue

        # Max iterations reached
        return f"⚠️ Max iterations ({self.config.max_iterations}) reached. The task may be too complex or an error occurred. Please try breaking it down into smaller steps."

    async def _execute_tools(self, tool_calls: list[ToolCall]) -> list[dict]:
        """
        Execute tool calls.

        Args:
            tool_calls: List of tool calls from LLM

        Returns:
            List of execution results
        """
        results = []

        for tool_call in tool_calls:
            if self.config.verbose:
                self.console.print(f"[cyan]Executing: {tool_call.name}[/cyan]")
                self.console.print(f"[dim]Parameters: {tool_call.parameters}[/dim]")

            try:
                result = await self.tools.execute_tool(
                    tool_call.name,
                    **tool_call.parameters
                )
                results.append(result)

                if self.config.verbose:
                    if result.get("success"):
                        self.console.print(f"[green][OK] {tool_call.name} succeeded[/green]")
                    else:
                        self.console.print(f"[yellow]⚠ {tool_call.name} failed: {result.get('error')}[/yellow]")

            except Exception as e:
                error_result = {
                    "success": False,
                    "result": None,
                    "error": f"Tool execution exception: {str(e)}"
                }
                results.append(error_result)

                if self.config.verbose:
                    self.console.print(f"[red][X] {tool_call.name} exception: {str(e)}[/red]")

        return results

    def _format_tool_result(self, result: dict) -> str:
        """
        Format tool result for LLM.

        Args:
            result: Tool execution result

        Returns:
            Formatted result string
        """
        if result.get("success"):
            output = f"Tool executed successfully.\n\n"
            if result.get("result"):
                output += str(result["result"])
            if result.get("metadata"):
                output += f"\n\nMetadata: {result['metadata']}"
            return output
        else:
            return f"Tool execution failed.\n\nError: {result.get('error', 'Unknown error')}"

    def _get_tool_format(self) -> str:
        """
        Get tool schema format based on LLM provider.

        Returns:
            Format string: "anthropic", "openai", or "generic"
        """
        provider_name = self.llm.__class__.__name__.lower()

        if "anthropic" in provider_name:
            return "anthropic"
        elif "openai" in provider_name:
            return "openai"
        else:
            return "anthropic"  # Default to Anthropic format

    def clear_conversation(self) -> None:
        """Clear conversation history (except system message)."""
        self.context.clear()

    def get_conversation_summary(self) -> str:
        """
        Get summary of conversation.

        Returns:
            Summary string
        """
        return f"Messages: {self.context.get_message_count()}, Tokens: ~{self.context.get_token_count_estimate()}"

    def __repr__(self) -> str:
        return f"Agent(llm={self.llm.__class__.__name__}, tools={len(self.tools)})"
