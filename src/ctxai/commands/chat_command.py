"""
Interactive chat command for ctxai agent.

Provides a REPL interface for conversing with the AI coding agent.
"""

import asyncio
import os
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt

from ..agent.config import AgentConfig, AgentLLMConfig
from ..agent.context import ConversationContext
from ..agent.core import Agent, AgentLoopConfig
from ..agent.llm.anthropic_provider import AnthropicProvider
from ..agent.tools.bash_tool import BashTool
from ..agent.tools.code_search import SemanticSearchTool
from ..agent.tools.file_ops import (
    EditFileTool,
    GlobTool,
    GrepTool,
    ListFilesTool,
    ReadFileTool,
    WriteFileTool,
)
from ..agent.tools.registry import ToolRegistry


console = Console(legacy_windows=False)


def print_banner():
    """Print welcome banner."""
    banner = """
    =========================================================
                 ctxai - AI Coding Agent

      Your autonomous coding assistant powered by AI

      Commands:
        /help    - Show help
        /clear   - Clear conversation
        /exit    - Exit chat
        /save    - Save session

      Just type your request and press Enter!
    =========================================================
    """
    console.print(banner, style="cyan")


def print_help():
    """Print help message."""
    help_text = """
# ctxai Commands

## Chat Commands
- `/help` - Show this help message
- `/clear` - Clear conversation history
- `/exit`, `/quit`, `/bye` - Exit the chat
- `/save` - Save current session
- `/status` - Show agent status
- `/tools` - List available tools

## Example Requests
- "Read the README.md file"
- "List all Python files in the src directory"
- "Search for authentication functions"
- "Create a new file with a hello world function"
- "What is the git status?"
- "Run the tests"

## Tips
- Be specific about what you want
- The agent can use multiple tools to complete tasks
- You can ask follow-up questions
- The agent remembers context within the session
    """
    console.print(Panel(Markdown(help_text), title="Help", border_style="green"))


async def interactive_chat(
    working_directory: Path,
    provider: str = "openrouter",
    model: Optional[str] = None,
    architect_editor: bool = False,
    architect_model: Optional[str] = None,
    editor_model: Optional[str] = None,
    preset: str = "default",
    use_repomap: bool = True,
    verbose: bool = False,
    max_iterations: int = 10,
):
    """
    Run interactive chat mode.

    Args:
        working_directory: Working directory for the agent
        provider: LLM provider (openrouter, ollama, anthropic, openai)
        model: Model name
        architect_editor: Use architect/editor pattern
        architect_model: Model for architect
        editor_model: Model for editor
        preset: Preset for architect/editor (default, budget, local, mixed)
        use_repomap: Use repository mapping for context
        verbose: Enable verbose output
        max_iterations: Max iterations per request
    """
    from ..agent.llm.factory import LLMProviderFactory

    # Check provider availability
    LLMProviderFactory.print_provider_status()

    available, message = LLMProviderFactory.check_provider_availability(provider)
    if not available:
        console.print(f"[red]{message}[/red]")
        console.print(LLMProviderFactory.get_setup_instructions())
        return

    # Print banner
    print_banner()

    # Initialize agent
    console.print("Initializing agent...", style="dim")

    # Create repository map if enabled
    repo_map = None
    if use_repomap:
        try:
            from ..agent.repomap import create_repository_map
            console.print("[cyan]Creating repository map...[/cyan]")
            repo_map = create_repository_map(working_directory, max_tokens=1000)
            console.print("[green]Repository map created[/green]")
        except Exception as e:
            console.print(f"[yellow]⚠ Could not create repository map: {e}[/yellow]")

    # Initialize LLM provider(s)
    if architect_editor:
        # Use architect/editor pattern
        from ..agent.architect_editor import ArchitectEditorAgent, ArchitectEditorConfig

        # Get architect and editor configs
        if architect_model and editor_model:
            # Custom models
            arch_config = AgentLLMConfig(
                provider=provider,
                model=architect_model,
                temperature=0.3,
            )
            edit_config = AgentLLMConfig(
                provider=provider,
                model=editor_model,
                temperature=0.7,
            )
        else:
            # Use preset
            arch_config, edit_config = LLMProviderFactory.get_architect_editor_pair(preset)

        # Create providers
        architect = LLMProviderFactory.create_provider(arch_config)
        editor = LLMProviderFactory.create_provider(edit_config)

        # Create architect/editor agent
        ae_config = ArchitectEditorConfig(
            architect_provider=architect,
            editor_provider=editor,
        )

        console.print(f"[green]Using architect/editor pattern[/green]")
        console.print(f"  Architect: {architect}")
        console.print(f"  Editor: {editor}")

        # Note: For chat, we'll primarily use editor, architect for complex tasks
        llm = editor

    else:
        # Single model
        llm_config = AgentLLMConfig(
            provider=provider,
            model=model,
            temperature=0.7,
            max_tokens=4096,
        )
        llm = LLMProviderFactory.create_provider(llm_config)
        console.print(f"[green]Using: {llm}[/green]")

    # Create agent config
    agent_config = AgentConfig()

    # Register tools
    tools = ToolRegistry(verbose=verbose)
    tools.register(ReadFileTool())
    tools.register(WriteFileTool())
    tools.register(EditFileTool())
    tools.register(ListFilesTool())
    tools.register(GlobTool())
    tools.register(GrepTool())
    tools.register(BashTool(agent_config.tools))
    tools.register(SemanticSearchTool())

    # Get available indexes (if any)
    available_indexes = []  # TODO: Load from config

    # Create agent
    loop_config = AgentLoopConfig(
        llm_provider=llm,
        tool_registry=tools,
        agent_config=agent_config,
        working_directory=working_directory,
        available_indexes=available_indexes,
        max_iterations=max_iterations,
        verbose=verbose,
    )
    agent = Agent(loop_config)

    console.print(f"Agent ready with {len(tools)} tools", style="green")
    console.print(f"Working directory: {working_directory}", style="dim")
    console.print()

    # Chat loop
    while True:
        try:
            # Get user input
            user_input = Prompt.ask("\n[bold cyan]You[/bold cyan]").strip()

            if not user_input:
                continue

            # Handle commands
            if user_input.startswith("/"):
                command = user_input.lower()

                if command in ["/exit", "/quit", "/bye"]:
                    console.print("\nGoodbye!", style="cyan")
                    break

                elif command == "/help":
                    print_help()
                    continue

                elif command == "/clear":
                    agent.clear_conversation()
                    console.print("Conversation cleared", style="green")
                    continue

                elif command == "/status":
                    summary = agent.get_conversation_summary()
                    console.print(f"\nAgent Status:", style="cyan")
                    console.print(f"  {summary}")
                    continue

                elif command == "/tools":
                    console.print("\nAvailable Tools:", style="cyan")
                    for tool_name in tools.list_tools():
                        tool_desc = tools.get_tool_description(tool_name)
                        console.print(f"  • {tool_name}: {tool_desc}")
                    continue

                elif command == "/save":
                    console.print("Session save not yet implemented", style="yellow")
                    continue

                else:
                    console.print(f"Unknown command: {command}", style="red")
                    console.print("Type /help for available commands", style="dim")
                    continue

            # Process message with agent
            console.print("\n[bold green]Agent[/bold green]:", end=" ")

            try:
                with console.status("[dim]Thinking...[/dim]"):
                    response = await agent.process_message(user_input)

                # Print response
                console.print(Markdown(response))

            except Exception as e:
                console.print(f"\n[red]Error: {str(e)}[/red]")
                if verbose:
                    console.print_exception()

        except KeyboardInterrupt:
            console.print("\n\nGoodbye!", style="cyan")
            break

        except EOFError:
            console.print("\n\nGoodbye!", style="cyan")
            break


def start_chat(
    working_directory: Optional[Path] = None,
    provider: str = "openrouter",
    model: Optional[str] = None,
    architect_editor: bool = False,
    architect_model: Optional[str] = None,
    editor_model: Optional[str] = None,
    preset: str = "default",
    use_repomap: bool = True,
    verbose: bool = False,
    max_iterations: int = 10,
):
    """
    Start interactive chat session.

    Args:
        working_directory: Working directory (default: current directory)
        provider: LLM provider (openrouter, ollama, anthropic, openai)
        model: Model name
        architect_editor: Use architect/editor pattern
        architect_model: Model for architect
        editor_model: Model for editor
        preset: Preset for architect/editor
        use_repomap: Use repository mapping
        verbose: Enable verbose output
        max_iterations: Max iterations per request
    """
    if working_directory is None:
        working_directory = Path.cwd()

    # Run async chat loop
    asyncio.run(
        interactive_chat(
            working_directory=working_directory,
            provider=provider,
            model=model,
            architect_editor=architect_editor,
            architect_model=architect_model,
            editor_model=editor_model,
            preset=preset,
            use_repomap=use_repomap,
            verbose=verbose,
            max_iterations=max_iterations,
        )
    )
