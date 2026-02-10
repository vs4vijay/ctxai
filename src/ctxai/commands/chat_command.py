"""
Interactive chat command for ctxai agent.

Provides a REPL interface for conversing with the AI coding agent.
"""

import asyncio
import os
from pathlib import Path
from typing import Optional

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.formatted_text import HTML
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

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


class ChatCommandCompleter(Completer):
    """Autocompleter for chat commands."""

    def __init__(self):
        self.commands = {
            "/help": "Show help message",
            "/clear": "Clear conversation history",
            "/model": "Change the LLM model",
            "/exit": "Exit the chat",
            "/quit": "Exit the chat",
            "/bye": "Exit the chat",
            "/save": "Save current session",
            "/status": "Show agent status",
            "/tools": "List available tools",
        }

    def get_completions(self, document, complete_event):
        """Get completions for the current input."""
        text = document.text_before_cursor

        # Only provide completions if text starts with /
        if text.startswith("/"):
            for cmd, description in self.commands.items():
                if cmd.startswith(text):
                    yield Completion(
                        cmd,
                        start_position=-len(text),
                        display_meta=description,
                    )


def print_banner(model: str, verbose: bool = False):
    """Print welcome banner."""
    if verbose:
        # Verbose mode: show full banner
        console.print("\n[bold cyan]ctxai[/bold cyan] [dim]-[/dim] [cyan]AI Coding Agent[/cyan]")
        console.print(f"[dim]Model:[/dim] {model}")
        console.print("[dim]Commands: /help /clear /model /status /tools /exit[/dim]\n")
    else:
        # Minimal mode: just show we're ready
        console.print(f"[dim]ctxai[/dim] [cyan]*[/cyan] [dim]{model}[/dim]")
        console.print("[dim]Type /help for commands, /exit to quit[/dim]\n")


def print_help():
    """Print help message."""
    help_text = """
# ctxai Commands

## Chat Commands
- `/help` - Show this help message
- `/clear` - Clear conversation history
- `/model [model-name]` - Change the LLM model (shows available models if no name)
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
    model: str | None = None,
    architect_editor: bool = False,
    architect_model: str | None = None,
    editor_model: str | None = None,
    preset: str = "default",
    use_repomap: bool = False,
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

    # Check provider availability (silently unless verbose)
    available, message = LLMProviderFactory.check_provider_availability(provider)
    if not available:
        console.print(f"\n[red]✗[/red] {message}\n")
        console.print(LLMProviderFactory.get_setup_instructions())
        return

    # Show provider status only if verbose
    if verbose:
        LLMProviderFactory.print_provider_status()

    # Create repository map if enabled
    repo_map = None
    if use_repomap:
        try:
            from ..agent.repomap import create_repository_map
            if verbose:
                console.print("[dim]Creating repository map...[/dim]")
            repo_map = create_repository_map(working_directory, max_tokens=1000)
            if verbose:
                console.print("[dim green]✓ Repository map created[/dim green]")
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

        if verbose:
            console.print("[dim green]✓ Architect/Editor pattern[/dim green]")
            console.print(f"[dim]  Architect: {architect}[/dim]")
            console.print(f"[dim]  Editor: {editor}[/dim]")

        # Note: For chat, we'll primarily use editor, architect for complex tasks
        llm = editor
        model_display = f"{architect.model} + {editor.model}"

    else:
        # Single model
        llm_config = AgentLLMConfig(
            provider=provider,
            model=model,
            temperature=0.7,
            max_tokens=4096,
        )
        llm = LLMProviderFactory.create_provider(llm_config)
        model_display = llm.model
        if verbose:
            console.print(f"[dim green]✓ Model: {llm}[/dim green]")

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

    # Print banner
    print_banner(model_display, verbose=verbose)

    if verbose:
        console.print(f"[dim]Ready • {len(tools)} tools • {working_directory}[/dim]\n")

    # Create prompt session with autocomplete
    session = PromptSession(completer=ChatCommandCompleter())

    # Chat loop
    while True:
        try:
            # Get user input with autocomplete
            user_input = (await session.prompt_async(
                HTML("\n<cyan><b>You</b></cyan>: ")
            )).strip()

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
                    console.print("\nAgent Status:", style="cyan")
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

                elif command.startswith("/model"):
                    # Parse model name
                    parts = user_input.split(maxsplit=1)
                    if len(parts) == 1:
                        # Show available models
                        from ..agent.llm.factory import LLMProviderFactory
                        from ..agent.llm.openrouter_provider import OPENROUTER_MODELS

                        console.print("\n[cyan]Available Models:[/cyan]")
                        console.print("\n[yellow]Quick aliases (use with /model <alias>):[/yellow]")
                        for alias, full_name in OPENROUTER_MODELS.items():
                            console.print(f"  • {alias:<20} → {full_name}")

                        console.print("\n[yellow]Or use full model name from OpenRouter:[/yellow]")
                        console.print("  • Visit: https://openrouter.ai/models")
                        console.print(f"\n[dim]Current model: {llm.model}[/dim]")
                        continue

                    # Change model
                    new_model = parts[1].strip()

                    # Check if it's an alias
                    from ..agent.llm.openrouter_provider import OPENROUTER_MODELS
                    if new_model in OPENROUTER_MODELS:
                        new_model = OPENROUTER_MODELS[new_model]

                    try:
                        # Create new LLM provider with the new model
                        llm_config = AgentLLMConfig(
                            provider=provider,
                            model=new_model,
                            temperature=0.7,
                            max_tokens=4096,
                        )
                        llm = LLMProviderFactory.create_provider(llm_config)

                        # Update agent with new provider
                        loop_config.llm_provider = llm

                        console.print(f"[green]✓ Switched to model: {new_model}[/green]")
                    except Exception as e:
                        console.print(f"[red]Error changing model: {str(e)}[/red]")
                    continue

                else:
                    console.print(f"Unknown command: {command}", style="red")
                    console.print("Type /help for available commands", style="dim")
                    continue

            # Process message with agent
            console.print()

            try:
                with console.status("[dim]•••[/dim]", spinner="dots"):
                    response = await agent.process_message(user_input)

                # Print response
                console.print(Markdown(response))

            except Exception as e:
                console.print(f"[red]✗ {str(e)}[/red]")
                if verbose:
                    console.print_exception()

        except KeyboardInterrupt:
            console.print("\n[dim]Bye![/dim]")
            break

        except EOFError:
            console.print("\n[dim]Bye![/dim]")
            break


def start_chat(
    working_directory: Path | None = None,
    provider: str = "openrouter",
    model: str | None = None,
    architect_editor: bool = False,
    architect_model: str | None = None,
    editor_model: str | None = None,
    preset: str = "default",
    use_repomap: bool = False,
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
