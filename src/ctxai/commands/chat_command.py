"""
Interactive chat command for ctxai agent.

Provides a REPL interface for conversing with the AI coding agent.
"""

import asyncio
import os
import logging
import sys
from pathlib import Path
from typing import Optional, Deque
from collections import deque

# Force UTF-8 encoding on Windows for Unicode support
if sys.platform == "win32":
    import io
    # Only wrap if not already wrapped
    if not isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# Disable httpx logging to prevent HTTP request logs from appearing in chat
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

# Disable other noisy loggers
logging.getLogger("openai").setLevel(logging.WARNING)


def _check_terminal_compatibility():
    """Check if terminal is compatible with prompt_toolkit and warn if not."""
    if sys.platform == "win32":
        term = os.getenv("TERM", "")
        if term in ("xterm-256color", "xterm", "cygwin", "mintty"):
            return (
                "WARNING: Terminal compatibility issue detected.\n"
                "prompt_toolkit doesn't fully support Git Bash/MinTTY terminals.\n"
                "\n"
                "Solutions:\n"
                "1. Run in cmd.exe: cmd.exe /c \"uv run ctxai chat\"\n"
                "2. Run in PowerShell: powershell -Command \"uv run ctxai chat\"\n"
                "3. Use winpty: winpty uv run ctxai chat\n"
                "4. Or set environment: set TERM=vt100"
            )
    return None


def _enable_cursor_blink():
    """Enable blinking cursor for Windows Terminal."""
    try:
        sys.stdout.write("\x1b[12h")
        sys.stdout.flush()
    except Exception:
        pass


def _disable_cursor_blink():
    """Disable blinking cursor for Windows Terminal."""
    try:
        sys.stdout.write("\x1b[12l")
        sys.stdout.flush()
    except Exception:
        pass


from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from ..agent.config import AgentConfig, AgentLLMConfig
from ..agent.context import ConversationContext
from ..agent.core import Agent, AgentLoopConfig
from ..agent.llm.anthropic_provider import AnthropicProvider
from ..agent.theme import (
    NEON_CYAN,
    NEON_BLUE,
    NEON_GREEN,
    NEON_GOLD,
    NEON_RED,
    NEON_PURPLE,
    NEON_WHITE,
    NEON_DIM,
    BG_DARK,
    BG_MID,
    NeonConsole,
    NeonCursor,
    create_prompt_text,  # type: ignore[attr-defined]
)
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
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.styles import Style
from prompt_toolkit.cursor_shapes import CursorShape
import sys

console = NeonConsole(Console(legacy_windows=False))


# ============================================================================
# PROVIDER & MODEL DATA
# ============================================================================

PROVIDERS_INFO = {
    "openrouter": {
        "name": "OpenRouter",
        "description": "100+ models (Claude, GPT-4o, o1, DeepSeek, etc.)",
        "auth": "ctxai login openrouter",
        "models": [
            ("anthropic/claude-3.5-sonnet", "Best coding assistant"),
            ("anthropic/claude-3-opus", "Most capable, slower"),
            ("openai/gpt-4o", "Strong all-around"),
            ("openai/o1-mini", "Fast reasoning"),
            ("openai/o1", "Best reasoning, slow"),
            ("deepseek/deepseek-r1", "Open source reasoning"),
            ("deepseek/deepseek-chat", "Fast & cheap"),
            ("google/gemini-pro-1.5", "Google's best"),
            ("meta-llama/llama-3-70b-instruct", "Open source"),
        ],
    },
    "anthropic": {
        "name": "Anthropic (Direct)",
        "description": "Claude models directly",
        "auth": "ANTHROPIC_API_KEY env var",
        "models": [
            ("claude-3-5-sonnet-20241022", "Best balance"),
            ("claude-3-opus-20240229", "Most capable"),
            ("claude-3-haiku-20240307", "Fastest, cheapest"),
        ],
    },
    "openai": {
        "name": "OpenAI (Direct)",
        "description": "GPT models directly",
        "auth": "OPENAI_API_KEY env var",
        "models": [
            ("gpt-4o", "Strong all-around"),
            ("gpt-4o-mini", "Fast & cheap"),
            ("gpt-4-turbo", "Previous best"),
            ("o1-mini", "Fast reasoning"),
            ("o1-preview", "Best reasoning"),
        ],
    },
    "ollama": {
        "name": "Ollama (Local)",
        "description": "Free local models",
        "auth": "ollama serve + pull model",
        "models": [
            ("codellama:34b", "Best coding (34B)"),
            ("codellama:13b", "Good coding (13B)"),
            ("codellama:7b", "Fast coding (7B)"),
            ("qwen2.5-coder:14b", "Qwen coding (14B)"),
            ("qwen2.5-coder:7b", "Qwen coding (7B)"),
            ("llama3.1:70b", "General (70B)"),
            ("llama3.1:8b", "Fast general (8B)"),
            ("deepseek-coder-v2:16b", "DeepSeek coding"),
        ],
    },
    "github-copilot": {
        "name": "GitHub Copilot",
        "description": "Via Copilot subscription",
        "auth": "ctxai login github-copilot",
        "models": [
            ("gpt-4", "GPT-4 via Copilot"),
            ("claude-3.5-sonnet", "Claude via Copilot"),
        ],
    },
    "custom": {
        "name": "Custom (Modal/etc)",
        "description": "OpenAI-compatible endpoints",
        "auth": "Configure in .ctxai/config.json",
        "models": [],  # User must specify
    },
}


# ============================================================================
# MESSAGE QUEUE
# ============================================================================

class MessageQueue:
    """Thread-safe message queue for concurrent chat."""
    
    def __init__(self):
        self._queue: Deque[str] = deque()
        self._lock = asyncio.Lock()
        self._not_empty = asyncio.Condition(self._lock)
    
    async def put(self, message: str):
        """Add a message to the queue."""
        async with self._not_empty:
            self._queue.append(message)
            self._not_empty.notify()
    
    async def get(self) -> str:
        """Get the next message from the queue (blocks if empty)."""
        async with self._not_empty:
            while not self._queue:
                await self._not_empty.wait()
            return self._queue.popleft()
    
    def is_empty(self) -> bool:
        """Check if queue is empty (non-blocking)."""
        return len(self._queue) == 0
    
    def clear(self):
        """Clear the queue."""
        self._queue.clear()


# ============================================================================
# CHAT COMMAND COMPLETER
# ============================================================================

class ChatCommandCompleter(Completer):
    """Autocompleter for chat commands."""

    def __init__(self):
        self.commands = {
            "/help": "Show help message",
            "/clear": "Clear conversation history",
            "/model": "Change provider/model",
            "/exit": "Exit the chat",
            "/quit": "Exit the chat",
            "/bye": "Exit the chat",
            "/stats": "Show statistics",
            "/context": "Show context info",
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


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def print_banner(provider: str, model: str, verbose: bool = False):
    """Print welcome banner."""
    console.print()
    
    width = 56
    
    console.print(f"[bold {NEON_CYAN}]╭{'─' * width}╮[/]")
    console.print(f"[bold {NEON_CYAN}]│ ctxai [bold {NEON_PURPLE}]◆[bold {NEON_PURPLE}] [white]{provider}/{model}[/]  [/]")
    console.print(f"[bold {NEON_CYAN}]│[dim] Type /help for commands, /exit to quit[/]")
    console.print(f"[bold {NEON_CYAN}]╰{'─' * width}╯[/]")
    console.print()


def print_help():
    """Print help message with neon panel styling."""
    help_text = """
[bold cyan]ctxai Commands[/bold cyan]

[bold cyan]Chat Commands[/bold cyan]
• [bold #FFD700]/help[/bold #FFD700] - Show this help message
• [bold #FFD700]/clear[/bold #FFD700] - Clear conversation history  
• [bold #FFD700]/model [provider/model][/bold #FFD700] - Change provider & model
• [bold #FFD700]/stats[/bold #FFD700] - Show session statistics
• [bold #FFD700]/context[/bold #FFD700] - Show context info
• [bold #FFD700]/exit[/bold #FFD700], [bold #FFD700]/quit[/bold #FFD700], [bold #FFD700]/bye[/bold #FFD700] - Exit the chat
• [bold #FFD700]/tools[/bold #FFD700] - List available tools

[bold cyan]/model Usage[/bold cyan]
• [bold #FFD700]/model[/bold #FFD700] - Show all providers & models
• [bold #FFD700]/model claude-3.5-sonnet[/bold #FFD700] - Switch model (current provider)
• [bold #FFD700]/model openrouter/claude-3.5-sonnet[/bold #FFD700] - Switch provider + model

[bold cyan]Example Requests[/bold cyan]
• "Read the README.md file"
• "List all Python files in the src directory"
• "Search for authentication functions"
• "Create a new file with a hello world function"
• "What is the git status?"
• "Run the tests"

[bold cyan]Tips[/bold cyan]
• Be specific about what you want
• The agent can use multiple tools to complete tasks
• You can ask follow-up questions
• The agent remembers context within the session
• Type while AI is thinking to queue messages
    """
    console.print_panel(
        Markdown(help_text),
        title=f"[bold {NEON_CYAN}]Help[bold {NEON_CYAN}]",
        border_style="primary",
        title_align="left",
        padding=(1, 2)
    )


def show_providers_and_models(current_provider: str, current_model: str):
    """Show all available providers and their models."""
    from ..agent.llm.factory import LLMProviderFactory
    
    console.print(f"\n[bold {NEON_CYAN}]Available Providers & Models[/bold {NEON_CYAN}]")
    console.print(f"[dim]Current: [bold]{current_provider}/{current_model}[/bold][/dim]\n")
    
    for provider_id, info in PROVIDERS_INFO.items():
        available, status_msg = LLMProviderFactory.check_provider_availability(provider_id)
        status = "[OK]" if available else "[X]"
        status_color = NEON_GREEN if available else NEON_RED
        
        marker = "●" if provider_id == current_provider else "○"
        
        console.print(f"[bold {NEON_CYAN}]{marker} {info['name']}[/bold {NEON_CYAN}] {status}")
        console.print(f"  [dim]{info['description']}[/dim]")
        
        if available:
            console.print(f"  [dim]Auth: {info['auth']}[/dim]")
            
            if info["models"]:
                console.print(f"  [dim]Popular models:[/dim]")
                for model_id, desc in info["models"][:5]:
                    if model_id == current_model and provider_id == current_provider:
                        console.print(f"    [bold {NEON_GOLD}]→[/bold {NEON_GOLD}] [bold {NEON_WHITE}]{model_id}[/bold {NEON_WHITE}] [dim]({desc})[/dim]")
                    else:
                        console.print(f"    [dim]• {model_id}[/dim] [dim]({desc})[/dim]")
        else:
            console.print(f"  [red]{status_msg}[/red]")
        
        console.print()


def change_model(
    input_str: str,
    config_manager,
    current_provider: str,
    current_model: str,
) -> tuple[str, str, bool]:
    """
    Handle model change request.
    
    Args:
        input_str: The /model argument (e.g., "claude-3.5-sonnet" or "openrouter/claude-3.5-sonnet")
        config_manager: ConfigManager instance
        current_provider: Current provider
        current_model: Current model
        
    Returns:
        Tuple of (new_provider, new_model, changed)
    """
    from ..agent.llm.factory import LLMProviderFactory
    
    new_provider = current_provider
    new_model = current_model
    
    # Parse input: "provider/model" or just "model"
    if "/" in input_str:
        parts = input_str.split("/", 1)
        new_provider = parts[0].strip()
        new_model = parts[1].strip()
    else:
        new_model = input_str.strip()
    
    # Check if provider exists
    if new_provider not in PROVIDERS_INFO:
        console.print_error(f"Unknown provider: {new_provider}")
        console.print(f"[dim]Available providers: {', '.join(PROVIDERS_INFO.keys())}[/dim]")
        return current_provider, current_model, False
    
    # Check provider availability
    available, msg = LLMProviderFactory.check_provider_availability(new_provider)
    if not available:
        console.print_error(f"{new_provider} not available: {msg}")
        return current_provider, current_model, False
    
    # Update config
    try:
        # Load current config
        config = config_manager.load()
        
        # Update provider config
        provider_config = config.get_provider_config(new_provider)
        if new_model and new_model != provider_config.model:
            # Set model for this provider
            config.set_provider_model(new_provider, new_model)
        
        # If provider changed, update default_provider
        if new_provider != current_provider:
            config.default_provider = new_provider
        
        # Save config
        config_manager.save(config)
        
        console.print_success(f"Model changed: {new_provider}/{new_model}")
        
        return new_provider, new_model, True
        
    except Exception as e:
        console.print_error(f"Error saving config: {e}")
        return current_provider, current_model, False


def show_stats(agent: Agent):
    """Show session statistics."""
    try:
        summary = agent.get_conversation_summary()
        
        # Get more detailed stats from context
        context = agent.context
        
        console.print(f"\n[bold {NEON_CYAN}]Session Statistics[/bold {NEON_CYAN}]")
        console.print(f"  [dim]{'─' * 40}[/dim]")
        
        # Parse summary
        parts = summary.split(',')
        for part in parts:
            if 'Messages' in part:
                console.print(f"  [bold {NEON_GOLD}]Messages:[/bold {NEON_GOLD}] {part.split(':')[1].strip()}")
            elif '~' in part:
                tokens = part.split('~')[1].replace(')', '').strip()
                console.print(f"  [bold {NEON_GOLD}]Tokens (est.):[/bold {NEON_GOLD}] ~{tokens}")
        
        console.print()
        
    except Exception as e:
        console.print_error(f"Error getting stats: {e}")


def show_context(agent: Agent):
    """Show context information."""
    try:
        context = agent.context
        
        console.print(f"\n[bold {NEON_CYAN}]Context Information[/bold {NEON_CYAN}]")
        console.print(f"  [dim]{'─' * 40}[/dim]")
        
        # Message count
        msg_count = context.get_message_count()
        console.print(f"  [bold {NEON_GOLD}]Total Messages:[/bold {NEON_GOLD}] {msg_count}")
        
        # Token estimate
        tokens = context.get_token_count_estimate()
        console.print(f"  [bold {NEON_GOLD}]Token Estimate:[/bold {NEON_GOLD}] ~{tokens}")
        
        # System prompt length
        if context.messages:
            system_msg = context.messages[0]
            if system_msg.role.value == "system":
                sys_tokens = len(system_msg.content) // 4  # Rough estimate
                console.print(f"  [bold {NEON_GOLD}]System Prompt:[/bold {NEON_GOLD}] ~{sys_tokens} tokens")
        
        # Conversation turns
        turns = max(0, (msg_count - 1) // 2)  # Subtract system message
        console.print(f"  [bold {NEON_GOLD}]Conversation Turns:[/bold {NEON_GOLD}] {turns}")
        
        # Available context space (assuming 100k context)
        context_limit = 100000
        used_pct = min(100, (tokens / context_limit) * 100)
        console.print(f"  [bold {NEON_GOLD}]Context Used:[/bold {NEON_GOLD}] {used_pct:.1f}%")
        
        # Show recent messages
        if len(context.messages) > 1:
            console.print(f"\n[bold {NEON_CYAN}]Recent Messages[/bold {NEON_CYAN}]")
            for msg in context.messages[-5:]:
                role = msg.role.value.upper()
                preview = msg.content[:50] + "..." if len(msg.content) > 50 else msg.content
                color = NEON_GREEN if role == "USER" else NEON_CYAN if role == "ASSISTANT" else NEON_DIM
                console.print(f"  [{color}]{role}:[/] [dim]{preview}[/dim]")
        
        console.print()
        
    except Exception as e:
        console.print_error(f"Error getting context: {e}")


# ============================================================================
# MAIN CHAT LOOP
# ============================================================================

async def interactive_chat(
    working_directory: Path,
    provider: str | None = None,
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
    Run interactive chat mode with concurrent message processing.
    
    Allows typing new messages while AI is processing previous ones.
    Messages are queued and processed in order.

    Args:
        working_directory: Working directory for the agent
        provider: LLM provider (uses config default_provider if None)
        model: Model name (uses provider config default if None)
        architect_editor: Use architect/editor pattern
        architect_model: Model for architect
        editor_model: Model for editor
        preset: Preset for architect/editor (default, budget, local, mixed)
        use_repomap: Use repository mapping for context
        verbose: Enable verbose output
        max_iterations: Max iterations per request
    """
    from ..agent.llm.factory import LLMProviderFactory
    from ..config import ConfigManager

    # Check terminal compatibility
    compat_warning = _check_terminal_compatibility()
    if compat_warning:
        console.print_warning(compat_warning)
        return

    # Load config
    config_manager = ConfigManager(working_directory)
    config = config_manager.load()

    # Determine provider: CLI arg > config default
    if provider is None:
        provider = config.default_provider

    # Get provider-specific config
    provider_config = config.get_provider_config(provider)

    # Determine model: CLI arg > provider config > default
    if model is None:
        model = provider_config.model

    # Check provider availability (silently unless verbose)
    available, message = LLMProviderFactory.check_provider_availability(provider)
    if not available:
        console.print_error(message)
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
                console.print_dim("Creating repository map...")
            repo_map = create_repository_map(working_directory, max_tokens=1000)
            if verbose:
                console.print_success("Repository map created")
        except Exception as e:
            console.print_warning(f"Could not create repository map: {e}")

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
            console.print_success("Architect/Editor pattern")
            console.print_dim(f"  Architect: {architect}")
            console.print_dim(f"  Editor: {editor}")

        # Note: For chat, we'll primarily use editor, architect for complex tasks
        llm = editor
        model_display = f"{architect.model} + {editor.model}"
        provider_display = "architect/editor"

    else:
        # Single model - use provider config values for api_key and base_url
        agent_llm_config = AgentLLMConfig(
            provider=provider,
            model=model,
            api_key=provider_config.api_key,
            base_url=provider_config.base_url,
            temperature=provider_config.temperature,
            max_tokens=provider_config.max_tokens,
        )
        llm = LLMProviderFactory.create_provider(agent_llm_config)
        model_display = llm.model if llm.model else "default"
        provider_display = provider
        if verbose:
            console.print_success(f"Model: {llm}")

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

    # Track current provider/model for /model command
    current_provider = provider
    current_model = model_display

    # Print banner
    print_banner(provider_display, model_display, verbose=verbose)

    if verbose:
        console.print_dim(f"Ready • {len(tools)} tools • {working_directory}\n")

    # Create message queue for concurrent processing
    message_queue = MessageQueue()
    
    # Track if we're currently processing
    processing_event = asyncio.Event()
    processing_event.set()  # Start ready to accept input
    
    # Create prompt session with autocomplete
    _enable_cursor_blink()
    
    # Create prompt_toolkit style for cyan cursor
    neon_style = Style.from_dict({
        "": "fg:ansicyan bold",
    })
    
    session = PromptSession(
        completer=ChatCommandCompleter(),
        style=neon_style,
        cursor=CursorShape.BEAM,
        message="You: ",
    )

    async def process_message(user_input: str):
        """Process a single message through the agent."""
        nonlocal tools, current_provider, current_model, llm, loop_config
        
        try:
            # Print thinking indicator
            console.print()
            console.print(f"[dim]● Thinking...[/dim]")
            
            # Create a Live display for thinking (if streaming is supported)
            from rich.live import Live
            from rich.text import Text
            
            thinking_lines = []
            
            def update_thinking(line: str):
                thinking_lines.append(line)
                return Text.from_markup(f"[dim]{chr(10).join(thinking_lines[-5:])}[/dim]")
            
            # Try streaming response for better UX
            try:
                full_response = ""
                with Live(console=console.console, transient=True, refresh_per_second=4) as live:
                    async for chunk in agent.stream_message(user_input):
                        full_response += chunk
                        live.update(Text.from_markup(f"[dim]{chunk}[/dim]"))
                
                # Clear thinking indicator
                console.print("\r" + " " * 50 + "\r", end="")
                
            except Exception:
                # Fallback to regular processing
                console.print("\r" + " " * 50 + "\r", end="")
                full_response = await agent.process_message(user_input)

            # Print response
            console.print()
            console.print(f"[bold {NEON_CYAN}]---[/]")
            console.print(Markdown(full_response))
            console.print(f"[bold {NEON_CYAN}]---[/]")
            console.print()

        except Exception as e:
            console.print_error(str(e))
            if verbose:
                console.print_exception()
        finally:
            processing_event.set()

    async def process_queue():
        """Process messages from the queue."""
        while True:
            try:
                # Wait for next message
                user_input = await message_queue.get()
                
                # Wait for any previous processing to complete
                await processing_event.wait()
                processing_event.clear()
                
                # Process this message
                await process_message(user_input)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                console.print_error(f"Queue error: {e}")
                processing_event.set()

    # Start queue processor
    queue_task = asyncio.create_task(process_queue())

    try:
        # Main input loop
        while True:
            try:
                # Get user input
                user_input = (await session.prompt_async()).strip()

                if not user_input:
                    continue

                # Handle commands
                if user_input.startswith("/"):
                    command = user_input.lower()

                    if command in ["/exit", "/quit", "/bye"]:
                        console.print(f"\n[bold {NEON_CYAN}]Goodbye![bold {NEON_CYAN}]")
                        break

                    elif command == "/help":
                        print_help()
                        continue

                    elif command == "/clear":
                        agent.clear_conversation()
                        message_queue.clear()
                        console.print_success("Conversation cleared")
                        continue

                    elif command == "/status":
                        summary = agent.get_conversation_summary()
                        console.print(f"\n[bold {NEON_CYAN}]Agent Status:[bold {NEON_CYAN}]")
                        console.print(f"  {summary}")
                        continue

                    elif command == "/stats":
                        show_stats(agent)
                        continue

                    elif command == "/context":
                        show_context(agent)
                        continue

                    elif command == "/tools":
                        console.print(f"\n[bold {NEON_CYAN}]Available Tools:[bold {NEON_CYAN}]")
                        for tool_name in tools.list_tools():
                            console.print(f"  [bold {NEON_GOLD}]•[bold {NEON_GOLD}] [bold {NEON_WHITE}]{tool_name}[bold {NEON_WHITE}]")
                        continue

                    elif command == "/save":
                        console.print_warning("Session save not yet implemented")
                        continue

                    elif command.startswith("/model"):
                        # Parse model name
                        parts = user_input.split(maxsplit=1)
                        
                        if len(parts) == 1:
                            # Show all providers and models
                            show_providers_and_models(current_provider, current_model)
                            continue

                        # Get the model/provider specification
                        model_spec = parts[1].strip()
                        
                        # Change model
                        new_provider, new_model, changed = change_model(
                            model_spec,
                            config_manager,
                            current_provider,
                            current_model,
                        )
                        
                        if changed:
                            current_provider = new_provider
                            current_model = new_model
                            
                            # Create new LLM provider
                            provider_config = config_manager.load().get_provider_config(new_provider)
                            llm_config = AgentLLMConfig(
                                provider=new_provider,
                                model=new_model,
                                api_key=provider_config.api_key,
                                base_url=provider_config.base_url,
                                temperature=provider_config.temperature,
                                max_tokens=provider_config.max_tokens,
                            )
                            llm = LLMProviderFactory.create_provider(llm_config)
                            
                            # Update agent
                            loop_config.llm_provider = llm
                            agent.llm = llm
                            
                            # Update banner display
                            print_banner(new_provider, new_model, verbose=False)
                        
                        continue

                    else:
                        console.print_error(f"Unknown command: {command}")
                        console.print_dim("Type /help for available commands")
                        continue

                # Queue message for concurrent processing
                await message_queue.put(user_input)
                console.print_dim("[dim]Message queued...[/dim]")

            except KeyboardInterrupt:
                _disable_cursor_blink()
                console.print(f"\n[bold {NEON_CYAN}]Bye![bold {NEON_CYAN}]")
                break

            except EOFError:
                _disable_cursor_blink()
                console.print(f"\n[bold {NEON_CYAN}]Bye![bold {NEON_CYAN}]")
                break

    finally:
        # Clean up
        queue_task.cancel()
        try:
            await queue_task
        except asyncio.CancelledError:
            pass


def start_chat(
    working_directory: Path | None = None,
    provider: str | None = None,
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
        provider: LLM provider (uses config default_provider if None)
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
