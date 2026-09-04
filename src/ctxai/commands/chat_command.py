"""
Interactive chat command for ctxai agent.

Provides a REPL interface for conversing with the AI coding agent.
"""

import asyncio
import io
import logging
import os
import sys
from collections import deque
from pathlib import Path

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.cursor_shapes import CursorShape
from prompt_toolkit.styles import Style
from rich.console import Console
from rich.markdown import Markdown
from rich.prompt import Confirm
from rich.syntax import Syntax

from ..agent.approvals import APPROVAL_MEMORY_KEY, ApprovalDecision
from ..agent.checkpoints import CheckpointManager
from ..agent.config import AgentConfig, AgentLLMConfig
from ..agent.core import Agent, AgentLoopConfig, format_compaction_notice
from ..agent.llm.base import ToolCall
from ..agent.resilience import format_retry_notice
from ..agent.sessions import SessionRecord, SessionStore
from ..agent.theme import (
    NEON_CYAN,
    NEON_DIM,
    NEON_GOLD,
    NEON_GREEN,
    NEON_PURPLE,
    NEON_WHITE,
    NeonConsole,  # type: ignore[attr-defined]
)
from ..agent.tools.bash_tool import BashTool
from ..agent.tools.code_search import SemanticSearchTool
from ..agent.tools.execution import ToolExecutionContext
from ..agent.tools.file_ops import (
    EditFileTool,
    GlobTool,
    GrepTool,
    ListFilesTool,
    ReadFileTool,
    WriteFileTool,
)
from ..agent.tools.git_tools import GitDiffTool, GitLogTool, GitStatusTool
from ..agent.tools.registry import ToolRegistry
from ..agent.workflow import validate_plan_mode
from ..repository_context import discover_repository_indexes

# Force UTF-8 encoding on Windows for Unicode support
if sys.platform == "win32" and not isinstance(sys.stdout, io.TextIOWrapper):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# Keep dependency request logs out of the interactive UI.
for logger_name in ("httpx", "httpcore", "openai"):
    logging.getLogger(logger_name).setLevel(logging.WARNING)


def _check_terminal_compatibility():
    """Check if terminal is compatible with prompt_toolkit and warn if not."""
    if sys.platform == "win32":
        term = os.getenv("TERM", "")
        if term in ("xterm-256color", "xterm", "cygwin", "mintty"):
            return (
                "WARNING: Terminal compatibility issue detected.\n"
                "prompt_toolkit doesn't fully support Git Bash/MinTTY terminals.\n\n"
                "Solutions:\n"
                '1. Run in cmd.exe: cmd.exe /c "uv run ctxai chat"\n'
                '2. Run in PowerShell: powershell -Command "uv run ctxai chat"\n'
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
    "nvidia": {
        "name": "NVIDIA NIM",
        "description": "NVIDIA hosted models (OpenAI-compatible)",
        "auth": "NVIDIA_API_KEY env var (optional for model list)",
        "models": [
            ("meta/llama-3.1-405b-instruct", "Llama 3.1 405B"),
            ("nvidia/llama-3.1-nemotron-70b-instruct", "Nemotron 70B"),
            ("mistralai/mistral-large", "Mistral Large"),
            ("01-ai/yi-large", "Yi Large"),
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
        self._queue: deque[str] = deque()
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
            "/provider": "Choose or add a provider (global config)",
            "/model": "Change provider/model",
            "/plan": "Set planning mode: /plan auto|force|off",
            "/exit": "Exit the chat",
            "/quit": "Exit the chat",
            "/bye": "Exit the chat",
            "/stats": "Show statistics",
            "/context": "Show context budget, usage, and compaction state",
            "/status": "Show agent status",
            "/tools": "List available tools",
            "/save": "Save session: /save [name]",
            "/resume": "Resume session: /resume [name]",
            "/export": "Export session: /export <path>",
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
# APPROVAL PROMPT (HH-07)
# ============================================================================


PLAN_MODE_HINTS = {
    "auto": "keyword classification decides when submit_plan is required",
    "force": "every task goes through submit_plan before mutations/verification",
    "off": "never plan; tools remain approval- and policy-gated",
}


def prompt_approval_decision(console: NeonConsole, call: ToolCall) -> ApprovalDecision:
    """Render an approval prompt with the proposed diff and collect a decision.

    The proposed diff is rendered with syntax highlighting when present, and
    the prompt offers ``[y] once / [a] always this session / [n] no``. A
    session grant is recorded by the agent loop into the session approval
    memory — this callback only renders and asks. Non-interactive stdin, EOF,
    and interrupts deny (fail closed).

    Args:
        console: The chat console.
        call: The approval-shaped tool call (``TaskRun._approval_call`` output).

    Returns:
        The human's ApprovalDecision.
    """
    target = call.parameters.get("approval_target") or call.name
    console.print(f"\n[bold {NEON_GOLD}]? Approve {call.name}: {target}?[/bold {NEON_GOLD}]")
    proposed_diff = call.parameters.get("proposed_diff")
    if proposed_diff:
        console.print(Syntax(str(proposed_diff), "diff", theme="ansi_dark", word_wrap=True))
    if sys.stdin is None or not sys.stdin.isatty():
        console.print("[dim]Non-interactive terminal; denying (fail closed).[/dim]")
        return ApprovalDecision.DENY
    try:
        answer = (
            console.console.input("[bold]?[/bold] [cyan][y] once / [a] always this session / [n] no[/cyan] ")
            .strip()
            .lower()
        )
    except (EOFError, KeyboardInterrupt):
        console.print("[dim]Denied (no answer).[/dim]")
        return ApprovalDecision.DENY
    if answer in {"y", "yes"}:
        return ApprovalDecision.APPROVE_ONCE
    if answer in {"a", "always", "session"}:
        return ApprovalDecision.APPROVE_SESSION
    console.print("[dim]Denied.[/dim]")
    return ApprovalDecision.DENY


def print_banner(provider: str, model: str, verbose: bool = False):
    """Print welcome banner."""
    console.print()

    width = 56

    console.print(f"[bold {NEON_CYAN}]╭{'─' * width}╮[/]")
    console.print(
        f"[bold {NEON_CYAN}]│ ctxai [bold {NEON_PURPLE}]◆[bold {NEON_PURPLE}] [white]{provider}/{model}[/]  [/]"
    )
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
• [bold #FFD700]/provider[/bold #FFD700] - Choose or add a provider (saved globally)
• [bold #FFD700]/model [provider/model][/bold #FFD700] - Change provider & model
• [bold #FFD700]/stats[/bold #FFD700] - Show session statistics
• [bold #FFD700]/context[/bold #FFD700] - Show context budget, measured usage, and compaction state
• [bold #FFD700]/plan [auto|force|off][/bold #FFD700] - Show or set planning mode for the next tasks
• [bold #FFD700]/exit[/bold #FFD700], [bold #FFD700]/quit[/bold #FFD700],
  [bold #FFD700]/bye[/bold #FFD700] - Exit the chat
• [bold #FFD700]/tools[/bold #FFD700] - List available tools
• [bold #FFD700]/save [name][/bold #FFD700] - Save a durable, redacted session
• [bold #FFD700]/resume [name][/bold #FFD700] - Resume a saved session
• [bold #FFD700]/export <path>[/bold #FFD700] - Export a redacted Markdown transcript

[bold cyan]/provider Usage[/bold cyan]
• [bold #FFD700]/provider[/bold #FFD700] - Interactive provider menu
• [bold #FFD700]/provider anthropic[/bold #FFD700] - Switch provider (prompts for model)
• [bold #FFD700]/provider add[/bold #FFD700] - Add a custom OpenAI-compatible provider

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
        padding=(1, 2),
    )


def show_providers_and_models(current_provider: str, current_model: str):
    """Show all available providers and their models."""
    from ..agent.llm.factory import LLMProviderFactory

    console.print(f"\n[bold {NEON_CYAN}]Available Providers & Models[/bold {NEON_CYAN}]")
    console.print(f"[dim]Current: [bold]{current_provider}/{current_model}[/bold][/dim]\n")

    for provider_id, info in PROVIDERS_INFO.items():
        available, status_msg = LLMProviderFactory.check_provider_availability(provider_id)
        status = "[OK]" if available else "[X]"
        marker = "●" if provider_id == current_provider else "○"

        console.print(f"[bold {NEON_CYAN}]{marker} {info['name']}[/bold {NEON_CYAN}] {status}")
        console.print(f"  [dim]{info['description']}[/dim]")

        if available:
            console.print(f"  [dim]Auth: {info['auth']}[/dim]")

            if info["models"]:
                console.print("  [dim]Popular models:[/dim]")
                for model_id, desc in info["models"][:5]:
                    if model_id == current_model and provider_id == current_provider:
                        console.print(
                            f"    [bold {NEON_GOLD}]→[/bold {NEON_GOLD}] "
                            f"[bold {NEON_WHITE}]{model_id}[/bold {NEON_WHITE}] [dim]({desc})[/dim]"
                        )
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


def handle_provider_command(
    input_str: str,
    global_config_manager,
    current_provider: str,
    current_model: str,
) -> tuple[str, str, bool]:
    """
    Handle /provider command — interactive selection, quick switch, or add new.

    Writes to the global config (~/.ctxai/config.toml or $CTXAI_HOME).

    Args:
        input_str: /provider argument (empty, provider name, or "add")
        global_config_manager: ConfigManager with use_global=True
        current_provider: Current provider name
        current_model: Current model name

    Returns:
        Tuple of (new_provider, new_model, changed)
    """

    from ..agent.llm.factory import LLMProviderFactory

    ps = PromptSession()

    # ── /provider (no args) — interactive menu ─────────────────────────────
    if not input_str:
        providers = list(PROVIDERS_INFO.keys())
        console.print(
            f"\n[bold {NEON_CYAN}]Providers[/bold {NEON_CYAN}]  [dim](current: [bold]{current_provider}[/bold])[/dim]\n"
        )

        numbered = []
        for pid in providers:
            available, msg = LLMProviderFactory.check_provider_availability(pid)
            status = "[green][ok][/green]" if available else "[red][--][/red]"
            marker = "▸" if pid == current_provider else " "
            info = PROVIDERS_INFO[pid]
            model_hint = info["models"][0][0] if info["models"] else "[dim]custom[/dim]"
            console.print(
                f"  {marker} [bold]{len(numbered) + 1}.[/bold] {info['name']:<24} {status}  [dim]{model_hint}[/dim]"
            )
            numbered.append(pid)

        console.print(f"  {len(numbered) + 1}. [bold]+ Add new provider...[/bold]\n")

        choice = ps.prompt(f"Pick [1-{len(numbered) + 1}]: ").strip()
        if not choice:
            return current_provider, current_model, False

        try:
            idx = int(choice) - 1
        except ValueError:
            console.print_error("Invalid choice")
            return current_provider, current_model, False

        if idx < 0 or idx > len(numbered):
            console.print_error("Invalid choice")
            return current_provider, current_model, False

        if idx == len(numbered):
            return _add_provider_interactive(global_config_manager, current_provider, current_model, ps)

        return handle_provider_command(numbered[idx], global_config_manager, current_provider, current_model)

    # ── /provider add — interactive new provider ───────────────────────────
    if input_str.lower() == "add":
        return _add_provider_interactive(global_config_manager, current_provider, current_model, ps)

    # ── /provider <name> — quick switch ────────────────────────────────────
    provider_name = input_str.strip().lower()

    if provider_name not in PROVIDERS_INFO and provider_name != "custom":
        console.print_error(f"Unknown provider: {provider_name}")
        console.print(f"[dim]Available: {', '.join(PROVIDERS_INFO.keys())} or /provider add[/dim]")
        return current_provider, current_model, False

    available, msg = LLMProviderFactory.check_provider_availability(provider_name)
    if not available:
        console.print_warning(f"{provider_name}: {msg}")
        if not Confirm.ask("Set as default anyway?", default=True):
            return current_provider, current_model, False

    # Ask for model — live discovery API first, static catalog fallback
    from ..agent.llm.model_discovery import discover_models

    discovered = discover_models(provider_name, global_config_manager)
    if discovered:
        console.print(
            f"\n[bold {NEON_CYAN}]Models for {provider_name}[/bold {NEON_CYAN}] "
            f"[dim](live — {len(discovered)} available)[/dim]"
        )
        for i, dm in enumerate(discovered[:25], 1):
            label = dm.name if dm.name and dm.name != dm.id else dm.id
            suffix = f"  [dim]({dm.id})[/dim]" if dm.name and dm.name != dm.id else ""
            ctx = f"  [dim]{dm.context_length:,} ctx[/dim]" if dm.context_length else ""
            console.print(f"  [bold]{i}.[/bold] {label}{suffix}{ctx}")
        if len(discovered) > 25:
            console.print(
                f"  [dim]… {len(discovered) - 25} more — pick any number 1-{len(discovered)} or type a model id[/dim]"
            )

        pick = ps.prompt(
            f"\nPick model [1-{len(discovered)}] or type id (Enter keeps [bold]{current_model}[/bold]): ",
        ).strip()

        if pick:
            try:
                idx = int(pick) - 1
                new_model = discovered[idx].id if 0 <= idx < len(discovered) else pick
            except ValueError:
                new_model = pick
        else:
            # Enter keeps the current model
            new_model = current_model
    else:
        info = PROVIDERS_INFO.get(provider_name, {})
        models = info.get("models", [])

        if models:
            console.print(f"\n[bold {NEON_CYAN}]Models for {provider_name}:[/bold {NEON_CYAN}]")
            for i, (mid, desc) in enumerate(models, 1):
                console.print(f"  [bold]{i}.[/bold] {mid}  [dim]{desc}[/dim]")

            pick = ps.prompt(
                f"\nPick model [1-{len(models)}] or type name (Enter keeps [bold]{current_model}[/bold]): ",
            ).strip()

            if pick:
                try:
                    idx = int(pick) - 1
                    new_model = models[idx][0] if 0 <= idx < len(models) else pick
                except ValueError:
                    new_model = pick
            else:
                # Enter keeps the current model
                new_model = current_model
        else:
            new_model = (
                ps.prompt(
                    f"Model name (Enter keeps [bold]{current_model}[/bold]): ",
                ).strip()
                or current_model
            )

    # Write to global config
    config = global_config_manager.load()
    config.default_provider = provider_name
    config.set_provider_model(provider_name, new_model)
    global_config_manager.save(config)

    console.print_success(f"Global default set: {provider_name}/{new_model}")
    return provider_name, new_model, True


def _add_provider_interactive(
    global_config_manager,
    current_provider: str,
    current_model: str,
    ps=None,
) -> tuple[str, str, bool]:
    """Interactive add a new custom provider and write to global config."""
    if ps is None:
        ps = PromptSession()

    console.print(f"\n[bold {NEON_CYAN}]Add Custom Provider[/bold {NEON_CYAN}]\n")

    name = ps.prompt("  Provider name: ").strip()
    if not name:
        return current_provider, current_model, False

    base_url = ps.prompt("  Base URL (OpenAI-compatible): ").strip()
    if not base_url:
        console.print_error("Base URL is required")
        return current_provider, current_model, False

    api_key = ps.prompt("  API key (or Enter to skip): ").strip() or None
    model = ps.prompt("  Default model: ").strip()
    if not model:
        console.print_error("Model name is required")
        return current_provider, current_model, False

    # Write to global config
    config = global_config_manager.load()
    config.set_provider_config(
        name,
        model=model,
        api_key=api_key,
        base_url=base_url,
        enabled=True,
    )
    config.default_provider = name
    global_config_manager.save(config)

    console.print_success(f"Added {name} ({base_url}) → global config")
    console.print_success(f"Now using: {name}/{model}")
    return name, model, True


def show_stats(agent: Agent):
    """Show session statistics."""
    try:
        summary = agent.get_conversation_summary()

        console.print(f"\n[bold {NEON_CYAN}]Session Statistics[/bold {NEON_CYAN}]")
        console.print(f"  [dim]{'─' * 40}[/dim]")

        # Parse summary
        parts = summary.split(",")
        for part in parts:
            if "Messages" in part:
                console.print(f"  [bold {NEON_GOLD}]Messages:[/bold {NEON_GOLD}] {part.split(':')[1].strip()}")
            elif "~" in part:
                tokens = part.split("~")[1].replace(")", "").strip()
                console.print(f"  [bold {NEON_GOLD}]Tokens (est.):[/bold {NEON_GOLD}] ~{tokens}")

        console.print()

    except Exception as e:
        console.print_error(f"Error getting stats: {e}")


def show_context(agent: Agent):
    """Show context budget, measured token usage, and compaction state (HH-03)."""
    try:
        context = agent.context
        behavior = agent.config.agent_config.behavior
        capabilities = agent.llm.get_capabilities()
        context_size = getattr(capabilities, "context_size", None)

        console.print(f"\n[bold {NEON_CYAN}]Context Information[/bold {NEON_CYAN}]")
        console.print(f"  [dim]{'─' * 40}[/dim]")

        # Message count
        msg_count = context.get_message_count()
        console.print(f"  [bold {NEON_GOLD}]Total Messages:[/bold {NEON_GOLD}] {msg_count}")

        # Measured (or estimated) context tokens
        measured = context.estimate_context_tokens()
        basis = (
            "measured from provider usage"
            if context.last_reported_prompt_tokens is not None
            else "estimated (~4 chars/token)"
        )
        console.print(f"  [bold {NEON_GOLD}]Context Tokens:[/bold {NEON_GOLD}] ~{measured} [dim]({basis})[/dim]")

        # Budget model: provider-declared context_size and the soft limit
        if isinstance(context_size, int) and context_size > 0:
            budget = int(context_size * behavior.context_soft_limit_ratio)
            used_pct = min(100.0, (measured / context_size) * 100)
            console.print(
                f"  [bold {NEON_GOLD}]Context Budget:[/bold {NEON_GOLD}] "
                f"{context_size:,} tokens ({agent.llm.__class__.__name__})"
            )
            console.print(
                f"  [bold {NEON_GOLD}]Soft Limit:[/bold {NEON_GOLD}] "
                f"{budget:,} tokens (ratio {behavior.context_soft_limit_ratio:g}) — compaction triggers above this"
            )
            console.print(f"  [bold {NEON_GOLD}]Context Used:[/bold {NEON_GOLD}] {used_pct:.1f}%")
        else:
            console.print(f"  [bold {NEON_GOLD}]Context Budget:[/bold {NEON_GOLD}] unknown (compaction disabled)")

        # Compaction state
        console.print(f"  [bold {NEON_GOLD}]Compactions:[/bold {NEON_GOLD}] {context.compaction_count}")
        console.print(f"  [bold {NEON_GOLD}]Elided Tool Results:[/bold {NEON_GOLD}] {context.elided_message_count}")

        # Per-run usage ledger (provider-reported, tokens only)
        run = agent.last_run
        if run is not None and run.usage.call_count:
            totals = run.usage.totals()
            console.print(
                f"  [bold {NEON_GOLD}]Last Run Usage:[/bold {NEON_GOLD}] "
                f"{totals['prompt_tokens']:,} prompt + {totals['completion_tokens']:,} completion = "
                f"{totals['total_tokens']:,} tokens over {totals['calls']} call(s)"
            )

        # System prompt size
        if context.messages:
            system_msg = context.messages[0]
            if system_msg.role.value == "system":
                sys_tokens = len(system_msg.content) // 4  # Rough estimate
                console.print(f"  [bold {NEON_GOLD}]System Prompt:[/bold {NEON_GOLD}] ~{sys_tokens} tokens")

        # Conversation turns
        turns = max(0, (msg_count - 1) // 2)  # Subtract system message
        console.print(f"  [bold {NEON_GOLD}]Conversation Turns:[/bold {NEON_GOLD}] {turns}")

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
    plan_mode: str = "auto",
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
        plan_mode: Planning override for tasks (HH-07): auto, force, or off
    """
    from ..agent.llm.factory import LLMProviderFactory
    from ..config import ConfigManager

    # Check terminal compatibility
    compat_warning = _check_terminal_compatibility()
    if compat_warning:
        console.print_warning(compat_warning)
        return

    try:
        plan_mode = validate_plan_mode(plan_mode)
    except ValueError as error:
        console.print_error(str(error))
        return

    # Load config (project layer)
    config_manager = ConfigManager(working_directory)
    global_config_manager = ConfigManager(working_directory, use_global=True)
    config = config_manager.load()

    if architect_editor:
        raise ValueError(
            "Architect/editor mode is disabled until a benchmark demonstrates better "
            "quality, latency, or cost than the validated single-model planning workflow."
        )

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
    if use_repomap:
        try:
            from ..agent.repomap import create_repository_map

            if verbose:
                console.print_dim("Creating repository map...")
            create_repository_map(working_directory, max_tokens=1000)
            if verbose:
                console.print_success("Repository map created")
        except Exception as e:
            console.print_warning(f"Could not create repository map: {e}")

    # Initialize LLM provider(s)
    if architect_editor:
        # Use architect/editor pattern
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
    execution_context = ToolExecutionContext.for_project(
        working_directory,
        allow_outside_project=agent_config.tools.allow_outside_project,
        timeout=agent_config.tools.bash_timeout,
        env_passthrough=agent_config.tools.env_passthrough,
    )
    tools.register(ReadFileTool(context=execution_context, max_output_chars=agent_config.tools.max_output_chars))
    tools.register(WriteFileTool(context=execution_context))
    tools.register(EditFileTool(context=execution_context))
    tools.register(ListFilesTool(context=execution_context))
    tools.register(GlobTool(context=execution_context))
    tools.register(GrepTool(context=execution_context))
    tools.register(BashTool(agent_config.tools, context=execution_context))
    tools.register(GitStatusTool(context=execution_context))
    tools.register(GitDiffTool(context=execution_context))
    tools.register(GitLogTool(context=execution_context))
    tools.register(SemanticSearchTool(project_path=working_directory))

    # Get available indexes (if any)
    available_indexes = discover_repository_indexes(working_directory)

    # Durable session state is repository-scoped and never stores provider credentials.
    session_store = SessionStore(working_directory)
    current_session = "default"

    # Create agent
    loop_config = AgentLoopConfig(
        llm_provider=llm,
        tool_registry=tools,
        agent_config=agent_config,
        working_directory=working_directory,
        available_indexes=available_indexes,
        planning_enabled=agent_config.behavior.planning_enabled,
        require_user_approval=agent_config.behavior.require_user_approval,
        max_iterations=max_iterations,
        verbose=verbose,
        # HH-07 decision ergonomics: the callback returns an ApprovalDecision
        # (once / always-this-session / no); the loop records session grants
        # into the session approval memory and adapts booleans elsewhere.
        approval_callback=(lambda call: prompt_approval_decision(console, call)),
        cancel_event=asyncio.Event(),
        on_retry=(lambda notice: console.print(f"[yellow]{format_retry_notice(notice)}[/yellow]")),
        on_compaction=(lambda notice: console.print(f"[yellow]{format_compaction_notice(notice)}[/yellow]")),
        session_store=session_store,
        session_name=current_session,
        # Local pre-mutation checkpoints (HH-06): one per run under
        # .ctxai/checkpoints/, bounded by the behavior retention/size config.
        checkpoint_manager=CheckpointManager.for_project(
            working_directory,
            retention=agent_config.behavior.checkpoint_retention,
            max_bytes=agent_config.behavior.checkpoint_max_bytes,
        ),
        plan_mode=plan_mode,
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
    neon_style = Style.from_dict(
        {
            "": "fg:ansicyan bold",
        }
    )

    session = PromptSession(
        completer=ChatCommandCompleter(),
        style=neon_style,
        cursor=CursorShape.BEAM,
        message="You: ",
    )

    async def process_message(user_input: str):
        """Process a single message through the agent, rendering live events (HH-05)."""
        nonlocal tools, current_provider, current_model, llm, loop_config

        from rich.live import Live
        from rich.text import Text

        from ..agent.events import AgentEventKind

        def render_stream_event(event, live):
            """Render one AgentEvent inside the streaming Live display.

            Token deltas update the transient preview; tool starts/results and
            approval brackets print as status lines above it; status lines
            (retries, compaction, fallback diagnostics) print in yellow.

            Args:
                event: The AgentEvent to render.
                live: The active Rich Live display.
            """
            if event.kind is AgentEventKind.TOKEN:
                streamed_chunks.append(event.text)
                live.update(Text("".join(streamed_chunks)[-1200:], style="dim"))
            elif event.kind is AgentEventKind.TOOL_CALL_STARTED:
                streamed_chunks.clear()
                live.update(Text("", style="dim"))
                console.print(f"[{NEON_GOLD}]→ {event.text}[/{NEON_GOLD}]")
            elif event.kind is AgentEventKind.TOOL_RESULT:
                if event.data.get("success"):
                    console.print(f"[dim]  ✓ {event.text}[/dim]")
                else:
                    console.print(f"[dim]  ✗ {event.text}: {event.data.get('error') or 'failed'}[/dim]")
            elif event.kind is AgentEventKind.APPROVAL_REQUIRED:
                # The decision prompt itself (HH-07) renders the diff with
                # syntax highlighting; here we print the compact question line.
                target = event.data.get("target")
                if target:
                    console.print(f"[{NEON_GOLD}]? Approval required: {event.data.get('tool')}: {target}[/{NEON_GOLD}]")
                else:
                    console.print(f"[{NEON_GOLD}]? {event.text}[/{NEON_GOLD}]")
            elif event.kind is AgentEventKind.APPROVAL_DECIDED:
                decision = event.data.get("decision")
                scope = f" ({decision})" if decision else ""
                console.print(f"[dim]  {'approved' if event.data.get('approved') else 'denied'}{scope}[/dim]")
            elif event.kind is AgentEventKind.STATUS:
                streamed_chunks.clear()
                live.update(Text("", style="dim"))
                console.print(f"[yellow]{event.text}[/yellow]")

        try:
            # Print thinking indicator
            console.print()

            console.print("[dim]● Thinking...[/dim]")

            # Stream live events (HH-05): token deltas render through Rich
            # Live, tool activity and approval prompts render as inline
            # status lines, and the final report is captured from the stream.
            streamed_chunks: list[str] = []
            full_response = ""
            try:
                with Live(console=console.console, transient=True, refresh_per_second=12) as live:
                    async for event in agent.stream_message(user_input):
                        if event.kind is AgentEventKind.FINAL_REPORT:
                            full_response = event.text
                            continue
                        render_stream_event(event, live)

            except Exception:
                # Fallback to regular processing
                full_response = await agent.process_message(user_input)

            # Print the final report as a panel
            console.print()
            console.print_panel(
                Markdown(full_response),
                title=f"[bold {NEON_CYAN}]Response[bold {NEON_CYAN}]",
                border_style="primary",
                title_align="left",
                padding=(1, 2),
            )

            # Append the per-run usage/cost line when the provider reported usage (HH-04).
            from .runs_command import format_usage_cost_line

            usage_line = format_usage_cost_line(agent.last_run)
            if usage_line:
                console.print(f"[dim]{usage_line}[/dim]")
            console.print()

            if agent_config.behavior.auto_save_context:
                session_store.save(
                    SessionRecord(
                        name=current_session,
                        context=agent.context,
                        provider=current_provider,
                        model=current_model,
                        project_root=str(working_directory.resolve()),
                    )
                )

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
                # Stop promptly once cancellation has been requested so a run
                # that swallowed asyncio cancellation still exits cleanly.
                if loop_config.cancel_event is not None and loop_config.cancel_event.is_set():
                    break

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
                        # Session-scope approvals die with the conversation.
                        agent.context.metadata.pop(APPROVAL_MEMORY_KEY, None)
                        message_queue.clear()
                        session_store.clear(current_session)
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

                    elif command == "/plan" or command.startswith("/plan "):
                        parts = user_input.split(maxsplit=1)
                        plan_arg = parts[1].strip().lower() if len(parts) > 1 else ""
                        if not plan_arg:
                            hint = PLAN_MODE_HINTS.get(loop_config.plan_mode, "")
                            console.print(
                                f"\n[bold {NEON_CYAN}]Plan Mode:[/bold {NEON_CYAN}] "
                                f"[bold]{loop_config.plan_mode}[/bold] [dim]({hint})[/dim]"
                            )
                            console.print("[dim]Set with /plan auto|force|off[/dim]\n")
                            continue
                        try:
                            loop_config.set_plan_mode(plan_arg)
                        except ValueError:
                            console.print_error("Plan mode must be one of: auto, force, off")
                            continue
                        console.print_success(
                            f"Plan mode set to {plan_arg} ({PLAN_MODE_HINTS[plan_arg]}) for the next tasks"
                        )
                        continue

                    elif command == "/tools":
                        console.print(f"\n[bold {NEON_CYAN}]Available Tools:[bold {NEON_CYAN}]")
                        for tool_name in tools.list_tools():
                            console.print(
                                f"  [bold {NEON_GOLD}]•[bold {NEON_GOLD}] "
                                f"[bold {NEON_WHITE}]{tool_name}[bold {NEON_WHITE}]"
                            )
                        continue

                    elif command.startswith("/save"):
                        parts = user_input.split(maxsplit=1)
                        current_session = parts[1].strip() if len(parts) > 1 else current_session
                        loop_config.session_name = current_session
                        path = session_store.save(
                            SessionRecord(
                                name=current_session,
                                context=agent.context,
                                provider=current_provider,
                                model=current_model,
                                project_root=str(working_directory.resolve()),
                            )
                        )
                        console.print_success(f"Session saved: {path}")
                        continue

                    elif command.startswith("/resume"):
                        parts = user_input.split(maxsplit=1)
                        name = parts[1].strip() if len(parts) > 1 else current_session
                        record = session_store.load(name)
                        agent.context = record.context
                        current_session = name
                        loop_config.session_name = current_session
                        console.print_success(f"Session resumed: {name} ({agent.context.get_message_count()} messages)")
                        continue

                    elif command.startswith("/export"):
                        parts = user_input.split(maxsplit=1)
                        if len(parts) == 1:
                            console.print_error("Usage: /export <path>")
                            continue
                        destination = Path(parts[1]).expanduser()
                        if not destination.is_absolute():
                            destination = working_directory / destination
                        path = session_store.export(
                            SessionRecord(
                                name=current_session,
                                context=agent.context,
                                provider=current_provider,
                                model=current_model,
                                project_root=str(working_directory.resolve()),
                            ),
                            destination,
                        )
                        console.print_success(f"Session exported: {path}")
                        continue

                    elif command.startswith("/provider"):
                        parts = user_input.split(maxsplit=1)
                        provider_arg = parts[1].strip() if len(parts) > 1 else ""

                        new_provider, new_model, changed = handle_provider_command(
                            provider_arg,
                            global_config_manager,
                            current_provider,
                            current_model,
                        )

                        if changed:
                            current_provider = new_provider
                            current_model = new_model

                            # Create new LLM provider
                            merged_config = config_manager.load()
                            provider_config = merged_config.get_provider_config(new_provider)
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

                            # Update banner
                            print_banner(new_provider, new_model, verbose=False)

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
                # Let any in-flight run unwind cleanly: the agent loop checks
                # this event between iterations and inside retry waits.
                loop_config.cancel_event.set()
                console.print(f"\n[bold {NEON_CYAN}]Bye![bold {NEON_CYAN}]")
                break

            except EOFError:
                _disable_cursor_blink()
                console.print(f"\n[bold {NEON_CYAN}]Bye![bold {NEON_CYAN}]")
                break

    finally:
        # Clean up: request cancellation so a run in flight finishes its
        # current tool call, persists session state, and stops iterating.
        loop_config.cancel_event.set()
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
    plan_mode: str = "auto",
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
        plan_mode: Planning override for tasks (HH-07): auto, force, or off
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
            plan_mode=plan_mode,
        )
    )
