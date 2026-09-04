import os
from pathlib import Path

import typer

app = typer.Typer(
    name="ctxai",
    help="A semantic code search engine for intelligent code discovery",
    no_args_is_help=False,
    invoke_without_command=True,
)

indexes_app = typer.Typer(help="Inspect and manage persistent code indexes")


@indexes_app.command("list")
def indexes_list(
    project_path: Path | None = typer.Option(None, "--project-path", "-p"),
):
    """List indexes and their repository, file, and chunk counts."""
    from rich.console import Console
    from rich.table import Table

    from .commands.indexes_command import list_indexes

    manifests = list_indexes(project_path)
    table = Table("Name", "Repository", "Files", "Chunks", "Updated")
    for manifest in manifests:
        table.add_row(
            manifest.index_name,
            manifest.repository_root,
            str(manifest.file_count),
            str(manifest.chunk_count),
            manifest.updated_at,
        )
    Console().print(table)


@indexes_app.command("info")
def indexes_info(
    name: str = typer.Argument(...),
    project_path: Path | None = typer.Option(None, "--project-path", "-p"),
):
    """Show the complete versioned manifest for an index."""
    import json
    from dataclasses import asdict

    from .commands.indexes_command import get_index_info

    typer.echo(json.dumps(asdict(get_index_info(name, project_path)), indent=2))


@indexes_app.command("doctor")
def indexes_doctor(
    name: str = typer.Argument(...),
    project_path: Path | None = typer.Option(None, "--project-path", "-p"),
):
    """Check manifest, vector storage, and repository freshness."""
    from .commands.indexes_command import doctor_index

    health = doctor_index(name, project_path)
    if health.healthy:
        typer.echo(f"[OK] Index '{name}' is healthy")
        return
    for problem in health.problems:
        typer.echo(f"[X] {problem}", err=True)
    raise typer.Exit(code=1)


@indexes_app.command("delete")
def indexes_delete(
    name: str = typer.Argument(...),
    yes: bool = typer.Option(False, "--yes", help="Delete without interactive confirmation"),
    project_path: Path | None = typer.Option(None, "--project-path", "-p"),
):
    """Permanently delete a named index."""
    from .commands.indexes_command import delete_index

    if not yes and not typer.confirm(f"Delete index '{name}'?"):
        raise typer.Abort()
    path = delete_index(name, project_path)
    typer.echo(f"Deleted index '{name}' at {path}")


app.add_typer(indexes_app, name="indexes")


runs_app = typer.Typer(help="Inspect and manage local agent run transcripts")


@runs_app.command("list")
def runs_list(
    limit: int | None = typer.Option(None, "--limit", "-l", help="Maximum number of runs to show (newest first)"),
    as_json: bool = typer.Option(False, "--json", help="Emit a versioned JSON envelope instead of a table"),
    project_path: Path | None = typer.Option(None, "--project-path", "-p"),
):
    """List past agent runs with status, token usage, and estimated cost."""
    import json

    from rich.console import Console
    from rich.table import Table

    from .commands.runs_command import list_runs

    summaries = list_runs(project_path, limit=limit)
    if as_json:
        envelope = {"schema_version": 1, "runs": [summary.to_dict() for summary in summaries]}
        typer.echo(json.dumps(envelope, indent=2))
        return
    table = Table("Run ID", "Started", "Status", "Events", "Tokens (prompt + completion)", "Calls", "Cost")
    for summary in summaries:
        table.add_row(
            summary.run_id,
            summary.started_at or "?",
            summary.status,
            str(summary.event_count),
            f"{summary.prompt_tokens:,} + {summary.completion_tokens:,}",
            str(summary.calls),
            summary.cost_display,
        )
    Console().print(table)


@runs_app.command("show")
def runs_show(
    run_id: str = typer.Argument(...),
    as_json: bool = typer.Option(False, "--json", help="Emit the raw on-disk events as versioned JSON"),
    kind: str | None = typer.Option(None, "--kind", "-k", help="Only show events of this kind (e.g. llm_call)"),
    project_path: Path | None = typer.Option(None, "--project-path", "-p"),
):
    """Render one run's transcript events, optionally filtered by kind."""
    import json

    from rich.console import Console

    from .agent.run_recorder import RUN_EVENT_KINDS
    from .commands.runs_command import read_run_events

    if kind is not None and kind not in RUN_EVENT_KINDS:
        typer.echo(f"Error: unknown event kind '{kind}'. Valid kinds: {', '.join(sorted(RUN_EVENT_KINDS))}", err=True)
        raise typer.Exit(code=1)
    try:
        events = read_run_events(run_id, project_path)
    except (FileNotFoundError, ValueError) as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(code=1) from error
    if kind is not None:
        events = [event for event in events if event.kind == kind]
    if as_json:
        envelope = {
            "schema_version": events[0].schema_version if events else 1,
            "run_id": run_id,
            "events": [event.to_dict() for event in events],
        }
        typer.echo(json.dumps(envelope, indent=2))
        return
    console = Console()
    for event in events:
        console.print(f"[bold]#{event.seq}[/bold] [cyan]{event.kind}[/cyan] [dim]{event.timestamp}[/dim]")
        console.print_json(json.dumps(event.to_dict()["payload"], sort_keys=True))
        if event.usage is not None:
            console.print("[dim]usage:[/dim]")
            console.print_json(json.dumps(event.usage, sort_keys=True))


@runs_app.command("delete")
def runs_delete(
    run_id: str | None = typer.Argument(None, help="Run id to delete (omit with --all)"),
    all_runs: bool = typer.Option(False, "--all", help="Delete every run transcript (asks for confirmation)"),
    project_path: Path | None = typer.Option(None, "--project-path", "-p"),
):
    """Delete one run transcript, or all of them with --all."""
    from .commands.runs_command import delete_all_runs, delete_run, resolve_runs_dir

    if all_runs:
        if run_id:
            typer.echo("Error: specify either RUN_ID or --all, not both", err=True)
            raise typer.Exit(code=1)
        runs_dir = resolve_runs_dir(project_path)
        count = len(list(runs_dir.glob("*.jsonl"))) if runs_dir.is_dir() else 0
        if count == 0:
            typer.echo(f"No run transcripts under {runs_dir}")
            return
        if not typer.confirm(f"Delete all {count} run transcript(s) under {runs_dir}?"):
            raise typer.Abort()
        deleted = delete_all_runs(project_path)
        typer.echo(f"Deleted {deleted} run transcript(s)")
        return

    if not run_id:
        typer.echo("Error: provide RUN_ID or --all", err=True)
        raise typer.Exit(code=1)
    try:
        path = delete_run(run_id, project_path)
    except (FileNotFoundError, ValueError) as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(code=1) from error
    typer.echo(f"Deleted run transcript {path}")


app.add_typer(runs_app, name="runs")


checkpoints_app = typer.Typer(help="Inspect and manage run checkpoints for rollback")


@checkpoints_app.command("list")
def checkpoints_list(
    run: str | None = typer.Option(None, "--run", help="Only show the checkpoint of this run id"),
    as_json: bool = typer.Option(False, "--json", help="Emit a versioned JSON envelope instead of a table"),
    project_path: Path | None = typer.Option(None, "--project-path", "-p"),
):
    """List captured run checkpoints with status, file count, and size."""
    import json

    from rich.console import Console
    from rich.table import Table

    from .commands.checkpoints_command import list_checkpoints

    try:
        checkpoints = list_checkpoints(project_path, run_id=run)
    except (FileNotFoundError, ValueError) as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(code=1) from error
    if as_json:
        envelope = {"schema_version": 1, "checkpoints": [checkpoint.to_dict() for checkpoint in checkpoints]}
        typer.echo(json.dumps(envelope, indent=2))
        return
    table = Table("Checkpoint", "Created", "Status", "Files", "Bytes")
    for checkpoint in checkpoints:
        status = "retained" if checkpoint.retained else checkpoint.status
        table.add_row(
            checkpoint.checkpoint_id,
            checkpoint.created_at or "?",
            status,
            str(len(checkpoint.files)),
            f"{checkpoint.bytes_captured:,}",
        )
    Console().print(table)


@checkpoints_app.command("restore")
def checkpoints_restore(
    checkpoint_id: str = typer.Argument(..., help="Checkpoint id (equals the run id)"),
    force: bool = typer.Option(
        False,
        "--force",
        help="Restore even when the working tree moved on since the run (per-file stale refusals)",
    ),
    project_path: Path | None = typer.Option(None, "--project-path", "-p"),
):
    """Restore the working tree to a checkpoint's pre-run state (asks for confirmation)."""
    from rich.console import Console
    from rich.table import Table

    from .commands.checkpoints_command import get_checkpoint, restore_checkpoint

    console = Console()
    try:
        checkpoint = get_checkpoint(checkpoint_id, project_path)
    except (FileNotFoundError, ValueError) as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(code=1) from error
    if not checkpoint.files:
        typer.echo(f"Checkpoint '{checkpoint_id}' has no captured files; nothing to restore")
        return
    table = Table("File", "Captured as")
    for entry in checkpoint.files:
        table.add_row(entry.path, entry.kind.value)
    console.print(table)
    if not typer.confirm(f"Restore {len(checkpoint.files)} file(s) to their pre-run state?"):
        raise typer.Abort()
    try:
        result = restore_checkpoint(checkpoint_id, project_path, force=force)
    except (FileNotFoundError, ValueError) as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(code=1) from error
    if not result.applied:
        console.print("[red]Restore refused: the working tree moved on since the run.[/red]")
        for item in result.refused:
            console.print(f"[red]- {item.path}: {item.detail}[/red]")
        console.print("Use --force to restore anyway.")
        raise typer.Exit(code=1)
    for item in result.results:
        suffix = f" ({item.detail})" if item.detail else ""
        console.print(f"[green][OK][/green] {item.path}: {item.action}{suffix}")
    typer.echo(f"Restored checkpoint '{checkpoint_id}'")


@checkpoints_app.command("delete")
def checkpoints_delete(
    checkpoint_id: str | None = typer.Argument(None, help="Checkpoint id to delete (omit with --all)"),
    all_checkpoints: bool = typer.Option(False, "--all", help="Delete every checkpoint (asks for confirmation)"),
    project_path: Path | None = typer.Option(None, "--project-path", "-p"),
):
    """Delete one checkpoint, or all of them with --all."""
    from .commands.checkpoints_command import delete_all_checkpoints, delete_checkpoint, resolve_checkpoints_dir

    if all_checkpoints:
        if checkpoint_id:
            typer.echo("Error: specify either CHECKPOINT_ID or --all, not both", err=True)
            raise typer.Exit(code=1)
        checkpoints_dir = resolve_checkpoints_dir(project_path)
        count = (
            len([path for path in checkpoints_dir.iterdir() if path.is_dir() and (path / "manifest.json").is_file()])
            if checkpoints_dir.is_dir()
            else 0
        )
        if count == 0:
            typer.echo(f"No checkpoints under {checkpoints_dir}")
            return
        if not typer.confirm(f"Delete all {count} checkpoint(s) under {checkpoints_dir}?"):
            raise typer.Abort()
        deleted = delete_all_checkpoints(project_path)
        typer.echo(f"Deleted {deleted} checkpoint(s)")
        return

    if not checkpoint_id:
        typer.echo("Error: provide CHECKPOINT_ID or --all", err=True)
        raise typer.Exit(code=1)
    try:
        path = delete_checkpoint(checkpoint_id, project_path)
    except (FileNotFoundError, ValueError) as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(code=1) from error
    typer.echo(f"Deleted checkpoint {path}")


app.add_typer(checkpoints_app, name="checkpoints")


@app.callback()
def main_callback(ctx: typer.Context):
    """
    ctxai - A semantic code search engine for intelligent code discovery

    When run without a command, starts interactive chat mode.
    Use 'ctxai --help' to see all available commands.
    """
    # If no subcommand was invoked, default to chat
    if ctx.invoked_subcommand is None:
        from .commands.chat_command import start_chat

        start_chat(
            working_directory=Path.cwd(),
            provider=None,  # Use default_provider from config
            model=None,  # Use model from config
            architect_editor=False,
            architect_model=None,
            editor_model=None,
            preset="default",
            use_repomap=False,
            verbose=False,
            max_iterations=10,
            plan_mode="auto",
        )


@app.command()
def index(
    path: Path = typer.Argument(
        ...,
        help="Path to the codebase directory to index",
        exists=True,
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
    ),
    name: str | None = typer.Argument(
        None,
        help="Name for the index (uses configured default if not provided)",
    ),
    include: list[str] | None = typer.Option(
        None,
        "--include",
        "-i",
        help="File patterns to include (e.g., '*.py', '*.js'). Can be specified multiple times.",
    ),
    exclude: list[str] | None = typer.Option(
        None,
        "--exclude",
        "-e",
        help="Additional file patterns to exclude beyond .gitignore. Can be specified multiple times.",
    ),
    follow_gitignore: bool = typer.Option(
        True,
        "--follow-gitignore/--no-follow-gitignore",
        help="Follow .gitignore patterns when traversing the codebase",
    ),
):
    """
    Index a codebase for semantic search.

    This command traverses your codebase, chunks the code intelligently,
    generates embeddings, and stores them in a local vector database.
    """
    from .commands.index_command import index_codebase

    if name:
        typer.echo(f"[*] Indexing codebase at: {path}")
        typer.echo(f"[*] Index name: {name}")
    else:
        typer.echo(f"[*] Indexing codebase at: {path}")
        typer.echo("[*] Using configured or default index name")

    index_codebase(
        path=path,
        index_name=name,
        include_patterns=include,
        exclude_patterns=exclude,
        follow_gitignore=follow_gitignore,
    )

    if name:
        typer.echo(f"[OK] Successfully indexed codebase as '{name}'")


@app.command()
def query(
    index_name: str | None = typer.Argument(
        None,
        help="Name of the index to query (uses configured default if not provided)",
    ),
    query_text: str = typer.Argument(
        ...,
        help="Natural language query to search the codebase",
    ),
    n_results: int = typer.Option(
        5,
        "--n-results",
        "-n",
        help="Number of results to return",
    ),
    no_content: bool = typer.Option(
        False,
        "--no-content",
        help="Don't show code content, only metadata",
    ),
):
    """
    Query an indexed codebase using natural language.

    This command searches the vector database using semantic similarity
    and returns the most relevant code chunks.
    """
    from .commands.query_command import query_codebase

    query_codebase(
        index_name=index_name,
        query=query_text,
        n_results=n_results,
        show_content=not no_content,
    )


@app.command()
def server(
    project_path: Path | None = typer.Option(
        None,
        "--project-path",
        "-p",
        help="Project path for configuration (uses CTXAI_HOME if not provided)",
        exists=True,
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
    ),
):
    """
    Start the MCP (Model Context Protocol) server for AI agents.

    Exposes ctxai functionality as MCP tools that can be used by LLMs
    and AI agents. The server communicates via stdio and can be integrated
    with MCP-compatible clients like Claude Desktop.

    Available tools:
    - list_indexes: List all available code indexes
    - index_codebase: Index a new codebase
    - query_codebase: Query indexed code with natural language
    - get_index_stats: Get detailed statistics about an index

    Example MCP client configuration (Claude Desktop):
    {
      "mcpServers": {
        "ctxai": {
          "command": "ctxai",
          "args": ["server"]
        }
      }
    }
    """
    from .commands.server_command import start_mcp_server

    start_mcp_server(project_path=project_path)


@app.command()
def dashboard(
    port: int = typer.Option(
        3000,
        "--port",
        "-p",
        help="Port to run the dashboard on",
    ),
    host: str = typer.Option(
        "127.0.0.1",
        "--host",
        help="Address to bind (remote addresses expose an unauthenticated dashboard)",
    ),
    allow_remote: bool = typer.Option(
        False,
        "--allow-remote",
        help="Acknowledge that remote binding has no authentication or TLS",
    ),
):
    """
    Start the web dashboard for browsing and querying indexed codebases.

    Provides a FastHTML-based web interface to:
    - View all indexes with statistics
    - Query codebases using natural language
    - Browse chunks and metadata
    - View configuration settings
    """
    from .commands.dashboard_command import start_dashboard

    start_dashboard(port=port, host=host, allow_remote=allow_remote)


@app.command()
def config(
    list_all: bool = typer.Option(
        False,
        "--list",
        "-l",
        help="List all configuration settings",
    ),
    get: str | None = typer.Option(
        None,
        "--get",
        "-g",
        help="Get a specific configuration value (e.g., 'embedding.provider')",
    ),
    set_key: str | None = typer.Option(
        None,
        "--set",
        "-s",
        help="Set a configuration key (requires --value)",
    ),
    value: str | None = typer.Option(
        None,
        "--value",
        "-v",
        help="Value to set for the configuration key (used with --set)",
    ),
    unset: str | None = typer.Option(
        None,
        "--unset",
        "-u",
        help="Unset a configuration value (e.g., 'embedding.api_key')",
    ),
    show_file: bool = typer.Option(
        False,
        "--show-file",
        help="Display the raw configuration file content",
    ),
    edit: bool = typer.Option(
        False,
        "--edit",
        "-e",
        help="Show configuration file location for manual editing",
    ),
    global_config: bool = typer.Option(
        False,
        "--global",
        help="Use the global configuration (~/.ctxai/config.toml or $CTXAI_HOME) instead of the project one",
    ),
    project_path: Path | None = typer.Option(
        None,
        "--project-path",
        "-p",
        help="Project path for configuration (uses current directory if not provided)",
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
    ),
):
    """
    Manage ctxai configuration settings (similar to git config).

    Configuration is stored in config.toml with two layers:
    - Global defaults: ~/.ctxai/config.toml (or $CTXAI_HOME/config.toml)
    - Project overrides: .ctxai/config.toml in the project/current directory

    Project values override global defaults key by key. Use --global to
    read/write the global layer instead.

    Examples:
        # List all configuration (merged global + project)
        ctxai config --list

        # Get a specific value
        ctxai config --get embedding.provider

        # Set a value (writes to the project .ctxai/config.toml)
        ctxai config --set embedding.provider --value openai
        ctxai config --set embedding.api_key --value sk-xxx
        ctxai config --set indexing.chunk_size --value 1500

        # Set a global default (writes to ~/.ctxai/config.toml)
        ctxai config --global --set default_provider --value openrouter

        # Unset a value (revert to global/default)
        ctxai config --unset embedding.api_key

        # View raw config file
        ctxai config --show-file

        # Get config file location
        ctxai config --edit
    """
    from .commands.config_command import (
        edit_config,
        get_config,
        list_config,
        set_config,
        show_config_file,
        unset_config,
    )

    # Handle different operations
    operations_count = sum(
        [
            list_all,
            get is not None,
            set_key is not None,
            unset is not None,
            show_file,
            edit,
        ]
    )

    if operations_count == 0:
        # Default to listing config
        list_config(project_path=project_path, global_config=global_config)
    elif operations_count > 1:
        typer.echo("Error: Please specify only one operation at a time")
        raise typer.Exit(code=1)
    elif list_all:
        list_config(project_path=project_path, global_config=global_config)
    elif get:
        get_config(key=get, project_path=project_path, global_config=global_config)
    elif set_key:
        if value is None:
            typer.echo("Error: --value is required when using --set")
            raise typer.Exit(code=1)
        set_config(key=set_key, value=value, project_path=project_path, global_config=global_config)
    elif unset:
        unset_config(key=unset, project_path=project_path, global_config=global_config)
    elif show_file:
        show_config_file(project_path=project_path, global_config=global_config)
    elif edit:
        edit_config(project_path=project_path, global_config=global_config)


@app.command()
def chat(
    provider: str = typer.Option(
        None,  # Changed from "openrouter" to None - use config default
        "--provider",
        "-p",
        help=(
            "LLM provider: openrouter, github-copilot, ollama, anthropic, openai, custom "
            "(uses config default if not specified)"
        ),
    ),
    model: str = typer.Option(
        None,
        "--model",
        "-m",
        help="Model name (provider-specific)",
    ),
    architect_editor: bool = typer.Option(
        False,
        "--architect-editor",
        "-ae",
        help="Experimental architect/editor mode (disabled pending benchmark evidence)",
    ),
    preset: str = typer.Option(
        "default",
        "--preset",
        help="Architect/editor preset: default, premium, budget, cheap, local, mixed",
    ),
    architect_model: str = typer.Option(
        None,
        "--architect-model",
        help="Custom architect model (requires --editor-model)",
    ),
    editor_model: str = typer.Option(
        None,
        "--editor-model",
        help="Custom editor model (requires --architect-model)",
    ),
    use_repomap: bool = typer.Option(
        False,
        "--repomap/--no-repomap",
        help="Use repository mapping for context (disabled by default)",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose output showing tool calls and iterations",
    ),
    max_iterations: int = typer.Option(
        10,
        "--max-iterations",
        help="Maximum iterations per request",
    ),
    plan_mode: str = typer.Option(
        "auto",
        "--plan",
        help="Planning mode for tasks: auto (keyword classification), force (always submit_plan), off (never plan)",
    ),
    working_directory: Path = typer.Option(
        None,
        "--working-directory",
        "-w",
        help="Working directory for the agent (default: current directory)",
        exists=True,
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
    ),
):
    """
    Start interactive chat mode with the AI coding agent.

    This provides a REPL (Read-Eval-Print Loop) interface where you can:
    - Ask questions about your codebase
    - Request code generation or modifications
    - Execute commands and tools
    - Get help with coding tasks

    Multi-Provider Support:
    - OpenRouter: 100+ models (Claude, GPT-4o, o1, DeepSeek, etc.)
    - GitHub Copilot: GPT-4, Claude, o1 via your Copilot subscription
    - Ollama: Local models (CodeLlama, DeepSeek Coder, etc.)
    - Anthropic: Direct Claude access
    - OpenAI: Direct GPT access

    Architect/Editor Pattern:
    Use two models for better quality + lower cost:
    - Architect (expensive): Plans and designs
    - Editor (cheaper): Implements

    Example usage:
        # OpenRouter with Claude
        ctxai chat --provider openrouter --model anthropic/claude-3.5-sonnet

        # GitHub Copilot (if you have subscription)
        ctxai chat --provider github-copilot --model gpt-4

        # Local Ollama model (free!)
        ctxai chat --provider ollama --model codellama:13b

        # Architect/Editor pattern (40-60% cost savings!)
        ctxai chat --architect-editor --preset default

        # Custom architect/editor
        ctxai chat --architect-editor --architect-model openai/o1-mini --editor-model anthropic/claude-3.5-sonnet

        # Mixed providers (cloud + local)
        ctxai chat --architect-editor --preset mixed

    Presets:
        default  - o1-mini + Claude Sonnet (best balance)
        premium  - o1 + Claude Opus (best quality)
        budget   - GPT-4o + GPT-4o-mini (lower cost)
        cheap    - DeepSeek R1 + DeepSeek Chat (cheapest)
        local    - CodeLlama 34B + 13B (fully local)
        mixed    - o1-mini + CodeLlama 13B (cloud + local)

    Commands within chat:
        /help    - Show available commands
        /clear   - Clear conversation history
        /plan    - Show or set planning mode (auto|force|off) for the session
        /exit    - Exit chat mode
        /status  - Show agent status
        /tools   - List available tools
    """
    from .commands.chat_command import start_chat

    wd = working_directory or Path.cwd()
    start_chat(
        working_directory=wd,
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


@app.command()
def code(
    task: str = typer.Argument(
        ...,
        help="Coding task to perform",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose output",
    ),
    max_iterations: int = typer.Option(
        10,
        "--max-iterations",
        "-m",
        help="Maximum iterations",
    ),
    plan_mode: str = typer.Option(
        "auto",
        "--plan",
        help="Planning mode for the task: auto (keyword classification), force (always submit_plan), off (never plan)",
    ),
):
    """
    Execute a one-shot coding task with the AI agent.

    This is useful for quick code generation or modification tasks
    without entering interactive mode.

    Examples:
        ctxai code "Create a Python function to validate email addresses"
        ctxai code "Add error handling to the main.py file"
        ctxai code "Generate unit tests for the User class"
        ctxai code --plan force "Update the VERSION constant"
    """
    import asyncio

    async def run_task():
        from rich.console import Console

        from .agent.checkpoints import CheckpointManager
        from .agent.config import AgentConfig, AgentLLMConfig
        from .agent.core import Agent, AgentLoopConfig, format_compaction_notice
        from .agent.llm.anthropic_provider import AnthropicProvider
        from .agent.resilience import format_retry_notice
        from .agent.tools.bash_tool import BashTool
        from .agent.tools.code_search import SemanticSearchTool
        from .agent.tools.execution import ToolExecutionContext
        from .agent.tools.file_ops import EditFileTool, GlobTool, GrepTool, ListFilesTool, ReadFileTool, WriteFileTool
        from .agent.tools.git_tools import GitDiffTool, GitLogTool, GitStatusTool
        from .agent.tools.registry import ToolRegistry
        from .agent.workflow import format_approval_prompt

        console = Console()

        if not os.getenv("ANTHROPIC_API_KEY"):
            console.print("[red]ANTHROPIC_API_KEY not set[/red]")
            return

        # Initialize agent
        llm_config = AgentLLMConfig(
            provider="anthropic",
            model="claude-3-5-sonnet-20241022",
            temperature=0.7,
            max_tokens=4096,
        )
        llm = AnthropicProvider(llm_config)
        # Agent tools configuration comes from the merged global + project
        # config so `ctxai config --set tools.sandbox ...` applies here too (HH-08).
        from .config import ConfigManager

        agent_config = AgentConfig(tools=ConfigManager(Path.cwd()).load().tools)

        tools = ToolRegistry(verbose=verbose)
        execution_context = ToolExecutionContext.for_project(
            Path.cwd(),
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
        from .repository_context import discover_repository_indexes

        tools.register(SemanticSearchTool(project_path=Path.cwd()))

        try:
            loop_config = AgentLoopConfig(
                llm_provider=llm,
                tool_registry=tools,
                agent_config=agent_config,
                working_directory=Path.cwd(),
                available_indexes=discover_repository_indexes(Path.cwd()),
                planning_enabled=agent_config.behavior.planning_enabled,
                require_user_approval=agent_config.behavior.require_user_approval,
                max_iterations=max_iterations,
                verbose=verbose,
                approval_callback=(lambda call: typer.confirm(format_approval_prompt(call))),
                on_retry=(lambda notice: console.print(f"[yellow]{format_retry_notice(notice)}[/yellow]")),
                on_compaction=(lambda notice: console.print(f"[yellow]{format_compaction_notice(notice)}[/yellow]")),
                # One-shot run: the process-wide ToolExecutionContext.request_id
                # is the transcript run id (HH-04). Chat passes none — every
                # message is a distinct run and gets a fresh id.
                run_id=execution_context.request_id,
                # Local pre-mutation checkpoints (HH-06), bounded by the
                # behavior retention/size config.
                checkpoint_manager=CheckpointManager.for_project(
                    Path.cwd(),
                    retention=agent_config.behavior.checkpoint_retention,
                    max_bytes=agent_config.behavior.checkpoint_max_bytes,
                ),
                plan_mode=plan_mode,
            )
        except ValueError as error:
            console.print(f"[red]{error}[/red]")
            return
        agent = Agent(loop_config)

        console.print(f"\n🤖 [cyan]Task:[/cyan] {task}\n")

        with console.status("[dim]Processing...[/dim]"):
            response = await agent.process_message(task)

        from rich.markdown import Markdown

        console.print("\n[bold green]Result:[/bold green]")
        console.print(Markdown(response))

        # Append the per-run usage/cost line when the provider reported usage (HH-04).
        from .commands.runs_command import format_usage_cost_line

        usage_line = format_usage_cost_line(agent.last_run)
        if usage_line:
            console.print(f"[dim]{usage_line}[/dim]")

    asyncio.run(run_task())


@app.command()
def login(
    provider: str = typer.Argument(
        "openrouter",
        help="Provider to authenticate with (openrouter, github-copilot)",
    ),
    port: int = typer.Option(
        8080,
        "--port",
        "-p",
        help="Port for OAuth callback server (OpenRouter only)",
    ),
):
    """
    Authenticate with an LLM provider using OAuth.

    This command uses OAuth flows to securely authenticate with
    supported providers without manually entering API keys.

    Currently supported providers:
    - openrouter: OAuth PKCE flow (browser-based)
    - github-copilot: OAuth device code flow (enter code at github.com/login/device)

    Examples:
        ctxai login openrouter
        ctxai login openrouter --port 3000
        ctxai login github-copilot

    After authentication, credentials are securely stored and
    automatically used for chat sessions.
    """
    from rich.console import Console

    from .auth.keystore import get_keystore

    console = Console()

    provider_lower = provider.lower()

    # OpenRouter OAuth PKCE
    if provider_lower == "openrouter":
        from .auth.oauth_pkce import authenticate_with_openrouter

        api_key = authenticate_with_openrouter(callback_port=port)

        if not api_key:
            console.print("\n[red]Login failed[/red]")
            raise typer.Exit(code=1)

        keystore = get_keystore()
        keystore.set_key("openrouter", api_key)

        console.print("\n[green]Successfully logged in to OpenRouter![/green]")
        console.print("\n[dim]You can now use:[/dim]")
        console.print("  [cyan]ctxai chat --provider openrouter[/cyan]")

    # GitHub Copilot OAuth Device Code
    elif provider_lower == "github-copilot":
        from .auth.github_copilot import authenticate_with_github_copilot

        token_data = authenticate_with_github_copilot()

        if not token_data:
            console.print("\n[red]Login failed[/red]")
            raise typer.Exit(code=1)

        keystore = get_keystore()
        keystore.set_key("github-copilot", token_data)

        console.print("\n[green]Successfully logged in to GitHub Copilot![/green]")
        console.print("\n[dim]You can now use:[/dim]")
        console.print("  [cyan]ctxai chat --provider github-copilot[/cyan]")

    else:
        console.print(f"[red]OAuth login not supported for '{provider}'[/red]")
        console.print("\n[dim]Supported providers:[/dim]")
        console.print("  [dim]• openrouter - OAuth PKCE flow[/dim]")
        console.print("  [dim]• github-copilot - OAuth device code flow[/dim]")
        console.print("\n[dim]For other providers, use environment variables:[/dim]")
        console.print("  [dim]• ANTHROPIC_API_KEY for Anthropic[/dim]")
        console.print("  [dim]• OPENAI_API_KEY for OpenAI[/dim]")
        raise typer.Exit(code=1)


models_app = typer.Typer(help="Manage and list LLM models")


@models_app.command("list")
def models_list(
    provider: str | None = typer.Option(
        None,
        "--provider",
        "-p",
        help="Specific provider to list models for (default: all providers)",
    ),
    limit: int = typer.Option(
        20,
        "--limit",
        "-l",
        help="Maximum number of models to show per provider",
    ),
    all: bool = typer.Option(
        False,
        "--all",
        "-a",
        help="Show all models (no limit)",
    ),
):
    """
    List available LLM models with their capabilities.

    This command shows which models are available for each configured
    provider. Some providers support model listing via API, while others
    don't provide this functionality.

    Supported providers:
    - openrouter: Lists 100+ models with pricing
    - ollama: Lists locally installed models
    - openai: Lists GPT/o1 models (requires API key)
    - anthropic: Model listing not available via API
    - github-copilot: Model listing not available via API

    Examples:
        ctxai models                    # List models for all providers
        ctxai models --provider openrouter  # List only OpenRouter models
        ctxai models --limit 10         # Show top 10 models
        ctxai models --all             # Show all available models
    """
    from .commands.models_command import list_models

    list_models(provider=provider, limit=limit, show_all=all)


@models_app.command("search")
def models_search(
    query: str = typer.Argument(
        ...,
        help="Search query (model name or description keywords)",
    ),
    provider: str | None = typer.Option(
        None,
        "--provider",
        "-p",
        help="Specific provider to search in",
    ),
    limit: int = typer.Option(
        20,
        "--limit",
        "-l",
        help="Maximum number of results to show",
    ),
):
    """
    Search for models by name or description.

    Examples:
        ctxai models search coding
        ctxai models search claude --provider openrouter
        ctxai models search vision --limit 10
    """
    from .commands.models_command import search_models

    search_models(query=query, provider=provider, limit=limit)


@models_app.command("info")
def models_info(
    model: str = typer.Argument(
        ...,
        help="Model ID to get details for (e.g., 'claude-3.5-sonnet', 'deepseek/deepseek-chat')",
    ),
    provider: str | None = typer.Option(
        None,
        "--provider",
        "-p",
        help="Specific provider to search in",
    ),
):
    """
    Show detailed information about a specific model.

    Examples:
        ctxai models info claude
        ctxai models info deepseek/deepseek-chat --provider openrouter
        ctxai models info codellama:13b --provider ollama
    """
    from .commands.models_command import show_model_details

    show_model_details(model_id=model, provider=provider)


@models_app.command("pull")
def models_pull(
    model: str = typer.Argument(
        ...,
        help="Model name to pull (e.g., 'codellama:13b', 'llama3.1')",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Show detailed download progress",
    ),
):
    """
    Pull a model from Ollama library.

    Examples:
        ctxai models pull codellama:13b
        ctxai models pull llama3.1:70b
        ctxai models pull qwen2.5-coder:7b --verbose
    """
    from .commands.models_command import pull_ollama_model

    success = pull_ollama_model(model_name=model, verbose=verbose)
    if not success:
        import typer

        raise typer.Exit(code=1)


@models_app.command("library")
def models_library(
    limit: int = typer.Option(
        30,
        "--limit",
        "-l",
        help="Maximum number of models to show",
    ),
):
    """
    List popular models available in Ollama library.

    These models can be downloaded using 'ctxai model-pull'.

    Examples:
        ctxai models library
        ctxai models library --limit 10
    """
    from rich.console import Console
    from rich.table import Table

    from .commands.models_command import list_ollama_library_models

    console = Console()
    models = list_ollama_library_models(limit=limit)

    console.print("\n[bold cyan]Popular Ollama Library Models:[/bold cyan]\n")

    table = Table(show_header=True, header_style="bold")
    table.add_column("Model Name", style="cyan")
    table.add_column("Description", max_width=60)

    for m in models:
        table.add_row(m["name"], m["description"])

    console.print(table)
    console.print("\n[dim]To pull a model: ctxai models pull <name>[/dim]")


app.add_typer(models_app, name="models")


@app.command()
def logout(
    provider: str = typer.Argument(
        ...,
        help="Provider to logout from (e.g., 'openrouter')",
    ),
):
    """
    Remove stored credentials for a provider.

    Examples:
        ctxai logout openrouter
        ctxai logout anthropic

    This removes the stored API key for the specified provider.
    You'll need to login again or set environment variables to use the provider.
    """
    from rich.console import Console

    from .auth.keystore import get_keystore

    console = Console()

    keystore = get_keystore()

    if keystore.delete_key(provider):
        console.print(f"\n[green]Logged out from {provider}[/green]")
    else:
        console.print(f"\n[yellow]No stored credentials for {provider}[/yellow]")
        raise typer.Exit(code=1)


def main():
    app()


if __name__ == "__main__":
    main()
