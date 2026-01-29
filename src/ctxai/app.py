from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(
    name="ctxai",
    help="A semantic code search engine for intelligent code discovery",
    no_args_is_help=True,
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

    start_dashboard(port=port)


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
    Manage ctxai configuration settings (similar to git config).

    Configuration is stored in .ctxai/config.json and can be managed at:
    - Global level (CTXAI_HOME environment variable)
    - Project level (.ctxai in current directory)

    Examples:
        # List all configuration
        ctxai config --list

        # Get a specific value
        ctxai config --get embedding.provider

        # Set a value
        ctxai config --set embedding.provider --value openai
        ctxai config --set embedding.api_key --value sk-xxx
        ctxai config --set indexing.chunk_size --value 1500

        # Unset a value (revert to default)
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
        list_config(project_path=project_path)
    elif operations_count > 1:
        typer.echo("Error: Please specify only one operation at a time")
        raise typer.Exit(code=1)
    elif list_all:
        list_config(project_path=project_path)
    elif get:
        get_config(key=get, project_path=project_path)
    elif set_key:
        if value is None:
            typer.echo("Error: --value is required when using --set")
            raise typer.Exit(code=1)
        set_config(key=set_key, value=value, project_path=project_path)
    elif unset:
        unset_config(key=unset, project_path=project_path)
    elif show_file:
        show_config_file(project_path=project_path)
    elif edit:
        edit_config(project_path=project_path)


@app.command()
def chat(
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose output showing tool calls and iterations",
    ),
    max_iterations: int = typer.Option(
        10,
        "--max-iterations",
        "-m",
        help="Maximum iterations per request",
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

    The agent has access to:
    - File operations (read, write, edit, search)
    - Bash command execution
    - Code search and analysis
    - Git operations (coming soon)

    Example requests:
        "Read the README.md file"
        "List all Python files in src/"
        "Create a new function to validate emails"
        "What is the current git status?"

    Commands within chat:
        /help    - Show available commands
        /clear   - Clear conversation history
        /exit    - Exit chat mode
        /status  - Show agent status
        /tools   - List available tools
    """
    from .commands.chat_command import start_chat

    wd = working_directory or Path.cwd()
    start_chat(working_directory=wd, verbose=verbose, max_iterations=max_iterations)


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
):
    """
    Execute a one-shot coding task with the AI agent.

    This is useful for quick code generation or modification tasks
    without entering interactive mode.

    Examples:
        ctxai code "Create a Python function to validate email addresses"
        ctxai code "Add error handling to the main.py file"
        ctxai code "Generate unit tests for the User class"
    """
    from .commands.chat_command import interactive_chat
    import asyncio

    async def run_task():
        from .agent.core import Agent, AgentLoopConfig
        from .agent.llm.anthropic_provider import AnthropicProvider
        from .agent.config import AgentLLMConfig, AgentConfig
        from .agent.tools.registry import ToolRegistry
        from .agent.tools.file_ops import (
            ReadFileTool, WriteFileTool, EditFileTool,
            ListFilesTool, GlobTool, GrepTool
        )
        from .agent.tools.bash_tool import BashTool
        from .agent.tools.code_search import SemanticSearchTool
        from rich.console import Console

        console = Console()

        if not os.getenv("ANTHROPIC_API_KEY"):
            console.print("❌ [red]ANTHROPIC_API_KEY not set[/red]")
            return

        # Initialize agent
        llm_config = AgentLLMConfig(
            provider="anthropic",
            model="claude-3-5-sonnet-20241022",
            temperature=0.7,
            max_tokens=4096,
        )
        llm = AnthropicProvider(llm_config)
        agent_config = AgentConfig()

        tools = ToolRegistry(verbose=verbose)
        tools.register(ReadFileTool())
        tools.register(WriteFileTool())
        tools.register(EditFileTool())
        tools.register(ListFilesTool())
        tools.register(GlobTool())
        tools.register(GrepTool())
        tools.register(BashTool(agent_config.tools))
        tools.register(SemanticSearchTool())

        loop_config = AgentLoopConfig(
            llm_provider=llm,
            tool_registry=tools,
            agent_config=agent_config,
            working_directory=Path.cwd(),
            available_indexes=[],
            max_iterations=max_iterations,
            verbose=verbose,
        )
        agent = Agent(loop_config)

        console.print(f"\n🤖 [cyan]Task:[/cyan] {task}\n")

        with console.status("[dim]Processing...[/dim]"):
            response = await agent.process_message(task)

        from rich.markdown import Markdown
        console.print("\n[bold green]Result:[/bold green]")
        console.print(Markdown(response))

    asyncio.run(run_task())


def main():
    app()


if __name__ == "__main__":
    main()
