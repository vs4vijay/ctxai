import typer
from pathlib import Path
from typing import Optional, List

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
    name: str = typer.Argument(
        ...,
        help="Name for the index",
    ),
    include: Optional[List[str]] = typer.Option(
        None,
        "--include",
        "-i",
        help="File patterns to include (e.g., '*.py', '*.js'). Can be specified multiple times.",
    ),
    exclude: Optional[List[str]] = typer.Option(
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
    
    typer.echo(f"🚀 Indexing codebase at: {path}")
    typer.echo(f"📝 Index name: {name}")
    
    index_codebase(
        path=path,
        index_name=name,
        include_patterns=include,
        exclude_patterns=exclude,
        follow_gitignore=follow_gitignore,
    )
    
    typer.echo(f"✅ Successfully indexed codebase as '{name}'")


@app.command()
def query(
    index_name: str = typer.Argument(
        ...,
        help="Name of the index to query",
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
    project_path: Optional[Path] = typer.Option(
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


def main():
    app()


if __name__ == "__main__":
    main()