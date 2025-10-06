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
def server(
    index_name: str = typer.Option(
        ...,
        "--index",
        "-i",
        help="Name of the index to serve",
    ),
    port: int = typer.Option(
        8000,
        "--port",
        "-p",
        help="Port to run the server on",
    ),
):
    """
    Start the MCP server for semantic code search.
    """
    typer.echo(f"🚀 Starting MCP server with index: {index_name}")
    typer.echo(f"📡 Server will run on port: {port}")
    
    # TODO: Implement server command
    typer.echo("⚠️  Server command not yet implemented")


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
    Start the web dashboard for browsing indexed codebases.
    """
    typer.echo(f"🚀 Starting dashboard on port: {port}")
    
    # TODO: Implement dashboard command
    typer.echo("⚠️  Dashboard command not yet implemented")


def main():
    app()


if __name__ == "__main__":
    main()