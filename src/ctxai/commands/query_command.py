"""
Query command implementation.
Search indexed codebase using natural language queries.
"""

from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

from ..config import ConfigManager
from ..embeddings import EmbeddingsFactory
from ..utils import get_indexes_dir
from ..vector_store import VectorStore

console = Console()


def query_codebase(
    index_name: Optional[str],
    query: str,
    project_path: Optional[Path] = None,
    n_results: int = 5,
    show_content: bool = True,
):
    """
    Query an indexed codebase using natural language.

    Args:
        index_name: Name of the index to query (uses config default if None)
        query: Natural language query
        project_path: Optional project path (uses CTXAI_HOME if not provided)
        n_results: Number of results to return
        show_content: Whether to show full code content
    """
    # Load configuration
    config_manager = ConfigManager(project_path)
    config = config_manager.load()
    
    # Determine index name: use provided or fall back to config
    if index_name is None:
        index_name = config.index_name
        if index_name is None:
            console.print("[red]✗[/red] No index name provided and none configured\n")
            console.print("[yellow]Tip:[/yellow] Either:\n")
            console.print("  1. Provide an index name: [cyan]ctxai query <index_name> \"your query\"[/cyan]\n")
            console.print("  2. Or set a default: [cyan]ctxai config --set index.name <index_name>[/cyan]\n")
            return
        console.print(f"[dim]Using configured index: {index_name}[/dim]")
    
    console.print(f"\n[bold blue]🔍 Searching index '{index_name}'...[/bold blue]\n")
    console.print(f"[dim]Query: {query}[/dim]\n")

    try:
        # Create embedding provider
        console.print(f"[dim]Using embedding provider: {config.embedding.provider}[/dim]")
        
        # Initialize embedding provider
        embeddings_generator = EmbeddingsFactory.create(config.embedding)
        
        # Load vector store
        indexes_dir = get_indexes_dir(project_path)
        storage_path = indexes_dir / index_name
        
        if not storage_path.exists():
            console.print(
                f"[red]✗[/red] Index '{index_name}' not found at {storage_path}\n"
            )
            console.print("[yellow]Tip:[/yellow] Run [cyan]ctxai index[/cyan] first to create an index\n")
            return
        
        vector_store = VectorStore(storage_path=storage_path, collection_name=index_name)
        
        # Generate query embedding
        console.print("[cyan]Generating query embedding...[/cyan]")
        query_embedding = embeddings_generator.generate_embedding(query)
        
        # Search
        console.print("[cyan]Searching vector database...[/cyan]\n")
        results = vector_store.search(
            query_embedding=query_embedding,
            n_results=n_results,
        )
        
        if not results:
            console.print("[yellow]No results found[/yellow]\n")
            return
        
        # Display results
        console.print(f"[bold green]✓ Found {len(results)} result(s)[/bold green]\n")
        
        for i, result in enumerate(results, 1):
            metadata = result["metadata"]
            content = result["content"]
            distance = result["distance"]
            
            # Calculate similarity score (lower distance = higher similarity)
            similarity = max(0, 1 - distance)
            
            # Create header
            file_path = Path(metadata["file_path"])
            header = f"Result {i}: {file_path.name}"
            
            # Create info table
            info_table = Table(show_header=False, box=None, padding=(0, 1))
            info_table.add_column("Key", style="cyan")
            info_table.add_column("Value", style="white")
            
            info_table.add_row("📁 File", str(file_path))
            info_table.add_row("📍 Lines", f"{metadata['start_line']}-{metadata['end_line']}")
            info_table.add_row("🏷️  Type", metadata["chunk_type"])
            info_table.add_row("💻 Language", metadata["language"])
            info_table.add_row("🎯 Similarity", f"{similarity:.1%}")
            
            # Add metadata if present
            for key, value in metadata.items():
                if key.startswith("meta_"):
                    clean_key = key.replace("meta_", "").title()
                    info_table.add_row(f"ℹ️  {clean_key}", value)
            
            console.print(Panel(info_table, title=header, border_style="blue"))
            
            # Show code content if requested
            if show_content:
                # Try to detect language for syntax highlighting
                language = metadata.get("language", "python")
                language_map = {
                    "python": "python",
                    "javascript": "javascript",
                    "typescript": "typescript",
                    "java": "java",
                    "cpp": "cpp",
                    "c": "c",
                    "go": "go",
                    "rust": "rust",
                    "ruby": "ruby",
                }
                
                syntax_lang = language_map.get(language, "text")
                
                # Limit content length for display
                display_content = content
                if len(content) > 1000:
                    display_content = content[:1000] + "\n... (truncated)"
                
                syntax = Syntax(
                    display_content,
                    syntax_lang,
                    theme="monokai",
                    line_numbers=True,
                    start_line=int(metadata["start_line"]),
                )
                console.print(syntax)
            
            console.print()  # Empty line between results
        
        # Show summary
        console.print("[dim]─" * 80 + "[/dim]")
        console.print(f"[dim]Showing {len(results)} of {len(results)} results[/dim]\n")
        
    except Exception as e:
        console.print(f"[red]✗ Error querying index: {e}[/red]\n")
        import traceback
        console.print(f"[dim]{traceback.format_exc()}[/dim]")
