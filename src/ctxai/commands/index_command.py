"""
Index command implementation.
Orchestrates the entire indexing pipeline: traversal, chunking, embedding, and storage.
"""

from pathlib import Path
from typing import List, Optional

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from ..chunking import CodeChunker
from ..embeddings import EmbeddingsGenerator
from ..traversal import CodeTraversal
from ..vector_store import VectorStore

console = Console()


def index_codebase(
    path: Path,
    index_name: str,
    include_patterns: Optional[List[str]] = None,
    exclude_patterns: Optional[List[str]] = None,
    follow_gitignore: bool = True,
):
    """
    Index a codebase for semantic search.

    Args:
        path: Path to the codebase directory
        index_name: Name for the index
        include_patterns: File patterns to include
        exclude_patterns: Additional patterns to exclude
        follow_gitignore: Whether to respect .gitignore
    """
    console.print("\n[bold blue]🚀 Starting codebase indexing...[/bold blue]\n")

    # Initialize components
    console.print("[cyan]Initializing components...[/cyan]")
    
    traversal = CodeTraversal(
        root_path=path,
        include_patterns=include_patterns,
        exclude_patterns=exclude_patterns,
        follow_gitignore=follow_gitignore,
    )
    
    chunker = CodeChunker(max_chunk_size=1000, overlap=100)
    
    try:
        embeddings_generator = EmbeddingsGenerator()
        console.print("[green]✓[/green] OpenAI API connection established\n")
    except ValueError as e:
        console.print(f"[red]✗[/red] Error: {e}\n")
        console.print(
            "[yellow]Tip:[/yellow] Set your OpenAI API key: "
            "[cyan]export OPENAI_API_KEY=your-key-here[/cyan]\n"
        )
        return

    # Storage path in .ctxai directory
    storage_path = path / ".ctxai" / "indexes" / index_name
    vector_store = VectorStore(storage_path=storage_path, collection_name=index_name)

    # Phase 1: Traverse and collect files
    console.print("[bold cyan]Phase 1: Traversing codebase[/bold cyan]")
    
    files_to_process = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Scanning files...", total=None)
        for file_path in traversal.traverse():
            files_to_process.append(file_path)
            progress.update(task, description=f"Found {len(files_to_process)} files...")
    
    console.print(f"[green]✓[/green] Found {len(files_to_process)} files to process\n")
    
    if not files_to_process:
        console.print("[yellow]⚠[/yellow] No files found to index. Check your include/exclude patterns.\n")
        return

    # Phase 2: Chunk files
    console.print("[bold cyan]Phase 2: Chunking code[/bold cyan]")
    
    all_chunks = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Chunking files...", total=len(files_to_process))
        
        for file_path in files_to_process:
            try:
                chunks = chunker.chunk_file(file_path)
                all_chunks.extend(chunks)
                progress.update(
                    task,
                    advance=1,
                    description=f"Chunking files... ({len(all_chunks)} chunks so far)",
                )
            except Exception as e:
                console.print(f"[red]✗[/red] Error chunking {file_path}: {e}")
    
    console.print(f"[green]✓[/green] Created {len(all_chunks)} code chunks\n")
    
    if not all_chunks:
        console.print("[yellow]⚠[/yellow] No chunks created. Nothing to index.\n")
        return

    # Phase 3: Generate embeddings
    console.print("[bold cyan]Phase 3: Generating embeddings[/bold cyan]")
    
    chunk_texts = [chunk.content for chunk in all_chunks]
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Generating embeddings...", total=len(chunk_texts))
        
        # Process in batches to show progress
        batch_size = 100
        all_embeddings = []
        
        for i in range(0, len(chunk_texts), batch_size):
            batch = chunk_texts[i : i + batch_size]
            try:
                batch_embeddings = embeddings_generator.generate_embeddings(batch)
                all_embeddings.extend(batch_embeddings)
                progress.update(task, advance=len(batch))
            except Exception as e:
                console.print(f"[red]✗[/red] Error generating embeddings: {e}")
                return
    
    console.print(f"[green]✓[/green] Generated {len(all_embeddings)} embeddings\n")

    # Phase 4: Store in vector database
    console.print("[bold cyan]Phase 4: Storing in vector database[/bold cyan]")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Storing embeddings...", total=None)
        vector_store.add_chunks(all_chunks, all_embeddings)
    
    console.print(f"[green]✓[/green] Stored embeddings in vector database\n")

    # Print summary
    stats = vector_store.get_stats()
    console.print("[bold green]✅ Indexing complete![/bold green]\n")
    console.print("[bold]Summary:[/bold]")
    console.print(f"  • Index name: [cyan]{index_name}[/cyan]")
    console.print(f"  • Storage path: [cyan]{storage_path}[/cyan]")
    console.print(f"  • Total chunks: [cyan]{stats.get('total_chunks', 0)}[/cyan]")
    console.print(f"  • Unique files: [cyan]{stats.get('unique_files', 0)}[/cyan]")
    
    languages = stats.get('languages', {})
    if languages:
        console.print(f"  • Languages: [cyan]{', '.join(languages.keys())}[/cyan]")
    
    console.print()
