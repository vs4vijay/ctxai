"""
Index command implementation.
Orchestrates the entire indexing pipeline: traversal, chunking, embedding, and storage.
"""

from pathlib import Path
from typing import List, Optional

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from ..chunking import CodeChunker
from ..config import ConfigManager, EmbeddingConfig
from ..embeddings import EmbeddingsFactory
from ..size_validator import ProjectSizeValidator, ProjectSizeLimitError
from ..traversal import CodeTraversal
from ..utils import get_ctxai_home, get_indexes_dir, is_using_global_home
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

    # Setup .ctxai directory and load configuration
    ctxai_home = get_ctxai_home(path)
    
    # Show where .ctxai is located
    if is_using_global_home():
        console.print(f"[dim]Using global CTXAI_HOME: {ctxai_home}[/dim]")
    else:
        console.print(f"[dim]Using project .ctxai: {ctxai_home}[/dim]")
    
    config_manager = ConfigManager(path)
    config = config_manager.load()
    
    console.print("[cyan]Initializing components...[/cyan]")
    
    # Show embedding provider info
    embedding_config = config.embedding
    console.print(f"[dim]Embedding provider: {embedding_config.provider}[/dim]")
    if embedding_config.model:
        console.print(f"[dim]Model: {embedding_config.model}[/dim]")
    
    # Initialize components
    traversal = CodeTraversal(
        root_path=path,
        include_patterns=include_patterns,
        exclude_patterns=exclude_patterns,
        follow_gitignore=follow_gitignore,
    )
    
    index_config = config.indexing
    chunker = CodeChunker(
        max_chunk_size=index_config.chunk_size,
        overlap=index_config.chunk_overlap,
    )
    
    # Initialize embedding provider
    try:
        embeddings_generator = EmbeddingsFactory.create(embedding_config)
        console.print(f"[green]✓[/green] Embedding provider '{embedding_config.provider}' initialized\n")
    except ImportError as e:
        console.print(f"[red]✗[/red] Error: {e}\n")
        console.print(
            f"[yellow]Tip:[/yellow] Install the required package for '{embedding_config.provider}' provider\n"
        )
        return
    except ValueError as e:
        console.print(f"[red]✗[/red] Error: {e}\n")
        if embedding_config.provider == "openai":
            console.print(
                "[yellow]Tip:[/yellow] Set your OpenAI API key: "
                "[cyan]export OPENAI_API_KEY=your-key-here[/cyan]\n"
            )
            console.print(
                "Or switch to local embeddings by editing [cyan].ctxai/config.json[/cyan]:\n"
                '  "embedding": {"provider": "local"}\n'
            )
        return

    # Storage path in .ctxai directory (respects CTXAI_HOME)
    indexes_dir = get_indexes_dir(path)
    storage_path = indexes_dir / index_name
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
    
    # Validate project size
    console.print("[bold cyan]Validating project size...[/bold cyan]")
    validator = ProjectSizeValidator(index_config)
    stats = validator.analyze_files(files_to_process)
    
    # Show summary
    for line in validator.get_summary(stats):
        console.print(f"[dim]{line}[/dim]")
    console.print()
    
    # Check limits
    is_valid, messages = validator.validate(stats)
    if messages:
        for message in messages:
            console.print(message)
        console.print()
    
    if not is_valid:
        console.print(
            "[red]❌ Project exceeds size limits. "
            "Please reduce the project size or adjust limits in .ctxai/config.json[/red]\n"
        )
        raise ProjectSizeLimitError(messages)
    
    # Filter out oversized files
    if stats.oversized_files:
        oversized_set = {f[0] for f in stats.oversized_files}
        files_to_process = [f for f in files_to_process if f not in oversized_set]
        console.print(f"[yellow]⚠[/yellow] Skipped {len(oversized_set)} oversized file(s)\n")

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
