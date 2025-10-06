"""
Dashboard command implementation.
Web-based interface for managing indexes and querying codebase.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

try:
    from fasthtml.common import (
        FastHTML,
        Body,
        Head,
        Html,
        Title,
        Div,
        H1,
        H2,
        H3,
        P,
        Form,
        Input,
        Button,
        Table,
        Thead,
        Tbody,
        Tr,
        Th,
        Td,
        A,
        Script,
        Style,
        serve,
        Link,
        Label,
        Textarea,
        Pre,
        Code,
        Select,
        Option,
    )
    FASTHTML_AVAILABLE = True
except ImportError:
    FASTHTML_AVAILABLE = False

from rich.console import Console

from ..config import ConfigManager
from ..embeddings import EmbeddingsFactory
from ..utils import get_indexes_dir, get_ctxai_home, get_ctxai_home_info
from ..vector_store import VectorStore

console = Console()


def start_dashboard(port: int = 3000, project_path: Optional[Path] = None):
    """
    Start the FastHTML dashboard server.

    Args:
        port: Port to run the dashboard on
        project_path: Optional project path (uses CTXAI_HOME if not provided)
    """
    if not FASTHTML_AVAILABLE:
        console.print("[red]✗ FastHTML is not installed[/red]\n")
        console.print(
            "[yellow]Install it with:[/yellow] [cyan]pip install python-fasthtml[/cyan]\n"
        )
        console.print(
            "[yellow]Or install all optional dependencies:[/yellow] [cyan]pip install ctxai[all][/cyan]\n"
        )
        return

    console.print(f"[bold blue]🚀 Starting dashboard on port {port}...[/bold blue]\n")

    # Initialize FastHTML app
    app = FastHTML()

    # Get CTXAI home info
    ctxai_home = get_ctxai_home(project_path)
    indexes_dir = get_indexes_dir(project_path)
    home_info = get_ctxai_home_info(project_path)

    # Styles
    app_styles = Style("""
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            background: #0f172a;
            color: #e2e8f0;
            line-height: 1.6;
        }
        .container { 
            max-width: 1200px; 
            margin: 0 auto; 
            padding: 2rem;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 2rem;
            border-radius: 1rem;
            margin-bottom: 2rem;
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        }
        .header h1 { 
            font-size: 2.5rem; 
            margin-bottom: 0.5rem;
            color: white;
        }
        .header p { 
            color: rgba(255,255,255,0.9); 
            font-size: 1.1rem;
        }
        .card {
            background: #1e293b;
            border-radius: 0.75rem;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 4px 6px rgba(0,0,0,0.2);
            border: 1px solid #334155;
        }
        .card h2 { 
            margin-bottom: 1rem; 
            color: #60a5fa;
            font-size: 1.5rem;
        }
        .card h3 { 
            margin: 1rem 0 0.5rem 0; 
            color: #818cf8;
        }
        .info-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin-bottom: 1rem;
        }
        .info-item {
            background: #0f172a;
            padding: 1rem;
            border-radius: 0.5rem;
            border-left: 4px solid #60a5fa;
        }
        .info-item strong {
            color: #94a3b8;
            display: block;
            margin-bottom: 0.25rem;
            font-size: 0.9rem;
        }
        .info-item span {
            color: #e2e8f0;
            font-size: 1.1rem;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            background: #0f172a;
            border-radius: 0.5rem;
            overflow: hidden;
        }
        th {
            background: #334155;
            padding: 1rem;
            text-align: left;
            color: #94a3b8;
            font-weight: 600;
            text-transform: uppercase;
            font-size: 0.85rem;
            letter-spacing: 0.05em;
        }
        td {
            padding: 1rem;
            border-top: 1px solid #334155;
        }
        tr:hover {
            background: #1e293b;
        }
        .form-group {
            margin-bottom: 1rem;
        }
        label {
            display: block;
            margin-bottom: 0.5rem;
            color: #94a3b8;
            font-weight: 500;
        }
        input, select, textarea {
            width: 100%;
            padding: 0.75rem;
            background: #0f172a;
            border: 1px solid #334155;
            border-radius: 0.5rem;
            color: #e2e8f0;
            font-size: 1rem;
            transition: all 0.2s;
        }
        input:focus, select:focus, textarea:focus {
            outline: none;
            border-color: #60a5fa;
            box-shadow: 0 0 0 3px rgba(96, 165, 250, 0.1);
        }
        textarea {
            min-height: 100px;
            font-family: monospace;
        }
        button, .btn {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 0.75rem 1.5rem;
            border: none;
            border-radius: 0.5rem;
            cursor: pointer;
            font-size: 1rem;
            font-weight: 600;
            transition: transform 0.2s, box-shadow 0.2s;
            display: inline-block;
            text-decoration: none;
        }
        button:hover, .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 20px rgba(102, 126, 234, 0.4);
        }
        .btn-secondary {
            background: #334155;
        }
        .btn-secondary:hover {
            background: #475569;
            box-shadow: 0 8px 20px rgba(0,0,0,0.3);
        }
        .code-block {
            background: #0f172a;
            padding: 1rem;
            border-radius: 0.5rem;
            overflow-x: auto;
            border: 1px solid #334155;
            margin: 0.5rem 0;
        }
        .code-block code {
            color: #e2e8f0;
            font-family: 'Courier New', monospace;
            font-size: 0.9rem;
        }
        .badge {
            display: inline-block;
            padding: 0.25rem 0.75rem;
            background: #334155;
            border-radius: 1rem;
            font-size: 0.85rem;
            color: #94a3b8;
        }
        .badge-success {
            background: #059669;
            color: white;
        }
        .badge-info {
            background: #0284c7;
            color: white;
        }
        .nav {
            display: flex;
            gap: 1rem;
            margin-bottom: 2rem;
        }
        .nav a {
            color: #94a3b8;
            text-decoration: none;
            padding: 0.5rem 1rem;
            border-radius: 0.5rem;
            transition: all 0.2s;
        }
        .nav a:hover {
            background: #1e293b;
            color: #60a5fa;
        }
        .result-card {
            background: #0f172a;
            padding: 1rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
            border-left: 4px solid #60a5fa;
        }
        .result-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 0.5rem;
        }
        .result-title {
            color: #60a5fa;
            font-weight: 600;
        }
        .result-similarity {
            background: #059669;
            color: white;
            padding: 0.25rem 0.75rem;
            border-radius: 1rem;
            font-size: 0.85rem;
        }
        .result-meta {
            display: flex;
            gap: 1rem;
            margin-bottom: 0.5rem;
            font-size: 0.9rem;
            color: #94a3b8;
        }
    """)

    @app.get("/")
    def home():
        """Home page showing all indexes."""
        # Get all indexes
        indexes = []
        if indexes_dir.exists():
            for index_path in indexes_dir.iterdir():
                if index_path.is_dir():
                    try:
                        # Get vector store stats
                        vector_store = VectorStore(
                            storage_path=index_path, collection_name=index_path.name
                        )
                        stats = vector_store.get_stats()

                        # Get creation time
                        creation_time = datetime.fromtimestamp(
                            index_path.stat().st_ctime
                        ).strftime("%Y-%m-%d %H:%M:%S")

                        indexes.append(
                            {
                                "name": index_path.name,
                                "path": str(index_path),
                                "chunks": stats["total_chunks"],
                                "created": creation_time,
                                "size_mb": sum(
                                    f.stat().st_size for f in index_path.rglob("*") if f.is_file()
                                ) / (1024 * 1024),
                            }
                        )
                    except Exception as e:
                        console.print(f"[yellow]Warning: Could not load index {index_path.name}: {e}[/yellow]")

        # Build index table
        if indexes:
            index_rows = [
                Tr(
                    Td(A(idx["name"], href=f"/index/{idx['name']}")),
                    Td(f"{idx['chunks']:,}"),
                    Td(f"{idx['size_mb']:.2f} MB"),
                    Td(idx["created"]),
                    Td(
                        A("View", href=f"/index/{idx['name']}", cls="btn btn-secondary", style="padding: 0.5rem 1rem;"),
                        " ",
                        A("Query", href=f"/query?index={idx['name']}", cls="btn", style="padding: 0.5rem 1rem;"),
                    ),
                )
                for idx in indexes
            ]
            index_table = Table(
                Thead(Tr(Th("Index Name"), Th("Chunks"), Th("Size"), Th("Created"), Th("Actions"))),
                Tbody(*index_rows),
            )
        else:
            index_table = P("No indexes found. Create one using the 'ctxai index' command.", style="color: #94a3b8;")

        return Html(
            Head(Title("CTXAI Dashboard"), app_styles),
            Body(
                Div(
                    # Header
                    Div(
                        H1("🤖 CTXAI Dashboard"),
                        P("Semantic Code Search Engine"),
                        cls="header",
                    ),
                    # Navigation
                    Div(
                        A("Home", href="/"),
                        A("Query", href="/query"),
                        A("Settings", href="/settings"),
                        cls="nav",
                    ),
                    # CTXAI Home Info
                    Div(
                        H2("📁 Configuration"),
                        Div(
                            Div(Div(P("Home Directory"), P(str(ctxai_home))), cls="info-item"),
                            Div(Div(P("Location Type"), P(home_info["location_type"])), cls="info-item"),
                            Div(Div(P("Total Indexes"), P(str(len(indexes)))), cls="info-item"),
                            cls="info-grid",
                        ),
                        cls="card",
                    ),
                    # Indexes
                    Div(H2("📊 Indexes"), index_table, cls="card"),
                    cls="container",
                ),
            ),
        )

    @app.get("/index/{name}")
    def view_index(name: str):
        """View details of a specific index."""
        index_path = indexes_dir / name

        if not index_path.exists():
            return Html(
                Head(Title(f"Index Not Found - CTXAI"), app_styles),
                Body(
                    Div(
                        Div(H1("❌ Index Not Found"), P(f"Index '{name}' does not exist."), cls="header"),
                        Div(A("← Back to Home", href="/", cls="btn")),
                        cls="container",
                    )
                ),
            )

        try:
            # Get vector store stats
            vector_store = VectorStore(storage_path=index_path, collection_name=name)
            stats = vector_store.get_stats()

            # Get all chunks (limited to first 100 for display)
            # This is a simplified approach - in production you'd want pagination
            results = vector_store.collection.get(limit=100)

            chunk_rows = []
            if results and results["ids"]:
                for i, (chunk_id, metadata, content) in enumerate(
                    zip(results["ids"], results["metadatas"], results["documents"])
                ):
                    file_path = Path(metadata.get("file_path", "Unknown"))
                    chunk_rows.append(
                        Tr(
                            Td(str(i + 1)),
                            Td(file_path.name, title=str(file_path)),
                            Td(metadata.get("language", "Unknown")),
                            Td(metadata.get("chunk_type", "Unknown")),
                            Td(f"{metadata.get('start_line', 0)}-{metadata.get('end_line', 0)}"),
                            Td(f"{len(content):,} chars"),
                        )
                    )

            chunk_table = Table(
                Thead(Tr(Th("#"), Th("File"), Th("Language"), Th("Type"), Th("Lines"), Th("Size"))),
                Tbody(*chunk_rows) if chunk_rows else Tbody(Tr(Td("No chunks found", colspan="6"))),
            )

        except Exception as e:
            return Html(
                Head(Title(f"Error - CTXAI"), app_styles),
                Body(
                    Div(
                        Div(H1("❌ Error"), P(f"Error loading index: {e}"), cls="header"),
                        Div(A("← Back to Home", href="/", cls="btn")),
                        cls="container",
                    )
                ),
            )

        return Html(
            Head(Title(f"{name} - CTXAI Dashboard"), app_styles),
            Body(
                Div(
                    Div(H1(f"📊 Index: {name}"), P(f"Total chunks: {stats['total_chunks']:,}"), cls="header"),
                    Div(A("← Back to Home", href="/", cls="btn"), A("Query This Index", href=f"/query?index={name}", cls="btn")),
                    Div(
                        H2("📈 Statistics"),
                        Div(
                            Div(Div(P("Total Chunks"), P(f"{stats['total_chunks']:,}")), cls="info-item"),
                            Div(Div(P("Index Name"), P(name)), cls="info-item"),
                            Div(Div(P("Storage Path"), P(str(index_path))), cls="info-item"),
                            cls="info-grid",
                        ),
                        cls="card",
                    ),
                    Div(
                        H2("📄 Chunks (First 100)"),
                        chunk_table,
                        cls="card",
                    ),
                    cls="container",
                ),
            ),
        )

    @app.get("/query")
    def query_page(index: Optional[str] = None):
        """Query interface."""
        # Get all indexes for dropdown
        index_options = []
        if indexes_dir.exists():
            for index_path in indexes_dir.iterdir():
                if index_path.is_dir():
                    selected = index_path.name == index if index else False
                    index_options.append(Option(index_path.name, value=index_path.name, selected=selected))

        return Html(
            Head(Title("Query - CTXAI Dashboard"), app_styles),
            Body(
                Div(
                    Div(H1("🔍 Query Codebase"), P("Search using natural language"), cls="header"),
                    Div(A("← Back to Home", href="/", cls="btn")),
                    Div(
                        H2("Search"),
                        Form(
                            Div(
                                Label("Index", **{"for": "index"}),
                                Select(*index_options, id="index", name="index", required=True),
                                cls="form-group",
                            ),
                            Div(
                                Label("Query", **{"for": "query"}),
                                Textarea(
                                    placeholder="e.g., Find authentication functions",
                                    id="query",
                                    name="query",
                                    required=True,
                                ),
                                cls="form-group",
                            ),
                            Div(
                                Label("Number of Results", **{"for": "n_results"}),
                                Input(type="number", id="n_results", name="n_results", value="5", min="1", max="20"),
                                cls="form-group",
                            ),
                            Button("🔍 Search", type="submit"),
                            action="/query/search",
                            method="post",
                        ),
                        cls="card",
                    ),
                    cls="container",
                ),
            ),
        )

    @app.post("/query/search")
    def query_search(index: str, query: str, n_results: int = 5):
        """Execute query and show results."""
        try:
            # Load configuration and create embedding provider
            config_manager = ConfigManager(project_path)
            config = config_manager.load()

            # Initialize embedding provider
            embeddings_generator = EmbeddingsFactory.create(config.embedding)

            # Load vector store
            index_path = indexes_dir / index
            if not index_path.exists():
                raise ValueError(f"Index '{index}' not found")

            vector_store = VectorStore(storage_path=index_path, collection_name=index)

            # Generate query embedding and search
            query_embedding = embeddings_generator.generate_embedding(query)
            results = vector_store.search(query_embedding=query_embedding, n_results=n_results)

            # Build result cards
            result_cards = []
            for i, result in enumerate(results, 1):
                metadata = result["metadata"]
                content = result["content"]
                distance = result["distance"]
                similarity = max(0, 1 - distance)

                file_path = Path(metadata["file_path"])

                # Truncate content for display
                display_content = content
                if len(content) > 500:
                    display_content = content[:500] + "\n... (truncated)"

                result_cards.append(
                    Div(
                        Div(
                            Div(f"{i}. {file_path.name}", cls="result-title"),
                            Div(f"{similarity:.0%}", cls="result-similarity"),
                            cls="result-header",
                        ),
                        Div(
                            P(f"📁 {file_path}"),
                            P(f"📍 Lines {metadata['start_line']}-{metadata['end_line']}"),
                            P(f"🏷️ {metadata['chunk_type']} ({metadata['language']})"),
                            cls="result-meta",
                        ),
                        Div(Pre(Code(display_content)), cls="code-block"),
                        cls="result-card",
                    )
                )

        except Exception as e:
            return Html(
                Head(Title("Query Error - CTXAI Dashboard"), app_styles),
                Body(
                    Div(
                        Div(H1("❌ Query Error"), P(f"Error executing query: {e}"), cls="header"),
                        Div(A("← Back to Query", href="/query", cls="btn")),
                        cls="container",
                    )
                ),
            )

        return Html(
            Head(Title("Query Results - CTXAI Dashboard"), app_styles),
            Body(
                Div(
                    Div(H1("🔍 Query Results"), P(f"Found {len(results)} result(s) for: {query}"), cls="header"),
                    Div(A("← New Query", href="/query", cls="btn")),
                    Div(H2(f"Results from '{index}'"), *result_cards, cls="card"),
                    cls="container",
                ),
            ),
        )

    @app.get("/settings")
    def settings_page():
        """Settings page."""
        try:
            config_manager = ConfigManager(project_path)
            config = config_manager.load()

            config_json = json.dumps(
                {
                    "embedding": {
                        "provider": config.embedding.provider,
                        "model": config.embedding.model,
                        "dimension": config.embedding.dimension,
                    },
                    "index": {
                        "max_files": config.index.max_files,
                        "max_size_mb": config.index.max_size_mb,
                        "chunk_size": config.index.chunk_size,
                        "chunk_overlap": config.index.chunk_overlap,
                    },
                },
                indent=2,
            )
        except Exception as e:
            config_json = f"Error loading configuration: {e}"

        return Html(
            Head(Title("Settings - CTXAI Dashboard"), app_styles),
            Body(
                Div(
                    Div(H1("⚙️ Settings"), P("Configuration and environment"), cls="header"),
                    Div(A("← Back to Home", href="/", cls="btn")),
                    Div(
                        H2("📁 CTXAI Home"),
                        Div(
                            Div(Div(P("Home Directory"), P(str(ctxai_home))), cls="info-item"),
                            Div(Div(P("Location Type"), P(home_info["location_type"])), cls="info-item"),
                            Div(Div(P("Indexes Directory"), P(str(indexes_dir))), cls="info-item"),
                            cls="info-grid",
                        ),
                        cls="card",
                    ),
                    Div(
                        H2("🔧 Configuration"),
                        P("Current configuration from .ctxai/config.json", style="color: #94a3b8; margin-bottom: 1rem;"),
                        Div(Pre(Code(config_json)), cls="code-block"),
                        cls="card",
                    ),
                    cls="container",
                ),
            ),
        )

    # Start server
    console.print(f"[green]✓ Dashboard started successfully![/green]")
    console.print(f"[cyan]Open in browser: http://localhost:{port}[/cyan]\n")
    console.print("[dim]Press Ctrl+C to stop the server[/dim]\n")

    serve(app=app, port=port)
