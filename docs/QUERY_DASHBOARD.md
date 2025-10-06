# Query and Dashboard Commands

This document describes the query and dashboard commands added to ctxai for interactive code search and management.

## Query Command

The `query` command allows you to search your indexed codebase using natural language queries.

### Basic Usage

```bash
ctxai query <index_name> "<query_text>"
```

### Examples

```bash
# Find authentication-related code
ctxai query my-project "Find functions that handle user authentication"

# Search for database operations
ctxai query my-project "How to connect to database"

# Look for error handling
ctxai query my-project "Find error handling and exception management code"

# Limit results
ctxai query my-project "Find API endpoints" --n-results 3

# Show only metadata (no code preview)
ctxai query my-project "Find configuration files" --no-content
```

### Options

- `index_name` (required): Name of the index to query
- `query_text` (required): Natural language query
- `--n-results, -n`: Number of results to return (default: 5)
- `--no-content`: Don't show code content, only metadata

### Output

The query command displays:

1. **Search Status**: Shows which index is being searched and the query
2. **Embedding Provider**: Displays which provider (Local/OpenAI/HuggingFace) is being used
3. **Results**: For each result:
   - **File Path**: Location of the code chunk
   - **Line Numbers**: Start and end lines
   - **Chunk Type**: Function, class, method, etc.
   - **Language**: Programming language detected
   - **Similarity Score**: How closely the chunk matches your query (0-100%)
   - **Code Preview**: Syntax-highlighted code (unless --no-content is used)

### How It Works

1. **Load Configuration**: Reads `.ctxai/config.json` to determine embedding provider
2. **Generate Query Embedding**: Converts your natural language query into a vector
3. **Search Vector Database**: Finds the most similar code chunks using cosine similarity
4. **Format Results**: Displays results with syntax highlighting and metadata

### Tips

- Be specific in your queries for better results
- Use natural language - describe what you're looking for
- Try different phrasings if results aren't relevant
- Use `--n-results` to see more options
- Local embeddings work offline but OpenAI may provide better quality

## Dashboard Command

The `dashboard` command starts a web-based interface for managing and querying your indexes.

### Basic Usage

```bash
ctxai dashboard
```

### Examples

```bash
# Start on default port (3000)
ctxai dashboard

# Use custom port
ctxai dashboard --port 8080

# Then open browser to http://localhost:3000 (or your custom port)
```

### Options

- `--port, -p`: Port to run the dashboard on (default: 3000)

### Features

#### 1. Home Page
- **View All Indexes**: See all your indexes with statistics
  - Chunk count
  - Total size (MB)
  - Creation timestamp
- **Quick Actions**: 
  - View index details
  - Query specific index
- **Configuration Info**: Shows CTXAI_HOME location and settings

#### 2. Index Detail Page
- **Statistics**: Comprehensive index information
  - Total number of chunks
  - Storage location
  - Index name
- **Browse Chunks**: View all code chunks with:
  - File names and paths
  - Languages and chunk types
  - Line numbers
  - Character counts
- **Quick Query**: Jump to query interface for this index

#### 3. Query Interface
- **Index Selection**: Choose which index to search
- **Natural Language Search**: Enter your query in plain English
- **Result Configuration**: Set number of results (1-20)
- **Interactive Results**: View:
  - Similarity scores with percentage
  - File paths and line numbers
  - Chunk metadata (type, language)
  - Full code content with formatting

#### 4. Settings Page
- **CTXAI Home Info**: 
  - Current home directory location
  - Location type (global/project)
  - Indexes directory path
- **Configuration View**: 
  - Current embedding provider and model
  - Index limits and settings
  - JSON formatted for easy reading

### User Interface

The dashboard features a modern, dark-themed UI with:
- 🎨 **Beautiful Design**: Gradient headers and smooth animations
- 📱 **Responsive Layout**: Works on desktop and mobile
- 🌙 **Dark Theme**: Easy on the eyes for long sessions
- ⚡ **Fast Navigation**: Quick links between pages
- 💻 **Code Highlighting**: Syntax-highlighted code blocks

### Requirements

The dashboard requires FastHTML to be installed:

```bash
# Install dashboard dependencies
pip install ctxai[dashboard]

# Or install all optional dependencies
pip install ctxai[all]
```

If FastHTML is not installed, the dashboard command will show an error with installation instructions.

### Architecture

The dashboard is built with:
- **FastHTML**: Lightweight Python web framework
- **No JavaScript Required**: Pure Python server-side rendering
- **ChromaDB Integration**: Direct access to vector database
- **Rich Formatting**: Consistent styling with terminal output

### Tips

- Leave the dashboard running while you work on your code
- Use it to explore your codebase visually
- Great for demos and presentations
- Access from any device on your network (use 0.0.0.0 binding if needed)
- Query multiple indexes without switching contexts

## Comparison: CLI vs Dashboard

| Feature | CLI Query | Dashboard |
|---------|-----------|-----------|
| **Speed** | ⚡ Instant | 🌐 Web-based |
| **Interface** | Terminal | Browser |
| **Best For** | Quick lookups | Exploration, demos |
| **Results** | Text output | Interactive UI |
| **Multiple Queries** | New command each time | Stay in browser |
| **Index Browsing** | No | Yes |
| **Configuration View** | No | Yes |
| **Installation** | Default | Requires FastHTML |

## Workflow Examples

### Development Workflow

```bash
# 1. Index your project
ctxai index ./src my-app

# 2. Quick CLI queries during development
ctxai query my-app "Find the login function"
ctxai query my-app "How is error logging implemented"

# 3. Detailed exploration with dashboard
ctxai dashboard
# Then open http://localhost:3000 and browse interactively
```

### Team Collaboration

```bash
# 1. Set up shared CTXAI_HOME
export CTXAI_HOME=~/shared/.ctxai

# 2. Index team's codebase
ctxai index /path/to/repo team-project

# 3. Start dashboard for team demos
ctxai dashboard --port 8080

# Team members can now:
# - Query the codebase visually
# - Explore code organization
# - Find examples and patterns
```

### Code Review

```bash
# 1. Index the feature branch
ctxai index ./feature-branch feature-xyz

# 2. Use dashboard to review changes
ctxai dashboard

# 3. Query specific patterns
# "Find new API endpoints"
# "Show authentication changes"
# "Find error handling additions"
```

## Troubleshooting

### Query Issues

**"Index not found"**
- Solution: Run `ctxai index` first to create the index
- Check CTXAI_HOME if using custom location

**"No results found"**
- Solution: Try different query phrasing
- Check that files were actually indexed
- Verify embedding provider is working

**Slow queries**
- Solution: Local embeddings may be slower than cloud
- Consider using OpenAI provider for speed
- Reduce n_results for faster responses

### Dashboard Issues

**"FastHTML is not installed"**
- Solution: `pip install ctxai[dashboard]` or `pip install python-fasthtml`

**Port already in use**
- Solution: Use `--port` to specify different port
- Or stop the existing service on that port

**Cannot access from other devices**
- Solution: Dashboard only binds to localhost by default
- Need to modify code to bind to 0.0.0.0 for network access

## Advanced Usage

### Python API

You can also use the query and dashboard functions programmatically:

```python
from pathlib import Path
from ctxai.commands.query_command import query_codebase
from ctxai.commands.dashboard_command import start_dashboard

# Query from Python
query_codebase(
    index_name="my-project",
    query="Find authentication code",
    n_results=5,
    show_content=True
)

# Start dashboard from Python
start_dashboard(
    port=3000,
    project_path=Path("/path/to/project")
)
```

### Custom Integration

The query function returns structured results that you can process:

```python
from ctxai.config import ConfigManager
from ctxai.embeddings import EmbeddingsFactory
from ctxai.vector_store import VectorStore
from ctxai.utils import get_indexes_dir

# Load configuration
config_manager = ConfigManager()
config = config_manager.load()

# Create embedding provider
embeddings = EmbeddingsFactory.create(config.embedding)

# Generate query embedding
query_embedding = embeddings.generate_embedding("your query")

# Search
index_path = get_indexes_dir() / "my-index"
vector_store = VectorStore(storage_path=index_path, collection_name="my-index")
results = vector_store.search(query_embedding=query_embedding, n_results=5)

# Process results
for result in results:
    print(f"File: {result['metadata']['file_path']}")
    print(f"Similarity: {1 - result['distance']:.2%}")
    print(f"Content: {result['content'][:100]}...")
```

## Future Enhancements

Planned improvements:

- [ ] **Query Filters**: Filter by language, file type, date
- [ ] **Query History**: Track and replay previous queries
- [ ] **Saved Queries**: Save frequently used queries
- [ ] **Export Results**: Export query results to file
- [ ] **Dashboard Auth**: Password protection for dashboard
- [ ] **Multi-Index Search**: Query across multiple indexes
- [ ] **Query Suggestions**: Auto-suggest query improvements
- [ ] **Result Ranking**: Customize ranking algorithms
- [ ] **Dashboard Indexing**: Trigger indexing from dashboard UI
- [ ] **Real-time Updates**: Auto-refresh when indexes change

## See Also

- [CONFIGURATION.md](./CONFIGURATION.md) - Configuration options
- [CTXAI_HOME.md](./CTXAI_HOME.md) - CTXAI_HOME environment variable
- [README.md](../README.md) - Main documentation
- [examples/](../examples/) - Code examples
