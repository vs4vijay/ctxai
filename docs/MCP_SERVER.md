# MCP Server Documentation

The ctxai MCP (Model Context Protocol) server exposes code indexing and querying functionality as tools that can be used by LLMs and AI agents.

## Overview

The MCP server allows AI agents to:
- List available code indexes
- Index new codebases
- Query code using natural language
- Get index statistics

This enables LLMs to autonomously search and understand your codebase through semantic search.

**Built with FastMCP**: The server uses the FastMCP framework for automatic tool registration and type-safe parameter handling.

## Installation

```bash
# Install with MCP support
pip install ctxai[mcp]

# Or install all features
pip install ctxai[all]
```

## Starting the Server

```bash
# Start MCP server (uses CTXAI_HOME or current directory)
ctxai server

# Use specific project path
ctxai server --project-path /path/to/project
```

The server communicates via stdio (standard input/output) and is designed to be used with MCP-compatible clients.

## MCP Client Configuration

### Claude Desktop

Add to your Claude Desktop configuration file:

**macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
**Windows**: `%APPDATA%\Claude\claude_desktop_config.json`

```json
{
  "mcpServers": {
    "ctxai": {
      "command": "ctxai",
      "args": ["server"]
    }
  }
}
```

With custom project path:

```json
{
  "mcpServers": {
    "ctxai": {
      "command": "ctxai",
      "args": ["server", "--project-path", "/path/to/project"]
    }
  }
}
```

### Other MCP Clients

Any MCP-compatible client can connect to ctxai by running:

```bash
ctxai server
```

The server will communicate via stdin/stdout following the MCP protocol.

## Available Tools

### 1. list_indexes

Lists all available code indexes with their statistics.

**Parameters**: None

**Returns**: List of indexes with chunk counts and paths

**Example usage in Claude**:
```
Can you list all the code indexes available?
```

### 2. index_codebase

Indexes a codebase for semantic search.

**Parameters**:
- `path` (required): Path to the codebase directory
- `name` (required): Name for the index
- `include_patterns` (optional): Array of file patterns to include (e.g., ["*.py", "*.js"])
- `exclude_patterns` (optional): Array of patterns to exclude beyond .gitignore
- `follow_gitignore` (optional): Whether to follow .gitignore (default: true)

**Returns**: Success message with index statistics

**Example usage in Claude**:
```
Please index the codebase at /Users/me/projects/myapp with the name "myapp"
```

### 3. query_codebase

Queries an indexed codebase using natural language.

**Parameters**:
- `index_name` (required): Name of the index to query
- `query` (required): Natural language query
- `n_results` (optional): Number of results to return (default: 5, max: 20)

**Returns**: Relevant code chunks with metadata and similarity scores

**Example usage in Claude**:
```
Search the "myapp" index for authentication functions
```

### 4. get_index_stats

Gets detailed statistics about a specific index.

**Parameters**:
- `index_name` (required): Name of the index

**Returns**: Index statistics including chunk count and storage size

**Example usage in Claude**:
```
Show me the statistics for the "myapp" index
```

## Usage Examples

### Example 1: Initial Setup

```
User: Can you help me search my codebase?

Claude: I'd be happy to help! First, let me see what indexes are available.
[Uses list_indexes tool]

Claude: I don't see any indexes yet. Would you like me to index your codebase? 
What's the path to your project?

User: /Users/me/projects/myapp

Claude: Great! Let me index that for you.
[Uses index_codebase tool with path="/Users/me/projects/myapp" and name="myapp"]

Claude: ✓ Successfully indexed your codebase! It has 1,234 code chunks.
Now you can ask me questions about your code.
```

### Example 2: Querying Code

```
User: Find all the database connection code

Claude: Let me search for that.
[Uses query_codebase tool with index_name="myapp" and query="database connection code"]

Claude: I found 3 relevant results:

1. **database.py** (Lines 10-45) - 89% similarity
   This file contains the main database connection class...
   [Shows code]

2. **config.py** (Lines 67-82) - 76% similarity
   Database configuration settings...
   [Shows code]

3. **models/__init__.py** (Lines 5-15) - 72% similarity
   Database initialization...
   [Shows code]
```

### Example 3: Multiple Projects

```
User: I have two projects I want to search. Can you help?

Claude: Absolutely! Let me index both projects for you.
What are the paths?

User: /Users/me/frontend and /Users/me/backend

Claude: [Uses index_codebase twice]
✓ Indexed frontend (523 chunks)
✓ Indexed backend (891 chunks)

You can now ask me about either project. Just mention which one!

User: Find authentication code in the backend

Claude: [Uses query_codebase with index_name="backend"]
Here's the authentication code I found in the backend project...
```

## How It Works

1. **Server Startup**: The FastMCP server starts and listens on stdio
2. **Tool Discovery**: Client discovers available tools via MCP protocol (automatic with FastMCP)
3. **Tool Execution**: LLM decides when to use tools based on user queries
4. **Async Processing**: Long-running operations (indexing, embedding generation) run asynchronously
5. **Results**: Formatted results are returned to the LLM for interpretation

**Key Features**:
- ✅ **FastMCP Framework**: Automatic tool registration using decorators
- ✅ **Type Hints**: Function signatures automatically generate tool schemas
- ✅ **Async/Await**: Non-blocking operations for better performance
- ✅ **Proper Logging**: Uses stderr to avoid corrupting stdio communication
- ✅ **Error Handling**: Graceful failures with informative messages

## Architecture

```
┌─────────────┐
│   LLM/AI    │
│   Client    │
│  (Claude)   │
└──────┬──────┘
       │ MCP Protocol (stdio)
       │
┌──────▼──────┐
│  FastMCP    │
│   Server    │
│   (ctxai)   │
└──────┬──────┘
       │
       ├─► @mcp.tool() list_indexes()
       ├─► @mcp.tool() index_codebase()
       ├─► @mcp.tool() query_codebase()
       └─► @mcp.tool() get_index_stats()
              │
              ▼
       ┌──────────────┐
       │  Vector DB   │
       │  (ChromaDB)  │
       └──────────────┘
```

## Logging Best Practices

The ctxai MCP server follows stdio logging best practices:

**✅ Good (Uses stderr)**:
```python
import logging
logging.basicConfig(
    level=logging.INFO,
    handlers=[logging.StreamHandler()]  # Uses stderr by default
)
logger.info("Processing request")

# Rich console also uses stderr
console = Console(stderr=True)
console.print("[blue]Status message[/blue]")
```

**❌ Bad (Would corrupt stdio)**:
```python
print("Processing request")  # DON'T DO THIS!
```

**Why**: STDIO-based MCP servers communicate via stdin/stdout. Any output to stdout will corrupt the JSON-RPC messages and break the server.

## Configuration

The MCP server uses the same configuration as the CLI:

- **CTXAI_HOME**: Controls where indexes are stored
- **.ctxai/config.json**: Embedding provider settings
- **Embedding Providers**: Local (default), OpenAI, HuggingFace

See [CONFIGURATION.md](./CONFIGURATION.md) for details.

## Limitations

1. **Stdio Communication**: Server runs in stdio mode (not HTTP)
2. **Single Client**: One client connection at a time
3. **No Streaming**: Results are returned all at once (not streamed)
4. **Synchronous Indexing**: Large codebases may take time to index
5. **Memory Usage**: Large indexes are loaded into memory

## Troubleshooting

### MCP Not Installed

**Error**: `MCP is not installed`

**Solution**: 
```bash
pip install ctxai[mcp]
```

### Server Not Starting

**Error**: Server hangs or doesn't respond

**Solutions**:
- Check that stdio is not being redirected
- Ensure no other process is using the server
- Verify MCP client configuration is correct

### Index Not Found

**Error**: `Index 'name' not found`

**Solutions**:
- Use `list_indexes` tool to see available indexes
- Index the codebase first with `index_codebase`
- Check CTXAI_HOME if using custom location

### Slow Queries

**Issue**: Queries take a long time

**Solutions**:
- Use local embeddings for faster responses (no API calls)
- Reduce `n_results` parameter
- Consider using OpenAI for better performance (but costs money)

### Memory Issues

**Issue**: Server crashes or uses too much memory

**Solutions**:
- Index smaller codebases
- Use project size limits in config
- Close and restart server periodically

## Security Considerations

### Local Execution
- Server runs locally (not exposed to network)
- Only accessible via stdio from parent process
- No authentication needed (process-level isolation)

### File System Access
- Server can read/write to CTXAI_HOME
- Can index any directory the user has access to
- No sandboxing (runs with user's permissions)

### API Keys
- Embedding provider API keys stored in config
- Not transmitted to MCP client
- Use environment variables for sensitive keys

**Best Practices**:
- Don't index sensitive codebases in shared CTXAI_HOME
- Use separate indexes for different security levels
- Rotate API keys regularly
- Review index contents before sharing

## Advanced Usage

### Custom Python API

You can use the MCP server programmatically:

```python
import asyncio
from pathlib import Path
from ctxai.mcp_server import run_mcp_server

# Run server with custom project path
asyncio.run(run_mcp_server(project_path=Path("/path/to/project")))
```

### Tool Integration

Use the individual handlers directly:

```python
from ctxai.mcp_server import (
    handle_list_indexes,
    handle_query_codebase,
    handle_index_codebase,
    handle_get_index_stats,
)

# List indexes
indexes = await handle_list_indexes()

# Query codebase
results = await handle_query_codebase({
    "index_name": "myapp",
    "query": "authentication code",
    "n_results": 5
})
```

## Performance Tips

1. **Index Once**: Index codebases once and reuse
2. **Local Embeddings**: Use local provider for speed (no API latency)
3. **Limit Results**: Use smaller `n_results` for faster queries
4. **Incremental Updates**: Re-index only when code changes significantly
5. **Separate Indexes**: Create separate indexes for different components

## Comparison with CLI

| Feature | MCP Server | CLI |
|---------|-----------|-----|
| **Access** | Via LLM | Direct terminal |
| **Interaction** | Conversational | Command-based |
| **Multi-step** | Automatic | Manual |
| **Best For** | AI agents | Developers |
| **Installation** | pip install ctxai[mcp] | pip install ctxai |

## Future Enhancements

Planned improvements:

- [ ] **Streaming Results**: Stream results as they're found
- [ ] **HTTP Server**: Optional HTTP API mode
- [ ] **Multi-client**: Support multiple concurrent clients
- [ ] **Progress Updates**: Real-time progress for long operations
- [ ] **Caching**: Cache query results for speed
- [ ] **Incremental Indexing**: Update only changed files
- [ ] **Resource Limits**: CPU/memory usage controls
- [ ] **Authentication**: Optional token-based auth
- [ ] **Webhooks**: Notify on index changes
- [ ] **Metrics**: Usage statistics and monitoring

## See Also

- [README.md](../README.md) - Main documentation
- [QUERY_DASHBOARD.md](./QUERY_DASHBOARD.md) - Query and dashboard commands
- [CONFIGURATION.md](./CONFIGURATION.md) - Configuration options
- [MCP Protocol](https://modelcontextprotocol.io/) - Official MCP specification
