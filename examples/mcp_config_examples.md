# Example MCP Server Configuration

This directory contains example configurations for using ctxai with MCP-compatible clients.

## Claude Desktop

### Basic Configuration

**File**: `claude_desktop_config.json`

Add this to your Claude Desktop configuration:

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

### With Custom Project Path

```json
{
  "mcpServers": {
    "ctxai": {
      "command": "ctxai",
      "args": ["server", "--project-path", "/Users/yourname/projects"]
    }
  }
}
```

### With Global CTXAI_HOME

```json
{
  "mcpServers": {
    "ctxai": {
      "command": "ctxai",
      "args": ["server"],
      "env": {
        "CTXAI_HOME": "/Users/yourname/.ctxai"
      }
    }
  }
}
```

### Using Python Directly

If ctxai is not in PATH:

```json
{
  "mcpServers": {
    "ctxai": {
      "command": "python",
      "args": ["-m", "ctxai.mcp_server"]
    }
  }
}
```

## Usage Examples with Claude

Once configured, you can interact with Claude like this:

### Example 1: List Indexes

**You**: What code indexes do I have available?

**Claude**: Let me check what indexes are available.
[Uses list_indexes tool]

**Claude**: You have 2 indexes:
- **myapp**: 1,234 chunks
- **library**: 456 chunks

### Example 2: Index a Codebase

**You**: Can you index my project at /Users/me/projects/newapp?

**Claude**: I'll index that codebase for you.
[Uses index_codebase tool]

**Claude**: ✓ Successfully indexed 'newapp' with 789 chunks.

### Example 3: Query Code

**You**: Find all the authentication code in myapp

**Claude**: Let me search for authentication code.
[Uses query_codebase tool]

**Claude**: I found 3 relevant results:

1. **auth.py** (Lines 10-45) - 89% similarity
   Contains the main authentication class...
   
2. **middleware.py** (Lines 67-82) - 76% similarity
   Authentication middleware...
   
3. **views.py** (Lines 123-145) - 72% similarity
   Login/logout views...

### Example 4: Get Statistics

**You**: Show me stats for the myapp index

**Claude**: [Uses get_index_stats tool]

**Claude**: Here are the statistics for myapp:
- Total chunks: 1,234
- Storage size: 45.2 MB
- Location: /Users/me/.ctxai/indexes/myapp

## Troubleshooting

### Server Not Showing Up in Claude

1. **Check Configuration File Location**
   - Verify the config file is in the correct location
   - Make sure JSON is valid (use a JSON validator)

2. **Restart Claude Desktop**
   - Close Claude completely
   - Open again to reload configuration

3. **Check ctxai Installation**
   ```bash
   # Verify ctxai is installed
   ctxai --version
   
   # Verify MCP is installed
   pip show mcp
   ```

4. **Check Logs**
   - Claude Desktop logs may show connection errors
   - On macOS: `~/Library/Logs/Claude/`

### Server Crashes or Disconnects

1. **Check MCP Installation**
   ```bash
   pip install ctxai[mcp]
   ```

2. **Test Server Manually**
   ```bash
   ctxai server
   # Should start without errors
   ```

3. **Check Permissions**
   - Ensure ctxai has read access to CTXAI_HOME
   - Verify write permissions for indexing

## Advanced Configuration

### Multiple Projects

You can run multiple ctxai instances for different projects:

```json
{
  "mcpServers": {
    "ctxai-frontend": {
      "command": "ctxai",
      "args": ["server", "--project-path", "/path/to/frontend"]
    },
    "ctxai-backend": {
      "command": "ctxai",
      "args": ["server", "--project-path", "/path/to/backend"]
    }
  }
}
```

Then specify which one to use:
- "Search ctxai-frontend for components"
- "Index the backend with ctxai-backend"

### With Environment Variables

Set additional environment variables:

```json
{
  "mcpServers": {
    "ctxai": {
      "command": "ctxai",
      "args": ["server"],
      "env": {
        "CTXAI_HOME": "/custom/path/.ctxai",
        "OPENAI_API_KEY": "sk-...",
        "LOG_LEVEL": "DEBUG"
      }
    }
  }
}
```

### Using Virtual Environment

If ctxai is in a virtual environment:

```json
{
  "mcpServers": {
    "ctxai": {
      "command": "/path/to/venv/bin/ctxai",
      "args": ["server"]
    }
  }
}
```

Or:

```json
{
  "mcpServers": {
    "ctxai": {
      "command": "/path/to/venv/bin/python",
      "args": ["-m", "ctxai", "server"]
    }
  }
}
```

## Security Considerations

### API Keys

- Store API keys in environment variables, not config files
- Use `.env` files for local development
- Don't commit keys to version control

### Access Control

- MCP server runs with your user permissions
- Can access any file you can access
- Consider creating a dedicated user for production

### Network Isolation

- Server uses stdio (no network exposure)
- Only accessible to parent process (Claude Desktop)
- No remote access by default

## See Also

- [MCP_SERVER.md](../docs/MCP_SERVER.md) - Complete MCP server documentation
- [README.md](../README.md) - Main documentation
- [Model Context Protocol](https://modelcontextprotocol.io/) - Official MCP docs
