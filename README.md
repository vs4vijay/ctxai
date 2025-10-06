# ctxai

A semantic code search engine that transforms your codebase into intelligent embeddings for fast, context-aware code retrieval. **ctxai** uses natural language processing to find code snippets, documentation, and examples through both CLI and MCP Server interfaces.

Available as both an MCP Server and CLI tool, **ctxai** integrates seamlessly with multi-agent systems and orchestration frameworks, allowing agents to discover relevant code through semantic queries.

TLDR; Intelligent semantic search across your entire codebase

Transform your code into searchable embeddings with advanced chunking and vector database indexing

## Quick Start

```bash
# 1. Install ctxai
pip install ctxai

# 2. Set your OpenAI API key
export OPENAI_API_KEY=your-api-key-here

# 3. Index your codebase
ctxai index /path/to/your/project "my-project"

# 4. Use with GitHub Copilot (coming soon)
# @ctxai find code for updating profile images
```

## Features

- **MCP Server Integration**: Works with any agent that supports MCP protocol
- **Smart Code Search**: Converts your code into searchable vectors using AI
- **Natural Language Queries**: Find code by describing what you want, not just keywords
- **CLI and Agent Ready**: Use from command line or integrate with AI agents
- **Fast Indexing**: Quickly processes large codebases

## Usage

### Prerequisites

Before using ctxai, you need to set up your OpenAI API key:

```bash
export OPENAI_API_KEY=your-api-key-here
```

### Indexing Your Codebase

Index your project to enable semantic search:

```bash
# Basic usage
ctxai index /path/to/codebase "index_name"

# With Python module
python -m ctxai index /path/to/codebase "index_name"

# Include only specific file patterns
ctxai index /path/to/codebase "my-index" --include "*.py" --include "*.js"

# Exclude additional patterns beyond .gitignore
ctxai index /path/to/codebase "my-index" --exclude "*.test.js" --exclude "migrations/*"

# Don't follow .gitignore
ctxai index /path/to/codebase "my-index" --no-follow-gitignore
```

The indexing process will:
1. Traverse your codebase recursively (respecting .gitignore by default)
2. Parse code using tree-sitter for semantic understanding
3. Chunk code intelligently (functions, classes, etc.)
4. Generate embeddings using OpenAI's embedding API
5. Store in a local ChromaDB vector database (`.ctxai` directory)

### CLI Commands

View all available commands:

```bash
ctxai --help
```

Available commands:
- `index` - Index a codebase for semantic search
- `server` - Start the MCP server (coming soon)
- `dashboard` - Start the web dashboard (coming soon)

### MCP Server Configuration

Configure the MCP server by creating an `mcp.json` file:

```json
{
  "inputs": [],
  "servers": {
    "ctxai": {
      "command": "python",
      "args": ["-m", "ctxai.server", "--index", "index_name"]
    }
  }
}
```

### Querying with GitHub Copilot

Use natural language queries through GitHub Copilot's Agent mode:

```
@ctxai find code for updating profile images
```

---

## Installation

Pre-requisites:

- Python 3.10+
- OpenAI API key (for generating embeddings)

```bash
pip install ctxai

# OR using uv
uv pip install ctxai

# OR run directly with uvx
uvx ctxai
```

## Running

```bash
# Run with uv
uv run ctxai index /path/to/codebase "index-name"

# Or install and run directly
pip install ctxai
ctxai --help
```

## Architecture

ctxai uses a multi-stage pipeline to transform your codebase into searchable vectors:

1. **Traversal**: Recursively walks through your codebase, respecting `.gitignore` patterns and custom include/exclude rules
2. **Parsing**: Uses tree-sitter to parse code and understand its structure (functions, classes, methods, etc.)
3. **Chunking**: Intelligently splits code into semantic chunks while preserving context and meaning
4. **Embedding**: Generates vector embeddings using OpenAI's embedding API
5. **Storage**: Stores embeddings in a local ChromaDB vector database (in `.ctxai` directory)

### Components

- **`traversal.py`**: File system traversal with gitignore support
- **`chunking.py`**: Tree-sitter based intelligent code chunking
- **`embeddings.py`**: OpenAI embedding generation
- **`vector_store.py`**: ChromaDB vector database management
- **`commands/index_command.py`**: Orchestrates the indexing pipeline

### Storage

Indexed codebases are stored locally in the `.ctxai/indexes/<index-name>` directory within your project. This directory contains:
- ChromaDB vector database
- Chunk metadata and embeddings
- Index configuration



## Development

### Setup Development Environment

```bash
# Clone the repository
git clone https://github.com/vs4vijay/ctxai.git
cd ctxai

# Install dependencies with uv
uv sync

# Or with pip
pip install -e ".[dev]"
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_indexing.py

# Run with coverage
pytest --cov=ctxai
```

### Code Quality

```bash
# Run linter
ruff check src/

# Format code
ruff format src/

# Type checking (if mypy is added)
mypy src/
```

### Project Structure

```
ctxai/
├── src/ctxai/
│   ├── __init__.py
│   ├── __main__.py
│   ├── app.py              # Typer CLI app
│   ├── chunking.py         # Code chunking logic
│   ├── embeddings.py       # Embedding generation
│   ├── traversal.py        # File system traversal
│   ├── vector_store.py     # Vector DB management
│   ├── server.py           # MCP server (coming soon)
│   └── commands/
│       ├── __init__.py
│       └── index_command.py
├── tests/
│   ├── __init__.py
│   ├── test_server.py
│   └── test_indexing.py
├── examples/
│   └── example_usage.py
├── pyproject.toml
└── README.md
```

## Releasing

- Bump version in pyproject.toml and push to main
- create a new release with tags pattern `vx.y.z` e.g. v0.0.1
- It would create a release on github and start a github action which would publish on pypi

## Troubleshooting

### OpenAI API Key Error

If you get an error about missing OpenAI API key:

```bash
export OPENAI_API_KEY=your-api-key-here  # Linux/Mac
set OPENAI_API_KEY=your-api-key-here     # Windows CMD
$env:OPENAI_API_KEY="your-api-key-here"  # Windows PowerShell
```

Or create a `.env` file in your project:

```bash
OPENAI_API_KEY=your-api-key-here
```

### No Files Found to Index

If the indexing process finds no files:
- Check your include/exclude patterns
- Verify the path is correct
- Use `--no-follow-gitignore` if files are being ignored
- Check that files are not binary

### Tree-sitter Parse Errors

If you see warnings about parsing errors:
- These are usually non-critical
- The tool will fall back to simple text chunking
- Only affects the semantic understanding, not the search capability

### Memory Issues with Large Codebases

For very large codebases:
- Index in smaller batches using include patterns
- Reduce `max_chunk_size` in the chunker
- Monitor the `.ctxai` directory size

## Contributing

We welcome all contributions to the project! Before submitting your pull request, please ensure you have run the tests and linters locally. This helps us maintain the quality of the project and makes the review process faster for everyone.

All contributions should adhere to the project's code of conduct. Let's work together to create a welcoming and inclusive environment for everyone.
