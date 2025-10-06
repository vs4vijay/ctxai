# ctxai

A semantic code search engine that transforms your codebase into intelligent embeddings for fast, context-aware code retrieval. **ctxai** uses natural language processing to find code snippets, documentation, and examples through both CLI and MCP Server interfaces.

Available as both an MCP Server and CLI tool, **ctxai** integrates seamlessly with multi-agent systems and orchestration frameworks, allowing agents to discover relevant code through semantic queries.

TLDR; Intelligent semantic search across your entire codebase

Transform your code into searchable embeddings with advanced chunking and vector database indexing

## Features

- **MCP Server Integration**: Works with any agent that supports MCP protocol
- **Smart Code Search**: Converts your code into searchable vectors using AI
- **Natural Language Queries**: Find code by describing what you want, not just keywords
- **CLI and Agent Ready**: Use from command line or integrate with AI agents
- **Fast Indexing**: Quickly processes large codebases

## Usage

### Indexing Your Codebase

Index your project to enable semantic search:

```bash
python -m ctxai.index /path/to/codebase "index_name"

# OR
ctxai index /path/to/codebase "index_name"
```

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

```bash
pip install ctxai

# OR using uv
uvx ctxai
```

## Running

```bash
uv run ctxai
```

### Development Notes

```bash

ctxai index
ctxai server
ctxai playgruond
ctxai shell


python -m ctxai.index /path/to/codebase "index_name"
/ctxai "Find the code related to Profile Image update"


tree-sitter
ast


perfect

autogen

pydentic ai

memory
mem0

spec kit

agent-framework


```

## Releasing

- Bump version in pyproject.toml and push to main
- create a new release with tags pattern `vx.y.z` e.g. v0.0.1
- It would create a release on github and start a github action which would publish on pypi

## Contributing

We welcome all contributions to the project! Before submitting your pull request, please ensure you have run the tests and linters locally. This helps us maintain the quality of the project and makes the review process faster for everyone.

All contributions should adhere to the project's code of conduct. Let's work together to create a welcoming and inclusive environment for everyone.
