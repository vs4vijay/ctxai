# ctxai Knowledge Base

## Comprehensive Codebase Overview

### 1. Project Purpose

**ctxai** is a semantic code search engine that transforms codebases into intelligent embeddings for fast, context-aware code retrieval. It bridges the gap between traditional keyword-based search and AI-powered semantic understanding by:

- Converting code into searchable vector embeddings using natural language processing
- Supporting multiple embedding providers (local, OpenAI, HuggingFace)
- Providing both CLI and AI agent integration (MCP Server) interfaces
- Enabling agents and developers to find relevant code through natural language queries
- Available as a PyPI package, CLI tool, web dashboard, and MCP server

**Key Insight**: The project is specifically designed to enhance AI agent capabilities by providing context-aware code discovery, allowing agents to understand and work with large codebases efficiently.

---

### 2. Main Components and Their Responsibilities

#### Core Processing Pipeline

**src/ctxai/traversal.py** - Code Traversal
- Recursively walks through codebases respecting .gitignore patterns
- Handles include/exclude patterns for selective file processing
- Detects and skips binary files
- Default exclusions: git, node_modules, venv, __pycache__, build, dist, etc.

**src/ctxai/chunking.py** - Code Chunking
- Uses tree-sitter for semantic parsing (20+ programming languages)
- Intelligently splits code into meaningful chunks (functions, classes, methods)
- Preserves context with configurable overlap between chunks
- Falls back to text-based chunking for unsupported languages
- Extracts metadata (function names, class names, node types)

**src/ctxai/embeddings.py** - Embedding Generation
- Abstract provider pattern supporting multiple backends
- **LocalEmbeddingProvider**: sentence-transformers (default, no API key needed)
- **OpenAIEmbeddingProvider**: text-embedding-3-small/large
- **HuggingFaceEmbeddingProvider**: Custom models via Inference API
- Factory pattern for easy provider switching

**src/ctxai/vector_store.py** - Vector Database
- ChromaDB-based persistent vector storage
- Stores code chunks with embeddings, metadata, and line numbers
- Semantic search with similarity scoring
- Batch processing for efficient indexing
- Collection-based organization for multiple indexes

#### Command Layer

**src/ctxai/app.py** - CLI Application
- Typer-based CLI with 5 main commands:
  - `index`: Index a codebase for semantic search
  - `query`: Search indexed codebase using natural language
  - `server`: Start MCP server for AI agent integration
  - `dashboard`: Start web UI for interactive exploration
  - `config`: Manage .ctxai/config.json settings

**src/ctxai/commands/** - Command Implementations
- **index_command.py**: Orchestrates the complete indexing pipeline
- **query_command.py**: Executes semantic searches and displays results
- **server_command.py**: Starts MCP protocol server for LLM integration
- **dashboard_command.py**: Launches FastHTML web interface
- **config_command.py**: Configuration management (get, set, unset operations)

#### Configuration and Utilities

**src/ctxai/config.py** - Configuration Management
- EmbeddingConfig: Provider settings, API keys, model selection
- IndexConfig: Size limits, chunk parameters
- ConfigManager: Load/save .ctxai/config.json
- CTXAI_HOME environment variable support for centralized configuration

**src/ctxai/utils.py** - Utility Functions
- Directory management (CTXAI_HOME, indexes directory)
- Config path resolution
- Home directory information

**src/ctxai/size_validator.py** - Size Validation
- Prevents indexing overly large projects
- File count and total size checks
- Configurable limits to control costs and performance

#### Agent System

**src/ctxai/agent/core.py** - Agent Loop
- Autonomous agent with tool-calling orchestration
- Multi-turn conversation management
- Iterative execution with error recovery
- Integration with LLM providers and tool registry

**src/ctxai/agent/llm/** - LLM Providers
- **base.py**: Abstract provider interface with Message, ToolCall, LLMResponse classes
- **anthropic_provider.py**: Full Claude integration with tool use support (claude-3-5-sonnet, opus, haiku)

**src/ctxai/agent/tools/** - Tool System
- **base.py**: BaseTool abstract class with schema system
- **registry.py**: Tool registry for managing and executing tools
- **file_ops.py**: 6 file operation tools (read, write, edit, list, glob, grep)
- **bash_tool.py**: Bash command execution with safety filtering
- **code_search.py**: Semantic code search using ctxai vector store

**src/ctxai/agent/** - Agent Components
- **config.py**: Agent configuration (LLM, tools, behavior settings)
- **context.py**: Conversation history and context management
- **prompts.py**: System prompts and error recovery templates

---

### 3. Overall Architecture and Structure

```
Indexing Pipeline Flow:
1. CodeTraversal → Walks codebase respecting .gitignore
2. CodeChunker → Parses with tree-sitter, splits into semantic chunks
3. EmbeddingsFactory → Generates vectors via selected provider
4. VectorStore → Persists to ChromaDB
5. ConfigManager → Stores metadata in config.json

Query/Search Flow:
1. User Query → Natural language text
2. Embeddings → Generate vector for query
3. VectorStore.search() → Semantic similarity matching
4. Results → Return chunks sorted by relevance score

Agent System Flow:
1. User Message → Agent.process_message()
2. Agent → Calls LLM with tool schemas
3. LLM → Responds with tool calls or text
4. Tool Execution → Registry executes requested tools
5. Results → Fed back to LLM for refinement
```

**Storage Structure**:
```
.ctxai/
├── config.json (configuration, embedding settings)
└── indexes/
    └── index-name/
        ├── chroma.db (ChromaDB storage)
        └── index metadata
```

---

### 4. Technologies and Frameworks

**Core Dependencies** (from pyproject.toml):

1. **Parsing & Chunking**
   - `tree-sitter` (0.25.2+): Semantic code parsing for 20+ languages
   - `tree-sitter-language-pack` (0.9.0+): Language parsers

2. **Embedding Generation**
   - `sentence-transformers` (3.3.1+): Local embeddings (default)
   - `openai` (1.58.1+): OpenAI API integration (optional)

3. **Vector Database**
   - `chromadb` (0.5.0+): Persistent vector storage

4. **CLI & Configuration**
   - `typer` (0.19.2+): CLI framework
   - `pydantic` (2.11.10+): Data validation and settings
   - `pydantic-ai` (1.0.15+): AI integration framework
   - `rich` (13.9.4+): Terminal formatting and progress bars

5. **File & Path Handling**
   - `pathspec` (0.12.1+): .gitignore pattern matching
   - `python-dotenv` (1.1.1+): Environment variable management

6. **Optional Dependencies**
   - `python-fasthtml` (0.9.3+): Web dashboard UI
   - `mcp` (1.16.0+): Model Context Protocol for agents

7. **Development**
   - `pytest`, `pytest-cov`, `pytest-asyncio`: Testing
   - `ruff`: Linting and formatting
   - `bandit`: Security checks

**Agent-Specific Technologies**:
- `anthropic`: Claude API client
- Async/await pattern for non-blocking operations
- Schema-based tool definitions for LLM compatibility

---

### 5. Key Features and Capabilities

#### Semantic Code Search
- Query code using natural language descriptions, not keywords
- Find relevant code by meaning rather than text matching
- Similarity scoring indicates relevance confidence

#### Multiple Embedding Providers
- **Local (Default)**: Free, offline-capable, ~80MB model download
- **OpenAI**: Higher quality, costs per token
- **HuggingFace**: Alternative cloud provider
- Easy switching via .ctxai/config.json

#### Smart Code Analysis
- Tree-sitter parses code understanding structure
- Chunks at semantic boundaries (functions, classes)
- Preserves context with configurable overlap
- 20+ programming language support

#### AI Agent Integration
- MCP Server exposes tools to Claude Desktop and agents
- Agents can search code, read/write files, execute bash
- Tool-calling orchestration with error recovery
- Multi-turn conversation support

#### Web Dashboard
- FastHTML-based dark-themed UI
- View all indexes with statistics
- Interactive query interface with results
- Chunk browser with pagination
- Configuration viewer

#### Flexible Configuration
- CTXAI_HOME for centralized configuration across projects
- Project-level .ctxai/config.json for overrides
- Size limits to prevent accidental massive indexing
- Automatic initialization with sensible defaults

#### Safety Features
- Bash command filtering to block dangerous operations
- File size limits to prevent memory issues
- Project size validation before indexing
- Binary file detection to skip non-code

#### Development Features
- Comprehensive test suite (pytest)
- Type hints throughout codebase
- Clean separation of concerns
- Extensible tool and provider systems

---

### 6. Project Statistics

- **Total Python Files**: 31 files
- **Main Source Files**: 30+ files in src/ctxai
- **Agent Implementation**: ~12 core components
- **Supported Languages**: 20+ (Python, JS/TS, Java, Go, Rust, C/C++, etc.)
- **Dependencies**: 10+ core, 3 optional feature groups
- **License**: MIT
- **Python**: 3.10+

---

### 7. Data Flow Summary

**Indexing Phase**:
1. Traverse codebase → Get file list
2. Validate project size → Ensure within limits
3. Parse files with tree-sitter → Extract AST
4. Create semantic chunks → Functions, classes, etc.
5. Generate embeddings → Vector representations
6. Store in ChromaDB → Persistent index

**Query Phase**:
1. Accept natural language query
2. Generate embedding for query
3. Search ChromaDB for similar vectors
4. Return chunks with similarity scores
5. Format results with syntax highlighting

**Agent Interaction**:
1. Agent receives user request
2. Calls LLM with available tools
3. LLM calls search/file tools
4. Tools execute and return results
5. Agent refines with more LLM calls
6. Returns final response to user

---

### 8. Configuration Example

```json
{
  "version": "1.0",
  "embedding": {
    "provider": "local",
    "model": "all-MiniLM-L6-v2",
    "api_key": null,
    "batch_size": 100
  },
  "indexing": {
    "max_files": 10000,
    "max_total_size_mb": 500,
    "max_file_size_mb": 5,
    "chunk_size": 1000,
    "chunk_overlap": 100
  },
  "index_name": "my-project",
  "index_status": "completed"
}
```

---

### 9. Project Structure

```
src/ctxai/
├── app.py                      # CLI entry point (Typer)
├── traversal.py                # Codebase walking with .gitignore
├── chunking.py                 # Semantic code splitting (tree-sitter)
├── embeddings.py               # Vector generation (multi-provider)
├── vector_store.py             # ChromaDB interface
├── config.py                   # Configuration management
├── size_validator.py           # Project size validation
├── utils.py                    # Utility functions
├── commands/
│   ├── index_command.py        # Index codebase command
│   ├── query_command.py        # Query command
│   ├── server_command.py       # MCP server command
│   ├── dashboard_command.py    # Web UI command
│   └── config_command.py       # Config management command
└── agent/
    ├── core.py                 # Agent orchestration loop
    ├── config.py               # Agent configuration
    ├── context.py              # Conversation history
    ├── prompts.py              # System prompts
    ├── llm/
    │   ├── base.py             # Abstract LLM provider
    │   └── anthropic_provider.py  # Claude integration
    └── tools/
        ├── base.py             # BaseTool abstract class
        ├── registry.py         # Tool registry
        ├── file_ops.py         # File operation tools
        ├── bash_tool.py        # Bash execution
        └── code_search.py      # Semantic code search
```

---

### 10. Usage Examples

#### CLI Usage

```bash
# Index a codebase
ctxai index /path/to/project --index-name my-project

# Query the index
ctxai query my-project "Where is authentication handled?"

# Start MCP server for AI agents
ctxai server my-project

# Launch web dashboard
ctxai dashboard my-project

# Configure settings
ctxai config set embedding.provider openai
ctxai config set embedding.api_key sk-...
```

#### Python API Usage

```python
from ctxai.vector_store import VectorStore
from ctxai.embeddings import EmbeddingsFactory
from ctxai.config import EmbeddingConfig

# Initialize components
config = EmbeddingConfig(provider="local")
embeddings = EmbeddingsFactory.create(config)
store = VectorStore(embeddings, index_name="my-project")

# Search
results = store.search("authentication logic", top_k=5)
for result in results:
    print(f"{result['file_path']}:{result['start_line']}")
    print(result['content'])
```

---

### 11. Development Workflow

**Testing**:
```bash
pytest                    # Run all tests
pytest --cov=src/ctxai   # Run with coverage
pytest -v                # Verbose output
```

**Linting**:
```bash
ruff check src/         # Check for issues
ruff format src/        # Format code
```

**Security**:
```bash
bandit -r src/ctxai/    # Security vulnerability scan
```

**Building**:
```bash
pip install -e .        # Install in development mode
pip install -e ".[dashboard]"  # With dashboard
pip install -e ".[server]"     # With MCP server
pip install -e ".[all]"        # All features
```

---

### Summary

**ctxai** is a sophisticated semantic code search engine with a well-architected multi-component system. It leverages tree-sitter for intelligent code parsing, vector embeddings for semantic understanding, and a modular design supporting multiple backends and interfaces (CLI, Web UI, MCP Server, Python API). The recent addition of an autonomous agent system with tool-calling capabilities positions it as a powerful platform for AI-assisted code discovery and manipulation, making it particularly valuable for AI agents and multi-agent orchestration frameworks.

The codebase demonstrates solid software engineering practices with clear separation of concerns, extensible provider patterns, comprehensive configuration management, and built-in safety features. It's designed to work seamlessly with large codebases while remaining lightweight and flexible for various deployment scenarios.
