# ctxai Production Roadmap: From v0.0.2 to Production-Ready Coding Agent

## Executive Summary

**ctxai** is a well-architected AI coding agent (v0.0.2) with semantic code search capabilities. This plan outlines 6 vertical phases to transform it into a production-ready, exportable coding agent harness with MCP support, long-running service architecture, and comprehensive export features.

**Current State:**
- ✅ Mature agent loop with 6 LLM providers, tool calling, streaming
- ✅ Complete semantic search pipeline (tree-sitter, embeddings, ChromaDB)
- ✅ FastMCP server with 4 tools (list, index, query, stats)
- ✅ 8+ tools (file ops, bash, git, code search)
- ✅ Architect-Editor pattern for cost optimization
- ⚠️ Planning system defined but not integrated
- ⚠️ Only 15 tests (needs 80%+ coverage)
- ⚠️ No long-running service architecture
- ⚠️ No core library extraction
- ❌ Missing: repo-to-text export, HTML code map

**Vision:**
1. Stable, production-ready coding agent
2. Exportable harness (ctxai-core) for building custom agents
3. Long-running daemon with REST/WebSocket API
4. Complete MCP integration (tools + resources + prompts)
5. Codebase export features (text, HTML, JSON)
6. Production deployment ready

---

## Phase 1: Stability & Comprehensive Testing ⭐ PRIORITY
**Duration:** 2-3 weeks | **Status:** Foundation phase, must complete first

### Goals
- Achieve 80%+ test coverage across all modules
- Fix existing bugs and enhance error handling
- Production-grade logging and monitoring
- Performance optimization and profiling
- Establish CI/CD foundation

### Key Deliverables

#### 1.1 Comprehensive Test Suite
**New Test Files:**
- `tests/unit/test_agent_core.py` - Agent loop logic (tool calling, iterations, context)
- `tests/unit/test_tools_registry.py` - Tool registration, execution, schema generation
- `tests/unit/test_llm_providers.py` - All 6 providers (Anthropic, OpenAI, OpenRouter, Ollama, GitHub Copilot, Custom)
- `tests/unit/test_context_management.py` - Message history, truncation, token counting
- `tests/unit/test_planning.py` - Plan, PlanStep, PlanExecutor classes
- `tests/integration/test_agent_tools_integration.py` - End-to-end agent + tools workflows
- `tests/integration/test_llm_fallback.py` - Provider fallback scenarios
- `tests/integration/test_mcp_integration.py` - MCP server tools
- `tests/performance/test_agent_performance.py` - Benchmarks (agent loop < 2s/iteration)

**Enhanced Fixtures:**
- `tests/conftest.py` - Mock LLM providers, temporary directories, sample codebases

**Coverage Target:** 80%+ measured with `pytest --cov=ctxai --cov-report=html`

#### 1.2 Error Handling Enhancements
**Files to Modify:**
- `src/ctxai/agent/core.py`
  - Exponential backoff for LLM retries
  - Circuit breaker pattern for failing providers
  - Context window overflow detection
  - Tool timeout enforcement (configurable, default 30s)
  - Better loop detection (prevent infinite tool calls)

- `src/ctxai/agent/llm/base.py`
  - Standardized error classes (`ProviderError`, `RateLimitError`, `ContextLengthError`)
  - Provider health checks before calls

- `src/ctxai/agent/tools/registry.py`
  - Graceful degradation when tools fail
  - Tool execution sandboxing improvements
  - Concurrent tool execution with timeout

- `src/ctxai/agent/context.py`
  - Automatic context pruning when approaching limits
  - Message priority system (keep system/recent, prune middle)

#### 1.3 Production Logging & Monitoring
**New Files:**
- `src/ctxai/logging.py` - Structured logging setup
  - JSON format with log levels (DEBUG, INFO, WARNING, ERROR)
  - Request ID tracking across agent loop
  - Rotating file handlers (max 10MB, 5 backups)
  - Environment-based configuration (DEV/PROD modes)

- `src/ctxai/monitoring.py` - Metrics collection
  - Performance metrics: LLM latency, tool execution time
  - Cost tracking: tokens used per request/provider
  - Error rates by provider and tool
  - Agent loop iteration counts

**Modified Files:**
Add structured logging to:
- `src/ctxai/agent/core.py` - Agent loop events
- `src/ctxai/agent/tools/base.py` - Tool execution logs
- All LLM providers - Request/response logging (with PII filtering)

#### 1.4 Bug Fixes & Optimizations
**Known Issues to Address:**
1. Agent loop memory leaks in long conversations (>100 messages)
2. Context truncation edge cases (messages split incorrectly)
3. Provider fallback logic gaps (doesn't try all available providers)
4. Tool call loop prevention needs refinement
5. File operations don't validate paths properly (security issue)

**Performance Optimizations:**
- Profile agent loop with `cProfile`
- Optimize embedding generation (batch processing)
- Cache tool schemas (no need to regenerate each call)
- Lazy load tree-sitter parsers

### Critical Files
- `src/ctxai/agent/core.py` (388 lines) - Core agent loop
- `src/ctxai/agent/llm/base.py` - Provider interface
- `src/ctxai/agent/tools/registry.py` - Tool system
- `src/ctxai/agent/context.py` - Context management
- `tests/conftest.py` - Test infrastructure
- `pyproject.toml` - Test configuration

### Verification Steps
1. Run `pytest --cov=ctxai --cov-report=html` → Verify 80%+ coverage
2. Run `pytest -m performance` → Verify agent loop < 2s per iteration
3. Test all 6 LLM providers with fallback scenarios
4. Stress test: 100-message conversation, monitor memory
5. Run `ruff check .` → No errors
6. Security scan: `bandit -r src/ctxai`

---

## Phase 2: Core Library Refactoring (ctxai-core)
**Duration:** 3-4 weeks | **Depends on:** Phase 1

### Goals
- Extract reusable agent harness (`ctxai-core` package)
- Clean public API for agent, tools, providers
- Plugin/extension system for customization
- Comprehensive API documentation
- Multiple working examples

### Key Deliverables

#### 2.1 Library Structure
**New Package Structure:**
```
ctxai/
├── src/
│   ├── ctxai/              # CLI application (depends on ctxai-core)
│   │   ├── commands/       # CLI commands (chat, index, query, server)
│   │   ├── cli_utils.py    # CLI-specific utilities
│   │   └── __main__.py     # Entry point
│   │
│   └── ctxai_core/         # 🆕 Core library (standalone, reusable)
│       ├── __init__.py     # Public API surface
│       ├── api.py          # 🆕 Clean API facade
│       │
│       ├── agent/          # Agent engine
│       │   ├── core.py
│       │   ├── context.py
│       │   ├── planning.py
│       │   └── config.py
│       │
│       ├── llm/            # LLM provider abstraction
│       │   ├── base.py
│       │   ├── factory.py
│       │   └── providers/
│       │       ├── anthropic.py
│       │       ├── openai.py
│       │       ├── openrouter.py
│       │       ├── ollama.py
│       │       ├── github_copilot.py
│       │       └── custom.py
│       │
│       ├── tools/          # Tool system
│       │   ├── base.py
│       │   ├── registry.py
│       │   └── builtin/    # Built-in tools
│       │       ├── file_ops.py
│       │       ├── bash.py
│       │       ├── git.py
│       │       └── search.py
│       │
│       ├── search/         # Semantic search (optional)
│       │   ├── indexing.py
│       │   ├── embeddings.py
│       │   └── vector_store.py
│       │
│       └── plugins/        # 🆕 Plugin system
│           ├── base.py
│           ├── loader.py
│           └── hooks.py
```

#### 2.2 Public API Design
**File:** `src/ctxai_core/api.py`

```python
from ctxai_core import (
    # Agent creation
    Agent,
    AgentConfig,
    create_agent,

    # Tool system
    BaseTool,
    ToolRegistry,
    create_tool,

    # LLM providers
    LLMProvider,
    create_provider,

    # Planning
    Plan,
    PlanStep,
    PlanExecutor,
)

# Example 1: Simple agent
agent = create_agent(
    provider="openrouter",
    model="anthropic/claude-3.5-sonnet",
    tools=["file_ops", "bash", "git"]
)
response = await agent.process("Create a Python function...")

# Example 2: Custom tool
@create_tool(name="my_tool", description="Does something")
async def my_custom_tool(param1: str) -> dict:
    return {"result": "..."}

agent.register_tool(my_custom_tool)

# Example 3: Architect-Editor pattern
agent = create_agent(
    preset="default",  # o1-mini + Claude Sonnet
    tools=["file_ops", "search"]
)
```

#### 2.3 Plugin System
**New Files:**
- `src/ctxai_core/plugins/base.py` - Plugin interface

```python
class PluginInterface:
    def on_agent_init(self, agent: Agent) -> None: ...
    def on_message_start(self, message: str) -> str: ...
    def on_message_end(self, response: str) -> str: ...
    def on_tool_call(self, tool_name: str, args: dict) -> dict: ...
```

- `src/ctxai_core/plugins/loader.py` - Plugin discovery and loading
  - Discover plugins from `~/.ctxai/plugins/` directory
  - Load plugins from entry points
  - Plugin dependency resolution

- `src/ctxai_core/plugins/hooks.py` - Lifecycle hooks
  - Pre/post message processing
  - Pre/post tool execution
  - Context modification hooks

**Plugin Types:**
1. **Tool plugins** - Custom tools (e.g., Jira integration, Slack notifications)
2. **Provider plugins** - Custom LLM providers (e.g., custom API endpoints)
3. **Planning plugins** - Custom planning strategies
4. **Context plugins** - Custom context managers (e.g., database-backed context)

#### 2.4 Documentation & Examples
**New Documentation:**
- `docs/api/agent.md` - Agent API reference
- `docs/api/tools.md` - Tool system API
- `docs/api/providers.md` - LLM provider API
- `docs/guides/custom_tools.md` - Building custom tools
- `docs/guides/custom_providers.md` - Building custom providers
- `docs/guides/plugins.md` - Plugin development guide
- `docs/examples/README.md` - Examples index

**New Examples:**
- `examples/basic_agent.py` - Simple agent usage
- `examples/custom_tool.py` - Custom tool implementation
- `examples/custom_provider.py` - Custom LLM provider
- `examples/architect_editor.py` - Two-model pattern
- `examples/planning_workflow.py` - Multi-step task with planning
- `examples/plugin_example.py` - Plugin development
- `examples/embedding_search.py` - Agent with semantic search

#### 2.5 Package Configuration
**Modify:** `pyproject.toml`

```toml
[project]
name = "ctxai-core"
version = "1.0.0"
description = "Reusable Python agent harness for building coding agents"

[project.optional-dependencies]
full = ["ctxai"]  # Includes CLI

[project.entry-points."ctxai.plugins"]
builtin_tools = "ctxai_core.tools.builtin"
```

### Critical Files
- `src/ctxai_core/api.py` - Public API facade (NEW)
- `src/ctxai_core/plugins/base.py` - Plugin system (NEW)
- `pyproject.toml` - Package configuration
- `docs/api/` - API documentation (NEW)
- `examples/` - Working examples (NEW)

### Verification Steps
1. Install ctxai-core: `pip install -e .[dev]`
2. Run all examples successfully
3. Create external test project using ctxai-core
4. Verify API documentation completeness
5. Test plugin system with sample plugin
6. Publish to PyPI (test.pypi.org first)

---

## Phase 3: Long-Running Service Architecture
**Duration:** 3-4 weeks | **Depends on:** Phase 2

### Goals
- Background daemon/service with systemd support
- REST and WebSocket APIs for agent interaction
- Session management with persistence
- Concurrent multi-session handling
- Health checks and Prometheus metrics
- Docker/Kubernetes deployment ready

### Key Deliverables

#### 3.1 Service Architecture
**New Directory:** `src/ctxai/service/`

**Architecture:**
```
Client (CLI/Web/API)
    ↓
REST API (FastAPI) → Session Manager → Agent Pool (5-10 agents)
    ↓                      ↓                    ↓
WebSocket Streaming   State Store       Agent Instances
                      (SQLite/Redis)          ↓
                                        Tool Execution
```

**New Files:**
- `src/ctxai/service/daemon.py` - Service lifecycle management
- `src/ctxai/service/api_server.py` - FastAPI REST server
- `src/ctxai/service/websocket_server.py` - WebSocket streaming
- `src/ctxai/service/session_manager.py` - Session lifecycle
- `src/ctxai/service/state_store.py` - Persistent state (SQLite/Redis)
- `src/ctxai/service/health.py` - Health checks and metrics

#### 3.2 REST API Design
**Endpoints:** `src/ctxai/service/api_server.py`

```
# Session management
POST   /api/v1/sessions                Create new session
GET    /api/v1/sessions                List all sessions
GET    /api/v1/sessions/{id}           Get session info
DELETE /api/v1/sessions/{id}           Delete session
PATCH  /api/v1/sessions/{id}           Update session config

# Messaging
POST   /api/v1/sessions/{id}/messages  Send message to agent
GET    /api/v1/sessions/{id}/messages  Get message history
DELETE /api/v1/sessions/{id}/messages  Clear history

# Planning
GET    /api/v1/sessions/{id}/plan      Get current plan
POST   /api/v1/sessions/{id}/plan      Update plan

# System
GET    /api/v1/health                  Health check
GET    /api/v1/metrics                 Prometheus metrics
GET    /api/v1/info                    System information
```

**Authentication:** API key (configurable, optional for local use)

#### 3.3 WebSocket Streaming
**Endpoint:** `ws://localhost:8000/ws/sessions/{session_id}`

**Event Types:**
```json
{
  "event": "message.start",
  "session_id": "uuid",
  "timestamp": "2026-05-21T10:00:00Z"
}

{
  "event": "message.chunk",
  "content": "Here is the code...",
  "chunk_id": 1
}

{
  "event": "tool.start",
  "tool_name": "ReadFileTool",
  "arguments": {"path": "main.py"}
}

{
  "event": "tool.complete",
  "tool_name": "ReadFileTool",
  "result": {...}
}

{
  "event": "plan.created",
  "plan": {...}
}

{
  "event": "message.complete",
  "total_tokens": 1500
}
```

#### 3.4 Session Management
**File:** `src/ctxai/service/session_manager.py`

**Features:**
- Session creation with UUID
- Session expiration (configurable TTL, default 24h)
- Session persistence to disk/database
- Concurrent session handling (up to 100 sessions)
- Session resource limits:
  - Max messages: 1000 (configurable)
  - Max tokens: 1M (configurable)
  - Max execution time: 1 hour (configurable)
- Session migration for upgrades

**Session Schema:**
```python
@dataclass
class Session:
    session_id: str
    created_at: datetime
    last_active: datetime
    config: AgentConfig
    agent_instance: Agent
    metadata: dict
    state: SessionState
```

#### 3.5 State Persistence
**File:** `src/ctxai/service/state_store.py`

**Storage Options:**
- SQLite (default, single-node)
- Redis (optional, distributed, high-performance)
- File-based (fallback, simple)

**Database Schema:**
```sql
CREATE TABLE sessions (
    session_id TEXT PRIMARY KEY,
    created_at TIMESTAMP,
    last_active TIMESTAMP,
    config_json TEXT,
    metadata_json TEXT,
    state TEXT
);

CREATE TABLE messages (
    message_id TEXT PRIMARY KEY,
    session_id TEXT REFERENCES sessions(session_id),
    role TEXT,
    content TEXT,
    tool_calls_json TEXT,
    timestamp TIMESTAMP
);

CREATE TABLE plans (
    plan_id TEXT PRIMARY KEY,
    session_id TEXT REFERENCES sessions(session_id),
    goal TEXT,
    steps_json TEXT,
    status TEXT,
    created_at TIMESTAMP,
    completed_at TIMESTAMP
);
```

#### 3.6 Daemon Management
**File:** `src/ctxai/service/daemon.py`

**Features:**
- Start/stop/restart/reload daemon
- PID file management (`/var/run/ctxai.pid` or `~/.ctxai/ctxai.pid`)
- Signal handling:
  - SIGTERM: Graceful shutdown (finish active requests)
  - SIGHUP: Reload configuration
  - SIGINT: Immediate shutdown
- Auto-restart on crash (optional)
- Log rotation (daily or size-based)

**CLI Commands:**
```bash
ctxai service start --port 8000 --host 0.0.0.0 [--workers 4]
ctxai service stop
ctxai service restart
ctxai service reload     # Reload config without restart
ctxai service status
ctxai service logs --follow [--tail 100]
```

#### 3.7 Health & Monitoring
**File:** `src/ctxai/service/health.py`

**Health Check Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "uptime_seconds": 3600,
  "active_sessions": 15,
  "providers": {
    "openrouter": "healthy",
    "anthropic": "degraded"
  },
  "database": "healthy",
  "memory_usage_mb": 512,
  "disk_space_gb": 25.5
}
```

**Prometheus Metrics:**
```
# Counters
ctxai_requests_total{endpoint="/api/v1/sessions", method="POST"} 150
ctxai_errors_total{provider="openrouter", error_type="rate_limit"} 5
ctxai_tokens_total{provider="anthropic", model="claude-3.5-sonnet"} 500000

# Histograms
ctxai_request_duration_seconds{endpoint="/api/v1/sessions/{id}/messages"} 2.5
ctxai_llm_latency_seconds{provider="openrouter"} 1.2
ctxai_tool_execution_seconds{tool="ReadFileTool"} 0.05

# Gauges
ctxai_active_sessions 15
ctxai_memory_usage_bytes 536870912
```

### Critical Files
- `src/ctxai/service/api_server.py` - REST API (NEW)
- `src/ctxai/service/session_manager.py` - Session management (NEW)
- `src/ctxai/service/state_store.py` - Persistence (NEW)
- `src/ctxai/service/daemon.py` - Daemon manager (NEW)
- `src/ctxai/service/health.py` - Health/metrics (NEW)
- `pyproject.toml` - Add FastAPI, uvicorn dependencies

### Verification Steps
1. Start service: `ctxai service start --port 8000`
2. Create 10 sessions via API: `POST /api/v1/sessions`
3. Send 100 messages across sessions
4. Test WebSocket streaming
5. Kill service (SIGTERM), verify graceful shutdown
6. Restart service, verify session recovery from state store
7. Load test: 100 concurrent requests with `locust`
8. Monitor Prometheus metrics dashboard

---

## Phase 4: MCP Enhancement & Planning Integration
**Duration:** 2-3 weeks | **Depends on:** Phase 1

### Goals
- Integrate planning system into agent loop
- Add MCP resources for file/repo access
- Add MCP prompts for common tasks
- Enhanced tool descriptions with examples
- Improved Claude Desktop integration

### Key Deliverables

#### 4.1 Planning System Integration
**Modify:** `src/ctxai/agent/core.py`

**Integration Logic:**
```python
async def process_message(self, user_message: str) -> str:
    # Determine if task needs planning
    if self._requires_planning(user_message):
        # Create plan using architect
        plan = await self._create_plan(user_message)
        self.context.set_current_plan(plan)

        # Execute plan steps
        executor = PlanExecutor(plan, agent=self)
        result = await executor.execute()

        return result
    else:
        # Standard agent loop
        return await self._standard_loop(user_message)

def _requires_planning(self, message: str) -> bool:
    """Check if message needs planning (multi-step task)"""
    planning_indicators = [
        "refactor", "migrate", "add feature", "implement",
        "create project", "setup", "multi-step"
    ]
    return any(indicator in message.lower() for indicator in planning_indicators)
```

**New Files:**
- `src/ctxai/agent/planners/simple_planner.py` - Rule-based planning
- `src/ctxai/agent/planners/llm_planner.py` - LLM-powered planning
- `src/ctxai/agent/planners/hierarchical_planner.py` - Multi-level plans

**Modify:** `src/ctxai/agent/architect_editor.py`
- Use planning system for architect's design phase
- Editor executes plan steps

#### 4.2 MCP Resources
**Modify:** `src/ctxai/commands/server_command.py`

**New Resources:**
```python
@mcp.resource("file://{path}")
async def get_file_content(uri: str) -> str:
    """Provide file contents as MCP resource"""
    path = uri.replace("file://", "")
    return Path(path).read_text()

@mcp.resource("repo://{name}")
async def get_repo_map(uri: str) -> str:
    """Provide repository map as MCP resource"""
    repo_name = uri.split("/")[-1]
    repo_map = RepositoryMap(path=Path.cwd())
    return repo_map.generate_map()

@mcp.resource("index://{name}/stats")
async def get_index_info(uri: str) -> str:
    """Provide index information as MCP resource"""
    index_name = uri.split("/")[-2]
    # ... get stats
    return json.dumps(stats)

@mcp.resource("session://{id}/context")
async def get_session_context(uri: str) -> str:
    """Provide session context as MCP resource (for service mode)"""
    session_id = uri.split("/")[-2]
    # ... get session context
    return json.dumps(context)
```

#### 4.3 MCP Prompts
**Add to:** `src/ctxai/commands/server_command.py`

```python
@mcp.prompt("analyze_codebase")
async def analyze_codebase_prompt(
    language: str = "python",
    focus: str = "architecture"
) -> str:
    """Prompt for analyzing codebase structure and patterns"""
    return f"""
    Analyze this {language} codebase with focus on {focus}.

    Examine:
    1. Project structure and organization
    2. Key architectural patterns
    3. Code quality and best practices
    4. Potential improvements

    Provide a detailed report with specific examples.
    """

@mcp.prompt("refactor_code")
async def refactor_code_prompt(
    file_path: str,
    refactor_type: str = "general"
) -> str:
    """Prompt for refactoring code"""
    return f"""
    Refactor the code in {file_path} ({refactor_type} refactoring).

    Steps:
    1. Read and understand current code
    2. Identify refactoring opportunities
    3. Apply improvements
    4. Verify functionality preserved
    5. Run tests to confirm
    """

@mcp.prompt("add_tests")
async def add_tests_prompt(
    file_path: str,
    test_framework: str = "pytest"
) -> str:
    """Prompt for adding comprehensive tests"""
    return f"""
    Add comprehensive tests for {file_path} using {test_framework}.

    Test coverage should include:
    1. Happy path scenarios
    2. Edge cases
    3. Error conditions
    4. Integration scenarios

    Aim for 90%+ coverage.
    """

@mcp.prompt("debug_issue")
async def debug_issue_prompt(description: str) -> str:
    """Prompt for debugging an issue"""
    return f"""
    Debug the following issue: {description}

    Debugging process:
    1. Reproduce the issue
    2. Examine relevant code
    3. Check logs and error messages
    4. Identify root cause
    5. Propose and implement fix
    6. Verify fix works
    """
```

#### 4.4 Enhanced Tool Descriptions
**Modify:** All tool files in `src/ctxai/agent/tools/`

Add examples and constraints:
```python
class ReadFileTool(BaseTool):
    name = "read_file"
    description = """
    Read contents of a file from the filesystem.

    Parameters:
    - path (str): Absolute or relative file path
    - start_line (int, optional): Start reading from line N
    - end_line (int, optional): Stop reading at line N

    Returns:
    - content (str): File contents with line numbers

    Examples:
    - Read entire file: {"path": "main.py"}
    - Read lines 10-20: {"path": "main.py", "start_line": 10, "end_line": 20}

    Constraints:
    - File size limit: 10MB
    - Binary files: Not supported (use specific tools)

    Common Errors:
    - FileNotFoundError: File doesn't exist
    - PermissionError: No read access
    - UnicodeDecodeError: Binary file or encoding issue
    """
```

### Critical Files
- `src/ctxai/agent/core.py` - Planning integration
- `src/ctxai/agent/planners/` - Planning implementations (NEW)
- `src/ctxai/commands/server_command.py` - MCP resources & prompts
- `src/ctxai/agent/tools/*.py` - Enhanced descriptions

### Verification Steps
1. Test planning integration: Give multi-step task, verify plan created
2. Configure Claude Desktop with ctxai MCP server
3. Test MCP prompts in Claude Desktop (use `/analyze_codebase`)
4. Test MCP resources (access `file://` and `repo://` URIs)
5. Verify enhanced tool descriptions in LLM responses
6. Execute complex multi-step workflow with planning

---

## Phase 5: Export Features
**Duration:** 2-3 weeks | **Independent, can run parallel to Phase 3**

### Goals
- Convert repository to single text file
- Generate HTML code map with navigation
- Export in multiple formats (JSON, Markdown)
- Configurable limits (max 500 files)
- Rich metadata and statistics

### Key Deliverables

#### 5.1 Repository-to-Text Export
**New Files:**
- `src/ctxai/export/repo_to_text.py` - Main export logic
- `src/ctxai/export/formatters.py` - Format handlers (text, markdown, XML)
- `src/ctxai/commands/export_command.py` - CLI command

**CLI Interface:**
```bash
ctxai export text /path/to/repo \
  --output repo.txt \
  --include "*.py" "*.md" \
  --exclude "tests/*" "__pycache__" \
  --max-files 500 \
  --max-size 50MB \
  --format markdown \
  --with-tree

# Formats: text, markdown, xml, json
```

**Output Format (Markdown):**
```markdown
# Repository Export: ctxai
Generated: 2026-05-21 15:30:00
Files: 245 / 500 (limit)
Total Size: 2.3 MB
Languages: Python (200 files), Markdown (45 files)

## Directory Structure
```
ctxai/
├── src/
│   ├── ctxai/
│   │   ├── __init__.py
│   │   ├── app.py (854 lines)
│   │   └── agent/
│   │       ├── core.py (388 lines)
│   │       └── tools/
│   │           ├── file_ops.py
│   │           └── bash_tool.py
├── tests/
└── README.md
```

## Files

### File 1/245: src/ctxai/__init__.py
- **Path:** src/ctxai/__init__.py
- **Language:** Python
- **Lines:** 10
- **Size:** 250 bytes
- **Last Modified:** 2026-05-20 10:00:00

```python
<file contents>
```

---

### File 2/245: src/ctxai/app.py
...
```

**Features:**
- Respects .gitignore by default
- Size and file count limits
- Progress bar for large repos
- Include directory tree
- Summary statistics
- Multiple output formats

#### 5.2 HTML Code Map
**New Files:**
- `src/ctxai/export/html_codemap.py` - HTML generator
- `src/ctxai/export/templates/codemap.html` - Jinja2 template
- `src/ctxai/export/static/` - CSS, JS for interactive features

**CLI Interface:**
```bash
ctxai export html /path/to/repo \
  --output codemap.html \
  --max-files 500 \
  --theme dark \
  --with-stats \
  --with-search
```

**HTML Features:**
- **File Tree Navigation:** Collapsible tree view (left sidebar)
- **Syntax Highlighting:** Pygments for all languages
- **Search:** Full-text search across files (client-side)
- **Dark/Light Theme:** Toggle switch
- **Statistics Dashboard:**
  - Total files, lines, size
  - Language breakdown (pie chart)
  - File type distribution
  - Largest files
- **Code View:**
  - Line numbers
  - Copy button
  - Permalink to specific lines
  - Breadcrumb navigation

**Template Structure:**
```html
<!DOCTYPE html>
<html>
<head>
    <title>Code Map - {{ repo_name }}</title>
    <style>/* Dark theme CSS */</style>
</head>
<body>
    <nav class="sidebar">
        <div class="stats">
            <h3>Statistics</h3>
            <p>Files: {{ total_files }}</p>
            <p>Lines: {{ total_lines }}</p>
            <p>Size: {{ total_size }}</p>
        </div>
        <div class="tree">
            <!-- File tree here -->
        </div>
    </nav>
    <main class="content">
        <div class="search">
            <input type="search" placeholder="Search code...">
        </div>
        <div class="code-view">
            <!-- Code display here -->
        </div>
    </main>
    <script>/* Interactive JS */</script>
</body>
</html>
```

#### 5.3 JSON Export
**New File:** `src/ctxai/export/json_export.py`

**Schema:**
```json
{
  "repository": {
    "name": "ctxai",
    "path": "/path/to/ctxai",
    "generated_at": "2026-05-21T15:30:00Z",
    "stats": {
      "total_files": 245,
      "total_lines": 15000,
      "total_size_bytes": 2400000,
      "languages": {
        "Python": {"files": 200, "lines": 12000},
        "Markdown": {"files": 45, "lines": 3000}
      }
    }
  },
  "files": [
    {
      "path": "src/ctxai/app.py",
      "language": "python",
      "lines": 854,
      "size_bytes": 25000,
      "last_modified": "2026-05-20T10:00:00Z",
      "content": "...",
      "chunks": [
        {
          "type": "class",
          "name": "MyClass",
          "start_line": 10,
          "end_line": 50,
          "content": "..."
        }
      ]
    }
  ]
}
```

#### 5.4 Export Configuration
**New File:** `src/ctxai/export/config.py`

```python
@dataclass
class ExportConfig:
    max_files: int = 500
    max_total_size_mb: int = 50
    max_file_size_mb: int = 5
    include_patterns: list[str] = field(default_factory=list)
    exclude_patterns: list[str] = field(default_factory=list)
    include_binary: bool = False
    include_hidden: bool = False
    follow_symlinks: bool = False
    follow_gitignore: bool = True
    output_format: str = "markdown"  # text, markdown, json, html, xml

    def validate(self) -> None:
        """Validate configuration"""
        if self.max_files < 1 or self.max_files > 10000:
            raise ValueError("max_files must be between 1 and 10000")
        # ... other validations
```

**Modify:** `src/ctxai/commands/export_command.py`

```bash
ctxai export --help
# Subcommands:
#   text     Export repository to single text file
#   html     Export repository to HTML code map
#   json     Export repository to JSON
#   stats    Show repository statistics only
```

### Critical Files
- `src/ctxai/export/repo_to_text.py` - Text export engine (NEW)
- `src/ctxai/export/html_codemap.py` - HTML generator (NEW)
- `src/ctxai/export/json_export.py` - JSON export (NEW)
- `src/ctxai/export/templates/codemap.html` - HTML template (NEW)
- `src/ctxai/commands/export_command.py` - CLI commands (NEW)

### Verification Steps
1. Export ctxai itself: `ctxai export text . --output ctxai.md`
2. Verify Markdown format correct (headers, code blocks, tree)
3. Generate HTML: `ctxai export html . --output ctxai.html`
4. Open HTML in browser, test search and navigation
5. Export to JSON, validate schema
6. Test with large repo (500+ files), verify limits enforced
7. Test with binary files, verify exclusion

---

## Phase 6: Production Hardening
**Duration:** 2-3 weeks | **Depends on:** All previous phases

### Goals
- Configuration validation and security
- Rate limiting and resource quotas
- Security hardening (input sanitization, auth)
- Docker and Kubernetes deployment
- Comprehensive deployment documentation
- Observability (tracing, metrics)

### Key Deliverables

#### 6.1 Configuration Validation
**New Files:**
- `src/ctxai/validation.py` - Config validation
- `src/ctxai/schemas.py` - Pydantic schemas for all configs

**Validations:**
- API key format validation (regex, length)
- Provider availability checks (network connectivity)
- Resource limits validation (memory, CPU, disk)
- Tool configuration validation
- Path traversal prevention
- Model availability checks

**Example:**
```python
class ConfigValidator:
    def validate_api_keys(self, config: AgentConfig) -> list[ValidationError]:
        errors = []
        if config.provider == "openrouter" and not config.api_key:
            errors.append(ValidationError("OpenRouter requires API key"))
        if config.api_key and not self._is_valid_api_key(config.api_key):
            errors.append(ValidationError("Invalid API key format"))
        return errors

    def validate_resources(self, config: ServiceConfig) -> list[ValidationError]:
        # Check system resources
        if config.max_sessions > 100:
            errors.append(ValidationError("max_sessions too high (>100)"))
        # ... more checks
```

#### 6.2 Rate Limiting
**New File:** `src/ctxai/service/rate_limiter.py`

**Strategies:**
- Per-session rate limits (e.g., 10 requests/minute)
- Per-IP rate limits (e.g., 100 requests/hour)
- Token budget limits (e.g., 100K tokens/day per session)
- Concurrent request limits (e.g., 5 concurrent per session)

**Implementation:**
```python
class RateLimiter:
    def __init__(self, redis_client=None):
        self.backend = redis_client or InMemoryBackend()

    async def check_rate_limit(
        self,
        key: str,
        max_requests: int,
        window_seconds: int
    ) -> tuple[bool, dict]:
        """Check if rate limit exceeded"""
        count = await self.backend.increment(key, window_seconds)
        allowed = count <= max_requests
        return allowed, {
            "limit": max_requests,
            "remaining": max(0, max_requests - count),
            "reset_at": time.time() + window_seconds
        }
```

**Integration in API:**
```python
@app.post("/api/v1/sessions/{id}/messages")
async def send_message(session_id: str, message: str):
    # Rate limit check
    allowed, info = await rate_limiter.check_rate_limit(
        key=f"session:{session_id}",
        max_requests=10,
        window_seconds=60
    )
    if not allowed:
        raise HTTPException(429, detail="Rate limit exceeded", headers={
            "X-RateLimit-Limit": str(info["limit"]),
            "X-RateLimit-Remaining": str(info["remaining"]),
            "X-RateLimit-Reset": str(info["reset_at"])
        })
    # ... process message
```

#### 6.3 Security Hardening
**Files to Modify:**
- `src/ctxai/agent/tools/bash_tool.py` - Enhanced command filtering
  - Blocklist: `rm -rf /`, `:(){ :|:& };:`, `mkfs`, `dd if=/dev/zero`, etc.
  - Whitelist mode option
  - Sandbox execution (chroot, containers)

- `src/ctxai/agent/tools/file_ops.py` - Path traversal prevention
  - Validate all paths against working directory
  - Reject `..` in paths
  - Symlink detection and handling

- `src/ctxai/service/api_server.py` - API security
  - API key authentication (X-API-Key header)
  - CORS configuration (whitelist origins)
  - Request size limits (max 10MB)
  - Input sanitization
  - SQL injection prevention (use parameterized queries)

**New File:** `src/ctxai/security.py`

```python
class SecurityManager:
    def validate_file_path(self, path: Path, base_dir: Path) -> Path:
        """Ensure path is within base directory"""
        resolved = path.resolve()
        if not resolved.is_relative_to(base_dir):
            raise SecurityError("Path traversal attempt detected")
        return resolved

    def sanitize_bash_command(self, command: str) -> str:
        """Remove dangerous patterns from bash commands"""
        dangerous = [";", "|", "&", ">", "<", "`", "$()"]
        # ... sanitization logic
```

#### 6.4 Deployment Infrastructure
**New Files:**
- `Dockerfile` - Multi-stage container image
- `docker-compose.yml` - Complete stack with Redis
- `.dockerignore` - Exclude unnecessary files
- `k8s/deployment.yaml` - Kubernetes deployment
- `k8s/service.yaml` - Kubernetes service
- `k8s/ingress.yaml` - Ingress configuration
- `systemd/ctxai.service` - Systemd service unit

**Dockerfile:**
```dockerfile
# Stage 1: Builder
FROM python:3.13-slim AS builder
WORKDIR /app
COPY pyproject.toml .
RUN pip install --no-cache-dir uv && \
    uv pip install --system .

# Stage 2: Runtime
FROM python:3.13-slim
WORKDIR /app
COPY --from=builder /usr/local /usr/local
COPY src/ src/
EXPOSE 8000
CMD ["ctxai", "service", "start", "--port", "8000", "--host", "0.0.0.0"]
```

**docker-compose.yml:**
```yaml
version: '3.8'
services:
  ctxai:
    build: .
    ports:
      - "8000:8000"
    environment:
      - CTXAI_HOME=/data
      - OPENROUTER_API_KEY=${OPENROUTER_API_KEY}
    volumes:
      - ./data:/data
    depends_on:
      - redis

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
```

#### 6.5 Deployment Documentation
**New Documentation:**
- `docs/deployment/docker.md` - Docker deployment guide
  - Build image
  - Run container
  - Environment variables
  - Volume mounts
  - Health checks

- `docs/deployment/kubernetes.md` - Kubernetes deployment
  - Manifests
  - ConfigMaps and Secrets
  - Scaling strategy
  - Monitoring setup

- `docs/deployment/systemd.md` - Systemd service
  - Service unit file
  - Installation steps
  - Log management
  - Auto-restart configuration

- `docs/deployment/cloud.md` - Cloud providers
  - AWS (ECS, Fargate)
  - GCP (Cloud Run, GKE)
  - Azure (Container Instances, AKS)

#### 6.6 Observability
**New Files:**
- `src/ctxai/observability/tracing.py` - OpenTelemetry tracing
- `src/ctxai/observability/metrics.py` - Prometheus metrics (expanded from Phase 3)

**Tracing:**
```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider

tracer = trace.get_tracer(__name__)

@tracer.start_as_current_span("agent.process_message")
async def process_message(message: str) -> str:
    span = trace.get_current_span()
    span.set_attribute("message.length", len(message))
    # ... processing
```

**Metrics (Expanded):**
```python
# Existing metrics from Phase 3 +
# Cost tracking
ctxai_cost_usd_total{provider="openrouter", model="claude-3.5-sonnet"} 5.25

# Tool usage
ctxai_tool_calls_total{tool="ReadFileTool", status="success"} 1500

# Planning
ctxai_plans_created_total 50
ctxai_plan_steps_executed_total{status="success"} 250
```

#### 6.7 Graceful Degradation
**Features:**
- LLM provider fallback (already exists, enhance with health checks)
- Read-only mode when database unavailable
- Cached responses when LLM unavailable (TTL cache)
- Tool failure handling (mark tool unavailable, continue without it)

**Configuration:**
```toml
[resilience]
enable_provider_fallback = true
enable_read_only_mode = true
enable_response_cache = true
cache_ttl_seconds = 300

[[resilience.provider_chain]]
provider = "openrouter"
fallback = "anthropic"

[[resilience.provider_chain]]
provider = "anthropic"
fallback = "ollama"
```

### Critical Files
- `Dockerfile` - Container image (NEW)
- `docker-compose.yml` - Stack definition (NEW)
- `k8s/*.yaml` - Kubernetes manifests (NEW)
- `src/ctxai/validation.py` - Config validation (NEW)
- `src/ctxai/service/rate_limiter.py` - Rate limiting (NEW)
- `src/ctxai/security.py` - Security utilities (NEW)
- `docs/deployment/` - Deployment docs (NEW)

### Verification Steps
1. Build Docker image: `docker build -t ctxai:1.0.0 .`
2. Run container: `docker-compose up -d`
3. Run security scan: `bandit -r src/ctxai`, `trivy image ctxai:1.0.0`
4. Deploy to Kubernetes: `kubectl apply -f k8s/`
5. Load test: `k6 run loadtest.js` (1000 req/s for 5 minutes)
6. Chaos testing: Kill pods, verify recovery
7. Test rate limiting: Exceed limits, verify 429 responses
8. Monitor metrics: Grafana dashboard with Prometheus data
9. Test graceful degradation: Disable providers, verify fallback

---

## Dependency Matrix

```
Phase 1 (Testing & Stability)     → REQUIRED BY: All other phases
Phase 2 (Core Library)            → REQUIRED BY: Phase 3, 6
Phase 3 (Service)                 → INDEPENDENT (can parallel with 4, 5)
Phase 4 (MCP & Planning)          → DEPENDS ON: Phase 1
Phase 5 (Export)                  → INDEPENDENT (can parallel with 3, 4)
Phase 6 (Production)              → DEPENDS ON: All previous phases
```

**Recommended Execution Order:**
1. **Phase 1** (MUST complete first - foundation)
2. **Phase 2 + Phase 4** (parallel - both enhance core)
3. **Phase 3 + Phase 5** (parallel - independent features)
4. **Phase 6** (final - requires all features complete)

**Timeline:** ~14-18 weeks total (3.5-4.5 months)

---

## Success Metrics

### Phase 1 (Stability)
- ✅ Test coverage ≥ 80%
- ✅ All 15 existing tests pass
- ✅ Agent loop < 2s per iteration (p95)
- ✅ Zero critical bugs
- ✅ Memory stable in 100-message conversations

### Phase 2 (Core Library)
- ✅ ctxai-core installable standalone
- ✅ All 7 examples run successfully
- ✅ 3+ external projects using ctxai-core
- ✅ API documentation complete (100% coverage)
- ✅ Plugin system functional with sample plugin

### Phase 3 (Service)
- ✅ Service starts/stops gracefully
- ✅ 100+ concurrent sessions supported
- ✅ Session recovery after crash
- ✅ API response time < 500ms (p95)
- ✅ WebSocket streaming functional

### Phase 4 (MCP & Planning)
- ✅ Planning integrated into agent loop
- ✅ 4 MCP resources implemented
- ✅ 4 MCP prompts implemented
- ✅ Claude Desktop integration working
- ✅ Multi-step tasks executed via planning

### Phase 5 (Export)
- ✅ Repo-to-text export functional
- ✅ HTML code map generation working
- ✅ JSON export with valid schema
- ✅ 500-file limit enforced
- ✅ Export time < 30s for 500 files

### Phase 6 (Production)
- ✅ Docker image builds successfully
- ✅ Kubernetes deployment works
- ✅ Security scan passes (Bandit, Trivy)
- ✅ Load test: 1000 req/s sustained
- ✅ Observability: all metrics collected

---

## Risk Mitigation

### Risk: Breaking Changes During Refactoring
**Mitigation:**
- Phase 1 comprehensive tests FIRST
- Feature flags for new functionality
- Semantic versioning (1.0.0 release)
- Maintain backward compatibility layer
- Extensive regression testing

### Risk: Performance Degradation
**Mitigation:**
- Performance tests in Phase 1 (baseline)
- Continuous benchmarking after changes
- Profile critical paths (agent loop, tool execution)
- Optimize hot paths identified via profiling
- Load testing before each release

### Risk: API Instability
**Mitigation:**
- API versioning (`/api/v1/`, `/api/v2/`)
- Deprecation warnings (6-month notice)
- Changelog for all API changes
- Backward compatibility guarantee for v1
- Clear migration guides

### Risk: Deployment Complexity
**Mitigation:**
- Docker images for easy deployment
- Comprehensive deployment documentation
- Example configurations for common scenarios
- Automated deployment scripts
- Health checks for early detection

### Risk: Security Vulnerabilities
**Mitigation:**
- Security-first design (input validation, sandboxing)
- Regular security scans (Bandit, Trivy)
- Dependency updates (automated via Dependabot)
- Penetration testing before 1.0.0 release
- Security disclosure policy

---

## Post-1.0.0 Roadmap (Future Enhancements)

### Incremental Indexing
- File watcher for automatic re-indexing
- Delta updates (only changed files)
- Index versioning and migration

### Multi-Repository Support
- Index multiple repos simultaneously
- Cross-repo search
- Workspace management

### Advanced Planning
- Hierarchical multi-level plans
- Plan visualization (Gantt charts)
- Plan templates for common tasks

### VS Code Extension
- Inline agent assistance
- Code suggestions
- Real-time indexing

### Enterprise Features
- Multi-tenancy
- RBAC (Role-Based Access Control)
- Audit logging
- SSO integration (SAML, OAuth2)

---

## Conclusion

This roadmap transforms **ctxai** from a promising v0.0.2 prototype into a production-ready, enterprise-grade coding agent platform. Each phase delivers working functionality incrementally, ensuring continuous value delivery while maintaining stability.

**Key Differentiators:**
1. **Exportable Harness** - ctxai-core enables building custom agents
2. **Long-Running Service** - Persistent sessions with API access
3. **Complete MCP Integration** - Tools, resources, prompts for Claude Desktop
4. **Export Capabilities** - Repo-to-text, HTML code maps for agent consumption
5. **Production-Ready** - Tests, monitoring, deployment, security

**Next Steps:**
1. Review and approve this plan
2. Begin Phase 1: Stability & Testing
3. Establish CI/CD pipeline
4. Set up project tracking (GitHub Projects, Linear, etc.)
5. Begin implementation!
