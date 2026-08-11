# Repository Guidelines

Local-first coding agent + semantic code search (`ctxai`). Python src-layout package managed with uv. Python 3.10+ syntax (dev and CI run 3.13; ruff targets py310). Run everything from the repo root via `uv run`.

## Project Overview

`ctxai` is a CLI that (1) indexes a codebase into a local ChromaDB vector store and answers semantic queries, and (2) runs an autonomous LLM coding agent (tool use, evidence-backed planning, exact-action approval) inside the user's repository. It ships as a PyPI package (`ctxai.__main__:app` console script) with an optional MCP stdio server, a FastHTML query dashboard, and OAuth/device-flow auth for openrouter and github-copilot.

## Architecture & Data Flow

- **Indexing pipeline**: `traversal.py` (gitignore-aware walk, symlink/size guards, `size_validator.py`) → `chunking.py` (tree-sitter parsing) → `embeddings.py` (`EmbeddingsFactory`; `BaseEmbeddingProvider` interface; local default, openai, huggingface) → `vector_store.py` (ChromaDB under `.ctxai/indexes/<name>`) → `index_operations.py` + `index_manifest.py` (verified writes; manifest published only after successful write — no-op reindex, delete propagation, fatal-write rollback are tested in `tests/e2e/test_vs01_trustworthy_index.py`).
- **Retrieval**: `repository_context.py` — `discover_repository_indexes`, `ContextAssembler` (dedup + token budget), `HybridRetriever` (hybrid beats vector-only per vs03 tests). `retrieval_eval.py` + `tests/fixtures/retrieval_benchmark.json` provide retrieval metrics (recall@5, MRR).
- **Agent**: `agent/core.py` `Agent` + `AgentLoopConfig` (dataclass carrying `llm_provider`, `tool_registry`, `agent_config`, `working_directory`, `available_indexes`, `planning_enabled`, `require_user_approval`, `max_iterations`, `approval_callback`). Loop = provider `chat()` with tools; complex tasks route through the `submit_plan` tool (`PLAN_TOOL_SCHEMA` in core.py) → approval (`workflow.py` `ApprovalCallback`, `format_approval_prompt`, `TaskState`, `FailureKind`, `discover_verification_commands`; `planning.py`). Context in `context.py` (`ConversationContext`); sessions persisted by `sessions.py` `SessionStore` with secret redaction. System prompts in `prompts.py`.
- **LLM providers** (`agent/llm/`): `base.py` (`BaseLLMProvider`, `Message`, `LLMResponse`, `ToolCall`), `contract.py` (`PROVIDER_SPECS` — drives `docs/PROVIDER_COMPATIBILITY.md`), `factory.py`, `fallback.py` (`FallbackProvider`, boundary crossing opt-in). Providers: anthropic, openai, openrouter, github-copilot, ollama, custom, nvidia.
- **Tools** (`agent/tools/`): `file_ops.py` (Read/Write/Edit/List/Glob/Grep), `bash_tool.py` (capability/command-policy enforcement), `code_search.py` (`SemanticSearchTool`), `git_tools.py` (read-only status/diff/log), `registry.py` (`ToolRegistry`), `execution.py` (`ToolExecutionContext.for_project(...)` — `allow_outside_project`, timeouts).
- **CLI** (`app.py`, Typer): bare `ctxai` (no args) starts interactive chat (`start_chat`). Commands: `index`, `query`, `server` (MCP, stdio), `dashboard`, `config`, `chat`, `code`, `login`, `logout`, plus sub-apps `indexes` (list/info/doctor/delete) and `models` (list/search/info/pull/library). `code` is a one-shot task that **hardcodes AnthropicProvider** and requires `ANTHROPIC_API_KEY`; `chat` uses the configured default provider. Entry: `__main__.py`.
- **MCP**: `commands/server_command.py` — FastMCP stdio server, 4 tools (list_indexes, index_codebase, query_codebase, get_index_stats), progress notifications, cooperative cancel/timeout, stable error codes (`mcp_protocol.py` envelope: invalid_input/not_found/cancelled/timeout/index_failed/query_failed/storage_failed/internal_error). `server.py` is a **separate** FastAPI health stub and the only `load_dotenv()` caller — do not confuse it with the MCP server.
- **Auth** (`auth/`): `keystore.py` (credentials in `~/.ctxai/keys.json`, 0600), `oauth_pkce.py` (OpenRouter PKCE), `github_copilot.py` (device flow). OAuth creds are stored via keystore, not env vars.
- **Dashboard**: `commands/dashboard_command.py` — FastHTML, binds 127.0.0.1 by default; remote requires `--host` + `--allow-remote`.

## Key Directories

- `src/ctxai/` — package root: CLI (`app.py`), pipeline (`traversal.py`, `chunking.py`, `embeddings.py`, `vector_store.py`), verified index writes (`index_operations.py`, `index_manifest.py`), retrieval assembly (`repository_context.py`), config (`config.py`), helpers (`utils.py`).
- `src/ctxai/commands/` — one module per CLI command.
- `src/ctxai/agent/` — agent loop (`core.py`, `workflow.py`, `planning.py`, `context.py`, `sessions.py`, `prompts.py`, `repomap.py`, `theme.py`); `agent/llm/` providers + factory + fallback + contract; `agent/tools/` tool implementations + registry + execution context.
- `src/ctxai/auth/` — keystore, OAuth PKCE, copilot device flow.
- `tests/` — unit/integration; `tests/e2e/` — end-to-end + acceptance specs (VS-01..VS-09); `tests/mocks/` (`MockLLMProvider`, `MockEmbeddingProvider`); `tests/fixtures/` (sample code, retrieval benchmark).
- `docs/` — current docs (MCP_SERVER, QUERY_DASHBOARD, GITHUB_COPILOT_AUTH, OAUTH_AUTHENTICATION, PROVIDER_COMPATIBILITY). `docs/IMPLEMENTATION_SUMMARY.md`, `MCP_IMPLEMENTATION_SUMMARY.md`, `MCP_REFACTORING.md` are historical.
- `examples/`, `scripts/` (`setup_providers.py`) — usage demos/helper.
- Root `*.md` are mostly historical/aspirational (CODING.md, plan*.md, IMPLEMENTATION*.md, COMPLETE_SUMMARY.md, PROOF.md, KNOWLEDGE.md, QUICKSTART.md, AI_AGENT.md). Trust source code and tests over those; `README.md`, `docs/`, `ROADMAP.md`, `plan.md` are the current references.

## Development Commands

```bash
uv sync --locked --all-extras --all-groups   # match CI exactly (default groups dev/mcp/server)
uv run pytest                                # full suite INCLUDING tests/e2e (testpaths=tests)
uv run pytest tests/ --ignore=tests/e2e/ -m "not slow" --cov=ctxai --cov-report=term-missing  # unit-only, as pr-gate
uv run pytest tests/e2e/                    # e2e only
uv run pytest tests/test_indexing.py::test_x # single test
uv run ruff check . --fix                   # lint (line-length 120, select E,F,I,UP)
uv run ruff format .                        # format (CI gates with --check)
uv run mypy                                 # type-check — ONLY the 6 files in [tool.mypy] files
uv run bandit -r src/                       # security scan (CI)
uv build && uv run twine check dist/*       # packaging gate
```

Run the tool itself: `uv run ctxai` (interactive chat), `ctxai index <path> <name>`, `ctxai query <name> "question"`, `ctxai code "task"`, `ctxai login openrouter`.

## Code Conventions & Common Patterns

- **Python 3.10 syntax**: builtin generics (`list[str]`), `X | None` unions, `from __future__ import annotations` in newer files. No 3.11+ syntax — CI runs 3.10.
- **Formatting**: ruff (120 cols, `E,F,I,UP`); `ruff format` for style. `.editorconfig` uses 4-space indent. Every public function has an Args/Returns docstring (project convention).
- **Data models**: dataclasses (not pydantic) for internal config/state — `EmbeddingConfig`/`IndexConfig`/`ProviderConfig` (config.py), `AgentLoopConfig`, `TaskRun`. pydantic/pydantic-ai are dependencies but internal models are dataclasses.
- **CLI**: Typer commands registered in `app.py` as thin wrappers delegating to `commands/<name>_command.py`; options via `typer.Option(...)` with help text; errors exit `raise typer.Exit(code=1)`.
- **Terminal output**: Rich (`Console(legacy_windows=False)`, tables/panels/markdown); the MCP server logs to stderr. Minimal `logging` usage.
- **Error handling**: typed exceptions + `FailureKind` taxonomy in the workflow; MCP surfaces errors through the envelope code table; user-facing messages via `console.print("[red]...[/red]")`.
- **Async**: agent loop and MCP handlers are async (`Agent.process_message`, `asyncio.run` inside CLI commands); LLM providers expose sync `chat`/`stream_chat`. Tests use pytest-asyncio `asyncio_mode = auto`.
- **DI pattern**: interfaces (`BaseLLMProvider`, `BaseEmbeddingProvider`) implemented by providers/adapters; construction via factory (`EmbeddingsFactory`, LLM provider factory) with config objects injected; `AgentLoopConfig` carries the full dependency set. Tests seam-patch factories (`patch_embeddings_factory`) or module symbols.
- **State**: config in `.ctxai/config.toml` — global defaults (`~/.ctxai/config.toml` or `$CTXAI_HOME/config.toml`) merged with project overrides (`<project>/.ctxai/config.toml`), project wins key by key; credentials in global home `keys.json`; indexes in `get_ctxai_home()/indexes`; sessions via `SessionStore`.
- **Untouchable/experimental**: `config_new.py` is an unused alternate TOML config draft (nothing imports it — don't build on it). `agent/architect_editor.py` architect/editor mode is experimental and disabled pending benchmark evidence; the validated path is single-agent planning + exact-action approval. Don't build around either.

## Important Files

- Entry points: `src/ctxai/__main__.py` (console script `ctxai = "ctxai.__main__:app"`), `src/ctxai/app.py`.
- Config: `pyproject.toml` (deps, extras openai/dashboard/mcp/all, dev group, ruff/mypy/uv config), `pytest.ini` (ALL pytest + coverage config), `uv.lock`, `.python-version` (3.13), `.env.example`.
- CI: `.github/workflows/ci.yml` (quality: ruff/mypy; tests matrix 3.10/3.13 with coverage; package build+twine), `pr-gate.yml` (lint + unit `--ignore=tests/e2e/`, e2e job, bandit), `release.yml` (tag → validate/build/release/PyPI).
- Key modules: `agent/core.py`, `agent/workflow.py`, `agent/llm/contract.py`, `agent/llm/factory.py`, `agent/tools/execution.py`, `traversal.py`, `chunking.py`, `embeddings.py`, `vector_store.py`, `index_operations.py`, `index_manifest.py`, `repository_context.py`, `config.py`, `commands/server_command.py`, `auth/keystore.py`.

## Runtime/Tooling Preferences

- **Python ≥3.10** (`requires-python`), dev/CI on 3.13; hatchling build backend.
- **uv only**: `uv sync --locked --all-extras --all-groups` (default groups dev/mcp/server via `[tool.uv] default-groups`). Never `pip install` into the venv.
- **Pinned constraints**: `mcp>=1.16,<1.17` (extra `mcp`); FastHTML (`dashboard` extra); ChromaDB for vectors; tree-sitter + `tree-sitter-language-pack` for parsing; sentence-transformers for local embeddings; `typer`, `rich`, `tomlkit`, `prompt-toolkit`.
- **mypy covers only 6 explicit files** (`[tool.mypy] files`): `agent/llm/contract.py`, `agent/tools/execution.py`, `agent/workflow.py`, `index_manifest.py`, `repository_context.py`, `retrieval_eval.py`. New files aren't type-checked unless added there.
- OAuth credentials live in the keystore (`auth/keystore.py`), not env vars; `server.py` is the only `load_dotenv()` caller. `CTXAI_HOME` overrides the project `.ctxai` dir.

## Testing & QA

- **Framework**: pytest + pytest-asyncio (`asyncio_mode = auto` — async tests need no decorator) + pytest-cov. pytest-mock and pytest-timeout are installed but unused.
- **Config** (`pytest.ini`): `testpaths = tests`; strict markers `e2e, slow, unit, integration, mcp, agent, indexing`; `filterwarnings = error` (Deprecation/Pending/User ignored) — a new warning fails the suite; addopts `-v --showlocals --tb=short`.
- **Full suite runs e2e**: `tests/e2e/` is inside `testpaths`, so plain `uv run pytest` runs it. Unit-only requires `--ignore=tests/e2e/`. Marker filtering is unreliable: `slow`/`unit`/`integration` are declared but never used, and some e2e files (`vs03`, `vs05`, `vs07`) lack the `e2e` marker — never use `-m "not e2e"` to exclude them.
- **Mocks**: `tests/mocks/mock_llm.py` (`MockLLMProvider` — scripted responses, `call_count`/`call_history` assertions, `create_mock_response` helper) and `tests/mocks/mock_embeddings.py` (MD5-seeded deterministic 384-dim vectors). Provider name `"mock"`. e2e patches `EmbeddingsFactory.create` via the `patch_embeddings_factory` fixture; real Agent loop, real tools, real Chroma, real MCP transport run underneath.
- **Isolation**: autouse `reset_environment_variables` fixture in `tests/conftest.py` snapshots/restores `os.environ` per test; per-test temp dirs. Prefer built-in `tmp_path` for new tests; `tests/conftest.py` `temp_dir` fixture and direct `tempfile.TemporaryDirectory()` (legacy `test_indexing.py`) also exist.
- **Coverage**: opt-in via CI flags (not in addopts); measures `src/ctxai`. pr-gate uploads XML to Codecov; `ci.yml` runs `--cov` on 3.10 + 3.13.
- **Gotchas**: root-level `test_*.py` files (`test_agent.py`, `test_copilot.py`, `test_oauth.py`, `test_openrouter.py`, `test_quick.py`) are **not collected** and some hit live APIs — don't run or extend them. `--strict-markers` errors on undeclared markers — add new ones to `pytest.ini`. MCP tests guard with `pytest.importorskip("mcp")`. `test_retrieval_eval.py` hard-asserts benchmark invariants (20 queries, recall@5 = 1.0, mrr ≥ 0.75) — update it when touching retrieval. Keep tests 3.10-compatible and warning-free.
