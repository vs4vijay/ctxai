# ctxai Product Plan

Last reviewed: 2026-07-23

This document is the product and engineering source of truth for ctxai. It organizes the work as vertical slices: each slice must deliver a complete user outcome across interface, domain logic, storage or integrations, safety, documentation, and tests.

## Product direction

ctxai should become the local-first coding agent that builds durable semantic understanding of a repository before it acts. It combines syntax-aware indexing, semantic retrieval, repository structure, and an autonomous tool loop so that users can search, understand, and safely change a codebase without repeatedly sending the entire repository to a model.

The primary workflow is:

```text
open repository
  -> discover or build its code index
  -> retrieve relevant code and structural context
  -> explain or plan the requested change
  -> obtain approval when the action requires it
  -> edit and run targeted verification
  -> report evidence, changes, and remaining risks
```

## Unique selling proposition

**Short version:** ctxai is a local-first coding agent with a persistent, syntax-aware semantic memory of your codebase.

**Positioning:** Most coding agents begin every task by rediscovering the repository through filenames, keyword search, or a large prompt. ctxai creates a reusable code intelligence layer from tree-sitter structure, semantic embeddings, repository maps, and metadata, then makes that context available to any supported model through CLI, chat, Python, or MCP.

The defensible differentiation should be:

1. **Durable repository understanding** — index once, reuse across tasks and model providers.
2. **Local-first privacy and cost control** — local embeddings, local vector storage, and optional local LLMs.
3. **Model independence** — the repository intelligence belongs to the user rather than one model vendor.
4. **Structure-aware retrieval** — retrieve functions, classes, symbols, and related context rather than arbitrary text fragments.
5. **Evidence-driven changes** — connect every plan and edit to retrieved code, tool results, and verification output.

Avoid positioning ctxai as only “a multi-provider chat CLI.” Provider choice is useful, but persistent code intelligence is the product advantage.

## Status legend

- **Validated** — implemented and covered by passing end-to-end acceptance tests.
- **Functional** — usable implementation exists, but the complete slice is not yet validated.
- **Partial** — important components exist, but the user outcome is incomplete or disconnected.
- **Planned** — approved direction with no complete implementation.
- **Exploratory** — requires product or technical validation before commitment.

## Current baseline

At this review, the CLI exposes indexing, querying, MCP server, dashboard, configuration, chat, one-shot coding, authentication, and model management. The full supported automated suite has 61 tests and all 61 pass. VS-01 through VS-07 acceptance suites pass.

This makes ctxai a broad alpha: the main components exist, but no autonomous change workflow should be called production-ready until the first four slices below are validated.

## Vertical slices

### VS-01: Trustworthy local index and query

**User outcome:** A user can index a repository, close the process, reopen ctxai, and retrieve relevant code with stable metadata.

**Status:** Validated (2026-07-22).

**Exists today:**

- Gitignore-aware traversal and include/exclude filters.
- Tree-sitter parsing and multi-language code chunks.
- Local and cloud embedding abstractions.
- Persistent ChromaDB vector store.
- CLI index/query flows and rich output.
- Unit coverage for indexing and querying.

**Validated implementation:**

- Canonical per-index storage paths persist across fresh `VectorStore` instances and operating-system processes.
- Typed storage failures propagate and prevent manifest publication or successful index status.
- Atomic schema-versioned manifests record repository root/revision, embedding identity, timestamps, and file/chunk state.
- File hashes and deterministic chunk IDs skip unchanged embedding work, replace changed chunks, and delete stale chunks.
- `ctxai indexes list|info|delete|doctor` provides lifecycle and integrity management.
- A 20-question retrieval fixture records Recall@5 and MRR baselines.

**Exit criteria:**

- Index -> process restart -> query passes on Python and JavaScript fixtures.
- Interrupted or failed writes cannot report a successful index.
- Re-indexing an unchanged repository performs no embedding work.
- Changed and deleted files update the index correctly.
- Retrieval quality has a recorded baseline such as Recall@5 or MRR.

### VS-02: Safe repository tools

**User outcome:** The agent can inspect and modify files and run commands inside the selected repository without silently escaping its boundary.

**Status:** Validated (2026-07-22).

**Exists today:**

- Read, write, edit, list, glob, grep, semantic-search, and bash tools.
- File-size limits, command timeout, blocked-command configuration, and tool registry.
- Structured tool schemas and results.

**Validated implementation:**

- A shared `ToolExecutionContext` carries the canonical project root, capabilities, environment, timeout, request ID, approval callback, and audit log.
- Every filesystem tool performs canonical containment checks and rejects path traversal and symlink escapes.
- Read, workspace-write, command, network, destructive, and outside-project capabilities are independently enforced.
- Command execution uses parsed argument vectors without a shell, rejects shell operators, and gates network, destructive, inline-code, and mutating Git operations by capability.
- File writes and edits return unified diffs and append request-linked mutation records to the shared audit log.
- Read-only Git status, diff, and log tools are project-rooted and registered in normal chat and one-shot coding tool sets; mutating Git tools remain separately gated.
- Dedicated VS-02 acceptance tests cover filesystem escapes, explicit outside access, capability and command denials, mutation diffs/auditing, and read-only Git behavior.

**Exit criteria:**

- Path traversal and symlink escape tests pass on every filesystem tool.
- Writes outside the project fail unless an explicit capability permits them.
- Dangerous commands are rejected by policy rather than a small blacklist alone.
- Every edit has a reviewable diff and every mutation is represented in the session log.

### VS-03: Grounded repository understanding

**User outcome:** A user asks a repository question and receives an answer grounded in the most relevant symbols, files, and relationships, with inspectable evidence.

**Status:** Validated (2026-07-22).

**Exists today:**

- Semantic search tool.
- Repository-map generation.
- Conversation system prompt with repository information.
- Code chunk metadata with file and line locations.

**Validated implementation:**

- Manifest-based discovery selects only healthy indexes whose canonical repository root matches the current repository; chat and one-shot coding expose those indexes automatically.
- Hybrid reciprocal-rank fusion combines semantic vectors, lexical matches, indexed symbol names, and repository-structure signals.
- A context assembler ranks and deduplicates chunks, emits `file:start-end` evidence, and enforces a configured approximate token budget.
- Debug retrieval explains each selected item's contributing ranks.
- VS-03 end-to-end acceptance tests verify repository matching, grounded citations, bounded context, deduplication, and hybrid ranking quality against vector-only retrieval.

**Exit criteria:**

- Chat uses the correct repository index without manual index-name entry.
- Answers cite source files and line locations.
- Hybrid retrieval beats the vector-only baseline on the evaluation set.
- Context selection remains within a configured token budget.

**Verification evidence (2026-07-22):**

- `uv run pytest -q tests/e2e/test_vs03_grounded_repository_understanding.py` — 2 passed.
- `uv run pytest -q` — 36 passed, 5 known VS-06 MCP compatibility failures; no VS-01, VS-02, or VS-03 failures.

### VS-04: Verified one-shot code change

**User outcome:** A user requests a bounded change; ctxai finds the relevant code, proposes or executes the change safely, runs focused checks, and reports the result.

**Status:** Validated (2026-07-22).

**Exists today:**

- One-shot `code` command and iterative agent loop.
- LLM tool calling, error recovery prompt, repeated-call detection, and iteration cap.
- File and shell tools needed for a basic edit-and-test workflow.

**Validated implementation:**

- A deterministic task ledger records understand, retrieve, plan, approve, execute, verify, summarize, and failed transitions.
- Existing files must be read successfully before overwrite or edit; mutation results supply reviewable diffs before completion.
- Planning and approval configuration now affects execution, with approvals bound to the exact mutation tool call and target.
- Conventional project markers provide focused verification suggestions, and a mutated task cannot report success without a successful command check.
- Tool failures are classified as recoverable errors, policy denials, test failures, infrastructure failures, approval denials, or incomplete workflows.
- Stable final reports are built from observed mutations and command results and contain status, changed files, checks, model outcome, and remaining risks.
- Mock-LLM acceptance scenarios cover read-only answers, one-file edits, multi-file edits, failed checks, and approval denial without live APIs.

**Exit criteria:**

- All agent workflow e2e tests pass without live model APIs.
- A task cannot report success after failed required checks.
- Unauthorized mutations are impossible through registered tools.
- Final reports can be checked against the actual diff and command results.

**Verification evidence (2026-07-22):**

- `uv run pytest -q tests/e2e/test_vs04_verified_code_change.py tests/e2e/test_e2e_agent_workflow.py` — 10 passed.
- `uv run pytest -q` — 41 passed, 5 known VS-06 MCP compatibility failures; no VS-01, VS-02, VS-03, or VS-04 failures.
- `uv run python -m compileall -q src tests` and `git diff --check` — passed.

### VS-05: Interactive coding session

**User outcome:** A user can hold a reliable multi-turn session, change models, inspect context and tools, and continue work without losing or corrupting state.

**Status:** Validated (2026-07-22).

**Exists today:**

- Rich interactive chat UI and slash commands.
- Conversation history and truncation.
- Provider/model switching.
- Message queue and response streaming interface.

**Validated implementation:**

- Chat, MCP, and dashboard dependencies for every advertised command are declared directly and import successfully from an isolated wheel installation.
- Repository-scoped, schema-versioned sessions support explicit save, resume, clear, and Markdown export controls plus atomic auto-save after completed turns.
- Credentials are recursively redacted by secret-bearing field names and common inline credential formats before persistence or export.
- Context compaction preserves the system prompt and summarizes older requests, decisions, changes, failures, verification results, risks, and open tasks within a bounded summary.
- A normalized provider capability model covers tools, streaming, images, structured output, and context size; the agent omits tool schemas for providers that cannot accept them.
- Terminal-independent mock-provider acceptance tests cover multi-turn continuity, model replacement, portable message serialization, durable sessions, repository boundaries, and redaction.

**Exit criteria:**

- Clean basic and `all` installations expose only commands whose dependencies are satisfied.
- A session can be saved, resumed, cleared, and exported.
- Model switching preserves valid conversation/tool-call structure.

**Verification evidence (2026-07-22):**

- `uv run pytest -q tests/e2e/test_vs05_interactive_coding_session.py` — 4 passed.
- Isolated installation of `dist/ctxai-0.0.2-py3-none-any.whl` imported chat, dashboard, MCP server, and `prompt_toolkit` successfully.
- `uv run pytest -q` — 45 passed, 5 known VS-06 MCP compatibility failures; no VS-01 through VS-05 failures.
- `uv run ruff check ...`, `uv run python -m compileall -q src tests`, and `git diff --check` — passed.

### VS-06: MCP code intelligence service

**User outcome:** An MCP client can list, build, inspect, and query ctxai indexes with stable schemas and helpful failures.

**Status:** Validated (2026-07-23).

**Exists today:**

- MCP server with list, index, query, and statistics tools.
- Stdio integration and documentation.
- Unit-level MCP coverage.

**Validated implementation:**

- The supported MCP Python SDK range is bounded to `>=1.16,<1.17` in base, optional, development, and lock-file dependencies.
- Protocol-level acceptance tests initialize a real `ClientSession` over in-memory MCP transport, discover every tool, and invoke list, index, query, and statistics workflows.
- Every tool returns a versioned `1.0` structured result envelope with stable success data or deterministic error codes.
- Index names and inputs are validated before storage or embedding-provider access, including traversal-resistant index lookup.
- Indexing publishes progress notifications across discovery, chunking, embedding, storage, and manifest stages.
- Indexing supports a bounded timeout and cooperative cancellation at safe boundaries; once storage commit begins, manifest publication completes to preserve index consistency.
- MCP documentation describes the supported SDK range, result envelope, error codes, progress, timeout, and cancellation contract.

**Exit criteria:**

- A real client can discover and invoke every tool in CI.
- Compatibility is verified for the documented MCP version range.
- Cancellation and invalid-input scenarios return deterministic errors.

**Verification evidence (2026-07-23):**

- `.venv/bin/pytest -q tests/e2e/test_e2e_mcp_server.py` — 5 passed through a real MCP client transport.
- `.venv/bin/pytest -q` — 49 passed; the former five VS-06 failures are resolved.
- Targeted Ruff checks for all VS-06 implementation and test files passed.
- Repository-wide Ruff still reports 66 pre-existing errors in unrelated legacy UI/model modules; this is not a VS-06 regression.

### VS-07: Provider-independent execution

**User outcome:** A user can choose a supported cloud or local model and receive consistent tool behavior, or a precise explanation of unsupported capabilities.

**Status:** Validated (2026-07-23).

**Exists today:**

- OpenRouter, GitHub Copilot, Ollama, Anthropic, OpenAI, and custom provider implementations.
- Provider factory, presets, OAuth/keystore support, and model commands.

**Validated implementation:**

- A shared provider contract normalizes messages, capabilities, usage-bearing responses, errors, streaming, and pre-request cancellation behavior.
- Unsupported tools, streaming, images, and structured-output requests are rejected before any provider transport is called.
- OpenAI and Ollama now satisfy the concrete provider interface and normalize structured agent messages before transport.
- Provider metadata identifies local/cloud boundaries, transport type, capabilities, and whether model information is dynamic, cached, static, or endpoint-defined.
- Fallback execution is disabled by default, records every attempt and outcome, and refuses to cross a local/cloud privacy boundary without explicit opt-in.
- A generated compatibility matrix is published from the executable provider metadata.
- A parameterized, non-network conformance suite covers every advertised provider plus cancellation, capability rejection, fallback observability, configuration persistence, and boundary enforcement.

**Exit criteria:**

- Every advertised provider passes the same non-network contract suite.
- Unsupported tool features fail before a request is sent.
- Fallback never silently crosses a local/cloud or cost boundary.

**Verification evidence (2026-07-23):**

- `uv run pytest -q tests/e2e/test_vs07_provider_independent_execution.py` — 12 passed.
- `uv run pytest -q` — 61 passed, including all VS-01 through VS-07 acceptance suites.
- Targeted Ruff checks for all VS-07 implementation and test files, `uv run python -m compileall -q src tests`, and `git diff --check` — passed.

### VS-08: Deliberate planning and approval

**User outcome:** For complex or risky changes, ctxai produces an evidence-backed plan, asks for approval at the right boundary, executes it, and updates progress.

**Status:** Validated (2026-07-23).

**Exists today:**

- Plan, step, status, dependency, and executor data structures.
- Architect/editor provider pairing and a planning prototype.

**Validated implementation:**

- The main single-model agent loop exposes a structured `submit_plan` contract for task-specific
  actions, exact tool parameters, repository evidence, reasoning, and measurable completion criteria.
- A deterministic scope, uncertainty, and risk policy requires structured plans for complex or risky
  requests while preserving the validated direct path for bounded low-risk changes.
- Plan citations must use `file:start-end` evidence from files successfully inspected during the current
  run; uninspected or malformed evidence is rejected before execution.
- Planned actions are matched against exact tool calls, and their pending, in-progress, completed, or
  failed status is derived from observed execution results rather than model claims.
- Approval callbacks receive the exact command or a preview unified diff and target for each mutation;
  approvals and denials are recorded per action.
- Interactive streaming uses the same verified task loop, preventing UI mode from bypassing planning,
  approval, mutation, or verification policy.
- Architect/editor mode is disabled pending benchmark evidence instead of silently using only the editor
  or publishing unmeasured cost-saving claims.

**Exit criteria:**

- Complex tasks produce task-specific structured plans.
- Approval decisions bind to the exact proposed action.
- Progress reflects actual tool and verification outcomes.
- Architect/editor remains only if measured results justify its complexity and claims.

**Verification evidence (2026-07-23):**

- `.venv/bin/pytest -q tests/e2e/test_vs08_deliberate_planning.py tests/e2e/test_vs04_verified_code_change.py`
  — 8 passed.
- `.venv/bin/pytest -q` — 64 passed, including all VS-01 through VS-08 acceptance suites.
- Targeted Ruff checks for the VS-08 agent, workflow, prompt, and acceptance-test files passed.
- Repository-wide Ruff continues to report the pre-existing legacy UI/app errors documented in VS-06;
  no VS-08 implementation file has a Ruff failure.

### VS-09: Web dashboard and index operations

**User outcome:** A user can inspect index health, search code, review retrieval evidence, and manage indexes from a browser.

**Status:** Validated (2026-07-23).

**Exists today:**

- FastHTML dashboard, index views, statistics, search, chunk browsing, and configuration display.

**Validated implementation:**

- A shared `IndexOperations` application service owns validated list, inspect, query, chunk, and delete
  behavior; CLI lifecycle commands and the dashboard use the same manifest and health semantics.
- The index overview prominently shows integrity, Git/file freshness, schema version, embedding identity,
  and storage chunk count.
- Query results include inspectable `file:start-end` evidence, similarity, and bounded source previews.
- Browser routes cover index listing, inspection, grounded query, and explicit per-index deletion, while
  index-name validation and the ASGI router prevent traversal outside index storage.
- Dashboard app construction is separate from server startup, enabling terminal-independent browser
  acceptance tests through a real ASGI client.
- The server binds to `127.0.0.1` by default. Non-loopback hosts are rejected unless the user explicitly
  supplies `--allow-remote`; startup states that the dashboard has no authentication or TLS and is only
  appropriate on a trusted network.

**Exit criteria:**

- Core dashboard workflows pass browser tests.
- CLI, MCP, and dashboard return consistent results for the same index/query.
- Remote exposure requires an explicit secure configuration.

**Verification evidence (2026-07-23):**

- `.venv/bin/pytest -q tests/e2e/test_vs09_dashboard_operations.py` — 3 passed.
- `.venv/bin/pytest -q` — 67 passed, including all VS-01 through VS-09 acceptance suites.
- Targeted Ruff checks for the VS-09 service, dashboard, CLI adapter, and acceptance tests passed.
- `.venv/bin/python -m compileall -q src tests` and `git diff --check` passed.

## Recommended additions

These additions strengthen the USP rather than expanding sideways:

### 1. Code intelligence graph

Build a lightweight symbol and relationship graph from tree-sitter and language metadata: definitions, imports, calls, inheritance, tests, and references. Use it to expand semantic hits into coherent implementation context. This is a stronger differentiator than adding more generic agent tools.

### 2. Retrieval evaluation and observability

Create a small benchmark framework that records query, expected symbols/files, retrieved context, rank, latency, and token count. Expose a `ctxai eval retrieval` workflow and a dashboard view. Retrieval quality needs to become a measurable product capability.

### 3. Change impact analysis

Before editing, show affected symbols, callers, tests, and likely documentation. After editing, compare predicted impact with actual test/diff evidence. This connects semantic intelligence directly to safer coding work.

### 4. Index freshness and automatic updates

Track Git revision and file hashes, detect stale indexes, and update incrementally through an explicit command or opt-in watcher. Users should always know whether an answer is based on current code.

### 5. Privacy and cost ledger

Show what context left the machine, which provider received it, approximate tokens/cost, and whether local-only mode was preserved. This makes local-first a verifiable promise rather than a configuration label.

### 6. Context packs

Allow users and agents to export a compact, evidence-linked context pack for a bug, feature, or subsystem. Packs can be reused across providers, shared in reviews, or served through MCP without sharing the full index.

### 7. Repository health command

Add `ctxai doctor` for installation, configuration, credentials, index freshness, provider capability, vector-store integrity, and tool-permission diagnostics. This will reduce support burden while the provider ecosystem changes.

## What not to prioritize yet

- More LLM providers before the existing providers pass one conformance suite.
- Multi-agent orchestration before one agent can complete a verified change safely.
- IDE extensions before CLI and MCP have stable application services.
- Enterprise collaboration before index identity, safety, and audit records are reliable.
- More generic web/search tools that do not improve repository understanding.
- Unmeasured claims about model quality, cost savings, or production readiness.

## Delivery sequence

### Milestone 1: Trustworthy foundation

Complete VS-01 and VS-02, repair the test suite, consolidate configuration, and reach zero high-signal lint/runtime errors.

### Milestone 2: The defining workflow

Complete VS-03 and VS-04 so a user can request a change that is grounded in the persistent index, safely executed, and verified.

### Milestone 3: Stable interfaces

Validate VS-05, VS-06, and VS-07 across clean installations and supported dependency versions.

### Milestone 4: Intelligence advantage

Add the code intelligence graph, impact analysis, retrieval evaluation, and index freshness. Validate planning from VS-08 only where it improves measured outcomes.

### Milestone 5: Product expansion

Harden the dashboard and then consider IDE, CI, team, and enterprise surfaces built on the same stable services.

## Near-term backlog

The next implementation cycle should produce:

1. A green e2e index persistence test.
2. A project-rooted tool execution context and path-escape tests.
3. Updated MCP protocol tests against the supported dependency version.
4. One consolidated configuration implementation and clean-save test.
5. Automatic current-repository index discovery in chat.
6. One passing mock-LLM change workflow: retrieve -> inspect -> edit -> diff -> test -> report.
7. A retrieval benchmark with at least 20 representative questions.
8. Updated README and roadmap claims derived from this plan.

## Product metrics

Track metrics that demonstrate the USP:

- Retrieval Recall@5 and MRR on repository questions.
- Percentage of useful context tokens versus duplicate or irrelevant tokens.
- Time from task request to first grounded answer.
- Index update time after a small commit.
- Verified task completion rate on the agent e2e suite.
- Escaped-boundary or unauthorized-action count; the target is always zero.
- Percentage of tasks completed in local-only mode.
- Model tokens and cost per successfully verified task.

## Definition of ready for beta

ctxai is ready for a beta label when:

- The full supported test suite passes from a clean installation.
- Indexes persist, migrate, update, and report freshness reliably.
- Filesystem and command boundaries are enforced and tested.
- The agent completes representative read, edit, and test workflows with evidence.
- CLI and MCP share stable application services and result schemas.
- Documentation accurately distinguishes validated, functional, experimental, and planned behavior.
- Telemetry is opt-in, secrets are redacted, and local-only behavior is auditable.
