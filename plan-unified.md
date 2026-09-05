# ctxai Unified Plan

Last reviewed: 2026-09-04

This document is the single source of truth for all remaining ctxai work. It merges the open slices
of `plan2.md` (intelligence phase) and `plan3.md` (harness-hardening phase) into one rationally
ordered, vertically sliced delivery sequence. `plan.md` remains the record of product direction and
of the nine validated product slices (VS-01..VS-09); its full text stays authoritative for history,
but every slice below is self-contained.

Slice IDs are preserved from their originating plans so references stay traceable: `HH-*` (harness
hardening, from plan3), `IG-*` (intelligence graph, from plan2), `RE-*` (retrieval evaluation and
observability, from plan2). The delivery order in this document supersedes the ordering sections of
both merged plans.

**Status legend**

- **Validated** — implemented and covered by passing end-to-end acceptance tests.
- **Planned** — approved direction with requirements in this document; not started.

**How to use this plan:** work slices one at a time, in the delivery order of Part IV (or an
explicitly marked parallel track). A slice is complete only when every acceptance criterion passes
from a clean installation and its status is updated to Validated with a date. See "Working a
slice" before starting one.

---

## Part I — Product direction and current state

### Product direction

ctxai is the local-first coding agent that builds durable semantic understanding of a repository
before it acts. It combines syntax-aware indexing, semantic retrieval, repository structure, and an
autonomous tool loop so users can search, understand, and safely change a codebase without
repeatedly sending the entire repository to a model. The defensible differentiation: durable
repository understanding, local-first privacy and cost control, model independence, structure-aware
retrieval, and evidence-driven changes. (Full positioning: `plan.md`.)

Primary workflow:

```text
open repository
  -> discover or build its code index
  -> retrieve relevant code and structural context
  -> explain or plan the requested change
  -> obtain approval when the action requires it
  -> edit and run targeted verification
  -> report evidence, changes, and remaining risks
```

### Slice catalog

| Slice | Outcome (abridged) | Status |
|---|---|---|
| VS-01 Trustworthy local index and query | Index, reopen, retrieve with stable metadata | Validated (2026-07-22) |
| VS-02 Safe repository tools | Inspect/modify files and run commands inside the boundary | Validated (2026-07-22) |
| VS-03 Grounded repository understanding | Answers grounded in relevant symbols/files with evidence | Validated (2026-07-22) |
| VS-04 Verified one-shot code change | Bounded change with checks and reporting | Validated (2026-07-22) |
| VS-05 Interactive coding session | Reliable multi-turn sessions without state corruption | Validated (2026-07-22) |
| VS-06 MCP code intelligence service | Stable MCP schemas and failures for index/query | Validated (2026-07-23) |
| VS-07 Provider-independent execution | Consistent tool behavior across providers | Validated (2026-07-23) |
| VS-08 Deliberate planning and approval | Evidence-backed plans approved at the right boundary | Validated (2026-07-23) |
| VS-09 Web dashboard and index operations | Index health, search, and management in a browser | Validated (2026-07-23) |
| HH-01 Hardened tool execution | No secret leaks, bounded output, deterministic edits | Validated (2026-09-04) |
| HH-02 Resilient agent loop | Retries, fail-fast, clean cancellation | Validated (2026-09-04) |
| HH-03 Context window management | Measured tokens, mid-loop compaction | Validated (2026-09-04) |
| HH-04 Run transcripts and cost ledger | Redacted local run records with usage/cost | Validated (2026-09-04) |
| HH-05 True streaming interaction | Live token and tool events in chat | Validated (2026-09-04) |
| HH-06 Checkpoint and rollback | Failed runs reversible byte-identically | Validated (2026-09-04) |
| HH-07 Approval ergonomics and planner control | once/session/deny approvals bound to exact diffs | Validated (2026-09-04) |
| HH-08 OS-sandboxed command execution | OS-level deny-by-default sandbox mode | Validated (2026-09-04) |
| HH-09 Agent task evaluation harness | Scored, gated agent benchmark + provider conformance | Validated (2026-09-05) |
| RE-01 Executable retrieval benchmark | One-command reproducible retrieval quality gates | Validated (2026-09-04) |
| IG-01 Inspectable symbol graph | Python definitions/relationships with evidence | Validated (2026-09-05) |
| IG-02 Multi-language graph + service contract | JS/TS parity, MCP + dashboard consumption | Validated (2026-09-05) |
| IG-03 Graph-expanded grounded retrieval | Graph evidence behind measured gates | Planned |
| RE-02 Privacy-preserving retrieval observability | Local, redacted retrieval traces | Planned |
| RE-03 Retrieval quality dashboard + CI gate | Baseline comparison and merge protection | Planned |

### Baseline facts the open slices build on or fix

Validated substrate (do not break): `CodeChunker` parses Python/JavaScript/TypeScript with
tree-sitter and stores `node_type` and symbol `name` in chunk metadata (`chunking.py:128-148`);
`IndexManifest` is the atomic, schema-versioned source of truth for index identity; `VectorStore`
persists per-index ChromaDB; `HybridRetriever` fuses semantic, lexical, and symbol rankings while
`ContextAssembler` emits bounded `file:start-end` evidence; `IndexOperations` is the shared service
layer for CLI/dashboard; the `evals/` package (RE-01) provides the versioned retrieval benchmark
(`tests/fixtures/retrieval_benchmark.json`, 20 questions) executed at runtime through
`ctxai eval retrieval` with recall@k/MRR/nDCG/latency/token metrics and baseline gates; sessions
persist atomically with secret redaction (`agent/sessions.py`).

Known defects and gaps the harness slices fix (verified in source, 2026-09-03):

- `agent/core.py` calls sync `llm.chat()` inside async `process_message` (blocks the event loop
  shared with MCP/dashboard), injects a recovery prompt into the conversation on *any* exception —
  including context overflow, which makes overflow worse — has no retry, detects loops only by
  comparing two consecutive tool-result lists, and `stream_message` awaits `process_message` and
  yields once (no real streaming).
- `agent/llm/base.py` already defines `ProviderErrorKind` (auth/rate_limit/timeout/cancelled/
  unsupported/transport/invalid_response), `normalize_error`, `ProviderCapabilities`
  (`context_size`, `streaming`), abstract `stream_chat`, and `validate_request(..., cancel_event)`.
  The agent loop currently ignores all of these.
- `agent/context.py` estimates tokens as `chars // 4` and calls `truncate_old_messages` only after a
  final no-tool response (never mid-loop).
- `agent/tools/execution.py` `command_environment()` inherits the **full `os.environ`** (secrets
  reach every subprocess); the audit log is in-memory only; command classification is an in-process
  blocklist with no OS-level backstop.
- `agent/tools/bash_tool.py` supports an executable allowlist and kills the child on timeout, but
  captures stdout/stderr without bounds.
- `agent/tools/file_ops.py` `EditFileTool` uses `re.subn`/`str.replace`, silently replacing **all**
  occurrences; `agent/workflow.py` `_approval_call` simulates edits with `str.replace` only, so the
  approved diff can diverge from the applied edit when `use_regex` is set.
- `agent/workflow.py` `TaskRun` (state, inspected/changed files, checks, approvals) exists only in
  memory.
- `agent/config.py` `AgentBehaviorConfig.stream_responses` is unused; the substring matcher
  `AgentToolsConfig.is_bash_command_allowed` overlaps `BashTool`'s exact-name allowlist.
- No graph storage, graph commands, evaluation CLI, trace schema, or cost ledger exists anywhere in
  `src/ctxai/` (verified 2026-09-03) — IG-*/RE-* start from zero.

### Hard constraints (apply to every slice)

- Python 3.10 syntax (no 3.11+), dev/CI on 3.13. `uv` only (`uv sync --locked --all-extras
  --all-groups`); never `pip install` into the venv.
- ruff (line-length 120, `E,F,I,UP`); `mypy` covers only files listed in `[tool.mypy] files` — each
  slice adds the new modules it introduces to that list. `pytest.ini` has `filterwarnings = error`:
  no new warnings.
- Internal models are dataclasses with `to_dict`/`from_dict` round-trip tests; no pydantic.
- Every new e2e file in `tests/e2e/` carries an explicit `pytest.mark.e2e` marker (marker filtering
  is otherwise unreliable in this repo). The full suite (`uv run pytest`) includes e2e and must stay
  green; unit-only runs use `--ignore=tests/e2e/`.
- Local-first: no telemetry and no outbound transport added by any slice; secrets are redacted via
  `sessions.redact_secrets` before anything is persisted; nothing is persisted where a privacy
  tradeoff exists without the default documented in the slice.
- Deterministic artifacts: identical source, parser versions, configuration, and embeddings produce
  stable identities and reproducible evaluation artifacts.
- Evidence-bearing: graph relationships, traces, and evaluation results always retain
  repository-relative `file:start-end` locations.
- Backward-safe: an older index/session/artifact is never silently interpreted as a newer schema;
  migrate explicitly or report that a rebuild is required.
- Interface consistency: CLI, MCP, dashboard, and agent call shared application services and use
  versioned result models rather than implementing semantics independently.
- No slice adds a required third-party dependency without an ADR-level justification; optional
  dependencies must degrade gracefully when absent.

---

## Part II — Capability contracts

Shared data models introduced across the remaining slices. Each persisted model is schema-versioned
with round-trip serialization tests.

### Graph data model (IG-01/IG-02)

Logical records, persisted in SQLite or another transactional local store:

- `GraphNode`: stable ID, kind, qualified name, display name, language, repository-relative file
  path, start/end line, optional parent ID, visibility when known, and source hash.
- `GraphEdge`: stable ID, kind, source node ID, target node ID when resolved, unresolved target text
  when unresolved, source file/line evidence, confidence (`exact`, `probable`, or `unresolved`), and
  resolver version.
- `GraphMetadata`: graph schema version, extractor/resolver versions, supported languages, build
  time, node/edge counts by kind, unresolved counts, and the index manifest generation or revision
  it matches.

Initial node kinds: `module`, `class`, `function`, `method`, `interface`, `test`. Initial edge
kinds: `contains`, `imports`, `calls`, `inherits`, `references`, `tests`. Every non-`contains` edge
records evidence. Stable IDs derive from repository identity plus canonical source identity, not a
database sequence. "Resolved" means statically supported by the language adapter; ambiguous or
dynamic references remain unresolved rather than being connected to an arbitrary candidate.

### Retrieval run and evaluation artifacts (RE-01/RE-02)

- `RetrievalRun`: schema version, run/query IDs, timestamp, index and graph identity, repository
  revision, and configuration; raw query or a deterministic redaction/hash when query recording is
  disabled; ordered candidates and selected context with chunk ID, citation, component ranks/scores,
  graph expansion reason/path, final rank, estimated tokens, and truncation/deduplication decisions;
  stage timings; total latency, candidate/selected counts, and errors; provider/network fields
  sufficient to prove local retrieval emitted no outbound data.
- `EvaluationArtifact`: benchmark identity/version, expected evidence, per-case judgments, aggregate
  metrics, environment metadata, configuration fingerprint, and comparison with an optional
  baseline. JSON is the canonical machine format; terminal and dashboard views are projections.

Required aggregate metrics: Recall@1/5/10, MRR, nDCG@10, evidence precision@5, successful-query
rate, p50/p95 latency, mean and p95 selected-context tokens, duplicate-token ratio, and graph
contribution rate. Metrics that cannot be computed are marked unavailable with a reason, never
reported as zero.

### Harness contracts (HH-02..HH-09)

- `RunEvent` (persisted, JSON Lines): `schema_version`, `run_id`, `seq`, `timestamp`, `kind`
  (`run_started | user_message | llm_call | tool_call | tool_result | approval | state_transition |
  check | compaction | cancellation | rollback | run_completed`), `payload` (redacted dict),
  optional `usage`.
- `RetryPolicy` (in-memory): `max_retries=3`, `base_delay_s=1.0`, `max_delay_s=30.0`,
  `retry_kinds={RATE_LIMIT, TIMEOUT, TRANSPORT}`.
- `AgentEvent` / `StreamEvent` (in-memory): streaming protocol events — `kind` (`token |
  tool_call_started | tool_result | approval_required | approval_decided | status | usage |
  final_report`), `text`, `data`.
- `UsageRecord` (persisted inside `RunEvent.payload`): `provider`, `model`, `prompt_tokens`,
  `completion_tokens`, `total_tokens`, `call_index`.
- `PriceTable` (static data): USD per 1M prompt/completion tokens per model id;
  `estimate_cost(model, usage) -> float | None` returns `None` for unknown models — never zero.
- `ApprovalDecision` (in-memory enum): `APPROVE_ONCE | APPROVE_SESSION | DENY`.
- `Checkpoint` (persisted): `checkpoint_id`, `run_id`, `created_at`, `files` (repo-relative paths),
  per-file pre-mutation content, `retained: bool`.
- `AgentTaskCase` / `AgentEvalArtifact` (persisted JSON): benchmark task schema and immutable run
  artifact, using the same artifact discipline as `EvaluationArtifact` (RE-01) — one fingerprinting,
  redaction, and baseline-comparison approach for both eval frameworks.

---

## Part III — Why this order

Three principles drive the sequence; each phase exists because of the one before it.

**1. Trust the execution layer before extending it (Phase A).** Every remaining slice — graph
work, evals, observability — executes through the agent loop and its tools. Today that layer can
leak environment secrets into subprocesses, flood the context with unbounded output, silently apply
ambiguous edits, die on the first rate limit, and overflow the context window mid-task with a
recovery path that makes things worse. Building intelligence on top of that would corrupt whatever
it touches, and measuring it would measure noise. HH-01 is deliberately first: it is small,
high-risk, and every later slice composes with its output caps and deterministic edits.

**2. Measure before you optimize (Phase B).** IG-03's own acceptance criteria require the RE-01
benchmark before graph-expanded retrieval can become default — the graph cannot honestly land
without measurement. So the measurement harnesses come first: RE-01 for retrieval, HH-09 for the
agent. They are built back-to-back deliberately, sharing one artifact discipline (fingerprinting,
redaction, baseline comparison) instead of designing two incompatible schemes. HH-08 (sandbox)
completes the safety story here because its composition with HH-01's environment/output policies is
what agent-eval cases run under.

**3. Intelligence behind gates (Phase C).** The graph lands in dependency order (Python first,
multi-language second, retrieval integration third behind a measured flag), then RE-02/RE-03 lock
the gains in with local observability and merge protection. Observability trails the refactor it
observes (RE-02 needs IG-03's candidate-stage boundaries for full ranking provenance).

**Parallelization.** Default is a single track. Explicitly parallel-safe once their dependencies
land: HH-06 after HH-01, HH-08 after HH-01, HH-07's domain layer before its CLI rendering. For a
second contributor, IG-01 can start any time after the validated VS baseline (it does not touch the
agent loop), but it should not merge ahead of RE-01 without its own manual quality review, since
its value claim is otherwise unmeasured.

---

## Part IV — Delivery sequence

### Phase A — Trustworthy execution

### HH-01: Hardened tool execution

**User outcome:** Tool calls cannot leak environment secrets into subprocesses, flood the context
with unbounded output, or silently corrupt files through ambiguous replacements; a failed edit is
impossible to mistake for a successful one.

**Status:** Planned.

**Scope**

- **CLI:** No new commands. `--verbose` output shows output-truncation and replacement-count
  diagnostics.
- **Domain:** `EditFileTool` requires exactly one match by default: non-regex mode errors when
  `count != 1` unless a new explicit `replace_all: bool = False` parameter is set; regex mode uses
  `re.subn` and applies the same rule. When the exact match fails, apply one bounded
  whitespace-tolerant fallback (collapse runs of spaces/tabs, strip trailing whitespace per line)
  that must also match exactly once; the applied strategy is returned in metadata
  (`"exact" | "normalized" | "replace_all"`). Extract the edit-simulation used for approval previews
  from `workflow._approval_call` into a shared function in a new `agent/editing.py`
  (`simulate_edit(tool_name: str, parameters: dict, before: str) -> tuple[str, int]`) that both the
  approval path and `EditFileTool` use, eliminating the `str.replace` divergence for regex edits.
  Consolidate command policy: `AgentToolsConfig.is_bash_command_allowed` is removed; the exact-name
  allowlist in `BashTool` plus `ToolExecutionContext.approve_command` become the single policy.
- **Storage:** No new storage.
- **Integration:** `ToolExecutionContext.command_environment()` returns an allowlist of
  `PATH, HOME, LANG, LC_ALL, TMPDIR, SHELL, TERM, USER, LOGNAME` plus
  `ToolExecutionContext.environment` and an explicit opt-in list
  `AgentToolsConfig.env_passthrough: list[str]`; `os.environ` is never inherited wholesale. New
  `agent/tools/output_limits.py` with
  `truncate_text(text: str, max_chars: int, *, label: str) -> str` (appends a
  `...[truncated N of M chars]` marker) applied to bash stdout/stderr and to `read_file` content
  before it enters the LLM context, bounded by `AgentToolsConfig.max_output_chars: int = 20_000`.
  New config fields get `to_dict`/`from_dict` support. CI (`pr-gate.yml`) gains a `pip-audit`
  dependency-vulnerability job alongside bandit.
- **Safety:** This is the safety slice. Environment leakage removal is verified by seeding a fake
  `ANTHROPIC_API_KEY` and asserting it never appears in a subprocess-visible environment or in any
  audit record. Truncation never throws and marks the direction of truncation. Edits fail closed:
  zero-match and multi-match without `replace_all` are errors, not best-effort writes.
- **Docs:** Update tool documentation: environment policy (exact allowlist, how to extend), output
  limits, edit semantics (uniqueness rule, fallback strategy, `replace_all`), and the threat-model
  note that command classification remains an in-process blocklist pending HH-08.
- **Tests:** Unit: env allowlist filtering; truncation markers and boundary sizes; edit uniqueness
  (0, 1, N matches; regex and literal; normalized-whitespace fallback; `replace_all`); approval
  preview equals applied result for regex edits. E2E (`tests/e2e/test_hh01_hardened_tools.py`):
  agent run through the real registry shows no env leakage, truncated huge command output, and a
  multi-match edit denied with a count-bearing error.

**Acceptance criteria**

1. A subprocess executed by `BashTool` observes only allowlisted variables; no secret from the test
   environment is reachable.
2. Command output and file reads larger than `max_output_chars` enter the context truncated, with an
   explicit marker; the audit record records original size.
3. `edit_file` with a pattern matching zero or multiple occurrences fails without writing unless
   `replace_all` is explicit; the failure names the match count.
4. For a `use_regex` edit, the diff shown at approval time is byte-identical to the diff of the
   applied change.
5. `pip-audit` runs in CI and fails the gate on a known-vulnerable pinned dependency.

**Dependencies:** None. First slice by design: small, high-risk fixes.

**Metrics:** truncation events per run; edit failures by match count; env-allowlist misses (target
zero unexplained); pip-audit findings.

**Non-goals:** OS sandboxing (HH-08); rewriting `write_file`; diff-format changes; shell interpreter
support.

### HH-02: Resilient agent loop

**User outcome:** Transient provider failures (rate limits, timeouts, transport blips) are retried
transparently; authentication and unsupported-capability errors fail fast with a precise message;
Ctrl+C cancels cleanly, preserving session state; long tasks no longer die on the first network
hiccup.

**Status:** Validated (2026-09-04).

**Scope**

- **CLI:** Chat and one-shot surfaces show retry attempts (`retry 2/3 after 2.1s (rate_limit)`) and
  fail fast on non-retryable errors with the provider-qualified reason.
- **Domain:** New `agent/resilience.py`: `RetryPolicy` and
  `async def call_with_retry(fn, *, policy, should_retry, sleep, rng) -> Any` with exponential
  backoff and jitter, retrying only `retry_kinds`, honoring a cancel event between attempts.
  `ProviderErrorKind` → outcome mapping in the loop: `RATE_LIMIT/TIMEOUT/TRANSPORT` retry;
  `AUTHENTICATION` and `UNSUPPORTED` fail fast (no recovery prompt, no iteration burn);
  `INVALID_RESPONSE` gets one recovery prompt then fails. `asyncio.CancelledError` marks the
  `TaskRun` failed with `FailureKind.INFRASTRUCTURE_FAILURE`, saves the session, and returns the
  final report — recovery prompts are never injected for cancellation. The sync `llm.chat()` call is
  offloaded via `asyncio.to_thread`. Loop detection keeps a hash window of the last three
  tool-result tuples and breaks on the third repeat (configurable
  `AgentBehaviorConfig.loop_break_threshold: int = 3`); the break path returns
  `run.final_report(...)` instead of a bare string so status/evidence survive. Max-iterations exit
  also returns `run.final_report(...)`. New fields on `AgentLoopConfig`: `retry_policy: RetryPolicy`,
  `cancel_event: asyncio.Event | None` (defaulted, backward compatible).
- **Storage:** Cancelled/interrupted runs persist state through the existing `SessionStore` and,
  once HH-04 lands, through run transcripts.
- **Integration:** `commands/chat_command.py` installs a `cancel_event` on KeyboardInterrupt paths;
  MCP `server_command.py` maps `ProviderErrorKind.CANCELLED` to the existing `cancelled` envelope
  code. `FailureKind.classify_failure` accepts provider errors.
- **Safety:** Retries never re-execute tools (only the LLM call is retried); cancellation cannot
  leave a half-written file from an in-flight tool because tool execution completes or fails
  atomically per call before the cancel check.
- **Docs:** Document retry defaults, the error-kind → behavior table, cancellation behavior, and the
  loop-detection rule; document that recovery prompts are reserved for malformed responses.
- **Tests:** Unit: backoff sequence with a fake clock/sleep; each `ProviderErrorKind` maps to the
  declared outcome; cancel between attempts raises cleanly; hash-window loop break at threshold.
  E2E (`tests/e2e/test_hh02_resilient_loop.py`): a scripted provider that fails twice with
  rate-limit errors then succeeds produces a completed run; an auth error produces a fast, precise
  failure; cancellation mid-loop preserves a reloadable session.

**Acceptance criteria**

1. A provider that returns `rate_limit` twice then succeeds completes the task; exactly two retry
   waits occur; backoff is exponential with jitter within declared bounds.
2. An `authentication` error ends the run within one iteration with a message naming the provider —
   no recovery prompt is added to the conversation.
3. Cancellation produces a `failed` TaskRun with `infrastructure_failure`, a saved session, and no
   injected recovery message.
4. The LLM call does not block the event loop (verified by a concurrent task progressing while
   `chat` sleeps).
5. Three identical consecutive tool-result tuples end the run with a status-bearing final report.

**Dependencies:** None strictly; lands before HH-03 because context-overflow handling depends on its
error classification.

**Metrics:** retry rate by kind; fast-fail rate for auth/unsupported; cancellation-to-clean-report
rate; iterations burned by avoidable errors (target zero).

**Non-goals:** provider-level streaming (HH-05); fallback-provider failover policy changes; cost
accounting (HH-04).

### HH-03: Context window management

**User outcome:** Long tool-heavy tasks stay under the model's context window automatically: old
tool results are elided deterministically, usage is measured from real provider reports, and a task
that would overflow compacts and continues instead of failing.

**Status:** Validated (2026-09-04).

**Scope**

- **CLI:** `chat` context command (`/context`) reports measured tokens, budget, compaction count,
  and elided message count; compaction prints a one-line notice.
- **Domain:** Capture `LLMResponse.usage` after every call into a `UsageLedger` (new dataclass in
  `workflow.py`: `record(provider, model, usage: dict)`; aggregates per run). Replace the
  chars-div-4 estimate with an estimator that uses the provider's reported usage when available and
  falls back to the existing heuristic otherwise. Before each LLM call, compare the estimated
  context size against `self.llm.get_capabilities().context_size`; when above a configurable
  threshold (`AgentBehaviorConfig.context_soft_limit_ratio: float = 0.8`), run
  `ConversationContext.compact(target_tokens, keep_recent: int = 6)`. Compaction (1) caps each
  tool-result message body at `max_output_chars` with an elision marker, (2) elides bodies of
  tool-result messages outside the recent window, keeping the
  `assistant(tool_calls) ↔ tool results` pairing atomic — an assistant message with tool calls and
  its results are elided or kept together, (3) summarizes elided turns with the existing
  `_summarize_messages`, and (4) records a `compaction` event (HH-04).
  `AgentBehaviorConfig.max_tokens` stays a completion-budget setting and is not conflated with
  context size.
- **Storage:** No new persistence (usage reaches disk via HH-04).
- **Integration:** All loop call sites use the same pre-call check; MCP and one-shot surfaces
  inherit the behavior unchanged.
- **Safety:** Compaction never removes the system prompt, never breaks tool-call pairing (verified
  against provider request validation), and is deterministic for identical history. Elision markers
  are honest about what was removed. Usage capture stores tokens only — no content.
- **Docs:** Document the budget model (context_size source, soft-limit ratio, keep_recent), what
  compaction preserves/elides, and the estimator's accuracy caveats per provider.
- **Tests:** Unit: pairing-preserving compaction (no orphan tool results for any provider message
  formatter); elision markers; usage-ledger aggregation; soft-limit trigger math; estimator
  fallback. E2E (`tests/e2e/test_hh03_context_management.py`): a scripted long tool session that
  would exceed a small injected `context_size` compacts mid-run and completes; a `length`
  finish-reason response surfaces as `invalid_response`-class handling, not a crash.

**Acceptance criteria**

1. A run whose cumulative context crosses the soft limit triggers compaction before the next call
   and completes with a correct final report.
2. After compaction, every retained assistant tool-call message has its paired tool results, for
   both openai and anthropic message formatting.
3. Usage totals per run equal the sum of per-call provider-reported usage.
4. Compaction is deterministic: identical history produces identical compacted messages.
5. The system prompt survives every compaction unchanged.

**Dependencies:** HH-02 (error classification for `length`/overflow responses).

**Metrics:** compactions per run; tokens elided; estimation error vs reported usage; runs ending in
context-overflow errors (target zero after this slice).

**Non-goals:** provider prompt-caching semantics; cross-session memory; retrieval-context budget
changes (owned by `repository_context.py`/RE slices).

### HH-04: Run transcripts and cost ledger

**User outcome:** Every agent run is recorded locally as a redacted JSON Lines transcript; a user
can list and inspect past runs from the CLI and see per-run token usage and estimated cost without
any data leaving the machine.

**Status:** Validated (2026-09-04).

**Scope**

- **CLI:** New `ctxai runs` sub-app (`commands/runs_command.py`, registered in `app.py`):
  `ctxai runs list [--limit N] [--json]`, `ctxai runs show RUN_ID [--json] [--kind KIND]`,
  `ctxai runs delete RUN_ID | --all` (explicit confirmation for `--all`). Final agent reports (chat
  and one-shot) append a usage/cost line when usage is known.
- **Domain:** New `agent/run_recorder.py`: `RunRecorder` writing one `RunEvent` per line with a
  monotonic `seq`, appending atomically and fsyncing on close; a no-op recorder for disabled mode.
  All payloads pass through `sessions.redact_secrets` before writing. New `agent/costing.py`:
  `PriceTable.estimate_cost(model: str, usage: dict) -> float | None` with a checked-in price table
  for documented models; unknown models yield `None` and are surfaced as "cost unknown", never 0.
  The agent loop records `run_started`, `user_message`, `llm_call` (with `UsageRecord`),
  `tool_call`, `tool_result`, `approval`, `state_transition`, `check`, `run_completed`; HH-02 adds
  `cancellation`, HH-03 adds `compaction`, and HH-06 adds `rollback` — the recorder accepts the
  full `RunEvent` kind set. `TaskRun` gains a `to_event_payloads()` helper so recording stays out of
  the loop's control flow. `run_id` is the existing `ToolExecutionContext.request_id` when present,
  else a fresh uuid4 hex.
- **Storage:** `.ctxai/runs/<run_id>.jsonl` inside the project (consistent with `sessions`);
  atomic per-line append, schema version at line 1. `AgentBehaviorConfig.record_runs: bool = True`
  (on by default: local-only, redacted; documented). Retention:
  `AgentBehaviorConfig.run_retention: int = 50` oldest-first cleanup at run start.
- **Integration:** Chat, one-shot, and MCP-driven agent runs all record through the same recorder.
  MCP query/tool responses may include `run_id` when recording is enabled. Dashboard consumption is
  explicitly deferred.
- **Safety:** Redaction covers tool parameters and results, messages, and approvals; transcripts
  never contain absolute home paths (repository-relative normalization). Deletion is scoped to the
  resolved runs directory. Recorder failures are surfaced as diagnostics and never fail the run.
- **Docs:** New `docs/RUN_TRANSCRIPTS.md`: event schema, kinds, redaction guarantees, retention,
  deletion, cost-table coverage and how to extend it, and the explicit statement that nothing is
  uploaded.
- **Tests:** Unit: seq monotonicity; redaction of seeded secrets in payloads; unknown-model cost
  returns None; retention pruning; recorder failure isolation. E2E
  (`tests/e2e/test_hh04_run_transcripts.py`): a full agent run produces a parseable transcript whose
  events reconstruct the TaskRun state transitions; `runs show` round-trips; a seeded API key string
  in tool output does not appear in the transcript.

**Acceptance criteria**

1. Any completed or failed run leaves a `.ctxai/runs/<run_id>.jsonl` that parses event-by-event with
   matching `run_id` and strictly increasing `seq`.
2. `ctxai runs show` renders kind-filtered events and `--json` matches the on-disk schema version.
3. Seeded secret patterns never appear in any persisted event.
4. Per-run usage totals match the ledger; known models get a cost estimate, unknown models get an
   explicit "unknown", and neither fabricates a zero.
5. With `record_runs: false`, no file is written under `.ctxai/runs/`.

**Dependencies:** HH-02 (event kinds include retries/cancellations). HH-09 depends on this slice.

**Metrics:** transcript write failures (target zero); redaction misses (target zero); bytes per run;
retention correctness.

**Non-goals:** dashboard views (later, mirrors RE-02's approach); OpenTelemetry export; cloud sync;
storing embeddings or full LLM responses.

### HH-05: True streaming interaction

**User outcome:** In interactive chat, tokens appear as the model generates them, tool activity and
approval prompts render live in the event stream, and the user is never staring at a silent spinner
while a long tool loop runs.

**Status:** Validated (2026-09-04).

**Scope**

- **CLI:** `commands/chat_command.py` (which already attempts `agent.stream_message`) renders
  `AgentEvent`s: token deltas via Rich Live, tool starts/results as dim status lines, approval
  prompts inline with the HH-07 decision UI, final report as a panel. Non-stream-capable providers
  fall back to current behavior with no UX regression.
- **Domain:** New `agent/events.py`: `AgentEvent` dataclass (kind, text, data) — the loop's event
  vocabulary. Extend `BaseLLMProvider` with
  `stream_chat_events(messages, tools, **kwargs) -> Generator[StreamEvent, None, LLMResponse]` where
  `StreamEvent` is `("text", str) | ("tool_call_delta", dict) | ("usage", dict)`; providers with
  real streaming emit deltas (implement for anthropic/openai/openrouter first); the default
  implementation falls back to `chat()` and emits one `text` event (graceful degradation, contract
  documented). Refactor the loop so `process_message` and `stream_message` share one core
  `_run_loop(event_sink: Callable[[AgentEvent], None])`; `process_message` keeps its exact current
  signature and return value (single final string) so MCP and one-shot surfaces stay stable;
  `stream_message` becomes a generator over real events ending with `final_report`. The
  `AgentBehaviorConfig.stream_responses` flag is finally honored: `false` forces the fallback path.
- **Storage:** None.
- **Integration:** Streaming respects HH-02 cancellation (deltas stop, cancel handler runs) and
  HH-03 budget checks (a compaction mid-stream emits a `status` event). Tool execution inside
  streaming turns follows the identical planning/approval workflow — no policy bypass.
- **Safety:** Approval-required events pause the stream until the decision is made; deltas are never
  persisted beyond what HH-04 already records (the final text, not every delta).
- **Docs:** Document the event protocol, per-provider streaming support matrix (generated from
  `PROVIDER_SPECS`), and the fallback semantics.
- **Tests:** Unit: event sequence for a scripted tool-turn (token events, tool events, final
  report); fallback path for providers without streaming; identical final report from
  `process_message` and `stream_message` for the same scripted conversation. E2E
  (`tests/e2e/test_hh05_streaming.py`): mock streaming provider drives chat; approval event blocks
  until decision; streaming run's final report equals non-streaming run's.

**Acceptance criteria**

1. A streaming-capable provider produces token deltas during the same turns where tools are
   advertised, and tool calls are still executed with the full approval workflow.
2. `stream_message` yields `AgentEvent`s (not one string) and its final event's report is identical
   to `process_message`'s return for the same inputs.
3. A provider without `stream_chat_events` support completes runs via the fallback with a documented
   diagnostic, and `capabilities.streaming` reflects reality.
4. Cancellation during streaming produces the HH-02 cancellation outcome.
5. Approval-required mutations never execute before the decision event.

**Dependencies:** HH-02, HH-03 (shared loop core).

**Metrics:** time-to-first-token; events per run; fallback rate by provider; divergence between
streamed and non-streamed finals (target zero).

**Non-goals:** token-level streaming of tool-result playback; server-sent events over MCP;
dashboard streaming; voice/TTY polish beyond event rendering.

### HH-06: Checkpoint and rollback

**User outcome:** A failed or cancelled verified run is reversible with one command: files are
restored byte-identical to their pre-run state, including files the run created or deleted.

**Status:** Validated (2026-09-04).

**Scope**

- **CLI:** New `ctxai checkpoints` sub-app: `ctxai checkpoints list [--run RUN_ID]`,
  `ctxai checkpoints restore CHECKPOINT_ID` (interactive confirmation; shows affected files), and
  `ctxai checkpoints delete CHECKPOINT_ID | --all`.
- **Domain:** New `agent/checkpoints.py`: `Checkpoint` and `CheckpointManager.for_project(root)`.
  Before the first mutation of each file in a run (hooked at `TaskRun.before_tool` mutation path),
  the manager stores the pre-mutation bytes (or a `created`/`deleted` marker) under
  `.ctxai/checkpoints/<run_id>/` with repository-relative paths and a content hash. `restore`
  writes back pre-run bytes, deletes files the run created, and recreates files the run deleted; it
  refuses when a target file's current hash differs from the post-run hash recorded at checkpoint
  finalization (the working tree moved on) unless `--force`. Checkpoints are marked `retained` when
  the run succeeds (kept for audit) and pruned by retention.
- **Storage:** `.ctxai/checkpoints/<run_id>/` with a manifest JSON (schema-versioned, atomic write,
  fsync — the `sessions.py` pattern). `AgentBehaviorConfig.checkpoint_retention: int = 20` runs;
  per-run size cap with a clear diagnostic when exceeded.
- **Integration:** Wired through the existing `TaskRun` mutation boundary — no tool changes. Works
  for git and non-git projects alike (pure shadow copy; git is used only to record HEAD for
  context). Rollback events are recorded as `rollback` events in HH-04 transcripts.
- **Safety:** Restoration is repo-path-contained and refuses symlink escape; restore never runs
  silently (confirmation + file list). Deleted-file recreation cannot resurrect paths outside the
  project. Checkpoint data is local-only.
- **Docs:** New docs section: what is captured, retention, the stale-worktree refusal, and the
  relationship to git (ctxai does not rewrite history or create commits).
- **Tests:** Unit: created/modified/deleted round-trips; stale-hash refusal; path-escape refusal;
  retention pruning. E2E (`tests/e2e/test_hh06_checkpoints.py`): a run whose verification fails is
  rolled back to byte-identical pre-run state across create/modify/delete; a post-run manual edit
  blocks restore without `--force`.

**Acceptance criteria**

1. After a failed run, `checkpoints restore` returns every touched file to byte-identical pre-run
   content, removes created files, and restores deleted files.
2. Every mutation is preceded by a checkpoint capture (verified by fault injection between capture
   and write).
3. Restore against a moved-on working tree fails with a per-file reason unless forced.
4. Checkpoints never contain paths outside the project and never follow symlinks out of it.
5. Retention prunes only checkpoints older than the configured window.

**Dependencies:** HH-04 (records rollback events) but functionally independent; can be built in
parallel.

**Metrics:** checkpoint capture failures (target zero); restores performed; stale-refusal rate;
checkpoint storage bytes.

**Non-goals:** git stash/branch automation; undo across multiple runs; interactive hunk-level
revert; conflict merging.

### HH-07: Approval ergonomics and planner control

**User outcome:** Approving an action can grant it once, for the session, or deny it; approvals bind
to the exact diff shown; users can force or suppress planning per task instead of relying on keyword
detection.

**Status:** Validated (2026-09-04).

**Scope**

- **CLI:** Chat approval prompts render the proposed diff with syntax highlighting and offer
  `[y] once / [a] always this session / [n] no`; `ctxai chat --plan auto|force|off` and a `/plan`
  chat command override planning for the next task; `ctxai code --plan ...` mirrors it one-shot.
- **Domain:** `ApprovalDecision` replaces the boolean callback; boolean callbacks are adapted
  (`True → APPROVE_ONCE`) for backward compatibility. New `ApprovalMemory` stored in
  `ConversationContext.metadata` (persisted with sessions automatically): scope decisions keyed by
  `(tool, target-pattern)` where the pattern is the exact path for mutations and the executable for
  commands. Approval binding closes the TOCTOU gap: `workflow._approval_call` attaches
  `proposed_diff_sha256` and the pre-approval content hash; before execution the loop re-verifies
  the file hash matches — a mismatch re-prompts instead of executing a stale approval.
  `TaskRun.requires_plan` gains an explicit override channel: `plan_mode: auto|force|off` on
  `AgentLoopConfig` (keyword classification remains the `auto` default).
- **Storage:** Approval decisions live in session metadata via the existing `SessionStore`; nothing
  new to persist.
- **Integration:** The CLI approval callback is the only place decisions are asked; MCP and one-shot
  keep current semantics (approve once via callback or deny). Session-scope memory expires with the
  session — never written to global config.
- **Safety:** "Always this session" can never escalate capabilities (scope is per tool+pattern, not
  global allow); stale-diff re-approval is mandatory; denied approvals keep the existing
  `APPROVAL_DENIAL` failure path.
- **Docs:** Document decision scopes, the binding/staleness rule, plan-mode flags, and what session
  memory persists (and that it is redacted like all session data).
- **Tests:** Unit: memory keying and expiry; boolean-callback adaptation; stale-hash re-prompt;
  plan-mode override matrix. E2E (`tests/e2e/test_hh07_approvals.py`): session-scope approval
  suppresses re-prompting for the same tool+path, deny produces the existing failure report, a file
  changed between approval and execution re-prompts, `--plan force` routes a simple task through
  `submit_plan`.

**Acceptance criteria**

1. Approve-session for a tool+path suppresses subsequent prompts for exactly that key and nothing
   else, within the session only.
2. An approval executed after the target file changed since the diff was shown re-prompts; the stale
   approval never executes.
3. `--plan force` triggers `submit_plan` for a task the keyword classifier would not flag; `--plan
   off` skips planning on a task it would flag (tools remain policy-gated).
4. Boolean approval callbacks continue to work unchanged in existing tests and integrations.
5. Approval decisions recorded in transcripts reflect the actual decision and scope.

**Dependencies:** HH-05 for inline rendering; the domain layer is independent and can land earlier.

**Metrics:** prompts per run; session-scope reuse rate; stale-approval re-prompts (evidence the
binding works); deny rate.

**Non-goals:** persistent ("forever") approvals; wildcard/global rules; approval delegation to
another process; auto-approval heuristics.

---

### Phase B — Measurement and isolation

### HH-08: OS-sandboxed command execution

**User outcome:** With sandboxing enabled, bash commands execute under an OS-level profile that
denies network and restricts writes by default — the in-process blocklist is no longer the only
line of defense.

**Status:** Validated (2026-09-04).

**Scope**

- **CLI:** `ctxai config set tools.sandbox auto|required|off`; chat shows a sandbox badge
  (`sandbox: seatbelt (network denied)`) or an explicit "sandbox unavailable" diagnostic.
- **Domain:** New `agent/tools/sandbox.py`: `SandboxBackend` protocol with `is_available() -> bool`
  and `wrap(argv: list[str], cwd: Path, *, network: bool) -> list[str]` plus environment
  adjustments. Backends: `MacOSSeatbeltBackend` (generates a minimal seatbelt profile in a temp
  file, invokes `sandbox-exec -f`), `BubblewrapBackend` (`bwrap` with `--unshare-net`, tmpfs on
  allowed write paths, read-only system binds), `NoopBackend` (default). Modes via
  `AgentToolsConfig.sandbox: str = "off"`: `auto` uses a backend when available with a visible
  diagnostic; `required` fails the command when no backend exists — it never silently unsands.
  `AgentToolsConfig.sandbox_network: bool = False` (deny by default); allowing network also
  satisfies `Capability.NETWORK` checks and vice versa.
- **Storage:** Generated profiles are temp files cleaned up after execution; nothing persisted
  beyond transcripts.
- **Integration:** `BashTool` wraps `argv` after `approve_command` classification; the blocklist and
  allowlist remain as backstops. Command timeouts and output caps behave identically under wrap.
- **Safety:** The backend contract is deny-by-default; wrap failures fail closed (command does not
  run unsandboxed in `required` mode). Documented honestly: seatbelt is deprecated-but-functional on
  macOS and its profile language is the scope boundary; bubblewrap availability depends on the host.
  Platform matrix documented; tests skip when no backend exists (guard like `pytest.importorskip`).
- **Docs:** New `docs/SANDBOXING.md`: threat model (what the sandbox does and does not prevent),
  backend support matrix, profile contents, mode semantics, and the explicit statement that `off`
  preserves today's behavior.
- **Tests:** Unit with a fake backend: wrap composition, fail-closed semantics, mode matrix. E2E
  (`tests/e2e/test_hh08_sandbox.py`): marked to run only when a backend is available; asserts a
  network-touching command fails under deny-network and a normal build command succeeds under wrap.

**Acceptance criteria**

1. Under `required` with no backend, commands fail with a precise diagnostic and nothing executes.
2. Under an available backend with network denied, a command attempting an outbound connection fails
   while a plain compile/test command succeeds with identical stdout capture.
3. `off` (default) produces byte-identical behavior to today.
4. Sandbox wrapping never bypasses allowlist/blocklist classification or audit recording.
5. Profile/temp cleanup leaves no artifacts behind.

**Dependencies:** HH-01 (env allowlist and output caps compose with the sandbox).

**Metrics:** sandbox mode distribution; wrap failures; commands denied by OS policy vs blocklist;
platform availability rate.

**Non-goals:** container-based isolation (Docker); seccomp/APPARMOR authoring; sandboxing file
tools (they are already path-contained); remote execution.

### RE-01: Executable, versioned retrieval benchmark

**User outcome:** A maintainer can run one command against a real local index and receive
reproducible quality, latency, and context-efficiency results, with a non-zero exit code when
declared gates regress.

**Status:** Validated (2026-09-04).

**Scope**

- **CLI:** Add `ctxai eval retrieval BENCHMARK --index INDEX [--output PATH] [--baseline PATH]
  [--fail-on-regression] [--repeat N] [--json]`. Add `ctxai eval retrieval validate BENCHMARK` for
  schema, duplicate ID, path, evidence-range, split, and expectation validation without running
  retrieval.
- **Domain:** Replace pre-populated retrieved locations with a versioned benchmark schema whose
  cases include stable ID, natural-language query, tags/cohort, expected files/symbols and optional
  line ranges, relevance grades, and train/dev/test split. The runner invokes the production
  retrieval and context-assembly service. Implement all required metrics, deterministic
  aggregation, bootstrap confidence intervals where useful, warm-up/repeat handling, and explicit
  unavailable metrics.
- **Storage:** Persist immutable JSON artifacts under `.ctxai/evaluations/retrieval/` by default,
  using atomic writes and content-derived benchmark/configuration fingerprints. User-selected output
  paths must pass the same project-boundary policy as other writes. Never modify the benchmark
  during execution.
- **Integration:** Reuse repository index discovery, embedding identity checks, `HybridRetriever`,
  `ContextAssembler`, and graph capability detection. Support deterministic mock embeddings in
  acceptance tests and real configured embeddings for local maintainer runs.
- **Safety:** No LLM or network call is allowed by the benchmark runner unless a future evaluator is
  explicitly selected and approved. Validate benchmark paths as repository-relative by default.
  Bound file size, case count, repeats, results, and concurrency. Artifacts redact secrets and
  absolute home paths.
- **Docs:** Provide benchmark authoring guidance, relevance grading, split discipline, metric
  definitions, comparison semantics, reproducibility caveats, and examples for adding a regression
  case without tuning on the test split.
- **Tests:** Unit tests validate metric math (including ties, empty cases, graded relevance,
  partial expectations, unavailable data), schema errors, fingerprints, redaction, and comparison
  thresholds. End-to-end tests build a fixture index, execute the command, parse the artifact,
  compare a baseline, and assert exit codes without network access.

**Acceptance criteria**

1. The existing 20 questions are migrated to the versioned schema with explicit IDs, tags,
   relevance, and splits; retrieved results are produced at runtime, not embedded in fixture data.
2. One clean-install command builds/uses the fixture index and reports all required available
   metrics plus per-case ranks, citations, timing, selected tokens, and configuration identity.
3. Repeated deterministic runs produce byte-stable semantic content apart from documented
   timestamps and measured durations; artifact comparison ignores only those volatile fields.
4. `--fail-on-regression` uses checked-in absolute and relative tolerances per metric/cohort,
   reports each failing gate, and exits non-zero. Missing/incompatible baselines fail clearly rather
   than silently pass.
5. Invalid expectations, missing evidence, unhealthy/stale indexes, embedding mismatch, empty
   cohorts, and partial runs are represented explicitly and cannot be mistaken for a successful
   benchmark.

**Dependencies:** Validated index/query path (VS-01/VS-03). It does not depend on the graph; graph
identity and metrics are optional fields until IG-03.

**Metrics:** benchmark case/cohort coverage; successful-query rate; Recall@1/5/10; MRR; nDCG@10;
evidence precision@5; p50/p95 latency; selected token mean/p95; duplicate-token ratio. Initial
gates are established from three clean deterministic runs and reviewed before enforcement.

**Non-goals:** claiming universal retrieval quality from one repository; live-provider or LLM-answer
evaluation; subjective judge-model scoring; benchmark auto-generation; hidden telemetry; tuning
against held-out test cases; making noisy wall-clock latency a hard cross-platform CI gate.

### HH-09: Agent task evaluation harness

**User outcome:** A maintainer runs one command and gets reproducible, scored results for the agent
on a curated task benchmark (deterministic mock provider in CI, configured providers locally), with
a non-zero exit when gates regress — the missing counterpart to RE-01's retrieval benchmark.

**Status:** Validated (2026-09-05).

**Scope**

- **CLI:** `ctxai eval agent BENCHMARK --provider mock|configured [--cases ID,...] [--output PATH]
  [--baseline PATH] [--fail-on-regression] [--json]` and `ctxai eval providers [--provider P]` for
  the executable conformance suite derived from `agent/llm/contract.py` `PROVIDER_SPECS` (auth
  presence, simple chat, tool round-trip, streaming, error normalization).
- **Domain:** New `evals/` package (`task_benchmark.py`, `runner.py`, `scoring.py`): versioned
  `AgentTaskCase` (stable id, instruction, setup fixture, `expected_checks` commands,
  `forbidden_paths`, `plan_required`, `max_iterations`, tags/cohort). The runner drives the real
  `Agent` loop against a fixture repository, applies HH-01..HH-06 behavior, and scores: all checks
  pass, forbidden paths untouched, completed within iteration budget, plan/approval workflow
  respected. Artifacts (immutable JSON under `.ctxai/evaluations/agent/`) follow the RE-01
  discipline: benchmark/config fingerprints, per-case judgments, aggregates (pass rate, mean/p95
  iterations, tokens, cost where known), and baseline comparison with declared tolerances.
  Mock-provider benchmark cases ship in `tests/fixtures/agent_benchmark/` and are fully
  deterministic.
- **Storage:** `.ctxai/evaluations/agent/<artifact>.json` with atomic writes and redaction; the
  benchmark file is never modified during execution.
- **Integration:** Reuses `MockLLMProvider` scripting for CI determinism; configured-provider runs
  are explicit maintainer actions (network, cost) and are never invoked by default test paths.
  Transcripts from HH-04 attach as evidence per case. Shares the artifact vocabulary, fingerprinting,
  and comparison code with RE-01 (built adjacently by design).
- **Safety:** CI runs require no credentials and no network; configured-provider runs require an
  explicit flag and print a cost warning. Artifacts redact secrets and absolute paths (same rules as
  HH-04).
- **Docs:** Benchmark authoring guide (case anatomy, checks, forbidden paths, cohort discipline,
  "never tune on the test split"), gate semantics, baseline refresh workflow, and the conformance
  suite's scope.
- **Tests:** Unit: scoring math, artifact fingerprints, tolerance comparison, forbidden-path
  detection. E2E (`tests/e2e/test_hh09_agent_evals.py`): mock-provider benchmark executes
  end-to-end from a clean install, produces an artifact, compares against a checked-in baseline, and
  exits non-zero on a seeded regression; the provider-conformance suite runs against
  `MockLLMProvider` without network.

**Acceptance criteria**

1. One command runs the deterministic benchmark from a clean installation with no network or
   credentials and produces a versioned artifact.
2. All shipped benchmark cases pass with `MockLLMProvider` in CI; a seeded regression (broken tool
   result handling) trips `--fail-on-regression` with a named gate.
3. Forbidden-path violations and budget overruns are scored as failures, not absorbed.
4. `eval providers` verifies each provider's declared `PROVIDER_SPECS` capabilities and reports
   drift as failures; mock conformance runs in CI, live providers run only on demand.
5. Baseline updates are deliberate, produce a reviewable artifact diff, and never ship silently
   bundled with the change they excuse.

**Dependencies:** HH-02, HH-03 (a stable loop is a precondition for meaningful scores), HH-04
(transcript evidence), and RE-01 (shares the artifact discipline; lands immediately before in the
delivery order).

**Metrics:** benchmark pass rate by cohort; iterations/token/cost per case; gate regressions
caught; conformance drift incidents; CI runtime.

**Non-goals:** LLM-as-judge scoring; SWE-bench as a CI gate (optional maintainer harness later);
auto-generating tasks from history; leaderboard; multi-repo quality claims.

---

### Phase C — Intelligence advantage

### IG-01: Inspectable symbol graph for one repository

**User outcome:** After indexing a supported repository, a user can inspect its definitions and
structural relationships and trace every result back to source.

**Status:** Validated (2026-09-05).

**Scope**

- **CLI:** Add `ctxai graph stats [INDEX]`, `ctxai graph symbol QUERY [--kind KIND] [--language
  LANG]`, and `ctxai graph neighbors SYMBOL_ID [--edge KIND] [--direction in|out|both] [--depth 1]
  [--limit N]`. Default output is human-readable; `--json` emits a versioned envelope.
- **Domain:** Introduce the graph records and language-adapter protocol. Implement deterministic
  extraction for Python first: modules, classes, functions/methods, containment, imports,
  inheritance, direct calls, references, and test definitions/relationships where statically
  resolvable. Preserve unresolved edges.
- **Storage:** Persist `graph.sqlite3` (preferred) inside the canonical index directory with foreign
  keys, indexes on qualified/display name and edge endpoints/kind, and transactional publication.
  Add graph identity and health fields to the index manifest through a schema migration or an
  explicit rebuild path.
- **Integration:** Graph generation is a stage of the existing index workflow, after parsing and
  before manifest publication. `IndexOperations.inspect` and `indexes doctor` report graph schema,
  counts, revision match, corruption, and missing graph data.
- **Safety:** Resolve only repository-relative canonical paths. Never import or execute indexed
  code. Parameterize storage queries. Bound symbol query length, traversal depth, and result count.
  A failed graph transaction must leave the prior healthy graph and manifest visible.
- **Docs:** Document supported Python constructs, edge confidence, unresolved edges,
  rebuild/migration behavior, CLI examples, and the distinction between static evidence and runtime
  behavior.
- **Tests:** Unit fixtures cover aliases, relative imports, nested definitions, methods,
  inheritance, direct calls, ambiguous names, dynamic calls, syntax errors, tests, duplicate symbol
  names, and stable IDs. Process-restart and corrupt-store acceptance tests exercise the CLI and
  doctor workflow.

**Acceptance criteria**

1. Indexing the Python fixture publishes matching vector and graph generations atomically.
2. A fresh process can locate a named definition and list its imports, callers/callees,
   parents/children, subclasses/base classes, references, and associated tests where the fixture
   makes them statically clear.
3. Every returned node and edge includes repository-relative `file:start-end` evidence and
   confidence.
4. Re-indexing unchanged files makes zero graph mutations; changing or deleting one file replaces
   only its owned nodes/edges and removes dangling relationships deterministically.
5. An injected extraction or storage failure cannot produce a healthy/current manifest.
6. `indexes doctor` detects graph/vector revision mismatch, unsupported schema, corruption, and
   count inconsistency and exits non-zero.

**Dependencies:** Validated incremental index and manifest behavior (VS-01), safe path handling
(VS-02), and shared index operations (VS-09). SQLite is preferred because Python ships its client
and transactions are local; choosing another engine requires an ADR demonstrating equivalent
clean-install behavior.

**Metrics:** extraction duration; nodes/edges per kind; exact/probable/unresolved edge rate;
incremental files reparsed; graph store size; doctor failure count. Establish baselines but set no
quality gate until RE-01 provides the labeled benchmark.

**Non-goals:** runtime tracing; whole-program type inference; cross-repository graphs;
framework-specific dependency injection; graph visualization; automatic code changes; perfect
resolution of reflection, monkey-patching, generated code, or dynamic imports.

### IG-02: Multi-language graph and stable service contract

**User outcome:** Python, JavaScript, and TypeScript users receive the same graph commands and
predictable capability reporting, while MCP and dashboard clients can consume the same results.

**Status:** Validated (2026-09-05).

**Scope**

- **CLI:** Extend IG-01 commands with `ctxai graph capabilities [INDEX]`; diagnostics explicitly
  list language support and unsupported edge kinds rather than returning incomplete data without
  warning.
- **Domain:** Add JavaScript and TypeScript adapters for ES/CommonJS imports, exports, functions,
  classes, methods, interfaces, inheritance/implementation, and statically named calls/references.
  Define a common `GraphOperations` application service and versioned query/result DTOs.
- **Storage:** Keep one graph generation per index. Store language and adapter version on records so
  an adapter upgrade marks only affected files stale and can trigger bounded incremental rebuild.
- **Integration:** Add versioned MCP tools for graph stats, symbol lookup, and neighbors. Add
  dashboard index graph summary, searchable symbol table, and accessible node detail with
  incoming/outgoing relationships. All adapters call `GraphOperations`; none may query graph storage
  directly.
- **Safety:** Apply MCP index-name validation, dashboard routing protections, depth/result bounds,
  escaped output, and the dashboard's loopback/explicit-remote policy. Graph endpoints are
  read-only.
- **Docs:** Publish a generated support matrix by language, construct, and edge kind. Document
  JSON/MCP schemas, deterministic error codes, limits, and examples for CLI, MCP, and dashboard.
- **Tests:** Shared contract tests run against every adapter and interface. Protocol tests invoke
  the real MCP transport; ASGI tests cover browser flows without opening a network socket. Fixtures
  include mixed JS/TS imports, re-exports, overloads/interfaces, aliases, and unresolved dynamic
  imports.

**Acceptance criteria**

1. Equivalent Python/JavaScript/TypeScript fixtures expose consistent node/edge semantics and
   evidence.
2. CLI JSON, MCP, and dashboard projections agree on identity, counts, confidence, and
   relationships for the same index and query.
3. Unsupported languages/constructs return explicit capability information and retain indexability
   as ordinary chunks; they do not fabricate edges or break indexing.
4. Malformed names, traversal attempts, excessive depth/limit, stale graph generation, and corrupt
   storage return deterministic errors without leaking paths outside the repository.
5. Adapter and schema compatibility tests pass from the built wheel with documented optional
   dependencies.

**Dependencies:** IG-01 and the validated MCP/dashboard application boundaries from VS-06/VS-09.

**Metrics:** supported-language file coverage; extraction failures by adapter; unresolved edges by
language; service latency p50/p95; interface contract failures (target zero).

**Non-goals:** editors/IDE extensions; remote graph service; cross-language call resolution beyond
explicit imports/exports; call-site control-flow analysis; user-authored graph mutation; graph
rendering beyond accessible tables and relationship lists.

### IG-03: Graph-expanded grounded retrieval

**User outcome:** A repository question returns coherent implementation context that includes
relevant definitions, callers/callees, imports, and tests when they add value, with an explanation
of why each item was selected.

**Status:** Planned.

**Scope**

- **CLI:** Add graph-aware behavior to the shared retrieval path and expose `ctxai query --explain`
  plus `--graph/--no-graph`. Explain output shows base ranks, graph paths, fusion contribution,
  exclusions, and context-budget decisions without changing normal concise output.
- **Domain:** Refactor `HybridRetriever` into independently measurable candidate generators and one
  deterministic fusion policy. Seed graph expansion from top base hits, use an allowlisted edge
  policy, decay score by depth/confidence, deduplicate by chunk/source identity, and assemble only
  within the configured token budget. Default traversal depth is one; depth two requires an explicit
  bounded option.
- **Storage:** Retrieval reads a graph only when its generation matches the vector/index manifest.
  No query mutates index or graph state. Configuration records graph enablement, edge weights, seed
  count, expansion cap, depth, and token budget with validated bounds.
- **Integration:** Agent semantic-search, one-shot coding, CLI query, MCP query, and dashboard query
  use the same retrieval service. Existing versioned response schemas gain optional explanation
  fields in a backward-compatible revision.
- **Safety:** Exclude graph evidence outside the repository; cap seeds, neighbors, depth,
  candidates, source preview, and tokens. If graph data is absent/stale/corrupt, fail explicitly
  when `--graph` is required and otherwise fall back with a visible diagnostic. Query logging
  follows the privacy controls in RE-02.
- **Docs:** Explain expansion policy, configuration, confidence, fallback semantics, token
  tradeoffs, and how to interpret `--explain` output.
- **Tests:** Deterministic fixtures prove useful one-hop expansion, cycle handling, confidence
  decay, deduplication, budget enforcement, stable ties, stale-graph behavior, and consistent
  results across interfaces. Add adversarial high-degree and cyclic graphs.

**Acceptance criteria**

1. A query seeded on an implementation symbol includes its directly relevant test or caller in the
   fixture when the configured edge policy permits it and explains the exact path.
2. Every selected item has a base or graph reason; graph-expanded items cite both source and
   relationship evidence.
3. Repeated runs against an unchanged index/configuration produce the same ordering and selected
   evidence.
4. Context never exceeds the configured approximate token budget, cycles never duplicate evidence,
   and high-degree nodes respect candidate caps.
5. On the RE-01 benchmark, graph-enabled retrieval must not regress Recall@5 or MRR beyond the
   declared tolerance and must improve at least one pre-registered relationship-oriented metric or
   case cohort before becoming the default.

**Dependencies:** IG-02 for stable graph services and RE-01 for honest quality gates. The
implementation may be developed behind a disabled feature flag before RE-01 is complete, but it
cannot become default first.

**Metrics:** graph contribution rate; useful graph expansion precision; Recall@5/MRR/nDCG delta
versus graph-disabled retrieval; token delta; duplicate-token ratio; stage p50/p95 latency;
fallback/error rate.

**Non-goals:** using graph proximity as proof of relevance; unlimited multi-hop traversal;
replacing semantic/lexical retrieval; model-based reranking in the default offline benchmark;
change-impact analysis; automatic tuning against the test set.

### RE-02: Privacy-preserving retrieval observability

**User outcome:** A user can inspect why a particular search selected its context and diagnose
slow, noisy, or graph-heavy retrieval locally without exposing source or queries.

**Status:** Planned.

**Scope**

- **CLI:** Add `ctxai retrieval runs list`, `ctxai retrieval runs show RUN_ID [--json]`, and
  `ctxai retrieval runs delete [RUN_ID|--all]`. Add `--trace` to query/evaluation commands. Normal
  queries produce in-memory metrics only unless local persistence is explicitly enabled in
  configuration or by flag.
- **Domain:** Instrument the production retrieval stages with a clock/recorder abstraction and the
  `RetrievalRun` schema. Record candidate provenance, component ranks, graph paths, final selection,
  deduplication/truncation, timing, and errors. Add configuration for `off|metrics|full` recording,
  query text `omit|hash|store`, source preview `omit|store`, retention count/days, and local
  artifact directory.
- **Storage:** Store local JSON Lines or SQLite traces atomically with a version and retention
  policy. Default is `off` for persistence; `metrics` stores no raw query, source, embeddings,
  credentials, or absolute home paths. Deletion is deterministic and scoped to the configured trace
  directory.
- **Integration:** The CLI, MCP, dashboard, agent, and evaluator emit through the same recorder. MCP
  responses return a run ID only when tracing is enabled. Dashboard adds a local retrieval-runs view
  with filters for index, status, time, and cohort and a detail view of the ranking funnel and
  timings.
- **Safety:** No automatic upload, telemetry SDK, or remote exporter. Redact recursively by
  secret-bearing field name and common credential formats before persistence. Apply project/index
  path normalization, bounded preview sizes, retention, and explicit confirmation for bulk deletion.
  Dashboard protections from VS-09 remain in force.
- **Docs:** State exact defaults and recorded fields for every mode; explain opt-in, storage
  location, retention, redaction limits, deletion, run IDs, and how to verify no outbound transport
  exists.
- **Tests:** Fake-clock tests cover stage timing and failures; snapshot/schema tests cover traces;
  privacy tests seed API keys, bearer tokens, URLs with credentials, absolute home paths, source,
  and raw queries; retention/concurrent-write/corruption recovery tests run locally. CLI/MCP/ASGI
  contract tests compare the same run projection.

**Acceptance criteria**

1. A traced query produces a versioned run showing every candidate generator, ordered candidates,
   graph paths if used, final items, exclusions, stage/total timings, token estimate, and
   index/config identity.
2. Default configuration writes no retrieval trace. `metrics` mode persists neither raw query nor
   source; `full` mode requires explicit opt-in and displays a privacy warning on enablement.
3. Privacy tests find no seeded secret or disallowed absolute path in persisted artifacts, terminal
   output, MCP results, or dashboard HTML.
4. Trace recording failure never changes retrieval ordering or turns a successful query into a
   failed one; the recording failure is surfaced as a diagnostic. Retrieval failures themselves
   remain observable.
5. Retention and delete commands remove only resolved trace targets, and concurrent writers cannot
   corrupt previously committed runs.

**Dependencies:** RE-01 artifact vocabulary and IG-03 candidate-stage boundaries. Basic
metrics-only instrumentation can land with RE-01; full ranking provenance follows the IG-03
refactor.

**Metrics:** trace overhead in latency and bytes; recorder failures; traces retained/deleted;
redaction failures (target zero); percentage of runs with complete stage timings; no outbound
transport count (target zero unless a separately approved future feature exists).

**Non-goals:** hosted telemetry; user tracking; remote log aggregation; session replay; storing
embeddings; capturing LLM prompts/responses; automatic source upload; observability of unrelated
agent/tool execution (owned by HH-04); distributed tracing standards unless a local
interoperability need is demonstrated.

### RE-03: Retrieval quality dashboard and CI regression gate

**User outcome:** Maintainers can compare benchmark runs, identify regressed cohorts or cases, and
prevent a retrieval change from merging when it violates reviewed quality gates.

**Status:** Planned.

**Scope**

- **CLI:** Add `ctxai eval retrieval compare BASELINE CANDIDATE [--json]` with metric/cohort/case
  deltas and compatible/incompatible status. Preserve the runner's gate-based exit codes for CI.
- **Domain:** Implement artifact compatibility checks for schema, benchmark fingerprint, case
  set/split, index/graph schema, embedding identity, and retrieval configuration. Comparison
  distinguishes quality, efficiency, correctness, and noisy timing dimensions and identifies newly
  passing/failing cases.
- **Storage:** Dashboard reads immutable evaluation artifacts through an `EvaluationOperations`
  service. It never scans arbitrary user paths; configured artifact roots and artifact IDs are
  validated.
- **Integration:** Add dashboard run list, run summary, aggregate/cohort comparison, worst
  regressions, and per-case ranking evidence. Add a GitHub Actions retrieval job using deterministic
  local/mock embeddings, a checked-in benchmark and baseline, artifact upload on success/failure,
  and `--fail-on-regression`.
- **Safety:** CI uses no provider credentials or network-dependent embeddings. Dashboard escapes
  query/source content, limits previews, and follows explicit remote binding rules. Uploaded CI
  artifacts contain fixture code only and pass the same redaction checks.
- **Docs:** Add maintainer workflow for refreshing a baseline, required review evidence,
  interpreting noisy latency, downloading CI artifacts, and recovering from schema/benchmark
  incompatibility. Document that a baseline update must not be bundled invisibly with the retrieval
  algorithm change it excuses.
- **Tests:** Comparison golden tests cover improvements, regressions, incompatible runs, missing
  metrics, cohort drift, and tolerance boundaries. ASGI tests cover lists/comparisons/case details.
  A workflow linter or parsed-YAML test verifies the CI job command, credential-free environment,
  and artifact publication.

**Acceptance criteria**

1. A dashboard comparison shows aggregate and cohort deltas, confidence/availability, changed cases,
   ranks, citations, graph contribution, latency, and context tokens from the same JSON artifacts as
   the CLI.
2. CI runs deterministically without secrets or external services and fails for a seeded
   Recall/MRR/nDCG or correctness regression beyond reviewed tolerances.
3. CI does not hard-fail on cross-run p95 latency noise; it records and flags reviewed efficiency
   thresholds separately unless a controlled runner makes the threshold reliable.
4. Incompatible artifacts cannot be compared as if equivalent. The output identifies every
   incompatible identity field and gives a rebuild/rerun action.
5. Updating a checked-in baseline is a deliberate documented command and produces a reviewable
   artifact diff plus benchmark/configuration fingerprints.

**Dependencies:** RE-01 is required. RE-02 supplies richer drill-down data but is not required for
the first aggregate comparison. IG-03 is required before graph contribution becomes a gate.

**Metrics:** CI pass/fail and runtime; number of gated quality regressions; baseline age; benchmark
cohort coverage; incompatible comparison count; dashboard comparison latency.

**Non-goals:** public leaderboard; comparing different repositories as equivalent; automatically
accepting baseline regressions; hard CI dependence on cloud embeddings; LLM answer correctness;
production telemetry; optimizing solely for one aggregate metric.

---

## Part V — Cross-slice engineering requirements

- New modules are added to `[tool.mypy] files` in the same PR that introduces them; `mypy` stays
  green in CI.
- Every new dataclass config field ships with `to_dict`/`from_dict` round-trip tests and a default
  that preserves current behavior (all `AgentLoopConfig`/`AgentConfig` additions are defaulted).
- Persisted artifacts (`runs`, `checkpoints`, `evaluations`, graphs, traces) are schema-versioned,
  written atomically with fsync (the `sessions.py` pattern), redacted via `sessions.redact_secrets`,
  and use repository-relative paths.
- New public result models and stored artifacts have round-trip serialization tests; new application
  services accept dependencies (clock, store, embedding provider, recorder) explicitly so tests do
  not require network, global configuration, or wall-clock timing.
- Configuration has one canonical implementation and rejects invalid depth, result, retention, and
  token limits before work begins.
- New user-facing failures reuse the `FailureKind` taxonomy and, where MCP-visible, the
  `mcp_protocol.py` envelope codes — no new ad-hoc error strings across surfaces. Human text may
  evolve; stable categories/codes do not.
- Database migrations (graph store, manifest schema) are forward-only, transactional,
  fixture-tested from every supported prior schema, and documented with rebuild fallback. No code
  silently drops an index, graph, trace, evaluation, or checkpoint.
- Every slice's e2e file carries `pytest.mark.e2e`, runs the real loop/tools with
  `patch_embeddings_factory` and `MockLLMProvider`, and passes in the full `uv run pytest` suite —
  including `filterwarnings = error`.
- Performance tests use bounded fixtures and declared budgets. A performance budget cannot override
  quality, correctness, privacy, or safety gates.
- Each slice updates the docs it names and lands with ruff/format/mypy/pytest/build/twine green in
  the supported Python matrix (3.10 and 3.13).

## Part Vbis — Working a slice

1. **Pick the next slice** in delivery order (Part III lists what is explicitly parallel-safe).
2. **Branch** named after the slice (e.g., `hh-01-hardened-tools`).
3. **Implement test-first** per repo conventions: unit tests named in the slice, then the e2e file;
   add new modules to `[tool.mypy] files`; ruff format + check; no new warnings.
4. **Verify acceptance criteria from a clean installation**: fresh `uv sync --locked --all-extras
   --all-groups`, a temp `CTXAI_HOME`, and each numbered criterion exercised through the CLI, MCP,
   or dashboard — not just through fixtures.
5. **Update the docs the slice names**, and `README.md` when a user-facing command changes.
6. **Flip the status** to `Validated (date)` in the slice catalog (Part I) and tick the
   corresponding ROADMAP entries.
7. **Land via PR** with CI green: lint, mypy, unit + e2e matrix, bandit, pip-audit, build + twine.

A slice with failing acceptance criteria is not partially done — it stays Planned until its full
user outcome passes from a clean install.

## Part VI — Definition of complete

The unified plan is complete when all fifteen open slices are validated and:

**Execution (Phase A):**

- a long, tool-heavy agent task survives transient provider errors, stays inside the model's context
  window, and reports real token usage and cost;
- no tool call can leak environment secrets, exceed output budgets, or apply an ambiguous edit;
- every run leaves a redacted, inspectable local transcript, and failed runs are reversible;
- approvals bind to exact diffs with once/session/deny ergonomics, and planning is
  user-controllable; and
- interactive sessions stream tokens and tool events live with no policy bypass.

**Measurement and isolation (Phase B):**

- optional OS sandboxing denies network by default with fail-closed semantics;
- one command reproduces retrieval quality, latency, and context-efficiency metrics with reviewed
  CI gates; and
- one command scores the agent on a deterministic benchmark with gates, plus executable provider
  conformance.

**Intelligence (Phase C):**

- a clean install can index a Python/JavaScript/TypeScript fixture, inspect its graph, and retrieve
  bounded graph-expanded evidence through CLI, MCP, and dashboard with consistent identities;
- graph-disabled and graph-enabled benchmark artifacts can be reproduced and honestly compared, and
  graph-enabled retrieval only becomes default after beating the declared gates; and
- a user can explain a retrieval run locally without opting into persistence or sending data
  externally.

**Explicitly not prioritized (merged from plan.md and ROADMAP):** multi-agent orchestration; IDE
extensions; enterprise collaboration surfaces; generic web/search tools unrelated to repository
understanding; architect/editor mode without benchmark evidence (HH-09 is the benchmark it waits
for); hosted telemetry; LLM-as-judge scoring; public leaderboards; automatic tuning against
held-out test sets.
