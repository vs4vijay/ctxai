# ctxai Agent Harness Plan

> **Superseded (2026-09-03):** The slices in this plan (HH-01..09) have been merged, with full detail
> and a unified delivery order, into `plan-unified.md` — the single source of truth for all remaining
> work. This file is kept as history; do not start work from here.

Last reviewed: 2026-09-03

This document defines implementation-ready requirements for hardening the agent harness: the machinery
wrapped around the model — the execution loop, context management, tool execution, observability,
recovery, and evaluation. It complements `plan.md` (validated product slices VS-01..VS-09) and
`plan2.md` (intelligence phase IG-01..03, RE-01..03). Slice IDs use the `HH-` prefix (harness
hardening) and do not collide with either.

Delivery is organized as vertical slices. Every slice must produce a user-visible outcome across the
CLI, domain model, persistence, integrations, safety, documentation, and tests. A slice is complete
only when its acceptance criteria pass from a clean installation.

## Baseline and constraints

The plan extends the validated architecture in `plan.md`, especially VS-02 (safe tools), VS-04
(verified changes), VS-05 (sessions), and VS-08 (planning and approval). Current-state facts the
slices build on or fix:

- `agent/core.py` calls the **sync** `llm.chat()` inside async `process_message` (blocks the event
  loop shared with MCP/dashboard), injects a recovery prompt into the conversation on *any*
  exception (including context overflow, which makes overflow worse), has no retry, and detects
  loops only by comparing two consecutive tool-result lists. `stream_message` awaits
  `process_message` and yields once — there is no real streaming.
- `agent/llm/base.py` already defines `ProviderErrorKind` (auth/rate_limit/timeout/cancelled/
  unsupported/transport/invalid_response), `normalize_error`, `ProviderCapabilities`
  (`context_size`, `streaming`), abstract `stream_chat`, and `validate_request(..., cancel_event)`.
  The agent loop currently ignores all of these.
- `agent/context.py` estimates tokens as `chars // 4`, calls `truncate_old_messages` only after a
  final no-tool response (never mid-loop), and summarizes elided messages with keyword heuristics.
- `agent/tools/execution.py` enforces capabilities in-process; `command_environment()` inherits the
  **full `os.environ`** (secrets reach every subprocess); the audit log is in-memory only.
- `agent/tools/bash_tool.py` supports an executable allowlist and kills the child on timeout, but
  captures stdout/stderr without bounds.
- `agent/tools/file_ops.py` `EditFileTool` uses `re.subn`/`str.replace`, silently replacing **all**
  occurrences; `agent/workflow.py` `_approval_call` simulates the edit with `str.replace` only, so
  the approved diff can diverge from the applied edit when `use_regex` is set.
- `agent/workflow.py` `TaskRun` (state, inspected/changed files, checks, approvals) exists only in
  memory; nothing survives the process.
- `agent/sessions.py` already provides atomic, fsynced, secret-redacted, repository-bound JSON
  persistence — the pattern new persistence must follow.
- `agent/config.py` `AgentBehaviorConfig.stream_responses` exists but is unused; the substring
  matcher `AgentToolsConfig.is_bash_command_allowed` overlaps `BashTool`'s exact-name allowlist and
  `ToolExecutionContext.approve_command`.

Hard constraints that apply to every slice:

- Python 3.10 syntax (no 3.11+), dev/CI on 3.13. `uv` only; never `pip install` into the venv.
- ruff (line-length 120, `E,F,I,UP`); `mypy` covers only files listed in `[tool.mypy] files` — each
  slice adds the new modules it introduces to that list. `pytest.ini` has `filterwarnings = error`:
  no new warnings.
- Internal models are dataclasses with `to_dict`/`from_dict` round-trip tests; no pydantic.
- Every new e2e file in `tests/e2e/` carries an explicit `pytest.mark.e2e` marker (marker filtering
  is otherwise unreliable in this repo).
- Local-first: no telemetry, no outbound transport added by any slice; secrets are redacted via
  `sessions.redact_secrets` before anything is persisted; nothing is persisted by default where a
  privacy tradeoff exists without an explicit default documented in the slice.

## Capability contracts

These are the shared data models the slices introduce. Each is schema-versioned where persisted.

- `RunEvent` (persisted, JSON Lines): `schema_version`, `run_id`, `seq`, `timestamp`,
  `kind` (`run_started | user_message | llm_call | tool_call | tool_result | approval |
  state_transition | check | compaction | cancellation | rollback | run_completed`), `payload`
  (redacted dict), optional `usage`.
- `RetryPolicy` (in-memory): `max_retries=3`, `base_delay_s=1.0`, `max_delay_s=30.0`,
  `retry_kinds={RATE_LIMIT, TIMEOUT, TRANSPORT}`.
- `AgentEvent` / `StreamEvent` (in-memory): streaming protocol events — `kind`
  (`token | tool_call_started | tool_result | approval_required | approval_decided | status |
  usage | final_report`), `text`, `data`.
- `UsageRecord` (persisted inside `RunEvent.payload`): `provider`, `model`, `prompt_tokens`,
  `completion_tokens`, `total_tokens`, `call_index`.
- `PriceTable` (static data): USD per 1M prompt/completion tokens per model id;
  `estimate_cost(model, usage) -> float | None` returns `None` for unknown models — never zero.
- `ApprovalDecision` (in-memory enum): `APPROVE_ONCE | APPROVE_SESSION | DENY`.
- `Checkpoint` (persisted): `checkpoint_id`, `run_id`, `created_at`, `files` (repo-relative paths),
  per-file pre-mutation content, `retained: bool`.
- `AgentTaskCase` / `AgentEvalArtifact` (persisted JSON): benchmark task schema and immutable run
  artifact, mirroring the RE-01 artifact discipline from `plan2.md`.

## Delivery sequence

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
  `PATH, HOME, LANG, LC_ALL, TMPDIR, SHELL, TERM, USER, LOGNAME` plus `ToolExecutionContext.environment`
  and an explicit opt-in list `AgentToolsConfig.env_passthrough: list[str]`; `os.environ` is never
  inherited wholesale. New `agent/tools/output_limits.py` with
  `truncate_text(text: str, max_chars: int, *, label: str) -> str` (appends a
  `...[truncated N of M chars]` marker) applied to bash stdout/stderr and to `read_file` content
  before it enters the LLM context, bounded by `AgentToolsConfig.max_output_chars: int = 20_000`.
  New config fields get `to_dict`/`from_dict` support. CI (`pr-gate.yml`) gains a `pip-audit`
  dependency-vulnerability job alongside bandit.
- **Safety:** This is the safety slice. Environment leakage removal is verified by seeding a fake
  `ANTHROPIC_API_KEY` and asserting it never appears in a subprocess-visible environment or in any
  audit record. Truncation never throws and marks the direction of truncation. Edits fail closed:
  zero-match and multi-match without `replace_all` are errors, not best-effort writes.
- **Docs:** Update `docs/` tool documentation: environment policy (exact allowlist, how to extend),
  output limits, edit semantics (uniqueness rule, fallback strategy, `replace_all`), and the
  threat-model note that command classification remains an in-process blocklist pending HH-08.
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

**Status:** Planned.

**Scope**

- **CLI:** Chat and one-shot surfaces show retry attempts (`retry 2/3 after 2.1s (rate_limit)`) and
  fail fast on non-retryable errors with the provider-qualified reason.
- **Domain:** New `agent/resilience.py`: `RetryPolicy` (above) and
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
- **Storage:** Cancelled/interrupted runs persist state through the existing `SessionStore` and, once
  HH-04 lands, through run transcripts.
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

**Status:** Planned.

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
  tool-result messages outside the recent window, keeping the `assistant(tool_calls) ↔ tool results`
  pairing atomic — an assistant message with tool calls and its results are elided or kept
  together, (3) summarizes elided turns with the existing `_summarize_messages`, and (4) records a
  `compaction` event (HH-04). `AgentBehaviorConfig.max_tokens` stays a completion-budget setting and
  is not conflated with context size.
- **Storage:** No new persistence (usage reaches disk via HH-04).
- **Integration:** All loop call sites use the same pre-call check; MCP and one-shot surfaces
  inherit the behavior unchanged.
- **Safety:** Compaction never removes the system prompt, never breaks tool-call pairing (verified
  against provider request validation), and is deterministic for identical history. Elision markers
  are honest about what was removed. Usage capture stores tokens only — no content.
- **Docs:** Document the budget model (context_size source, soft-limit ratio, keep_recent), what
  compaction preserves/elides, and the estimator's accuracy caveats per provider.
- **Tests:** Unit: pairing-preserving compaction (no orphan tool results for any provider message
  formatter); elision markers; usage-ledger aggregation; soft-limit trigger math; estimator fallback.
  E2E (`tests/e2e/test_hh03_context_management.py`): a scripted long tool session that would exceed
  a small injected `context_size` compacts mid-run and completes; a `length` finish-reason response
  surfaces as `invalid_response`-class handling, not a crash.

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

**Status:** Planned.

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
  The agent loop records `run_started`, `user_message`, `llm_call` (with `UsageRecord`), `tool_call`,
  `tool_result`, `approval`, `state_transition`, `check`, `run_completed`; HH-02 adds `cancellation`,
  HH-03 adds `compaction`, and HH-06 adds `rollback` — the recorder accepts the full `RunEvent` kind
  set. `TaskRun` gains a `to_event_payloads()` helper so recording stays out of the loop's control
  flow. `run_id` is the
  existing `ToolExecutionContext.request_id` when present, else a fresh uuid4 hex.
- **Storage:** `.ctxai/runs/<run_id>.jsonl` inside the project (consistent with `sessions`);
  atomic per-line append, schema version at line 1. `AgentBehaviorConfig.record_runs: bool = True`
  (on by default: local-only, redacted; documented). Retention: `AgentBehaviorConfig.run_retention:
  int = 50` oldest-first cleanup at run start.
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
  events reconstruct the TaskRun state transitions; `runs show` round-trips; a seeded API key
  string in tool output does not appear in the transcript.

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

**Non-goals:** dashboard views (later slice, mirrors RE-02's approach); OpenTelemetry export; cloud
sync; storing embeddings or full LLM responses.

### HH-05: True streaming interaction

**User outcome:** In interactive chat, tokens appear as the model generates them, tool activity and
approval prompts render live in the event stream, and the user is never staring at a silent spinner
while a long tool loop runs.

**Status:** Planned.

**Scope**

- **CLI:** `commands/chat_command.py` (which already attempts `agent.stream_message`) renders
  `AgentEvent`s: token deltas via Rich Live, tool starts/results as dim status lines, approval
  prompts inline with the HH-07 decision UI, final report as a panel. Non-stream-capable providers
  fall back to current behavior with no UX regression.
- **Domain:** New `agent/events.py`: `AgentEvent` dataclass (kind, text, data) — the loop's event
  vocabulary. Extend `BaseLLMProvider` with
  `stream_chat_events(messages, tools, **kwargs) -> Generator[StreamEvent, None, LLMResponse]` where
  `StreamEvent` is `("text", str) | ("tool_call_delta", dict) | ("usage", dict)`; providers with real
  streaming emit deltas (implement for anthropic/openai/openrouter first); the default
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
- **Tests:** Unit: event sequence for a scripted tool-turn (token events, tool events, final report);
  fallback path for providers without streaming; identical final report from `process_message` and
  `stream_message` for the same scripted conversation. E2E
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

**Non-goals:** token-level streaming of tool-result playback; server-sent events over MCP; dashboard
streaming; voice/TTY polish beyond event rendering.

### HH-06: Checkpoint and rollback

**User outcome:** A failed or cancelled verified run is reversible with one command: files are
restored byte-identical to their pre-run state, including files the run created or deleted.

**Status:** Planned.

**Scope**

- **CLI:** New `ctxai checkpoints` sub-app: `ctxai checkpoints list [--run RUN_ID]`,
  `ctxai checkpoints restore CHECKPOINT_ID` (interactive confirmation; shows affected files), and
  `ctxai checkpoints delete CHECKPOINT_ID | --all`.
- **Domain:** New `agent/checkpoints.py`: `Checkpoint` (above) and `CheckpointManager.for_project(root)`.
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

**Status:** Planned.

**Scope**

- **CLI:** Chat approval prompts render the proposed diff with syntax highlighting and offer
  `[y] once / [a] always this session / [n] no`; `ctxai chat --plan auto|force|off` and a `/plan`
  chat command override planning for the next task; `ctxai code --plan ...` mirrors it one-shot.
- **Domain:** `ApprovalDecision` enum replaces the boolean callback; boolean callbacks are adapted
  (`True → APPROVE_ONCE`) for backward compatibility. New `ApprovalMemory` stored in
  `ConversationContext.metadata` (persisted with sessions automatically): scope decisions keyed by
  `(tool, target-pattern)` where the pattern is the exact path for mutations and the executable for
  commands. Approval binding closes the TOCTOU gap: `workflow._approval_call` attaches
  `proposed_diff_sha256` and the pre-approval content hash; before execution the loop re-verified
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
2. An approval executed after the target file changed since the diff was shown re-prompts; the
   stale approval never executes.
3. `--plan force` triggers `submit_plan` for a task the keyword classifier would not flag; `--plan
   off` skips planning on a task it would flag (tools remain policy-gated).
4. Boolean approval callbacks continue to work unchanged in existing tests and integrations.
5. Approval decisions recorded in transcripts reflect the actual decision and scope.

**Dependencies:** HH-05 for inline rendering; the domain layer is independent and can land earlier.

**Metrics:** prompts per run; session-scope reuse rate; stale-approval re-prompts (evidence the
binding works); deny rate.

**Non-goals:** persistent ("forever") approvals; wildcard/global rules; approval delegation to
another process; auto-approval heuristics.

### HH-08: OS-sandboxed command execution

**User outcome:** With sandboxing enabled, bash commands execute under an OS-level profile that
denies network and restricts writes by default — the in-process blocklist is no longer the only
line of defense.

**Status:** Planned.

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
  `AgentToolsConfig.sandbox_network: bool = False` (deny by default); allowing network also satisfies
  `Capability.NETWORK` checks and vice versa.
- **Storage:** Generated profiles are temp files cleaned up after execution; nothing persisted
  beyond transcripts.
- **Integration:** `BashTool` wraps `argv` after `approve_command` classification; the blocklist and
  allowlist remain as backstops. Command timeouts and output caps behave identically under wrap.
- **Safety:** The backend contract is deny-by-default; wrap failures fail closed (command does not
  run unsandboxed in `required` mode). Documented honestly: seatbelt is deprecated-but-functional on
  macOS and its profile language is the scope boundary; bubblewrap availability depends on the host.
  Platform matrix documented; tests skip when no backend exists (guard like `pytest.importorskip`).
- **Docs:** New `docs/SANDBOXING.md`: threat model (what the sandbox does and does not prevent),
  backend support matrix, profile contents, mode semantics, and the explicit statement that
  `off` preserves today's behavior.
- **Tests:** Unit with a fake backend: wrap composition, fail-closed semantics, mode matrix. E2E
  (`tests/e2e/test_hh08_sandbox.py`): marked to run only when a backend is available; asserts a
  network-touching command fails under deny-network and a normal build command succeeds under wrap.

**Acceptance criteria**

1. Under `required` with no backend, commands fail with a precise diagnostic and nothing executes.
2. Under an available backend with network denied, a command attempting an outbound connection
   fails while a plain compile/test command succeeds with identical stdout capture.
3. `off` (default) produces byte-identical behavior to today.
4. Sandbox wrapping never bypasses allowlist/blocklist classification or audit recording.
5. Profile/temp cleanup leaves no artifacts behind.

**Dependencies:** HH-01 (env allowlist and output caps compose with the sandbox).

**Metrics:** sandbox mode distribution; wrap failures; commands denied by OS policy vs blocklist;
platform availability rate.

**Non-goals:** container-based isolation (Docker); seccomp/APPARMOR authoring; sandboxing file
tools (they are already path-contained); remote execution.

### HH-09: Agent task evaluation harness

**User outcome:** A maintainer runs one command and gets reproducible, scored results for the agent
on a curated task benchmark (deterministic mock provider in CI, configured providers locally), with
a non-zero exit when gates regress — the missing counterpart to RE-01's retrieval benchmark.

**Status:** Planned.

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
  Transcripts from HH-04 attach as evidence per case.
- **Safety:** CI runs require no credentials and no network; configured-provider runs require an
  explicit flag and print a cost warning. Artifacts redact secrets and absolute paths (same rules as
  HH-04).
- **Docs:** Benchmark authoring guide (case anatomy, checks, forbidden paths, cohort discipline,
  "never tune on the test split"), gate semantics, baseline refresh workflow, and the conformance
  suite's scope.
- **Tests:** Unit: scoring math, artifact fingerprints, tolerance comparison, forbidden-path
  detection. E2E (`tests/e2e/test_hh09_agent_evals.py`): mock-provider benchmark executes end-to-end
  from a clean install, produces an artifact, compares against a checked-in baseline, and exits
  non-zero on a seeded regression; the provider-conformance suite runs against `MockLLMProvider`
  without network.
- **Acceptance criteria**

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
(transcript evidence), and conceptually RE-01's artifact vocabulary from `plan2.md` (independent of
its implementation).

**Metrics:** benchmark pass rate by cohort; iterations/token/cost per case; gate regressions caught;
conformance drift incidents; CI runtime.

**Non-goals:** LLM-as-judge scoring; SWE-bench as a CI gate (optional maintainer harness later);
auto-generating tasks from history; leaderboard; multi-repo quality claims.

## Cross-slice engineering requirements

- New modules are added to `[tool.mypy] files` in the same PR that introduces them; `mypy` stays
  green in CI.
- Every new dataclass config field ships with `to_dict`/`from_dict` round-trip tests and a default
  that preserves current behavior (all `AgentLoopConfig`/`AgentConfig` additions are defaulted).
- Persisted artifacts (`runs`, `checkpoints`, `evaluations`) are schema-versioned, written atomically
  with fsync (the `sessions.py` pattern), redacted via `sessions.redact_secrets`, and use
  repository-relative paths.
- New user-facing failures reuse the `FailureKind` taxonomy and, where MCP-visible, the
  `mcp_protocol.py` envelope codes — no new ad-hoc error strings across surfaces.
- Every slice's e2e file carries `pytest.mark.e2e`, runs the real loop/tools with
  `patch_embeddings_factory` and `MockLLMProvider`, and passes in the full `uv run pytest` suite —
  including `filterwarnings = error`.
- No slice adds a required third-party dependency without an ADR-level justification in the slice;
  optional dependencies must degrade gracefully when absent.
- Each slice updates the docs it names and lands with ruff/format/mypy/pytest/build/twine green in
  the supported Python matrix.

## Recommended implementation order

1. **HH-01** — small, high-risk safety fixes first; every later slice builds on truncated output and
   deterministic edits.
2. **HH-02** — the loop's error classification is the foundation for context and streaming work.
3. **HH-03** — context management depends on HH-02's classification and usage plumbing.
4. **HH-04** — transcripts/costs unlock evidence for everything after, including HH-09.
5. **HH-05** — streaming on top of the now-stable loop core.
6. **HH-06** and **HH-07** — independent of each other; either may follow HH-05 in either order.
7. **HH-08** — sandboxing composes with HH-01's environment/output policies.
8. **HH-09** — the evaluation harness last, once the loop it measures is stable and observable.

HH-06 may proceed in parallel with HH-04/HH-05 (its only coupling is recording rollback events).
HH-09 must not start before HH-02/HH-03 land: evaluating an unstable loop produces noise, not
signal.

## Definition of complete

The harness phase is complete when all nine slices are validated and:

- a long, tool-heavy agent task survives transient provider errors, stays inside the model's context
  window, and reports real token usage and cost;
- no tool call can leak environment secrets, exceed output budgets, or apply an ambiguous edit, and
  optional OS sandboxing denies network by default;
- every run leaves a redacted, inspectable local transcript, and failed runs are reversible;
- approvals bind to exact diffs with once/session/deny ergonomics, and planning is user-controllable;
- interactive sessions stream tokens and tool events live with no policy bypass;
- and a deterministic agent benchmark with CI gates protects the loop against regressions, the same
  way RE-01 protects retrieval.
