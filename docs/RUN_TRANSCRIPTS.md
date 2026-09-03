# Run Transcripts and Cost Ledger

This document describes the local run transcripts and cost ledger (slice HH-04): every agent run
records a redacted JSON Lines transcript under the project, the CLI can list, inspect, and delete
past runs, and final agent reports carry a per-run token-usage and cost line.

**Nothing is uploaded.** Transcripts are written to `.ctxai/runs/` inside your repository, redacted
before anything is persisted, and read back only by the local CLI. The recorder has no network
code, no telemetry, and no exporter — storage is a local file and nothing else.

## Where transcripts live

```
<project>/.ctxai/runs/<run_id>.jsonl
```

- One file per agent run; the file name is the run id.
- The directory is repository-scoped, exactly like `.ctxai/sessions/`. It is always inside the
  project — there is no global (`~/.ctxai`) fallback for runs, so transcripts never leave the
  repository they describe.
- `run_id` is the existing `ToolExecutionContext.request_id` when the caller pins it (the one-shot
  `ctxai code` command does this: one process, one run), otherwise a fresh `uuid4` hex per run. In
  interactive chat every message is a distinct run and gets a fresh id, so transcripts never
  overwrite each other.

## Event schema

Each line is one JSON object — a `RunEvent` (`agent/run_recorder.py`). **Every line carries
`schema_version`; there is no separate schema header line. Line 1 is the `run_started` event**, which
also carries the schema version. (Documented implementation choice: this keeps the file a pure event
stream — every line parses identically, and schema checks are per-line.)

| Field            | Type     | Meaning                                                        |
|------------------|----------|----------------------------------------------------------------|
| `schema_version` | int      | On-disk schema version (currently `1`).                        |
| `run_id`         | string   | Identifier shared by every event of the run.                   |
| `seq`            | int      | Monotonic per-run sequence number starting at `1`.             |
| `timestamp`      | string   | ISO-8601 UTC timestamp of the event.                           |
| `kind`           | string   | One of the event kinds below.                                  |
| `payload`        | object   | Redacted, path-normalized event payload.                       |
| `usage`          | object?  | Optional `UsageRecord` for `llm_call` events (tokens only).    |

A `UsageRecord` holds `provider`, `model`, `prompt_tokens`, `completion_tokens`, `total_tokens` —
token counts only, never message content. The `usage` records on `llm_call` events are exactly the
records in the run's `UsageLedger`, so per-run totals always equal the sum of transcript usage.

### Event kinds

| Kind               | Recorded when                                                              |
|--------------------|----------------------------------------------------------------------------|
| `run_started`      | The run begins (line 1). Carries the goal, initial state, provider/model, and loop settings. |
| `user_message`     | The user message that started the run.                                     |
| `llm_call`         | Every LLM call that returned a response, with its `UsageRecord`. Includes calls that returned an error/`length` response (tokens were still billed). |
| `tool_call`        | The loop is about to execute a tool call (parameters included, redacted).  |
| `tool_result`      | The tool finished (success flag, result, error, metadata — redacted). Policy denials also appear here. |
| `approval`         | A human approval was requested for a mutation/command, with the decision.  |
| `state_transition` | The `TaskRun` workflow state changed (retrieve → approve → execute → …).   |
| `check`            | A verification command ran against a mutated tree, with its outcome.       |
| `compaction`       | The loop compacted the context (HH-03): token counts before/after, elided messages. |
| `cancellation`     | The run was cancelled (HH-02 cancellation path).                           |
| `rollback`         | Reserved for HH-06 (checkpoint/rollback); the recorder accepts the kind already. |
| `run_completed`    | The run finished: final status, failure kind, changed files, checks, plan progress, and usage totals. |

Transcripts are append-only per run: writes are one complete JSON line per event (flushed
immediately) and fsynced on close. An interrupted run therefore leaves a shorter but still
parseable transcript — every line is a whole event, and every completed *or* failed run ends with a
`run_completed` event.

## Redaction guarantees

Before any event is written, its payload passes through two stages (in
`agent/run_recorder.py`):

1. **Path normalization** — absolute paths inside payloads are rewritten: paths inside the project
   become repository-relative (`/home/me/project/src/x.py` → `src/x.py`) and the user's home
   directory becomes `~`. Transcripts never contain absolute home paths.
2. **Secret redaction** — the shared `sessions.redact_secrets` primitive redacts
   secret-bearing keys (`api_key`, `token`, `password`, `authorization`, …) and common credential
   shapes (`sk-…`/`ghp_…`/`github_pat_…` tokens, `Bearer` headers, `key=value`/`key: value`
   assignments) recursively across tool parameters, tool results, messages, and approvals.

Seeded-secret acceptance tests (`tests/e2e/test_hh04_run_transcripts.py`) prove that API keys
appearing in tool output never reach the transcript.

## Retention

```toml
[behavior]
record_runs = true    # default: on (local-only, redacted)
run_retention = 50    # default: keep the newest 50 transcripts per project
```

At every run start the oldest transcripts are deleted (oldest first, by file modification time)
until at most `run_retention - 1` old transcripts remain, so the fresh run fits inside the window.
Retention cleanup is scoped to the resolved `.ctxai/runs/` directory — nothing outside it is ever
touched. A non-positive `run_retention` is rejected at configuration load.

## CLI

```bash
ctxai runs list                     # newest-first table: status, events, tokens, calls, cost
ctxai runs list --limit 10 --json   # versioned JSON envelope
ctxai runs show RUN_ID              # every event, in order
ctxai runs show RUN_ID --kind llm_call
ctxai runs show RUN_ID --json       # raw on-disk events (matching schema version)
ctxai runs delete RUN_ID            # delete one transcript (no confirmation)
ctxai runs delete --all             # delete all transcripts (asks for confirmation)
```

All commands accept `--project-path/-p` to target a project other than the current directory.
Deletion is scoped to the resolved runs directory; run ids are validated against path traversal.
`runs list --json` emits `{"schema_version": 1, "runs": [...]}`; `runs show --json` emits
`{"schema_version": <on-disk>, "run_id": ..., "events": [...]}` matching the file byte-for-byte.

## Usage and cost on final reports

When the provider reported usage, chat and one-shot final reports append one dim line:

```
usage: 1,234 prompt + 567 completion tokens over 3 call(s); cost: $0.0123
```

When the model has no price-table entry the line says so explicitly instead of showing a number:

```
usage: 300 prompt + 10 completion tokens over 1 call(s); cost: unknown (no price entry for mock-model-v1)
```

No usage reported → no usage line at all. Unknown cost is **never** rendered as `$0.00`.

## Cost table coverage

`agent/costing.py` holds a small checked-in table (`PRICES_PER_1M_TOKENS`) of USD list prices per
1M prompt/completion tokens, last reviewed 2026-09. Covered model ids:

- **Anthropic**: `claude-3-5-sonnet-20241022`, `claude-3-opus-20240229`, `claude-3-haiku-20240307` (and
  the short aliases).
- **OpenAI**: `gpt-4o`, `gpt-4o-mini`, `gpt-4-turbo`, `o1`, `o1-preview`, `o1-mini`.
- **OpenRouter** (as documented in the chat provider catalog): `anthropic/claude-3.5-sonnet`,
  `anthropic/claude-3-opus`, `openai/gpt-4o`, `openai/o1`, `openai/o1-mini`,
  `deepseek/deepseek-r1`, `deepseek/deepseek-chat`, `google/gemini-pro-1.5`,
  `meta-llama/llama-3-70b-instruct`.

Vendor-prefixed ids also resolve through the documented alias rule: an id like
`openai/gpt-4o-mini` that is not in the table is retried once with its vendor prefix
(`anthropic/`, `openai/`, `deepseek/`, `google/`, `meta-llama/`) stripped. Everything else is
unknown → `None` → "cost unknown".

**How to extend:** add one line to `PRICES_PER_1M_TOKENS` with the model id exactly as the provider
reports it in usage records, as `(prompt_price, completion_price)` per 1M tokens, and update the
"last reviewed" comment. Prices are indicative list prices; verify against the provider's current
pricing page.

## Disabling

Set `record_runs = false` under `[behavior]` in `.ctxai/config.toml` (project or global). No file
is written under `.ctxai/runs/` — the agent loop uses a no-op recorder. Recording failures (disk
full, permissions, unparsable payload) are surfaced as log diagnostics and never fail a run.

## Where the pieces live

- `src/ctxai/agent/run_recorder.py` — `RunEvent`, `RunEventKind`, `RunRecorder` (atomic line
  appends, fsync on close, failure isolation), `NullRunRecorder`, `create_recorder`, `prune_runs`.
- `src/ctxai/agent/costing.py` — `PriceTable.estimate_cost`, `estimate_run_cost`,
  `format_unknown_cost`.
- `src/ctxai/agent/core.py` — the loop's recording hooks (`_record_event`, `_start_recording`,
  `_finalize`) and `AgentLoopConfig.run_id`.
- `src/ctxai/agent/workflow.py` — `TaskRun.to_event_payloads()` (transitions/approvals/checks
  payloads, drained at tool-batch boundaries so recording stays out of the loop's control flow) and
  `UsageRecord.to_dict`/`from_dict`.
- `src/ctxai/commands/runs_command.py` — runs CLI logic and `format_usage_cost_line`.
- `src/ctxai/app.py` — the `ctxai runs` sub-app.
- Tests: `tests/test_run_recorder.py`, `tests/test_costing.py` (unit),
  `tests/e2e/test_hh04_run_transcripts.py` (acceptance, marked `e2e`).

MCP note: the MCP server's query tools execute retrieval only and do not run the agent loop, so
there is no MCP-driven agent run to record today. When agent runs become reachable through MCP they
will record through the same recorder; responses may then include the `run_id`.
