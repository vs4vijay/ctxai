# Retrieval Traces (RE-02)

Local, privacy-preserving observability for retrieval: `ctxai query --trace`
(or the `retrieval.trace_mode` configuration) persists one versioned JSON
Lines record per query under `<project>/.ctxai/traces/<run_id>.jsonl`, so a
user can inspect why a search selected its context and diagnose slow, noisy,
or graph-heavy retrieval — without exposing source or queries.

**Nothing is uploaded.** The recorder's only transport is the local file
system; every persisted record carries the proof block
`"network": {"recorder_transport": "local-file-only", "outbound_transports": []}`.
You can verify no outbound transport exists: `grep -rn "http" src/ctxai/retrieval_traces.py`
shows no client code, and no telemetry SDK is a dependency.

## Modes and defaults

| Setting (retrieval.* in config.toml) | Default | Values | Meaning |
|---|---|---|---|
| `trace_mode` | `off` | `off` \| `metrics` \| `full` | Whether runs are persisted at all |
| `trace_query_text` | `hash` | `omit` \| `hash` \| `store` | Query recording (`store` honored only in `full`) |
| `trace_source_preview` | `omit` | `omit` \| `store` | Source previews (`store` honored only in `full`) |
| `trace_retention` | `100` | ≥ 1 | Maximum retained trace files (oldest pruned first) |
| `trace_retention_days` | `30` | ≥ 1 | Maximum age of retained traces |
| `trace_dir` | — | path | Optional override; must stay inside the project |

- **off (default)** — nothing is written. Per-query insight stays in the
  terminal (`ctxai query --explain`), which never persists.
- **metrics** — identity, candidates (chunk ids, citations, component
  ranks/scores, graph paths, decisions, token estimates), selected ids,
  exclusions, stage timings, counts, errors. **No raw query, no source
  content, no embeddings, no credentials.** A configured `store` is coerced
  to `hash`/`omit`.
- **full** — everything above plus raw query text (when
  `trace_query_text="store"`) and bounded source previews (when
  `trace_source_preview="store"`, capped at `trace_preview_chars`, default
  500). Enabling `full` prints a one-line privacy warning on first use.

The query echo in the terminal is itself redacted (credential-shaped
substrings become `[REDACTED]`), so pasting a secret into a query does not
put it on screen or on disk.

## CLI

```bash
ctxai query my-index "why does the scheduler retry" --trace          # one traced query (config mode, off -> metrics)
ctxai eval retrieval BENCHMARK --index IDX --trace                   # trace every benchmark case
ctxai retrieval runs list [--limit N] [--index NAME] [--json]        # newest first; corrupt files skipped with diagnostics
ctxai retrieval runs show RUN_ID [--json]                            # stage timings + ranking funnel
ctxai retrieval runs delete RUN_ID | --all                           # deletion is scoped to the trace directory
```

MCP `query_codebase` responses include `trace_run_id` only when tracing is
enabled; the dashboard adds `/retrieval-runs` (filterable list) and
`/retrieval-runs/{run_id}` (ranking funnel and timings), both read-only and
escaped.

## Redaction guarantees and limits

Every payload passes through the shared `sessions.redact_secrets` (secret-bearing
key names, `sk-`/`ghp_`-style tokens, bearer tokens, key=value pairs, URLs
with credentials) and the run-recorder path normalization (project paths to
repository-relative, home directories to `~`, any POSIX user-home rewritten).
Absolute paths outside the repository are never persisted; paths that cannot
be contained are dropped.

Limits of redaction: content that embeds a secret in a non-recognizable
encoding may not be pattern-matched — keep `trace_source_preview` at `omit`
(default) when indexing untrusted content.

## Retention, deletion, corruption

- Retention runs best-effort after each recorded trace (count, then age).
- `delete RUN_ID` removes exactly the resolved file; `--all` asks for
  confirmation. Deletion never scans beyond the resolved trace directory.
- A corrupt trace file is skipped by `list`/`show` with a diagnostic instead
  of failing; one run per file means concurrent writers cannot corrupt
  previously committed runs.
- Recording failures (permissions, disk) surface as diagnostics on the
  result and never change retrieval ordering or turn a query into a failure.
