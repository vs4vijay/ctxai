# Agent Loop Resilience

This document describes the resilient agent-loop guarantees (slice HH-02): how transient provider
failures are retried, which errors fail fast, how cancellation unwinds, and how the loop detects
repetitive tool use. Together with [docs/TOOLS.md](TOOLS.md) (hardened tool execution) it defines
the behavioral contract of the agent loop.

## Retry policy

Provider calls made by the agent loop are wrapped in a retry policy
(`RetryPolicy` in `src/ctxai/agent/resilience.py`) with these defaults:

| Field           | Default                                     | Meaning                                        |
|-----------------|---------------------------------------------|------------------------------------------------|
| `max_retries`   | `3`                                         | Retries after the initial attempt              |
| `base_delay_s`  | `1.0`                                       | Delay ceiling for the first retry wait         |
| `max_delay_s`   | `30.0`                                      | Upper bound for any single delay               |
| `retry_kinds`   | `{RATE_LIMIT, TIMEOUT, TRANSPORT}`          | Provider error kinds eligible for retry        |

Backoff is exponential with **full jitter**: the ceiling for attempt `n` (0-based failed attempt)
is `min(max_delay_s, base_delay_s * 2 ** n)` and the actual wait is drawn uniformly from
`[0, ceiling]`. Jitter avoids synchronized retry storms across concurrent clients.

**Only the LLM call is retried.** Tools are never re-executed by a retry: a retry replays the
provider request only, after the previous tool batch has completed and been recorded. The sync
provider call also runs in a worker thread (`asyncio.to_thread`) so it never blocks the event loop
shared with the MCP server and dashboard.

Chat and one-shot surfaces print every wait as a structured line so runs are observable:

```
retry 2/3 after 2.1s (rate_limit)
```

## Error-kind → behavior

Every exception raised by the provider call — and every response with
`finish_reason == "error"` — is normalized through `BaseLLMProvider.normalize_error` into a
`ProviderErrorKind` before the loop decides what to do:

| ProviderErrorKind | Behavior |
|-------------------|----------|
| `RATE_LIMIT`      | Retry with bounded exponential backoff. On exhaustion the run fails with a provider-qualified reason (no recovery prompt). |
| `TIMEOUT`         | Same as `RATE_LIMIT`. |
| `TRANSPORT`       | Same as `RATE_LIMIT`. |
| `AUTHENTICATION`  | **Fail fast** within one iteration: no retries, no iteration burn, no recovery prompt. The message names the provider, e.g. `Provider AnthropicProvider error (authentication): ...`. |
| `UNSUPPORTED`     | **Fail fast** with a provider-qualified message (same rules as authentication). |
| `INVALID_RESPONSE`| Exactly **one** recovery prompt is injected so the model can correct the malformed exchange; a second malformed response fails the run without further recovery. |
| `CANCELLED`       | Treated as cancellation (below), never as a failure to recover from. |

Exceptions raised elsewhere in the loop (bugs, context bookkeeping) keep the historical behavior:
one recovery prompt is injected and the loop continues. Recovery prompts are therefore reserved for
malformed responses and internal errors — they are never injected for retries, cancellation, or
fail-fast terminations.

## Cancellation

Cancellation is cooperative and clean, on both surfaces:

- **Chat (Ctrl+C):** `commands/chat_command.py` installs an `asyncio.Event` on
  `AgentLoopConfig.cancel_event`. A `KeyboardInterrupt` sets the event; the loop checks it before
  every iteration and inside every retry wait, completes or fails the current tool call atomically,
  and unwinds without injecting a recovery prompt.
- **Task cancellation:** an `asyncio.CancelledError` raised in the loop is caught in its own branch
  (it is a `BaseException`, never swallowed by `except Exception`).

On cancellation the loop:

1. Marks the `TaskRun` failed with `FailureKind.INFRASTRUCTURE_FAILURE`.
2. Persists the conversation through the existing `SessionStore`
   (`AgentLoopConfig.session_store` / `session_name`) so interactive sessions survive.
3. Returns the normal status-bearing final report (`Status: failed`, evidence intact).

Because tools complete or fail atomically per call before the next cancel check, cancellation cannot
leave a half-written file from an in-flight tool. Retries likewise never re-execute tools.

MCP surfaces map `ProviderErrorKind.CANCELLED` onto the existing `cancelled` envelope code
(`_provider_error_code` in `commands/server_command.py`), with `TIMEOUT` mapped to `timeout` and all
other provider errors to `internal_error`.

## Loop detection

The loop keeps a hash window of the last `AgentBehaviorConfig.loop_break_threshold` (default `3`)
tool-result tuples (SHA-256 of the formatted results per iteration). When the window fills with one
identical hash — that is, the same tool calls keep producing identical results — the loop breaks and
returns `run.final_report(...)` so status, changed files, checks, and plan progress survive, instead
of the historical bare string.

The max-iterations exit also returns `run.final_report(...)`.

Set a different threshold in the agent behavior configuration:

```toml
[behavior]
loop_break_threshold = 5
```

## Where the pieces live

- `src/ctxai/agent/resilience.py` — `RetryPolicy`, `RetryNotice`, `backoff_delay`, `call_with_retry`,
  `format_retry_notice`.
- `src/ctxai/agent/core.py` — `_call_llm` (retry + `asyncio.to_thread`), the `ProviderErrorKind`
  mapping, cancellation handling, hash-window loop detection, session snapshot on cancellation.
- `src/ctxai/agent/workflow.py` — `classify_provider_failure` maps provider error kinds into the
  shared `FailureKind` taxonomy (all map to `infrastructure_failure`).
- `src/ctxai/agent/config.py` — `AgentBehaviorConfig.loop_break_threshold`.
- Tests: `tests/test_resilience.py` (policy/backoff/cancellation), `tests/test_agent_resilience.py`
  (loop mapping, cancellation, loop detection, event-loop responsiveness),
  `tests/e2e/test_hh02_resilient_loop.py` (acceptance criteria end to end).
