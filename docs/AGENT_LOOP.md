# Agent Loop Resilience

This document describes the resilient agent-loop guarantees (slices HH-02 and HH-03): how transient
provider failures are retried, which errors fail fast, how cancellation unwinds, how the loop
detects repetitive tool use, and how long tool-heavy tasks stay under the model's context window.
Together with [docs/TOOLS.md](TOOLS.md) (hardened tool execution) it defines the behavioral contract
of the agent loop.

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
| `INVALID_RESPONSE`| Exactly **one** recovery prompt is injected so the model can correct the malformed exchange; a second malformed response fails the run without further recovery. A `finish_reason == "length"` response (the model hit its output limit and the payload is truncated) is mapped to this kind, so it gets the same recovery-then-fail treatment instead of returning a cut-off answer or crashing. |
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

## Context window management

Long tool-heavy runs stay under the model's context window automatically (slice HH-03). The loop
performs a budget check **before every LLM call** — including the first call after a resumed
session — so a task that would overflow compacts and continues instead of failing.

### Budget model

```
budget = context_size × context_soft_limit_ratio
```

- **`context_size`** comes from the provider's `get_capabilities().context_size`
  (`ProviderCapabilities`, declared per provider in `agent/llm/contract.py`'s specs; the default is
  100,000). A missing or non-positive value disables the check entirely — compaction never runs on
  an unknown budget.
- **`context_soft_limit_ratio`** (`AgentBehaviorConfig`, default `0.8`) is the fraction of the
  window above which compaction triggers. It is validated to `(0, 1]` at construction. This is a
  *context window* setting; `AgentLLMConfig.max_tokens` remains the separate completion budget and
  the two are never conflated.
- **`keep_recent`** (default `6`) is the number of most recent message **groups** kept verbatim by a
  compaction. A group is an assistant message with tool calls together with all of its paired
  tool-result messages, or any single non-system message — groups are the atomic unit everywhere.

The **estimator** is measured-first: when the provider reports usage, the `prompt_tokens` value from
the most recent call is the context-size basis (an honest measurement of the exact payload the
provider accepted, including tool schemas and formatting overhead). When nothing has been reported —
or after a compaction invalidates the measurement until the next call reports — the loop falls back
to the historical `chars ÷ 4` heuristic over message contents.

Estimator accuracy caveats per provider:

- **Anthropic**: `input_tokens` is reported per call; the measured basis is exact for the request
  that produced it.
- **OpenAI / OpenRouter / GitHub Copilot / custom (OpenAI-compatible)**: usage is extracted from the
  response payload (`usage.prompt_tokens`); exact for the reported request.
- **Ollama**: `prompt_eval_count`/`eval_count` are reported by local models; some models serve
  approximate counts.
- **Providers that report no usage** fall back to `chars ÷ 4`, which ignores tool schemas, message
  envelopes, and tokenizer differences — it can undershoot real prompt tokens significantly. Configure
  a more conservative `context_soft_limit_ratio` for such providers.

### What compaction preserves and elides

`ConversationContext.compact(target_tokens, keep_recent=6, max_output_chars=20_000)` is a pure
function of history + config — no wall-clock, no randomness — and never removes messages:

1. **Caps** every tool-result body at `max_output_chars` (from `AgentToolsConfig`) with an elision
   marker such as `[elided 400 chars of tool result for read_file]`.
2. **Elides** tool-result bodies outside the recent window, replacing them with honest markers that
   name the exact character count removed. The elision decision is **per group**: an
   `assistant(tool_calls)` message and its results move between capped/elided states together, so a
   group is never split. Bodies already elided — or smaller than their own marker would be — are
   left untouched, which makes repeated compactions idempotent.
3. **Summarizes** the turns elided by this compaction with the deterministic `_summarize_messages`
   helper into a single user-role message placed after the leading system messages; a later
   compaction replaces that summary in place (there is exactly one summary slot).
4. **Records** the event as counters/attributes on the context — `compaction_count`,
   `elided_message_count`, `last_compaction` — which are persisted as `compaction` transcript
   events (see [docs/RUN_TRANSCRIPTS.md](RUN_TRANSCRIPTS.md)). No-op compactions (nothing new to
   elide) are not counted.

Always preserved, byte-for-byte: the system prompt, every assistant `tool_calls` payload, and the
`tool_call_id`/`tool_use_id` pairing the OpenAI and Anthropic request validators require. Compaction
never breaks tool-call pairing or removes the system prompt; elision markers are honest about what
was removed; usage capture stores token counts only — never content.

The historical `truncate_old_messages` path (after a final no-tool response) is group-aware for the
same reason: it drops or keeps whole groups, so it can never orphan a tool result.

### Observability

- Chat prints a one-line notice per effective compaction (via the `on_compaction` hook, mirroring
  `on_retry`):

  ```
  context compacted: ~11500 -> ~3100 tokens (14 tool results elided, soft limit 8000)
  ```

- The `/context` chat command reports measured tokens (and their basis), the budget and soft limit,
  compaction count, elided tool-result count, and the last run's provider-reported usage totals.
- Per-run usage is aggregated by `UsageLedger` (`agent/workflow.py`); totals equal the sum of
  per-call provider-reported usage. It is held on `TaskRun.usage` and persisted per run as
  `llm_call` transcript events with a cost estimate ([docs/RUN_TRANSCRIPTS.md](RUN_TRANSCRIPTS.md)).



## Streaming and the event protocol (HH-05)

`process_message` and `stream_message` share one core, `Agent._run_loop`, which emits every
occurrence of interest to a synchronous sink as an `AgentEvent` (`agent/events.py`). The buffered
surface discards the events and returns only the final report, so MCP and one-shot behavior are
unchanged; `stream_message` bridges the sink into an async generator and yields the events live.

- Event kinds (closed vocabulary): `token | tool_call_started | tool_result | approval_required |
  approval_decided | status | usage | final_report`. The stream always ends with exactly one
  `final_report` whose text is identical to `process_message`'s return for the same conversation.
- Provider level: `BaseLLMProvider.stream_chat_events(messages, tools)` yields `StreamEvent` tuples
  (`("text", str) | ("tool_call_delta", dict) | ("usage", dict)`) and returns the complete
  `LLMResponse`. The default implementation falls back to `chat()` and emits one `text` event —
  graceful degradation documented here and reflected honestly in
  `ProviderCapabilities.streaming` (True only when a provider implements `stream_chat_events`).
  Real delta streaming is implemented for anthropic, openai, and openrouter; github-copilot, ollama,
  custom, and nvidia use the buffered fallback.
- `AgentBehaviorConfig.stream_responses: false` forces the buffered provider path (events are still
  emitted, one `token` per call).
- Approval-required tool calls emit `approval_required` (with the proposed diff), invoke the same
  approval callback as the buffered path (the stream pauses until the decision is made), emit
  `approval_decided`, and only then execute — the streaming UI cannot bypass planning/approval
  policy.
- Cancellation during streaming produces the HH-02 outcome (`Status: failed`,
  `infrastructure_failure`); compaction mid-stream emits a `status` event; each LLM call's reported
  usage emits a `usage` event.
- Token deltas are transient UI state: HH-04 transcripts record the same events in streaming and
  buffered mode (final texts, not deltas).
- Chat renders the events via Rich `Live` (token deltas inline, tool activity as dim status lines,
  the final report as a panel). Non-stream-capable providers fall back to buffered rendering with no
  UX regression.

Support matrix (generated from `PROVIDER_SPECS`): see
[docs/PROVIDER_COMPATIBILITY.md](PROVIDER_COMPATIBILITY.md), column "streaming".



## Where the pieces live

- `src/ctxai/agent/resilience.py` — `RetryPolicy`, `RetryNotice`, `backoff_delay`, `call_with_retry`,
  `format_retry_notice`.
- `src/ctxai/agent/events.py` — `AgentEvent`, `AgentEventKind`, `StreamEvent` (the loop's event
  vocabulary).
- `src/ctxai/agent/core.py` — `_run_loop` (the shared core), `_call_llm` (retry + `asyncio.to_thread`,
  usage capture, the `finish_reason == "length"` mapping, streaming drain), the `ProviderErrorKind`
  mapping, cancellation handling, hash-window loop detection, session snapshot on cancellation,
  `_enforce_context_budget`, `CompactionNotice` / `format_compaction_notice`.
- `src/ctxai/agent/context.py` — `ConversationContext.compact` (group-aware elision), the measured
  estimator (`estimate_context_tokens` / `note_reported_usage`), group-aware `truncate_old_messages`,
  compaction counters.
- `src/ctxai/agent/workflow.py` — `UsageLedger`/`UsageRecord` (per-run token aggregation) and
  `classify_provider_failure`, which maps provider error kinds into the shared `FailureKind` taxonomy
  (all map to `infrastructure_failure`).
- `src/ctxai/agent/config.py` — `AgentBehaviorConfig.loop_break_threshold`,
  `AgentBehaviorConfig.context_soft_limit_ratio`, `AgentBehaviorConfig.stream_responses`.
- Tests: `tests/test_resilience.py` (policy/backoff/cancellation), `tests/test_agent_resilience.py`
  (loop mapping, cancellation, loop detection, event-loop responsiveness),
  `tests/test_context_management.py` (compaction, pairing, estimator, ledger, soft-limit math),
  `tests/test_agent_events.py` (event contract, fallback, mock streaming),
  `tests/test_streaming_loop.py` (event sequences, approvals, cancellation, transcripts),
  `tests/test_provider_streaming.py` (anthropic/openai/openrouter `stream_chat_events`),
  `tests/e2e/test_hh02_resilient_loop.py` (HH-02 acceptance end to end),
  `tests/e2e/test_hh03_context_management.py` (HH-03 acceptance end to end),
  `tests/e2e/test_hh05_streaming.py` (HH-05 acceptance end to end).
