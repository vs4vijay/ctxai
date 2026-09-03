# Agent Tool Safety

This document describes the hardened tool-execution guarantees (slice HH-01): how environment
variables reach subprocesses, how tool output is bounded before it enters the model context, how
`edit_file` applies replacements deterministically, and which command policy is enforced. It also
states the threat model honestly, including what is still out of scope.

## Environment policy for subprocesses

`BashTool` never passes `os.environ` wholesale to a subprocess. `ToolExecutionContext.command_environment()`
builds the child environment from three explicit sources:

1. **Fixed allowlist** — only these variables are copied from `os.environ` (when present):

   `PATH`, `HOME`, `LANG`, `LC_ALL`, `TMPDIR`, `SHELL`, `TERM`, `USER`, `LOGNAME`

   The constant lives in `src/ctxai/agent/tools/execution.py` as `ALLOWED_ENVIRONMENT_KEYS`. `PATH`
   must stay on the list or plain executable names can no longer be resolved by the child.

2. **Opt-in passthrough** — `AgentToolsConfig.env_passthrough: list[str]` names additional variables
   that are copied from `os.environ` when present. This is the supported extension point; add a name
   to the list in `.ctxai/config.toml` (or code) instead of weakening the allowlist. Values are still
   sourced from `os.environ`, never invented.

3. **Explicit extras** — `ToolExecutionContext.environment: dict[str, str]` sets additional variables
   with explicit values and wins over both sources above on conflicts.

Consequently, secrets present in the user's environment (API keys, tokens) are unreachable by any
subprocess unless they are explicitly named in `env_passthrough` or set via `environment`. This is
verified by tests that seed a fake `ANTHROPIC_API_KEY` and assert it never appears in
subprocess-visible environment or audit records.

### How to extend

- **Per project/user (recommended):** set `tools.env_passthrough = ["MY_VAR"]` in the agent tools
  configuration.
- **In code:** construct `ToolExecutionContext.for_project(..., env_passthrough=[...])` or set
  `environment={...}` for explicit values.
- **New baseline variables** (rare, security-relevant): extend `ALLOWED_ENVIRONMENT_KEYS` in
  `execution.py` and update this document and the tests in `tests/test_hh01_tool_policy.py`.

## Output limits

Tool results that can grow without bound are truncated **before** they enter the LLM context:

- `bash` stdout and stderr
- `read_file` formatted content

The bound is `AgentToolsConfig.max_output_chars` (default `20_000` characters). Truncation keeps the
first `max_chars` characters and appends an explicit marker stating the direction and size of the
loss:

```
...[truncated N of M chars]
```

where `N` is the number of characters removed from the tail and `M` is the original character count.
The helper is `truncate_text(text, max_chars, *, label)` in `src/ctxai/agent/tools/output_limits.py`;
it never raises on any input (including empty strings, missing trailing newlines, or invalid limits).

Original sizes are always recorded:

- `bash` audit records include `stdout_chars`, `stderr_chars`, and a `truncated` flag; the tool
  metadata additionally carries `original_stdout_chars`, `original_stderr_chars`,
  `stdout_truncated`, and `stderr_truncated`.
- `read_file` metadata carries `truncated` and `original_chars` (reads do not write audit records —
  the audit log records mutations and commands).
- With `--verbose`, the agent prints truncation and replacement-count diagnostics per tool result.

## Edit semantics

`edit_file` replaces text deterministically and fails closed:

- **Uniqueness rule.** The pattern must match exactly once. Zero matches and multiple matches are
  errors that name the match count (for example, `Edit failed: pattern matched 2 occurrence(s) ...`);
  nothing is written. The error is a recoverable tool error — the model can retry with a more
  specific pattern.
- **`replace_all`.** Set the explicit `replace_all: bool` parameter (default `false`) to replace every
  occurrence; the result then reports the number of replacements.
- **Regex mode.** With `use_regex: true` the pattern is applied via `re.subn` under the same
  uniqueness rule.
- **Whitespace-tolerant fallback.** When the exact match finds nothing, one bounded fallback is
  attempted: runs of spaces/tabs collapse to a single space and trailing whitespace is stripped per
  line — on both the content and the pattern. The fallback must also match exactly once; the
  replacement is then spliced into the **original** bytes at the matched region (the normalized
  content is never written back), so surrounding whitespace is preserved.
- **Strategy metadata.** Successful edits report how the match was located:
  `"exact" | "normalized" | "replace_all"` in the tool metadata and audit details, next to the
  replacement count.

### Approval previews cannot diverge

The approval preview and the applied edit share one implementation. `agent/editing.py` exposes
`apply_edit` (the core routine), `simulate_edit` (tool-call level simulation used by approval
previews), and `edit_diff` (unified diff rendering). `workflow.TaskRun._approval_call` previews via
`simulate_edit`, and `EditFileTool` applies via `apply_edit` — the diff shown at approval time is
byte-identical to the diff of the applied change, including regex edits. If an edit is ambiguous the
preview carries no diff, and execution fails closed with the count-bearing error.

## Command policy

Command policy is consolidated in exactly two places; there is no separate substring matcher
(`AgentToolsConfig.is_bash_command_allowed` and `bash_blocked_commands` were removed):

1. `ToolExecutionContext.approve_command` classifies one shell-operator-free command: destructive,
   network, shell, and inline-code invocations require the matching capability; git mutations are
   denied unless read-only.
2. `AgentToolsConfig.bash_allowed_commands` optionally restricts the executable to an exact-name
   allowlist (matched on the executable's basename).

### Threat-model note

Command classification is an **in-process blocklist/allowlist**. It is not an OS-level sandbox: a
determined or confused model may still ask an allowlisted executable to do harmful things inside the
project boundary. Deny-by-default OS sandboxing (seatbelt/bubblewrap), including network denial, is
tracked separately as HH-08 and is explicitly out of scope here.

## Configuration reference

`AgentToolsConfig` (dataclass, serialized via `to_dict`/`from_dict`; unknown keys from older
configurations are ignored):

| Field | Default | Meaning |
|---|---|---|
| `enabled_tools` | `None` | Tool allowlist; `None` enables all registered tools. |
| `bash_allowed_commands` | `None` | Exact executable-name allowlist; `None` = classification only. |
| `bash_timeout` | `30` | Per-command timeout in seconds. |
| `max_file_size_mb` | `10` | Maximum file size `read_file` will open. |
| `allow_outside_project` | `False` | Allow file operations outside the project root. |
| `max_output_chars` | `20000` | Truncation bound for bash stdout/stderr and read_file content. |
| `env_passthrough` | `[]` | Opt-in `os.environ` variable names forwarded to subprocesses. |

Removed in HH-01: `is_bash_command_allowed()` and `bash_blocked_commands` (the substring matcher) —
see "Command policy" above.
