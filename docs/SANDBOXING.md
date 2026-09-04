# Sandboxed Command Execution (HH-08)

This document describes the optional OS-level sandbox for agent bash commands: what it does, what
it does not prevent, which backends exist on which platforms, and what each mode means. The
in-process command policy (exact-name allowlist plus `ToolExecutionContext.approve_command`
classification) always runs first and remains in force; the sandbox is a second, OS-enforced layer
behind it.

## What the sandbox does

With sandboxing enabled, every command executed by the `bash` tool runs under an OS-level,
deny-by-default profile that:

- **Denies outbound network access** by default (`tools.sandbox_network: false`). A socket open or
  connect fails immediately with an OS permission error instead of reaching the network.
- **Restricts writes** to the command's working directory, the temp directories (`TMPDIR` and the
  system temp locations), and `/dev/null`. Writes anywhere else fail visibly rather than silently.

Timeouts, output caps (`max_output_chars`), environment allowlisting (HH-01), and stdout/stderr
capture behave identically under wrap.

## What the sandbox does not prevent (threat model, stated honestly)

- **The profile language is the scope boundary.** The seatbelt profile used here allows everything
  by default and then denies network and out-of-bounds writes. It does not attempt to jail the
  filesystem like a container, and it does not restrict reads: any file the user can read, the
  sandboxed command can read.
- **Seatbelt is deprecated-but-functional on macOS.** Apple has deprecated `sandbox-exec`; it
  still works on current macOS versions (validated on this host), but Apple may remove it. The
  write-restriction portion of the profile is best-effort: a command that legitimately needs to
  write outside the project (for example a build that writes into a virtualenv under `$HOME`) will
  fail visibly — the fix is to relax the mode, run with the sandbox off, or restructure the
  command.
- **Bubblewrap availability depends on the host.** On Linux, `bwrap` must be installed; there is
  no auto-install. Kernel/user-namespace restrictions on some hosts can make bwrap unusable.
- **No container isolation.** Docker/container-based isolation, seccomp/AppArmor profile authoring,
  sandboxing of file tools (they are already path-contained), and remote execution are explicit
  non-goals.
- **The in-process policy is still the first line of defense.** Classification (`approve_command`)
  and the allowlist run *before* wrapping; the sandbox is a backstop, not a replacement. Approval
  flow (HH-07) is unchanged.

## Backend support matrix

| Backend | Name | Platform | Availability | Enforcement notes |
|---|---|---|---|---|
| `MacOSSeatbeltBackend` | `seatbelt` | macOS | `sandbox-exec` on `PATH` (present by default) | Denies `network*` and `file-write*` outside allowed subpaths; profile language is the scope boundary; deprecated-but-functional |
| `BubblewrapBackend` | `bwrap` | Linux (and other POSIX with bwrap) | `bwrap` on `PATH` (must be installed) | `--unshare-net` (when network denied), read-only root bind, writable cwd bind, tmpfs `/tmp` |
| `NoopBackend` | `none` | any | always | Identity wrapper: runs the command exactly as before HH-08; used by mode `off` and by `auto` when nothing better exists |

Seatbelt is preferred on macOS; bwrap is the fallback where installed.

### Seatbelt profile contents

Generated per command into a temp file (`ctxai-seatbelt-*.sb`), deleted after execution:

```
(version 1)
(allow default)
(deny network*)              ; omitted when tools.sandbox_network is true
(deny file-write*)
(allow file-write* (subpath "<resolved cwd>") (subpath "<resolved TMPDIR>")
      (subpath "/tmp") ... (literal "/dev/null"))
```

All paths are embedded symlink-resolved (seatbelt evaluates canonical paths such as
`/private/tmp`). Strings are escaped, so project paths containing spaces or quotes are safe.

### Bubblewrap argv composition

```
bwrap --ro-bind / /  --bind <cwd> <cwd>  --dev /dev  --proc /proc  --tmpfs /tmp
      [--unshare-net]  -- <argv>
```

The child's `TMPDIR` is adjusted to `/tmp` (the tmpfs mount) since the host temp directory is not
bound into the jail.

## Mode semantics

Configured via `AgentToolsConfig.sandbox` (project `.ctxai/config.toml` under `[tools]`, or
`ctxai config --set tools.sandbox --value <mode>`):

| Mode | Meaning |
|---|---|
| `off` (default) | **Preserves today's behavior byte for byte.** No OS sandbox is applied; the in-process allowlist/classification policy and audit trail are unchanged. |
| `auto` | Use a backend when one is available on the host, with a visible diagnostic (the chat badge and the per-command audit record) when none is; commands then run unsandboxed exactly as in `off` mode. A *wrap failure* never falls back to unsandboxed execution — it fails the command. |
| `required` | Commands fail closed when no backend exists or wrap fails: a precise diagnostic is returned and recorded, and **nothing executes**. |

`AgentToolsConfig.sandbox_network: bool = False` controls network inside an enforcing sandbox:

- `false` (default): the backend profile denies network access.
- `true`: the profile allows network access *and* the `NETWORK` capability check in
  `approve_command` is satisfied (commands like `curl` pass classification). Conversely, network is
  also allowed inside the sandbox when the `NETWORK` capability is satisfied for the command in any
  other way — a context pre-granted `Capability.NETWORK`, or a human approving the network command
  through the HH-07 approval flow (that grant is per-command: it applies to exactly the approved
  command). Without an enforcing backend, `sandbox_network` grants nothing — the in-process
  NETWORK check still applies.

## Audit integration

Sandbox wrapping never bypasses classification or audit recording. Every `bash` audit record
carries `sandbox` (backend name or `None`), `sandbox_diagnostic` (when a mode found no backend),
and `sandbox_network` (when an enforcing backend wrapped the command) alongside the existing
policy fields.

## Configuration reference

| Field | Default | Meaning |
|---|---|---|
| `tools.sandbox` | `"off"` | `off`, `auto`, or `required` (invalid values are rejected at load). |
| `tools.sandbox_network` | `false` | Allow outbound network inside an enforcing sandbox. |

Set with:

```bash
ctxai config --set tools.sandbox --value auto       # or required / off
ctxai config --set tools.sandbox_network --value true
```

or by editing `[tools]` in `.ctxai/config.toml` (project) or the global config file.

## Chat badge

When sandboxing is enabled, chat prints a status line under the banner:

- `sandbox: seatbelt (network denied)` — an enforcing backend is active.
- `sandbox unavailable: ...` — mode is enabled but no backend exists (`auto` will run unsandboxed;
  `required` will fail commands).

With mode `off` nothing is printed.

## Testing

Unit tests use a fake backend to pin the mode matrix, wrap composition, and fail-closed semantics
(`tests/test_sandbox.py`); seatbelt-specific tests are skipped on hosts without `sandbox-exec`.
E2E tests (`tests/e2e/test_hh08_sandbox.py`) run the real loop and tools under an enforcing
backend and skip (importorskip-style guard) when no backend exists, so CI hosts without one stay
green.
