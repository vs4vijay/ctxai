# Checkpoints and Rollback

This document describes local run checkpoints and rollback (slice HH-06): before the agent mutates
a file for the first time in a run, ctxai captures the file's pre-mutation bytes (or a `created`
marker) under the project, and one command — `ctxai checkpoints restore` — returns every touched
file byte-identically to its pre-run state, including files the run created or deleted.

**Nothing is uploaded.** Checkpoints are written to `.ctxai/checkpoints/` inside your repository
and read back only by the local CLI. There is no network code, no telemetry, and no exporter.

## Relationship to git

ctxai does **not** rewrite history, create commits, stash, or branch. The checkpoint is a pure
shadow copy inside `.ctxai/` — it works identically in git and non-git projects. The only thing
git is used for is recording the repository `HEAD` commit hash in the manifest, for context when
auditing a checkpoint. Your working tree and index are never touched by capture; only an explicit
`checkpoints restore` writes to the working tree.

## What is captured

Checkpointing is wired through the existing `TaskRun` mutation boundary — no tool changes. When a
run's `write_file` or `edit_file` call has passed policy **and** approval and is about to execute,
the loop captures the target file's pre-mutation state — but only on the **first touch** of that
file in the run (later mutations of the same file reuse the first capture).

| Capture kind | When | Restores to |
|--------------|------|-------------|
| `file` | The target existed at first touch | Pre-mutation bytes are written back (or the file is recreated if it was deleted later in the run) |
| `created` | The target did not exist at first touch | The file is deleted (pre-run state was "absent") |

**Honest scope limit:** only the structured file tools name a target path before mutating, so only
they can be pre-captured. Files deleted through a bash command (e.g. `rm`) are restorable only when
a structured tool touched them earlier in the same run — the earlier capture holds the pre-run
bytes and `restore` recreates them. A bash command that creates or modifies a *never-captured* file
is outside checkpoint capture. Capture failures and refusals (symlinks, paths escaping the project
root, the size cap) are logged diagnostics that never block the run; the checkpoint is then
honestly partial.

## Where checkpoints live

```
<project>/.ctxai/checkpoints/<run_id>/
    manifest.json          schema-versioned, atomic write + fsync
    files/<sha256>.blob    content-addressed pre-mutation bytes
```

- One checkpoint per run; `checkpoint_id == run_id` (1:1 by design). `ctxai runs show RUN_ID` and
  `ctxai checkpoints restore RUN_ID` therefore address the same run.
- The manifest is atomically rewritten (temp file + fsync + rename — the `sessions.py` pattern)
  every time a file is added and again at finalization. A run that never mutates a file leaves no
  directory behind.
- Per-file content is stored under its content hash, and restore always looks blobs up by hash —
  manifest paths can never be used for path traversal.
- Manifest metadata is redacted with the same `sessions.redact_secrets` primitive as transcripts.
  Blob files store the raw pre-mutation bytes **by design**: restore must be byte-identical, so
  redaction would defeat the purpose. Blobs are only ever read back into the file they were
  captured from.
- A run that crashes without finalizing leaves an `open` (unfinalized) checkpoint; restoring it
  skips the stale-worktree check (there is no post-run hash to compare against) and is a valid
  recovery path.

### Manifest fields

`schema_version`, `checkpoint_id`, `run_id`, `created_at`, `updated_at`, `retained` (the run
succeeded — kept for audit), `status` (`open` | `finalized`), `git_head`, `project_root`,
`bytes_captured`, `cap_exceeded`, and `files`: each entry carries `path` (repository-relative
POSIX), `kind` (`file` | `created`), `sha256`/`blob`/`size` of the pre-mutation content, and —
recorded at finalization — `post_run_present`/`post_run_sha256` of the file at run end.

## Restore semantics and the stale-worktree refusal

```
ctxai checkpoints restore CHECKPOINT_ID
```

shows the affected files with their capture kinds and asks for confirmation — restore never runs
silently. The restore is planned for every captured file **before** anything is modified:

1. **Stale check** — for finalized checkpoints, the current content hash of each target is
   compared with the post-run hash recorded at finalization. A mismatch (or a file that appeared /
   disappeared after the run) means the working tree moved on; the restore is **refused** with a
   per-file reason and nothing is modified. `--force` bypasses exactly this staleness refusal.
2. **Apply** — pre-run bytes are written back (`restored`), files the run created are deleted
   (`deleted`), files the run deleted after capturing are recreated (`recreated`), and no-ops are
   reported (`skipped`).

Safety invariants:

- Targets are validated as safe repository-relative paths (no absolute paths, no `..`, no
  backslashes or drive prefixes) and resolved inside the project root; a symlink at a target path
  is refused, so restore can never follow a link out of the project or clobber an unexpected
  destination. Deleted-file recreation cannot resurrect paths outside the project.
- Hard safety refusals (unsafe path, symlink target, missing checkpoint blob) apply regardless of
  `--force`; only the staleness refusal is forceable.
- The restore is recorded as a `rollback` event appended to the run's HH-04 transcript (with the
  sequence numbering continued) when that transcript exists — see
  [docs/RUN_TRANSCRIPTS.md](RUN_TRANSCRIPTS.md).

## CLI

```bash
ctxai checkpoints list                    # newest-first table: created, status, files, bytes
ctxai checkpoints list --run RUN_ID       # only one run's checkpoint
ctxai checkpoints list --json             # versioned JSON envelope
ctxai checkpoints restore CHECKPOINT_ID   # shows the file list, asks for confirmation
ctxai checkpoints restore CHECKPOINT_ID --force   # bypass the stale-worktree refusal
ctxai checkpoints delete CHECKPOINT_ID    # delete one checkpoint
ctxai checkpoints delete --all            # delete every checkpoint (asks for confirmation)
```

All commands accept `--project-path/-p` to target a project other than the current directory.
Checkpoint ids are validated against path traversal; deletion is scoped to the resolved
`.ctxai/checkpoints/` directory.

## Retention and size cap

```toml
[behavior]
checkpoint_retention = 20          # keep at most this many checkpoint directories per project
checkpoint_max_bytes = 52428800    # per-run capture cap (50 MB)
```

At every run start the oldest checkpoint directories beyond `checkpoint_retention - 1` are deleted
(oldest first, by manifest `created_at`), so the fresh run fits inside the window — the same
pattern as run transcripts. Retention is scoped to the resolved `.ctxai/checkpoints/` directory;
unrelated files are never touched.

When a run's captured bytes would exceed `checkpoint_max_bytes`, further captures stop with a
diagnostic and `cap_exceeded` is recorded in the manifest: the run proceeds and the checkpoint is
honestly partial. Both values are validated at configuration load.

## What restore does not do

Non-goals (per the plan): no git stash/branch automation, no undo across multiple runs (each
checkpoint restores exactly its own run's first-touch state), no interactive hunk-level revert, and
no conflict merging. Restoring is a plain byte-for-byte write-back, not a three-way merge.

## Where the pieces live

- `src/ctxai/agent/checkpoints.py` — `Checkpoint`, `CheckpointFile`, `CheckpointManager`
  (capture, finalize, restore, list, delete, prune), `CaptureKind`, `CaptureOutcome`,
  `RestoreResult`/`FileRestore`.
- `src/ctxai/agent/workflow.py` — the `TaskRun.before_tool` mutation boundary: `_capture_mutation`
  captures pre-mutation bytes after approval and before execution.
- `src/ctxai/agent/core.py` — `AgentLoopConfig.checkpoint_manager` (explicit DI; surfaces that do
  not wire it get no checkpoints) plus the loop's `_start_checkpointing`/`_finalize_checkpoints`.
- `src/ctxai/agent/config.py` — `AgentBehaviorConfig.checkpoint_retention`/`checkpoint_max_bytes`.
- `src/ctxai/commands/checkpoints_command.py` — CLI logic and `record_rollback_event`.
- `src/ctxai/app.py` — the `ctxai checkpoints` sub-app.
- Tests: `tests/test_checkpoints.py` (unit), `tests/e2e/test_hh06_checkpoints.py` (acceptance,
  marked `e2e`).
