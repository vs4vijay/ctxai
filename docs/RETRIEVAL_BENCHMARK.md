# Retrieval Benchmark (RE-01)

`ctxai eval retrieval` measures retrieval quality, latency, and context
efficiency of a real local index against a versioned, checked-in benchmark,
and produces an immutable JSON artifact. Declared gates can fail a run with a
non-zero exit code when metrics regress beyond tolerance, which makes the
benchmark usable as a merge-protection gate (full CI wiring lands with RE-03).

The runner invokes the **production retrieval path** — repository index
discovery, `HybridRetriever`, `ContextAssembler`, and embedding identity
checks — never a simplified in-process reimplementation. No LLM, network, or
evaluator call happens anywhere in the benchmark runner
(`environment.network_access` in every artifact is `"none"`).

## CLI

```bash
# Validate a benchmark without running retrieval
ctxai eval retrieval validate tests/fixtures/retrieval_benchmark.json
ctxai eval retrieval validate BENCHMARK --project-path /path/to/project   # also verifies evidence files/ranges
ctxai eval retrieval validate BENCHMARK --json

# Run the benchmark against an index
ctxai eval retrieval BENCHMARK --index INDEX
ctxai eval retrieval BENCHMARK --index INDEX --project-path /path/to/project
ctxai eval retrieval BENCHMARK --index INDEX --output .ctxai/evaluations/retrieval/my-run.json
ctxai eval retrieval BENCHMARK --index INDEX --baseline PRIOR_ARTIFACT.json
ctxai eval retrieval BENCHMARK --index INDEX --baseline PRIOR_ARTIFACT.json --fail-on-regression
ctxai eval retrieval BENCHMARK --index INDEX --repeat 3   # first repeat warms up, excluded from latency
ctxai eval retrieval BENCHMARK --index INDEX --json       # print the exact on-disk artifact
```

`ctxai eval retrieval run ...` is an explicit alias of the bare form.

**Exit codes:** `0` success (including a baseline comparison that shows no
regression). `1` any failure: benchmark validation errors, missing/unhealthy/
stale index, embedding identity mismatch, out-of-bounds options, unreadable
or incompatible baselines, partial runs, and — with `--fail-on-regression` —
any failing gate. Every failure prints a specific message; none passes
silently.

## Benchmark schema (version 1)

A benchmark is an immutable JSON document. The shipped benchmark for ctxai
itself lives at `tests/fixtures/retrieval_benchmark.json` (20 questions,
cohort `agent`/`pipeline`/`interface`, splits `train`/`dev`/`test`); it is a
test fixture by location but a real product benchmark by schema.

```json
{
  "schema_version": 1,
  "name": "ctxai-retrieval-core",
  "description": "...",
  "cases": [
    {
      "id": "q04-provider-factory",
      "query": "provider factory",
      "tags": ["agent", "llm"],
      "cohort": "agent",
      "split": "train",
      "expected": {
        "files": ["src/ctxai/agent/llm/factory.py"],
        "symbols": [],
        "line_ranges": {"src/ctxai/agent/llm/factory.py": [1, 60]}
      },
      "relevance": {"src/ctxai/agent/llm/factory.py": 3, "src/ctxai/agent/llm/base.py": 2}
    }
  ]
}
```

Field rules (all enforced by `ctxai eval retrieval validate`):

- `schema_version` must be `1`; `name` must be a non-empty string.
- Cases: unique non-empty `id`s; non-empty `query` (≤ 2000 chars); non-empty
  `cohort`; `split` in `train|dev|test`; at most 1000 cases; at most 1 MB per
  benchmark file.
- Evidence paths (`expected.files`, `expected.line_ranges` keys,
  `relevance` keys) must be repository-relative POSIX paths — absolute paths,
  `~`, `..`, and backslashes are rejected.
- `line_ranges` values are `[start_line, end_line]` with `start >= 1` and
  `end >= start`. Format is checked at validation time; whether the range
  fits the actual file is checked at run time (a range beyond the file length
  is a case error and makes the run `partial`).
- `relevance` grades are integers 0–3; an expected file may not be graded 0.

The benchmark is never modified during execution — the runner only reads it.

### Relevance grading

| Grade | Meaning |
|---|---|
| 0 | not relevant (only meaningful for non-expected paths) |
| 1 | marginally relevant (mentions the concept) |
| 2 | relevant (default for expected files without an explicit grade) |
| 3 | core answer (the file you would open first) |

### Split discipline

- `train` — authoring ground; free to edit while wording queries.
- `dev` — the cohort you iterate against when changing retrieval behavior.
- `test` — held out. Never tune retrieval parameters against the test split;
  a change justified on dev data ships without touching test expectations.

Adding a regression case: write the case, assign it to `dev` while you verify
it fails for the right reason, then move it to `test` and record it as a new
held-out expectation. Adjusting the test split and the fix in the same change
defeats the benchmark — land the case first, review the baseline update
separately (see RE-03).

## Metrics

Aggregates are computed per cohort, per split, and overall. A metric that
cannot be computed is marked `"available": false` with a `reason` — it is
never reported as zero.

| Metric | Definition |
|---|---|
| `recall@1/5/10` | Mean over executed cases of the fraction of expected files found in the top-k ranks. |
| `mrr` | Mean of `1/rank` of the first relevant hit (grade ≥ 1) in the candidate ranking. |
| `ndcg@10` | Graded nDCG using exponential gain `2^grade − 1` and log2 discount; ideal ranking is the case's grades sorted descending, truncated to 10. |
| `evidence_precision@5` | Relevant citations (grade ≥ 1) among the top-5 selected context items, divided by 5 — under-filled context is measured as wasted budget. |
| `successful_query_rate` | Fraction of cases where retrieval produced ≥ 1 candidate without an error. |
| `latency_p50_ms` / `latency_p95_ms` | Percentiles of measured per-case end-to-end retrieval latency (index discovery + hybrid ranking + assembly). Reported, never gated. |
| `selected_token_mean` / `selected_token_p95` | Mean/p95 of the assembler's estimated tokens of the selected context. |
| `duplicate_token_ratio` | Across the selected chunks: fraction of cross-chunk identifier-token occurrences that repeat a token already seen in another chunk (see `ctxai/evals/metrics.py` for the exact formula). 0.0 for a single chunk. |
| `graph_contribution_rate` | Explicitly unavailable (`"graph expansion not enabled"`) until IG-03 lands. |

Quality metrics average over cases that executed without error; error cases
are excluded from quality denominators and surface in
`successful_query_rate` and the `errored` counts instead. A cohort whose
cases all failed reports its quality metrics as unavailable
(`"no successful cases"`).

**Confidence intervals:** `recall@5` and `mrr` ship deterministic percentile
bootstrap confidence intervals (default 1000 resamples, fixed seed
`20260904`, `random.Random`). Identical inputs always produce identical
intervals. CIs are reported per group in `confidence_intervals`.

**Ties:** all rank metrics are position-based, so equal scores are handled by
the retriever's deterministic ordering (`-score, file_path, start_line`).

**Concurrency:** cases execute sequentially — each case runs in an isolated
worker with a hard per-case timeout (`per_case_timeout_s`, default 60s), so a
hung case becomes an explicit case error and a `partial` run instead of
hanging the benchmark.

## Artifact

Artifacts are written atomically (fsync + rename) under
`.ctxai/evaluations/retrieval/` by default, named
`<benchmark>-<timestamp>-<runid8>.json`. A user-selected `--output` path must
stay inside the project boundary (same policy as session exports).

An artifact contains:

- `schema_version` (1), `kind` (`retrieval`), `run_id`, `created_at`,
  `duration_ms`, `status` (`complete` or `partial`).
- `benchmark`: name, schema version, case count, and the content-derived
  sha256 **benchmark fingerprint** over the whole canonicalized document.
- `configuration`: sha256 **configuration fingerprint** over the embedding
  identity (provider/model/dimension) plus result-affecting retrieval
  settings (`token_budget`, `candidate_limit`, query recording, bootstrap
  settings). Operational-only settings (`repeats`, per-case timeout) are
  excluded so they do not invalidate baselines.
- `index`: manifest identity — embedding provider/model/dimension, repository
  revision, chunk/file counts, health and staleness at run time.
- `environment`: python/platform/ctxai version and the explicit
  `network_access: "none"` / `evaluator: "local-retrieval"` proof fields.
- `runs`: one record per case — ordered candidates with chunk id,
  repository-relative `file:start-end` citation, component ranks (`reasons`),
  fused score, final rank, estimated tokens, and per-candidate decisions
  (`selected` / `duplicate` / `budget`, with `truncated` flags on selected
  items); per-case metrics; measured latencies and stage timings; expected
  evidence; line-range overlap findings; query text, or a deterministic
  sha256 of it when query recording is disabled.
- `aggregates`: `overall`, `by_cohort`, `by_split`.
- `comparison`: baseline comparison when `--baseline` was given.
- `errors`: run-level errors (a non-empty list means a partial run).

Payloads pass the shared redaction pipeline (`sessions.redact_secrets` plus
absolute-path prefix replacement for the project root, ctxai home, and user
home) before anything is written.

## Baseline comparison and gates

`--baseline PRIOR_ARTIFACT.json` compares the fresh run against the baseline.
Compatibility is checked first; **incompatible artifacts are never compared
as equivalent** and the run exits 1 naming every mismatch:

- artifact `schema_version` differs → rebuild the baseline;
- evaluation `kind` differs;
- benchmark fingerprint differs (the benchmark document changed);
- configuration fingerprint differs (embedding identity or retrieval
  settings changed);
- case set differs (added/removed ids are named);
- cohort set differs.

When compatible, every gated metric is compared per cohort (`overall` plus
each cohort) with checked-in tolerances. A metric regresses when it is worse
than the baseline by more than `max(absolute, relative × |baseline|)`:

| Metric | Direction | abs | rel |
|---|---|---|---|
| recall@1 | higher | 0.05 | 0.05 |
| recall@5 | higher | 0.02 | 0.02 |
| recall@10 | higher | 0.02 | 0.02 |
| mrr | higher | 0.05 | 0.05 |
| ndcg@10 | higher | 0.05 | 0.05 |
| evidence_precision@5 | higher | 0.05 | 0.05 |
| successful_query_rate | higher | 0.0 | 0.0 |
| duplicate_token_ratio | lower | 0.05 | 0.10 |
| selected_token_mean | lower | 50.0 | 0.10 |
| selected_token_p95 | lower | 100.0 | 0.10 |
| latency p50/p95 | reported only | — | — |

Latency deltas are reported but intentionally not gated: noisy wall-clock
timings must not become a hard cross-platform CI gate (a controlled-runner
efficiency threshold is RE-03's concern). Unavailable metrics produce an
`unavailable` gate with the reason, never a failure or a zero.

Without `--fail-on-regression`, regressions are reported in the table and in
the artifact's `comparison.status` but do not change the exit code. With it,
each failing gate is printed (`Gate failed: overall/recall@5 baseline … ->
current …`) and the process exits 1.

## Reproducibility caveats

- Deterministic given: identical repository content, index, embedding
  identity, and configuration. The shipped CI path uses
  `MockEmbeddingProvider` (MD5-seeded vectors) to guarantee this.
- `repeats > 1` treats the first execution as a warm-up: quality comes from
  the first successful execution, latency statistics exclude the warm-up.
- Two artifacts from deterministic runs differ **only** in these documented
  volatile fields: `run_id`, `created_at`, `duration_ms`, per-run
  `timestamp`, `timings`, `latency` (and the derived `latency_p50_ms` /
  `latency_p95_ms`). `ctxai.evals.common.strip_volatile` removes exactly
  those keys — artifact comparison ignores them and nothing else.
- Real embedding models are not bit-deterministic across versions and
  hardware; compare artifacts only when the configuration fingerprint
  matches.
- The benchmark measures retrieval of *this* repository's benchmark cases;
  it makes no universal quality claim (see the slice non-goals).

## Authoring workflow

1. Write or extend the benchmark JSON; run
   `ctxai eval retrieval validate BENCHMARK --project-path .` — all schema
   and repository-level evidence errors are reported together.
2. Add cases to `dev` first; run the benchmark; verify failures mean what
   you think (inspect `runs[].candidates[].reasons` for provenance).
3. Promote reviewed cases to `test`. Never tune on the test split.
4. After a deliberate retrieval change, refresh the baseline in a separate
   reviewable change: run once, store the artifact, commit it as the new
   baseline, and reference both fingerprints in the review.

## Maintainer workflow: baseline refresh and the CI gate (RE-03)

The retrieval-quality CI job (`.github/workflows/pr-gate.yml`,
`retrieval-quality`) runs `scripts/ci_retrieval_eval.py` on every PR: it
builds a fixture project from the checked-in benchmark's expected files,
indexes it with the registered deterministic `mock` embedding provider (no
network, no credentials, no model downloads), evaluates, and compares the
fresh artifact against the checked-in baseline
(`tests/fixtures/retrieval_baseline.json`) with `--fail-on-regression`
semantics. Exit codes: 0 pass, 1 gated regression, 2 incompatible. The gate
never fails on latency: timing is reported and flagged as noisy, never
gated (criterion 3).

### Refreshing the baseline (deliberate, separate change)

```bash
uv run python scripts/ci_retrieval_eval.py \
    --update-baseline tests/fixtures/retrieval_baseline.json
```

The script regenerates the checked-in baseline and prints the reviewable
evidence: the benchmark fingerprint and configuration fingerprint, plus an
artifact diff. Required review evidence for a baseline-update PR:

- the benchmark fingerprint is UNCHANGED (a changed benchmark fingerprint is
  a benchmark change, not a baseline refresh);
- the configuration fingerprint change is explained by the retrieval change
  it accompanies;
- the per-metric/cohort deltas are reviewed and accepted;
- latency deltas are interpreted as noise unless a controlled runner makes
  them reliable.

A baseline update must never be bundled invisibly with the retrieval change
it excuses — land it as a separate reviewed commit/PR.

### Downloading CI artifacts

Failed or successful `retrieval-quality` jobs publish their artifacts
(candidate run, comparison) via `actions/upload-artifact@v4` (job page →
Artifacts). Compare them locally with
`ctxai eval retrieval compare BASELINE CANDIDATE --json`.

### Recovering from incompatible artifacts

When the comparator reports `incompatible` (exit 2), it names every
mismatched identity field — schema version, benchmark fingerprint, case
set/split, embedding identity, retrieval configuration. The remedy is one
of: re-run both evaluations against the same benchmark document (transient
mismatch), refresh the baseline deliberately after an intended benchmark or
configuration change (see above), or rebuild the index if the embedding
identity changed. Incompatible artifacts are never compared as if
equivalent.
