# Agent Task Evaluation Harness (HH-09)

`ctxai eval agent` runs the real agent loop against a curated task benchmark and
scores the results reproducibly. It is the agent-side counterpart to the
retrieval benchmark ([docs/RETRIEVAL_BENCHMARK.md](RETRIEVAL_BENCHMARK.md)) and
shares one artifact discipline with it: canonical-JSON content fingerprints,
secret redaction, atomic writes, unavailable-with-reason metrics, and
declared-tolerance baseline gates.

```bash
ctxai eval agent tests/fixtures/agent_benchmark/benchmark.json                    # deterministic mock run (CI path)
ctxai eval agent BENCHMARK --cases hello-file,fix-typo                            # subset
ctxai eval agent BENCHMARK --baseline PATH --fail-on-regression                   # gate against a prior artifact
ctxai eval agent BENCHMARK --json                                                 # print the exact artifact JSON
ctxai eval agent BENCHMARK --provider configured                                  # live provider: network + cost, maintainer-only
ctxai eval providers                                                              # mock provider-conformance suite (CI path)
ctxai eval providers --provider anthropic                                          # live conformance: network + cost, maintainer-only
ctxai eval agent validate BENCHMARK                                               # schema validation without running
```

## Benchmark case anatomy

A benchmark document is versioned JSON (`schema_version: 1`) with named cases:

```json
{
  "schema_version": 1,
  "name": "ctxai-agent-core",
  "cases": [
    {
      "id": "hello-file",
      "instruction": "Create a file named hello.txt containing exactly the word hello",
      "cohort": "file-ops",
      "split": "test",
      "setup": {"files": {"README.md": "# fixture\n"}},
      "expected_checks": [
        {"command": "cat hello.txt", "description": "content matches", "expect_output": "hello"}
      ],
      "forbidden_paths": ["README.md"],
      "plan_required": false,
      "max_iterations": 8,
      "mock_script": []
    }
  ]
}
```

- **id** — stable slug; artifact comparison matches cases by id.
- **instruction** — the natural-language task given to the agent.
- **setup.files** — repository-relative files written into a fresh per-case
  fixture project (paths must be repo-relative; `..`, absolute paths, and `~`
  are rejected).
- **expected_checks** — shell commands that must exit 0 after the run (run
  through the real bash policy); `expect_output` additionally requires the
  marker in stdout.
- **forbidden_paths** — paths that must not be created and, when part of the
  setup, must stay byte-identical. Violations are scored as failures, never
  absorbed.
- **plan_required** — the run must have gone through `submit_plan`.
- **max_iterations** — the loop budget; overruns fail the `budget` judgment.
- **mock_script** — mock-provider-only scripted responses (see below);
  validation rejects it only when it is malformed, and the runner ignores it
  for configured providers.

Validation without execution: `ctxai eval agent validate BENCHMARK` (or
`ctxai eval retrieval validate BENCHMARK` for the retrieval schema).

## Scoring

Each case is executed by the real `Agent` loop (real registry, real tools, real
HH-01..HH-07 behavior) inside a fresh fixture project, then judged on five
named dimensions, each recorded in the artifact with a reason when it fails:

| Judgment | Passes when |
|---|---|
| `checks` | every `expected_checks` command exits 0 (and prints `expect_output` when given) |
| `forbidden_paths` | every forbidden path is absent, or byte-identical to its setup bytes |
| `budget` | the run reached a succeeded state within `max_iterations` |
| `plan_workflow` | `plan_required` cases used `submit_plan` |
| `approvals` | every mutation has a matching recorded approval |

Aggregates: pass rate, mean/p95 iterations, mean/p95 selected tokens, and cost
where the model is priced (unavailable-with-reason otherwise — never zero).
Per-case records attach the HH-04 transcript (run id, path, event count) as
evidence.

Artifacts are written to `.ctxai/evaluations/agent/<run_id>.json` (atomic,
redacted, schema-versioned); `--output` selects another project-contained
path. The benchmark file itself is never modified during execution.

## Mock determinism (CI path)

`--provider mock` (the default) drives every case with the packaged
`MockLLMProvider` using the case's `mock_script`. Same scripts, same
deterministic mock embeddings, same result: the shipped benchmark passes
byte-stably apart from volatile fields (timestamps, durations, run ids) —
`ctxai evals.common.strip_volatile` defines exactly what comparisons ignore.
CI requires no network and no credentials.

`--provider configured` uses the real configured provider: it is an explicit
maintainer action, prints a cost/network warning, and is never invoked by
default test paths.

## Provider conformance suite

`ctxai eval providers` executes the conformance suite derived from
`agent/llm/contract.py` `PROVIDER_SPECS`: auth presence, simple chat,
tool round-trip, streaming, and declared-vs-observed capability checks.
Any drift between declared and observed behavior is reported as a failure.
The mock suite runs offline (CI path); `--provider P` runs a live provider on
demand with an explicit cost warning.

## Gates and baselines

- Gate tolerances (checked in `src/ctxai/evals/agent_artifacts.py`):
  `pass_rate` (exact — zero tolerance), `mean_iterations` and
  `p95_iterations` (±1.0 absolute / 20% relative), `token_mean` (±200
  absolute / 20% relative). Latency is reported, never gated.
- `--baseline PATH` compares against a prior artifact; incompatible baselines
  (schema/kind/benchmark fingerprint/config fingerprint/case set) fail
  clearly instead of silently passing. `--fail-on-regression` exits non-zero
  naming every failing gate.
- A fresh deterministic run compares clean against the checked-in baseline
  (`tests/fixtures/agent_benchmark/baseline.json`).

### Baseline refresh workflow

1. Land the agent-behavior change in its own PR with its e2e evidence.
2. Regenerate the baseline in a dedicated follow-up commit:
   run the benchmark with `--json`, save the artifact as
   `tests/fixtures/agent_benchmark/baseline.json`.
3. The PR description must include the artifact diff (fingerprints, per-gate
   deltas). A baseline update is never bundled silently with the change it
   excuses.

## Authoring discipline

- Add regression cases with ids/tags/cohorts; never tune cases against the
  held-out `test` split — use `train`/`dev` while iterating.
- Keep cases deterministic: check file contents with `cat`/`test`, not fuzzy
  judgments; there is no LLM-as-judge scoring in this harness.
- Bounds: at most 1000 cases, 1 MB benchmark document, iteration budgets
  within 1–50.
