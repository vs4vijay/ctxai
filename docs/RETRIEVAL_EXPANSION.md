# Graph-Expanded Grounded Retrieval (IG-03)

Graph expansion seeds from the top base hits of the shared hybrid retrieval
service and adds relationship evidence (callers, callees, imports, tests)
within strict caps. It is **disabled by default** and cannot become the
default until the RE-01 benchmark gate passes (see "The default-enablement
gate" below).

## How expansion works

1. **Seed selection** — the top `seed_count` (default 3) base candidates are
   mapped to graph symbols by chunk metadata (symbol name + repository-relative
   file). Seeds that do not resolve to a graph symbol are skipped with a
   diagnostic.
2. **Bounded traversal** — from each seed, the resolver walks allowlisted edge
   kinds only (`calls`, `imports`, `inherits`, `tests`, `references`;
   `contains` is structural and never boosts). Depth is 1 by default; depth 2
   requires an explicit bounded option. At most `max_neighbors_per_seed`
   (default 8) neighbors per seed and `expansion_cap` (default 24) expanded
   symbols per query are accepted. Visited symbols are tracked, so cycles
   never duplicate evidence.
3. **Deterministic contribution** — an expanded symbol's contribution is
   `seed_base_score × edge_weight^depth × confidence_factor`, with confidence
   factors `exact > probable > unresolved`. Unresolved edges carry evidence
   but never fabricate connections.
4. **Chunk mapping** — expanded symbols map back to indexed chunks (same
   file + symbol name). Symbols without an indexed chunk are counted in
   diagnostics, never added.
5. **Assembly** — expansion only adds candidates; the final context is
   assembled by the existing `ContextAssembler` within the configured token
   budget. Deduplication is by chunk identity; one chunk never enters the
   context twice.

All settings live in `RetrievalConfig` (`graph_*` fields, validated bounds)
and are disabled by default. Identical index + configuration + query produce
identical ordering and selection (stable tie-breaking by score, then chunk id).

## Using it

```bash
ctxai query my-index "why does the scheduler retry" --graph            # expansion on (required: healthy graph)
ctxai query my-index "scheduler retries" --explain                     # selection rationale (base ranks, fusion, exclusions)
ctxai query my-index "scheduler retries" --graph --explain             # both
ctxai eval retrieval BENCHMARK --index IDX --graph --json              # benchmark with expansion
ctxai eval retrieval compare-graph NO_GRAPH.json GRAPH.json --json     # the default-enablement gate
```

`--graph` (explicit) **fails** when the graph is absent, stale, unsupported,
or corrupt. Without flags, expansion follows configuration: when enabled but
unavailable, retrieval falls back to base behavior with a visible WARNING
diagnostic on stderr. Every selected item carries a base or graph reason;
graph-expanded items cite both the source citation and the relationship path
(`seed -[edge]-> expanded`, with confidence, depth, and contribution).

## The default-enablement gate (acceptance criterion 5)

`ctxai eval retrieval compare-graph` compares a graph-enabled artifact
against its no-graph baseline on the same benchmark/index/embeddings:

- **Compatibility** — schema, evaluation kind, benchmark fingerprint, case
  set, cohort set, embedding identity. The configuration fingerprint is
  expected to differ (the flag changes behavior); anything else incompatible
  fails the comparison.
- **Regression gates** — per metric/cohort, using the checked-in absolute and
  relative tolerances. Recall@5 and MRR are the headline quality gates.
- **Improvement requirement** — at least one pre-registered
  relationship-oriented metric must improve. The derived `graph-relationship`
  cohort (benchmark cases whose expected evidence participates in cross-file
  relationship edges) and the `graph_contribution_rate` metric are
  pre-registered for this purpose.
- **Verdict** — `passed` is true only when compatible, no gate regresses, and
  a relationship improvement exists. The comparator exits non-zero while the
  gate does not pass, so default-enablement cannot ship silently.

**Current honest verdict** (shipped benchmark, deterministic mock embeddings):
the relationship cohort shows real graph contribution (≈0.20 of selected
items entered via expansion) with MRR and nDCG improving and Recall@5/Recall@10
unchanged, but overall Recall@5/MRR regress at the tolerance boundary because
mock embeddings carry no semantic relationship to graph structure. The gate
therefore reports `passed: false` and graph expansion **remains non-default**.
Re-evaluate with real embeddings before proposing default enablement; never
tune expansion parameters against the held-out test split.

## Privacy

Expansion adds no persistence: explain output is per-query terminal text
(escaped for rendering), nothing about the query or the expansion is written
to disk, and no network access participates. Graph reads are read-only and
never mutate index or graph state.
