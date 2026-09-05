# ctxai Roadmap

Last reviewed: 2026-07-23

This roadmap summarizes validated capabilities and upcoming product work. [plan.md](plan.md) is the detailed source of truth; [plan2.md](plan2.md) defines requirements for the next intelligence phase.

## Validated foundation

- [x] Persistent, incremental, gitignore-aware syntax and semantic indexing
- [x] Stable manifests, index integrity checks, lifecycle commands, and retrieval baselines
- [x] Project-rooted filesystem, command, semantic-search, and read-only Git tools
- [x] Hybrid grounded retrieval with file-and-line evidence and token budgeting
- [x] Verified one-shot changes with diffs, approvals, focused checks, and audit records
- [x] Durable, redacted interactive sessions and provider/model switching
- [x] Versioned MCP index and query service with real-client protocol tests
- [x] Provider capability contracts and privacy-boundary-aware fallback behavior
- [x] Evidence-backed planning and exact-action approval for risky or complex work
- [x] Local dashboard for index health, grounded search, inspection, and deletion

## Next: intelligence advantage

- [x] Build a persistent symbol and relationship graph for definitions, imports, calls, inheritance, tests, and references (`ctxai graph` over `graph.sqlite3`, Python first — IG-01; see [docs/SYMBOL_GRAPH.md](docs/SYMBOL_GRAPH.md))
- [ ] Expand semantic retrieval with graph relationships while preserving bounded, inspectable evidence
- [x] Add an executable retrieval-evaluation CLI with Recall@K, MRR, latency, and context-efficiency metrics (`ctxai eval retrieval`, RE-01)
- [ ] Add privacy-preserving retrieval traces and dashboard observability
- [ ] Enforce retrieval-quality regression gates in CI
- [ ] Add change-impact analysis using callers, references, tests, and documentation relationships
- [ ] Detect index freshness and support explicit or opt-in automatic updates

The first five items above are specified as vertical slices in [plan2.md](plan2.md).

## Product hardening

- [x] Add a privacy and provider-cost ledger for context leaving the machine (redacted local run transcripts with usage/cost, [docs/RUN_TRANSCRIPTS.md](docs/RUN_TRANSCRIPTS.md))
- [ ] Export reusable evidence-linked context packs
- [ ] Add a repository-wide `ctxai doctor` command
- [ ] Validate clean installations and supported platforms continuously
- [ ] Expand static typing from the validated core boundary to the complete package
- [ ] Maintain zero repository-wide lint and formatting errors

## Deferred until the foundation justifies them

- Multi-agent orchestration
- IDE extensions
- Enterprise collaboration surfaces
- Generic web/search tools unrelated to repository understanding
- Architect/editor mode or cost-saving claims without benchmark evidence

## Release standard

A feature is marked validated only when its complete user outcome has passing acceptance tests. Functional or experimental code is not presented as production-ready, and documentation must distinguish validated behavior from planned work.
