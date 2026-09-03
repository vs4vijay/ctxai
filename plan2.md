# ctxai Intelligence Advantage Plan

> **Superseded (2026-09-03):** The slices in this plan (IG-01..03, RE-01..03) have been merged, with
> full detail and a unified delivery order, into `plan-unified.md` — the single source of truth for
> all remaining work. This file is kept as history; do not start work from here.

Last reviewed: 2026-07-23

This document defines implementation-ready requirements for two connected product capabilities:

1. a persistent code intelligence graph; and
2. retrieval evaluation and observability.

Delivery is organized as vertical slices. Every slice must produce a user-visible outcome across the
CLI, domain model, persistence, integrations, safety, documentation, and tests. A slice is complete only
when its acceptance criteria pass from a clean installation; landing domain-only foundations does not
count as completing a slice.

## Baseline and constraints

The plan extends the validated architecture in `plan.md`, especially VS-01, VS-03, VS-06, and VS-09:

- `CodeChunker` already parses Python, JavaScript, and TypeScript with tree-sitter and stores symbol names
  in chunk metadata where available.
- `IndexManifest` is the atomic, schema-versioned source of truth for index identity and file/chunk state.
- `VectorStore` persists chunks and embeddings in a per-index ChromaDB directory.
- `HybridRetriever` fuses semantic, lexical, symbol, and repository-map rankings with reciprocal-rank
  fusion, while `ContextAssembler` emits bounded `file:start-end` evidence.
- `IndexOperations` provides shared application services to CLI and dashboard adapters.
- `retrieval_eval.py` currently calculates Recall@5 and MRR from pre-populated retrieved locations, and
  `tests/fixtures/retrieval_benchmark.json` contains 20 deterministic questions. It is a test fixture,
  not yet an executable product benchmark.

The implementation must preserve these constraints:

- Local-first: graph construction and evaluation require no network and emit no telemetry by default.
- Deterministic: identical source, parser versions, configuration, and embeddings produce stable node and
  edge identities and reproducible evaluation artifacts.
- Evidence-bearing: graph relationships and evaluation results always retain repository-relative file and
  line locations.
- Incremental: re-indexing an unchanged repository performs no graph or embedding work; changed and deleted
  files update both stores without leaving dangling records.
- Backward-safe: an older index is never silently interpreted as a graph-capable index. It is either
  migrated by an explicit supported migration or reported as requiring rebuild.
- Interface consistency: CLI, MCP, dashboard, and agent retrieval call shared application services and use
  versioned result models rather than independently implementing graph or metric semantics.

## Capability contracts

### Graph data model

The first graph schema version must use these logical records, whether persisted in SQLite or another
transactional local store:

- `GraphNode`: stable ID, kind, qualified name, display name, language, repository-relative file path,
  start/end line, optional parent ID, visibility when known, and source hash.
- `GraphEdge`: stable ID, kind, source node ID, target node ID when resolved, unresolved target text when
  unresolved, source file/line evidence, confidence (`exact`, `probable`, or `unresolved`), and resolver
  version.
- `GraphMetadata`: graph schema version, extractor/resolver versions, supported languages, build time,
  node/edge counts by kind, unresolved counts, and the index manifest generation or revision it matches.

Initial node kinds are `module`, `class`, `function`, `method`, `interface`, and `test`. Initial edge kinds
are `contains`, `imports`, `calls`, `inherits`, `references`, and `tests`. Every non-`contains` edge must
record evidence. Stable IDs must be derived from repository identity plus canonical source identity, not a
database sequence.

“Resolved” means statically supported by the language adapter; it must not imply runtime certainty.
Ambiguous or dynamic references remain unresolved rather than being connected to an arbitrary candidate.

### Retrieval run and evaluation artifact model

Every observable retrieval execution must be representable as a `RetrievalRun` containing:

- schema version, run/query IDs, timestamp, index and graph identity, repository revision, and configuration;
- raw query or a deterministic redaction/hash when query recording is disabled;
- ordered candidates and selected context with chunk ID, citation, component ranks/scores, graph expansion
  reason/path, final rank, estimated tokens, and truncation/deduplication decisions;
- stage timings for embedding, vector search, lexical/symbol ranking, graph expansion, fusion, and assembly;
- total latency, candidate/selected counts, and errors; and
- provider/network fields sufficient to prove that local retrieval emitted no outbound data.

An `EvaluationArtifact` adds benchmark identity/version, expected evidence, per-case judgments, aggregate
metrics, environment metadata, configuration fingerprint, and comparison with an optional baseline. JSON
is the canonical machine format; terminal and dashboard views are projections of the same schema.

Required aggregate metrics are Recall@1/5/10, MRR, nDCG@10, evidence precision@5, successful-query rate,
p50/p95 latency, mean and p95 selected-context tokens, duplicate-token ratio, and graph contribution rate.
Metrics that cannot be computed must be marked unavailable with a reason, never reported as zero.

## Delivery sequence

### IG-01: Inspectable symbol graph for one repository

**User outcome:** After indexing a supported repository, a user can inspect its definitions and structural
relationships and trace every result back to source.

**Scope**

- **CLI:** Add `ctxai graph stats [INDEX]`, `ctxai graph symbol QUERY [--kind KIND] [--language LANG]`, and
  `ctxai graph neighbors SYMBOL_ID [--edge KIND] [--direction in|out|both] [--depth 1] [--limit N]`.
  Default output is human-readable; `--json` emits a versioned envelope.
- **Domain:** Introduce the graph records and language-adapter protocol. Implement deterministic extraction
  for Python first: modules, classes, functions/methods, containment, imports, inheritance, direct calls,
  references, and test definitions/relationships where statically resolvable. Preserve unresolved edges.
- **Storage:** Persist `graph.sqlite3` (preferred) inside the canonical index directory with foreign keys,
  indexes on qualified/display name and edge endpoints/kind, and transactional publication. Add graph
  identity and health fields to the index manifest through a schema migration or an explicit rebuild path.
- **Integration:** Graph generation is a stage of the existing index workflow, after parsing and before
  manifest publication. `IndexOperations.inspect` and `indexes doctor` report graph schema, counts,
  revision match, corruption, and missing graph data.
- **Safety:** Resolve only repository-relative canonical paths. Never import or execute indexed code.
  Parameterize storage queries. Bound symbol query length, traversal depth, and result count. A failed graph
  transaction must leave the prior healthy graph and manifest visible.
- **Docs:** Document supported Python constructs, edge confidence, unresolved edges, rebuild/migration
  behavior, CLI examples, and the distinction between static evidence and runtime behavior.
- **Tests:** Unit fixtures cover aliases, relative imports, nested definitions, methods, inheritance, direct
  calls, ambiguous names, dynamic calls, syntax errors, tests, duplicate symbol names, and stable IDs.
  Process-restart and corrupt-store acceptance tests exercise the CLI and doctor workflow.

**Acceptance criteria**

1. Indexing the Python fixture publishes matching vector and graph generations atomically.
2. A fresh process can locate a named definition and list its imports, callers/callees, parents/children,
   subclasses/base classes, references, and associated tests where the fixture makes them statically clear.
3. Every returned node and edge includes repository-relative `file:start-end` evidence and confidence.
4. Re-indexing unchanged files makes zero graph mutations; changing or deleting one file replaces only its
   owned nodes/edges and removes dangling relationships deterministically.
5. An injected extraction or storage failure cannot produce a healthy/current manifest.
6. `indexes doctor` detects graph/vector revision mismatch, unsupported schema, corruption, and count
   inconsistency and exits non-zero.

**Dependencies:** Validated incremental index and manifest behavior (VS-01), safe path handling (VS-02),
and shared index operations (VS-09). SQLite is preferred because Python ships its client and transactions
are local; choosing another engine requires an ADR demonstrating equivalent clean-install behavior.

**Metrics:** extraction duration; nodes/edges per kind; exact/probable/unresolved edge rate; incremental
files reparsed; graph store size; doctor failure count. Establish baselines but set no quality gate until
IG-03 provides a labeled benchmark.

**Non-goals:** runtime tracing; whole-program type inference; cross-repository graphs; framework-specific
dependency injection; graph visualization; automatic code changes; perfect resolution of reflection,
monkey-patching, generated code, or dynamic imports.

### IG-02: Multi-language graph and stable service contract

**User outcome:** Python, JavaScript, and TypeScript users receive the same graph commands and predictable
capability reporting, while MCP and dashboard clients can consume the same results.

**Scope**

- **CLI:** Extend IG-01 commands with `ctxai graph capabilities [INDEX]`; diagnostics explicitly list
  language support and unsupported edge kinds rather than returning incomplete data without warning.
- **Domain:** Add JavaScript and TypeScript adapters for ES/CommonJS imports, exports, functions, classes,
  methods, interfaces, inheritance/implementation, and statically named calls/references. Define a common
  `GraphOperations` application service and versioned query/result DTOs.
- **Storage:** Keep one graph generation per index. Store language and adapter version on records so an
  adapter upgrade marks only affected files stale and can trigger bounded incremental rebuild.
- **Integration:** Add versioned MCP tools for graph stats, symbol lookup, and neighbors. Add dashboard index
  graph summary, searchable symbol table, and accessible node detail with incoming/outgoing relationships.
  All adapters call `GraphOperations`; none may query graph storage directly.
- **Safety:** Apply MCP index-name validation, dashboard routing protections, depth/result bounds, escaped
  output, and the dashboard's loopback/explicit-remote policy. Graph endpoints are read-only.
- **Docs:** Publish a generated support matrix by language, construct, and edge kind. Document JSON/MCP
  schemas, deterministic error codes, limits, and examples for CLI, MCP, and dashboard.
- **Tests:** Shared contract tests run against every adapter and interface. Protocol tests invoke the real
  MCP transport; ASGI tests cover browser flows without opening a network socket. Fixtures include mixed
  JS/TS imports, re-exports, overloads/interfaces, aliases, and unresolved dynamic imports.

**Acceptance criteria**

1. Equivalent Python/JavaScript/TypeScript fixtures expose consistent node/edge semantics and evidence.
2. CLI JSON, MCP, and dashboard projections agree on identity, counts, confidence, and relationships for
   the same index and query.
3. Unsupported languages/constructs return explicit capability information and retain indexability as
   ordinary chunks; they do not fabricate edges or break indexing.
4. Malformed names, traversal attempts, excessive depth/limit, stale graph generation, and corrupt storage
   return deterministic errors without leaking paths outside the repository.
5. Adapter and schema compatibility tests pass from the built wheel with documented optional dependencies.

**Dependencies:** IG-01 and the validated MCP/dashboard application boundaries from VS-06/VS-09.

**Metrics:** supported-language file coverage; extraction failures by adapter; unresolved edges by language;
service latency p50/p95; interface contract failures (target zero).

**Non-goals:** editors/IDE extensions; remote graph service; cross-language call resolution beyond explicit
imports/exports; call-site control-flow analysis; user-authored graph mutation; graph rendering beyond
accessible tables and relationship lists.

### IG-03: Graph-expanded grounded retrieval

**User outcome:** A repository question returns coherent implementation context that includes relevant
definitions, callers/callees, imports, and tests when they add value, with an explanation of why each item
was selected.

**Scope**

- **CLI:** Add graph-aware behavior to the shared retrieval path and expose `ctxai query --explain` plus
  `--graph/--no-graph`. Explain output shows base ranks, graph paths, fusion contribution, exclusions, and
  context-budget decisions without changing normal concise output.
- **Domain:** Refactor `HybridRetriever` into independently measurable candidate generators and one
  deterministic fusion policy. Seed graph expansion from top base hits, use an allowlisted edge policy,
  decay score by depth/confidence, deduplicate by chunk/source identity, and assemble only within the
  configured token budget. Default traversal depth is one; depth two requires an explicit bounded option.
- **Storage:** Retrieval reads a graph only when its generation matches the vector/index manifest. No query
  mutates index or graph state. Configuration records graph enablement, edge weights, seed count, expansion
  cap, depth, and token budget with validated bounds.
- **Integration:** Agent semantic-search, one-shot coding, CLI query, MCP query, and dashboard query use the
  same retrieval service. Existing versioned response schemas gain optional explanation fields in a
  backward-compatible revision.
- **Safety:** Exclude graph evidence outside the repository; cap seeds, neighbors, depth, candidates, source
  preview, and tokens. If graph data is absent/stale/corrupt, fail explicitly when `--graph` is required and
  otherwise fall back with a visible diagnostic. Query logging follows the privacy controls in RE-02.
- **Docs:** Explain expansion policy, configuration, confidence, fallback semantics, token tradeoffs, and how
  to interpret `--explain` output.
- **Tests:** Deterministic fixtures prove useful one-hop expansion, cycle handling, confidence decay,
  deduplication, budget enforcement, stable ties, stale-graph behavior, and consistent results across
  interfaces. Add adversarial high-degree and cyclic graphs.

**Acceptance criteria**

1. A query seeded on an implementation symbol includes its directly relevant test or caller in the fixture
   when the configured edge policy permits it and explains the exact path.
2. Every selected item has a base or graph reason; graph-expanded items cite both source and relationship
   evidence.
3. Repeated runs against an unchanged index/configuration produce the same ordering and selected evidence.
4. Context never exceeds the configured approximate token budget, cycles never duplicate evidence, and
   high-degree nodes respect candidate caps.
5. On the versioned benchmark created in RE-01, graph-enabled retrieval must not regress Recall@5 or MRR
   beyond the declared tolerance and must improve at least one pre-registered relationship-oriented metric
   or case cohort before becoming the default.

**Dependencies:** IG-02 for stable graph services and RE-01 for honest quality gates. The implementation may
be developed behind a disabled feature flag before RE-01 is complete, but it cannot become default first.

**Metrics:** graph contribution rate; useful graph expansion precision; Recall@5/MRR/nDCG delta versus
graph-disabled retrieval; token delta; duplicate-token ratio; stage p50/p95 latency; fallback/error rate.

**Non-goals:** using graph proximity as proof of relevance; unlimited multi-hop traversal; replacing
semantic/lexical retrieval; model-based reranking in the default offline benchmark; change-impact analysis;
automatic tuning against the test set.

### RE-01: Executable, versioned retrieval benchmark

**User outcome:** A maintainer can run one command against a real local index and receive reproducible
quality, latency, and context-efficiency results, with a non-zero exit code when declared gates regress.

**Scope**

- **CLI:** Add `ctxai eval retrieval BENCHMARK --index INDEX [--output PATH] [--baseline PATH]
  [--fail-on-regression] [--repeat N] [--json]`. Add `ctxai eval retrieval validate BENCHMARK` for schema,
  duplicate ID, path, evidence-range, split, and expectation validation without running retrieval.
- **Domain:** Replace pre-populated retrieved locations with a versioned benchmark schema whose cases include
  stable ID, natural-language query, tags/cohort, expected files/symbols and optional line ranges, relevance
  grades, and train/dev/test split. The runner invokes the production retrieval and context-assembly service.
  Implement all required metrics, deterministic aggregation, bootstrap confidence intervals where useful,
  warm-up/repeat handling, and explicit unavailable metrics.
- **Storage:** Persist immutable JSON artifacts under `.ctxai/evaluations/retrieval/` by default, using atomic
  writes and content-derived benchmark/configuration fingerprints. User-selected output paths must pass the
  same project-boundary policy as other writes. Never modify the benchmark during execution.
- **Integration:** Reuse repository index discovery, embedding identity checks, `HybridRetriever`,
  `ContextAssembler`, and graph capability detection. Support deterministic mock embeddings in acceptance
  tests and real configured embeddings for local maintainer runs.
- **Safety:** No LLM or network call is allowed by the benchmark runner unless a future evaluator is
  explicitly selected and approved. Validate benchmark paths as repository-relative by default. Bound file
  size, case count, repeats, results, and concurrency. Artifacts redact secrets and absolute home paths.
- **Docs:** Provide benchmark authoring guidance, relevance grading, split discipline, metric definitions,
  comparison semantics, reproducibility caveats, and examples for adding a regression case without tuning
  on the test split.
- **Tests:** Unit tests validate metric math (including ties, empty cases, graded relevance, partial
  expectations, unavailable data), schema errors, fingerprints, redaction, and comparison thresholds.
  End-to-end tests build a fixture index, execute the command, parse the artifact, compare a baseline, and
  assert exit codes without network access.

**Acceptance criteria**

1. The existing 20 questions are migrated to the versioned schema with explicit IDs, tags, relevance, and
   splits; retrieved results are produced at runtime, not embedded in fixture data.
2. One clean-install command builds/uses the fixture index and reports all required available metrics plus
   per-case ranks, citations, timing, selected tokens, and configuration identity.
3. Repeated deterministic runs produce byte-stable semantic content apart from documented timestamps and
   measured durations; artifact comparison ignores only those volatile fields.
4. `--fail-on-regression` uses checked-in absolute and relative tolerances per metric/cohort, reports each
   failing gate, and exits non-zero. Missing/incompatible baselines fail clearly rather than silently pass.
5. Invalid expectations, missing evidence, unhealthy/stale indexes, embedding mismatch, empty cohorts, and
   partial runs are represented explicitly and cannot be mistaken for a successful benchmark.

**Dependencies:** Validated index/query path (VS-01/VS-03). It does not depend on the graph; graph identity
and metrics are optional fields until IG-03.

**Metrics:** benchmark case/cohort coverage; successful-query rate; Recall@1/5/10; MRR; nDCG@10; evidence
precision@5; p50/p95 latency; selected token mean/p95; duplicate-token ratio. Initial gates are established
from three clean deterministic runs and reviewed before enforcement.

**Non-goals:** claiming universal retrieval quality from one repository; live-provider or LLM-answer
evaluation; subjective judge-model scoring; benchmark auto-generation; hidden telemetry; tuning against
held-out test cases; making noisy wall-clock latency a hard cross-platform CI gate.

### RE-02: Privacy-preserving retrieval observability

**User outcome:** A user can inspect why a particular search selected its context and diagnose slow, noisy,
or graph-heavy retrieval locally without exposing source or queries.

**Scope**

- **CLI:** Add `ctxai retrieval runs list`, `ctxai retrieval runs show RUN_ID [--json]`, and
  `ctxai retrieval runs delete [RUN_ID|--all]`. Add `--trace` to query/evaluation commands. Normal queries
  produce in-memory metrics only unless local persistence is explicitly enabled in configuration or by flag.
- **Domain:** Instrument the production retrieval stages with a clock/recorder abstraction and the
  `RetrievalRun` schema. Record candidate provenance, component ranks, graph paths, final selection,
  deduplication/truncation, timing, and errors. Add configuration for `off|metrics|full` recording, query
  text `omit|hash|store`, source preview `omit|store`, retention count/days, and local artifact directory.
- **Storage:** Store local JSON Lines or SQLite traces atomically with a version and retention policy. Default
  is `off` for persistence; `metrics` stores no raw query, source, embeddings, credentials, or absolute home
  paths. Deletion is deterministic and scoped to the configured trace directory.
- **Integration:** The CLI, MCP, dashboard, agent, and evaluator emit through the same recorder. MCP responses
  return a run ID only when tracing is enabled. Dashboard adds a local retrieval-runs view with filters for
  index, status, time, and cohort and a detail view of the ranking funnel and timings.
- **Safety:** No automatic upload, telemetry SDK, or remote exporter. Redact recursively by secret-bearing
  field name and common credential formats before persistence. Apply project/index path normalization,
  bounded preview sizes, retention, and explicit confirmation for bulk deletion. Dashboard protections from
  VS-09 remain in force.
- **Docs:** State exact defaults and recorded fields for every mode; explain opt-in, storage location,
  retention, redaction limits, deletion, run IDs, and how to verify no outbound transport exists.
- **Tests:** Fake-clock tests cover stage timing and failures; snapshot/schema tests cover traces; privacy
  tests seed API keys, bearer tokens, URLs with credentials, absolute home paths, source, and raw queries;
  retention/concurrent-write/corruption recovery tests run locally. CLI/MCP/ASGI contract tests compare the
  same run projection.

**Acceptance criteria**

1. A traced query produces a versioned run showing every candidate generator, ordered candidates, graph
   paths if used, final items, exclusions, stage/total timings, token estimate, and index/config identity.
2. Default configuration writes no retrieval trace. `metrics` mode persists neither raw query nor source;
   `full` mode requires explicit opt-in and displays a privacy warning on enablement.
3. Privacy tests find no seeded secret or disallowed absolute path in persisted artifacts, terminal output,
   MCP results, or dashboard HTML.
4. Trace recording failure never changes retrieval ordering or turns a successful query into a failed one;
   the recording failure is surfaced as a diagnostic. Retrieval failures themselves remain observable.
5. Retention and delete commands remove only resolved trace targets, and concurrent writers cannot corrupt
   previously committed runs.

**Dependencies:** RE-01 artifact vocabulary and IG-03 candidate-stage boundaries. Basic metrics-only
instrumentation can land with RE-01; full ranking provenance follows the IG-03 refactor.

**Metrics:** trace overhead in latency and bytes; recorder failures; traces retained/deleted; redaction
failures (target zero); percentage of runs with complete stage timings; no outbound transport count (target
zero unless a separately approved future feature exists).

**Non-goals:** hosted telemetry; user tracking; remote log aggregation; session replay; storing embeddings;
capturing LLM prompts/responses; automatic source upload; observability of unrelated agent/tool execution;
distributed tracing standards unless a local interoperability need is demonstrated.

### RE-03: Retrieval quality dashboard and CI regression gate

**User outcome:** Maintainers can compare benchmark runs, identify regressed cohorts or cases, and prevent a
retrieval change from merging when it violates reviewed quality gates.

**Scope**

- **CLI:** Add `ctxai eval retrieval compare BASELINE CANDIDATE [--json]` with metric/cohort/case deltas and
  compatible/incompatible status. Preserve the runner's gate-based exit codes for CI.
- **Domain:** Implement artifact compatibility checks for schema, benchmark fingerprint, case set/split,
  index/graph schema, embedding identity, and retrieval configuration. Comparison distinguishes quality,
  efficiency, correctness, and noisy timing dimensions and identifies newly passing/failing cases.
- **Storage:** Dashboard reads immutable evaluation artifacts through an `EvaluationOperations` service.
  It never scans arbitrary user paths; configured artifact roots and artifact IDs are validated.
- **Integration:** Add dashboard run list, run summary, aggregate/cohort comparison, worst regressions, and
  per-case ranking evidence. Add a GitHub Actions retrieval job using deterministic local/mock embeddings,
  a checked-in benchmark and baseline, artifact upload on success/failure, and `--fail-on-regression`.
- **Safety:** CI uses no provider credentials or network-dependent embeddings. Dashboard escapes query/source
  content, limits previews, and follows explicit remote binding rules. Uploaded CI artifacts contain fixture
  code only and pass the same redaction checks.
- **Docs:** Add maintainer workflow for refreshing a baseline, required review evidence, interpreting noisy
  latency, downloading CI artifacts, and recovering from schema/benchmark incompatibility. Document that a
  baseline update must not be bundled invisibly with the retrieval algorithm change it excuses.
- **Tests:** Comparison golden tests cover improvements, regressions, incompatible runs, missing metrics,
  cohort drift, and tolerance boundaries. ASGI tests cover lists/comparisons/case details. A workflow linter
  or parsed-YAML test verifies the CI job command, credential-free environment, and artifact publication.

**Acceptance criteria**

1. A dashboard comparison shows aggregate and cohort deltas, confidence/availability, changed cases, ranks,
   citations, graph contribution, latency, and context tokens from the same JSON artifacts as the CLI.
2. CI runs deterministically without secrets or external services and fails for a seeded Recall/MRR/nDCG or
   correctness regression beyond reviewed tolerances.
3. CI does not hard-fail on cross-run p95 latency noise; it records and flags reviewed efficiency thresholds
   separately unless a controlled runner makes the threshold reliable.
4. Incompatible artifacts cannot be compared as if equivalent. The output identifies every incompatible
   identity field and gives a rebuild/rerun action.
5. Updating a checked-in baseline is a deliberate documented command and produces a reviewable artifact
   diff plus benchmark/configuration fingerprints.

**Dependencies:** RE-01 is required. RE-02 supplies richer drill-down data but is not required for the first
aggregate comparison. IG-03 is required before graph contribution becomes a gate.

**Metrics:** CI pass/fail and runtime; number of gated quality regressions; baseline age; benchmark cohort
coverage; incompatible comparison count; dashboard comparison latency.

**Non-goals:** public leaderboard; comparing different repositories as equivalent; automatically accepting
baseline regressions; hard CI dependence on cloud embeddings; LLM answer correctness; production telemetry;
optimizing solely for one aggregate metric.

## Cross-slice engineering requirements

- New public result models and stored artifacts are schema-versioned and have round-trip serialization tests.
- New application services accept dependencies (clock, store, embedding provider, recorder) explicitly so
  tests do not require network, global configuration, or wall-clock timing.
- Configuration has one canonical implementation and rejects invalid depth, result, retention, and token
  limits before work begins.
- User-facing errors use stable categories/codes across CLI JSON, MCP, and dashboard; human text may evolve.
- Database migrations are forward-only, transactional, fixture-tested from every supported prior schema,
  and documented with rebuild fallback. No code silently drops an index, graph, trace, or evaluation.
- All paths persisted in portable artifacts are repository-relative where possible. Machine-specific paths
  belong only in explicitly local diagnostics and are redacted from exported/CI artifacts.
- Every slice passes formatting, lint, static type checking, unit tests, relevant end-to-end tests, build,
  isolated-wheel smoke tests, and `git diff --check` in the supported Python matrix.
- Performance tests use bounded fixtures and declared budgets. A performance budget cannot override quality,
  correctness, privacy, or safety gates.

## Recommended implementation order

1. RE-01, so all retrieval and graph-ranking decisions have an executable baseline.
2. IG-01, delivering useful Python graph inspection and durable graph/index integrity.
3. IG-02, stabilizing adapters and interface contracts for Python, JavaScript, and TypeScript.
4. IG-03, integrating graph evidence behind a measured feature flag and enabling it by default only after
   benchmark evidence meets the acceptance gate.
5. RE-02, completing opt-in local traces on the refactored retrieval stages.
6. RE-03, adding human comparison views and the deterministic CI quality gate.

RE-01 and IG-01 may be developed in parallel because the evaluation artifact permits a graph identity to be
absent. Their schemas should be reviewed together before either is frozen.

## Definition of complete

These capabilities are complete when all six slices are validated and:

- a clean install can index a Python/JavaScript/TypeScript fixture, inspect its graph, and retrieve bounded
  graph-expanded evidence through CLI, MCP, and dashboard with consistent identities;
- graph-disabled and graph-enabled benchmark artifacts can be reproduced and honestly compared;
- reviewed quality gates protect Recall, ranking, evidence precision, correctness, and context efficiency;
- a user can explain a retrieval run locally without opting into persistence or sending data externally;
- incremental indexing keeps vector, graph, and manifest generations consistent across changes, deletions,
  failures, and process restarts; and
- documentation distinguishes exact, probable, unresolved, measured, experimental, and unsupported behavior.
