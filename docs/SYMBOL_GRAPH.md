# Symbol Graph (IG-01)

After indexing a repository, ctxai builds an **inspectable symbol graph** next
to the vector store: definitions, structural relationships, and test
associations, every one traceable back to source with repository-relative
`file:start-end` evidence. This document describes what the Python adapter
extracts, what resolves and what stays unresolved, how the graph is stored and
kept consistent with the index manifest, and how to drive it from the CLI.

The graph is **static evidence only**. It is derived from parsing (tree-sitter,
the same parser the chunker uses) and never imports or executes indexed code.
It does not tell you what code does at runtime — dynamic dispatch,
monkey-patching, reflection, and generated code are out of scope by design
(see Non-goals in the delivery plan).

## Data model

- `GraphNode` — stable id, kind, qualified name, display name, language,
  repository-relative file path, start/end line, optional parent id,
  visibility (`public`/`private`), and a source hash (sha256 of the
  definition's own text; modules hash the whole file).
- `GraphEdge` — stable id, kind, source node id, target node id when resolved,
  unresolved target text otherwise, `file:line` evidence, confidence
  (`exact`, `probable`, or `unresolved`), and the resolver version.
- `GraphMetadata` — graph schema version, extractor/resolver versions,
  supported languages, build time, node/edge counts by kind, unresolved edge
  count, and the generation it publishes.

Node kinds: `module`, `class`, `function`, `method`, `interface` (reserved for
future adapters), `test`. Edge kinds: `contains`, `imports`, `calls`,
`inherits`, `references`, `tests`.

**Stable identity.** Node and edge ids are sha256 digests over repository
identity plus canonical source identity — never a database sequence:

- node id = `sha256("node\0<repository_root>\0<repo-relative path>\0<kind>\0<qualified_name>")`
- edge id = `sha256("edge\0<repository_root>\0<kind>\0<source id>\0<target id or text>\0<evidence file>\0<evidence line>")`

Identical input therefore always produces identical ids (and unchanged files
keep their ids across re-indexes). Module names come from paths: `pkg/mod.py`
is `pkg.mod`; `pkg/__init__.py` is `pkg`.

## What the Python adapter resolves

Extraction is deterministic and conservative. A target is connected only when
it is *statically unambiguous* under the resolution ladder below; everything
else stays unresolved rather than being attached to an arbitrary candidate.

| Construct | Resolves when | Confidence | Otherwise |
| --- | --- | --- | --- |
| Modules, classes, functions, methods | Always (syntax-tolerant) | `exact` | A file with syntax errors still yields its module node and whatever definitions parse; the error count is recorded at build time |
| `contains` (parent → child) | Always | `exact` | — |
| `imports` (`import a.b`, `import a.b as c`, `from a.b import x`, `from . import m`, `from .m import x`) | The target module, or the symbol/submodule inside it, exists in the repository | `exact` | Unresolved edge preserving the import target text (e.g. `a.b.x`, `not_a_module`) |
| `inherits` (class bases) | The base resolves through the ladder below | `exact`/`probable` | Unresolved edge with the base expression text |
| `calls` | Same ladder; a dotted chain must fully resolve | `exact`/`probable` | Unresolved edge with the call expression text (dynamic/ambiguous calls are preserved, not guessed) |
| `references` | Same ladder, non-call name usages | `exact`/`probable` | Not recorded (references are only recorded when statically clear) |
| `tests` | A `test` node's resolved call into a non-test symbol | mirrors the call | — |

Test detection is name-based: functions named `test` or `test_*`, and classes
named `Test*`, become `test` nodes. A resolved call made from inside a test
into a non-test symbol additionally creates a `tests` edge (the association
used to answer "which tests exercise this symbol?").

### The resolution ladder

For each usage/import/base expression, tried in order — first hit wins:

1. **Import bindings** of the current module (`from x import y as z`, module
   aliases): `exact`.
2. **`self`/`cls` attributes** inside a class body: the method/attribute is
   looked up on the enclosing class: `exact`.
3. **Lexical scope** — enclosing definition's qualified name: `exact` (this is
   how nested functions and methods resolve).
4. **Module scope** — the file's own top-level definitions: `exact`.
5. **Unique display name** across the whole repository: `probable`. Duplicate
   names (two modules both defining `process`) are ambiguous and stay
   unresolved.
6. **Runtime builtins** (`print`, `len`, `open`, ...) produce **no edge at
   all** when nothing in the repository defines the same name.

Dotted chains (`pkg.service.run`) resolve by walking every component through
the module/symbol index; a partially-resolvable chain does not resolve.
Locally bound names (parameters, assignment targets, loop variables) are never
treated as symbol references — which is exactly why `dog.speak()` (where `dog`
is a local) stays an unresolved call.

**Not resolved by design:** wildcard `import *` targets, calls through call
results (`factory()()`), calls through locals/attributes of unknown objects,
star-args shuffling, decorator parameter semantics, and anything outside the
repository. These appear as unresolved edges (calls/imports/inheritance) or
are omitted (references) — never as guessed links.

## Storage

The graph lives in `graph.sqlite3` inside the canonical index directory
(`.ctxai/indexes/<name>/`), alongside ChromaDB and `manifest.json`:

- `nodes` and `edges` tables with **foreign keys** (`ON DELETE CASCADE`) and
  indexes on qualified/display name, file path, kind, parent, and edge
  endpoints/kind — all queries are parameterized.
- A `graph_meta` table holds schema version (currently `1`), extractor and
  resolver versions (`python/1`), supported languages, build time, generation,
  and per-kind counts.
- **Full publications** (first build, legacy manifest, extractor upgrade,
  corrupt store) build `graph.sqlite3.tmp`, fsync it, and atomically rename it
  over the live file. **Incremental publications** run in a single
  transaction: a failure rolls back and the prior healthy graph stays visible.

## Index integration

Graph generation is a stage of `ctxai index`, after parsing/embedding
validation and **before any storage mutation and before manifest
publication**:

1. unchanged files: skipped — **zero graph mutations**;
2. changed/deleted files: their owned nodes/edges are replaced/removed;
3. *dependent* files — any file owning an edge that points into a changed
   file's nodes — are re-extracted so their relationships are rebuilt against
   the new node set (the dependency closure is computed to a fixed point, so
   it is deterministic and complete);
4. the graph store is written and fsynced, then the manifest is updated with
   `graph_schema_version`, `graph_extractor_version`, `graph_generation`,
   `graph_node_count`, and `graph_edge_count`.

Because the manifest is published last, a failed graph transaction (or an
injected extraction/storage error) leaves the prior healthy graph *and* the
prior manifest visible; nothing partial becomes current.

**Migration/rebuild.** Manifests written before this feature simply lack the
graph fields; they load unchanged (`None`) and `indexes doctor` reports the
graph as *not built* — a diagnostic, not corruption. The next `ctxai index`
run performs a full graph rebuild (generation 1). A future graph schema bump
will be rejected explicitly with a rebuild message rather than silently
misinterpreted.

## Health checking

`ctxai indexes doctor <INDEX>` (backed by `IndexOperations.inspect`) reports
graph schema, counts, and revision match, and exits non-zero on:

- **generation mismatch** — the manifest references a different generation
  than the store (e.g. a crashed run between graph write and manifest publish);
- **unsupported schema** — the manifest or store declares a schema this build
  does not support;
- **corruption** — unreadable SQLite file or failed `integrity_check`;
- **count inconsistency** — node/edge counts disagree with the manifest or
  with the stored rows.

A *missing* graph (legacy index) is reported as `[i] Graph: not built yet` and
does not fail the doctor; the vector store and manifest health rules are
unchanged.

## CLI

All commands accept an optional index name (the configured
`index_name` default is used otherwise) and `--project-path/-p` for the project
scope. Every command supports `--json`, emitting a versioned envelope
(`{"schema_version": 1, ...}`).

```bash
# Index as usual — graph generation happens automatically
ctxai index ./my-repo my-repo

# Counts by kind, unresolved rate, schema/extractor versions, generation
ctxai graph stats my-repo
ctxai graph stats my-repo --json

# Find definitions by qualified/display name substring
ctxai graph symbol calculate
ctxai graph symbol calculate --kind function --language python --limit 50
ctxai graph symbol Calculator --json

# Relationships: callers/callees, parents/children, base classes/subclasses,
# references, and associated tests
ctxai graph neighbors <SYMBOL_ID>                     # both directions, depth 1
ctxai graph neighbors <SYMBOL_ID> --direction in      # who calls/tests me
ctxai graph neighbors <SYMBOL_ID> --edge tests        # associated tests only
ctxai graph neighbors <SYMBOL_ID> --depth 2 --limit 100 --json
```

Bounds (enforced before any work begins): symbol query length ≤ 200
characters, traversal depth ≤ 3, result limit ≤ 500. `SYMBOL_ID` may be the
full 64-character id or a unique prefix of at least 8 characters; ambiguous
prefixes are rejected rather than guessed.

Human output always shows the evidence: nodes as `file:start-end` rows, edges
with `file:line` and confidence, so every result can be traced back to source.

## Multi-language support (IG-02)

JavaScript and TypeScript adapters join the Python adapter behind one
`LanguageAdapter` protocol (`src/ctxai/graph/adapters.py`). The support matrix
below is generated from the same constants the CLI reports with
`ctxai graph capabilities [INDEX] [--json]`; a test asserts the two stay in
sync.

| Language | Adapter | Node kinds | Edge kinds | Extensions |
|---|---|---|---|---|
| python | `python/1` | module, class, function, method, test | contains, imports, calls, inherits, references, tests | `.py` |
| javascript | `javascript/1` | module, class, function, method, test | contains, imports, calls, inherits, references, tests | `.js`, `.jsx`, `.mjs`, `.cjs` |
| typescript | `typescript/1` | module, class, function, method, interface, test | contains, imports, calls, inherits, references, tests | `.ts`, `.tsx`, `.mts`, `.cts` |

JavaScript/TypeScript specifics: ES imports (`import x from`, named, namespace),
CommonJS `require()`, re-exports, functions/classes/methods, TS
`interface`/`type` declarations (node kind `interface`), `extends`/`implements`
(inherits), statically named calls and member calls on local/imported symbols,
`this.` within the class. Dynamic imports/requires, template paths, and
ambiguous names stay **unresolved** — never guessed.

Unsupported languages (e.g. `go`, `rust`) and constructs are reported
explicitly by `ctxai graph capabilities` (and in per-row detail with
`adapter_version: null`); their files remain fully indexable as ordinary
chunks with no fabricated graph nodes. Graph records store each node's
`language` and `adapter_version`; an adapter upgrade marks affected files
stale for bounded incremental rebuild. Graph store schema v2 migrates
forward-only from IG-01's v1 by rebuilding on the next index run.

## Service contract: GraphOperations, MCP, and dashboard (IG-02)

CLI, MCP, and dashboard all read through the `GraphOperations` application
service (versioned DTOs in `src/ctxai/graph/dto.py`); nothing outside the
graph package touches the SQLite store. The three surfaces therefore agree on
identity, counts, confidence, and relationships for the same index (asserted
by e2e tests on all three).

MCP tools (read-only, versioned envelopes, deterministic error codes
`invalid_input` / `not_found` / `storage_failed`):

- `graph_stats(index_name)` — counts by kind, unresolved rate, capabilities,
  health, generation.
- `graph_symbol(index_name, query, kind?, language?, limit?)` — bounded
  definition search with stable ids and `file:start-end` evidence.
- `graph_neighbors(index_name, symbol_id, edge_kind?, direction?, depth?,
  limit?)` — bounded traversal (depth ≤ 3, limit ≤ 500) with per-edge
  evidence and confidence.

Dashboard (loopback-bound by default, all output escaped, endpoints read-only):
`/index/{name}/graph` summary (counts, generation, health),
`/index/{name}/graph/symbols` searchable symbol table, and
`/index/{name}/graph/node/{node_id}` node detail with incoming/outgoing
relationships. Malformed names, traversal attempts, excessive depth/limit,
stale graph generations, and corrupt stores return deterministic errors on
every surface without leaking paths outside the repository.

## Limits and non-goals

- No runtime tracing, no whole-program type inference, no cross-repository
  graphs, no framework-specific dependency injection.
- Node/edge records are local-only, stored in the index directory, and never
  leave the machine (no telemetry, no outbound transport).
