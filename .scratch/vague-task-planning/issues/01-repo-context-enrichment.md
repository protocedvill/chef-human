Type: task
Status: closed
Blocked by: none

## Question

Build deterministic repo-context enrichment: replace `ReActLoop._get_repo_context()`'s bare
`self._context._repo_map.generate_tree()[:1000]` with a richer, still-cheap-and-deterministic
`repo_context` string, composed of:

1. A language/file-type breakdown (extension counts across the workspace) — cheap, deterministic, tells
   the planning LLM up front "this is a C firmware/host project" instead of it discovering that fact via
   several `read`/`glob` tool calls during execution.
2. A short excerpt (first N lines, or a summary) of top-level doc files (README and similar) —
   unconditional, unlike today's `_get_referenced_document_contents`, which only fires when the task
   *names* a doc. A vague task ("add a web interface") never names one.
3. A top-level directory listing with per-directory purpose hints where cheaply inferable (e.g. via
   `SymbolIndex`'s per-file language tags, since the target repo class takes the symbol path — see the
   map's Notes) — not a full recursive tree dump.

Keep `planning_facts` untouched (existence-check scope only, decided). Cap the whole enriched
`repo_context` to a fixed, low-hundreds-of-tokens budget — it shares `settings.max_context_tokens`
(32768) with conversation history and file context, so it must not grow unbounded on a large repo.

Land this as a real code change with test coverage (unit tests against a fixture workspace with mixed
file types + a README, asserting the composed `repo_context` contains the expected sections and respects
the token cap) — this ticket's job is the code, not yet the live benchmark run (that's ticket 02).

## Answer

Implemented in `chef_human/agent/react_loop.py`'s `_get_repo_context`, which now composes
`_build_repo_context_enrichment()` (extension-count file-type breakdown, unconditional README
excerpt, top-level directory listing with per-directory dominant-extension hints) ahead of the
existing bare `[:1000]`-truncated tree. Enrichment is capped to
`_REPO_CONTEXT_ENRICHMENT_TOKEN_BUDGET` (300 tokens) via a binary-search truncation against the
`RepoMap`'s tokenizer (falls back to a chars-per-token estimate if unreachable), independent of repo
size. Directory language hints use per-extension counts over `WorkspaceManager.list_files()` rather
than `SymbolIndex` per-file language tags, so it works identically on both the symbol path and the
RAG path (RAG-path `ContextAssembler`s don't populate `symbol_index`). `planning_facts` untouched.

Test coverage: `tests/test_agent/test_react_loop.py::TestRepoContextEnrichment` (real `WorkspaceManager`
+ `RepoMap` + `ApproxTokenizer` rooted at a `tmp_path` fixture) — file-type breakdown, unconditional
README excerpt, top-level directory listing with language hints, token-budget enforcement on a 200-dir
synthetic repo, and empty-workspace fallback. Full suite: 1586 passed, 6 pre-existing failures unrelated
to this change (embeddings extra not installed, asyncio-subprocess integration teardown, lint_fix) — all
failing identically on `main` before this change.
