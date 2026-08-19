# Architecture

This is a reader-facing map of how chef-human is put together: component boundaries, the request
sequence for a single task, and where to extend it. For environment/testing mechanics, see
[DEVELOPMENT.md](DEVELOPMENT.md); for what the safety guardrails do and don't cover, see
[SAFETY.md](SAFETY.md).

## Component map

```
                         ┌─────────────────────────┐
 CLI (main.py)  ───────► │      create_agent()      │
 REPL / TUI              │  create_context_assembler │
                         └────────────┬─────────────┘
                                      │ wires together
        ┌─────────────┬──────────────┼──────────────┬─────────────┐
        ▼             ▼              ▼               ▼             ▼
   WorkspaceManager  Tokenizer  ContextAssembler  ToolRegistry  LLMBackend
                                      │                              │
                                      │      symbol index / RAG      │
                                      ▼                              ▼
                                 ReActLoop  ◄───────────────────  Planner
                                      │
                    parse tool calls, dispatch, verify step,
                    retry / replan / escalate  (retry.py)
                                      │
                                      ▼
                                 ReActUI (protocol.py)
                    DebugTUI / StreamingUI / ReplUI / Textual TUI / NoopUI
```

- **`chef_human/main.py`** — the Click CLI (`run`, `repl`, `tui`, `session ...`, `show-config`,
  `doctor`, `recommend-model`). CLI flags like `--model`/`--temperature`/`--config` are applied by
  monkeypatching the module-level `chef_human.config.settings` singleton for the duration of agent
  construction — settings are not assumed stable across the process lifetime.
- **`chef_human/agent/__init__.py`** — `create_agent()` / `create_context_assembler()` wire the
  workspace, tokenizer, context assembler, tool registry, LLM backend, and planner into a
  `ReActLoop`. `create_context_assembler()` branches on repo size
  (`len(files) > settings.max_index_files`): small repos get a tree-sitter `SymbolIndex` +
  `DependencyGraph` (exact-match code intelligence); large repos get a `RAGRetriever` (embeddings +
  FAISS) instead, and **do not** get `symbol_index`/`dep_graph` populated — `create_tool_registry()`
  silently skips registering the symbol-aware tools (`lookup_symbol`, `refactor_symbol`,
  `reference_finder`, `goto_definition`) in that case.
- **`chef_human/agent/react_loop.py`** — the core loop (see below).
- **`chef_human/agent/planner.py`** — produces the initial `Plan` (ordered `PlanStep`s) and, later,
  `verify_step` — the LLM fallback used when a step's completion can't be checked deterministically.
- **`chef_human/agent/retry.py`** — `RetryManager` decides, from a step's iteration history, whether
  to retry, replan, or escalate/fail, bounded by `max_retries_per_step` and `max_replans`.
- **`chef_human/tools/`** — plain classes (`name`, `description`, JSON-schema `parameters`, async
  `run()`) registered into a `ToolRegistry` by `create_tool_registry()`. Each tool declares a
  `ToolPolicy` (`mutation_scope`: none/path/workspace/history; `read_requirement`) that the loop's
  deterministic step-verification guards use to decide what evidence a tool call can supply.
- **`chef_human/llm/`** — `create_backend()` picks `OllamaBackend` or `LlamaCppBackend` based on
  `settings.llm_backend`; both implement a shared `LLMBackend` protocol (`complete`,
  `complete_stream`, `embed`, `count_tokens`).
- **`chef_human/ui/`** — `ReActUI` is a protocol the loop calls into for all presentation
  (`DebugTUI`, `StreamingUI`, `ReplUI`, `ChefHumanTUI` (Textual), `NoopUI` for headless runs) — the
  core loop has no direct knowledge of terminal rendering.

## Request sequence: one `ReActLoop.run()` call

1. **Assemble context.** `ContextAssembler.assemble()` combines the system prompt, token-budget-
   aware conversation history, a `RepoMap` (directory-tree-like summary), relevant open-file
   contents, and — depending on which path was built — symbol lookups or RAG-retrieved chunks, all
   trimmed to `settings.max_context_tokens`.
2. **Call the LLM.** Through the active `LLMBackend`.
3. **Parse tool calls** (`agent/parser.py`). `OllamaBackend` first checks Ollama's native
   `response.message.tool_calls` field (populated for models whose template supports structured
   tool-calling, e.g. Qwen3.x); if empty, it falls back to scraping ChatML-style
   `<tool_call>{...}</tool_call>` tags out of raw content. `LlamaCppBackend` has no native path and
   always expects the tag form.
4. **Dispatch.** Non-`finish` tool calls run one at a time, in model order, each wrapped in
   `asyncio.wait_for(..., timeout=config.tool_timeout)`; `finish` runs serially/terminally.
   Dispatch within one response batch is deliberately *not* `asyncio.gather` — this is specifically
   so two mutating calls in the same turn can't race on the same path. The tool classes themselves
   have no per-file locking of their own, so this guarantee only holds as long as every caller goes
   through `ReActLoop`.
5. **Post-write lint (optional).** If `lint_after_write` is set, `run_lint` lints the touched file;
   a lint error triggers `_rollback_file`, restoring content captured before dispatch
   (`_capture_file_content`).
6. **Verify the step** (`_verify_and_mark_step`). Deterministic guards run first — file-creation,
   mutation, and execution-step checks against real disk state and tool exit codes — falling back to
   the LLM verifier (`Planner.verify_step`) only for what isn't objectively checkable.
7. **Retry / replan / escalate.** Results feed back into `RetryManager`, which decides whether to
   retry the current step, trigger a replan (`Planner`), or escalate to failure after
   `max_retries_per_step` consecutive failures.
8. Results feed back into context, and the loop continues until `finish` or an exit condition.

## Design decisions worth calling out

**Why tagged tool calls, not just native structured calling.** Not every local model's chat
template supports native tool-calling reliably, and even models that do sometimes emit malformed
JSON under pressure from small parameter counts. Falling back to a scraped `<tool_call>` tag format
keeps the agent usable across a wider range of open-weight models than requiring native support
would, at the cost of a second parser path to maintain.

**Why context selection differs for small vs. large repos.** Tree-sitter symbol extraction gives
exact-match navigation (find this function, rename this symbol) that's strictly better than semantic
retrieval when it's affordable, but its `SymbolIndex` is built synchronously at startup and doesn't
scale to large trees. RAG's chunk-embed-retrieve pipeline scales further but only approximates
relevance and currently has no incremental update path (`RAGRetriever.build()` always clears and
rebuilds the whole store) — hence RAG stays labeled experimental and symbol-aware tools are simply
absent rather than degraded on the large-repo path.

**How retries and replanning are bounded.** `RetryManager` caps both dimensions independently
(`max_retries_per_step`, `max_replans`) so a stuck step can't loop forever and a pathologically
hard task can't replan indefinitely either — it escalates to a failed result instead of hanging.
Two real false-escalation bugs were found this way (a step phrasing that skipped valid `bash`
evidence, and an ambiguous no-op-edit message the LLM verifier misread as "not fixed") — both are
documented in `CLAUDE.md`'s architecture notes with root cause and fix. A known, not-yet-fixed
follow-up: per-step evidence is keyed by exact step-description text
(`_step_evidence_key`), so a replan that rewords a step orphans evidence accumulated under the old
wording even when the step's real-world goal was already met — this can still surface as a false
escalation on long, multiply-replanned tasks.

**How the UI stays independent of the core.** `ReActUI` is a protocol, not a base class the loop
depends on concretely — the loop calls generic methods (render a plan, show a tool call, request
approval) and any of the five implementations can be swapped in without touching `react_loop.py`.
This is what let the Textual TUI and headless/JSON output ship as alternate front ends onto the same
loop rather than forks of it.

**What failed / changed during development** (see `CLAUDE.md` for full detail):
- Blocking I/O inside the async Textual TUI caused stalls under load; fixed by routing tool
  execution properly through the event loop rather than the UI thread.
- Undo/redo needed real transaction semantics — `DiffStore.record_transaction()` — once
  multi-file operations (`refactor_symbol`) needed to be reversible as one atomic unit rather than
  file-by-file.
- A prior working-tree state accumulated conflicting backup file generations and untracked
  implementations that diverged from what was committed; recovering a single canonical source of
  truth was a prerequisite phase before any of the above stabilization work could be trusted (see
  `docs/finish-line/00-repository-recovery.md`).

## What would be required for production

This is a prototype, not a production tool — see [SAFETY.md](SAFETY.md) for the full threat model.
Getting to production would require, at minimum:

- **Real sandboxing.** `BashTool`'s guardrails are string-matching tripwires, not isolation; an
  indirect invocation (`python3 -c "..."`) bypasses them entirely. Production use needs actual
  process/filesystem/network isolation (container, VM, or OS-level sandbox), not a blacklist.
- **Stronger evals.** The benchmark suite (`chef_human/benchmark.py`) covers seven hand-verified
  cases against one or two models — useful as a regression signal, not a statistically meaningful
  claim about reliability across tasks or models.
- **Concurrency isolation.** Serialized per-turn dispatch prevents same-turn races but assumes a
  single `ReActLoop` instance per workspace; nothing prevents two separate processes/sessions from
  operating on the same workspace concurrently.
- **Secrets handling.** No redaction of credentials from prompts, tool output, logs, or saved
  sessions.
- **Telemetry/privacy decisions.** None currently collected — deliberate for a local-first tool, but
  a production deployment would need an explicit policy either way.
- **Cross-platform validation.** Only Linux is supported/tested.

## Extension points

- **New tool:** implement the tool-class shape (`name`, `description`, `parameters`, async `run()`,
  a `ToolPolicy`) in `chef_human/tools/`, register it in `create_tool_registry()`
  (`chef_human/tools/__init__.py`).
- **New LLM backend:** implement the `LLMBackend` protocol (`chef_human/llm/backend.py`) and wire it
  into `create_backend()` (`chef_human/llm/__init__.py`).
- **New UI:** implement `ReActUI` (`chef_human/ui/protocol.py`); nothing in `react_loop.py` needs to
  change.
- **New language for symbol indexing:** requires updating four tables in lockstep —
  `grammars.py`'s package map, `extractor.py`'s `_LANG_MAP`/`_TS_QUERIES`/`_LANG_PATTERNS`, and
  `dependencies.py`'s `_IMPORT_PATTERNS`. Missing one degrades that language silently (extraction
  returns nothing, no error) — see `CLAUDE.md` for the current gaps (ruby/c/cpp lack `_TS_QUERIES`).
