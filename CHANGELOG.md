# Changelog

## 0.1.0 — prototype release

First public-facing release. Chef Human is an experimental, local-first coding agent — a working
prototype for exploring how agent loops, tool use, code retrieval, and human approval fit together
against open-weight local models, not a production replacement for commercial coding assistants.

### Highlights

- Plan/act/observe agent loop (`ReActLoop`) with deterministic + LLM step verification, bounded
  retry and replan, and tool-call timeouts.
- Ollama and llama.cpp backends behind a shared `LLMBackend` protocol, including native structured
  tool-calling for Qwen3.x-family models and a scraped-tag fallback for models without it.
- Workspace-scoped file, search, shell, patch, diff, undo/redo, refactor, and navigation tools with
  reversible, transactional edits.
- Tree-sitter symbol indexing and dependency graphing for smaller repositories; embedding/FAISS RAG
  retrieval (experimental) for larger ones.
- Session persistence (`chef-human session ...`, `--resume`/`--continue`) and four UI surfaces:
  headless/JSON, streaming CLI, REPL, and a split-pane Textual TUI with a live diff preview and
  destructive-command approval modal.
- `chef-human doctor` preflight command and a `chef-human recommend-model` hardware-aware model
  advisor.
- A seven-tier, externally-verified capability benchmark (`chef_human/benchmark.py`) run against
  disposable workspaces, from `smoke` through `marathon`.

### Known limitations

See [docs/SAFETY.md](docs/SAFETY.md) for the full threat model. In short: the shell tool is not a
sandbox, guardrails are string-matching tripwires rather than a policy engine, RAG has no
incremental update path, and Linux is the only validated platform.
