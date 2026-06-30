# chef-human: Local AI Software Development Tool

## Vision

A fully local AI coding assistant that can understand codebases, write code, run commands, and execute multi-step software engineering tasks -- all running on consumer hardware with open-weight models.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                     CLI / TUI / IDE Plugin              │
├─────────────────────────────────────────────────────────┤
│                    Agent Orchestrator                    │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐  │
│  │  Plan    │ │  Code    │ │  Tool    │ │  Context │  │
│  │  Module  │ │  Gen     │ │  Exec    │ │  Manager │  │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘  │
├──────────────────────┬──────────────────────────────────┤
│   LLM Backend        │   Tool Layer                     │
│  (llama.cpp/ollama)  │  (read, write, grep, bash, etc) │
├──────────────────────┴──────────────────────────────────┤
│                  Sandbox / Workspace                     │
└─────────────────────────────────────────────────────────┘
```

---

## Phase 1: Core Engine (Month 1-2)

### 1.1 LLM Backend Integration

Wrap an open-source inference engine. Recommended stack:

| Component        | Choice                          |
|------------------|---------------------------------|
| Inference Engine | `llama.cpp` (via `llama-cpp-python` or subprocess) |
| Model           | `Qwen2.5-Coder-7B-Instruct` (via Ollama/GGUF) |
| Quantization     | Q4_K_M for 6-8GB VRAM, Q5_K_M for 12GB+ |
| Embeddings       | `bge-small-en-v1.5` for RAG |

**Why Qwen2.5-Coder-7B**: Best-in-class 7B coding model, strong function calling, Apache 2.0 license, fits on consumer hardware.

**Function calling protocol**: ChatML format with tool definitions as JSON schema. The orchestrator parses `<tool_call>` blocks from model output.

### 1.2 Context Manager

Manages conversation state, token budgets, and context window (~32K tokens with sliding window).

- **Token tracker**: Counts tokens per message, enforces limits
- **Window strategy**: Sliding window of recent messages + compressed summaries of older turns
- **File context**: On-demand file reading with LRU cache
- **Repository map**: Structured summary of project structure (tree, key symbols)

### 1.3 Tool Layer

Core tools the agent can invoke:

| Tool            | Description                              |
|-----------------|------------------------------------------|
| `read`          | Read file contents (with line ranges)    |
| `write`         | Write/overwrite file                     |
| `edit`          | Find-and-replace edit (like sed)         |
| `glob`          | Pattern-based file search                |
| `grep`          | Regex content search                     |
| `bash`          | Run shell command (with timeout)         |
| `ls`            | List directory                           |
| `ls_tree`       | Show git-friendly project tree           |
| `ask_user`      | Request user input/clarification         |
| `finish`        | Signal task complete                     |

**Safety**: Each bash tool runs in a sandboxed workspace with:
- Directory jail (can't access files outside workspace without explicit user approval)
- Command blacklist (rm -rf /, etc.)
- Timeout enforcement
- User approval for destructive operations (write, bash, edit)

---

## Phase 2: Agent Loop (Month 3)

### 2.1 ReAct Loop

Implement a structured ReAct (Reasoning + Acting) loop:

```
1. User sends task
2. System generates a plan
3. For each step:
   a. Model reasons about what to do
   b. Model emits tool call
   c. System executes tool, returns result
   d. Model observes result, continues
4. Model emits finish when done
```

**Plan-then-execute** strategy: First generate a high-level plan, then execute each step with tools. Re-plan if subtasks fail or unexpected results occur.

### 2.2 Self-Correction

- If a tool call fails (bad path, syntax error), model automatically retries with corrected input
- Max retry counter per step to prevent infinite loops
- If retries exhausted, escalate to user

### 2.3 Structured Output Parsing

- Parse tool calls from model output using regex on `<tool_name>...</tool_name>` tags
- Validate JSON arguments before execution
- Fallback: If model outputs code blocks without tool calls, extract and apply as edits

---

## Phase 3: Code Understanding (Month 4)

### 3.1 Repository Indexing

- Language-agnostic symbol extraction (Tree-sitter)
- Build a lightweight index of:
  - Functions, classes, imports
  - File-level dependencies
  - Key type definitions
- On-demand retrieval: when model references a symbol, fetch its definition

### 3.2 RAG for Large Codebases

- Chunk codebase into ~512-token chunks
- Embed with `bge-small-en-v1.5`
- Store in FAISS vector index
- Retrieve top-k chunks relevant to current task and inject into context
- Re-index on file changes

### 3.3 Diff-Aware Editing

- Instead of re-writing entire files, model outputs unified diffs
- Apply with `patch` or custom diff parser
- Reduces token usage and preserves file history

---

## Phase 4: Advanced Features (Month 5+)

### 4.1 Multi-File Refactoring

- Model can propose refactoring plans across multiple files
- Each file change is a separate tool call
- Final verification step: run linter/tests

### 4.2 Test Generation & Execution

- Model writes tests based on existing patterns
- Runs test suite, observes failures, iterates
- Supports pytest, vitest, cargo test, etc.

### 4.3 Interactive Debugging

- Model reads stack traces, browses source, proposes fixes
- Can re-run failing commands to verify fixes

### 4.4 Git Integration

- Create commits with meaningful messages
- Create PR descriptions summarizing changes
- Review diffs and suggest improvements

---

## Phase 5: UI & Packaging (Month 6)

### 5.1 CLI

- Interactive REPL-style interface
- Non-interactive mode for CI (`chef-human "fix this bug"`)
- Streaming output of model reasoning
- Colorized diff output

### 5.2 TUI (optional)

- Terminal UI with split panes (chat, file tree, diff view)
- Built with Textual or Bubble Tea (Go)

### 5.3 IDE Extension (optional)

- VS Code extension using Language Server Protocol
- Inline suggestions, chat panel, file editing

### 5.4 Packaging

- Single binary distribution with embedded models (via Ollama/llama.cpp bundling)
- Docker image for server mode
- pip/npm installer

---

## Model Selection Guide

| Model                     | Size  | RAM/VRAM | Quality | License     |
|---------------------------|-------|----------|---------|-------------|
| Qwen2.5-Coder-7B-Instruct | 7B    | 6-8 GB   | ★★★★☆  | Apache 2.0  |
| DeepSeek-Coder-V2-Lite    | 16B   | 10-12 GB | ★★★★☆  | MIT         |
| CodeLlama-13B-Instruct    | 13B   | 8-10 GB  | ★★★☆☆  | Llama 2     |
| Qwen2.5-Coder-14B         | 14B   | 10-12 GB | ★★★★★  | Apache 2.0  |
| DeepSeek-Coder-33B        | 33B   | 20-24 GB | ★★★★★  | MIT         |
| Mixtral-8x7B              | 47B   | 24-32 GB | ★★★★☆  | Apache 2.0  |
| StarCoder2-15B            | 15B   | 10-12 GB | ★★★☆☆  | StarCode    |
| Qwen2.5-Coder-32B         | 32B   | 20-24 GB | ★★★★★  | Apache 2.0  |

Start with Qwen2.5-Coder-7B (Ollama + Q4_K_M). Scale up as hardware allows.

---

## Technology Stack

| Layer        | Technology                                 |
|--------------|--------------------------------------------|
| Language     | Python 3.12+                               |
| LLM Backend  | Ollama (Python SDK) / llama-cpp-python     |
| Embeddings   | sentence-transformers (BGE)                |
| Vector Store | FAISS (in-memory)                          |
| Code Parsing | Tree-sitter                                |
| CLI          | click + rich                               |
| Config       | pydantic-settings + TOML                   |
| Sandboxing   | subprocess + resource limits               |
| Testing      | pytest                                     |
| Linting      | ruff + pyright                             |

---

## File Structure

```
chef-human/
├── pyproject.toml
├── README.md
├── LICENSE
├── config.toml              # Model path, workspace, behavior settings
├── chef_human/
│   ├── __init__.py
│   ├── main.py              # Entry point, CLI dispatch
│   ├── agent/
│   │   ├── __init__.py
│   │   ├── loop.py          # ReAct loop
│   │   ├── planner.py       # High-level plan generation
│   │   └── context.py       # Context window manager
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── backend.py       # Abstract LLM interface
│   │   ├── ollama.py        # Ollama backend
│   │   └── llamacpp.py      # llama.cpp backend
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── registry.py      # Tool registration & dispatch
│   │   ├── filesystem.py    # read, write, edit, grep, glob, ls
│   │   ├── shell.py         # bash execution (sandboxed)
│   │   └── user.py          # ask_user, finish
│   ├── codebase/
│   │   ├── __init__.py
│   │   ├── indexer.py       # Symbol extraction & indexing
│   │   ├── parser.py        # Tree-sitter wrappers
│   │   └── retriever.py     # RAG retrieval
│   └── ui/
│       ├── __init__.py
│       └── cli.py           # Rich-based CLI
└── tests/
    ├── test_agent/
    ├── test_tools/
    └── test_codebase/
```

---

## First Steps (MVP)

1. **Set up Python project** with pyproject.toml, ruff, pytest
2. **Implement LLM backend** -- Ollama wrapper with ChatML format
3. **Build 4 core tools**: `read`, `write`, `bash`, `glob`
4. **Write minimal ReAct loop**: prompt → tool call → execute → observe → repeat
5. **Test end-to-end**: "Create a Python script that prints Fibonacci numbers"

---

## Constraints & Risks

| Risk                     | Mitigation                              |
|--------------------------|-----------------------------------------|
| Context window too small | Sliding window + summarization          |
| Model hallucinates paths | Validate paths before tool execution    |
| Infinite loops           | Max steps (default 25), interruptible   |
| Destructive commands     | Sandbox + user approval for write/bash  |
| Slow on CPU              | Support GPU offload, quantized models   |
| Poor code quality        | Self-review loop, lint-then-edit        |
