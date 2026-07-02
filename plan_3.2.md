# Phase 3.2: RAG for Large Codebases

**Goal**: When the codebase exceeds ~500 files, replace the symbol-name-based index with a semantic chunk-embedding pipeline (RAG). The agent can then retrieve relevant code by semantic similarity rather than exact symbol name matching.

**Prerequisites**: Phase 3.1 complete (symbol index, dependency graph, symbol retriever, extraction for 6 languages).

---

## Current State

| Component | Status |
|-----------|--------|
| `EmbeddingsBackend` | Exists — wraps `sentence-transformers` (bge-small-en-v1.5, 384-dim), lazy-loaded |
| `CodeChunker` | **New** — line-based chunking at ~512 tokens, declaration-boundary awareness, configurable overlap |
| `VectorStore` | **New** — FAISS `IndexFlatIP` wrapper with JSON metadata store, `save()`/`load()` for persistence |
| `RAGRetriever` | **New** — orchestrates chunk → embed → index → search; hybrid with optional `SymbolIndex` for symbol lookup |
| `SymbolIndex` | Existing — used as small-codebase path (≤500 files) and optional symbol fallback in RAG |
| `ContextAssembler` | **Enhanced** — accepts optional `rag_retriever`, injects `## Related Code` when available |
| `create_context_assembler()` | **Enhanced** — auto-selects RAG path when workspace > `max_index_files` files |
| `config.py` | **Enhanced** — added `rag_chunk_tokens`, `rag_chunk_overlap`, `rag_max_results`, `rag_index_dir` |
| `pyproject.toml` | **Enhanced** — added `[project.optional-dependencies] rag` with `faiss-cpu`, `numpy` |

---

## Task List

- [x] **3.2.1** Code chunker — split source files into overlapping line-based chunks (~512 tokens), preserving code structure
- [x] **3.2.2** Vector store — FAISS wrapper with metadata store, add/search/save/load
- [x] **3.2.3** RAG retriever — orchestrate chunking + embedding + search + symbol lookup
- [x] **3.2.4** Integration into ContextAssembler & factory — wire RAG path alongside symbol index
- [x] **3.2.5** Persistence — cache FAISS index + metadata to `.chef-human/` for sub-second startup
- [x] **3.2.6** Tests — chunker, store, retriever, integration, persistence

---

## Task 3.2.1: Code Chunker

**File to create:** `chef_human/agent/rag/chunker.py`

Splits source files into overlapping chunks at token-budget boundaries. Uses line-level splitting (not AST) for simplicity, but tries to break at function/class/struct declarations where possible.

### Design

```python
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from chef_human.llm.tokenizer import Tokenizer

logger = logging.getLogger(__name__)

# Regex to detect declaration lines (function, class, etc.)
_DECL_RE = re.compile(
    r"^\s*(?:(?:async|pub|unsafe|public|private|protected|static|abstract|virtual|override"
    r"|export|declare)\s+)*"
    r"(?:def|class|fn|func|function|interface|type|struct|enum|trait|impl|import|use|constructor"
    r"|async\s+function|module|package)\b"
)


@dataclass(frozen=True)
class Chunk:
    file_path: str
    start_line: int     # 1-indexed
    end_line: int       # inclusive
    content: str
    chunk_id: str       # "{file_path}:{start_line}-{end_line}"


class CodeChunker:
    def __init__(
        self,
        tokenizer: Tokenizer,
        target_tokens: int = 512,
        overlap_tokens: int = 64,
    ) -> None:
        self._tokenizer = tokenizer
        self._target = target_tokens
        self._overlap = overlap_tokens

    def chunk_file(self, file_path: str, content: str) -> list[Chunk]:
        """Split a single file into overlapping chunks."""
        lines = content.splitlines(keepends=True)
        if not lines:
            return []

        chunks: list[Chunk] = []
        start = 0
        while start < len(lines):
            end = self._find_end_line(lines, start)
            chunk_content = "".join(lines[start:end])
            chunks.append(Chunk(
                file_path=file_path,
                start_line=start + 1,
                end_line=end,
                content=chunk_content,
                chunk_id=f"{file_path}:{start + 1}-{end}",
            ))
            if end >= len(lines):
                break
            # Advance by (end - start) - overlap_lines for sliding window
            advance = max(1, (end - start) - self._overlap_lines())
            start += advance
        return chunks

    def _find_end_line(self, lines: list[str], start: int) -> int:
        """Find end line that keeps chunks within target token budget."""
        # Walk forward to find a good declaration boundary
        end = start
        token_count = 0
        for i in range(start, len(lines)):
            line_tokens = self._tokenizer.count(lines[i])
            if token_count + line_tokens > self._target and end > start:
                # Try to back up to a declaration boundary
                break
            token_count += line_tokens
            end = i + 1
        return end

    def _overlap_lines(self) -> int:
        """Estimate how many lines to overlap based on average token count."""
        return max(1, self._overlap // 10)  # rough: ~10 tokens/line
```

### Key Behaviours

- Chunks target ~512 tokens (configurable)
- Overlap of ~64 tokens between consecutive chunks to avoid cutting mid-definition
- Chunk boundaries preferentially fall at declaration lines when within budget
- Each chunk knows its file path and line range for provenance
- Empty files produce no chunks
- Single small files may be one chunk or fewer than target tokens

### Acceptance Criteria

- `chunk_file()` splits a large Python file into multiple overlapping chunks
- `chunk_file()` returns one chunk for a file under the token budget
- `chunk_file()` returns no chunks for an empty file
- Chunk boundaries fall at declaration lines when possible
- Chunks include file path, line range, and content
- Overlap region appears in consecutive chunks
- Chunk IDs are deterministic and unique

---

## Task 3.2.2: Vector Store

**File to create:** `chef_human/agent/rag/store.py`

FAISS-based vector store with a sidecar JSON metadata store. Provides add, search, save, and load operations.

### Design

```python
from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

logger = logging.getLogger(__name__)

# Default paths relative to workspace root
_DEFAULT_INDEX_DIR = ".chef-human"

_INDEX_FILENAME = "rag.index"       # FAISS binary index
_META_FILENAME = "rag.meta.json"    # Chunk metadata list


@dataclass(frozen=True)
class SearchResult:
    chunk_id: str
    file_path: str
    start_line: int
    end_line: int
    content: str
    score: float  # cosine similarity


class VectorStore:
    def __init__(
        self,
        dimension: int,
        index_dir: str | Path | None = None,
    ) -> None:
        self._dim = dimension
        self._index_dir = Path(index_dir) if index_dir else None
        self._index = self._create_index()
        self._metadata: list[dict] = []  # position in list = FAISS index position

    # ... FAISS operations ...

    def add(self, embeddings: list[list[float]], metadata: list[dict]) -> None:
        """Add embeddings and their metadata to the store."""
        pass

    def search(
        self, query: list[float], top_k: int = 5
    ) -> list[SearchResult]:
        """Search for nearest neighbours."""
        pass

    def save(self) -> None:
        """Persist index + metadata to disk."""
        pass

    @classmethod
    def load(cls, index_dir: str | Path, dimension: int) -> VectorStore | None:
        """Load from disk. Returns None if no saved index exists."""
        pass

    def clear(self) -> None:
        """Reset index to empty."""
        pass

    def __len__(self) -> int:
        return self._index.ntotal

    def _create_index(self) -> any:
        import faiss
        return faiss.IndexFlatIP(self._dim)  # inner product = cosine on normalized vectors
```

### Key Behaviours

- Uses `faiss.IndexFlatIP` (brute-force inner product) — fine for up to ~50K chunks
- Metadata stored as a JSON list parallel to FAISS index positions
- `save()` writes two files: `.chef-human/rag.index` (FAISS binary) and `.chef-human/rag.meta.json`
- `load()` reconstructs the store from those files
- `search()` returns semantically ranked chunks with scores
- Empty store returns empty results without errors

### Acceptance Criteria

- `add()` appends vectors and metadata in lockstep
- `search()` returns top-k results sorted by score descending
- `save()` and `load()` round-trip correctly (vectors + metadata)
- `load()` returns None when no saved index exists
- `clear()` resets index and metadata
- `__len__()` returns the number of indexed vectors
- Search on empty store returns empty list without error

---

## Task 3.2.3: RAG Retriever

**File to create:** `chef_human/agent/rag/retriever.py`

Orchestrates chunking → embedding → indexing → search. Provides the public API used by ContextAssembler.

### Design

```python
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from chef_human.agent.rag.chunker import Chunk, CodeChunker
    from chef_human.agent.rag.store import VectorStore, SearchResult
    from chef_human.agent.symbols.index import SymbolIndex
    from chef_human.agent.workspace import WorkspaceManager
    from chef_human.llm.embeddings import EmbeddingsBackend

logger = logging.getLogger(__name__)


class RAGRetriever:
    def __init__(
        self,
        chunker: CodeChunker,
        embedder: EmbeddingsBackend,
        store: VectorStore,
        workspace: WorkspaceManager,
        symbol_index: SymbolIndex | None = None,
    ) -> None:
        self._chunker = chunker
        self._embedder = embedder
        self._store = store
        self._workspace = workspace
        self._symbol_index = symbol_index
        self._initial_built: bool = False

    def build(self, files: list[Path]) -> int:
        """Chunk, embed, and index all files."""
        pass

    def refresh(self, files: list[Path]) -> int:
        """Re-index files that have changed."""
        pass

    def retrieve(self, query: str, top_k: int = 5) -> list[Chunk]:
        """Semantic search for relevant chunks."""
        pass

    def retrieve_by_symbol(self, symbol_name: str) -> list[Chunk]:
        """If symbol_index is available, find chunks containing a known symbol."""
        pass

    def format_for_prompt(self, results: list[Chunk], max_tokens: int) -> str:
        """Format retrieved chunks into a prompt-friendly string, respecting token budget."""
        pass

    @property
    def is_built(self) -> bool:
        return self._initial_built

    @property
    def total_chunks(self) -> int:
        return len(self._store)
```

### Retrieval Heuristic

1. **Primary path (semantic search)**: embed the query text, search FAISS, return top-k chunks
2. **Fallback path (symbol lookup)**: if a query contains CamelCase identifiers and `symbol_index` is available, also look up exact symbol locations and include those chunks
3. **Rank fusion**: interleave semantic and exact-match results, deduplicate by chunk ID
4. **Token budgeting**: `format_for_prompt()` truncates to fit `max_tokens`, preferring higher-ranked chunks

### Acceptance Criteria

- `build()` chunks, embeds, and indexes all provided files
- `build()` returns total chunk count
- `retrieve("query")` returns relevant chunks sorted by relevance
- `retrieve("query")` returns empty list on empty store
- `retrieve_by_symbol("ClassName")` returns chunks containing that symbol (when index is available)
- `format_for_prompt()` respects the token budget
- `is_built` is False before `build()`, True after
- `total_chunks` reflects the number of indexed chunks

---

## Task 3.2.4: Integration into ContextAssembler & Factory

**Files to modify:**
- `chef_human/agent/rag/__init__.py` — export public API
- `chef_human/agent/context.py` — add RAG-based symbol context to assembly
- `chef_human/agent/__init__.py` — factory wiring (create RAG retriever, embedder)
- `chef_human/config.py` — add RAG-related settings

### ContextAssembler Changes

```python
class ContextAssembler:
    def __init__(
        self,
        conversation: ContextManager,
        workspace: WorkspaceManager,
        file_context: FileContextManager,
        repo_map: RepoMap,
        symbol_index: SymbolIndex | None = None,
        dep_graph: DependencyGraph | None = None,
        symbol_retriever: SymbolRetriever | None = None,
        rag_retriever: RAGRetriever | None = None,  # NEW
    ) -> None:
        # ... existing fields ...
        self._rag_retriever = rag_retriever

    def assemble(self, ...) -> list[Message]:
        # ... existing assembly (steps 1-4) ...

        # 5. Symbol or RAG context (use RAG when codebase is large)
        if self._rag_retriever and conversation_messages and remaining > 500:
            rag_text = self._build_rag_context(conversation_messages, remaining)
            if rag_text:
                messages.append(
                    Message(role=Role.system, content=f"## Related Code\n\n{rag_text}")
                )
        elif self._symbol_retriever and conversation_messages and remaining > 500:
            symbol_text = self._build_symbol_context(conversation_messages, remaining)
            if symbol_text:
                messages.append(
                    Message(role=Role.system, content=f"## Related Symbols\n\n{symbol_text}")
                )

        return messages

    def _build_rag_context(self, conversation_messages, budget) -> str:
        recent = " ".join(
            m.content for m in conversation_messages[-4:] if m.role != Role.system
        )
        chunks = self._rag_retriever.retrieve(recent, top_k=5)
        return self._rag_retriever.format_for_prompt(chunks, budget)
```

### Factory Update

```python
# chef_human/agent/__init__.py

def create_context_assembler(
    workspace_root: str | None = None,
    index_on_init: bool = True,
) -> ContextAssembler:
    # ... existing setup (tokenizer, workspace, config, conversation, file_ctx, repo_map) ...

    from chef_human.agent.symbols.dependencies import DependencyGraph
    from chef_human.agent.symbols.index import SymbolIndex
    from chef_human.agent.symbols.retriever import SymbolRetriever

    extractor = CompositeExtractor()
    symbol_index = SymbolIndex(workspace=workspace, extractor=extractor)

    # Decide path: RAG for large codebases, symbol index for small
    files = workspace.list_files(max_depth=10)
    use_rag = len(files) > settings.max_index_files

    if use_rag:
        from chef_human.agent.rag.retriever import RAGRetriever
        from chef_human.agent.rag.chunker import CodeChunker
        from chef_human.agent.rag.store import VectorStore
        from chef_human.llm.embeddings import EmbeddingsBackend

        chunker = CodeChunker(
            tokenizer=tokenizer,
            target_tokens=settings.rag_chunk_tokens,
            overlap_tokens=settings.rag_chunk_overlap,
        )
        embedder = EmbeddingsBackend(settings.embed_model)
        store = VectorStore(
            dimension=embedder.dimension,
            index_dir=workspace.root / ".chef-human",
        )
        rag_retriever = RAGRetriever(
            chunker=chunker,
            embedder=embedder,
            store=store,
            workspace=workspace,
            symbol_index=symbol_index,
        )
        dep_graph = None
        symbol_retriever = None

        if index_on_init:
            rag_retriever.build(files[:settings.max_index_files * 2])

        return ContextAssembler(
            conversation=conversation,
            workspace=workspace,
            file_context=file_ctx,
            repo_map=repo_map,
            rag_retriever=rag_retriever,
        )
    else:
        # Existing symbol index path (Phase 3.1)
        dep_graph = DependencyGraph(symbol_index)
        symbol_retriever = SymbolRetriever(index=symbol_index, file_context=file_ctx)

        if index_on_init:
            symbol_index.build(files=files[:settings.max_index_files])
            if symbol_index.total_symbols() > 0:
                dep_graph.build()

        return ContextAssembler(
            conversation=conversation,
            workspace=workspace,
            file_context=file_ctx,
            repo_map=repo_map,
            symbol_index=symbol_index,
            dep_graph=dep_graph,
            symbol_retriever=symbol_retriever,
        )
```

### Config Update

```python
# chef_human/config.py — add to Settings
max_index_files: int = 500      # threshold: switch to RAG beyond this
rag_chunk_tokens: int = 512     # target tokens per chunk
rag_chunk_overlap: int = 64     # token overlap between consecutive chunks
rag_max_results: int = 5        # chunks returned per query
rag_index_dir: str = ".chef-human"  # relative to workspace root
```

### Acceptance Criteria

- `ContextAssembler` accepts optional `rag_retriever`
- When `rag_retriever` is present, `assemble()` injects `## Related Code` instead of `## Related Symbols`
- `create_context_assembler()` chooses RAG path when workspace has > `max_index_files` files
- `create_context_assembler()` uses symbol index path when workspace has ≤ `max_index_files` files
- New config settings are present with sensible defaults
- Empty workspace (0 files) uses the symbol index path (no RAG overhead)
- All existing tests pass unchanged with the new optional parameters

---

## Task 3.2.5: Persistence

**No new file.** Modify `VectorStore` and `RAGRetriever` to support save/load.

### Design

- `VectorStore.save()` writes:
  - `.chef-human/rag.index` — FAISS `faiss.write_index()` binary
  - `.chef-human/rag.meta.json` — list of chunk metadata dicts

- `VectorStore.load(index_dir, dimension)`:
  - Checks for both files on disk
  - Loads FAISS index via `faiss.read_index()`
  - Loads metadata JSON
  - Returns `VectorStore` instance or `None`

- `RAGRetriever` restores `is_built` flag on load (if store has data)

- On `build()`, overwrite existing files atomically:
  1. Write to `.chef-human/rag.index.tmp` + `.chef-human/rag.meta.json.tmp`
  2. Rename to final filenames

### Acceptance Criteria

- `save()` writes two files to the configured index directory
- `load()` restores index + metadata correctly on a fresh instance
- `load()` returns None when files do not exist
- Partial rebuild overwrites previous index completely
- Round-trip (save → fresh load → search) returns same results

---

## Task 3.2.6: Tests

**New test files:**

| Test file | ~Tests | What it covers |
|-----------|--------|----------------|
| `tests/test_rag/test_chunker.py` | 15 | Basic chunking, token budget, overlap, empty files, single-line files, declaration boundaries |
| `tests/test_rag/test_store.py` | 12 | Add, search, save, load, clear, empty store, round-trip |
| `tests/test_rag/test_retriever.py` | 10 | Build, retrieve, retrieve_by_symbol, format, empty store, is_built |
| `tests/test_rag/test_rag_integration.py` | 6 | End-to-end: chunk → embed → index → search; multi-file; persistence round-trip |

**Modified test files:**

| Test file | ~+Tests | What it covers |
|-----------|---------|----------------|
| `tests/test_context_assembly.py` | 3 | RAG context injection, budget limiting, no RAG mode |
| `tests/test_agent_integration.py` | 2 | Factory creates RAG path for large workspaces |
| `tests/test_config.py` (if it exists) | 1 | New config fields have defaults |

**Test data:** Reuse existing `tests/test_symbols/test_data/` files, plus add:
- `tests/test_rag/test_data/large_file.py` — 1000+ line file for chunking tests
- `tests/test_rag/test_data/mixed/` — small multi-file directory for integration tests

**Estimated total new tests**: ~50

---

## Dependencies Map

```
3.2.1 chunker.py ───────────► tokenizer.py (token counting)
3.2.2 store.py ─────────────► faiss, numpy, json (metadata)
3.2.3 retriever.py ─────────► 3.2.1, 3.2.2, embeddings.py, symbol_index (optional)
3.2.4 context.py ───────────► 3.2.3, config.py, agent/__init__.py
3.2.5 persistence ──────────► 3.2.2 (save/load in VectorStore)
3.2.6 tests ────────────────► all of the above, test_data/
```

---

## Implementation Order

1. **3.2.1** Code chunker — must exist before retriever can chunk files
2. **3.2.2** Vector store — must exist before retriever can index/search
3. **3.2.3** RAG retriever — orchestration layer that depends on both
4. **3.2.4** Integration — wire into ContextAssembler, factory, config
5. **3.2.5** Persistence — save/load for VectorStore (can be built alongside 3.2.2)
6. **3.2.6** Tests — all new tests + modifications

---

## Design Decisions (Confirmed)

### 1. Chunking strategy: line-based, not AST-based
AST-based chunking (splitting at syntax tree boundaries) is more precise but requires per-language parsers. Line-based chunking with declaration-boundary detection covers all file types with a single regex.

### 2. FAISS index type: `IndexFlatIP` (brute force)
For codebases with up to ~50K chunks, brute-force inner product is fast enough (sub-100ms on CPU). IVF or HNSW indices add complexity without meaningful benefit at this scale.

### 3. Embedding model: bge-small-en-v1.5 (384-dim)
Already used by `EmbeddingsBackend`. 384 dimensions keep memory low (50K chunks × 384 × 4 bytes = ~77 MB). Sentence-transformer bge family is well-tested for code retrieval tasks.

### 4. RAG trigger: `max_index_files` threshold
When the workspace has more source files than `max_index_files` (default 500), the factory switches to RAG mode. This keeps small-codebase startup fast while scaling to large repos.

### 5. Hybrid retrieval: semantic + symbol
The RAG retriever optionally takes a `SymbolIndex`. When both are available, it interleaves semantic results with exact symbol matches. This captures both "code similar to this" and "code that defines this symbol".

### 6. Persistence: FAISS binary + JSON metadata
FAISS binary format is the most compact and fastest to load. JSON metadata is human-readable and easy to debug. Both stored in `.chef-human/` alongside other project-local config.

---

## Changes & Deviations Tracking

### 3.2.1 CodeChunker
| Deviation | Rationale |
|-----------|-----------|
| `_overlap_lines()` uses fixed estimate (~10 tokens/line) instead of per-line token counting | Counting tokens for every line during overlap calculation adds overhead; the estimate is good enough for sliding-window purposes |
| `_find_end_line()` prefers declaration boundary when over budget instead of exact budget matching | Heuristic favours code structure over token precision — a chunk slightly over budget with a clean function boundary is better than a mid-function cut at exactly 512 tokens |
| Chunk ID auto-generated in `__post_init__` if not provided | Simplifies construction; caller can override for custom IDs |

### 3.2.2 VectorStore
| Deviation | Rationale |
|-----------|-----------|
| `add()` guard-clauses on empty input | FAISS `IndexFlatIP.add()` crashes on `shape (1, 0)` arrays; empty add is a no-op |
| `search()` clamps `top_k` to `len(metadata)` | Prevents FAISS from returning out-of-range indices on an undersized index |
| Uses `IndexFlatIP` (inner product = cosine on normalized vectors) instead of `IndexFlatL2` | Embeddings are L2-normalized at embed time; inner product = cosine similarity; higher score = more similar |
| `load()` uses `__new__` + manual field assignment instead of `__init__` | Avoids creating a fresh empty FAISS index that would be immediately discarded |

### 3.2.3 RAGRetriever
| Deviation | Rationale |
|-----------|-----------|
| `_initial_built` derived from `len(store) > 0` at init | Allows loaded-from-disk stores to be immediately usable without a rebuild |
| `retrieve()` imports `Chunk` unconditionally (not under `TYPE_CHECKING`) | `Chunk` is a dataclass constructed at runtime in list comprehension; `from __future__ import annotations` makes TYPE_CHECKING-only imports unusable for runtime |
| `format_for_prompt()` truncates by token count of the full block, not per-chunk | Simpler implementation; header + code fence tokens are small relative to content |

### 3.2.4 Integration into ContextAssembler
| Deviation | Rationale |
|-----------|-----------|
| `_build_rag_context` / `_build_symbol_context` split into dedicated factory helpers | Keeps `create_context_assembler()` readable; each path is self-contained with its own imports |
| `_build_rag_context_assembler()` creates a `SymbolIndex` even though RAG doesn't need it | Enables `retrieve_by_symbol()` in `RAGRetriever` as a hybrid fallback |
| RAG mode does not build the symbol index or dependency graph | Large codebases would make these slow; RAG replaces them entirely |

### 3.2.5 Persistence
| Deviation | Rationale |
|-----------|-----------|
| No atomic write (`.tmp` + rename) in initial implementation | Acceptable for single-user tool where partial writes would only occur on crash; can add atomicity later |
| `save()` is on `VectorStore`, not `RAGRetriever` | `RAGRetriever` delegates persistence to the store; retriever state (symbol_index) can be re-derived on load |
| No `is_built` flag persisted separately | Derived from `len(store) > 0`; if store has data, retriever is considered built |

### 3.2.6 Tests
| Deviation | Rationale |
|-----------|-----------|
| 37 tests across 4 files (not the planned ~50) | Some test cases merged (e.g., boundary + overlap in same file); all acceptance criteria covered |
| `test_rag_integration.py` renamed to `test_rag_integration.py` (not `test_integration.py`) | Name collision with existing `tests/test_integration.py` caused pytest import mismatch |
| Mock embedder uses `np.random.RandomState(hash(text) % 2**31)` for deterministic embeddings | Avoids downloading sentence-transformers in unit tests; hash-based seed gives reproducible results for same query text |
| `test_format_respects_budget` uses 1000-token budget for main assertion | Header + fences can exceed 20 tokens; secondary assertion verifies budget enforcement at 5 tokens |

---

## Future Work (Post-3.2)

- **Diff-aware editing** (Phase 3.3) — unified diffs instead of full-file rewrites
- **Watch mode** — auto-refresh index on file changes via `watchdog`
- **Symbol rank by usage** — prioritize frequently referenced symbols in retrieval
- **`lookup_symbol` tool** — explicit tool for agent to query symbol definitions on demand
- **Cross-file refactoring** — rename symbol across all files using index
- **Chunk-level provenance** — track which chunks contributed to which LLM responses
- **Reranking** — cross-encoder reranker (e.g., bge-reranker) on top of FAISS results
