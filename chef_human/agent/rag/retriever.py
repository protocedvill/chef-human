from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from chef_human.agent.rag.chunker import Chunk

if TYPE_CHECKING:
    from chef_human.agent.rag.chunker import CodeChunker
    from chef_human.agent.rag.store import VectorStore
    from chef_human.agent.symbols.index import SymbolIndex
    from chef_human.agent.workspace import WorkspaceManager
    from chef_human.llm.embeddings import EmbeddingsBackend
    from chef_human.llm.tokenizer import Tokenizer

logger = logging.getLogger(__name__)


class RAGRetriever:
    def __init__(
        self,
        chunker: CodeChunker,
        embedder: EmbeddingsBackend,
        store: VectorStore,
        workspace: WorkspaceManager,
        tokenizer: Tokenizer,
        symbol_index: SymbolIndex | None = None,
    ) -> None:
        self._chunker = chunker
        self._embedder = embedder
        self._store = store
        self._workspace = workspace
        self._tokenizer = tokenizer
        self._symbol_index = symbol_index
        self._initial_built: bool = len(self._store) > 0

    def build(self, files: list[Path]) -> int:
        self._store.clear()

        chunks: list[Chunk] = []
        for f in files:
            try:
                content = f.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue
            file_chunks = self._chunker.chunk_file(str(f), content)
            chunks.extend(file_chunks)

        if not chunks:
            self._initial_built = True
            return 0

        texts = [c.content for c in chunks]
        embeddings = self._embedder.embed(texts)
        metadata = [
            {
                "chunk_id": c.chunk_id,
                "file_path": c.file_path,
                "start_line": c.start_line,
                "end_line": c.end_line,
                "content": c.content,
            }
            for c in chunks
        ]
        self._store.add(embeddings, metadata)
        self._initial_built = True
        logger.info("RAG built: %d chunks from %d files", len(chunks), len(files))
        return len(chunks)

    def retrieve(self, query: str, top_k: int = 5) -> list[Chunk]:
        if not self._initial_built or len(self._store) == 0:
            return []
        query_emb = self._embedder.embed_single(query)
        results = self._store.search(query_emb, top_k=top_k)
        return [
            Chunk(
                file_path=r.file_path,
                start_line=r.start_line,
                end_line=r.end_line,
                content=r.content,
                chunk_id=r.chunk_id,
            )
            for r in results
        ]

    def retrieve_by_symbol(self, symbol_name: str, top_k: int = 5) -> list[Chunk]:
        if self._symbol_index is None:
            return []
        entries = self._symbol_index.lookup(symbol_name)
        if not entries:
            return []

        chunks: list[Chunk] = []
        seen: set[str] = set()
        for entry in entries:
            file_chunks = self._chunker.chunk_file(
                entry.file_path,
                Path(entry.file_path).read_text(encoding="utf-8", errors="replace"),
            )
            for c in file_chunks:
                if c.chunk_id not in seen:
                    chunks.append(c)
                    seen.add(c.chunk_id)
                    if len(chunks) >= top_k:
                        break
            if len(chunks) >= top_k:
                break
        return chunks

    def format_for_prompt(self, chunks: list[Chunk], max_tokens: int) -> str:
        sections: list[str] = []
        remaining = max_tokens
        for c in chunks:
            header = f"File: {c.file_path}:{c.start_line}-{c.end_line}"
            block = f"{header}\n```\n{c.content}\n```"
            tokens = self._tokenizer.count(block)
            if tokens <= remaining:
                sections.append(block)
                remaining -= tokens
            else:
                break
        return "\n\n".join(sections)

    @property
    def is_built(self) -> bool:
        return self._initial_built

    @property
    def total_chunks(self) -> int:
        return len(self._store)
