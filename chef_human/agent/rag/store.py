from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_INDEX_FILENAME = "rag.index"
_META_FILENAME = "rag.meta.json"


@dataclass
class SearchResult:
    chunk_id: str
    file_path: str
    start_line: int
    end_line: int
    content: str
    score: float


class VectorStore:
    def __init__(
        self,
        dimension: int,
        index_dir: str | Path | None = None,
    ) -> None:
        self._dim = dimension
        self._index_dir = Path(index_dir) if index_dir else None
        self._index = self._create_index()
        self._metadata: list[dict[str, Any]] = []

    def add(self, embeddings: list[list[float]], metadata: list[dict[str, Any]]) -> None:
        if not embeddings:
            return
        import numpy as np

        arr = np.array(embeddings, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        self._index.add(arr)
        self._metadata.extend(metadata)

    def search(self, query: list[float], top_k: int = 5) -> list[SearchResult]:
        import numpy as np

        if len(self._metadata) == 0:
            return []

        arr = np.array([query], dtype=np.float32)
        scores, indices = self._index.search(arr, min(top_k, len(self._metadata)))

        results: list[SearchResult] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self._metadata):
                continue
            meta = self._metadata[int(idx)]
            results.append(SearchResult(
                chunk_id=meta.get("chunk_id", ""),
                file_path=meta.get("file_path", ""),
                start_line=meta.get("start_line", 0),
                end_line=meta.get("end_line", 0),
                content=meta.get("content", ""),
                score=float(score),
            ))
        return results

    def save(self) -> None:
        if self._index_dir is None:
            return
        self._index_dir.mkdir(parents=True, exist_ok=True)
        import faiss

        faiss.write_index(self._index, str(self._index_dir / _INDEX_FILENAME))
        meta_path = self._index_dir / _META_FILENAME
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(self._metadata, f, ensure_ascii=False)

    @classmethod
    def load(
        cls,
        index_dir: str | Path,
        dimension: int,
    ) -> VectorStore | None:
        import faiss

        index_path = Path(index_dir) / _INDEX_FILENAME
        meta_path = Path(index_dir) / _META_FILENAME
        if not index_path.exists() or not meta_path.exists():
            return None

        index = faiss.read_index(str(index_path))
        with open(meta_path, "r", encoding="utf-8") as f:
            metadata: list[dict[str, Any]] = json.load(f)

        store = cls.__new__(cls)
        store._dim = dimension
        store._index_dir = Path(index_dir)
        store._index = index
        store._metadata = metadata
        return store

    def clear(self) -> None:
        self._index = self._create_index()
        self._metadata.clear()

    def __len__(self) -> int:
        return self._index.ntotal

    def _create_index(self) -> Any:
        import faiss

        return faiss.IndexFlatIP(self._dim)
