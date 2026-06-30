from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_EMBED_MODEL = "BAAI/bge-small-en-v1.5"


class EmbeddingsBackend:
    def __init__(self, model_name: str = DEFAULT_EMBED_MODEL) -> None:
        self._model_name = model_name
        self._model: Any = None

    def _lazy_load(self) -> None:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            logger.info("Loading embedding model: %s", self._model_name)
            self._model = SentenceTransformer(self._model_name)

    def embed(self, texts: list[str]) -> list[list[float]]:
        self._lazy_load()
        embeddings = self._model.encode(texts, normalize_embeddings=True)
        return embeddings.tolist()

    def embed_single(self, text: str) -> list[float]:
        return self.embed([text])[0]

    @property
    def dimension(self) -> int:
        self._lazy_load()
        return self._model.get_sentence_embedding_dimension()
