from __future__ import annotations

from unittest.mock import patch

import pytest

from chef_human.llm.embeddings import DEFAULT_EMBED_MODEL, EmbeddingsBackend


class _FakeTensor:
    """Mimics a numpy array returned by SentenceTransformer.encode()."""

    def __init__(self, data: list[list[float]]) -> None:
        self._data = data

    def tolist(self) -> list[list[float]]:
        return self._data


class TestEmbeddingsBackend:
    def test_default_model_name(self):
        assert DEFAULT_EMBED_MODEL == "BAAI/bge-small-en-v1.5"

    def test_init_with_default_model(self):
        backend = EmbeddingsBackend()
        assert backend._model_name == DEFAULT_EMBED_MODEL
        assert backend._model is None

    def test_init_with_custom_model(self):
        backend = EmbeddingsBackend(model_name="custom-model")
        assert backend._model_name == "custom-model"

    def test_lazy_model_is_none_before_call(self):
        backend = EmbeddingsBackend()
        assert backend._model is None

    def test_embed_raises_importerror_when_not_installed(self):
        backend = EmbeddingsBackend()
        with pytest.raises(ImportError):
            backend.embed(["hello"])

    def test_embed_single_raises_importerror(self):
        backend = EmbeddingsBackend()
        with pytest.raises(ImportError):
            backend.embed_single("hello")

    def test_dimension_raises_importerror(self):
        backend = EmbeddingsBackend()
        with pytest.raises(ImportError):
            _ = backend.dimension


class TestEmbeddingsBackendWithMock:
    def test_embed_returns_list_of_lists(self):
        backend = EmbeddingsBackend()
        mock = _MockModel(embeddings=[[0.1, 0.2], [0.3, 0.4]])
        with patch.object(backend, "_model", mock):
            result = backend.embed(["text1", "text2"])
        assert isinstance(result, list)
        assert len(result) == 2
        assert all(isinstance(v, list) for v in result)

    def test_embed_single_returns_single_vector(self):
        backend = EmbeddingsBackend()
        mock = _MockModel(embeddings=[[0.5, 0.6]])
        with patch.object(backend, "_model", mock):
            result = backend.embed_single("hello")
        assert result == [0.5, 0.6]

    def test_dimension_returns_int(self):
        backend = EmbeddingsBackend()
        mock = _MockModel(dimension=384)
        with patch.object(backend, "_model", mock):
            assert backend.dimension == 384

    def test_embed_single_empty_string(self):
        backend = EmbeddingsBackend()
        mock = _MockModel(embeddings=[[0.0, 0.0]])
        with patch.object(backend, "_model", mock):
            result = backend.embed_single("")
        assert isinstance(result, list)


class _MockModel:
    def __init__(
        self,
        embeddings: list[list[float]] | None = None,
        dimension: int = 384,
    ) -> None:
        self._embeddings = embeddings
        self._dimension = dimension

    def encode(self, texts: list[str], normalize_embeddings: bool = True) -> _FakeTensor:
        if self._embeddings is not None:
            return _FakeTensor(self._embeddings)
        return _FakeTensor([[0.0] * self._dimension] * len(texts))

    def get_sentence_embedding_dimension(self) -> int:
        return self._dimension
