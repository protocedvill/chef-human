from __future__ import annotations

import pytest

from chef_human.agent.rag.store import VectorStore


@pytest.fixture
def store() -> VectorStore:
    return VectorStore(dimension=4)


class TestVectorStoreAdd:
    def test_add_single(self, store: VectorStore):
        store.add([[0.1, 0.2, 0.3, 0.4]], [{"chunk_id": "c1"}])
        assert len(store) == 1

    def test_add_multiple(self, store: VectorStore):
        embeddings = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ]
        metadata = [{"chunk_id": "c1"}, {"chunk_id": "c2"}]
        store.add(embeddings, metadata)
        assert len(store) == 2

    def test_add_empty_list(self, store: VectorStore):
        store.add([], [])
        assert len(store) == 0


class TestVectorStoreSearch:
    def test_search_returns_top_k(self, store: VectorStore):
        embeddings = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ]
        metadata = [
            {"chunk_id": "c1", "content": "foo"},
            {"chunk_id": "c2", "content": "bar"},
            {"chunk_id": "c3", "content": "baz"},
        ]
        store.add(embeddings, metadata)
        results = store.search([0.9, 0.1, 0.0, 0.0], top_k=2)
        assert len(results) == 2
        assert results[0].chunk_id == "c1"

    def test_search_empty_store(self, store: VectorStore):
        results = store.search([1.0, 0.0, 0.0, 0.0])
        assert results == []

    def test_search_scores_decreasing(self, store: VectorStore):
        embeddings = [
            [1.0, 0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5, 0.5],
            [0.0, 0.0, 0.0, 1.0],
        ]
        metadata = [
            {"chunk_id": "c1"},
            {"chunk_id": "c2"},
            {"chunk_id": "c3"},
        ]
        store.add(embeddings, metadata)
        results = store.search([1.0, 0.0, 0.0, 0.0], top_k=3)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_search_k_larger_than_store(self, store: VectorStore):
        store.add([[1.0, 0.0, 0.0, 0.0]], [{"chunk_id": "c1"}])
        results = store.search([1.0, 0.0, 0.0, 0.0], top_k=10)
        assert len(results) == 1


class TestVectorStoreRemoveByFile:
    def test_removes_only_matching_file(self, store: VectorStore):
        store.add(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
            [
                {"chunk_id": "c1", "file_path": "a.py"},
                {"chunk_id": "c2", "file_path": "a.py"},
                {"chunk_id": "c3", "file_path": "b.py"},
            ],
        )
        removed = store.remove_by_file("a.py")
        assert removed == 2
        assert len(store) == 1
        results = store.search([0.0, 0.0, 1.0, 0.0], top_k=5)
        assert [r.chunk_id for r in results] == ["c3"]

    def test_removing_nonexistent_file_is_a_noop(self, store: VectorStore):
        store.add([[1.0, 0.0, 0.0, 0.0]], [{"chunk_id": "c1", "file_path": "a.py"}])
        removed = store.remove_by_file("nope.py")
        assert removed == 0
        assert len(store) == 1

    def test_remove_then_readd_same_file(self, store: VectorStore):
        """A file re-indexed after an edit shouldn't leave stale duplicate
        chunks around, and the fresh chunk must still be searchable."""
        store.add([[1.0, 0.0, 0.0, 0.0]], [{"chunk_id": "old", "file_path": "a.py"}])
        store.remove_by_file("a.py")
        store.add([[0.0, 1.0, 0.0, 0.0]], [{"chunk_id": "new", "file_path": "a.py"}])
        assert len(store) == 1
        results = store.search([0.0, 1.0, 0.0, 0.0], top_k=5)
        assert [r.chunk_id for r in results] == ["new"]


class TestVectorStoreClear:
    def test_clear_empties_store(self, store: VectorStore):
        store.add([[0.1, 0.2, 0.3, 0.4]], [{"chunk_id": "c1"}])
        store.clear()
        assert len(store) == 0

    def test_clear_then_add(self, store: VectorStore):
        store.add([[0.1, 0.2, 0.3, 0.4]], [{"chunk_id": "c1"}])
        store.clear()
        store.add([[0.5, 0.6, 0.7, 0.8]], [{"chunk_id": "c2"}])
        assert len(store) == 1


class TestVectorStorePersistence:
    def test_save_and_load(self, store: VectorStore, tmp_path):
        store.add(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
            [{"chunk_id": "c1", "content": "foo"}, {"chunk_id": "c2", "content": "bar"}],
        )
        store._index_dir = tmp_path
        store.save()

        loaded = VectorStore.load(tmp_path, dimension=4)
        assert loaded is not None
        assert len(loaded) == 2
        results = loaded.search([1.0, 0.0, 0.0, 0.0], top_k=1)
        assert len(results) == 1
        assert results[0].chunk_id == "c1"

    def test_load_nonexistent(self, tmp_path):
        loaded = VectorStore.load(tmp_path, dimension=4)
        assert loaded is None

    def test_roundtrip_search(self, store: VectorStore, tmp_path):
        embeddings = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
        metadata = [{"chunk_id": "c1"}, {"chunk_id": "c2"}]
        store.add(embeddings, metadata)
        store._index_dir = tmp_path
        store.save()

        loaded = VectorStore.load(tmp_path, dimension=4)
        assert loaded is not None
        results = loaded.search([0.0, 1.0, 0.0, 0.0], top_k=1)
        assert results[0].chunk_id == "c2"

    def test_roundtrip_survives_remove_by_file(self, store: VectorStore, tmp_path):
        """save/load must preserve ids, not just insertion order, since ids
        are what remove_by_file keys off of after a reload."""
        store.add(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
            [{"chunk_id": "c1", "file_path": "a.py"}, {"chunk_id": "c2", "file_path": "b.py"}],
        )
        store._index_dir = tmp_path
        store.save()

        loaded = VectorStore.load(tmp_path, dimension=4)
        assert loaded is not None
        removed = loaded.remove_by_file("a.py")
        assert removed == 1
        assert len(loaded) == 1
        results = loaded.search([0.0, 1.0, 0.0, 0.0], top_k=5)
        assert [r.chunk_id for r in results] == ["c2"]

    def test_loads_pre_incremental_update_format(self, store: VectorStore, tmp_path):
        """Older on-disk metadata was a plain JSON list (no explicit ids) --
        loading it must still work."""
        import json

        import faiss

        store.add(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
            [{"chunk_id": "c1"}, {"chunk_id": "c2"}],
        )
        store._index_dir = tmp_path
        faiss.write_index(store._index, str(tmp_path / "rag.index"))
        with open(tmp_path / "rag.meta.json", "w") as f:
            json.dump([{"chunk_id": "c1"}, {"chunk_id": "c2"}], f)

        loaded = VectorStore.load(tmp_path, dimension=4)
        assert loaded is not None
        assert len(loaded) == 2
        results = loaded.search([1.0, 0.0, 0.0, 0.0], top_k=1)
        assert results[0].chunk_id == "c1"
