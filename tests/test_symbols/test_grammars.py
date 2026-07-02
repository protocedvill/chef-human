from __future__ import annotations

from chef_human.agent.symbols.grammars import GrammarLoader


class TestGrammarLoaderBasics:
    def test_supported_languages_returns_all(self):
        langs = GrammarLoader.supported_languages()
        assert isinstance(langs, list)
        assert len(langs) > 0
        assert "python" in langs
        assert "javascript" in langs
        assert "typescript" in langs
        assert "rust" in langs
        assert "go" in langs
        assert "java" in langs

    def test_loaded_languages_empty_initially(self):
        loader = GrammarLoader()
        assert loader.loaded_languages() == []

    def test_load_nonexistent_language_returns_none(self):
        loader = GrammarLoader()
        result = loader.load("nonexistent")
        assert result is None

    def test_load_nonexistent_does_not_retry(self):
        loader = GrammarLoader()
        assert loader.load("nonexistent") is None
        assert loader.load("nonexistent") is None

    def test_reset_clears_caches(self):
        loader = GrammarLoader()
        py = loader.load("python")
        assert py is not None
        loader.reset()
        assert loader.loaded_languages() == []
        assert loader._checked == set()
        assert loader._core_available is None


class TestGrammarLoaderReal:
    def test_is_available_true(self):
        loader = GrammarLoader()
        assert loader.is_available is True

    def test_load_python_returns_language(self):
        loader = GrammarLoader()
        result = loader.load("python")
        assert result is not None
        assert type(result).__name__ == "Language"

    def test_load_installed_languages(self):
        loader = GrammarLoader()
        installed = ["python", "javascript", "typescript", "rust", "go", "java"]
        for lang in installed:
            result = loader.load(lang)
            assert result is not None, f"Failed to load {lang}"
            assert type(result).__name__ == "Language"

    def test_load_missing_grammar_returns_none(self):
        loader = GrammarLoader()
        assert loader.load("ruby") is None

    def test_load_caches_language(self):
        loader = GrammarLoader()
        first = loader.load("python")
        second = loader.load("python")
        assert first is second

    def test_loaded_languages_after_load(self):
        loader = GrammarLoader()
        loader.load("python")
        assert loader.loaded_languages() == ["python"]

    def test_load_different_languages(self):
        loader = GrammarLoader()
        py = loader.load("python")
        js = loader.load("javascript")
        assert py is not js
        assert py is not None
        assert js is not None
        assert set(loader.loaded_languages()) == {"python", "javascript"}

    def test_reset_after_load(self):
        loader = GrammarLoader()
        loader.load("python")
        assert loader.loaded_languages() == ["python"]
        loader.reset()
        assert loader.loaded_languages() == []


class TestGrammarLoaderEdgeCases:
    def test_is_available_caches_result(self):
        loader = GrammarLoader()
        loader._core_available = False
        assert loader.is_available is False

    def test_load_skips_core_check_when_core_unavailable(self):
        loader = GrammarLoader()
        loader._core_available = False
        loader._checked.add("python")
        assert loader.load("python") is None

    def test_concurrent_instance_independence(self):
        a = GrammarLoader()
        b = GrammarLoader()
        a.load("python")
        assert "python" in a._cache
        assert "python" not in b._cache
