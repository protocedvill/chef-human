from __future__ import annotations

import importlib
import logging
from typing import Any

from tree_sitter import Language

logger = logging.getLogger(__name__)

_LANGUAGE_PACKAGES: dict[str, tuple[str, str]] = {
    "python": ("tree_sitter_python", "language"),
    "javascript": ("tree_sitter_javascript", "language"),
    "typescript": ("tree_sitter_typescript", "language_typescript"),
    "rust": ("tree_sitter_rust", "language"),
    "go": ("tree_sitter_go", "language"),
    "java": ("tree_sitter_java", "language"),
    "ruby": ("tree_sitter_ruby", "language"),
    "c": ("tree_sitter_c", "language"),
    "cpp": ("tree_sitter_cpp", "language"),
}


class GrammarLoader:
    def __init__(self) -> None:
        self._cache: dict[str, Any] = {}
        self._checked: set[str] = set()
        self._core_available: bool | None = None

    @property
    def is_available(self) -> bool:
        if self._core_available is not None:
            return self._core_available
        try:
            import tree_sitter  # noqa: F401
            self._core_available = True
        except ImportError:
            self._core_available = False
            logger.warning(
                "tree-sitter core is not installed. "
                "Symbol extraction will use regex fallback. "
                "Install with: pip install chef-human[indexing]"
            )
        return self._core_available

    def load(self, language: str) -> Any | None:
        if language in self._cache:
            return self._cache[language]
        if language in self._checked:
            return None

        if not self.is_available:
            self._checked.add(language)
            return None

        entry = _LANGUAGE_PACKAGES.get(language)
        if entry is None:
            self._checked.add(language)
            logger.debug("No grammar package mapping for language: %s", language)
            return None

        pkg_name, func_name = entry
        try:
            mod = importlib.import_module(pkg_name)
            raw = getattr(mod, func_name)()
            lang = Language(raw) if not isinstance(raw, Language) else raw
            self._cache[language] = lang
            logger.info("Loaded tree-sitter grammar for '%s'", language)
            return lang
        except (ImportError, AttributeError):
            self._checked.add(language)
            logger.info(
                "Tree-sitter grammar for '%s' is not installed. "
                "Install with: pip install tree-sitter-%s",
                language,
                language,
            )
            return None

    def loaded_languages(self) -> list[str]:
        return list(self._cache.keys())

    def reset(self) -> None:
        self._cache.clear()
        self._checked.clear()
        self._core_available = None

    @staticmethod
    def supported_languages() -> list[str]:
        return list(_LANGUAGE_PACKAGES.keys())
