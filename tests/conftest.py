"""Shared test taxonomy and live-backend prerequisite checks."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import pytest


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Apply the repository's test taxonomy and reject ambiguous integrations.

    Unit tests are the default category. UI and optional-RAG are useful
    cross-cutting labels: those tests remain units unless they also carry an
    explicit integration marker.
    """
    for item in items:
        path = Path(str(item.path)).as_posix()
        is_integration = item.get_closest_marker("integration") is not None
        backend_markers = {
            name
            for name in ("integration_ollama", "integration_llamacpp")
            if item.get_closest_marker(name) is not None
        }

        if is_integration and len(backend_markers) != 1:
            raise pytest.UsageError(
                f"{item.nodeid} must have exactly one live-backend marker: "
                "integration_ollama or integration_llamacpp"
            )
        if backend_markers and not is_integration:
            raise pytest.UsageError(
                f"{item.nodeid} has a live-backend marker but is missing integration"
            )

        if not is_integration:
            item.add_marker("unit")
        if "/test_ui/" in path or path.endswith("/test_agent/test_tui.py"):
            item.add_marker("ui")
        if "/test_rag/" in path or path.endswith("/test_embeddings.py"):
            item.add_marker("rag")


def pytest_runtest_setup(item: pytest.Item) -> None:
    """Skip selected live integrations with an actionable prerequisite reason."""
    if item.get_closest_marker("integration_ollama") is not None:
        reason = _ollama_unavailable_reason()
        if reason:
            pytest.skip(reason)

    if item.get_closest_marker("integration_llamacpp") is not None:
        reason = _llamacpp_unavailable_reason()
        if reason:
            pytest.skip(reason)


@lru_cache(maxsize=1)
def _ollama_unavailable_reason() -> str | None:
    from chef_human.config import settings

    try:
        import ollama

        client = ollama.Client(host=settings.ollama_host)
        client.show(settings.ollama_model)
    except Exception as exc:
        return (
            "Ollama integration prerequisite unavailable: start Ollama and run "
            f"`ollama pull {settings.ollama_model}` ({type(exc).__name__}: {exc})"
        )
    return None


@lru_cache(maxsize=1)
def _llamacpp_unavailable_reason() -> str | None:
    model_path = os.environ.get("CHEF_TEST_LLAMACPP_MODEL")
    if not model_path:
        return (
            "llama.cpp integration prerequisite unavailable: set "
            "CHEF_TEST_LLAMACPP_MODEL to a readable GGUF model"
        )
    if not Path(model_path).is_file():
        return f"llama.cpp integration model does not exist: {model_path}"
    try:
        import llama_cpp  # noqa: F401
    except ImportError:
        return "llama.cpp integration prerequisite unavailable: install chef-human[llamacpp]"
    return None
