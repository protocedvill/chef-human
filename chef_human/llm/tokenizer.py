from __future__ import annotations

import logging
from typing import Protocol

logger = logging.getLogger(__name__)

BACKEND_ENCODING_MAP: dict[str, str] = {
    "qwen": "gpt-4o",
    "deepseek": "cl100k_base",
    "codellama": "cl100k_base",
    "default": "cl100k_base",
}


class Tokenizer(Protocol):
    def count(self, text: str) -> int: ...


class TiktokenTokenizer:
    def __init__(self, encoding_name: str = "cl100k_base") -> None:
        import tiktoken

        self._enc = tiktoken.get_encoding(encoding_name)

    def count(self, text: str) -> int:
        return len(self._enc.encode(text))


class ApproxTokenizer:
    def count(self, text: str) -> int:
        return max(1, len(text) // 4)


def create_tokenizer(model_name: str = "") -> Tokenizer:
    try:
        import tiktoken  # noqa: F401

        key = "default"
        for prefix in BACKEND_ENCODING_MAP:
            if prefix in model_name.lower():
                key = prefix
                break
        return TiktokenTokenizer(BACKEND_ENCODING_MAP[key])
    except ImportError:
        logger.info("tiktoken not available, using approximate tokenizer")
        return ApproxTokenizer()
