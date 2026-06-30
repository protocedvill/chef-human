from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from chef_human.llm.backend import (
    CompletionRequest,
    CompletionResponse,
    EmbeddingRequest,
    EmbeddingResponse,
    LLMBackend,
    Message,
    Role,
)
from chef_human.llm.chatml import tool_to_dict

logger = logging.getLogger(__name__)


class LlamaCppBackend(LLMBackend):
    def __init__(
        self,
        model_path: str | Path,
        n_ctx: int = 32768,
        n_gpu_layers: int = 0,
        n_threads: int | None = None,
        verbose: bool = False,
    ) -> None:
        self._model_path = Path(model_path)
        if not self._model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        try:
            from llama_cpp import Llama
        except ImportError as e:
            raise ImportError(
                "llama-cpp-python is required for the LlamaCppBackend. "
                "Install it with: pip install chef-human[llamacpp]"
            ) from e

        self._llm = Llama(
            model_path=str(self._model_path),
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            n_threads=n_threads,
            verbose=verbose,
        )

    @property
    def model_name(self) -> str:
        return self._model_path.name

    @property
    def context_length(self) -> int:
        return self._llm.context_params.n_ctx

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        prompt = self.format_chatml(request)

        response = self._llm.create_completion(
            prompt=prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            stop=request.stop or ["<|im_end|>"],
        )

        content = response["choices"][0]["text"].strip()
        tool_calls = self.parse_tool_calls(content)

        return CompletionResponse(
            message=Message(
                role=Role.assistant,
                content=self.strip_tool_calls(content),
                tool_calls=tool_calls,
            ),
            usage={
                "prompt_tokens": response["usage"]["prompt_tokens"],
                "completion_tokens": response["usage"]["completion_tokens"],
            },
        )

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        embeddings = []
        for text in request.texts:
            emb = self._llm.create_embedding(input=text)
            embeddings.append(emb["data"][0]["embedding"])
        return EmbeddingResponse(embeddings=embeddings)

    def count_tokens(self, text: str) -> int:
        return len(self._llm.tokenize(text.encode("utf-8")))

    @staticmethod
    def format_chatml(request: CompletionRequest) -> str:
        parts: list[str] = []

        if request.tools:
            tools_json = json.dumps(
                [tool_to_dict(t) for t in request.tools], indent=2
            )
            parts.append(
                f"<|im_start|>system\nAvailable tools:\n{tools_json}\n<|im_end|>"
            )

        for msg in request.messages:
            if msg.role == Role.system:
                parts.append(f"<|im_start|>system\n{msg.content}<|im_end|>")
            elif msg.role == Role.user:
                parts.append(f"<|im_start|>user\n{msg.content}<|im_end|>")
            elif msg.role == Role.assistant:
                content = msg.content or ""
                if msg.tool_calls:
                    for tc in msg.tool_calls:
                        content += f"\n<|tool_call|>{json.dumps(tc)}<|/tool_call|>"
                parts.append(f"<|im_start|>assistant\n{content}<|im_end|>")
            elif msg.role == Role.tool:
                parts.append(f"<|im_start|>tool\n{msg.content}<|im_end|>")

        parts.append("<|im_start|>assistant\n")
        return "\n".join(parts)

    @staticmethod
    def parse_tool_calls(text: str) -> list[dict[str, Any]] | None:
        calls: list[dict[str, Any]] = []
        for match in re.finditer(
            r"<tool_call>(.*?)</tool_call>", text, re.DOTALL
        ):
            try:
                calls.append(json.loads(match.group(1)))
            except json.JSONDecodeError:
                logger.warning("Failed to parse tool call: %s", match.group(1))
        return calls if calls else None

    @staticmethod
    def strip_tool_calls(text: str) -> str:
        return re.sub(r"<tool_call>.*?</tool_call>", "", text, flags=re.DOTALL).strip()

