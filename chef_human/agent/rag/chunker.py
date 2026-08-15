from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from chef_human.llm.tokenizer import Tokenizer

_DECL_RE = re.compile(
    r"^\s*(?:(?:async|pub|unsafe|public|private|protected|static|abstract|virtual|override"
    r"|export|declare)\s+)*"
    r"(?:def|class|fn|func|function|interface|type|struct|enum|trait|impl|import|use|constructor"
    r"|async\s+function|module|package)\b"
)


@dataclass(frozen=True)
class Chunk:
    file_path: str
    start_line: int
    end_line: int
    content: str
    chunk_id: str

    def __post_init__(self) -> None:
        if not self.chunk_id:
            object.__setattr__(
                self, "chunk_id", f"{self.file_path}:{self.start_line}-{self.end_line}"
            )


class CodeChunker:
    def __init__(
        self,
        tokenizer: Tokenizer,
        target_tokens: int = 512,
        overlap_tokens: int = 64,
    ) -> None:
        self._tokenizer = tokenizer
        self._target = target_tokens
        self._overlap = overlap_tokens

    def chunk_file(self, file_path: str, content: str) -> list[Chunk]:
        lines = content.splitlines(keepends=True)
        if not lines:
            return []

        chunks: list[Chunk] = []
        start = 0
        while start < len(lines):
            end = self._find_end_line(lines, start)
            chunk_content = "".join(lines[start:end])
            chunk_id = f"{file_path}:{start + 1}-{end}"
            chunks.append(Chunk(
                file_path=file_path,
                start_line=start + 1,
                end_line=end,
                content=chunk_content,
                chunk_id=chunk_id,
            ))
            if end >= len(lines):
                break
            advance = max(1, (end - start) - self._overlap_lines(len(lines)))
            start += advance
        return chunks

    def _find_end_line(self, lines: list[str], start: int) -> int:
        end = start
        token_count = 0
        decl_boundary = -1
        for i in range(start, len(lines)):
            line_tokens = self._tokenizer.count(lines[i])
            if token_count + line_tokens > self._target and end > start:
                if decl_boundary > start:
                    return decl_boundary
                break
            token_count += line_tokens
            end = i + 1
            if _DECL_RE.search(lines[i]):
                decl_boundary = end
        return end

    def _overlap_lines(self, total_lines: int) -> int:
        est = max(1, self._overlap // 10)
        return min(est, total_lines // 2)
