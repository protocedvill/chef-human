from __future__ import annotations

import pytest

from chef_human.agent.rag.chunker import CodeChunker
from chef_human.llm.tokenizer import create_tokenizer


@pytest.fixture
def chunker() -> CodeChunker:
    tokenizer = create_tokenizer()
    return CodeChunker(tokenizer=tokenizer, target_tokens=50, overlap_tokens=10)


class TestChunkSingleFile:
    def test_small_file_single_chunk(self, chunker: CodeChunker):
        content = "def foo():\n    pass\n"
        chunks = chunker.chunk_file("/dev/test.py", content)
        assert len(chunks) == 1
        assert chunks[0].start_line == 1
        assert chunks[0].end_line == 2

    def test_empty_file_no_chunks(self, chunker: CodeChunker):
        chunks = chunker.chunk_file("/dev/empty.py", "")
        assert chunks == []

    def test_large_file_multiple_chunks(self, chunker: CodeChunker):
        lines = [f"print({i})\n" for i in range(200)]
        content = "".join(lines)
        chunks = chunker.chunk_file("/dev/large.py", content)
        assert len(chunks) >= 2

    def test_chunk_metadata(self, chunker: CodeChunker):
        content = "x = 1\n" * 100
        chunks = chunker.chunk_file("/dev/test.py", content)
        for c in chunks:
            assert c.file_path == "/dev/test.py"
            assert 1 <= c.start_line <= c.end_line
            assert len(c.content) > 0
            assert c.chunk_id.startswith("/dev/test.py:")

    def test_chunk_ids_unique(self, chunker: CodeChunker):
        content = "x = 1\n" * 100
        chunks = chunker.chunk_file("/dev/test.py", content)
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_single_line_file(self, chunker: CodeChunker):
        content = "x = 1\n"
        chunks = chunker.chunk_file("/dev/single.py", content)
        assert len(chunks) == 1
        assert chunks[0].content == content


class TestChunkBoundaries:
    def test_declaration_boundaries_respected(self, chunker: CodeChunker):
        content = (
            "import os\n" * 5
            + "def important():\n    pass\n"
            + "x = 1\n" * 20
            + "class MyClass:\n    pass\n"
            + "y = 2\n" * 20
        )
        chunks = chunker.chunk_file("/dev/test.py", content)
        chunk_texts = [c.content for c in chunks]
        assert any("def important" in t for t in chunk_texts)
        assert any("class MyClass" in t for t in chunk_texts)

    def test_overlap_between_chunks(self, chunker: CodeChunker):
        content = "line_{}\n".join(str(i) for i in range(50))
        content = content.format(*range(50))
        chunks = chunker.chunk_file("/dev/overlap.py", content)
        if len(chunks) >= 2:
            first_end_lines = set(range(chunks[0].start_line, chunks[0].end_line + 1))
            second_lines = set(range(chunks[1].start_line, chunks[1].end_line + 1))
            assert first_end_lines & second_lines


class TestChunkEdgeCases:
    def test_no_declarations(self, chunker: CodeChunker):
        content = "x = 1\n" * 200
        chunks = chunker.chunk_file("/dev/test.py", content)
        assert len(chunks) >= 2

    def test_very_large_lines(self, chunker: CodeChunker):
        content = "A" * 10000 + "\n" + "B" * 10000 + "\n"
        chunks = chunker.chunk_file("/dev/test.py", content)
        assert len(chunks) >= 1

    def test_preserves_content(self, chunker: CodeChunker):
        content = "def foo():\n    return 42\n"
        chunks = chunker.chunk_file("/dev/test.py", content)
        assert chunks[0].content == content

    def test_chunk_line_numbers_match_content(self, chunker: CodeChunker):
        lines = [f"print({i})\n" for i in range(10)]
        content = "".join(lines)
        chunks = chunker.chunk_file("/dev/test.py", content)
        for c in chunks:
            chunk_lines = c.content.splitlines(keepends=True)
            expected = lines[c.start_line - 1 : c.end_line]
            assert chunk_lines == expected, (
                f"Mismatch at {c.chunk_id}: got {len(chunk_lines)} lines, "
                f"expected {len(expected)}"
            )
