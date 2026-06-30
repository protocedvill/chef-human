from __future__ import annotations

from pathlib import Path

import pytest

from chef_human.agent import create_context_assembler
from chef_human.agent.context import ContextAssembler, ContextConfig, ContextManager, Role
from chef_human.agent.file_context import FileContextManager
from chef_human.agent.repo_map import RepoMap
from chef_human.agent.symbols.extractor import RegexExtractor, Symbol, create_extractor
from chef_human.agent.workspace import WorkspaceManager
from chef_human.llm.tokenizer import ApproxTokenizer, create_tokenizer


# ---------------------------------------------------------------------------
# Factory sanity checks
# ---------------------------------------------------------------------------

class TestCreateContextAssembler:
    def test_creates_without_error(self):
        ca = create_context_assembler()
        assert isinstance(ca, ContextAssembler)

    def test_assemble_does_not_raise(self):
        ca = create_context_assembler()
        result = ca.assemble(system_prompt="You are a helpful assistant.")
        assert isinstance(result, list)
        assert len(result) > 0
        assert result[0].role == Role.system

    def test_accepts_tool_definitions(self):
        ca = create_context_assembler()
        result = ca.assemble(system_prompt="Be helpful.", tool_definitions="**tools**")
        assert "**tools**" in result[0].content


# ---------------------------------------------------------------------------
# End-to-end with temporary workspace
# ---------------------------------------------------------------------------

class TestEndToEnd:
    def test_with_real_files(self, tmp_path: Path):
        (tmp_path / "greet.py").write_text("def hello():\n    print('hi')")
        (tmp_path / "utils.py").write_text("class Helper:\n    pass")
        tokenizer = ApproxTokenizer()
        workspace = WorkspaceManager(root=tmp_path)
        file_ctx = FileContextManager(workspace=workspace, tokenizer=tokenizer)
        file_ctx.get("greet.py")
        file_ctx.get("utils.py")
        repo_map = RepoMap(workspace=workspace, tokenizer=tokenizer)
        config = ContextConfig(max_tokens=1000, max_response_tokens=100, summary_tokens=50)
        conversation = ContextManager(config=config, tokenizer=tokenizer)
        conversation.add_message(type("FakeMessage", (), {"role": "user", "content": "hello"})())
        ca = ContextAssembler(
            conversation=conversation,
            workspace=workspace,
            file_context=file_ctx,
            repo_map=repo_map,
        )
        result = ca.assemble(system_prompt="You are a bot.")
        assert len(result) >= 3
        assert result[0].role == Role.system
        assert "greet.py" in str(result)

    def test_empty_workspace(self, tmp_path: Path):
        tokenizer = ApproxTokenizer()
        workspace = WorkspaceManager(root=tmp_path)
        file_ctx = FileContextManager(workspace=workspace, tokenizer=tokenizer)
        repo_map = RepoMap(workspace=workspace, tokenizer=tokenizer)
        config = ContextConfig(max_tokens=500, max_response_tokens=50, summary_tokens=25)
        conversation = ContextManager(config=config, tokenizer=tokenizer)
        ca = ContextAssembler(
            conversation=conversation,
            workspace=workspace,
            file_context=file_ctx,
            repo_map=repo_map,
        )
        result = ca.assemble(system_prompt="sys")
        system_msgs = [m for m in result if m.role == Role.system]
        repo_msgs = [m for m in result if "Repository Structure" in (m.content or "")]
        file_msgs = [m for m in result if "File Context" in (m.content or "")]
        assert len(repo_msgs) == 0
        assert len(file_msgs) == 0
        assert len(system_msgs) >= 1


# ---------------------------------------------------------------------------
# Factory edge cases
# ---------------------------------------------------------------------------

class TestFactoryEdgeCases:
    def test_create_extractor_fallback(self):
        extractor = create_extractor()
        assert isinstance(extractor, RegexExtractor)

    def test_create_tokenizer_fallback(self):
        tokenizer = create_tokenizer()
        assert isinstance(tokenizer, ApproxTokenizer)


# ---------------------------------------------------------------------------
# Symbol extractor integration with repo map
# ---------------------------------------------------------------------------

class TestSymbolExtractorIntegration:
    def test_extracts_from_python(self):
        extractor = RegexExtractor()
        symbols = extractor.extract("test.py", "def f():\n    pass\n\nclass C:\n    pass")
        assert len(symbols) == 2
        kinds = [s.kind for s in symbols]
        assert "function" in kinds
        assert "class" in kinds

    def test_extracts_from_rust(self):
        extractor = RegexExtractor()
        symbols = extractor.extract("lib.rs", "pub fn foo() {}\nstruct Bar {}")
        assert len(symbols) == 2
        assert symbols[0].name == "foo"
        assert symbols[1].name == "Bar"
