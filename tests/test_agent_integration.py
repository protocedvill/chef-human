from __future__ import annotations

from pathlib import Path

from chef_human.agent import create_context_assembler
from chef_human.agent.context import ContextAssembler, ContextConfig, ContextManager, Role
from chef_human.agent.file_context import FileContextManager
from chef_human.agent.repo_map import RepoMap
from chef_human.agent.symbols.extractor import CompositeExtractor, RegexExtractor, create_extractor
from chef_human.agent.workspace import WorkspaceManager
from chef_human.llm.tokenizer import ApproxTokenizer, create_tokenizer
from chef_human.tools import create_tool_registry
from chef_human.tools.view_diff import ViewDiffTool


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
    def test_create_extractor_returns_composite(self):
        extractor = create_extractor()
        assert isinstance(extractor, CompositeExtractor)

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


# ---------------------------------------------------------------------------
# Tool registry integration (Phase 3.3)
# ---------------------------------------------------------------------------

class TestDiffToolRegistry:
    def test_view_diff_tool_registered(self, tmp_path: Path):
        ws = WorkspaceManager(root=tmp_path)
        registry = create_tool_registry(ws)
        tool = registry.get("view_diff")
        assert tool is not None
        assert isinstance(tool, ViewDiffTool)

    def test_edit_tool_has_diff_store(self, tmp_path: Path):
        ws = WorkspaceManager(root=tmp_path)
        registry = create_tool_registry(ws)
        tool = registry.get("edit")
        assert tool is not None
        assert hasattr(tool, "_diff_store")
        assert tool._diff_store is not None

    def test_write_tool_has_diff_store(self, tmp_path: Path):
        ws = WorkspaceManager(root=tmp_path)
        registry = create_tool_registry(ws)
        tool = registry.get("write")
        assert tool is not None
        assert hasattr(tool, "_diff_store")
        assert tool._diff_store is not None

    def test_diff_store_shared_across_tools(self, tmp_path: Path):
        ws = WorkspaceManager(root=tmp_path)
        registry = create_tool_registry(ws)
        edit_tool = registry.get("edit")
        write_tool = registry.get("write")
        view_tool = registry.get("view_diff")
        assert edit_tool._diff_store is write_tool._diff_store
        assert edit_tool._diff_store is view_tool._store
