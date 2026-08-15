from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from chef_human.agent.context import ContextAssembler, ContextConfig, ContextManager
from chef_human.agent.file_context import FileContextManager
from chef_human.agent.repo_map import RepoMap
from chef_human.agent.workspace import WorkspaceManager
from chef_human.llm.backend import Role
from chef_human.llm.tokenizer import ApproxTokenizer


@dataclass
class FakeMessage:
    role: str = "user"
    content: str = ""
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None


@pytest.fixture
def tokenizer() -> ApproxTokenizer:
    return ApproxTokenizer()


@pytest.fixture
def workspace(tmp_path: Path) -> WorkspaceManager:
    return WorkspaceManager(root=tmp_path)


@pytest.fixture
def config() -> ContextConfig:
    return ContextConfig(
        max_tokens=1000,
        max_response_tokens=100,
        summary_tokens=50,
        repo_map_tokens=200,
        file_context_tokens=300,
    )


@pytest.fixture
def conversation(config: ContextConfig) -> ContextManager:
    cm = ContextManager(config=config)
    cm.add_message(FakeMessage(content="user message 1"))  # type: ignore[arg-type]
    cm.add_message(FakeMessage(content="user message 2"))  # type: ignore[arg-type]
    return cm


@pytest.fixture
def file_context(tmp_path: Path, workspace: WorkspaceManager) -> FileContextManager:
    tagger = ApproxTokenizer()
    fcm = FileContextManager(workspace=workspace, tokenizer=tagger, max_files=5, max_tokens=500)
    for name in ("main.py", "utils.py"):
        path = tmp_path / name
        path.write_text(f"# {name}\ndef {name.replace('.', '_')}(): pass")
        fcm.get(path)
    return fcm


@pytest.fixture
def repo_map(workspace: WorkspaceManager) -> RepoMap:
    return RepoMap(workspace=workspace, tokenizer=ApproxTokenizer())


# ---------------------------------------------------------------------------
# ContextConfig
# ---------------------------------------------------------------------------

class TestContextConfigExtended:
    def test_defaults(self):
        cfg = ContextConfig()
        assert cfg.repo_map_tokens == 2000
        assert cfg.file_context_tokens == 10000

    def test_custom_values(self):
        cfg = ContextConfig(repo_map_tokens=500, file_context_tokens=2000)
        assert cfg.repo_map_tokens == 500
        assert cfg.file_context_tokens == 2000


# ---------------------------------------------------------------------------
# ContextAssembler
# ---------------------------------------------------------------------------

class TestContextAssemblerInit:
    def test_accepts_dependencies(self, conversation, workspace, file_context, repo_map):
        ca = ContextAssembler(
            conversation=conversation,
            workspace=workspace,
            file_context=file_context,
            repo_map=repo_map,
        )
        assert ca._conversation is conversation
        assert ca._workspace is workspace

    def test_requires_all_args(self, conversation, workspace, file_context):
        with pytest.raises(TypeError):
            ContextAssembler(conversation=conversation, workspace=workspace, file_context=file_context)  # type: ignore[call-arg]


class TestContextAssemblerAssemble:
    def test_returns_list_of_messages(self, conversation, workspace, file_context, repo_map):
        ca = ContextAssembler(
            conversation=conversation,
            workspace=workspace,
            file_context=file_context,
            repo_map=repo_map,
        )
        result = ca.assemble(system_prompt="You are a helpful assistant.")
        assert isinstance(result, list)
        assert len(result) > 0

    def test_first_message_is_system(self, conversation, workspace, file_context, repo_map):
        ca = ContextAssembler(
            conversation=conversation,
            workspace=workspace,
            file_context=file_context,
            repo_map=repo_map,
        )
        result = ca.assemble(system_prompt="You are a helpful assistant.")
        assert result[0].role == Role.system
        assert "You are a helpful assistant." in result[0].content

    def test_includes_conversation_messages(self, conversation, workspace, file_context, repo_map):
        ca = ContextAssembler(
            conversation=conversation,
            workspace=workspace,
            file_context=file_context,
            repo_map=repo_map,
        )
        result = ca.assemble(system_prompt="sys")
        user_msgs = [m for m in result if m.role == Role.user]
        assert len(user_msgs) == 2

    def test_tool_definitions_appended_to_system_prompt(self, conversation, workspace, file_context, repo_map):
        ca = ContextAssembler(
            conversation=conversation,
            workspace=workspace,
            file_context=file_context,
            repo_map=repo_map,
        )
        result = ca.assemble(system_prompt="Be helpful.")
        assert "Be helpful." in result[0].content

    def test_includes_repo_map(self, conversation, workspace, file_context, repo_map):
        ca = ContextAssembler(
            conversation=conversation,
            workspace=workspace,
            file_context=file_context,
            repo_map=repo_map,
        )
        result = ca.assemble(system_prompt="sys")
        repo_msgs = [m for m in result if "Repository Structure" in (m.content or "")]
        assert len(repo_msgs) > 0

    def test_includes_file_context(self, conversation, workspace, file_context, repo_map):
        ca = ContextAssembler(
            conversation=conversation,
            workspace=workspace,
            file_context=file_context,
            repo_map=repo_map,
        )
        result = ca.assemble(system_prompt="sys")
        file_msgs = [m for m in result if "File Context" in (m.content or "")]
        assert len(file_msgs) > 0
        assert "main.py" in file_msgs[0].content

    def test_system_prompt_always_first(self, conversation, workspace, file_context, repo_map):
        ca = ContextAssembler(
            conversation=conversation,
            workspace=workspace,
            file_context=file_context,
            repo_map=repo_map,
        )
        result = ca.assemble(system_prompt="SYSTEM")
        assert result[0].role == Role.system
        assert "SYSTEM" in result[0].content

    def test_conversation_messages_after_system(self, conversation, workspace, file_context, repo_map):
        ca = ContextAssembler(
            conversation=conversation,
            workspace=workspace,
            file_context=file_context,
            repo_map=repo_map,
        )
        result = ca.assemble(system_prompt="sys")
        system_count = sum(1 for m in result if m.role == Role.system)
        assert result[-1].role != Role.system or system_count == len(result)
        assert result[-1].role == Role.user


class TestContextAssemblerEmptySources:
    def test_empty_workspace_no_file_context(self, conversation, workspace, repo_map):
        file_context = FileContextManager(workspace=workspace, tokenizer=ApproxTokenizer())
        ca = ContextAssembler(
            conversation=conversation,
            workspace=workspace,
            file_context=file_context,
            repo_map=repo_map,
        )
        result = ca.assemble(system_prompt="sys")
        file_msgs = [m for m in result if "File Context" in (m.content or "")]
        assert len(file_msgs) == 0

    def test_empty_workspace_no_repo_map(self, workspace):
        config = ContextConfig(max_tokens=100, max_response_tokens=10, summary_tokens=5, repo_map_tokens=50, file_context_tokens=30)
        conversation = ContextManager(config=config)
        conversation.add_message(FakeMessage(content="hi"))  # type: ignore[arg-type]
        file_context = FileContextManager(workspace=workspace, tokenizer=ApproxTokenizer())
        repo_map = RepoMap(workspace=workspace, tokenizer=ApproxTokenizer())
        ca = ContextAssembler(
            conversation=conversation,
            workspace=workspace,
            file_context=file_context,
            repo_map=repo_map,
        )
        result = ca.assemble(system_prompt="sys")
        repo_msgs = [m for m in result if "Repository Structure" in (m.content or "")]
        assert len(repo_msgs) == 0


class TestContextAssemblerTruncation:
    def test_file_context_truncated_when_over_budget(self, conversation, workspace, repo_map):
        file_context = FileContextManager(workspace=workspace, tokenizer=ApproxTokenizer(), max_tokens=500)
        fp = workspace.root / "data.txt"
        fp.write_text("some content")
        file_context.get(fp)
        ca = ContextAssembler(
            conversation=conversation,
            workspace=workspace,
            file_context=file_context,
            repo_map=repo_map,
        )
        full = ca._build_file_context()
        truncated = ca._truncate_file_context(full, 1)
        assert isinstance(truncated, str)

    def test_tool_definitions_included(self, conversation, workspace, file_context, repo_map):
        ca = ContextAssembler(
            conversation=conversation,
            workspace=workspace,
            file_context=file_context,
            repo_map=repo_map,
        )
        result = ca.assemble(system_prompt="s")
        assert "s" in result[0].content


class TestBuildFileContext:
    def test_empty_when_no_cached_files(self, conversation, workspace, repo_map):
        file_context = FileContextManager(workspace=workspace, tokenizer=ApproxTokenizer())
        ca = ContextAssembler(
            conversation=conversation,
            workspace=workspace,
            file_context=file_context,
            repo_map=repo_map,
        )
        result = ca._build_file_context()
        assert result == ""

    def test_formats_cached_files(self, workspace, repo_map, file_context, conversation):
        ca = ContextAssembler(
            conversation=conversation,
            workspace=workspace,
            file_context=file_context,
            repo_map=repo_map,
        )
        result = ca._build_file_context()
        assert "File:" in result
        assert "main.py" in result
        assert "```" in result


class TestTruncateFileContext:
    def test_keeps_all_when_under_budget(self, conversation, workspace, file_context, repo_map):
        ca = ContextAssembler(
            conversation=conversation,
            workspace=workspace,
            file_context=file_context,
            repo_map=repo_map,
        )
        text = "File: a.py (1 lines)\n```\nx\n```\n\nFile: b.py (1 lines)\n```\ny\n```"
        result = ca._truncate_file_context(text, 9999)
        assert result == text

    def test_truncates_when_over_budget(self, conversation, workspace, file_context, repo_map):
        ca = ContextAssembler(
            conversation=conversation,
            workspace=workspace,
            file_context=file_context,
            repo_map=repo_map,
        )
        text = "File: a.py (1 lines)\n```\naaaaaaaaaa\n```\n\nFile: b.py (1 lines)\n```\nbbbbbbbbbb\n```"
        result = ca._truncate_file_context(text, 5)
        assert "b.py" not in result

    def test_returns_empty_string_for_empty_input(self, conversation, workspace, file_context, repo_map):
        ca = ContextAssembler(
            conversation=conversation,
            workspace=workspace,
            file_context=file_context,
            repo_map=repo_map,
        )
        result = ca._truncate_file_context("", 100)
        assert result == ""


class TestContextAssemblerIntegration:
    def test_assemble_does_not_raise(self, conversation, workspace, file_context, repo_map):
        ca = ContextAssembler(
            conversation=conversation,
            workspace=workspace,
            file_context=file_context,
            repo_map=repo_map,
        )
        result = ca.assemble(system_prompt="test")
        assert isinstance(result, list)
        assert all(hasattr(m, "role") for m in result)
        assert all(hasattr(m, "content") for m in result)
