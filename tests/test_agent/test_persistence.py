from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from chef_human.agent.context import ContextConfig, ContextManager
from chef_human.agent.persistence import (
    delete_session,
    load_conversation,
    load_session_data,
    list_sessions,
    save_conversation,
)
from chef_human.agent.planner import Plan
from chef_human.agent.react_loop import AgentResult, ReActConfig, ReActLoop
from chef_human.llm.backend import Message, Role


@pytest.fixture
def runner():
    from click.testing import CliRunner
    return CliRunner()


# ── ContextManager serialization ──────────────────────────────────────

class TestContextManagerSerialization:
    def test_to_dict_round_trip(self):
        cm = ContextManager(config=ContextConfig(max_tokens=5000))
        cm.add_message(Message(role=Role.user, content="hello"))
        cm.add_message(
            Message(
                role=Role.assistant,
                content="hi",
                tool_calls=[{"function": {"name": "read", "arguments": {"path": "x.py"}}}],
            )
        )
        cm.add_message(Message(role=Role.tool, content="file content"))

        data = cm.to_dict()
        assert data["max_tokens"] == 5000
        assert len(data["messages"]) == 3
        assert data["messages"][0]["role"] == "user"
        assert data["messages"][0]["content"] == "hello"
        assert data["messages"][1]["tool_calls"] is not None
        assert data["messages"][2]["role"] == "tool"

        cm2 = ContextManager.from_dict(data, config=ContextConfig(max_tokens=5000))
        assert len(cm2.messages) == 3
        assert cm2.messages[0].content == "hello"
        assert cm2.messages[0].role == Role.user
        assert cm2.messages[1].tool_calls is not None
        assert cm2.messages[2].role == Role.tool

    def test_to_dict_empty(self):
        cm = ContextManager(config=ContextConfig(max_tokens=32000))
        data = cm.to_dict()
        assert data["messages"] == []
        assert data["max_tokens"] == 32000

    def test_from_dict_restores_messages(self):
        data = {
            "max_tokens": 4000,
            "messages": [
                {"role": "user", "content": "task", "tool_calls": None, "tool_call_id": None},
                {"role": "assistant", "content": "thinking", "tool_calls": None, "tool_call_id": None},
            ],
        }
        cm = ContextManager.from_dict(data, config=ContextConfig(max_tokens=4000))
        assert len(cm.messages) == 2
        assert cm.messages[0].role == Role.user
        assert cm.messages[0].content == "task"

    def test_from_dict_with_tool_calls(self):
        data = {
            "max_tokens": 4000,
            "messages": [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{"function": {"name": "bash", "arguments": {"command": "ls"}}}],
                    "tool_call_id": None,
                },
            ],
        }
        cm = ContextManager.from_dict(data)
        assert cm.messages[0].tool_calls == [{"function": {"name": "bash", "arguments": {"command": "ls"}}}]

    def test_save_and_load_integration(self, tmp_path):
        cm = ContextManager(config=ContextConfig(max_tokens=3000))
        cm.add_message(Message(role=Role.user, content="do something"))

        conv_data = cm.to_dict()
        path = save_conversation(conv_data, task="my task", save_dir=tmp_path)
        assert path.exists()

        # Need the session_id from the save filename
        session_id = path.stem.replace("session_", "")
        loaded = load_conversation(session_id, save_dir=tmp_path)
        assert loaded is not None
        assert loaded["messages"][0]["content"] == "do something"


# ── Persistence module ────────────────────────────────────────────────

class TestPersistence:
    def test_save_conversation_creates_file(self, tmp_path):
        conv = {"messages": [{"role": "user", "content": "hello"}]}
        path = save_conversation(conv, task="test", save_dir=tmp_path)
        assert path.exists()
        assert path.suffix == ".json"

    def test_save_conversation_with_session_id(self, tmp_path):
        conv = {"messages": []}
        path = save_conversation(conv, task="test", save_dir=tmp_path, session_id="abc123")
        assert path.name == "session_abc123.json"

    def test_save_conversation_creates_dir(self, tmp_path):
        nested = tmp_path / "a" / "b" / "c"
        conv = {"messages": []}
        path = save_conversation(conv, task="test", save_dir=nested)
        assert path.exists()
        assert nested.exists()

    def test_save_conversation_content(self, tmp_path):
        conv = {"messages": [{"role": "user", "content": "hello"}]}
        path = save_conversation(conv, task="my task", save_dir=tmp_path)
        data = json.loads(path.read_text())
        assert data["task"] == "my task"
        assert data["conversation"]["messages"][0]["content"] == "hello"
        assert "session_id" in data

    def test_load_conversation_returns_none_for_missing(self, tmp_path):
        result = load_conversation("nonexistent", save_dir=tmp_path)
        assert result is None

    def test_load_conversation_returns_dict(self, tmp_path):
        conv = {"messages": [{"role": "user", "content": "hi"}]}
        path = save_conversation(conv, task="t", save_dir=tmp_path)
        session_id = path.stem.replace("session_", "")
        result = load_conversation(session_id, save_dir=tmp_path)
        assert result == conv

    def test_load_session_data(self, tmp_path):
        conv = {"messages": []}
        path = save_conversation(conv, task="my task", save_dir=tmp_path)
        session_id = path.stem.replace("session_", "")
        data = load_session_data(session_id, save_dir=tmp_path)
        assert data is not None
        assert data["task"] == "my task"
        assert data["session_id"] == session_id

    def test_load_session_data_returns_none(self, tmp_path):
        assert load_session_data("missing", save_dir=tmp_path) is None

    def test_list_sessions_empty_dir(self, tmp_path):
        assert list_sessions(save_dir=tmp_path) == []

    def test_list_sessions_returns_sorted(self, tmp_path):
        save_conversation({"messages": []}, task="first", save_dir=tmp_path, session_id="aaa")
        save_conversation({"messages": []}, task="second", save_dir=tmp_path, session_id="bbb")
        sessions = list_sessions(save_dir=tmp_path)
        assert len(sessions) == 2
        assert sessions[0]["session_id"] == "bbb"
        assert sessions[1]["session_id"] == "aaa"
        assert sessions[0]["task"] == "second"
        assert "path" in sessions[0]

    def test_list_sessions_ignores_non_session_files(self, tmp_path):
        (tmp_path / "other.json").write_text("{}")
        save_conversation({"messages": []}, task="t", save_dir=tmp_path, session_id="s1")
        sessions = list_sessions(save_dir=tmp_path)
        assert len(sessions) == 1


class TestDeleteSession:
    def test_delete_existing(self, tmp_path):
        save_conversation({"messages": []}, task="t", save_dir=tmp_path, session_id="abc")
        assert delete_session("abc", save_dir=tmp_path) is True
        assert not (tmp_path / "session_abc.json").exists()

    def test_delete_nonexistent(self, tmp_path):
        assert delete_session("nonexistent", save_dir=tmp_path) is False

    def test_delete_then_list(self, tmp_path):
        save_conversation({"messages": []}, task="t1", save_dir=tmp_path, session_id="s1")
        save_conversation({"messages": []}, task="t2", save_dir=tmp_path, session_id="s2")
        delete_session("s1", save_dir=tmp_path)
        sessions = list_sessions(save_dir=tmp_path)
        assert len(sessions) == 1
        assert sessions[0]["session_id"] == "s2"


# ── ReActLoop save-on-exit ────────────────────────────────────────────

class TestReActLoopPersistence:
    @pytest.mark.asyncio
    async def test_save_conversation_called_on_completion(self):
        backend = MagicMock()
        backend.complete = AsyncMock(return_value=MagicMock(
            message=MagicMock(
                content='<tool_call>{"name": "finish", "arguments": {"summary": "done"}}</tool_call>'
            )
        ))
        planner = MagicMock()
        planner.generate_plan = AsyncMock(return_value=Plan(goal="test", steps=[]))
        context = MagicMock()
        context.conversation.to_dict.return_value = {"messages": []}
        registry = MagicMock()
        finish_tool = MagicMock()
        finish_tool.name = "finish"
        finish_tool.parameters = {"type": "object", "properties": {"summary": {"type": "string"}}}
        finish_tool.run = AsyncMock(return_value=MagicMock(output="done", success=True, error=None))
        registry.get.side_effect = lambda name: {"finish": finish_tool}.get(name)

        with patch("chef_human.agent.persistence.save_conversation") as mock_save:
            loop = ReActLoop(
                llm_backend=backend,
                tool_registry=registry,
                context_assembler=context,
                planner=planner,
                config=ReActConfig(max_steps=1, save_sessions=True),
            )
            result = await loop.run("test task")

            assert result.success is True
            mock_save.assert_called_once()
            args, kwargs = mock_save.call_args
            assert kwargs["task"] == "test task"

    @pytest.mark.asyncio
    async def test_save_conversation_not_called_when_disabled(self):
        backend = MagicMock()
        backend.complete = AsyncMock(return_value=MagicMock(
            message=MagicMock(
                content='<tool_call>{"name": "finish", "arguments": {"summary": "done"}}</tool_call>'
            )
        ))
        planner = MagicMock()
        planner.generate_plan = AsyncMock(return_value=Plan(goal="test", steps=[]))
        context = MagicMock()
        registry = MagicMock()
        finish_tool = MagicMock()
        finish_tool.name = "finish"
        finish_tool.parameters = {"type": "object", "properties": {"summary": {"type": "string"}}}
        finish_tool.run = AsyncMock(return_value=MagicMock(output="done", success=True, error=None))
        registry.get.side_effect = lambda name: {"finish": finish_tool}.get(name)

        with patch("chef_human.agent.persistence.save_conversation") as mock_save:
            loop = ReActLoop(
                llm_backend=backend,
                tool_registry=registry,
                context_assembler=context,
                planner=planner,
                config=ReActConfig(max_steps=1, save_sessions=False),
            )
            await loop.run("test task")
            mock_save.assert_not_called()

    @pytest.mark.asyncio
    async def test_save_conversation_called_even_on_failure(self):
        backend = MagicMock()
        backend.complete = AsyncMock(return_value=MagicMock(
            message=MagicMock(content="")
        ))
        planner = MagicMock()
        planner.generate_plan = AsyncMock(return_value=Plan(goal="test", steps=[]))
        context = MagicMock()
        context.conversation.to_dict.return_value = {"messages": []}
        registry = MagicMock()

        with patch("chef_human.agent.persistence.save_conversation") as mock_save:
            loop = ReActLoop(
                llm_backend=backend,
                tool_registry=registry,
                context_assembler=context,
                planner=planner,
                config=ReActConfig(max_steps=1, save_sessions=True),
            )
            result = await loop.run("test task")
            assert result.success is False
            mock_save.assert_called_once()

    @pytest.mark.asyncio
    async def test_save_conversation_passes_save_dir(self):
        backend = MagicMock()
        backend.complete = AsyncMock(return_value=MagicMock(
            message=MagicMock(
                content='<tool_call>{"name": "finish", "arguments": {"summary": "done"}}</tool_call>'
            )
        ))
        planner = MagicMock()
        planner.generate_plan = AsyncMock(return_value=Plan(goal="test", steps=[]))
        context = MagicMock()
        context.conversation.to_dict.return_value = {"messages": []}
        registry = MagicMock()
        finish_tool = MagicMock()
        finish_tool.name = "finish"
        finish_tool.parameters = {"type": "object", "properties": {"summary": {"type": "string"}}}
        finish_tool.run = AsyncMock(return_value=MagicMock(output="done", success=True, error=None))
        registry.get.side_effect = lambda name: {"finish": finish_tool}.get(name)

        with patch("chef_human.agent.persistence.save_conversation") as mock_save:
            loop = ReActLoop(
                llm_backend=backend,
                tool_registry=registry,
                context_assembler=context,
                planner=planner,
                config=ReActConfig(max_steps=1, save_sessions=True, save_dir="/tmp/sessions"),
            )
            await loop.run("test task")
            _, kwargs = mock_save.call_args
            assert kwargs["save_dir"] == "/tmp/sessions"


# ── CLI resume / save-dir ─────────────────────────────────────────────

class TestCLIPersistence:
    def test_resume_and_save_dir_in_help(self, runner):
        from chef_human.main import cli

        result = runner.invoke(cli, ["run", "--help"])
        assert result.exit_code == 0
        assert "--resume" in result.output
        assert "--save-dir" in result.output

    def test_resume_passes_to_execute_task(self, runner):
        from chef_human.main import cli

        with patch("chef_human.main._execute_task", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = AgentResult(
                plan=Plan(goal="test", steps=[]),
                steps_taken=1,
                message="Done",
                success=True,
            )
            with patch("chef_human.main.load_session_data") as mock_load:
                mock_load.return_value = {"task": "saved task", "conversation": {"messages": []}}
                runner.invoke(cli, ["run", "--resume", "abc123"])
                kwargs = mock_exec.call_args[1]
                assert kwargs["resume"] == "abc123"
                assert kwargs["task"] == "saved task"

    def test_resume_override_task(self, runner):
        from chef_human.main import cli

        with patch("chef_human.main._execute_task", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = AgentResult(
                plan=Plan(goal="test", steps=[]),
                steps_taken=1,
                message="Done",
                success=True,
            )
            with patch("chef_human.main.load_session_data") as mock_load:
                mock_load.return_value = {"task": "saved task", "conversation": {"messages": []}}
                runner.invoke(cli, ["run", "override task", "--resume", "abc123"])
                kwargs = mock_exec.call_args[1]
                assert kwargs["task"] == "override task"

    def test_resume_session_not_found(self, runner):
        from chef_human.main import cli

        with patch("chef_human.main.load_session_data", return_value=None):
            result = runner.invoke(cli, ["run", "--resume", "missing"])
            assert result.exit_code == 1
            assert "not found" in result.output

    def test_save_dir_passed_to_execute_task(self, runner):
        from chef_human.main import cli

        with patch("chef_human.main._execute_task", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = AgentResult(
                plan=Plan(goal="test", steps=[]),
                steps_taken=1,
                message="Done",
                success=True,
            )
            runner.invoke(cli, ["run", "task", "--save-dir", "/tmp/mysessions"])
            kwargs = mock_exec.call_args[1]
            assert kwargs["save_dir"] == "/tmp/mysessions"
