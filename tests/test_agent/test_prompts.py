from __future__ import annotations

from chef_human.agent.planner import Plan, PlanStep
from chef_human.agent.prompts import (
    AGENT_FINISH_PROMPT,
    AGENT_SYSTEM_PROMPT,
    PLANNER_SYSTEM_PROMPT,
    build_agent_prompt,
    build_planner_prompt,
)
from chef_human.llm.backend import ToolDefinition


class TestPromptConstants:
    def test_planner_prompt_has_instructions(self):
        assert "planning assistant" in PLANNER_SYSTEM_PROMPT
        assert "JSON array" in PLANNER_SYSTEM_PROMPT

    def test_agent_prompt_has_guidelines(self):
        assert "chef-human" in AGENT_SYSTEM_PROMPT
        assert "tool_call" in AGENT_SYSTEM_PROMPT
        assert "finish" in AGENT_SYSTEM_PROMPT

    def test_finish_prompt_has_summary_instructions(self):
        assert "Summarize" in AGENT_FINISH_PROMPT
        assert "changes" in AGENT_FINISH_PROMPT


class TestBuildPlannerPrompt:
    def test_basic_prompt(self):
        prompt = build_planner_prompt("Fix the bug")
        assert "Fix the bug" in prompt
        assert "planning assistant" in prompt

    def test_with_repo_context(self):
        prompt = build_planner_prompt("Fix the bug", repo_context="src/main.py")
        assert "Project context" in prompt
        assert "src/main.py" in prompt

    def test_repo_context_optional(self):
        prompt = build_planner_prompt("Fix")
        assert "Project context" not in prompt

    def test_empty_task(self):
        prompt = build_planner_prompt("")
        assert "User task:" in prompt


class TestBuildAgentPrompt:
    def test_includes_tool_definitions(self):
        plan = Plan(goal="Test", steps=[])
        tool_defs = [
            ToolDefinition(name="read", description="Read", parameters={"type": "object"})
        ]
        prompt = build_agent_prompt(plan=plan, tool_defs=tool_defs)
        assert "read" in prompt

    def test_includes_plan(self):
        plan = Plan(goal="Test", steps=[PlanStep(index=1, description="Do something")])
        tool_defs: list[ToolDefinition] = []
        prompt = build_agent_prompt(plan=plan, tool_defs=tool_defs)
        assert "Step 1" in prompt
        assert "Do something" in prompt

    def test_with_both(self):
        plan = Plan(goal="Test", steps=[PlanStep(index=1, description="Do something")])
        tool_defs = [
            ToolDefinition(name="read", description="Read", parameters={"type": "object"})
        ]
        prompt = build_agent_prompt(plan=plan, tool_defs=tool_defs)
        assert "read" in prompt
        assert "Step 1" in prompt

    def test_repo_map_empty_uses_fallback(self):
        plan = Plan(goal="Test", steps=[])
        tool_defs: list[ToolDefinition] = []
        prompt = build_agent_prompt(plan=plan, tool_defs=tool_defs)
        assert "no project context loaded" in prompt

    def test_repo_map_included_when_provided(self):
        plan = Plan(goal="Test", steps=[])
        tool_defs: list[ToolDefinition] = []
        prompt = build_agent_prompt(plan=plan, tool_defs=tool_defs, repo_map="src/\n  main.py")
        assert "src/" in prompt
        assert "no project context loaded" not in prompt

    def test_tool_call_format_example_present(self):
        plan = Plan(goal="Test", steps=[])
        tool_defs: list[ToolDefinition] = []
        prompt = build_agent_prompt(plan=plan, tool_defs=tool_defs)
        assert "tool_name" in prompt
