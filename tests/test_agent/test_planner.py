from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from chef_human.agent.planner import (
    Plan,
    PlanStep,
    Planner,
    StepStatus,
    StepVerdict,
)
from chef_human.llm.backend import CompletionResponse, Message, Role


class TestStepStatus:
    def test_enum_values(self):
        assert StepStatus.pending.value == "pending"
        assert StepStatus.in_progress.value == "in_progress"
        assert StepStatus.completed.value == "completed"
        assert StepStatus.failed.value == "failed"
        assert StepStatus.skipped.value == "skipped"

    def test_is_string_enum(self):
        assert isinstance(StepStatus.pending, str)


class TestPlanStep:
    def test_default_status(self):
        step = PlanStep(index=1, description="Do something")
        assert step.index == 1
        assert step.description == "Do something"
        assert step.status == StepStatus.pending

    def test_custom_status(self):
        step = PlanStep(index=2, description="Done", status=StepStatus.completed)
        assert step.status == StepStatus.completed

    def test_equality(self):
        a = PlanStep(index=1, description="Task")
        b = PlanStep(index=1, description="Task")
        c = PlanStep(index=2, description="Other")
        assert a == b
        assert a != c


class TestPlan:
    def test_creation(self):
        plan = Plan(goal="Fix the bug")
        assert plan.goal == "Fix the bug"
        assert plan.steps == []

    def test_with_steps(self):
        steps = [
            PlanStep(index=1, description="Find the bug"),
            PlanStep(index=2, description="Fix it"),
        ]
        plan = Plan(goal="Fix the bug", steps=steps)
        assert len(plan.steps) == 2
        assert plan.steps[0].description == "Find the bug"


class TestCurrentStep:
    def test_returns_first_pending_step(self):
        plan = Plan(goal="g", steps=[
            PlanStep(index=1, description="a", status=StepStatus.completed),
            PlanStep(index=2, description="b", status=StepStatus.pending),
            PlanStep(index=3, description="c", status=StepStatus.pending),
        ])
        step = plan.current_step()
        assert step is not None
        assert step.description == "b"

    def test_returns_none_when_all_completed(self):
        plan = Plan(goal="g", steps=[
            PlanStep(index=1, description="a", status=StepStatus.completed),
        ])
        assert plan.current_step() is None

    def test_returns_none_for_empty_plan(self):
        plan = Plan(goal="g", steps=[])
        assert plan.current_step() is None

    def test_ignores_in_progress_and_failed_steps(self):
        """Only 'pending' counts as the current step -- in_progress is a
        transient marker set during verification, and failed/skipped steps
        are done being worked on."""
        plan = Plan(goal="g", steps=[
            PlanStep(index=1, description="a", status=StepStatus.failed),
            PlanStep(index=2, description="b", status=StepStatus.in_progress),
            PlanStep(index=3, description="c", status=StepStatus.pending),
        ])
        step = plan.current_step()
        assert step is not None
        assert step.description == "c"


class TestParseSteps:
    def test_json_array_of_strings(self):
        planner = Planner(_make_mock_backend([]))
        content = '["Step 1: Read the file", "Step 2: Edit the file"]'
        steps = planner._parse_steps(content)
        assert len(steps) == 2
        assert steps[0].index == 1
        assert steps[0].description == "Step 1: Read the file"
        assert steps[1].description == "Step 2: Edit the file"

    def test_json_array_of_objects(self):
        planner = Planner(_make_mock_backend([]))
        content = '[{"description": "Read file"}, {"description": "Write file"}]'
        steps = planner._parse_steps(content)
        assert len(steps) == 2
        assert steps[0].description == "Read file"
        assert steps[1].description == "Write file"

    def test_json_array_of_objects_without_description_key(self):
        planner = Planner(_make_mock_backend([]))
        content = '[{"name": "Read file"}, {"name": "Write file"}]'
        steps = planner._parse_steps(content)
        assert len(steps) == 2
        # Falls back to str(item) for unknown keys
        assert "{'name': 'Read file'}" in steps[0].description

    def test_json_with_extra_text(self):
        planner = Planner(_make_mock_backend([]))
        content = 'Here is the plan:\n["Step A", "Step B"]\nLet me know if correct.'
        steps = planner._parse_steps(content)
        assert len(steps) == 2
        assert steps[0].description == "Step A"
        assert steps[1].description == "Step B"

    def test_plain_text_lines(self):
        planner = Planner(_make_mock_backend([]))
        content = "Step 1: Read the file\nStep 2: Edit the file"
        steps = planner._parse_steps(content)
        assert len(steps) == 2
        assert "Step 1: Read the file" in steps[0].description
        assert "Step 2: Edit the file" in steps[1].description

    def test_plain_text_with_empty_lines(self):
        planner = Planner(_make_mock_backend([]))
        content = "First step\n\nSecond step\n\n\nThird step"
        steps = planner._parse_steps(content)
        assert len(steps) == 3

    def test_malformed_json(self):
        planner = Planner(_make_mock_backend([]))
        content = "[not valid json at all"
        steps = planner._parse_steps(content)
        # Falls back to line-by-line
        assert len(steps) == 1
        assert "[not valid json at all" in steps[0].description

    def test_empty_content(self):
        planner = Planner(_make_mock_backend([]))
        steps = planner._parse_steps("")
        assert steps == []

    def test_single_object(self):
        planner = Planner(_make_mock_backend([]))
        content = '{"description": "Only one step"}'
        steps = planner._parse_steps(content)
        assert len(steps) == 1
        assert "Only one step" in steps[0].description

    def test_json_array_with_extra_whitespace(self):
        planner = Planner(_make_mock_backend([]))
        content = '  \n  ["a", "b"]  \n  '
        steps = planner._parse_steps(content)
        assert len(steps) == 2
        assert steps[0].description == "a"

    def test_mixed_list_items_returns_single_step(self):
        planner = Planner(_make_mock_backend([]))
        content = '["string step", {"description": "object step"}]'
        steps = planner._parse_steps(content)
        # Does not match "all strings" or "all dicts", so falls to final return
        assert len(steps) >= 1


class TestFormatPlanForPrompt:
    def test_empty_plan(self):
        plan = Plan(goal="test")
        result = Planner.format_plan_for_prompt(plan)
        assert result == "## Plan\n"

    def test_single_pending_step(self):
        plan = Plan(goal="test", steps=[PlanStep(index=1, description="Do it")])
        result = Planner.format_plan_for_prompt(plan)
        assert "[ ] Step 1: Do it" in result
        assert "## Plan" in result

    def test_mixed_statuses(self):
        steps = [
            PlanStep(index=1, description="Done", status=StepStatus.completed),
            PlanStep(index=2, description="In progress", status=StepStatus.in_progress),
            PlanStep(index=3, description="Pending"),
        ]
        plan = Plan(goal="test", steps=steps)
        result = Planner.format_plan_for_prompt(plan)
        assert "[✓]" in result
        assert "[→]" in result
        assert "[ ]" in result

    def test_all_status_markers_present(self):
        steps = [
            PlanStep(index=1, description="P", status=StepStatus.pending),
            PlanStep(index=2, description="I", status=StepStatus.in_progress),
            PlanStep(index=3, description="C", status=StepStatus.completed),
            PlanStep(index=4, description="F", status=StepStatus.failed),
            PlanStep(index=5, description="S", status=StepStatus.skipped),
        ]
        plan = Plan(goal="test", steps=steps)
        result = Planner.format_plan_for_prompt(plan)
        assert "[ ]" in result
        assert "[→]" in result
        assert "[✓]" in result
        assert "[✗]" in result
        assert "[-]" in result


class TestGeneratePlan:
    @pytest.mark.asyncio
    async def test_basic_generation(self):
        mock_llm = _make_mock_backend(
            [PlanStep(index=1, description="Read"), PlanStep(index=2, description="Write")]
        )
        planner = Planner(mock_llm)
        plan = await planner.generate_plan("Fix the bug")

        assert plan.goal == "Fix the bug"
        assert len(plan.steps) == 2
        assert plan.steps[0].description == "Read"
        assert plan.steps[1].description == "Write"

    @pytest.mark.asyncio
    async def test_with_repo_context(self):
        mock_llm = _make_mock_backend([PlanStep(index=1, description="Do it")])
        planner = Planner(mock_llm)
        plan = await planner.generate_plan("Fix the bug", repo_context="src/main.py")

        assert plan.goal == "Fix the bug"
        assert len(plan.steps) == 1

    @pytest.mark.asyncio
    async def test_llm_called_with_correct_messages(self):
        mock_complete = AsyncMock(return_value=CompletionResponse(
            message=Message(role=Role.assistant, content='["Step 1"]'),
        ))
        mock_llm = MagicMock()
        mock_llm.complete = mock_complete

        planner = Planner(mock_llm)
        await planner.generate_plan("Do the thing")

        call_args = mock_complete.await_args
        assert call_args is not None
        request = call_args.args[0]
        messages = request.messages
        assert len(messages) == 2  # system + user
        assert messages[0].role == Role.system
        assert messages[1].role == Role.user
        assert "Do the thing" in messages[1].content

    @pytest.mark.asyncio
    async def test_llm_called_with_repo_context(self):
        mock_complete = AsyncMock(return_value=CompletionResponse(
            message=Message(role=Role.assistant, content='["Step 1"]'),
        ))
        mock_llm = MagicMock()
        mock_llm.complete = mock_complete

        planner = Planner(mock_llm)
        await planner.generate_plan("Do the thing", repo_context="src/")

        call_args = mock_complete.await_args
        assert call_args is not None
        request = call_args.args[0]
        messages = request.messages
        assert len(messages) == 3  # system + repo + user
        assert "src/" in messages[1].content


class TestUpdatePlan:
    @pytest.mark.asyncio
    async def test_merges_completed_steps(self):
        original_steps = [
            PlanStep(index=1, description="Read", status=StepStatus.completed),
            PlanStep(index=2, description="Fix", status=StepStatus.failed),
        ]
        plan = Plan(goal="Fix bug", steps=original_steps)

        # LLM returns revised remaining steps (without the completed one)
        mock_llm = _make_mock_backend([
            PlanStep(index=1, description="Fix properly"),
            PlanStep(index=2, description="Test"),
        ])
        planner = Planner(mock_llm)
        revised = await planner.update_plan(plan, failure_context="Could not find the bug")

        assert revised.goal == "Fix bug"
        # Completed step preserved
        assert len(revised.steps) == 3
        assert revised.steps[0].description == "Read"
        assert revised.steps[0].status == StepStatus.completed
        # New steps added
        assert revised.steps[1].description == "Fix properly"
        assert revised.steps[2].description == "Test"

    @pytest.mark.asyncio
    async def test_skips_duplicates(self):
        """If LLM returns a step matching an already-completed step, skip it."""
        original_steps = [
            PlanStep(index=1, description="Read", status=StepStatus.completed),
        ]
        plan = Plan(goal="Fix bug", steps=original_steps)

        mock_llm = _make_mock_backend([
            PlanStep(index=1, description="Read"),  # duplicate
            PlanStep(index=2, description="Write"),
        ])
        planner = Planner(mock_llm)
        revised = await planner.update_plan(plan, failure_context="")

        assert len(revised.steps) == 2  # Read (kept) + Write (new)
        assert revised.steps[0].description == "Read"
        assert revised.steps[1].description == "Write"

    @pytest.mark.asyncio
    async def test_no_completed_steps(self):
        plan = Plan(goal="Fix bug", steps=[
            PlanStep(index=1, description="Read", status=StepStatus.failed),
        ])
        mock_llm = _make_mock_backend([
            PlanStep(index=1, description="Try again"),
        ])
        planner = Planner(mock_llm)
        revised = await planner.update_plan(plan, failure_context="Error")

        assert len(revised.steps) == 1
        assert revised.steps[0].description == "Try again"
        assert revised.steps[0].index == 1

    @pytest.mark.asyncio
    async def test_sends_failure_context(self):
        mock_complete = AsyncMock(return_value=CompletionResponse(
            message=Message(role=Role.assistant, content='["Revised step"]'),
        ))
        mock_llm = MagicMock()
        mock_llm.complete = mock_complete

        plan = Plan(goal="Fix", steps=[
            PlanStep(index=1, description="Do it", status=StepStatus.failed),
        ])
        planner = Planner(mock_llm)
        await planner.update_plan(plan, failure_context="Permission denied")

        call_args = mock_complete.await_args
        assert call_args is not None
        request = call_args.args[0]
        messages = request.messages
        assert len(messages) == 2
        user_msg = messages[1].content
        assert "Permission denied" in user_msg
        assert "Fix" in user_msg
        assert "[✗]" in user_msg


class TestParseVerdict:
    def test_complete(self):
        verdict, reason = Planner._parse_verdict("VERDICT: COMPLETE\nREASON: file was created")
        assert verdict == StepVerdict.complete
        assert reason == "file was created"

    def test_partial(self):
        verdict, reason = Planner._parse_verdict("VERDICT: PARTIAL\nREASON: only half done")
        assert verdict == StepVerdict.partial
        assert reason == "only half done"

    def test_not_complete(self):
        verdict, reason = Planner._parse_verdict("VERDICT: NOT_COMPLETE\nREASON: nothing happened")
        assert verdict == StepVerdict.not_complete
        assert reason == "nothing happened"

    def test_not_complete_with_space_variant(self):
        verdict, _ = Planner._parse_verdict("VERDICT: NOT COMPLETE\nREASON: no evidence")
        assert verdict == StepVerdict.not_complete

    def test_case_insensitive(self):
        verdict, _ = Planner._parse_verdict("verdict: complete\nreason: done")
        assert verdict == StepVerdict.complete

    def test_unparseable_defaults_to_not_complete(self):
        verdict, reason = Planner._parse_verdict("I'm not sure what happened here.")
        assert verdict == StepVerdict.not_complete
        assert reason == "Could not parse verifier response"

    def test_missing_reason_is_empty_string(self):
        verdict, reason = Planner._parse_verdict("VERDICT: COMPLETE")
        assert verdict == StepVerdict.complete
        assert reason == ""


class TestVerifyStep:
    @pytest.mark.asyncio
    async def test_returns_parsed_verdict(self):
        mock_complete = AsyncMock(return_value=CompletionResponse(
            message=Message(role=Role.assistant, content="VERDICT: COMPLETE\nREASON: evidence shows it"),
        ))
        mock_llm = MagicMock()
        mock_llm.complete = mock_complete

        planner = Planner(mock_llm)
        plan = Plan(goal="Add a function", steps=[])
        step = PlanStep(index=1, description="Write the function")
        verdict, reason = await planner.verify_step(plan, step, "wrote function foo() in utils.py")

        assert verdict == StepVerdict.complete
        assert reason == "evidence shows it"

    @pytest.mark.asyncio
    async def test_sends_goal_step_and_evidence(self):
        mock_complete = AsyncMock(return_value=CompletionResponse(
            message=Message(role=Role.assistant, content="VERDICT: PARTIAL\nREASON: half done"),
        ))
        mock_llm = MagicMock()
        mock_llm.complete = mock_complete

        planner = Planner(mock_llm)
        plan = Plan(goal="Add a function", steps=[])
        step = PlanStep(index=1, description="Write the function")
        await planner.verify_step(plan, step, "created empty utils.py")

        call_args = mock_complete.await_args
        request = call_args.args[0]
        prompt = request.messages[0].content
        assert "Add a function" in prompt
        assert "Write the function" in prompt
        assert "created empty utils.py" in prompt


class TestUsageCallback:
    """Planner's plan-building/verification LLM calls run on a separate call
    path from ReActLoop's main reasoning loop, so token usage from them was
    previously invisible to any UI -- on_usage lets a caller (ReActLoop)
    observe every completion's usage regardless of which Planner method
    triggered it."""

    @pytest.mark.asyncio
    async def test_generate_plan_reports_usage(self):
        mock_complete = AsyncMock(return_value=CompletionResponse(
            message=Message(role=Role.assistant, content='["Step 1"]'),
            usage={"prompt_tokens": 30, "completion_tokens": 5},
        ))
        mock_llm = MagicMock()
        mock_llm.complete = mock_complete

        planner = Planner(mock_llm)
        received: list[tuple[int, int]] = []
        planner.on_usage = lambda p, c: received.append((p, c))

        await planner.generate_plan("Do the thing")

        assert received == [(30, 5)]

    @pytest.mark.asyncio
    async def test_verify_step_reports_usage(self):
        mock_complete = AsyncMock(return_value=CompletionResponse(
            message=Message(role=Role.assistant, content="VERDICT: COMPLETE\nREASON: ok"),
            usage={"prompt_tokens": 12, "completion_tokens": 3},
        ))
        mock_llm = MagicMock()
        mock_llm.complete = mock_complete

        planner = Planner(mock_llm)
        received: list[tuple[int, int]] = []
        planner.on_usage = lambda p, c: received.append((p, c))

        plan = Plan(goal="g", steps=[])
        step = PlanStep(index=1, description="step")
        await planner.verify_step(plan, step, "evidence")

        assert received == [(12, 3)]

    @pytest.mark.asyncio
    async def test_update_plan_reports_usage(self):
        mock_complete = AsyncMock(return_value=CompletionResponse(
            message=Message(role=Role.assistant, content='["New step"]'),
            usage={"prompt_tokens": 20, "completion_tokens": 8},
        ))
        mock_llm = MagicMock()
        mock_llm.complete = mock_complete

        planner = Planner(mock_llm)
        received: list[tuple[int, int]] = []
        planner.on_usage = lambda p, c: received.append((p, c))

        plan = Plan(goal="g", steps=[])
        await planner.update_plan(plan, "it failed")

        assert received == [(20, 8)]

    @pytest.mark.asyncio
    async def test_no_usage_no_callback_when_none_set(self):
        mock_complete = AsyncMock(return_value=CompletionResponse(
            message=Message(role=Role.assistant, content='["Step 1"]'),
            usage={"prompt_tokens": 30, "completion_tokens": 5},
        ))
        mock_llm = MagicMock()
        mock_llm.complete = mock_complete

        planner = Planner(mock_llm)
        # on_usage defaults to None -- must not raise.
        await planner.generate_plan("Do the thing")

    @pytest.mark.asyncio
    async def test_no_callback_when_usage_is_none(self):
        mock_complete = AsyncMock(return_value=CompletionResponse(
            message=Message(role=Role.assistant, content='["Step 1"]'),
            usage=None,
        ))
        mock_llm = MagicMock()
        mock_llm.complete = mock_complete

        planner = Planner(mock_llm)
        received: list[tuple[int, int]] = []
        planner.on_usage = lambda p, c: received.append((p, c))

        await planner.generate_plan("Do the thing")

        assert received == []


def _make_mock_backend(steps: list[PlanStep]) -> MagicMock:
    """Create a mock LLMBackend that returns parsed steps."""
    descriptions = [s.description for s in steps]
    content = json.dumps(descriptions)

    mock_complete = AsyncMock(return_value=CompletionResponse(
        message=Message(role=Role.assistant, content=content),
    ))
    mock_llm = MagicMock()
    mock_llm.complete = mock_complete
    return mock_llm
