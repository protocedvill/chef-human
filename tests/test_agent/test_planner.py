from __future__ import annotations

import json
import logging
from unittest.mock import AsyncMock, MagicMock

import pytest

from chef_human.agent.planner import (
    Plan,
    PlanNode,
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


class TestPlanNode:
    def test_default_status(self):
        step = PlanNode(index=1, description="Do something")
        assert step.index == 1
        assert step.description == "Do something"
        assert step.status == StepStatus.pending

    def test_custom_status(self):
        step = PlanNode(index=2, description="Done", status=StepStatus.completed)
        assert step.status == StepStatus.completed

    def test_identity_is_node_id_not_description(self):
        """Two separately-created nodes are distinct even with identical
        description/index/status -- identity is the node_id, assigned once
        and carried forward across replans, not the descriptive fields."""
        a = PlanNode(index=1, description="Task")
        b = PlanNode(index=1, description="Task")
        assert a != b
        assert a.node_id != b.node_id

    def test_same_object_is_equal_to_itself(self):
        a = PlanNode(index=1, description="Task")
        assert a == a

    def test_is_leaf_true_with_no_children(self):
        assert PlanNode(description="leaf").is_leaf

    def test_is_leaf_false_with_children(self):
        branch = PlanNode(description="branch", children=[PlanNode(description="child")])
        assert not branch.is_leaf

    def test_set_children_links_parent(self):
        parent = PlanNode(description="parent")
        child_a = PlanNode(description="a")
        child_b = PlanNode(description="b")
        parent.set_children([child_a, child_b])
        assert parent.children == [child_a, child_b]
        assert child_a.parent is parent
        assert child_b.parent is parent


class TestPlan:
    def test_creation(self):
        plan = Plan(goal="Fix the bug")
        assert plan.goal == "Fix the bug"
        assert plan.steps == []

    def test_with_steps(self):
        steps = [
            PlanNode(index=1, description="Find the bug"),
            PlanNode(index=2, description="Fix it"),
        ]
        plan = Plan(goal="Fix the bug", steps=steps)
        assert len(plan.steps) == 2
        assert plan.steps[0].description == "Find the bug"

    def test_construction_links_parent_pointers(self):
        steps = [PlanNode(index=1, description="Find the bug")]
        plan = Plan(goal="Fix the bug", steps=steps)
        assert steps[0].parent is plan.root


class TestReadyRollupBranches:
    def test_root_is_never_a_rollup_branch(self):
        """The root represents the whole plan, already gated by
        unresolved_steps()/the finish flow -- it's excluded from rollup
        so a simple depth-1 plan doesn't need an extra rollup call."""
        leaf = PlanNode(description="leaf", status=StepStatus.completed)
        plan = Plan(goal="g", steps=[leaf])
        assert plan.ready_rollup_branches() == []
        assert plan.is_complete()

    def test_branch_ready_only_once_all_children_complete(self):
        child_a = PlanNode(description="a", status=StepStatus.completed)
        child_b = PlanNode(description="b", status=StepStatus.pending)
        branch = PlanNode(description="branch")
        branch.set_children([child_a, child_b])
        plan = Plan(goal="g", steps=[branch])

        assert plan.ready_rollup_branches() == []

        child_b.status = StepStatus.completed
        assert plan.ready_rollup_branches() == [branch]

    def test_branch_not_ready_again_once_completed(self):
        child = PlanNode(description="a", status=StepStatus.completed)
        branch = PlanNode(description="branch", status=StepStatus.completed)
        branch.set_children([child])
        plan = Plan(goal="g", steps=[branch])
        assert plan.ready_rollup_branches() == []

    def test_deepest_branches_come_first(self):
        grandchild = PlanNode(description="gc", status=StepStatus.completed)
        inner_branch = PlanNode(description="inner")
        inner_branch.set_children([grandchild])
        outer_branch = PlanNode(description="outer")
        outer_branch.set_children([inner_branch])
        plan = Plan(goal="g", steps=[outer_branch])

        # Only inner_branch's children are all complete so far -- outer_branch
        # isn't ready until inner_branch itself is marked complete.
        assert plan.ready_rollup_branches() == [inner_branch]

        inner_branch.status = StepStatus.completed
        assert plan.ready_rollup_branches() == [outer_branch]


class TestCurrentStep:
    def test_returns_first_pending_step(self):
        plan = Plan(goal="g", steps=[
            PlanNode(index=1, description="a", status=StepStatus.completed),
            PlanNode(index=2, description="b", status=StepStatus.pending),
            PlanNode(index=3, description="c", status=StepStatus.pending),
        ])
        step = plan.current_leaf()
        assert step is not None
        assert step.description == "b"

    def test_returns_none_when_all_completed(self):
        plan = Plan(goal="g", steps=[
            PlanNode(index=1, description="a", status=StepStatus.completed),
        ])
        assert plan.current_leaf() is None

    def test_returns_none_for_empty_plan(self):
        plan = Plan(goal="g", steps=[])
        assert plan.current_leaf() is None

    def test_failed_and_skipped_steps_keep_plan_incomplete(self):
        plan = Plan(goal="g", steps=[
            PlanNode(index=1, description="failed", status=StepStatus.failed),
            PlanNode(index=2, description="skipped", status=StepStatus.skipped),
        ])
        assert plan.current_leaf() is None
        assert not plan.is_complete()
        assert [step.description for step in plan.unresolved_steps()] == [
            "failed",
            "skipped",
        ]

    def test_ignores_in_progress_and_failed_steps(self):
        """Only 'pending' counts as the current step -- in_progress is a
        transient marker set during verification, and failed/skipped steps
        are done being worked on."""
        plan = Plan(goal="g", steps=[
            PlanNode(index=1, description="a", status=StepStatus.failed),
            PlanNode(index=2, description="b", status=StepStatus.in_progress),
            PlanNode(index=3, description="c", status=StepStatus.pending),
        ])
        step = plan.current_leaf()
        assert step is not None
        assert step.description == "c"

    def test_walks_multi_level_tree_in_dfs_pre_order(self):
        grandchild_a = PlanNode(description="branch-1a", status=StepStatus.completed)
        grandchild_b = PlanNode(description="branch-1b", status=StepStatus.pending)
        branch_1 = PlanNode(
            description="branch 1",
            children=[grandchild_a, grandchild_b],
        )
        leaf_2 = PlanNode(description="leaf 2", status=StepStatus.pending)
        plan = Plan(goal="g", steps=[branch_1, leaf_2])

        step = plan.current_leaf()
        assert step is not None
        assert step.description == "branch-1b"

        grandchild_b.status = StepStatus.completed
        step = plan.current_leaf()
        assert step is not None
        assert step.description == "leaf 2"

        leaf_2.status = StepStatus.completed
        assert plan.current_leaf() is None
        # All leaves are complete, but branch_1 hasn't passed rollup
        # verification yet -- is_complete() requires that too.
        assert not plan.is_complete()
        assert plan.ready_rollup_branches() == [branch_1]

        branch_1.status = StepStatus.completed
        assert plan.is_complete()


class TestParseSteps:
    def test_json_array_of_strings(self):
        planner = Planner(_make_mock_backend([]))
        content = '["Step 1: Read the file", "Step 2: Edit the file"]'
        steps = planner._parse_steps(content)
        assert len(steps) == 2
        assert steps[0].index == 1
        # A leading "Step N:" echoed back by the model is stripped -- it
        # would otherwise double up wherever the step is displayed (e.g.
        # "Step 1 ('Step 1: Read the file')").
        assert steps[0].description == "Read the file"
        assert steps[1].description == "Edit the file"

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
        assert "Read the file" in steps[0].description
        assert "Edit the file" in steps[1].description

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

    def test_string_items_default_to_leaf(self):
        planner = Planner(_make_mock_backend([]))
        steps = planner._parse_steps('["Step A"]')
        assert steps[0].requested_branch is False

    def test_object_with_branch_type_is_flagged(self):
        planner = Planner(_make_mock_backend([]))
        content = '[{"description": "Implement the module", "type": "branch"}, {"description": "Run tests", "type": "leaf"}]'
        steps = planner._parse_steps(content)
        assert steps[0].requested_branch is True
        assert steps[1].requested_branch is False

    def test_object_without_type_defaults_to_leaf(self):
        planner = Planner(_make_mock_backend([]))
        steps = planner._parse_steps('[{"description": "Read file"}]')
        assert steps[0].requested_branch is False


class TestNormalizeSteps:
    def test_drops_environment_setup_hallucination_when_task_does_not_request_it(self):
        steps = [
            PlanNode(index=1, description="Install Python if it is not already installed"),
            PlanNode(index=2, description="Write hello.py with the required content"),
            PlanNode(index=3, description="Run hello.py and verify the output"),
        ]
        normalized = Planner._normalize_steps(
            "Create hello.py that prints Hello, world!",
            steps,
        )

        assert [step.description for step in normalized] == [
            "Write hello.py with the required content",
            "Run hello.py and verify the output",
        ]

    def test_keeps_environment_setup_steps_when_task_explicitly_requests_setup(self):
        steps = [
            PlanNode(index=1, description="Create a virtual environment"),
            PlanNode(index=2, description="Install dependencies from requirements.txt"),
        ]
        normalized = Planner._normalize_steps(
            "Set up a Python virtualenv and install the project requirements",
            steps,
        )

        assert [step.description for step in normalized] == [
            "Create a virtual environment",
            "Install dependencies from requirements.txt",
        ]

    def test_drops_editor_mechanics_steps(self):
        steps = [
            PlanNode(index=1, description="Open hello.py for editing using nano"),
            PlanNode(index=2, description="Save and close the file"),
            PlanNode(index=3, description="Write the required content to hello.py"),
        ]
        normalized = Planner._normalize_steps(
            "Create hello.py that prints Hello, world!",
            steps,
        )

        assert [step.description for step in normalized] == [
            "Write the required content to hello.py",
        ]

    def test_resolves_conditional_create_for_missing_file(self):
        steps = [
            PlanNode(index=1, description="Create slugify.py if it does not exist"),
            PlanNode(index=2, description="Implement slugify.py"),
        ]
        normalized = Planner._normalize_steps(
            "Implement slugify.py without modifying tests",
            steps,
            planning_facts={"slugify.py": False},
        )

        assert [step.description for step in normalized] == [
            "Create slugify.py",
            "Implement slugify.py",
        ]

    def test_drops_conditional_create_and_optional_explore_for_existing_or_missing_fact(self):
        steps = [
            PlanNode(index=1, description="Create slugify.py if it does not exist"),
            PlanNode(index=2, description="Explore the existing code in slugify.py to understand its current state (if any)"),
            PlanNode(index=3, description="Implement slugify.py"),
        ]
        normalized = Planner._normalize_steps(
            "Implement slugify.py without modifying tests",
            steps,
            planning_facts={"slugify.py": True},
        )

        assert [step.description for step in normalized] == [
            "Explore the existing code in slugify.py to understand its current state (if any)",
            "Implement slugify.py",
        ]

        normalized_missing = Planner._normalize_steps(
            "Implement slugify.py without modifying tests",
            steps,
            planning_facts={"slugify.py": False},
        )

        assert [step.description for step in normalized_missing] == [
            "Create slugify.py",
            "Implement slugify.py",
        ]

    def test_drops_write_tests_when_task_forbids_modifying_tests(self):
        steps = [
            PlanNode(index=1, description="Implement slugify.py"),
            PlanNode(index=2, description="Write unit tests for slugify in test_slugify.py"),
            PlanNode(index=3, description="Run the tests"),
        ]
        normalized = Planner._normalize_steps(
            "Implement slugify.py. Do not modify the specification or tests.",
            steps,
        )

        assert [step.description for step in normalized] == [
            "Implement slugify.py",
            "Run the tests",
        ]


class TestFormatPlanForPrompt:
    def test_empty_plan(self):
        plan = Plan(goal="test")
        result = Planner.format_plan_for_prompt(plan)
        assert result == "## Plan\n"

    def test_single_pending_step(self):
        plan = Plan(goal="test", steps=[PlanNode(index=1, description="Do it")])
        result = Planner.format_plan_for_prompt(plan)
        assert "[ ] Step 1: Do it" in result
        assert "## Plan" in result

    def test_mixed_statuses(self):
        steps = [
            PlanNode(index=1, description="Done", status=StepStatus.completed),
            PlanNode(index=2, description="In progress", status=StepStatus.in_progress),
            PlanNode(index=3, description="Pending"),
        ]
        plan = Plan(goal="test", steps=steps)
        result = Planner.format_plan_for_prompt(plan)
        assert "[✓]" in result
        assert "[→]" in result
        assert "[ ]" in result

    def test_all_status_markers_present(self):
        steps = [
            PlanNode(index=1, description="P", status=StepStatus.pending),
            PlanNode(index=2, description="I", status=StepStatus.in_progress),
            PlanNode(index=3, description="C", status=StepStatus.completed),
            PlanNode(index=4, description="F", status=StepStatus.failed),
            PlanNode(index=5, description="S", status=StepStatus.skipped),
        ]
        plan = Plan(goal="test", steps=steps)
        result = Planner.format_plan_for_prompt(plan)
        assert "[ ]" in result
        assert "[→]" in result
        assert "[✓]" in result
        assert "[✗]" in result
        assert "[-]" in result


class TestFormatPlanForPromptTreeAware:
    def test_collapses_non_active_siblings_to_one_line(self):
        root = PlanNode(description="root")
        leaf_a = PlanNode(index=1, description="Leaf A", status=StepStatus.completed)
        branch_b = PlanNode(index=2, description="Branch B")
        leaf_c = PlanNode(index=3, description="Leaf C")
        root.set_children([leaf_a, branch_b, leaf_c])
        grandchild_1 = PlanNode(index=1, description="Grandchild 1", status=StepStatus.completed)
        grandchild_2 = PlanNode(index=2, description="Grandchild 2")
        branch_b.set_children([grandchild_1, grandchild_2])

        plan = Plan(goal="test", root=root)
        result = Planner.format_plan_for_prompt(plan)

        # Active path: root -> branch_b -> grandchild_2 (current leaf).
        assert "Step 2: Branch B" in result
        assert "Grandchild 1" in result
        assert "Grandchild 2" in result
        # Siblings of active-path nodes still get a full one-line entry.
        assert "Leaf A" in result
        assert "Leaf C" in result

    def test_deep_subtree_off_path_collapses_without_descendants(self):
        root = PlanNode(description="root")
        branch_active = PlanNode(index=1, description="Active branch")
        branch_other = PlanNode(index=2, description="Other branch")
        root.set_children([branch_active, branch_other])
        branch_active.set_children([PlanNode(index=1, description="Pending leaf")])
        deep_child = PlanNode(index=1, description="Deep hidden leaf")
        branch_other.set_children([deep_child])
        deep_child.set_children([PlanNode(index=1, description="Deeper hidden leaf")])

        plan = Plan(goal="test", root=root)
        result = Planner.format_plan_for_prompt(plan)

        assert "Other branch" in result
        assert "Deep hidden leaf" not in result
        assert "Deeper hidden leaf" not in result


class TestGeneratePlan:
    @pytest.mark.asyncio
    async def test_basic_generation(self):
        mock_llm = _make_mock_backend(
            [PlanNode(index=1, description="Read"), PlanNode(index=2, description="Write")]
        )
        planner = Planner(mock_llm)
        plan = await planner.generate_plan("Fix the bug")

        assert plan.goal == "Fix the bug"
        assert len(plan.steps) == 2
        assert plan.steps[0].description == "Read"
        assert plan.steps[1].description == "Write"

    @pytest.mark.asyncio
    async def test_with_repo_context(self):
        mock_llm = _make_mock_backend([PlanNode(index=1, description="Do it")])
        planner = Planner(mock_llm)
        plan = await planner.generate_plan("Fix the bug", repo_context="src/main.py")

        assert plan.goal == "Fix the bug"
        assert len(plan.steps) == 1

    @pytest.mark.asyncio
    async def test_llm_called_with_correct_messages(self, monkeypatch):
        monkeypatch.setattr(Planner, "_check_atomicity", AsyncMock(return_value=(False, "")))
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
    async def test_llm_called_with_repo_context(self, monkeypatch):
        monkeypatch.setattr(Planner, "_check_atomicity", AsyncMock(return_value=(False, "")))
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

    @pytest.mark.asyncio
    async def test_planning_facts_influence_normalization(self):
        mock_complete = AsyncMock(return_value=CompletionResponse(
            message=Message(
                role=Role.assistant,
                content='["Create slugify.py if it does not exist", "Implement slugify.py"]',
            ),
        ))
        mock_llm = MagicMock()
        mock_llm.complete = mock_complete

        planner = Planner(mock_llm)
        plan = await planner.generate_plan(
            "Implement slugify.py without modifying tests",
            planning_facts={"slugify.py": False},
        )

        assert [step.description for step in plan.steps] == [
            "Create slugify.py",
            "Implement slugify.py",
        ]

    @pytest.mark.asyncio
    async def test_branch_child_is_recursively_expanded(self, monkeypatch):
        monkeypatch.setattr(Planner, "_check_atomicity", AsyncMock(return_value=(False, "")))
        root_response = CompletionResponse(
            message=Message(
                role=Role.assistant,
                content=json.dumps(
                    [
                        "Explore the existing code",
                        {"description": "Implement the scheduler module", "type": "branch"},
                        "Run the tests",
                    ]
                ),
            )
        )
        branch_response = CompletionResponse(
            message=Message(
                role=Role.assistant,
                content=json.dumps(["Write scheduler.py", "Write scheduler tests"]),
            )
        )
        mock_complete = AsyncMock(side_effect=[root_response, branch_response])
        mock_llm = MagicMock()
        mock_llm.complete = mock_complete

        planner = Planner(mock_llm)
        plan = await planner.generate_plan("Build a task scheduler")

        assert mock_complete.await_count == 2
        assert len(plan.steps) == 3
        branch = plan.steps[1]
        assert branch.description == "Implement the scheduler module"
        assert not branch.is_leaf
        assert [c.description for c in branch.children] == [
            "Write scheduler.py",
            "Write scheduler tests",
        ]
        # Leaves elsewhere in the tree are untouched by the branch's own expansion.
        assert plan.steps[0].is_leaf
        assert plan.steps[2].is_leaf

    @pytest.mark.asyncio
    async def test_branch_expansion_receives_ancestor_chain(self, monkeypatch):
        monkeypatch.setattr(Planner, "_check_atomicity", AsyncMock(return_value=(False, "")))
        root_response = CompletionResponse(
            message=Message(
                role=Role.assistant,
                content=json.dumps(
                    [{"description": "Implement the scheduler module", "type": "branch"}]
                ),
            )
        )
        branch_response = CompletionResponse(
            message=Message(role=Role.assistant, content='["Write scheduler.py"]'),
        )
        mock_complete = AsyncMock(side_effect=[root_response, branch_response])
        mock_llm = MagicMock()
        mock_llm.complete = mock_complete

        planner = Planner(mock_llm)
        await planner.generate_plan("Build a task scheduler")

        second_call_messages = mock_complete.await_args_list[1].args[0].messages
        user_content = second_call_messages[-1].content
        assert "Build a task scheduler" in user_content
        assert "Implement the scheduler module" in user_content

    @pytest.mark.asyncio
    async def test_no_hard_depth_cap_recurses_multiple_levels(self, monkeypatch):
        monkeypatch.setattr(Planner, "_check_atomicity", AsyncMock(return_value=(False, "")))
        # Each level returns exactly one branch child until the third level,
        # which finally returns a leaf -- nothing in generate_plan should
        # stop this early.
        level1 = CompletionResponse(
            message=Message(
                role=Role.assistant,
                content=json.dumps([{"description": "Level 1", "type": "branch"}]),
            )
        )
        level2 = CompletionResponse(
            message=Message(
                role=Role.assistant,
                content=json.dumps([{"description": "Level 2", "type": "branch"}]),
            )
        )
        level3 = CompletionResponse(
            message=Message(role=Role.assistant, content='["Level 3 leaf"]'),
        )
        mock_complete = AsyncMock(side_effect=[level1, level2, level3])
        mock_llm = MagicMock()
        mock_llm.complete = mock_complete

        planner = Planner(mock_llm)
        plan = await planner.generate_plan("Deeply nested task")

        assert mock_complete.await_count == 3
        level1_node = plan.steps[0]
        level2_node = level1_node.children[0]
        level3_node = level2_node.children[0]
        assert level1_node.description == "Level 1"
        assert level2_node.description == "Level 2"
        assert level3_node.description == "Level 3 leaf"
        assert level3_node.is_leaf
        assert plan.current_leaf() is level3_node

    @pytest.mark.asyncio
    async def test_slow_convergence_logs_warning_without_changing_behavior(
        self, monkeypatch, caplog
    ):
        monkeypatch.setattr(Planner, "_SLOW_CONVERGENCE_DEPTH", 2)
        monkeypatch.setattr(Planner, "_SLOW_CONVERGENCE_NODE_COUNT", 999)
        monkeypatch.setattr(Planner, "_check_atomicity", AsyncMock(return_value=(False, "")))

        level1 = CompletionResponse(
            message=Message(
                role=Role.assistant,
                content=json.dumps([{"description": "Level 1", "type": "branch"}]),
            )
        )
        level2 = CompletionResponse(
            message=Message(
                role=Role.assistant,
                content=json.dumps([{"description": "Level 2", "type": "branch"}]),
            )
        )
        level3 = CompletionResponse(
            message=Message(role=Role.assistant, content='["Level 3 leaf"]'),
        )
        mock_complete = AsyncMock(side_effect=[level1, level2, level3])
        mock_llm = MagicMock()
        mock_llm.complete = mock_complete

        planner = Planner(mock_llm)
        with caplog.at_level("WARNING", logger="chef_human.agent.planner"):
            plan = await planner.generate_plan("Deeply nested task")

        # No behavior change: generation still recursed all the way to a leaf.
        assert mock_complete.await_count == 3
        assert plan.steps[0].children[0].children[0].is_leaf
        assert any("not converged" in record.message for record in caplog.records)

    @pytest.mark.asyncio
    async def test_fast_convergence_does_not_warn(self, caplog):
        mock_llm = _make_mock_backend(
            [PlanNode(index=1, description="Read"), PlanNode(index=2, description="Write")]
        )
        planner = Planner(mock_llm)
        with caplog.at_level("WARNING", logger="chef_human.agent.planner"):
            await planner.generate_plan("Fix the bug")

        assert not any("not converged" in record.message for record in caplog.records)

    @pytest.mark.asyncio
    async def test_cleanup_filters_apply_per_expansion_call(self, monkeypatch):
        monkeypatch.setattr(Planner, "_check_atomicity", AsyncMock(return_value=(False, "")))
        # The env-setup filter should apply independently to the branch's
        # own expansion call, not just the root call.
        root_response = CompletionResponse(
            message=Message(
                role=Role.assistant,
                content=json.dumps(
                    [{"description": "Implement the feature", "type": "branch"}]
                ),
            )
        )
        branch_response = CompletionResponse(
            message=Message(
                role=Role.assistant,
                content=json.dumps(
                    [
                        "Install Python if it is not already installed",
                        "Write feature.py",
                    ]
                ),
            )
        )
        mock_complete = AsyncMock(side_effect=[root_response, branch_response])
        mock_llm = MagicMock()
        mock_llm.complete = mock_complete

        planner = Planner(mock_llm)
        plan = await planner.generate_plan("Add a feature")

        branch = plan.steps[0]
        assert [c.description for c in branch.children] == ["Write feature.py"]


class TestAtomicityCheck:
    """A leaf the generation call proposed is not trusted as atomic on its
    own say-so -- _classify_children runs an independent LLM check per leaf
    (_check_atomicity) and can reclassify it as a branch."""

    @pytest.mark.asyncio
    async def test_leaf_reclassified_as_branch_gets_expanded(self, monkeypatch):
        async def fake_atomicity(self, goal, step, tree_context):
            if step == "Implement subscribe, publish, and retries":
                return True, "bundles three features"
            return False, "fine"

        monkeypatch.setattr(Planner, "_check_atomicity", fake_atomicity)

        root_response = CompletionResponse(
            message=Message(
                role=Role.assistant,
                content=json.dumps(["Implement subscribe, publish, and retries"]),
            )
        )
        breakdown_response = CompletionResponse(
            message=Message(
                role=Role.assistant,
                content=json.dumps(["Implement subscribe", "Implement publish", "Implement retries"]),
            )
        )
        mock_complete = AsyncMock(side_effect=[root_response, breakdown_response])
        mock_llm = MagicMock()
        mock_llm.complete = mock_complete

        planner = Planner(mock_llm)
        plan = await planner.generate_plan("Build a notification bus")

        # Root expansion + the reclassified leaf's own breakdown expansion --
        # the atomicity checks themselves are stubbed, not real completions.
        assert mock_complete.await_count == 2
        node = plan.steps[0]
        assert not node.is_leaf
        assert [c.description for c in node.children] == [
            "Implement subscribe",
            "Implement publish",
            "Implement retries",
        ]

    @pytest.mark.asyncio
    async def test_leaf_confirmed_atomic_stays_a_leaf(self):
        root_response = CompletionResponse(
            message=Message(role=Role.assistant, content=json.dumps(["Write hello.py"])),
        )
        atomicity_response = CompletionResponse(
            message=Message(
                role=Role.assistant,
                content="VERDICT: ATOMIC\nREASON: one file, one write",
            )
        )
        mock_complete = AsyncMock(side_effect=[root_response, atomicity_response])
        mock_llm = MagicMock()
        mock_llm.complete = mock_complete

        planner = Planner(mock_llm)
        plan = await planner.generate_plan("Print hello")

        assert mock_complete.await_count == 2
        assert plan.steps[0].is_leaf

    @pytest.mark.asyncio
    async def test_branch_marked_children_skip_the_atomicity_check(self, monkeypatch):
        checked_steps: list[str] = []

        async def fake_atomicity(self, goal, step, tree_context):
            checked_steps.append(step)
            return False, "fine"

        monkeypatch.setattr(Planner, "_check_atomicity", fake_atomicity)

        root_response = CompletionResponse(
            message=Message(
                role=Role.assistant,
                content=json.dumps(
                    [{"description": "Implement the scheduler module", "type": "branch"}]
                ),
            )
        )
        branch_response = CompletionResponse(
            message=Message(role=Role.assistant, content=json.dumps(["Write scheduler.py"])),
        )
        mock_complete = AsyncMock(side_effect=[root_response, branch_response])
        mock_llm = MagicMock()
        mock_llm.complete = mock_complete

        planner = Planner(mock_llm)
        await planner.generate_plan("Build a task scheduler")

        # A step the generation call already marked "branch" is trusted
        # as-is and never sent through the atomicity check -- only its
        # eventual leaf descendant ("Write scheduler.py") is.
        assert checked_steps == ["Write scheduler.py"]

    @pytest.mark.asyncio
    async def test_atomicity_prompt_includes_goal_and_step(self):
        root_response = CompletionResponse(
            message=Message(role=Role.assistant, content=json.dumps(["Do the one thing"])),
        )
        atomicity_response = CompletionResponse(
            message=Message(role=Role.assistant, content="VERDICT: ATOMIC\nREASON: fine"),
        )
        mock_complete = AsyncMock(side_effect=[root_response, atomicity_response])
        mock_llm = MagicMock()
        mock_llm.complete = mock_complete

        planner = Planner(mock_llm)
        await planner.generate_plan("Some overall goal")

        second_call_messages = mock_complete.await_args_list[1].args[0].messages
        content = second_call_messages[-1].content
        assert "Some overall goal" in content
        assert "Do the one thing" in content

    @pytest.mark.asyncio
    async def test_unparseable_atomicity_response_fails_open_to_leaf(self):
        root_response = CompletionResponse(
            message=Message(role=Role.assistant, content=json.dumps(["Do the one thing"])),
        )
        atomicity_response = CompletionResponse(
            message=Message(role=Role.assistant, content="not a verdict at all"),
        )
        mock_complete = AsyncMock(side_effect=[root_response, atomicity_response])
        mock_llm = MagicMock()
        mock_llm.complete = mock_complete

        planner = Planner(mock_llm)
        plan = await planner.generate_plan("Some overall goal")

        assert mock_complete.await_count == 2
        assert plan.steps[0].is_leaf


class TestNearbyTreeContext:
    """_collect_nearby_nodes/_render_nearby_tree give the atomicity check
    visibility into the surrounding tree, not just the one step's own
    wording -- added after an isolated-per-node check produced an infinite
    oscillation: 'ls' judged NEEDS_BREAKDOWN into 'list current directory
    contents', which was itself then judged NEEDS_BREAKDOWN back into 'ls',
    forever, because neither call had any way to see it was about to
    recreate a step that already existed one level up."""

    def _chain(self, *descriptions):
        """Builds a straight-line chain of PlanNodes, each the sole child of
        the previous one, returning (root, deepest_node)."""
        root = PlanNode(description="root")
        current = root
        for desc in descriptions:
            child = PlanNode(description=desc)
            current.set_children([child])
            current = child
        return root, current

    def test_collect_nearby_nodes_is_nearest_first_and_respects_limit(self):
        root, leaf = self._chain("a", "b", "c", "d", "e")
        # leaf's description is "e"; walk to "c" (two hops up from leaf).
        target = leaf.parent.parent
        assert target.description == "c"

        collected = Planner._collect_nearby_nodes(target, limit=3)

        # "c" itself, then its immediate neighbors (child "d", parent "b") --
        # order between same-distance neighbors follows children-then-parent,
        # not semantically meaningful, but the *set* of closest 3 is exact.
        assert [n.description for n in collected] == ["c", "d", "b"]

    def test_collect_nearby_nodes_includes_siblings_and_ancestors(self):
        parent = PlanNode(description="parent")
        target = PlanNode(description="target")
        sibling = PlanNode(description="sibling")
        parent.set_children([target, sibling])
        grandparent = PlanNode(description="grandparent")
        grandparent.set_children([parent])

        collected = Planner._collect_nearby_nodes(target, limit=40)

        assert {n.description for n in collected} == {
            "target",
            "parent",
            "sibling",
            "grandparent",
        }

    def test_render_nearby_tree_marks_the_checked_node(self):
        parent = PlanNode(description="List current directory contents")
        target = PlanNode(description="ls")
        parent.set_children([target])

        nearby = Planner._collect_nearby_nodes(target, limit=40)
        rendered = Planner._render_nearby_tree(target, nearby)

        assert "List current directory contents" in rendered
        assert "ls  <-- the step being checked" in rendered

    def test_render_nearby_tree_shows_root_as_overall_task(self):
        root, target = self._chain("only step")

        nearby = Planner._collect_nearby_nodes(target, limit=40)
        rendered = Planner._render_nearby_tree(target, nearby)

        assert "(overall task)" in rendered
        assert "root" not in rendered.replace("(overall task)", "")

    @pytest.mark.asyncio
    async def test_check_atomicity_prompt_includes_nearby_structure(self):
        """End-to-end: a leaf's sibling must actually show up in the prompt
        text sent to the atomicity-check LLM call, not just in the internal
        tree-context string that never reaches the model."""
        root_response = CompletionResponse(
            message=Message(
                role=Role.assistant,
                content=json.dumps(["List current directory contents", "ls"]),
            )
        )
        atomicity_response = CompletionResponse(
            message=Message(role=Role.assistant, content="VERDICT: ATOMIC\nREASON: fine"),
        )
        mock_complete = AsyncMock(
            side_effect=[root_response, atomicity_response, atomicity_response]
        )
        mock_llm = MagicMock()
        mock_llm.complete = mock_complete

        planner = Planner(mock_llm)
        await planner.generate_plan("Explore the workspace")

        # The second call checks "List current directory contents"; its
        # sibling "ls" should be visible in that call's prompt.
        second_call_content = mock_complete.await_args_list[1].args[0].messages[0].content
        assert "List current directory contents" in second_call_content
        assert "ls" in second_call_content


class TestSingleChildBreakdownGuard:
    """A breakdown call for an atomicity-reclassified node that produces
    only one child is not a real decomposition -- that child is forced to
    stay a leaf instead of being sent through another atomicity check,
    which previously oscillated forever (a flagged node's breakdown kept
    regenerating a single reworded paraphrase of itself, re-flagged every
    time). An *explicitly* branch-tagged step (the generation call's own
    "type": "branch", not an atomicity reclassification) is trusted as-is
    even if its own breakdown also yields exactly one child -- that case is
    covered by TestGeneratePlan's existing multi-level recursion tests."""

    @pytest.mark.asyncio
    async def test_reclassified_node_with_single_child_breakdown_stays_bounded(self, monkeypatch):
        async def fake_atomicity(self, goal, step, tree_context):
            # Every leaf looks like it bundles too much, so every leaf gets
            # reclassified -- the guard, not the check itself, must be what
            # stops this from recursing forever.
            return True, "bundles multiple concerns"

        monkeypatch.setattr(Planner, "_check_atomicity", fake_atomicity)

        root_response = CompletionResponse(
            message=Message(role=Role.assistant, content=json.dumps(["Do a big thing"])),
        )
        # Every subsequent expansion call also returns exactly one
        # (reworded) child -- the failure pattern actually observed.
        breakdown_response = CompletionResponse(
            message=Message(
                role=Role.assistant, content=json.dumps(["Do a slightly different big thing"])
            ),
        )
        mock_complete = AsyncMock(side_effect=[root_response, breakdown_response])
        mock_llm = MagicMock()
        mock_llm.complete = mock_complete

        planner = Planner(mock_llm)
        plan = await planner.generate_plan("Some overall goal")

        # Exactly 2 calls: the root's own generation, then the one
        # breakdown of the reclassified leaf. No third call -- the
        # guard stopped the resulting single child from being
        # atomicity-checked (and thus re-flagged) again.
        assert mock_complete.await_count == 2
        node = plan.steps[0]
        assert not node.is_leaf
        grandchild = node.children[0]
        assert grandchild.description == "Do a slightly different big thing"
        assert grandchild.is_leaf

    @pytest.mark.asyncio
    async def test_explicit_branch_with_single_child_is_still_trusted(self):
        """Mirrors TestGeneratePlan.test_branch_child_is_recursively_expanded
        but with only one item in the branch's own breakdown -- an
        explicitly-tagged branch is trusted even when its breakdown yields
        just one child, unlike an atomicity-reclassified one."""
        root_response = CompletionResponse(
            message=Message(
                role=Role.assistant,
                content=json.dumps(
                    [{"description": "Implement the scheduler module", "type": "branch"}]
                ),
            )
        )
        branch_response = CompletionResponse(
            message=Message(role=Role.assistant, content=json.dumps(["Write scheduler.py"])),
        )
        atomicity_response = CompletionResponse(
            message=Message(role=Role.assistant, content="VERDICT: ATOMIC\nREASON: fine"),
        )
        mock_complete = AsyncMock(
            side_effect=[root_response, branch_response, atomicity_response]
        )
        mock_llm = MagicMock()
        mock_llm.complete = mock_complete

        planner = Planner(mock_llm)
        plan = await planner.generate_plan("Build a task scheduler")

        # 3 calls: root generation, the explicit branch's own breakdown
        # (yielding one child), and the atomicity check on that child --
        # unlike the reclassified case, this child's parent was never
        # itself reclassified, so _classify_children still runs normally.
        assert mock_complete.await_count == 3
        branch = plan.steps[0]
        assert not branch.is_leaf
        assert [c.description for c in branch.children] == ["Write scheduler.py"]


class TestUpdatePlan:
    @pytest.mark.asyncio
    async def test_merges_completed_steps(self):
        original_steps = [
            PlanNode(index=1, description="Read", status=StepStatus.completed),
            PlanNode(index=2, description="Fix", status=StepStatus.failed),
        ]
        plan = Plan(goal="Fix bug", steps=original_steps)

        # LLM returns revised remaining steps (without the completed one)
        mock_llm = _make_mock_backend([
            PlanNode(index=1, description="Fix properly"),
            PlanNode(index=2, description="Test"),
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
            PlanNode(index=1, description="Read", status=StepStatus.completed),
        ]
        plan = Plan(goal="Fix bug", steps=original_steps)

        mock_llm = _make_mock_backend([
            PlanNode(index=1, description="Read"),  # duplicate
            PlanNode(index=2, description="Write"),
        ])
        planner = Planner(mock_llm)
        revised = await planner.update_plan(plan, failure_context="")

        assert len(revised.steps) == 2  # Read (kept) + Write (new)
        assert revised.steps[0].description == "Read"
        assert revised.steps[1].description == "Write"

    @pytest.mark.asyncio
    async def test_no_completed_steps(self):
        plan = Plan(goal="Fix bug", steps=[
            PlanNode(index=1, description="Read", status=StepStatus.failed),
        ])
        mock_llm = _make_mock_backend([
            PlanNode(index=1, description="Try again"),
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
            PlanNode(index=1, description="Do it", status=StepStatus.failed),
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


class TestReplanSubtree:
    @pytest.mark.asyncio
    async def test_leaf_replan_keeps_node_id_and_gets_fresh_children(self):
        leaf = PlanNode(index=1, description="Write feature.py")
        original_id = leaf.node_id
        plan = Plan(goal="Add a feature", steps=[leaf])

        mock_llm = _make_mock_backend([PlanNode(index=1, description="Write feature.py properly")])
        planner = Planner(mock_llm)
        await planner.replan_subtree(plan, leaf, failure_context="syntax error")

        assert leaf.node_id == original_id
        assert leaf.is_leaf is False  # gained children
        assert [c.description for c in leaf.children] == ["Write feature.py properly"]

    @pytest.mark.asyncio
    async def test_branch_replan_discards_old_children_keeps_node_id(self):
        old_child = PlanNode(index=1, description="Old child", status=StepStatus.completed)
        branch = PlanNode(index=1, description="Build the subsystem")
        branch.set_children([old_child])
        original_id = branch.node_id
        plan = Plan(goal="Big task", steps=[branch])

        mock_llm = _make_mock_backend([PlanNode(index=1, description="New child")])
        planner = Planner(mock_llm)
        await planner.replan_subtree(plan, branch, failure_context="rollup rejected")

        assert branch.node_id == original_id
        assert [c.description for c in branch.children] == ["New child"]
        assert old_child not in branch.children

    @pytest.mark.asyncio
    async def test_only_targets_own_subtree_siblings_untouched(self):
        sibling = PlanNode(index=1, description="Untouched sibling")
        target = PlanNode(index=2, description="Failing part")
        plan = Plan(goal="Task", steps=[sibling, target])
        sibling_id = sibling.node_id
        sibling_children_before = list(sibling.children)

        mock_llm = _make_mock_backend([PlanNode(index=1, description="Fixed part")])
        planner = Planner(mock_llm)
        await planner.replan_subtree(plan, target, failure_context="failed")

        assert plan.steps[0] is sibling
        assert sibling.node_id == sibling_id
        assert sibling.children == sibling_children_before
        assert [c.description for c in target.children] == ["Fixed part"]

    @pytest.mark.asyncio
    async def test_sends_ancestor_chain_and_failure_context(self, monkeypatch):
        monkeypatch.setattr(Planner, "_check_atomicity", AsyncMock(return_value=(False, "")))
        mock_complete = AsyncMock(return_value=CompletionResponse(
            message=Message(role=Role.assistant, content='["Fixed sub-step"]'),
        ))
        mock_llm = MagicMock()
        mock_llm.complete = mock_complete

        branch = PlanNode(index=1, description="Build the subsystem")
        root_leaf = PlanNode(index=1, description="Build the subsystem")
        plan = Plan(goal="Big task", steps=[root_leaf])
        root_leaf.set_children([branch])

        planner = Planner(mock_llm)
        await planner.replan_subtree(plan, branch, failure_context="rollup said no coverage")

        call_args = mock_complete.await_args
        request = call_args.args[0]
        user_msg = request.messages[1].content
        assert "Big task" in user_msg
        assert "Build the subsystem" in user_msg
        assert "rollup said no coverage" in user_msg


class TestParseVerdict:
    def test_complete(self):
        verdict, reason = Planner._parse_verdict("VERDICT: COMPLETE\nREASON: file was created")
        assert verdict == StepVerdict.complete
        assert reason == "file was created"

    def test_complete_verdict_ignores_partial_word_in_reason(self):
        verdict, reason = Planner._parse_verdict(
            "VERDICT: COMPLETE\nREASON: previously partial, now done"
        )
        assert verdict == StepVerdict.complete
        assert reason == "previously partial, now done"

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

    def test_bare_verdict_with_prose_reason(self):
        verdict, reason = Planner._parse_verdict(
            "NOT_COMPLETE\n\nThe evidence shows that slugify.py was created but not implemented."
        )
        assert verdict == StepVerdict.not_complete
        assert reason == "The evidence shows that slugify.py was created but not implemented."

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
        step = PlanNode(index=1, description="Write the function")
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
        step = PlanNode(index=1, description="Write the function")
        await planner.verify_step(plan, step, "created empty utils.py")

        call_args = mock_complete.await_args
        request = call_args.args[0]
        prompt = request.messages[0].content
        assert "Add a function" in prompt
        assert "Write the function" in prompt
        assert "created empty utils.py" in prompt

    @pytest.mark.asyncio
    async def test_repairs_invalid_verifier_response(self):
        mock_complete = AsyncMock(side_effect=[
            CompletionResponse(
                message=Message(
                    role=Role.assistant,
                    content="I think the file was probably created.",
                ),
            ),
            CompletionResponse(
                message=Message(
                    role=Role.assistant,
                    content="VERDICT: COMPLETE\nREASON: evidence shows it",
                ),
            ),
        ])
        mock_llm = MagicMock()
        mock_llm.complete = mock_complete

        planner = Planner(mock_llm)
        plan = Plan(goal="Add a function", steps=[])
        step = PlanNode(index=1, description="Write the function")

        verdict, reason = await planner.verify_step(plan, step, "created empty utils.py")

        assert verdict == StepVerdict.complete
        assert reason == "evidence shows it"
        assert mock_complete.await_count == 2
        repair_request = mock_complete.await_args_list[1].args[0]
        assert "did not follow the required format" in repair_request.messages[0].content
        assert "I think the file was probably created." in repair_request.messages[0].content

    @pytest.mark.asyncio
    async def test_logs_raw_response_when_verdict_cannot_be_repaired(self, caplog):
        mock_complete = AsyncMock(side_effect=[
            CompletionResponse(
                message=Message(
                    role=Role.assistant,
                    content="I think the file was probably created.",
                ),
            ),
            CompletionResponse(
                message=Message(
                    role=Role.assistant,
                    content="Still looks complete to me.",
                ),
            ),
        ])
        mock_llm = MagicMock()
        mock_llm.complete = mock_complete

        planner = Planner(mock_llm)
        plan = Plan(goal="Add a function", steps=[])
        step = PlanNode(index=1, description="Write the function")

        with caplog.at_level("DEBUG", logger="chef_human.agent.planner"):
            verdict, reason = await planner.verify_step(plan, step, "created empty utils.py")

        assert verdict == StepVerdict.not_complete
        assert reason == "Could not parse verifier response"
        assert any(
            "Raw verifier response could not be parsed" in record.message
            for record in caplog.records
        )
        assert any(
            "Verifier repair response could not be parsed" in record.message
            for record in caplog.records
        )


class TestVerifyRollup:
    @pytest.mark.asyncio
    async def test_returns_parsed_verdict(self):
        mock_complete = AsyncMock(return_value=CompletionResponse(
            message=Message(role=Role.assistant, content="VERDICT: COMPLETE\nREASON: covered"),
        ))
        mock_llm = MagicMock()
        mock_llm.complete = mock_complete

        planner = Planner(mock_llm)
        plan = Plan(goal="Build the scheduler", steps=[])
        branch = PlanNode(description="Implement the scheduler module")
        verdict, reason = await planner.verify_rollup(plan, branch, "utils.py exists and works")

        assert verdict == StepVerdict.complete
        assert reason == "covered"

    @pytest.mark.asyncio
    async def test_sends_goal_branch_evidence_and_children_summary(self):
        mock_complete = AsyncMock(return_value=CompletionResponse(
            message=Message(role=Role.assistant, content="VERDICT: NOT_COMPLETE\nREASON: gap"),
        ))
        mock_llm = MagicMock()
        mock_llm.complete = mock_complete

        planner = Planner(mock_llm)
        plan = Plan(goal="Build the scheduler", steps=[])
        branch = PlanNode(description="Implement the scheduler module")

        verdict, reason = await planner.verify_rollup(
            plan,
            branch,
            "scheduler.py is missing the topo-sort function",
            children_summary="- Write scheduler.py: completed -- created the file",
        )

        call_args = mock_complete.await_args
        prompt = call_args.args[0].messages[0].content
        assert "Build the scheduler" in prompt
        assert "Implement the scheduler module" in prompt
        assert "scheduler.py is missing the topo-sort function" in prompt
        assert "Write scheduler.py: completed -- created the file" in prompt
        assert verdict == StepVerdict.not_complete
        assert reason == "gap"

    @pytest.mark.asyncio
    async def test_can_reject_branch_even_with_no_children_summary(self):
        mock_complete = AsyncMock(return_value=CompletionResponse(
            message=Message(role=Role.assistant, content="VERDICT: NOT_COMPLETE\nREASON: missing coverage"),
        ))
        mock_llm = MagicMock()
        mock_llm.complete = mock_complete

        planner = Planner(mock_llm)
        plan = Plan(goal="Build the scheduler", steps=[])
        branch = PlanNode(description="Implement the scheduler module")

        verdict, reason = await planner.verify_rollup(plan, branch, "")

        assert verdict == StepVerdict.not_complete
        assert reason == "missing coverage"


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

        # One completion for the expansion call, one for the atomicity check
        # on its single leaf child -- both go through _complete, so both
        # report usage.
        assert received == [(30, 5), (30, 5)]

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
        step = PlanNode(index=1, description="step")
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


class TestThinkingDebugLogging:
    """Ollama's think mode returns a separate `thinking` field alongside the
    real answer -- Planner._complete logs it at DEBUG so it's visible under
    debug logging without being folded into the normal response content."""

    @pytest.mark.asyncio
    async def test_thinking_is_logged_when_present(self, caplog):
        mock_complete = AsyncMock(return_value=CompletionResponse(
            message=Message(role=Role.assistant, content='["Step 1", "Step 2"]'),
            thinking="weighing how to split this goal",
        ))
        mock_llm = MagicMock()
        mock_llm.complete = mock_complete

        planner = Planner(mock_llm)
        with caplog.at_level(logging.DEBUG, logger="chef_human.agent.planner"):
            await planner.generate_plan("Do the thing")

        assert any(
            "weighing how to split this goal" in record.message
            for record in caplog.records
        )

    @pytest.mark.asyncio
    async def test_nothing_logged_when_thinking_absent(self, caplog):
        mock_complete = AsyncMock(return_value=CompletionResponse(
            message=Message(role=Role.assistant, content='["Step 1", "Step 2"]'),
        ))
        mock_llm = MagicMock()
        mock_llm.complete = mock_complete

        planner = Planner(mock_llm)
        with caplog.at_level(logging.DEBUG, logger="chef_human.agent.planner"):
            await planner.generate_plan("Do the thing")

        assert not any("LLM thinking" in record.message for record in caplog.records)


def _make_mock_backend(steps: list[PlanNode]) -> MagicMock:
    """Create a mock LLMBackend that returns parsed steps."""
    descriptions = [s.description for s in steps]
    content = json.dumps(descriptions)

    mock_complete = AsyncMock(return_value=CompletionResponse(
        message=Message(role=Role.assistant, content=content),
    ))
    mock_llm = MagicMock()
    mock_llm.complete = mock_complete
    return mock_llm
