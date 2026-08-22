from __future__ import annotations

import json
import types
import subprocess
from pathlib import Path

import pytest

from chef_human import benchmark
from chef_human.benchmark import CASES, BenchmarkCase, Verification
from chef_human.agent.planner import Plan, PlanNode, StepStatus


def _case(*, protected: tuple[str, ...] = ()) -> BenchmarkCase:
    return BenchmarkCase(
        case_id="test_case",
        level="smoke",
        title="Test case",
        task="Create result.py",
        seed_files={"SPEC.md": "keep me\n"} if protected else {},
        verification=Verification(
            command=("{python}", "result.py"),
            expected_stdout="ok\n",
        ),
        protected_files=protected,
    )


def _completed(command, returncode=0, stdout="", stderr=""):
    return subprocess.CompletedProcess(command, returncode, stdout, stderr)


class TestCaseSelection:
    def test_levels_are_cumulative(self):
        assert [case.case_id for case in benchmark.select_cases("smoke", [])] == [
            "hello_world"
        ]
        assert [case.case_id for case in benchmark.select_cases("core", [])] == [
            "hello_world",
            "slugify_contract",
        ]
        assert benchmark.select_cases("all", []) == list(CASES)

    def test_explicit_case_selection(self):
        selected = benchmark.select_cases("smoke", ["inventory_refactor"])
        assert [case.case_id for case in selected] == ["inventory_refactor"]

    def test_unknown_case_is_rejected(self):
        with pytest.raises(ValueError, match="Unknown benchmark"):
            benchmark.select_cases("smoke", ["missing"])


class TestSourceRepoResolution:
    def test_tilde_source_repo_falls_back_to_real_home_when_sandbox_home_differs(
        self, tmp_path, monkeypatch
    ):
        sandbox_home = tmp_path / "sandbox-home"
        real_home = tmp_path / "real-home"
        sandbox_home.mkdir()
        real_home.mkdir()
        (real_home / "ubertooth").mkdir()
        monkeypatch.setenv("HOME", str(sandbox_home))
        monkeypatch.setattr(
            benchmark,
            "pwd",
            types.SimpleNamespace(
                getpwuid=lambda _uid: types.SimpleNamespace(pw_dir=str(real_home))
            ),
        )

        resolved = benchmark._resolve_source_repo_path("~/ubertooth")

        assert resolved == (real_home / "ubertooth").resolve()


class TestRunCase:
    def test_prepare_agent_path_exposes_python_shim(self):
        env = benchmark._prepare_agent_path()
        shim_dir = Path(env["PATH"].split(":")[0])
        shim = shim_dir / "python"

        assert shim.exists()
        assert shim.stat().st_mode & 0o111

    def test_prepare_agent_path_shim_lives_outside_any_workspace(self, tmp_path):
        # The shim dir must not sit inside a workspace being planned over --
        # WorkspaceManager's IGNORE_PATTERNS doesn't know its name, so a
        # workspace-local shim dir would show up in the agent's own repo map
        # and make an empty greenfield workspace look non-empty.
        env = benchmark._prepare_agent_path()
        shim_dir = Path(env["PATH"].split(":")[0])

        assert shim_dir.is_absolute()
        assert tmp_path not in shim_dir.parents

    def test_pass_requires_agent_and_external_verifier(self, tmp_path, monkeypatch):
        calls: list[list[str]] = []

        def fake_run(command, *, cwd, timeout, env=None):
            calls.append(list(command))
            if len(calls) == 1:
                (cwd / "result.py").write_text("print('ok')\n")
                return _completed(
                    command,
                    stdout=json.dumps(
                        {
                            "success": True,
                            "steps_taken": 2,
                            "total_prompt_tokens": 10,
                            "total_completion_tokens": 4,
                        }
                    ),
                )
            return _completed(command, stdout="ok\n")

        monkeypatch.setattr(benchmark, "_run_process", fake_run)
        result = benchmark.run_case(
            _case(),
            tmp_path / "workspace",
            model="test-model",
            agent_timeout=60,
        )

        assert result.passed
        assert result.changed_files == ["result.py"]
        assert result.steps_taken == 2
        assert result.prompt_tokens == 10
        assert "--headless" in calls[0]
        assert calls[0][-2:] == ["--model", "test-model"]
        assert calls[1][0] == benchmark.sys.executable

    def test_agent_receives_absolute_workspace_path(self, tmp_path, monkeypatch):
        commands: list[list[str]] = []

        def fake_run(command, *, cwd, timeout, env=None):
            commands.append(list(command))
            if len(commands) == 1:
                (cwd / "result.py").write_text("print('ok')\n")
                return _completed(command, stdout='{"success": true}')
            return _completed(command, stdout="ok\n")

        monkeypatch.setattr(benchmark, "_run_process", fake_run)
        workspace = tmp_path / "nested" / "workspace"
        benchmark.run_case(_case(), workspace, model=None, agent_timeout=60)

        workspace_index = commands[0].index("--workspace") + 1
        assert commands[0][workspace_index] == str(workspace.resolve())
        log_index = commands[0].index("--log-file") + 1
        assert commands[0][log_index] == str((workspace / ".chef-human" / "agent.log").resolve())

    def test_verifier_can_fail_after_agent_reports_success(self, tmp_path, monkeypatch):
        responses = iter(
            [
                _completed([], stdout='{"success": true, "steps_taken": 1}'),
                _completed([], stdout="wrong\n"),
            ]
        )
        monkeypatch.setattr(
            benchmark,
            "_run_process",
            lambda command, *, cwd, timeout, env=None: next(responses),
        )

        result = benchmark.run_case(
            _case(), tmp_path / "workspace", model=None, agent_timeout=60
        )

        assert result.agent_success
        assert not result.verifier_success
        assert not result.passed
        assert "External verifier failed" in (result.error or "")

    def test_modifying_protected_test_fails_integrity(self, tmp_path, monkeypatch):
        calls = 0

        def fake_run(command, *, cwd: Path, timeout, env=None):
            nonlocal calls
            calls += 1
            if calls == 1:
                (cwd / "SPEC.md").write_text("cheated\n")
                return _completed(command, stdout='{"success": true}')
            return _completed(command, stdout="ok\n")

        monkeypatch.setattr(benchmark, "_run_process", fake_run)
        result = benchmark.run_case(
            _case(protected=("SPEC.md",)),
            tmp_path / "workspace",
            model=None,
            agent_timeout=60,
        )

        assert not result.integrity_success
        assert not result.passed
        assert "protected" in (result.error or "")

    def test_generated_cache_files_are_not_reported_as_changes(self, tmp_path, monkeypatch):
        calls = 0

        def fake_run(command, *, cwd: Path, timeout, env=None):
            nonlocal calls
            calls += 1
            if calls == 1:
                (cwd / "result.py").write_text("print('ok')\n")
                cache = cwd / ".ruff_cache"
                cache.mkdir()
                (cache / "CACHEDIR.TAG").write_text("cache\n")
                return _completed(command, stdout='{"success": true}')
            return _completed(command, stdout="ok\n")

        monkeypatch.setattr(benchmark, "_run_process", fake_run)
        result = benchmark.run_case(
            _case(), tmp_path / "workspace", model=None, agent_timeout=60
        )

        assert result.changed_files == ["result.py"]

    def test_timeout_still_runs_external_verifier(self, tmp_path, monkeypatch):
        calls = 0

        def fake_run(command, *, cwd: Path, timeout, env=None):
            nonlocal calls
            calls += 1
            if calls == 1:
                (cwd / "result.py").write_text("print('ok')\n")
                raise subprocess.TimeoutExpired(command, timeout)
            return _completed(command, stdout="ok\n")

        monkeypatch.setattr(benchmark, "_run_process", fake_run)
        result = benchmark.run_case(
            _case(), tmp_path / "workspace", model=None, agent_timeout=15
        )

        assert not result.agent_success
        assert result.verifier_success
        assert not result.passed
        assert "exceeded the 15s" in (result.error or "")


def test_list_command_does_not_run_agent(capsys):
    assert benchmark.main(["--list"]) == 0
    output = capsys.readouterr().out
    assert "hello_world" in output
    assert "inventory_refactor" in output


def _init_source_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "source_repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "a@b.c"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "tester"], cwd=repo, check=True)
    (repo / "existing.py").write_text("x = 1\n")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=repo, check=True)
    return repo


def _review_case(**overrides) -> BenchmarkCase:
    defaults = dict(
        case_id="review_case",
        level="review",
        title="Review case",
        task="Review the code and write REVIEW.md. Do not modify anything else.",
        seed_files={},
        verification=None,
        protect_all_existing_files=True,
        workspace_kind="worktree",
    )
    defaults.update(overrides)
    return BenchmarkCase(**defaults)


def _planner_case(**overrides) -> BenchmarkCase:
    defaults = dict(
        case_id="planner_case",
        level="frontier",
        title="Planner case",
        task="Let's add a web interface to this project",
        seed_files={"README.md": "# demo\n"},
        verification=None,
        protect_all_existing_files=True,
        runner_kind="planner",
    )
    defaults.update(overrides)
    return BenchmarkCase(**defaults)


class TestWorktreeCase:
    def test_worktree_is_checked_out_from_source_repo(self, tmp_path, monkeypatch):
        source_repo = _init_source_repo(tmp_path)

        def fake_run(command, *, cwd, timeout, env=None):
            assert (cwd / "existing.py").read_text() == "x = 1\n"
            return _completed(command, stdout='{"success": true, "message": "found nothing"}')

        monkeypatch.setattr(benchmark, "_run_process", fake_run)
        result = benchmark.run_case(
            _review_case(),
            tmp_path / "workspace",
            model=None,
            agent_timeout=60,
            source_repo=source_repo,
        )

        assert result.agent_success
        assert result.verifier_success
        assert result.passed
        assert result.agent_message == "found nothing"
        benchmark._prune_worktrees(source_repo)

    def test_no_external_verifier_is_invoked_when_verification_is_none(self, tmp_path, monkeypatch):
        source_repo = _init_source_repo(tmp_path)
        calls: list[list[str]] = []

        def fake_run(command, *, cwd, timeout, env=None):
            calls.append(list(command))
            return _completed(command, stdout='{"success": true}')

        monkeypatch.setattr(benchmark, "_run_process", fake_run)
        benchmark.run_case(
            _review_case(),
            tmp_path / "workspace",
            model=None,
            agent_timeout=60,
            source_repo=source_repo,
        )

        assert len(calls) == 1
        benchmark._prune_worktrees(source_repo)

    def test_new_report_file_does_not_break_integrity(self, tmp_path, monkeypatch):
        source_repo = _init_source_repo(tmp_path)

        def fake_run(command, *, cwd, timeout, env=None):
            (cwd / "REVIEW.md").write_text("- nothing found\n")
            return _completed(command, stdout='{"success": true}')

        monkeypatch.setattr(benchmark, "_run_process", fake_run)
        result = benchmark.run_case(
            _review_case(),
            tmp_path / "workspace",
            model=None,
            agent_timeout=60,
            source_repo=source_repo,
        )

        assert result.integrity_success
        assert result.changed_files == ["REVIEW.md"]
        benchmark._prune_worktrees(source_repo)

    def test_modifying_an_existing_file_fails_integrity(self, tmp_path, monkeypatch):
        source_repo = _init_source_repo(tmp_path)

        def fake_run(command, *, cwd, timeout, env=None):
            (cwd / "existing.py").write_text("x = 2\n")
            return _completed(command, stdout='{"success": true}')

        monkeypatch.setattr(benchmark, "_run_process", fake_run)
        result = benchmark.run_case(
            _review_case(),
            tmp_path / "workspace",
            model=None,
            agent_timeout=60,
            source_repo=source_repo,
        )

        assert not result.integrity_success
        assert not result.passed
        benchmark._prune_worktrees(source_repo)

    def test_relative_workspace_resolves_against_process_cwd_not_source_repo(self, tmp_path, monkeypatch):
        source_repo = _init_source_repo(tmp_path)
        monkeypatch.chdir(tmp_path)

        def fake_run(command, *, cwd, timeout, env=None):
            assert (cwd / "existing.py").read_text() == "x = 1\n"
            return _completed(command, stdout='{"success": true, "message": "found nothing"}')

        monkeypatch.setattr(benchmark, "_run_process", fake_run)
        relative_workspace = Path("nested") / "workspace"
        result = benchmark.run_case(
            _review_case(),
            relative_workspace,
            model=None,
            agent_timeout=60,
            source_repo=source_repo,
        )

        assert result.passed
        assert (tmp_path / "nested" / "workspace" / "existing.py").exists()
        assert not (source_repo / "nested").exists()
        benchmark._prune_worktrees(source_repo)

    def test_worktree_checks_out_the_requested_ref(self, tmp_path, monkeypatch):
        source_repo = _init_source_repo(tmp_path)
        (source_repo / "existing.py").write_text("x = 2\n")
        subprocess.run(["git", "commit", "-q", "-am", "second"], cwd=source_repo, check=True)
        first_commit = subprocess.run(
            ["git", "rev-list", "--max-parents=0", "HEAD"],
            cwd=source_repo,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()

        seen = {}

        def fake_run(command, *, cwd, timeout, env=None):
            seen["content"] = (cwd / "existing.py").read_text()
            return _completed(command, stdout='{"success": true}')

        monkeypatch.setattr(benchmark, "_run_process", fake_run)
        benchmark.run_case(
            _review_case(worktree_ref=first_commit),
            tmp_path / "workspace",
            model=None,
            agent_timeout=60,
            source_repo=source_repo,
        )

        assert seen["content"] == "x = 1\n"
        benchmark._prune_worktrees(source_repo)


class TestPlannerCase:
    def test_prepare_agent_path_enables_think_mode(self):
        env = benchmark._prepare_agent_path()

        assert env["CHEF_OLLAMA_THINK"] == "true"

    def test_plan_from_data_preserves_children_and_checkpoint_type(self):
        plan = benchmark._plan_from_data(
            {
                "goal": "g",
                "steps": [
                    {
                        "index": 6,
                        "description": "Explore the repo first",
                        "status": "completed",
                        "type": "checkpoint",
                        "children": [
                            {
                                "index": 1,
                                "description": "Read the README",
                                "status": "completed",
                                "type": "leaf",
                            }
                        ],
                    }
                ],
            }
        )
        node = plan.steps[0]

        assert node.declared_type == "checkpoint"
        assert node.children[0].parent is node
        assert node.children[0].description == "Read the README"

    def test_plan_from_data_loads_old_payload_without_node_id(self):
        """A replay `planner_state` written before the invalidation feature
        has no `node_id` on any step (and no `archived_subtrees` key): the
        benchmark loader must still build an executable tree for it."""
        plan = benchmark._plan_from_data(
            {
                "goal": "g",
                "steps": [
                    {
                        "index": 1,
                        "description": "Read the README",
                        "status": "completed",
                        "type": "leaf",
                    }
                ],
            }
        )
        node = plan.steps[0]

        assert node.status is StepStatus.completed
        assert node.node_id
        assert node.children == []

    def test_plan_from_data_round_trips_invalidation_metadata(self):
        """When the replay state does carry the new metadata, the benchmark
        loader round-trips it: `node_id` on the steps, the new `invalidated`
        status, and the optional `archived_subtrees` off-tree field."""
        branch = PlanNode(index=1, description="Build the transport layer",
                          status=StepStatus.invalidated, node_id="d" * 32)
        discarded = [
            PlanNode(index=1, description="Stale child", status=StepStatus.completed,
                     node_id="e" * 32),
        ]
        plan = Plan(goal="g", steps=[branch])
        plan.archive_subtree(branch, discarded, reason="r", evidence_summary="e")
        payload = plan.to_dict()
        assert "archived_subtrees" in payload  # the field under test is present

        reloaded = benchmark._plan_from_data(payload)

        assert reloaded.goal == "g"
        assert reloaded.steps[0].node_id == branch.node_id
        assert reloaded.steps[0].status is StepStatus.invalidated
        assert reloaded.archived_nodes[branch.node_id].children[0].node_id == discarded[0].node_id
        assert reloaded.archived_nodes[branch.node_id].reason == "r"
        # Round-trip through the benchmark's own loader is lossless.
        assert reloaded.to_dict() == payload

    def test_planner_runner_returns_plan_and_llm_trace(self, tmp_path, monkeypatch):
        captured = {}

        def fake_run_planner_case(case, workspace, *, model, timeout_seconds):
            captured["timeout_seconds"] = timeout_seconds
            captured["workspace"] = workspace
            return {
                "success": True,
                "steps_taken": 2,
                "total_prompt_tokens": 11,
                "total_completion_tokens": 7,
                "message": "Planner produced 2 top-level step(s)",
                "planner_plan": {
                    "goal": case.task,
                    "steps": [
                        {
                            "index": 1,
                            "description": "Explore",
                            "status": "pending",
                            "type": "checkpoint",
                        }
                    ],
                },
                "planner_llm_calls": [
                    {
                        "activity": "planning",
                        "request": {"messages": [{"role": "user", "content": case.task}]},
                        "response": {"content": "[]", "thinking": "first think", "usage": None},
                    }
                ],
                "planner_trace_file": str(workspace / ".chef-human" / "planner-trace.json"),
            }

        monkeypatch.setattr(benchmark, "_run_planner_case", fake_run_planner_case)
        result = benchmark.run_case(
            _planner_case(),
            tmp_path / "workspace",
            model="test-model",
            agent_timeout=45,
        )

        assert result.passed
        assert result.agent_success
        assert result.steps_taken == 2
        assert result.planner_plan == {
            "goal": "Let's add a web interface to this project",
            "steps": [
                {
                    "index": 1,
                    "description": "Explore",
                    "status": "pending",
                    "type": "checkpoint",
                }
            ],
        }
        assert result.planner_llm_calls[0]["response"]["thinking"] == "first think"
        assert result.planner_trace_file is not None
        assert captured["timeout_seconds"] == 45

    def test_replay_benchmark_case_is_registered(self):
        case = next(c for c in CASES if c.case_id == "vague_feature_request_checkpoint_replay")

        assert case.runner_kind == "planner"
        assert case.planner_operation == "continue_from_checkpoint"
        assert case.planner_state is not None

    def test_planner_timeout_still_writes_trace_file(self, tmp_path, monkeypatch):
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        (workspace / ".chef-human").mkdir()

        class FakePlanner:
            def __init__(self):
                self._complete = None

        class FakeLoop:
            def __init__(self):
                self._planner = FakePlanner()
                self._total_prompt_tokens = 3
                self._total_completion_tokens = 4

            async def _plan_task(self, task):
                await self._planner._complete(
                    types.SimpleNamespace(
                        messages=[types.SimpleNamespace(role="user", content=task)],
                        temperature=0.0,
                        max_tokens=10,
                    ),
                    activity="planning",
                )
                raise TimeoutError("planner timed out")

        async def fake_complete(request, activity="planning"):
            return types.SimpleNamespace(
                message=types.SimpleNamespace(content="[]"),
                thinking="partial reasoning",
                usage={"prompt_tokens": 1, "completion_tokens": 2},
            )

        fake_loop = FakeLoop()
        fake_loop._planner._complete = fake_complete

        monkeypatch.setattr(
            "chef_human.agent.create_agent",
            lambda max_steps, workspace_root, settings: (fake_loop, None),
        )

        with pytest.raises(TimeoutError):
            benchmark._run_planner_case(
                _planner_case(),
                workspace,
                model=None,
                timeout_seconds=1,
            )

        trace_path = workspace / ".chef-human" / "planner-trace.json"
        assert trace_path.exists()
        payload = json.loads(trace_path.read_text())
        assert payload["plan"] is None
        assert payload["llm_calls"][0]["response"]["thinking"] == "partial reasoning"
        assert payload["tree_snapshots"] == []

    def test_checkpoint_replay_timeout_writes_seed_tree_snapshot(self, tmp_path, monkeypatch):
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        (workspace / ".chef-human").mkdir()

        class FakePlanner:
            def __init__(self):
                self._complete = None

            async def continue_from_checkpoint(self, plan, checkpoint, evidence):
                raise TimeoutError("checkpoint continuation timed out")

            async def expand_spliced_steps(self, plan, steps):
                raise AssertionError("expand_spliced_steps should not run")

        class FakeLoop:
            def __init__(self):
                self._planner = FakePlanner()
                self._total_prompt_tokens = 0
                self._total_completion_tokens = 0

        monkeypatch.setattr(
            "chef_human.agent.create_agent",
            lambda max_steps, workspace_root, settings: (FakeLoop(), None),
        )

        case = next(c for c in CASES if c.case_id == "vague_feature_request_checkpoint_replay")

        with pytest.raises(TimeoutError):
            benchmark._run_planner_case(
                case,
                workspace,
                model=None,
                timeout_seconds=1,
            )

        payload = json.loads((workspace / ".chef-human" / "planner-trace.json").read_text())
        assert payload["plan"]["steps"][5]["type"] == "checkpoint"
        assert payload["tree_snapshots"][0]["phase"] == "seeded_checkpoint_state"
        assert payload["tree_snapshots"][0]["plan"]["steps"][5]["type"] == "checkpoint"

    def test_replay_state_with_new_metadata_round_trips(self, tmp_path, monkeypatch):
        """Ticket 06, benchmark/replay seam: when a replay case's
        `planner_state` carries the new invalidation metadata (`node_id` on
        steps, an `archived_subtrees` block), the seeded plan restores both
        losslessly -- node ids survive so the off-tree archive stays keyed
        to the live nodes, and the archive itself (structure, reason,
        evidence) comes back. A state without the metadata still loads
        (the pre-feature replay case above proves that)."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        (workspace / ".chef-human").mkdir()

        branch_id = "b" * 32
        child_id = "c" * 32
        checkpoint_id = "k" * 32

        class FakePlanner:
            def __init__(self):
                self._complete = None

            async def continue_from_checkpoint(self, plan, checkpoint, evidence):
                return []

            async def expand_spliced_steps(self, plan, steps):
                return None

        class FakeLoop:
            def __init__(self):
                self._planner = FakePlanner()
                self._total_prompt_tokens = 0
                self._total_completion_tokens = 0

        monkeypatch.setattr(
            "chef_human.agent.create_agent",
            lambda max_steps, workspace_root, settings: (FakeLoop(), None),
        )

        case = BenchmarkCase(
            case_id="replay_with_metadata",
            level="smoke",
            title="Replay state carrying invalidation metadata",
            task="Build the transport layer",
            seed_files={},
            runner_kind="planner",
            planner_operation="continue_from_checkpoint",
            planner_state={
                "steps": [
                    {"index": 1, "description": "Read the README", "status": "completed",
                     "type": "leaf"},
                    {"index": 2, "description": "Stale framing plan", "status": "invalidated",
                     "type": "branch", "node_id": branch_id},
                    {"index": 3, "description": "Explore checkpoint", "status": "completed",
                     "type": "checkpoint", "node_id": checkpoint_id},
                ],
                "checkpoint_index": 2,
                "evidence": "framing assumption disproved",
                "archived_subtrees": {
                    branch_id: {
                        "node_id": branch_id,
                        "reason": "tcp framing disproved",
                        "evidence_summary": "protocol.py:42",
                        "children": [
                            {"index": 1, "description": "Write the TCP framing",
                             "status": "failed", "type": "leaf", "node_id": child_id},
                        ],
                    }
                },
            },
        )

        result = benchmark._run_planner_case(
            case, workspace, model=None, timeout_seconds=5
        )

        seeded = result["planner_plan"]
        # Node ids round-tripped (the checkpoint and the stale node kept
        # their exact identities; the legacy step got a fresh one).
        by_desc = {s["description"]: s for s in seeded["steps"]}
        assert by_desc["Stale framing plan"]["node_id"] == branch_id
        assert by_desc["Explore checkpoint"]["node_id"] == checkpoint_id
        assert by_desc["Read the README"]["node_id"]
        assert by_desc["Stale framing plan"]["status"] == "invalidated"
        # The off-tree archive came back keyed to the restored node id.
        entry = seeded["archived_subtrees"][branch_id]
        assert entry["reason"] == "tcp framing disproved"
        assert entry["evidence_summary"] == "protocol.py:42"
        assert entry["children"][0]["node_id"] == child_id
        assert entry["children"][0]["status"] == "failed"
        # The continuation was seeded from the loaded plan, not a fresh one:
        # the plan object the planner continued from carries the archive.
        trace = json.loads((workspace / ".chef-human" / "planner-trace.json").read_text())
        seeded_snapshot = trace["tree_snapshots"][0]["plan"]
        assert seeded_snapshot["archived_subtrees"][branch_id]["evidence_summary"] == "protocol.py:42"
