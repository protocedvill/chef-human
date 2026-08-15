from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from chef_human import benchmark
from chef_human.benchmark import CASES, BenchmarkCase, Verification


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


class TestRunCase:
    def test_pass_requires_agent_and_external_verifier(self, tmp_path, monkeypatch):
        calls: list[list[str]] = []

        def fake_run(command, *, cwd, timeout):
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
            lambda command, *, cwd, timeout: next(responses),
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

        def fake_run(command, *, cwd: Path, timeout):
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

        def fake_run(command, *, cwd: Path, timeout):
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

        def fake_run(command, *, cwd: Path, timeout):
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
