from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence


LEVELS = ("smoke", "core", "stretch")
IGNORED_SNAPSHOT_PARTS = {
    ".chef-human",
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
}


@dataclass(frozen=True)
class Verification:
    command: tuple[str, ...]
    expected_stdout: str | None = None
    timeout_seconds: int = 30


@dataclass(frozen=True)
class BenchmarkCase:
    case_id: str
    level: str
    title: str
    task: str
    seed_files: dict[str, str]
    verification: Verification
    protected_files: tuple[str, ...] = ()
    max_steps: int = 15


@dataclass
class BenchmarkResult:
    case_id: str
    level: str
    title: str
    passed: bool
    agent_success: bool
    verifier_success: bool
    integrity_success: bool
    agent_exit_code: int | None
    duration_seconds: float
    steps_taken: int | None
    prompt_tokens: int | None
    completion_tokens: int | None
    changed_files: list[str]
    verifier_output: str
    error: str | None = None
    workspace: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


CASES: tuple[BenchmarkCase, ...] = (
    BenchmarkCase(
        case_id="hello_world",
        level="smoke",
        title="Create and run one file",
        task=(
            "Create hello.py. When run with Python it must print exactly "
            "Hello, world! followed by a newline. Run it to verify the output, then finish."
        ),
        seed_files={},
        verification=Verification(
            command=("{python}", "hello.py"),
            expected_stdout="Hello, world!\n",
        ),
        max_steps=10,
    ),
    BenchmarkCase(
        case_id="slugify_contract",
        level="core",
        title="Implement a tested contract",
        task=(
            "Read SPEC.md and test_slugify.py, then implement slugify.py so the supplied tests "
            "pass. Do not modify the specification or tests. Run the tests before finishing."
        ),
        seed_files={
            "SPEC.md": (
                "# Slugify contract\n\n"
                "Implement `slugify(value: str) -> str` in `slugify.py`.\n\n"
                "- Trim surrounding whitespace and lowercase the value.\n"
                "- Replace each run of non-alphanumeric characters with one hyphen.\n"
                "- Remove leading and trailing hyphens.\n"
                "- Raise `ValueError` when the result would be empty.\n"
            ),
            "test_slugify.py": (
                "import unittest\n\n"
                "from slugify import slugify\n\n\n"
                "class SlugifyTests(unittest.TestCase):\n"
                "    def test_words_and_punctuation(self):\n"
                "        self.assertEqual(slugify('  Hello, Local Agent!  '), 'hello-local-agent')\n\n"
                "    def test_collapses_separators(self):\n"
                "        self.assertEqual(slugify('one___two...three'), 'one-two-three')\n\n"
                "    def test_keeps_digits(self):\n"
                "        self.assertEqual(slugify('Version 2'), 'version-2')\n\n"
                "    def test_rejects_empty_result(self):\n"
                "        with self.assertRaises(ValueError):\n"
                "            slugify('---')\n\n\n"
                "if __name__ == '__main__':\n"
                "    unittest.main()\n"
            ),
        },
        verification=Verification(
            command=("{python}", "-m", "unittest", "-v", "test_slugify.py"),
        ),
        protected_files=("SPEC.md", "test_slugify.py"),
        max_steps=15,
    ),
    BenchmarkCase(
        case_id="inventory_refactor",
        level="stretch",
        title="Repair and extend a multi-file program",
        task=(
            "Implement every requirement in SPEC.md. Inspect the existing inventory.py and "
            "test_inventory.py, repair the inventory behavior, and add report.py. Do not modify "
            "the specification or tests. Run the full test suite before finishing."
        ),
        seed_files={
            "SPEC.md": (
                "# Inventory requirements\n\n"
                "1. `Inventory.add(name, quantity)` accumulates positive integer quantities.\n"
                "2. `Inventory.remove(name, quantity)` removes a positive integer quantity.\n"
                "3. Invalid quantities raise `ValueError`; unknown or insufficient stock raises `KeyError`.\n"
                "4. Items at zero stock disappear.\n"
                "5. `report.inventory_report(inventory)` returns sorted `name: quantity` lines, "
                "or `(empty)` when no stock remains.\n"
            ),
            "inventory.py": (
                "class Inventory:\n"
                "    def __init__(self):\n"
                "        self.stock = {}\n\n"
                "    def add(self, name, quantity):\n"
                "        self.stock[name] = quantity\n\n"
                "    def remove(self, name, quantity):\n"
                "        self.stock[name] -= quantity\n\n"
                "    def total_items(self):\n"
                "        return sum(self.stock.values())\n"
            ),
            "test_inventory.py": (
                "import unittest\n\n"
                "from inventory import Inventory\n"
                "from report import inventory_report\n\n\n"
                "class InventoryTests(unittest.TestCase):\n"
                "    def test_add_accumulates(self):\n"
                "        inv = Inventory()\n"
                "        inv.add('pear', 2)\n"
                "        inv.add('pear', 3)\n"
                "        self.assertEqual(inv.stock, {'pear': 5})\n\n"
                "    def test_remove_and_delete_zero(self):\n"
                "        inv = Inventory()\n"
                "        inv.add('pear', 2)\n"
                "        inv.remove('pear', 2)\n"
                "        self.assertEqual(inv.stock, {})\n\n"
                "    def test_validation(self):\n"
                "        inv = Inventory()\n"
                "        for bad in (0, -1, 1.5, True):\n"
                "            with self.assertRaises(ValueError):\n"
                "                inv.add('pear', bad)\n"
                "        with self.assertRaises(KeyError):\n"
                "            inv.remove('missing', 1)\n"
                "        inv.add('pear', 1)\n"
                "        with self.assertRaises(KeyError):\n"
                "            inv.remove('pear', 2)\n\n"
                "    def test_report_is_sorted(self):\n"
                "        inv = Inventory()\n"
                "        inv.add('pear', 2)\n"
                "        inv.add('apple', 1)\n"
                "        self.assertEqual(inventory_report(inv), 'apple: 1\\npear: 2')\n\n"
                "    def test_empty_report(self):\n"
                "        self.assertEqual(inventory_report(Inventory()), '(empty)')\n\n\n"
                "if __name__ == '__main__':\n"
                "    unittest.main()\n"
            ),
        },
        verification=Verification(
            command=("{python}", "-m", "unittest", "discover", "-v"),
        ),
        protected_files=("SPEC.md", "test_inventory.py"),
        max_steps=25,
    ),
)


def _digest(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _snapshot(workspace: Path) -> dict[str, str]:
    snapshot: dict[str, str] = {}
    for path in workspace.rglob("*"):
        relative = path.relative_to(workspace)
        if (
            not path.is_file()
            or any(part in IGNORED_SNAPSHOT_PARTS for part in relative.parts)
            or path.name == "agent.log"
        ):
            continue
        snapshot[str(relative)] = _digest(path.read_bytes())
    return snapshot


def _write_seed(workspace: Path, files: dict[str, str]) -> None:
    for relative, content in files.items():
        path = workspace / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")


def _parse_agent_json(stdout: str) -> dict[str, Any]:
    try:
        value = json.loads(stdout)
        return value if isinstance(value, dict) else {}
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        for index, char in enumerate(stdout):
            if char != "{":
                continue
            try:
                value, _ = decoder.raw_decode(stdout[index:])
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                return value
    return {}


def _run_process(
    command: Sequence[str], *, cwd: Path, timeout: int
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def run_case(
    case: BenchmarkCase,
    workspace: Path,
    *,
    model: str | None,
    agent_timeout: int,
) -> BenchmarkResult:
    workspace.mkdir(parents=True, exist_ok=False)
    _write_seed(workspace, case.seed_files)
    before = _snapshot(workspace)
    protected = {name: before[name] for name in case.protected_files}
    command = [
        sys.executable,
        "-m",
        "chef_human",
        "run",
        case.task,
        "--headless",
        "--no-stream",
        "--workspace",
        str(workspace),
        "--max-steps",
        str(case.max_steps),
        "--log-file",
        str(workspace / "agent.log"),
    ]
    if model:
        command.extend(("--model", model))

    started = time.monotonic()
    agent_exit_code: int | None = None
    agent_data: dict[str, Any] = {}
    errors: list[str] = []
    try:
        agent = _run_process(command, cwd=workspace, timeout=agent_timeout)
        agent_exit_code = agent.returncode
        agent_data = _parse_agent_json(agent.stdout)
        if not agent_data:
            errors.append("Agent did not emit a JSON result")
        if agent.returncode != 0:
            detail = agent.stderr.strip().splitlines()
            errors.append(
                f"Agent exited with {agent.returncode}"
                + (f": {detail[-1]}" if detail else "")
            )
    except subprocess.TimeoutExpired:
        errors.append(f"Agent exceeded the {agent_timeout}s case timeout")

    verify_command = [
        sys.executable if part == "{python}" else part
        for part in case.verification.command
    ]
    verifier_output = ""
    verifier_success = False
    try:
        verified = _run_process(
            verify_command,
            cwd=workspace,
            timeout=case.verification.timeout_seconds,
        )
        verifier_output = (verified.stdout + verified.stderr).strip()
        verifier_success = verified.returncode == 0
        if case.verification.expected_stdout is not None:
            verifier_success = (
                verifier_success
                and verified.stdout == case.verification.expected_stdout
            )
        if not verifier_success:
            errors.append("External verifier failed")
    except subprocess.TimeoutExpired:
        errors.append(
            f"Verifier exceeded the {case.verification.timeout_seconds}s timeout"
        )

    after = _snapshot(workspace)
    changed_files = sorted(
        name for name in set(before) | set(after) if before.get(name) != after.get(name)
    )
    integrity_success = all(after.get(name) == digest for name, digest in protected.items())
    if not integrity_success:
        errors.append("A protected specification or test file was modified")
    agent_success = agent_exit_code == 0 and agent_data.get("success") is True
    if agent_exit_code == 0 and agent_data and not agent_success:
        errors.append("Agent reported that the task failed")
    passed = agent_success and verifier_success and integrity_success
    return BenchmarkResult(
        case_id=case.case_id,
        level=case.level,
        title=case.title,
        passed=passed,
        agent_success=agent_success,
        verifier_success=verifier_success,
        integrity_success=integrity_success,
        agent_exit_code=agent_exit_code,
        duration_seconds=round(time.monotonic() - started, 3),
        steps_taken=agent_data.get("steps_taken"),
        prompt_tokens=agent_data.get("total_prompt_tokens"),
        completion_tokens=agent_data.get("total_completion_tokens"),
        changed_files=changed_files,
        verifier_output=verifier_output[-4000:],
        error="; ".join(errors) or None,
        workspace=str(workspace),
    )


def select_cases(through: str, case_ids: Sequence[str]) -> list[BenchmarkCase]:
    if case_ids:
        requested = set(case_ids)
        selected = [case for case in CASES if case.case_id in requested]
        missing = requested - {case.case_id for case in selected}
        if missing:
            raise ValueError(f"Unknown benchmark case(s): {', '.join(sorted(missing))}")
        return selected
    maximum = len(LEVELS) - 1 if through == "all" else LEVELS.index(through)
    return [case for case in CASES if LEVELS.index(case.level) <= maximum]


def _report(results: Sequence[BenchmarkResult]) -> dict[str, Any]:
    passed = sum(result.passed for result in results)
    return {
        "schema_version": 1,
        "passed": passed,
        "total": len(results),
        "score_percent": round(100 * passed / len(results), 1) if results else 0.0,
        "results": [result.to_dict() for result in results],
    }


def _print_human(report: dict[str, Any]) -> None:
    print("Chef Human capability benchmark")
    for result in report["results"]:
        mark = "PASS" if result["passed"] else "FAIL"
        print(
            f"[{mark}] {result['level']}/{result['case_id']} — "
            f"{result['duration_seconds']:.1f}s, steps={result['steps_taken'] or '?'}"
        )
        if result["error"]:
            print(f"       {result['error']}")
    print(
        f"Score: {report['passed']}/{report['total']} "
        f"({report['score_percent']:.1f}%)"
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run opt-in, real-model Chef Human capability benchmarks."
    )
    parser.add_argument("--through", choices=(*LEVELS, "all"), default="smoke")
    parser.add_argument("--case", action="append", default=[], dest="case_ids")
    parser.add_argument("--model", help="Override the configured model")
    parser.add_argument("--timeout", type=int, default=600, help="Seconds per agent case")
    parser.add_argument("--work-dir", type=Path, help="Keep case workspaces under this directory")
    parser.add_argument("--keep-workspaces", action="store_true")
    parser.add_argument("--json", action="store_true", dest="json_output")
    parser.add_argument("--output", type=Path, help="Write the JSON report to this path")
    parser.add_argument("--list", action="store_true", dest="list_cases")
    args = parser.parse_args(argv)

    if args.list_cases:
        for case in CASES:
            print(f"{case.level:7} {case.case_id:20} {case.title}")
        return 0

    try:
        cases = select_cases(args.through, args.case_ids)
    except ValueError as exc:
        parser.error(str(exc))

    temporary_root: Path | None = None
    if args.work_dir:
        root = args.work_dir.resolve()
        root.mkdir(parents=True, exist_ok=True)
    elif args.keep_workspaces:
        root = Path("benchmark-runs") / time.strftime("%Y%m%d-%H%M%S")
        root.mkdir(parents=True, exist_ok=False)
    else:
        temporary_root = Path(tempfile.mkdtemp(prefix="chef-human-benchmark-"))
        root = temporary_root

    results: list[BenchmarkResult] = []
    try:
        for case in cases:
            result = run_case(
                case,
                root / case.case_id,
                model=args.model,
                agent_timeout=args.timeout,
            )
            if temporary_root is not None:
                result.workspace = None
            results.append(result)
    finally:
        if temporary_root is not None:
            shutil.rmtree(temporary_root, ignore_errors=True)

    report = _report(results)
    rendered = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    if args.json_output:
        print(rendered)
    else:
        _print_human(report)
    return 0 if report["passed"] == report["total"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
