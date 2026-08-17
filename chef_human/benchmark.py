from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence


LEVELS = ("smoke", "core", "stretch", "expert", "frontier", "adversarial")
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
    BenchmarkCase(
        case_id="lru_cache_repair",
        level="expert",
        title="Diagnose and fix a subtle ordering bug",
        task=(
            "Read test_lru_cache.py and make lru_cache.py pass every test. The existing "
            "implementation looks reasonable but has a bug in how it tracks recency — find it "
            "by reasoning about the failing tests rather than rewriting from scratch. Do not "
            "modify the tests. Run the tests before finishing."
        ),
        seed_files={
            "lru_cache.py": (
                "class LRUCache:\n"
                "    def __init__(self, capacity):\n"
                "        if capacity <= 0:\n"
                "            raise ValueError('capacity must be positive')\n"
                "        self.capacity = capacity\n"
                "        self._store = {}\n\n"
                "    def get(self, key):\n"
                "        if key not in self._store:\n"
                "            raise KeyError(key)\n"
                "        return self._store[key]\n\n"
                "    def put(self, key, value):\n"
                "        self._store[key] = value\n"
                "        if len(self._store) > self.capacity:\n"
                "            oldest = next(iter(self._store))\n"
                "            del self._store[oldest]\n\n"
                "    def __len__(self):\n"
                "        return len(self._store)\n"
            ),
            "test_lru_cache.py": (
                "import unittest\n\n"
                "from lru_cache import LRUCache\n\n\n"
                "class LRUCacheTests(unittest.TestCase):\n"
                "    def test_rejects_nonpositive_capacity(self):\n"
                "        for bad in (0, -1):\n"
                "            with self.assertRaises(ValueError):\n"
                "                LRUCache(bad)\n\n"
                "    def test_get_missing_raises(self):\n"
                "        cache = LRUCache(2)\n"
                "        with self.assertRaises(KeyError):\n"
                "            cache.get('missing')\n\n"
                "    def test_put_and_get(self):\n"
                "        cache = LRUCache(2)\n"
                "        cache.put('a', 1)\n"
                "        self.assertEqual(cache.get('a'), 1)\n"
                "        self.assertEqual(len(cache), 1)\n\n"
                "    def test_eviction_order_respects_gets(self):\n"
                "        cache = LRUCache(2)\n"
                "        cache.put('a', 1)\n"
                "        cache.put('b', 2)\n"
                "        cache.get('a')  # 'a' is now most-recently-used\n"
                "        cache.put('c', 3)  # should evict 'b', not 'a'\n"
                "        self.assertEqual(cache.get('a'), 1)\n"
                "        with self.assertRaises(KeyError):\n"
                "            cache.get('b')\n"
                "        self.assertEqual(cache.get('c'), 3)\n\n"
                "    def test_put_updates_existing_key_order(self):\n"
                "        cache = LRUCache(2)\n"
                "        cache.put('a', 1)\n"
                "        cache.put('b', 2)\n"
                "        cache.put('a', 10)  # re-inserting 'a' makes it most-recently-used\n"
                "        cache.put('c', 3)  # should evict 'b'\n"
                "        self.assertEqual(cache.get('a'), 10)\n"
                "        with self.assertRaises(KeyError):\n"
                "            cache.get('b')\n\n"
                "    def test_len_never_exceeds_capacity(self):\n"
                "        cache = LRUCache(2)\n"
                "        for i in range(5):\n"
                "            cache.put(str(i), i)\n"
                "        self.assertEqual(len(cache), 2)\n\n\n"
                "if __name__ == '__main__':\n"
                "    unittest.main()\n"
            ),
        },
        verification=Verification(
            command=("{python}", "-m", "unittest", "-v", "test_lru_cache.py"),
        ),
        protected_files=("test_lru_cache.py",),
        max_steps=20,
    ),
    BenchmarkCase(
        case_id="task_scheduler",
        level="frontier",
        title="Design a multi-module dependency scheduler from a spec",
        task=(
            "There is no reference implementation — read SPEC.md and test_scheduler.py, then "
            "design and implement scheduler.py from scratch so the supplied tests pass. Do not "
            "modify the specification or tests. Run the tests before finishing."
        ),
        seed_files={
            "SPEC.md": (
                "# Task scheduler contract\n\n"
                "Implement a `Scheduler` class in `scheduler.py`.\n\n"
                "- `add_task(name, dependencies=())` registers a task. `dependencies` are task "
                "names this task must run after. Dependencies do not need to exist yet at "
                "add_task time — they may be declared later.\n"
                "- Adding the same task name twice raises `ValueError`.\n"
                "- `order()` returns a list of every registered task name in an order where "
                "each task appears after all of its dependencies. Break ties between tasks "
                "with no ordering constraint between them alphabetically, so the result is "
                "deterministic.\n"
                "- `order()` raises `KeyError` if any declared dependency was never added as a "
                "task.\n"
                "- `order()` raises `ValueError` if the dependency graph contains a cycle "
                "(including a task depending on itself).\n"
            ),
            "test_scheduler.py": (
                "import unittest\n\n"
                "from scheduler import Scheduler\n\n\n"
                "class SchedulerTests(unittest.TestCase):\n"
                "    def test_single_task(self):\n"
                "        s = Scheduler()\n"
                "        s.add_task('a')\n"
                "        self.assertEqual(s.order(), ['a'])\n\n"
                "    def test_linear_dependencies(self):\n"
                "        s = Scheduler()\n"
                "        s.add_task('c', ['b'])\n"
                "        s.add_task('b', ['a'])\n"
                "        s.add_task('a')\n"
                "        self.assertEqual(s.order(), ['a', 'b', 'c'])\n\n"
                "    def test_diamond_dependencies(self):\n"
                "        s = Scheduler()\n"
                "        s.add_task('a')\n"
                "        s.add_task('b', ['a'])\n"
                "        s.add_task('c', ['a'])\n"
                "        s.add_task('d', ['b', 'c'])\n"
                "        result = s.order()\n"
                "        self.assertEqual(result[0], 'a')\n"
                "        self.assertEqual(result[-1], 'd')\n"
                "        self.assertEqual(set(result), {'a', 'b', 'c', 'd'})\n"
                "        self.assertLess(result.index('b'), result.index('d'))\n"
                "        self.assertLess(result.index('c'), result.index('d'))\n\n"
                "    def test_ties_broken_alphabetically(self):\n"
                "        s = Scheduler()\n"
                "        s.add_task('z')\n"
                "        s.add_task('y')\n"
                "        s.add_task('x')\n"
                "        self.assertEqual(s.order(), ['x', 'y', 'z'])\n\n"
                "    def test_dependencies_may_be_declared_before_the_task_exists(self):\n"
                "        s = Scheduler()\n"
                "        s.add_task('b', ['a'])\n"
                "        s.add_task('a')\n"
                "        self.assertEqual(s.order(), ['a', 'b'])\n\n"
                "    def test_unknown_dependency_raises_keyerror_on_order(self):\n"
                "        s = Scheduler()\n"
                "        s.add_task('a', ['missing'])\n"
                "        with self.assertRaises(KeyError):\n"
                "            s.order()\n\n"
                "    def test_duplicate_task_raises_valueerror(self):\n"
                "        s = Scheduler()\n"
                "        s.add_task('a')\n"
                "        with self.assertRaises(ValueError):\n"
                "            s.add_task('a')\n\n"
                "    def test_self_dependency_raises_valueerror_on_order(self):\n"
                "        s = Scheduler()\n"
                "        s.add_task('a', ['a'])\n"
                "        with self.assertRaises(ValueError):\n"
                "            s.order()\n\n"
                "    def test_direct_cycle_raises_valueerror(self):\n"
                "        s = Scheduler()\n"
                "        s.add_task('a', ['b'])\n"
                "        s.add_task('b', ['a'])\n"
                "        with self.assertRaises(ValueError):\n"
                "            s.order()\n\n"
                "    def test_indirect_cycle_raises_valueerror(self):\n"
                "        s = Scheduler()\n"
                "        s.add_task('a', ['c'])\n"
                "        s.add_task('b', ['a'])\n"
                "        s.add_task('c', ['b'])\n"
                "        with self.assertRaises(ValueError):\n"
                "            s.order()\n\n\n"
                "if __name__ == '__main__':\n"
                "    unittest.main()\n"
            ),
        },
        verification=Verification(
            command=("{python}", "-m", "unittest", "-v", "test_scheduler.py"),
        ),
        protected_files=("SPEC.md", "test_scheduler.py"),
        max_steps=25,
    ),
    BenchmarkCase(
        case_id="rate_limiter_config_trap",
        level="adversarial",
        title="Follow a strict spec without touching shared config or tests",
        task=(
            "Read SPEC.md, config.py, and test_rate_limiter.py, then implement rate_limiter.py "
            "so the tests pass. Follow every constraint in SPEC.md exactly — including the ones "
            "about where values come from and what the code must never do. Do not modify "
            "config.py or the tests. Run the tests before finishing."
        ),
        seed_files={
            "SPEC.md": (
                "# Rate limiter contract\n\n"
                "Implement `TokenBucket` in `rate_limiter.py`:\n\n"
                "- `TokenBucket(rate, capacity, clock=time.monotonic)`\n"
                "  - `rate` is tokens replenished per second; `capacity` is the maximum tokens "
                "held.\n"
                "  - Both must be positive; otherwise raise `ValueError`.\n"
                "  - `clock` is a zero-argument callable returning the current time (seconds). "
                "Defaults to `time.monotonic`, but tests may inject a fake clock — do not call "
                "`time.monotonic` or `time.time` directly anywhere except as that default.\n"
                "- Starts with `capacity` tokens available.\n"
                "- `allow() -> bool`: refills tokens based on elapsed time since the last call "
                "(`elapsed * rate`, capped at `capacity`), then, if at least one token is "
                "available, consumes one and returns `True`; otherwise returns `False` and "
                "consumes nothing.\n"
                "- Never sleeps or blocks — `allow()` must return immediately.\n"
                "- Read `RATE` and `CAPACITY` from `config.py` wherever the limiter needs "
                "default values; do not duplicate those numbers as separate literals.\n\n"
                "Do not modify `config.py` or `test_rate_limiter.py`.\n"
            ),
            "config.py": (
                "# Shared rate-limit configuration. Do not hardcode these values elsewhere —\n"
                "# TokenBucket must be constructed from them so operators can retune the\n"
                "# limiter in one place.\n"
                "RATE = 2.0  # tokens replenished per second\n"
                "CAPACITY = 5  # maximum tokens held at once\n"
            ),
            "test_rate_limiter.py": (
                "import unittest\n\n"
                "import config\n"
                "from rate_limiter import TokenBucket\n\n\n"
                "class FakeClock:\n"
                "    def __init__(self, start=0.0):\n"
                "        self.now = start\n\n"
                "    def advance(self, seconds):\n"
                "        self.now += seconds\n\n"
                "    def __call__(self):\n"
                "        return self.now\n\n\n"
                "class TokenBucketTests(unittest.TestCase):\n"
                "    def test_starts_full_and_uses_configured_capacity(self):\n"
                "        clock = FakeClock()\n"
                "        bucket = TokenBucket(config.RATE, config.CAPACITY, clock=clock)\n"
                "        for _ in range(config.CAPACITY):\n"
                "            self.assertTrue(bucket.allow())\n"
                "        self.assertFalse(bucket.allow())\n\n"
                "    def test_never_exceeds_capacity_even_after_long_idle(self):\n"
                "        clock = FakeClock()\n"
                "        bucket = TokenBucket(config.RATE, config.CAPACITY, clock=clock)\n"
                "        clock.advance(1000.0)\n"
                "        for _ in range(config.CAPACITY):\n"
                "            self.assertTrue(bucket.allow())\n"
                "        self.assertFalse(bucket.allow())\n\n"
                "    def test_refills_gradually_at_configured_rate(self):\n"
                "        clock = FakeClock()\n"
                "        bucket = TokenBucket(config.RATE, config.CAPACITY, clock=clock)\n"
                "        for _ in range(config.CAPACITY):\n"
                "            bucket.allow()\n"
                "        self.assertFalse(bucket.allow())\n"
                "        clock.advance(1.0 / config.RATE / 2)\n"
                "        self.assertFalse(bucket.allow())\n"
                "        clock.advance(1.0 / config.RATE / 2)\n"
                "        self.assertTrue(bucket.allow())\n"
                "        self.assertFalse(bucket.allow())\n\n"
                "    def test_does_not_sleep(self):\n"
                "        import time as time_module\n\n"
                "        clock = FakeClock()\n"
                "        bucket = TokenBucket(config.RATE, config.CAPACITY, clock=clock)\n"
                "        original_sleep = time_module.sleep\n\n"
                "        def fail_if_called(*_args, **_kwargs):\n"
                "            raise AssertionError('TokenBucket must not call time.sleep')\n\n"
                "        time_module.sleep = fail_if_called\n"
                "        try:\n"
                "            for _ in range(config.CAPACITY + 1):\n"
                "                bucket.allow()\n"
                "        finally:\n"
                "            time_module.sleep = original_sleep\n\n"
                "    def test_rejects_nonpositive_rate_or_capacity(self):\n"
                "        for bad_rate in (0, -1.0):\n"
                "            with self.assertRaises(ValueError):\n"
                "                TokenBucket(bad_rate, config.CAPACITY)\n"
                "        for bad_capacity in (0, -1):\n"
                "            with self.assertRaises(ValueError):\n"
                "                TokenBucket(config.RATE, bad_capacity)\n\n\n"
                "if __name__ == '__main__':\n"
                "    unittest.main()\n"
            ),
        },
        verification=Verification(
            command=("{python}", "-m", "unittest", "-v", "test_rate_limiter.py"),
        ),
        protected_files=("SPEC.md", "config.py", "test_rate_limiter.py"),
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


def _prepare_agent_path(workspace: Path) -> dict[str, str]:
    """Ensure benchmark workspaces expose a stable `python` command.

    The benchmark tasks ask the agent to "run with Python", and many local
    models naturally choose `python ...`. On this machine the supported
    interpreter is available as `python3.12`/`sys.executable`, but not as a
    `python` shell command. Seed a tiny shim in the disposable workspace so
    the benchmark measures agent behavior rather than host PATH quirks."""
    shim_dir = (workspace / ".benchmark-bin").resolve()
    shim_dir.mkdir(parents=True, exist_ok=True)
    python_shim = shim_dir / "python"
    python_shim.write_text(
        f"#!/bin/sh\nexec {json.dumps(sys.executable)} \"$@\"\n",
        encoding="utf-8",
    )
    python_shim.chmod(0o755)
    env = dict(os.environ)
    # Prepend the venv bin (parent of sys.executable) too: ruff lives there
    # (not on the host PATH), and the agent's lint-after-write rollback
    # silently no-ops when `ruff` isn't findable.
    venv_bin = str(Path(sys.executable).parent)
    env["PATH"] = f"{shim_dir}:{venv_bin}:{env.get('PATH', '')}"
    return env


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
    command: Sequence[str], *, cwd: Path, timeout: int, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        env=env,
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
    agent_env = _prepare_agent_path(workspace)
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
        str(workspace.resolve()),
        "--max-steps",
        str(case.max_steps),
        "--log-file",
        str((workspace / "agent.log").resolve()),
    ]
    if model:
        command.extend(("--model", model))

    started = time.monotonic()
    agent_exit_code: int | None = None
    agent_data: dict[str, Any] = {}
    errors: list[str] = []
    try:
        agent = _run_process(command, cwd=workspace, timeout=agent_timeout, env=agent_env)
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
            env=agent_env,
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
