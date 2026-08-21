from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import pwd
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Literal, Sequence


LEVELS = (
    "smoke",
    "core",
    "stretch",
    "expert",
    "frontier",
    "adversarial",
    "marathon",
    "review",
)
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
    verification: Verification | None = None
    protected_files: tuple[str, ...] = ()
    max_steps: int = 15
    # "seed" (default) writes seed_files into an empty workspace. "worktree"
    # instead checks out a real git worktree of this repository — used for
    # benchmarks that exercise chef-human against its own, much larger,
    # codebase rather than a small synthetic project.
    workspace_kind: Literal["seed", "worktree"] = "seed"
    # "agent" runs the full ReAct loop via `chef_human run`; "planner"
    # only executes the planning phase and returns its plan/trace.
    runner_kind: Literal["agent", "planner"] = "agent"
    # Planner-only cases can either exercise the initial task planning call
    # (`generate_plan`) or replay a later planner seam from a real run
    # (`continue_from_checkpoint`).
    planner_operation: Literal["generate_plan", "continue_from_checkpoint"] = "generate_plan"
    # Optional serialized planner state for planner-only replay cases.
    planner_state: dict[str, Any] | None = None
    worktree_ref: str = "HEAD"
    # Absolute path to an external repo to worktree instead of this one.
    # None (default) means "this repository" (chef-human itself), resolved
    # via _repo_root() -- set this for cases that need a real, unfamiliar
    # codebase (as opposed to self-review cases, which deliberately use
    # chef-human's own source so the agent can be told what project it's
    # looking at).
    source_repo: str | None = None
    # When set, every file present before the agent runs must stay
    # byte-identical afterward (new files, e.g. a written report, are still
    # allowed). Use this instead of enumerating protected_files one by one
    # for cases — like a read-only code review — where nothing existing
    # should be touched at all.
    protect_all_existing_files: bool = False


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
    agent_message: str | None = None
    planner_plan: dict[str, Any] | None = None
    planner_llm_calls: list[dict[str, Any]] = field(default_factory=list)
    planner_trace_file: str | None = None
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
        case_id="scroll_grid_navigation_repair",
        level="expert",
        title="Diagnose a directional-navigation bug in a larger module",
        task=(
            "Read test_scroll_grid.py and make scroll_grid.py pass every test. The module "
            "implements column/row focus navigation for a scrolling tiling window grid; most "
            "of it is correct. Find the one method with a wrong fallback direction by reasoning "
            "about the failing tests and comparing it against its correct symmetric sibling "
            "method, rather than rewriting the module. Do not modify the tests. Run the tests "
            "before finishing."
        ),
        seed_files={
            "scroll_grid.py": 'class ScrollGrid:\n    """A scrolling-tiling window grid: columns of vertically-stacked tiles,\n    modeled after niri\'s layout. Each column remembers its own last-focused\n    row so switching columns and back restores your place."""\n\n    def __init__(self):\n        self.columns = []  # list[list[tile_id]]\n        self._active_row = {}  # col_idx -> remembered row index\n        self.active_col = None\n        self.active_row = None\n\n    def add_column(self):\n        self.columns.append([])\n        return len(self.columns) - 1\n\n    def add_window(self, col_idx, window_id):\n        self.columns[col_idx].append(window_id)\n        if self.active_col is None:\n            self.active_col = col_idx\n            self.active_row = 0\n            self._active_row[col_idx] = 0\n\n    def focused(self):\n        if self.active_col is None:\n            return None\n        return (self.active_col, self.active_row)\n\n    def _remember(self):\n        if self.active_col is not None:\n            self._active_row[self.active_col] = self.active_row\n\n    def focus_left(self):\n        if self.active_col is None or self.active_col == 0:\n            return False\n        self._remember()\n        self.active_col -= 1\n        remembered = self._active_row.get(self.active_col, 0)\n        self.active_row = min(remembered, len(self.columns[self.active_col]) - 1)\n        return True\n\n    def focus_right(self):\n        if self.active_col is None or self.active_col == len(self.columns) - 1:\n            return False\n        self._remember()\n        self.active_col += 1\n        remembered = self._active_row.get(self.active_col, 0)\n        self.active_row = min(remembered, len(self.columns[self.active_col]) - 1)\n        return True\n\n    def focus_up(self):\n        if self.active_col is None or self.active_row == 0:\n            return False\n        self.active_row -= 1\n        self._remember()\n        return True\n\n    def focus_down(self):\n        if self.active_col is None:\n            return False\n        if self.active_row >= len(self.columns[self.active_col]) - 1:\n            return False\n        self.active_row += 1\n        self._remember()\n        return True\n\n    def focus_up_or_column_right(self):\n        if self.focus_up():\n            return True\n        return self.focus_left()\n\n    def focus_down_or_column_left(self):\n        if self.focus_down():\n            return True\n        return self.focus_left()\n\n    def remove_window(self, col_idx, window_id):\n        column = self.columns[col_idx]\n        removed_row = column.index(window_id)\n        column.pop(removed_row)\n\n        if not column:\n            del self.columns[col_idx]\n            self._active_row.pop(col_idx, None)\n            self._active_row = {\n                (c - 1 if c > col_idx else c): row\n                for c, row in self._active_row.items()\n                if c != col_idx\n            }\n            if not self.columns:\n                self.active_col = None\n                self.active_row = None\n                return\n            if self.active_col is not None:\n                if self.active_col > col_idx:\n                    self.active_col -= 1\n                elif self.active_col == col_idx:\n                    self.active_col = min(col_idx, len(self.columns) - 1)\n                    remembered = self._active_row.get(self.active_col, 0)\n                    self.active_row = min(remembered, len(self.columns[self.active_col]) - 1)\n            return\n\n        if self.active_col == col_idx and self.active_row >= len(column):\n            self.active_row = len(column) - 1\n            self._remember()\n',
            "test_scroll_grid.py": 'import unittest\n\nfrom scroll_grid import ScrollGrid\n\n\ndef make_grid(shape):\n    """shape: list of column sizes, e.g. [2, 3, 1] -> 3 columns with that\n    many tiles each, named \'c{col}t{row}\'."""\n    grid = ScrollGrid()\n    for col_idx, size in enumerate(shape):\n        grid.add_column()\n        for row in range(size):\n            grid.add_window(col_idx, f"c{col_idx}t{row}")\n    return grid\n\n\nclass BasicFocusTests(unittest.TestCase):\n    def test_first_window_added_becomes_focused(self):\n        grid = ScrollGrid()\n        grid.add_column()\n        grid.add_window(0, "a")\n        self.assertEqual(grid.focused(), (0, 0))\n\n    def test_empty_grid_has_no_focus(self):\n        grid = ScrollGrid()\n        self.assertIsNone(grid.focused())\n\n    def test_focus_up_within_column(self):\n        grid = make_grid([3])\n        grid.active_row = 2\n        self.assertTrue(grid.focus_up())\n        self.assertEqual(grid.focused(), (0, 1))\n\n    def test_focus_up_at_top_of_column_fails(self):\n        grid = make_grid([3])\n        self.assertFalse(grid.focus_up())\n        self.assertEqual(grid.focused(), (0, 0))\n\n    def test_focus_down_within_column(self):\n        grid = make_grid([3])\n        self.assertTrue(grid.focus_down())\n        self.assertEqual(grid.focused(), (0, 1))\n\n    def test_focus_down_at_bottom_of_column_fails(self):\n        grid = make_grid([3])\n        grid.active_row = 2\n        self.assertFalse(grid.focus_down())\n        self.assertEqual(grid.focused(), (0, 2))\n\n    def test_focus_left_and_right(self):\n        grid = make_grid([2, 2, 2])\n        self.assertTrue(grid.focus_right())\n        self.assertEqual(grid.focused(), (1, 0))\n        self.assertTrue(grid.focus_right())\n        self.assertEqual(grid.focused(), (2, 0))\n        self.assertFalse(grid.focus_right())\n        self.assertTrue(grid.focus_left())\n        self.assertEqual(grid.focused(), (1, 0))\n\n    def test_focus_left_at_leftmost_column_fails(self):\n        grid = make_grid([2, 2])\n        self.assertFalse(grid.focus_left())\n\n    def test_column_switch_remembers_row(self):\n        grid = make_grid([3, 3])\n        grid.active_row = 2  # focus bottom tile of column 0\n        grid.focus_right()  # switches to column 1, remembered row 0\n        self.assertEqual(grid.focused(), (1, 0))\n        grid.active_row = 2\n        grid._remember()\n        grid.focus_left()  # back to column 0, should restore row 2\n        self.assertEqual(grid.focused(), (0, 2))\n\n    def test_column_switch_clamps_remembered_row_to_shorter_column(self):\n        grid = make_grid([1, 3])\n        grid.focus_right()\n        grid.active_row = 2\n        grid._remember()\n        grid.focus_left()  # column 0 only has 1 row -- must clamp to 0\n        self.assertEqual(grid.focused(), (0, 0))\n\n\nclass UpOrColumnRightTests(unittest.TestCase):\n    # These mirror niri\'s real focus-window-up-or-column-right /\n    # focus-window-down-or-column-left actions: moving further in the\n    # primary direction takes priority, and only when that\'s impossible\n    # does it fall through to switching columns -- in a specific direction\n    # (right for "up", left for "down"), not just "the other" column.\n\n    def test_moves_up_within_column_when_possible(self):\n        grid = make_grid([3, 3])\n        grid.active_row = 2\n        self.assertTrue(grid.focus_up_or_column_right())\n        self.assertEqual(grid.focused(), (0, 1))\n\n    def test_falls_through_to_column_right_at_top_of_column(self):\n        grid = make_grid([2, 2])\n        # already at row 0 (top) of column 0\n        self.assertTrue(grid.focus_up_or_column_right())\n        self.assertEqual(grid.focused(), (1, 0))\n\n    def test_falls_through_to_column_right_not_left(self):\n        # Regression test for a real-world bug (niri#686) where the\n        # fallback direction was implemented backwards: "up-or-column-right"\n        # switched to the *left* column instead of the right one.\n        grid = make_grid([2, 2, 2])\n        grid.focus_right()  # active column is now the middle one (index 1)\n        self.assertEqual(grid.focused(), (1, 0))\n        self.assertTrue(grid.focus_up_or_column_right())\n        self.assertEqual(grid.focused()[0], 2, "should move to the column on the right")\n\n    def test_fails_at_top_of_rightmost_column(self):\n        grid = make_grid([2, 2])\n        grid.focus_right()\n        self.assertFalse(grid.focus_up_or_column_right())\n        self.assertEqual(grid.focused(), (1, 0))\n\n\nclass DownOrColumnLeftTests(unittest.TestCase):\n    def test_moves_down_within_column_when_possible(self):\n        grid = make_grid([3, 3])\n        self.assertTrue(grid.focus_down_or_column_left())\n        self.assertEqual(grid.focused(), (0, 1))\n\n    def test_falls_through_to_column_left_at_bottom_of_column(self):\n        grid = make_grid([2, 2])\n        grid.focus_right()\n        grid.active_row = 1  # bottom of column 1\n        self.assertTrue(grid.focus_down_or_column_left())\n        self.assertEqual(grid.focused()[0], 0, "should move to the column on the left")\n\n    def test_fails_at_bottom_of_leftmost_column(self):\n        grid = make_grid([2, 2])\n        grid.active_row = 1\n        self.assertFalse(grid.focus_down_or_column_left())\n\n\nclass RemoveWindowTests(unittest.TestCase):\n    def test_remove_non_active_window_keeps_focus(self):\n        grid = make_grid([2, 2])\n        grid.remove_window(1, "c1t0")\n        self.assertEqual(grid.focused(), (0, 0))\n        self.assertEqual(grid.columns[1], ["c1t1"])\n\n    def test_remove_last_window_in_column_deletes_column(self):\n        grid = make_grid([1, 2])\n        grid.remove_window(0, "c0t0")\n        self.assertEqual(len(grid.columns), 1)\n        self.assertEqual(grid.focused(), (0, 0))\n        self.assertEqual(grid.columns[0], ["c1t0", "c1t1"])\n\n    def test_remove_active_window_clamps_row(self):\n        grid = make_grid([3])\n        grid.active_row = 2\n        grid.remove_window(0, "c0t2")\n        self.assertEqual(grid.focused(), (0, 1))\n\n    def test_remove_column_after_active_column_keeps_active_index(self):\n        grid = make_grid([2, 1, 2])\n        # active is column 0\n        grid.remove_window(1, "c1t0")\n        self.assertEqual(len(grid.columns), 2)\n        self.assertEqual(grid.focused(), (0, 0))\n        self.assertEqual(grid.columns[1], ["c2t0", "c2t1"])\n\n    def test_remove_all_windows_clears_focus(self):\n        grid = make_grid([1])\n        grid.remove_window(0, "c0t0")\n        self.assertIsNone(grid.focused())\n        self.assertEqual(grid.columns, [])\n\n\nif __name__ == "__main__":\n    unittest.main()\n',
        },
        verification=Verification(
            command=("{python}", "-m", "unittest", "-v", "test_scroll_grid.py"),
            timeout_seconds=45,
        ),
        protected_files=("test_scroll_grid.py",),
        max_steps=25,
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
        case_id="notification_bus_greenfield",
        level="frontier",
        title="Design a pub/sub notification bus from a vague prose brief",
        task=(
            "There's no starter code, no spec file, and no test file — you're building this from "
            "scratch and need to design the module yourself, then write and run your own tests to "
            "convince yourself it works before finishing. Put everything in a single file, notify.py.\n\n"
            "The goal: an in-memory pub/sub notification bus for a job queue. Producers publish "
            "events by type; subscribers register interest in one or more event types and get "
            "called back when a matching event is published. A Bus class is the right shape for "
            "this, with roughly this surface: bus.subscribe(name, handler, event_types) registers "
            "handler (a callable taking one positional event argument) for a subscriber identified "
            "by name, against an iterable of event type strings. bus.publish(event_type, payload) "
            "delivers an event — a dict with at least 'type' and 'payload' keys — to every handler "
            "currently subscribed to that event_type. bus.unsubscribe(name) removes all of that "
            "subscriber's registrations. A subscriber can listen for more than one event type, and "
            "more than one subscriber can listen for the same event type.\n\n"
            "Every event handed to a handler, or stored in a dead letter, must be a plain dict — not "
            "a namedtuple, dataclass, or any other object — accessed the ordinary dict way "
            "(event['type'], not event.type), since other parts of this job system that aren't "
            "yours to change will do dict-style access on whatever you deliver.\n\n"
            "Handlers are unreliable in this system — some will raise exceptions when called. A "
            "subscriber whose handler raises should be retried a few times (three total attempts is "
            "the right ballpark) before the bus gives up on delivering that particular event to that "
            "particular subscriber; if it eventually succeeds on a later attempt it should NOT be "
            "treated as failed. Retries must be immediate — don't sleep or block real time for this, "
            "and one subscriber's failures must never stop delivery to the other subscribers of the "
            "same event. Once the bus gives up on a subscriber for an event, record it somewhere "
            "durable and inspectable — bus.dead_letters as a list of dicts, each with at least a "
            "'subscriber' key (the subscriber's name) and an 'event' key (the event dict that "
            "failed), is the expected shape.\n\n"
            "Two more things this needs to support, because real producers in this job system need "
            "them. First, publish_many(events), where events is a list of (event_type, payload) "
            "pairs — it should deliver them in the order given, going through the same per-subscriber "
            "retry/dead-letter handling as publish() does one at a time, and a subscriber blowing up "
            "permanently on one event in the batch must not stop later events in the same batch from "
            "reaching that subscriber or any other subscriber. Second, late subscribers need to be "
            "able to catch up: subscribe(..., replay=True) should, before subscribe() returns, walk "
            "the bus's full history of everything published so far (via publish or publish_many) and "
            "deliver every past event matching this subscriber's event_types to it in the original "
            "publish order — through that same retry/dead-letter path, so a past event this new "
            "subscriber can never handle still ends up in dead_letters rather than being silently "
            "dropped. replay defaults to False, meaning no catch-up, just future events as normal.\n\n"
            "Beyond that interface, the internal design — how you track subscriptions and history, "
            "how retries are structured — is your call. Write tests covering ordinary delivery, "
            "multiple subscribers, retry-then-succeed, permanent failure landing in dead_letters, "
            "unsubscribe, publish_many's ordering and failure isolation, and replay's catch-up "
            "(including a replayed event that can never succeed still landing in dead_letters), and "
            "run them before finishing."
        ),
        seed_files={},
        verification=Verification(
            command=(
                "{python}",
                "-c",
                (
                    "import sys; sys.path.insert(0, '.')\n"
                    "from notify import Bus\n"
                    "\n"
                    "received = []\n"
                    "bus = Bus()\n"
                    "bus.subscribe('sub1', lambda e: received.append(e), ['order.created'])\n"
                    "bus.subscribe('sub2', lambda e: received.append(('other', e)), ['order.shipped'])\n"
                    "bus.publish('order.created', {'id': 1})\n"
                    "assert received == [{'type': 'order.created', 'payload': {'id': 1}}], received\n"
                    "\n"
                    "received2 = []\n"
                    "bus2 = Bus()\n"
                    "bus2.subscribe('a', lambda e: received2.append('a'), ['x'])\n"
                    "bus2.subscribe('b', lambda e: received2.append('b'), ['x'])\n"
                    "bus2.publish('x', {})\n"
                    "assert sorted(received2) == ['a', 'b'], received2\n"
                    "\n"
                    "calls = {'n': 0}\n"
                    "def flaky(e):\n"
                    "    calls['n'] += 1\n"
                    "    if calls['n'] < 3:\n"
                    "        raise RuntimeError('boom')\n"
                    "bus3 = Bus()\n"
                    "bus3.subscribe('flaky', flaky, ['e'])\n"
                    "bus3.publish('e', {})\n"
                    "assert calls['n'] == 3, calls\n"
                    "assert bus3.dead_letters == [], bus3.dead_letters\n"
                    "\n"
                    "def always_fails(e):\n"
                    "    raise ValueError('nope')\n"
                    "bus4 = Bus()\n"
                    "bus4.subscribe('broken', always_fails, ['e'])\n"
                    "bus4.publish('e', {'k': 'v'})\n"
                    "assert len(bus4.dead_letters) == 1, bus4.dead_letters\n"
                    "entry = bus4.dead_letters[0]\n"
                    "assert entry['subscriber'] == 'broken', entry\n"
                    "assert entry['event']['type'] == 'e', entry\n"
                    "assert entry['event']['payload'] == {'k': 'v'}, entry\n"
                    "\n"
                    "received5 = []\n"
                    "bus5 = Bus()\n"
                    "bus5.subscribe('s', lambda e: received5.append(e), ['y'])\n"
                    "bus5.unsubscribe('s')\n"
                    "bus5.publish('y', {})\n"
                    "assert received5 == [], received5\n"
                    "\n"
                    "received6 = []\n"
                    "bus6 = Bus()\n"
                    "bus6.subscribe('multi', lambda e: received6.append(e['type']), ['p', 'q'])\n"
                    "bus6.publish('p', 1)\n"
                    "bus6.publish('q', 2)\n"
                    "bus6.publish('r', 3)\n"
                    "assert received6 == ['p', 'q'], received6\n"
                    "\n"
                    "assert type(entry['event']) is dict, entry['event']\n"
                    "\n"
                    "received7 = []\n"
                    "def sometimes_fails(e):\n"
                    "    if e['payload'] == 'bad':\n"
                    "        raise RuntimeError('bad payload')\n"
                    "    received7.append(e['payload'])\n"
                    "bus7 = Bus()\n"
                    "bus7.subscribe('batch', sometimes_fails, ['b'])\n"
                    "bus7.publish_many([('b', 'first'), ('b', 'bad'), ('b', 'third')])\n"
                    "assert received7 == ['first', 'third'], received7\n"
                    "assert len(bus7.dead_letters) == 1, bus7.dead_letters\n"
                    "assert bus7.dead_letters[0]['event']['payload'] == 'bad', bus7.dead_letters\n"
                    "\n"
                    "bus8 = Bus()\n"
                    "bus8.publish('r1', 'early')\n"
                    "bus8.publish_many([('r1', 'batched1'), ('r2', 'batched2')])\n"
                    "bus8.publish('r1', 'late')\n"
                    "received8 = []\n"
                    "bus8.subscribe('catchup', lambda e: received8.append(e['payload']), ['r1'], replay=True)\n"
                    "assert received8 == ['early', 'batched1', 'late'], received8\n"
                    "\n"
                    "bus9 = Bus()\n"
                    "bus9.publish('r1', 'doomed')\n"
                    "def always_fails9(e):\n"
                    "    raise ValueError('nope')\n"
                    "bus9.subscribe('catchup9', always_fails9, ['r1'], replay=True)\n"
                    "assert len(bus9.dead_letters) == 1, bus9.dead_letters\n"
                    "assert bus9.dead_letters[0]['event']['payload'] == 'doomed', bus9.dead_letters\n"
                    "\n"
                    "print('OK')\n"
                ),
            ),
            expected_stdout="OK\n",
            timeout_seconds=30,
        ),
        max_steps=40,
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
    BenchmarkCase(
        case_id="library_system",
        level="marathon",
        title="Build a multi-module library system from a large spec",
        task=(
            "Read SPEC.md and test_library.py, then design and implement models.py, library.py, "
            "and search.py from scratch so every test passes. This spec is large — read it in "
            "full before writing code, and check off each requirement (borrowing, waitlists, "
            "fines, search) rather than guessing at the interface. Do not modify the "
            "specification or tests. Run the full test suite before finishing."
        ),
        seed_files={
            "SPEC.md": '# Library system contract\n\nImplement a small library management system across three new files: `models.py`, `library.py`,\nand `search.py`. There is no reference implementation — design it from this spec. Days are plain\nintegers (no real dates), so the whole system stays deterministic and testable.\n\n## models.py\n\n- `Book(isbn, title, author, total_copies)` — `total_copies` must be a positive int, else raise\n  `ValueError`. Exposes `available_copies`, which starts equal to `total_copies`.\n- `Member(member_id, name, borrow_limit=3)` — `borrow_limit` must be a positive int, else raise\n  `ValueError`.\n\n## library.py\n\n`Library()` holds books and members and coordinates borrowing.\n\n- `add_book(book)` / `add_member(member)` — raise `ValueError` if the isbn / member_id is already\n  registered.\n- `borrow(member_id, isbn, day) -> str` — `day` is the current integer day.\n  - Raise `KeyError` if the member or isbn is unknown.\n  - Raise `RuntimeError` if the member already has an active (unreturned) loan for that isbn.\n  - Raise `RuntimeError` if the member already has `borrow_limit` active loans.\n  - If the book has an available copy: decrement `available_copies`, record a loan with a due day\n    of `day + 14`, and return `"borrowed"`.\n  - If no copies are available: add the member to that book\'s FIFO waitlist (raise `ValueError` if\n    they\'re already on it) and return `"waitlisted"`.\n- `return_book(member_id, isbn, day) -> int` — returns the fine owed for this specific loan (0 if\n  not late).\n  - Raise `KeyError` if the member or isbn is unknown.\n  - Raise `LookupError` if the member has no active loan for that isbn.\n  - Fine is `max(0, day - due_day) * FINE_RATE_PER_DAY` (see below).\n  - Mark the loan returned and increment `available_copies`.\n  - If the book\'s waitlist is non-empty, pop the first waiting member and immediately create a new\n    loan for them (due day `day + 14`), decrementing `available_copies` again — a returned book\n    goes straight to whoever\'s next in line rather than sitting on the shelf.\n- `member_fines(member_id) -> int` — the sum of every fine ever charged to that member across all\n  returned loans. Raise `KeyError` for an unknown member.\n- `active_loans(member_id) -> list[str]` — isbns the member currently has out, in the order they\n  were borrowed. Raise `KeyError` for an unknown member.\n- `all_books() -> list[Book]` — every registered book, in any order.\n\n`FINE_RATE_PER_DAY` must be importable from `library.py` and equal `2`.\n\n## search.py\n\n- `search_books(library, query=None, author=None, available_only=False) -> list[Book]` — books\n  from `library`, filtered by:\n  - `query`: case-insensitive substring match against the book\'s title (skip filter if `None`).\n  - `author`: case-insensitive substring match against the book\'s author (skip filter if `None`).\n  - `available_only`: when `True`, only include books with `available_copies > 0`.\n  - Results are sorted by title, ascending.\n\n## Constraints\n\n- Do not modify `test_library.py`.\n- All three modules must be plain, dependency-free Python (standard library only).\n',
            "test_library.py": 'import unittest\n\nfrom library import Library, FINE_RATE_PER_DAY\nfrom models import Book, Member\nfrom search import search_books\n\n\ndef make_library():\n    lib = Library()\n    lib.add_book(Book("111", "The Hobbit", "Tolkien", 1))\n    lib.add_book(Book("222", "Dune", "Herbert", 2))\n    lib.add_book(Book("333", "Foundation", "Asimov", 1))\n    lib.add_member(Member("m1", "Ann"))\n    lib.add_member(Member("m2", "Bo"))\n    lib.add_member(Member("m3", "Cy", borrow_limit=1))\n    return lib\n\n\nclass ModelTests(unittest.TestCase):\n    def test_book_rejects_nonpositive_copies(self):\n        for bad in (0, -1):\n            with self.assertRaises(ValueError):\n                Book("x", "t", "a", bad)\n\n    def test_book_available_copies_starts_full(self):\n        b = Book("x", "t", "a", 3)\n        self.assertEqual(b.available_copies, 3)\n\n    def test_member_rejects_nonpositive_limit(self):\n        for bad in (0, -1):\n            with self.assertRaises(ValueError):\n                Member("m", "n", borrow_limit=bad)\n\n\nclass LibraryRegistrationTests(unittest.TestCase):\n    def test_duplicate_isbn_rejected(self):\n        lib = make_library()\n        with self.assertRaises(ValueError):\n            lib.add_book(Book("111", "Dup", "X", 1))\n\n    def test_duplicate_member_rejected(self):\n        lib = make_library()\n        with self.assertRaises(ValueError):\n            lib.add_member(Member("m1", "Dup"))\n\n\nclass BorrowTests(unittest.TestCase):\n    def test_unknown_member_or_isbn_raises_keyerror(self):\n        lib = make_library()\n        with self.assertRaises(KeyError):\n            lib.borrow("missing", "111", day=0)\n        with self.assertRaises(KeyError):\n            lib.borrow("m1", "missing", day=0)\n\n    def test_borrow_available_book(self):\n        lib = make_library()\n        self.assertEqual(lib.borrow("m1", "222", day=0), "borrowed")\n        self.assertEqual(lib.active_loans("m1"), ["222"])\n\n    def test_double_borrow_same_book_rejected(self):\n        lib = make_library()\n        lib.borrow("m1", "222", day=0)\n        with self.assertRaises(RuntimeError):\n            lib.borrow("m1", "222", day=1)\n\n    def test_borrow_limit_enforced(self):\n        lib = make_library()\n        self.assertEqual(lib.borrow("m3", "111", day=0), "borrowed")\n        with self.assertRaises(RuntimeError):\n            lib.borrow("m3", "222", day=0)\n\n    def test_borrow_when_unavailable_waitlists(self):\n        lib = make_library()\n        lib.borrow("m1", "111", day=0)  # takes the only copy\n        self.assertEqual(lib.borrow("m2", "111", day=0), "waitlisted")\n\n    def test_duplicate_waitlist_entry_rejected(self):\n        lib = make_library()\n        lib.borrow("m1", "111", day=0)\n        lib.borrow("m2", "111", day=0)\n        with self.assertRaises(ValueError):\n            lib.borrow("m2", "111", day=1)\n\n\nclass ReturnAndFineTests(unittest.TestCase):\n    def test_return_unknown_raises_keyerror(self):\n        lib = make_library()\n        with self.assertRaises(KeyError):\n            lib.return_book("missing", "111", day=0)\n        with self.assertRaises(KeyError):\n            lib.return_book("m1", "missing", day=0)\n\n    def test_return_without_active_loan_raises_lookuperror(self):\n        lib = make_library()\n        with self.assertRaises(LookupError):\n            lib.return_book("m1", "111", day=0)\n\n    def test_on_time_return_has_no_fine(self):\n        lib = make_library()\n        lib.borrow("m1", "222", day=0)\n        fine = lib.return_book("m1", "222", day=10)\n        self.assertEqual(fine, 0)\n        self.assertEqual(lib.member_fines("m1"), 0)\n\n    def test_late_return_charges_fine(self):\n        lib = make_library()\n        lib.borrow("m1", "222", day=0)  # due day 14\n        fine = lib.return_book("m1", "222", day=17)\n        self.assertEqual(fine, 3 * FINE_RATE_PER_DAY)\n        self.assertEqual(lib.member_fines("m1"), 3 * FINE_RATE_PER_DAY)\n\n    def test_fines_accumulate_across_loans(self):\n        lib = make_library()\n        lib.borrow("m1", "222", day=0)\n        lib.return_book("m1", "222", day=16)  # 2 days late\n        lib.borrow("m1", "333", day=16)\n        lib.return_book("m1", "333", day=33)  # 3 days late\n        self.assertEqual(lib.member_fines("m1"), 5 * FINE_RATE_PER_DAY)\n\n    def test_return_frees_a_copy_for_borrowing(self):\n        lib = make_library()\n        lib.borrow("m1", "111", day=0)\n        lib.return_book("m1", "111", day=5)\n        self.assertEqual(lib.borrow("m2", "111", day=5), "borrowed")\n\n    def test_return_services_waitlist_automatically(self):\n        lib = make_library()\n        lib.borrow("m1", "111", day=0)\n        lib.borrow("m2", "111", day=0)  # waitlisted\n        lib.return_book("m1", "111", day=5)\n        # m2 should now hold the book without calling borrow() again.\n        self.assertEqual(lib.active_loans("m2"), ["111"])\n\n    def test_active_loans_removed_after_return(self):\n        lib = make_library()\n        lib.borrow("m1", "222", day=0)\n        lib.return_book("m1", "222", day=1)\n        self.assertEqual(lib.active_loans("m1"), [])\n\n\nclass SearchTests(unittest.TestCase):\n    def test_search_by_title_substring_case_insensitive(self):\n        lib = make_library()\n        result = search_books(lib, query="dun")\n        self.assertEqual([b.isbn for b in result], ["222"])\n\n    def test_search_by_author_substring_case_insensitive(self):\n        lib = make_library()\n        result = search_books(lib, author="ASIMOV")\n        self.assertEqual([b.isbn for b in result], ["333"])\n\n    def test_search_available_only(self):\n        lib = make_library()\n        lib.borrow("m1", "111", day=0)  # exhausts the only copy of 111\n        result = search_books(lib, available_only=True)\n        self.assertNotIn("111", [b.isbn for b in result])\n        self.assertIn("222", [b.isbn for b in result])\n\n    def test_search_results_sorted_by_title(self):\n        lib = make_library()\n        result = search_books(lib)\n        titles = [b.title for b in result]\n        self.assertEqual(titles, sorted(titles))\n\n    def test_search_with_no_filters_returns_everything(self):\n        lib = make_library()\n        result = search_books(lib)\n        self.assertEqual(len(result), 3)\n\n\nif __name__ == "__main__":\n    unittest.main()\n',
        },
        verification=Verification(
            command=("{python}", "-m", "unittest", "-v", "test_library.py"),
            timeout_seconds=60,
        ),
        protected_files=("SPEC.md", "test_library.py"),
        max_steps=45,
    ),
    BenchmarkCase(
        case_id="chef_human_tools_self_review",
        level="review",
        title="Code-review chef-human's own tool implementations",
        task=(
            "You are in a git worktree of the chef-human repository (the project this very "
            "agent is part of). Perform a careful code review of every file directly inside "
            "the chef_human/tools/ directory — do not review any other directory. Look for "
            "real correctness bugs: places where the code's actual behavior would surprise a "
            "caller, not style nitpicks or missing features. For each bug you find, write one "
            "entry to a new file, REVIEW.md, at the repository root, with: the file and "
            "function/method, what is wrong, and a concrete input or call sequence that shows "
            "the failure. Do not fix any bugs and do not modify any existing file — only "
            "create REVIEW.md. When you are done, finish with a one- or two-sentence summary "
            "of how many issues you found."
        ),
        seed_files={},
        verification=None,
        protect_all_existing_files=True,
        max_steps=35,
        workspace_kind="worktree",
    ),
    BenchmarkCase(
        case_id="vague_feature_request_small_repo",
        level="frontier",
        title="Vague feature request against a small unfamiliar codebase (fast checkpoint smoke test)",
        # Same shape and intent as vague_feature_request_real_repo below --
        # a bare, scope-free request against existing code the planner has
        # never seen, meant to reproduce the explore-only-collapse failure
        # and exercise the checkpoint mechanism -- but against a tiny seeded
        # project instead of a full external git worktree checkout, so a
        # run finishes in a couple of minutes of wall-clock exploration
        # instead of a large real C/host codebase. Use this to shake out
        # checkpoint bugs cheaply before spending a full
        # vague_feature_request_real_repo run.
        #
        # First cut of this case named "a browser" directly in the task and
        # seeded a single flat notes.py -- against qwen3.6:35b-a3b it
        # planned straight to "Implement a Flask web interface" without
        # ever considering a checkpoint (confirmed via agent.log: the model's
        # own reasoning explicitly said "I don't need a checkpoint here. I
        # know what to do"). That's a real, valid outcome (checkpoints are
        # for genuine uncertainty, not mandatory), but it means that version
        # exercised no checkpoint code at all. This version widens the
        # ambiguity on both axes real vague tasks actually have: the task
        # names no implementation shape (web app? REST API? shared server?
        # sync tool?), and the seed is now four small interdependent modules
        # (storage backend, core notes, tag filtering, CLI) that must be
        # read together to know what "notes" even means structurally before
        # picking an approach -- closer to the real case's need to explore
        # multiple files before implementing.
        task=(
            "Let's make this something the whole team can actually use "
            "together, not just from the command line."
        ),
        seed_files={
            "storage.py": (
                '"""Pluggable storage backends for notes."""\n'
                "import json\n"
                "from pathlib import Path\n\n\n"
                "class JSONFileStore:\n"
                '    def __init__(self, path="notes.json"):\n'
                "        self.path = Path(path)\n\n"
                "    def load(self):\n"
                "        if self.path.exists():\n"
                "            return json.loads(self.path.read_text())\n"
                "        return []\n\n"
                "    def save(self, notes):\n"
                "        self.path.write_text(json.dumps(notes, indent=2))\n"
            ),
            "notes.py": (
                '"""Core note operations, backed by a pluggable storage backend."""\n'
                "from storage import JSONFileStore\n\n"
                "_store = JSONFileStore()\n\n\n"
                "def add_note(text, tags=None):\n"
                "    notes = _store.load()\n"
                '    notes.append({"text": text, "tags": tags or []})\n'
                "    _store.save(notes)\n\n\n"
                "def list_notes():\n"
                "    return _store.load()\n\n\n"
                "def remove_note(index):\n"
                "    notes = _store.load()\n"
                "    del notes[index]\n"
                "    _store.save(notes)\n"
            ),
            "tags.py": (
                '"""Helpers for filtering notes by tag."""\n'
                "from notes import list_notes\n\n\n"
                "def notes_with_tag(tag):\n"
                '    return [n for n in list_notes() if tag in n.get("tags", [])]\n'
            ),
            "cli.py": (
                '"""Command-line entry point for the notes tool."""\n'
                "import sys\n\n"
                "from notes import add_note, list_notes, remove_note\n"
                "from tags import notes_with_tag\n\n\n"
                "def main():\n"
                "    if len(sys.argv) < 2:\n"
                '        print("Usage: cli.py [add TEXT | list | remove INDEX | tag TAG]")\n'
                "        return\n"
                "    command = sys.argv[1]\n"
                '    if command == "add":\n'
                '        add_note(" ".join(sys.argv[2:]))\n'
                '    elif command == "list":\n'
                "        for i, note in enumerate(list_notes()):\n"
                '            print(f"{i}: {note[\'text\']} {note.get(\'tags\', [])}")\n'
                '    elif command == "remove":\n'
                "        remove_note(int(sys.argv[2]))\n"
                '    elif command == "tag":\n'
                "        for note in notes_with_tag(sys.argv[2]):\n"
                '            print(note["text"])\n'
                "    else:\n"
                '        print(f"Unknown command: {command}")\n\n\n'
                'if __name__ == "__main__":\n'
                "    main()\n"
            ),
            "README.md": (
                "# notes\n\n"
                "A small command-line note-taking tool with tag support.\n\n"
                "Usage:\n\n"
                '    python cli.py add "buy milk"\n'
                "    python cli.py list\n"
                "    python cli.py remove 0\n"
                "    python cli.py tag groceries\n\n"
                "Notes are stored via a pluggable storage backend (see storage.py); "
                "the default backend keeps everything in notes.json in the current "
                "directory.\n"
            ),
        },
        verification=Verification(
            # Mirrors vague_feature_request_real_repo's "did the agent do
            # anything beyond exploring" check, but this workspace is a
            # plain seed dir (workspace_kind="seed"), not a git worktree --
            # no `git status` available. Instead, name every known seed
            # file explicitly and check for any other real file outside
            # .chef-human/ (this harness's own log/session directory) --
            # notes.json and __pycache__ are also excluded even though
            # neither is a seed file: they're both side effects of merely
            # running the seeded code to understand it (a completely
            # legitimate way to explore -- `python cli.py list` writes
            # notes.json and leaves .pyc files behind), and a pure-
            # exploration run must not look like it did real implementation
            # work just because it ran the existing program.
            command=(
                "sh",
                "-c",
                "find . -mindepth 1 "
                "\\( -path './.chef-human' -o -path './.chef-human/*' "
                "-o -path './__pycache__' -o -path './__pycache__/*' \\) -prune -o "
                "-type f -not -name 'storage.py' -not -name 'notes.py' "
                "-not -name 'tags.py' -not -name 'cli.py' -not -name 'README.md' "
                "-not -name 'notes.json' "
                "-print | grep -q .",
            ),
            timeout_seconds=15,
        ),
        max_steps=25,
    ),
    BenchmarkCase(
        case_id="vague_feature_request_real_repo_planner",
        level="frontier",
        title="Planner-only vague web interface request against an unfamiliar real codebase",
        task="Let's add a web interface to this project",
        seed_files={},
        verification=None,
        protect_all_existing_files=True,
        max_steps=40,
        workspace_kind="worktree",
        runner_kind="planner",
        source_repo="~/ubertooth",
        worktree_ref="master",
    ),
    BenchmarkCase(
        case_id="vague_feature_request_checkpoint_replay",
        level="frontier",
        title="Planner-only replay of the vague web-interface checkpoint continuation",
        task="Let's add a web interface to this project",
        seed_files={
            "README.md": (
                "# Ubertooth\n\n"
                "Bluetooth experimentation platform with CLI host tools and firmware.\n"
            ),
            "docs/software.rst": (
                "Host tools include ubertooth-rx for packet capture/decoding and "
                "ubertooth-specan for spectrum analysis.\n"
            ),
            "host/CMakeLists.txt": (
                "option(ENABLE_PYTHON \"Enable Python support\" ON)\n"
                "add_subdirectory(ubertooth-tools)\n"
            ),
            "host/ubertooth-tools/ubertooth-rx.c": (
                "/* CLI packet capture and decode tool */\n"
                "int main(void) { return 0; }\n"
            ),
            "host/ubertooth-tools/ubertooth-specan.c": (
                "/* CLI spectrum analyzer tool */\n"
                "int main(void) { return 0; }\n"
            ),
            "host/libubertooth/ubertooth.h": (
                "/* libubertooth USB communication API */\n"
            ),
        },
        verification=None,
        protect_all_existing_files=True,
        max_steps=40,
        runner_kind="planner",
        planner_operation="continue_from_checkpoint",
        planner_state={
            "steps": [
                {
                    "description": "Read the README.md for project overview and architecture",
                    "status": "completed",
                    "type": "leaf",
                },
                {
                    "description": "List the host/ directory contents to understand existing host-side software",
                    "status": "completed",
                    "type": "leaf",
                },
                {
                    "description": "Read key files in host/ (e.g., any main programs or library code)",
                    "status": "completed",
                    "type": "leaf",
                },
                {
                    "description": "Read docs/software.rst to understand what host tools are provided",
                    "status": "completed",
                    "type": "leaf",
                },
                {
                    "description": "Check if there's any existing web-related code or configuration",
                    "status": "completed",
                    "type": "leaf",
                },
                {
                    "description": "Explore the codebase to learn what it does and how it's structured, so the shape of a web interface can be decided from what's actually there rather than guessed",
                    "status": "completed",
                    "type": "checkpoint",
                },
            ],
            "checkpoint_index": 5,
            "evidence": (
                "The exploration phase established the following grounded facts from the repo:\n"
                "- Project: Ubertooth, a Bluetooth sniffer/decoder platform for a USB device.\n"
                "- Languages/build: primarily C with a CMake build; ENABLE_PYTHON is available.\n"
                "- Key host tools: ubertooth-rx for passive Bluetooth discovery/decode and "
                "ubertooth-specan for spectrum analysis.\n"
                "- Existing capabilities: live USB capture, file-based capture/export, survey "
                "mode, and configuration via CLI arguments.\n"
                "- Existing infrastructure: libubertooth handles USB communication; host-side "
                "tools expose capture modes and runtime parameters that a web interface would "
                "need to mirror.\n"
                "- Implication for web architecture: browsers cannot access the USB device "
                "directly, so a local server bridge is required; the previous run chose a Python "
                "web server plus WebSocket streaming as the natural continuation."
            ),
        },
    ),
    BenchmarkCase(
        case_id="vague_feature_request_real_repo",
        level="frontier",
        title="Vague feature request against an unfamiliar real codebase",
        task="Let's add a web interface to this project",
        seed_files={},
        # Deliberately just the bare, everyday-phrased request with no
        # scope, no hints about the target directory, and no repo
        # description -- this is verbatim what was actually typed against a
        # real project (ubertooth, a USB Bluetooth-sniffing tool, mostly C
        # firmware/host code) and reproduced a specific failure: the planner
        # collapsed the whole task into pure exploration (list the
        # directory, glob for source files, search for existing web
        # dirs, read the README) and finished without ever writing a single
        # line of implementation. Adding scope hints here would make the
        # planner's job easier and risk not reproducing that failure at
        # all, which defeats the point of this case.
        verification=Verification(
            # No fixed expected output is possible for a task this open-
            # ended -- what this case actually checks is the specific
            # failure it was built to catch: did the agent do *anything*
            # beyond reading/exploring the repo? A real git worktree makes
            # this a one-line check: `git status --porcelain` is empty iff
            # nothing was created or modified. The pathspec exclude is load-
            # bearing, not cosmetic -- .chef-human/ (this harness's own log/
            # session directory, seeded into the workspace before the agent
            # runs) is untracked by the target repo's git, so plain `git
            # status --porcelain` is *always* non-empty regardless of what
            # the agent did. Confirmed the hard way: an agent run that
            # generated zero plan steps still reported PASS under the
            # unfiltered command.
            command=(
                "sh",
                "-c",
                'test -n "$(git status --porcelain -- . \':(exclude).chef-human\')"',
            ),
            timeout_seconds=15,
        ),
        max_steps=40,
        workspace_kind="worktree",
        source_repo="~/ubertooth",
        worktree_ref="master",
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


def _repo_root() -> Path:
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=Path(__file__).resolve().parent,
        capture_output=True,
        text=True,
        check=True,
    )
    return Path(result.stdout.strip())


def _resolve_source_repo_path(source_repo: str) -> Path:
    candidate = Path(source_repo).expanduser()
    if candidate.exists():
        return candidate.resolve()
    if source_repo.startswith("~/"):
        real_home = Path(pwd.getpwuid(os.getuid()).pw_dir)
        return (real_home / source_repo[2:]).resolve()
    return candidate.resolve()


def _create_worktree(source_repo: Path, workspace: Path, ref: str) -> None:
    """Check out a detached-HEAD git worktree of `source_repo` at `workspace`.

    `workspace` must not already exist — `git worktree add` creates it. Using
    `--detach` avoids "branch already checked out" errors when `ref` is the
    branch checked out in the primary worktree (e.g. `main`)."""
    workspace.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["git", "worktree", "add", "--detach", str(workspace), ref],
        cwd=source_repo,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"git worktree add failed: {result.stderr.strip()}")


def _prune_worktrees(source_repo: Path) -> None:
    """Clean up git's worktree metadata after a case's directory was deleted
    directly (e.g. by the benchmark's temporary-root cleanup) instead of via
    `git worktree remove`. Safe to call even if nothing needs pruning."""
    subprocess.run(
        ["git", "worktree", "prune"],
        cwd=source_repo,
        capture_output=True,
        text=True,
        check=False,
    )


def _prepare_agent_path() -> dict[str, str]:
    """Ensure benchmark workspaces expose a stable `python` command.

    The benchmark tasks ask the agent to "run with Python", and many local
    models naturally choose `python ...`. On this machine the supported
    interpreter is available as `python3.12`/`sys.executable`, but not as a
    `python` shell command. Seed a tiny shim so the benchmark measures agent
    behavior rather than host PATH quirks.

    The shim directory only needs to be an absolute path on PATH -- it does
    not need to live inside the workspace, and living there was actively
    harmful: WorkspaceManager's IGNORE_PATTERNS doesn't know about this
    harness-specific directory name, so it showed up as a real file in the
    repo map the agent's own planner sees, making an otherwise-empty
    greenfield workspace look like "an existing codebase" (same failure mode
    as the workspace-root agent.log fixed alongside this)."""
    shim_dir = Path(tempfile.mkdtemp(prefix="chef-human-benchmark-shim-"))
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
    env["CHEF_OLLAMA_THINK"] = "true"
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


def _serialize_messages(messages: Sequence[Any]) -> list[dict[str, str]]:
    serialized: list[dict[str, str]] = []
    for message in messages:
        role = getattr(message, "role", "")
        serialized.append(
            {
                "role": getattr(role, "value", str(role)),
                "content": getattr(message, "content", ""),
            }
        )
    return serialized


def _plan_node_from_data(data: dict[str, Any]):
    from chef_human.agent.planner import PlanNode, StepStatus

    node = PlanNode(
        index=int(data.get("index", 0)),
        description=str(data["description"]),
        status=StepStatus(str(data.get("status", "pending"))),
        declared_type=str(data.get("type", "leaf")),
    )
    children = [_plan_node_from_data(child) for child in data.get("children", [])]
    if children:
        node.set_children(children)
    return node


def _run_planner_case(
    case: BenchmarkCase,
    workspace: Path,
    *,
    model: str | None,
    timeout_seconds: int,
) -> dict[str, Any]:
    from chef_human import config
    from chef_human.agent import create_agent
    from chef_human.agent.planner import Plan

    settings = replace(
        config.settings,
        ollama_model=model or config.settings.ollama_model,
        ollama_think=True,
    )
    loop, _ = create_agent(
        max_steps=case.max_steps,
        workspace_root=str(workspace),
        settings=settings,
    )
    planner = loop._planner
    trace_path = (workspace / ".chef-human" / "planner-trace.json").resolve()
    llm_calls: list[dict[str, Any]] = []
    original_complete = planner._complete
    result_plan: Plan | None = None
    message: str | None = None

    async def traced_complete(request, activity="planning"):
        response = await original_complete(request, activity)
        llm_calls.append(
            {
                "activity": activity,
                "request": {
                    "messages": _serialize_messages(request.messages),
                    "temperature": request.temperature,
                    "max_tokens": request.max_tokens,
                },
                "response": {
                    "content": response.message.content,
                    "thinking": response.thinking,
                    "usage": response.usage,
                },
            }
        )
        return response

    planner._complete = traced_complete
    try:
        if case.planner_operation == "continue_from_checkpoint":
            state = case.planner_state or {}
            root_steps = [_plan_node_from_data(step) for step in state.get("steps", [])]
            plan = Plan(goal=case.task, steps=root_steps)
            checkpoint = plan.steps[int(state["checkpoint_index"])]
            continued_steps = asyncio.run(
                asyncio.wait_for(
                    planner.continue_from_checkpoint(
                        plan,
                        checkpoint,
                        evidence=str(state.get("evidence", "")),
                    ),
                    timeout=timeout_seconds,
                )
            )
            plan.root.set_children(plan.steps + continued_steps)
            asyncio.run(
                asyncio.wait_for(
                    planner.expand_spliced_steps(plan, continued_steps),
                    timeout=timeout_seconds,
                )
            )
            result_plan = plan
            message = (
                "Planner replayed checkpoint continuation and produced "
                f"{len(continued_steps)} continuation step(s)"
            )
        else:
            plan = asyncio.run(
                asyncio.wait_for(loop._plan_task(case.task), timeout=timeout_seconds)
            )
            result_plan = plan
            message = f"Planner produced {len(plan.steps)} top-level step(s)"
    finally:
        planner._complete = original_complete
        trace_payload = {
            "task": case.task,
            "planner_operation": case.planner_operation,
            "planner_state": case.planner_state,
            "plan": result_plan.to_dict() if result_plan is not None else None,
            "llm_calls": llm_calls,
            "message": message,
        }
        trace_path.write_text(json.dumps(trace_payload, indent=2) + "\n", encoding="utf-8")
    return {
        "success": True,
        "steps_taken": len(result_plan.steps),
        "total_prompt_tokens": loop._total_prompt_tokens,
        "total_completion_tokens": loop._total_completion_tokens,
        "message": message,
        "planner_plan": result_plan.to_dict(),
        "planner_llm_calls": llm_calls,
        "planner_trace_file": str(trace_path),
    }


def run_case(
    case: BenchmarkCase,
    workspace: Path,
    *,
    model: str | None,
    agent_timeout: int,
    source_repo: Path | None = None,
) -> BenchmarkResult:
    # Must be absolute before use: `_create_worktree` runs `git worktree add`
    # with cwd=source_repo, so a relative `workspace` would be resolved
    # against the *source repo's* directory instead of this process's cwd,
    # landing the actual worktree somewhere entirely different from the path
    # `--workspace` below (which resolves the same Path against this
    # process's cwd) tells the agent to operate in.
    workspace = workspace.resolve()
    if case.workspace_kind == "worktree":
        if case.source_repo:
            repo = _resolve_source_repo_path(case.source_repo)
        else:
            repo = source_repo or _repo_root()
        _create_worktree(repo, workspace, case.worktree_ref)
    else:
        workspace.mkdir(parents=True, exist_ok=False)
        _write_seed(workspace, case.seed_files)
    agent_env = _prepare_agent_path()
    # The agent process configures its log file before it creates
    # .chef-human/ itself (that happens later, during create_agent()'s
    # symbol-index setup), so the directory must exist up front or
    # logging.basicConfig(filename=...) fails outright.
    (workspace / ".chef-human").mkdir(parents=True, exist_ok=True)
    before = _snapshot(workspace)
    if case.protect_all_existing_files:
        protected = dict(before)
    else:
        protected = {name: before[name] for name in case.protected_files}
    started = time.monotonic()
    agent_exit_code: int | None = None
    agent_data: dict[str, Any] = {}
    errors: list[str] = []
    try:
        if case.runner_kind == "planner":
            agent_data = _run_planner_case(
                case,
                workspace,
                model=model,
                timeout_seconds=agent_timeout,
            )
            agent_exit_code = 0
        else:
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
                # Inside .chef-human/, not the workspace root:
                # WorkspaceManager's IGNORE_PATTERNS already excludes that
                # directory from the repo map the agent's own planner sees.
                # A log file sitting in the workspace root instead makes an
                # otherwise-empty greenfield workspace look like "an existing
                # codebase" to the planner, which then applies the system
                # prompt's mandatory explore-before-implementing rule and
                # (observed against qwen3.6:35b-a3b) collapses the whole plan
                # into a single unproductive "explore the project structure"
                # step.
                str((workspace / ".chef-human" / "agent.log").resolve()),
            ]
            if model:
                command.extend(("--model", model))
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
    except TimeoutError:
        errors.append(f"Planner exceeded the {agent_timeout}s case timeout")
    except Exception as exc:
        errors.append(str(exc))

    verifier_output = ""
    verifier_success = True
    if case.verification is not None:
        verify_command = [
            sys.executable if part == "{python}" else part
            for part in case.verification.command
        ]
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
    agent_message = agent_data.get("message")
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
        agent_message=agent_message[-4000:] if isinstance(agent_message, str) else None,
        planner_plan=agent_data.get("planner_plan"),
        planner_llm_calls=agent_data.get("planner_llm_calls") or [],
        planner_trace_file=agent_data.get("planner_trace_file"),
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
        "schema_version": 2,
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
        if result["agent_message"] and not result["verifier_output"]:
            print(f"       {result['agent_message'][:200]}")
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

    worktree_cases = [c for c in cases if c.workspace_kind == "worktree"]
    default_source_repo = _repo_root() if any(c.source_repo is None for c in worktree_cases) else None
    used_repos = {
        _resolve_source_repo_path(c.source_repo) if c.source_repo else default_source_repo
        for c in worktree_cases
    }
    used_repos.discard(None)

    results: list[BenchmarkResult] = []
    try:
        for case in cases:
            result = run_case(
                case,
                root / case.case_id,
                model=args.model,
                agent_timeout=args.timeout,
                source_repo=default_source_repo,
            )
            if temporary_root is not None:
                result.workspace = None
            results.append(result)
    finally:
        if temporary_root is not None:
            shutil.rmtree(temporary_root, ignore_errors=True)
        for repo in used_repos:
            _prune_worktrees(repo)

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
