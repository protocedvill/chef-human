from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path
from typing import Sequence

logger = logging.getLogger(__name__)


def _find_ruff() -> str | None:
    return shutil.which("ruff")


def _detect_linter(file_path: str) -> str | None:
    ext = Path(file_path).suffix
    if ext == ".py":
        return "ruff"
    return None


def run_lint(file_path: str) -> str:
    linter = _detect_linter(file_path)
    if linter is None:
        return ""
    if linter == "ruff":
        return _run_ruff(file_path)
    return ""


def _run_ruff(file_path: str) -> str:
    ruff_path = _find_ruff()
    if ruff_path is None:
        return ""
    try:
        result = subprocess.run(
            [ruff_path, "check", file_path],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        logger.warning("Ruff check failed: %s", exc)
        return ""

    output = result.stdout.strip()
    if not output:
        return ""
    lines = output.splitlines()
    filtered = _filter_lint_lines(lines, file_path)
    if not filtered:
        return ""
    return "\n".join(filtered)


def _filter_lint_lines(lines: Sequence[str], file_path: str) -> list[str]:
    result: list[str] = []
    for line in lines:
        if file_path in line:
            result.append(line.strip())
    return result


def format_lint_result(lint_output: str) -> str:
    if not lint_output:
        return ""
    issue_count = len(lint_output.splitlines())
    return (
        f"\nLint results ({issue_count} issue{'s' if issue_count != 1 else ''}):\n"
        f"{lint_output}"
    )
