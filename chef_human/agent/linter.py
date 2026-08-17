from __future__ import annotations

import logging
import re
import shutil
import subprocess
from pathlib import Path
from typing import Sequence

logger = logging.getLogger(__name__)

_LINT_LINE_RE = re.compile(r"^(.+?):(\d+):(\d+):\s*(\S+)\s+(.+)$")
_HUNK_HEADER_RE = re.compile(r"^@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@.*")


def annotate_diff_with_lint(
    diff: str,
    lint_output: str,
    linter_name: str = "ruff",
) -> str:
    """Overlay lint warnings on a unified diff.

    For each lint warning that references a line in the diff's + side,
    append an annotation to the relevant hunk line.
    """
    if not lint_output or not diff:
        return diff

    lint_by_line: dict[int, list[str]] = {}
    for line in lint_output.splitlines():
        m = _LINT_LINE_RE.match(line)
        if m:
            line_num = int(m.group(2))
            code = m.group(4)
            message = m.group(5)
            lint_by_line.setdefault(line_num, []).append(f"# {linter_name}: {code} {message}")

    if not lint_by_line:
        return diff

    diff_lines = diff.splitlines(keepends=True)
    result: list[str] = []
    hunk_new_start = 0

    for raw_line in diff_lines:
        m = _HUNK_HEADER_RE.match(raw_line)
        if m:
            hunk_new_start = int(m.group(2))
            result.append(raw_line)
            continue

        if raw_line.startswith("+") and not raw_line.startswith("+++"):
            current_line = hunk_new_start
            hunk_new_start += 1
            annotations = lint_by_line.get(current_line, [])
            if annotations:
                stripped = raw_line.rstrip("\n\r")
                annotation = "; ".join(annotations)
                result.append(f"{stripped}  {annotation}\n")
                continue
            result.append(raw_line)
        elif raw_line.startswith(" "):
            hunk_new_start += 1
            result.append(raw_line)
        else:
            result.append(raw_line)

    return "".join(result)


def _find_ruff() -> str | None:
    return shutil.which("ruff")


def _detect_linter(file_path: str) -> str | None:
    ext = Path(file_path).suffix
    if ext == ".py":
        return "ruff"
    return None


def syntax_error(source: str, file_path: str) -> str:
    """ruff-style `path:line:col: E999 ...` line if `source` (treated as the
    content of `file_path`) does not compile, else ''."""
    try:
        compile(source, file_path, "exec")
    except (SyntaxError, ValueError) as exc:
        if isinstance(exc, SyntaxError):
            line = exc.lineno or 0
            col = exc.offset or 0
            msg = exc.msg or "syntax error"
        else:
            line, col, msg = 0, 0, str(exc)
        return f"{file_path}:{line}:{col}: E999 SyntaxError: {msg}"
    return ""


def check_python_syntax(file_path: str) -> str:
    """Deterministic syntax-only check for a Python file, independent of any
    external linter binary being installed/on PATH. Returns ruff-style
    `path:line:col: E999 ...` output when the file does not compile, else ''."""
    try:
        source = Path(file_path).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return ""
    return syntax_error(source, file_path)


def run_lint(file_path: str) -> str:
    linter = _detect_linter(file_path)
    if linter is None:
        return ""
    if linter == "ruff":
        syntax_errors = check_python_syntax(file_path)
        if syntax_errors:
            return syntax_errors
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
