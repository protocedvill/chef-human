from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Any

from chef_human.agent.linter import _detect_linter, _find_ruff
from chef_human.tools.diff import compute_diff
from chef_human.tools.registry import ToolResult

if TYPE_CHECKING:
    from chef_human.agent.workspace import WorkspaceManager
    from chef_human.tools.diff import DiffStore

logger = logging.getLogger(__name__)


class LintFixTool:
    name = "lint_fix"
    description = "Auto-fix lint warnings in the specified file(s) using ruff --fix (or other supported linter)."
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Single file or directory to fix. If not provided, fixes all workspace Python files.",
                "default": None,
            },
            "check_only": {
                "type": "boolean",
                "description": "Only check for lint issues, don't apply fixes. Returns the issue list.",
                "default": False,
            },
            "select": {
                "type": "string",
                "description": "Comma-separated list of rule codes to fix (e.g., 'F401,F841'). Fixes all by default.",
                "default": None,
            },
        },
    }

    def __init__(
        self,
        workspace: WorkspaceManager,
        diff_store: DiffStore | None = None,
    ) -> None:
        self._workspace = workspace
        self._diff_store = diff_store

    async def run(
        self,
        path: str | None = None,
        check_only: bool = False,
        select: str | None = None,
    ) -> ToolResult:
        target = self._resolve_target(path)
        if target is None:
            return ToolResult(
                success=False,
                error=f"Path not found or outside workspace: {path}",
            )

        linter = _detect_linter(str(target))
        if linter is None:
            ext = target.suffix if target.is_file() else "files in " + str(target)
            return ToolResult(output=f"No supported linter for {ext}.")

        if linter == "ruff":
            return await self._run_ruff(target, check_only, select)

        return ToolResult(output=f"Linter '{linter}' is not yet supported for auto-fix.")

    def _resolve_target(self, path: str | None) -> Path | None:
        if path is None:
            return self._workspace.root
        resolved = self._workspace.resolve(path)
        if not resolved.exists():
            return None
        if not self._workspace.is_within_workspace(resolved):
            return None
        return resolved

    async def _run_ruff(
        self,
        target: Path,
        check_only: bool,
        select: str | None,
    ) -> ToolResult:
        ruff_path = _find_ruff()
        if ruff_path is None:
            return ToolResult(
                success=False,
                error="ruff not found on PATH. Install it with `pip install ruff`.",
            )

        output_parts: list[str] = []

        if check_only:
            cmd = [ruff_path, "check", str(target)]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            issues = result.stdout.strip()
            if not issues:
                return ToolResult(output="No lint issues found.")
            output_parts.append(f"Lint issues in {target}:")
            output_parts.append(issues)
            return ToolResult(output="\n".join(output_parts))

        # Read before state
        before_map = self._read_files(target)

        # Build fix command
        cmd = [ruff_path, "check", "--fix", str(target)]
        if select:
            cmd.extend(["--select", select])

        fix_result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

        # Read after state
        after_map = self._read_files(target)

        # Compute diffs
        changed = 0
        all_paths = set(before_map) | set(after_map)
        for fp in sorted(all_paths):
            before = before_map.get(fp, "")
            after = after_map.get(fp, "")
            if before != after:
                diff = compute_diff(before, after, path=fp)
                if self._diff_store and diff:
                    self._diff_store.record(
                        fp,
                        diff,
                        "lint_fix",
                        old_content=before,
                        new_content=after,
                    )
                changed += 1

        if changed == 0:
            return ToolResult(output="No lint issues found or fixable.")

        output_parts.append(f"Fixed lint issues in {changed} file{'s' if changed != 1 else ''}.")
        report = fix_result.stdout.strip()
        if report:
            output_parts.append(report)

        return ToolResult(output="\n".join(output_parts))

    def _read_files(self, target: Path) -> dict[str, str]:
        """Recursively read all Python files under target, returning {path: content}."""
        result: dict[str, str] = {}
        if target.is_file():
            try:
                result[str(target)] = target.read_text(encoding="utf-8")
            except Exception:
                pass
        else:
            for f in sorted(target.rglob("*.py")):
                try:
                    result[str(f)] = f.read_text(encoding="utf-8")
                except Exception:
                    continue
        return result
