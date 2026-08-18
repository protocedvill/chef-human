"""Execution-backed verification pass for review_tree findings.

Motivation (see docs/review-tree-report.md and the follow-up experiment that
produced this module): a reasoning-only re-check of a finding against the
source text is not reliable for claims about *library/runtime behavior*
(e.g. "difflib.unified_diff produces double newlines here") -- a verifier
model asked to just re-read the code tends to repeat the same plausible-
sounding wrong reasoning as the original leaf, because the error was never a
misreading of the code, it was a wrong belief about what running it does.
Executing an actual repro is the only thing that reliably catches that
class of false positive; verifying by re-reading only helps for the *other*
class (claims contradicted by something explicit in the code itself, e.g. a
documented docstring behavior).

This module asks the model to produce a short, self-contained Python check
for empirically-checkable findings (an input/output claim on a specific
function), executes it in a subprocess with a timeout and no network, and
feeds the real stdout/stderr back to the model for a final verdict --
falling back to reasoning-only verification when the model itself says a
finding isn't checkable by execution (e.g. modularity/simplification
findings, or a claim about un-runnable surrounding infrastructure).
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from chef_human.llm.backend import CompletionRequest, LLMBackend, Message, Role
from chef_human.review_tree import Finding, _parse_json_object

Verdict = Literal["confirmed", "refuted", "uncertain"]

CHECK_TIMEOUT_SECONDS = 10

PROPOSE_CHECK_SYSTEM_PROMPT = (
    "You are checking whether a code-review finding is real by trying to write an "
    "executable Python repro for it. You will see the finding and the full real source "
    "of the file it refers to. If the finding makes an empirically checkable claim about "
    "runtime behavior (a function returns/raises/produces something specific for some "
    "input), write a short, self-contained Python script that imports the file at "
    "TARGET_PATH (an absolute path already provided to you -- use importlib to load it "
    "by that exact path, the file may not be on sys.path as a package) and PRINTS one "
    "line: either 'FINDING_CONFIRMED: <what actually happened>' or "
    "'FINDING_REFUTED: <what actually happened instead>'. The script must be fully "
    "self-contained, deterministic, have no network access requirement, and finish in "
    "under a few seconds. If the finding is NOT empirically checkable this way (a design/"
    "modularity/simplification claim, or a claim about code that can't be exercised in "
    "isolation), set check_code to null instead -- do not force a fake check. "
    'Respond with a single JSON object: {"checkable": bool, "check_code": str or null, '
    '"reasoning": str}'
)

FINAL_VERDICT_SYSTEM_PROMPT = (
    "You are finalizing verification of a code-review finding. You'll see the finding, "
    "and either (a) the real stdout/stderr from actually executing a repro script against "
    "the real code, or (b) a note that the finding wasn't empirically checkable and your "
    "own prior reasoning about it. Trust actual execution output over any prior reasoning "
    "-- if a script printed FINDING_REFUTED or crashed in a way that contradicts the "
    "claim, refute it regardless of how plausible the original claim sounded. "
    'Respond with a single JSON object: {"verdict": "confirmed"|"refuted"|"uncertain", '
    '"reasoning": str, "corrected_failure_scenario": str or null}'
)


@dataclass(frozen=True)
class VerifiedFinding:
    finding: Finding
    verdict: Verdict
    reasoning: str
    corrected_failure_scenario: str | None
    checked_by_execution: bool
    check_stdout: str = ""
    check_stderr: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.finding.to_dict(),
            "verdict": self.verdict,
            "verify_reasoning": self.reasoning,
            "corrected_failure_scenario": self.corrected_failure_scenario,
            "checked_by_execution": self.checked_by_execution,
        }


def _propose_check_prompt(finding: Finding, source: str, target_path: Path) -> str:
    return (
        f"## Finding to verify\nCategory: {finding.category}\n"
        f"Location: {finding.file}:{finding.line}\nSummary: {finding.summary}\n"
        f"Claimed failure scenario: {finding.failure_scenario}\n\n"
        f"## TARGET_PATH\n{target_path}\n\n"
        f"## Full real source of {finding.file}\n```python\n{source}\n```"
    )


def _run_check_script(code: str, workdir: Path) -> tuple[str, str, bool]:
    """Runs `code` as a subprocess with a timeout, no shell, cwd pinned to a
    scratch dir. Returns (stdout, stderr, timed_out). Never raises -- a
    crashing/timing-out check script is itself evidence, not a harness bug."""
    script_path = workdir / "check.py"
    script_path.write_text(code, encoding="utf-8")
    try:
        proc = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            timeout=CHECK_TIMEOUT_SECONDS,
            cwd=workdir,
        )
        return proc.stdout, proc.stderr, False
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout.decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr = exc.stderr.decode() if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        return stdout, stderr + "\n[TIMED OUT]", True


async def _complete_json(backend: LLMBackend, system_prompt: str, user_prompt: str) -> dict[str, Any]:
    request = CompletionRequest(
        messages=[
            Message(role=Role.system, content=system_prompt),
            Message(role=Role.user, content=user_prompt),
        ],
        temperature=0.0,
        max_tokens=6000,
    )
    response = await backend.complete(request)
    return _parse_json_object(response.message.content)


async def verify_finding(
    backend: LLMBackend, finding: Finding, source: str, target_path: Path
) -> VerifiedFinding:
    propose = await _complete_json(
        backend, PROPOSE_CHECK_SYSTEM_PROMPT, _propose_check_prompt(finding, source, target_path)
    )
    check_code = propose.get("check_code")
    checkable = bool(propose.get("checkable")) and isinstance(check_code, str) and check_code.strip()

    if not checkable:
        result = await _complete_json(
            backend,
            FINAL_VERDICT_SYSTEM_PROMPT,
            "Not empirically checkable by execution. Prior reasoning about checkability: "
            f"{propose.get('reasoning', '')}\n\nFinding: [{finding.category}] {finding.file}:"
            f"{finding.line} -- {finding.summary} ({finding.failure_scenario})",
        )
        verdict = result.get("verdict")
        return VerifiedFinding(
            finding=finding,
            verdict=verdict if verdict in ("confirmed", "refuted", "uncertain") else "uncertain",
            reasoning=result.get("reasoning", ""),
            corrected_failure_scenario=result.get("corrected_failure_scenario"),
            checked_by_execution=False,
        )

    with tempfile.TemporaryDirectory(prefix="review_verify_") as tmp:
        stdout, stderr, timed_out = _run_check_script(check_code, Path(tmp))

    outcome = (
        f"[TIMED OUT after {CHECK_TIMEOUT_SECONDS}s]" if timed_out else
        f"stdout:\n{stdout}\nstderr:\n{stderr}"
    )
    result = await _complete_json(
        backend,
        FINAL_VERDICT_SYSTEM_PROMPT,
        f"Finding: [{finding.category}] {finding.file}:{finding.line} -- {finding.summary} "
        f"({finding.failure_scenario})\n\nExecuted repro script:\n```python\n{check_code}\n```\n\n"
        f"Real execution result:\n{outcome}",
    )
    verdict = result.get("verdict")
    return VerifiedFinding(
        finding=finding,
        verdict=verdict if verdict in ("confirmed", "refuted", "uncertain") else "uncertain",
        reasoning=result.get("reasoning", ""),
        corrected_failure_scenario=result.get("corrected_failure_scenario"),
        checked_by_execution=True,
        check_stdout=stdout,
        check_stderr=stderr,
    )


async def verify_findings(
    backend: LLMBackend, findings: list[Finding], source_by_file: dict[str, str], path_by_file: dict[str, Path]
) -> list[VerifiedFinding]:
    """Sequential, not gathered in parallel: each check_code execution is a
    subprocess launch, and keeping this predictable/serial matches the rest
    of review_tree's traversal (see the comment on ReActLoop's own dispatch
    loop for the same reasoning in the main agent)."""
    results: list[VerifiedFinding] = []
    for finding in findings:
        source = source_by_file.get(finding.file, "")
        target_path = path_by_file.get(finding.file, Path(finding.file))
        results.append(await verify_finding(backend, finding, source, target_path))
    return results
