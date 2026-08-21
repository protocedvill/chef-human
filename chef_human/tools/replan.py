from __future__ import annotations

import logging
from typing import Any

from chef_human.tools.registry import ToolResult

logger = logging.getLogger(__name__)

# Machine-distinct action name carried in `ToolResult.control_payload` so the
# ReAct loop can branch on it without parsing prose.
REQUEST_REPLAN_ACTION = "replan_requested"


class RequestReplanTool:
    """Model-visible control tool that requests an evidence-driven replan.

    This tool is a *control-plane* action, not a work-plane tool: it never
    touches the workspace or the plan itself. It only returns a structured
    acknowledgment (`ToolResult.control == True`) that the ReAct loop
    interprets. The loop -- not this tool -- owns all execution-time plan
    mutation (evidence gating, sole-call enforcement, archiving the old
    subtree, and invoking the planner). See
    `.scratch/evidence-driven-subtree-replan/spec.md`.

    Argument contract: `reason` plus `evidence_summary`. The model never
    supplies a target node id -- the scope is always the current acting
    node's subtree, which only the loop knows.
    """

    name = "request_replan"
    description = (
        "Request that the current plan subtree be replanned because fresh, "
        "objective read evidence contradicts the current step or subtree. "
        "Use only when you have just read source that proves the current "
        "step's premise is wrong -- not for ordinary uncertainty, a failed "
        "attempt, or a change of mind with no new evidence. Cite the specific "
        "file and line (or equivalent current fact) in evidence_summary. This "
        "is a control action: it must be the only tool call in your response, "
        "and the current turn ends when it is accepted. It does not modify any "
        "file or the plan itself."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "reason": {
                "type": "string",
                "description": (
                    "Which specific assumption, child step, or subtree idea is "
                    "invalidated, and why continuing the current decomposition "
                    "would be wrong or wasteful."
                ),
            },
            "evidence_summary": {
                "type": "string",
                "description": (
                    "The fresh, objective evidence that disproved the "
                    "assumption -- cite specific file paths and line numbers "
                    "(or equivalent current facts) you actually read this turn."
                ),
            },
        },
        "required": ["reason", "evidence_summary"],
    }

    async def run(self, reason: str = "", evidence_summary: str = "") -> ToolResult:
        reason = (reason or "").strip()
        evidence_summary = (evidence_summary or "").strip()
        logger.info(
            "request_replan invoked: reason=%r evidence=%r",
            reason[:200],
            evidence_summary[:200],
        )
        # A single template; the reason is only surfaced when present so the
        # empty-argument case still reads cleanly.
        detail = f" ({reason})" if reason else ""
        ack = (
            "Replan requested. The current acting turn ends now and the "
            f"planner will rebuild this subtree from the evidence you cited{detail}. "
            "Do not issue further tool calls this turn."
        )
        # Structured acknowledgment. `control=True` is the machine-distinct
        # signal; the loop branches on it (and on `control_payload["action"]`)
        # rather than reading `output`. `output` stays a short, model-legible
        # confirmation so the transcript reads sensibly too.
        return ToolResult(
            success=True,
            output=ack,
            control=True,
            control_payload={
                "action": REQUEST_REPLAN_ACTION,
                "reason": reason,
                "evidence_summary": evidence_summary,
            },
        )
