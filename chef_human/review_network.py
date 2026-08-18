"""Network layer for review_tree: clusters densely-connected files into
groups of up to `ReviewConfig.network_max_cluster_size` files, then runs an
agentic, tool-using validation pass over each cluster.

Sits between the file layer and root in the tree:
    root -> network (this module) -> file (review_tree._execute_file_node) -> leaf

Unlike every other layer in review_tree.py, a network node's LLM call is
multi-turn: it can request specific code spans from files inside its own
cluster via a `read_code_span` tool before answering, rather than seeing
only text descriptions of findings. Built directly on chef_human.llm.backend's
provider-agnostic tool-call primitives (`CompletionRequest.tools`,
`ToolDefinition`, `Message.tool_calls`) plus the standalone
`chef_human.agent.parser.parse_native_tool_calls` -- deliberately NOT
`chef_human.agent.react_loop`/`ToolRegistry`/`Planner`, matching
review_tree.py's own stated principle of staying independent of that
orchestration stack. `parser.parse_native_tool_calls` is a leaf utility with
no dependency on the rest of `chef_human.agent`, so importing it doesn't
violate that principle.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from chef_human.agent.parser import parse_native_tool_calls
from chef_human.llm.backend import CompletionRequest, Message, Role, ToolDefinition
from chef_human.llm.tokenizer import Tokenizer
from chef_human.review_tree import (
    CodeUnit,
    NodeResult,
    ReviewConfig,
    ReviewNode,
    _merge_child_findings,
    _parse_dropped,
    _parse_findings,
    _parse_json_object,
    _parse_margin_references,
)

# Each read_code_span result is hard-capped at this many lines regardless of
# what the model asked for, so one call can't single-handedly blow the
# cluster's whole network_token_budget.
MAX_READ_LINES = 150


def cluster_files(
    file_paths: list[Path], edges: dict[frozenset[Path], int], max_cluster_size: int
) -> list[list[Path]]:
    """Greedy agglomerative merge on file-level edge weights (see
    review_tree.build_file_edges): repeatedly merges the two clusters
    connected by the highest total edge weight, as long as the merged size
    stays <= max_cluster_size, until no eligible merge remains. Files with
    no cross-file edges to anything end up as singleton clusters -- never
    blocked, just reviewed alone. Deterministic: ties broken by
    (-weight, sorted file paths) so the same input always clusters the same
    way.

    O(n^3) in file count via repeated O(n^2) best-pair scans -- fine for the
    small file counts this tool is actually run against; not meant to scale
    to whole-monorepo runs."""
    clusters: list[set[Path]] = [{p} for p in sorted(file_paths, key=str)]

    while True:
        best_key: tuple[int, tuple[str, ...]] | None = None
        best_pair: tuple[int, int] | None = None
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                if len(clusters[i]) + len(clusters[j]) > max_cluster_size:
                    continue
                weight = sum(
                    edges.get(frozenset((a, b)), 0) for a in clusters[i] for b in clusters[j]
                )
                if weight <= 0:
                    continue
                candidate_key = (-weight, tuple(sorted(str(p) for p in clusters[i] | clusters[j])))
                if best_key is None or candidate_key < best_key:
                    best_key = candidate_key
                    best_pair = (i, j)
        if best_pair is None:
            break
        i, j = best_pair
        clusters[i] |= clusters[j]
        del clusters[j]

    return [sorted(c, key=str) for c in clusters]


NETWORK_SYSTEM_PROMPT = (
    "You are validating and cross-checking code-review findings for one cluster of "
    "densely-connected files -- files that call into or are called by each other. Focus on this "
    "cluster's own dense core: combine and re-examine the per-file findings below, escalating "
    "severity when combining two files' findings reveals something worse than either looked "
    "alone (same severity/dropped-with-reasons contract used at every other layer -- do not just "
    "omit a finding you reject, list it explicitly in \"dropped\" with a reason, or it will be "
    "added back automatically). You may call read_code_span(file, start_line, end_line) to pull "
    "up specific code from files IN THIS CLUSTER if you need to verify something beyond what the "
    "per-file summaries already told you -- it will refuse files outside the cluster. If you "
    "notice a finding, docstring, or signature reference something in a file OUTSIDE this "
    "cluster, do NOT guess about it or try to resolve it without being able to read it -- name it "
    "explicitly in a \"margin_references\" array with a one-sentence note instead. When finished "
    "(with or without tool calls), respond with a single JSON object and nothing else, shaped "
    'like: {"summary": str, "findings": [{"file": str, "line": int or null, "category": str, '
    '"summary": str, "failure_scenario": str, "severity": "low"|"medium"|"high"}], "dropped": '
    '[{"file": str, "line": int or null, "category": str, "reason": str}], "margin_references": '
    '[{"file": str, "note": str}]}.'
)

READ_CODE_SPAN_TOOL = ToolDefinition(
    name="read_code_span",
    description=(
        "Read a specific line range from a file in this network's cluster (listed in the prompt "
        "above). Returns an error if the file isn't part of this cluster -- report those under "
        "margin_references instead of calling this tool for them."
    ),
    parameters={
        "type": "object",
        "properties": {
            "file": {"type": "string", "description": "Path of the file, as listed in the cluster."},
            "start_line": {"type": "integer", "description": "1-indexed first line to read."},
            "end_line": {"type": "integer", "description": "1-indexed last line to read (inclusive)."},
        },
        "required": ["file", "start_line", "end_line"],
    },
)


def _format_file_findings(file_nodes: list[ReviewNode]) -> str:
    parts = []
    for fn in file_nodes:
        result = fn.result
        assert result is not None
        findings_text = (
            "\n".join(
                f"- [{f.severity}/{f.category}] {f.file}:{f.line} -- {f.summary} "
                f"({f.failure_scenario})"
                for f in result.findings
            )
            or "(no findings)"
        )
        parts.append(
            f"--- File: {fn.node_id} (validation_status={result.validation_status}) ---\n"
            f"Summary: {result.summary}\nFindings:\n{findings_text}\n"
        )
    return "\n".join(parts)


def _format_docstrings(units: list[CodeUnit]) -> str:
    lines = [
        f"- `{u.path}` {u.signature.strip()}\n  \"\"\"{u.docstring}\"\"\""
        for u in sorted(units, key=lambda u: (str(u.path), u.name))
        if u.docstring
    ]
    return "\n".join(lines) or "(no docstrings)"


def _initial_prompt(goal: str, cluster_paths: list[Path], file_nodes: list[ReviewNode], units: list[CodeUnit]) -> str:
    parts = [
        f"Goal: {goal}\n",
        f"Cluster files (the ONLY files read_code_span can read): "
        f"{', '.join(str(p) for p in cluster_paths)}\n",
        "## Per-file validated findings\n" + _format_file_findings(file_nodes),
        "## Docstrings for units in this cluster\n" + _format_docstrings(units),
    ]
    return "\n\n".join(parts)


def _dispatch_read_code_span(
    arguments: dict[str, Any],
    cluster_paths: list[Path],
    source_by_path: dict[Path, str],
    tokenizer: Tokenizer,
    budget_state: dict[str, int],
    network_token_budget: int,
) -> str:
    file_arg = arguments.get("file")
    start_line = arguments.get("start_line")
    end_line = arguments.get("end_line")

    matched = next(
        (p for p in cluster_paths if str(p) == file_arg or p.name == file_arg), None
    )
    if matched is None:
        return (
            f"ERROR: '{file_arg}' is not part of this network's cluster "
            f"({', '.join(str(p) for p in cluster_paths)}) -- out of scope for this tool. "
            "Report it under margin_references instead."
        )
    if not isinstance(start_line, int) or not isinstance(end_line, int) or start_line < 1 or end_line < start_line:
        return "ERROR: start_line/end_line must be positive integers with end_line >= start_line."

    lines = source_by_path[matched].splitlines(keepends=True)
    end_line = min(end_line, start_line + MAX_READ_LINES - 1)
    span_text = "".join(lines[start_line - 1 : end_line])
    cost = tokenizer.count(span_text)
    if budget_state["used"] + cost > network_token_budget:
        return (
            "ERROR: network_token_budget exhausted for this cluster -- no further reads "
            "available. Finalize your answer with what you already have."
        )
    budget_state["used"] += cost
    return f"{matched}:{start_line}-{end_line}\n{span_text}"


async def run_network_node(
    node: ReviewNode, cluster_paths: list[Path], source_by_path: dict[Path, str], config: ReviewConfig
) -> None:
    """Runs the bounded multi-turn tool loop for one network node, against
    config.synthesis_backend. node.children must already be resolved file
    nodes (see review_tree._execute_file_node) -- this only reads their
    results, never re-executes them."""
    units_in_cluster = [u for u in config.units if u.path in cluster_paths]
    messages = [
        Message(role=Role.system, content=NETWORK_SYSTEM_PROMPT),
        Message(
            role=Role.user,
            content=_initial_prompt(node.goal, cluster_paths, node.children, units_in_cluster),
        ),
    ]
    budget_state = {"used": 0}
    if config.on_progress:
        config.on_progress(f"{node.node_id} -- dispatching (network, up to {config.network_max_tool_turns} turns)")

    final_response = None
    for turn in range(config.network_max_tool_turns):
        response = await config.synthesis_backend.complete(
            CompletionRequest(
                messages=messages,
                tools=[READ_CODE_SPAN_TOOL],
                temperature=0.0,
                max_tokens=config.max_completion_tokens,
            )
        )
        tool_calls = (
            parse_native_tool_calls(response.message.tool_calls) if response.message.tool_calls else []
        )
        if not tool_calls:
            final_response = response
            break
        messages.append(
            Message(
                role=Role.assistant,
                content=response.message.content or "",
                tool_calls=[{"function": {"name": tc.name, "arguments": tc.arguments}} for tc in tool_calls],
            )
        )
        for tc in tool_calls:
            if tc.name != "read_code_span":
                result_text = f"ERROR: unknown tool '{tc.name}' -- only read_code_span is available."
            else:
                result_text = _dispatch_read_code_span(
                    tc.arguments, cluster_paths, source_by_path, config.tokenizer,
                    budget_state, config.network_token_budget,
                )
            messages.append(Message(role=Role.tool, content=result_text))
    else:
        # Turn cap reached while the model was still requesting tools --
        # force one last no-tools request so it answers with what it has
        # rather than the loop just silently ending with no result.
        messages.append(
            Message(
                role=Role.user,
                content=(
                    "Tool call budget reached. Finalize your answer now with the JSON object "
                    "described in the system prompt, using only what you've already gathered. "
                    "Do not request any more tool calls."
                ),
            )
        )
        final_response = await config.synthesis_backend.complete(
            CompletionRequest(messages=messages, tools=None, temperature=0.0, max_tokens=config.max_completion_tokens)
        )

    assert final_response is not None
    data = _parse_json_object(final_response.message.content)
    usage = final_response.usage or {}
    completion_tokens = usage.get("completion_tokens", 0)
    summary = data.get("summary")
    node.result = NodeResult(
        summary=summary if isinstance(summary, str) else "",
        findings=_parse_findings(data.get("findings")),
        prompt_tokens=usage.get("prompt_tokens", 0),
        completion_tokens=completion_tokens,
        truncated=completion_tokens >= config.max_completion_tokens,
        dropped=_parse_dropped(data.get("dropped")),
        margin_references=_parse_margin_references(data.get("margin_references")),
    )
    node.result.findings, node.result.recovered_from_children = _merge_child_findings(
        node.result.findings, node.children, node.result.dropped
    )
    if config.on_progress:
        status = "TRUNCATED" if node.result.truncated else f"{len(node.result.findings)} finding(s)"
        config.on_progress(f"{node.node_id} -- done ({status}, tool budget used={budget_state['used']})")
