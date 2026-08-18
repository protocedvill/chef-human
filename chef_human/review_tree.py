"""Goal-tree code review: a standalone prototype of a decompose-into-subgoals
architecture, applied to code review as its first concrete use case.

This is deliberately independent of chef_human.agent.planner / react_loop --
it does not use the Planner, ReActLoop, or ToolRegistry. A goal is
decomposed into subgoals (here: analysis lenses, then one function/class per
leaf) until each subgoal's scope fits in one context window (an "atom"),
executed with a single direct LLM call per atom, then results are
synthesized back up the tree: each non-leaf node re-reasons over its
children's summaries and structured findings rather than just concatenating
them, so a subtree/layer review can surface issues no single leaf would see
alone.

A leaf reviews exactly one function or class body, given full context of
that body plus (budget-permitting) the signatures and docstrings -- never
full bodies -- of functions it calls and functions that call it, resolved
across every file loaded for the run, so a leaf isn't reviewing code in a
vacuum.

Run as: `python -m chef_human.review_tree --target-dir chef_human/tools`
"""

from __future__ import annotations

import argparse
import ast
import asyncio
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal

from chef_human import config as chef_config
from chef_human.llm import create_backend
from chef_human.llm.backend import CompletionRequest, LLMBackend, Message, Role
from chef_human.llm.tokenizer import Tokenizer, create_tokenizer

DEFAULT_LEAF_TOKEN_BUDGET = 6000

# Ollama's `think` chat option: False/True disable/enable reasoning
# entirely; "low"/"medium"/"high" (supported by some models, including
# qwen3.6) trade reasoning depth against latency and, more importantly for
# this tool, against how much of max_completion_tokens gets eaten before the
# JSON answer ever appears. "low" is the default here because leaf/synthesis
# tasks are closer to structured classification than open-ended problem
# solving -- deep reasoning is rarely worth its own truncation risk.
ThinkLevel = bool | Literal["low", "medium", "high"]


def _default_backend(model: str | None, think: ThinkLevel) -> LLMBackend:
    """Builds the default backend for run_review(), honoring `think` as an
    explicit override rather than only the ambient CHEF_OLLAMA_THINK config
    -- review_tree is a separate system from the main agent and shouldn't
    depend on that global being set correctly for its own calls to behave."""
    settings = chef_config.settings
    if settings.llm_backend != "ollama":
        return create_backend(model_override=model)
    from chef_human.llm.ollama_backend import OllamaBackend

    return OllamaBackend(
        model=model or settings.ollama_model, host=settings.ollama_host, think=think
    )


@dataclass(frozen=True)
class ReviewMethod:
    """An analysis lens: one first-layer branch of the tree."""

    id: str
    description: str


DEFAULT_METHODS: tuple[ReviewMethod, ...] = (
    ReviewMethod(
        "correctness",
        "Wrong behavior on valid or boundary inputs: off-by-one errors, wrong return "
        "values, incorrect control flow, broken invariants.",
    ),
    ReviewMethod(
        "error_handling",
        "Exceptions raised, swallowed, or reported inconsistently with the code's own "
        "contracts and docstrings; ambiguous success/failure signaling.",
    ),
    ReviewMethod(
        "state_and_side_effects",
        "Bugs in mutable state, caching, undo/redo/diff history, or partial writes -- "
        "places where one call's effect on shared state surprises a later call.",
    ),
    ReviewMethod(
        "security",
        "Unsafe handling of untrusted input: shell/command injection, path traversal, "
        "unsafe eval/exec, or guard logic that can be bypassed.",
    ),
    ReviewMethod(
        "resource_and_concurrency",
        "File handle leaks, non-atomic writes, missing timeouts, or races between "
        "concurrently dispatched operations.",
    ),
    ReviewMethod(
        "modularity",
        "Poor separation of concerns: duplicated logic that should be one shared "
        "function, a module doing two unrelated jobs, an abstraction that leaks its "
        "internals across a boundary it's supposed to own, or coupling that makes one "
        "file impossible to change without also changing an unrelated one.",
    ),
    ReviewMethod(
        "simplification",
        "Unnecessary complexity for what the code actually needs to do: dead code, "
        "unreachable branches, a general mechanism built for one caller, redundant "
        "indirection, or logic that could be materially shorter and clearer without "
        "changing behavior.",
    ),
)


@dataclass(frozen=True)
class CodeChunk:
    path: Path
    text: str
    start_line: int = 1
    end_line: int | None = None

    def label(self) -> str:
        if self.end_line is None:
            return str(self.path)
        return f"{self.path}:{self.start_line}-{self.end_line}"


@dataclass(frozen=True)
class CodeUnit:
    """One reviewable function or class -- the leaf granularity of the tree.

    `kind="method_group"` is used only for pieces produced by
    _split_oversized_unit when a class is too big to review as a whole: one
    piece per method, each still carrying the class's own header/docstring
    baked into `chunk.text` as free context so the piece isn't a method
    floating with no idea what class it belongs to."""

    path: Path
    name: str
    kind: Literal["function", "class", "method_group"]
    chunk: CodeChunk
    signature: str
    docstring: str
    calls: frozenset[str]


@dataclass(frozen=True)
class ConnectedRef:
    """A signature+docstring-only reference shown to a leaf as background --
    the leaf never sees this unit's body, so findings must never be reported
    against it."""

    path: Path
    name: str
    signature: str
    docstring: str
    relation: Literal["callee", "caller"]


@dataclass(frozen=True)
class Finding:
    file: str
    line: int | None
    category: str
    summary: str
    failure_scenario: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "file": self.file,
            "line": self.line,
            "category": self.category,
            "summary": self.summary,
            "failure_scenario": self.failure_scenario,
        }


@dataclass
class NodeResult:
    summary: str
    findings: list[Finding] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    # True when the completion hit max_completion_tokens -- for a thinking
    # model this usually means it was still inside <think> reasoning and
    # never emitted the JSON answer, so an empty summary/findings here means
    # "truncated", not "reviewed and found nothing". Surfaced explicitly so
    # that distinction doesn't get silently lost.
    truncated: bool = False
    # Findings recovered from children that the synthesis call itself failed
    # to include in its own findings list -- see _merge_child_findings. 0 for
    # leaves (no children to lose anything from).
    recovered_from_children: int = 0
    # How many connected callee/caller signatures were included vs. couldn't
    # fit the remaining token budget. 0/0 for non-leaf nodes.
    connected_included: int = 0
    connected_omitted: int = 0


@dataclass
class ReviewNode:
    node_id: str
    goal: str
    method: str | None
    depth: int
    scope: tuple[CodeChunk, ...]
    children: list["ReviewNode"] = field(default_factory=list)
    result: NodeResult | None = None
    # Leaf-only: the module-level background (imports/constants) shown for
    # this leaf's file, and the connected callee/caller signatures actually
    # included in its prompt. Empty for non-leaf nodes.
    header_context: CodeChunk | None = None
    connected: tuple[ConnectedRef, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "goal": self.goal,
            "method": self.method,
            "depth": self.depth,
            "scope": [chunk.label() for chunk in self.scope],
            "header_context": self.header_context.label() if self.header_context else None,
            "connected": [
                {
                    "name": c.name,
                    "path": str(c.path),
                    "relation": c.relation,
                    "signature": c.signature,
                    "docstring": c.docstring,
                }
                for c in self.connected
            ],
            "summary": self.result.summary if self.result else None,
            "findings": [f.to_dict() for f in self.result.findings] if self.result else [],
            "prompt_tokens": self.result.prompt_tokens if self.result else 0,
            "completion_tokens": self.result.completion_tokens if self.result else 0,
            "truncated": self.result.truncated if self.result else False,
            "recovered_from_children": self.result.recovered_from_children if self.result else 0,
            "connected_included": self.result.connected_included if self.result else 0,
            "connected_omitted": self.result.connected_omitted if self.result else 0,
            "children": [child.to_dict() for child in self.children],
        }

    def all_findings(self) -> list[Finding]:
        """Findings from every leaf in this subtree. _merge_child_findings
        guarantees a synthesized node's own result.findings already contains
        everything from its children (deduped), so this only recurses into
        leaves -- including internal nodes' own results here would double
        count everything they already absorbed from their children."""
        if not self.children:
            return list(self.result.findings) if self.result else []
        collected: list[Finding] = []
        for child in self.children:
            collected.extend(child.all_findings())
        return collected

    def truncated_node_ids(self) -> list[str]:
        """node_ids where the completion hit max_completion_tokens -- an
        empty summary/findings there means "truncated", not "reviewed and
        found nothing", so this is worth checking before trusting a clean
        result."""
        ids = [self.node_id] if self.result and self.result.truncated else []
        for child in self.children:
            ids.extend(child.truncated_node_ids())
        return ids


# Atom criterion: swappable so alternative strategies (LLM self-assessment,
# fixed depth, ...) can replace the default token-budget check without
# touching the traversal logic below. Operates on one already
# structurally-atomic unit (a single function/class) plus its file's header
# context -- not a whole multi-file scope -- since decomposition now stops
# at function/class boundaries by construction; this only decides whether
# that one unit needs the oversized-unit fallback.
AtomCheck = Callable[[CodeUnit, "CodeChunk | None", Tokenizer, int], bool]


def token_budget_atom_check(
    unit: CodeUnit, header: CodeChunk | None, tokenizer: Tokenizer, leaf_token_budget: int
) -> bool:
    total = tokenizer.count(unit.chunk.text)
    if header is not None:
        total += tokenizer.count(header.text)
    return total <= leaf_token_budget


@dataclass
class ReviewConfig:
    # Used for leaf calls -- one function/class body plus context, closer to
    # structured classification than open-ended reasoning, so a quicker/
    # smaller model is usually enough.
    backend: LLMBackend
    # Used for every synthesis call (method-level, oversized-unit-fallback,
    # and root) -- these have to weigh multiple sub-reviews against each
    # other and catch cross-cutting issues, which benefits more from a
    # stronger model. Defaults to the same backend as leaves when no
    # separate synthesis model/backend is configured, so nothing changes
    # unless a caller opts in.
    synthesis_backend: LLMBackend
    tokenizer: Tokenizer
    # Computed once per run (lens-independent) and reused across every
    # ReviewMethod -- the same functions/classes and cross-references apply
    # no matter which analysis lens is reviewing them.
    units: list[CodeUnit] = field(default_factory=list)
    header_by_path: dict[Path, CodeChunk | None] = field(default_factory=dict)
    definition_map: dict[str, list[CodeUnit]] = field(default_factory=dict)
    caller_index: dict[str, list[CodeUnit]] = field(default_factory=dict)
    leaf_token_budget: int = DEFAULT_LEAF_TOKEN_BUDGET
    is_atom: AtomCheck = token_budget_atom_check
    # Thinking models (Ollama's `think` option) spend part of this budget on
    # <think> reasoning before ever emitting the JSON answer. A prior version
    # of this tool started low and retried with an escalating budget on
    # truncation, but at qwen3.6 scale that meant ~2-3 LLM calls per node on
    # 93% of nodes -- expensive and slow for what it bought. Simpler: just
    # give every call enough headroom up front and accept the one-shot cost.
    max_completion_tokens: int = 30000
    # Called with a short label right before each LLM call dispatches, and
    # again with the result summary right after it returns. Traversal is
    # strictly sequential (one call at a time), so this is the only signal
    # available that a long run is still alive rather than hung.
    on_progress: Callable[[str], None] | None = None


def _parse_json_object(text: str) -> dict[str, Any]:
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else {}
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        for index, char in enumerate(text):
            if char != "{":
                continue
            try:
                value, _ = decoder.raw_decode(text[index:])
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                return value
    return {}


def _parse_findings(raw: Any) -> list[Finding]:
    if not isinstance(raw, list):
        return []
    findings: list[Finding] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        file = item.get("file")
        summary = item.get("summary")
        if not isinstance(file, str) or not isinstance(summary, str):
            continue
        line = item.get("line")
        category = item.get("category")
        failure_scenario = item.get("failure_scenario")
        findings.append(
            Finding(
                file=file,
                line=line if isinstance(line, int) else None,
                category=category if isinstance(category, str) else "uncategorized",
                summary=summary,
                failure_scenario=failure_scenario if isinstance(failure_scenario, str) else "",
            )
        )
    return findings


async def _complete_result(
    config: ReviewConfig, backend: LLMBackend, system_prompt: str, user_prompt: str, label: str
) -> NodeResult:
    if config.on_progress:
        config.on_progress(f"{label} -- dispatching (budget={config.max_completion_tokens})")
    request = CompletionRequest(
        messages=[
            Message(role=Role.system, content=system_prompt),
            Message(role=Role.user, content=user_prompt),
        ],
        temperature=0.0,
        max_tokens=config.max_completion_tokens,
    )
    response = await backend.complete(request)
    data = _parse_json_object(response.message.content)
    usage = response.usage or {}
    completion_tokens = usage.get("completion_tokens", 0)
    truncated = completion_tokens >= config.max_completion_tokens

    summary = data.get("summary")
    result = NodeResult(
        summary=summary if isinstance(summary, str) else "",
        findings=_parse_findings(data.get("findings")),
        prompt_tokens=usage.get("prompt_tokens", 0),
        completion_tokens=completion_tokens,
        truncated=truncated,
    )
    if config.on_progress:
        status = "TRUNCATED" if truncated else f"{len(result.findings)} finding(s)"
        config.on_progress(f"{label} -- done ({status})")
    return result


def _merge_child_findings(
    synthesized: list[Finding], children: list[ReviewNode]
) -> tuple[list[Finding], int]:
    """Synthesis calls reliably write a good narrative summary but
    unreliably populate the structured findings array: measured against
    qwen3.6, several non-truncated synthesis calls returned findings=[] even
    though their children found real, specific bugs. Union the synthesized
    findings with every immediate child's own (already-merged) findings,
    deduped by (file, line, category), so a synthesized node can never
    report fewer distinct findings than its children did -- synthesis can
    still add its own cross-cutting findings and rewrite descriptions, it
    just can't silently make real ones disappear. Returns (merged, recovered_count)."""
    seen = {(f.file, f.line, f.category.lower()) for f in synthesized}
    merged = list(synthesized)
    recovered = 0
    for child in children:
        for finding in (child.result.findings if child.result else []):
            key = (finding.file, finding.line, finding.category.lower())
            if key not in seen:
                seen.add(key)
                merged.append(finding)
                recovered += 1
    return merged, recovered


LEAF_SYSTEM_PROMPT = (
    "You are a meticulous code reviewer. You read source code and report only real, "
    "concrete issues that fall strictly under the given lens -- never vague opinions "
    "or hypotheticals. For a correctness-style lens (bugs, error handling, state, "
    "security, resources/concurrency), evidence means a specific input or call "
    "sequence that demonstrates the bug. For a design-quality lens (modularity, "
    "simplification), evidence means pointing at the exact duplicated/tangled/"
    "over-complicated code and describing concretely what a better version would "
    "look like -- not just 'this could be cleaner'. The prompt below may include a "
    "background section (imports/module-level code) and a related-signatures section "
    "(other functions this code calls or is called by, shown as signature+docstring "
    "only, never their bodies) -- these are context only. Only report findings against "
    "the section explicitly marked as the code to review; never report a finding "
    "against the background or related-signatures sections. Respond with a single "
    'JSON object and nothing else, shaped like: {"summary": str, "findings": '
    '[{"file": str, "line": int or null, "category": str, "summary": str, '
    '"failure_scenario": str}]} -- "failure_scenario" holds that concrete evidence '
    "regardless of lens. If you find nothing real, return an empty findings list -- "
    "do not invent one to seem thorough."
)

SYNTHESIS_SYSTEM_PROMPT = (
    "You are synthesizing several code-review sub-reviews into one coherent review "
    "for this subtree. Merge duplicate or overlapping findings, drop anything that "
    "isn't backed by concrete evidence (a failing input for a bug lens, or the "
    "specific code in question for a design-quality lens), and call out any issue "
    "that only becomes visible by combining two sub-reviews together. Respond with "
    'a single JSON object, shaped like: {"summary": str, "findings": [...]}, using '
    "the same finding fields as the sub-reviews."
)


def _leaf_prompt(
    method: ReviewMethod, unit: CodeUnit, header: CodeChunk | None, connected: list[ConnectedRef]
) -> str:
    parts = [f"Review lens: {method.id}\n{method.description}\n"]
    if header is not None and header.text.strip():
        parts.append(
            f"## Background context from {header.path} (imports/module-level code -- "
            "for context only, do NOT report findings against this section)\n"
            f"```python\n{header.text}\n```"
        )
    parts.append(
        f"## Code to review: {unit.name} ({unit.chunk.label()})\n"
        "This is the code to actually review. Report findings ONLY against lines in "
        f"this section.\n```python\n{unit.chunk.text}\n```"
    )
    if connected:
        callees = [c for c in connected if c.relation == "callee"]
        callers = [c for c in connected if c.relation == "caller"]
        section = [
            "## Related signatures (context only -- you have NOT seen these bodies, "
            "only their signature and docstring; NEVER report a finding against code "
            "in this section, only use it to judge whether the reviewed code above "
            "correctly calls/is called by them)"
        ]
        for ref in callees:
            section.append(f"- (called by the code above) `{ref.path}`: {ref.signature.strip()}")
            if ref.docstring:
                section.append(f'  """{ref.docstring}"""')
        for ref in callers:
            section.append(f"- (calls the code above) `{ref.path}`: {ref.signature.strip()}")
            if ref.docstring:
                section.append(f'  """{ref.docstring}"""')
        parts.append("\n".join(section))
    return "\n\n".join(parts)


def _synthesis_prompt(goal: str, cross_axis: str, children: list[ReviewNode]) -> str:
    parts = [f"Goal: {goal}\nYou are combining {len(children)} sub-reviews across {cross_axis}.\n"]
    for child in children:
        result = child.result
        assert result is not None
        findings_text = (
            "\n".join(
                f"- [{f.category}] {f.file}:{f.line} -- {f.summary} ({f.failure_scenario})"
                for f in result.findings
            )
            or "(no findings)"
        )
        parts.append(
            f"--- Sub-review: {child.node_id} ---\nSummary: {result.summary}\n"
            f"Findings:\n{findings_text}\n"
        )
    return "\n".join(parts)


def _split_oversized_line(chunk: CodeChunk, tokenizer: Tokenizer, budget: int) -> list[CodeChunk]:
    """A single line (or a chunk with no newlines at all) that alone exceeds
    the budget can't be shrunk by splitting on lines -- fall back to a raw
    character split so decomposition still makes guaranteed progress."""
    text = chunk.text
    if tokenizer.count(text) <= budget:
        return [chunk]
    # Binary-search-free approximate split: halve characters until each half
    # fits, which converges in O(log n) since tokens scale with length.
    mid = len(text) // 2
    if mid == 0:
        return [chunk]
    left = CodeChunk(chunk.path, text[:mid], chunk.start_line, chunk.end_line)
    right = CodeChunk(chunk.path, text[mid:], chunk.start_line, chunk.end_line)
    return _split_oversized_line(left, tokenizer, budget) + _split_oversized_line(
        right, tokenizer, budget
    )


def _split_chunk_by_lines(chunk: CodeChunk, tokenizer: Tokenizer, budget: int) -> list[CodeChunk]:
    lines = chunk.text.splitlines(keepends=True)
    groups: list[CodeChunk] = []
    current: list[str] = []
    current_tokens = 0
    start = chunk.start_line
    line_no = chunk.start_line
    for line in lines:
        line_tokens = tokenizer.count(line)
        if line_tokens > budget:
            if current:
                groups.append(CodeChunk(chunk.path, "".join(current), start, line_no - 1))
                current, current_tokens = [], 0
            groups.extend(
                _split_oversized_line(
                    CodeChunk(chunk.path, line, line_no, line_no), tokenizer, budget
                )
            )
            start = line_no + 1
            line_no += 1
            continue
        if current and current_tokens + line_tokens > budget:
            groups.append(CodeChunk(chunk.path, "".join(current), start, line_no - 1))
            current, current_tokens, start = [], 0, line_no
        current.append(line)
        current_tokens += line_tokens
        line_no += 1
    if current:
        groups.append(CodeChunk(chunk.path, "".join(current), start, line_no - 1))
    return groups


_DefOrClass = ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef


def _iter_top_level_blocks(
    body: list[ast.stmt], chunk: CodeChunk
) -> list[tuple[_DefOrClass | None, CodeChunk]]:
    """Slice `chunk` at the boundaries of each statement in `body` (an AST
    statement list whose linenos are 1-indexed from the top of chunk.text).
    Slicing from where the previous statement ended (not the next node's own
    lineno) means decorators, leading comments, and blank lines between
    statements are absorbed into whichever unit follows them. Returns
    (ast_node, chunk) pairs; node is None for a non-def/class statement."""
    lines = chunk.text.splitlines(keepends=True)
    blocks: list[tuple[_DefOrClass | None, CodeChunk]] = []
    prev_end = 0
    for node in body:
        end_lineno = getattr(node, "end_lineno", None) or node.lineno
        block_chunk = CodeChunk(
            chunk.path,
            "".join(lines[prev_end:end_lineno]),
            chunk.start_line + prev_end,
            chunk.start_line + end_lineno - 1,
        )
        kind_node = node if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) else None
        blocks.append((kind_node, block_chunk))
        prev_end = end_lineno
    if prev_end < len(lines):
        blocks.append(
            (
                None,
                CodeChunk(
                    chunk.path,
                    "".join(lines[prev_end:]),
                    chunk.start_line + prev_end,
                    chunk.start_line + len(lines) - 1,
                ),
            )
        )
    return blocks


def _extract_signature(chunk: CodeChunk, node: ast.AST) -> str:
    """The header line(s) of a def/class -- from the start of `chunk`
    (which already includes any decorators via _iter_top_level_blocks'
    prev_end slicing) through the line before the body starts. Raw source
    text, not ast.unparse, to preserve exact original formatting."""
    body = getattr(node, "body", None)
    if not body:
        return chunk.text
    lines = chunk.text.splitlines(keepends=True)
    # +1 guards one-line defs/classes (`def f(): return 1`) where the body
    # starts on the same physical line as the header -- can't cleanly
    # separate them, so take at least that first line as the signature.
    relative_end = max(1, body[0].lineno - chunk.start_line)
    return "".join(lines[:relative_end]) if relative_end <= len(lines) else chunk.text


def _collect_call_names(node: ast.AST) -> set[str]:
    """Best-effort, unqualified call-name collection: `foo()` -> "foo",
    `obj.method()` -> "method". No type resolution -- consistent with how
    lightweight the rest of this tool's approach already is."""
    names: set[str] = set()
    for sub in ast.walk(node):
        if isinstance(sub, ast.Call):
            func = sub.func
            if isinstance(func, ast.Name):
                names.add(func.id)
            elif isinstance(func, ast.Attribute):
                names.add(func.attr)
    return names


def _make_unit(
    path: Path, node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef, chunk: CodeChunk
) -> CodeUnit:
    kind: Literal["function", "class"] = "class" if isinstance(node, ast.ClassDef) else "function"
    return CodeUnit(
        path=path,
        name=node.name,
        kind=kind,
        chunk=chunk,
        signature=_extract_signature(chunk, node),
        docstring=ast.get_docstring(node) or "",
        calls=frozenset(_collect_call_names(node)),
    )


def _extract_file_units(path: Path, text: str) -> tuple[CodeChunk | None, list[CodeUnit]]:
    """Parses one file into (header, units): `header` is the module-level
    code that isn't a function/class definition (imports, constants, module
    docstring) -- always shown as free background context, never its own
    leaf. `units` is one CodeUnit per top-level function/class (a class's
    methods stay inside its single chunk, not extracted separately).

    If the file has no top-level def/class at all (a pure-constants module,
    an empty file, or one that fails to parse), a single pseudo-unit is
    synthesized from whatever content exists so the file still gets
    reviewed as one leaf instead of silently dropping out."""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        tree = None

    whole_chunk = CodeChunk(path, text)
    header_parts: list[str] = []
    header_start: int | None = None
    header_end: int | None = None
    units: list[CodeUnit] = []

    if tree is not None:
        for node, block_chunk in _iter_top_level_blocks(tree.body, whole_chunk):
            if node is None:
                if not block_chunk.text.strip():
                    continue
                header_parts.append(block_chunk.text)
                header_start = header_start if header_start is not None else block_chunk.start_line
                header_end = block_chunk.end_line
                continue
            units.append(_make_unit(path, node, block_chunk))

    header = (
        CodeChunk(path, "".join(header_parts), header_start or 1, header_end)
        if header_parts
        else None
    )

    if units:
        return header, units

    pseudo_chunk = header if header is not None else whole_chunk
    if not pseudo_chunk.text.strip():
        return None, []
    pseudo = CodeUnit(
        path=path, name=path.stem, kind="function", chunk=pseudo_chunk, signature="",
        docstring="", calls=frozenset(),
    )
    return None, [pseudo]


def _extract_all_units(target_files: list[Path]) -> tuple[dict[Path, CodeChunk | None], list[CodeUnit]]:
    header_by_path: dict[Path, CodeChunk | None] = {}
    all_units: list[CodeUnit] = []
    for path in target_files:
        header, units = _extract_file_units(path, path.read_text(encoding="utf-8"))
        header_by_path[path] = header
        all_units.extend(units)
    return header_by_path, all_units


def _build_definition_map(
    units: list[CodeUnit],
) -> tuple[dict[str, list[CodeUnit]], dict[str, list[CodeUnit]]]:
    """(definition_map, caller_index), both name -> list[CodeUnit], built in
    one O(units + total call-sites) pass -- no O(n^2) reverse scan.
    definition_map[name] = units defining that name (usually one, possibly
    several across files with a name collision). caller_index[name] = units
    whose body calls that name, i.e. the callers of whatever defines it."""
    definition_map: dict[str, list[CodeUnit]] = {}
    caller_index: dict[str, list[CodeUnit]] = {}
    for unit in units:
        definition_map.setdefault(unit.name, []).append(unit)
    for unit in units:
        for name in unit.calls:
            caller_index.setdefault(name, []).append(unit)
    return definition_map, caller_index


def _assemble_connected_context(
    unit: CodeUnit,
    definition_map: dict[str, list[CodeUnit]],
    caller_index: dict[str, list[CodeUnit]],
    tokenizer: Tokenizer,
    remaining_budget: int,
) -> tuple[list[ConnectedRef], int, int]:
    """Callees (what `unit` calls) then callers (what calls `unit`), each
    group alphabetical by (path, name) for determinism, deduped against
    `unit` itself and against each other. Greedily included while they fit
    remaining_budget -- a candidate that doesn't fit doesn't stop the scan,
    since a smaller one later might still fit. Returns
    (included_refs, included_count, omitted_count)."""
    seen = {(unit.path, unit.name)}
    callees: list[CodeUnit] = []
    for name in sorted(unit.calls):
        for candidate in definition_map.get(name, []):
            key = (candidate.path, candidate.name)
            if key in seen:
                continue
            seen.add(key)
            callees.append(candidate)
    callers: list[CodeUnit] = []
    for candidate in caller_index.get(unit.name, []):
        key = (candidate.path, candidate.name)
        if key in seen:
            continue
        seen.add(key)
        callers.append(candidate)
    callees.sort(key=lambda u: (str(u.path), u.name))
    callers.sort(key=lambda u: (str(u.path), u.name))

    included: list[ConnectedRef] = []
    included_count = 0
    omitted_count = 0
    used = 0
    for relation, group in (("callee", callees), ("caller", callers)):
        for candidate in group:
            ref = ConnectedRef(candidate.path, candidate.name, candidate.signature, candidate.docstring, relation)  # type: ignore[arg-type]
            cost = tokenizer.count(f"{ref.path}: {ref.signature}\n{ref.docstring}")
            if used + cost > remaining_budget:
                omitted_count += 1
                continue
            included.append(ref)
            used += cost
            included_count += 1
    return included, included_count, omitted_count


def _wrap_as_units(parent: CodeUnit, chunks: list[CodeChunk]) -> list[CodeUnit]:
    return [
        CodeUnit(
            path=parent.path,
            name=f"{parent.name} (part {i + 1}/{len(chunks)})",
            kind=parent.kind,
            chunk=chunk,
            signature=parent.signature,
            docstring=parent.docstring,
            calls=parent.calls,
        )
        for i, chunk in enumerate(chunks)
    ]


def _split_oversized_unit(
    unit: CodeUnit, tokenizer: Tokenizer, budget: int
) -> list[CodeUnit]:
    """A single unit too big to review as-is. A class splits into one piece
    per method (each carrying the class's own header/docstring baked in as
    free context, so a method-piece still knows what class it's in); a
    function, or a still-oversized method piece, falls back to the existing
    raw line/character splitters, which already guarantee termination."""
    if unit.kind == "class":
        try:
            tree = ast.parse(unit.chunk.text)
        except SyntaxError:
            tree = None
        class_node = next(
            (n for n in tree.body if isinstance(n, ast.ClassDef)), None
        ) if tree is not None else None
        if class_node is not None and class_node.body:
            class_header = _extract_signature(unit.chunk, class_node)
            class_doc = ast.get_docstring(class_node) or ""
            prefix = class_header + (f'    """{class_doc}"""\n' if class_doc else "")
            pieces: list[CodeUnit] = []
            for node, block_chunk in _iter_top_level_blocks(class_node.body, unit.chunk):
                if node is None:
                    continue  # class-level non-method statement -- dropped, not a leaf
                combined = CodeChunk(
                    unit.path, prefix + block_chunk.text, block_chunk.start_line, block_chunk.end_line
                )
                piece = CodeUnit(
                    path=unit.path,
                    name=f"{unit.name}.{getattr(node, 'name', '?')}",
                    kind="method_group",
                    chunk=combined,
                    signature=_extract_signature(block_chunk, node),
                    docstring=ast.get_docstring(node) or "",
                    calls=frozenset(_collect_call_names(node)),
                )
                if tokenizer.count(combined.text) > budget:
                    pieces.extend(_wrap_as_units(piece, _split_chunk_by_lines(block_chunk, tokenizer, budget)))
                else:
                    pieces.append(piece)
            if pieces:
                return pieces
    return _wrap_as_units(unit, _split_chunk_by_lines(unit.chunk, tokenizer, budget))


async def _execute_leaf(
    node_id: str,
    method: ReviewMethod,
    unit: CodeUnit,
    header: CodeChunk | None,
    depth: int,
    config: ReviewConfig,
) -> ReviewNode:
    node = ReviewNode(
        node_id=node_id,
        goal=f"[{method.id}] review {unit.name} ({unit.path})",
        method=method.id,
        depth=depth,
        scope=(unit.chunk,),
        header_context=header,
    )
    remaining = config.leaf_token_budget - config.tokenizer.count(unit.chunk.text)
    if header is not None:
        remaining -= config.tokenizer.count(header.text)
    connected, included, omitted = _assemble_connected_context(
        unit, config.definition_map, config.caller_index, config.tokenizer, max(remaining, 0)
    )
    node.connected = tuple(connected)
    node.result = await _complete_result(
        config, config.backend, LEAF_SYSTEM_PROMPT, _leaf_prompt(method, unit, header, connected), node_id
    )
    node.result.connected_included = included
    node.result.connected_omitted = omitted
    return node


async def _build_unit_subtree(
    node_id: str,
    method: ReviewMethod,
    unit: CodeUnit,
    header: CodeChunk | None,
    depth: int,
    config: ReviewConfig,
) -> ReviewNode:
    if config.is_atom(unit, header, config.tokenizer, config.leaf_token_budget):
        return await _execute_leaf(node_id, method, unit, header, depth, config)

    pieces = _split_oversized_unit(unit, config.tokenizer, config.leaf_token_budget)
    if len(pieces) <= 1:
        # Couldn't actually shrink further -- execute directly rather than
        # recurse on an unchanged unit.
        return await _execute_leaf(node_id, method, unit, header, depth, config)

    node = ReviewNode(
        node_id=node_id,
        goal=f"[{method.id}] review oversized {unit.kind} {unit.name} ({unit.path})",
        method=method.id,
        depth=depth,
        scope=(unit.chunk,),
    )
    for index, piece in enumerate(pieces):
        child = await _build_unit_subtree(f"{node_id}/p{index}", method, piece, header, depth + 1, config)
        node.children.append(child)
    node.result = await _complete_result(
        config,
        config.synthesis_backend,
        SYNTHESIS_SYSTEM_PROMPT,
        _synthesis_prompt(node.goal, f"pieces of the oversized {unit.kind} '{unit.name}'", node.children),
        f"{node_id} (synthesis)",
    )
    node.result.findings, node.result.recovered_from_children = _merge_child_findings(
        node.result.findings, node.children
    )
    return node


async def _build_method_subtree(node_id: str, method: ReviewMethod, config: ReviewConfig, depth: int) -> ReviewNode:
    if len(config.units) == 1:
        # Only one function/class in scope for this whole run -- synthesizing
        # "across 1 sub-review" adds nothing and doubles the LLM calls for
        # what's otherwise a trivial case. Promote the single unit's own
        # subtree straight to the method level, same shortcut the old
        # token-budget-atomic scope check used to take.
        unit = config.units[0]
        header = config.header_by_path.get(unit.path)
        return await _build_unit_subtree(node_id, method, unit, header, depth, config)

    node = ReviewNode(
        node_id=node_id,
        goal=f"[{method.id}] review {len(config.units)} function/class unit(s)",
        method=method.id,
        depth=depth,
        scope=tuple(u.chunk for u in config.units),
    )
    for unit in config.units:
        header = config.header_by_path.get(unit.path)
        child_id = f"{node_id}/{unit.path.stem}.{unit.name}"
        child = await _build_unit_subtree(child_id, method, unit, header, depth + 1, config)
        node.children.append(child)

    node.result = await _complete_result(
        config,
        config.synthesis_backend,
        SYNTHESIS_SYSTEM_PROMPT,
        _synthesis_prompt(node.goal, f"functions/classes reviewed under the '{method.id}' lens", node.children),
        f"{node_id} (synthesis)",
    )
    node.result.findings, node.result.recovered_from_children = _merge_child_findings(
        node.result.findings, node.children
    )
    return node


async def run_review(
    target_files: list[Path],
    *,
    methods: tuple[ReviewMethod, ...] = DEFAULT_METHODS,
    backend: LLMBackend | None = None,
    synthesis_backend: LLMBackend | None = None,
    tokenizer: Tokenizer | None = None,
    leaf_token_budget: int = DEFAULT_LEAF_TOKEN_BUDGET,
    max_completion_tokens: int = 30000,
    is_atom: AtomCheck = token_budget_atom_check,
    model: str | None = None,
    synthesis_model: str | None = None,
    think: ThinkLevel = "low",
    synthesis_think: ThinkLevel | None = None,
    on_progress: Callable[[str], None] | None = None,
) -> ReviewNode:
    backend = backend or _default_backend(model, think)
    # synthesis_backend defaults to the same backend as leaves -- nothing
    # changes unless a caller explicitly asks for a stronger model on
    # synthesis nodes via synthesis_model/synthesis_backend.
    if synthesis_backend is None:
        synthesis_backend = (
            _default_backend(synthesis_model, synthesis_think if synthesis_think is not None else think)
            if synthesis_model is not None
            else backend
        )
    tokenizer = tokenizer or create_tokenizer(getattr(backend, "model_name", ""))

    header_by_path, units = _extract_all_units(target_files)
    definition_map, caller_index = _build_definition_map(units)
    config = ReviewConfig(
        backend=backend,
        synthesis_backend=synthesis_backend,
        tokenizer=tokenizer,
        units=units,
        header_by_path=header_by_path,
        definition_map=definition_map,
        caller_index=caller_index,
        leaf_token_budget=leaf_token_budget,
        max_completion_tokens=max_completion_tokens,
        on_progress=on_progress,
        is_atom=is_atom,
    )

    root = ReviewNode(
        node_id="root",
        goal=f"Review {len(target_files)} file(s): {', '.join(str(p) for p in target_files)}",
        method=None,
        depth=0,
        scope=tuple(u.chunk for u in units),
    )
    for method in methods:
        method_node = await _build_method_subtree(f"root/{method.id}", method, config, 1)
        root.children.append(method_node)

    root.result = await _complete_result(
        config,
        config.synthesis_backend,
        SYNTHESIS_SYSTEM_PROMPT,
        _synthesis_prompt(root.goal, "different analysis lenses over the same code", root.children),
        "root (synthesis)",
    )
    root.result.findings, root.result.recovered_from_children = _merge_child_findings(
        root.result.findings, root.children
    )
    return root


def _iter_nodes(node: ReviewNode) -> list[ReviewNode]:
    nodes = [node]
    for child in node.children:
        nodes.extend(_iter_nodes(child))
    return nodes


def render_markdown(root: ReviewNode) -> str:
    lines = ["# Code review", "", root.result.summary if root.result else "", ""]
    all_nodes = _iter_nodes(root)
    truncated_nodes = [n for n in all_nodes if n.result and n.result.truncated]
    recovered_nodes = [n for n in all_nodes if n.result and n.result.recovered_from_children]
    omitted_nodes = [n for n in all_nodes if n.result and n.result.connected_omitted]
    if truncated_nodes or recovered_nodes or omitted_nodes:
        lines.append("## Diagnostics")
        lines.append("")
    if truncated_nodes:
        lines.append(
            f"**{len(truncated_nodes)} node(s) hit max_completion_tokens** -- their answer "
            "may be missing, not clean:"
        )
        for node in truncated_nodes:
            lines.append(f"- `{node.node_id}` ({node.goal})")
        lines.append("")
    if recovered_nodes:
        lines.append(
            "**Findings recovered from children that synthesis itself dropped** "
            "(synthesis wrote a summary but an incomplete findings list):"
        )
        for node in recovered_nodes:
            count = node.result.recovered_from_children if node.result else 0
            lines.append(f"- `{node.node_id}`: {count} finding(s) recovered")
        lines.append("")
    if omitted_nodes:
        lines.append("**Connected context omitted due to budget on some node(s):**")
        for node in omitted_nodes:
            included = node.result.connected_included if node.result else 0
            omitted = node.result.connected_omitted if node.result else 0
            lines.append(f"- `{node.node_id}`: {included} included / {omitted} omitted")
        lines.append("")
    findings = root.result.findings if root.result else []
    if not findings:
        lines.append("No findings survived synthesis.")
    else:
        lines.append(f"## Findings ({len(findings)})")
        lines.append("")
        for finding in findings:
            location = f"{finding.file}:{finding.line}" if finding.line else finding.file
            lines.append(f"### [{finding.category}] {location}")
            lines.append("")
            lines.append(finding.summary)
            lines.append("")
            if finding.failure_scenario:
                lines.append(f"**Failure scenario:** {finding.failure_scenario}")
                lines.append("")
    return "\n".join(lines)


def _iter_python_files(target_dir: Path) -> list[Path]:
    return sorted(p for p in target_dir.glob("*.py") if p.is_file())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Goal-tree code review prototype.")
    parser.add_argument("--target-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("review-runs"))
    parser.add_argument("--model", default=None, help="Model for leaf calls (one function/class + context)")
    parser.add_argument(
        "--synthesis-model",
        default=None,
        help="Model for synthesis calls (method/oversized-unit/root) -- defaults to --model "
        "when unset, so a stronger model can review sub-reviews without changing leaf cost",
    )
    parser.add_argument("--leaf-token-budget", type=int, default=DEFAULT_LEAF_TOKEN_BUDGET)
    parser.add_argument("--max-completion-tokens", type=int, default=30000)
    parser.add_argument(
        "--think",
        choices=("none", "low", "medium", "high"),
        default="low",
        help="Ollama reasoning effort for qwen3.x-style thinking models (default: low)",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress per-node progress output")
    args = parser.parse_args(argv)

    target_files = _iter_python_files(args.target_dir)
    if not target_files:
        parser.error(f"No .py files found directly in {args.target_dir}")

    def _report_progress(message: str) -> None:
        print(f"[{time.strftime('%H:%M:%S')}] {message}", file=sys.stderr)

    think: ThinkLevel = False if args.think == "none" else args.think
    root = asyncio.run(
        run_review(
            target_files,
            leaf_token_budget=args.leaf_token_budget,
            max_completion_tokens=args.max_completion_tokens,
            model=args.model,
            synthesis_model=args.synthesis_model,
            think=think,
            on_progress=None if args.quiet else _report_progress,
        )
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "tree.json").write_text(
        json.dumps(root.to_dict(), indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "REVIEW.md").write_text(render_markdown(root) + "\n", encoding="utf-8")
    print(f"Wrote {args.output_dir / 'tree.json'} and {args.output_dir / 'REVIEW.md'}")
    truncated = root.truncated_node_ids()
    if truncated:
        print(
            f"WARNING: {len(truncated)} node(s) hit max_completion_tokens "
            f"(--max-completion-tokens {args.max_completion_tokens}) and may not have "
            "produced a real answer -- consider raising the budget and re-running.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
