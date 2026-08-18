from __future__ import annotations

import json

import pytest

from chef_human import review_tree
from chef_human.llm.backend import CompletionRequest, CompletionResponse, Message, Role
from chef_human.review_tree import (
    CodeChunk,
    CodeUnit,
    Finding,
    ReviewMethod,
    render_markdown,
    run_review,
)


class ApproxTokenizer:
    def count(self, text: str) -> int:
        return max(1, len(text) // 4)


class ScriptedBackend:
    """Returns a scripted JSON response per call, keyed by call order."""

    def __init__(self, responses: list[dict]) -> None:
        self._responses = list(responses)
        self.calls: list[CompletionRequest] = []
        self.model_name = "scripted"

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        self.calls.append(request)
        data = self._responses.pop(0)
        return CompletionResponse(
            message=Message(role=Role.assistant, content=json.dumps(data)),
            usage={"prompt_tokens": 10, "completion_tokens": 5},
        )


class CountingBackend:
    """Returns a distinct, deterministic response per call without needing
    the test to predict exact call counts up front."""

    def __init__(self) -> None:
        self.calls: list[CompletionRequest] = []
        self.model_name = "counting"

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        self.calls.append(request)
        n = len(self.calls)
        data = {"summary": f"call-{n}", "findings": [_finding(summary=f"bug-{n}", line=n)]}
        return CompletionResponse(
            message=Message(role=Role.assistant, content=json.dumps(data)),
            usage={"prompt_tokens": 10, "completion_tokens": 5},
        )


def _finding(file: str = "a.py", summary: str = "bug", line: int = 3) -> dict:
    return {
        "file": file,
        "line": line,
        "category": "correctness",
        "summary": summary,
        "failure_scenario": "call f(0)",
    }


def _make_functions_source(n: int, body_lines: int = 2) -> str:
    parts = []
    for i in range(n):
        body = "\n".join(f"    x{j} = {j}" for j in range(body_lines))
        parts.append(f"def func_{i}():\n{body}\n    return x0\n")
    return "\n\n".join(parts) + "\n"


class TestUnitExtraction:
    def test_functions_and_classes_become_units(self, tmp_path):
        path = tmp_path / "a.py"
        source = "def f():\n    return 1\n\n\nclass C:\n    def m(self):\n        return 2\n"
        header, units = review_tree._extract_file_units(path, source)

        assert header is None
        names = {u.name for u in units}
        assert names == {"f", "C"}
        c_unit = next(u for u in units if u.name == "C")
        assert c_unit.kind == "class"
        assert "def m" in c_unit.chunk.text

    def test_decorators_attach_to_their_function(self, tmp_path):
        path = tmp_path / "a.py"
        source = "@dec_one\n@dec_two\ndef decorated():\n    return 1\n"
        _, units = review_tree._extract_file_units(path, source)

        assert len(units) == 1
        unit = units[0]
        assert "@dec_one" in unit.chunk.text
        assert "@dec_two" in unit.chunk.text
        assert "@dec_one" in unit.signature
        assert "@dec_two" in unit.signature

    def test_docstrings_extracted_for_function_and_class(self, tmp_path):
        path = tmp_path / "a.py"
        source = (
            'def f():\n    """f doc"""\n    return 1\n\n\n'
            'class C:\n    """C doc"""\n    def m(self):\n        return 2\n'
        )
        _, units = review_tree._extract_file_units(path, source)

        f_unit = next(u for u in units if u.name == "f")
        c_unit = next(u for u in units if u.name == "C")
        assert f_unit.docstring == "f doc"
        assert c_unit.docstring == "C doc"

    def test_nested_function_is_not_separately_extracted(self, tmp_path):
        path = tmp_path / "a.py"
        source = "def outer():\n    def inner():\n        return 1\n    return inner()\n"
        _, units = review_tree._extract_file_units(path, source)

        assert len(units) == 1
        assert units[0].name == "outer"
        assert "def inner" in units[0].chunk.text

    def test_imports_and_constants_fold_into_header_not_units(self, tmp_path):
        path = tmp_path / "a.py"
        source = '"""module doc"""\nimport os\nX = 1\n\n\ndef f():\n    return os.getcwd()\n'
        header, units = review_tree._extract_file_units(path, source)

        assert header is not None
        assert "import os" in header.text
        assert "X = 1" in header.text
        assert "module doc" in header.text
        assert len(units) == 1
        assert "import os" not in units[0].chunk.text

    def test_multiline_signature_captured_correctly(self, tmp_path):
        path = tmp_path / "a.py"
        source = "def f(\n    a=1,\n    b=2,\n):\n    return a + b\n"
        _, units = review_tree._extract_file_units(path, source)

        signature = units[0].signature
        assert "def f(" in signature
        assert "b=2," in signature
        assert "return a" not in signature

    def test_file_with_no_functions_becomes_one_pseudo_unit(self, tmp_path):
        path = tmp_path / "consts.py"
        source = "X = 1\nY = 2\n"
        header, units = review_tree._extract_file_units(path, source)

        assert header is None
        assert len(units) == 1
        assert units[0].name == "consts"
        assert "X = 1" in units[0].chunk.text

    def test_empty_file_produces_no_units(self, tmp_path):
        path = tmp_path / "empty.py"
        header, units = review_tree._extract_file_units(path, "")
        assert header is None
        assert units == []

    def test_unparseable_file_falls_back_to_one_opaque_unit(self, tmp_path):
        path = tmp_path / "broken.py"
        source = "def f(\n"  # syntax error
        header, units = review_tree._extract_file_units(path, source)

        assert header is None
        assert len(units) == 1
        assert units[0].chunk.text == source


class TestDefinitionMapResolution:
    def test_cross_file_callee_and_caller_resolution(self, tmp_path):
        file_a = tmp_path / "a.py"
        file_b = tmp_path / "b.py"
        _, units_a = review_tree._extract_file_units(file_a, "def caller():\n    return callee()\n")
        _, units_b = review_tree._extract_file_units(file_b, "def callee():\n    return 1\n")
        units = units_a + units_b

        definition_map, caller_index = review_tree._build_definition_map(units)

        assert [u.path for u in definition_map["callee"]] == [file_b]
        assert [u.path for u in caller_index["callee"]] == [file_a]

    def test_self_calls_excluded_from_connected_context(self, tmp_path):
        path = tmp_path / "a.py"
        _, units = review_tree._extract_file_units(path, "def f():\n    return f()\n")
        definition_map, caller_index = review_tree._build_definition_map(units)

        refs, included, omitted = review_tree._assemble_connected_context(
            units[0], definition_map, caller_index, ApproxTokenizer(), remaining_budget=10_000
        )

        assert refs == []
        assert included == 0
        assert omitted == 0

    def test_name_collisions_across_files_produce_multiple_candidates(self, tmp_path):
        file_a = tmp_path / "a.py"
        file_b = tmp_path / "b.py"
        _, units_a = review_tree._extract_file_units(file_a, "def helper():\n    return 1\n")
        _, units_b = review_tree._extract_file_units(file_b, "def helper():\n    return 2\n")

        definition_map, _ = review_tree._build_definition_map(units_a + units_b)

        assert len(definition_map["helper"]) == 2

    def test_unresolvable_calls_produce_no_connected_ref(self, tmp_path):
        path = tmp_path / "a.py"
        _, units = review_tree._extract_file_units(path, "def f():\n    return len(x)\n")
        definition_map, caller_index = review_tree._build_definition_map(units)

        refs, included, omitted = review_tree._assemble_connected_context(
            units[0], definition_map, caller_index, ApproxTokenizer(), remaining_budget=10_000
        )

        assert refs == []
        assert included == 0
        assert omitted == 0


class TestConnectedContextBudget:
    def _unit_and_maps(self, tmp_path):
        file_a = tmp_path / "a.py"
        file_b = tmp_path / "b.py"
        _, units_a = review_tree._extract_file_units(
            file_a, "def caller():\n    one()\n    two()\n    three()\n    return None\n"
        )
        _, units_b = review_tree._extract_file_units(
            file_b,
            'def one():\n    """one doc"""\n    return 1\n\n\n'
            'def two():\n    """two doc"""\n    return 2\n\n\n'
            'def three():\n    """three doc"""\n    return 3\n',
        )
        units = units_a + units_b
        definition_map, caller_index = review_tree._build_definition_map(units)
        caller_unit = next(u for u in units if u.name == "caller")
        return caller_unit, definition_map, caller_index

    def test_tight_budget_omits_some_candidates(self, tmp_path):
        unit, definition_map, caller_index = self._unit_and_maps(tmp_path)

        refs, included, omitted = review_tree._assemble_connected_context(
            unit, definition_map, caller_index, ApproxTokenizer(), remaining_budget=1
        )

        assert included + omitted == 3
        assert omitted > 0

    def test_included_and_omitted_counts_sum_to_total_candidates(self, tmp_path):
        unit, definition_map, caller_index = self._unit_and_maps(tmp_path)

        refs, included, omitted = review_tree._assemble_connected_context(
            unit, definition_map, caller_index, ApproxTokenizer(), remaining_budget=5
        )

        assert len(refs) == included
        assert included + omitted == 3

    def test_generous_budget_includes_all_candidates_alphabetically(self, tmp_path):
        unit, definition_map, caller_index = self._unit_and_maps(tmp_path)

        refs, included, omitted = review_tree._assemble_connected_context(
            unit, definition_map, caller_index, ApproxTokenizer(), remaining_budget=10_000
        )

        assert omitted == 0
        assert [r.name for r in refs] == ["one", "three", "two"]
        assert all(r.relation == "callee" for r in refs)

    def test_zero_remaining_budget_omits_everything(self, tmp_path):
        unit, definition_map, caller_index = self._unit_and_maps(tmp_path)

        refs, included, omitted = review_tree._assemble_connected_context(
            unit, definition_map, caller_index, ApproxTokenizer(), remaining_budget=0
        )

        assert refs == []
        assert included == 0
        assert omitted == 3


class TestOversizedUnitFallback:
    def test_oversized_class_splits_into_per_method_leaves_with_class_context(self, tmp_path):
        path = tmp_path / "a.py"
        source = (
            'class Big:\n    """class doc"""\n\n'
            "    def method_a(self):\n        return 1\n\n"
            "    def method_b(self):\n        return 2\n\n"
            "    def method_c(self):\n        return 3\n"
        )
        _, units = review_tree._extract_file_units(path, source)
        class_unit = units[0]
        assert class_unit.kind == "class"

        pieces = review_tree._split_oversized_unit(class_unit, ApproxTokenizer(), budget=20)

        assert len(pieces) == 3
        assert all(p.kind == "method_group" for p in pieces)
        for piece in pieces:
            assert "class Big" in piece.chunk.text
            assert "class doc" in piece.chunk.text
        assert {p.name for p in pieces} == {"Big.method_a", "Big.method_b", "Big.method_c"}

    def test_oversized_standalone_function_falls_back_to_line_splitting(self, tmp_path):
        tokenizer = ApproxTokenizer()
        huge_body = "\n".join(f"    x{j} = {j}" for j in range(100))
        source = f"def huge():\n{huge_body}\n    return 0\n"
        unit = CodeUnit(
            path=tmp_path / "a.py", name="huge", kind="function",
            chunk=CodeChunk(tmp_path / "a.py", source), signature="def huge():\n",
            docstring="", calls=frozenset(),
        )

        pieces = review_tree._split_oversized_unit(unit, tokenizer, budget=20)

        assert len(pieces) > 1
        assert all(p.kind == "function" for p in pieces)
        all_text = "".join(p.chunk.text for p in pieces)
        assert "def huge" in all_text
        assert "return 0" in all_text

    def test_oversized_method_within_oversized_class_recurses_to_line_splitting(self, tmp_path):
        path = tmp_path / "a.py"
        huge_body = "\n".join(f"        x{j} = {j}" for j in range(100))
        source = f"class Big:\n    def small(self):\n        return 1\n\n    def huge(self):\n{huge_body}\n        return 0\n"
        _, units = review_tree._extract_file_units(path, source)
        class_unit = units[0]

        pieces = review_tree._split_oversized_unit(class_unit, ApproxTokenizer(), budget=15)

        # small() stays as one method_group piece, huge() gets split further
        assert len(pieces) > 2
        names = [p.name for p in pieces]
        assert any(n == "Big.small" for n in names)
        assert sum(1 for n in names if n.startswith("Big.huge")) > 1


class TestHeaderContextPropagation:
    def test_header_populated_for_file_with_imports_and_function(self, tmp_path):
        path = tmp_path / "a.py"
        source = "import os\n\n\ndef f():\n    return os.getcwd()\n"
        header, units = review_tree._extract_file_units(path, source)

        assert header is not None
        assert "import os" in header.text
        assert len(units) == 1

    def test_functions_only_file_yields_no_header(self, tmp_path):
        path = tmp_path / "a.py"
        source = "def f():\n    return 1\n\n\ndef g():\n    return 2\n"
        header, units = review_tree._extract_file_units(path, source)

        assert header is None
        assert len(units) == 2

    @pytest.mark.asyncio
    async def test_header_never_becomes_its_own_leaf(self, tmp_path):
        target = tmp_path / "a.py"
        target.write_text("import os\n\n\ndef f():\n    return os.getcwd()\n")
        methods = (ReviewMethod("m1", "desc1"),)
        backend = ScriptedBackend(
            [
                {"summary": "leaf", "findings": []},
                {"summary": "root", "findings": []},
            ]
        )

        root = await run_review(
            [target], methods=methods, backend=backend, tokenizer=ApproxTokenizer(),
            leaf_token_budget=10_000,
        )

        # one unit (f) -> the single-unit shortcut means method_node IS the
        # leaf directly, no separate header-only node anywhere
        method_node = root.children[0]
        assert method_node.children == []
        assert method_node.header_context is not None
        assert "import os" in method_node.header_context.text


class TestRunReview:
    @pytest.mark.asyncio
    async def test_single_function_uses_method_level_shortcut(self, tmp_path):
        target = tmp_path / "a.py"
        target.write_text("def f():\n    return 1\n")
        methods = (ReviewMethod("m1", "desc1"), ReviewMethod("m2", "desc2"))
        backend = ScriptedBackend(
            [
                {"summary": "m1 leaf", "findings": [_finding(summary="m1 bug")]},
                {"summary": "m2 leaf", "findings": []},
                {"summary": "root synthesis", "findings": [_finding(summary="m1 bug")]},
            ]
        )

        root = await run_review(
            [target], methods=methods, backend=backend, tokenizer=ApproxTokenizer(),
            leaf_token_budget=10_000,
        )

        # one call per method (leaf, no separate method-synthesis since
        # there's only one unit) + one root synthesis call
        assert len(backend.calls) == 3
        assert len(root.children) == 2
        assert root.children[0].children == []
        assert root.result is not None
        assert root.result.summary == "root synthesis"
        assert [f.summary for f in root.result.findings] == ["m1 bug"]

    @pytest.mark.asyncio
    async def test_multiple_functions_produce_one_leaf_each(self, tmp_path):
        target = tmp_path / "multi.py"
        target.write_text(_make_functions_source(4))
        methods = (ReviewMethod("m1", "desc1"),)
        backend = CountingBackend()

        root = await run_review(
            [target], methods=methods, backend=backend, tokenizer=ApproxTokenizer(),
            leaf_token_budget=10_000,
        )

        method_node = root.children[0]
        assert len(method_node.children) == 4
        assert all(child.children == [] for child in method_node.children)
        assert all(child.result and child.result.findings for child in method_node.children)
        # 4 leaves, then method synthesis (since >1 unit), then root synthesis
        assert method_node.result.summary == "call-5"
        assert root.result.summary == "call-6"

    @pytest.mark.asyncio
    async def test_custom_atom_check_that_cannot_shrink_falls_back_to_leaf(self, tmp_path):
        target = tmp_path / "a.py"
        target.write_text("def f():\n    return 1\n")
        methods = (ReviewMethod("m1", "desc1"),)
        backend = ScriptedBackend(
            [
                {"summary": "leaf", "findings": []},
                {"summary": "root synthesis", "findings": []},
            ]
        )

        root = await run_review(
            [target], methods=methods, backend=backend, tokenizer=ApproxTokenizer(),
            leaf_token_budget=10_000,
            is_atom=lambda unit, header, tok, budget: False,
        )

        # single-unit shortcut applies regardless of is_atom (only 1 unit in
        # scope), and the unit can't actually be split further, so it still
        # executes directly as a leaf rather than looping forever
        method_node = root.children[0]
        assert method_node.children == []
        assert method_node.result is not None
        assert method_node.result.summary == "leaf"

    @pytest.mark.asyncio
    async def test_no_functions_content_still_terminates(self, tmp_path):
        # A pseudo-unit with a single huge "line" (no newlines) larger than
        # the budget used to make the old splitter return content
        # unchanged, causing infinite recursion -- this is the regression
        # test for that, now via the no-functions-found pseudo-unit path.
        target = tmp_path / "minified.py"
        target.write_text("x" * 2000)
        methods = (ReviewMethod("m1", "desc1"),)
        backend = CountingBackend()

        root = await run_review(
            [target], methods=methods, backend=backend, tokenizer=ApproxTokenizer(),
            leaf_token_budget=100,
        )

        assert root.result is not None
        method_node = root.children[0]
        assert len(method_node.children) > 1
        assert all(child.children == [] for child in method_node.children)

    @pytest.mark.asyncio
    async def test_functions_across_multiple_files_get_cross_file_context(self, tmp_path):
        file_a = tmp_path / "a.py"
        file_b = tmp_path / "b.py"
        file_a.write_text("def caller():\n    return callee()\n")
        file_b.write_text('def callee():\n    """callee doc"""\n    return 1\n')
        methods = (ReviewMethod("m1", "desc1"),)
        backend = CountingBackend()

        root = await run_review(
            [file_a, file_b], methods=methods, backend=backend, tokenizer=ApproxTokenizer(),
            leaf_token_budget=10_000,
        )

        method_node = root.children[0]
        caller_leaf = next(c for c in method_node.children if "caller" in c.node_id)
        assert any(ref.name == "callee" for ref in caller_leaf.connected)


class TestMalformedResponses:
    @pytest.mark.asyncio
    async def test_non_json_response_degrades_to_empty_result(self, tmp_path):
        target = tmp_path / "a.py"
        target.write_text("x = 1\n")
        methods = (ReviewMethod("m1", "desc1"),)

        class GarbageBackend:
            model_name = "garbage"

            async def complete(self, request: CompletionRequest) -> CompletionResponse:
                return CompletionResponse(
                    message=Message(role=Role.assistant, content="not json at all"),
                    usage={"prompt_tokens": 1, "completion_tokens": 1},
                )

        root = await run_review(
            [target], methods=methods, backend=GarbageBackend(), tokenizer=ApproxTokenizer(),
            leaf_token_budget=10_000,
        )

        assert root.result is not None
        assert root.result.summary == ""
        assert root.result.findings == []

    @pytest.mark.asyncio
    async def test_malformed_finding_entries_are_skipped(self, tmp_path):
        target = tmp_path / "a.py"
        target.write_text("x = 1\n")
        methods = (ReviewMethod("m1", "desc1"),)
        backend = ScriptedBackend(
            [
                {
                    "summary": "leaf",
                    "findings": [
                        {"file": "a.py", "summary": "valid"},
                        {"file": "a.py"},  # missing summary -- skipped
                        "not even a dict",  # skipped
                        {"summary": "missing file"},  # skipped
                    ],
                },
                {"summary": "root synthesis", "findings": []},
            ]
        )

        root = await run_review(
            [target], methods=methods, backend=backend, tokenizer=ApproxTokenizer(),
            leaf_token_budget=10_000,
        )

        leaf = root.children[0]
        assert leaf.result is not None
        assert [f.summary for f in leaf.result.findings] == ["valid"]
        assert leaf.result.findings[0].category == "uncategorized"
        assert leaf.result.findings[0].line is None


class TestDefaultMethods:
    def test_includes_modularity_and_simplification_lenses(self):
        ids = {m.id for m in review_tree.DEFAULT_METHODS}
        assert "modularity" in ids
        assert "simplification" in ids


class TestProgressCallback:
    @pytest.mark.asyncio
    async def test_on_progress_fires_before_and_after_each_call(self, tmp_path):
        target = tmp_path / "a.py"
        target.write_text("x = 1\n")
        methods = (ReviewMethod("m1", "desc1"),)
        backend = ScriptedBackend(
            [
                {"summary": "leaf", "findings": []},
                {"summary": "root", "findings": []},
            ]
        )
        messages: list[str] = []

        await run_review(
            [target], methods=methods, backend=backend, tokenizer=ApproxTokenizer(),
            leaf_token_budget=10_000, on_progress=messages.append,
        )

        assert len(messages) == 4  # dispatch+done for each of the 2 calls
        assert "root/m1" in messages[0]
        assert "dispatching" in messages[0]
        assert "done" in messages[1]
        assert "root (synthesis)" in messages[2]

    @pytest.mark.asyncio
    async def test_no_progress_callback_by_default(self, tmp_path):
        target = tmp_path / "a.py"
        target.write_text("x = 1\n")
        backend = ScriptedBackend(
            [{"summary": "leaf", "findings": []} for _ in review_tree.DEFAULT_METHODS]
            + [{"summary": "root", "findings": []}]
        )

        # Should not raise even though nothing is listening.
        await run_review([target], backend=backend, tokenizer=ApproxTokenizer(), leaf_token_budget=10_000)


class TestFindingsMergeRecovery:
    @pytest.mark.asyncio
    async def test_synthesis_dropping_findings_are_recovered_from_children(self, tmp_path):
        target = tmp_path / "multi.py"
        target.write_text(_make_functions_source(4))
        methods = (ReviewMethod("m1", "desc1"),)

        class DropsFindingsBackend:
            model_name = "drops-findings"

            def __init__(self) -> None:
                self.calls = 0

            async def complete(self, request: CompletionRequest) -> CompletionResponse:
                self.calls += 1
                is_leaf = "Review lens" in request.messages[-1].content
                if is_leaf:
                    finding = _finding(file="multi.py", summary=f"bug-{self.calls}", line=self.calls)
                    data = {"summary": f"leaf {self.calls}", "findings": [finding]}
                else:
                    # synthesis: writes a summary but drops the findings array
                    data = {"summary": "synthesis dropped findings", "findings": []}
                return CompletionResponse(
                    message=Message(role=Role.assistant, content=json.dumps(data)),
                    usage={"prompt_tokens": 5, "completion_tokens": 10},
                )

        root = await run_review(
            [target], methods=methods, backend=DropsFindingsBackend(), tokenizer=ApproxTokenizer(),
            leaf_token_budget=10_000,
        )

        method_node = root.children[0]
        leaf_count = len(method_node.children)
        assert leaf_count == 4
        assert len(method_node.result.findings) == leaf_count
        assert method_node.result.recovered_from_children == leaf_count
        assert len(root.result.findings) == leaf_count
        assert root.result.recovered_from_children == leaf_count

    @pytest.mark.asyncio
    async def test_duplicate_finding_key_is_not_double_recovered(self, tmp_path):
        target = tmp_path / "a.py"
        target.write_text("def f():\n    return 1\n")
        methods = (ReviewMethod("m1", "desc1"),)
        shared = _finding(file="a.py", summary="same bug")
        backend = ScriptedBackend(
            [
                {"summary": "leaf", "findings": [shared]},
                {"summary": "root", "findings": [shared]},
            ]
        )

        root = await run_review(
            [target], methods=methods, backend=backend, tokenizer=ApproxTokenizer(),
            leaf_token_budget=10_000,
        )

        assert len(root.result.findings) == 1
        assert root.result.recovered_from_children == 0

    @pytest.mark.asyncio
    async def test_explicitly_dropped_finding_is_not_recovered(self, tmp_path):
        """A synthesis call that lists a child's finding in "dropped" (with
        a reason) is making a deliberate prioritization call -- it must NOT
        get force-recovered like an accidental JSON-omission would."""
        target = tmp_path / "multi.py"
        target.write_text(_make_functions_source(2))
        methods = (ReviewMethod("m1", "desc1"),)
        real_bug = _finding(file="multi.py", summary="real bug", line=1)
        false_positive = _finding(file="multi.py", summary="not actually a bug", line=2)

        class DropOneBackend:
            model_name = "drop-one"

            def __init__(self) -> None:
                self.calls = 0

            async def complete(self, request: CompletionRequest) -> CompletionResponse:
                self.calls += 1
                is_leaf = "Review lens" in request.messages[-1].content
                if is_leaf:
                    finding = real_bug if self.calls == 1 else false_positive
                    data = {"summary": f"leaf {self.calls}", "findings": [finding]}
                else:
                    # synthesis keeps the real bug, explicitly drops the
                    # false positive with a reason, and doesn't re-list it.
                    data = {
                        "summary": "synthesis escalated one, dropped one",
                        "findings": [{**real_bug, "severity": "high"}],
                        "dropped": [
                            {
                                "file": "multi.py",
                                "line": 2,
                                "category": false_positive["category"],
                                "reason": "not backed by concrete evidence",
                            }
                        ],
                    }
                return CompletionResponse(
                    message=Message(role=Role.assistant, content=json.dumps(data)),
                    usage={"prompt_tokens": 5, "completion_tokens": 10},
                )

        root = await run_review(
            [target], methods=methods, backend=DropOneBackend(), tokenizer=ApproxTokenizer(),
            leaf_token_budget=10_000,
        )

        method_node = root.children[0]
        assert len(method_node.result.findings) == 1
        assert method_node.result.findings[0].summary == "real bug"
        assert method_node.result.findings[0].severity == "high"
        assert method_node.result.recovered_from_children == 0
        assert len(method_node.result.dropped) == 1
        assert method_node.result.dropped[0]["reason"] == "not backed by concrete evidence"


class TrackingBackend:
    """Records which prompts it received so tests can verify routing."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.model_name = name
        self.calls: list[CompletionRequest] = []

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        self.calls.append(request)
        is_leaf = "Review lens" in request.messages[-1].content
        data = {"summary": f"{self.name}-leaf" if is_leaf else f"{self.name}-synth", "findings": []}
        return CompletionResponse(
            message=Message(role=Role.assistant, content=json.dumps(data)),
            usage={"prompt_tokens": 5, "completion_tokens": 5},
        )


class TestDualModelRouting:
    @pytest.mark.asyncio
    async def test_leaves_and_synthesis_use_separate_backends_when_configured(self, tmp_path):
        target = tmp_path / "multi.py"
        target.write_text(_make_functions_source(3))
        methods = (ReviewMethod("m1", "desc1"),)
        leaf_backend = TrackingBackend("quick")
        synthesis_backend = TrackingBackend("powerful")

        root = await run_review(
            [target], methods=methods, backend=leaf_backend, synthesis_backend=synthesis_backend,
            tokenizer=ApproxTokenizer(), leaf_token_budget=10_000,
        )

        # 3 leaf calls all went to the quick backend
        assert len(leaf_backend.calls) == 3
        assert all("Review lens" in c.messages[-1].content for c in leaf_backend.calls)
        # method synthesis + root synthesis went to the powerful backend
        assert len(synthesis_backend.calls) == 2
        assert all("Review lens" not in c.messages[-1].content for c in synthesis_backend.calls)
        assert root.result.summary == "powerful-synth"

    @pytest.mark.asyncio
    async def test_defaults_to_same_backend_for_both_when_unconfigured(self, tmp_path):
        target = tmp_path / "multi.py"
        target.write_text(_make_functions_source(3))
        methods = (ReviewMethod("m1", "desc1"),)
        backend = TrackingBackend("shared")

        await run_review(
            [target], methods=methods, backend=backend, tokenizer=ApproxTokenizer(),
            leaf_token_budget=10_000,
        )

        # 3 leaves + method synthesis + root synthesis, all on the one backend
        assert len(backend.calls) == 5

    @pytest.mark.asyncio
    async def test_synthesis_model_builds_a_separate_backend(self, monkeypatch, tmp_path):
        target = tmp_path / "multi.py"
        target.write_text(_make_functions_source(2))
        methods = (ReviewMethod("m1", "desc1"),)
        leaf_backend = TrackingBackend("quick")
        built_models: list[str | None] = []

        def fake_default_backend(model, think):
            built_models.append(model)
            return TrackingBackend(f"synth-{model}")

        monkeypatch.setattr(review_tree, "_default_backend", fake_default_backend)

        await run_review(
            [target], methods=methods, backend=leaf_backend, synthesis_model="big-model",
            tokenizer=ApproxTokenizer(), leaf_token_budget=10_000,
        )

        assert built_models == ["big-model"]

    @pytest.mark.asyncio
    async def test_oversized_unit_synthesis_also_uses_synthesis_backend(self, tmp_path):
        target = tmp_path / "a.py"
        huge_body = "\n".join(f"    x{j} = {j}" for j in range(200))
        target.write_text(f"def huge():\n{huge_body}\n    return 0\n")
        methods = (ReviewMethod("m1", "desc1"),)
        leaf_backend = TrackingBackend("quick")
        synthesis_backend = TrackingBackend("powerful")

        await run_review(
            [target], methods=methods, backend=leaf_backend, synthesis_backend=synthesis_backend,
            tokenizer=ApproxTokenizer(), leaf_token_budget=20,
        )

        assert len(leaf_backend.calls) > 1  # huge() got split into multiple leaf pieces
        assert len(synthesis_backend.calls) >= 1  # at least the oversized-unit synthesis


class TestThinkEffort:
    def test_default_backend_uses_low_think_by_default(self, monkeypatch, tmp_path):
        captured: dict = {}

        class FakeSettings:
            llm_backend = "ollama"
            ollama_model = "test-model"
            ollama_host = "http://localhost:11434"

        class FakeOllamaBackend:
            def __init__(self, model, host, think):
                captured["model"] = model
                captured["host"] = host
                captured["think"] = think

        monkeypatch.setattr(review_tree.chef_config, "settings", FakeSettings())
        monkeypatch.setattr("chef_human.llm.ollama_backend.OllamaBackend", FakeOllamaBackend)

        review_tree._default_backend(None, "low")

        assert captured == {"model": "test-model", "host": "http://localhost:11434", "think": "low"}

    def test_default_backend_honors_explicit_think_level(self, monkeypatch):
        captured: dict = {}

        class FakeSettings:
            llm_backend = "ollama"
            ollama_model = "test-model"
            ollama_host = "http://localhost:11434"

        class FakeOllamaBackend:
            def __init__(self, model, host, think):
                captured["think"] = think

        monkeypatch.setattr(review_tree.chef_config, "settings", FakeSettings())
        monkeypatch.setattr("chef_human.llm.ollama_backend.OllamaBackend", FakeOllamaBackend)

        review_tree._default_backend("some-model", False)

        assert captured["think"] is False


class TestTruncationDetection:
    @pytest.mark.asyncio
    async def test_hitting_completion_cap_marks_result_truncated(self, tmp_path):
        target = tmp_path / "a.py"
        target.write_text("x = 1\n")
        methods = (ReviewMethod("m1", "desc1"),)

        class CappedBackend:
            model_name = "capped"

            async def complete(self, request: CompletionRequest) -> CompletionResponse:
                return CompletionResponse(
                    message=Message(role=Role.assistant, content=""),
                    usage={"prompt_tokens": 5, "completion_tokens": request.max_tokens},
                )

        root = await run_review(
            [target], methods=methods, backend=CappedBackend(), tokenizer=ApproxTokenizer(),
            leaf_token_budget=10_000, max_completion_tokens=50,
        )

        leaf = root.children[0]
        assert leaf.result is not None
        assert leaf.result.truncated is True
        assert leaf.result.summary == ""
        assert leaf.truncated_node_ids() == [leaf.node_id]
        assert leaf.node_id in root.truncated_node_ids()

    @pytest.mark.asyncio
    async def test_truncation_does_not_retry_makes_exactly_one_call_per_node(self, tmp_path):
        target = tmp_path / "a.py"
        target.write_text("x = 1\n")
        methods = (ReviewMethod("m1", "desc1"),)

        class AlwaysTruncatedBackend:
            model_name = "always-truncated"

            def __init__(self) -> None:
                self.call_count = 0

            async def complete(self, request: CompletionRequest) -> CompletionResponse:
                self.call_count += 1
                return CompletionResponse(
                    message=Message(role=Role.assistant, content=""),
                    usage={"prompt_tokens": 5, "completion_tokens": request.max_tokens},
                )

        backend = AlwaysTruncatedBackend()
        await run_review(
            [target], methods=methods, backend=backend, tokenizer=ApproxTokenizer(),
            leaf_token_budget=10_000, max_completion_tokens=50,
        )

        # exactly 2 calls for this tree: 1 leaf (single-unit shortcut) + 1
        # root synthesis -- no retries
        assert backend.call_count == 2

    def test_default_max_completion_tokens_is_30000(self):
        import inspect

        sig = inspect.signature(run_review)
        assert sig.parameters["max_completion_tokens"].default == 30000

    @pytest.mark.asyncio
    async def test_under_cap_completion_is_not_marked_truncated(self, tmp_path):
        target = tmp_path / "a.py"
        target.write_text("x = 1\n")
        backend = ScriptedBackend(
            [{"summary": "leaf", "findings": []} for _ in review_tree.DEFAULT_METHODS]
            + [{"summary": "root", "findings": []}]
        )

        root = await run_review(
            [target], backend=backend, tokenizer=ApproxTokenizer(), leaf_token_budget=10_000
        )

        assert root.truncated_node_ids() == []

    def test_render_markdown_warns_about_truncated_nodes(self):
        node = review_tree.ReviewNode(node_id="root", goal="g", method=None, depth=0, scope=())
        node.result = review_tree.NodeResult(summary="", findings=[], truncated=True)

        markdown = render_markdown(node)

        assert "Diagnostics" in markdown
        assert "hit max_completion_tokens" in markdown
        assert "`root`" in markdown


class TestConnectedContextDiagnostics:
    def test_render_markdown_shows_omitted_connected_context(self):
        node = review_tree.ReviewNode(node_id="root", goal="g", method=None, depth=0, scope=())
        node.result = review_tree.NodeResult(
            summary="ok", findings=[], connected_included=1, connected_omitted=2
        )

        markdown = render_markdown(node)

        assert "Diagnostics" in markdown
        assert "connected" in markdown.lower()
        assert "1 included / 2 omitted" in markdown


class TestRenderAndSerialize:
    @pytest.mark.asyncio
    async def test_to_dict_round_trips_structure(self, tmp_path):
        target = tmp_path / "a.py"
        target.write_text("x = 1\n")
        backend = ScriptedBackend(
            [{"summary": "leaf", "findings": []} for _ in review_tree.DEFAULT_METHODS]
            + [{"summary": "root", "findings": [_finding()]}]
        )

        root = await run_review(
            [target], backend=backend, tokenizer=ApproxTokenizer(), leaf_token_budget=10_000
        )
        payload = root.to_dict()

        assert payload["node_id"] == "root"
        assert len(payload["children"]) == len(review_tree.DEFAULT_METHODS)
        assert payload["findings"][0]["file"] == "a.py"
        assert "header_context" in payload
        assert "connected" in payload
        assert "connected_included" in payload
        assert "connected_omitted" in payload

    def test_render_markdown_lists_findings(self):
        node = review_tree.ReviewNode(
            node_id="root", goal="Review 1 file(s)", method=None, depth=0, scope=(),
        )
        node.result = review_tree.NodeResult(
            summary="overall summary",
            findings=[Finding("a.py", 3, "correctness", "bug here", "call f(0)")],
        )

        markdown = render_markdown(node)

        assert "overall summary" in markdown
        assert "a.py:3" in markdown
        assert "bug here" in markdown
        assert "call f(0)" in markdown

    def test_render_markdown_with_no_findings(self):
        node = review_tree.ReviewNode(node_id="root", goal="g", method=None, depth=0, scope=())
        node.result = review_tree.NodeResult(summary="all clear", findings=[])

        markdown = render_markdown(node)

        assert "No findings survived synthesis." in markdown


def test_iter_python_files_only_direct_children(tmp_path):
    (tmp_path / "a.py").write_text("x = 1\n")
    (tmp_path / "b.py").write_text("x = 2\n")
    (tmp_path / "not_python.txt").write_text("skip\n")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "c.py").write_text("x = 3\n")

    files = review_tree._iter_python_files(tmp_path)

    assert [f.name for f in files] == ["a.py", "b.py"]
