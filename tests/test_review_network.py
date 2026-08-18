from __future__ import annotations

import json
from pathlib import Path

import pytest

from chef_human import review_network, review_tree
from chef_human.llm.backend import CompletionRequest, CompletionResponse, Message, Role
from chef_human.review_tree import CodeChunk, CodeUnit, Finding, NodeResult, ReviewNode


class ApproxTokenizer:
    def count(self, text: str) -> int:
        return max(1, len(text) // 4)


def _unit(path: Path, name: str, calls: frozenset[str] = frozenset()) -> CodeUnit:
    return CodeUnit(
        path=path, name=name, kind="function",
        chunk=CodeChunk(path, f"def {name}(): pass\n"),
        signature=f"def {name}():", docstring="", calls=calls,
    )


class TestClusterFiles:
    def test_connected_files_cluster_together(self):
        a, b, c = Path("a.py"), Path("b.py"), Path("c.py")
        # a calls into b heavily, c is isolated
        edges = {frozenset((a, b)): 5}
        clusters = review_network.cluster_files([a, b, c], edges, max_cluster_size=6)

        cluster_sets = [set(cl) for cl in clusters]
        assert {a, b} in cluster_sets
        assert {c} in cluster_sets

    def test_isolated_files_become_singletons(self):
        a, b = Path("a.py"), Path("b.py")
        clusters = review_network.cluster_files([a, b], {}, max_cluster_size=6)

        assert sorted(clusters, key=str) == [[a], [b]]

    def test_cluster_size_cap_is_respected(self):
        paths = [Path(f"f{i}.py") for i in range(8)]
        # fully connected -- without a cap this would merge into one cluster
        edges = {
            frozenset((paths[i], paths[j])): 1
            for i in range(len(paths))
            for j in range(i + 1, len(paths))
        }
        clusters = review_network.cluster_files(paths, edges, max_cluster_size=3)

        assert all(len(c) <= 3 for c in clusters)
        # every file still accounted for exactly once
        assert sorted(p for cl in clusters for p in cl) == sorted(paths)

    def test_deterministic_ordering(self):
        a, b, c, d = Path("a.py"), Path("b.py"), Path("c.py"), Path("d.py")
        edges = {frozenset((a, b)): 3, frozenset((c, d)): 3}
        first = review_network.cluster_files([d, c, b, a], edges, max_cluster_size=6)
        second = review_network.cluster_files([a, b, c, d], edges, max_cluster_size=6)

        assert first == second


class TestFileLayerHardFail:
    @pytest.mark.asyncio
    async def test_oversized_file_skips_llm_call_entirely(self, tmp_path):
        target = tmp_path / "huge.py"
        # source alone will exceed a tiny file_token_budget
        target.write_text("def f():\n    return 1\n" * 500)
        methods = (review_tree.ReviewMethod("m1", "desc1"),)

        class RecordingBackend:
            model_name = "recording"

            def __init__(self) -> None:
                self.calls = 0

            async def complete(self, request: CompletionRequest) -> CompletionResponse:
                self.calls += 1
                is_leaf = "Review lens" in request.messages[-1].content
                data = {"summary": "leaf" if is_leaf else "synth", "findings": []}
                return CompletionResponse(
                    message=Message(role=Role.assistant, content=json.dumps(data)),
                    usage={"prompt_tokens": 5, "completion_tokens": 5},
                )

        backend = RecordingBackend()
        root = await review_tree.run_review(
            [target], methods=methods, backend=backend, tokenizer=ApproxTokenizer(),
            leaf_token_budget=10_000, file_token_budget=10,  # impossibly small
        )

        file_node = root.children[0].children[0]
        assert file_node.result.validation_status == "too_large_skipped"
        assert "too large" in file_node.result.summary.lower()
        # the file node's own call never happened -- only leaves + network +
        # root touched the backend
        leaf_count = len(file_node.children)
        assert backend.calls == leaf_count + 2  # leaves + network + root, no file call


class ToolCallingBackend:
    """Scripted: first response requests a tool call, second gives the
    final JSON answer -- mirrors the Message.tool_calls shape OllamaBackend
    produces for native tool-calling models."""

    def __init__(self, final_data: dict) -> None:
        self._final_data = final_data
        self.calls: list[CompletionRequest] = []
        self.model_name = "tool-calling"

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        self.calls.append(request)
        is_leaf = "Review lens" in request.messages[-1].content
        if is_leaf:
            return CompletionResponse(
                message=Message(role=Role.assistant, content=json.dumps({"summary": "leaf", "findings": []})),
                usage={"prompt_tokens": 5, "completion_tokens": 5},
            )
        if request.tools and len(self.calls) == 1:  # network node's first turn
            return CompletionResponse(
                message=Message(
                    role=Role.assistant,
                    content="",
                    tool_calls=[
                        {"function": {"name": "read_code_span", "arguments": {"file": "a.py", "start_line": 1, "end_line": 2}}}
                    ],
                ),
                usage={"prompt_tokens": 5, "completion_tokens": 5},
            )
        return CompletionResponse(
            message=Message(role=Role.assistant, content=json.dumps(self._final_data)),
            usage={"prompt_tokens": 5, "completion_tokens": 5},
        )


class TestNetworkToolLoop:
    @pytest.mark.asyncio
    async def test_read_code_span_dispatches_and_final_answer_parses(self, tmp_path):
        path = tmp_path / "a.py"
        path.write_text("def f():\n    return 1\n")
        node = ReviewNode(node_id="root/net0", goal="Validate cluster", method=None, depth=1, scope=())
        file_node = ReviewNode(node_id="root/net0/a", goal="file", method=None, depth=2, scope=())
        file_node.result = NodeResult(summary="file summary", findings=[])
        node.children = [file_node]

        config = review_tree.ReviewConfig(
            backend=None, synthesis_backend=ToolCallingBackend(
                {"summary": "network done", "findings": [], "margin_references": [{"file": "out.py", "note": "referenced but out of cluster"}]}
            ),
            tokenizer=ApproxTokenizer(), units=[_unit(path, "f")],
        )
        # backend field is unused by run_network_node (only synthesis_backend is)
        config.backend = config.synthesis_backend

        await review_network.run_network_node(node, [path], {path: path.read_text()}, config)

        assert node.result is not None
        assert node.result.summary == "network done"
        assert node.result.margin_references == [{"file": "out.py", "note": "referenced but out of cluster"}]
        # 2 calls: the tool-call turn, then the final answer
        assert len(config.synthesis_backend.calls) == 2
        assert config.synthesis_backend.calls[0].tools is not None

    @pytest.mark.asyncio
    async def test_out_of_cluster_read_is_rejected(self, tmp_path):
        path = tmp_path / "a.py"
        path.write_text("def f():\n    return 1\n")
        budget_state = {"used": 0}

        result = review_network._dispatch_read_code_span(
            {"file": "not_in_cluster.py", "start_line": 1, "end_line": 2},
            [path], {path: path.read_text()}, ApproxTokenizer(), budget_state, network_token_budget=1000,
        )

        assert "ERROR" in result
        assert "not part of this network's cluster" in result

    def test_network_token_budget_exhaustion_rejects_further_reads(self, tmp_path):
        path = tmp_path / "a.py"
        path.write_text("x = 1\n" * 1000)
        budget_state = {"used": 0}

        first = review_network._dispatch_read_code_span(
            {"file": "a.py", "start_line": 1, "end_line": 100},
            [path], {path: path.read_text()}, ApproxTokenizer(), budget_state, network_token_budget=10,
        )

        assert "ERROR" in first
        assert "budget exhausted" in first.lower()
