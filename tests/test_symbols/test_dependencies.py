from __future__ import annotations

from pathlib import Path

import pytest

from chef_human.agent.symbols.dependencies import DependencyGraph
from chef_human.agent.symbols.extractor import CompositeExtractor
from chef_human.agent.symbols.index import SymbolIndex
from chef_human.agent.workspace import WorkspaceManager


@pytest.fixture
def workspace(tmp_path: Path) -> WorkspaceManager:
    return WorkspaceManager(root=str(tmp_path))


@pytest.fixture
def index_and_files(
    workspace: WorkspaceManager, tmp_path: Path
) -> tuple[SymbolIndex, list[Path]]:
    extractor = CompositeExtractor()
    index = SymbolIndex(workspace=workspace, extractor=extractor)

    main_py = tmp_path / "main.py"
    main_py.write_text("from utils import helper\nfrom models import Model\n\ndef run():\n    pass\n")

    utils_py = tmp_path / "utils.py"
    utils_py.write_text(
        "from models import Model\n\ndef helper():\n    return Model()\n"
    )

    models_py = tmp_path / "models.py"
    models_py.write_text("class Model:\n    pass\n")

    standalone_py = tmp_path / "standalone.py"
    standalone_py.write_text("def no_deps():\n    pass\n")

    files = [main_py, utils_py, models_py, standalone_py]
    index.build(files=files)
    return index, files


class TestDependencyGraphBuild:
    def test_build_from_index(self, index_and_files):
        index, files = index_and_files
        graph = DependencyGraph(index)
        graph.build()

        main_py, utils_py, models_py, standalone_py = files
        assert models_py in graph.dependencies(main_py)
        assert utils_py in graph.dependencies(main_py)
        assert models_py in graph.dependencies(utils_py)
        assert graph.dependencies(models_py) == []
        assert graph.dependencies(standalone_py) == []

    def test_reverse_deps(self, index_and_files):
        index, files = index_and_files
        graph = DependencyGraph(index)
        graph.build()

        main_py, _utils_py, models_py, _standalone_py = files
        assert main_py in graph.dependents(models_py)
        assert _utils_py in graph.dependents(models_py)

    def test_empty_index(self, workspace: WorkspaceManager):
        extractor = CompositeExtractor()
        index = SymbolIndex(workspace=workspace, extractor=extractor)
        index.build(files=[])
        graph = DependencyGraph(index)
        graph.build()

        assert graph.dependencies("nonexistent.py") == []
        assert graph.dependents("nonexistent.py") == []
        assert graph.format_for_prompt() == ""


class TestDependencyGraphQuery:
    def test_dependencies_by_string_path(self, index_and_files):
        index, files = index_and_files
        graph = DependencyGraph(index)
        graph.build()

        deps = graph.dependencies("main.py")
        assert len(deps) >= 2

    def test_dependencies_nonexistent(self, index_and_files):
        index, _files = index_and_files
        graph = DependencyGraph(index)
        graph.build()

        assert graph.dependencies("/nonexistent/path.py") == []


class TestDependencyGraphTransitive:
    def test_transitive_dependencies(self, index_and_files):
        index, files = index_and_files
        graph = DependencyGraph(index)
        graph.build()

        main_py, _utils_py, models_py, _standalone_py = files
        transitive = graph.transitive_dependencies(main_py, max_depth=2)
        assert models_py in transitive

    def test_transitive_zero_depth(self, index_and_files):
        index, files = index_and_files
        graph = DependencyGraph(index)
        graph.build()

        main_py, _utils_py, _models_py, _standalone_py = files
        transitive = graph.transitive_dependencies(main_py, max_depth=0)
        assert transitive == set()

    def test_transitive_nonexistent(self, index_and_files):
        index, _files = index_and_files
        graph = DependencyGraph(index)
        graph.build()

        assert graph.transitive_dependencies("/missing.py") == set()


class TestDependencyGraphFormat:
    def test_format_for_prompt(self, index_and_files):
        index, _files = index_and_files
        graph = DependencyGraph(index)
        graph.build()

        output = graph.format_for_prompt()
        assert "# Dependency Graph" in output
        assert "→" in output
        assert "←" in output

    def test_format_empty_graph(self, workspace: WorkspaceManager):
        extractor = CompositeExtractor()
        index = SymbolIndex(workspace=workspace, extractor=extractor)
        index.build(files=[])
        graph = DependencyGraph(index)
        graph.build()

        assert graph.format_for_prompt() == ""

    def test_format_respects_max_files(self, index_and_files):
        index, _files = index_and_files
        graph = DependencyGraph(index)
        graph.build()

        output = graph.format_for_prompt(max_files=1)
        lines = output.splitlines()
        assert len(lines) < 10


class TestDependencyGraphResolution:
    def test_relative_import(self, workspace: WorkspaceManager, tmp_path: Path):
        (tmp_path / "sub").mkdir()
        mod = tmp_path / "sub" / "mod.py"
        mod.write_text("class SubModel:\n    pass\n")
        lib = tmp_path / "lib.py"
        lib.write_text("from sub.mod import SubModel\n\ndef use():\n    pass\n")

        extractor = CompositeExtractor()
        index = SymbolIndex(workspace=workspace, extractor=extractor)
        index.build(files=[mod, lib])

        graph = DependencyGraph(index)
        graph.build()

        deps = graph.dependencies(lib)
        assert mod in deps

    def test_external_import_excluded(self, workspace: WorkspaceManager, tmp_path: Path):
        lib = tmp_path / "lib.py"
        lib.write_text("import sys\nimport json\n\ndef f():\n    pass\n")

        extractor = CompositeExtractor()
        index = SymbolIndex(workspace=workspace, extractor=extractor)
        index.build(files=[lib])

        graph = DependencyGraph(index)
        graph.build()

        assert graph.dependencies(lib) == []
