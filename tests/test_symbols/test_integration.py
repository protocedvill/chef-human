from __future__ import annotations

from pathlib import Path

import pytest

from chef_human.agent.file_context import FileContextManager
from chef_human.agent.symbols.dependencies import DependencyGraph
from chef_human.agent.symbols.extractor import CompositeExtractor
from chef_human.agent.symbols.index import SymbolIndex
from chef_human.agent.symbols.retriever import SymbolRetriever
from chef_human.agent.workspace import WorkspaceManager
from chef_human.llm.tokenizer import create_tokenizer


@pytest.fixture
def workspace(tmp_path: Path) -> WorkspaceManager:
    return WorkspaceManager(root=str(tmp_path))


@pytest.fixture
def multi_lang_codebase(workspace: WorkspaceManager, tmp_path: Path) -> list[Path]:
    py_utils = tmp_path / "utils.py"
    py_utils.write_text(
        "from models import Config\n\n"
        "def parse(data: str) -> Config:\n"
        "    return Config(name=data)\n"
    )

    py_models = tmp_path / "models.py"
    py_models.write_text("class Config:\n    name: str\n\nclass Result:\n    pass\n")

    js_helpers = tmp_path / "helpers.js"
    js_helpers.write_text(
        "import { readFile } from 'fs';\n"
        "import { Config } from './config.js';\n"
        "\n"
        "function load(path) {\n"
        "    return readFile(path);\n"
        "}\n"
        "\n"
        "class Parser {\n"
        "    parse(input) {\n"
        "        return JSON.parse(input);\n"
        "    }\n"
        "}\n"
    )

    js_config = tmp_path / "config.js"
    js_config.write_text("export class Config {\n    constructor(name) {\n        this.name = name;\n    }\n}\n")

    rs_lib = tmp_path / "lib.rs"
    rs_lib.write_text(
        "use std::collections::HashMap;\n"
        "\n"
        "pub struct Server {\n"
        "    pub port: u16,\n"
        "}\n"
        "\n"
        "pub fn start() -> Server {\n"
        "    Server { port: 8080 }\n"
        "}\n"
    )

    return [py_utils, py_models, js_helpers, js_config, rs_lib]


def test_full_pipeline(workspace: WorkspaceManager, multi_lang_codebase: list[Path]):
    extractor = CompositeExtractor()
    index = SymbolIndex(workspace=workspace, extractor=extractor)
    index.build(files=multi_lang_codebase)

    assert index.total_symbols() > 0
    assert index.is_built

    dep_graph = DependencyGraph(index)
    dep_graph.build()

    tokenizer = create_tokenizer()
    file_ctx = FileContextManager(workspace=workspace, tokenizer=tokenizer)
    retriever = SymbolRetriever(index=index, file_context=file_ctx)

    assert retriever.detect_symbol_references("Use Config") == ["Config"]

    result = retriever.retrieve("Config")
    assert result is not None
    assert "Config" in result

    cached = file_ctx.cached_files()
    assert len(cached) >= 1


def test_empty_codebase(workspace: WorkspaceManager):
    extractor = CompositeExtractor()
    index = SymbolIndex(workspace=workspace, extractor=extractor)
    index.build(files=[])

    assert index.total_symbols() == 0
    assert index.is_built

    dep_graph = DependencyGraph(index)
    dep_graph.build()
    assert dep_graph.format_for_prompt() == ""

    tokenizer = create_tokenizer()
    file_ctx = FileContextManager(workspace=workspace, tokenizer=tokenizer)
    retriever = SymbolRetriever(index=index, file_context=file_ctx)

    assert retriever.detect_symbol_references("Use Config") == []
    assert retriever.retrieve("Config") is None


def test_retrieval_deduplicates(workspace: WorkspaceManager, multi_lang_codebase: list[Path]):
    extractor = CompositeExtractor()
    index = SymbolIndex(workspace=workspace, extractor=extractor)
    index.build(files=multi_lang_codebase)

    tokenizer = create_tokenizer()
    file_ctx = FileContextManager(workspace=workspace, tokenizer=tokenizer)
    retriever = SymbolRetriever(index=index, file_context=file_ctx)

    names = retriever.detect_symbol_references("Config and Config and Config")
    assert len(names) <= 1


def test_multi_file_dependencies(workspace: WorkspaceManager, multi_lang_codebase: list[Path]):
    extractor = CompositeExtractor()
    index = SymbolIndex(workspace=workspace, extractor=extractor)
    index.build(files=multi_lang_codebase)

    dep_graph = DependencyGraph(index)
    dep_graph.build()

    _, py_models, js_helpers, js_config, _rs_lib = multi_lang_codebase

    assert py_models in dep_graph.dependencies(multi_lang_codebase[0])
    assert js_config in dep_graph.dependencies(js_helpers)


def test_format_output(workspace: WorkspaceManager, multi_lang_codebase: list[Path]):
    extractor = CompositeExtractor()
    index = SymbolIndex(workspace=workspace, extractor=extractor)
    index.build(files=multi_lang_codebase)

    dep_graph = DependencyGraph(index)
    dep_graph.build()

    output = dep_graph.format_for_prompt()
    assert "# Dependency Graph" in output
