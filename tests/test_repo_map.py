from __future__ import annotations

from pathlib import Path

import pytest

from chef_human.agent.repo_map import RepoMap
from chef_human.agent.symbols.extractor import (
    RegexExtractor,
    Symbol,
    TreeSitterExtractor,
    create_extractor,
)
from chef_human.agent.workspace import WorkspaceManager
from chef_human.llm.tokenizer import ApproxTokenizer


def create_file(directory: Path, name: str, content: str = "") -> Path:
    path = directory / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return path


@pytest.fixture
def workspace(tmp_path: Path) -> WorkspaceManager:
    return WorkspaceManager(root=tmp_path)


@pytest.fixture
def tokenizer() -> ApproxTokenizer:
    return ApproxTokenizer()


@pytest.fixture
def repo_map(workspace: WorkspaceManager, tokenizer: ApproxTokenizer) -> RepoMap:
    return RepoMap(workspace=workspace, tokenizer=tokenizer)


# ---------------------------------------------------------------------------
# Symbol dataclass
# ---------------------------------------------------------------------------

class TestSymbol:
    def test_fields(self):
        s = Symbol(name="foo", kind="function", line=1, signature="def foo():")
        assert s.name == "foo"
        assert s.kind == "function"
        assert s.line == 1
        assert s.signature == "def foo():"

    def test_frozen(self):
        s = Symbol(name="foo", kind="function", line=1, signature="def foo():")
        with pytest.raises(AttributeError):
            s.name = "bar"  # type: ignore[misc]

    def test_hashable(self):
        s1 = Symbol(name="foo", kind="function", line=1, signature="def foo():")
        s2 = Symbol(name="foo", kind="function", line=1, signature="def foo():")
        assert hash(s1) == hash(s2)


# ---------------------------------------------------------------------------
# RegexExtractor
# ---------------------------------------------------------------------------

class TestRegexExtractorPython:
    def setup_method(self) -> None:
        self.extractor = RegexExtractor()

    def test_extract_function(self):
        symbols = self.extractor.extract("test.py", "def hello():\n    pass\n")
        assert len(symbols) == 1
        assert symbols[0].name == "hello"
        assert symbols[0].kind == "function"
        assert symbols[0].line == 1
        assert symbols[0].signature == "def hello():"

    def test_extract_async_function(self):
        symbols = self.extractor.extract("test.py", "async def fetch(): ...")
        assert len(symbols) == 1
        assert symbols[0].name == "fetch"
        assert symbols[0].kind == "function"

    def test_extract_function_with_return_type(self):
        symbols = self.extractor.extract("test.py", "def add(a: int, b: int) -> int:")
        assert len(symbols) == 1
        assert symbols[0].name == "add"

    def test_extract_class(self):
        symbols = self.extractor.extract("test.py", "class MyClass:")
        assert len(symbols) == 1
        assert symbols[0].name == "MyClass"
        assert symbols[0].kind == "class"

    def test_extract_class_with_inheritance(self):
        symbols = self.extractor.extract("test.py", "class MyClass(Base):")
        assert len(symbols) == 1
        assert symbols[0].name == "MyClass"

    def test_extract_import(self):
        symbols = self.extractor.extract("test.py", "import os")
        assert len(symbols) == 1
        assert symbols[0].kind == "import"
        assert "import os" in symbols[0].signature

    def test_extract_from_import(self):
        symbols = self.extractor.extract("test.py", "from pathlib import Path")
        assert len(symbols) == 1
        assert symbols[0].kind == "import"

    def test_no_false_positives_on_comment(self):
        symbols = self.extractor.extract("test.py", "# def old_func():")
        assert len(symbols) == 0

    def test_multiple_symbols_per_file(self):
        content = "\n".join(["def a():", "def b():", "class C:"])
        symbols = self.extractor.extract("test.py", content)
        assert len(symbols) == 3
        names = [s.name for s in symbols]
        assert names == ["a", "b", "C"]

    def test_empty_content(self):
        symbols = self.extractor.extract("test.py", "")
        assert symbols == []

    def test_only_comments_and_blank_lines(self):
        content = "\n".join(["# comment", "", "# another"])
        symbols = self.extractor.extract("test.py", content)
        assert symbols == []


class TestRegexExtractorRust:
    def setup_method(self) -> None:
        self.extractor = RegexExtractor()

    def test_fn(self):
        symbols = self.extractor.extract("main.rs", "fn main() {")
        assert len(symbols) == 1
        assert symbols[0].name == "main"
        assert symbols[0].kind == "function"

    def test_pub_fn(self):
        symbols = self.extractor.extract("lib.rs", "pub fn helper() -> i32 {")
        assert len(symbols) == 1
        assert symbols[0].name == "helper"

    def test_struct(self):
        symbols = self.extractor.extract("types.rs", "pub struct Config {")
        assert len(symbols) == 1
        assert symbols[0].name == "Config"
        assert symbols[0].kind == "struct"

    def test_enum(self):
        symbols = self.extractor.extract("types.rs", "enum Color {")
        assert len(symbols) == 1
        assert symbols[0].name == "Color"
        assert symbols[0].kind == "enum"

    def test_trait(self):
        symbols = self.extractor.extract("traits.rs", "trait Display {")
        assert len(symbols) == 1
        assert symbols[0].name == "Display"
        assert symbols[0].kind == "trait"

    def test_unsafe_fn(self):
        symbols = self.extractor.extract("ffi.rs", "pub unsafe fn transmute<T>(x: T) -> U {")
        assert len(symbols) == 1
        assert symbols[0].name == "transmute"


class TestRegexExtractorGo:
    def setup_method(self) -> None:
        self.extractor = RegexExtractor()

    def test_func(self):
        symbols = self.extractor.extract("main.go", "func main() {")
        assert len(symbols) == 1
        assert symbols[0].name == "main"
        assert symbols[0].kind == "function"

    def test_method_receiver(self):
        symbols = self.extractor.extract("handler.go", "func (h *Handler) ServeHTTP(w ResponseWriter, r *Request) {")
        assert len(symbols) == 1
        assert symbols[0].name == "ServeHTTP"

    def test_struct_type(self):
        symbols = self.extractor.extract("types.go", "type Config struct {")
        assert len(symbols) == 1
        assert symbols[0].name == "Config"
        assert symbols[0].kind == "struct"

    def test_interface_type(self):
        symbols = self.extractor.extract("types.go", "type Handler interface {")
        assert len(symbols) == 1
        assert symbols[0].name == "Handler"
        assert symbols[0].kind == "interface"


class TestRegexExtractorJsTs:
    def setup_method(self) -> None:
        self.extractor = RegexExtractor()

    def test_js_function(self):
        symbols = self.extractor.extract("app.js", "function greet(name) {")
        assert len(symbols) == 1
        assert symbols[0].name == "greet"

    def test_async_js_function(self):
        symbols = self.extractor.extract("app.js", "async function fetchData(url) {")
        assert len(symbols) == 1
        assert symbols[0].name == "fetchData"

    def test_js_class(self):
        symbols = self.extractor.extract("app.js", "class Animal {")
        assert len(symbols) == 1
        assert symbols[0].name == "Animal"

    def test_ts_interface(self):
        symbols = self.extractor.extract("types.ts", "interface User {")
        assert len(symbols) == 1
        assert symbols[0].name == "User"
        assert symbols[0].kind == "interface"


class TestRegexExtractorJava:
    def setup_method(self) -> None:
        self.extractor = RegexExtractor()

    def test_class(self):
        symbols = self.extractor.extract("Main.java", "public class Main {")
        assert len(symbols) == 1
        assert symbols[0].name == "Main"

    def test_interface(self):
        symbols = self.extractor.extract("Service.java", "public interface Service {")
        assert len(symbols) == 1
        assert symbols[0].name == "Service"
        assert symbols[0].kind == "interface"

    def test_method(self):
        symbols = self.extractor.extract("Main.java", "public void run() {")
        assert len(symbols) == 1
        assert symbols[0].name == "run"
        assert symbols[0].kind == "method"

    def test_private_method(self):
        symbols = self.extractor.extract("Main.java", "private int calculate() throws Exception {")
        assert len(symbols) == 1
        assert symbols[0].name == "calculate"


class TestRegexExtractorDefaultPatterns:
    def setup_method(self) -> None:
        self.extractor = RegexExtractor()

    def test_unknown_extension_default_function(self):
        symbols = self.extractor.extract("script.rb", "def hello()")
        assert len(symbols) == 1
        assert symbols[0].name == "hello"
        assert symbols[0].kind == "function"

    def test_unknown_extension_default_class(self):
        symbols = self.extractor.extract("script.rb", "class Foo")
        assert len(symbols) == 1
        assert symbols[0].name == "Foo"
        assert symbols[0].kind == "class"


# ---------------------------------------------------------------------------
# TreeSitterExtractor
# ---------------------------------------------------------------------------

class TestTreeSitterExtractor:
    def test_import_error_when_not_installed(self):
        with pytest.raises(ImportError, match="tree-sitter"):
            TreeSitterExtractor()

    def test_create_extractor_fallback(self):
        extractor = create_extractor()
        assert isinstance(extractor, RegexExtractor)


# ---------------------------------------------------------------------------
# RepoMap — Tree Generation
# ---------------------------------------------------------------------------

class TestRepoMapTree:
    def test_empty_workspace(self, repo_map, tmp_path):
        result = repo_map.generate_tree()
        assert result == ""

    def test_single_root_file(self, repo_map, tmp_path):
        create_file(tmp_path, "README.md", "content")
        result = repo_map.generate_tree()
        assert "Project tree (1 files shown):" in result
        assert "README.md" in result

    def test_multiple_root_files(self, repo_map, tmp_path):
        create_file(tmp_path, "a.py", "x = 1")
        create_file(tmp_path, "b.py", "y = 2")
        result = repo_map.generate_tree()
        assert "a.py" in result
        assert "b.py" in result

    def test_single_subdirectory(self, repo_map, tmp_path):
        create_file(tmp_path, "src/main.py", "def f(): pass")
        result = repo_map.generate_tree()
        assert "src/" in result
        assert "main.py" in result

    def test_nested_directories(self, repo_map, tmp_path):
        create_file(tmp_path, "src/utils/helper.py", "def h(): pass")
        result = repo_map.generate_tree()
        assert "src/" in result
        assert "utils/" in result
        assert "helper.py" in result

    def test_root_and_subdir_files(self, repo_map, tmp_path):
        create_file(tmp_path, "README.md", "docs")
        create_file(tmp_path, "src/main.py", "print('hi')")
        result = repo_map.generate_tree()
        assert "README.md" in result
        assert "src/" in result
        assert "main.py" in result

    def test_subdirectory_with_multiple_files(self, repo_map, tmp_path):
        create_file(tmp_path, "src/a.py", "x = 1")
        create_file(tmp_path, "src/b.py", "y = 2")
        create_file(tmp_path, "src/c.py", "z = 3")
        result = repo_map.generate_tree()
        assert "a.py" in result
        assert "b.py" in result
        assert "c.py" in result

    def test_multiple_directories(self, repo_map, tmp_path):
        create_file(tmp_path, "src/main.py", "x = 1")
        create_file(tmp_path, "tests/test_main.py", "def test(): pass")
        result = repo_map.generate_tree()
        assert "src/" in result
        assert "tests/" in result
        assert "main.py" in result
        assert "test_main.py" in result

    def test_ignored_files_excluded(self, repo_map, tmp_path):
        create_file(tmp_path, "main.py", "x = 1")
        create_file(tmp_path, "__pycache__/cache.pyc", "binary")
        create_file(tmp_path, ".git/config", "stuff")
        result = repo_map.generate_tree()
        assert "main.py" in result
        assert "__pycache__" not in result
        assert ".git" not in result


# ---------------------------------------------------------------------------
# RepoMap — Symbol Map
# ---------------------------------------------------------------------------

class TestRepoMapSymbolMap:
    def test_no_files(self, repo_map):
        result = repo_map.generate_symbol_map(files=[])
        assert result == ""

    def test_file_with_no_symbols(self, repo_map, tmp_path):
        f = create_file(tmp_path, "data.txt", "just text")
        result = repo_map.generate_symbol_map(files=[f])
        assert result == ""

    def test_single_file_with_symbols(self, repo_map, tmp_path):
        f = create_file(tmp_path, "main.py", "def hello():\n    pass\n")
        result = repo_map.generate_symbol_map(files=[f])
        assert "main.py" in result
        assert "def hello():" in result

    def test_multiple_symbols_per_file(self, repo_map, tmp_path):
        content = "def a():\n    pass\n\ndef b():\n    pass\n"
        f = create_file(tmp_path, "mod.py", content)
        result = repo_map.generate_symbol_map(files=[f])
        assert "def a():" in result
        assert "def b():" in result

    def test_max_ten_symbols_per_file(self, repo_map, tmp_path):
        lines = [f"def f{i}():\n    pass" for i in range(15)]
        f = create_file(tmp_path, "many.py", "\n".join(lines))
        result = repo_map.generate_symbol_map(files=[f])
        for i in range(10):
            assert f"def f{i}():" in result
        assert "def f10():" not in result

    def test_no_double_info_dicts(self, repo_map, tmp_path):
        f = create_file(tmp_path, "mod.py", "def f(): pass")
        result = repo_map.generate_symbol_map(files=[f])
        assert result.count("###") == 1


# ---------------------------------------------------------------------------
# RepoMap — Combined Generate
# ---------------------------------------------------------------------------

class TestRepoMapGenerate:
    def test_empty_workspace(self, repo_map):
        result = repo_map.generate(max_tokens=2000)
        assert result == ""

    def test_includes_tree(self, repo_map, tmp_path):
        create_file(tmp_path, "main.py", "def f(): pass")
        result = repo_map.generate(max_tokens=2000)
        assert "Project tree" in result
        assert "main.py" in result

    def test_includes_symbols_with_content(self, repo_map, tmp_path):
        create_file(tmp_path, "main.py", "def hello():\n    pass\n")
        result = repo_map.generate(max_tokens=2000)
        assert "## Symbols" in result
        assert "def hello():" in result

    def test_truncates_when_over_budget(self, repo_map, tmp_path):
        long_content = "\n".join([f"def func_{i}():\n    pass" for i in range(200)])
        create_file(tmp_path, "big.py", long_content)
        result = repo_map.generate(max_tokens=50)
        total_tokens = ApproxTokenizer().count(result)
        assert total_tokens <= 50 or "(truncated)" in result

    def test_omits_symbols_when_remaining_too_small(self, repo_map, tmp_path):
        create_file(tmp_path, "main.py", "def f(): pass")
        result = repo_map.generate(max_tokens=10)
        assert "## Symbols" not in result

    def test_symbol_cache_used(self, repo_map, tmp_path):
        extractor = RegexExtractor()
        repo_map._extractor = extractor
        f = create_file(tmp_path, "mod.py", "def f(): pass")
        first = repo_map._extract_symbols(f, "def f(): pass")
        second = repo_map._extract_symbols(f, "def f(): pass")
        assert first is second


# ---------------------------------------------------------------------------
# RepoMap — Helpers
# ---------------------------------------------------------------------------

class TestRepoSafeRead:
    def test_reads_text_file(self, repo_map, tmp_path):
        f = create_file(tmp_path, "a.py", "hello")
        result = RepoMap._safe_read(f)
        assert result == "hello"

    def test_none_for_nonexistent(self, repo_map, tmp_path):
        result = RepoMap._safe_read(tmp_path / "nonexistent.py")
        assert result is None

    def test_none_for_directory(self, repo_map, tmp_path):
        d = tmp_path / "mydir"
        d.mkdir()
        result = RepoMap._safe_read(d)
        assert result is None


class TestRepoTruncate:
    def test_below_limit(self):
        text = "hello world"
        result = RepoMap._truncate_to_tokens(text, 100)
        assert result == text

    def test_above_limit(self):
        text = "a" * 100
        result = RepoMap._truncate_to_tokens(text, 10)
        assert result.endswith("(truncated)")
        assert len(result) < len(text)

    def test_at_exact_limit(self):
        text = "a" * 40
        result = RepoMap._truncate_to_tokens(text, 10)
        assert result == text

    def test_empty_string(self):
        result = RepoMap._truncate_to_tokens("", 10)
        assert result == ""


# ---------------------------------------------------------------------------
# RepoMap — tree with subdirectory parameter
# ---------------------------------------------------------------------------

class TestRepoMapSubdir:
    def test_subdirectory_tree(self, repo_map, tmp_path):
        create_file(tmp_path, "src/main.py", "x = 1")
        create_file(tmp_path, "src/utils/helper.py", "y = 2")
        create_file(tmp_path, "tests/test_main.py", "z = 3")
        result = repo_map.generate_tree(directory="src")
        assert "src/" not in result or "utils/" in result
        assert "test_main.py" not in result
