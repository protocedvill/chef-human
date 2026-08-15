from __future__ import annotations

from pathlib import Path

import pytest

from chef_human.agent.symbols.extractor import (
    CompositeExtractor,
    RegexExtractor,
    Symbol,
    TreeSitterExtractor,
    create_extractor,
)

_HERE = Path(__file__).parent
_TEST_DATA = _HERE / "test_data"


class TestSymbolDataclass:
    def test_symbol_creation(self):
        s = Symbol(name="foo", kind="function", line=1, signature="def foo()")
        assert s.name == "foo"
        assert s.kind == "function"
        assert s.line == 1
        assert s.signature == "def foo()"

    def test_symbol_is_frozen(self):
        s = Symbol(name="foo", kind="function", line=1, signature="def foo()")
        with pytest.raises(AttributeError):
            s.name = "bar"  


class TestRegexExtractor:
    @pytest.fixture
    def extractor(self):
        return RegexExtractor()

    def test_extract_python_function(self, extractor):
        result = extractor.extract("test.py", "def foo(x, y):\n    pass\n")
        assert len(result) == 1
        assert result[0].name == "foo"
        assert result[0].kind == "function"
        assert result[0].signature == "def foo(x, y):"

    def test_extract_python_class(self, extractor):
        result = extractor.extract("test.py", "class MyClass:\n    pass\n")
        assert len(result) == 1
        assert result[0].name == "MyClass"
        assert result[0].kind == "class"

    def test_extract_unknown_extension_falls_back_to_defaults(self, extractor):
        result = extractor.extract("test.xyz", "function foo() {}")
        assert len(result) >= 1

    def test_extract_empty_content(self, extractor):
        result = extractor.extract("test.py", "")
        assert result == []


class TestTreeSitterExtractor:
    @pytest.fixture
    def extractor(self):
        return TreeSitterExtractor()

    def test_unknown_extension_returns_empty(self, extractor):
        result = extractor.extract("test.xyz", "def foo():\n    pass\n")
        assert result == []

    def test_empty_content(self, extractor):
        result = extractor.extract("test.py", "")
        assert result == []

    def test_no_symbols(self, extractor):
        result = extractor.extract("test.py", "# just a comment\n")
        assert result == []

    def test_python_function(self, extractor):
        result = extractor.extract("test.py", "def foo(x, y):\n    pass\n")
        assert len(result) == 1
        s = result[0]
        assert s.name == "foo"
        assert s.kind == "function"
        assert s.line == 1

    def test_python_class(self, extractor):
        result = extractor.extract("test.py", "class MyClass:\n    pass\n")
        assert len(result) == 1
        s = result[0]
        assert s.name == "MyClass"
        assert s.kind == "class"
        assert s.line == 1

    def test_python_async_function(self, extractor):
        code = "@decorator\nasync def foo(a: int, b: str) -> bool:\n    pass\n"
        result = extractor.extract("test.py", code)
        assert len(result) == 1
        s = result[0]
        assert s.name == "foo"
        assert s.kind == "function"
        assert "@decorator" in s.signature
        assert "async def foo" in s.signature

    def test_python_import(self, extractor):
        result = extractor.extract("test.py", "import os, sys")
        assert len(result) == 2
        assert result[0].kind == "import"
        assert result[1].kind == "import"

    def test_python_from_import(self, extractor):
        result = extractor.extract("test.py", "from pathlib import Path")
        assert len(result) == 1
        assert result[0].name == "Path"
        assert result[0].kind == "from_import"

    def test_python_method_in_class(self, extractor):
        code = "class Foo:\n    def method(self):\n        pass\n"
        result = extractor.extract("test.py", code)
        kinds = {s.kind for s in result}
        assert "class" in kinds
        assert "function" in kinds
        names = {s.name for s in result}
        assert "Foo" in names
        assert "method" in names

    def test_javascript_function(self, extractor):
        result = extractor.extract("test.js", "function foo(x) { return x; }")
        assert len(result) >= 1
        assert result[0].name == "foo"
        assert result[0].kind == "function"

    def test_javascript_class(self, extractor):
        code = "class Bar {\n  constructor() {}\n  method() {}\n}\n"
        result = extractor.extract("test.js", code)
        kinds = {s.kind for s in result}
        assert "class" in kinds
        assert "method" in kinds

    def test_typescript_interface(self, extractor):
        code = "interface Foo {\n  bar: number;\n}\n"
        result = extractor.extract("test.ts", code)
        assert len(result) >= 1
        assert result[0].name == "Foo"
        assert result[0].kind == "interface"

    def test_typescript_type_alias(self, extractor):
        code = "type Foo = string;\n"
        result = extractor.extract("test.ts", code)
        assert len(result) >= 1
        assert result[0].name == "Foo"
        assert result[0].kind == "type_alias"

    def test_rust_function(self, extractor):
        result = extractor.extract("test.rs", "fn foo(x: i32) -> i32 { x }")
        assert len(result) >= 1
        assert result[0].name == "foo"
        assert result[0].kind == "function"

    def test_rust_struct(self, extractor):
        result = extractor.extract("test.rs", "struct Bar { x: i32 }")
        assert len(result) >= 1
        assert result[0].name == "Bar"
        assert result[0].kind == "struct"

    def test_rust_enum(self, extractor):
        result = extractor.extract("test.rs", "enum Baz { A, B }")
        assert len(result) >= 1
        assert result[0].name == "Baz"
        assert result[0].kind == "enum"

    def test_rust_trait(self, extractor):
        result = extractor.extract("test.rs", "trait Qux { fn quux(&self); }")
        assert len(result) >= 1
        assert result[0].name == "Qux"
        assert result[0].kind == "trait"

    def test_rust_use(self, extractor):
        result = extractor.extract("test.rs",
                                    "use std::collections::HashMap;")
        assert len(result) >= 1
        assert result[0].name == "HashMap"
        assert result[0].kind == "use"

    def test_go_function(self, extractor):
        result = extractor.extract("test.go", "func foo(x int) int { return x }")
        assert len(result) >= 1
        assert result[0].name == "foo"
        assert result[0].kind == "function"

    def test_go_method(self, extractor):
        code = "func (r *Receiver) method() {\n}\n"
        result = extractor.extract("test.go", code)
        assert len(result) >= 1
        assert result[0].name == "method"
        assert result[0].kind == "method"

    def test_go_struct(self, extractor):
        result = extractor.extract("test.go", "type Bar struct { x int }")
        assert len(result) >= 1
        assert result[0].name == "Bar"
        assert result[0].kind == "struct"

    def test_go_interface(self, extractor):
        result = extractor.extract("test.go",
                                    "type Baz interface { Method() }")
        assert len(result) >= 1
        assert result[0].name == "Baz"
        assert result[0].kind == "interface"

    def test_go_import(self, extractor):
        result = extractor.extract("test.go", 'import "fmt"')
        assert len(result) >= 1
        assert result[0].kind == "import"

    def test_java_class(self, extractor):
        result = extractor.extract("test.java", "class Foo {}")
        assert len(result) >= 1
        assert result[0].name == "Foo"
        assert result[0].kind == "class"

    def test_java_interface(self, extractor):
        result = extractor.extract("test.java", "interface Bar {}")
        assert len(result) >= 1
        assert result[0].name == "Bar"
        assert result[0].kind == "interface"

    def test_java_method(self, extractor):
        code = "class Foo {\n  void bar() {}\n}\n"
        result = extractor.extract("test.java", code)
        assert len(result) >= 2
        methods = [s for s in result if s.kind == "method"]
        assert len(methods) >= 1
        assert methods[0].name == "bar"

    def test_java_import(self, extractor):
        result = extractor.extract("test.java",
                                    "import java.util.List;")
        assert len(result) >= 1
        assert result[0].kind == "import"

    def test_full_python_file(self, extractor):
        path = _TEST_DATA / "example.py"
        content = path.read_text()
        result = extractor.extract(str(path), content)
        names = {s.name for s in result}
        assert "Config" in names
        assert "BaseHandler" in names
        assert "compute" in names
        assert "__init__" in names
        assert "handle" in names
        assert "_process" in names

    def test_full_javascript_file(self, extractor):
        path = _TEST_DATA / "example.js"
        content = path.read_text()
        result = extractor.extract(str(path), content)
        names = {s.name for s in result}
        assert "greet" in names
        assert "Calculator" in names
        assert "multiply" in names

    def test_full_rust_file(self, extractor):
        path = _TEST_DATA / "example.rs"
        content = path.read_text()
        result = extractor.extract(str(path), content)
        names = {s.name for s in result}
        assert "Config" in names
        assert "Handler" in names
        assert "Status" in names
        assert "compute" in names

    def test_full_go_file(self, extractor):
        path = _TEST_DATA / "example.go"
        content = path.read_text()
        result = extractor.extract(str(path), content)
        names = {s.name for s in result}
        assert "Config" in names
        assert "Handler" in names
        assert "compute" in names
        assert "Handle" in names

    def test_full_java_file(self, extractor):
        path = _TEST_DATA / "example.java"
        content = path.read_text()
        result = extractor.extract(str(path), content)
        names = {s.name for s in result}
        assert "Application" in names
        assert "compute" in names


class TestCreateExtractor:
    def test_create_returns_composite_extractor(self):
        ext = create_extractor()
        assert isinstance(ext, CompositeExtractor)

    def test_create_extractor_extracts(self):
        ext = create_extractor()
        result = ext.extract("test.py", "def foo(x): pass")
        assert len(result) >= 1
        assert result[0].name == "foo"
