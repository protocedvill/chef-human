from __future__ import annotations

from pathlib import Path

import pytest

from chef_human.agent.workspace import WorkspaceManager


class TestWorkspaceManagerInit:
    def test_default_root_is_cwd(self):
        wm = WorkspaceManager()
        assert wm.root == Path.cwd().resolve()

    def test_custom_root(self, tmp_path: Path):
        wm = WorkspaceManager(root=tmp_path)
        assert wm.root == tmp_path.resolve()

    def test_root_property(self, tmp_path: Path):
        wm = WorkspaceManager(root=tmp_path)
        assert wm.root == wm._root

    def test_root_resolves_relative(self, tmp_path: Path):
        sub = tmp_path / "sub"
        sub.mkdir()
        wm = WorkspaceManager(root=sub)
        assert wm.root == sub.resolve()


class TestResolve:
    def test_resolve_relative_path(self, tmp_path: Path):
        wm = WorkspaceManager(root=tmp_path)
        result = wm.resolve("some/file.txt")
        assert result == (tmp_path / "some/file.txt").resolve()

    def test_resolve_absolute_path(self, tmp_path: Path):
        wm = WorkspaceManager(root=tmp_path)
        target = tmp_path / "target.txt"
        result = wm.resolve(str(target))
        assert result == target.resolve()

    def test_resolve_outside_path(self, tmp_path: Path):
        wm = WorkspaceManager(root=tmp_path)
        outside = Path("/tmp/outside.txt")
        result = wm.resolve(str(outside))
        assert str(result) == "/tmp/outside.txt"

    def test_resolve_dot(self, tmp_path: Path):
        wm = WorkspaceManager(root=tmp_path)
        result = wm.resolve(".")
        assert result == tmp_path.resolve()


class TestIsWithinWorkspace:
    def test_root_is_within(self, tmp_path: Path):
        wm = WorkspaceManager(root=tmp_path)
        assert wm.is_within_workspace(str(tmp_path))

    def test_subdirectory_is_within(self, tmp_path: Path):
        sub = tmp_path / "subdir"
        sub.mkdir()
        wm = WorkspaceManager(root=tmp_path)
        assert wm.is_within_workspace(str(sub))

    def test_outside_path_is_not_within(self, tmp_path: Path):
        wm = WorkspaceManager(root=tmp_path)
        assert not wm.is_within_workspace("/tmp")

    def test_sibling_path_is_not_within(self, tmp_path: Path):
        wm = WorkspaceManager(root=tmp_path)
        sibling = tmp_path.parent / "other"
        sibling.mkdir(exist_ok=True)
        assert not wm.is_within_workspace(str(sibling))

    def test_relative_dotdot_outside_is_not_within(self, tmp_path: Path):
        wm = WorkspaceManager(root=tmp_path / "inner")
        (tmp_path / "inner").mkdir(exist_ok=True)
        assert not wm.is_within_workspace("..")


class TestIsIgnored:
    def test_dotgit_is_ignored(self, tmp_path: Path):
        (tmp_path / ".git").mkdir(exist_ok=True)
        wm = WorkspaceManager(root=tmp_path)
        assert wm.is_ignored(str(tmp_path / ".git"))

    def test_pycache_is_ignored(self, tmp_path: Path):
        (tmp_path / "__pycache__").mkdir(exist_ok=True)
        wm = WorkspaceManager(root=tmp_path)
        assert wm.is_ignored(str(tmp_path / "__pycache__"))

    def test_node_modules_is_ignored(self, tmp_path: Path):
        (tmp_path / "node_modules").mkdir(exist_ok=True)
        wm = WorkspaceManager(root=tmp_path)
        assert wm.is_ignored(str(tmp_path / "node_modules"))

    def test_regular_file_is_not_ignored(self, tmp_path: Path):
        (tmp_path / "main.py").touch()
        wm = WorkspaceManager(root=tmp_path)
        assert not wm.is_ignored(str(tmp_path / "main.py"))

    def test_pyc_file_is_ignored(self, tmp_path: Path):
        (tmp_path / "module.pyc").touch()
        wm = WorkspaceManager(root=tmp_path)
        assert wm.is_ignored(str(tmp_path / "module.pyc"))

    def test_dot_venv_is_ignored(self, tmp_path: Path):
        (tmp_path / ".venv").mkdir(exist_ok=True)
        wm = WorkspaceManager(root=tmp_path)
        assert wm.is_ignored(str(tmp_path / ".venv"))

    def test_ignored_in_subdirectory(self, tmp_path: Path):
        (tmp_path / "src/__pycache__").mkdir(parents=True)
        wm = WorkspaceManager(root=tmp_path)
        assert wm.is_ignored(str(tmp_path / "src/__pycache__"))

    def test_not_ignored_when_outside_workspace(self, tmp_path: Path):
        wm = WorkspaceManager(root=tmp_path)
        assert not wm.is_ignored("/tmp")

    def test_gitignore_pattern_respected(self, tmp_path: Path):
        (tmp_path / ".gitignore").write_text("*.log\n")
        (tmp_path / "debug.log").touch()
        wm = WorkspaceManager(root=tmp_path)
        assert wm.is_ignored(str(tmp_path / "debug.log"))

    def test_gitignore_slash_dir(self, tmp_path: Path):
        (tmp_path / ".gitignore").write_text("build/\n")
        (tmp_path / "build").mkdir(exist_ok=True)
        wm = WorkspaceManager(root=tmp_path)
        assert wm.is_ignored(str(tmp_path / "build"))

    def test_gitignore_comment_ignored(self, tmp_path: Path):
        (tmp_path / ".gitignore").write_text("# comment\n*.o\n")
        (tmp_path / "main.o").touch()
        wm = WorkspaceManager(root=tmp_path)
        assert wm.is_ignored(str(tmp_path / "main.o"))

    def test_normal_file_not_ignored_by_gitignore(self, tmp_path: Path):
        (tmp_path / ".gitignore").write_text("*.log\n")
        (tmp_path / "main.py").touch()
        wm = WorkspaceManager(root=tmp_path)
        assert not wm.is_ignored(str(tmp_path / "main.py"))


class TestMatchGitignore:
    @pytest.mark.parametrize(
        ("name", "pattern", "expected"),
        [
            ("main.py", "*.py", True),
            ("main.pyc", "*.pyc", True),
            ("main.py", "*.pyc", False),
            ("build", "build/", True),
            ("build_dir", "build/", False),
            ("foo", "foo", True),
            ("foo", "bar", False),
            ("debug.log", "*.log", True),
            ("log.txt", "*.log", False),
        ],
    )
    def test_matches(self, name: str, pattern: str, expected: bool):
        assert WorkspaceManager._match_gitignore(name, pattern) == expected


class TestListFiles:
    def test_empty_directory(self, tmp_path: Path):
        wm = WorkspaceManager(root=tmp_path)
        assert wm.list_files() == []

    def test_lists_all_files(self, tmp_path: Path):
        (tmp_path / "a.py").touch()
        (tmp_path / "b.py").touch()
        wm = WorkspaceManager(root=tmp_path)
        files = wm.list_files()
        assert len(files) == 2

    def test_ignores_git_dir(self, tmp_path: Path):
        (tmp_path / "a.py").touch()
        (tmp_path / ".git/heads/main").parent.mkdir(parents=True)
        (tmp_path / ".git/heads/main").touch()
        wm = WorkspaceManager(root=tmp_path)
        files = wm.list_files()
        assert len(files) == 1
        assert files[0].name == "a.py"

    def test_respects_max_depth(self, tmp_path: Path):
        (tmp_path / "a/b/c/d/e/f.txt").parent.mkdir(parents=True)
        (tmp_path / "a/b/c/d/e/f.txt").touch()
        wm = WorkspaceManager(root=tmp_path)
        assert wm.list_files(max_depth=2) == []
        assert len(wm.list_files(max_depth=6)) == 1

    def test_returns_sorted(self, tmp_path: Path):
        (tmp_path / "z.py").touch()
        (tmp_path / "a.py").touch()
        (tmp_path / "m.py").touch()
        wm = WorkspaceManager(root=tmp_path)
        names = [f.name for f in wm.list_files()]
        assert names == ["a.py", "m.py", "z.py"]

    def test_nonexistent_directory(self, tmp_path: Path):
        wm = WorkspaceManager(root=tmp_path)
        assert wm.list_files(directory="nonexistent") == []

    def test_nested_respects_gitignore(self, tmp_path: Path):
        (tmp_path / ".gitignore").write_text("*.pyc\n")
        (tmp_path / "src/main.py").parent.mkdir()
        (tmp_path / "src/main.py").touch()
        (tmp_path / "src/module.pyc").touch()
        wm = WorkspaceManager(root=tmp_path)
        files = wm.list_files()
        # .gitignore itself is listed, main.py is listed, module.pyc is ignored
        assert len(files) == 2
        assert all(f.suffix != ".pyc" for f in files)


class TestDiscoverRoot:
    def test_discover_from_git(self, tmp_path: Path):
        (tmp_path / ".git").mkdir()
        inner = tmp_path / "sub/deep"
        inner.mkdir(parents=True)
        root = WorkspaceManager.discover_root(inner)
        assert root == tmp_path.resolve()

    def test_discover_from_pyproject(self, tmp_path: Path):
        (tmp_path / "pyproject.toml").touch()
        inner = tmp_path / "src/pkg"
        inner.mkdir(parents=True)
        root = WorkspaceManager.discover_root(inner)
        assert root == tmp_path.resolve()

    def test_returns_current_when_no_marker(self, tmp_path: Path):
        root = WorkspaceManager.discover_root(tmp_path)
        assert root == tmp_path.resolve()

    def test_discover_from_package_json(self, tmp_path: Path):
        (tmp_path / "package.json").touch()
        root = WorkspaceManager.discover_root(tmp_path / "sub")
        assert root == tmp_path.resolve()

    def test_deeply_nested_discovery(self, tmp_path: Path):
        (tmp_path / ".git").mkdir()
        deep = tmp_path / "a/b/c/d/e/f"
        deep.mkdir(parents=True)
        root = WorkspaceManager.discover_root(deep)
        assert root == tmp_path.resolve()
