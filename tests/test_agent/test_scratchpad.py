from __future__ import annotations

from chef_human.agent.scratchpad import Scratchpad


class TestScratchpadEmpty:
    def test_is_empty_initially(self):
        assert Scratchpad().is_empty() is True

    def test_render_shows_hint(self):
        result = Scratchpad().render()
        assert "empty" in result.lower()
        assert "decision" in result.lower()


class TestScratchpadAddLine:
    def test_tagged_decision(self):
        sp = Scratchpad()
        sp.add_line("[decision] Used SQLite because no DB is configured")
        assert sp.entries["decision"] == ["Used SQLite because no DB is configured"]
        assert sp.is_empty() is False

    def test_tagged_file(self):
        sp = Scratchpad()
        sp.add_line("[file] created db.py")
        assert sp.entries["file"] == ["created db.py"]

    def test_tagged_assumption(self):
        sp = Scratchpad()
        sp.add_line("[assumption] user wants Python 3.12+")
        assert sp.entries["assumption"] == ["user wants Python 3.12+"]

    def test_tagged_question(self):
        sp = Scratchpad()
        sp.add_line("[question] should schema be normalized?")
        assert sp.entries["question"] == ["should schema be normalized?"]

    def test_tag_is_case_insensitive(self):
        sp = Scratchpad()
        sp.add_line("[DECISION] uppercase tag")
        assert sp.entries["decision"] == ["uppercase tag"]

    def test_untagged_line_becomes_note(self):
        sp = Scratchpad()
        sp.add_line("just a plain note, old format")
        assert sp.entries["note"] == ["just a plain note, old format"]

    def test_empty_line_ignored(self):
        sp = Scratchpad()
        sp.add_line("   ")
        assert sp.is_empty() is True

    def test_duplicate_entry_not_added_twice(self):
        sp = Scratchpad()
        sp.add_line("[decision] pick SQLite")
        sp.add_line("[decision] pick SQLite")
        assert sp.entries["decision"] == ["pick SQLite"]


class TestScratchpadAddLines:
    def test_accumulates_across_calls(self):
        sp = Scratchpad()
        sp.add_lines(["[decision] first decision"])
        sp.add_lines(["[decision] second decision", "[file] created a.py"])
        assert sp.entries["decision"] == ["first decision", "second decision"]
        assert sp.entries["file"] == ["created a.py"]

    def test_entries_survive_across_many_updates(self):
        """The core fix: notes accumulate instead of the last update wiping
        out everything that came before."""
        sp = Scratchpad()
        for i in range(5):
            sp.add_lines([f"[decision] decision {i}"])
        assert len(sp.entries["decision"]) == 5


class TestScratchpadRender:
    def test_render_groups_by_category(self):
        sp = Scratchpad()
        sp.add_line("[decision] use SQLite")
        sp.add_line("[file] created db.py")
        rendered = sp.render()
        assert "Decisions:" in rendered
        assert "use SQLite" in rendered
        assert "Files touched:" in rendered
        assert "created db.py" in rendered

    def test_render_omits_empty_categories(self):
        sp = Scratchpad()
        sp.add_line("[decision] only this")
        rendered = sp.render()
        assert "Decisions:" in rendered
        assert "Files touched:" not in rendered
        assert "Open questions:" not in rendered

    def test_render_includes_untagged_notes(self):
        sp = Scratchpad()
        sp.add_line("legacy free-text note")
        rendered = sp.render()
        assert "Notes:" in rendered
        assert "legacy free-text note" in rendered
