from __future__ import annotations

import os
import tempfile
from pathlib import Path


def count_lines(content: str) -> int:
    """Line count matching what an editor would show: 0 for empty content,
    and no phantom extra line for content ending in a trailing newline
    (unlike `content.count("\\n") + 1`, which gives 1 for "" and N+1 for
    N lines each terminated by "\\n")."""
    if not content:
        return 0
    return content.count("\n") + (0 if content.endswith("\n") else 1)


def atomic_write_text(path: Path, content: str, encoding: str = "utf-8") -> None:
    """Write `content` to `path` without ever leaving a truncated/partial file on disk.

    Writes to a temp file in the same directory (so the final os.replace is on the
    same filesystem and therefore atomic), then renames it into place. A crash or
    kill mid-write leaves either the old file or the new one, never a half-written
    one, unlike Path.write_text()'s truncate-then-write.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding=encoding) as f:
            f.write(content)
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
