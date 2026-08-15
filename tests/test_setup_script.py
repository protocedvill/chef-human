from __future__ import annotations

import os
import subprocess
from pathlib import Path


def _fake_python(path: Path, version: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"#!/usr/bin/env bash\necho {version}\n")
    path.chmod(0o755)


def test_setup_rejects_existing_venv_from_different_python(tmp_path: Path) -> None:
    selected = tmp_path / "python3.12"
    venv_dir = tmp_path / "venv"
    _fake_python(selected, "3.12")
    _fake_python(venv_dir / "bin" / "python", "3.13")
    script = Path(__file__).parents[1] / "scripts" / "setup.sh"
    env = {
        **os.environ,
        "CH_INSTALL_PYTHON": str(selected),
        "CH_INSTALL_VENV": str(venv_dir),
    }

    result = subprocess.run(
        ["bash", str(script)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert "uses Python 3.13" in result.stderr
    assert "selected Python 3.12" in result.stderr
