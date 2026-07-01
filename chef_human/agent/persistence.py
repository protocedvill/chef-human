from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_SAVE_DIR = Path.home() / ".cache" / "chef-human" / "sessions"


def save_conversation(
    conversation: dict,
    task: str,
    save_dir: str | Path = DEFAULT_SAVE_DIR,
    session_id: str | None = None,
) -> Path:
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if session_id is None:
        session_id = hashlib.sha256(
            f"{task}-{time.time()}".encode()
        ).hexdigest()[:12]

    path = save_dir / f"session_{session_id}.json"
    data = {
        "session_id": session_id,
        "task": task,
        "conversation": conversation,
    }
    path.write_text(json.dumps(data, indent=2))
    logger.info("Conversation saved to %s", path)
    return path


def load_conversation(
    session_id: str,
    save_dir: str | Path = DEFAULT_SAVE_DIR,
) -> dict | None:
    path = Path(save_dir) / f"session_{session_id}.json"
    if not path.exists():
        logger.warning("Session file not found: %s", path)
        return None
    data = json.loads(path.read_text())
    return data.get("conversation")


def load_session_data(
    session_id: str,
    save_dir: str | Path = DEFAULT_SAVE_DIR,
) -> dict | None:
    path = Path(save_dir) / f"session_{session_id}.json"
    if not path.exists():
        logger.warning("Session file not found: %s", path)
        return None
    return json.loads(path.read_text())


def delete_session(
    session_id: str,
    save_dir: str | Path = DEFAULT_SAVE_DIR,
) -> bool:
    path = Path(save_dir) / f"session_{session_id}.json"
    if not path.exists():
        return False
    path.unlink()
    logger.info("Session deleted: %s", session_id)
    return True


def list_sessions(
    save_dir: str | Path = DEFAULT_SAVE_DIR,
) -> list[dict]:
    save_dir = Path(save_dir)
    if not save_dir.exists():
        return []
    sessions = []
    for f in sorted(save_dir.glob("session_*.json"), reverse=True):
        data = json.loads(f.read_text())
        sessions.append({
            "session_id": data.get("session_id"),
            "task": data.get("task"),
            "path": str(f),
        })
    return sessions
