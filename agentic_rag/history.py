from __future__ import annotations

import datetime as dt
import json
from pathlib import Path


SESSIONS_DIR = Path("artifacts/sessions")


def _safe_session_id(session_id: str) -> str:
    cleaned = "".join(ch for ch in session_id if ch.isalnum() or ch in ("-", "_"))
    return cleaned or "default"


def append_session_event(session_id: str, event_type: str, payload: dict) -> dict:
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    sid = _safe_session_id(session_id)
    event = {
        "timestamp": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "type": event_type,
        "payload": payload,
    }
    out = SESSIONS_DIR / f"{sid}.jsonl"
    with out.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")
    return event


def read_session_history(session_id: str, limit: int = 200) -> list[dict]:
    sid = _safe_session_id(session_id)
    path = SESSIONS_DIR / f"{sid}.jsonl"
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    if limit > 0:
        lines = lines[-limit:]
    events: list[dict] = []
    for line in lines:
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events


def list_sessions() -> list[str]:
    if not SESSIONS_DIR.exists():
        return []
    files = sorted(
        [p for p in SESSIONS_DIR.glob("*.jsonl") if p.is_file()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return [p.stem for p in files]

