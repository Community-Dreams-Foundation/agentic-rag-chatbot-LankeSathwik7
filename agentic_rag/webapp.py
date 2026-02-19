from __future__ import annotations

import json
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from agentic_rag.history import append_session_event, list_sessions, read_session_history
from agentic_rag.memory import select_high_signal_memory, write_memories
from agentic_rag.pipeline import RAGPipeline


INDEX_PATH = "artifacts/index.json"
UPLOAD_DIR = Path("artifacts/uploads")
WEB_ROOT = Path(__file__).parent / "web"


class AppState:
    def __init__(self, index_path: str):
        self.index_path = index_path
        self.pipeline = RAGPipeline.load(index_path)
        self.lock = threading.Lock()


def _json_response(handler: BaseHTTPRequestHandler, data: dict, code: int = 200) -> None:
    payload = json.dumps(data).encode("utf-8")
    handler.send_response(code)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(payload)))
    handler.end_headers()
    handler.wfile.write(payload)


def _read_json(handler: BaseHTTPRequestHandler) -> dict:
    length = int(handler.headers.get("Content-Length", "0"))
    raw = handler.rfile.read(length).decode("utf-8") if length > 0 else "{}"
    return json.loads(raw or "{}")


def _safe_name(filename: str) -> str:
    raw = Path(filename).name
    keep = "".join(ch for ch in raw if ch.isalnum() or ch in ("-", "_", ".", " "))
    return keep.strip() or "uploaded.txt"


def _write_uploaded_files(files: list[dict]) -> list[str]:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []
    for item in files:
        name = _safe_name(str(item.get("name", "uploaded.txt")))
        content = str(item.get("content", ""))
        out = UPLOAD_DIR / name
        out.write_text(content, encoding="utf-8")
        saved.append(str(out))
    return saved


def make_handler(state: AppState):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args) -> None:
            return

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path == "/":
                html = (WEB_ROOT / "index.html").read_bytes()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(html)))
                self.end_headers()
                self.wfile.write(html)
                return
            if parsed.path == "/api/sessions":
                _json_response(self, {"sessions": list_sessions()})
                return
            if parsed.path == "/api/history":
                params = parse_qs(parsed.query)
                session_id = params.get("session_id", ["default"])[0]
                history = read_session_history(session_id)
                _json_response(self, {"session_id": session_id, "history": history})
                return
            _json_response(self, {"error": "Not found"}, code=404)

        def do_POST(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            try:
                body = _read_json(self)
            except json.JSONDecodeError:
                _json_response(self, {"error": "Invalid JSON body"}, code=400)
                return

            if parsed.path == "/api/upload":
                files = body.get("files", [])
                if not isinstance(files, list) or len(files) == 0:
                    _json_response(self, {"error": "files[] is required"}, code=400)
                    return
                paths = _write_uploaded_files(files)
                with state.lock:
                    stats = state.pipeline.ingest(paths, append=True)
                    state.pipeline.save(state.index_path)
                _json_response(self, {"status": "ok", "saved_paths": paths, "stats": stats})
                return

            if parsed.path == "/api/ask":
                question = str(body.get("question", "")).strip()
                session_id = str(body.get("session_id", "default")).strip() or "default"
                if not question:
                    _json_response(self, {"error": "question is required"}, code=400)
                    return
                with state.lock:
                    result = state.pipeline.ask(question)
                payload = result.to_dict()
                append_session_event(session_id, "qa", payload)
                _json_response(self, payload)
                return

            if parsed.path == "/api/memory":
                text = str(body.get("text", "")).strip()
                session_id = str(body.get("session_id", "default")).strip() or "default"
                if not text:
                    _json_response(self, {"error": "text is required"}, code=400)
                    return
                decisions = select_high_signal_memory(text)
                with state.lock:
                    writes = write_memories(decisions)
                event_payload = {
                    "text": text,
                    "decisions": [d.to_dict() for d in decisions],
                    "writes": writes,
                }
                append_session_event(session_id, "memory", event_payload)
                _json_response(self, {"status": "ok", **event_payload})
                return

            _json_response(self, {"error": "Not found"}, code=404)

    return Handler


def run_server(host: str = "127.0.0.1", port: int = 7860, index_path: str = INDEX_PATH) -> None:
    state = AppState(index_path=index_path)
    server = ThreadingHTTPServer((host, port), make_handler(state))
    print(f"Web UI running at http://{host}:{port}")
    server.serve_forever()

