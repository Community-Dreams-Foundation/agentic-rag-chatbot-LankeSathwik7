import json
import sys
import threading
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agentic_rag.memory import select_high_signal_memory, write_memories
from agentic_rag.pipeline import RAGPipeline
from agentic_rag.sanity import run_sanity
from agentic_rag.weather import analyze_open_meteo_timeseries
from agentic_rag.webapp import run_server


def _assert(name: str, cond: bool, detail: str, results: list[dict]) -> None:
    if not cond:
        raise AssertionError(f"{name} failed: {detail}")
    results.append({"test": name, "status": "PASS", "detail": detail})


def _post(path: str, body: dict, port: int) -> dict:
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=20) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _get(path: str, port: int) -> tuple[str, str]:
    with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=20) as resp:
        return resp.headers.get("Content-Type", ""), resp.read().decode("utf-8")


def main() -> None:
    results: list[dict] = []

    # Backend path
    pipeline = RAGPipeline()
    ingest_stats = pipeline.ingest(
        ["sample_docs/solar_finance_brief.txt", "sample_docs/operations_notes.txt"],
        append=False,
    )
    _assert(
        "backend_ingest",
        ingest_stats["documents"] == 2 and ingest_stats["chunks"] >= 2,
        str(ingest_stats),
        results,
    )
    pipeline.save("artifacts/index_test_full.json")

    qa_ok = pipeline.ask("What are the key assumptions or limitations?")
    _assert(
        "backend_qa_grounded",
        bool(qa_ok.citations) and "cannot find" not in qa_ok.answer.lower(),
        f"citations={len(qa_ok.citations)}",
        results,
    )

    qa_refusal = pipeline.ask("What is the CEO phone number?")
    _assert(
        "backend_refusal",
        "cannot find" in qa_refusal.answer.lower(),
        qa_refusal.answer,
        results,
    )

    # Memory path
    mem_secret = select_high_signal_memory("My API key is abc and password is 1234")
    _assert("memory_secret_filter", len(mem_secret) == 0, "secret blocked", results)

    mem = select_high_signal_memory("I prefer Friday digests at 7 AM.")
    writes = write_memories(mem)
    _assert("memory_write", isinstance(writes, list), f"writes={len(writes)}", results)

    # Weather tool
    weather = analyze_open_meteo_timeseries(40.71, -74.01, "2026-02-01", "2026-02-05")
    _assert("weather_tool", bool(weather.explanation), weather.explanation[:100], results)

    # Sanity artifact contract
    sanity = run_sanity("artifacts/sanity_output_test_full.json")
    _assert(
        "sanity_contract",
        all(k in sanity for k in ("implemented_features", "qa", "demo")),
        "top-level keys present",
        results,
    )

    # Frontend->middleware->backend path
    port = 7873
    thread = threading.Thread(
        target=run_server,
        kwargs={"host": "127.0.0.1", "port": port, "index_path": "artifacts/index_test_full.json"},
        daemon=True,
    )
    thread.start()
    time.sleep(0.8)

    ct, html = _get("/", port)
    _assert("web_frontend_root", "text/html" in ct and "Agentic RAG Arena" in html, "ui served", results)

    upload = _post(
        "/api/upload",
        {"files": [{"name": "web_flow.txt", "content": "Planning variance dropped 22 percent."}]},
        port,
    )
    _assert("web_upload", upload.get("status") == "ok", json.dumps(upload), results)

    ask = _post("/api/ask", {"session_id": "e2e-ui", "question": "What numeric detail is mentioned?"}, port)
    _assert("web_ask", len(ask.get("citations", [])) > 0, f"citations={len(ask.get('citations', []))}", results)

    mem_out = _post(
        "/api/memory",
        {"session_id": "e2e-ui", "text": "I prefer monthly summaries every first Monday."},
        port,
    )
    _assert("web_memory", mem_out.get("status") == "ok", f"writes={len(mem_out.get('writes', []))}", results)

    _, history_raw = _get("/api/history?session_id=e2e-ui", port)
    history = json.loads(history_raw)
    _assert("web_history", len(history.get("history", [])) >= 2, f"events={len(history.get('history', []))}", results)

    _, sessions_raw = _get("/api/sessions", port)
    sessions = json.loads(sessions_raw)
    _assert(
        "web_sessions",
        "e2e-ui" in sessions.get("sessions", []),
        f"sessions={sessions.get('sessions', [])}",
        results,
    )

    print(json.dumps({"summary": {"passed": len(results), "failed": 0}, "results": results}, indent=2))


if __name__ == "__main__":
    main()
