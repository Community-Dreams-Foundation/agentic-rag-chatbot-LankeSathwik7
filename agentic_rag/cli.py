from __future__ import annotations

import argparse
import json

from agentic_rag.memory import select_high_signal_memory, write_memories
from agentic_rag.pipeline import RAGPipeline
from agentic_rag.sanity import run_sanity
from agentic_rag.weather import analyze_open_meteo_timeseries
from agentic_rag.webapp import run_server


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Agentic RAG Chatbot CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="Ingest files and build index")
    p_ingest.add_argument("--paths", nargs="+", required=True, help="File or folder paths")
    p_ingest.add_argument("--index", default="artifacts/index.json")

    p_ask = sub.add_parser("ask", help="Ask grounded question")
    p_ask.add_argument("--question", required=True)
    p_ask.add_argument("--index", default="artifacts/index.json")
    p_ask.add_argument("--top-k", type=int, default=5)

    p_memory = sub.add_parser("remember", help="Extract and write high-signal memory")
    p_memory.add_argument("--text", required=True)
    p_memory.add_argument("--user-memory", default="USER_MEMORY.md")
    p_memory.add_argument("--company-memory", default="COMPANY_MEMORY.md")

    p_weather = sub.add_parser("weather", help="Open-Meteo analytics")
    p_weather.add_argument("--lat", required=True, type=float)
    p_weather.add_argument("--lon", required=True, type=float)
    p_weather.add_argument("--start-date", required=True)
    p_weather.add_argument("--end-date", required=True)

    p_sanity = sub.add_parser("sanity", help="Run end-to-end sanity flow")
    p_sanity.add_argument("--output", default="artifacts/sanity_output.json")

    p_serve = sub.add_parser("serve", help="Run lightweight web UI")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=7860)
    p_serve.add_argument("--index", default="artifacts/index.json")

    p_hist = sub.add_parser("history", help="Read session history")
    p_hist.add_argument("--session-id", default="default")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "ingest":
        pipeline = RAGPipeline()
        stats = pipeline.ingest(args.paths)
        pipeline.save(args.index)
        print(json.dumps({"status": "ok", **stats, "index": args.index}, indent=2))
        return

    if args.command == "ask":
        pipeline = RAGPipeline.load(args.index)
        result = pipeline.ask(args.question, top_k=args.top_k)
        print(json.dumps(result.to_dict(), indent=2))
        return

    if args.command == "remember":
        decisions = select_high_signal_memory(args.text)
        writes = write_memories(decisions, args.user_memory, args.company_memory)
        print(
            json.dumps(
                {
                    "status": "ok",
                    "decisions": [d.to_dict() for d in decisions],
                    "written": writes,
                },
                indent=2,
            )
        )
        return

    if args.command == "weather":
        try:
            result = analyze_open_meteo_timeseries(
                latitude=args.lat,
                longitude=args.lon,
                start_date=args.start_date,
                end_date=args.end_date,
            )
            print(json.dumps(result.to_dict(), indent=2))
        except ValueError as exc:
            print(json.dumps({"error": f"Invalid weather input: {exc}"}, indent=2))
        return

    if args.command == "sanity":
        payload = run_sanity(args.output)
        print(json.dumps(payload, indent=2))
        return

    if args.command == "serve":
        run_server(host=args.host, port=args.port, index_path=args.index)
        return

    if args.command == "history":
        from agentic_rag.history import read_session_history

        events = read_session_history(args.session_id)
        print(json.dumps({"session_id": args.session_id, "history": events}, indent=2))
        return


if __name__ == "__main__":
    main()
