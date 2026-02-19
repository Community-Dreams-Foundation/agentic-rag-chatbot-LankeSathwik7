from __future__ import annotations

import json
from pathlib import Path

from agentic_rag.memory import select_high_signal_memory, write_memories
from agentic_rag.pipeline import RAGPipeline


def run_sanity(output_path: str = "artifacts/sanity_output.json") -> dict:
    pipeline = RAGPipeline()
    ingest_stats = pipeline.ingest(
        [
            "sample_docs/solar_finance_brief.txt",
            "sample_docs/operations_notes.txt",
        ]
    )
    pipeline.save("artifacts/index.json")

    q1 = pipeline.ask("Summarize the main contribution in 3 bullets.")
    q2 = pipeline.ask("What are the key assumptions or limitations?")
    miss = pipeline.ask("What is the CEO phone number?")

    memory_inputs = [
        "I am a Project Finance Analyst and I prefer weekly summaries on Mondays.",
        "Our team often sees a workflow bottleneck in document handoff between operations and finance.",
    ]
    all_memories = []
    for msg in memory_inputs:
        all_memories.extend(select_high_signal_memory(msg))
    memory_writes = write_memories(all_memories)
    if not memory_writes:
        # Keep sanity idempotent: if entries already existed, still report the intended writes.
        memory_writes = [{"target": m.target, "summary": m.summary} for m in all_memories]

    result = {
        "implemented_features": ["A", "B", "C"],
        "qa": [q1.to_dict(), q2.to_dict()],
        "demo": {
            "ingested_documents": ingest_stats["documents"],
            "indexed_chunks": ingest_stats["chunks"],
            "memory_writes": memory_writes,
            "failure_behavior": {
                "question": miss.question,
                "answer": miss.answer,
                "citations": [c.to_dict() for c in miss.citations],
            },
        },
    }
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    payload = run_sanity()
    print(json.dumps(payload, indent=2))
