from __future__ import annotations

import json
from pathlib import Path

from agentic_rag.chunking import chunk_documents
from agentic_rag.ingestion import ingest_paths
from agentic_rag.models import DocumentChunk, QAResult
from agentic_rag.qa import generate_grounded_answer
from agentic_rag.retrieval import HybridRetriever


class RAGPipeline:
    def __init__(self, chunks: list[DocumentChunk] | None = None):
        self.chunks = chunks or []
        self.retriever = HybridRetriever(self.chunks)

    def ingest(self, paths: list[str], append: bool = False) -> dict[str, int]:
        docs = ingest_paths(paths)
        new_chunks = chunk_documents(docs)
        if append:
            existing_index = {c.chunk_id: i for i, c in enumerate(self.chunks)}
            replaced = 0
            added = 0
            for chunk in new_chunks:
                if chunk.chunk_id in existing_index:
                    self.chunks[existing_index[chunk.chunk_id]] = chunk
                    replaced += 1
                else:
                    self.chunks.append(chunk)
                    added += 1
        else:
            self.chunks = new_chunks
            replaced = 0
            added = len(new_chunks)
        self.retriever = HybridRetriever(self.chunks)
        return {
            "documents": len(docs),
            "new_chunks": len(new_chunks),
            "added_chunks": added,
            "replaced_chunks": replaced,
            "chunks": len(self.chunks),
            "append_mode": append,
        }

    def ask(self, question: str, top_k: int = 5) -> QAResult:
        hits = self.retriever.search(question, top_k=top_k)
        return generate_grounded_answer(question, hits)

    def save(self, index_path: str = "artifacts/index.json") -> None:
        path = Path(index_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"chunks": [c.to_dict() for c in self.chunks]}
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, index_path: str = "artifacts/index.json") -> "RAGPipeline":
        path = Path(index_path)
        if not path.exists():
            return cls(chunks=[])
        data = json.loads(path.read_text(encoding="utf-8"))
        chunks = [DocumentChunk.from_dict(d) for d in data.get("chunks", [])]
        return cls(chunks=chunks)
