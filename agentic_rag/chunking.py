from __future__ import annotations

import hashlib
from collections.abc import Iterable

from agentic_rag.ingestion import RawDocument
from agentic_rag.models import DocumentChunk
from agentic_rag.utils import tokenize


def _is_heading(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if stripped.startswith("#"):
        return True
    if stripped.endswith(":") and len(stripped.split()) <= 8:
        return True
    if stripped.isupper() and len(stripped.split()) <= 8:
        return True
    return False


def _sectioned_lines(doc: RawDocument) -> Iterable[tuple[str, int, str]]:
    section = "Document"
    for i, line in enumerate(doc.lines, start=1):
        if _is_heading(line):
            section = line.strip().lstrip("#").strip() or "Document"
        yield section, i, line


def chunk_document(
    doc: RawDocument,
    chunk_token_size: int = 130,
    overlap_tokens: int = 30,
) -> list[DocumentChunk]:
    chunks: list[DocumentChunk] = []
    doc_key = hashlib.md5(doc.source_path.encode("utf-8")).hexdigest()[:8]
    rolling: list[tuple[int, str]] = []
    rolling_section = "Document"
    chunk_index = 0

    def flush_chunk() -> None:
        nonlocal rolling, chunk_index
        if not rolling:
            return
        start_line = rolling[0][0]
        end_line = rolling[-1][0]
        text = "\n".join(line for _, line in rolling).strip()
        if not text:
            rolling = []
            return
        chunk_id = f"{doc.source}::{doc_key}::chunk_{chunk_index:03d}"
        chunks.append(
            DocumentChunk(
                chunk_id=chunk_id,
                source=doc.source,
                section=rolling_section,
                start_line=start_line,
                end_line=end_line,
                text=text,
            )
        )
        chunk_index += 1
        if overlap_tokens <= 0:
            rolling = []
            return
        backfill: list[tuple[int, str]] = []
        token_budget = 0
        for line_no, line_text in reversed(rolling):
            t_count = len(tokenize(line_text))
            if token_budget + t_count > overlap_tokens and backfill:
                break
            backfill.append((line_no, line_text))
            token_budget += t_count
            if token_budget >= overlap_tokens:
                break
        rolling = list(reversed(backfill))

    for section, line_no, line in _sectioned_lines(doc):
        if not rolling:
            rolling_section = section
        rolling.append((line_no, line))
        current_tokens = len(tokenize("\n".join(txt for _, txt in rolling)))
        if current_tokens >= chunk_token_size:
            flush_chunk()
    flush_chunk()
    return chunks


def chunk_documents(docs: list[RawDocument]) -> list[DocumentChunk]:
    all_chunks: list[DocumentChunk] = []
    for doc in docs:
        all_chunks.extend(chunk_document(doc))
    return all_chunks
