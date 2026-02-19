# Architecture Overview

## Goal
Deliver a practical agentic RAG assistant with:
- grounded Q&A over uploaded files with citations,
- selective durable memory,
- and an optional safe analytics tool call.

## High-Level Flow

### 1) Ingestion (Upload -> Parse -> Chunk)
- Inputs: `.txt`, `.md`, `.rst`, `.log`.
- Discovery: accepts file paths and folder paths (recursive scan).
- Parsing: UTF-8 text parsing with binary/non-UTF8 skip behavior.
- Chunking: section-aware chunking with overlap.
  - Heading heuristics: markdown headings (`#`), short uppercase headings, short section labels ending with `:`.
  - Chunk metadata:
    - source filename
    - inferred section
    - line-range locator
    - deterministic chunk id

### 2) Indexing / Storage
- Index is persisted as JSON at `artifacts/index.json`.
- Stored item format is chunk-centric (text + metadata).
- Current stack is dependency-light and deterministic, so no external vector DB is required for the baseline challenge flow.

### 3) Retrieval + Grounded Answering
- Hybrid retrieval score combines:
  - BM25-style lexical relevance,
  - character trigram Jaccard similarity (semantic-ish robustness),
  - query-term coverage bonus.
- Top-k chunks are reranked by final weighted score.
- Answering is extractive and grounded:
  - sentence candidates come only from retrieved chunks,
  - most relevant sentences are selected by overlap with query terms,
  - citations are attached with:
    - `source`
    - `locator` (section + line range + chunk id)
    - `snippet`
- Failure behavior:
  - if retrieval confidence is below threshold, returns "cannot find in uploaded documents" instead of guessing.

### 4) Memory System (Selective)
- Memory extraction works in steps:
  - regex patterns pick up "I am a …", "I prefer …", "Our team …", etc.
  - each match gets a confidence score,
  - only writes above 0.80 confidence,
  - anything that looks like a password/key/SSN is blocked.
- High-signal user memory examples:
  - role/job context,
  - stable communication preferences.
- High-signal company memory examples:
  - recurring workflow bottlenecks,
  - reusable process learnings.
- Explicitly not stored:
  - raw transcript dumps,
  - secrets (passwords, API keys, tokens, SSN-like patterns).
- Durable write targets:
  - `USER_MEMORY.md`
  - `COMPANY_MEMORY.md`
- Duplicate prevention:
  - summary-level dedupe against existing memory file content.

### 5) Optional Safe Tooling (Open-Meteo)
- Tool endpoint is fixed to Open-Meteo archive/forecast APIs.
- Inputs are constrained to numeric lat/lon and ISO dates.
- Safety boundaries:
  - fixed request timeout,
  - no arbitrary code execution,
  - no dynamic import or shell handoff.
- Outputs include:
  - missingness checks,
  - average and volatility,
  - anomaly flags using z-score threshold,
  - concise explanation text.

## Tradeoffs & Next Steps

### Why this design
- No API keys or external services needed — just clone and run. Makes it easy for judges to reproduce.
- BM25 alone misses typos and paraphrases, so I added trigram Jaccard as a cheap second signal.
- Answers come directly from chunk sentences, not generated text, so there's less room for making things up.
- Didn't want to require React/Flask/etc. A stdlib HTTP server is enough to show upload, chat, and history.

### What I would improve next
- Support PDF and HTML ingestion (right now it's only plain text / markdown).
- Swap in a real embedding model and a cross-encoder reranker for better retrieval.
- Add file management in the UI — re-index, delete, inspect individual chunks.
- Build a small eval harness that checks expected citations against actual output.
- Per-user memory isolation so multiple people don't clobber each other's files.
