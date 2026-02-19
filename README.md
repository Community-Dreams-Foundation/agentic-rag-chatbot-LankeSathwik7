[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/P5MLsfQv)
# Agentic RAG Chatbot - Competition Submission

This project implements:
- Feature A: file-grounded Q&A with citations (hybrid retrieval + reranking).
- Feature B: selective persistent memory writing to markdown.
- Extra: optional Open-Meteo time-series analytics tool (weather API + basic stats).
- Extra: lightweight web UI with multi-session history and file upload.

## Participant Info (Required)
- Full Name: Lanke Sathwik
- Email: lankesathwik7@gmail.com
- GitHub Username: LankeSathwik7

## Video Walkthrough
- Link: TODO_ADD_VIDEO_LINK

## Quick Start (Required)
Run from repo root. **No external dependencies** — stdlib only (Python 3.10+).

```bash
# 1) Python version
python --version

# 2) Ingest sample docs and build index
python -m agentic_rag ingest --paths sample_docs/solar_finance_brief.txt sample_docs/operations_notes.txt --index artifacts/index.json

# 3) Ask grounded question with citations
python -m agentic_rag ask --index artifacts/index.json --question "What are the key assumptions or limitations?"

# 4) Write selective memory
python -m agentic_rag remember --text "I am a Project Finance Analyst and I prefer weekly summaries on Mondays."

# 5) Optional weather analytics tool (Open-Meteo)
python -m agentic_rag weather --lat 40.71 --lon -74.01 --start-date 2025-12-01 --end-date 2025-12-10

# 6) Optional web UI
python -m agentic_rag serve --host 127.0.0.1 --port 7860
# open http://127.0.0.1:7860
```

Judge command:

```bash
make sanity
```

This generates:
- `artifacts/sanity_output.json` (required)

Optional validator:

```bash
bash scripts/sanity_check.sh
```

Full local E2E check (frontend/API/backend roundtrip):

```bash
python scripts/e2e_full_check.py
```

## What Was Built

### Feature A: RAG + Citations
- Ingestion: UTF-8 text/markdown files from file or folder paths.
- Chunking: section-aware chunking with token overlap.
- Retrieval: hybrid scoring:
  - lexical BM25-style score
  - semantic character-trigram Jaccard score
  - query-term coverage bonus
- Grounded answering:
  - extractive answer from retrieved chunks only
  - citations include `source`, `locator`, `snippet`
  - refusal behavior when retrieval confidence is low
- Prompt-injection awareness:
  - if a doc says "ignore previous instructions", that line gets skipped in the answer.

### Feature B: Persistent Memory
- Selective memory extraction (regex + confidence gating).
- Sensitive-content filter avoids writing secrets.
- Writes only high-signal summaries to:
  - `USER_MEMORY.md`
  - `COMPANY_MEMORY.md`
- Deduplicates prior memories.

### Extra: Feature C-style Tooling
- Optional Open-Meteo analytics command.
- Computes:
  - missingness
  - mean temperature
  - volatility (population std dev)
  - anomaly flags by z-score threshold
- Timeouts on API calls, no arbitrary code execution.

### Extra: Web UI + Session History
- Built-in web server (no external framework dependency).
- Browser file upload -> local indexed documents.
- Session-scoped Q&A and memory events persisted as JSONL logs under `artifacts/sessions/`.
- Session history is queryable via CLI and UI.

## Project Structure

```text
agentic_rag/
  cli.py            # CLI entrypoint and commands
  ingestion.py      # file loading/discovery
  chunking.py       # section-aware chunking
  retrieval.py      # hybrid retriever
  qa.py             # grounded answer generation + citations
  memory.py         # selective memory decisions and writes
  weather.py        # optional Open-Meteo analytics
  history.py        # session event persistence
  webapp.py         # lightweight HTTP server + APIs
  web/index.html    # frontend UI
  pipeline.py       # ingestion/retrieval orchestration
  sanity.py         # required e2e sanity output generator
scripts/
  sanity_check.sh
  verify_output.py
sample_docs/
  solar_finance_brief.txt
  operations_notes.txt
```

## Sanity Output Contract
`make sanity` produces `artifacts/sanity_output.json` with:
- `implemented_features`
- `qa`
- `demo`

Schema is compatible with `scripts/verify_output.py`.

## Suggested Demo Questions
See `EVAL_QUESTIONS.md`.
