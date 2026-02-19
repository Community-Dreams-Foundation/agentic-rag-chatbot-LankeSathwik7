#!/usr/bin/env bash
set -euo pipefail

echo "== Agentic RAG Sanity Check =="

rm -rf artifacts
mkdir -p artifacts

echo "Running: make sanity"
make sanity

OUT="artifacts/sanity_output.json"
if [[ ! -f "$OUT" ]]; then
  echo "ERROR: Missing $OUT"
  echo "Your 'make sanity' must generate: artifacts/sanity_output.json"
  exit 1
fi

if command -v python3 >/dev/null 2>&1; then
  python3 scripts/verify_output.py "$OUT"
else
  python scripts/verify_output.py "$OUT"
fi

echo "OK: sanity check passed"
