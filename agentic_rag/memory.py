from __future__ import annotations

import datetime as dt
import re
from pathlib import Path

from agentic_rag.models import MemoryWrite
from agentic_rag.utils import normalize_whitespace


USER_PATTERNS = (
    r"\bi am an? ([^.]+)",
    r"\bi'?m an? ([^.]+)",
    r"\bi prefer ([^.]+)",
    r"\bplease (?:send|share) ([^.]+)",
    r"\bi work as ([^.]+)",
)

COMPANY_PATTERNS = (
    r"\bour (?:team|org|organization|company) ([^.]+)",
    r"\bthe workflow bottleneck is ([^.]+)",
    r"\bwe usually ([^.]+)",
    r"\bwe often ([^.]+)",
)

SECRET_PATTERNS = (
    r"\bpassword\b",
    r"\bapi[_ -]?key\b",
    r"\bsecret\b",
    r"\btoken\b",
    r"\bssn\b",
    r"\b\d{3}[-.\s]?\d{2}[-.\s]?\d{4}\b",
)


def _contains_secret(text: str) -> bool:
    lowered = text.lower()
    return any(re.search(p, lowered) for p in SECRET_PATTERNS)


def _extract_patterns(text: str, patterns: tuple[str, ...], target: str) -> list[MemoryWrite]:
    decisions: list[MemoryWrite] = []
    for pattern in patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            summary = normalize_whitespace(match.group(0)).rstrip(".")
            if len(summary) < 12:
                continue
            confidence = 0.92 if "prefer" in summary.lower() else 0.85
            decisions.append(
                MemoryWrite(
                    target=target,
                    summary=summary[0].upper() + summary[1:],
                    confidence=confidence,
                    reason=f"Matched pattern: {pattern}",
                )
            )
    return decisions


def select_high_signal_memory(user_text: str) -> list[MemoryWrite]:
    text = normalize_whitespace(user_text)
    if not text or _contains_secret(text):
        return []
    decisions = []
    decisions.extend(_extract_patterns(text, USER_PATTERNS, target="USER"))
    decisions.extend(_extract_patterns(text, COMPANY_PATTERNS, target="COMPANY"))
    # Confidence gate and simple dedupe.
    result: list[MemoryWrite] = []
    seen = set()
    for d in decisions:
        key = d.summary.lower()
        if d.confidence < 0.80 or key in seen:
            continue
        seen.add(key)
        result.append(d)
    # Remove weaker summaries that are substrings of stronger, longer summaries.
    result.sort(key=lambda x: (x.target, -len(x.summary), -x.confidence))
    filtered: list[MemoryWrite] = []
    for candidate in result:
        lowered = candidate.summary.lower()
        if any(
            candidate.target == kept.target and lowered in kept.summary.lower() for kept in filtered
        ):
            continue
        filtered.append(candidate)
    return filtered


def _append_if_new(path: Path, summary: str, confidence: float) -> bool:
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    if summary.lower() in existing.lower():
        return False
    stamp = dt.date.today().isoformat()
    line = f"- {stamp} | confidence={confidence:.2f} | {summary}\n"
    with path.open("a", encoding="utf-8") as f:
        if existing and not existing.endswith("\n"):
            f.write("\n")
        f.write(line)
    return True


def write_memories(
    memories: list[MemoryWrite],
    user_memory_path: str = "USER_MEMORY.md",
    company_memory_path: str = "COMPANY_MEMORY.md",
) -> list[dict[str, str]]:
    writes: list[dict[str, str]] = []
    user_path = Path(user_memory_path)
    company_path = Path(company_memory_path)
    for mem in memories:
        target_path = user_path if mem.target == "USER" else company_path
        if _append_if_new(target_path, mem.summary, mem.confidence):
            writes.append({"target": mem.target, "summary": mem.summary})
    return writes
