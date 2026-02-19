from __future__ import annotations

import math
import re
from collections import Counter


WORD_RE = re.compile(r"[a-zA-Z0-9']+")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "have",
    "how",
    "i",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "this",
    "to",
    "was",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
    "you",
    "your",
    "give",
    "one",
}


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in WORD_RE.findall(text) if t.lower() not in STOPWORDS]


def sentence_split(text: str) -> list[str]:
    text = normalize_whitespace(text)
    if not text:
        return []
    return [s.strip() for s in SENTENCE_SPLIT_RE.split(text) if s.strip()]


def token_counts(tokens: list[str]) -> Counter[str]:
    return Counter(tokens)


# def cosine_sim(v1: dict[str, float], v2: dict[str, float]) -> float:
#     if not v1 or not v2:
#         return 0.0
#     dot = 0.0
#     for key, val in v1.items():
#         dot += val * v2.get(key, 0.0)
#     n1 = math.sqrt(sum(x * x for x in v1.values()))
#     n2 = math.sqrt(sum(x * x for x in v2.values()))
#     if n1 == 0.0 or n2 == 0.0:
#         return 0.0
#     return dot / (n1 * n2)


def char_ngrams(text: str, n: int = 3) -> set[str]:
    cleaned = re.sub(r"\s+", " ", text.lower()).strip()
    if len(cleaned) < n:
        return {cleaned} if cleaned else set()
    return {cleaned[i : i + n] for i in range(len(cleaned) - n + 1)}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)
