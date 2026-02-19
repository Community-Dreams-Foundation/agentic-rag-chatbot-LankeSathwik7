from __future__ import annotations

import re

from agentic_rag.models import Citation, QAResult
from agentic_rag.retrieval import RetrievalHit
from agentic_rag.utils import normalize_whitespace, sentence_split, tokenize

_HEADING_MARKER_RE = re.compile(r"#{1,6}\s*")


INJECTION_PATTERNS = (
    "ignore previous instructions",
    "reveal secrets",
    "system prompt",
    "developer message",
    "exfiltrate",
)

SENSITIVE_QUERY_TERMS = ("phone", "number", "email", "password", "ssn", "secret", "api key")
NUMERIC_QUERY_TERMS = ("numeric", "number", "percent", "percentage", "metric", "value")


def _is_malicious_sentence(sentence: str) -> bool:
    lowered = sentence.lower()
    return any(p in lowered for p in INJECTION_PATTERNS)


def _sentence_relevance(question_tokens: list[str], sentence: str) -> float:
    stoks = set(tokenize(sentence))
    if not question_tokens or not stoks:
        return 0.0
    overlap = len(set(question_tokens) & stoks)
    return overlap / max(1, len(set(question_tokens)))


def _has_strong_grounding(question_tokens: list[str], hits: list[RetrievalHit]) -> bool:
    if not hits or not question_tokens:
        return False
    top = hits[0]
    chunk_tokens = set(tokenize(top.chunk.text))
    overlap = len(set(question_tokens) & chunk_tokens)
    return overlap > 0 and top.score >= 0.12


def _contains_numeric_request(question: str) -> bool:
    lowered = question.lower()
    return any(term in lowered for term in NUMERIC_QUERY_TERMS)


def _contains_sensitive_request(question: str) -> bool:
    lowered = question.lower()
    return any(term in lowered for term in SENSITIVE_QUERY_TERMS)


def generate_grounded_answer(question: str, hits: list[RetrievalHit]) -> QAResult:
    q_tokens = tokenize(question)
    if not hits:
        return QAResult(
            question=question,
            answer="I cannot find this in the uploaded documents. Please add more relevant files.",
            citations=[],
        )

    if _contains_sensitive_request(question) and not _has_strong_grounding(q_tokens, hits):
        return QAResult(
            question=question,
            answer="I cannot find this in the uploaded documents. Please add more relevant files.",
            citations=[],
        )

    if hits[0].score < 0.08 and not _contains_numeric_request(question):
        return QAResult(
            question=question,
            answer="I cannot find this in the uploaded documents. Please add more relevant files.",
            citations=[],
        )

    candidate_sentences: list[tuple[float, str]] = []
    for hit in hits:
        for sentence in sentence_split(hit.chunk.text):
            if _is_malicious_sentence(sentence):
                continue
            rel = _sentence_relevance(q_tokens, sentence)
            if rel > 0:
                candidate_sentences.append((0.7 * rel + 0.3 * hit.score, sentence))

    candidate_sentences.sort(key=lambda x: x[0], reverse=True)
    selected = [s for _, s in candidate_sentences[:3]]
    if _contains_numeric_request(question):
        numeric_sentences = []
        for hit in hits:
            for sentence in sentence_split(hit.chunk.text):
                if any(ch.isdigit() for ch in sentence):
                    numeric_sentences.append(sentence)
        if numeric_sentences:
            selected = numeric_sentences[:3]

    # Pad to at least 3 sentences from top chunks when candidates are sparse.
    if len(selected) < 3:
        seen = {normalize_whitespace(s) for s in selected}
        for hit in hits:
            for sentence in sentence_split(hit.chunk.text):
                if _is_malicious_sentence(sentence):
                    continue
                clean = normalize_whitespace(sentence)
                if clean not in seen and len(clean) > 20:
                    selected.append(clean)
                    seen.add(clean)
                    if len(selected) >= 3:
                        break
            if len(selected) >= 3:
                break

    if not selected:
        selected = [normalize_whitespace(hits[0].chunk.text)[:280]]

    lines = [f"- {_HEADING_MARKER_RE.sub('', normalize_whitespace(s)).strip()}" for s in selected]
    answer = "Based on the uploaded documents:\n" + "\n".join(lines)

    citations: list[Citation] = []
    for hit in hits[:2]:
        snippet = normalize_whitespace(hit.chunk.text)[:220]
        citations.append(
            Citation(
                source=hit.chunk.source,
                locator=hit.chunk.locator,
                snippet=snippet,
            )
        )
    return QAResult(question=question, answer=answer, citations=citations)
