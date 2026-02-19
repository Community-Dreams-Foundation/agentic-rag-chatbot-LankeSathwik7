from __future__ import annotations

import math
from dataclasses import dataclass

from agentic_rag.models import DocumentChunk
from agentic_rag.utils import char_ngrams, jaccard, token_counts, tokenize


@dataclass
class RetrievalHit:
    chunk: DocumentChunk
    score: float
    lexical_score: float
    semantic_score: float


class HybridRetriever:
    def __init__(self, chunks: list[DocumentChunk]):
        self.chunks = chunks
        self.doc_tokens = [tokenize(c.text) for c in chunks]
        self.df: dict[str, int] = {}
        self.avg_doc_len = 0.0
        self._build_stats()

    def _build_stats(self) -> None:
        if not self.doc_tokens:
            self.avg_doc_len = 0.0
            return
        total_len = 0
        for toks in self.doc_tokens:
            total_len += len(toks)
            seen = set(toks)
            for t in seen:
                self.df[t] = self.df.get(t, 0) + 1
        self.avg_doc_len = total_len / len(self.doc_tokens)

    def _bm25(self, query_tokens: list[str], doc_tokens: list[str]) -> float:
        if not query_tokens or not doc_tokens or self.avg_doc_len == 0.0:
            return 0.0
        tf = token_counts(doc_tokens)
        score = 0.0
        n_docs = len(self.doc_tokens)
        k1 = 1.5
        b = 0.75
        doc_len = len(doc_tokens)
        for token in query_tokens:
            df = self.df.get(token, 0)
            if df == 0:
                continue
            idf = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
            f = tf.get(token, 0)
            denom = f + k1 * (1 - b + b * (doc_len / self.avg_doc_len))
            if denom == 0:
                continue
            score += idf * ((f * (k1 + 1)) / denom)
        return score

    def search(self, query: str, top_k: int = 5) -> list[RetrievalHit]:
        q_tokens = tokenize(query)
        q_ngrams = char_ngrams(query, n=3)
        hits: list[RetrievalHit] = []
        for chunk, doc_toks in zip(self.chunks, self.doc_tokens):
            lexical = self._bm25(q_tokens, doc_toks)
            semantic = jaccard(q_ngrams, char_ngrams(chunk.text, n=3))
            coverage = 0.0
            if q_tokens:
                overlap = len(set(q_tokens) & set(doc_toks))
                coverage = overlap / len(set(q_tokens))
            score = 0.60 * lexical + 0.30 * semantic + 0.10 * coverage
            hits.append(
                RetrievalHit(
                    chunk=chunk,
                    score=score,
                    lexical_score=lexical,
                    semantic_score=semantic,
                )
            )
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:top_k]

