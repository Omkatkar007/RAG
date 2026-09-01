"""
Stage 5: Reranking via a local cross-encoder (ms-marco-MiniLM-L-6-v2).
Runs entirely locally - no Groq calls for reranking, no extra API cost
(blueprint 2.2 row 6).
"""
from __future__ import annotations

import threading

from sentence_transformers import CrossEncoder

from app.config import settings
from app.schemas import RetrievedChunk

_reranker: CrossEncoder | None = None
_lock = threading.Lock()


def get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        with _lock:
            if _reranker is None:
                _reranker = CrossEncoder(settings.reranker_model)
    return _reranker


def rerank(query: str, candidates: list[RetrievedChunk], top_n: int | None = None) -> list[RetrievedChunk]:
    if not candidates:
        return []

    top_n = top_n or settings.rerank_top_n
    model = get_reranker()

    pairs = [(query, c.text) for c in candidates]
    scores = model.predict(pairs)

    for chunk, score in zip(candidates, scores):
        chunk.rerank_score = float(score)

    ranked = sorted(candidates, key=lambda c: c.rerank_score or float("-inf"), reverse=True)
    return ranked[:top_n]
