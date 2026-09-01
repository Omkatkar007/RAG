"""
Stage 4: Fusion via Reciprocal Rank Fusion, k=60 (blueprint 2.1/2.2).

RRF merges the dense and lexical candidate lists purely by rank position
(not raw score, which isn't comparable across a cosine-similarity list and
a BM25 list): fused_score(doc) = sum over lists of 1 / (k + rank_in_list).
"""
from __future__ import annotations

from app.config import settings
from app.schemas import RetrievedChunk


def reciprocal_rank_fusion(
    dense_results: list[RetrievedChunk],
    lexical_results: list[RetrievedChunk],
    k: int | None = None,
) -> list[RetrievedChunk]:
    k = k or settings.rrf_k

    fused: dict[str, RetrievedChunk] = {}
    scores: dict[str, float] = {}

    for rank, chunk in enumerate(dense_results, start=1):
        scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + 1.0 / (k + rank)
        fused[chunk.chunk_id] = chunk

    for rank, chunk in enumerate(lexical_results, start=1):
        scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + 1.0 / (k + rank)
        if chunk.chunk_id in fused:
            # Merge lexical score onto the already-seen (dense) chunk record.
            fused[chunk.chunk_id].lexical_score = chunk.lexical_score
        else:
            fused[chunk.chunk_id] = chunk

    for chunk_id, chunk in fused.items():
        chunk.fused_score = scores[chunk_id]

    ranked = sorted(fused.values(), key=lambda c: c.fused_score or 0.0, reverse=True)
    return ranked
