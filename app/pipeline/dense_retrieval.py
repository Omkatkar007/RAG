"""
Stage 3 (dense half): Qdrant ANN search over cosine similarity (blueprint 2.2 row 3).

The lexical half lives in lexical_retrieval.py; fusion.py combines the two
result lists via Reciprocal Rank Fusion.
"""
from __future__ import annotations

import logging

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from app.config import settings
from app.pipeline.embeddings import embed_query
from app.schemas import RetrievedChunk

logger = logging.getLogger(__name__)

_client: QdrantClient | None = None


def get_qdrant_client() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key or None)
    return _client


def ensure_collection() -> None:
    client = get_qdrant_client()
    existing = [c.name for c in client.get_collections().collections]
    if settings.qdrant_collection not in existing:
        client.create_collection(
            collection_name=settings.qdrant_collection,
            vectors_config=qmodels.VectorParams(
                size=settings.embedding_dim,
                distance=qmodels.Distance.COSINE,
            ),
        )
        logger.info("Created Qdrant collection '%s'", settings.qdrant_collection)


def dense_search(query: str, top_k: int | None = None) -> list[RetrievedChunk]:
    top_k = top_k or settings.dense_top_k
    client = get_qdrant_client()
    query_vector = embed_query(query)

    results = client.query_points(
        collection_name=settings.qdrant_collection,
        query=query_vector.tolist(),
        limit=top_k,
        with_payload=True,
    ).points

    chunks: list[RetrievedChunk] = []
    for hit in results:
        payload = hit.payload or {}
        chunks.append(
            RetrievedChunk(
                chunk_id=str(hit.id),
                scheme_name=payload.get("scheme_name", "Unknown scheme"),
                ministry=payload.get("ministry"),
                text=payload.get("text", ""),
                source_url=payload.get("source_url"),
                last_verified=payload.get("last_verified"),
                dense_score=hit.score,
            )
        )
    return chunks
