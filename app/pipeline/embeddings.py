"""
Embedding model: local MiniLM, singleton, 384-dim (blueprint 2.2 row 2).

Used by: dense retrieval (Stage 3), the off-topic guardrail's embedding
similarity check, and the grounding guardrail's embedding similarity check.
Loaded once per process - sentence-transformers handles its own internal
caching, but we still guard against re-instantiating the model object,
since that's the expensive part (model weights + tokenizer init).
"""
from __future__ import annotations

import threading

import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import settings

_model: SentenceTransformer | None = None
_lock = threading.Lock()


def get_embedding_model() -> SentenceTransformer:
    global _model
    if _model is None:
        with _lock:
            if _model is None:
                _model = SentenceTransformer(settings.embedding_model)
    return _model


def embed_texts(texts: list[str]) -> np.ndarray:
    """Returns an (n, embedding_dim) float32 array of L2-normalized embeddings."""
    model = get_embedding_model()
    vectors = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=False,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return vectors.astype(np.float32)


def embed_query(text: str) -> np.ndarray:
    return embed_texts([text])[0]


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    # Vectors from embed_texts are already L2-normalized, so dot product == cosine similarity.
    return float(np.dot(a, b))
