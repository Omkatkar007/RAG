"""
Stage 3 (lexical half): in-memory BM25 (Okapi), genuine BM25 formula rather
than a full-text-search wrapper (blueprint 2.2 row 4). Good for exact matches
on scheme names, numbers, and Acts that dense embeddings tend to blur.

This index is process-local. `build_index` is called once at startup (or by
the ingestion script) and the result is pickled to disk so the API process
can load it without recomputing tokenization on every restart.
"""
from __future__ import annotations

import pickle
import re
import threading
from pathlib import Path

from rank_bm25 import BM25Okapi

from app.config import settings
from app.schemas import RetrievedChunk

_TOKEN_RE = re.compile(r"[a-z0-9]+")

_INDEX_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "bm25_index.pkl"

_lock = threading.Lock()
_bm25: BM25Okapi | None = None
_corpus_metadata: list[dict] = []  # parallel to the tokenized corpus fed into BM25Okapi


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def build_index(chunks: list[dict]) -> None:
    """
    chunks: list of dicts with keys chunk_id, scheme_name, ministry, text,
    source_url, last_verified (same shape ingestion writes to Qdrant payload).
    """
    global _bm25, _corpus_metadata
    tokenized_corpus = [tokenize(c["text"]) for c in chunks]
    with _lock:
        _bm25 = BM25Okapi(tokenized_corpus)
        _corpus_metadata = chunks

    _INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_INDEX_PATH, "wb") as f:
        pickle.dump({"bm25": _bm25, "metadata": _corpus_metadata}, f)


def _load_index_if_needed() -> None:
    global _bm25, _corpus_metadata
    if _bm25 is not None:
        return
    with _lock:
        if _bm25 is not None:
            return
        if not _INDEX_PATH.exists():
            raise RuntimeError(
                "BM25 index not built yet. Run `python scripts/run_ingestion.py` first "
                "(blueprint 2.7 phase 1: data pipeline before retrieval)."
            )
        with open(_INDEX_PATH, "rb") as f:
            data = pickle.load(f)
        _bm25 = data["bm25"]
        _corpus_metadata = data["metadata"]


def lexical_search(query: str, top_k: int | None = None) -> list[RetrievedChunk]:
    top_k = top_k or settings.lexical_top_k
    _load_index_if_needed()
    assert _bm25 is not None

    tokenized_query = tokenize(query)
    scores = _bm25.get_scores(tokenized_query)

    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

    results: list[RetrievedChunk] = []
    for idx in ranked_indices:
        if scores[idx] <= 0:
            continue
        meta = _corpus_metadata[idx]
        results.append(
            RetrievedChunk(
                chunk_id=meta["chunk_id"],
                scheme_name=meta.get("scheme_name", "Unknown scheme"),
                ministry=meta.get("ministry"),
                text=meta.get("text", ""),
                source_url=meta.get("source_url"),
                last_verified=meta.get("last_verified"),
                lexical_score=float(scores[idx]),
            )
        )
    return results
