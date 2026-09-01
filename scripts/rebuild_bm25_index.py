#!/usr/bin/env python3
"""
Rebuilds data/bm25_index.pkl from whatever's already in Qdrant, without
re-running ingestion from the original source. Use this:
  - After moving to a new machine/environment where Qdrant already has data
    but the local BM25 pickle file doesn't exist yet.
  - As the container startup step (see Dockerfile) instead of re-ingesting,
    so a fresh container matches your real data instead of the 8-scheme
    sample corpus.

Usage:
    python scripts/rebuild_bm25_index.py
"""
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.ingestion.ingest import rebuild_bm25_index_from_qdrant  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

if __name__ == "__main__":
    count = rebuild_bm25_index_from_qdrant()
    print(f"Rebuilt BM25 index from {count} existing Qdrant points.")
