#!/usr/bin/env python3
"""
Run this once (and again on every scheduled refresh - blueprint 2.4) to
populate Qdrant + the BM25 index from data/sample_schemes.json.

Usage:
    python scripts/run_ingestion.py

Requires QDRANT_URL (and QDRANT_API_KEY if using Qdrant Cloud) to be set,
either in .env or the environment. Does NOT require GROQ_API_KEY - that's
only needed at query time (Stage 7).
"""
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.ingestion.ingest import ingest_sample_corpus  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

if __name__ == "__main__":
    count = ingest_sample_corpus()
    print(f"Ingested {count} chunks from the sample corpus.")
    print("Next: start the API with `uvicorn app.main:app --reload` and POST to /query.")
