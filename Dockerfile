# Multi-stage build: keep the final image lean by not shipping pip's build cache.
FROM python:3.11-slim AS base

WORKDIR /app

# System deps needed by torch/sentence-transformers at build time (CPU-only).
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY data ./data
COPY scripts ./scripts

# Pre-download the embedding and reranker models at build time, not on first
# request - avoids a slow/failing cold start on a platform with a request
# timeout (Render, Railway, etc. often kill a request that takes too long).
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
    SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'); \
    CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

EXPOSE 8000

# On container start: rebuild the local BM25 index from whatever's already
# in Qdrant (never re-run source ingestion here - see
# rebuild_bm25_index_from_qdrant()'s docstring for why re-ingesting on every
# cold start would duplicate data and silently lose your real corpus).
# If Qdrant is empty (first-ever deploy, nothing ingested yet), this is a
# no-op - run scripts/run_ingestion.py or ingest_myscheme_csv.py manually
# once, from your own machine or a one-off shell on the platform, before
# expecting real answers.
CMD sh -c "python scripts/rebuild_bm25_index.py; \
    uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"
