"""
Ingestion pipeline (build order phase 1, blueprint 2.7): load -> chunk ->
embed -> upsert to Qdrant, and build the parallel in-memory BM25 index.

Uses metadata-aware chunking (2.3/2.4 - recommended primary strategy for
structured scheme data), chunking per eligibility criterion rather than
per paragraph.
"""
from __future__ import annotations

import logging
import uuid

from qdrant_client.http import models as qmodels
from tenacity import retry, stop_after_attempt, wait_exponential

from app.ingestion.chunking import chunk_metadata_aware
from app.ingestion.loaders import load_sample_schemes, load_data_gov_schemes
from app.pipeline import lexical_retrieval
from app.pipeline.dense_retrieval import ensure_collection, get_qdrant_client
from app.pipeline.embeddings import embed_texts
from app.config import settings

logger = logging.getLogger(__name__)

# Upserting thousands of points in one HTTP request is what causes dropped
# connections (seen in practice against Qdrant Cloud's free tier) - batch
# instead. Also lets you see progress on a large real-world corpus instead
# of staring at a silent terminal for minutes.
UPSERT_BATCH_SIZE = 100


@retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=2, min=2, max=30))
def _upsert_batch(client, collection_name: str, points: list) -> None:
    client.upsert(collection_name=collection_name, points=points)


def ingest_schemes(schemes: list[dict]) -> int:
    """Chunks + embeds + upserts every scheme dict. Returns total chunk count."""
    all_chunks = []
    for scheme in schemes:
        chunks = chunk_metadata_aware(
            scheme_name=scheme["scheme_name"],
            ministry=scheme.get("ministry"),
            eligibility_clauses=scheme.get("eligibility_clauses", []),
            benefit_amount=scheme.get("benefit_amount"),
            category_state_applicability=scheme.get("category_state_applicability"),
            source_url=scheme.get("source_url"),
            last_verified=scheme.get("last_verified"),
            description=scheme.get("description"),
        )
        all_chunks.extend(chunks)

    if not all_chunks:
        logger.warning("No chunks produced from ingestion input.")
        return 0

    logger.info("Embedding %d chunks from %d schemes...", len(all_chunks), len(schemes))

    # --- Dense: embed + upsert to Qdrant, in batches ---
    ensure_collection()
    client = get_qdrant_client()
    texts = [c.text for c in all_chunks]
    vectors = embed_texts(texts)  # sentence-transformers batches internally too

    points = [
        qmodels.PointStruct(
            id=str(uuid.uuid4()),
            vector=vector.tolist(),
            payload=chunk.to_payload(),
        )
        for chunk, vector in zip(all_chunks, vectors)
    ]
    # Use the point's generated UUID as the authoritative chunk_id so dense
    # and lexical results reference the same identifier.
    for point, chunk in zip(points, all_chunks):
        point.payload["chunk_id"] = point.id
        chunk.chunk_id = point.id

    total_batches = (len(points) + UPSERT_BATCH_SIZE - 1) // UPSERT_BATCH_SIZE
    for i in range(0, len(points), UPSERT_BATCH_SIZE):
        batch = points[i:i + UPSERT_BATCH_SIZE]
        batch_num = i // UPSERT_BATCH_SIZE + 1
        _upsert_batch(client, settings.qdrant_collection, batch)
        logger.info("Upserted batch %d/%d (%d points)", batch_num, total_batches, len(batch))

    logger.info("Upserted %d chunks total to Qdrant collection '%s'", len(points), settings.qdrant_collection)

    # --- Lexical: build BM25 index over the same chunks ---
    bm25_payloads = [c.to_payload() for c in all_chunks]
    lexical_retrieval.build_index(bm25_payloads)
    logger.info("Built BM25 index over %d chunks", len(bm25_payloads))

    return len(all_chunks)


def ingest_sample_corpus() -> int:
    schemes = load_sample_schemes()
    logger.info("Loaded %d sample schemes", len(schemes))
    return ingest_schemes(schemes)


def ingest_data_gov_corpus(resource_id: str, limit: int = 100) -> int:
    schemes = load_data_gov_schemes(resource_id, limit=limit)
    logger.info("Loaded %d schemes from data.gov.in (resource: %s)", len(schemes), resource_id)
    return ingest_schemes(schemes)


def rebuild_bm25_index_from_qdrant(scroll_batch_size: int = 500) -> int:
    """
    Rebuilds the local BM25 index by reading whatever chunks already exist
    in Qdrant, instead of re-running ingestion from the original source.

    Why this matters: `data/bm25_index.pkl` is a local file, excluded from
    the Docker image (.dockerignore) and never committed to git - so a
    fresh container/environment starts with no BM25 index, even though
    Qdrant Cloud already has your real ingested data. If a container
    naively re-ran `ingest_sample_corpus()` on every cold start, it would
    (a) create duplicate points in Qdrant every restart, since each chunk
    gets a fresh UUID, and (b) rebuild BM25 from only the 8-scheme sample
    corpus, silently losing lexical search over whatever larger corpus
    (e.g. a 3,400-scheme CSV import) you'd actually ingested.

    This function fixes both: it treats Qdrant as the source of truth and
    derives BM25 from it, so BM25 always matches what's actually
    searchable, and nothing gets re-inserted.
    """
    client = get_qdrant_client()
    all_payloads: list[dict] = []
    next_offset = None

    while True:
        records, next_offset = client.scroll(
            collection_name=settings.qdrant_collection,
            limit=scroll_batch_size,
            offset=next_offset,
            with_payload=True,
            with_vectors=False,
        )
        all_payloads.extend(r.payload for r in records if r.payload)
        if next_offset is None:
            break

    if not all_payloads:
        logger.warning(
            "No points found in Qdrant collection '%s' - nothing to build BM25 from. "
            "Run ingestion first.", settings.qdrant_collection,
        )
        return 0

    lexical_retrieval.build_index(all_payloads)
    logger.info("Rebuilt BM25 index from %d existing Qdrant points", len(all_payloads))
    return len(all_payloads)
