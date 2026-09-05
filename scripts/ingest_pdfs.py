import argparse
import logging
import os
import sys
import uuid
from pathlib import Path

from pypdf import PdfReader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings
from app.ingestion.chunking import chunk_fixed_window
from app.pipeline.embeddings import embed_texts
from app.pipeline.dense_retrieval import ensure_collection, get_qdrant_client
from app.pipeline import lexical_retrieval
from qdrant_client.http import models as qmodels

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

UPSERT_BATCH_SIZE = 100

def _upsert_batch(client, collection_name: str, points: list) -> None:
    client.upsert(collection_name=collection_name, points=points)

def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extracts text from a given PDF file using pypdf."""
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text.strip()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=5, help="Number of PDFs to process")
    parser.add_argument("--dir", type=str, default="text_data", help="Directory containing PDFs")
    args = parser.parse_args()

    pdf_dir = Path(args.dir)
    if not pdf_dir.exists():
        logger.error(f"Directory {pdf_dir} does not exist.")
        sys.exit(1)

    pdf_files = list(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        logger.error(f"No PDFs found in {pdf_dir}")
        sys.exit(1)

    logger.info(f"Found {len(pdf_files)} PDFs. Processing up to {args.limit}...")
    
    all_chunks = []
    
    for i, pdf_path in enumerate(pdf_files[:args.limit]):
        if i % 100 == 0 and i > 0:
            logger.info(f"Processed {i}/{args.limit} PDFs...")
            
        raw_text = extract_text_from_pdf(pdf_path)
        if not raw_text:
            continue
            
        scheme_name = pdf_path.stem.replace("-", " ").title()
        
        # Bypass LLM: chunk the raw text directly
        chunks = chunk_fixed_window(
            scheme_name=scheme_name,
            text=raw_text,
            window_chars=1000,
            overlap_chars=200,
            source_url=f"file://{pdf_path.name}"
        )
        all_chunks.extend(chunks)

    if not all_chunks:
        logger.error("No text could be extracted.")
        sys.exit(1)

    logger.info(f"Extracted {len(all_chunks)} chunks total. Embedding and upserting...")
    
    # Dense: embed + upsert
    ensure_collection()
    client = get_qdrant_client()
    texts = [c.text for c in all_chunks]
    
    # Batch embeddings to save memory
    vectors = []
    batch_size = 256
    for i in range(0, len(texts), batch_size):
        logger.info(f"Embedding batch {i//batch_size + 1}/{(len(texts)+batch_size-1)//batch_size}...")
        batch_texts = texts[i:i+batch_size]
        batch_vectors = embed_texts(batch_texts)
        vectors.extend(batch_vectors.tolist())
    
    points = [
        qmodels.PointStruct(
            id=str(uuid.uuid4()),
            vector=vector,
            payload=chunk.to_payload(),
        )
        for chunk, vector in zip(all_chunks, vectors)
    ]
    
    for point, chunk in zip(points, all_chunks):
        point.payload["chunk_id"] = point.id
        chunk.chunk_id = point.id
        
    total_batches = (len(points) + UPSERT_BATCH_SIZE - 1) // UPSERT_BATCH_SIZE
    for i in range(0, len(points), UPSERT_BATCH_SIZE):
        batch = points[i:i + UPSERT_BATCH_SIZE]
        batch_num = i // UPSERT_BATCH_SIZE + 1
        _upsert_batch(client, settings.qdrant_collection, batch)
        logger.info(f"Upserted batch {batch_num}/{total_batches} ({len(batch)} points)")

    logger.info(f"Upserted {len(points)} chunks total to Qdrant collection '{settings.qdrant_collection}'")

    # Lexical: BM25
    bm25_payloads = [c.to_payload() for c in all_chunks]
    lexical_retrieval.build_index(bm25_payloads)
    logger.info(f"Built BM25 index over {len(bm25_payloads)} chunks")
    
if __name__ == "__main__":
    main()
