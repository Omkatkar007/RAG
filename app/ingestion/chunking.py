"""
Chunking strategies (blueprint 2.3). Pick per document type; metadata-aware
is the recommended primary strategy for myscheme.gov.in-style structured
scheme data (2.4).

Note on metadata-aware chunking (2.3 callout): the fields preserved per
chunk here are scheme name, ministry/department, eligibility clause,
benefit amount, category/state applicability, and source link +
last-verified date - NOT an MSMARCO-style field set (passage ID, query ID,
relevance labels), which is built for a search-ranking benchmark, not
scheme documents.
"""
from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field

from app.pipeline.embeddings import cosine_similarity, embed_texts

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass
class Chunk:
    chunk_id: str
    scheme_name: str
    text: str
    ministry: str | None = None
    eligibility_clause: str | None = None
    benefit_amount: str | None = None
    category_state_applicability: str | None = None
    source_url: str | None = None
    last_verified: str | None = None
    extra_metadata: dict = field(default_factory=dict)

    def to_payload(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "scheme_name": self.scheme_name,
            "text": self.text,
            "ministry": self.ministry,
            "source_url": self.source_url,
            "last_verified": self.last_verified,
        }


def _split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]


def chunk_semantic(scheme_name: str, text: str, similarity_threshold: float = 0.55, **meta) -> list[Chunk]:
    """
    Topic-boundary detection via sentence-embedding similarity. Best for
    long-form scheme descriptions with multiple sub-topics.
    Groups consecutive sentences while similarity to the running group stays
    above threshold; starts a new chunk when it drops.
    """
    sentences = _split_sentences(text)
    if not sentences:
        return []

    embeddings = embed_texts(sentences)
    chunks: list[Chunk] = []
    current_sentences = [sentences[0]]
    current_embedding = embeddings[0]

    for sentence, emb in zip(sentences[1:], embeddings[1:]):
        sim = cosine_similarity(current_embedding, emb)
        if sim >= similarity_threshold:
            current_sentences.append(sentence)
            # running average embedding (cheap approximation of topic centroid)
            current_embedding = (current_embedding + emb) / 2
        else:
            chunks.append(_make_chunk(scheme_name, " ".join(current_sentences), meta))
            current_sentences = [sentence]
            current_embedding = emb

    chunks.append(_make_chunk(scheme_name, " ".join(current_sentences), meta))
    return chunks


def chunk_sentence_window(scheme_name: str, text: str, group_size: int = 4, overlap: int = 1, **meta) -> list[Chunk]:
    """Groups of `group_size` sentences with `overlap`-sentence overlap. General-purpose fallback."""
    sentences = _split_sentences(text)
    if not sentences:
        return []

    chunks: list[Chunk] = []
    step = max(group_size - overlap, 1)
    for i in range(0, len(sentences), step):
        window = sentences[i:i + group_size]
        if not window:
            continue
        chunks.append(_make_chunk(scheme_name, " ".join(window), meta))
        if i + group_size >= len(sentences):
            break
    return chunks


def chunk_fixed_window(scheme_name: str, text: str, window_chars: int = 800, overlap_chars: int = 150, **meta) -> list[Chunk]:
    """Character-based windows with configurable overlap. For unstructured/poorly formatted source text."""
    if not text:
        return []
    chunks: list[Chunk] = []
    step = max(window_chars - overlap_chars, 1)
    for i in range(0, len(text), step):
        window = text[i:i + window_chars].strip()
        if window:
            chunks.append(_make_chunk(scheme_name, window, meta))
        if i + window_chars >= len(text):
            break
    return chunks


def chunk_metadata_aware(
    scheme_name: str,
    ministry: str | None,
    eligibility_clauses: list[str],
    benefit_amount: str | None,
    category_state_applicability: str | None,
    source_url: str | None,
    last_verified: str | None,
    description: str | None = None,
) -> list[Chunk]:
    """
    Recommended primary strategy for structured scheme databases (2.4).
    Chunks per eligibility criterion rather than per paragraph, which makes
    retrieval much more precise for eligibility questions specifically.
    Preserves full metadata per chunk.
    """
    chunks: list[Chunk] = []

    if description:
        chunks.append(Chunk(
            chunk_id=str(uuid.uuid4()),
            scheme_name=scheme_name,
            text=f"{scheme_name}: {description}",
            ministry=ministry,
            benefit_amount=benefit_amount,
            category_state_applicability=category_state_applicability,
            source_url=source_url,
            last_verified=last_verified,
        ))

    for clause in eligibility_clauses:
        chunk_text = f"{scheme_name} eligibility criterion: {clause}"
        if category_state_applicability:
            chunk_text += f" (Applicability: {category_state_applicability})"
        chunks.append(Chunk(
            chunk_id=str(uuid.uuid4()),
            scheme_name=scheme_name,
            text=chunk_text,
            ministry=ministry,
            eligibility_clause=clause,
            benefit_amount=benefit_amount,
            category_state_applicability=category_state_applicability,
            source_url=source_url,
            last_verified=last_verified,
        ))

    if benefit_amount:
        chunks.append(Chunk(
            chunk_id=str(uuid.uuid4()),
            scheme_name=scheme_name,
            text=f"{scheme_name} benefit: {benefit_amount}",
            ministry=ministry,
            benefit_amount=benefit_amount,
            source_url=source_url,
            last_verified=last_verified,
        ))

    return chunks


def _make_chunk(scheme_name: str, text: str, meta: dict) -> Chunk:
    return Chunk(chunk_id=str(uuid.uuid4()), scheme_name=scheme_name, text=text, **meta)
