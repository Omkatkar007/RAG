"""
Stage 6: Context Builder. Token-budgeted, 3-5 chunks, max 1500 tokens
(blueprint 2.1/2.2). Greedily adds reranked chunks until either the chunk
count ceiling or the token budget is hit, whichever comes first, but never
goes below MIN_CONTEXT_CHUNKS if enough candidates exist.
"""
from __future__ import annotations

import tiktoken

from app.config import settings
from app.schemas import RetrievedChunk

_encoder = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_encoder.encode(text))


def build_context(reranked_chunks: list[RetrievedChunk]) -> tuple[str, list[RetrievedChunk]]:
    """
    Returns (context_string, chunks_used). context_string is formatted with
    inline source tags so the LLM (and the grounding guardrail afterwards)
    can trace each sentence back to a specific scheme + source.
    """
    selected: list[RetrievedChunk] = []
    total_tokens = 0

    for chunk in reranked_chunks:
        if len(selected) >= settings.max_context_chunks:
            break

        chunk_text = _format_chunk(chunk)
        chunk_tokens = count_tokens(chunk_text)

        if total_tokens + chunk_tokens > settings.max_context_tokens:
            if len(selected) >= settings.min_context_chunks:
                break
            # Still under the minimum chunk floor - allow one more chunk even
            # if it slightly exceeds budget, then stop.
            selected.append(chunk)
            total_tokens += chunk_tokens
            break

        selected.append(chunk)
        total_tokens += chunk_tokens

    context_string = "\n\n---\n\n".join(_format_chunk(c) for c in selected)
    return context_string, selected


def _format_chunk(chunk: RetrievedChunk) -> str:
    header = f"[Source: {chunk.scheme_name}"
    if chunk.ministry:
        header += f" | {chunk.ministry}"
    if chunk.last_verified:
        header += f" | last verified {chunk.last_verified}"
    header += f" | chunk_id={chunk.chunk_id}]"
    return f"{header}\n{chunk.text}"
