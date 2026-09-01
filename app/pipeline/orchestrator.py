"""
Ties together stages 1-8 exactly as diagrammed in blueprint 2.1:

User (Voice or Text)
  -> [1] STT (voice only)
  -> [2] Query Processor
  -> [3] Hybrid Retrieval (Dense + Lexical)
  -> [4] Fusion (RRF)
  -> [5] Reranking (Cross-Encoder)
  -> [6] Context Builder
  -> [7] LLM Generation
  -> [8] Guardrails (off-topic / safety / sufficiency / grounding)
  -> Final Answer + Citations

The structured eligibility-verification addition (2.5) runs after
generation passes guardrails, once per scheme that appears in the
final context.
"""
from __future__ import annotations

import logging

from app.pipeline import context_builder, dense_retrieval, fusion, generation, guardrails, lexical_retrieval, stt
from app.pipeline.query_processor import extract_entities
from app.schemas import InputType, QueryRequest, QueryResponse, SchemeVerdict

logger = logging.getLogger(__name__)


def run_pipeline(request: QueryRequest) -> QueryResponse:
    # --- Stage 1: STT (voice only) ---
    if request.input_type == InputType.VOICE:
        if not request.audio_base64:
            return QueryResponse(answer="", blocked=True, block_reason="No audio provided for voice input.")
        raw_text = stt.transcribe_audio(request.audio_base64, request.audio_language_hint)
    else:
        if not request.text:
            return QueryResponse(answer="", blocked=True, block_reason="No text provided.")
        raw_text = request.text

    # --- Stage 2: Query processing ---
    entities = extract_entities(raw_text)

    # --- Guardrails, pass 1: off-topic + safety (cheap, run before spending
    # retrieval/generation compute on a query we're going to reject anyway) ---
    early_ok, early_trace = guardrails.run_guardrail_pipeline(raw_text, chunks=[])
    # run_guardrail_pipeline needs chunks for the sufficiency check, so we only
    # trust the first two entries (off_topic, safety) from this early call.
    offtopic_and_safety = [r for r in early_trace if r.layer in ("off_topic", "safety")]
    if any(not r.passed for r in offtopic_and_safety):
        failed = next(r for r in offtopic_and_safety if not r.passed)
        return QueryResponse(
            answer="I can only help with questions about Indian government welfare schemes and eligibility.",
            blocked=True,
            block_reason=failed.reason,
            guardrail_trace=offtopic_and_safety,
        )

    # --- Stage 3: Hybrid retrieval ---
    dense_results = dense_retrieval.dense_search(raw_text)
    lexical_results = lexical_retrieval.lexical_search(raw_text)

    # --- Stage 4: Fusion (RRF) ---
    fused_results = fusion.reciprocal_rank_fusion(dense_results, lexical_results)

    # --- Stage 5: Reranking ---
    reranked = guardrails_safe_rerank(raw_text, fused_results)

    # --- Guardrails: sufficiency (needs the reranked chunks) ---
    sufficiency_result = guardrails.check_sufficiency(reranked)
    if not sufficiency_result.passed:
        return QueryResponse(
            answer=(
                "I don't have enough verified information in my current scheme database "
                "to answer this confidently. Please check the official myscheme.gov.in "
                "portal or your nearest Common Service Centre."
            ),
            blocked=True,
            block_reason=sufficiency_result.reason,
            guardrail_trace=offtopic_and_safety + [sufficiency_result],
            citations=reranked,
        )

    # --- Stage 6: Context builder ---
    context_string, chunks_used = context_builder.build_context(reranked)

    # --- Stage 7: Generation ---
    answer = generation.generate_answer(raw_text, context_string)

    # --- Guardrails: grounding ---
    grounding_result = guardrails.check_grounding(answer, chunks_used)
    full_trace = offtopic_and_safety + [sufficiency_result, grounding_result]

    if not grounding_result.passed:
        return QueryResponse(
            answer=(
                "I found some potentially relevant scheme information, but I couldn't "
                "generate an answer that's reliably grounded in it. Please check "
                "myscheme.gov.in directly for accurate details."
            ),
            blocked=True,
            block_reason=grounding_result.reason,
            guardrail_trace=full_trace,
            citations=chunks_used,
        )

    # --- Structured eligibility verification (2.5 recommended addition) ---
    verdicts: list[SchemeVerdict] = []
    seen_schemes: set[str] = set()
    for chunk in chunks_used:
        if chunk.scheme_name in seen_schemes:
            continue
        seen_schemes.add(chunk.scheme_name)
        try:
            verdict = guardrails.verify_eligibility_for_scheme(
                scheme_name=chunk.scheme_name,
                scheme_context_text=chunk.text,
                entities=entities,
                source_url=chunk.source_url,
                last_verified=chunk.last_verified,
            )
            verdicts.append(verdict)
        except Exception:
            logger.exception("Eligibility verification failed for scheme %s", chunk.scheme_name)

    return QueryResponse(
        answer=answer,
        verdicts=verdicts,
        citations=chunks_used,
        guardrail_trace=full_trace,
        blocked=False,
    )


def guardrails_safe_rerank(query: str, candidates):
    """Thin wrapper so orchestrator only imports one module for the hot path."""
    from app.pipeline.reranker import rerank
    return rerank(query, candidates)
