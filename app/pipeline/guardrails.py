"""
Stage 8: Guardrails layer (blueprint 2.5).

Four layers, run in this order:
  1. Off-topic filter   - pattern matching + embedding similarity to a
                           "known topic" reference set.
  2. Safety filter       - pattern matching + keyword/embedding classifiers.
  3. Sufficiency check   - retrieved chunk relevance vs. a minimum threshold.
  4. Grounding check     - word-overlap + embedding similarity between the
                           generated answer and the source chunks.

Plus the recommended addition: structured eligibility verification, which
extracts each scheme's conditions into discrete fields and checks the
user's stated attributes against each one individually (a checklist,
rather than a fuzzy one-shot LLM judgment). This runs after generation
passes grounding, using the LLM only for structured extraction/comparison,
with the query processor's regex-extracted entities as the user-attribute
source of truth.
"""
from __future__ import annotations

import json
import re

from app.config import settings
from app.pipeline.embeddings import cosine_similarity, embed_query, embed_texts
from app.pipeline.generation import _get_client
from app.pipeline.lexical_retrieval import tokenize as qp_tokenize
from app.pipeline.query_processor import looks_offtopic_by_keyword
from app.schemas import EligibilityConditionCheck, ExtractedEntities, GuardrailResult, RetrievedChunk, SchemeVerdict

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Reference sentences that define "on-topic" for the off-topic guardrail.
_TOPIC_REFERENCE_SET = [
    "Indian government welfare schemes, subsidies, pensions, and eligibility criteria",
    "Applying for scholarships, farmer subsidies, housing schemes, or health insurance from the government",
    "Eligibility rules for central and state government schemes in India",
]
_topic_reference_embeddings = None  # lazy-computed


def _get_topic_reference_embeddings():
    global _topic_reference_embeddings
    if _topic_reference_embeddings is None:
        _topic_reference_embeddings = embed_texts(_TOPIC_REFERENCE_SET)
    return _topic_reference_embeddings


_UNSAFE_PATTERNS = [
    r"\bhow to fake\b", r"\bforge (a |an )?document\b", r"\bfalse (income|caste) certificate\b",
    r"\bbribe\b", r"\bignore (previous|prior) instructions\b", r"\bsystem prompt\b",
]


def check_offtopic(raw_query: str) -> GuardrailResult:
    if looks_offtopic_by_keyword(raw_query):
        return GuardrailResult(passed=False, layer="off_topic", reason="Query matched an off-topic keyword.")

    query_embedding = embed_query(raw_query)
    ref_embeddings = _get_topic_reference_embeddings()
    best_similarity = max(cosine_similarity(query_embedding, ref) for ref in ref_embeddings)

    if best_similarity < settings.offtopic_min_similarity:
        return GuardrailResult(
            passed=False, layer="off_topic",
            reason="Query is not sufficiently related to government schemes.",
            score=best_similarity,
        )
    return GuardrailResult(passed=True, layer="off_topic", score=best_similarity)


def check_safety(raw_query: str) -> GuardrailResult:
    lower = raw_query.lower()
    for pattern in _UNSAFE_PATTERNS:
        if re.search(pattern, lower):
            return GuardrailResult(
                passed=False, layer="safety",
                reason="Query matched a disallowed pattern (fraud facilitation or prompt injection).",
            )
    return GuardrailResult(passed=True, layer="safety")


def check_sufficiency(chunks: list[RetrievedChunk]) -> GuardrailResult:
    """
    Cross-encoder (ms-marco-MiniLM-L-6-v2) scores are raw, uncalibrated
    logits - NOT probabilities. They commonly sit in the negative range
    (roughly -12 to +12) even for genuinely relevant matches, because the
    model was trained on general web passages (MS MARCO), not this domain.
    A sigmoid-based 0-1 threshold crushes these to near-zero regardless of
    relevance, so we compare the raw score directly against a raw-score
    floor instead. SUFFICIENCY_MIN_RERANK_SCORE is empirical - if you swap
    corpora or reranker models, re-check a few known-good queries and
    adjust it (a good top match will usually be > -8; unrelated chunks
    usually score well below -10).
    """
    if not chunks:
        return GuardrailResult(passed=False, layer="sufficiency", reason="No chunks retrieved.", score=0.0)

    top_chunk = chunks[0]

    if top_chunk.rerank_score is not None:
        score = top_chunk.rerank_score
        threshold = settings.sufficiency_min_rerank_score
    else:
        # Fallback if reranking was skipped: RRF fused scores are small
        # positive numbers, already on a sensible relative scale.
        score = top_chunk.fused_score or 0.0
        threshold = settings.sufficiency_min_fused_score

    if score < threshold:
        return GuardrailResult(
            passed=False, layer="sufficiency",
            reason="Top retrieved chunk relevance is below the minimum confidence threshold.",
            score=score,
        )
    return GuardrailResult(passed=True, layer="sufficiency", score=score)


def check_grounding(answer: str, chunks: list[RetrievedChunk]) -> GuardrailResult:
    if not chunks:
        return GuardrailResult(passed=False, layer="grounding", reason="No source chunks to ground against.")

    answer_tokens = set(qp_tokenize(answer))
    context_tokens: set[str] = set()
    for c in chunks:
        context_tokens |= set(qp_tokenize(c.text))

    if not answer_tokens:
        return GuardrailResult(passed=False, layer="grounding", reason="Empty answer.")

    overlap = len(answer_tokens & context_tokens) / len(answer_tokens)

    if overlap < settings.grounding_min_overlap:
        return GuardrailResult(
            passed=False, layer="grounding",
            reason="Answer content does not sufficiently overlap with retrieved source chunks.",
            score=overlap,
        )
    return GuardrailResult(passed=True, layer="grounding", score=overlap)


def run_guardrail_pipeline(
    raw_query: str, chunks: list[RetrievedChunk], answer: str | None = None
) -> tuple[bool, list[GuardrailResult]]:
    """
    Runs off-topic -> safety -> sufficiency in sequence (short-circuits on
    first failure). Grounding is checked separately, after generation, by
    calling check_grounding directly once `answer` exists.
    """
    trace: list[GuardrailResult] = []

    for check in (check_offtopic, check_safety):
        result = check(raw_query)
        trace.append(result)
        if not result.passed:
            return False, trace

    result = check_sufficiency(chunks)
    trace.append(result)
    if not result.passed:
        return False, trace

    if answer is not None:
        result = check_grounding(answer, chunks)
        trace.append(result)
        if not result.passed:
            return False, trace

    return True, trace


# --- Structured eligibility verification (recommended addition, 2.5) ---

_ELIGIBILITY_SYSTEM_PROMPT = """You extract and check scheme eligibility conditions.

Given a scheme's source text and a user's stated attributes, do two things:
1. Identify the discrete eligibility conditions stated in the source text \
(age range, income ceiling, occupation, category, land holding, state, \
gender, life event, etc.) - list only conditions that are actually present \
in the text.
2. For each condition, compare it against the user's stated attributes and \
mark it as "met", "not_met", or "unclear" (if the user didn't state the \
relevant attribute).

Respond ONLY with valid JSON, no markdown fences, no preamble, in this exact shape:
{
  "conditions": [
    {"condition": "<short description>", "status": "met|not_met|unclear", "explanation": "<one sentence>"}
  ]
}
"""


def verify_eligibility_for_scheme(
    scheme_name: str, scheme_context_text: str, entities: ExtractedEntities,
    source_url: str | None = None, last_verified: str | None = None,
) -> SchemeVerdict:
    client = _get_client()

    user_attrs = entities.model_dump(exclude={"raw_query"})
    user_prompt = (
        f"SCHEME: {scheme_name}\n\nSOURCE TEXT:\n{scheme_context_text}\n\n"
        f"USER ATTRIBUTES (JSON, missing fields mean the user did not mention them):\n"
        f"{json.dumps(user_attrs)}"
    )

    completion = client.chat.completions.create(
        model=settings.groq_model,
        max_tokens=400,
        temperature=0.0,
        messages=[
            {"role": "system", "content": _ELIGIBILITY_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )
    raw = completion.choices[0].message.content.strip()
    raw = re.sub(r"^```(json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()

    try:
        parsed = json.loads(raw)
        conditions = [
            EligibilityConditionCheck(**c) for c in parsed.get("conditions", [])
        ]
    except (json.JSONDecodeError, TypeError, KeyError):
        conditions = []

    statuses = [c.status for c in conditions]
    if conditions and all(s == "met" for s in statuses):
        eligible = "eligible"
    elif any(s == "not_met" for s in statuses):
        eligible = "not_eligible"
    else:
        eligible = "possibly_eligible_needs_verification"

    return SchemeVerdict(
        scheme_name=scheme_name,
        eligible=eligible,
        conditions=conditions,
        source_url=source_url,
        last_verified=last_verified,
    )
