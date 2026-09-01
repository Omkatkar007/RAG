"""
Stage 7: Generation. Groq-hosted Llama model, <=256 output tokens
(blueprint 2.2 row 8).

IMPORTANT: verify GROQ_MODEL is still live in Groq's model list before
deploying - Groq regularly deprecates older Llama versions in favor of
newer ones (3.3, 4-series, etc.), as the blueprint itself flags.

The system prompt enforces "only answer from provided context" (build
order step 4) so the grounding guardrail in Stage 8 has something
meaningful to check against.
"""
from __future__ import annotations

from groq import Groq
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import settings

_SYSTEM_PROMPT = """You are an assistant that explains Indian government welfare \
scheme eligibility to ordinary citizens in plain language.

STRICT RULES:
1. Only use information present in the CONTEXT block below. Never use outside \
knowledge about schemes, even if you believe you know it.
2. If the context does not contain enough information to answer confidently, say \
so explicitly instead of guessing.
3. For every claim about eligibility or benefits, mention which scheme it comes \
from (as named in the context's [Source: ...] tags).
4. Do not state a firm "you are eligible" or "you are not eligible" verdict \
yourself - that judgment is made separately by a structured eligibility \
checker. Instead, summarize what the context says about the relevant \
conditions (age limits, income ceilings, occupation, land holding, category, \
state, etc.) so the person understands what to check next.
5. Keep the answer concise, plain-language, and free of bureaucratic jargon.
6. Never invent scheme names, benefit amounts, or eligibility conditions that \
are not in the context.
"""

_client: Groq | None = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        if not settings.groq_api_key:
            raise RuntimeError("GROQ_API_KEY is not configured.")
        _client = Groq(api_key=settings.groq_api_key)
    return _client


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
def generate_answer(user_query: str, context: str) -> str:
    client = _get_client()

    user_prompt = f"CONTEXT:\n{context}\n\nUSER QUESTION:\n{user_query}"

    completion = client.chat.completions.create(
        model=settings.groq_model,
        max_tokens=settings.groq_max_output_tokens,
        temperature=0.2,
        reasoning_effort="low",  # gpt-oss models spend tokens "thinking" before
                                  # answering; low keeps that budget small so
                                  # max_tokens isn't consumed before any answer
                                  # text is produced.
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )
    content = completion.choices[0].message.content
    return (content or "").strip()
