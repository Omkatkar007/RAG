from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class InputType(str, Enum):
    TEXT = "text"
    VOICE = "voice"


class QueryRequest(BaseModel):
    input_type: InputType = InputType.TEXT
    text: Optional[str] = Field(default=None, description="Required if input_type == text")
    audio_base64: Optional[str] = Field(default=None, description="Required if input_type == voice")
    audio_language_hint: Optional[str] = Field(default=None, description="e.g. 'hi-IN', optional hint for STT")
    user_id: Optional[str] = None


class ExtractedEntities(BaseModel):
    """Slots pulled out of the free-text query by the query processor (2.2)."""
    age: Optional[int] = None
    occupation: Optional[str] = None
    income_annual_inr: Optional[float] = None
    category: Optional[str] = None  # SC/ST/OBC/General/EWS etc.
    state: Optional[str] = None
    gender: Optional[str] = None
    land_holding_acres: Optional[float] = None
    life_events: list[str] = Field(default_factory=list)  # e.g. "daughter's marriage", "child's education"
    raw_query: str = ""


class RetrievedChunk(BaseModel):
    chunk_id: str
    scheme_name: str
    ministry: Optional[str] = None
    text: str
    source_url: Optional[str] = None
    last_verified: Optional[str] = None
    dense_score: Optional[float] = None
    lexical_score: Optional[float] = None
    fused_score: Optional[float] = None
    rerank_score: Optional[float] = None


class GuardrailResult(BaseModel):
    passed: bool
    layer: str
    reason: Optional[str] = None
    score: Optional[float] = None


class EligibilityConditionCheck(BaseModel):
    condition: str
    status: str  # "met" | "not_met" | "unclear" | "not_applicable"
    explanation: str


class SchemeVerdict(BaseModel):
    scheme_name: str
    eligible: str  # "eligible" | "not_eligible" | "possibly_eligible_needs_verification"
    conditions: list[EligibilityConditionCheck] = Field(default_factory=list)
    source_url: Optional[str] = None
    last_verified: Optional[str] = None


class QueryResponse(BaseModel):
    answer: str
    verdicts: list[SchemeVerdict] = Field(default_factory=list)
    citations: list[RetrievedChunk] = Field(default_factory=list)
    guardrail_trace: list[GuardrailResult] = Field(default_factory=list)
    disclaimer: str = (
        "This is an information aid, not a legal determination of eligibility. "
        "Please verify against the official scheme page or your nearest government "
        "office / Common Service Centre before acting."
    )
    blocked: bool = False
    block_reason: Optional[str] = None
