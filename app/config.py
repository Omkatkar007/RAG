"""
Centralized configuration. Every external-service knob lives here so the
rest of the pipeline never reads os.environ directly (blueprint 2.6).
"""
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- Groq (Stage 7: Generation) ---
    groq_api_key: str = ""
    groq_model: str = "openai/gpt-oss-120b"
    groq_max_output_tokens: int = 900

    # --- Sarvam (Stage 1: STT, voice input only) ---
    sarvam_api_key: str = ""
    sarvam_stt_model: str = "saaras:v3"
    sarvam_base_url: str = "https://api.sarvam.ai"

    # --- Qdrant (Stage 3: Dense retrieval) ---
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""
    qdrant_collection: str = "gov_schemes"

    # --- Embeddings ---
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384

    # --- Reranker (Stage 5) ---
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # --- Context builder (Stage 6) ---
    max_context_tokens: int = 1500
    min_context_chunks: int = 3
    max_context_chunks: int = 5

    # --- Fusion / retrieval (Stage 3-4) ---
    rrf_k: int = 60
    dense_top_k: int = 20
    lexical_top_k: int = 20
    rerank_top_n: int = 5

    # --- Guardrails (Stage 8) ---
    # See guardrails.py::check_sufficiency for why these are raw-score
    # thresholds, not 0-1 probabilities.
    sufficiency_min_rerank_score: float = -8.0
    sufficiency_min_fused_score: float = 0.01
    grounding_min_overlap: float = 0.30
    offtopic_min_similarity: float = 0.28

    # --- data.gov.in (secondary source) ---
    data_gov_in_api_key: str | None = None
    data_gov_in_resource_id: str | None = None

    # --- Production hardening ---
    cors_allowed_origins: list[str] = ["*"]  # comma-separated in env, e.g. "https://myapp.com,https://www.myapp.com"
    rate_limit_max_requests: int = 20
    rate_limit_window_seconds: int = 60

    log_level: str = "INFO"

    @field_validator("cors_allowed_origins", mode="before")
    @classmethod
    def _split_cors_origins(cls, v):
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",") if origin.strip()]
        return v


settings = Settings()
