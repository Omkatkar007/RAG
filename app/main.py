from __future__ import annotations

import logging
import time
from collections import defaultdict, deque
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.pipeline.orchestrator import run_pipeline
from app.schemas import QueryRequest, QueryResponse

logging.basicConfig(level=settings.log_level)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Government Scheme Eligibility RAG",
    description=(
        "Ask about Indian government welfare scheme eligibility in plain language. "
        "See blueprint 2.1 for the full 8-stage pipeline this API wraps."
    ),
    version="0.1.0",
)

# CORS: settings.cors_allowed_origins defaults to "*" for local dev. In
# production, set CORS_ALLOWED_ORIGINS in .env to your actual frontend
# domain(s), comma-separated (e.g. "https://yourapp.com") - allow_origins=["*"]
# is fine for a public read-mostly API like this, but tighten it if you add
# cookies/auth later.
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allowed_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

# --- Minimal in-memory rate limiting ---
# Per-IP sliding window. This is intentionally simple: fine for a single
# process/instance (e.g. one Render/Railway dyno). If you ever scale to
# multiple instances behind a load balancer, replace this with a shared
# store (Redis) since each instance would otherwise track its own counts.
_request_log: dict[str, deque] = defaultdict(deque)


def _client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    if request.url.path in ("/health",) or request.url.path.startswith("/static"):
        return await call_next(request)

    ip = _client_ip(request)
    now = time.monotonic()
    window = _request_log[ip]

    while window and now - window[0] > settings.rate_limit_window_seconds:
        window.popleft()

    if len(window) >= settings.rate_limit_max_requests:
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=429, content={"detail": "Too many requests. Please slow down."})

    window.append(now)
    return await call_next(request)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    if request.input_type.value == "text" and not request.text:
        raise HTTPException(status_code=400, detail="text is required when input_type is 'text'")
    if request.input_type.value == "voice" and not request.audio_base64:
        raise HTTPException(status_code=400, detail="audio_base64 is required when input_type is 'voice'")

    try:
        return run_pipeline(request)
    except RuntimeError as e:
        # e.g. BM25 index not built, or GROQ_API_KEY missing - operator error,
        # not something the end user should see a raw stack trace for.
        logger.error("Pipeline configuration error: %s", e)
        raise HTTPException(status_code=503, detail=str(e))
    except Exception:
        logger.exception("Unhandled error in pipeline")
        raise HTTPException(status_code=500, detail="Internal error while processing your query.")


# --- Serve the chat frontend from the same origin as the API ---
_static_dir = Path(__file__).resolve().parent / "static"


@app.get("/")
def serve_frontend():
    return FileResponse(_static_dir / "index.html")


app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")
