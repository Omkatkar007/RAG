# Government Scheme Eligibility RAG

Working implementation of the 8-stage pipeline from the project blueprint:

```
User (Voice/Text) -> STT -> Query Processor -> Hybrid Retrieval (Dense + Lexical)
  -> RRF Fusion -> Cross-Encoder Reranking -> Context Builder -> LLM Generation
  -> Guardrails (off-topic / safety / sufficiency / grounding) -> Answer + Citations
```

Plus the recommended structured eligibility-verification step (checklist-style
condition-by-condition checking, run after grounding passes).

## 1. Setup

```bash
cd scheme-rag
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Fill in `.env`:
- `GROQ_API_KEY` — required for Stage 7 (generation) and eligibility verification.
  **Verify `GROQ_MODEL` is still live** in Groq's model list before running — Groq
  regularly deprecates older Llama versions.
- `QDRANT_URL` / `QDRANT_API_KEY` — a local Docker Qdrant (`docker run -p 6333:6333
  qdrant/qdrant`) works fine for development, or use Qdrant Cloud's free tier.
- `SARVAM_API_KEY` — only needed if you're testing voice input.

## 2. Ingest the sample corpus

```bash
python scripts/run_ingestion.py
```

This chunks `data/sample_schemes.json` (8 real central schemes: PM-KISAN,
Ayushman Bharat PM-JAY, Sukanya Samriddhi Yojana, PMAY, Atal Pension Yojana,
PMMY, NSP Post-Matric Scholarship, PMFBY) using the metadata-aware strategy,
embeds them with a local MiniLM model, upserts to Qdrant, and builds the BM25
index at `data/bm25_index.pkl`.

**The sample corpus is a seed set for testing the pipeline, not a production
data source.** Several fields (income ceilings, PMAY slabs, NSP amounts) are
flagged in the JSON as needing re-verification against the live source before
being used for real eligibility decisions — this mirrors the blueprint's own
"stale data" risk (2.8). Real ingestion from myscheme.gov.in still needs to be
implemented in `app/ingestion/loaders.py::fetch_myscheme_schemes` after you've
verified current API/scraper compliance (see that function's docstring —
deliberately left unimplemented until you've done that check).

## 3. Run the API

```bash
uvicorn app.main:app --reload
```

Then:

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
        "input_type": "text",
        "text": "I am a farmer, I have 2 acres of land, my daughter is getting married next year - what help can I get from the government?"
      }'
```

Or bypass HTTP for a quick manual check:

```bash
python scripts/test_pipeline.py "your question here"
```

## Project layout

```
app/
  config.py              # all settings, one place
  schemas.py              # request/response models
  main.py                  # FastAPI app (/query, /health)
  pipeline/
    stt.py                 # Stage 1 - Sarvam speech-to-text (voice only)
    query_processor.py     # Stage 2 - cleaning + entity extraction
    embeddings.py            # local MiniLM singleton
    dense_retrieval.py       # Stage 3a - Qdrant
    lexical_retrieval.py     # Stage 3b - in-memory BM25
    fusion.py                 # Stage 4 - Reciprocal Rank Fusion
    reranker.py                # Stage 5 - local cross-encoder
    context_builder.py          # Stage 6 - token-budgeted context
    generation.py                 # Stage 7 - Groq LLM call
    guardrails.py                  # Stage 8 - 4 layers + eligibility checklist
    orchestrator.py                 # wires stages 1-8 together
  ingestion/
    chunking.py             # all 4 chunking strategies from 2.3
    loaders.py                # myscheme.gov.in (stub) / data.gov.in / sample corpus
    ingest.py                  # load -> chunk -> embed -> upsert + BM25 build
data/
  sample_schemes.json      # 8 real, verified-as-of-writing central schemes
scripts/
  run_ingestion.py          # CLI: populate Qdrant + BM25 from sample corpus
  test_pipeline.py            # CLI: run one query through the full pipeline
```

## What's implemented vs. what's a deliberate stub

| Component | Status |
|---|---|
| All 8 pipeline stages | Implemented, wired end-to-end, **verified working against live Groq/Qdrant/Sarvam** |
| 4 chunking strategies | Implemented |
| 4 guardrail layers + eligibility checklist | Implemented, verified passing correctly end-to-end |
| Sample corpus (8 schemes) | Implemented, web-verified at write time |
| Qdrant / BM25 / RRF / reranker / MiniLM | Implemented, all run against real libraries |
| Groq generation + eligibility check | Implemented, verified working (`openai/gpt-oss-120b`) |
| Sarvam STT | Implemented — needs your API key to run |
| Chat frontend (`app/static/index.html`) | Implemented — text + voice, served at `/` |
| Rate limiting | Implemented — simple in-memory per-IP sliding window |
| Dockerfile / deploy config | Implemented — see Deployment section below |
| `myscheme.gov.in` ingestion | **Deliberately stubbed** — raises `NotImplementedError`
until you've verified current API/ToS status (blueprint 2.4 explicitly flags this as unverified) |
| Scheduled re-ingestion / "last verified" freshness display | Not yet built — `ingest_schemes` can be re-run on a cron/scheduler as-is, but there's no scheduler wiring yet |

## Using the chat frontend

Once the API is running (`uvicorn app.main:app --reload`), open **http://localhost:8000/** in a
browser — that's the chat UI, served by the same FastAPI app (no separate frontend server needed).
It supports both typed and voice (mic button) input, and renders the eligibility verdicts and
source citations under each answer.

## Getting real scheme data (myScheme dataset mirror)

Direct myscheme.gov.in API access goes through India's API Setu marketplace, which requires
organizational documents (GST certificate, Certificate of Incorporation, domain email) — not
practical for an individual/student project. Instead, this project supports ingesting a
**structured dataset mirror** of myScheme data (e.g. a public Kaggle dataset), exactly as the
blueprint's own fallback describes (2.4).

```bash
# 1. Download a myScheme dataset mirror (e.g. from Kaggle) to your machine.

# 2. Inspect its actual columns first - don't assume they match any example.
python scripts/inspect_csv.py path/to/downloaded_schemes.csv

# 3. If the printed column names don't match DEFAULT_COLUMN_MAP in
#    app/ingestion/loaders.py, edit that dict (or pass column_map explicitly)
#    so each key points at your file's real column name.

# 4. Preview parsed output before writing anything (no Qdrant/BM25 writes yet).
python scripts/ingest_myscheme_csv.py path/to/downloaded_schemes.csv --dry-run

# 5. Once the preview looks right, ingest for real (asks for confirmation).
python scripts/ingest_myscheme_csv.py path/to/downloaded_schemes.csv

# Optional: test with a small subset first.
python scripts/ingest_myscheme_csv.py path/to/downloaded_schemes.csv --limit 50
```

This **adds to** whatever's already in Qdrant/BM25 — it doesn't clear the existing 8-scheme
sample corpus first. If you want a clean slate, delete the Qdrant collection (via the Qdrant
Cloud dashboard) and `data/bm25_index.pkl` before running ingestion.

**Caveats to keep in mind:**
- Dataset mirrors are a snapshot — check the dataset's own "last updated" date. `last_verified`
  is intentionally left `null` for CSV-ingested schemes (unlike the hand-checked sample corpus)
  since it wasn't independently verified against the live site.
- Check the dataset's license before using it beyond a personal/portfolio project.
- If you ever get real API Setu access later, implement `fetch_myscheme_schemes()` in
  `app/ingestion/loaders.py` (currently a documented stub) — the rest of the ingestion pipeline
  doesn't need to change, since it just expects a list of scheme dicts in the same shape.

## Deployment

The project ships with a `Dockerfile` that:
- Installs all dependencies
- **Pre-downloads the embedding and reranker models at build time** (not on first request —
  avoids slow/failing cold starts on platforms with request timeouts)
- Runs ingestion automatically on first container start if no BM25 index exists yet
- Serves the API + frontend together on `$PORT` (defaults to 8000)

```bash
docker build -t scheme-rag .
docker run -p 8000:8000 --env-file .env scheme-rag
```

**Deploying to a platform** (Render, Railway, Fly.io, a VM — any Docker-friendly host works):
1. Push this repo to GitHub.
2. Point your platform at it and let it build from the `Dockerfile`.
3. Set the same environment variables from `.env` in the platform's dashboard —
   `render.yaml` is included as a ready-made Render blueprint if you use Render specifically,
   but it's not required for other platforms.
4. Set `CORS_ALLOWED_ORIGINS` to your real frontend domain(s) once you have one, instead of `*`.
5. Point `QDRANT_URL` at Qdrant Cloud (already cloud-hosted, no extra step needed there).

**What deployment does *not* solve by itself:**
- Real scheme data — you're still on the 8-scheme sample corpus until `myscheme.gov.in`
  ingestion is implemented (see table above).
- Multi-instance scaling — the rate limiter is in-memory per process; if you ever run more
  than one instance behind a load balancer, swap it for a shared store (Redis) so limits are
  enforced consistently across instances.
- Secrets management — don't commit `.env`; use your platform's secret/env var storage.

## Suggested next steps (per blueprint 2.7)

1. Run ingestion + a handful of manual test queries against the sample corpus,
   confirm retrieval quality feels right before touching `myscheme.gov.in`.
2. Verify `GROQ_MODEL` against Groq's current docs.
3. Implement `fetch_myscheme_schemes` after doing the required API/ToS check.
4. Add the scheduled re-ingestion job (weekly, per 2.4) once real ingestion works.
5. Wire up voice input via Sarvam last, once the text pipeline is stable (2.7 step 7).
