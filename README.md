<p align="center">
  <h1 align="center">🏛️ Scheme RAG</h1>
  <p align="center">
    <strong>AI-powered eligibility checker for Indian government welfare schemes</strong>
  </p>
  <p align="center">
    <a href="https://schema-rag.streamlit.app/">
      <img src="https://img.shields.io/badge/Live_Demo-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit App">
    </a>
    <img src="https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
    <img src="https://img.shields.io/badge/RAG-8_Stage_Pipeline-blueviolet?style=for-the-badge" alt="RAG Pipeline">
    
  </p>
</p>

---

Scheme RAG is a full-stack **Retrieval-Augmented Generation** system that lets Indian citizens ask about government welfare scheme eligibility in plain, natural language — including Hindi via voice input. It retrieves real scheme data from a curated corpus, cross-checks eligibility conditions against the user's stated profile, and provides grounded, cited answers with per-scheme eligibility verdicts.

> **🔗 Try it live:** [https://schema-rag.streamlit.app](https://schema-rag.streamlit.app/)

---

## ✨ Key Features

| Feature | Description |
|---|---|
| 🗣️ **Voice Input (Hindi/Regional)** | Sarvam AI's `saaras:v3` STT converts spoken queries to text (In Production) |
| 🔍 **Hybrid Retrieval** | Dense (Qdrant + MiniLM embeddings) + Lexical (BM25 Okapi) search |
| 🔀 **Reciprocal Rank Fusion** | Merges dense & lexical results by rank, not raw score |
| 🎯 **Cross-Encoder Reranking** | `ms-marco-MiniLM-L-6-v2` reranker runs locally — zero API cost |
| 🤖 **LLM Generation** | Groq-hosted model with strict context-only system prompt |
| 🛡️ **4-Layer Guardrails** | Off-topic filter → Safety filter → Sufficiency check → Grounding check |
| ✅ **Structured Eligibility Verdicts** | Per-scheme condition-by-condition verification (✅ met / ❌ not met / ⚠️ unclear) |
| 📊 **Entity Extraction** | Regex-based slot extraction (age, income, occupation, state, category, etc.) — no LLM cost |
| 🐳 **Docker Ready** | Multi-stage build with pre-downloaded models for fast cold starts |

---

## 🏗️ Architecture

The system follows an **8-stage pipeline**, orchestrated end-to-end by a single function call:

```
User Query (Text or Voice)
  │
  ├─ [Stage 1] Speech-to-Text (Sarvam AI — voice input only)
  │
  ├─ [Stage 2] Query Processor (regex entity extraction)
  │
  ├─ [Stage 3] Hybrid Retrieval
  │     ├── Dense Search (Qdrant ANN, cosine similarity)
  │     └── Lexical Search (BM25 Okapi, in-memory)
  │
  ├─ [Stage 4] Fusion (Reciprocal Rank Fusion, k=60)
  │
  ├─ [Stage 5] Reranking (Cross-Encoder, local inference)
  │
  ├─ [Stage 6] Context Builder (token-budgeted, 3–5 chunks, ≤1500 tokens)
  │
  ├─ [Stage 7] LLM Generation (Groq API, context-only prompt)
  │
  ├─ [Stage 8] Guardrails
  │     ├── Off-topic filter (keyword + embedding similarity)
  │     ├── Safety filter (fraud/injection pattern matching)
  │     ├── Sufficiency check (rerank score threshold)
  │     └── Grounding check (answer ↔ source token overlap)
  │
  └─ Structured Eligibility Verification (per-scheme condition checklist)
        │
        ▼
  Final Answer + Eligibility Verdicts + Citations + Disclaimer
```

### Guardrails in Detail

Guardrails run at **two points** to avoid wasting compute:

1. **Pre-retrieval** (cheap): Off-topic and safety checks reject bad queries before any retrieval or generation happens.
2. **Post-generation**: Sufficiency and grounding checks ensure the answer is backed by real evidence.

---

## 📁 Project Structure

```
scheme-rag/
├── streamlit_app.py              # Streamlit frontend (deployed at streamlit.app)
├── app/
│   ├── main.py                   # FastAPI backend with CORS, rate limiting, /query endpoint
│   ├── config.py                 # Centralized Pydantic settings (reads .env)
│   ├── schemas.py                # Request/response models, eligibility verdict schemas
│   ├── static/
│   │   └── index.html            # Standalone chat UI served by FastAPI
│   ├── pipeline/
│   │   ├── orchestrator.py       # Ties stages 1–8 together
│   │   ├── stt.py                # Stage 1: Sarvam speech-to-text client
│   │   ├── query_processor.py    # Stage 2: Regex entity extraction
│   │   ├── dense_retrieval.py    # Stage 3a: Qdrant ANN search
│   │   ├── lexical_retrieval.py  # Stage 3b: BM25 Okapi (in-memory, pickled)
│   │   ├── fusion.py             # Stage 4: Reciprocal Rank Fusion
│   │   ├── reranker.py           # Stage 5: Cross-encoder reranking
│   │   ├── context_builder.py    # Stage 6: Token-budgeted context assembly
│   │   ├── generation.py         # Stage 7: Groq LLM generation
│   │   ├── guardrails.py         # Stage 8: 4-layer guardrails + eligibility verification
│   │   └── embeddings.py         # Shared MiniLM embedding model (singleton)
│   └── ingestion/
│       ├── loaders.py            # Data source loaders (sample JSON, CSV mirror, data.gov.in)
│       ├── chunking.py           # 4 chunking strategies (metadata-aware, semantic, sentence-window, fixed)
│       └── ingest.py             # Ingestion pipeline: load → chunk → embed → upsert
├── scripts/
│   ├── run_ingestion.py          # One-shot: ingest sample corpus into Qdrant + BM25
│   ├── ingest_myscheme_csv.py    # Ingest a CSV dataset mirror of myscheme.gov.in
│   ├── ingest_pdfs.py            # Ingest scheme data from PDF documents
│   ├── rebuild_bm25_index.py     # Rebuild BM25 from existing Qdrant data (no re-ingestion)
│   ├── inspect_csv.py            # Inspect CSV column names before ingestion
│   └── test_pipeline.py          # Quick end-to-end pipeline smoke test
├── data/                         # Sample schemes JSON + BM25 pickle (gitignored)
├── Dockerfile                    # Multi-stage build, pre-downloads models
├── render.yaml                   # One-click Render.com deployment blueprint
├── requirements.txt              # Python dependencies
└── .env                          # Environment variables (gitignored)
```

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.11+**
- API keys for:
  - [Groq](https://console.groq.com/) — LLM generation
  - [Qdrant Cloud](https://cloud.qdrant.io/) (or a local Qdrant instance) — vector store
  - [Sarvam AI](https://www.sarvam.ai/) *(optional)* — voice input STT
  - [data.gov.in](https://data.gov.in/) *(optional)* — secondary data source

### 1. Clone & Install

```bash
git clone https://github.com/Omkatkar007/scheme-rag.git
cd scheme-rag

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure Environment

Create a `.env` file in the project root:

```env
# ---- LLM Generation ----
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=openai/gpt-oss-120b
GROQ_MAX_OUTPUT_TOKENS=900

# ---- Vector Store ----
QDRANT_URL=https://your-cluster.cloud.qdrant.io
QDRANT_API_KEY=your_qdrant_api_key
QDRANT_COLLECTION=gov_schemes

# ---- Embedding & Reranker (local, no API key needed) ----
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_DIM=384
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2

# ---- Speech-to-Text (optional, for voice input) ----
SARVAM_API_KEY=your_sarvam_api_key
SARVAM_STT_MODEL=saaras:v3

# ---- data.gov.in (optional, secondary source) ----
DATA_GOV_IN_API_KEY=your_data_gov_in_api_key

LOG_LEVEL=INFO
```

### 3. Ingest Data

Run the ingestion pipeline to populate Qdrant and build the BM25 index:

```bash
# Option A: Ingest the bundled sample corpus (8 schemes — good for testing)
python scripts/run_ingestion.py

# Option B: Ingest a CSV dataset mirror (e.g., from Kaggle)
# First inspect the CSV to check column names:
python scripts/inspect_csv.py path/to/schemes.csv
# Then ingest:
python scripts/ingest_myscheme_csv.py path/to/schemes.csv
```

### 4. Run the App

**Streamlit UI** (deployed version):
```bash
streamlit run streamlit_app.py
```

**FastAPI backend** (with built-in chat UI at `/`):
```bash
uvicorn app.main:app --reload
# API docs at http://localhost:8000/docs
# Chat UI at http://localhost:8000
```

---

## 🐳 Docker Deployment

```bash
docker build -t scheme-rag .
docker run -p 8000:8000 --env-file .env scheme-rag
```

The Dockerfile uses a multi-stage build that **pre-downloads embedding and reranker models** at build time, avoiding slow/failing cold starts on platforms with request timeouts.

On container startup, the BM25 index is automatically rebuilt from existing Qdrant data (no re-ingestion, no data duplication).

### One-Click Deploy on Render

The included `render.yaml` supports [Render Blueprint](https://render.com/docs/blueprint-spec) deployment — just connect your GitHub repo and Render handles the rest. Set your API keys as environment variables in the Render dashboard.

---

## 🔌 API Reference

### `POST /query`

Send a natural language query and receive an eligibility analysis.

**Request:**
```json
{
  "input_type": "text",
  "text": "I am a 25 year old female farmer from Maharashtra with 2 acres of land and annual income of 1.5 lakhs. What schemes can help me?"
}
```

**Response:**
```json
{
  "answer": "Based on the available information...",
  "verdicts": [
    {
      "scheme_name": "PM-KISAN",
      "eligible": "eligible",
      "conditions": [
        { "condition": "Must be a farmer", "status": "met", "explanation": "User stated they are a farmer." },
        { "condition": "Land holding ≤ 2 hectares", "status": "met", "explanation": "2 acres ≈ 0.8 hectares, within limit." }
      ]
    }
  ],
  "citations": [ ... ],
  "guardrail_trace": [ ... ],
  "disclaimer": "This is an information aid, not a legal determination of eligibility...",
  "blocked": false
}
```

### `GET /health`

Returns `{"status": "ok"}` — used for platform health checks.

---

## ⚙️ Configuration Reference

All settings are managed via environment variables (or `.env` file) and centralized in [`app/config.py`](app/config.py):

| Variable | Default | Description |
|---|---|---|
| `GROQ_API_KEY` | — | Groq API key (required) |
| `GROQ_MODEL` | `openai/gpt-oss-120b` | LLM model identifier |
| `GROQ_MAX_OUTPUT_TOKENS` | `900` | Max tokens for LLM response |
| `QDRANT_URL` | `http://localhost:6333` | Qdrant instance URL |
| `QDRANT_API_KEY` | — | Qdrant API key |
| `QDRANT_COLLECTION` | `gov_schemes` | Qdrant collection name |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Sentence embedding model |
| `EMBEDDING_DIM` | `384` | Embedding vector dimensionality |
| `RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder reranker |
| `DENSE_TOP_K` | `20` | Dense retrieval candidates |
| `LEXICAL_TOP_K` | `20` | BM25 retrieval candidates |
| `RERANK_TOP_N` | `5` | Final reranked results |
| `RRF_K` | `60` | RRF smoothing constant |
| `MAX_CONTEXT_TOKENS` | `1500` | Token budget for context window |
| `MIN_CONTEXT_CHUNKS` | `3` | Minimum chunks in context |
| `MAX_CONTEXT_CHUNKS` | `5` | Maximum chunks in context |
| `OFFTOPIC_MIN_SIMILARITY` | `0.28` | Embedding similarity threshold for off-topic filter |
| `GROUNDING_MIN_OVERLAP` | `0.30` | Token overlap threshold for grounding check |
| `SARVAM_API_KEY` | — | Sarvam API key (optional, voice only) |
| `DATA_GOV_IN_API_KEY` | — | data.gov.in API key (optional) |
| `CORS_ALLOWED_ORIGINS` | `*` | Comma-separated allowed origins |
| `RATE_LIMIT_MAX_REQUESTS` | `20` | Max requests per IP per window |
| `RATE_LIMIT_WINDOW_SECONDS` | `60` | Rate limit sliding window (seconds) |
| `LOG_LEVEL` | `INFO` | Logging level |

---

## 🧠 Tech Stack

| Layer | Technology |
|---|---|
| **Frontend** | [Streamlit](https://streamlit.io/) (deployed), HTML/JS chat UI (FastAPI-served) |
| **Backend** | [FastAPI](https://fastapi.tiangolo.com/) with rate limiting & CORS |
| **LLM** | [Groq](https://groq.com/) (cloud inference) |
| **Embeddings** | [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) (local, 384-dim) |
| **Reranker** | [cross-encoder/ms-marco-MiniLM-L-6-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2) (local) |
| **Vector Store** | [Qdrant](https://qdrant.tech/) (cloud or self-hosted) |
| **Lexical Search** | [BM25 Okapi](https://github.com/dorianbrown/rank_bm25) (in-memory, pickled) |
| **STT** | [Sarvam AI](https://www.sarvam.ai/) `saaras:v3` (Hindi/regional languages) |
| **Config** | [Pydantic Settings](https://docs.pydantic.dev/latest/concepts/pydantic_settings/) |
| **Containerization** | Docker (multi-stage build) |
| **Deployment** | Streamlit Community Cloud, Render, Docker |

---

## 📊 Data Sources

| Source | Type | Usage |
|---|---|---|
| [myscheme.gov.in](https://www.myscheme.gov.in/) | Structured dataset mirror (CSV) | **Primary** — scheme names, eligibility, benefits |
| `data/sample_schemes.json` | Hand-seeded JSON | Bootstrap / testing (8 schemes) |
| [data.gov.in](https://data.gov.in/) | REST API | **Secondary** — cross-checking income/category statistics |
| PDF documents | Uploaded files | Additional scheme data via `scripts/ingest_pdfs.py` |

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## ⚠️ Disclaimer

> This tool is an **information aid**, not a legal determination of eligibility. Please verify all results against the [official myScheme portal](https://www.myscheme.gov.in/) or your nearest government office / Common Service Centre before acting on the information provided.

---



---

<p align="center">
  Built with ❤️ for making government schemes accessible to every Indian citizen
</p>
