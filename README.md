<div align="center">

# Scheme RAG 🏛️

*An intelligent Retrieval-Augmented Generation (RAG) system for verifying eligibility against thousands of Indian Government Welfare Schemes.*

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.36+-FF4B4B.svg)](https://streamlit.io/)
[![Groq](https://img.shields.io/badge/Groq-Llama_3-f55036.svg)](https://groq.com/)
[![Qdrant](https://img.shields.io/badge/Vector_DB-Qdrant-red.svg)](https://qdrant.tech/)

</div>

---

## 📖 Overview

**Scheme RAG** is a highly structured RAG application designed to parse, ingest, and query thousands of government welfare schemes. By inputting demographic details (such as occupation, income, state, and category), the AI assistant accurately cross-references eligibility conditions to recommend matching schemes.

This project was built to address the unstructured nature of scheme documents by extracting the core `eligibility_clauses` and mapping them into a high-performance dense and lexical search pipeline.

### Key Features
- **Intelligent RAG Pipeline**: Uses a hybrid approach combining Dense Vector Search (Qdrant + `sentence-transformers/all-MiniLM-L6-v2`) and Lexical Search (BM25) for hyper-accurate retrieval.
- **Automated PDF Ingestion**: A custom parsing engine that extracts raw text from thousands of PDFs and structures them into smart semantic chunks.
- **Groq-Powered LLM**: Utilizes Llama-3 120b on Groq for ultra-fast, intelligent verification of user criteria against scheme rules.
- **Streamlit Interface**: A clean, accessible chat interface where users can ask questions in natural language.

---

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/Omkatkar007/RAG.git
cd RAG
```

### 2. Install Dependencies
Ensure you have Python 3.10+ installed, then run:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Environment Variables
Create a `.env` file in the root directory and add your API keys:
```env
# LLM Generation
GROQ_API_KEY="your-groq-api-key"

# Vector Database (Qdrant)
QDRANT_URL="your-qdrant-cluster-url"
QDRANT_API_KEY="your-qdrant-api-key"
```

### 4. Run the Application
Launch the Streamlit frontend locally:
```bash
streamlit run streamlit_app.py
```

---

## 🛠️ Data Ingestion Pipeline

To ingest your own raw PDFs into the RAG database, place your `.pdf` files in the `text_data/` directory and run the automated ingestion script:

```bash
python scripts/ingest_pdfs.py --limit 1000
```
*Note: The script bypasses LLM overhead and uses fixed-window chunking (`app/ingestion/chunking.py`) to process large volumes of documents entirely locally at maximum speed.*

---

## 🏗️ System Architecture

1. **Frontend**: Streamlit Chat Interface (`streamlit_app.py`).
2. **Retrieval Module**: 
   - **Dense Retriever**: Qdrant Vector DB with Sentence-Transformers embeddings.
   - **Lexical Retriever**: BM25 In-Memory Index.
3. **Orchestrator (`app/pipeline/orchestrator.py`)**: Merges retrieved chunks, scores them, and filters out low-relevance documents.
4. **Generation Module**: Passes the top retrieved documents and user queries to the Groq API (Llama 3) for the final eligibility verdict and response formatting.

---
<div align="center">
<i>Developed and Maintained for the Scheme RAG Initiative.</i>
</div>
