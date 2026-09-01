#!/usr/bin/env python3
"""
Quick manual smoke test, bypassing FastAPI. Requires ingestion to have run
first (`python scripts/run_ingestion.py`) and GROQ_API_KEY to be set.

Usage:
    python scripts/test_pipeline.py "I am a farmer, I have 2 acres of land, \
my daughter is getting married next year - what help can I get?"
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.pipeline.orchestrator import run_pipeline  # noqa: E402
from app.schemas import InputType, QueryRequest  # noqa: E402

if __name__ == "__main__":
    query_text = sys.argv[1] if len(sys.argv) > 1 else (
        "I am a farmer, I have 2 acres of land, my daughter is getting "
        "married next year - what help can I get from the government?"
    )
    request = QueryRequest(input_type=InputType.TEXT, text=query_text)
    response = run_pipeline(request)
    print(json.dumps(response.model_dump(), indent=2, ensure_ascii=False))
