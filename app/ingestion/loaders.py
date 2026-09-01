"""
Data source loaders (blueprint 2.4).

Primary source: myscheme.gov.in - No stable, fully documented public API is
guaranteed to exist at any given time; this MUST be verified directly
before relying on `fetch_myscheme_schemes` below. If unavailable, use a
compliant scraper (check robots.txt and ToS first) or an existing
structured dataset mirror to bootstrap the corpus.

Secondary source: data.gov.in - general open-data portal (census, income,
agricultural, budget data). Useful for cross-checking a stated income
threshold or category classification against real statistics, but NOT the
primary scheme/eligibility corpus.

`load_sample_schemes` reads the small hand-seeded corpus in
data/sample_schemes.json, used to get the pipeline running end-to-end
before the real ingestion source is wired up (build order phase 1: "10-20
schemes before scaling up").
"""
from __future__ import annotations

import json
from pathlib import Path

import requests

from app.config import settings

_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
_SAMPLE_SCHEMES_PATH = _DATA_DIR / "sample_schemes.json"


def load_sample_schemes() -> list[dict]:
    """Loads the hand-seeded sample corpus (data/sample_schemes.json)."""
    with open(_SAMPLE_SCHEMES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def fetch_myscheme_schemes(category: str | None = None) -> list[dict]:
    """
    NOT YET IMPLEMENTED - deliberately.

    myScheme's real APIs are published through India's API Setu marketplace,
    but access requires an organizational registration (GST certificate,
    Certificate of Incorporation, domain email) that an individual/student
    project won't have. See `load_myscheme_csv_mirror` below for the
    practical alternative used in this project instead - a structured
    dataset mirror, exactly as the blueprint's own fallback describes (2.4).

    If you ever do get API Setu access (e.g. through a registered
    organization), implement the real call here and the ingestion pipeline
    (app/ingestion/ingest.py) won't need to change - it just needs a list
    of scheme dicts in the same shape as data/sample_schemes.json.
    """
    raise NotImplementedError(
        "Direct myscheme.gov.in API access requires API Setu organizational "
        "registration (see this function's docstring). Use "
        "load_myscheme_csv_mirror() instead for now."
    )


def load_myscheme_csv_mirror(csv_path: str | Path, column_map: dict[str, str] | None = None) -> list[dict]:
    """
    Loads scheme data from a structured dataset mirror (CSV) of
    myscheme.gov.in - the blueprint's own suggested fallback when a direct
    API isn't accessible (2.4): "an existing structured dataset mirror ...
    to bootstrap the corpus."

    IMPORTANT: every public CSV mirror of this data uses different column
    names. This function does NOT guess your file's columns - run
    `python scripts/inspect_csv.py <path>` first to see the real column
    names and a sample row, then pass `column_map` (or edit
    DEFAULT_COLUMN_MAP below) so it points at the right columns.

    column_map keys (our schema) -> values (your CSV's actual column names):
        scheme_name, ministry, description, eligibility_text,
        benefit_amount, category_state_applicability, source_url

    `eligibility_text` is expected to be a single free-text field (common
    in scraped datasets); it gets split into individual clauses by
    sentence, matching the metadata-aware chunking strategy's expectation
    of a list of clauses (2.3).
    """
    import csv as csv_module

    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} not found. Download the dataset (e.g. from Kaggle) "
            f"and pass its path here, or run scripts/inspect_csv.py first."
        )

    col_map = {**DEFAULT_COLUMN_MAP, **(column_map or {})}

    schemes: list[dict] = []
    with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv_module.DictReader(f)
        missing_cols = [v for v in col_map.values() if v and v not in (reader.fieldnames or [])]
        if missing_cols:
            raise KeyError(
                f"Column(s) {missing_cols} not found in {csv_path}. "
                f"Actual columns are: {reader.fieldnames}. "
                f"Run scripts/inspect_csv.py {csv_path} to inspect, then pass "
                f"the correct column_map."
            )

        for row in reader:
            name = row.get(col_map["scheme_name"], "").strip()
            if not name:
                continue

            eligibility_text = row.get(col_map.get("eligibility_text", ""), "") or ""
            eligibility_clauses = [
                clause.strip() for clause in _split_into_clauses(eligibility_text) if clause.strip()
            ]

            schemes.append({
                "scheme_name": name,
                "ministry": (row.get(col_map.get("ministry", ""), "") or "").strip() or None,
                "description": (row.get(col_map.get("description", ""), "") or "").strip() or None,
                "eligibility_clauses": eligibility_clauses,
                "benefit_amount": (row.get(col_map.get("benefit_amount", ""), "") or "").strip() or None,
                "category_state_applicability": (
                    row.get(col_map.get("category_state_applicability", ""), "") or ""
                ).strip() or None,
                "source_url": (row.get(col_map.get("source_url", ""), "") or "").strip() or None,
                "last_verified": None,  # dataset mirror - not live-verified; see README caveat
            })

    return schemes


# Best-effort default guesses for common scraped-scheme-dataset column
# naming conventions. Verify against your actual file with
# scripts/inspect_csv.py and override via the column_map argument if these
# don't match - do not assume these are correct without checking.
DEFAULT_COLUMN_MAP = {
    "scheme_name": "scheme_name",
    "ministry": "ministry",
    "description": "description",
    "eligibility_text": "eligibility",
    "benefit_amount": "benefits",
    "category_state_applicability": "category",
    "source_url": "url",
}


def _split_into_clauses(text: str) -> list[str]:
    """Splits a free-text eligibility blob into individual clauses."""
    import re
    if not text:
        return []
    # Split on sentence boundaries and common list separators (bullets, semicolons).
    parts = re.split(r"(?<=[.!?])\s+|\n+|;\s*|•\s*", text)
    return [p.strip() for p in parts if p.strip()]


def fetch_data_gov_in_dataset(resource_id: str, filters: dict | None = None, limit: int = 100) -> list[dict]:
    """
    data.gov.in exposes a documented, registration-key-based REST API
    (https://data.gov.in/help/api). Used here only as a *secondary* source
    for cross-checking things like income thresholds or category stats
    against real statistics - not for the core scheme corpus (2.4).
    """
    if not settings.data_gov_in_api_key:
        raise RuntimeError("DATA_GOV_IN_API_KEY is not configured.")

    params = {
        "api-key": settings.data_gov_in_api_key,
        "format": "json",
        "limit": limit,
    }
    if filters:
        for key, value in filters.items():
            params[f"filters[{key}]"] = value

    resp = requests.get(f"https://api.data.gov.in/resource/{resource_id}", params=params, timeout=30)
    resp.raise_for_status()
    return resp.json().get("records", [])
