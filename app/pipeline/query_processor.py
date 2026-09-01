"""
Stage 2: Query Processor.

Sits between STT and retrieval (blueprint note under 2.2): "not a separate
external dependency, just a logic step before Stage 3." Cleans the raw
question and extracts entities (age, income, occupation, location, category)
that the structured eligibility-verification layer (2.5) will later check
scheme conditions against.

Implemented as fast, dependency-free rule/regex extraction rather than an
LLM call, so this stage never costs an API request. If extraction quality
becomes a problem for messier queries, swap `extract_entities` for a Groq
call with a strict JSON-only system prompt - the return type doesn't need
to change.
"""
from __future__ import annotations

import re

from app.schemas import ExtractedEntities

_WHITESPACE_RE = re.compile(r"\s+")

_AGE_RE = re.compile(r"\b(\d{1,3})\s*(?:years?\s*old|yrs?\s*old|year[- ]old|yo)\b", re.IGNORECASE)
_LAND_RE = re.compile(r"\b(\d+(?:\.\d+)?)\s*(?:acres?|acre)\b", re.IGNORECASE)
_INCOME_RE = re.compile(
    r"\b(?:income|earn|earning|salary)\s*(?:of|is|:)?\s*(?:rs\.?|inr|₹)?\s*"
    r"(\d+(?:,\d{2,3})*(?:\.\d+)?)\s*(lakh|lakhs|crore|k|thousand)?",
    re.IGNORECASE,
)

_CATEGORY_KEYWORDS = {
    "sc": "SC", "scheduled caste": "SC",
    "st": "ST", "scheduled tribe": "ST",
    "obc": "OBC", "other backward class": "OBC",
    "ews": "EWS", "economically weaker section": "EWS",
    "general category": "General", "general": "General",
}

_OCCUPATION_KEYWORDS = [
    "farmer", "student", "laborer", "labourer", "construction worker", "domestic worker",
    "street vendor", "artisan", "weaver", "fisherman", "self-employed", "unemployed",
    "government employee", "private employee", "widow", "senior citizen", "disabled",
    "pensioner", "entrepreneur", "shopkeeper", "daily wage worker",
]

_LIFE_EVENT_KEYWORDS = {
    "marriage": ["marriage", "getting married", "wedding"],
    "education": ["education", "college", "school fees", "scholarship", "studying"],
    "childbirth": ["pregnant", "newborn", "childbirth", "delivery"],
    "housing": ["build a house", "buy a house", "own house", "pucca house"],
    "medical": ["hospital", "surgery", "medical treatment", "illness"],
    "death_of_earner": ["passed away", "death of", "husband died", "widow"],
}

_INDIAN_STATES = [
    "andhra pradesh", "arunachal pradesh", "assam", "bihar", "chhattisgarh", "goa",
    "gujarat", "haryana", "himachal pradesh", "jharkhand", "karnataka", "kerala",
    "madhya pradesh", "maharashtra", "manipur", "meghalaya", "mizoram", "nagaland",
    "odisha", "punjab", "rajasthan", "sikkim", "tamil nadu", "telangana", "tripura",
    "uttar pradesh", "uttarakhand", "west bengal", "delhi", "jammu and kashmir",
    "ladakh", "puducherry", "chandigarh",
]

_OFFTOPIC_HINT_WORDS = {
    "weather", "cricket", "movie", "recipe", "joke", "song lyrics", "stock price",
}


def clean_text(raw: str) -> str:
    text = raw.strip()
    text = _WHITESPACE_RE.sub(" ", text)
    return text


def _parse_income_to_annual_inr(value: str, unit: str | None) -> float:
    amount = float(value.replace(",", ""))
    unit = (unit or "").lower()
    if unit in ("lakh", "lakhs"):
        amount *= 100_000
    elif unit == "crore":
        amount *= 10_000_000
    elif unit in ("k", "thousand"):
        amount *= 1_000
    return amount


def extract_entities(raw_query: str) -> ExtractedEntities:
    text = clean_text(raw_query)
    lower = text.lower()

    entities = ExtractedEntities(raw_query=text)

    if m := _AGE_RE.search(lower):
        entities.age = int(m.group(1))

    if m := _LAND_RE.search(lower):
        entities.land_holding_acres = float(m.group(1))

    if m := _INCOME_RE.search(lower):
        entities.income_annual_inr = _parse_income_to_annual_inr(m.group(1), m.group(2))

    for keyword, normalized in _CATEGORY_KEYWORDS.items():
        if keyword in lower:
            entities.category = normalized
            break

    for occ in _OCCUPATION_KEYWORDS:
        if occ in lower:
            entities.occupation = occ
            break

    for state in _INDIAN_STATES:
        if state in lower:
            entities.state = state.title()
            break

    if "woman" in lower or "female" in lower or "daughter" in lower or "mother" in lower:
        entities.gender = "female"
    elif "man" in lower or "male" in lower or "son" in lower or "father" in lower:
        entities.gender = "male"

    events: list[str] = []
    for event, keywords in _LIFE_EVENT_KEYWORDS.items():
        if any(kw in lower for kw in keywords):
            events.append(event)
    entities.life_events = events

    return entities


def looks_offtopic_by_keyword(raw_query: str) -> bool:
    """Cheap pre-filter used alongside the embedding-based off-topic guardrail (2.5)."""
    lower = raw_query.lower()
    return any(word in lower for word in _OFFTOPIC_HINT_WORDS)
