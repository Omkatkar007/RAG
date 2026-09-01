#!/usr/bin/env python3
"""
Ingests a downloaded structured-dataset-mirror CSV of myscheme.gov.in data
(blueprint 2.4 fallback) into Qdrant + BM25, replacing/extending the
8-scheme sample corpus.

ALWAYS run scripts/inspect_csv.py on your file first and check the column
names actually match DEFAULT_COLUMN_MAP in app/ingestion/loaders.py before
running this for real - use --dry-run to preview parsed output first.

Usage:
    python scripts/ingest_myscheme_csv.py path/to/schemes.csv --dry-run
    python scripts/ingest_myscheme_csv.py path/to/schemes.csv
    python scripts/ingest_myscheme_csv.py path/to/schemes.csv --limit 50
"""
import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.ingestion.ingest import ingest_schemes  # noqa: E402
from app.ingestion.loaders import load_myscheme_csv_mirror  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", help="Path to the downloaded schemes CSV")
    parser.add_argument("--dry-run", action="store_true", help="Parse and print without writing to Qdrant/BM25")
    parser.add_argument("--limit", type=int, default=None, help="Only ingest the first N schemes (for testing)")
    args = parser.parse_args()

    schemes = load_myscheme_csv_mirror(args.csv_path)
    print(f"Parsed {len(schemes)} schemes from {args.csv_path}")

    if not schemes:
        print("No schemes parsed - check your column mapping (scripts/inspect_csv.py).")
        return

    print("\n--- First parsed scheme (sanity check before ingesting) ---")
    print(json.dumps(schemes[0], indent=2, ensure_ascii=False)[:1500])

    if args.limit:
        schemes = schemes[:args.limit]
        print(f"\n--limit {args.limit} applied - ingesting first {len(schemes)} schemes only.")

    if args.dry_run:
        print("\n--dry-run: nothing was written to Qdrant/BM25. Remove --dry-run to actually ingest.")
        return

    confirm = input(f"\nProceed to embed and upsert {len(schemes)} schemes into Qdrant? [y/N] ")
    if confirm.strip().lower() != "y":
        print("Aborted.")
        return

    count = ingest_schemes(schemes)
    print(f"\nIngested {count} chunks from {len(schemes)} schemes.")


if __name__ == "__main__":
    main()
