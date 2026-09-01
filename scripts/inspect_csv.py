#!/usr/bin/env python3
"""
Run this FIRST on any downloaded dataset CSV, before trying to ingest it.
Prints the real column names and one sample row so you can build an
accurate column_map for load_myscheme_csv_mirror() in
app/ingestion/loaders.py - don't guess at column names.

Usage:
    python scripts/inspect_csv.py path/to/downloaded_schemes.csv
"""
import csv
import sys


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/inspect_csv.py <path_to_csv>")
        sys.exit(1)

    path = sys.argv[1]
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        columns = reader.fieldnames or []
        print(f"Found {len(columns)} columns:\n")
        for col in columns:
            print(f"  - {col}")

        print("\n--- Sample row ---")
        first_row = next(reader, None)
        if first_row:
            for col in columns:
                value = str(first_row.get(col, ""))
                preview = value[:150] + ("..." if len(value) > 150 else "")
                print(f"  {col}: {preview}")
        else:
            print("  (file appears to have no data rows)")

        # Count total rows without loading everything into memory twice.
        row_count = 1 if first_row else 0
        for _ in reader:
            row_count += 1
        print(f"\nTotal data rows: {row_count}")

    print(
        "\nNext step: edit DEFAULT_COLUMN_MAP in app/ingestion/loaders.py "
        "(or pass column_map explicitly) so each key points at the correct "
        "column name from the list above, then run "
        "scripts/ingest_myscheme_csv.py."
    )


if __name__ == "__main__":
    main()
