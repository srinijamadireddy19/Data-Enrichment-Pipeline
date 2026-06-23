"""
exporter.py — CSV (and optional JSON) export for the Data Enrichment Pipeline

Reads enriched records from the database and writes them to output files.

Usage:
    from exporter import export_csv, export_json

    export_csv(records, "output/results.csv")
    export_json(records, "output/results.json")

    # Or export directly from the database:
    from database import Database
    with Database("pipeline.db") as db:
        export_csv_from_db(db, "output/results.csv")
"""

from __future__ import annotations

import csv
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from models import EnrichedRecord

log = logging.getLogger("exporter")

# ── Column order for CSV ──────────────────────────────────────────────────────

CSV_COLUMNS = [
    "company_name",
    "industry",
    "description",
    "country",
    "employee_size",
    "website",
]

CSV_COLUMNS_FULL = CSV_COLUMNS + [
    "scrape_method",
    "completeness",
    "ai_enriched",
]


# ── CSV export ────────────────────────────────────────────────────────────────

def export_csv(
    records: list[EnrichedRecord],
    path: str | Path,
    include_metadata: bool = False,
) -> Path:
    """
    Export a list of EnrichedRecords to CSV.

    Args:
        records:          Records to export.
        path:             Output file path (.csv).
        include_metadata: If True, adds scrape_method / completeness / ai_enriched columns.

    Returns:
        Path to the written file.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    columns = CSV_COLUMNS_FULL if include_metadata else CSV_COLUMNS

    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for r in records:
            row = r.to_csv_row()
            if include_metadata:
                row["scrape_method"] = r.scrape_method
                row["completeness"]  = f"{r.completeness:.1f}"
                row["ai_enriched"]   = "yes" if r.ai_enriched else "no"
            writer.writerow(row)

    log.info("Exported %d record(s) → %s", len(records), out)
    return out


def export_csv_from_db(
    db,            # Database instance — avoid circular import
    path: str | Path,
    include_metadata: bool = False,
    country: Optional[str] = None,
    industry: Optional[str] = None,
) -> Path:
    """
    Export directly from the database with optional filters.

    Args:
        db:               Open Database instance.
        path:             Output file path.
        include_metadata: Include pipeline metadata columns.
        country:          Filter by country (exact match).
        industry:         Filter by industry (exact match).

    Returns:
        Path to the written file.
    """
    if country:
        rows = db.get_by_country(country)
    elif industry:
        rows = db.get_by_industry(industry)
    else:
        rows = db.get_all()

    # Convert raw dicts → EnrichedRecord so we reuse the same export logic
    records = _rows_to_records(rows)
    return export_csv(records, path, include_metadata=include_metadata)


# ── JSON export ───────────────────────────────────────────────────────────────

def export_json(
    records: list[EnrichedRecord],
    path: str | Path,
    include_metadata: bool = False,
) -> Path:
    """
    Export records to a JSON array.

    Args:
        records:          Records to export.
        path:             Output file path (.json).
        include_metadata: If True, adds pipeline metadata fields.

    Returns:
        Path to the written file.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    data = []
    for r in records:
        row = r.to_csv_row()   # 6 user-facing fields
        if include_metadata:
            row["scrape_method"] = r.scrape_method
            row["completeness"]  = r.completeness
            row["ai_enriched"]   = r.ai_enriched
        data.append(row)

    with out.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    log.info("Exported %d record(s) → %s", len(records), out)
    return out


def export_json_from_db(
    db,
    path: str | Path,
    include_metadata: bool = False,
) -> Path:
    """Export all records from DB to JSON."""
    rows    = db.get_all()
    records = _rows_to_records(rows)
    return export_json(records, path, include_metadata=include_metadata)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _rows_to_records(rows: list[dict]) -> list[EnrichedRecord]:
    """Convert raw DB dicts → EnrichedRecord objects."""
    records = []
    for row in rows:
        try:
            records.append(EnrichedRecord(
                company_name  = row["company_name"],
                industry      = row["industry"],
                description   = row["description"],
                country       = row["country"],
                employee_size = row["employee_size"],
                website       = row["website"],
                scrape_method = row.get("scrape_method", "unknown"),
                completeness  = row.get("completeness", 0.0),
                ai_enriched   = bool(row.get("ai_enriched", 0)),
                created_at    = row.get("created_at", 0.0),
            ))
        except Exception as exc:
            log.warning("Skipping malformed DB row for %s: %s", row.get("website"), exc)
    return records


def timestamped_path(directory: str | Path, prefix: str = "results", ext: str = "csv") -> Path:
    """
    Generate an output path with a timestamp suffix.
    e.g. output/results_20240615_143022.csv
    """
    ts  = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out = Path(directory) / f"{prefix}_{ts}.{ext}"
    out.parent.mkdir(parents=True, exist_ok=True)
    return out