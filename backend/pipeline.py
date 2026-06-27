"""
pipeline.py — Orchestrator for the Data Enrichment Pipeline

Wires all six modules together into a single run() call:
  scraper → extractor → cleaner → enricher → database → exporter

Usage:
    # Programmatic
    from pipeline import run
    results = run(["https://openai.com", "https://github.com"])

    # CLI
    python pipeline.py https://openai.com https://github.com
    python pipeline.py --file urls.txt
    python pipeline.py --file urls.txt --output-dir results/ --no-csv --json
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
from pathlib import Path
from typing import Optional

from config import cfg, setup_logging
from scraper import scrape_urls
from extractor import extract_all
from cleaner import clean
from enricher import enrich_all
from database import Database
from exporter import export_csv, export_json, timestamped_path
from models import EnrichedRecord, PipelineRun

log = logging.getLogger("pipeline")


# ── Result container ──────────────────────────────────────────────────────────

class PipelineResult:
    """
    Return value of run(). Holds all outputs and a stats summary.
    """
    def __init__(
        self,
        records:  list[EnrichedRecord],
        run_meta: PipelineRun,
        csv_path:  Optional[Path] = None,
        json_path: Optional[Path] = None,
    ):
        self.records   = records
        self.run_meta  = run_meta
        self.csv_path  = csv_path
        self.json_path = json_path

    def __repr__(self) -> str:
        return (
            f"PipelineResult("
            f"{len(self.records)} records | "
            f"{self.run_meta.avg_completeness:.1f}% avg completeness"
            f")"
        )

    def print_summary(self) -> None:
        print(f"\n{'='*60}")
        print(f"  Pipeline complete")
        print(f"{'='*60}")
        print(f"  URLs processed : {self.run_meta.total_urls}")
        print(f"  Scraped        : {self.run_meta.scraped_ok}")
        print(f"  Extracted      : {self.run_meta.extracted_ok}")
        print(f"  Enriched       : {self.run_meta.enriched_ok}")
        print(f"  Failed         : {self.run_meta.failed}")
        print(f"  Avg completeness: {self.run_meta.avg_completeness:.1f}%")
        print(f"  Duration       : {self.run_meta.duration_seconds():.1f}s")
        if self.csv_path:
            print(f"  CSV            : {self.csv_path}")
        if self.json_path:
            print(f"  JSON           : {self.json_path}")
        print(f"{'='*60}\n")

        if self.records:
            print(f"  {'COMPANY':<30} {'COUNTRY':<18} {'INDUSTRY':<25} {'COMP%':>5}")
            print(f"  {'-'*30} {'-'*18} {'-'*25} {'-'*5}")
            for r in self.records:
                name     = r.company_name[:28]
                country  = r.country[:16]
                industry = r.industry[:23]
                print(f"  {name:<30} {country:<18} {industry:<25} {r.completeness:>4.0f}%")
            print()


# ── Core async pipeline ───────────────────────────────────────────────────────

async def _run_async(
    urls: list[str],
    groq_api_key: Optional[str] = None,
    db_path: Optional[Path]     = None,
    output_dir: Optional[Path]  = None,
    export_csv_flag:  Optional[bool] = None,
    export_json_flag: Optional[bool] = None,
    export_metadata:  Optional[bool] = None,
    save_to_db: bool = True,
) -> PipelineResult:
    """
    Full async pipeline execution.
    All parameters default to cfg values — override only what you need.
    """
    api_key    = groq_api_key   or cfg.groq_api_key
    db_path    = db_path        or cfg.db_path
    out_dir    = output_dir     or cfg.output_dir
    do_csv     = export_csv_flag  if export_csv_flag  is not None else cfg.export_csv
    do_json    = export_json_flag if export_json_flag is not None else cfg.export_json
    do_meta    = export_metadata  if export_metadata  is not None else cfg.export_metadata

    run = PipelineRun(total_urls=len(urls))
    deduped = list(dict.fromkeys(u.strip() for u in urls if u.strip()))

    if not deduped:
        log.warning("No valid URLs provided")
        run.finished_at = time.time()
        return PipelineResult(records=[], run_meta=run)

    log.info("Starting pipeline for %d URL(s)", len(deduped))

    # ── Stage 1: Scrape ───────────────────────────────────────────────────────
    log.info("── Stage 1/4: Scraping …")
    scrape_results = await scrape_urls(deduped, concurrency=cfg.scrape_concurrency)
    run.scraped_ok = sum(1 for r in scrape_results if r.success)
    log.info("   Scraped %d/%d", run.scraped_ok, len(deduped))

    # ── Stage 2: Extract ──────────────────────────────────────────────────────
    log.info("── Stage 2/4: Extracting …")
    extract_pairs = await extract_all(
        scrape_results, api_key=api_key, concurrency=cfg.extract_concurrency
    )
    # Keep only pairs where extraction succeeded; track failures
    valid_pairs  = [(sr, cr) for sr, cr in extract_pairs if cr is not None]
    run.extracted_ok = len(valid_pairs)
    run.failed      += (len(deduped) - run.extracted_ok)
    log.info("   Extracted %d/%d", run.extracted_ok, len(deduped))

    if not valid_pairs:
        log.error("Extraction failed for all URLs — aborting")
        run.finished_at = time.time()
        return PipelineResult(records=[], run_meta=run)

    # ── Stage 3: Clean ────────────────────────────────────────────────────────
    log.info("── Stage 3/4: Cleaning + Enriching …")
    clean_results = [clean(cr) for _, cr in valid_pairs]
    scrape_results_valid = [sr for sr, _ in valid_pairs]

    # ── Stage 4: Enrich ───────────────────────────────────────────────────────
    enriched_records = await enrich_all(
        list(zip(clean_results, scrape_results_valid)),
        api_key=api_key,
        concurrency=cfg.enrich_concurrency,
    )
    run.enriched_ok = len(enriched_records)

    # Compute avg completeness
    if enriched_records:
        run.avg_completeness = round(
            sum(r.completeness for r in enriched_records) / len(enriched_records), 1
        )

    log.info("   Enriched %d records | avg completeness %.1f%%",
             run.enriched_ok, run.avg_completeness)

    # ── Stage 5: Store in DB ──────────────────────────────────────────────────
    csv_path  = None
    json_path = None

    if save_to_db:
        log.info("── Stage 5/5: Saving to DB + Exporting …")
        with Database(db_path) as db:
            db.upsert_batch(enriched_records)
            db.save_run(run)

            # ── Export ────────────────────────────────────────────────────────
            if do_csv:
                csv_path = export_csv(
                    enriched_records,
                    timestamped_path(out_dir, prefix="results", ext="csv"),
                    include_metadata=do_meta,
                )
            if do_json:
                json_path = export_json(
                    enriched_records,
                    timestamped_path(out_dir, prefix="results", ext="json"),
                    include_metadata=do_meta,
                )
    else:
        if do_csv:
            csv_path = export_csv(
                enriched_records,
                timestamped_path(out_dir, prefix="results", ext="csv"),
                include_metadata=do_meta,
            )
        if do_json:
            json_path = export_json(
                enriched_records,
                timestamped_path(out_dir, prefix="results", ext="json"),
                include_metadata=do_meta,
            )

    run.finished_at = time.time()
    log.info("Pipeline finished in %.1fs", run.duration_seconds())

    return PipelineResult(
        records=enriched_records,
        run_meta=run,
        csv_path=csv_path,
        json_path=json_path,
    )


# ── Public sync API ───────────────────────────────────────────────────────────

def run(
    urls: list[str],
    groq_api_key: Optional[str] = None,
    db_path: Optional[Path]     = None,
    output_dir: Optional[Path]  = None,
    export_csv_flag:  Optional[bool] = None,
    export_json_flag: Optional[bool] = None,
    export_metadata:  Optional[bool] = None,
    save_to_db: bool = True,
) -> PipelineResult:
    """
    Run the full enrichment pipeline synchronously.

    Args:
        urls:             List of company website URLs to process.
        groq_api_key:     Override cfg.groq_api_key.
        db_path:          Override cfg.db_path.
        output_dir:       Override cfg.output_dir.
        export_csv_flag:  Override cfg.export_csv.
        export_json_flag: Override cfg.export_json.
        export_metadata:  Include pipeline metadata columns in exports.
        save_to_db:       Set False to skip DB write (useful for testing).

    Returns:
        PipelineResult with .records, .run_meta, .csv_path, .json_path
    """
    cfg.validate()
    return asyncio.run(_run_async(
        urls=urls,
        groq_api_key=groq_api_key,
        db_path=db_path,
        output_dir=output_dir,
        export_csv_flag=export_csv_flag,
        export_json_flag=export_json_flag,
        export_metadata=export_metadata,
        save_to_db=save_to_db,
    ))


# ── URL file loader ───────────────────────────────────────────────────────────

def load_urls_from_file(path: str | Path) -> list[str]:
    """
    Read URLs from a plain-text file — one URL per line.
    Skips blank lines and lines starting with #.
    """
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    urls  = [
        line.strip() for line in lines
        if line.strip() and not line.strip().startswith("#")
    ]
    log.info("Loaded %d URL(s) from %s", len(urls), path)
    return urls


# ── CLI ───────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pipeline",
        description="AI-powered company data enrichment pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python pipeline.py https://openai.com https://github.com
  python pipeline.py --file urls.txt
  python pipeline.py --file urls.txt --json --metadata
  python pipeline.py --file urls.txt --no-csv --json --output-dir results/
  python pipeline.py --file urls.txt --db pipeline.db --log-level DEBUG
        """,
    )
    p.add_argument("urls", nargs="*", help="Company website URLs to process")
    p.add_argument("--file",       "-f", metavar="PATH",
                   help="Text file with one URL per line")
    p.add_argument("--output-dir", "-o", metavar="DIR",  default=None,
                   help=f"Output directory (default: {cfg.output_dir})")
    p.add_argument("--db",               metavar="PATH", default=None,
                   help=f"SQLite database path (default: {cfg.db_path})")
    p.add_argument("--no-csv",     action="store_true",
                   help="Skip CSV export")
    p.add_argument("--json",       action="store_true",
                   help="Also export JSON")
    p.add_argument("--metadata",   action="store_true",
                   help="Include pipeline metadata columns in exports")
    p.add_argument("--no-db",      action="store_true",
                   help="Skip writing to database")
    p.add_argument("--log-level",  default=None,
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                   help=f"Log verbosity (default: {cfg.log_level})")
    return p


def main(argv: list[str] | None = None) -> int:
    args   = _build_parser().parse_args(argv)
    setup_logging(args.log_level)

    # Collect URLs
    urls: list[str] = list(args.urls)
    if args.file:
        urls.extend(load_urls_from_file(args.file))

    if not urls:
        print("Error: provide URLs as arguments or via --file", file=sys.stderr)
        return 1

    try:
        cfg.validate()
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    result = asyncio.run(_run_async(
        urls=urls,
        db_path=Path(args.db) if args.db else None,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        export_csv_flag=not args.no_csv,
        export_json_flag=args.json,
        export_metadata=args.metadata,
        save_to_db=not args.no_db,
    ))

    result.print_summary()
    return 0 if result.run_meta.enriched_ok > 0 else 1


if __name__ == "__main__":
    sys.exit(main())