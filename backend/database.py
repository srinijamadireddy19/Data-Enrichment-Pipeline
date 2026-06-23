"""
database.py — SQLite persistence layer for the Data Enrichment Pipeline

Schema:
  companies      — one row per enriched company (upsert by website)
  pipeline_runs  — one row per pipeline execution with summary stats

Usage:
    db = Database("pipeline.db")
    db.init()
    db.upsert(enriched_record)
    db.upsert_batch(records)
    rows = db.get_all()
    db.close()

    # or as a context manager:
    with Database("pipeline.db") as db:
        db.upsert_batch(records)
"""

from __future__ import annotations

import logging
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

from models import EnrichedRecord, PipelineRun

log = logging.getLogger("database")

# ── Schema ────────────────────────────────────────────────────────────────────

_CREATE_COMPANIES = """
CREATE TABLE IF NOT EXISTS companies (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    website       TEXT    NOT NULL UNIQUE,   -- dedup key
    company_name  TEXT    NOT NULL,
    industry      TEXT    NOT NULL,
    description   TEXT    NOT NULL,
    country       TEXT    NOT NULL,
    employee_size TEXT    NOT NULL,
    scrape_method TEXT    NOT NULL DEFAULT 'unknown',
    completeness  REAL    NOT NULL DEFAULT 0.0,
    ai_enriched   INTEGER NOT NULL DEFAULT 0,   -- SQLite has no BOOL; 0/1
    created_at    REAL    NOT NULL,
    updated_at    REAL    NOT NULL
);
"""

_CREATE_PIPELINE_RUNS = """
CREATE TABLE IF NOT EXISTS pipeline_runs (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    total_urls       INTEGER NOT NULL DEFAULT 0,
    scraped_ok       INTEGER NOT NULL DEFAULT 0,
    extracted_ok     INTEGER NOT NULL DEFAULT 0,
    enriched_ok      INTEGER NOT NULL DEFAULT 0,
    failed           INTEGER NOT NULL DEFAULT 0,
    avg_completeness REAL    NOT NULL DEFAULT 0.0,
    started_at       REAL    NOT NULL,
    finished_at      REAL
);
"""

_CREATE_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_companies_website ON companies(website);",
    "CREATE INDEX IF NOT EXISTS idx_companies_country ON companies(country);",
    "CREATE INDEX IF NOT EXISTS idx_companies_industry ON companies(industry);",
    "CREATE INDEX IF NOT EXISTS idx_pipeline_runs_started ON pipeline_runs(started_at);",
]


# ── Database class ────────────────────────────────────────────────────────────

class Database:
    """
    SQLite wrapper for the pipeline.

    Thread-safety: SQLite connections are not thread-safe by default.
    This class uses check_same_thread=False with WAL mode for read
    concurrency, but write operations should be serialised externally
    (the pipeline's asyncio.Semaphore on the enricher is sufficient).
    """

    def __init__(self, path: str | Path = "pipeline.db"):
        self.path = Path(path)
        self._conn: Optional[sqlite3.Connection] = None

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def connect(self) -> "Database":
        """Open connection and enable WAL + foreign keys."""
        self._conn = sqlite3.connect(
            str(self.path),
            check_same_thread=False,
            detect_types=sqlite3.PARSE_DECLTYPES,
        )
        self._conn.row_factory = sqlite3.Row   # column access by name
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA foreign_keys=ON;")
        self._conn.execute("PRAGMA synchronous=NORMAL;")
        log.info("Connected to %s", self.path)
        return self

    def init(self) -> "Database":
        """Connect (if needed) and create tables + indexes."""
        if not self._conn:
            self.connect()
        with self._tx():
            self._conn.execute(_CREATE_COMPANIES)
            self._conn.execute(_CREATE_PIPELINE_RUNS)
            for idx in _CREATE_INDEXES:
                self._conn.execute(idx)
        log.info("Schema initialised")
        return self

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None
            log.info("Connection closed")

    def __enter__(self) -> "Database":
        return self.init()

    def __exit__(self, *_) -> None:
        self.close()

    # ── Transaction helper ────────────────────────────────────────────────────

    @contextmanager
    def _tx(self):
        """Yield inside a transaction; auto-commit or rollback."""
        try:
            yield
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

    def _require_conn(self) -> sqlite3.Connection:
        if not self._conn:
            raise RuntimeError("Database not initialised — call .init() or use as context manager")
        return self._conn

    # ── Companies table ───────────────────────────────────────────────────────

    def upsert(self, record: EnrichedRecord) -> int:
        """
        Insert or update a company record (keyed by website URL).
        Returns the row id.
        """
        conn = self._require_conn()
        now  = time.time()

        sql = """
        INSERT INTO companies
            (website, company_name, industry, description, country,
             employee_size, scrape_method, completeness, ai_enriched,
             created_at, updated_at)
        VALUES
            (:website, :company_name, :industry, :description, :country,
             :employee_size, :scrape_method, :completeness, :ai_enriched,
             :created_at, :updated_at)
        ON CONFLICT(website) DO UPDATE SET
            company_name  = excluded.company_name,
            industry      = excluded.industry,
            description   = excluded.description,
            country       = excluded.country,
            employee_size = excluded.employee_size,
            scrape_method = excluded.scrape_method,
            completeness  = excluded.completeness,
            ai_enriched   = excluded.ai_enriched,
            updated_at    = excluded.updated_at
        """

        params = {
            "website":       record.website,
            "company_name":  record.company_name,
            "industry":      record.industry,
            "description":   record.description,
            "country":       record.country,
            "employee_size": record.employee_size,
            "scrape_method": record.scrape_method,
            "completeness":  record.completeness,
            "ai_enriched":   int(record.ai_enriched),
            "created_at":    record.created_at,
            "updated_at":    now,
        }

        with self._tx():
            cursor = conn.execute(sql, params)
            row_id = cursor.lastrowid

        log.debug("Upserted %s → row %d", record.website, row_id)
        return row_id

    def upsert_batch(self, records: list[EnrichedRecord]) -> list[int]:
        """
        Upsert multiple records in a single transaction.
        Much faster than calling upsert() in a loop.
        """
        if not records:
            return []

        conn = self._require_conn()
        now  = time.time()

        sql = """
        INSERT INTO companies
            (website, company_name, industry, description, country,
             employee_size, scrape_method, completeness, ai_enriched,
             created_at, updated_at)
        VALUES
            (:website, :company_name, :industry, :description, :country,
             :employee_size, :scrape_method, :completeness, :ai_enriched,
             :created_at, :updated_at)
        ON CONFLICT(website) DO UPDATE SET
            company_name  = excluded.company_name,
            industry      = excluded.industry,
            description   = excluded.description,
            country       = excluded.country,
            employee_size = excluded.employee_size,
            scrape_method = excluded.scrape_method,
            completeness  = excluded.completeness,
            ai_enriched   = excluded.ai_enriched,
            updated_at    = excluded.updated_at
        """

        params_list = [
            {
                "website":       r.website,
                "company_name":  r.company_name,
                "industry":      r.industry,
                "description":   r.description,
                "country":       r.country,
                "employee_size": r.employee_size,
                "scrape_method": r.scrape_method,
                "completeness":  r.completeness,
                "ai_enriched":   int(r.ai_enriched),
                "created_at":    r.created_at,
                "updated_at":    now,
            }
            for r in records
        ]

        row_ids: list[int] = []
        with self._tx():
            for params in params_list:
                cursor = conn.execute(sql, params)
                row_ids.append(cursor.lastrowid)

        log.info("Upserted %d record(s) in batch", len(row_ids))
        return row_ids

    def get_all(self) -> list[dict]:
        """Return all companies as a list of dicts."""
        conn = self._require_conn()
        rows = conn.execute(
            "SELECT * FROM companies ORDER BY created_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]

    def get_by_website(self, website: str) -> Optional[dict]:
        """Fetch a single company by its website URL."""
        conn = self._require_conn()
        row = conn.execute(
            "SELECT * FROM companies WHERE website = ?", (website,)
        ).fetchone()
        return dict(row) if row else None

    def get_by_country(self, country: str) -> list[dict]:
        conn = self._require_conn()
        rows = conn.execute(
            "SELECT * FROM companies WHERE country = ? ORDER BY company_name",
            (country,),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_by_industry(self, industry: str) -> list[dict]:
        conn = self._require_conn()
        rows = conn.execute(
            "SELECT * FROM companies WHERE industry = ? ORDER BY company_name",
            (industry,),
        ).fetchall()
        return [dict(r) for r in rows]

    def delete_by_website(self, website: str) -> bool:
        """Delete a company record. Returns True if a row was deleted."""
        conn = self._require_conn()
        with self._tx():
            cursor = conn.execute(
                "DELETE FROM companies WHERE website = ?", (website,)
            )
        return cursor.rowcount > 0

    def count(self) -> int:
        return self._require_conn().execute(
            "SELECT COUNT(*) FROM companies"
        ).fetchone()[0]

    # ── Pipeline runs table ───────────────────────────────────────────────────

    def save_run(self, run: PipelineRun) -> int:
        """Persist a PipelineRun summary. Returns the run row id."""
        conn = self._require_conn()
        sql = """
        INSERT INTO pipeline_runs
            (total_urls, scraped_ok, extracted_ok, enriched_ok,
             failed, avg_completeness, started_at, finished_at)
        VALUES
            (:total_urls, :scraped_ok, :extracted_ok, :enriched_ok,
             :failed, :avg_completeness, :started_at, :finished_at)
        """
        with self._tx():
            cursor = conn.execute(sql, {
                "total_urls":       run.total_urls,
                "scraped_ok":       run.scraped_ok,
                "extracted_ok":     run.extracted_ok,
                "enriched_ok":      run.enriched_ok,
                "failed":           run.failed,
                "avg_completeness": run.avg_completeness,
                "started_at":       run.started_at,
                "finished_at":      run.finished_at,
            })
        log.info("Saved pipeline run → row %d", cursor.lastrowid)
        return cursor.lastrowid

    def get_runs(self, limit: int = 20) -> list[dict]:
        """Return recent pipeline runs."""
        rows = self._require_conn().execute(
            "SELECT * FROM pipeline_runs ORDER BY started_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]

    # ── Stats helpers ─────────────────────────────────────────────────────────

    def stats(self) -> dict:
        """Return a quick summary of the companies table."""
        conn = self._require_conn()
        total        = conn.execute("SELECT COUNT(*) FROM companies").fetchone()[0]
        ai_enriched  = conn.execute("SELECT COUNT(*) FROM companies WHERE ai_enriched=1").fetchone()[0]
        avg_complete = conn.execute("SELECT AVG(completeness) FROM companies").fetchone()[0] or 0.0
        by_country   = conn.execute(
            "SELECT country, COUNT(*) as n FROM companies GROUP BY country ORDER BY n DESC LIMIT 10"
        ).fetchall()
        by_industry  = conn.execute(
            "SELECT industry, COUNT(*) as n FROM companies GROUP BY industry ORDER BY n DESC LIMIT 10"
        ).fetchall()
        return {
            "total":            total,
            "ai_enriched":      ai_enriched,
            "avg_completeness": round(avg_complete, 1),
            "by_country":       [dict(r) for r in by_country],
            "by_industry":      [dict(r) for r in by_industry],
        }