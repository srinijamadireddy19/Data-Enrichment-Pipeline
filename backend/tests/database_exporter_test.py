"""
test_database_exporter.py — Tests for database.py and exporter.py

Structure:
  TestDatabaseInit        — connection, schema, WAL mode
  TestDatabaseUpsert      — insert, update, dedup by website
  TestDatabaseBatch       — upsert_batch, single transaction
  TestDatabaseQueries     — get_all, get_by_website, get_by_country, get_by_industry
  TestDatabaseDelete      — delete_by_website
  TestDatabasePipelineRun — save_run, get_runs
  TestDatabaseStats       — stats() aggregation
  TestExporterCSV         — column order, content, metadata flag
  TestExporterJSON        — structure, metadata flag
  TestExporterFromDB      — export_csv_from_db / export_json_from_db with filters
  TestTimestampedPath     — path generation helper

Run:
  python test_database_exporter.py
  python test_database_exporter.py unit
"""

from __future__ import annotations

import csv
import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

from database import Database
from exporter import (
    export_csv, export_json,
    export_csv_from_db, export_json_from_db,
    timestamped_path, CSV_COLUMNS, CSV_COLUMNS_FULL,
)
from models import EnrichedRecord, PipelineRun


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _record(
    website       = "https://acme.com",
    company_name  = "Acme Corp",
    industry      = "Software / SaaS",
    description   = "Acme builds widgets for small businesses.",
    country       = "United States",
    employee_size = "51-200",
    scrape_method = "browser",
    completeness  = 100.0,
    ai_enriched   = False,
) -> EnrichedRecord:
    return EnrichedRecord(
        website=website, company_name=company_name, industry=industry,
        description=description, country=country, employee_size=employee_size,
        scrape_method=scrape_method, completeness=completeness, ai_enriched=ai_enriched,
    )


def _run(**kwargs) -> PipelineRun:
    base = dict(total_urls=3, scraped_ok=3, extracted_ok=3,
                enriched_ok=3, failed=0, avg_completeness=95.0,
                finished_at=time.time() + 10)
    base.update(kwargs)
    return PipelineRun(**base)


# ─────────────────────────────────────────────────────────────────────────────
# DATABASE TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestDatabaseInit(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.tmp.close()
        self.db_path = self.tmp.name

    def tearDown(self):
        os.unlink(self.db_path)

    def test_context_manager_opens_and_closes(self):
        with Database(self.db_path) as db:
            self.assertIsNotNone(db._conn)
        self.assertIsNone(db._conn)

    def test_creates_companies_table(self):
        with Database(self.db_path) as db:
            tables = db._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
            names = {r[0] for r in tables}
            self.assertIn("companies", names)

    def test_creates_pipeline_runs_table(self):
        with Database(self.db_path) as db:
            tables = db._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
            names = {r[0] for r in tables}
            self.assertIn("pipeline_runs", names)

    def test_creates_indexes(self):
        with Database(self.db_path) as db:
            indexes = db._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index'"
            ).fetchall()
            names = {r[0] for r in indexes}
            self.assertIn("idx_companies_website", names)
            self.assertIn("idx_companies_country", names)
            self.assertIn("idx_companies_industry", names)

    def test_init_idempotent(self):
        # Calling init() twice should not raise
        db = Database(self.db_path)
        db.init()
        db.init()
        db.close()

    def test_requires_init_before_queries(self):
        db = Database(self.db_path)
        with self.assertRaises(RuntimeError):
            db.get_all()


class TestDatabaseUpsert(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.tmp.close()
        self.db = Database(self.tmp.name).init()

    def tearDown(self):
        self.db.close()
        os.unlink(self.tmp.name)

    def test_insert_returns_row_id(self):
        row_id = self.db.upsert(_record())
        self.assertIsInstance(row_id, int)
        self.assertGreater(row_id, 0)

    def test_record_retrievable_after_insert(self):
        self.db.upsert(_record())
        row = self.db.get_by_website("https://acme.com")
        self.assertIsNotNone(row)
        self.assertEqual(row["company_name"], "Acme Corp")

    def test_all_fields_persisted(self):
        rec = _record(
            company_name="TestCo", industry="Fintech",
            country="Germany", employee_size="201-500",
            scrape_method="httpx", completeness=80.0, ai_enriched=True,
        )
        self.db.upsert(rec)
        row = self.db.get_by_website("https://acme.com")
        self.assertEqual(row["company_name"],  "TestCo")
        self.assertEqual(row["industry"],      "Fintech")
        self.assertEqual(row["country"],       "Germany")
        self.assertEqual(row["employee_size"], "201-500")
        self.assertEqual(row["scrape_method"], "httpx")
        self.assertAlmostEqual(row["completeness"], 80.0)
        self.assertEqual(row["ai_enriched"], 1)

    def test_upsert_updates_existing(self):
        self.db.upsert(_record(company_name="Old Name"))
        self.db.upsert(_record(company_name="New Name"))   # same URL
        rows = self.db.get_all()
        self.assertEqual(len(rows), 1)                     # still one row
        self.assertEqual(rows[0]["company_name"], "New Name")

    def test_upsert_deduplicates_by_website(self):
        self.db.upsert(_record(website="https://a.com"))
        self.db.upsert(_record(website="https://a.com"))   # same URL again
        self.assertEqual(self.db.count(), 1)

    def test_different_websites_create_separate_rows(self):
        self.db.upsert(_record(website="https://a.com"))
        self.db.upsert(_record(website="https://b.com"))
        self.assertEqual(self.db.count(), 2)

    def test_ai_enriched_stored_as_integer(self):
        self.db.upsert(_record(ai_enriched=True))
        row = self.db.get_by_website("https://acme.com")
        # SQLite stores bool as 0/1
        self.assertIn(row["ai_enriched"], (0, 1, True, False))
        self.assertTrue(bool(row["ai_enriched"]))


class TestDatabaseBatch(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.tmp.close()
        self.db = Database(self.tmp.name).init()

    def tearDown(self):
        self.db.close()
        os.unlink(self.tmp.name)

    def test_batch_inserts_all(self):
        records = [
            _record(website="https://a.com"),
            _record(website="https://b.com"),
            _record(website="https://c.com"),
        ]
        ids = self.db.upsert_batch(records)
        self.assertEqual(len(ids), 3)
        self.assertEqual(self.db.count(), 3)

    def test_batch_returns_row_ids(self):
        records = [_record(website=f"https://co{i}.com") for i in range(5)]
        ids = self.db.upsert_batch(records)
        self.assertEqual(len(ids), 5)
        self.assertTrue(all(isinstance(i, int) and i > 0 for i in ids))

    def test_batch_upserts_duplicates(self):
        records = [
            _record(website="https://a.com", company_name="Old"),
            _record(website="https://a.com", company_name="New"),   # dup → update
        ]
        self.db.upsert_batch(records)
        self.assertEqual(self.db.count(), 1)
        row = self.db.get_by_website("https://a.com")
        self.assertEqual(row["company_name"], "New")

    def test_empty_batch_returns_empty(self):
        result = self.db.upsert_batch([])
        self.assertEqual(result, [])


class TestDatabaseQueries(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.tmp.close()
        self.db = Database(self.tmp.name).init()
        self.db.upsert_batch([
            _record(website="https://a.com", country="United States", industry="Software / SaaS"),
            _record(website="https://b.com", country="Germany",       industry="Software / SaaS"),
            _record(website="https://c.com", country="United States", industry="Fintech"),
        ])

    def tearDown(self):
        self.db.close()
        os.unlink(self.tmp.name)

    def test_get_all_returns_all(self):
        rows = self.db.get_all()
        self.assertEqual(len(rows), 3)

    def test_get_by_website_found(self):
        row = self.db.get_by_website("https://b.com")
        self.assertIsNotNone(row)
        self.assertEqual(row["country"], "Germany")

    def test_get_by_website_not_found(self):
        row = self.db.get_by_website("https://missing.com")
        self.assertIsNone(row)

    def test_get_by_country(self):
        rows = self.db.get_by_country("United States")
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(r["country"] == "United States" for r in rows))

    def test_get_by_industry(self):
        rows = self.db.get_by_industry("Software / SaaS")
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(r["industry"] == "Software / SaaS" for r in rows))

    def test_count(self):
        self.assertEqual(self.db.count(), 3)

    def test_get_all_returns_dicts(self):
        rows = self.db.get_all()
        self.assertIsInstance(rows[0], dict)
        self.assertIn("company_name", rows[0])


class TestDatabaseDelete(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.tmp.close()
        self.db = Database(self.tmp.name).init()
        self.db.upsert(_record())

    def tearDown(self):
        self.db.close()
        os.unlink(self.tmp.name)

    def test_delete_existing(self):
        deleted = self.db.delete_by_website("https://acme.com")
        self.assertTrue(deleted)
        self.assertEqual(self.db.count(), 0)

    def test_delete_nonexistent_returns_false(self):
        deleted = self.db.delete_by_website("https://nothere.com")
        self.assertFalse(deleted)

    def test_delete_leaves_others_intact(self):
        self.db.upsert(_record(website="https://other.com"))
        self.db.delete_by_website("https://acme.com")
        self.assertEqual(self.db.count(), 1)
        self.assertIsNotNone(self.db.get_by_website("https://other.com"))


class TestDatabasePipelineRun(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.tmp.close()
        self.db = Database(self.tmp.name).init()

    def tearDown(self):
        self.db.close()
        os.unlink(self.tmp.name)

    def test_save_run_returns_id(self):
        run_id = self.db.save_run(_run())
        self.assertIsInstance(run_id, int)
        self.assertGreater(run_id, 0)

    def test_run_retrievable(self):
        self.db.save_run(_run(total_urls=5, failed=1))
        runs = self.db.get_runs()
        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0]["total_urls"], 5)
        self.assertEqual(runs[0]["failed"], 1)

    def test_get_runs_limit(self):
        for i in range(5):
            self.db.save_run(_run())
        runs = self.db.get_runs(limit=3)
        self.assertEqual(len(runs), 3)

    def test_multiple_runs_stored(self):
        self.db.save_run(_run(total_urls=3))
        self.db.save_run(_run(total_urls=7))
        runs = self.db.get_runs()
        self.assertEqual(len(runs), 2)


class TestDatabaseStats(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.tmp.close()
        self.db = Database(self.tmp.name).init()
        self.db.upsert_batch([
            _record(website="https://a.com", country="United States", industry="Software / SaaS", ai_enriched=True,  completeness=100.0),
            _record(website="https://b.com", country="Germany",       industry="Software / SaaS", ai_enriched=False, completeness=80.0),
            _record(website="https://c.com", country="United States", industry="Fintech",         ai_enriched=True,  completeness=60.0),
        ])

    def tearDown(self):
        self.db.close()
        os.unlink(self.tmp.name)

    def test_total_count(self):
        stats = self.db.stats()
        self.assertEqual(stats["total"], 3)

    def test_ai_enriched_count(self):
        stats = self.db.stats()
        self.assertEqual(stats["ai_enriched"], 2)

    def test_avg_completeness(self):
        stats = self.db.stats()
        self.assertAlmostEqual(stats["avg_completeness"], 80.0, places=0)

    def test_by_country_top_entry(self):
        stats = self.db.stats()
        # United States has 2 entries → should be first
        self.assertEqual(stats["by_country"][0]["country"], "United States")
        self.assertEqual(stats["by_country"][0]["n"], 2)

    def test_by_industry_present(self):
        stats = self.db.stats()
        industries = {r["industry"] for r in stats["by_industry"]}
        self.assertIn("Software / SaaS", industries)
        self.assertIn("Fintech", industries)


# ─────────────────────────────────────────────────────────────────────────────
# EXPORTER TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestExporterCSV(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.records = [
            _record(website="https://a.com", company_name="Alpha", country="Germany"),
            _record(website="https://b.com", company_name="Beta",  country="France", ai_enriched=True, completeness=80.0),
        ]

    def test_creates_file(self):
        out = export_csv(self.records, Path(self.tmp_dir) / "out.csv")
        self.assertTrue(out.exists())

    def test_correct_column_order(self):
        out = export_csv(self.records, Path(self.tmp_dir) / "out.csv")
        with out.open() as f:
            reader = csv.DictReader(f)
            self.assertEqual(reader.fieldnames, CSV_COLUMNS)

    def test_correct_row_count(self):
        out = export_csv(self.records, Path(self.tmp_dir) / "out.csv")
        with out.open() as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 2)

    def test_correct_values(self):
        out = export_csv(self.records, Path(self.tmp_dir) / "out.csv")
        with out.open() as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(rows[0]["company_name"], "Alpha")
        self.assertEqual(rows[0]["country"], "Germany")
        self.assertEqual(rows[1]["company_name"], "Beta")

    def test_no_metadata_columns_by_default(self):
        out = export_csv(self.records, Path(self.tmp_dir) / "out.csv")
        with out.open() as f:
            reader = csv.DictReader(f)
            self.assertNotIn("scrape_method", reader.fieldnames)
            self.assertNotIn("completeness", reader.fieldnames)
            self.assertNotIn("ai_enriched", reader.fieldnames)

    def test_metadata_columns_when_flagged(self):
        out = export_csv(self.records, Path(self.tmp_dir) / "meta.csv", include_metadata=True)
        with out.open() as f:
            reader = csv.DictReader(f)
            self.assertIn("scrape_method", reader.fieldnames)
            self.assertIn("completeness", reader.fieldnames)
            self.assertIn("ai_enriched", reader.fieldnames)

    def test_ai_enriched_as_yes_no(self):
        out = export_csv(self.records, Path(self.tmp_dir) / "meta.csv", include_metadata=True)
        with out.open() as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(rows[0]["ai_enriched"], "no")
        self.assertEqual(rows[1]["ai_enriched"], "yes")

    def test_creates_parent_dirs(self):
        nested = Path(self.tmp_dir) / "a" / "b" / "out.csv"
        out = export_csv(self.records, nested)
        self.assertTrue(out.exists())

    def test_empty_records_creates_header_only(self):
        out = export_csv([], Path(self.tmp_dir) / "empty.csv")
        with out.open() as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 0)

    def test_utf8_characters_exported(self):
        rec = _record(company_name="Société Générale", country="France")
        out = export_csv([rec], Path(self.tmp_dir) / "utf8.csv")
        content = out.read_text(encoding="utf-8")
        self.assertIn("Société Générale", content)


class TestExporterJSON(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.records = [
            _record(website="https://a.com", company_name="Alpha"),
            _record(website="https://b.com", company_name="Beta", ai_enriched=True),
        ]

    def test_creates_valid_json(self):
        out = export_json(self.records, Path(self.tmp_dir) / "out.json")
        data = json.loads(out.read_text())
        self.assertIsInstance(data, list)

    def test_correct_record_count(self):
        out = export_json(self.records, Path(self.tmp_dir) / "out.json")
        data = json.loads(out.read_text())
        self.assertEqual(len(data), 2)

    def test_correct_fields(self):
        out = export_json(self.records, Path(self.tmp_dir) / "out.json")
        data = json.loads(out.read_text())
        self.assertIn("company_name", data[0])
        self.assertIn("website", data[0])
        self.assertEqual(data[0]["company_name"], "Alpha")

    def test_no_metadata_by_default(self):
        out = export_json(self.records, Path(self.tmp_dir) / "out.json")
        data = json.loads(out.read_text())
        self.assertNotIn("scrape_method", data[0])
        self.assertNotIn("ai_enriched", data[0])

    def test_metadata_included_when_flagged(self):
        out = export_json(self.records, Path(self.tmp_dir) / "out.json", include_metadata=True)
        data = json.loads(out.read_text())
        self.assertIn("scrape_method", data[0])
        self.assertIn("ai_enriched", data[0])
        self.assertIn("completeness", data[0])

    def test_ai_enriched_is_boolean_in_json(self):
        out = export_json(self.records, Path(self.tmp_dir) / "out.json", include_metadata=True)
        data = json.loads(out.read_text())
        self.assertIsInstance(data[1]["ai_enriched"], bool)
        self.assertTrue(data[1]["ai_enriched"])

    def test_unicode_preserved(self):
        rec = _record(company_name="日本企業")
        out = export_json([rec], Path(self.tmp_dir) / "unicode.json")
        data = json.loads(out.read_text(encoding="utf-8"))
        self.assertEqual(data[0]["company_name"], "日本企業")


class TestExporterFromDB(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.db_file = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.db_file.close()
        self.db = Database(self.db_file.name).init()
        self.db.upsert_batch([
            _record(website="https://a.com", country="United States", industry="Fintech"),
            _record(website="https://b.com", country="Germany",       industry="Software / SaaS"),
            _record(website="https://c.com", country="United States", industry="Software / SaaS"),
        ])

    def tearDown(self):
        self.db.close()
        os.unlink(self.db_file.name)

    def test_csv_from_db_all_records(self):
        out = export_csv_from_db(self.db, Path(self.tmp_dir) / "all.csv")
        with out.open() as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 3)

    def test_csv_from_db_filter_country(self):
        out = export_csv_from_db(self.db, Path(self.tmp_dir) / "us.csv", country="United States")
        with out.open() as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(r["country"] == "United States" for r in rows))

    def test_csv_from_db_filter_industry(self):
        out = export_csv_from_db(self.db, Path(self.tmp_dir) / "saas.csv", industry="Software / SaaS")
        with out.open() as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 2)

    def test_json_from_db(self):
        out = export_json_from_db(self.db, Path(self.tmp_dir) / "all.json")
        data = json.loads(out.read_text())
        self.assertEqual(len(data), 3)


class TestTimestampedPath(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()

    def test_creates_csv_path(self):
        path = timestamped_path(self.tmp_dir, prefix="results", ext="csv")
        self.assertEqual(path.suffix, ".csv")

    def test_contains_timestamp(self):
        path = timestamped_path(self.tmp_dir)
        # Timestamp format: YYYYMMDD_HHMMSS
        import re
        self.assertTrue(re.search(r"\d{8}_\d{6}", path.name))

    def test_contains_prefix(self):
        path = timestamped_path(self.tmp_dir, prefix="export")
        self.assertTrue(path.name.startswith("export_"))

    def test_creates_directory_if_missing(self):
        nested = Path(self.tmp_dir) / "new_subdir"
        path = timestamped_path(nested)
        self.assertTrue(nested.exists())

    def test_two_calls_produce_different_paths(self):
        # Sleep 1s to guarantee different timestamps
        p1 = timestamped_path(self.tmp_dir)
        time.sleep(1.1)
        p2 = timestamped_path(self.tmp_dir)
        self.assertNotEqual(p1, p2)


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"

    loader = unittest.TestLoader()
    all_classes = [
        TestDatabaseInit, TestDatabaseUpsert, TestDatabaseBatch,
        TestDatabaseQueries, TestDatabaseDelete, TestDatabasePipelineRun,
        TestDatabaseStats, TestExporterCSV, TestExporterJSON,
        TestExporterFromDB, TestTimestampedPath,
    ]

    if mode == "unit":
        suite = unittest.TestSuite()
        for cls in all_classes:
            suite.addTests(loader.loadTestsFromTestCase(cls))
    else:
        suite = unittest.TestSuite()
        for cls in all_classes:
            suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)