"""
test_extractor.py — Tests for extractor.py

Covers:
  1. Unit tests  — prompt builder, JSON parser (no API calls)
  2. Integration — live Groq call against a real ScrapeResult

Run all:           python test_extractor.py
Run unit only:     python test_extractor.py unit
Run integration:   python test_extractor.py integration   (needs GROQ_API_KEY)
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import textwrap
import time
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

# ── make sure local modules resolve ──────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

from extractor import _build_prompt, _parse_response, extract_one, extract_all
from models import CompanyRecord
from backend.scraper import ScrapeResult



# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_scrape(
    url: str = "https://example.com",
    markdown: str = "Example Corp builds SaaS software for small businesses. Founded in 2015, headquartered in London, UK. The company has around 200 employees.",
    title: str = "Example Corp — Home",
    meta: str = "Example Corp: small business SaaS platform.",
    success: bool = True,
) -> ScrapeResult:
    return ScrapeResult(
        url=url, success=success,
        markdown=markdown, title=title,
        meta_description=meta, scrape_method="browser",
    )


def _mock_groq_response(json_payload: dict) -> MagicMock:
    """Return a mock Groq client whose completions.create returns json_payload."""
    choice     = MagicMock()
    choice.message.content = json.dumps(json_payload)
    completion = MagicMock()
    completion.choices = [choice]

    client = MagicMock()
    client.chat = MagicMock()
    client.chat.completions = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=completion)
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__  = AsyncMock(return_value=False)
    return client


# ─────────────────────────────────────────────────────────────────────────────
# 1. Unit tests — no API calls
# ─────────────────────────────────────────────────────────────────────────────

class TestPromptBuilder(unittest.TestCase):

    def test_prompt_contains_url(self):
        sr     = _make_scrape(url="https://stripe.com")
        prompt = _build_prompt(sr)
        self.assertIn("https://stripe.com", prompt)

    def test_prompt_contains_title(self):
        sr     = _make_scrape(title="Stripe — Financial Infrastructure")
        prompt = _build_prompt(sr)
        self.assertIn("Stripe — Financial Infrastructure", prompt)

    def test_prompt_contains_meta(self):
        sr     = _make_scrape(meta="Stripe powers internet commerce.")
        prompt = _build_prompt(sr)
        self.assertIn("Stripe powers internet commerce.", prompt)

    def test_prompt_contains_content_truncated(self):
        long_md = "x " * 5000          # 10 000 chars
        sr      = _make_scrape(markdown=long_md)
        prompt  = _build_prompt(sr)
        # Content window capped at 6000 chars in prompt
        self.assertLessEqual(len(prompt), 15_000)

    def test_prompt_has_json_instruction(self):
        prompt = _build_prompt(_make_scrape())
        self.assertIn("Return JSON only", prompt)

    def test_prompt_has_all_field_names(self):
        prompt = _build_prompt(_make_scrape())
        for field in ["company_name", "industry", "description",
                      "country", "employee_size", "website", "confidence"]:
            self.assertIn(field, prompt)

    def test_no_title_no_meta_still_builds(self):
        sr     = _make_scrape(title="", meta="")
        prompt = _build_prompt(sr)
        self.assertIn("Website:", prompt)


class TestResponseParser(unittest.TestCase):

    def _good_json(self, **overrides) -> str:
        base = {
            "company_name":  "Acme Corp",
            "industry":      "Software / SaaS",
            "description":   "Acme builds widgets.",
            "country":       "United States",
            "employee_size": "51-200",
            "website":       "https://acme.com",
            "confidence":    {"company_name": 0.95, "industry": 0.8},
        }
        base.update(overrides)
        return json.dumps(base)

    def test_clean_json_parses(self):
        record = _parse_response(self._good_json(), "https://acme.com")
        self.assertIsNotNone(record)
        self.assertEqual(record.company_name, "Acme Corp")
        self.assertEqual(record.country, "United States")

    def test_markdown_fenced_json_parses(self):
        fenced = f"```json\n{self._good_json()}\n```"
        record = _parse_response(fenced, "https://acme.com")
        self.assertIsNotNone(record)
        self.assertEqual(record.company_name, "Acme Corp")

    def test_json_with_preamble_parses(self):
        preamble = "Here is the extracted data:\n" + self._good_json()
        record   = _parse_response(preamble, "https://acme.com")
        self.assertIsNotNone(record)

    def test_null_fields_become_none(self):
        payload = self._good_json(country=None, employee_size=None)
        record  = _parse_response(payload, "https://acme.com")
        self.assertIsNone(record.country)
        self.assertIsNone(record.employee_size)

    def test_placeholder_strings_become_none(self):
        payload = self._good_json(country="N/A", description="none")
        record  = _parse_response(payload, "https://acme.com")
        self.assertIsNone(record.country)
        self.assertIsNone(record.description)

    def test_missing_website_falls_back_to_url(self):
        payload = self._good_json(website=None)
        record  = _parse_response(payload, "https://fallback.com")
        self.assertEqual(record.website, "https://fallback.com")

    def test_url_without_scheme_gets_https(self):
        payload = self._good_json(website="acme.com")
        record  = _parse_response(payload, "https://acme.com")
        self.assertTrue(record.website.startswith("https://"))

    def test_confidence_scores_attached(self):
        record = _parse_response(self._good_json(), "https://acme.com")
        self.assertIn("company_name", record.confidence)
        self.assertAlmostEqual(record.confidence["company_name"], 0.95)

    def test_employee_size_normalised(self):
        payload = self._good_json(employee_size="approximately 150 employees")
        record  = _parse_response(payload, "https://acme.com")
        self.assertEqual(record.employee_size, "51-200")

    def test_completely_empty_response_returns_none(self):
        self.assertIsNone(_parse_response("", "https://acme.com"))

    def test_malformed_json_returns_none(self):
        self.assertIsNone(_parse_response("{not valid json!!", "https://acme.com"))

    def test_partial_json_salvaged(self):
        # Missing some fields — should still build a partial record
        partial = json.dumps({
            "company_name": "Partial Co",
            "website":      "https://partial.co",
            "confidence":   {},
        })
        record = _parse_response(partial, "https://partial.co")
        self.assertIsNotNone(record)
        self.assertEqual(record.company_name, "Partial Co")
        self.assertIsNone(record.industry)

    def test_missing_fields_reported(self):
        partial = json.dumps({
            "company_name": "Incomplete",
            "confidence": {},
        })
        record = _parse_response(partial, "https://x.com")
        missing = record.missing_fields()
        self.assertIn("industry", missing)
        self.assertIn("country", missing)
        self.assertNotIn("company_name", missing)

    def test_completeness_percent(self):
        full_json = self._good_json()
        record    = _parse_response(full_json, "https://acme.com")
        self.assertEqual(record.completeness_pct(), 100.0)

        partial   = json.dumps({"company_name": "X", "confidence": {}})
        record2   = _parse_response(partial, "https://x.com")
        self.assertEqual(record2.completeness_pct(), 20.0)   # 1 of 5 core fields


class TestExtractOneMocked(unittest.IsolatedAsyncioTestCase):
    """extract_one() with a mocked Groq client — no network calls."""

    async def test_successful_extraction(self):
        payload = {
            "company_name":  "GitHub",
            "industry":      "Developer Tools",
            "description":   "GitHub is a developer platform.",
            "country":       "United States",
            "employee_size": "1001-5000",
            "website":       "https://github.com",
            "confidence":    {"company_name": 0.99},
        }
        sr     = _make_scrape(url="https://github.com")
        client = _mock_groq_response(payload)

        result_sr, record = await extract_one(client, sr)

        self.assertIs(result_sr, sr)
        self.assertIsNotNone(record)
        self.assertEqual(record.company_name, "GitHub")
        self.assertEqual(record.industry, "Developer Tools")
        self.assertEqual(record.completeness_pct(), 100.0)

    async def test_failed_scrape_returns_none_record(self):
        sr     = _make_scrape(success=False, markdown="")
        client = _mock_groq_response({})

        _, record = await extract_one(client, sr)
        self.assertIsNone(record)
        # Groq should not have been called
        client.chat.completions.create.assert_not_called()

    async def test_groq_error_returns_none_after_retries(self):
        sr     = _make_scrape()
        client = MagicMock()
        client.chat = MagicMock()
        client.chat.completions = MagicMock()
        client.chat.completions.create = AsyncMock(side_effect=Exception("API down"))
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__  = AsyncMock(return_value=False)

        # Speed up retry delay for tests
        import extractor as ext_mod
        original_delay = ext_mod.RETRY_DELAY
        ext_mod.RETRY_DELAY = 0.0
        try:
            _, record = await extract_one(client, sr)
        finally:
            ext_mod.RETRY_DELAY = original_delay

        self.assertIsNone(record)

    async def test_malformed_groq_response_returns_none(self):
        choice            = MagicMock()
        choice.message.content = "Sorry, I cannot extract that."
        completion        = MagicMock()
        completion.choices = [choice]
        client            = MagicMock()
        client.chat.completions.create = AsyncMock(return_value=completion)
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__  = AsyncMock(return_value=False)

        import extractor as ext_mod
        original_delay = ext_mod.RETRY_DELAY
        ext_mod.RETRY_DELAY = 0.0
        try:
            _, record = await extract_one(client, _make_scrape())
        finally:
            ext_mod.RETRY_DELAY = original_delay

        self.assertIsNone(record)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Integration test — real Groq call (skipped if no API key)
# ─────────────────────────────────────────────────────────────────────────────

GROQ_KEY = os.environ.get("GROQ_API_KEY")

GITHUB_CONTENT = textwrap.dedent("""
    GitHub is a developer platform that allows developers to create, store,
    manage, and share their code. It uses Git software, providing the distributed
    version control of Git plus access control, bug tracking, software feature
    requests, task management, continuous integration, and wikis for every project.
    GitHub is headquartered in San Francisco, California, United States. It was
    founded in 2008 and acquired by Microsoft in 2018. GitHub has over 100 million
    developers and more than 90% of the Fortune 100 use GitHub. The company employs
    approximately 3,000 people.
""").strip()


@unittest.skipUnless(GROQ_KEY, "GROQ_API_KEY not set — skipping integration tests")
class TestExtractorIntegration(unittest.IsolatedAsyncioTestCase):

    async def test_extract_github_like_content(self):
        """Real Groq call — validates end-to-end extraction quality."""
        from groq import AsyncGroq

        sr = ScrapeResult(
            url="https://github.com",
            success=True,
            markdown=GITHUB_CONTENT,
            title="GitHub: Let's build from here",
            meta_description="GitHub is where over 100 million developers shape the future of software.",
            scrape_method="test",
        )

        async with AsyncGroq(api_key=GROQ_KEY) as client:
            _, record = await extract_one(client, sr)

        self.assertIsNotNone(record, "Extraction returned None — check Groq API key / response")

        # Flexible assertions — model should get these right
        self.assertIsNotNone(record.company_name)
        self.assertIn("GitHub", record.company_name or "")

        self.assertIsNotNone(record.country)
        self.assertIn("United States", record.country or "")

        self.assertIsNotNone(record.description)
        self.assertGreater(len(record.description or ""), 30)

        self.assertIsNotNone(record.employee_size)

        self.assertGreaterEqual(record.completeness_pct(), 80.0,
            f"Expected ≥80% completeness, got {record.completeness_pct()}%")

        print(f"\n  ── Integration result ──")
        print(f"  company_name  : {record.company_name}")
        print(f"  industry      : {record.industry}")
        print(f"  description   : {record.description}")
        print(f"  country       : {record.country}")
        print(f"  employee_size : {record.employee_size}")
        print(f"  website       : {record.website}")
        print(f"  completeness  : {record.completeness_pct()}%")
        print(f"  confidence    : {record.confidence}")

    async def test_extract_all_batch(self):
        """Batch extraction of two records."""
        sr1 = ScrapeResult(
            url="https://github.com", success=True, scrape_method="test",
            markdown=GITHUB_CONTENT,
            title="GitHub", meta_description="Developer platform.",
        )
        sr2 = ScrapeResult(
            url="https://stripe.com", success=True, scrape_method="test",
            markdown=(
                "Stripe is a financial infrastructure platform for businesses. "
                "Millions of companies—from the world's largest enterprises to the most "
                "ambitious startups—use Stripe to accept payments, grow their revenue, "
                "and accelerate new business opportunities. Stripe is headquartered in "
                "South San Francisco, California. The company was founded in 2010 by "
                "Irish brothers Patrick and John Collison. Stripe employs over 8,000 people."
            ),
            title="Stripe", meta_description="Financial infrastructure platform.",
        )
        sr_failed = ScrapeResult(url="https://blocked.com", success=False, markdown="")

        pairs = await extract_all(
            [sr1, sr2, sr_failed],
            api_key=GROQ_KEY,
            concurrency=2,
        )

        self.assertEqual(len(pairs), 3)

        _, r1 = pairs[0]
        _, r2 = pairs[1]
        _, r3 = pairs[2]

        self.assertIsNotNone(r1)
        self.assertIsNotNone(r2)
        self.assertIsNone(r3)   # failed scrape → no extraction

        self.assertIn("GitHub", r1.company_name or "")
        self.assertIn("Stripe", r2.company_name or "")

        print(f"\n  ── Batch results ──")
        for sr, rec in pairs:
            status = f"{rec.company_name} ({rec.completeness_pct()}%)" if rec else "FAILED"
            print(f"  {sr.url}: {status}")


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"

    if mode == "unit":
        suite = unittest.TestSuite()
        for cls in [TestPromptBuilder, TestResponseParser, TestExtractOneMocked]:
            suite.addTests(unittest.TestLoader().loadTestsFromTestCase(cls))
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)

    elif mode == "integration":
        suite = unittest.TestLoader().loadTestsFromTestCase(TestExtractorIntegration)
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)

    else:  # all
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(unittest.TestLoader().loadTestsFromModule(
            __import__(__name__)
        ))

    sys.exit(0 if result.wasSuccessful() else 1)