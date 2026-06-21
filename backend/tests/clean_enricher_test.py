"""
test_cleaner_enricher.py — Tests for cleaner.py and enricher.py

Structure:
  TestCountryNormalisation   — country alias map
  TestIndustryNormalisation  — keyword → canonical industry
  TestDescriptionCleaning    — boilerplate stripping, length trim
  TestWebsiteCleaning        — URL normalisation
  TestEmployeeCleaning       — band mapping
  TestCleanFull              — full clean() integration
  TestEnricherPrompt         — prompt builder (no API)
  TestEnricherMerge          — _merge() logic
  TestEnricherMocked         — enrich_one() with mocked Groq
  TestEnricherIntegration    — real Groq call (needs GROQ_API_KEY)

Run:
  python test_cleaner_enricher.py           # all tests
  python test_cleaner_enricher.py unit      # skip integration
  python test_cleaner_enricher.py integration
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import unittest
from unittest.mock import AsyncMock, MagicMock

sys.path.insert(0, os.path.dirname(__file__))

from cleaner import (
    clean, clean_batch,
    _clean_company_name, _clean_country, _clean_industry,
    _clean_description, _clean_website, _clean_employee_size,
    _low_confidence_fields, CleanResult,
)
from enricher import (
    _build_enrichment_prompt, _parse_enrichment, _merge,
    enrich_one, enrich_all,
)
from models import CompanyRecord, EnrichedRecord, Industry, EmployeeSize
from scraper import ScrapeResult


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _record(**kwargs) -> CompanyRecord:
    base = dict(
        company_name="Acme Corp",
        industry="Software / SaaS",
        description="Acme builds widgets for small businesses.",
        country="United States",
        employee_size="51-200",
        website="https://acme.com",
        confidence={},
    )
    base.update(kwargs)
    return CompanyRecord(**base)


def _scrape(url="https://acme.com", markdown="Acme Corp is a SaaS company based in the US.", success=True) -> ScrapeResult:
    return ScrapeResult(url=url, success=success, markdown=markdown,
                        title="Acme", meta_description="", scrape_method="browser")


def _mock_groq(response_json: dict) -> MagicMock:
    choice = MagicMock()
    choice.message.content = json.dumps(response_json)
    completion = MagicMock()
    completion.choices = [choice]
    client = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=completion)
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__  = AsyncMock(return_value=False)
    return client


# ─────────────────────────────────────────────────────────────────────────────
# CLEANER UNIT TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestCountryNormalisation(unittest.TestCase):

    def test_us_abbreviation(self):
        self.assertEqual(_clean_country("US"), "United States")

    def test_usa_abbreviation(self):
        self.assertEqual(_clean_country("USA"), "United States")

    def test_full_name_passthrough(self):
        self.assertEqual(_clean_country("United States"), "United States")

    def test_uk_abbreviation(self):
        self.assertEqual(_clean_country("UK"), "United Kingdom")

    def test_city_maps_to_country(self):
        self.assertEqual(_clean_country("london"), "United Kingdom")
        self.assertEqual(_clean_country("San Francisco"), "United States")

    def test_unknown_titlecased(self):
        result = _clean_country("new zealand")
        self.assertEqual(result, "New Zealand")

    def test_none_returns_none(self):
        self.assertIsNone(_clean_country(None))

    def test_uae_alias(self):
        self.assertEqual(_clean_country("UAE"), "United Arab Emirates")

    def test_india_abbreviation(self):
        self.assertEqual(_clean_country("IN"), "India")


class TestIndustryNormalisation(unittest.TestCase):

    def test_exact_canonical_match(self):
        self.assertEqual(_clean_industry("Software / SaaS"), "Software / SaaS")

    def test_ai_keyword(self):
        self.assertEqual(_clean_industry("Artificial Intelligence"), Industry.AI_ML.value)

    def test_ml_keyword(self):
        self.assertEqual(_clean_industry("Machine Learning platform"), Industry.AI_ML.value)

    def test_developer_tools_keyword(self):
        self.assertEqual(_clean_industry("Developer Tools & CI/CD"), Industry.DEVELOPER_TOOLS.value)

    def test_fintech_keyword(self):
        self.assertEqual(_clean_industry("Online Payment Processing"), Industry.FINTECH.value)

    def test_healthtech_keyword(self):
        self.assertEqual(_clean_industry("Medical SaaS"), Industry.HEALTHTECH.value)

    def test_ecommerce_keyword(self):
        self.assertEqual(_clean_industry("E-commerce marketplace"), Industry.ECOMMERCE.value)

    def test_cloud_keyword(self):
        self.assertEqual(_clean_industry("Cloud Infrastructure"), Industry.CLOUD.value)

    def test_unrecognised_returns_none(self):
        self.assertIsNone(_clean_industry("XYZ Unrecognisable Sector 9000"))

    def test_none_returns_none(self):
        self.assertIsNone(_clean_industry(None))

    def test_case_insensitive(self):
        self.assertEqual(_clean_industry("ARTIFICIAL INTELLIGENCE"), Industry.AI_ML.value)


class TestDescriptionCleaning(unittest.TestCase):

    def test_strips_cookie_text(self):
        desc = "Acme builds great software. We use cookies. Accept all cookies to continue."
        result = _clean_description(desc)
        self.assertNotIn("cookie", result.lower())

    def test_strips_copyright(self):
        desc = "Acme Corp builds great software. © 2024 All rights reserved."
        result = _clean_description(desc)
        self.assertNotIn("©", result)

    def test_strips_cta_phrases(self):
        desc = "Best SaaS platform. Get started today. Sign up for free."
        result = _clean_description(desc)
        self.assertNotIn("Get started today", result)

    def test_collapses_whitespace(self):
        desc = "Acme   Corp   builds   things."
        result = _clean_description(desc)
        self.assertNotIn("   ", result)

    def test_truncates_at_sentence_boundary(self):
        desc = ("Great company. " * 40).strip()   # ~600 chars
        result = _clean_description(desc)
        self.assertLessEqual(len(result), 510)
        self.assertTrue(result.endswith(".") or result.endswith("…"))

    def test_short_result_returns_none(self):
        self.assertIsNone(_clean_description("Hi."))

    def test_none_returns_none(self):
        self.assertIsNone(_clean_description(None))

    def test_strips_stray_urls(self):
        desc = "Visit us at https://example.com/tracking?id=123 for more info about our products."
        result = _clean_description(desc)
        self.assertNotIn("https://", result)


class TestWebsiteCleaning(unittest.TestCase):

    def test_adds_https(self):
        self.assertEqual(_clean_website("example.com"), "https://example.com")

    def test_strips_trailing_slash(self):
        self.assertEqual(_clean_website("https://example.com/"), "https://example.com")

    def test_strips_query_string(self):
        result = _clean_website("https://example.com/page?utm_source=google")
        self.assertNotIn("utm_source", result)

    def test_strips_fragment(self):
        result = _clean_website("https://example.com/page#section")
        self.assertNotIn("#section", result)

    def test_none_returns_none(self):
        self.assertIsNone(_clean_website(None))

    def test_already_clean_passthrough(self):
        self.assertEqual(_clean_website("https://stripe.com"), "https://stripe.com")


class TestEmployeeSizeCleaning(unittest.TestCase):

    def test_already_canonical(self):
        self.assertEqual(_clean_employee_size("51-200"), "51-200")

    def test_free_text_number(self):
        self.assertEqual(_clean_employee_size("about 75 employees"), "51-200")

    def test_large_number(self):
        self.assertEqual(_clean_employee_size("12,000 staff"), "10000+")

    def test_solo(self):
        self.assertEqual(_clean_employee_size("1"), "1")

    def test_none_returns_none(self):
        self.assertIsNone(_clean_employee_size(None))

    def test_unreadable_returns_none(self):
        self.assertIsNone(_clean_employee_size("many people"))

    def test_tilde_prefix(self):
        self.assertEqual(_clean_employee_size("~500"), "201-500")


class TestCompanyNameCleaning(unittest.TestCase):

    def test_strips_pipe_suffix(self):
        self.assertEqual(_clean_company_name("OpenAI | Home"), "OpenAI")

    def test_strips_dash_suffix(self):
        self.assertEqual(_clean_company_name("GitHub — Build software better"), "GitHub")

    def test_strips_inc_suffix(self):
        self.assertEqual(_clean_company_name("Stripe, Inc."), "Stripe")

    def test_clean_name_passthrough(self):
        self.assertEqual(_clean_company_name("Anthropic"), "Anthropic")

    def test_none_returns_none(self):
        self.assertIsNone(_clean_company_name(None))


class TestLowConfidenceDetection(unittest.TestCase):

    def test_flags_low_confidence_field(self):
        rec = _record(confidence={"country": 0.4, "industry": 0.9})
        flagged = _low_confidence_fields(rec)
        self.assertIn("country", flagged)
        self.assertNotIn("industry", flagged)

    def test_no_flags_when_all_high(self):
        rec = _record(confidence={"company_name": 0.99, "industry": 0.85})
        self.assertEqual(_low_confidence_fields(rec), [])

    def test_none_field_not_flagged_as_low_conf(self):
        # None fields → already in missing_fields, not low-confidence
        rec = _record(country=None, confidence={"country": 0.3})
        flagged = _low_confidence_fields(rec)
        self.assertNotIn("country", flagged)   # it's None, not low-conf


class TestCleanFull(unittest.TestCase):

    def test_complete_record_no_enrichment_needed(self):
        rec = _record()
        result = clean(rec)
        self.assertIsInstance(result, CleanResult)
        self.assertEqual(result.fields_to_enrich, [])

    def test_missing_fields_flagged_for_enrichment(self):
        rec = _record(country=None, industry=None)
        result = clean(rec)
        self.assertIn("country", result.fields_to_enrich)
        self.assertIn("industry", result.fields_to_enrich)

    def test_country_alias_cleaned(self):
        rec = _record(country="USA")
        result = clean(rec)
        self.assertEqual(result.record.country, "United States")
        self.assertTrue(any("country" in c for c in result.changes))

    def test_industry_normalised(self):
        rec = _record(industry="artificial intelligence startup")
        result = clean(rec)
        self.assertEqual(result.record.industry, Industry.AI_ML.value)

    def test_url_cleaned(self):
        rec = _record(website="github.com/")
        result = clean(rec)
        self.assertEqual(result.record.website, "https://github.com")

    def test_low_confidence_added_to_enrich_list(self):
        rec = _record(confidence={"country": 0.3})
        result = clean(rec)
        self.assertIn("country", result.fields_to_enrich)

    def test_clean_batch_processes_all(self):
        records = [_record(), _record(country=None), _record(industry="AI")]
        results = clean_batch(records)
        self.assertEqual(len(results), 3)
        self.assertIsInstance(results[0], CleanResult)

    def test_unrecognised_industry_flagged_for_enrichment(self):
        rec = _record(industry="Quantum Nanotech Wizardry")
        result = clean(rec)
        # Industry unmappable → becomes None → flagged
        self.assertIsNone(result.record.industry)
        self.assertIn("industry", result.fields_to_enrich)


# ─────────────────────────────────────────────────────────────────────────────
# ENRICHER UNIT TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestEnricherPrompt(unittest.TestCase):

    def _make_clean_result(self, **kwargs) -> CleanResult:
        rec = _record(**kwargs)
        return CleanResult(
            record=rec,
            changes=[],
            fields_to_enrich=[f for f in ["company_name","industry","description","country","employee_size"]
                               if getattr(rec, f) is None],
        )

    def test_prompt_includes_known_fields(self):
        cr = self._make_clean_result(country=None)
        prompt = _build_enrichment_prompt(cr, _scrape(), ["country"])
        self.assertIn("Acme Corp", prompt)     # known company_name included as context
        self.assertIn("country", prompt)

    def test_prompt_only_asks_for_missing(self):
        cr = self._make_clean_result(country=None, industry=None)
        prompt = _build_enrichment_prompt(cr, _scrape(), ["country", "industry"])
        self.assertIn("country", prompt)
        self.assertIn("industry", prompt)
        # Should not re-ask for fields we already have
        self.assertNotIn('"company_name":', prompt.split("MISSING")[1])

    def test_prompt_includes_scrape_content(self):
        sr = _scrape(markdown="Founded in Austin Texas. 500 employees. B2B SaaS.")
        cr = self._make_clean_result(country=None)
        prompt = _build_enrichment_prompt(cr, sr, ["country"])
        self.assertIn("Austin Texas", prompt)

    def test_industry_prompt_includes_enum_list(self):
        cr = self._make_clean_result(industry=None)
        prompt = _build_enrichment_prompt(cr, _scrape(), ["industry"])
        self.assertIn("Software / SaaS", prompt)
        self.assertIn("AI / Machine Learning", prompt)

    def test_employee_prompt_includes_band_list(self):
        cr = self._make_clean_result(employee_size=None)
        prompt = _build_enrichment_prompt(cr, _scrape(), ["employee_size"])
        self.assertIn("51-200", prompt)
        self.assertIn("10000+", prompt)


class TestParseEnrichment(unittest.TestCase):

    def test_parses_valid_json(self):
        raw = json.dumps({"country": "Germany", "industry": "Software / SaaS"})
        result = _parse_enrichment(raw, ["country", "industry"])
        self.assertEqual(result["country"], "Germany")
        self.assertEqual(result["industry"], "Software / SaaS")

    def test_strips_markdown_fences(self):
        raw = "```json\n{\"country\": \"France\"}\n```"
        result = _parse_enrichment(raw, ["country"])
        self.assertEqual(result["country"], "France")

    def test_null_string_becomes_none(self):
        raw = json.dumps({"country": "null"})
        result = _parse_enrichment(raw, ["country"])
        self.assertIsNone(result.get("country"))

    def test_only_returns_requested_fields(self):
        raw = json.dumps({"country": "India", "employee_size": "51-200", "surprise": "extra"})
        result = _parse_enrichment(raw, ["country"])
        self.assertIn("country", result)
        self.assertNotIn("surprise", result)
        self.assertNotIn("employee_size", result)

    def test_empty_response_returns_empty_dict(self):
        self.assertEqual(_parse_enrichment("", ["country"]), {})

    def test_malformed_json_returns_empty_dict(self):
        self.assertEqual(_parse_enrichment("{not json!!", ["country"]), {})


class TestMerge(unittest.TestCase):

    def test_fills_missing_field(self):
        rec = _record(country=None)
        new_rec, filled = _merge(rec, {"country": "Canada"})
        self.assertEqual(new_rec.country, "Canada")
        self.assertIn("country", filled)

    def test_does_not_overwrite_existing(self):
        rec = _record(country="United States")
        new_rec, filled = _merge(rec, {"country": "Canada"})
        self.assertEqual(new_rec.country, "United States")   # unchanged
        self.assertNotIn("country", filled)

    def test_none_value_not_written(self):
        rec = _record(country=None)
        new_rec, filled = _merge(rec, {"country": None})
        self.assertIsNone(new_rec.country)
        self.assertNotIn("country", filled)

    def test_multiple_fields_filled(self):
        rec = _record(country=None, industry=None)
        new_rec, filled = _merge(rec, {"country": "Japan", "industry": "Software / SaaS"})
        self.assertEqual(new_rec.country, "Japan")
        self.assertEqual(new_rec.industry, "Software / SaaS")
        self.assertEqual(sorted(filled), ["country", "industry"])

    def test_empty_dict_changes_nothing(self):
        rec = _record()
        new_rec, filled = _merge(rec, {})
        self.assertEqual(filled, [])
        self.assertEqual(new_rec.company_name, rec.company_name)


class TestEnricherMocked(unittest.IsolatedAsyncioTestCase):

    def _clean_result(self, **kwargs) -> CleanResult:
        rec = _record(**kwargs)
        return CleanResult(
            record=rec,
            changes=[],
            fields_to_enrich=[f for f in ["company_name","industry","description","country","employee_size"]
                               if getattr(rec, f) is None],
        )

    async def test_complete_record_skips_api(self):
        cr     = self._clean_result()    # all fields present
        sr     = _scrape()
        client = _mock_groq({})

        result = await enrich_one(client, cr, sr)

        self.assertIsInstance(result, EnrichedRecord)
        self.assertFalse(result.ai_enriched)
        client.chat.completions.create.assert_not_called()

    async def test_missing_field_triggers_api(self):
        cr     = self._clean_result(country=None)
        sr     = _scrape()
        client = _mock_groq({"country": "United Kingdom"})

        result = await enrich_one(client, cr, sr)

        self.assertEqual(result.country, "United Kingdom")
        self.assertTrue(result.ai_enriched)
        client.chat.completions.create.assert_called_once()

    async def test_null_from_api_uses_default(self):
        cr     = self._clean_result(country=None)
        sr     = _scrape()
        client = _mock_groq({"country": None})

        result = await enrich_one(client, cr, sr)

        # AI returned null → EnrichedRecord default kicks in
        self.assertEqual(result.country, "Unknown")
        self.assertFalse(result.ai_enriched)

    async def test_api_error_still_returns_enriched_record(self):
        cr     = self._clean_result(country=None)
        sr     = _scrape()
        client = MagicMock()
        client.chat.completions.create = AsyncMock(side_effect=Exception("Groq down"))
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__  = AsyncMock(return_value=False)

        result = await enrich_one(client, cr, sr)

        self.assertIsInstance(result, EnrichedRecord)
        self.assertEqual(result.country, "Unknown")   # default
        self.assertFalse(result.ai_enriched)

    async def test_enriched_record_has_correct_metadata(self):
        # Use a record without a preset website so scrape URL is used
        cr     = self._clean_result(country=None, website=None)
        sr     = ScrapeResult(url="https://test.io", success=True, markdown="test", title="", meta_description="", scrape_method="httpx")
        client = _mock_groq({"country": "Australia"})

        result = await enrich_one(client, cr, sr)

        self.assertEqual(result.website, "https://test.io")
        self.assertEqual(result.scrape_method, "httpx")
        self.assertTrue(result.ai_enriched)

    async def test_enrich_all_batch(self):
        cr1 = self._clean_result(country=None)
        cr2 = self._clean_result()             # complete — no API call needed

        # Mock returns country for the first call
        choice = MagicMock()
        choice.message.content = json.dumps({"country": "Brazil"})
        completion = MagicMock()
        completion.choices = [choice]
        client = MagicMock()
        client.chat.completions.create = AsyncMock(return_value=completion)
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__  = AsyncMock(return_value=False)

        with unittest.mock.patch("enricher.AsyncGroq", return_value=client):
            results = await enrich_all(
                [(cr1, _scrape()), (cr2, _scrape())],
                api_key="fake-key",
                concurrency=2,
            )

        self.assertEqual(len(results), 2)
        self.assertIsInstance(results[0], EnrichedRecord)
        self.assertIsInstance(results[1], EnrichedRecord)


# ─────────────────────────────────────────────────────────────────────────────
# INTEGRATION — real Groq calls (skipped without API key)
# ─────────────────────────────────────────────────────────────────────────────

GROQ_KEY = os.environ.get("GROQ_API_KEY")


@unittest.skipUnless(GROQ_KEY, "GROQ_API_KEY not set — skipping integration tests")
class TestEnricherIntegration(unittest.IsolatedAsyncioTestCase):

    async def test_enriches_missing_country(self):
        from groq import AsyncGroq

        rec = CompanyRecord(
            company_name="Stripe",
            industry="Fintech",
            description="Stripe is a payment infrastructure company.",
            country=None,       # ← missing, should be filled
            employee_size="1001-5000",
            website="https://stripe.com",
            confidence={},
        )
        cr = CleanResult(record=rec, changes=[], fields_to_enrich=["country"])
        sr = ScrapeResult(
            url="https://stripe.com", success=True, scrape_method="test",
            markdown=(
                "Stripe is a financial infrastructure platform headquartered in "
                "South San Francisco, California, United States. Founded in 2010 "
                "by Patrick and John Collison."
            ),
            title="Stripe", meta_description="",
        )

        async with AsyncGroq(api_key=GROQ_KEY) as client:
            result = await enrich_one(client, cr, sr)

        self.assertIsInstance(result, EnrichedRecord)
        self.assertIn("United States", result.country)
        self.assertTrue(result.ai_enriched)

        print(f"\n  ── Integration result ──")
        print(f"  country (filled): {result.country}")
        print(f"  ai_enriched     : {result.ai_enriched}")
        print(f"  completeness    : {result.completeness}%")

    async def test_complete_record_not_enriched(self):
        from groq import AsyncGroq

        rec = _record()
        cr  = CleanResult(record=rec, changes=[], fields_to_enrich=[])
        sr  = _scrape()

        call_count = 0
        async with AsyncGroq(api_key=GROQ_KEY) as client:
            result = await enrich_one(client, cr, sr)

        self.assertFalse(result.ai_enriched)
        print(f"\n  ── Complete record skipped enrichment correctly ──")


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import unittest.mock   # ensure patch is importable in mock tests

    mode = sys.argv[1] if len(sys.argv) > 1 else "all"

    loader = unittest.TestLoader()

    unit_classes = [
        TestCountryNormalisation, TestIndustryNormalisation,
        TestDescriptionCleaning, TestWebsiteCleaning,
        TestEmployeeSizeCleaning, TestCompanyNameCleaning,
        TestLowConfidenceDetection, TestCleanFull,
        TestEnricherPrompt, TestParseEnrichment, TestMerge,
        TestEnricherMocked,
    ]

    if mode == "unit":
        suite = unittest.TestSuite()
        for cls in unit_classes:
            suite.addTests(loader.loadTestsFromTestCase(cls))
    elif mode == "integration":
        suite = loader.loadTestsFromTestCase(TestEnricherIntegration)
    else:
        suite = unittest.TestSuite()
        for cls in unit_classes:
            suite.addTests(loader.loadTestsFromTestCase(cls))
        suite.addTests(loader.loadTestsFromTestCase(TestEnricherIntegration))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)