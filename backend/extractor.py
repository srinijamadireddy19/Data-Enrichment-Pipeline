"""
extractor.py — AI-powered field extractor for the Data Enrichment Pipeline

Takes a ScrapeResult (raw website text) and calls Groq (LLaMA 3.3 70B) to
extract structured company data into a CompanyRecord.

Responsibilities:
  - Build a tight extraction prompt
  - Parse + validate JSON from LLM
  - Attach per-field confidence scores
  - Handle malformed / partial responses gracefully
  - Retry on transient failures

Does NOT enrich missing fields — that's enricher.py.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from typing import Optional

from groq import AsyncGroq

from models import CompanyRecord, Industry
from backend.scraper import ScrapeResult

log = logging.getLogger("extractor")

# ── Config ────────────────────────────────────────────────────────────────────

MODEL        = "llama-3.3-70b-versatile"   # Fast + accurate; free tier friendly
MAX_TOKENS   = 512                          # JSON response is small
TEMPERATURE  = 0.1                          # Low = deterministic extraction
MAX_RETRIES  = 3
RETRY_DELAY  = 2.0                          # seconds between retries

INDUSTRY_LIST = "\n".join(f"  - {i.value}" for i in Industry)


# ── Prompt ────────────────────────────────────────────────────────────────────

def _build_prompt(result: ScrapeResult) -> str:
    """
    Build the extraction prompt from a ScrapeResult.
    Injects title + meta description at the top for signal density.
    """
    header_parts = []
    if result.title:
        header_parts.append(f"Page title: {result.title}")
    if result.meta_description:
        header_parts.append(f"Meta description: {result.meta_description}")
    header_parts.append(f"Website: {result.url}")
    header = "\n".join(header_parts)

    return f"""You are a company data extraction engine. Extract structured information from the website content below.

{header}

---WEBSITE CONTENT---
{result.markdown[:6000]}
---END CONTENT---

Extract the following fields and return ONLY valid JSON. No explanation, no markdown, no code fences.

Fields to extract:
- company_name: Official company name (string or null)
- industry: Best matching industry from this list:
{INDUSTRY_LIST}
- description: 2-3 sentence summary of what the company does (string or null)
- country: Country where company is headquartered (string or null)
- employee_size: Employee count or range, e.g. "500", "1000-5000", "10,000+" (string or null)
- website: Primary website URL (string or null)
- confidence: Object with a score 0.0-1.0 for each field above (how confident you are)

Rules:
- Use null for any field you cannot find — do not guess
- For industry, you MUST pick from the list above or use "Unknown"
- description must be factual, based only on the content provided
- country should be the full country name, e.g. "United States" not "US"

Return JSON only:"""


# ── JSON parser ───────────────────────────────────────────────────────────────

def _parse_response(raw: str, url: str) -> Optional[CompanyRecord]:
    """
    Parse LLM response into a CompanyRecord.
    Handles: bare JSON, JSON wrapped in markdown fences, partial objects.
    """
    # Strip markdown code fences if present
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.MULTILINE)
    cleaned = cleaned.strip()

    # Find first {...} block in case there's preamble text
    m = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not m:
        log.warning("No JSON object found in LLM response for %s", url)
        return None

    try:
        data = json.loads(m.group())
    except json.JSONDecodeError as exc:
        log.warning("JSON parse failed for %s: %s", url, exc)
        # Last resort: try to fix common issues
        try:
            fixed = m.group().replace("'", '"').replace("None", "null").replace("True", "true").replace("False", "false")
            data = json.loads(fixed)
        except Exception:
            return None

    # Pull confidence out before passing to CompanyRecord
    confidence = data.pop("confidence", {})
    if not isinstance(confidence, dict):
        confidence = {}

    # Ensure website falls back to the input URL
    if not data.get("website"):
        data["website"] = url

    try:
        record = CompanyRecord(**data, confidence=confidence)
        return record
    except Exception as exc:
        log.warning("CompanyRecord validation failed for %s: %s", url, exc)
        # Build a partial record from whatever we can salvage
        safe = {k: v for k, v in data.items()
                if k in CompanyRecord.model_fields and isinstance(v, (str, type(None)))}
        safe["website"] = url
        safe["confidence"] = confidence
        try:
            return CompanyRecord(**safe)
        except Exception:
            return None


# ── Core extractor ────────────────────────────────────────────────────────────

async def extract_one(
    client: AsyncGroq,
    result: ScrapeResult,
) -> tuple[ScrapeResult, Optional[CompanyRecord]]:
    """
    Extract a CompanyRecord from a single ScrapeResult.

    Returns (result, record) — record is None if extraction failed completely.
    Always returns the original ScrapeResult so callers can log the URL.
    """
    if not result.success or not result.markdown:
        log.warning("Skipping extraction for failed scrape: %s", result.url)
        return result, None

    prompt = _build_prompt(result)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            log.info("Extracting %s (attempt %d/%d) …", result.url, attempt, MAX_RETRIES)
            t0 = time.time()

            response = await client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
            )

            elapsed  = round(time.time() - t0, 2)
            raw_text = response.choices[0].message.content or ""
            log.info("  ← Groq responded in %.2fs for %s", elapsed, result.url)

            record = _parse_response(raw_text, result.url)

            if record is not None:
                log.info(
                    "  ✓ %s | completeness=%.0f%% | missing=%s",
                    result.url,
                    record.completeness_pct(),
                    record.missing_fields() or "none",
                )
                return result, record

            log.warning("  ⚠ Parsing failed on attempt %d for %s", attempt, result.url)

        except Exception as exc:
            log.error("  ✗ Groq error on attempt %d for %s: %s", attempt, result.url, exc)

        if attempt < MAX_RETRIES:
            await asyncio.sleep(RETRY_DELAY)

    log.error("  ✗ All %d attempts failed for %s", MAX_RETRIES, result.url)
    return result, None


# ── Batch extraction ──────────────────────────────────────────────────────────

async def extract_all(
    scrape_results: list[ScrapeResult],
    api_key: Optional[str] = None,
    concurrency: int = 3,
) -> list[tuple[ScrapeResult, Optional[CompanyRecord]]]:
    """
    Extract CompanyRecords from a batch of ScrapeResults concurrently.

    Args:
        scrape_results: Output from scraper.scrape_urls()
        api_key:        Groq API key (falls back to GROQ_API_KEY env var)
        concurrency:    Max parallel Groq calls (default 3, stay inside rate limits)

    Returns:
        List of (ScrapeResult, CompanyRecord | None) in the same order as input.
    """
    if not scrape_results:
        return []

    key = api_key or os.environ.get("GROQ_API_KEY")
    if not key:
        raise ValueError("Groq API key required — set GROQ_API_KEY env var or pass api_key=")

    sem     = asyncio.Semaphore(concurrency)
    results = [None] * len(scrape_results)

    async def _bounded(idx: int, sr: ScrapeResult) -> None:
        async with sem:
            results[idx] = await extract_one(client, sr)

    async with AsyncGroq(api_key=key) as client:
        await asyncio.gather(*[_bounded(i, sr) for i, sr in enumerate(scrape_results)])

    return results  # type: ignore[return-value]


def extract_all_sync(
    scrape_results: list[ScrapeResult],
    api_key: Optional[str] = None,
    concurrency: int = 3,
) -> list[tuple[ScrapeResult, Optional[CompanyRecord]]]:
    """Synchronous wrapper — use when no event loop is running."""
    return asyncio.run(extract_all(scrape_results, api_key, concurrency))