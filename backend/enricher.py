"""
enricher.py — AI-powered field enrichment for the Data Enrichment Pipeline

Takes a CleanResult (from cleaner.py) and uses Groq to fill only the
fields that are still missing or low-confidence after cleaning.

Responsibilities:
  - Build targeted prompts for only the missing fields
  - Merge AI-filled values back into the CompanyRecord
  - Produce a final EnrichedRecord ready for database.py
  - Skip enrichment entirely if record is already complete
  - Fall back to general knowledge when scraped content is thin

Does NOT clean / normalise values — that's cleaner.py's job.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from typing import Optional

from groq import AsyncGroq

from cleaner import CleanResult
from models import CompanyRecord, EnrichedRecord, Industry, EmployeeSize
from scraper import ScrapeResult

log = logging.getLogger("enricher")

# ── Config ────────────────────────────────────────────────────────────────────

MODEL       = "llama-3.3-70b-versatile"
MAX_TOKENS  = 256
TEMPERATURE = 0.1
MAX_RETRIES = 2

INDUSTRY_LIST  = "\n".join(f"  - {i.value}" for i in Industry if i != Industry.UNKNOWN)
EMPLOYEE_BANDS = "\n".join(f"  - {e.value}" for e in EmployeeSize if e != EmployeeSize.UNKNOWN)

# Threshold: if scraped text is shorter than this, the model almost certainly
# lacks enough evidence → switch to knowledge-based prompt
THIN_CONTENT_THRESHOLD = 500


# ── Prompt builders ───────────────────────────────────────────────────────────

def _field_specs(fields_needed: list[str]) -> str:
    """Build the field spec block for a given list of fields."""
    specs: list[str] = []
    for f in fields_needed:
        if f == "company_name":
            specs.append('- "company_name": Official company name (string or null)')
        elif f == "industry":
            specs.append(
                f'- "industry": One value from this list ONLY:\n{INDUSTRY_LIST}\n  or null'
            )
        elif f == "description":
            specs.append(
                '- "description": 2-3 sentence summary of what the company does (string or null)'
            )
        elif f == "country":
            specs.append(
                '- "country": Full country name where company is headquartered (string or null)'
            )
        elif f == "employee_size":
            specs.append(
                f'- "employee_size": One value from this list ONLY:\n{EMPLOYEE_BANDS}\n  or null'
            )
    return "\n".join(specs)


def _context_block(record: CompanyRecord) -> str:
    """Summarise what we already know — given to model as context."""
    known: list[str] = []
    if record.company_name:  known.append(f"Company name: {record.company_name}")
    if record.industry:      known.append(f"Industry: {record.industry}")
    if record.country:       known.append(f"Country: {record.country}")
    if record.description:   known.append(f"Description: {record.description}")
    if record.employee_size: known.append(f"Employee size: {record.employee_size}")
    if record.website:       known.append(f"Website: {record.website}")
    return "\n".join(known) if known else "(nothing known yet)"


def _build_evidence_prompt(
    clean_result: CleanResult,
    scrape_result: ScrapeResult,
    fields_needed: list[str],
) -> str:
    """
    Prompt when scraped content is rich enough to serve as evidence.
    Instructs the model to extract ONLY from the provided content.
    """
    return f"""You are a company data enrichment engine. Fill in the missing fields for a company.

WHAT WE ALREADY KNOW:
{_context_block(clean_result.record)}

WEBSITE: {scrape_result.url}
PAGE TITLE: {scrape_result.title or '(none)'}

---WEBSITE CONTENT (evidence)---
{scrape_result.markdown[:5000]}
---END CONTENT---

MISSING FIELDS TO FILL:
{_field_specs(fields_needed)}

Rules:
- Extract ONLY from the content above — do not guess
- Use null if you cannot determine a value from the content
- For industry and employee_size, use exact values from the lists provided
- Return ONLY valid JSON with the missing field keys. No explanation, no markdown fences.

JSON:"""


def _build_knowledge_prompt(
    clean_result: CleanResult,
    scrape_result: ScrapeResult,
    fields_needed: list[str],
) -> str:
    """
    Prompt when scraped content is thin (product page, Cloudflare block, etc).
    Allows the model to use general world knowledge about the company.
    Used as fallback after evidence prompt returns nothing useful.
    """
    return f"""You are a company data enrichment engine with access to general world knowledge.

COMPANY DETAILS:
{_context_block(clean_result.record)}
Website: {scrape_result.url}

The website content was not informative enough to extract the missing fields.
Use your general knowledge about this company to fill in what you know.

MISSING FIELDS TO FILL:
{_field_specs(fields_needed)}

Rules:
- Use your knowledge about well-known companies (e.g. WhatsApp → Meta → United States)
- Use null only if you genuinely do not know — do not fabricate
- For industry and employee_size, use exact values from the lists provided
- Return ONLY valid JSON with the missing field keys. No explanation, no markdown fences.

JSON:"""


# ── Response parser ───────────────────────────────────────────────────────────

def _parse_enrichment(raw: str, fields_needed: list[str]) -> dict[str, Optional[str]]:
    """
    Parse enrichment JSON response → dict of {field: value | None}.

    FIX: previously discarded null values entirely (returning {}).
    Now we correctly include None for null fields so _merge knows
    the model tried but found nothing (vs. the field not being in response).
    """
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.MULTILINE).strip()
    m = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not m:
        return {}
    try:
        data = json.loads(m.group())
    except json.JSONDecodeError:
        try:
            fixed = m.group().replace("'", '"').replace("None", "null")
            data = json.loads(fixed)
        except Exception:
            return {}

    result: dict[str, Optional[str]] = {}
    for f in fields_needed:
        if f not in data:
            continue
        v = data[f]
        # Normalise placeholder strings → None
        if isinstance(v, str) and v.strip().lower() in {"", "null", "none", "n/a", "unknown", "not found"}:
            v = None
        elif not isinstance(v, str):
            v = None
        # Include both filled values AND explicit nulls (was the bug: only str values were kept)
        result[f] = v

    return result


def _has_useful_data(parsed: dict[str, Optional[str]]) -> bool:
    """Return True if the parsed response has at least one non-None value."""
    return any(v is not None for v in parsed.values())


# ── Merge helper ──────────────────────────────────────────────────────────────

def _merge(record: CompanyRecord, filled: dict[str, Optional[str]]) -> tuple[CompanyRecord, list[str]]:
    """
    Merge AI-filled values into a CompanyRecord.
    Returns (new_record, list_of_fields_actually_filled).
    Only writes non-None values into fields that were previously None.
    """
    data = record.model_dump()
    actually_filled: list[str] = []

    for field, value in filled.items():
        if value and not data.get(field):
            data[field] = value
            actually_filled.append(field)

    return CompanyRecord(**data), actually_filled


# ── Core enricher ─────────────────────────────────────────────────────────────

async def enrich_one(
    client: AsyncGroq,
    clean_result: CleanResult,
    scrape_result: ScrapeResult,
) -> EnrichedRecord:
    """
    Enrich a single CleanResult → EnrichedRecord.

    Strategy:
      1. If complete → promote directly, no API call.
      2. Try evidence-based prompt (extract from scraped content).
      3. If that yields nothing → try knowledge-based prompt (use LLM world knowledge).
      4. Merge whatever was found; apply defaults for anything still missing.
    """
    record  = clean_result.record
    to_fill = clean_result.fields_to_enrich
    website = scrape_result.url
    method  = scrape_result.scrape_method

    # ── Already complete ──────────────────────────────────────────────────────
    if not to_fill or record.is_complete():
        log.info("  ✓ %s — complete, no enrichment needed", website)
        return EnrichedRecord.from_company_record(
            record, website=website, scrape_method=method, ai_enriched=False
        )

    log.info("  ↑ Enriching %s — filling: %s", website, to_fill)

    content_is_thin = len(scrape_result.markdown or "") < THIN_CONTENT_THRESHOLD

    # ── Phase 1: evidence-based prompt ───────────────────────────────────────
    filled_data: dict[str, Optional[str]] = {}
    used_knowledge = False

    if not content_is_thin:
        filled_data = await _call_groq(
            client, website,
            _build_evidence_prompt(clean_result, scrape_result, to_fill),
        )

    # ── Phase 2: knowledge-based fallback ────────────────────────────────────
    # Trigger if: content was thin OR evidence prompt found nothing useful
    if content_is_thin or not _has_useful_data(filled_data):
        if not content_is_thin:
            log.info("  ↩ Evidence prompt empty — trying knowledge fallback for %s", website)
        else:
            log.info("  ↩ Thin content — using knowledge prompt for %s", website)

        knowledge_data = await _call_groq(
            client, website,
            _build_knowledge_prompt(clean_result, scrape_result, to_fill),
        )
        if _has_useful_data(knowledge_data):
            filled_data = knowledge_data
            used_knowledge = True

    # ── Merge and finalise ────────────────────────────────────────────────────
    enriched_record, actually_filled = _merge(record, filled_data)
    ai_enriched = bool(actually_filled)

    if actually_filled:
        source = "knowledge" if used_knowledge else "evidence"
        log.info("  ✓ %s — AI filled [%s]: %s", website, source, actually_filled)
    else:
        log.warning("  ⚠ %s — both prompts returned nothing useful", website)

    return EnrichedRecord.from_company_record(
        enriched_record,
        website=website,
        scrape_method=method,
        ai_enriched=ai_enriched,
    )


async def _call_groq(
    client: AsyncGroq,
    url: str,
    prompt: str,
) -> dict[str, Optional[str]]:
    """Fire a single Groq call with retries. Returns parsed dict (may be empty)."""
    # Extract fields_needed from prompt to pass to parser
    fields_needed = ["company_name", "industry", "description", "country", "employee_size"]

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = await client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
            )
            raw = response.choices[0].message.content or ""
            result = _parse_enrichment(raw, fields_needed)
            if result:   # got at least one key back (even if value is None)
                return result
            log.warning("    ⚠ Attempt %d parse returned empty for %s", attempt, url)
        except Exception as exc:
            log.error("    ✗ Groq error attempt %d for %s: %s", attempt, url, exc)
            if attempt < MAX_RETRIES:
                await asyncio.sleep(1.5)
    return {}


# ── Batch enrichment ──────────────────────────────────────────────────────────

async def enrich_all(
    pairs: list[tuple[CleanResult, ScrapeResult]],
    api_key: Optional[str] = None,
    concurrency: int = 3,
) -> list[EnrichedRecord]:
    """
    Enrich a batch of (CleanResult, ScrapeResult) pairs concurrently.

    Args:
        pairs:       Zipped output of cleaner + scraper.
        api_key:     Groq API key (falls back to GROQ_API_KEY env var).
        concurrency: Max parallel Groq calls (default 3).

    Returns:
        List[EnrichedRecord] in the same order as input.
    """
    if not pairs:
        return []

    key = api_key or os.environ.get("GROQ_API_KEY")
    if not key:
        raise ValueError("Groq API key required — set GROQ_API_KEY or pass api_key=")

    sem     = asyncio.Semaphore(concurrency)
    results: list[Optional[EnrichedRecord]] = [None] * len(pairs)

    async def _bounded(idx: int, clean_res: CleanResult, scrape_res: ScrapeResult) -> None:
        async with sem:
            results[idx] = await enrich_one(client, clean_res, scrape_res)

    async with AsyncGroq(api_key=key) as client:
        await asyncio.gather(*[
            _bounded(i, cr, sr) for i, (cr, sr) in enumerate(pairs)
        ])

    return results  # type: ignore[return-value]


def enrich_all_sync(
    pairs: list[tuple[CleanResult, ScrapeResult]],
    api_key: Optional[str] = None,
    concurrency: int = 3,
) -> list[EnrichedRecord]:
    """Synchronous wrapper."""
    return asyncio.run(enrich_all(pairs, api_key, concurrency))