"""
cleaner.py — Data cleaning layer for the Data Enrichment Pipeline

Takes a raw CompanyRecord (from extractor.py) and applies deterministic
rule-based cleaning before AI enrichment. Fast, zero API cost, zero
network calls.

Responsibilities:
  - Normalise country names (aliases → canonical)
  - Normalise industry labels (free-text → Industry enum value)
  - Clean description text (strip boilerplate, fix whitespace)
  - Canonicalise website URLs
  - Flag low-confidence fields so enricher knows what to target

Does NOT fill missing fields — that's enricher.py's job.
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field
from typing import Optional
from urllib.parse import urlparse, urlunparse

from models import CompanyRecord, Industry, EmployeeSize

log = logging.getLogger("cleaner")


# ── Country normalisation map ─────────────────────────────────────────────────
# alias (lowercase) → canonical full name

COUNTRY_ALIASES: dict[str, str] = {
    # United States
    "us": "United States", "usa": "United States", "u.s.": "United States",
    "u.s.a.": "United States", "united states of america": "United States",
    "america": "United States", "san francisco": "United States",
    "new york": "United States", "california": "United States",
    "seattle": "United States", "boston": "United States",
    # United Kingdom
    "uk": "United Kingdom", "u.k.": "United Kingdom", "great britain": "United Kingdom",
    "britain": "United Kingdom", "england": "United Kingdom",
    "london": "United Kingdom",
    # Common EU
    "de": "Germany", "deutschland": "Germany",
    "fr": "France",
    "nl": "Netherlands", "the netherlands": "Netherlands",
    "se": "Sweden",
    "ch": "Switzerland",
    "be": "Belgium",
    # Asia-Pacific
    "cn": "China", "prc": "China",
    "jp": "Japan",
    "kr": "South Korea", "republic of korea": "South Korea",
    "in": "India",
    "au": "Australia",
    "sg": "Singapore",
    # Americas
    "ca": "Canada",
    "br": "Brazil",
    # Middle East / Africa
    "il": "Israel",
    "ae": "United Arab Emirates", "uae": "United Arab Emirates",
}

# ── Industry normalisation map ────────────────────────────────────────────────
# keyword pattern (lowercase) → canonical Industry value
# Checked in order — put more specific patterns first

INDUSTRY_PATTERNS: list[tuple[str, str]] = [
    # AI / ML
    (r"artificial intelligence|machine learning|\bai\b|\bml\b|deep learning|llm|generative",
     Industry.AI_ML.value),
    # Developer Tools
    (r"developer tool|devtool|dev tool|version control|code hosting|ide|ci/cd|devops",
     Industry.DEVELOPER_TOOLS.value),
    # Cybersecurity
    (r"cyber|security|infosec|firewall|siem|zero.?trust|penetration|antivirus",
     Industry.CYBERSECURITY.value),
    # Cloud / Infrastructure
    (r"cloud|infrastructure|iaas|paas|kubernetes|serverless|data center|cdn",
     Industry.CLOUD.value),
    # Fintech
    (r"fintech|payment|banking|finance|lending|insurance|crypto|blockchain|trading",
     Industry.FINTECH.value),
    # Healthtech
    (r"health|medical|clinical|pharma|biomedical|telemedicine|ehr|hospital",
     Industry.HEALTHTECH.value),
    # Biotech
    (r"biotech|genomics|life science|drug discovery|therapeutics|biopharma",
     Industry.BIOTECH.value),
    # E-commerce
    (r"e.?commerce|ecommerce|retail|marketplace|shopping|store|consumer goods",
     Industry.ECOMMERCE.value),
    # Education
    (r"education|edtech|learning|online course|tutoring|university|school",
     Industry.EDUCATION.value),
    # Media
    (r"media|content|news|publishing|streaming|podcast|entertainment|social",
     Industry.MEDIA.value),
    # Energy
    (r"energy|cleantech|renewable|solar|wind|climate|sustainability|ev|battery",
     Industry.ENERGY.value),
    # Logistics
    (r"logistics|supply chain|shipping|freight|warehouse|delivery|transport",
     Industry.LOGISTICS.value),
    # Hardware / Semiconductors
    (r"hardware|semiconductor|chip|processor|robotics|iot|embedded|electronics",
     Industry.HARDWARE.value),
    # Consulting
    (r"consulting|professional service|advisory|outsourcing|staffing|recruitment",
     Industry.CONSULTING.value),
    # Software / SaaS (broad — keep last before OTHER)
    (r"software|saas|platform|application|app|api|tool|productivity|crm|erp|b2b",
     Industry.SOFTWARE.value),
]

# ── Boilerplate patterns to strip from descriptions ───────────────────────────

_BOILERPLATE: list[re.Pattern] = [
    re.compile(p, re.I) for p in [
        r"cookie[s]?(\s+policy)?",
        r"accept\s+all\s+cookies?",
        r"privacy\s+policy",
        r"terms\s+(of\s+)?(service|use)",
        r"all\s+rights\s+reserved",
        r"copyright\s+©?\s*\d{4}",
        r"subscribe\s+to\s+(our\s+)?newsletter",
        r"sign\s+up\s+(for\s+)?free",
        r"get\s+started\s+today",
        r"learn\s+more",
        r"click\s+here",
        r"read\s+more",
        r"follow\s+us\s+on",
        r"©\s*\d{4}",
        r"\[.*?\]",           # markdown link text leftovers
        r"!\[.*?\]\(.*?\)",   # markdown images
        r"https?://\S+",      # stray URLs
    ]
]


# ── Return type ───────────────────────────────────────────────────────────────

@dataclass
class CleanResult:
    """Output of the cleaner — a cleaned record plus a change log."""
    record: CompanyRecord
    changes: list[str] = field(default_factory=list)   # human-readable changelog
    fields_to_enrich: list[str] = field(default_factory=list)  # still None after cleaning

    def __repr__(self) -> str:
        return (
            f"CleanResult(completeness={self.record.completeness_pct()}% | "
            f"changes={len(self.changes)} | to_enrich={self.fields_to_enrich})"
        )


# ── Individual cleaning functions ─────────────────────────────────────────────

def _clean_company_name(name: Optional[str]) -> Optional[str]:
    if not name:
        return None
    # Strip common suffixes if the name is very long (likely a page title)
    name = re.sub(r"\s*[|–—-]\s*.+$", "", name).strip()
    # Strip trailing legal suffixes that are overly verbose
    name = re.sub(r",?\s+(Inc\.?|LLC\.?|Ltd\.?|Corp\.?|Co\.?|GmbH|S\.A\.)$", "", name, flags=re.I).strip()
    return name or None


def _clean_country(country: Optional[str]) -> Optional[str]:
    if not country:
        return None
    key = country.strip().lower()
    # Direct alias lookup
    if key in COUNTRY_ALIASES:
        return COUNTRY_ALIASES[key]
    # Title-case the input as a fallback (handles "united states" → "United States")
    return country.strip().title()


def _clean_industry(industry: Optional[str]) -> Optional[str]:
    if not industry:
        return None

    # Direct enum value match (already canonical)
    canonical_values = {i.value.lower(): i.value for i in Industry}
    if industry.strip().lower() in canonical_values:
        return canonical_values[industry.strip().lower()]

    # Keyword pattern matching
    text = industry.lower()
    for pattern, canonical in INDUSTRY_PATTERNS:
        if re.search(pattern, text):
            return canonical

    # Couldn't map — return None so enricher can try
    log.debug("Could not map industry %r to a canonical value", industry)
    return None


def _clean_description(desc: Optional[str]) -> Optional[str]:
    if not desc:
        return None

    # Strip boilerplate phrases
    for pat in _BOILERPLATE:
        desc = pat.sub(" ", desc)

    # Collapse whitespace
    desc = re.sub(r"\s{2,}", " ", desc).strip()

    # Trim to 500 chars at a sentence boundary
    if len(desc) > 500:
        truncated = desc[:500]
        last_period = truncated.rfind(".")
        if last_period > 200:
            desc = truncated[: last_period + 1]
        else:
            desc = truncated.rstrip() + "…"

    return desc if len(desc) >= 20 else None


def _clean_website(url: Optional[str]) -> Optional[str]:
    if not url:
        return None
    url = url.strip()
    if not url.startswith("http"):
        url = "https://" + url
    try:
        parsed = urlparse(url)
        # Keep only scheme + netloc + path, drop query/fragment
        clean = urlunparse((parsed.scheme, parsed.netloc, parsed.path.rstrip("/"), "", "", ""))
        return clean
    except Exception:
        return url


def _clean_employee_size(emp: Optional[str]) -> Optional[str]:
    """Map free-text to canonical EmployeeSize band if not already mapped."""
    if not emp:
        return None

    # Already a valid enum value?
    valid = {e.value for e in EmployeeSize}
    if emp in valid:
        return emp

    # Try to extract a number and re-map
    digits = re.sub(r"[,\s]", "", emp)
    m = re.search(r"(\d+)", digits)
    if not m:
        return None   # Unrecognisable — let enricher handle

    n = int(m.group(1))
    if n == 1:         return EmployeeSize.SOLO.value
    if n <= 10:        return EmployeeSize.MICRO.value
    if n <= 50:        return EmployeeSize.SMALL.value
    if n <= 200:       return EmployeeSize.MEDIUM.value
    if n <= 500:       return EmployeeSize.MID_LARGE.value
    if n <= 1_000:     return EmployeeSize.LARGE.value
    if n <= 5_000:     return EmployeeSize.ENTERPRISE.value
    if n <= 10_000:    return EmployeeSize.GIANT.value
    return EmployeeSize.MEGA.value


# ── Low-confidence field detection ───────────────────────────────────────────

CONFIDENCE_THRESHOLD = 0.6   # below this → flag for re-enrichment

def _low_confidence_fields(record: CompanyRecord) -> list[str]:
    """Return fields where AI confidence was below threshold."""
    flagged = []
    for f, score in record.confidence.items():
        if score < CONFIDENCE_THRESHOLD and getattr(record, f, None) is not None:
            flagged.append(f)
    return flagged


# ── Public API ────────────────────────────────────────────────────────────────

def clean(record: CompanyRecord) -> CleanResult:
    """
    Apply all deterministic cleaning rules to a CompanyRecord.

    Returns a CleanResult with:
    - record: the cleaned CompanyRecord (new object, original unchanged)
    - changes: list of what was changed (for logging / audit)
    - fields_to_enrich: fields that are still None OR low-confidence
    """
    changes: list[str] = []
    data = record.model_dump()

    # ── company_name ──────────────────────────────────────────────────────────
    cleaned_name = _clean_company_name(data["company_name"])
    if cleaned_name != data["company_name"]:
        changes.append(f"company_name: {data['company_name']!r} → {cleaned_name!r}")
    data["company_name"] = cleaned_name

    # ── country ───────────────────────────────────────────────────────────────
    cleaned_country = _clean_country(data["country"])
    if cleaned_country != data["country"]:
        changes.append(f"country: {data['country']!r} → {cleaned_country!r}")
    data["country"] = cleaned_country

    # ── industry ──────────────────────────────────────────────────────────────
    cleaned_industry = _clean_industry(data["industry"])
    if cleaned_industry != data["industry"]:
        changes.append(f"industry: {data['industry']!r} → {cleaned_industry!r}")
    data["industry"] = cleaned_industry

    # ── description ───────────────────────────────────────────────────────────
    cleaned_desc = _clean_description(data["description"])
    if cleaned_desc != data["description"]:
        changes.append(
            f"description: cleaned "
            f"({len(data['description'] or '')} → {len(cleaned_desc or '')} chars)"
        )
    data["description"] = cleaned_desc

    # ── website ───────────────────────────────────────────────────────────────
    cleaned_url = _clean_website(data["website"])
    if cleaned_url != data["website"]:
        changes.append(f"website: {data['website']!r} → {cleaned_url!r}")
    data["website"] = cleaned_url

    # ── employee_size ─────────────────────────────────────────────────────────
    cleaned_emp = _clean_employee_size(data["employee_size"])
    if cleaned_emp != data["employee_size"]:
        changes.append(f"employee_size: {data['employee_size']!r} → {cleaned_emp!r}")
    data["employee_size"] = cleaned_emp

    # ── Build cleaned record ──────────────────────────────────────────────────
    cleaned = CompanyRecord(**data)

    # ── Determine what still needs enrichment ─────────────────────────────────
    still_missing  = cleaned.missing_fields()
    low_conf       = _low_confidence_fields(cleaned)
    fields_to_enrich = list(dict.fromkeys(still_missing + low_conf))  # dedup, preserve order

    if changes:
        log.info("Cleaned %s: %d change(s)", record.website or "?", len(changes))
        for c in changes:
            log.debug("  %s", c)

    return CleanResult(
        record=cleaned,
        changes=changes,
        fields_to_enrich=fields_to_enrich,
    )


def clean_batch(records: list[CompanyRecord]) -> list[CleanResult]:
    """Clean a list of CompanyRecords. Synchronous — no I/O."""
    return [clean(r) for r in records]