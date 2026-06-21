"""
models.py — Pydantic schemas for the Data Enrichment Pipeline

Defines the canonical data shapes that flow through every stage:
  ScrapeResult  (scraper.py output)   → already in scraper.py, imported here
  CompanyRecord (extractor output)    → structured company data from AI
  EnrichedRecord (final output)       → cleaned + enriched, ready for DB/CSV

Usage:
    from models import CompanyRecord, EnrichedRecord, EmployeeSize
"""

from __future__ import annotations

import re
import time
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# ── Enums ─────────────────────────────────────────────────────────────────────

class EmployeeSize(str, Enum):
    """Standardised headcount bands."""
    SOLO        = "1"
    MICRO       = "2-10"
    SMALL       = "11-50"
    MEDIUM      = "51-200"
    MID_LARGE   = "201-500"
    LARGE       = "501-1000"
    ENTERPRISE  = "1001-5000"
    GIANT       = "5001-10000"
    MEGA        = "10000+"
    UNKNOWN     = "unknown"


class Industry(str, Enum):
    """Top-level industry buckets."""
    AI_ML           = "AI / Machine Learning"
    SOFTWARE        = "Software / SaaS"
    DEVELOPER_TOOLS = "Developer Tools"
    CLOUD           = "Cloud / Infrastructure"
    CYBERSECURITY   = "Cybersecurity"
    FINTECH         = "Fintech"
    HEALTHTECH      = "Healthtech"
    ECOMMERCE       = "E-commerce"
    MEDIA           = "Media / Content"
    EDUCATION       = "Education / Edtech"
    CONSULTING      = "Consulting / Professional Services"
    HARDWARE        = "Hardware / Semiconductors"
    BIOTECH         = "Biotech / Life Sciences"
    ENERGY          = "Energy / CleanTech"
    LOGISTICS       = "Logistics / Supply Chain"
    OTHER           = "Other"
    UNKNOWN         = "Unknown"


# ── Raw extraction from AI (permissive) ──────────────────────────────────────

class CompanyRecord(BaseModel):
    """
    Direct output from the AI extractor.
    Fields are Optional/lenient — AI may not find everything.
    """
    company_name:  Optional[str] = Field(None, description="Official company name")
    industry:      Optional[str] = Field(None, description="Primary industry or sector")
    description:   Optional[str] = Field(None, description="1-3 sentence company summary")
    country:       Optional[str] = Field(None, description="Country of headquarters")
    employee_size: Optional[str] = Field(None, description="Headcount or employee range")
    website:       Optional[str] = Field(None, description="Primary website URL")

    # Confidence signals (0.0–1.0) — set by extractor, used by enricher
    confidence: dict[str, float] = Field(
        default_factory=dict,
        description="Per-field extraction confidence scores",
    )

    @field_validator("company_name", "description", "country", "industry", mode="before")
    @classmethod
    def clean_str(cls, v: object) -> Optional[str]:
        """Strip whitespace, discard placeholder strings."""
        if not isinstance(v, str):
            return None
        v = v.strip()
        if v.lower() in {"", "n/a", "na", "none", "unknown", "not found", "null"}:
            return None
        return v

    @field_validator("website", mode="before")
    @classmethod
    def normalise_url(cls, v: object) -> Optional[str]:
        if not isinstance(v, str):
            return None
        v = v.strip().rstrip("/")
        if v and not v.startswith("http"):
            v = "https://" + v
        return v or None

    @field_validator("employee_size", mode="before")
    @classmethod
    def normalise_employees(cls, v: object) -> Optional[str]:
        """
        Accept messy strings like '~500 employees', '10,000+', 'over 1000'
        and map them to an EmployeeSize band label.
        """
        if not isinstance(v, str):
            return None
        v = v.strip()
        if not v or v.lower() in {"unknown", "n/a", "none"}:
            return None

        # Extract the first number we can find
        digits = re.sub(r"[,\s]", "", v)
        m = re.search(r"(\d+)", digits)
        if not m:
            return v  # Return as-is; enricher will tidy later

        n = int(m.group(1))
        if n == 1:             return EmployeeSize.SOLO.value
        if n <= 10:            return EmployeeSize.MICRO.value
        if n <= 50:            return EmployeeSize.SMALL.value
        if n <= 200:           return EmployeeSize.MEDIUM.value
        if n <= 500:           return EmployeeSize.MID_LARGE.value
        if n <= 1_000:         return EmployeeSize.LARGE.value
        if n <= 5_000:         return EmployeeSize.ENTERPRISE.value
        if n <= 10_000:        return EmployeeSize.GIANT.value
        return EmployeeSize.MEGA.value

    def missing_fields(self) -> list[str]:
        """Return field names that are None — used by enricher to know what to fill."""
        core = ["company_name", "industry", "description", "country", "employee_size"]
        return [f for f in core if getattr(self, f) is None]

    def is_complete(self) -> bool:
        return len(self.missing_fields()) == 0

    def completeness_pct(self) -> float:
        core = ["company_name", "industry", "description", "country", "employee_size"]
        filled = sum(1 for f in core if getattr(self, f) is not None)
        return round(filled / len(core) * 100, 1)


# ── Final enriched record (strict, for DB + CSV) ──────────────────────────────

class EnrichedRecord(BaseModel):
    """
    Fully validated output record.
    Written to SQLite and exported to CSV.
    All fields have a guaranteed non-null value.
    """
    # Core output columns
    company_name:  str          = Field(...,   description="Official company name")
    industry:      str          = Field(...,   description="Industry bucket")
    description:   str          = Field(...,   description="Company summary")
    country:       str          = Field(...,   description="HQ country")
    employee_size: str          = Field(...,   description="Headcount band")
    website:       str          = Field(...,   description="Primary URL")

    # Pipeline metadata (stored in DB, excluded from CSV by default)
    scrape_method:   str   = Field("unknown", description="browser | httpx | wikipedia")
    completeness:    float = Field(0.0,       description="% of core fields filled (0–100)")
    ai_enriched:     bool  = Field(False,     description="True if AI filled missing fields")
    created_at:      float = Field(default_factory=time.time)

    @model_validator(mode="before")
    @classmethod
    def apply_defaults(cls, data: dict) -> dict:
        """Fill any remaining None / missing with safe fallback strings."""
        defaults = {
            "company_name":  "Unknown Company",
            "industry":      Industry.UNKNOWN.value,
            "description":   "No description available.",
            "country":       "Unknown",
            "employee_size": EmployeeSize.UNKNOWN.value,
        }
        for field, fallback in defaults.items():
            if not data.get(field):
                data[field] = fallback
        return data

    @classmethod
    def from_company_record(
        cls,
        record: CompanyRecord,
        website: str,
        scrape_method: str = "unknown",
        ai_enriched: bool = False,
    ) -> "EnrichedRecord":
        """Promote a CompanyRecord → EnrichedRecord, filling gaps with defaults."""
        core = ["company_name", "industry", "description", "country", "employee_size"]
        filled = sum(1 for f in core if getattr(record, f) is not None)
        completeness = round(filled / len(core) * 100, 1)

        return cls(
            company_name  = record.company_name  or "Unknown Company",
            industry      = record.industry      or Industry.UNKNOWN.value,
            description   = record.description   or "No description available.",
            country       = record.country       or "Unknown",
            employee_size = record.employee_size or EmployeeSize.UNKNOWN.value,
            website       = record.website       or website,
            scrape_method = scrape_method,
            completeness  = completeness,
            ai_enriched   = ai_enriched,
        )

    def to_csv_row(self) -> dict[str, str]:
        """Return only the 6 user-facing output fields for CSV export."""
        return {
            "company_name":  self.company_name,
            "industry":      self.industry,
            "description":   self.description,
            "country":       self.country,
            "employee_size": self.employee_size,
            "website":       self.website,
        }

    def __repr__(self) -> str:
        return (
            f"EnrichedRecord({self.company_name!r} | "
            f"{self.industry} | {self.country} | "
            f"{self.employee_size} | {self.completeness}%)"
        )


# ── Pipeline summary (for logging / reporting) ────────────────────────────────

class PipelineRun(BaseModel):
    """Tracks stats for a single pipeline execution."""
    total_urls:      int   = 0
    scraped_ok:      int   = 0
    extracted_ok:    int   = 0
    enriched_ok:     int   = 0
    failed:          int   = 0
    avg_completeness: float = 0.0
    started_at:      float = Field(default_factory=time.time)
    finished_at:     Optional[float] = None

    def duration_seconds(self) -> Optional[float]:
        if self.finished_at:
            return round(self.finished_at - self.started_at, 2)
        return None

    def summary(self) -> str:
        dur = self.duration_seconds()
        dur_str = f"{dur}s" if dur else "in progress"
        return (
            f"PipelineRun | {self.total_urls} URLs | "
            f"scraped={self.scraped_ok} extracted={self.extracted_ok} "
            f"enriched={self.enriched_ok} failed={self.failed} | "
            f"avg_completeness={self.avg_completeness}% | {dur_str}"
        )


# ── Quick self-test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n── CompanyRecord validation ──")

    # Messy AI output — test validators
    raw = CompanyRecord(
        company_name="  OpenAI  ",
        industry="Artificial Intelligence",
        description="OpenAI is an AI safety company.",
        country="United States",
        employee_size="~1,500 employees",
        website="openai.com",             # missing https
    )
    print(raw)
    print(f"  website       → {raw.website}")
    print(f"  employee_size → {raw.employee_size}")
    print(f"  missing       → {raw.missing_fields()}")
    print(f"  completeness  → {raw.completeness_pct()}%")

    print("\n── Placeholder rejection ──")
    dirty = CompanyRecord(
        company_name="N/A",
        industry="none",
        description="  ",
        country=None,
        employee_size="unknown",
    )
    print(f"  company_name → {dirty.company_name!r}  (should be None)")
    print(f"  missing      → {dirty.missing_fields()}")

    print("\n── EnrichedRecord.from_company_record ──")
    enriched = EnrichedRecord.from_company_record(
        raw, website="https://openai.com", scrape_method="browser", ai_enriched=False
    )
    print(enriched)
    print(f"  csv row → {enriched.to_csv_row()}")

    print("\n── Partial record with defaults ──")
    partial = CompanyRecord(company_name="GitHub", website="https://github.com")
    enriched2 = EnrichedRecord.from_company_record(
        partial, website="https://github.com", ai_enriched=True
    )
    print(enriched2)
    print(f"  ai_enriched  → {enriched2.ai_enriched}")
    print(f"  completeness → {enriched2.completeness}%")

    print("\n── EmployeeSize bands ──")
    for raw_val in ["1", "5 people", "~50", "500 employees", "10,000+", "over 2000"]:
        rec = CompanyRecord(employee_size=raw_val)
        print(f"  {raw_val!r:25} → {rec.employee_size!r}")

    print("\n── PipelineRun summary ──")
    import time as _time
    run = PipelineRun(
        total_urls=3, scraped_ok=2, extracted_ok=2,
        enriched_ok=2, failed=1, avg_completeness=78.3,
        finished_at=_time.time() + 12.4,
    )
    print(f"  {run.summary()}")
    print()