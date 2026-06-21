from backend.models import CompanyRecord, EnrichedRecord, PipelineRun

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