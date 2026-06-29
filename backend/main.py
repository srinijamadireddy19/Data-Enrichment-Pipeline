"""
main.py — FastAPI server for the Data Enrichment Pipeline

Routes:
  POST /enrich                 — run pipeline on a list of URLs
  GET  /companies              — list all enriched companies (with filters)
  GET  /companies/{website}    — get single company by website URL
  DELETE /companies/{website}  — delete a company record
  GET  /export/csv             — download enriched data as CSV
  GET  /export/json            — download enriched data as JSON
  GET  /runs                   — list recent pipeline run history
  GET  /stats                  — DB summary stats
  GET  /health                 — health check + config status

Run:
  uvicorn main:app --reload --port 8000
"""

from __future__ import annotations

import asyncio
import logging
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional
from urllib.parse import quote

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field, field_validator

from config import cfg, setup_logging
from database import Database
from exporter import export_csv_from_db, export_json_from_db, timestamped_path
from models import EnrichedRecord
from pipeline import PipelineResult, _run_async

setup_logging()
log = logging.getLogger("main")


# ── Lifespan — init DB on startup ─────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Starting up — initialising database …")
    with Database(cfg.db_path) as db:
        pass   # creates tables if not exist
    log.info("Database ready at %s", cfg.db_path)
    log.info(cfg.summary())
    yield
    log.info("Shutting down")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Data Enrichment Pipeline API",
    description=(
        "AI-powered pipeline that scrapes company websites, "
        "extracts and enriches structured data using Groq LLaMA, "
        "stores results in SQLite, and exports to CSV/JSON."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── DB dependency ─────────────────────────────────────────────────────────────

def get_db() -> Database:
    db = Database(cfg.db_path)
    db.init()
    try:
        yield db
    finally:
        db.close()


# ── Request / Response schemas ────────────────────────────────────────────────

class EnrichRequest(BaseModel):
    urls: list[str] = Field(
        ...,
        min_length=1,
        description="List of company website URLs to enrich",
        examples=[["https://openai.com", "https://github.com"]],
    )
    export_csv:      bool = Field(True,  description="Write a CSV file after enrichment")
    export_json:     bool = Field(False, description="Write a JSON file after enrichment")
    export_metadata: bool = Field(False, description="Include pipeline metadata columns in exports")

    @field_validator("urls")
    @classmethod
    def validate_urls(cls, urls: list[str]) -> list[str]:
        cleaned = [u.strip() for u in urls if u.strip()]
        if not cleaned:
            raise ValueError("At least one non-empty URL is required")
        for url in cleaned:
            if not url.startswith(("http://", "https://")):
                raise ValueError(f"Invalid URL (must start with http/https): {url!r}")
        return cleaned


class EnrichResponse(BaseModel):
    status:           str
    total_urls:       int
    scraped_ok:       int
    extracted_ok:     int
    enriched_ok:      int
    failed:           int
    avg_completeness: float
    duration_seconds: Optional[float]
    csv_path:         Optional[str]
    json_path:        Optional[str]
    records:          list[dict]


class CompanyResponse(BaseModel):
    company_name:  str
    industry:      str
    description:   str
    country:       str
    employee_size: str
    website:       str
    scrape_method: str
    completeness:  float
    ai_enriched:   bool
    created_at:    float


class StatsResponse(BaseModel):
    total:            int
    ai_enriched:      int
    avg_completeness: float
    by_country:       list[dict]
    by_industry:      list[dict]


# ── In-progress job tracker ───────────────────────────────────────────────────
# Keyed by job_id (float timestamp). Simple in-memory store —
# good enough for MVP; swap for Redis in production.

_jobs: dict[str, dict] = {}


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
async def health():
    """Health check — confirms API is running and DB is reachable."""
    try:
        with Database(cfg.db_path) as db:
            count = db.count()
        db_status = "ok"
    except Exception as exc:
        db_status = f"error: {exc}"

    return {
        "status":      "ok",
        "db_status":   db_status,
        "db_path":     str(cfg.db_path),
        "groq_model":  cfg.groq_model,
        "api_key_set": bool(cfg.groq_api_key),
        "companies":   count if db_status == "ok" else None,
    }


@app.post("/enrich", response_model=EnrichResponse, tags=["Pipeline"])
async def enrich(req: EnrichRequest):
    """
    Run the full enrichment pipeline on a list of URLs.

    Scrapes each website, extracts structured company data using Groq LLaMA,
    cleans and enriches missing fields, stores in SQLite, and optionally exports.

    Returns the enriched records immediately (synchronous — for async/background
    use the /enrich/async endpoint).
    """
    if not cfg.groq_api_key:
        raise HTTPException(
            status_code=503,
            detail="GROQ_API_KEY is not configured on the server. Set it as an environment variable.",
        )

    log.info("POST /enrich — %d URL(s)", len(req.urls))
    t0 = time.time()

    try:
        result: PipelineResult = await _run_async(
            urls=req.urls,
            groq_api_key=cfg.groq_api_key,
            db_path=cfg.db_path,
            output_dir=cfg.output_dir,
            export_csv_flag=req.export_csv,
            export_json_flag=req.export_json,
            export_metadata=req.export_metadata,
            save_to_db=True,
        )
    except Exception as exc:
        log.exception("Pipeline error")
        raise HTTPException(status_code=500, detail=f"Pipeline error: {exc}")

    return EnrichResponse(
        status="complete",
        total_urls=result.run_meta.total_urls,
        scraped_ok=result.run_meta.scraped_ok,
        extracted_ok=result.run_meta.extracted_ok,
        enriched_ok=result.run_meta.enriched_ok,
        failed=result.run_meta.failed,
        avg_completeness=result.run_meta.avg_completeness,
        duration_seconds=result.run_meta.duration_seconds(),
        csv_path=str(result.csv_path) if result.csv_path else None,
        json_path=str(result.json_path) if result.json_path else None,
        records=[r.to_csv_row() for r in result.records],
    )


@app.post("/enrich/async", tags=["Pipeline"])
async def enrich_async(req: EnrichRequest, background_tasks: BackgroundTasks):
    """
    Start enrichment in the background and return a job_id immediately.
    Poll GET /jobs/{job_id} to check progress.
    """
    if not cfg.groq_api_key:
        raise HTTPException(status_code=503, detail="GROQ_API_KEY not configured")

    job_id = str(time.time())
    _jobs[job_id] = {"status": "running", "started_at": time.time()}

    async def _run_job():
        try:
            result = await _run_async(
                urls=req.urls,
                groq_api_key=cfg.groq_api_key,
                db_path=cfg.db_path,
                output_dir=cfg.output_dir,
                export_csv_flag=req.export_csv,
                export_json_flag=req.export_json,
                export_metadata=req.export_metadata,
                save_to_db=True,
            )
            _jobs[job_id].update({
                "status":           "complete",
                "enriched_ok":      result.run_meta.enriched_ok,
                "failed":           result.run_meta.failed,
                "avg_completeness": result.run_meta.avg_completeness,
                "duration_seconds": result.run_meta.duration_seconds(),
                "csv_path":         str(result.csv_path) if result.csv_path else None,
                "json_path":        str(result.json_path) if result.json_path else None,
                "records":          [r.to_csv_row() for r in result.records],
            })
        except Exception as exc:
            log.exception("Background job %s failed", job_id)
            _jobs[job_id].update({"status": "failed", "error": str(exc)})

    background_tasks.add_task(_run_job)
    return {"job_id": job_id, "status": "running", "total_urls": len(req.urls)}


@app.get("/jobs/{job_id}", tags=["Pipeline"])
async def get_job(job_id: str):
    """Check status of a background enrichment job."""
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id!r} not found")
    return {"job_id": job_id, **job}


@app.get("/companies", response_model=list[CompanyResponse], tags=["Companies"])
async def list_companies(
    country:  Optional[str] = Query(None, description="Filter by country (exact)"),
    industry: Optional[str] = Query(None, description="Filter by industry (exact)"),
    limit:    int           = Query(100,  ge=1, le=1000, description="Max rows to return"),
    db: Database = Depends(get_db),
):
    """
    List all enriched companies.

    Supports filtering by country and industry. Returns up to `limit` records,
    ordered by most recently created.
    """
    if country:
        rows = db.get_by_country(country)
    elif industry:
        rows = db.get_by_industry(industry)
    else:
        rows = db.get_all()

    return [
        CompanyResponse(
            company_name  = r["company_name"],
            industry      = r["industry"],
            description   = r["description"],
            country       = r["country"],
            employee_size = r["employee_size"],
            website       = r["website"],
            scrape_method = r.get("scrape_method", "unknown"),
            completeness  = r.get("completeness", 0.0),
            ai_enriched   = bool(r.get("ai_enriched", 0)),
            created_at    = r.get("created_at", 0.0),
        )
        for r in rows[:limit]
    ]


@app.get("/companies/{website:path}", response_model=CompanyResponse, tags=["Companies"])
async def get_company(website: str, db: Database = Depends(get_db)):
    """
    Get a single company by its website URL.

    Pass the URL without encoding — FastAPI handles it via `{website:path}`.
    Example: GET /companies/https://openai.com
    """
    # Normalise — ensure https:// prefix
    if not website.startswith("http"):
        website = "https://" + website

    row = db.get_by_website(website)
    if not row:
        raise HTTPException(
            status_code=404,
            detail=f"No record found for website: {website!r}",
        )
    return CompanyResponse(
        company_name  = row["company_name"],
        industry      = row["industry"],
        description   = row["description"],
        country       = row["country"],
        employee_size = row["employee_size"],
        website       = row["website"],
        scrape_method = row.get("scrape_method", "unknown"),
        completeness  = row.get("completeness", 0.0),
        ai_enriched   = bool(row.get("ai_enriched", 0)),
        created_at    = row.get("created_at", 0.0),
    )


@app.delete("/companies/{website:path}", tags=["Companies"])
async def delete_company(website: str, db: Database = Depends(get_db)):
    """Delete a company record by website URL."""
    if not website.startswith("http"):
        website = "https://" + website

    deleted = db.delete_by_website(website)
    if not deleted:
        raise HTTPException(
            status_code=404,
            detail=f"No record found for website: {website!r}",
        )
    return {"status": "deleted", "website": website}


@app.get("/export/csv", tags=["Export"])
async def download_csv(
    country:          Optional[str] = Query(None),
    industry:         Optional[str] = Query(None),
    include_metadata: bool          = Query(False),
    db: Database = Depends(get_db),
):
    """
    Download all enriched companies as a CSV file.

    Supports optional country/industry filters and a metadata columns flag.
    """
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    export_csv_from_db(
        db, tmp_path,
        include_metadata=include_metadata,
        country=country,
        industry=industry,
    )

    filename = "companies"
    if country:  filename += f"_{country.replace(' ', '_')}"
    if industry: filename += f"_{industry.replace(' ', '_').replace('/', '-')}"
    filename += ".csv"

    return FileResponse(
        path=str(tmp_path),
        media_type="text/csv",
        filename=filename,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/export/json", tags=["Export"])
async def download_json(
    include_metadata: bool = Query(False),
    db: Database = Depends(get_db),
):
    """Download all enriched companies as a JSON file."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    export_json_from_db(db, tmp_path, include_metadata=include_metadata)

    return FileResponse(
        path=str(tmp_path),
        media_type="application/json",
        filename="companies.json",
        headers={"Content-Disposition": 'attachment; filename="companies.json"'},
    )


@app.get("/runs", tags=["System"])
async def list_runs(
    limit: int = Query(20, ge=1, le=100),
    db: Database = Depends(get_db),
):
    """List recent pipeline run history."""
    return db.get_runs(limit=limit)


@app.get("/stats", response_model=StatsResponse, tags=["System"])
async def stats(db: Database = Depends(get_db)):
    """Database summary — total records, AI enrichment rate, breakdowns by country and industry."""
    return db.stats()




