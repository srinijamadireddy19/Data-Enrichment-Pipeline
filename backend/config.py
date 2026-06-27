"""
config.py — Centralised configuration for the Data Enrichment Pipeline

All tuneable settings live here. Every other module imports from this file
instead of having magic numbers scattered across the codebase.

Environment variables always override defaults.
A .env file is loaded automatically if python-dotenv is installed.

Usage:
    from config import cfg

    print(cfg.groq_api_key)
    print(cfg.db_path)
    print(cfg.scrape_concurrency)
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

# Load .env file if python-dotenv is available (optional dep)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

log = logging.getLogger("config")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _env(key: str, default: str) -> str:
    return os.environ.get(key, default)

def _env_int(key: str, default: int) -> int:
    try:
        return int(os.environ.get(key, default))
    except (TypeError, ValueError):
        return default

def _env_float(key: str, default: float) -> float:
    try:
        return float(os.environ.get(key, default))
    except (TypeError, ValueError):
        return default

def _env_bool(key: str, default: bool) -> bool:
    val = os.environ.get(key)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


# ── Config dataclass ──────────────────────────────────────────────────────────

@dataclass
class Config:
    """
    Single source of truth for all pipeline settings.
    Instantiate once via cfg = Config() and import everywhere.
    """

    # ── API ───────────────────────────────────────────────────────────────────
    groq_api_key: str = field(
        default_factory=lambda: _env("GROQ_API_KEY", "")
    )
    groq_model: str = field(
        default_factory=lambda: _env("GROQ_MODEL", "llama-3.3-70b-versatile")
    )

    # ── Paths ─────────────────────────────────────────────────────────────────
    db_path: Path = field(
        default_factory=lambda: Path(_env("DB_PATH", "pipeline.db"))
    )
    output_dir: Path = field(
        default_factory=lambda: Path(_env("OUTPUT_DIR", "output"))
    )

    # ── Scraper ───────────────────────────────────────────────────────────────
    scrape_concurrency: int = field(
        default_factory=lambda: _env_int("SCRAPE_CONCURRENCY", 3)
    )
    scrape_min_content: int = field(
        default_factory=lambda: _env_int("SCRAPE_MIN_CONTENT", 200)
    )
    scrape_max_content: int = field(
        default_factory=lambda: _env_int("SCRAPE_MAX_CONTENT", 8_000)
    )

    # ── Extractor ─────────────────────────────────────────────────────────────
    extract_concurrency: int = field(
        default_factory=lambda: _env_int("EXTRACT_CONCURRENCY", 3)
    )
    extract_max_tokens: int = field(
        default_factory=lambda: _env_int("EXTRACT_MAX_TOKENS", 512)
    )
    extract_temperature: float = field(
        default_factory=lambda: _env_float("EXTRACT_TEMPERATURE", 0.1)
    )
    extract_max_retries: int = field(
        default_factory=lambda: _env_int("EXTRACT_MAX_RETRIES", 3)
    )

    # ── Enricher ──────────────────────────────────────────────────────────────
    enrich_concurrency: int = field(
        default_factory=lambda: _env_int("ENRICH_CONCURRENCY", 3)
    )
    enrich_max_tokens: int = field(
        default_factory=lambda: _env_int("ENRICH_MAX_TOKENS", 256)
    )
    enrich_temperature: float = field(
        default_factory=lambda: _env_float("ENRICH_TEMPERATURE", 0.1)
    )
    enrich_max_retries: int = field(
        default_factory=lambda: _env_int("ENRICH_MAX_RETRIES", 2)
    )
    enrich_thin_threshold: int = field(
        default_factory=lambda: _env_int("ENRICH_THIN_THRESHOLD", 500)
    )

    # ── Export ────────────────────────────────────────────────────────────────
    export_csv: bool = field(
        default_factory=lambda: _env_bool("EXPORT_CSV", True)
    )
    export_json: bool = field(
        default_factory=lambda: _env_bool("EXPORT_JSON", False)
    )
    export_metadata: bool = field(
        default_factory=lambda: _env_bool("EXPORT_METADATA", False)
    )

    # ── Logging ───────────────────────────────────────────────────────────────
    log_level: str = field(
        default_factory=lambda: _env("LOG_LEVEL", "INFO").upper()
    )

    # ── Post-init validation ──────────────────────────────────────────────────

    def __post_init__(self) -> None:
        self.db_path     = Path(self.db_path)
        self.output_dir  = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def validate(self) -> "Config":
        """
        Raise ValueError if required settings are missing.
        Call this at pipeline startup — not at import time so tests can
        import config without needing a real API key.
        """
        if not self.groq_api_key:
            raise ValueError(
                "GROQ_API_KEY is not set.\n"
                "  Option 1: export GROQ_API_KEY=your_key\n"
                "  Option 2: create a .env file with GROQ_API_KEY=your_key\n"
                "  Get a free key at: https://console.groq.com"
            )
        return self

    def summary(self) -> str:
        """Human-readable config dump for startup logging."""
        key_preview = f"{self.groq_api_key[:6]}…" if self.groq_api_key else "NOT SET"
        return (
            f"Config:\n"
            f"  groq_model          = {self.groq_model}\n"
            f"  groq_api_key        = {key_preview}\n"
            f"  db_path             = {self.db_path}\n"
            f"  output_dir          = {self.output_dir}\n"
            f"  scrape_concurrency  = {self.scrape_concurrency}\n"
            f"  extract_concurrency = {self.extract_concurrency}\n"
            f"  enrich_concurrency  = {self.enrich_concurrency}\n"
            f"  export_csv          = {self.export_csv}\n"
            f"  export_json         = {self.export_json}\n"
            f"  export_metadata     = {self.export_metadata}\n"
            f"  log_level           = {self.log_level}\n"
        )


# ── Module-level singleton ────────────────────────────────────────────────────
# Import this in every module: `from config import cfg`

cfg = Config()


# ── Logging setup helper ──────────────────────────────────────────────────────

def setup_logging(level: str | None = None) -> None:
    """Configure root logger. Call once from pipeline.py entry point."""
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        level=getattr(logging, level or cfg.log_level, logging.INFO),
        force=True,
    )
    # Silence noisy third-party loggers
    for noisy in ("httpx", "httpcore", "crawl4ai", "asyncio", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)