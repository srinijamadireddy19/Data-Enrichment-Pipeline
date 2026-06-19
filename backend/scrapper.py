"""
scraper.py — Async website scraper for the Data Enrichment Pipeline

Three-layer scraping strategy:
  Layer 1 → crawl4ai headless browser (handles JS-heavy sites)
  Layer 2 → httpx with real browser headers (bypasses some bot detection)
  Layer 3 → Wikipedia API lookup (fallback for Cloudflare-blocked sites)

Usage:
    from scraper import scrape_urls_sync
    results = scrape_urls_sync(["https://openai.com", "https://github.com"])
"""

import asyncio
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Optional
from urllib.parse import urlparse

import httpx
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger("scraper")
logging.getLogger("httpx").setLevel(logging.WARNING)


# ── Data model ────────────────────────────────────────────────────────────────
@dataclass
class ScrapeResult:
    """Raw scrape output for one URL, ready for the AI extractor."""
    url: str
    success: bool
    markdown: str = ""            # Cleaned text content
    title: str = ""               # Page / article title
    meta_description: str = ""    # Meta description (if available)
    scrape_method: str = ""       # "browser" | "httpx" | "wikipedia"
    scraped_at: float = field(default_factory=time.time)
    error: Optional[str] = None

    @property
    def domain(self) -> str:
        return urlparse(self.url).netloc

    @property
    def text_length(self) -> int:
        return len(self.markdown)

    def __repr__(self) -> str:
        status = "✓" if self.success else "✗"
        detail = f"{self.text_length:,} chars [{self.scrape_method}]" if self.success else self.error
        return f"ScrapeResult({status} {self.domain} | {detail})"


# ── Config ────────────────────────────────────────────────────────────────────
FALLBACK_PATHS   = ["/about", "/about-us", "/company"]
MIN_CONTENT_LEN  = 200
MAX_CONTENT_LEN  = 8_000
MAX_CONCURRENCY  = 3

_BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "DNT": "1",
    "Connection": "keep-alive",
}


# ── crawl4ai config ───────────────────────────────────────────────────────────
def _browser_config() -> BrowserConfig:
    return BrowserConfig(
        headless=True,
        verbose=False,
        extra_args=[
            "--no-sandbox", "--disable-dev-shm-usage", "--disable-gpu",
            "--disable-extensions", "--blink-settings=imagesEnabled=false",
        ],
    )


def _run_config() -> CrawlerRunConfig:
    return CrawlerRunConfig(
        cache_mode=CacheMode.ENABLED,
        word_count_threshold=10,
        only_text=False,
        remove_forms=True,
        excluded_tags=["nav", "footer", "script", "style", "iframe", "noscript"],
        excluded_selector="[class*='cookie'], [class*='banner'], [id*='cookie']",
    )


# ── Layer 1: crawl4ai browser ─────────────────────────────────────────────────
async def _browser_fetch(
    crawler: AsyncWebCrawler, url: str
) -> tuple[bool, str, str, str, Optional[str]]:
    """Returns (ok, text, title, desc, error)."""
    try:
        r = await crawler.arun(url=url, config=_run_config())
        if not r.success:
            return False, "", "", "", r.error_message or "Crawl failed"
        md   = (r.markdown or "").strip()
        meta = r.metadata or {}
        return True, md, meta.get("title", ""), meta.get("description", ""), None
    except Exception as exc:
        return False, "", "", "", str(exc)


# ── Layer 2: httpx with browser headers ──────────────────────────────────────
def _strip_html(html: str) -> str:
    html = re.sub(r"<(script|style|noscript|iframe)[^>]*>.*?</\1>", "", html, flags=re.S | re.I)
    text = re.sub(r"<[^>]+>", " ", html)
    for ent, ch in [("&amp;","&"),("&lt;","<"),("&gt;",">"),
                    ("&nbsp;"," "),("&#39;","'"),("&quot;",'"')]:
        text = text.replace(ent, ch)
    return re.sub(r"\s{2,}", " ", text).strip()


def _meta(html: str, name: str) -> str:
    for pat in [
        rf'<meta[^>]+name=["\']?{name}["\']?[^>]+content=["\']([^"\']+)["\']',
        rf'<meta[^>]+content=["\']([^"\']+)["\'][^>]+name=["\']?{name}["\']?',
    ]:
        m = re.search(pat, html, re.I)
        if m:
            return m.group(1).strip()
    return ""


async def _httpx_fetch(url: str) -> tuple[bool, str, str, str, Optional[str]]:
    try:
        async with httpx.AsyncClient(
            headers=_BROWSER_HEADERS, follow_redirects=True,
            timeout=15.0, verify=False,
        ) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            html  = resp.text
            m_title = re.search(r"<title[^>]*>(.*?)</title>", html, re.I | re.S)
            title = m_title.group(1).strip() if m_title else ""
            desc  = _meta(html, "description") or _meta(html, "og:description")
            return True, _strip_html(html), title, desc, None
    except Exception as exc:
        return False, "", "", "", str(exc)


# ── Layer 3: Wikipedia API (graceful fallback for Cloudflare-blocked sites) ───
def _domain_to_company(url: str) -> str:
    """
    Turn 'https://openai.com' → 'OpenAI'
    Turn 'https://huggingface.co' → 'Hugging Face'
    """
    domain = urlparse(url).netloc.lower()
    domain = re.sub(r"^www\.", "", domain)
    name   = domain.split(".")[0]  # 'openai', 'github', 'huggingface'

    # Known aliases
    aliases = {
        "huggingface": "Hugging Face",
        "openai": "OpenAI",
        "github": "GitHub",
        "anthropic": "Anthropic",
        "deepmind": "DeepMind",
        "mistral": "Mistral AI",
    }
    return aliases.get(name, name.title())


async def _wikipedia_fetch(url: str) -> tuple[bool, str, str, str, Optional[str]]:
    """
    Search Wikipedia for the company and return its introduction.
    Returns (ok, text, title, desc, error).
    """
    query = _domain_to_company(url)
    api   = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query", "format": "json",
        "generator": "search", "gsrsearch": query, "gsrlimit": "1",
        "prop": "extracts", "exintro": "1", "explaintext": "1",
    }
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(api, params=params)
            resp.raise_for_status()
            data  = resp.json()
            pages = data.get("query", {}).get("pages", {})
            if not pages:
                return False, "", "", "", "No Wikipedia article found"
            page  = next(iter(pages.values()))
            title = page.get("title", "")
            text  = page.get("extract", "").strip()
            if not text:
                return False, "", "", "", "Empty Wikipedia extract"
            note = f"\n\n[Source: Wikipedia article '{title}']"
            return True, text + note, title, "", None
    except Exception as exc:
        return False, "", "", "", str(exc)


# ── Core per-URL scraper ──────────────────────────────────────────────────────
async def _scrape_one(crawler: AsyncWebCrawler, url: str) -> ScrapeResult:
    """
    Full three-layer scraping for a single URL.
    """

    async def _try(target: str) -> tuple[bool, str, str, str, str, Optional[str]]:
        """Try browser → httpx. Returns (ok, text, title, desc, method, error)."""
        ok, txt, ti, de, err = await _browser_fetch(crawler, target)
        if ok and len(txt) >= MIN_CONTENT_LEN:
            return True, txt, ti, de, "browser", None

        b_err = err or "Thin content"
        log.info("    ↩ httpx fallback for %s", target)
        ok, txt, ti, de, err = await _httpx_fetch(target)
        if ok and len(txt) >= MIN_CONTENT_LEN:
            return True, txt, ti, de, "httpx", None

        return False, txt, ti, de, "httpx", f"browser={b_err} | httpx={err}"

    log.info("Scraping %s …", url)

    # ── Layer 1 + 2 on root URL ───────────────────────────────────────────────
    ok, text, title, desc, method, error = await _try(url)
    if ok:
        trunc = text[:MAX_CONTENT_LEN]
        log.info("  ✓ %s [%s] — %s chars", url, method, f"{len(trunc):,}")
        return ScrapeResult(url=url, success=True, markdown=trunc,
                            title=title, meta_description=desc, scrape_method=method)

    root_text, root_title, root_desc = text, title, desc

    # ── Layer 1 + 2 on /about fallbacks ──────────────────────────────────────
    for path in FALLBACK_PATHS:
        fb_url = url.rstrip("/") + path
        log.info("  ↩ Fallback: %s", fb_url)
        ok, txt, ti, de, mth, _ = await _try(fb_url)
        if ok:
            combined = "\n\n".join(filter(None, [root_text, txt]))[:MAX_CONTENT_LEN]
            log.info("  ✓ %s (via %s) [%s] — %s chars", url, path, mth, f"{len(combined):,}")
            return ScrapeResult(url=url, success=True, markdown=combined,
                                title=root_title or ti,
                                meta_description=root_desc or de,
                                scrape_method=mth)

    # ── Layer 3: Wikipedia ────────────────────────────────────────────────────
    log.info("  ↩ Wikipedia fallback for %s", url)
    ok, txt, ti, de, err = await _wikipedia_fetch(url)
    if ok:
        log.info("  ✓ %s [wikipedia] — %s chars", url, f"{len(txt):,}")
        return ScrapeResult(url=url, success=True,
                            markdown=txt[:MAX_CONTENT_LEN],
                            title=ti, meta_description=de,
                            scrape_method="wikipedia")

    # ── Complete failure — return partial data if any ─────────────────────────
    if root_text:
        log.warning("  ⚠ %s — thin content (%d chars), using anyway", url, len(root_text))
        return ScrapeResult(url=url, success=True,
                            markdown=root_text[:MAX_CONTENT_LEN],
                            title=root_title, meta_description=root_desc,
                            scrape_method=method)

    log.error("  ✗ %s — all layers failed", url)
    return ScrapeResult(url=url, success=False, error=f"All layers failed: {error}")


# ── Public API ────────────────────────────────────────────────────────────────
async def scrape_urls(
    urls: list[str],
    concurrency: int = MAX_CONCURRENCY,
) -> list[ScrapeResult]:
    """
    Scrape a list of URLs concurrently with three-layer fallback.

    Args:
        urls:        Company website URLs to scrape.
        concurrency: Max simultaneous browser tabs (default 3).

    Returns:
        List[ScrapeResult] in the same order as input.
    """
    if not urls:
        return []

    sem     = asyncio.Semaphore(concurrency)
    results: list[Optional[ScrapeResult]] = [None] * len(urls)

    async def _bounded(idx: int, url: str) -> None:
        async with sem:
            results[idx] = await _scrape_one(crawler, url)

    async with AsyncWebCrawler(config=_browser_config()) as crawler:
        await asyncio.gather(*[_bounded(i, u) for i, u in enumerate(urls)])

    return results  # type: ignore[return-value]


def scrape_urls_sync(
    urls: list[str],
    concurrency: int = MAX_CONCURRENCY,
) -> list[ScrapeResult]:
    """Synchronous wrapper — use when no event loop is running."""
    return asyncio.run(scrape_urls(urls, concurrency))


# ── CLI smoke test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    test_urls = sys.argv[1:] or [
        "https://openai.com",
        "https://github.com",
        "https://huggingface.co",
    ]

    print(f"\n{'='*60}")
    print(f"  Scraping {len(test_urls)} URL(s) …")
    print(f"{'='*60}\n")

    results = scrape_urls_sync(test_urls)

    for r in results:
        print(r)
        if r.success:
            print(f"  Title   : {r.title or '—'}")
            print(f"  Meta    : {(r.meta_description or '—')[:100]}")
            print(f"  Preview : {r.markdown[:400]} …")
        else:
            print(f"  Error   : {r.error}")
        print()

    passed = sum(1 for r in results if r.success)
    print(f"{'='*60}")
    print(f"  Result: {passed}/{len(results)} succeeded")
    print(f"{'='*60}\n")