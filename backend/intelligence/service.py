"""Orchestrates the scrape -> extract pipeline to produce a ProductProfile."""
import asyncio
import logging
import os

from backend.intelligence.extractor import ExtractionError, extract_profile
from backend.intelligence.models import ProductProfile
from backend.intelligence.scraper import ScraperError, scrape_url

logger = logging.getLogger(__name__)

PIPELINE_TIMEOUT_SECONDS = float(os.getenv("PIPELINE_TIMEOUT_SECONDS", "60"))


async def analyze_url(url: str) -> ProductProfile:
    """Run the full scrape -> extract pipeline for a URL, enforcing an overall timeout."""
    try:
        return await asyncio.wait_for(_run_pipeline(url), timeout=PIPELINE_TIMEOUT_SECONDS)
    except asyncio.TimeoutError as exc:
        raise ScraperError(url, f"pipeline exceeded {PIPELINE_TIMEOUT_SECONDS}s timeout") from exc


async def _run_pipeline(url: str) -> ProductProfile:
    """Scrape the URL, then extract a ProductProfile from the resulting page content."""
    try:
        page = await asyncio.to_thread(scrape_url, url)
    except ScraperError:
        raise
    except Exception as exc:
        raise ScraperError(url, str(exc)) from exc

    try:
        profile = await asyncio.to_thread(extract_profile, page)
    except ExtractionError:
        raise
    except Exception as exc:
        raise ExtractionError(str(exc)) from exc

    return profile
