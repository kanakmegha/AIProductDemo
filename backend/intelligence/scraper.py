"""Raw HTML fetching and cleaning for the Website Intelligence module."""
import logging
import os
import time
from dataclasses import dataclass, field

import httpx
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

DEFAULT_USER_AGENT = os.getenv(
    "SCRAPER_USER_AGENT", "Mozilla/5.0 (compatible; WebsiteIntelligenceBot/1.0)"
)
MAX_RETRIES = 3
REQUEST_TIMEOUT_SECONDS = 30.0


class ScraperError(Exception):
    """Raised when a target URL cannot be scraped successfully."""

    def __init__(self, url: str, reason: str):
        """Store the offending URL and a human-readable reason for the failure."""
        self.url = url
        self.reason = reason
        super().__init__(f"Failed to scrape {url}: {reason}")


@dataclass
class ScrapedPage:
    """Cleaned, structured content extracted from a single HTML page."""

    url: str
    title: str = ""
    meta_description: str = ""
    og_title: str = ""
    og_description: str = ""
    headings: list[str] = field(default_factory=list)
    paragraphs: list[str] = field(default_factory=list)
    image_urls: list[str] = field(default_factory=list)


def _fetch_with_retries(url: str) -> httpx.Response:
    """Fetch a URL with up to MAX_RETRIES attempts using exponential backoff."""
    headers = {"User-Agent": DEFAULT_USER_AGENT}
    last_error: Exception | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        start = time.monotonic()
        try:
            with httpx.Client(timeout=REQUEST_TIMEOUT_SECONDS, follow_redirects=True) as client:
                response = client.get(url, headers=headers)
            duration = time.monotonic() - start
            logger.info(
                "GET %s attempt=%d status=%d duration=%.3fs",
                url, attempt, response.status_code, duration,
            )
            response.raise_for_status()
            return response
        except httpx.HTTPStatusError as exc:
            duration = time.monotonic() - start
            logger.info(
                "GET %s attempt=%d status=%d duration=%.3fs",
                url, attempt, exc.response.status_code, duration,
            )
            raise ScraperError(url, f"HTTP {exc.response.status_code}") from exc
        except httpx.HTTPError as exc:
            duration = time.monotonic() - start
            logger.warning(
                "GET %s attempt=%d failed=%s duration=%.3fs",
                url, attempt, exc, duration,
            )
            last_error = exc
            if attempt < MAX_RETRIES:
                time.sleep(2 ** (attempt - 1))

    raise ScraperError(url, f"network error after {MAX_RETRIES} attempts: {last_error}")


def scrape_url(url: str) -> ScrapedPage:
    """Fetch and parse a URL into a ScrapedPage with cleaned text and image URLs."""
    response = _fetch_with_retries(url)

    soup = BeautifulSoup(response.text, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()

    title = soup.title.string.strip() if soup.title and soup.title.string else ""

    meta_description = ""
    meta_tag = soup.find("meta", attrs={"name": "description"})
    if meta_tag and meta_tag.get("content"):
        meta_description = meta_tag["content"].strip()

    og_title = ""
    og_title_tag = soup.find("meta", attrs={"property": "og:title"})
    if og_title_tag and og_title_tag.get("content"):
        og_title = og_title_tag["content"].strip()

    og_description = ""
    og_description_tag = soup.find("meta", attrs={"property": "og:description"})
    if og_description_tag and og_description_tag.get("content"):
        og_description = og_description_tag["content"].strip()

    headings = [
        h.get_text(strip=True)
        for h in soup.find_all(["h1", "h2", "h3"])
        if h.get_text(strip=True)
    ]

    paragraphs = [
        p.get_text(strip=True)
        for p in soup.find_all("p")
        if p.get_text(strip=True)
    ]

    image_urls = [img["src"] for img in soup.find_all("img") if img.get("src")]

    return ScrapedPage(
        url=url,
        title=title,
        meta_description=meta_description,
        og_title=og_title,
        og_description=og_description,
        headings=headings,
        paragraphs=paragraphs,
        image_urls=image_urls,
    )
