"""Tests for backend.intelligence.scraper."""
import httpx
import pytest
import respx

from backend.intelligence.scraper import ScraperError, scrape_url

SAMPLE_HTML = """
<html>
<head>
  <title>Linear - Issue tracking</title>
  <meta name="description" content="The issue tracker built for modern teams.">
  <meta property="og:title" content="Linear">
  <meta property="og:description" content="Plan and build products.">
  <style>.hidden { display: none; }</style>
  <script>console.log("should be stripped");</script>
</head>
<body>
  <h1>Linear</h1>
  <h2>Built for speed</h2>
  <p>Linear is a tool for managing software projects.</p>
  <p>Teams use Linear to plan, track, and build.</p>
  <img src="https://linear.app/screenshot1.png" />
  <img src="https://linear.app/screenshot2.png" />
</body>
</html>
"""


@respx.mock
def test_successful_scrape_returns_populated_page():
    """A successful fetch should populate all ScrapedPage fields."""
    respx.get("https://linear.app").mock(
        return_value=httpx.Response(200, text=SAMPLE_HTML)
    )

    page = scrape_url("https://linear.app")

    assert page.title == "Linear - Issue tracking"
    assert page.meta_description == "The issue tracker built for modern teams."
    assert page.og_title == "Linear"
    assert page.og_description == "Plan and build products."
    assert "Linear" in page.headings
    assert "Built for speed" in page.headings
    assert len(page.paragraphs) == 2
    assert "https://linear.app/screenshot1.png" in page.image_urls
    assert "https://linear.app/screenshot2.png" in page.image_urls


@respx.mock
def test_404_response_raises_scraper_error():
    """A 404 HTTP response should raise ScraperError."""
    respx.get("https://example.com/missing").mock(
        return_value=httpx.Response(404, text="Not Found")
    )

    with pytest.raises(ScraperError):
        scrape_url("https://example.com/missing")


@respx.mock
def test_network_timeout_triggers_retry_and_raises(monkeypatch):
    """Repeated network timeouts should trigger retries and ultimately raise ScraperError."""
    monkeypatch.setattr("backend.intelligence.scraper.time.sleep", lambda _: None)

    route = respx.get("https://timeout.example.com").mock(
        side_effect=httpx.ConnectTimeout("connection timed out")
    )

    with pytest.raises(ScraperError):
        scrape_url("https://timeout.example.com")

    assert route.call_count == 3


@respx.mock
def test_script_and_style_tags_are_stripped():
    """Script and style tag contents must not appear in extracted text."""
    respx.get("https://linear.app").mock(
        return_value=httpx.Response(200, text=SAMPLE_HTML)
    )

    page = scrape_url("https://linear.app")

    full_text = " ".join(page.headings + page.paragraphs)
    assert "console.log" not in full_text
    assert "display: none" not in full_text
