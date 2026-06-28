"""Integration tests for backend.intelligence.service."""
import asyncio
from unittest.mock import patch

import pytest

from backend.intelligence.extractor import ExtractionError
from backend.intelligence.models import ProductProfile
from backend.intelligence.scraper import ScrapedPage, ScraperError
from backend.intelligence.service import analyze_url

SAMPLE_PAGE = ScrapedPage(
    url="https://linear.app",
    title="Linear",
    headings=["Linear"],
    paragraphs=["Linear is a project management tool."],
    image_urls=[],
)

SAMPLE_PROFILE = ProductProfile(
    url="https://linear.app",
    product_name="Linear",
    tagline="Issue tracking for modern teams",
    confidence_score=0.9,
)


@pytest.mark.asyncio
@patch("backend.intelligence.service.extract_profile")
@patch("backend.intelligence.service.scrape_url")
async def test_full_pipeline_returns_valid_profile(mock_scrape, mock_extract):
    """The pipeline should return a valid ProductProfile when scraping and extraction succeed."""
    mock_scrape.return_value = SAMPLE_PAGE
    mock_extract.return_value = SAMPLE_PROFILE

    profile = await analyze_url("https://linear.app")

    assert profile.product_name == "Linear"
    mock_scrape.assert_called_once_with("https://linear.app")
    mock_extract.assert_called_once_with(SAMPLE_PAGE)


@pytest.mark.asyncio
@patch("backend.intelligence.service.extract_profile")
@patch("backend.intelligence.service.scrape_url")
async def test_scraper_failure_propagates_as_scraper_error(mock_scrape, mock_extract):
    """A scraper failure should propagate as ScraperError without calling the extractor."""
    mock_scrape.side_effect = ScraperError("https://bad.example.com", "connection refused")

    with pytest.raises(ScraperError):
        await analyze_url("https://bad.example.com")

    mock_extract.assert_not_called()


@pytest.mark.asyncio
@patch("backend.intelligence.service.extract_profile")
@patch("backend.intelligence.service.scrape_url")
async def test_extraction_failure_propagates_as_extraction_error(mock_scrape, mock_extract):
    """An extraction failure should propagate as ExtractionError."""
    mock_scrape.return_value = SAMPLE_PAGE
    mock_extract.side_effect = ExtractionError("LLM unavailable")

    with pytest.raises(ExtractionError):
        await analyze_url("https://linear.app")


@pytest.mark.asyncio
@patch("backend.intelligence.service.PIPELINE_TIMEOUT_SECONDS", 0.05)
@patch("backend.intelligence.service.extract_profile")
@patch("backend.intelligence.service.scrape_url")
async def test_total_timeout_is_enforced(mock_scrape, mock_extract):
    """The pipeline should raise ScraperError if it exceeds the total timeout."""

    def slow_scrape(_url):
        import time
        time.sleep(0.2)
        return SAMPLE_PAGE

    mock_scrape.side_effect = slow_scrape

    with pytest.raises(ScraperError):
        await analyze_url("https://linear.app")
