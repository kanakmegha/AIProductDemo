"""FastAPI routes for the Website Intelligence module."""
import logging
import uuid

from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel, HttpUrl

from backend.intelligence.extractor import ExtractionError
from backend.intelligence.models import ProductProfile
from backend.intelligence.scraper import ScraperError
from backend.intelligence.service import analyze_url

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/intelligence", tags=["intelligence"])

SERVICE_VERSION = "1.0.0"


class AnalyzeRequest(BaseModel):
    """Request body for the analyze endpoint."""

    url: HttpUrl


class HealthResponse(BaseModel):
    """Response body for the health check endpoint."""

    status: str
    version: str


@router.post("/analyze", response_model=ProductProfile)
async def analyze(payload: AnalyzeRequest, request: Request, response: Response):
    """Scrape and extract a structured ProductProfile for the given product URL."""
    request_id = str(uuid.uuid4())
    response.headers["X-Request-ID"] = request_id

    url = str(payload.url)
    logger.info("request_id=%s analyzing url=%s", request_id, url)

    try:
        return await analyze_url(url)
    except ScraperError as exc:
        logger.warning("request_id=%s scrape failed: %s", request_id, exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ExtractionError as exc:
        logger.error("request_id=%s extraction failed: %s", request_id, exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/health", response_model=HealthResponse)
async def health(response: Response):
    """Return the service's health status and version."""
    response.headers["X-Request-ID"] = str(uuid.uuid4())
    return HealthResponse(status="ok", version=SERVICE_VERSION)
