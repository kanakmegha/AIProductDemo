"""Pydantic models for the Website Intelligence module."""
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class Feature(BaseModel):
    """A single product feature extracted from a page."""

    name: str
    description: str
    importance: Literal["core", "secondary", "tertiary"]


class ProductProfile(BaseModel):
    """Structured intelligence profile for a SaaS product, derived from its website."""

    url: str
    product_name: str = ""
    tagline: str = ""
    target_audience: list[str] = Field(default_factory=list)
    core_features: list[Feature] = Field(default_factory=list)
    use_cases: list[str] = Field(default_factory=list)
    pricing_tiers: list[str] = Field(default_factory=list)
    screenshots_found: list[str] = Field(default_factory=list)
    raw_text_summary: str = ""
    confidence_score: float = 0.0
    scraped_at: datetime = Field(default_factory=datetime.utcnow)
