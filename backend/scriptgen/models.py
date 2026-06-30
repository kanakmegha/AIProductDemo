"""Pydantic models for the Demo Script Generator module."""
import uuid
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from backend.intelligence.models import ProductProfile


class Narration(BaseModel):
    """Spoken narration and delivery guidance for one demo step."""

    text: str
    tone_notes: str
    word_count: int = 0  # computed from text, not supplied by the LLM


class DemoStep(BaseModel):
    """A single step in a product demo script."""

    step_number: int
    title: str
    action: str
    narration: Narration
    expected_url_pattern: str
    screenshot_hint: str
    duration_seconds: int
    approved: bool = False  # set to True via HumanFeedback "approve"


class DemoScript(BaseModel):
    """A complete, LLM-generated product demo script."""

    script_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    product_profile: ProductProfile
    steps: list[DemoStep]
    total_estimated_duration_seconds: int
    target_audience: str
    tone: Literal["professional", "casual", "technical"]
    created_at: datetime = Field(default_factory=datetime.utcnow)
    version: int = 1


class GenerationConfig(BaseModel):
    """Parameters that control script generation."""

    tone: Literal["professional", "casual", "technical"] = "professional"
    target_audience: str
    max_steps: int = Field(default=5, ge=3, le=7)


class HumanFeedback(BaseModel):
    """Reviewer feedback on a generated script or an individual step."""

    script_id: str
    step_number: int | None = None  # None = whole-script feedback
    feedback_type: Literal["approve", "edit", "reject"]
    edited_step: DemoStep | None = None
    comment: str = ""
