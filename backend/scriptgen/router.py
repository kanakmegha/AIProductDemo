"""FastAPI routes for the Demo Script Generator module."""
import asyncio
import logging
import uuid
from typing import Literal

from fastapi import APIRouter, HTTPException, Response
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

from backend.intelligence.models import ProductProfile
from backend.scriptgen.generator import GenerationError
from backend.scriptgen.models import DemoScript, DemoStep, GenerationConfig, HumanFeedback
from backend.scriptgen.service import apply_feedback, generate_script, get_script

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/scripts", tags=["scripts"])


class GenerateRequest(BaseModel):
    """Request body for the script generation endpoint."""

    product_profile: ProductProfile
    tone: Literal["professional", "casual", "technical"] = "professional"
    target_audience: str


@router.post("/generate", response_model=DemoScript)
async def generate_script_endpoint(payload: GenerateRequest, response: Response):
    """Generate a demo script from a ProductProfile."""
    request_id = str(uuid.uuid4())
    response.headers["X-Request-ID"] = request_id
    logger.info("request_id=%s generating script for product=%s", request_id, payload.product_profile.product_name)

    config = GenerationConfig(tone=payload.tone, target_audience=payload.target_audience)
    try:
        return await asyncio.to_thread(generate_script, payload.product_profile, config)
    except GenerationError as exc:
        logger.error("request_id=%s generation failed: %s", request_id, exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/{script_id}", response_model=DemoScript)
async def get_script_endpoint(script_id: str, response: Response):
    """Retrieve a previously generated script by ID."""
    response.headers["X-Request-ID"] = str(uuid.uuid4())
    try:
        return get_script(script_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Script '{script_id}' not found")


@router.post("/{script_id}/feedback", response_model=DemoScript)
async def feedback_endpoint(script_id: str, feedback: HumanFeedback, response: Response):
    """Apply reviewer feedback to a script step and return the updated script."""
    response.headers["X-Request-ID"] = str(uuid.uuid4())
    try:
        script = get_script(script_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Script '{script_id}' not found")

    try:
        return await asyncio.to_thread(apply_feedback, script, feedback)
    except (ValueError, GenerationError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get("/{script_id}/export", response_class=PlainTextResponse)
async def export_script_endpoint(script_id: str, response: Response):
    """Return a human-readable plain-text version of the script."""
    response.headers["X-Request-ID"] = str(uuid.uuid4())
    try:
        script = get_script(script_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Script '{script_id}' not found")

    return _format_script(script)


def _format_script(script: DemoScript) -> str:
    """Render a DemoScript as readable plain text with step numbers and narration."""
    minutes, seconds = divmod(script.total_estimated_duration_seconds, 60)
    duration_str = f"{minutes}m {seconds}s" if minutes else f"{seconds}s"

    lines = [
        f"=== Demo Script: {script.product_profile.product_name} ===",
        f"Tone: {script.tone}  |  Audience: {script.target_audience}",
        f"Total Duration: {duration_str}  |  Steps: {len(script.steps)}  |  Version: {script.version}",
        "",
    ]

    for step in script.steps:
        approved_tag = "  [APPROVED]" if step.approved else ""
        lines += [
            f"Step {step.step_number}: {step.title} ({step.duration_seconds}s){approved_tag}",
            f"  Action:      {step.action}",
            f"  Narration:   \"{step.narration.text}\"",
            f"  Tone notes:  {step.narration.tone_notes}",
            f"  URL pattern: {step.expected_url_pattern}",
            f"  Screenshot:  {step.screenshot_hint}",
            "",
        ]

    return "\n".join(lines).rstrip()
