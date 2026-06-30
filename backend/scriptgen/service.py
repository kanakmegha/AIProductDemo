"""Orchestration layer for demo script generation, feedback, and in-memory storage."""
import logging
import time

from backend.intelligence.models import ProductProfile
from backend.scriptgen.generator import GenerationError, generate_demo_script, regenerate_step
from backend.scriptgen.models import DemoScript, DemoStep, GenerationConfig, HumanFeedback

logger = logging.getLogger(__name__)

# In-memory stores keyed by script_id — no database in this milestone.
_scripts: dict[str, DemoScript] = {}
_journeys: dict[str, dict] = {}


def generate_script(profile: ProductProfile, config: GenerationConfig) -> DemoScript:
    """Run the full planning + writing pipeline, cache the result, and return the script."""
    start = time.monotonic()
    script, journey = generate_demo_script(profile, config)
    duration = time.monotonic() - start

    _scripts[script.script_id] = script
    _journeys[script.script_id] = journey

    logger.info(
        "Generated script_id=%s product=%s steps=%d total_duration=%ds elapsed=%.3fs",
        script.script_id,
        profile.product_name,
        len(script.steps),
        script.total_estimated_duration_seconds,
        duration,
    )
    return script


def get_script(script_id: str) -> DemoScript:
    """Return a stored script by ID, raising KeyError if it does not exist."""
    if script_id not in _scripts:
        raise KeyError(script_id)
    return _scripts[script_id]


def apply_feedback(script: DemoScript, feedback: HumanFeedback) -> DemoScript:
    """
    Apply reviewer feedback to a script and persist the result.

    - "edit"    → replace the specified step with feedback.edited_step; increment version
    - "reject"  → regenerate the specified step using Call 2 only (no re-planning)
    - "approve" → mark the specified step as approved; version unchanged
    """
    steps = list(script.steps)

    if feedback.feedback_type == "edit":
        if feedback.edited_step is None:
            raise ValueError("HumanFeedback.edited_step is required for feedback_type='edit'")
        idx = _find_step_index(steps, feedback.step_number)
        steps[idx] = feedback.edited_step
        updated = script.model_copy(update={"steps": steps, "version": script.version + 1})

    elif feedback.feedback_type == "reject":
        idx = _find_step_index(steps, feedback.step_number)
        journey = _journeys.get(script.script_id, {})
        new_step = regenerate_step(script, feedback.step_number, journey, feedback.comment)
        steps[idx] = new_step
        updated = script.model_copy(update={"steps": steps})

    elif feedback.feedback_type == "approve":
        idx = _find_step_index(steps, feedback.step_number)
        steps[idx] = steps[idx].model_copy(update={"approved": True})
        updated = script.model_copy(update={"steps": steps})

    else:
        raise ValueError(f"Unknown feedback_type: {feedback.feedback_type!r}")

    _scripts[script.script_id] = updated
    return updated


def _find_step_index(steps: list[DemoStep], step_number: int | None) -> int:
    """Return the list index for the given step_number, raising ValueError if not found."""
    if step_number is None:
        raise ValueError("step_number is required for step-level feedback")
    idx = next((i for i, s in enumerate(steps) if s.step_number == step_number), None)
    if idx is None:
        raise ValueError(f"Step {step_number} not found in script")
    return idx
