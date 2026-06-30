"""Tests for backend.scriptgen.service (and the router's 404 behaviour)."""
import json
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.intelligence.models import Feature, ProductProfile
from backend.scriptgen import service
from backend.scriptgen.models import (
    DemoScript,
    DemoStep,
    GenerationConfig,
    HumanFeedback,
    Narration,
)
from backend.scriptgen.router import router
from backend.scriptgen.service import apply_feedback

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

LINEAR_PROFILE = ProductProfile(
    url="https://linear.app",
    product_name="Linear",
    tagline="The issue tracker for modern teams",
    target_audience=["software teams", "startups"],
    core_features=[
        Feature(name="Issue tracking", description="Track issues", importance="core"),
    ],
    use_cases=["sprint planning", "bug tracking"],
    pricing_tiers=["Free", "Plus"],
    raw_text_summary="Linear is a fast issue tracker.",
    confidence_score=0.86,
)


def _make_step(step_number: int, title: str = "Demo step") -> DemoStep:
    return DemoStep(
        step_number=step_number,
        title=title,
        action=f"Navigate to step {step_number}",
        narration=Narration(
            text=f"This is step {step_number} of the demo showing core product value.",
            tone_notes="confident",
            word_count=13,
        ),
        expected_url_pattern=r"linear\.app/.*",
        screenshot_hint=f"Screen for step {step_number}",
        duration_seconds=30,
    )


STEP_1 = _make_step(1, "Sign up and onboard")
STEP_2 = _make_step(2, "Create your first project")
STEP_3 = _make_step(3, "Track and manage issues")

VALID_SCRIPT = DemoScript(
    product_profile=LINEAR_PROFILE,
    steps=[STEP_1, STEP_2, STEP_3],
    total_estimated_duration_seconds=90,
    target_audience="software engineering teams",
    tone="professional",
)

JOURNEY_DATA = {
    "features": ["Issue tracking", "Cycles"],
    "journey": [
        {"step": 1, "title": "Sign up", "action_summary": "...", "why": "..."},
        {"step": 2, "title": "Create project", "action_summary": "...", "why": "..."},
        {"step": 3, "title": "Track issues", "action_summary": "...", "why": "..."},
    ],
}


@pytest.fixture(autouse=True)
def clear_store():
    """Reset in-memory stores before and after each test."""
    service._scripts.clear()
    service._journeys.clear()
    yield
    service._scripts.clear()
    service._journeys.clear()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_apply_feedback_edit_replaces_step_and_increments_version():
    """Editing a step should replace it in the script and increment the version number."""
    service._scripts[VALID_SCRIPT.script_id] = VALID_SCRIPT

    edited = STEP_1.model_copy(update={"title": "Revised Sign Up Flow"})
    feedback = HumanFeedback(
        script_id=VALID_SCRIPT.script_id,
        step_number=1,
        feedback_type="edit",
        edited_step=edited,
    )

    updated = apply_feedback(VALID_SCRIPT, feedback)

    assert updated.steps[0].title == "Revised Sign Up Flow"
    assert updated.version == VALID_SCRIPT.version + 1


def test_apply_feedback_approve_marks_step_approved():
    """Approving a step should set approved=True on that step without changing the version."""
    service._scripts[VALID_SCRIPT.script_id] = VALID_SCRIPT

    feedback = HumanFeedback(
        script_id=VALID_SCRIPT.script_id,
        step_number=2,
        feedback_type="approve",
    )

    updated = apply_feedback(VALID_SCRIPT, feedback)

    assert updated.steps[1].approved is True
    assert updated.steps[0].approved is False  # other steps unchanged
    assert updated.version == VALID_SCRIPT.version  # version does not increment on approve


@patch("backend.scriptgen.generator.InferenceClient")
def test_apply_feedback_reject_calls_only_call2(mock_client_cls):
    """Rejecting a step should invoke the LLM exactly once (Call 2 only — no re-planning)."""
    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client

    single_step_data = {
        "step_number": 1,
        "title": "Revised Sign Up",
        "action": "Navigate to linear.app/signup, fill in your email",
        "narration": {
            "text": "Getting started with Linear is instant. Sign up with your email and have your workspace ready in under a minute.",
            "tone_notes": "energetic, conversational",
        },
        "expected_url_pattern": r"linear\.app/signup",
        "screenshot_hint": "Sign-up form with email field highlighted",
        "duration_seconds": 25,
    }
    mock_client.chat_completion.return_value = _mock_hf_response(json.dumps(single_step_data))

    service._scripts[VALID_SCRIPT.script_id] = VALID_SCRIPT
    service._journeys[VALID_SCRIPT.script_id] = JOURNEY_DATA

    feedback = HumanFeedback(
        script_id=VALID_SCRIPT.script_id,
        step_number=1,
        feedback_type="reject",
        comment="Step feels too generic — make it more specific",
    )

    updated = apply_feedback(VALID_SCRIPT, feedback)

    # Only one LLM call: Call 2 (step rewrite). Call 1 (planning) must NOT be invoked.
    assert mock_client.chat_completion.call_count == 1
    assert updated.steps[0].title == "Revised Sign Up"


def test_get_nonexistent_script_returns_404():
    """GET /api/v1/scripts/{id} with an unknown ID must return HTTP 404."""
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    response = client.get("/api/v1/scripts/does-not-exist-99999")

    assert response.status_code == 404


# ---------------------------------------------------------------------------
# Acceptance criterion: static Linear.app fixture round-trip
# ---------------------------------------------------------------------------


@patch("backend.scriptgen.generator.InferenceClient")
def test_linear_fixture_script_acceptance_criteria(mock_client_cls):
    """
    Acceptance test: generate a script from a Linear.app profile fixture and
    assert steps >= 3, total_duration <= 300, no empty narration texts.
    """
    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client

    call1 = {
        "features": ["Issue tracking", "Cycles", "Roadmaps"],
        "journey": [
            {"step": i, "title": f"Step {i}", "action_summary": "...", "why": "..."}
            for i in range(1, 4)
        ],
    }
    call2 = {
        "steps": [
            {
                "step_number": i,
                "title": f"Demo Step {i}",
                "action": f"Perform action {i} in Linear",
                "narration": {
                    "text": "Linear helps software teams move faster by combining issue tracking cycles and roadmaps in one beautiful interface.",
                    "tone_notes": "professional",
                },
                "expected_url_pattern": r"linear\.app/.*",
                "screenshot_hint": f"Screen {i} showing core feature",
                "duration_seconds": 35,
            }
            for i in range(1, 4)
        ],
        "total_estimated_duration_seconds": 105,
    }

    mock_client.chat_completion.side_effect = [
        _mock_hf_response(json.dumps(call1)),
        _mock_hf_response(json.dumps(call2)),
    ]

    config = GenerationConfig(tone="professional", target_audience="engineering teams")
    script, _ = __import__(
        "backend.scriptgen.generator", fromlist=["generate_demo_script"]
    ).generate_demo_script(LINEAR_PROFILE, config)

    assert len(script.steps) >= 3
    assert script.total_estimated_duration_seconds <= 300
    for step in script.steps:
        assert step.narration.text.strip(), f"Step {step.step_number} has empty narration"


def _mock_hf_response(content: str):
    mock = MagicMock()
    mock.choices = [MagicMock()]
    mock.choices[0].message.content = content
    return mock
