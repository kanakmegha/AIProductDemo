"""Tests for backend.scriptgen.generator."""
import json
from unittest.mock import MagicMock, patch

import pytest

from backend.intelligence.models import Feature, ProductProfile
from backend.scriptgen.generator import GenerationError, generate_demo_script
from backend.scriptgen.models import GenerationConfig

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

LINEAR_PROFILE = ProductProfile(
    url="https://linear.app",
    product_name="Linear",
    tagline="The issue tracker built for modern software teams",
    target_audience=["software teams", "startups", "engineers"],
    core_features=[
        Feature(name="Issue tracking", description="Track issues across teams", importance="core"),
        Feature(name="Cycles", description="Sprint planning with cycles", importance="core"),
        Feature(name="Roadmaps", description="Visual project roadmaps", importance="secondary"),
    ],
    use_cases=["sprint planning", "bug tracking", "feature development"],
    pricing_tiers=["Free", "Plus", "Business"],
    raw_text_summary="Linear is a fast, opinionated issue tracker for modern software teams.",
    confidence_score=0.86,
)

DEFAULT_CONFIG = GenerationConfig(
    tone="professional",
    target_audience="software engineering teams",
    max_steps=5,
)

# Call 1 — planning response
CALL1_DATA = {
    "features": ["Issue tracking", "Cycles", "Roadmaps"],
    "journey": [
        {"step": 1, "title": "Sign up and onboard", "action_summary": "Navigate to homepage and sign up", "why": "Establishes the product instantly"},
        {"step": 2, "title": "Create your first project", "action_summary": "Set up a new project with a team", "why": "Shows how simple setup is"},
        {"step": 3, "title": "Track and manage issues", "action_summary": "Create an issue, set priority, assign it", "why": "Core workflow"},
    ],
}

# Call 2 — writing response (narration texts must be 20-50 words each)
CALL2_DATA = {
    "steps": [
        {
            "step_number": 1,
            "title": "Sign up and onboard",
            "action": "Navigate to https://linear.app, click 'Start for free', complete the sign-up form",
            "narration": {
                # 26 words
                "text": "Welcome to Linear, the issue tracker built for modern software teams. In seconds you can set up a workspace and invite your entire team.",
                "tone_notes": "enthusiastic, emphasize speed of onboarding",
            },
            "expected_url_pattern": r"linear\.app/.*",
            "screenshot_hint": "Linear homepage with hero tagline and sign-up button visible",
            "duration_seconds": 30,
        },
        {
            "step_number": 2,
            "title": "Create your first project",
            "action": "Click 'New Project', enter project name, invite team members, click 'Create'",
            "narration": {
                # 25 words
                "text": "Setting up a project takes seconds. Name it, add your teammates, and start organizing work right away with intuitive labels and priorities.",
                "tone_notes": "confident, highlight simplicity and speed",
            },
            "expected_url_pattern": r"linear\.app/.*/projects",
            "screenshot_hint": "New project dialog with team member invite field",
            "duration_seconds": 35,
        },
        {
            "step_number": 3,
            "title": "Track and manage issues",
            "action": "Press 'C' to create a new issue, set priority to High, assign to a teammate, add a label",
            "narration": {
                # 27 words
                "text": "Creating and triaging issues is where Linear shines. Set priorities, assign teammates, and track every detail from a beautiful keyboard-first interface designed for speed.",
                "tone_notes": "technical, demonstrate keyboard shortcut, emphasize developer experience",
            },
            "expected_url_pattern": r"linear\.app/.*/issues",
            "screenshot_hint": "Issue detail view with priority, assignee, and label set",
            "duration_seconds": 40,
        },
    ],
    "total_estimated_duration_seconds": 105,
}


def _mock_hf_response(content: str):
    """Build a fake huggingface_hub chat_completion response."""
    mock = MagicMock()
    mock.choices = [MagicMock()]
    mock.choices[0].message.content = content
    return mock


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@patch("backend.scriptgen.generator.InferenceClient")
def test_generated_script_has_3_to_7_steps(mock_client_cls):
    """A successful generation should produce between 3 and 7 steps."""
    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client
    mock_client.chat_completion.side_effect = [
        _mock_hf_response(json.dumps(CALL1_DATA)),
        _mock_hf_response(json.dumps(CALL2_DATA)),
    ]

    script, _ = generate_demo_script(LINEAR_PROFILE, DEFAULT_CONFIG)

    assert 3 <= len(script.steps) <= 7


@patch("backend.scriptgen.generator.InferenceClient")
def test_each_narration_is_20_to_50_words(mock_client_cls):
    """Every step's narration should contain 20–50 words (computed from text)."""
    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client
    mock_client.chat_completion.side_effect = [
        _mock_hf_response(json.dumps(CALL1_DATA)),
        _mock_hf_response(json.dumps(CALL2_DATA)),
    ]

    script, _ = generate_demo_script(LINEAR_PROFILE, DEFAULT_CONFIG)

    for step in script.steps:
        wc = step.narration.word_count
        assert 20 <= wc <= 50, (
            f"Step {step.step_number} narration has {wc} words: {step.narration.text!r}"
        )


@patch("backend.scriptgen.generator.InferenceClient")
def test_total_duration_is_90_to_300_seconds(mock_client_cls):
    """Total estimated duration should fall within the 90–300 s target range."""
    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client
    mock_client.chat_completion.side_effect = [
        _mock_hf_response(json.dumps(CALL1_DATA)),
        _mock_hf_response(json.dumps(CALL2_DATA)),
    ]

    script, _ = generate_demo_script(LINEAR_PROFILE, DEFAULT_CONFIG)

    assert 90 <= script.total_estimated_duration_seconds <= 300


@patch("backend.scriptgen.generator.InferenceClient")
def test_all_demo_step_fields_are_non_empty(mock_client_cls):
    """Every DemoStep field expected to carry content must be non-empty/non-zero."""
    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client
    mock_client.chat_completion.side_effect = [
        _mock_hf_response(json.dumps(CALL1_DATA)),
        _mock_hf_response(json.dumps(CALL2_DATA)),
    ]

    script, _ = generate_demo_script(LINEAR_PROFILE, DEFAULT_CONFIG)

    for step in script.steps:
        assert step.title, f"Step {step.step_number} title is empty"
        assert step.action, f"Step {step.step_number} action is empty"
        assert step.narration.text, f"Step {step.step_number} narration text is empty"
        assert step.narration.tone_notes, f"Step {step.step_number} tone_notes is empty"
        assert step.expected_url_pattern, f"Step {step.step_number} url pattern is empty"
        assert step.screenshot_hint, f"Step {step.step_number} screenshot hint is empty"
        assert step.duration_seconds > 0, f"Step {step.step_number} duration is 0"


@patch("backend.scriptgen.generator.InferenceClient")
def test_exactly_2_llm_calls_per_generation(mock_client_cls):
    """Generating a script should make exactly 2 chat_completion calls (Call 1 + Call 2)."""
    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client
    mock_client.chat_completion.side_effect = [
        _mock_hf_response(json.dumps(CALL1_DATA)),
        _mock_hf_response(json.dumps(CALL2_DATA)),
    ]

    generate_demo_script(LINEAR_PROFILE, DEFAULT_CONFIG)

    assert mock_client.chat_completion.call_count == 2


@patch("backend.scriptgen.generator.InferenceClient")
def test_markdown_fenced_json_in_call2_is_stripped_and_parsed(mock_client_cls):
    """A markdown-fenced JSON response from Call 2 should be stripped and parsed successfully."""
    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client
    fenced_call2 = f"```json\n{json.dumps(CALL2_DATA)}\n```"
    mock_client.chat_completion.side_effect = [
        _mock_hf_response(json.dumps(CALL1_DATA)),
        _mock_hf_response(fenced_call2),
    ]

    script, _ = generate_demo_script(LINEAR_PROFILE, DEFAULT_CONFIG)

    assert script.steps[0].title == "Sign up and onboard"
    assert len(script.steps) == 3


@patch("backend.scriptgen.generator.InferenceClient")
def test_call2_failure_triggers_fallback_model(mock_client_cls):
    """If Call 2 fails on the primary model, the fallback model should be used."""
    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client
    mock_client.chat_completion.side_effect = [
        _mock_hf_response(json.dumps(CALL1_DATA)),  # Call 1 succeeds on primary
        RuntimeError("primary model unavailable"),   # Call 2 fails on primary
        _mock_hf_response(json.dumps(CALL2_DATA)),  # Call 2 succeeds on fallback
    ]

    script, _ = generate_demo_script(LINEAR_PROFILE, DEFAULT_CONFIG)

    assert script.product_profile.product_name == "Linear"
    # 3 InferenceClient instances: Call1(primary) + Call2(primary) + Call2(fallback)
    assert mock_client_cls.call_count == 3


@patch("backend.scriptgen.generator.InferenceClient")
def test_invalid_backslash_escapes_in_call2_are_sanitized(mock_client_cls):
    """
    LLMs often embed bare regex patterns inside JSON strings (e.g. "linear\\.app/.*"),
    producing invalid JSON escape sequences such as \\. or \\-.
    The parser must sanitize these and return a valid DemoScript.
    """
    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client

    # Build a JSON string that contains invalid \\. escapes in the url pattern fields.
    # In a Python string literal, "\\\\" is two chars: backslash + backslash, and "\\." is
    # backslash + dot.  The resulting *string value* is  "linear\.app/.*"  which is invalid JSON.
    bad_escape_call2 = (
        '{"steps": ['
        '{"step_number": 1, "title": "Sign up and onboard",'
        ' "action": "Navigate to https://linear.app, click Start for free",'
        ' "narration": {"text": "Welcome to Linear the issue tracker built for modern'
        ' software teams. Get started in seconds and start shipping faster today.",'
        ' "tone_notes": "enthusiastic, emphasize speed"},'
        ' "expected_url_pattern": "linear\\.app\\/.*",'   # \. is an invalid JSON escape
        ' "screenshot_hint": "Linear homepage with hero tagline",'
        ' "duration_seconds": 30},'
        '{"step_number": 2, "title": "Create your first project",'
        ' "action": "Click New Project, name it, invite teammates",'
        ' "narration": {"text": "Setting up a project in Linear takes only seconds.'
        ' Name it add your teammates and start organizing work with intuitive priorities.",'
        ' "tone_notes": "confident, highlight simplicity"},'
        ' "expected_url_pattern": "linear\\.app\\/projects\\/.*",'  # \. again
        ' "screenshot_hint": "Project creation dialog",'
        ' "duration_seconds": 35},'
        '{"step_number": 3, "title": "Track and manage issues",'
        ' "action": "Press C to create issue, set priority High, assign to teammate",'
        ' "narration": {"text": "Issues in Linear are powerful yet simple. Set priorities'
        ' assign teammates and track every detail from a beautiful keyboard-first interface.",'
        ' "tone_notes": "technical, show keyboard shortcut"},'
        ' "expected_url_pattern": "linear\\.app\\/.*\\/issues",'  # \. twice
        ' "screenshot_hint": "Issue detail with priority and assignee set",'
        ' "duration_seconds": 40}'
        '], "total_estimated_duration_seconds": 105}'
    )

    mock_client.chat_completion.side_effect = [
        _mock_hf_response(json.dumps(CALL1_DATA)),
        _mock_hf_response(bad_escape_call2),
    ]

    script, _ = generate_demo_script(LINEAR_PROFILE, DEFAULT_CONFIG)

    assert len(script.steps) == 3
    assert script.steps[0].title == "Sign up and onboard"
    # The sanitizer doubles the backslash so the stored value contains a literal backslash
    assert r"\." in script.steps[0].expected_url_pattern
