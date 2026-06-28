"""Tests for backend.intelligence.extractor."""
import json
from unittest.mock import MagicMock, patch

import pytest

from backend.intelligence.extractor import ExtractionError, extract_profile
from backend.intelligence.scraper import ScrapedPage

FULL_PAGE = ScrapedPage(
    url="https://linear.app",
    title="Linear",
    meta_description="Issue tracking for modern teams",
    og_title="Linear",
    og_description="Plan and build products",
    headings=["Linear", "Built for speed"],
    paragraphs=["Linear is a tool for managing software projects."],
    image_urls=["https://linear.app/shot.png"],
)

VALID_RESPONSE_DATA = {
    "product_name": "Linear",
    "tagline": "The issue tracker built for modern teams",
    "target_audience": ["software teams", "startups"],
    "core_features": [
        {"name": "Issue tracking", "description": "Track issues across teams", "importance": "core"}
    ],
    "use_cases": ["sprint planning", "bug tracking"],
    "pricing_tiers": ["Free", "Plus", "Business"],
    "raw_text_summary": "Linear is an issue tracking tool for modern software teams.",
}


def _mock_hf_response(content: str):
    """Build a fake huggingface_hub chat_completion response containing the given text content."""
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = content
    return mock_response


@patch("backend.intelligence.extractor.InferenceClient")
def test_valid_llm_response_parsed_into_profile(mock_client_cls):
    """A valid JSON response from the LLM should produce a populated ProductProfile."""
    mock_client = MagicMock()
    mock_client.chat_completion.return_value = _mock_hf_response(json.dumps(VALID_RESPONSE_DATA))
    mock_client_cls.return_value = mock_client

    profile = extract_profile(FULL_PAGE)

    assert profile.product_name == "Linear"
    assert profile.tagline == "The issue tracker built for modern teams"
    assert len(profile.core_features) == 1
    assert profile.core_features[0].importance == "core"
    assert profile.screenshots_found == ["https://linear.app/shot.png"]


@patch("backend.intelligence.extractor.InferenceClient")
def test_markdown_fenced_json_is_stripped_and_parsed(mock_client_cls):
    """A response wrapped in markdown code fences should be stripped and parsed correctly."""
    mock_client = MagicMock()
    fenced_content = f"```json\n{json.dumps(VALID_RESPONSE_DATA)}\n```"
    mock_client.chat_completion.return_value = _mock_hf_response(fenced_content)
    mock_client_cls.return_value = mock_client

    profile = extract_profile(FULL_PAGE)

    assert profile.product_name == "Linear"
    assert mock_client.chat_completion.call_count == 1


@patch("backend.intelligence.extractor.InferenceClient")
def test_malformed_json_triggers_retry_with_truncated_context(mock_client_cls):
    """Malformed JSON on the first call should trigger a retry with a truncated prompt."""
    mock_client = MagicMock()
    mock_client.chat_completion.side_effect = [
        _mock_hf_response("not valid json{{{"),
        _mock_hf_response(json.dumps(VALID_RESPONSE_DATA)),
    ]
    mock_client_cls.return_value = mock_client

    profile = extract_profile(FULL_PAGE)

    assert profile.product_name == "Linear"
    assert mock_client.chat_completion.call_count == 2


@patch("backend.intelligence.extractor.InferenceClient")
def test_confidence_score_full_vs_partial(mock_client_cls):
    """Confidence score should be 1.0 when all fields populated and lower when fields are missing."""
    mock_client = MagicMock()
    mock_client.chat_completion.return_value = _mock_hf_response(json.dumps(VALID_RESPONSE_DATA))
    mock_client_cls.return_value = mock_client

    full_profile = extract_profile(FULL_PAGE)
    assert full_profile.confidence_score == 1.0

    partial_data = dict(VALID_RESPONSE_DATA)
    partial_data["tagline"] = ""
    partial_data["pricing_tiers"] = []
    partial_data["use_cases"] = []
    mock_client.chat_completion.return_value = _mock_hf_response(json.dumps(partial_data))

    partial_profile = extract_profile(FULL_PAGE)
    assert partial_profile.confidence_score < 1.0


@patch("backend.intelligence.extractor.InferenceClient")
def test_primary_model_failure_falls_back_to_secondary_model(mock_client_cls):
    """If the primary model call raises, the extractor should retry with the fallback model."""
    mock_client = MagicMock()
    mock_client.chat_completion.side_effect = [
        RuntimeError("primary model timed out"),
        _mock_hf_response(json.dumps(VALID_RESPONSE_DATA)),
    ]
    mock_client_cls.return_value = mock_client

    profile = extract_profile(FULL_PAGE)

    assert profile.product_name == "Linear"
    assert mock_client_cls.call_count == 2


@patch("backend.intelligence.extractor.InferenceClient")
def test_extraction_error_raised_after_failed_retry(mock_client_cls):
    """If both the primary call and the retry fail, ExtractionError should be raised."""
    mock_client = MagicMock()
    mock_client.chat_completion.side_effect = RuntimeError("LLM unavailable")
    mock_client_cls.return_value = mock_client

    with pytest.raises(ExtractionError):
        extract_profile(FULL_PAGE)
