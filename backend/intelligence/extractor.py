"""LLM-powered feature/use-case extraction from a scraped page."""
import json
import logging
import os
import re
import time

from huggingface_hub import InferenceClient

from backend.intelligence.models import ProductProfile
from backend.intelligence.scraper import ScrapedPage
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

HF_TOKEN = os.getenv("HF_TOKEN")
HF_MODEL = os.getenv("HF_MODEL", "Qwen/Qwen2.5-7B-Instruct")
HF_FALLBACK_MODEL = os.getenv("HF_FALLBACK_MODEL", "meta-llama/Meta-Llama-3.1-8B-Instruct")
# Route through HuggingFace Inference Providers; "auto" picks an available provider
# for the requested model (replaces the deprecated serverless inference endpoint).
HF_PROVIDER = os.getenv("HF_PROVIDER", "auto")

MAX_PARAGRAPHS_FULL = 40
MAX_PARAGRAPHS_TRUNCATED = 10
MAX_PROMPT_CHARS = 6000

SYSTEM_PROMPT = (
    "You are a precise extraction engine for SaaS product websites. "
    "Always respond with a single JSON object and nothing else."
)

MARKDOWN_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)

PROFILE_FIELDS = [
    "product_name",
    "tagline",
    "target_audience",
    "core_features",
    "use_cases",
    "pricing_tiers",
    "raw_text_summary",
]


class ExtractionError(Exception):
    """Raised when the LLM extraction step fails, optionally carrying partial data."""

    def __init__(self, reason: str, partial_profile: ProductProfile | None = None):
        """Store the failure reason and any partially built ProductProfile."""
        self.reason = reason
        self.partial_profile = partial_profile
        super().__init__(reason)


def _build_prompt(page: ScrapedPage, truncated: bool = False) -> str:
    """Build the extraction prompt from a ScrapedPage's content."""
    paragraph_limit = MAX_PARAGRAPHS_TRUNCATED if truncated else MAX_PARAGRAPHS_FULL
    paragraphs = "\n".join(page.paragraphs[:paragraph_limit])
    headings = "\n".join(page.headings)

    return f"""You are analyzing a SaaS product's website to extract a structured profile.

URL: {page.url}
Title: {page.title}
Meta description: {page.meta_description}
OG title: {page.og_title}
OG description: {page.og_description}

Headings:
{headings}

Body paragraphs:
{paragraphs}

Return ONLY a JSON object (no markdown fences, no commentary) with exactly these keys:
- product_name: string
- tagline: string
- target_audience: array of strings
- core_features: array of objects with keys "name", "description", "importance" (one of "core", "secondary", "tertiary")
- use_cases: array of strings
- pricing_tiers: array of strings
- raw_text_summary: string, a 2-3 sentence summary of what the product does

If a field cannot be determined, use an empty string or empty array."""


def _strip_markdown_fences(text: str) -> str:
    """Remove leading/trailing markdown code fences (e.g. ```json ... ```) from a response."""
    return MARKDOWN_FENCE_RE.sub("", text.strip()).strip()


def _parse_json_response(text: str) -> dict:
    """Parse a JSON object from the LLM response, retrying once with markdown fences stripped."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return json.loads(_strip_markdown_fences(text))


def _chat_completion(model: str, prompt: str):
    """Issue a single chat completion call against the given HuggingFace model."""
    client = InferenceClient(provider=HF_PROVIDER, api_key=HF_TOKEN)
    return client.chat_completion(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        max_tokens=1024,
        temperature=0.1,
    )


def _call_llm(prompt: str) -> dict:
    """Call the HuggingFace Inference API, falling back to a smaller model on failure."""
    truncated_prompt = prompt[:MAX_PROMPT_CHARS]

    start = time.monotonic()
    model = HF_MODEL
    try:
        response = _chat_completion(model, truncated_prompt)
    except Exception as exc:
        logger.warning("Primary model %s failed: %s. Falling back to %s.", HF_MODEL, exc, HF_FALLBACK_MODEL)
        model = HF_FALLBACK_MODEL
        response = _chat_completion(model, truncated_prompt)
    duration = time.monotonic() - start

    text = response.choices[0].message.content.strip()
    logger.info(
        "LLM call model=%s input_chars=%d output_len=%d duration=%.3fs",
        model, len(truncated_prompt), len(text), duration,
    )

    return _parse_json_response(text)


def _compute_confidence(data: dict) -> float:
    """Compute a confidence score based on the proportion of populated profile fields."""
    populated = 0
    for field_name in PROFILE_FIELDS:
        value = data.get(field_name)
        if isinstance(value, str) and value.strip():
            populated += 1
        elif isinstance(value, list) and len(value) > 0:
            populated += 1
    return round(populated / len(PROFILE_FIELDS), 2)


def extract_profile(page: ScrapedPage) -> ProductProfile:
    """Extract a ProductProfile from a ScrapedPage using a single LLM call, retrying once on failure."""
    try:
        data = _call_llm(_build_prompt(page))
    except Exception as exc:
        logger.warning("Primary LLM extraction failed for %s: %s. Retrying with truncated context.", page.url, exc)
        try:
            data = _call_llm(_build_prompt(page, truncated=True))
        except Exception as retry_exc:
            raise ExtractionError(f"LLM extraction failed after retry: {retry_exc}") from retry_exc

    confidence_score = _compute_confidence(data)

    try:
        return ProductProfile(
            url=page.url,
            product_name=data.get("product_name", ""),
            tagline=data.get("tagline", ""),
            target_audience=data.get("target_audience", []),
            core_features=data.get("core_features", []),
            use_cases=data.get("use_cases", []),
            pricing_tiers=data.get("pricing_tiers", []),
            screenshots_found=page.image_urls,
            raw_text_summary=data.get("raw_text_summary", ""),
            confidence_score=confidence_score,
        )
    except Exception as exc:
        raise ExtractionError(f"Failed to build ProductProfile: {exc}") from exc
