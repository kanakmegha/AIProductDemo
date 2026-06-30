"""LLM-powered demo script generation using HuggingFace InferenceClient."""
import json
import logging
import os
import re
import time

from huggingface_hub import InferenceClient

from backend.intelligence.models import ProductProfile
from backend.scriptgen.models import DemoScript, DemoStep, GenerationConfig, Narration
from backend.scriptgen.templates import (
    PLANNER_SYSTEM,
    STEP_REWRITER_SYSTEM,
    WRITER_SYSTEM,
    build_planning_prompt,
    build_step_rewrite_prompt,
    build_writing_prompt,
)

logger = logging.getLogger(__name__)

HF_TOKEN = os.getenv("HF_TOKEN")
HF_MODEL = os.getenv("HF_MODEL", "Qwen/Qwen2.5-72B-Instruct")
HF_FALLBACK_MODEL = os.getenv("HF_FALLBACK_MODEL", "Qwen/Qwen2.5-7B-Instruct")
HF_PROVIDER = os.getenv("HF_PROVIDER", "auto")

MARKDOWN_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)

# Characters that may legally follow a backslash in a JSON string value.
_JSON_ESCAPE_CHARS = frozenset('"\\\/bfnrtu')

MIN_STEPS = 3
MAX_STEPS = 7
MIN_DURATION_S = 90
MAX_DURATION_S = 300


class GenerationError(Exception):
    """Raised when LLM-based script generation fails."""


def _chat_completion(model: str, system: str, user_prompt: str, temperature: float, max_tokens: int):
    """Issue a single chat completion call via HuggingFace Inference Providers."""
    client = InferenceClient(provider=HF_PROVIDER, api_key=HF_TOKEN)
    return client.chat_completion(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
    )


def _fix_json_escapes(text: str) -> str:
    """
    Fix invalid JSON escape sequences produced by LLMs.

    LLMs often embed regex patterns directly inside JSON string values, e.g.
    ``"expected_url_pattern": "linear\\.app/.*"``, where ``\\.`` is not a valid
    JSON escape.  A naïve regex substitution that doubles every backslash would
    corrupt already-valid two-char sequences such as ``\\\\`` (escaped backslash)
    or ``\\"`` (escaped quote), because it would double the *second* ``\\`` of
    ``\\\\``, which is itself followed by a non-escape char.

    This function walks the text one character at a time so it can treat any
    valid two-char escape (``\\X`` where ``X`` in ``"\\\\/bfnrtu"``) as a single
    unit and leave it untouched, while still doubling a lone ``\\X`` where ``X``
    is *not* a legal JSON escape character.

    Examples
    --------
    ``linear\\.app/.*``  (invalid: lone ``\\.``)  →  ``linear\\\\.app/.*``  ✓
    ``linear\\\\.app/.*`` (valid: ``\\\\`` then ``.``) →  unchanged            ✓
    """
    parts: list[str] = []
    i = 0
    while i < len(text):
        ch = text[i]
        if ch != '\\':
            parts.append(ch)
            i += 1
            continue

        nxt = text[i + 1] if i + 1 < len(text) else ''
        if nxt in _JSON_ESCAPE_CHARS:
            # Valid two-char JSON escape — keep both and skip past them as a unit.
            parts.append(ch)
            parts.append(nxt)
            i += 2
        else:
            # Lone backslash with an invalid following character — double the
            # backslash to make it a valid ``\\`` escape.  Leave ``nxt`` for the
            # next iteration so it is processed independently (it may itself be
            # another backslash that needs the same treatment).
            parts.append('\\')
            parts.append('\\')
            i += 1
    return ''.join(parts)


def _parse_json(text: str) -> dict:
    """
    Parse JSON from an LLM response with two-pass sanitization.

    Pass 1 — fast path: try json.loads directly (valid JSON skips all overhead).
    Pass 2 — sanitize: strip markdown fences, then run ``_fix_json_escapes`` to
             repair lone backslashes that are invalid in JSON (e.g. ``\\.`` or
             ``\\-`` from regex patterns embedded in url/pattern fields).
             Re-raise the final JSONDecodeError unchanged so the caller can fall
             back to the secondary model.
    """
    # Pass 1 — fast path
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        logger.warning(
            "Direct JSON parse failed (%s). Attempting sanitization. "
            "Raw response (first 500 chars):\n%.500s",
            exc, text,
        )

    # Pass 2 — strip markdown fences + fix invalid backslash escapes
    cleaned = MARKDOWN_FENCE_RE.sub("", text.strip()).strip()
    sanitized = _fix_json_escapes(cleaned)
    return json.loads(sanitized)  # let JSONDecodeError propagate if still broken


def _call_with_fallback(
    call_num: int,
    system: str,
    user_prompt: str,
    temperature: float,
    max_tokens: int,
) -> dict:
    """Execute one LLM call, retrying with HF_FALLBACK_MODEL on any failure."""
    start = time.monotonic()
    model = HF_MODEL
    try:
        response = _chat_completion(model, system, user_prompt, temperature, max_tokens)
    except Exception as exc:
        logger.warning("Call %d: primary model %s failed (%s). Retrying with fallback.", call_num, model, exc)
        model = HF_FALLBACK_MODEL
        response = _chat_completion(model, system, user_prompt, temperature, max_tokens)

    duration = time.monotonic() - start
    text = response.choices[0].message.content.strip()
    logger.info(
        "LLM call=%d model=%s input_chars=%d output_len=%d duration=%.3fs",
        call_num, model, len(user_prompt), len(text), duration,
    )
    logger.debug("LLM call=%d raw response:\n%s", call_num, text)
    return _parse_json(text)


def _build_step(raw: dict) -> DemoStep:
    """Construct a DemoStep from a raw LLM dict, computing word_count from narration text."""
    narration_raw = raw.get("narration", {})
    text = narration_raw.get("text", "")
    narration = Narration(
        text=text,
        tone_notes=narration_raw.get("tone_notes", ""),
        word_count=len(text.split()),
    )
    return DemoStep(
        step_number=int(raw["step_number"]),
        title=raw["title"],
        action=raw["action"],
        narration=narration,
        expected_url_pattern=raw.get("expected_url_pattern", ".*"),
        screenshot_hint=raw.get("screenshot_hint", ""),
        duration_seconds=int(raw["duration_seconds"]),
    )


def _plan_journey(profile: ProductProfile, config: GenerationConfig) -> dict:
    """Call 1 — Planning: identify key features and design the user journey."""
    prompt = build_planning_prompt(
        profile_json=profile.model_dump_json(indent=2),
        max_steps=config.max_steps,
    )
    return _call_with_fallback(1, PLANNER_SYSTEM, prompt, temperature=0.3, max_tokens=512)


def _write_script(journey: dict, profile: ProductProfile, config: GenerationConfig) -> dict:
    """Call 2 — Writing: produce the full demo script from the planned journey."""
    prompt = build_writing_prompt(
        journey_json=json.dumps(journey, indent=2),
        tone=config.tone,
        audience=config.target_audience,
        product_name=profile.product_name or profile.url,
    )
    return _call_with_fallback(2, WRITER_SYSTEM, prompt, temperature=0.2, max_tokens=2048)


def _write_single_step(
    step_number: int,
    journey: dict,
    profile: ProductProfile,
    config: GenerationConfig,
    comment: str = "",
) -> dict:
    """Call 2 variant — rewrite one step using the stored journey (no re-planning)."""
    prompt = build_step_rewrite_prompt(
        step_number=step_number,
        journey_json=json.dumps(journey, indent=2),
        tone=config.tone,
        audience=config.target_audience,
        product_name=profile.product_name or profile.url,
        comment=comment,
    )
    return _call_with_fallback(2, STEP_REWRITER_SYSTEM, prompt, temperature=0.2, max_tokens=512)


def generate_demo_script(
    profile: ProductProfile,
    config: GenerationConfig,
) -> tuple[DemoScript, dict]:
    """
    Run the two-call generation pipeline.
    Returns (DemoScript, journey_dict) so the caller can cache the journey for future
    step-level regenerations without needing to re-run Call 1.
    """
    try:
        journey = _plan_journey(profile, config)
    except Exception as exc:
        raise GenerationError(f"Planning call failed: {exc}") from exc

    try:
        script_data = _write_script(journey, profile, config)
    except Exception as exc:
        raise GenerationError(f"Writing call failed: {exc}") from exc

    raw_steps = script_data.get("steps", [])
    raw_steps = raw_steps[:MAX_STEPS]
    if len(raw_steps) < MIN_STEPS:
        raise GenerationError(
            f"LLM returned only {len(raw_steps)} step(s); minimum is {MIN_STEPS}."
        )

    steps = [_build_step(r) for r in raw_steps]
    total = script_data.get("total_estimated_duration_seconds") or sum(
        s.duration_seconds for s in steps
    )

    if not (MIN_DURATION_S <= total <= MAX_DURATION_S):
        logger.warning(
            "Total duration %ds is outside expected range %d–%ds",
            total, MIN_DURATION_S, MAX_DURATION_S,
        )

    script = DemoScript(
        product_profile=profile,
        steps=steps,
        total_estimated_duration_seconds=total,
        target_audience=config.target_audience,
        tone=config.tone,
    )
    return script, journey


def regenerate_step(
    script: DemoScript,
    step_number: int,
    journey: dict,
    comment: str = "",
) -> DemoStep:
    """Regenerate a single step using Call 2 only — no re-planning (Call 1) needed."""
    config = GenerationConfig(tone=script.tone, target_audience=script.target_audience)
    try:
        step_data = _write_single_step(
            step_number, journey, script.product_profile, config, comment
        )
    except Exception as exc:
        raise GenerationError(f"Step {step_number} regeneration failed: {exc}") from exc
    return _build_step(step_data)
