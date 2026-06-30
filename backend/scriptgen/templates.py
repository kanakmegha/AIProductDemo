"""Prompt templates for the Demo Script Generator. No business logic — strings only."""

PLANNER_SYSTEM = (
    "You are a product demo strategist. "
    "Identify the most impactful features and design a compelling, story-driven user journey."
)

WRITER_SYSTEM = (
    "You are a product demo scriptwriter. "
    "Write concise, engaging narration and precise browser actions. "
    "Always respond with a single valid JSON object and nothing else."
)

STEP_REWRITER_SYSTEM = (
    "You are a product demo scriptwriter. "
    "Rewrite a single demo step based on reviewer feedback. "
    "Always respond with a single valid JSON object and nothing else."
)


def build_planning_prompt(profile_json: str, max_steps: int) -> str:
    return f"""Analyze this SaaS product profile and select the {min(5, max_steps)} most impactful features to demo.
Design a logical user journey that tells a compelling story — not just a feature list.

Product Profile:
{profile_json}

Respond with ONLY a valid JSON object (no markdown fences, no commentary):
{{
  "features": ["feature name 1", "feature name 2"],
  "journey": [
    {{"step": 1, "title": "...", "action_summary": "...", "why": "why this step matters"}},
    {{"step": 2, "title": "...", "action_summary": "...", "why": "..."}}
  ]
}}"""


def build_writing_prompt(
    journey_json: str,
    tone: str,
    audience: str,
    product_name: str,
) -> str:
    return f"""Write a complete demo script for {product_name} based on the planned journey below.

Audience: {audience}
Tone: {tone}

Planned Journey:
{journey_json}

Requirements:
- narration.text: 20-50 words per step at a conversational pace (~130 words/min)
- action: specific browser instructions (URL to navigate to, exact element to click, text to type)
- duration_seconds: realistic per step; total MUST be 90-300 seconds
- expected_url_pattern: a regex string the browser URL should match after completing this step
- screenshot_hint: one sentence describing what should be visible on screen

Respond with ONLY a valid JSON object (no markdown fences, no commentary):
{{
  "steps": [
    {{
      "step_number": 1,
      "title": "...",
      "action": "...",
      "narration": {{
        "text": "20-50 word narration here",
        "tone_notes": "delivery guidance, e.g. enthusiastic, pause after key point"
      }},
      "expected_url_pattern": "regex string",
      "screenshot_hint": "...",
      "duration_seconds": 30
    }}
  ],
  "total_estimated_duration_seconds": 120
}}"""


def build_step_rewrite_prompt(
    step_number: int,
    journey_json: str,
    tone: str,
    audience: str,
    product_name: str,
    comment: str,
) -> str:
    feedback_note = comment if comment.strip() else "Make it more compelling and specific."
    return f"""Rewrite step {step_number} of the {product_name} demo script.

Audience: {audience}
Tone: {tone}
Reviewer feedback: {feedback_note}

Journey context:
{journey_json}

Requirements:
- narration.text: 20-50 words, conversational pace (~130 words/min)
- action: specific browser instructions

Respond with ONLY a valid JSON object for this single step (no markdown fences):
{{
  "step_number": {step_number},
  "title": "...",
  "action": "...",
  "narration": {{
    "text": "20-50 word narration",
    "tone_notes": "delivery guidance"
  }},
  "expected_url_pattern": "regex string",
  "screenshot_hint": "...",
  "duration_seconds": 30
}}"""
