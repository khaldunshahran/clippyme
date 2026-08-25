"""AI-powered speaker identification, trending hashtags, and caption generation.

Uses Gemini to analyze a clip's transcript, source metadata, and dialogue context
to extract the speaker's identity, viral title, relevant trending hashtags, and
rich social media descriptions/captions for YouTube Shorts, TikTok, and Instagram.
"""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

from google import genai
from google.genai import types

from clippyme.domain.errors import ValidationError
from clippyme.pipeline.gemini_service import get_auxiliary_gemini_model

logger = logging.getLogger(__name__)

METADATA_PROMPT_TEMPLATE = """You are a senior social media strategist and YouTube Shorts / TikTok virality expert.
Analyze this video clip's transcript and video source context to generate high-performing publication metadata.

VIDEO TITLE / SOURCE: {video_title}
SOURCE CHANNEL / UPLOADER: {uploader}
CLIP TIMESTAMPS: {start:.2f}s to {end:.2f}s (duration: {duration:.1f}s)

CLIP TRANSCRIPT:
\"\"\"
{clip_transcript}
\"\"\"

ADDITIONAL CONTEXT / INSTRUCTIONS:
{instructions}

YOUR TASKS:
1. IDENTIFY SPEAKER / SUBJECT: Detect who is speaking or who is the main subject in this clip from the dialogue and video title (e.g. 'John Kiriakou', 'Lex Fridman', 'Andrew Huberman', etc.). If uncertain, provide a descriptive role (e.g. 'Former CIA Officer', 'Guest') or leave blank.
2. TITLE: Create a scroll-stopping, high-converting YouTube Shorts title (max 100 chars, engagement-first bait, grounded in what happens).
3. HASHTAGS: Provide 5 to 8 high-traffic, trending, and topical hashtags formatted with '#' (e.g. ["#shorts", "#cia", "#johnkiriakou", "#whistleblower", "#podcast", "#trending", "#history"]).
4. CAPTION / DESCRIPTION: Write an engaging 2-4 sentence caption. Include:
   - Strong curiosity hook or key insight.
   - Mention the speaker / context naturally (e.g. 'Former CIA officer John Kiriakou reveals...').
   - An open question or discussion starter to provoke comments.
   - Space and include the generated hashtags at the bottom.

OUTPUT FORMAT:
Emit ONLY valid JSON:
{{
  "speaker_name": "<Identified speaker name or role, or empty string if unknown>",
  "title": "<High-converting title, max 100 characters>",
  "hashtags": ["#shorts", "#trending", "#tag1", "#tag2", "#tag3", "#tag4"],
  "caption": "<Complete formatted caption with hook, speaker mention, conversation question, and hashtags at the end>"
}}
"""


def generate_clip_metadata(
    *,
    api_key: str,
    model: Optional[str] = None,
    clip_transcript: str,
    start: float,
    end: float,
    video_title: str = "",
    uploader: str = "",
    instructions: str = "",
) -> Dict[str, Any]:
    """Call Gemini to generate speaker name, title, hashtags, and caption for a clip."""
    if not api_key:
        raise ValidationError("Gemini API key is required")

    duration = max(0.0, end - start)
    prompt = METADATA_PROMPT_TEMPLATE.format(
        video_title=video_title or "Unknown Source",
        uploader=uploader or "Unknown Channel",
        start=start,
        end=end,
        duration=duration,
        clip_transcript=clip_transcript or "(No dialogue detected)",
        instructions=instructions or "Generate optimal viral metadata for social publishing.",
    )

    client = genai.Client(api_key=api_key)

    # Prefer lightweight auxiliary models for cost efficiency
    primary_lite = model or get_auxiliary_gemini_model()
    candidate_models = [
        primary_lite,
        "gemini-2.5-flash-lite",
        "gemini-2.5-flash",
        "gemini-3.5-flash",
    ]
    # Deduplicate while preserving order
    seen = set()
    models_to_try = [m for m in candidate_models if m and not (m in seen or seen.add(m))]

    last_exc = None
    for target_model in models_to_try:
        try:
            logger.info("Generating AI metadata with %s for clip (%.1fs - %.1fs)", target_model, start, end)
            response = client.models.generate_content(
                model=target_model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.3,
                ),
            )
            raw_text = response.text or ""
            parsed = json.loads(raw_text)
            if isinstance(parsed, dict):
                # Clean up hashtags to ensure all start with '#'
                raw_tags = parsed.get("hashtags") or []
                cleaned_tags = []
                if isinstance(raw_tags, list):
                    for tag in raw_tags:
                        if isinstance(tag, str) and tag.strip():
                            t = tag.strip()
                            cleaned_tags.append(t if t.startswith("#") else f"#{t}")
                if not cleaned_tags:
                    cleaned_tags = ["#shorts", "#trending", "#viral"]

                title = (parsed.get("title") or "").strip()[:100]
                speaker = (parsed.get("speaker_name") or "").strip()
                caption = (parsed.get("caption") or "").strip()

                # If caption didn't include hashtags, append them nicely
                tags_str = " ".join(cleaned_tags)
                if tags_str and tags_str not in caption and not any(t in caption for t in cleaned_tags):
                    caption = f"{caption}\n\n{tags_str}".strip()

                return {
                    "speaker_name": speaker,
                    "title": title,
                    "hashtags": cleaned_tags,
                    "caption": caption,
                }
        except Exception as exc:
            logger.warning("generate_clip_metadata with %s failed: %s", target_model, exc)
            last_exc = exc
            continue

    # Fallback if Gemini fails
    logger.error("All Gemini models failed for metadata generation: %s", last_exc)
    fallback_tags = ["#shorts", "#trending", "#viral"]
    return {
        "speaker_name": "",
        "title": video_title[:100] if video_title else "Viral Moment",
        "hashtags": fallback_tags,
        "caption": f"{video_title}\n\n{' '.join(fallback_tags)}" if video_title else "Watch this moment!\n\n#shorts #trending",
    }
