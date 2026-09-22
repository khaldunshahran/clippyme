"""AI-powered speaker identification, platform captions, and situational metadata generation.

Leverages full Video Brain knowledge (source context, creator bio, description,
top viewer comments, and community debates from audience_intel.json) to generate
high-performing, platform-specific publication copy for TikTok, Instagram Reels,
and YouTube Shorts.
"""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

from google import genai
from google.genai import types

from clippyme.domain.errors import ClippyMeError, NotFoundError, ValidationError
from clippyme.domain.job_artifacts import load_job_metadata, save_job_metadata
from clippyme.domain.smartcut import clip_transcript_segments
from clippyme.pipeline.audience_intel import format_audience_intel_prompt, load_audience_intel
from clippyme.pipeline.gemini_service import get_auxiliary_gemini_model
from clippyme.storage.config_store import load_persistent_config

logger = logging.getLogger(__name__)

SINGLE_CLIP_PROMPT_TEMPLATE = """You are an elite short-form video copywriter and social media strategist.
Analyze this video clip's transcript, along with the source video's audience intelligence, creator background, and viewer reactions to generate high-performing publication metadata tailored specifically to each platform.

{brain_context}

SOURCE VIDEO TITLE: {video_title}
CREATOR / CHANNEL: {uploader}
CLIP TIMESTAMPS: {start:.2f}s to {end:.2f}s (duration: {duration:.1f}s)

CLIP TRANSCRIPT:
\"\"\"
{clip_transcript}
\"\"\"

ADDITIONAL INSTRUCTIONS:
{instructions}

CRITICAL SITUATIONAL AWARENESS:
- Ground all copy in who is speaking, the true context of the discussion, and the community's emotional reaction.
- Draw directly upon what viewers resonated with in the comments or creator description.

PLATFORM REQUIREMENTS:
1. TikTok:
   - Punchy scroll-stopping opener, curiosity gap, conversational or controversial question.
   - 3 to 5 trending tags (e.g. #fyp, #trending, #viral, plus topic tags).
2. Instagram Reels:
   - Clean visual opener before the "...more" fold, storytelling context, save/share prompt ("Drop your thoughts below 👇" or "Save for later").
   - 4 to 6 clean topic tags.
3. YouTube Shorts:
   - High-CTR viral title (maximum 100 characters, ideally under 65).
   - Structured 2-3 sentence description mentioning the speaker, core insight, and open question for comments.
   - Include #shorts and 3 to 5 relevant tags.

OUTPUT FORMAT:
Emit ONLY valid JSON:
{{
  "speaker_name": "<Identified speaker name or role (e.g. 'John Kiriakou' or 'Former CIA Officer')>",
  "title": "<High-CTR YouTube Shorts title, max 100 chars>",
  "hashtags": ["#shorts", "#trending", "#topic1", "#topic2"],
  "caption": "<Universal fallback caption with hook and tags>",
  "platforms": {{
    "tiktok": {{
      "caption": "<TikTok-optimized caption with hook, debate question, and tags>",
      "hashtags": ["#fyp", "#trending", "#tag1", "#tag2"]
    }},
    "instagram": {{
      "caption": "<Instagram-optimized caption with narrative context, CTA, and clean tags>",
      "hashtags": ["#reels", "#tag1", "#tag2"]
    }},
    "youtube": {{
      "title": "<YouTube Shorts title, max 100 chars>",
      "description": "<Structured YouTube description with context, question, and #shorts tags>",
      "hashtags": ["#shorts", "#tag1", "#tag2"]
    }}
  }}
}}
"""

BATCH_CLIPS_PROMPT_TEMPLATE = """You are an elite short-form video copywriter and social media strategist.
Analyze the following {clip_count} extracted video clips from this source video.
Using the provided audience intelligence, creator background, and top viewer comments, generate high-converting, platform-specific publication copy for EVERY clip.

{brain_context}

SOURCE VIDEO TITLE: "{video_title}"
CREATOR / CHANNEL: "{uploader}"

CLIPS TO ANALYZE:
{clips_json}

ADDITIONAL INSTRUCTIONS:
{instructions}

CRITICAL SITUATIONAL AWARENESS:
- Ground each clip in the specific moment being discussed, who is speaking, and the emotional resonance proven in the audience comments.
- Give each clip a unique, non-generic title and hook.

PLATFORM REQUIREMENTS FOR EACH CLIP:
1. TikTok:
   - Punchy scroll-stopping opener, curiosity gap, conversational or controversial question.
   - 3 to 5 trending tags (e.g. #fyp, #trending, plus topic tags).
2. Instagram Reels:
   - Clean visual opener before the fold, storytelling context, save/share call-to-action ("Drop your thoughts below 👇", "Save for later").
   - 4 to 6 clean topic tags.
3. YouTube Shorts:
   - High-CTR viral title (maximum 100 characters, ideally under 65).
   - Structured 2-3 sentence description mentioning the speaker, core insight, and open discussion question.
   - Include #shorts and 3 to 5 relevant tags.

OUTPUT FORMAT:
Emit ONLY valid JSON:
{{
  "clips": [
    {{
      "clip_index": 1,
      "speaker_name": "<Speaker name or role>",
      "title": "<High-converting title, max 100 chars>",
      "hashtags": ["#shorts", "#trending", "#topic1", "#topic2"],
      "caption": "<Universal fallback caption>",
      "platforms": {{
        "tiktok": {{
          "caption": "<TikTok caption with hook, debate question, and tags>",
          "hashtags": ["#fyp", "#trending", "#tag1", "#tag2"]
        }},
        "instagram": {{
          "caption": "<Instagram caption with narrative context, CTA, and clean tags>",
          "hashtags": ["#reels", "#tag1", "#tag2"]
        }},
        "youtube": {{
          "title": "<YouTube Shorts title, max 100 chars>",
          "description": "<Structured YouTube description with context, question, and #shorts tags>",
          "hashtags": ["#shorts", "#tag1", "#tag2"]
        }}
      }}
    }}
  ]
}}
"""


def _clean_tags(tags: Any, default_tag: str = "#shorts") -> List[str]:
    """Clean and ensure a list of hashtags all start with '#'."""
    cleaned = []
    if isinstance(tags, list):
        for tag in tags:
            if isinstance(tag, str) and tag.strip():
                t = tag.strip()
                cleaned.append(t if t.startswith("#") else f"#{t}")
    if not cleaned:
        cleaned = [default_tag, "#trending", "#viral"]
    return cleaned


def _normalize_platform_dict(
    parsed: Dict[str, Any],
    fallback_title: str,
    fallback_speaker: str = "",
) -> Dict[str, Any]:
    """Normalize and validate platform captions from Gemini response."""
    speaker = (parsed.get("speaker_name") or fallback_speaker or "").strip()
    title = (parsed.get("title") or fallback_title or "Viral Moment").strip()[:100]
    hashtags = _clean_tags(parsed.get("hashtags"), "#shorts")

    platforms = parsed.get("platforms")
    if not isinstance(platforms, dict):
        platforms = {}

    # TikTok
    tiktok = platforms.get("tiktok") if isinstance(platforms.get("tiktok"), dict) else {}
    tiktok_tags = _clean_tags(tiktok.get("hashtags") or hashtags, "#fyp")
    tiktok_caption = (tiktok.get("caption") or "").strip()
    if not tiktok_caption:
        speaker_mention = f" featuring {speaker}" if speaker else ""
        tiktok_caption = f"{title}! What do you think about this?{speaker_mention}\n\n{' '.join(tiktok_tags)}"
    elif not any(t in tiktok_caption for t in tiktok_tags):
        tiktok_caption = f"{tiktok_caption}\n\n{' '.join(tiktok_tags)}"

    # Instagram
    instagram = platforms.get("instagram") if isinstance(platforms.get("instagram"), dict) else {}
    ig_tags = _clean_tags(instagram.get("hashtags") or hashtags, "#reels")
    ig_caption = (instagram.get("caption") or "").strip()
    if not ig_caption:
        speaker_lead = f"{speaker}: " if speaker else ""
        ig_caption = f"{speaker_lead}{title}\n\nDrop your thoughts in the comments below 👇\n\n{' '.join(ig_tags)}"
    elif not any(t in ig_caption for t in ig_tags):
        ig_caption = f"{ig_caption}\n\n{' '.join(ig_tags)}"

    # YouTube Shorts
    youtube = platforms.get("youtube") if isinstance(platforms.get("youtube"), dict) else {}
    yt_tags = _clean_tags(youtube.get("hashtags") or hashtags, "#shorts")
    yt_title = (youtube.get("title") or title).strip()[:100]
    yt_desc = (youtube.get("description") or "").strip()
    if not yt_desc:
        speaker_ctx = f"Former CIA Officer {speaker}" if "kiriakou" in speaker.lower() else (f"Featuring {speaker}." if speaker else "")
        yt_desc = f"{yt_title}\n\n{speaker_ctx} Watch the full discussion and share your perspective below!\n\n{' '.join(yt_tags)}"
    elif not any(t in yt_desc for t in yt_tags):
        yt_desc = f"{yt_desc}\n\n{' '.join(yt_tags)}"

    universal_caption = (parsed.get("caption") or tiktok_caption or yt_desc).strip()

    return {
        "speaker_name": speaker,
        "title": title,
        "hashtags": hashtags,
        "caption": universal_caption,
        "platforms": {
            "tiktok": {
                "caption": tiktok_caption,
                "hashtags": tiktok_tags,
            },
            "instagram": {
                "caption": ig_caption,
                "hashtags": ig_tags,
            },
            "youtube": {
                "title": yt_title,
                "description": yt_desc,
                "hashtags": yt_tags,
            },
        },
    }


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
    audience_intel: Optional[Dict[str, Any]] = None,
    source_info: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Call Gemini to generate speaker name, title, hashtags, and platform captions for a single clip."""
    if not api_key:
        raise ValidationError("Gemini API key is required")

    duration = max(0.0, end - start)
    brain_context = format_audience_intel_prompt(audience_intel)

    prompt = SINGLE_CLIP_PROMPT_TEMPLATE.format(
        brain_context=brain_context,
        video_title=video_title or "Unknown Source",
        uploader=uploader or "Unknown Channel",
        start=start,
        end=end,
        duration=duration,
        clip_transcript=clip_transcript or "(No dialogue detected)",
        instructions=instructions or "Generate optimal viral metadata for social publishing.",
    )

    client = genai.Client(api_key=api_key)

    primary_lite = model or get_auxiliary_gemini_model()
    candidate_models = [
        primary_lite,
        "gemini-3.5-flash-lite",
        "gemini-3.1-flash-lite",
        "gemini-3.5-flash",
        "gemini-3.6-flash",
    ]
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
                return _normalize_platform_dict(parsed, fallback_title=video_title)
        except Exception as exc:
            logger.warning("generate_clip_metadata with %s failed: %s", target_model, exc)
            last_exc = exc
            continue

    logger.error("All Gemini models failed for metadata generation: %s", last_exc)
    fallback_tags = ["#shorts", "#trending", "#viral"]
    fallback_title = video_title[:100] if video_title else "Viral Moment"
    return _normalize_platform_dict(
        {"title": fallback_title, "hashtags": fallback_tags},
        fallback_title=fallback_title,
    )


def generate_all_clips_metadata(
    job_id: str,
    output_dir: str = "output",
    api_key: Optional[str] = None,
    force: bool = False,
) -> Dict[str, Any]:
    """Generate platform-specific captions and descriptions for all clips in a job.

    Ingests the job's audience_intel.json, extracts transcript segments for each
    clip, and batch-queries Gemini with model fallback. Atomically updates
    *_metadata.json on disk.
    """
    import glob

    try:
        metadata_path, metadata = load_job_metadata(job_id, output_dir)
    except FileNotFoundError:
        # Check if output_dir is already the job directory itself
        direct_matches = glob.glob(os.path.join(output_dir, "*_metadata.json"))
        if direct_matches:
            metadata_path = max(direct_matches, key=os.path.getmtime)
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        else:
            raise NotFoundError(f"Metadata not found for job {job_id}")

    shorts: List[dict] = metadata.get("shorts", [])
    if not shorts:
        return {"success": True, "job_id": job_id, "clips_count": 0, "message": "No clips to generate metadata for"}

    # Check if all clips already have complete platform captions
    if not force and all(isinstance(c.get("platforms"), dict) and c.get("platforms").get("tiktok") for c in shorts):
        logger.info("Job %s already has complete platform metadata; skipping generation", job_id)
        return {"success": True, "job_id": job_id, "clips_count": len(shorts), "cached": True}

    job_folder = os.path.dirname(metadata_path)
    audience_intel = load_audience_intel(job_folder)
    source_info = metadata.get("source_info") or {}

    cfg = load_persistent_config() or {}
    active_key = api_key or cfg.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not active_key:
        logger.warning("GEMINI_API_KEY not configured for auto metadata generation on job %s", job_id)
        return {"success": False, "error": "GEMINI_API_KEY not configured"}

    video_title = (
        audience_intel.get("title")
        if audience_intel
        else (source_info.get("title") or metadata.get("title") or metadata.get("video_title") or "Video Clips")
    )
    uploader = (
        audience_intel.get("channel")
        if audience_intel
        else (source_info.get("channel") or source_info.get("uploader") or metadata.get("uploader") or "")
    )

    # Extract segments
    segments = metadata.get("transcript", {}).get("segments", [])
    if not segments:
        checkpoint_tr = os.path.join(job_folder, ".clippyme_checkpoint", "transcript.json")
        if os.path.exists(checkpoint_tr):
            try:
                with open(checkpoint_tr, "r", encoding="utf-8") as f:
                    checkpoint_data = json.load(f)
                    segments = checkpoint_data.get("segments", [])
            except Exception as e:
                logger.warning("Could not read checkpoint transcript: %s", e)

    clips_payload = []
    for i, clip in enumerate(shorts):
        c_start = float(clip.get("start", 0.0))
        c_end = float(clip.get("end", 0.0))
        if segments:
            matched = clip_transcript_segments({"segments": segments}, c_start, c_end)
            c_text = " ".join(s.get("text", "") for s in matched if s.get("text"))
        else:
            c_text = clip.get("hook", "") or clip.get("video_title_for_youtube_short", "")

        clips_payload.append({
            "clip_index": i + 1,
            "start": round(c_start, 1),
            "end": round(c_end, 1),
            "duration": round(c_end - c_start, 1),
            "existing_title": clip.get("video_title_for_youtube_short") or clip.get("title") or "",
            "transcript_snippet": c_text[:800],
        })

    brain_context = format_audience_intel_prompt(audience_intel)
    prompt = BATCH_CLIPS_PROMPT_TEMPLATE.format(
        clip_count=len(clips_payload),
        brain_context=brain_context,
        video_title=video_title,
        uploader=uploader,
        clips_json=json.dumps(clips_payload, indent=2),
        instructions="Ensure captions are grounded in the topic, speaker, and community reactions.",
    )

    client = genai.Client(api_key=active_key)
    target_model = cfg.get("GEMINI_LITE_MODEL") or get_auxiliary_gemini_model()
    candidate_models = [
        target_model,
        "gemini-3.5-flash-lite",
        "gemini-3.1-flash-lite",
        "gemini-3.5-flash",
        "gemini-3.6-flash",
    ]
    seen = set()
    models_to_try = [m for m in candidate_models if m and not (m in seen or seen.add(m))]

    parsed_clips_map: Dict[int, Dict[str, Any]] = {}
    last_exc = None

    for model_name in models_to_try:
        try:
            logger.info("Batch generating platform metadata for %d clips in job %s using %s", len(shorts), job_id, model_name)
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.3,
                ),
            )
            raw_text = response.text or ""
            data = json.loads(raw_text)
            if isinstance(data, dict) and isinstance(data.get("clips"), list):
                for item in data["clips"]:
                    if isinstance(item, dict) and "clip_index" in item:
                        parsed_clips_map[int(item["clip_index"])] = item
                if len(parsed_clips_map) >= max(1, len(shorts) // 2):
                    break
        except Exception as exc:
            logger.warning("Batch metadata generation failed with %s: %s", model_name, exc)
            last_exc = exc
            continue

    # Apply generated platform metadata to clips
    for i, clip in enumerate(shorts):
        idx = i + 1
        item = parsed_clips_map.get(idx)
        if item:
            normalized = _normalize_platform_dict(
                item,
                fallback_title=clip.get("video_title_for_youtube_short") or video_title,
                fallback_speaker=clip.get("speaker_name") or "",
            )
        else:
            # Fallback per-clip if batch missed this index
            fallback_title = clip.get("video_title_for_youtube_short") or f"{video_title} - Moment {idx}"
            normalized = _normalize_platform_dict(
                {"title": fallback_title},
                fallback_title=fallback_title,
                fallback_speaker=clip.get("speaker_name") or "",
            )

        clip["platforms"] = normalized["platforms"]
        clip["speaker_name"] = normalized["speaker_name"] or clip.get("speaker_name", "")
        clip["hashtags"] = normalized["hashtags"]
        clip["caption"] = normalized["caption"]
        clip["video_description_for_tiktok"] = normalized["platforms"]["tiktok"]["caption"]
        clip["video_description_for_instagram"] = normalized["platforms"]["instagram"]["caption"]
        clip["video_description"] = normalized["platforms"]["youtube"]["description"]
        if not clip.get("video_title_for_youtube_short") and normalized.get("title"):
            clip["video_title_for_youtube_short"] = normalized["title"]

    # Save updated metadata atomically
    save_job_metadata(metadata_path, metadata)
    logger.info("Successfully saved platform metadata for %d clips in job %s", len(shorts), job_id)

    return {
        "success": True,
        "job_id": job_id,
        "clips_count": len(shorts),
        "clips_updated": len(parsed_clips_map),
    }
