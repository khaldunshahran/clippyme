"""Domain service for AI rescoring existing video clips.

Recovers jobs where clip extraction succeeded but AI scoring fell back to 0
(e.g., due to temporary Gemini 503 capacity spikes or rate limits).
"""
import json
import logging
import os
from typing import Optional, Dict, Any, List

from google import genai
from clippyme.domain.job_artifacts import load_job_metadata, save_job_metadata
from clippyme.domain.errors import NotFoundError, ValidationError, ClippyMeError
from clippyme.storage.config_store import load_persistent_config
from clippyme.pipeline.gemini_request import build_model_chain, generate_with_model_fallback
from clippyme.pipeline.gemini_parser import parse_gemini_response

logger = logging.getLogger("clippyme")


def rescore_job(
    job_id: str,
    output_dir: str = "output",
    api_key: Optional[str] = None,
    primary_model: Optional[str] = None,
) -> Dict[str, Any]:
    """Rescore all clips in an existing job with AI virality scores and titles.

    Reads the job metadata, extracts the transcript text slice for each clip,
    queries Gemini via the resilient fallback chain, updates clip virality
    scores, reasons, hooks, and titles, and atomically writes back to disk.
    """
    try:
        metadata_path, metadata = load_job_metadata(job_id, output_dir)
    except FileNotFoundError:
        raise NotFoundError(f"Metadata not found for job {job_id}")

    shorts: List[dict] = metadata.get("shorts", [])
    if not shorts:
        raise ValidationError(f"Job {job_id} has no clips to rescore")

    cfg = load_persistent_config()
    active_key = api_key or cfg.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not active_key:
        raise ValidationError("GEMINI_API_KEY is not configured")

    target_model = primary_model or cfg.get("GEMINI_MODEL") or "gemini-3.5-flash-lite"
    model_chain = build_model_chain(target_model)

    video_title = (
        metadata.get("source_info", {}).get("title")
        or metadata.get("video_title")
        or "Video Clips"
    )

    # Extract segments
    segments = metadata.get("transcript", {}).get("segments", [])
    if not segments:
        # Check checkpoint transcript if present
        checkpoint_tr = os.path.join(output_dir, job_id, ".clippyme_checkpoint", "transcript.json")
        if os.path.exists(checkpoint_tr):
            try:
                with open(checkpoint_tr, "r", encoding="utf-8") as f:
                    checkpoint_data = json.load(f)
                    segments = checkpoint_data.get("segments", [])
            except Exception as e:
                logger.warning("Could not read checkpoint transcript: %s", e)

    clip_payload = []
    for i, clip in enumerate(shorts):
        c_start = float(clip.get("start", 0.0))
        c_end = float(clip.get("end", 0.0))
        # Match segments within clip range
        if segments:
            c_text = " ".join(
                s.get("text", "").strip()
                for s in segments
                if s.get("end", 0) >= c_start and s.get("start", 0) <= c_end
            )
        else:
            c_text = clip.get("tiktok_caption", "") or clip.get("video_title_for_youtube_short", "")

        clip_payload.append({
            "clip_index": i + 1,
            "start": round(c_start, 1),
            "end": round(c_end, 1),
            "duration_seconds": round(c_end - c_start, 1),
            "transcript_text": c_text[:600],
        })

    learned_patterns_block = ""
    try:
        from clippyme.domain.performance_feedback import get_learned_patterns_prompt
        learned_patterns_block = get_learned_patterns_prompt()
    except Exception:
        learned_patterns_block = ""

    prompt = f"""You are an elite short-form video viral strategist for YouTube Shorts, TikTok, and Instagram Reels.
Video Title: "{video_title}"

## NARRATIVE COMPLETENESS & WHOLE-STORY INTEGRITY (MANDATORY RULE)
Evaluate each clip's self-contained narrative arc:
1. SETUP & CONTEXT: Establish the premise clearly so the viewer understands the scenario without confusion.
2. ESCALATION & CONFLICT: Tension, debate, question, or dramatic buildup.
3. CLIMAX / PEAK BEAT: The shocking quote, confrontation, revelation, or critical point.
4. RESOLUTION & PUNCHLINE: The outcome, consequence, reaction, or punchline.
A complete story arc that resolves its premise with a punchline achieves far higher viral shareability and completion rate. Penalize truncated fragments cut off before payoff.
{learned_patterns_block}

Analyze the following {len(clip_payload)} extracted video clips.
For EACH clip, evaluate its viral performance and provide:
- viral_score: Integer between 65 and 99 based on hook strength, emotional intensity, narrative completeness, controversy/curiosity, and shareability.
- viral_reason: A sharp, persuasive 1-2 sentence explanation of why this moment captures viewer retention and drives comments/shares. Highlight narrative setup, payoff, and story completeness.
- video_title_for_youtube_short: A high-CTR viral YouTube Short title with curiosity hook (under 60 characters).
- hook: The opening punchline or tension-building hook in the first 3 seconds of the clip.
- duration_tier: "short" (<60s), "mid" (60s-120s), or "extended" (>120s).
- target_platforms: Array of best-fit platforms, e.g. ["tiktok", "youtube_shorts", "instagram_reels"].

Clips:
{json.dumps(clip_payload, indent=2)}

Respond with ONLY a JSON object matching this schema:
{{
  "clips": [
    {{
      "clip_index": 1,
      "viral_score": 88,
      "viral_reason": "...",
      "video_title_for_youtube_short": "...",
      "hook": "...",
      "duration_tier": "mid",
      "target_platforms": ["tiktok", "youtube_shorts"]
    }}
  ]
}}
"""

    client = genai.Client(api_key=active_key)
    logger.info("Rescoring %d clips for job %s using model chain %s", len(shorts), job_id, model_chain)
    response, model_used = generate_with_model_fallback(
        client, prompt, model_chain, max_attempts=2, log_fn=logger.info
    )

    parse_result = parse_gemini_response(response.text or "")
    if not parse_result.data or "clips" not in parse_result.data:
        raise ClippyMeError("Failed to parse AI rescoring response", status_code=502)

    rescored_map = {item.get("clip_index"): item for item in parse_result.data.get("clips", [])}

    # Update metadata shorts
    for i, clip in enumerate(shorts):
        idx = i + 1
        ai_data = rescored_map.get(idx)
        if ai_data:
            clip["viral_score"] = int(ai_data.get("viral_score", 85))
            clip["viral_reason"] = str(ai_data.get("viral_reason", "")).strip()
            if ai_data.get("video_title_for_youtube_short"):
                clip["video_title_for_youtube_short"] = str(ai_data["video_title_for_youtube_short"]).strip()
            if ai_data.get("hook"):
                clip["hook"] = str(ai_data["hook"]).strip()
                clip["viral_hook_text"] = str(ai_data["hook"]).strip()
            if ai_data.get("duration_tier"):
                clip["duration_tier"] = str(ai_data["duration_tier"]).strip()
            if ai_data.get("target_platforms") and isinstance(ai_data["target_platforms"], list):
                clip["target_platforms"] = [str(p).strip() for p in ai_data["target_platforms"] if p]

    # Atomically save updated metadata
    save_job_metadata(metadata_path, metadata)
    logger.info("Successfully updated metadata for job %s with model %s", job_id, model_used)

    # Ensure platform metadata and situational captions are populated
    try:
        from clippyme.domain.metadata_generator import generate_all_clips_metadata
        generate_all_clips_metadata(job_id, output_dir=output_dir, api_key=active_key, force=False)
        _, metadata = load_job_metadata(job_id, output_dir)
        shorts = metadata.get("shorts", [])
    except Exception as exc:
        logger.warning("Auto metadata generation during rescore notice: %s", exc)

    # Also update .clippyme_runtime.json if it exists
    job_dir = os.path.dirname(metadata_path)
    runtime_path = os.path.join(job_dir, ".clippyme_runtime.json")
    if os.path.exists(runtime_path):
        try:
            with open(runtime_path, "r", encoding="utf-8") as f:
                rt_data = json.load(f)
            if "preflight" in rt_data and isinstance(rt_data["preflight"], dict):
                rt_data["preflight"]["model"] = model_used
            with open(runtime_path, "w", encoding="utf-8") as f:
                json.dump(rt_data, f, indent=2)
        except Exception as e:
            logger.warning("Failed to update runtime file after rescore: %s", e)

    return {
        "success": True,
        "job_id": job_id,
        "model_used": model_used,
        "rescored_clips": len(shorts),
        "clips": [
            {
                "index": i + 1,
                "viral_score": c.get("viral_score"),
                "viral_reason": c.get("viral_reason"),
                "video_title_for_youtube_short": c.get("video_title_for_youtube_short"),
                "hook": c.get("hook"),
                "viral_hook_text": c.get("viral_hook_text") or c.get("hook"),
                "duration_tier": c.get("duration_tier"),
                "target_platforms": c.get("target_platforms"),
            }
            for i, c in enumerate(shorts)
        ],
    }
