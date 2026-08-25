"""Pure logic for AI Highlights / Supercut Reel generation.

Host-testable with no heavy ML or cv2 dependencies.
"""
import json
import logging
import re
from typing import List, Dict, Any, Optional

from clippyme.pipeline.cut_ops import snap_clip_to_words

logger = logging.getLogger("clippyme.highlights")


MULTI_TIER_HIGHLIGHT_PROMPT_TEMPLATE = """You are a world-class documentary director and master video recap editor.
Your task is to analyze the full video transcript and construct a series of TRUE RECAP HIGHLIGHT REELS that summarize the entire video narrative from beginning to end.

A viewer watching these reels should understand the full story, key arguments, progression, and final conclusion without having to watch the full video.

## CRITICAL NARRATIVE & RECAP RULES:
1. **COMPLETE CHRONOLOGICAL STORY ARC (0% to 100%)**:
   - Every highlight reel MUST progress chronologically forward in time.
   - **Beginning / Setup (0% - 25% of video)**: Establish the premise, core question, stakes, or opening context.
   - **Middle / Progression (25% - 75% of video)**: The pivotal turning points, biggest revelations, key challenges, or breakthrough moments.
   - **Climax / Resolution (75% - 100% of video)**: The final outcome, verdict, emotional climax, or overarching takeaway.
   - **DO NOT** jump backward in time or focus only on a single isolated segment. The cuts must span across the entire video timeline.

2. **DURATION TIERS & RECAP INTENTS**:
   - **"micro" (TikTok, Shorts, IG Reels - Target: 35s - 55s)**: High-energy, fast-paced 3-4 milestone recap: [Hook & Premise] -> [Biggest Action / Turning Point] -> [Final Result / Climax]. Total duration strictly under 60s.
   - **"recap" (YouTube / Feed / Story Digest - Target: 90s - 140s)**: The Definitive Story Digest. 5-8 sequential chapter moments capturing the full end-to-end narrative with context and complete thoughts.
   {extended_tier_desc}

3. **NATURAL SENTENCE BOUNDARIES & AUDIO FLOW**:
   - Every cut MUST start at the natural beginning of a sentence and end on a clean sentence completion.
   - Never cut mid-word, mid-syllable, or mid-sentence.
   - Provide exact "start" and "end" timestamps in seconds (float).

Video Title: {video_title}
Video Total Duration: ~{video_duration}s

TRANSCRIPT WITH WORD-LEVEL TIMESTAMPS:
{transcript_text}

## REQUIRED OUTPUT FORMAT (STRICT JSON ONLY):
{{
  "reels": [
    {{
      "tier": "micro",
      "title": "Short punchy title summarizing the whole story",
      "viral_score": 96,
      "summary": "High-level summary of the end-to-end recap",
      "cuts": [
        {{ "role": "setup", "start": 0.0, "end": 12.5, "summary": "The opening challenge premise" }},
        {{ "role": "beat", "start": 215.2, "end": 230.0, "summary": "The midpoint escalation" }},
        {{ "role": "outro", "start": 580.0, "end": 592.5, "summary": "The final winner and conclusion" }}
      ]
    }},
    {{
      "tier": "recap",
      "title": "Comprehensive Story Digest Title",
      "viral_score": 93,
      "summary": "Full narrative journey from setup to conclusion",
      "cuts": [
        {{ "role": "setup", "start": 0.0, "end": 18.0, "summary": "Premise and rules" }},
        {{ "role": "beat", "start": 85.0, "end": 105.0, "summary": "First obstacle" }},
        {{ "role": "beat", "start": 215.0, "end": 240.0, "summary": "Major twist / breakthrough" }},
        {{ "role": "beat", "start": 410.0, "end": 435.0, "summary": "High-stakes battle" }},
        {{ "role": "outro", "start": 575.0, "end": 595.0, "summary": "Final outcome and wrap-up" }}
      ]
    }}
  ]
}}
"""


SINGLE_SUPERCUT_PROMPT_TEMPLATE = """You are a world-class documentary director and master video recap editor.
Your task is to analyze the full video transcript and construct a single, cohesive RECAP HIGHLIGHT REEL that summarizes the entire video narrative from beginning to end in chronological order.

## RECAP RULES:
1. **Target Duration**: ~{target_duration} seconds (strict window: {min_dur}s to {max_dur}s).
2. **Complete Chronological Story Arc (0% to 100%)**:
   - Setup / Premise (0% - 25% of source video)
   - Progression / Turning Points (25% - 75% of source video)
   - Climax / Resolution (75% - 100% of source video)
3. Cuts MUST be ordered chronologically forward in time and span the video progression.
4. Clean sentence boundaries (do not cut mid-sentence or mid-word).

Video Title: {video_title}
Video Total Duration: ~{video_duration}s

TRANSCRIPT WITH WORD-LEVEL TIMESTAMPS:
{transcript_text}

## REQUIRED OUTPUT FORMAT (STRICT JSON ONLY):
{{
  "title": "Highlight Reel Title Summarizing Video",
  "viral_score": 95,
  "summary": "Summary of the complete narrative arc from start to finish",
  "cuts": [
    {{ "role": "setup", "start": 0.0, "end": 12.0, "summary": "Opening premise" }},
    {{ "role": "beat", "start": 180.0, "end": 210.0, "summary": "Key turning point" }},
    {{ "role": "outro", "start": 540.0, "end": 565.0, "summary": "Final outcome" }}
  ]
}}
"""


def _parse_safe_score(val: Any, default: int = 85) -> int:
    """Safely parse a viral score integer clamped between 0 and 100."""
    try:
        if val is None:
            return default
        if isinstance(val, (int, float)):
            return max(0, min(100, int(val)))
        if isinstance(val, str):
            cleaned = re.sub(r"[^\d.]", "", val)
            if cleaned:
                return max(0, min(100, int(float(cleaned))))
        return default
    except (ValueError, TypeError):
        return default


def _deduplicate_and_trim_overlaps(cuts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Sort cuts chronologically and trim/discard overlapping adjacent segments.

    Note: The engulfment check (end <= prev_end + 1.0) intentionally drops cuts that extend
    only marginally beyond the previous cut (<= 1.0s) to avoid awkward 1-second fragment cuts.
    """
    if not cuts:
        return []

    # Sort strictly chronologically by start timestamp
    sorted_cuts = sorted(cuts, key=lambda x: float(x.get("start", 0.0)))
    cleaned: List[Dict[str, Any]] = []

    for c in sorted_cuts:
        start = round(float(c.get("start", 0.0)), 2)
        end = round(float(c.get("end", 0.0)), 2)

        if end <= start or (end - start) < 1.0:
            continue

        if cleaned:
            prev_end = cleaned[-1]["end"]
            if start < prev_end:
                # If this cut is engulfed (or extends only marginally <= 1.0s), skip it
                if end <= prev_end + 1.0:
                    continue
                # Shift start to immediately follow previous cut
                start = round(prev_end + 0.05, 2)

        duration = round(end - start, 2)
        if duration >= 1.0:
            cleaned.append({
                **c,
                "start": start,
                "end": end,
                "duration": duration,
            })

    return cleaned


TIER_DURATION_BOUNDS = {
    "micro": (25.0, 58.0),
    "recap": (70.0, 150.0),
    "extended": (150.0, 720.0),
}


def _enforce_duration_bounds(
    cuts: List[Dict[str, Any]],
    min_d: float = 10.0,
    max_d: float = 720.0,
    tier_name: str = "",
) -> List[Dict[str, Any]]:
    """Enforce duration bounds while preserving the chronological story arc (Setup + Climax).

    Strategy:
    1. Always protect the opening cut (Setup/Hook) and closing cut (Climax/Outro) as narrative anchors.
    2. When budget is exceeded, shed or trim from middle filler beats first.
    3. If anchors alone exceed budget or only 1-2 cuts exist, proportionally scale cuts down so both opening and resolution remain represented.
    4. Log warnings if total duration falls below min_d.
    """
    if not cuts:
        return []

    total = sum(c["duration"] for c in cuts)

    if total <= max_d:
        if total < min_d:
            logger.warning(
                "Highlight reel %s is under-filled (total duration %.1fs < min %.1fs)",
                tier_name or "supercut", total, min_d
            )
        return cuts

    # Case 1: Single cut that exceeds max_d -> hard trim that cut to max_d
    if len(cuts) == 1:
        c0 = cuts[0]
        trimmed_end = round(c0["start"] + min(c0["duration"], max_d), 2)
        trimmed_dur = round(trimmed_end - c0["start"], 2)
        return [{
            **c0,
            "end": trimmed_end,
            "duration": trimmed_dur,
        }]

    # Case 2: 2 cuts (Setup + Outro)
    if len(cuts) == 2:
        # Scale both cuts proportionally to fit within max_d
        scale = max_d / total
        result = []
        for c in cuts:
            scaled_dur = max(1.5, round(c["duration"] * scale, 2))
            trimmed_end = round(c["start"] + scaled_dur, 2)
            result.append({
                **c,
                "end": trimmed_end,
                "duration": round(trimmed_end - c["start"], 2),
            })
        return result

    # Case 3: 3+ cuts (Setup, Middle Beats..., Outro)
    # Anchor protection: opening cut (idx 0) and closing cut (idx -1)
    head = cuts[0]
    tail = cuts[-1]
    middle = cuts[1:-1]

    anchor_dur = head["duration"] + tail["duration"]
    if anchor_dur >= max_d:
        # The anchors alone exceed max_d -> keep anchors only and proportionally scale them
        scale = max_d / anchor_dur
        result = []
        for c in [head, tail]:
            scaled_dur = max(1.5, round(c["duration"] * scale, 2))
            trimmed_end = round(c["start"] + scaled_dur, 2)
            result.append({
                **c,
                "end": trimmed_end,
                "duration": round(trimmed_end - c["start"], 2),
            })
        return result

    # Anchors fit with budget remaining for middle beats
    middle_budget = max_d - anchor_dur
    selected_middle = []
    current_mid_dur = 0.0

    for m in middle:
        d = m["duration"]
        if current_mid_dur + d <= middle_budget:
            selected_middle.append(m)
            current_mid_dur += d
        elif (middle_budget - current_mid_dur) >= 2.0:
            # Trim this middle cut to use the remaining middle budget
            avail = middle_budget - current_mid_dur
            trimmed_end = round(m["start"] + avail, 2)
            if (trimmed_end - m["start"]) >= 1.5:
                selected_middle.append({
                    **m,
                    "end": trimmed_end,
                    "duration": round(trimmed_end - m["start"], 2),
                })
                current_mid_dur += (trimmed_end - m["start"])
            break
        else:
            break

    # Assemble: Head (Setup) + Selected Middle Beats + Tail (Climax/Outro preserved!)
    assembled = [head] + selected_middle + [tail]
    return assembled


def _enforce_tier_duration(cuts: List[Dict[str, Any]], tier: str) -> List[Dict[str, Any]]:
    """Enforce duration bounds on cuts for a given tier while preserving the setup and climax."""
    if not cuts:
        return []
    min_d, max_d = TIER_DURATION_BOUNDS.get(tier, (10.0, 720.0))
    return _enforce_duration_bounds(cuts, min_d=min_d, max_d=max_d, tier_name=tier)


def _chunk_transcript(
    transcript_words: List[Dict[str, Any]],
    max_chunks: int = 4000,
) -> str:
    """Format word stream into timestamped sentence chunks with uniform proportional sampling if oversized."""
    if not transcript_words:
        return ""

    lines = []
    current_chunk = []
    chunk_start = 0.0

    for w in transcript_words:
        word_text = w.get("word", "").strip()
        start = float(w.get("start", 0.0))
        end = float(w.get("end", start + 0.3))

        if not current_chunk:
            chunk_start = start
        current_chunk.append(word_text)

        if len(current_chunk) >= 12 or word_text.endswith((".", "!", "?", "\n")):
            lines.append(f"[{chunk_start:.2f}s - {end:.2f}s] {' '.join(current_chunk)}")
            current_chunk = []

    if current_chunk:
        last_end = float(transcript_words[-1].get("end", chunk_start + 1.0))
        lines.append(f"[{chunk_start:.2f}s - {last_end:.2f}s] {' '.join(current_chunk)}")

    if len(lines) > max_chunks:
        logger.warning(
            "Transcript chunk count (%d) exceeded budget (%d); sampling proportionally across duration",
            len(lines), max_chunks
        )
        step = len(lines) / max_chunks
        sampled = [lines[int(i * step)] for i in range(max_chunks)]
        return "\n".join(sampled)

    return "\n".join(lines)


def build_multi_tier_highlights_prompt(
    transcript_words: List[Dict[str, Any]],
    video_duration_sec: float = 600.0,
    video_title: str = "",
) -> str:
    """Build the prompt for Gemini to generate multiple highlight packages across duration tiers."""
    dur = float(video_duration_sec or 600.0)
    include_extended = dur >= 300.0

    if include_extended:
        target_ext = min(720, max(180, int(dur * 0.35)))
        extended_tier_desc = f'- **"extended" (Long-form / Documentary Supercut - Target: 180s - {min(720, int(dur * 0.5))}s)**: Deep-dive chronological compilation covering all key chapters and in-depth discussions from start to finish.'
    else:
        extended_tier_desc = ""

    transcript_text = _chunk_transcript(transcript_words, max_chunks=4000)

    return MULTI_TIER_HIGHLIGHT_PROMPT_TEMPLATE.format(
        video_title=video_title or "Untitled Video",
        video_duration=int(dur),
        extended_tier_desc=extended_tier_desc,
        transcript_text=transcript_text,
    )


def parse_multi_tier_response(raw_response: str) -> List[Dict[str, Any]]:
    """Parse and validate the multi-tier JSON response from Gemini."""
    text = (raw_response or "").strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)

    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("Multi-tier response must be a JSON object")

    reels_data = data.get("reels", [])
    if not isinstance(reels_data, list) or not reels_data:
        raise ValueError("Multi-tier response contains no reels")

    parsed_reels = []
    for idx, r in enumerate(reels_data):
        if not isinstance(r, dict):
            continue

        try:
            tier = str(r.get("tier", "recap")).strip().lower()
            if tier not in ("micro", "recap", "extended"):
                tier = "micro" if idx == 0 else ("recap" if idx == 1 else "extended")

            title = str(r.get("title", f"Highlight Reel {idx + 1}")).strip()
            viral_score = _parse_safe_score(r.get("viral_score"), default=85)
            summary = str(r.get("summary", "")).strip()

            raw_cuts = r.get("cuts", [])
            validated_cuts = []
            for c in raw_cuts:
                if not isinstance(c, dict):
                    continue
                try:
                    start = float(c.get("start", 0.0))
                    end = float(c.get("end", 0.0))
                    role = str(c.get("role", "beat")).strip().lower()
                    c_summary = str(c.get("summary", "")).strip()

                    if end > start and (end - start) >= 1.0:
                        validated_cuts.append({
                            "start": round(start, 2),
                            "end": round(end, 2),
                            "duration": round(end - start, 2),
                            "role": role,
                            "summary": c_summary,
                        })
                except (ValueError, TypeError):
                    continue

            if validated_cuts:
                # 1. Deduplicate & trim overlapping spans
                deduped = _deduplicate_and_trim_overlaps(validated_cuts)
                # 2. Enforce duration limits for tier with anchor protection
                bounded = _enforce_tier_duration(deduped, tier)

                if bounded:
                    total_dur = sum(c["duration"] for c in bounded)
                    parsed_reels.append({
                        "tier": tier,
                        "title": title,
                        "viral_score": viral_score,
                        "summary": summary,
                        "total_duration": round(total_dur, 2),
                        "cuts": bounded,
                    })
        except Exception as e:
            logger.warning("Failed to parse reel at index %d: %s", idx, e)
            continue

    if not parsed_reels:
        raise ValueError("No valid highlight reels parsed from multi-tier response")

    return parsed_reels


def build_supercut_prompt(
    transcript_words: List[Dict[str, Any]],
    target_duration_sec: int = 60,
    video_duration_sec: float = 600.0,
    video_title: str = "",
) -> str:
    """Build the prompt for Gemini to select an ordered highlight reel timeline for a single target duration."""
    target_duration = max(30, min(300, int(target_duration_sec)))
    min_dur = max(20, target_duration - 10)
    max_dur = target_duration + 10
    dur = float(video_duration_sec or 600.0)

    transcript_text = _chunk_transcript(transcript_words, max_chunks=4000)

    return SINGLE_SUPERCUT_PROMPT_TEMPLATE.format(
        video_title=video_title or "Untitled Video",
        video_duration=int(dur),
        target_duration=target_duration,
        min_dur=min_dur,
        max_dur=max_dur,
        transcript_text=transcript_text,
    )


def parse_supercut_response(
    raw_response: str,
    target_duration_sec: Optional[int] = None,
) -> Dict[str, Any]:
    """Parse and validate the JSON response from Gemini for a single supercut timeline."""
    text = (raw_response or "").strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)

    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("Supercut response must be a JSON object")

    title = str(data.get("title", "Highlight Reel")).strip()
    viral_score = _parse_safe_score(data.get("viral_score"), default=85)
    summary = str(data.get("summary", "")).strip()

    cuts = data.get("cuts")
    # Fallback if Gemini returned nested reels array
    if not cuts and isinstance(data.get("reels"), list) and data["reels"]:
        first_reel = data["reels"][0]
        if isinstance(first_reel, dict):
            cuts = first_reel.get("cuts", [])
            title = str(first_reel.get("title", title)).strip()
            viral_score = _parse_safe_score(first_reel.get("viral_score"), default=viral_score)
            summary = str(first_reel.get("summary", summary)).strip()

    if not isinstance(cuts, list) or not cuts:
        raise ValueError("Supercut response contains no cuts")

    validated_cuts = []
    for c in cuts:
        if not isinstance(c, dict):
            continue
        try:
            start = float(c.get("start", 0.0))
            end = float(c.get("end", 0.0))
            role = str(c.get("role", "beat")).strip().lower()
            c_summary = str(c.get("summary", "")).strip()

            if end > start and (end - start) >= 1.0:
                validated_cuts.append({
                    "start": round(start, 2),
                    "end": round(end, 2),
                    "duration": round(end - start, 2),
                    "role": role,
                    "summary": c_summary,
                })
        except (ValueError, TypeError):
            continue

    if not validated_cuts:
        raise ValueError("No valid cuts with positive duration found in response")

    # 1. Deduplicate and trim overlapping adjacent cuts
    cleaned_cuts = _deduplicate_and_trim_overlaps(validated_cuts)
    if not cleaned_cuts:
        raise ValueError("No valid non-overlapping cuts found in response")

    # 2. Enforce duration bounds with anchor protection if target duration provided
    if target_duration_sec is not None:
        target_d = int(target_duration_sec)
        min_d = max(15.0, float(target_d - 15))
        max_d = float(target_d + 15)
        bounded_cuts = _enforce_duration_bounds(cleaned_cuts, min_d=min_d, max_d=max_d, tier_name="supercut")
    else:
        bounded_cuts = cleaned_cuts

    total_dur = sum(c["duration"] for c in bounded_cuts)
    return {
        "title": title,
        "viral_score": viral_score,
        "summary": summary,
        "total_duration": round(total_dur, 2),
        "cuts": bounded_cuts,
    }


def snap_supercut_timeline(
    cuts: List[Dict[str, Any]],
    transcript_words: List[Dict[str, Any]],
    source_video_duration_sec: Optional[float] = None,
    max_duration_sec: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Snap each micro-cut to natural word boundaries and clamp within source video bounds."""
    if not cuts:
        return []

    # Support both named params for backward compatibility
    effective_source_dur = source_video_duration_sec if source_video_duration_sec is not None else max_duration_sec

    snapped_cuts = []
    for c in cuts:
        raw_start = float(c.get("start", 0.0))
        raw_end = float(c.get("end", 0.0))

        if transcript_words:
            start, end = snap_clip_to_words(
                raw_start,
                raw_end,
                transcript_words,
                source_duration=effective_source_dur,
            )
        else:
            start, end = raw_start, raw_end

        if effective_source_dur is not None:
            start = max(0.0, min(start, effective_source_dur))
            end = max(0.0, min(end, effective_source_dur))

        if end > start + 0.5:
            snapped_cuts.append({
                **c,
                "start": round(start, 3),
                "end": round(end, 3),
                "duration": round(end - start, 3),
            })

    return snapped_cuts


def fuse_supercut_subtitles(
    cuts: List[Dict[str, Any]],
    transcript_words: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Shift word timestamps from original video time into the new concatenated timeline.

    Returns a new list of word objects whose timestamps start at 0.0s and progress
    continuously across all stitched cuts.
    """
    if not cuts or not transcript_words:
        return []

    fused_words: List[Dict[str, Any]] = []
    timeline_offset = 0.0

    for cut in cuts:
        cut_start = cut["start"]
        cut_end = cut["end"]
        cut_duration = cut_end - cut_start

        # Extract words that belong to this cut
        for w in transcript_words:
            w_start = float(w.get("start", 0.0))
            w_end = float(w.get("end", w_start + 0.2))

            # Check if word falls inside the cut interval
            if w_end > cut_start and w_start < cut_end:
                # Clamp within cut bounds
                clamped_start = max(cut_start, w_start)
                clamped_end = min(cut_end, w_end)

                # Shift by cut start and add timeline offset
                new_start = round(timeline_offset + (clamped_start - cut_start), 3)
                new_end = round(timeline_offset + (clamped_end - cut_start), 3)

                if new_end > new_start:
                    fused_words.append({
                        **w,
                        "start": new_start,
                        "end": new_end,
                    })

        timeline_offset += cut_duration

    return fused_words


ASPECT_DIMENSIONS = {
    "9:16": (1080, 1920),
    "16:9": (1920, 1080),
    "1:1": (1080, 1080),
    "4:5": (1080, 1350),
}


def get_aspect_dimensions(aspect: str) -> tuple[int, int]:
    """Return target (width, height) for the given aspect ratio."""
    return ASPECT_DIMENSIONS.get(aspect, (1080, 1920))


def build_highlight_video_filter(
    aspect: str = "9:16",
    reframe_mode: str = "blur_pad",
) -> str:
    """Generate FFmpeg video filter for sizing and reframing the highlight video.

    Modes:
    - blur_pad: Blurred background copy behind the scaled source video
    - letterbox / fit: Fit inside canvas with clean black bars
    - crop / center_crop: Center-cropped to fill the canvas
    - 16:9: Standard 16:9 output with aspect fit and black padding if source is non-16:9
    """
    target_w, target_h = get_aspect_dimensions(aspect)

    if aspect == "16:9":
        return f"scale={target_w}:{target_h}:force_original_aspect_ratio=decrease,pad={target_w}:{target_h}:(ow-iw)/2:(oh-ih)/2:black"

    mode = (reframe_mode or "blur_pad").lower()

    if mode in ("letterbox", "fit"):
        return f"scale={target_w}:{target_h}:force_original_aspect_ratio=decrease,pad={target_w}:{target_h}:(ow-iw)/2:(oh-ih)/2:black"

    if mode in ("crop", "center_crop"):
        return f"scale={target_w}:{target_h}:force_original_aspect_ratio=increase,crop={target_w}:{target_h}"

    # Default to modern high-retention blur_pad for vertical/square/portrait
    # Uses robust static blur radius to guarantee FFmpeg portability across platforms
    return (
        f"[0:v]scale={target_w}:{target_h}:force_original_aspect_ratio=increase,crop={target_w}:{target_h},"
        f"boxblur=15:2[bg];"
        f"[0:v]scale={target_w}:{target_h}:force_original_aspect_ratio=decrease[fg];"
        f"[bg][fg]overlay=(W-w)/2:(H-h)/2"
    )
