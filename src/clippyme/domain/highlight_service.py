"""Highlight Reel / Supercut Generation Service.

Orchestrates Gemini narrative selection, micro-cut snapping, aspect ratio sizing
(9:16, 16:9, 1:1, 4:5), FFmpeg fast seeking concatenation, subtitle fusion, and multi-layer composition
(Subtitles, Color Grading, Hook Overlays, and Brand Logos).
"""
import asyncio
import json
import logging
import os
import subprocess
import tempfile
import time
import uuid
from typing import Dict, Any, Optional, List, Tuple

from clippyme.domain.clip_locks import job_metadata_lock
from clippyme.domain.encode import ffmpeg_timeout, x264_video_args
from clippyme.domain.errors import ValidationError, NotFoundError, ComposeError, ClippyMeError
from clippyme.domain.highlights_ops import (
    build_supercut_prompt,
    parse_supercut_response,
    build_multi_tier_highlights_prompt,
    parse_multi_tier_response,
    snap_supercut_timeline,
    fuse_supercut_subtitles,
    get_aspect_dimensions,
    build_highlight_video_filter,
)
from clippyme.domain.job_artifacts import load_job_metadata, save_job_metadata
from clippyme.domain.subtitles import generate_ass_karaoke, burn_subtitles
from clippyme.storage.config_store import load_persistent_config

logger = logging.getLogger("clippyme.highlights")


def _find_source_video_path(job_dir: str, job_data: Optional[dict] = None) -> str:
    """Locate the best source video file in the job directory."""
    if not os.path.isdir(job_dir):
        raise NotFoundError(f"Job directory not found: {job_dir}")

    # Check job_data artifacts first if available
    if job_data:
        artifacts = job_data.get("artifacts") or {}
        input_vid = artifacts.get("input_video") or job_data.get("source_video")
        if input_vid and os.path.isfile(input_vid):
            return input_vid

    # Also check .clippyme_runtime.json
    runtime_path = os.path.join(job_dir, ".clippyme_runtime.json")
    if os.path.isfile(runtime_path):
        try:
            with open(runtime_path, "r", encoding="utf-8") as f:
                rt = json.load(f)
                input_vid = (rt.get("artifacts") or {}).get("input_video")
                if input_vid and os.path.isfile(input_vid):
                    return input_vid
        except Exception:
            pass

    candidates = []
    allowed_exts = (".mp4", ".mov", ".mkv", ".webm", ".m4v", ".avi")
    for f in os.listdir(job_dir):
        if not any(f.lower().endswith(ext) for ext in allowed_exts):
            continue
        # Skip generated clips, covers, or temporary files
        if f.startswith(("composed_", "dubbed_", "highlight_reel_", "temp_")):
            continue
        candidates.append(os.path.join(job_dir, f))

    if not candidates:
        if os.path.isfile(runtime_path):
            try:
                with open(runtime_path, "r", encoding="utf-8") as f:
                    rt = json.load(f)
                if not rt.get("completed_at"):
                    raise ValidationError("Video is still being downloaded or processed. Please wait for completion.")
            except (ValidationError, NotFoundError):
                raise
            except Exception:
                pass
        raise NotFoundError(f"No source video file found in {job_dir}")

    # Prefer source_* if present, else largest / newest media candidate
    sources = [c for c in candidates if os.path.basename(c).startswith("source_")]
    if sources:
        return sources[0]
    return max(candidates, key=os.path.getsize)


def _extract_transcript_words_from_job(job_data: dict, job_dir: str) -> List[Dict[str, Any]]:
    """Extract flat list of word dicts from metadata, checkpoint, or cached transcript."""
    from clippyme.pipeline.cut_ops import flatten_words

    # 1. Check metadata dictionary directly
    full_transcript = job_data.get("full_transcript") or job_data.get("transcript")
    if isinstance(full_transcript, dict):
        words = flatten_words(full_transcript)
        if words:
            return words

    # 2. Check .clippyme_checkpoint/transcript.json
    chk_path = os.path.join(job_dir, ".clippyme_checkpoint", "transcript.json")
    if os.path.isfile(chk_path):
        try:
            with open(chk_path, "r", encoding="utf-8") as f:
                chk_data = json.load(f)
                if isinstance(chk_data, dict):
                    words = flatten_words(chk_data)
                    if words:
                        return words
        except Exception:
            pass

    # 3. Check results.json
    results_path = os.path.join(job_dir, "results.json")
    if os.path.isfile(results_path):
        try:
            with open(results_path, "r", encoding="utf-8") as f:
                res_data = json.load(f)
                full_t = res_data.get("full_transcript") or res_data.get("transcript")
                if isinstance(full_t, dict):
                    words = flatten_words(full_t)
                    if words:
                        return words
        except Exception:
            pass

    # 4. Check shorts word arrays
    words: List[Dict[str, Any]] = []
    shorts = job_data.get("shorts", [])
    if isinstance(shorts, list):
        for s in shorts:
            if isinstance(s, dict) and "words" in s:
                for w in s.get("words", []):
                    if isinstance(w, dict) and "word" in w and "start" in w:
                        words.append(w)

    if words:
        words.sort(key=lambda x: float(x.get("start", 0.0)))
        return words

    # 5. Check if segments without words exist in any candidate and synthesize word timings
    for candidate_t in [full_transcript]:
        if isinstance(candidate_t, dict) and "segments" in candidate_t:
            for seg in candidate_t.get("segments", []):
                text = (seg.get("text") or "").strip()
                s_start = float(seg.get("start", 0.0))
                s_end = float(seg.get("end", s_start + 1.0))
                raw_words = text.split()
                if not raw_words:
                    continue
                step = (s_end - s_start) / len(raw_words)
                for i, rw in enumerate(raw_words):
                    words.append({
                        "word": rw,
                        "start": round(s_start + i * step, 2),
                        "end": round(s_start + (i + 1) * step, 2),
                    })

    if not words:
        runtime_path = os.path.join(job_dir, ".clippyme_runtime.json")
        if os.path.isfile(runtime_path):
            try:
                with open(runtime_path, "r", encoding="utf-8") as f:
                    rt = json.load(f)
                if not rt.get("completed_at"):
                    raise ValidationError("Video transcription is still in progress. Please wait for completion.")
            except (ValidationError, NotFoundError):
                raise
            except Exception:
                pass

    return words


def load_or_create_job_metadata(job_id: str, output_root: str = "output") -> Tuple[str, dict]:
    """Load job metadata, or synthesize a clean metadata dictionary if *_metadata.json hasn't been generated yet."""
    job_dir = os.path.join(output_root, job_id)
    if not os.path.isdir(job_dir):
        raise NotFoundError(f"Job directory not found: {job_dir}")

    try:
        return load_job_metadata(job_id, output_root)
    except FileNotFoundError:
        pass

    # Check metadata.json
    meta_path = os.path.join(job_dir, "metadata.json")
    if os.path.isfile(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                return meta_path, json.load(f)
        except Exception:
            pass

    # Synthesize from source_info.json or .clippyme_runtime.json
    data: Dict[str, Any] = {
        "job_id": job_id,
        "title": "Video Highlights",
        "highlights": [],
    }

    src_info_path = os.path.join(job_dir, "source_info.json")
    if os.path.isfile(src_info_path):
        try:
            with open(src_info_path, "r", encoding="utf-8") as f:
                src_data = json.load(f)
                data["title"] = src_data.get("title") or data["title"]
                data["duration"] = src_data.get("duration") or 600.0
                data["source_url"] = src_data.get("webpage_url") or src_data.get("url")
        except Exception:
            pass

    rt_path = os.path.join(job_dir, ".clippyme_runtime.json")
    if os.path.isfile(rt_path):
        try:
            with open(rt_path, "r", encoding="utf-8") as f:
                rt_data = json.load(f)
                art = rt_data.get("artifacts") or {}
                data["title"] = art.get("video_title") or data["title"]
                data["source_url"] = art.get("source_url") or data.get("source_url")
                if "preflight" in rt_data:
                    data["duration"] = rt_data["preflight"].get("duration_seconds", data.get("duration", 600.0))
        except Exception:
            pass

    save_job_metadata(meta_path, data)
    return meta_path, data


def plan_highlights_sync(
    job_id: str,
    target_duration: int = 60,
    content_mode: Optional[str] = "podcast",
    output_style: Optional[str] = "recap",
    theme: Optional[str] = None,
    merge_gap_seconds: float = 1.5,
    output_root: str = "output",
    api_key: Optional[str] = None,
    model_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Stage 1: Query Gemini to plan a highlight reel timeline without rendering."""
    job_dir = os.path.join(output_root, job_id)
    _, data = load_or_create_job_metadata(job_id, output_root)

    words = _extract_transcript_words_from_job(data, job_dir)
    if not words:
        raise ValidationError("No transcript with word-level timestamps available for this job")

    cfg = load_persistent_config()
    gemini_key = api_key or cfg.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not gemini_key:
        raise ValidationError("Gemini API key is required to analyze and create highlights")

    model = model_name or cfg.get("GEMINI_MODEL") or os.getenv("GEMINI_MODEL") or "gemini-3.5-flash"
    video_title = data.get("title", data.get("video_title", "Video"))
    video_dur = float(data.get("duration", 0.0))
    if not video_dur and words:
        video_dur = float(words[-1].get("end", 600.0))

    prompt = build_supercut_prompt(
        words,
        target_duration_sec=target_duration,
        video_duration_sec=video_dur,
        video_title=video_title,
        content_mode=content_mode,
        output_style=output_style,
        theme=theme,
    )

    from google import genai
    from google.genai import types

    client = genai.Client(api_key=gemini_key)
    response = client.models.generate_content(
        model=model,
        contents=[prompt],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
        ),
    )

    parsed = parse_supercut_response(
        response.text,
        target_duration_sec=target_duration,
        merge_gap_seconds=merge_gap_seconds,
    )
    raw_cuts = parsed["cuts"]
    snapped_cuts = snap_supercut_timeline(raw_cuts, words, source_video_duration_sec=video_dur)
    if not snapped_cuts:
        raise ComposeError("Failed to snap valid highlight cuts from transcript")

    total_dur = sum(c["duration"] for c in snapped_cuts)
    return {
        "title": parsed.get("title", f"Highlights ({target_duration}s)"),
        "target_duration": target_duration,
        "total_duration": round(total_dur, 2),
        "cuts": snapped_cuts,
    }


def render_highlight_reel_sync(
    job_id: str,
    target_duration: int = 60,
    cuts: Optional[List[Dict[str, Any]]] = None,
    aspect: str = "9:16",
    reframe_mode: str = "blur_pad",
    subtitles: Optional[Dict[str, Any]] = None,
    hook: Optional[Dict[str, Any]] = None,
    logo: Optional[Dict[str, Any]] = None,
    grade_preset: str = "none",
    content_mode: Optional[str] = "podcast",
    output_style: Optional[str] = "recap",
    theme: Optional[str] = None,
    merge_gap_seconds: float = 1.5,
    output_root: str = "output",
    api_key: Optional[str] = None,
    model_name: Optional[str] = None,
    title: Optional[str] = None,
    save_metadata: bool = True,
) -> Dict[str, Any]:
    """Stage 2: Full multi-layer rendering of the highlight reel with aspect ratio sizing, fast seeking, subtitles, hook, and logo."""
    job_dir = os.path.join(output_root, job_id)

    with job_metadata_lock(job_dir):
        metadata_path, data = load_or_create_job_metadata(job_id, output_root)
        source_video = _find_source_video_path(job_dir, data)
        words = _extract_transcript_words_from_job(data, job_dir)
        video_dur = float(data.get("duration", 0.0)) or (float(words[-1].get("end", 0.0)) if words else None)

        # 1. Resolve cuts (use provided custom cuts or generate plan)
        if cuts and len(cuts) > 0:
            snapped_cuts = snap_supercut_timeline(cuts, words, source_video_duration_sec=video_dur)
            reel_title = title or f"Highlights ({target_duration}s)"
        else:
            plan = plan_highlights_sync(
                job_id=job_id,
                target_duration=target_duration,
                content_mode=content_mode,
                output_style=output_style,
                theme=theme,
                merge_gap_seconds=merge_gap_seconds,
                output_root=output_root,
                api_key=api_key,
                model_name=model_name,
            )
            snapped_cuts = plan["cuts"]
            reel_title = title or plan.get("title", f"Highlights ({target_duration}s)")

        if not snapped_cuts:
            raise ComposeError("No valid cuts found to render highlight reel")

        target_w, target_h = get_aspect_dimensions(aspect)
        total_duration = sum(c["duration"] for c in snapped_cuts)

        # 2. Build FFmpeg command with input-level seeking for fast demuxer extraction
        temp_concat_out = os.path.join(job_dir, f"temp_hl_{uuid.uuid4().hex[:8]}.mp4")
        input_args = []
        filter_parts = []
        concat_inputs = []

        for i, cut in enumerate(snapped_cuts):
            s = cut["start"]
            e = cut["end"]
            # Input-side fast seek before each source input
            input_args.extend(["-ss", f"{s:.3f}", "-to", f"{e:.3f}", "-i", source_video])
            filter_parts.append(f"[{i}:v]setpts=PTS-STARTPTS[v{i}]")
            filter_parts.append(f"[{i}:a]asetpts=PTS-STARTPTS[a{i}]")
            concat_inputs.append(f"[v{i}][a{i}]")

        n = len(snapped_cuts)
        filter_parts.append(f"{''.join(concat_inputs)}concat=n={n}:v=1:a=1[rawv][outa]")

        # Apply aspect sizing and reframing filter
        if aspect == "16:9" or reframe_mode in ("disabled", "letterbox", "fit"):
            reframe_filter = f"[rawv]scale={target_w}:{target_h}:force_original_aspect_ratio=decrease,pad={target_w}:{target_h}:(ow-iw)/2:(oh-ih)/2:black[outv]"
        elif reframe_mode in ("crop", "center_crop"):
            reframe_filter = f"[rawv]scale={target_w}:{target_h}:force_original_aspect_ratio=increase,crop={target_w}:{target_h}[outv]"
        else:  # blur_pad, auto, subject default to high-retention blurred background
            reframe_filter = (
                f"[rawv]split=2[raw_bg][raw_fg];"
                f"[raw_bg]scale={target_w}:{target_h}:force_original_aspect_ratio=increase,crop={target_w}:{target_h},"
                f"boxblur=15:2[bg];"
                f"[raw_fg]scale={target_w}:{target_h}:force_original_aspect_ratio=decrease[fg];"
                f"[bg][fg]overlay=(W-w)/2:(H-h)/2[outv]"
            )

        filter_parts.append(reframe_filter)
        filter_str = ";".join(filter_parts)

        cmd = [
            "ffmpeg", "-y",
            *input_args,
            "-filter_complex", filter_str,
            "-map", "[outv]",
            "-map", "[outa]",
            *x264_video_args(),
            temp_concat_out,
        ]

        logger.info("🎬 Rendering highlight cuts & sizing via fast input-seeking FFmpeg...")
        try:
            res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=ffmpeg_timeout())
        except subprocess.TimeoutExpired:
            logger.error("FFmpeg highlight render timed out after %ss", ffmpeg_timeout())
            raise ComposeError(f"FFmpeg highlight concatenation timed out after {ffmpeg_timeout()}s", status_code=504)

        if res.returncode != 0:
            err_tail = res.stderr.decode(errors="replace")[-400:]
            logger.error("FFmpeg highlight concat error: %s", err_tail)
            raise ComposeError(f"FFmpeg highlight concatenation failed: {err_tail}", status_code=500)

        # 3. Subtitles & Karaoke Burning
        final_id = uuid.uuid4().hex[:6]
        final_filename = f"highlight_reel_{target_duration}s_{aspect.replace(':', 'x')}_{final_id}.mp4"
        final_output_path = os.path.join(job_dir, final_filename)

        sub_opts = subtitles or {}
        sub_enabled = sub_opts.get("enabled", True)
        fused_words = fuse_supercut_subtitles(snapped_cuts, words)

        current_video_file = temp_concat_out

        if sub_enabled and fused_words:
            ass_path = os.path.join(job_dir, f"temp_hl_sub_{final_id}.ass")
            fake_transcript = {"segments": [{"words": fused_words}]}
            sub_preset = sub_opts.get("preset", "classic_white")
            try:
                generate_ass_karaoke(
                    transcript=fake_transcript,
                    clip_start=0.0,
                    clip_end=total_duration,
                    output_path=ass_path,
                    preset=sub_preset,
                    font_name=sub_opts.get("font_name"),
                    font_color=sub_opts.get("font_color"),
                    highlight_color=sub_opts.get("highlight_color"),
                    font_size=sub_opts.get("font_size"),
                    outline_width=sub_opts.get("outline_width"),
                    position=sub_opts.get("position", "bottom"),
                    offset_y=sub_opts.get("offset_y", 0),
                    align=sub_opts.get("align", "center"),
                    animate=sub_opts.get("animate"),
                    keywords=sub_opts.get("keywords"),
                )
                subbed_temp = os.path.join(job_dir, f"temp_hl_subbed_{final_id}.mp4")
                burn_subtitles(
                    video_path=current_video_file,
                    srt_path=ass_path,
                    output_path=subbed_temp,
                )
                if os.path.exists(current_video_file):
                    os.remove(current_video_file)
                current_video_file = subbed_temp
                if os.path.exists(ass_path):
                    os.remove(ass_path)
            except Exception as se:
                logger.warning("Subtitle burn failed on highlight reel: %s", se)

        # 4. Color Grading Filter (if requested)
        if grade_preset and grade_preset.lower() not in ("none", ""):
            from clippyme.domain.grade import apply_grade
            graded_temp = os.path.join(job_dir, f"temp_hl_graded_{final_id}.mp4")
            try:
                if apply_grade(current_video_file, graded_temp, grade_preset):
                    if os.path.exists(current_video_file):
                        os.remove(current_video_file)
                    current_video_file = graded_temp
            except Exception as ge:
                logger.warning("Color grade failed on highlight reel: %s", ge)

        # 5. Hook Text Overlay (if requested)
        if hook:
            hook_dict = hook if isinstance(hook, dict) else {"text": str(hook)}
            hook_text = hook_dict.get("text", "")
            hook_enabled = hook_dict.get("enabled", True) if "enabled" in hook_dict else bool(hook_text)
            if hook_enabled and hook_text:
                from clippyme.domain.hooks import add_hook_to_video
                hooked_temp = os.path.join(job_dir, f"temp_hl_hooked_{final_id}.mp4")
                try:
                    pos = hook_dict.get("position", "top")
                    size_map = {"S": 0.8, "M": 1.0, "L": 1.25, "XL": 1.5}
                    font_scale = size_map.get(hook_dict.get("size", "M"), 1.0)
                    offset_y = hook_dict.get("offset_y", 0)
                    style_keys = (
                        "text_color", "bg_enabled", "bg_color", "bg_opacity",
                        "corner_radius", "outline_color", "outline_width", "font", "shadow", "animate"
                    )
                    style = {k: hook_dict[k] for k in style_keys if k in hook_dict}
                    hook_duration = min(4, max(1, int(total_duration)))
                    if add_hook_to_video(
                        video_path=current_video_file,
                        text=hook_text,
                        output_path=hooked_temp,
                        position=pos,
                        font_scale=font_scale,
                        offset_y=offset_y,
                        style=style or None,
                        hook_duration=hook_duration,
                    ):
                        if os.path.exists(current_video_file):
                            os.remove(current_video_file)
                        current_video_file = hooked_temp
                except Exception as he:
                    logger.warning("Hook overlay failed on highlight reel: %s", he)

        # 6. Brand Logo Overlay (if requested)
        if logo:
            logo_dict = logo if isinstance(logo, dict) else {}
            logo_enabled = logo_dict.get("enabled", True)
            logo_path = logo_dict.get("path") or os.path.join("data", "branding", "logo.png")
            if logo_enabled and os.path.exists(logo_path):
                from clippyme.domain.logo import add_logo_to_video, DEFAULT_POSITION
                logo_temp = os.path.join(job_dir, f"temp_hl_logo_{final_id}.mp4")
                try:
                    pos = logo_dict.get("position", DEFAULT_POSITION)
                    scale = float(logo_dict.get("scale", 0.18))
                    opacity = float(logo_dict.get("opacity", 1.0))
                    margin = float(logo_dict.get("margin", 0.04))
                    if add_logo_to_video(
                        video_path=current_video_file,
                        logo_path=logo_path,
                        output_path=logo_temp,
                        position=pos,
                        scale=scale,
                        opacity=opacity,
                        margin=margin,
                    ):
                        if os.path.exists(current_video_file):
                            os.remove(current_video_file)
                        current_video_file = logo_temp
                except Exception as le:
                    logger.warning("Logo overlay failed on highlight reel: %s", le)

        # Move to final destination
        if os.path.exists(current_video_file):
            os.replace(current_video_file, final_output_path)

        # 7. Register in metadata (if requested)
        highlight_entry = {
            "id": final_id,
            "title": reel_title,
            "target_duration": target_duration,
            "actual_duration": round(total_duration, 2),
            "aspect": aspect,
            "reframe_mode": reframe_mode,
            "content_mode": content_mode,
            "output_style": output_style,
            "theme": theme,
            "filename": final_filename,
            "video_url": f"/videos/{job_id}/{final_filename}",
            "cuts": snapped_cuts,
            "subtitles": subtitles,
            "hook": hook,
            "logo": logo,
            "grade_preset": grade_preset,
            "created_at": time.time(),
        }

        if save_metadata:
            highlights_list = data.get("highlights", [])
            highlights_list.insert(0, highlight_entry)
            data["highlights"] = highlights_list
            save_job_metadata(metadata_path, data)

        return {
            "success": True,
            "job_id": job_id,
            "highlight": highlight_entry,
        }


def delete_highlight_reel_sync(job_id: str, filename: str, output_root: str = "output") -> Dict[str, Any]:
    """Delete a generated highlight reel from disk and job metadata with thread safety."""
    job_dir = os.path.join(output_root, job_id)

    with job_metadata_lock(job_dir):
        metadata_path, data = load_or_create_job_metadata(job_id, output_root)

        safe_filename = os.path.basename(filename)
        file_path = os.path.join(job_dir, safe_filename)
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except OSError as e:
                logger.warning("Could not delete highlight reel file: %s", e)

        highlights_list = data.get("highlights", [])
        data["highlights"] = [h for h in highlights_list if h.get("filename") != safe_filename]
        save_job_metadata(metadata_path, data)

    return {"success": True, "deleted": safe_filename}


def generate_multi_tier_highlights_sync(
    job_id: str,
    aspect: str = "9:16",
    reframe_mode: str = "blur_pad",
    subtitles: Optional[Dict[str, Any]] = None,
    hook: Optional[Dict[str, Any]] = None,
    logo: Optional[Dict[str, Any]] = None,
    grade_preset: str = "none",
    content_mode: Optional[str] = "podcast",
    output_style: Optional[str] = "recap",
    theme: Optional[str] = None,
    merge_gap_seconds: float = 1.5,
    output_root: str = "output",
    api_key: Optional[str] = None,
    model_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Analyze full video and generate all duration tier packages (<60s, ~120s, 180s-720s) with atomic metadata locking."""
    job_dir = os.path.join(output_root, job_id)

    with job_metadata_lock(job_dir):
        metadata_path, data = load_or_create_job_metadata(job_id, output_root)

        words = _extract_transcript_words_from_job(data, job_dir)
        if not words:
            raise ValidationError("No transcript with word-level timestamps available for this job")

        cfg = load_persistent_config()
        gemini_key = api_key or cfg.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not gemini_key:
            raise ValidationError("Gemini API key is required to analyze and generate highlights")

        model = model_name or cfg.get("GEMINI_MODEL") or os.getenv("GEMINI_MODEL") or "gemini-3.5-flash"
        video_title = data.get("title", data.get("video_title", "Video"))
        video_dur = float(data.get("duration", 600.0))

        prompt = build_multi_tier_highlights_prompt(
            words,
            video_duration_sec=video_dur,
            video_title=video_title,
            content_mode=content_mode,
            output_style=output_style,
            theme=theme,
        )

        from google import genai
        from google.genai import types

        client = genai.Client(api_key=gemini_key)
        response = client.models.generate_content(
            model=model,
            contents=[prompt],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
            ),
        )

        parsed_reels = parse_multi_tier_response(
            response.text,
            merge_gap_seconds=merge_gap_seconds,
        )
        rendered_highlights = []

        for reel_spec in parsed_reels:
            tier = reel_spec["tier"]
            title = reel_spec["title"]
            viral_score = reel_spec["viral_score"]
            cuts = reel_spec["cuts"]
            target_dur = int(reel_spec["total_duration"])

            effective_subs = subtitles if subtitles is not None else {
                "enabled": True,
                "preset": "hormozi_bold",
                "position": "bottom",
            }

            try:
                # Pass save_metadata=False so orchestrator writes final state atomically
                render_res = render_highlight_reel_sync(
                    job_id=job_id,
                    target_duration=target_dur,
                    cuts=cuts,
                    aspect=aspect,
                    reframe_mode=reframe_mode,
                    subtitles=effective_subs,
                    hook=hook,
                    logo=logo,
                    grade_preset=grade_preset,
                    content_mode=content_mode,
                    output_style=output_style,
                    theme=theme,
                    merge_gap_seconds=merge_gap_seconds,
                    output_root=output_root,
                    title=title,
                    save_metadata=False,
                )
                hl = render_res["highlight"]
                hl["tier"] = tier
                hl["viral_score"] = viral_score
                hl["summary"] = reel_spec.get("summary", "")
                rendered_highlights.append(hl)
            except Exception as re:
                logger.error("Failed to render highlight tier %s: %s", tier, re)

        # Single atomic metadata update
        existing = data.get("highlights", [])
        data["highlights"] = rendered_highlights + [
            h for h in existing if h.get("id") not in [r["id"] for r in rendered_highlights]
        ]
        save_job_metadata(metadata_path, data)

    return {
        "success": True,
        "job_id": job_id,
        "highlights": rendered_highlights,
    }


def reprocess_highlight_reel_sync(
    job_id: str,
    highlight_id: str,
    edit_params: Dict[str, Any],
    output_root: str = "output",
) -> Dict[str, Any]:
    """Reprocess an existing highlight reel with updated edit parameters (reframe, subtitles, grade, hook, logo, trim)."""
    job_dir = os.path.join(output_root, job_id)

    with job_metadata_lock(job_dir):
        metadata_path, data = load_or_create_job_metadata(job_id, output_root)

        highlights = data.get("highlights", [])
        target_reel = None
        for h in highlights:
            if h.get("id") == highlight_id or h.get("filename") == highlight_id:
                target_reel = h
                break

        if not target_reel:
            raise NotFoundError(f"Highlight reel {highlight_id} not found in job {job_id}")

        aspect = edit_params.get("aspect") or target_reel.get("aspect", "9:16")
        reframe_mode = edit_params.get("reframe_mode") or target_reel.get("reframe_mode", "blur_pad")
        subtitles = edit_params.get("subtitles") or target_reel.get("subtitles")
        hook = edit_params.get("hook") or target_reel.get("hook")
        logo = edit_params.get("logo") or target_reel.get("logo")
        grade_preset = edit_params.get("grade_preset") or target_reel.get("grade_preset", "none")
        content_mode = edit_params.get("content_mode") or target_reel.get("content_mode", "podcast")
        output_style = edit_params.get("output_style") or target_reel.get("output_style", "recap")
        theme = edit_params.get("theme") or target_reel.get("theme")
        merge_gap_seconds = float(edit_params.get("merge_gap_seconds", 1.5))
        cuts = edit_params.get("cuts") or target_reel.get("cuts")
        title = edit_params.get("title") or target_reel.get("title")
        target_duration = int(target_reel.get("target_duration", 60))

        # Old filename to clean up after success
        old_filename = target_reel.get("filename")

        render_res = render_highlight_reel_sync(
            job_id=job_id,
            target_duration=target_duration,
            cuts=cuts,
            aspect=aspect,
            reframe_mode=reframe_mode,
            subtitles=subtitles,
            hook=hook,
            logo=logo,
            grade_preset=grade_preset,
            content_mode=content_mode,
            output_style=output_style,
            theme=theme,
            merge_gap_seconds=merge_gap_seconds,
            output_root=output_root,
            title=title,
            save_metadata=False,
        )

        new_highlight = render_res["highlight"]
        new_highlight["id"] = target_reel["id"]  # keep stable ID
        new_highlight["tier"] = target_reel.get("tier", "recap")
        new_highlight["viral_score"] = target_reel.get("viral_score", 85)

        # Replace old entry in metadata atomically
        updated_list = []
        for h in data.get("highlights", []):
            if h.get("id") == target_reel["id"] or h.get("filename") == old_filename:
                continue
            updated_list.append(h)
        updated_list.insert(0, new_highlight)
        data["highlights"] = updated_list
        save_job_metadata(metadata_path, data)

        # Clean old file if different
        if old_filename and old_filename != new_highlight.get("filename"):
            old_path = os.path.join(job_dir, old_filename)
            if os.path.exists(old_path):
                try:
                    os.remove(old_path)
                except OSError as e:
                    logger.warning("Could not clean old highlight reel file %s: %s", old_path, e)

    return {
        "success": True,
        "job_id": job_id,
        "highlight": new_highlight,
    }
