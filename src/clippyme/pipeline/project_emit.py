"""Phase 1 (clip editor): emit the auto-pipeline's "default edit" as clip-project.json.

Called once per clip from ``orchestrator._render_one_clip`` right after the clip
passes QA and its metadata is finalized (including the resume path, so re-run
jobs backfill missing projects).

Purely additive:
- Reads: the clip dict, ``clips_data`` (metadata + transcript), the source slice
  and final mp4 (probed, never modified).
- Writes: exactly one new file, ``clip-project-<index>.json``, next to
  ``_metadata.json``.
- Never raises: any failure is logged and the job continues. A missing or
  invalid project must never break a working pipeline.

Derivation (1:1 from existing data):
- ``segments``: single segment from the clip's snapped ``start``/``end``.
  ``crop`` is the static 9:16 center-crop box in source-slice pixel coordinates.
  (The AI's per-frame face-track trajectory is not persisted by the pipeline;
  the center box is the geometric summary the v1 schema supports.)
- ``captions.words``: clip-relative word timings, using the same window rule as
  ``domain.smartcut_ops.clip_transcript_segments`` (words overlapping
  ``[start, end)``, times relative to clip start, end clamped to clip end).
  ``style``/``position`` mirror the dashboard ``seedClipParams`` canonical
  defaults (``classic_white`` / ``bottom``).
- ``overlays``: the AI's hook text (``clip["hook"]`` or ``clip["viral_hook_text"]``)
  as a top-anchored hook overlay, only when non-empty.
- ``audio``: single null track (no music in v1; schema-ready).
- ``grade``: default preset.
- ``render``: width/height/fps probed from the actual final mp4; encoder
  resolved via ``domain.encode.video_encoder()`` (same as the render itself).
- ``source``: the kept ``source_<clip>.mp4`` slice name + probed dimensions.
- ``version`` = 1, ``origin`` = "auto", fresh ``idempotency_key`` (uuid4).

Filename convention: ``clip-project-<clip_index>.json`` (documented here;
Phase 2 serves it via ``GET /api/project/{job_id}/{clip_index}``).
"""

from __future__ import annotations

import logging
import os
from typing import Any

from clippyme.domain.clip_project import SCHEMA_TAG, project_to_json, validate_project
from clippyme.domain.encode import video_encoder
from clippyme.pipeline.media_qa import probe_media

logger = logging.getLogger(__name__)

PROJECT_FILENAME_TEMPLATE = "clip-project-{index}.json"
"""Per-clip project filename, e.g. ``clip-project-0.json``."""

# Canonical caption defaults, mirroring dashboard seedClipParams.js
# (seedSubtitleParams: preset 'classic_white', position 'bottom').
CAPTION_STYLE_DEFAULT = "classic_white"
CAPTION_POSITION_DEFAULT = "bottom"


def _as_int(value: Any, default: int) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _clip_relative_words(
    transcript: dict[str, Any], start: float, end: float
) -> list[dict[str, Any]]:
    """Clip-relative word timings.

    Same window rule as ``domain.smartcut_ops.clip_transcript_segments``:
    words overlapping ``[start, end)`` contribute, times made relative to the
    clip start, end clamped to the clip end.
    """
    out: list[dict[str, Any]] = []
    segments = transcript.get("segments", []) if isinstance(transcript, dict) else []
    for seg in segments:
        for w in seg.get("words") or []:
            ws, we = w.get("start"), w.get("end")
            if ws is None or we is None:
                continue
            if we <= start or ws >= end:
                continue
            text = (w.get("word") or "").strip()
            if not text:
                continue
            rs_raw = max(0.0, ws - start)
            re_raw = max(rs_raw, min(we, end) - start)
            rs, re = round(rs_raw, 3), round(re_raw, 3)
            if re <= rs:
                continue  # sub-millisecond after rounding; not displayable
            out.append({"w": text, "start": rs, "end": re})
    return out


def _center_crop_916(src_w: int, src_h: int) -> dict[str, float]:
    """Static 9:16 center-crop box in source pixel coordinates."""
    crop_w = src_h * 9.0 / 16.0
    return {
        "x": round((src_w - crop_w) / 2.0, 2),
        "y": 0.0,
        "w": round(crop_w, 2),
        "h": float(src_h),
    }


def _build_project(
    *,
    job_id: str,
    index: int,
    start: float,
    end: float,
    clip: dict[str, Any],
    clips_data: dict[str, Any],
    clip_source: str,
    clip_final: str,
) -> dict[str, Any]:
    if not (end > start >= 0):
        raise ValueError(f"invalid clip bounds start={start} end={end}")

    src_probe = probe_media(clip_source)
    src_w, src_h = src_probe.get("width"), src_probe.get("height")
    if not src_w or not src_h:
        raise ValueError(
            f"cannot determine source slice dimensions for {clip_source!r}"
        )

    final_probe = probe_media(clip_final)
    render_w = _as_int(final_probe.get("width"), 608)
    render_h = _as_int(final_probe.get("height"), 1080)
    render_fps = _as_int(final_probe.get("fps"), 30)

    transcript = clips_data.get("transcript") or {}
    words = _clip_relative_words(transcript, start, end)

    hook_text = (clip.get("hook") or clip.get("viral_hook_text") or "").strip()
    duration = end - start
    overlays: list[dict[str, Any]] = []
    if hook_text:
        overlays.append(
            {
                "type": "hook",
                "text": hook_text,
                "style": "viral",
                "start": 0.0,
                "end": round(min(3.0, duration), 3),
                "position": "top",
            }
        )

    return {
        "schema": SCHEMA_TAG,
        "job_id": job_id,
        "clip_index": index,
        "version": 1,
        "origin": "auto",
        "source": {
            "file": os.path.basename(clip_source),
            "width": int(src_w),
            "height": int(src_h),
            "duration": round(duration, 3),
        },
        # Phase 2: segments are in SOURCE-FILE coordinates (0 = first frame
        # of source.file, the already-cut slice). The slice is cut at exactly
        # [start, end], so the single default segment is [0, duration].
        "segments": [
            {
                "id": "seg-1",
                "start": 0.0,
                "end": round(duration, 3),
                "crop": _center_crop_916(int(src_w), int(src_h)),
                "transition_in": "cut",
            }
        ],
        "captions": {
            "style": CAPTION_STYLE_DEFAULT,
            "position": CAPTION_POSITION_DEFAULT,
            "words": words,
            "edits": {},
        },
        "overlays": overlays,
        "audio": [{"file": None, "volume": 1.0, "offset": 0.0}],
        "grade": {"preset": "default"},
        "render": {
            "width": render_w,
            "height": render_h,
            "fps": render_fps,
            "encoder": video_encoder(),
        },
    }


def emit_clip_project(
    *,
    output_dir: str,
    job_id: str,
    index: int,
    start: float,
    end: float,
    clip: dict[str, Any],
    clips_data: dict[str, Any],
    clip_source: str,
    clip_final: str,
) -> str | None:
    """Derive and write ``clip-project-<index>.json`` for one clip.

    Returns the written path, or ``None`` if emission failed (never raises).
    """
    try:
        data = _build_project(
            job_id=job_id,
            index=index,
            start=start,
            end=end,
            clip=clip,
            clips_data=clips_data,
            clip_source=clip_source,
            clip_final=clip_final,
        )
        project = validate_project(data)
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, PROJECT_FILENAME_TEMPLATE.format(index=index))
        with open(path, "w", encoding="utf-8") as f:
            f.write(project_to_json(project))
        print(f"clip-project: wrote {os.path.basename(path)}", flush=True)
        return path
    except Exception as exc:  # never fail a good clip over the project file
        logger.warning(
            "clip-project emission failed for clip %d (job %s): %s",
            index,
            job_id,
            exc,
        )
        return None
