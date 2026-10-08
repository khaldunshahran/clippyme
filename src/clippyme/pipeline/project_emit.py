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
- ``framing`` (Task 13): the accepted Task 12 turn-aware plan, validated and
  normalized to clip-relative frames by ``domain.project_framing`` -- explicit
  version, stable job-relative sidecar reference, content hash (so
  same-size/same-mtime replacement is detected), source fingerprint, render
  geometry, ordered scene strategies, and a normalized timeline. Absent when
  no usable sidecar exists or the domain helper is unavailable.
- ``version`` = 1, ``origin`` = "auto", fresh ``idempotency_key`` (uuid4).

Filename convention: ``clip-project-<clip_index>.json`` (documented here;
Phase 2 serves it via ``GET /api/project/{job_id}/{clip_index}``).
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from clippyme.domain.clip_project import SCHEMA_TAG, project_to_json, validate_project
from clippyme.domain.project_lineage import build_lineage
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
    job_dir: str | None = None,
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

    result = {
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
            "style": clips_data.get("caption_style_default") or CAPTION_STYLE_DEFAULT,
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
    # Task 13: stable framing trajectory bridge. Build the project framing
    # from the Task 12 sidecar the caller promoted to the stable final
    # basename. Never raises: a missing/unusable sidecar or an unavailable
    # domain helper simply leaves the field out.
    # Built BEFORE the Task 05 lineage block so the lineage baseline can
    # capture the framing identity.
    framing: dict[str, Any] | None = None
    if job_dir:
        try:
            from clippyme.domain import project_framing as _pf
        except ImportError:
            _pf = None
        if _pf is not None:
            # Task 13 review-01: bind the sidecar's Task 12 source identity
            # to the ACTUAL current source slice. A structurally valid plan
            # from a different source is rejected (no framing field), never
            # trusted.
            _expected_fp = None
            try:
                from clippyme.pipeline.framing_plan import (
                    source_fingerprint as _task12_fp,
                )
                _expected_fp = _task12_fp(clip_source)
            except Exception:
                _expected_fp = None
            try:
                framing = _pf.build_project_framing(
                    job_dir=job_dir,
                    clip_final=clip_final,
                    render_w=render_w,
                    render_h=render_h,
                    render_fps=render_fps,
                    # The render used render_fps (probed from the accepted
                    # final mp4); the clip is the already-cut source slice,
                    # so its frame count is duration * render fps.
                    clip_frame_count=int(round(duration * render_fps)),
                    expected_source_fingerprint=_expected_fp,
                )
            except Exception:
                framing = None
    if framing:
        result["framing"] = framing
    # Task 05: reframed-base lineage for automatic framing preservation.
    # Only for auto-emitted projects; job_dir needed for safe references.
    if job_dir:
        _source = result["source"]
        _lineage = build_lineage(
            clip_final=clip_final,
            job_dir=job_dir,
            source=_source,
            segments=result["segments"],
            render=result["render"],
            framing=framing,
        )
        if _lineage:
            result["reframed_base"] = _lineage
    return result


def project_has_user_edits(project_path: str) -> bool:
    """True when a working project file must be preserved, not overwritten.

    Real project schema (see ``_build_project``):
    - ``origin``: "auto" (fresh) vs "user" (edited in the clip editor)
    - ``version``: 1 (fresh); the editor bumps it on every user save
    - ``captions.edits``: {} (fresh) vs non-empty dict (caption edits)
    - ``segments`` / ``overlays``: user may also edit these nested structures

    Conservative: a missing file returns False (nothing to preserve, safe
    to create). An UNREADABLE or malformed existing file returns True --
    it must NOT be classified as safe to overwrite.
    """
    import json

    try:
        with open(project_path, encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        return False
    except (OSError, ValueError):
        # Unreadable existing file: preserve, do not overwrite.
        return True
    if not isinstance(data, dict):
        return True
    if data.get("origin") == "user":
        return True
    try:
        if int(data.get("version", 1)) > 1:
            return True
    except (TypeError, ValueError):
        return True
    captions = data.get("captions")
    if isinstance(captions, dict) and captions.get("edits"):
        return True
    # Nested timeline/segment edits: any segment with a user origin marker
    segments = data.get("segments")
    if isinstance(segments, list):
        for seg in segments:
            if isinstance(seg, dict) and seg.get("origin") == "user":
                return True
    return False



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
            job_dir=output_dir,
        )
        project = validate_project(data)
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, PROJECT_FILENAME_TEMPLATE.format(index=index))
        # Task 13: atomic write -- a crash mid-emit must never leave a
        # partial project behind; the temp file is cleaned in finally on all
        # paths. The project_has_user_edits preservation contract (checked by
        # the caller before emission) is unchanged.
        # Task 13 review-01: when there is no usable sidecar the framing
        # field is ABSENT from the emitted JSON (not null) -- the
        # pre-Task-13 shape. model_dump_json would otherwise serialize
        # `"framing": null`, which still counts as "present".
        payload = project_to_json(project)
        if project.framing is None:
            _doc = json.loads(payload)
            _doc.pop("framing", None)
            payload = json.dumps(_doc, indent=2, ensure_ascii=False)
        tmp_path = path + ".tmp"
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                f.write(payload)
            os.replace(tmp_path, path)
        finally:
            try:
                if os.path.isfile(tmp_path):
                    os.remove(tmp_path)
            except OSError:
                pass
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
