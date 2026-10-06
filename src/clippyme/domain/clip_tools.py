"""Clip-level utility operations for the chat-UI clip popup.

Owns the sync filesystem/ffmpeg work behind four additive endpoints:

  - POST /api/clips/{job_id}/{clip_index}/duplicate   (deep-copy a clip)
  - POST /api/clips/{job_id}/{clip_index}/schedule    (persist a Zernio schedule record)
  - POST /api/clips/{job_id}/{clip_index}/upscale     (ffmpeg 2x upscale, long edge <= 2560px)
  - GET  /api/clips/{job_id}/{clip_index}/export-xml  (FCP7 xmeml timeline)

Pure sync code — handlers wrap these in ``asyncio.to_thread``. Raises
``ClippyMeError`` subclasses (``NotFoundError`` -> 404, ``ValidationError`` ->
400) which the app-level handler maps to HTTP responses.
"""

import copy
import json
import logging
import os
import re
import shutil
import subprocess
import time
import uuid
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from xml.sax.saxutils import escape as _xml_escape

from clippyme.domain.clip_resolve import resolve_clip
from clippyme.domain.encode import (
    ffmpeg_timeout,
    video_encoder,
    x264_crf,
    x264_preset,
    x264_video_args,
)
from clippyme.domain.errors import ClippyMeError, NotFoundError, ValidationError
from clippyme.domain.job_artifacts import save_job_metadata

logger = logging.getLogger("clippyme")

# Upscale caps the long edge at this many pixels (matches the platform-safe
# ceiling used elsewhere); a clip already at/above it is rejected with 400.
_MAX_LONG_EDGE = 2560
# FCP7 xmeml timebase used for the exported timeline.
_XML_FPS = 30
# A single upscale pass on a short clip finishes in seconds; reuse the shared
# compose-layer timeout so a hung ffmpeg can't pin the API thread pool.
_FFMPEG_TIMEOUT = ffmpeg_timeout()

_PUBLISH_STATE_KEYS = ("published", "deleted_after_publish")
_ID_KEYS = ("id", "clip_id", "uuid")
_TITLE_KEYS = ("video_title_for_youtube_short", "title")


# ---------------------------------------------------------------------------
# Small shared helpers
# ---------------------------------------------------------------------------

def _atomic_write_json(path: str, data: dict) -> None:
    """Write JSON atomically (unique tmp sibling + os.replace)."""
    tmp_path = f"{path}.{os.getpid()}_{time.time_ns()}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")
    os.replace(tmp_path, path)


def _atomic_write_text(path: str, text: str) -> None:
    tmp_path = f"{path}.{os.getpid()}_{time.time_ns()}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp_path, path)


def _windows_safe(name: str) -> str:
    """Strip characters Windows forbids in filenames."""
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "", name).strip().rstrip(".")
    return cleaned


def _unique_filename(job_dir: str, stem: str, ext: str) -> str:
    """Return a non-colliding filename in the job dir (stem, stem_v2, ...)."""
    candidate = f"{stem}{ext}"
    n = 2
    while os.path.exists(os.path.join(job_dir, candidate)):
        candidate = f"{stem}_v{n}{ext}"
        n += 1
    return candidate


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Duplicate
# ---------------------------------------------------------------------------

def duplicate_clip(job_id: str, clip_index: int, output_root: str) -> dict:
    """Deep-copy a clip: new metadata entry, new rendered mp4, copied project.

    The source clip (metadata entry and file) is never modified. Returns
    ``{"job_id", "source_index", "new_index", "new_filename"}``.
    """
    resolved = resolve_clip(job_id, clip_index, output_root, require_file=True)
    job_dir = resolved.job_dir
    shorts = resolved.metadata.get("shorts") or []
    if clip_index < 0 or clip_index >= len(shorts):
        raise NotFoundError("Clip not found")
    new_index = len(shorts)

    src = copy.deepcopy(shorts[clip_index])

    # Drop publish/volatile state so the copy is a fresh, unpublished clip.
    for key in _PUBLISH_STATE_KEYS:
        src.pop(key, None)
    # Fresh ids wherever the schema carries any.
    for key in _ID_KEYS:
        if isinstance(src.get(key), str):
            src[key] = uuid.uuid4().hex

    # "(copy)" title suffix on the canonical title fields.
    for tkey in _TITLE_KEYS:
        title = src.get(tkey)
        if isinstance(title, str) and title and not title.rstrip().endswith("(copy)"):
            src[tkey] = f"{title} (copy)"

    # New filename: keep the title-stem convention, swap the positional suffix
    # for the copy's index so on-disk fallback scans stay unambiguous.
    src_stem = os.path.splitext(os.path.basename(resolved.clip_filename))[0]
    base = re.sub(r"_clip_\d+$", "", src_stem)
    new_stem = _windows_safe(f"{base} (copy)_clip_{new_index + 1}".strip()) \
        or f"clip_copy_{new_index + 1}"
    new_filename = f"{new_stem}.mp4"
    src["clip_filename"] = new_filename
    src["duplicated_from"] = clip_index

    # Point stale QA metrics at the copy (they describe the source render).
    qa = src.get("qa")
    if isinstance(qa, dict):
        metrics = qa.get("metrics")
        if isinstance(metrics, dict) and metrics.get("path"):
            metrics["path"] = new_filename

    # Copy the rendered file first (a failure here leaves no dangling
    # metadata reference), then persist the metadata atomically.
    new_path = os.path.join(job_dir, new_filename)
    if os.path.exists(new_path):
        raise ClippyMeError(
            f"Duplicate target already exists: {new_filename}", status_code=409)
    shutil.copy2(resolved.clip_path, new_path)
    shorts.append(src)
    save_job_metadata(resolved.metadata_path, resolved.metadata)

    # Copy the clip-project JSON if the source has one, retargeted at the copy.
    try:
        from clippyme.domain.project_render import get_project_path
        src_proj_path = get_project_path(job_dir, clip_index)
        if src_proj_path:
            with open(src_proj_path, encoding="utf-8") as f:
                proj = json.load(f)
            if isinstance(proj, dict):
                proj["clip_index"] = new_index
                proj_source = proj.get("source")
                if isinstance(proj_source, dict):
                    proj_source["file"] = new_filename
                _atomic_write_json(
                    os.path.join(job_dir, f"clip-project-{new_index}.json"), proj)
    except ClippyMeError:
        raise
    except Exception as exc:
        # The clip itself is duplicated; a project-copy hiccup must not fail
        # the response (the project can be regenerated from the clip).
        logger.warning("duplicate: project copy failed for %s/%d: %s",
                       job_id, clip_index, exc)

    logger.info("duplicate: job=%s clip=%d -> new_index=%d file=%s",
                job_id, clip_index, new_index, new_filename)
    return {
        "job_id": job_id,
        "source_index": clip_index,
        "new_index": new_index,
        "new_filename": new_filename,
    }


# ---------------------------------------------------------------------------
# Schedule records (the Zernio timed post itself is done by publish_clip_flow)
# ---------------------------------------------------------------------------

def _schedule_filename(clip_index: int) -> str:
    return f"clip-schedule-{clip_index}.json"


def read_schedule_record(job_dir: str, clip_index: int):
    """Return the persisted schedule record, or None when never scheduled."""
    path = os.path.join(job_dir, _schedule_filename(clip_index))
    if not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def write_schedule_record(job_dir: str, clip_index: int, record: dict) -> str:
    """Persist a schedule record atomically; returns the record path."""
    payload = dict(record or {})
    payload["scheduled"] = True
    payload.setdefault("clip_index", clip_index)
    payload.setdefault("created_at", _utcnow_iso())
    path = os.path.join(job_dir, _schedule_filename(clip_index))
    _atomic_write_json(path, payload)
    return path


# ---------------------------------------------------------------------------
# Upscale
# ---------------------------------------------------------------------------

def _probe_streams(path: str) -> dict:
    """ffprobe width/height/audio codec for one file."""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "stream=width,height,codec_name,codec_type",
        "-of", "json", path,
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    except FileNotFoundError as exc:
        raise ClippyMeError(f"ffprobe not available: {exc}", status_code=500)
    except subprocess.TimeoutExpired:
        raise ClippyMeError("ffprobe timed out probing the clip", status_code=500)
    if proc.returncode != 0:
        raise ClippyMeError(
            f"Could not probe clip dimensions: {(proc.stderr or '')[-300:]}",
            status_code=500)
    try:
        data = json.loads(proc.stdout or "{}")
    except ValueError:
        raise ClippyMeError("Could not parse ffprobe output", status_code=500)
    streams = data.get("streams") or []
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
    if not video or not video.get("width") or not video.get("height"):
        raise ValidationError("Clip file has no readable video stream")
    return {
        "width": int(video["width"]),
        "height": int(video["height"]),
        "audio_codec": (audio or {}).get("codec_name"),
    }


@lru_cache(maxsize=1)
def _nvenc_available() -> bool:
    """True when this ffmpeg build ships the h264_nvenc encoder (cached)."""
    try:
        proc = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            capture_output=True, text=True, timeout=30)
    except Exception:
        return False
    return proc.returncode == 0 and "h264_nvenc" in (proc.stdout or "")


def _upscale_video_args() -> list:
    """NVENC args via the shared encoder settings, libx264 fallback.

    ``x264_video_args()`` already resolves h264_nvenc (p4, cq 20, yuv420p,
    faststart) from the environment; the only thing it can't know is whether
    this ffmpeg build actually ships NVENC — hence the availability probe.
    """
    if video_encoder() == "nvenc" and _nvenc_available():
        return x264_video_args()
    logger.info("upscale: h264_nvenc unavailable, falling back to libx264")
    return [
        "-c:v", "libx264",
        "-preset", x264_preset(),
        "-crf", str(x264_crf()),
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
    ]


def _even(n: int) -> int:
    return n if n % 2 == 0 else n + 1


def upscale_clip(job_id: str, clip_index: int, output_root: str) -> dict:
    """ffmpeg 2x lanczos upscale of the resolved clip, long edge <= 2560px.

    Writes ``upscaled_<stem>.mp4`` next to the original (never overwrites it),
    keeps audio, yuv420p + faststart. Runs synchronously — handlers execute it
    in ``asyncio.to_thread``.
    """
    resolved = resolve_clip(job_id, clip_index, output_root, require_file=True)
    job_dir = resolved.job_dir
    probe = _probe_streams(resolved.clip_path)
    width, height = probe["width"], probe["height"]
    long_edge = max(width, height)
    if long_edge >= _MAX_LONG_EDGE:
        raise ValidationError(
            f"Clip is already at maximum resolution "
            f"(long edge {long_edge}px >= {_MAX_LONG_EDGE}px)")

    factor = min(2.0, _MAX_LONG_EDGE / long_edge)
    out_w = _even(max(2, int(round(width * factor))))
    out_h = _even(max(2, int(round(height * factor))))
    # Even-rounding can push the long edge one step over the cap.
    while max(out_w, out_h) > _MAX_LONG_EDGE:
        if out_w >= out_h:
            out_w -= 2
        else:
            out_h -= 2

    src_stem = os.path.splitext(os.path.basename(resolved.clip_filename))[0]
    out_stem = _windows_safe(f"upscaled_{src_stem}") or "upscaled_clip"
    out_name = _unique_filename(job_dir, out_stem, ".mp4")
    out_path = os.path.join(job_dir, out_name)

    if probe.get("audio_codec") == "aac":
        audio_args = ["-c:a", "copy"]
    else:
        audio_args = ["-c:a", "aac", "-b:a", "192k"]

    cmd = [
        "ffmpeg", "-y",
        "-i", resolved.clip_path,
        "-vf", f"scale={out_w}:{out_h}:flags=lanczos",
        *_upscale_video_args(),
        *audio_args,
        out_path,
    ]
    logger.info("upscale: job=%s clip=%d %dx%d -> %dx%d",
                job_id, clip_index, width, height, out_w, out_h)
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=_FFMPEG_TIMEOUT)
    except subprocess.TimeoutExpired:
        raise ClippyMeError("Upscale timed out", status_code=500)
    if proc.returncode != 0:
        raise ClippyMeError(
            f"Upscale failed (ffmpeg exited {proc.returncode}): "
            f"{(proc.stderr or '')[-500:]}",
            status_code=500)
    if not os.path.isfile(out_path):
        raise ClippyMeError("Upscale failed: no output file produced",
                            status_code=500)

    return {
        "file": out_name,
        "download_url": f"/videos/{job_id}/{out_name}",
        "width": out_w,
        "height": out_h,
    }


# ---------------------------------------------------------------------------
# FCP7 (xmeml) XML export
# ---------------------------------------------------------------------------

def _select_segments(project, clip_info) -> list:
    """Timeline segments as (start_s, end_s): project segments first, then
    the clip's start/end bounds. Empty list when neither exists."""
    segments = []
    if isinstance(project, dict):
        for seg in project.get("segments") or []:
            if not isinstance(seg, dict):
                continue
            start, end = seg.get("start"), seg.get("end")
            if (isinstance(start, (int, float)) and isinstance(end, (int, float))
                    and end > start >= 0):
                segments.append((float(start), float(end)))
    if not segments:
        start, end = clip_info.get("start"), clip_info.get("end")
        if (isinstance(start, (int, float)) and isinstance(end, (int, float))
                and end > start >= 0):
            segments = [(float(start), float(end))]
    return segments


def _clipitem_xml(item_id: str, file_id: str, name: str, file_uri: str,
                  tl_start: int, tl_end: int, src_in: int, src_out: int,
                  total_frames: int, width: int, height: int) -> str:
    esc_name = _xml_escape(name)
    return f"""      <clipitem id="{item_id}">
        <name>{esc_name}</name>
        <duration>{total_frames}</duration>
        <rate>
          <timebase>{_XML_FPS}</timebase>
          <ntsc>FALSE</ntsc>
        </rate>
        <start>{tl_start}</start>
        <end>{tl_end}</end>
        <in>{src_in}</in>
        <out>{src_out}</out>
        <file id="{file_id}">
          <name>{esc_name}</name>
          <pathurl>{_xml_escape(file_uri)}</pathurl>
          <rate>
            <timebase>{_XML_FPS}</timebase>
            <ntsc>FALSE</ntsc>
          </rate>
          <duration>{total_frames}</duration>
          <media>
            <video>
              <duration>{total_frames}</duration>
              <samplecharacteristics>
                <rate>
                  <timebase>{_XML_FPS}</timebase>
                  <ntsc>FALSE</ntsc>
                </rate>
                <width>{width}</width>
                <height>{height}</height>
                <anamorphic>FALSE</anamorphic>
                <pixelaspectratio>square</pixelaspectratio>
                <fielddominance>none</fielddominance>
              </samplecharacteristics>
            </video>
            <audio>
              <samplecharacteristics>
                <depth>16</depth>
                <samplerate>48000</samplerate>
              </samplecharacteristics>
              <channelcount>2</channelcount>
            </audio>
          </media>
        </file>
      </clipitem>
"""


def _build_xmeml(sequence_name: str, clip_name: str, file_uri: str,
                 segments: list, width: int, height: int) -> str:
    """Minimal valid FCP7 xmeml: one sequence, video + audio tracks, one
    clipitem per timeline segment placed back-to-back from frame 0."""
    items = []
    cursor = 0
    for i, (start_s, end_s) in enumerate(segments, start=1):
        src_in = int(round(start_s * _XML_FPS))
        src_out = int(round(end_s * _XML_FPS))
        dur = max(1, src_out - src_in)
        tl_start = cursor
        tl_end = cursor + dur
        cursor = tl_end
        items.append((f"clipitem-{i}", f"file-{i}", tl_start, tl_end,
                      src_in, src_in + dur))
    total_frames = cursor

    video_items = "".join(
        _clipitem_xml(iid, fid, clip_name, file_uri, ts, te, si, so,
                      total_frames, width, height)
        for iid, fid, ts, te, si, so in items)
    audio_items = "".join(
        _clipitem_xml(f"{iid}-a", f"{fid}-a", clip_name, file_uri, ts, te,
                      si, so, total_frames, width, height)
        for iid, fid, ts, te, si, so in items)

    esc_seq = _xml_escape(sequence_name)
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE xmeml>
<xmeml version="4">
  <sequence id="sequence-1">
    <name>{esc_seq}</name>
    <duration>{total_frames}</duration>
    <rate>
      <timebase>{_XML_FPS}</timebase>
      <ntsc>FALSE</ntsc>
    </rate>
    <media>
      <video>
        <format>
          <samplecharacteristics>
            <rate>
              <timebase>{_XML_FPS}</timebase>
              <ntsc>FALSE</ntsc>
            </rate>
            <width>{width}</width>
            <height>{height}</height>
            <anamorphic>FALSE</anamorphic>
            <pixelaspectratio>square</pixelaspectratio>
            <fielddominance>none</fielddominance>
          </samplecharacteristics>
        </format>
        <track>
{video_items}        </track>
      </video>
      <audio>
        <track>
{audio_items}        </track>
      </audio>
    </media>
  </sequence>
</xmeml>
"""


def export_timeline_xml(job_id: str, clip_index: int, output_root: str):
    """Generate the FCP7 xmeml timeline for a clip; returns (path, filename).

    Source data: the clip's project JSON segments, falling back to the clip's
    start/end bounds. Raises ``ValidationError`` (400) when neither exists —
    never emits an empty timeline.
    """
    resolved = resolve_clip(job_id, clip_index, output_root, require_file=True)
    job_dir = resolved.job_dir

    project = None
    try:
        from clippyme.domain.project_render import get_project_path
        project_path = get_project_path(job_dir, clip_index)
        if project_path:
            with open(project_path, encoding="utf-8") as f:
                project = json.load(f)
    except Exception as exc:
        logger.warning("export-xml: could not load project for %s/%d: %s",
                       job_id, clip_index, exc)
        project = None

    segments = _select_segments(project, resolved.clip_info)
    if not segments:
        raise ValidationError(
            "No timeline data for this clip: no clip-project segments and "
            "no start/end bounds in the clip metadata")

    probe = _probe_streams(resolved.clip_path)
    width, height = probe["width"], probe["height"]
    clip_name = os.path.splitext(os.path.basename(resolved.clip_filename))[0]
    file_uri = Path(os.path.abspath(resolved.clip_path)).as_uri()
    xml = _build_xmeml(f"{clip_name} timeline", clip_name, file_uri,
                       segments, width, height)

    filename = _unique_filename(
        job_dir, _windows_safe(f"{clip_name}_timeline") or "clip_timeline",
        ".xml")
    path = os.path.join(job_dir, filename)
    _atomic_write_text(path, xml)
    logger.info("export-xml: job=%s clip=%d -> %s (%d segments)",
                job_id, clip_index, filename, len(segments))
    return path, filename
