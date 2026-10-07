"""Pure helpers for the pipeline entrypoint (host-testable).

Extracted from ``pipeline.main``'s ``__main__`` block, which imports
cv2/torch/mediapipe at module top and therefore can't run on the dev host.
Only stdlib + ``domain.encode`` here.
"""
import os
import re

from clippyme.domain.encode import x264_video_args

_VIDEO_SUFFIXES = {".mp4", ".mkv", ".mov", ".webm", ".avi"}

# Windows-forbidden filename characters + ASCII control chars + URL-breaking chars (#, %).
_FORBIDDEN_CHARS_RE = re.compile(r'[<>:"/\\|?*#%\x00-\x1f]')
_WHITESPACE_RE = re.compile(r"\s+")
_RESERVED_NAMES = frozenset(
    {"CON", "PRN", "AUX", "NUL"}
    | {f"COM{i}" for i in range(1, 10)}
    | {f"LPT{i}" for i in range(1, 10)}
)


def sanitize_windows_basename(title: str | None, max_len: int = 80) -> str | None:
    """Windows-safe basename (no extension, no suffix) or None.

    Strips forbidden chars (<>:"/\\|?* + control), collapses whitespace,
    trims trailing dots/spaces, rejects reserved names (CON/PRN/AUX/NUL/
    COM1-9/LPT1-9). Truncates on a word boundary at ``max_len``. Returns
    None when nothing usable survives (caller supplies its own fallback).
    """
    if not title or not isinstance(title, str):
        return None
    cleaned = _FORBIDDEN_CHARS_RE.sub("", title)
    cleaned = _WHITESPACE_RE.sub(" ", cleaned).strip()
    cleaned = cleaned.strip(". ")
    if not cleaned or cleaned.upper() in _RESERVED_NAMES:
        return None
    if len(cleaned) > max_len:
        cut = cleaned[:max_len]
        boundary, _, _ = cut.rpartition(" ")
        if boundary:
            cut = boundary
        cleaned = cut.strip(". ")
        if not cleaned:
            return None
    return cleaned


def clip_output_basename(
    title: str | None, index: int, fallback_base: str, max_len: int = 80
) -> str:
    """Windows-safe on-disk basename (no extension) for one clip.

    Prefers the clip's Gemini title, sanitized for Windows filesystems;
    falls back to the legacy ``{fallback_base}_clip_{index+1}`` convention
    for a missing/empty/reserved/all-forbidden title. Every result — title-
    or fallback-derived — carries the ``_clip_{index+1}`` suffix, so
    filenames stay unique across a job's clips.
    """
    fallback = f"{fallback_base}_clip_{index + 1}"
    cleaned = sanitize_windows_basename(title, max_len=max_len)
    if cleaned is None:
        return fallback
    return f"{cleaned}_clip_{index + 1}"


def resolve_output_dir(out: str | None, default: str) -> str:
    """Treat ``out`` as a directory unless it has a video suffix.

    Fixes the edge case where a user passes a new (non-existent) directory
    and the old logic called ``os.path.dirname`` on it, landing the output
    one level above the intended dir. Creates the directory when needed.
    """
    if not out:
        return default
    if os.path.splitext(out)[1].lower() in _VIDEO_SUFFIXES:
        return os.path.dirname(out) or default
    os.makedirs(out, exist_ok=True)
    return out


def should_use_fallback(monitor_mode: bool) -> bool:
    """Whether to use the no-AI fallbacks (TextTiling / whole-video).

    Monitor jobs must NEVER publish fallback clips (topic "part N" chunks or a
    30-min whole-video render), so they get zero clips instead. Normal jobs
    keep the fallbacks.
    """
    return not monitor_mode


def build_vfr_normalization_command(input_video: str, dest: str) -> list[str]:
    """ffmpeg argv for converting a VFR source to CFR before reframe."""
    return [
        "ffmpeg", "-y", "-i", input_video,
        "-vsync", "cfr",
        *x264_video_args(faststart=False),
        "-c:a", "copy", dest,
    ]


# Bump when the slice contract changes in a way that invalidates slices
# cut by an older implementation (resume must re-cut them).
# v4: timing contract changed (normalized shared-clock trim; no input
# seek, no per-stream offset additions). v3 slices used incorrect content
# selection and must be re-cut, not reused.
_SLICE_PROVENANCE_VERSION = 4

# --- Timing convention (single source of truth) ------------------------------
# A source slice is a half-open interval [start, end) on the SOURCE timeline,
# re-based so the slice's own presentation timeline starts at exactly 0.
#
# START policy: the first retained frame is the first decoded source frame
#   whose presentation timestamp is >= the requested start (ceil to frame
#   boundary). ``-ss`` is placed AFTER ``-i`` (output seeking): ffmpeg
#   decodes from the nearest seek point and discards everything before
#   ``start``. Head offset is in [0, 1/fps) -- sub-frame, never a GOP.
# END policy: the last retained frame is the last frame with presentation
#   timestamp < the requested end. Callers pass ``frames`` (computed from
#   probed source fps as floor((end - start) * fps)) so the frame count is
#   deterministic; a frame exactly AT ``end`` is NOT retained.
# PRESENTATION ORIGIN (normalized shared-clock baseline):
#   ``-i source -vf trim=start=S:end=E,setpts=PTS-S/TB
#     -af atrim=start=S:end=E,asetpts=PTS-S/TB -fps_mode passthrough``
#   Both streams are trimmed to the SAME [S, E) on the normalized
#   container-elapsed clock, then both are re-based by subtracting the SAME
#   requested S. FFmpeg's container-origin normalization handles genuine
#   stream start offsets; we do NOT add stream start_times to trim
#   boundaries (that selects different source intervals per stream).
#   No input -ss: seeking changes the clock the trim filters see.
#   ``fps_mode passthrough`` preserves VFR frame timing.
#   Slice time T corresponds to source container-elapsed time (T + S).
#   Encoder sub-frame offsets (B-frame delay, AAC priming) may appear as
#   small positive first timestamps; consumers map against requested S.
#
# All downstream consumers (word timings, compose, Smart Cut, project emit,
# reframe audio extraction) use the REQUESTED start as slice time 0, which
# is the shared rebase origin. No consumer re-rounds independently.


def build_cut_command(
    input_video: str, start: float, end: float, dest: str,
) -> list[str]:
    """ffmpeg argv for cutting the 16:9 source slice of one clip.

    Normalized shared-clock baseline (verified reference):
    ``-i source -vf trim=start=S:end=E,setpts=PTS-S/TB
      -af atrim=start=S:end=E,asetpts=PTS-S/TB -fps_mode passthrough``

    Both streams trim to the SAME [S, E) and rebase by the SAME S.
    No input -ss (it changes the clock trim sees). ``fps_mode passthrough``
    preserves VFR timing. Video/audio re-encoded with the shared policy
    (``domain.encode.x264_video_args``); ``-map 0:a?`` keeps audio
    optional without probing. Correct timing has priority over cut speed.
    """
    start_f, end_f = float(start), float(end)
    if not (end_f - start_f > 0):
        raise ValueError(f"invalid cut interval start={start_f} end={end_f}")
    if start_f < 0:
        raise ValueError(f"cut start must be >= 0, got {start_f}")
    s = f"{start_f:.3f}"
    e = f"{end_f:.3f}"
    return [
        'ffmpeg', '-y',
        '-i', input_video,
        '-map', '0:v:0',
        '-map', '0:a?',
        '-vf', f'trim=start={s}:end={e},setpts=PTS-{s}/TB',
        '-af', f'atrim=start={s}:end={e},asetpts=PTS-{s}/TB',
        '-fps_mode', 'passthrough',
        *x264_video_args(faststart=False),
        '-c:a', 'aac',
        dest,
    ]



def source_fingerprint(path: str) -> dict:
    """Identity of a source file, captured BEFORE a cut.

    Taken pre-cut so a source that changes mid-cut invalidates the result
    instead of being described only by its post-cut state.
    """
    try:
        st = os.stat(path)
        return {
            "path": os.path.abspath(path),
            "size": st.st_size,
            "mtime": st.st_mtime,
        }
    except OSError:
        return {"path": os.path.abspath(path), "size": None, "mtime": None}


def slice_provenance_path(slice_path: str) -> str:
    """Sidecar path for a source slice's provenance record."""
    return slice_path + ".slice.json"


def _output_identity(path: str) -> dict | None:
    """Identity of the produced slice file itself (binds stamp to artifact)."""
    try:
        st = os.stat(path)
        return {"size": st.st_size, "mtime": st.st_mtime}
    except OSError:
        return None


def slice_provenance(
    *, start: float, end: float, source_fp: dict, output_path: str | None = None
) -> dict:
    """Fingerprint binding a source slice to its requested interval.

    ``source_fp`` must be captured via ``source_fingerprint`` BEFORE the cut.
    When ``output_path`` is given, the record also binds to the produced
    artifact's identity, so a torn media/provenance pair (new media, old
    stamp after an interrupted swap) fails validation instead of falsely
    validating.
    """
    return {
        "version": _SLICE_PROVENANCE_VERSION,
        "start": round(float(start), 3),
        "end": round(float(end), 3),
        "source": source_fp,
        "output": _output_identity(output_path) if output_path else None,
    }


def write_slice_provenance(
    slice_path: str, *, start: float, end: float, source_fp: dict
) -> None:
    """Persist the provenance sidecar atomically, bound to the slice file.

    The record is written to a temp file and atomically renamed, and it
    captures the slice file's own identity -- so the stamp always describes
    the exact bytes on disk, never a previous generation. Call AFTER the
    media swap (``os.replace``) has completed.
    """
    import json

    record = slice_provenance(
        start=start, end=end, source_fp=source_fp, output_path=slice_path
    )
    tmp_path = slice_provenance_path(slice_path) + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(record, f)
    os.replace(tmp_path, slice_provenance_path(slice_path))


def read_slice_provenance(slice_path: str) -> dict | None:
    """Read a slice's provenance record, or None if missing/unparseable."""
    import json

    try:
        with open(slice_provenance_path(slice_path), encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def slice_provenance_valid(
    slice_path: str, *, start: float, end: float, source_path: str
) -> bool:
    """True when the on-disk slice was cut for this exact request.

    Guards resume: an old slice (pre-accurate-cut ``version``), a changed
    interval, a changed source file, a missing/unparseable record, or a
    torn media/provenance pair (stamp's output identity != current file)
    all invalidate reuse. A slice with no provenance record is treated as
    legacy and re-cut.
    """
    record = read_slice_provenance(slice_path)
    if record is None:
        return False
    source_fp = source_fingerprint(source_path)
    expected = slice_provenance(
        start=start, end=end, source_fp=source_fp, output_path=slice_path
    )
    for key in ("version", "start", "end", "source"):
        if record.get(key) != expected.get(key):
            return False
    stamped_output = record.get("output")
    current_output = expected.get("output")
    if not stamped_output or not current_output:
        return False
    return stamped_output == current_output


def validate_cut_output(
    path: str, expected_duration: float, *, min_size: int = 10_000
) -> tuple[bool, str]:
    """Verify a freshly cut temp file before it replaces a known-good slice.

    Returns (True, "") when the file is usable, else (False, reason).
    Catches the "ffmpeg exit 0 but empty/truncated output" case: a
    successful return code alone must never authorize replacing good media.
    Never raises.
    """
    try:
        size = os.path.getsize(path)
    except OSError as exc:
        return False, f"unreadable: {exc}"
    if size < min_size:
        return False, f"too small ({size} bytes < {min_size})"
    try:
        from clippyme.pipeline.media_qa import probe_media
    except Exception as exc:
        return False, f"probe unavailable: {exc}"
    report = probe_media(path)
    if report.get("probe_error"):
        return False, f"probe failed: {report['probe_error']}"
    if not report.get("has_video"):
        return False, "no usable video stream"
    dur = report.get("duration")
    if dur is None:
        return False, "unknown duration"
    # Tight validation: duration must match the requested interval.
    # The normalized shared-clock cut selects exactly [S, E); incorrect
    # content selection must fail, not be accommodated.
    if abs(dur - expected_duration) > 0.2:
        return False, f"duration {dur:.3f}s not within tolerance of {expected_duration:.3f}s"
    return True, ""


_CLIP_SUFFIX_RE = re.compile(r"_clip_\d+\.mp4$", re.IGNORECASE)
_YT_VIDEO_ID_RE = re.compile(r"(?:v=|\/embed\/|\/live\/|\/shorts\/|youtu\.be\/|\/v\/)([A-Za-z0-9_-]{11})")


def is_clip_artifact(filename: str) -> bool:
    """Return True if filename is an output or intermediate clip artifact, not a source video."""
    base = os.path.basename(filename)
    low = base.lower()
    return bool(
        low.startswith("source_")
        or low.startswith("clip_")
        or low.startswith("composed_")
        or low.startswith("highlight_reel_")
        or low.endswith(".tmp.mp4")
        or low.endswith(".render.tmp.mp4")
        or low.endswith(".tmp")
        or bool(_CLIP_SUFFIX_RE.search(low))
    )


def find_source_video_candidate(output_dir: str, min_size: int = 10_000) -> str | None:
    """Find the most likely original or trimmed source video in output_dir, ignoring clips.

    If multiple candidates exist, files starting with 'trimmed_' (head-trimmed source)
    take priority, followed by the largest file by byte size.
    """
    if not os.path.isdir(output_dir):
        return None
    candidates: list[tuple[str, int, str]] = []
    try:
        entries = os.listdir(output_dir)
    except OSError:
        return None

    for item in entries:
        if not item.lower().endswith(".mp4"):
            continue
        if is_clip_artifact(item):
            continue
        full_path = os.path.join(output_dir, item)
        try:
            sz = os.path.getsize(full_path)
            if sz >= min_size:
                candidates.append((item, sz, full_path))
        except OSError:
            continue

    if not candidates:
        return None

    trimmed = [c for c in candidates if c[0].lower().startswith("trimmed_")]
    if trimmed:
        return max(trimmed, key=lambda c: c[1])[2]
    return max(candidates, key=lambda c: c[1])[2]


def extract_youtube_video_id(url: str | None) -> str | None:
    """Extract standard 11-char YouTube video id from various URL shapes, or None."""
    if not url or not isinstance(url, str):
        return None
    match = _YT_VIDEO_ID_RE.search(url)
    return match.group(1) if match else None


def is_same_video_source(url_a: str | None, url_b: str | None) -> bool:
    """Compare two source URLs, treating matching YouTube video IDs as identical."""
    if not url_a or not url_b:
        return False
    if url_a == url_b:
        return True
    id_a = extract_youtube_video_id(url_a)
    id_b = extract_youtube_video_id(url_b)
    if id_a and id_b:
        return id_a == id_b
    return False

