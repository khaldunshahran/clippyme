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
_SLICE_PROVENANCE_VERSION = 2

# --- Timing convention (single source of truth) ------------------------------
# The slice's time 0 corresponds to the first decoded source frame whose
# presentation timestamp is >= the requested start. ``-ss`` is placed AFTER
# ``-i`` (output seeking): ffmpeg decodes from the nearest seek point and
# discards everything before ``start``, so the retained head offset is in
# [0, 1/fps) -- sub-frame, never a GOP. A requested edge that falls between
# two source frames resolves to the NEXT frame (ceil to frame boundary).
# All downstream consumers (word timings, compose, Smart Cut, project emit)
# use the REQUESTED start as time 0; the residual (< 1 frame, < ~33ms at
# 30fps) is below the caption display threshold and inside the QA duration
# tolerance, so no consumer re-rounds independently.


def build_cut_command(input_video: str, start: float, end: float, dest: str) -> list[str]:
    """ffmpeg argv for cutting the 16:9 source slice of one clip.

    Accurate cut: ``-ss`` AFTER ``-i`` seeks accurately by decoding and
    discarding pre-start frames, then video AND audio are re-encoded with
    the shared encode policy (``domain.encode.x264_video_args``). The
    persisted slice starts at the first frame at/after ``start`` -- no
    keyframe-granularity head preroll. ``-avoid_negative_ts make_zero``
    re-bases timestamps so the slice starts at zero. Slower than the old
    stream-copy cut; correct timing has priority over cut speed.

    Audio: re-encoded to AAC so the audio track is trimmed to the same
    accurate boundaries as the video. ``-map 0:a?`` makes the audio
    stream optional -- sources without audio cut cleanly instead of
    failing.
    """
    start_f, end_f = float(start), float(end)
    clip_duration = end_f - start_f
    if not (clip_duration > 0):
        raise ValueError(f"invalid cut interval start={start_f} end={end_f}")
    if start_f < 0:
        raise ValueError(f"cut start must be >= 0, got {start_f}")
    return [
        'ffmpeg', '-y',
        '-i', input_video,
        '-ss', f'{start_f:.3f}',
        '-t', f'{clip_duration:.3f}',
        '-map', '0:v:0',
        '-map', '0:a?',
        *x264_video_args(faststart=False),
        '-c:a', 'aac',
        '-avoid_negative_ts', 'make_zero',
        dest,
    ]


def slice_provenance_path(slice_path: str) -> str:
    """Sidecar path for a source slice's provenance record."""
    return slice_path + ".slice.json"


def slice_provenance(*, start: float, end: float, source_path: str) -> dict:
    """Fingerprint binding a source slice to its requested interval + source.

    Resume decisions compare this against the current request; any change
    to the interval or the source file invalidates reuse.
    """
    try:
        st = os.stat(source_path)
        src_id: dict = {
            "path": os.path.abspath(source_path),
            "size": st.st_size,
            "mtime": st.st_mtime,
        }
    except OSError:
        src_id = {"path": os.path.abspath(source_path), "size": None, "mtime": None}
    return {
        "version": _SLICE_PROVENANCE_VERSION,
        "start": round(float(start), 3),
        "end": round(float(end), 3),
        "source": src_id,
    }


def write_slice_provenance(slice_path: str, *, start: float, end: float, source_path: str) -> None:
    """Persist the provenance sidecar next to a freshly cut slice."""
    import json

    with open(slice_provenance_path(slice_path), "w", encoding="utf-8") as f:
        json.dump(slice_provenance(start=start, end=end, source_path=source_path), f)


def read_slice_provenance(slice_path: str) -> dict | None:
    """Read a slice's provenance record, or None if missing/unparseable."""
    import json

    try:
        with open(slice_provenance_path(slice_path), encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def slice_provenance_valid(slice_path: str, *, start: float, end: float, source_path: str) -> bool:
    """True when the on-disk slice was cut for this exact request.

    Guards resume: an old slice (pre-accurate-cut ``version``), a changed
    interval, or a changed source file all invalidate reuse. A slice with
    no provenance record is treated as legacy and re-cut.
    """
    record = read_slice_provenance(slice_path)
    if record is None:
        return False
    expected = slice_provenance(start=start, end=end, source_path=source_path)
    return record == expected


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

