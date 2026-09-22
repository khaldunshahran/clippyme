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


def build_cut_command(input_video: str, start: float, end: float, dest: str) -> list[str]:
    """ffmpeg argv for cutting the 16:9 source slice of one clip.

    ``-ss`` BEFORE ``-i`` uses fast input seek (jump to the keyframe before
    ``start``, decode forward to the exact time). ``-pix_fmt yuv420p`` +
    ``-vsync cfr`` guarantee the persisted slice is universally decodable and
    constant-frame-rate, so the downstream reframe render (raw frames at a
    fixed ``-r``) can't drift against audio even if the original download was
    VFR. Shared x264 settings (CRF 18 / medium): this slice feeds every later
    generation, so it must not be the weak link.
    """
    clip_duration = float(end) - float(start)
    return [
        'ffmpeg', '-y',
        '-ss', f'{float(start):.3f}',
        '-i', input_video,
        '-t', f'{clip_duration:.3f}',
        *x264_video_args(faststart=False),
        '-vsync', 'cfr',
        '-c:a', 'aac',
        dest,
    ]


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

