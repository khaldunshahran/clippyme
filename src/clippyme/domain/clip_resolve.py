"""Shared job/clip resolution for the per-clip endpoints.

Every per-clip endpoint (smartcut, transcript, edit-ai, compose, publish) needs
the same chain: job dir → latest ``*_metadata.json`` → clip entry by index →
clip filename (from ``video_url``, falling back to the ``_clip_{i+1}.mp4``
naming convention) → absolute clip path. This module is the single owner of
that chain; the handlers call :func:`resolve_clip` and stay thin.

Pure sync filesystem code (handlers wrap it in ``asyncio.to_thread``); raises
``NotFoundError`` which the app-level ``ClippyMeError`` handler maps to 404.
"""
import json
import os
from dataclasses import dataclass

from clippyme.domain.errors import NotFoundError
from clippyme.domain.job_artifacts import find_job_metadata_path
from clippyme.domain.url_utils import filename_from_video_url


@dataclass(frozen=True)
class ResolvedClip:
    metadata_path: str
    metadata: dict
    clip_info: dict
    clip_filename: str
    clip_path: str

    @property
    def job_dir(self) -> str:
        return os.path.dirname(self.metadata_path)


def clip_filename_for(metadata_path: str, clip_info: dict, clip_index: int) -> str:
    """Clip filename, preferring (a) the ``clip_filename`` basename the
    pipeline persists into metadata (pipeline/main.py, task 4b) when it is a
    non-empty string with no path separators or ``..`` (defence against a
    tampered/legacy metadata file smuggling a path), then (b) the legacy
    metadata ``video_url``, then (c) the positional ``<base>_clip_{i+1}.mp4``
    convention when neither is present."""
    raw = clip_info.get("clip_filename")
    if isinstance(raw, str) and raw and "/" not in raw and "\\" not in raw and ".." not in raw:
        return raw
    filename = filename_from_video_url(clip_info.get("video_url"))
    if not filename:
        base_name = os.path.basename(metadata_path).replace("_metadata.json", "")
        filename = f"{base_name}_clip_{clip_index + 1}.mp4"
    return filename


def composed_clip_basename(clip_info: dict, clip_index: int) -> str:
    """Filename for a clip's final composed (hook/subtitles/banner/…) output.

    Title-based and Windows-safe, prefixed with ``composed_`` and disambiguated
    with a positional ``_clip_{N}`` suffix so it NEVER collides with the base
    clip filename or wipes the base clip before/during composition.
    """
    from clippyme.pipeline.run_ops import sanitize_windows_basename

    title = (clip_info or {}).get("video_title_for_youtube_short") or (clip_info or {}).get("title")
    base = sanitize_windows_basename(title)
    suffix = f"_clip_{clip_index + 1}"
    if base:
        if not base.endswith(suffix):
            max_len = 80 - len(suffix)
            base = f"{base[:max_len]}{suffix}"
        return f"composed_{base}.mp4"
    return f"composed_clip_{clip_index + 1}.mp4"


def _is_composed_basename(basename: str) -> bool:
    """True for filenames produced by the compose pipeline itself.

    Composing ONTO one of these burns a second caption layer over the
    already burned-in one — the doubled-caption defect — so compose callers
    must never use them as the base clip.
    """
    name = basename.lower()
    return name.startswith("composed_") and name.endswith(".mp4")


def resolve_clip(job_id: str, clip_index: int, output_root: str,
                 *, require_file: bool = True, for_compose: bool = False) -> ResolvedClip:
    """Resolve a job's clip to its metadata + on-disk path.

    Raises ``NotFoundError`` (→ 404) when the job dir, metadata, clip index or
    — with ``require_file=True`` — the rendered mp4 is missing. Callers that
    can proceed without the base file (transcript/edit-ai read only metadata;
    publish may fall back to a composed file) pass ``require_file=False``.

    ``for_compose=True`` marks a composition input: composed files
    (``composed_*.mp4``) are then EXCLUDED from the on-disk fallback chain
    and a composed file is never returned, because composing onto one burns
    a second caption layer over the first (the doubled-caption defect).
    When only a composed file exists, ``NotFoundError`` is raised instead of
    silently returning it.
    """
    job_dir = os.path.join(output_root, job_id)
    if not os.path.isdir(job_dir):
        raise NotFoundError("Job not found")

    try:
        metadata_path = find_job_metadata_path(job_id, output_root)
    except FileNotFoundError:
        raise NotFoundError("No metadata found")

    with open(metadata_path, encoding="utf-8") as f:
        metadata = json.load(f)

    clips = metadata.get("shorts", [])
    if clip_index < 0 or clip_index >= len(clips):
        raise NotFoundError("Clip not found")
    clip_info = clips[clip_index]

    clip_filename = clip_filename_for(metadata_path, clip_info, clip_index)
    clip_path = os.path.join(job_dir, clip_filename)

    # If the exact clip_filename doesn't exist on disk, look for on-disk fallback variants
    if not os.path.exists(clip_path):
        candidates = [
            os.path.join(job_dir, composed_clip_basename(clip_info, clip_index)),
            os.path.join(job_dir, f"composed_{clip_filename}"),
            os.path.join(job_dir, f"source_{clip_filename}"),
            os.path.join(job_dir, f"reframe_{clip_filename}"),
        ]
        for cand in candidates:
            if for_compose and _is_composed_basename(os.path.basename(cand)):
                # Compose input must be the clean base clip: a composed file
                # already carries burned-in layers (double-burn defect).
                continue
            if os.path.isfile(cand):
                clip_path = cand
                break
        else:
            if os.path.isdir(job_dir):
                suffix = f"_clip_{clip_index + 1}.mp4"
                for entry in sorted(os.listdir(job_dir)):
                    if for_compose and _is_composed_basename(entry):
                        continue
                    if entry.endswith(suffix):
                        cand = os.path.join(job_dir, entry)
                        if os.path.isfile(cand):
                            clip_path = cand
                            break

    if require_file and not os.path.exists(clip_path):
        raise NotFoundError(f"Clip file not found: {clip_filename}")

    if for_compose and _is_composed_basename(os.path.basename(clip_path)):
        # Covers the primary path too: metadata's clip_filename itself may
        # point at an already-composed file (dirty metadata). Refuse loudly
        # instead of burning a second caption layer over the first.
        raise NotFoundError(
            f"Clean base clip missing for compose (only already-composed "
            f"file {os.path.basename(clip_path)!r} found); refusing to "
            "compose onto it - that would double-burn the captions")

    return ResolvedClip(
        metadata_path=metadata_path,
        metadata=metadata,
        clip_info=clip_info,
        clip_filename=clip_filename,
        clip_path=clip_path,
    )
