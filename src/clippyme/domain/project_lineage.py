"""Reframed-base lineage for automatic framing preservation.

When the auto-pipeline emits a clip project, the original ``clip_final``
already contains the AI's tracked-speaker/split-layout framing. This module
provides pure helpers to:

- Build lineage at emission: reference to the reframed base + geometry/
  timeline baseline + file identity for staleness detection.
- Validate eligibility at render: semantic equivalence of source, timeline,
  and render geometry to the stored baseline.

Eligibility is semantic, not incidental:
- Source: file identity (name + size + mtime), dimensions.
- Timeline: ordered segment (id, start, end, crop x/y/w/h, transition).
- Render: width, height, fps.
- Metadata-only changes (labels, IDs, version, origin) do NOT invalidate.
- Any crop, trim, reorder, transition, or render-geometry change DOES.

References resolve safely inside the job directory only. Malformed
references never read unrelated files.
"""

from __future__ import annotations

import hashlib
import os
from typing import Any


def _file_identity(path: str) -> dict[str, Any] | None:
    """File identity for staleness detection: name, size, mtime.

    Returns None if the file doesn't exist or can't be statted.
    """
    try:
        st = os.stat(path)
        return {
            "file": os.path.basename(path),
            "size": st.st_size,
            "mtime": st.st_mtime,
        }
    except (OSError, ValueError):
        return None


def build_lineage(
    *,
    clip_final: str,
    job_dir: str,
    source: dict[str, Any],
    segments: list[dict[str, Any]],
    render: dict[str, Any],
) -> dict[str, Any] | None:
    """Build reframed-base lineage for a newly emitted auto project.

    Args:
        clip_final: path to the original reframed clip (before overlay
            composition). May be absolute or relative to job_dir.
        job_dir: job directory for safe reference resolution.
        source: source dict with file/width/height/duration.
        segments: segment list with id/start/end/crop/transition.
        render: render spec with width/height/fps/encoder.

    Returns lineage dict, or None if the base file is missing.
    The lineage stores a job-relative reference + identity + baselines.
    """
    # Resolve to job-relative reference safely.
    try:
        abs_final = os.path.abspath(clip_final)
        abs_job = os.path.abspath(job_dir)
        # Must be inside job_dir.
        if not abs_final.startswith(abs_job + os.sep):
            return None
        rel_ref = os.path.relpath(abs_final, abs_job)
        # Reject path traversal.
        if rel_ref.startswith("..") or os.path.isabs(rel_ref):
            return None
    except (OSError, ValueError):
        return None

    identity = _file_identity(abs_final)
    if identity is None:
        return None

    # Geometry/timeline baseline: the fields that affect base rendering.
    # Source identity.
    source_baseline = {
        "file": source.get("file"),
        "width": source.get("width"),
        "height": source.get("height"),
    }
    # Ordered segment geometry.
    segment_baseline = []
    for seg in segments:
        crop = seg.get("crop") or {}
        segment_baseline.append({
            "id": seg.get("id"),
            "start": seg.get("start"),
            "end": seg.get("end"),
            "crop": {
                "x": crop.get("x"),
                "y": crop.get("y"),
                "w": crop.get("w"),
                "h": crop.get("h"),
            },
            "transition_in": seg.get("transition_in", "cut"),
        })
    # Render geometry.
    render_baseline = {
        "width": render.get("width"),
        "height": render.get("height"),
        "fps": render.get("fps"),
    }

    return {
        "base_ref": rel_ref,
        "identity": identity,
        "source": source_baseline,
        "segments": segment_baseline,
        "render": render_baseline,
    }


def _safe_resolve(ref: str, job_dir: str) -> str | None:
    """Resolve a lineage reference safely inside job_dir.

    Returns absolute path, or None if malformed/escaping.
    """
    if not isinstance(ref, str) or not ref:
        return None
    # Reject absolute paths and traversal.
    if os.path.isabs(ref) or ".." in ref.split(os.sep):
        return None
    try:
        abs_job = os.path.abspath(job_dir)
        abs_path = os.path.abspath(os.path.join(abs_job, ref))
        if not abs_path.startswith(abs_job + os.sep):
            return None
        return abs_path
    except (OSError, ValueError):
        return None


def check_lineage_eligible(
    lineage: dict[str, Any],
    project: dict[str, Any],
    job_dir: str,
) -> tuple[bool, str | None, str | None]:
    """Check if a lineage-bearing project is eligible for base reuse.

    Args:
        lineage: the stored lineage dict.
        project: current project dict (may have user edits).
        job_dir: job directory for reference resolution.

    Returns (eligible, base_path, reason):
        - eligible True: base_path is the validated reframed clip.
        - eligible False: reason explains why (for diagnostics).
          reason is None only when eligible.

    Eligibility requires semantic equivalence:
    - Base file exists and identity matches (not missing/stale/replaced).
    - Source file, dimensions match.
    - Ordered segments match (id, start, end, crop, transition).
    - Render width/height/fps match.
    - Metadata-only changes (version, origin, labels, caption edits,
      overlays, grade, audio) do NOT affect eligibility.
    """
    if not isinstance(lineage, dict):
        return (False, None, "lineage malformed")

    # 1. Resolve and validate base file.
    base_ref = lineage.get("base_ref")
    base_path = _safe_resolve(base_ref, job_dir)
    if base_path is None:
        return (False, None, f"invalid base reference: {base_ref!r}")
    if not os.path.isfile(base_path):
        return (False, None, f"base file missing: {base_ref!r}")

    # 2. Check identity (stale/replaced detection).
    stored_identity = lineage.get("identity") or {}
    current_identity = _file_identity(base_path)
    if current_identity is None:
        return (False, None, "cannot stat base file")
    for key in ("size", "mtime"):
        if stored_identity.get(key) != current_identity.get(key):
            return (False, None,
                    f"base file {key} changed (stale/replaced)")

    # 3. Source equivalence.
    stored_source = lineage.get("source") or {}
    proj_source = project.get("source") or {}
    for key in ("file", "width", "height"):
        if stored_source.get(key) != proj_source.get(key):
            return (False, None, f"source {key} changed")

    # 4. Segment equivalence (ordered, geometry-sensitive).
    stored_segs = lineage.get("segments") or []
    proj_segs = project.get("segments") or []
    if len(stored_segs) != len(proj_segs):
        return (False, None,
                f"segment count changed ({len(stored_segs)} -> {len(proj_segs)})")
    for i, (s_stored, s_proj) in enumerate(zip(stored_segs, proj_segs)):
        for key in ("id", "start", "end", "transition_in"):
            if s_stored.get(key) != s_proj.get(key):
                return (False, None,
                        f"segment {i} {key} changed")
        c_stored = s_stored.get("crop") or {}
        c_proj = (s_proj.get("crop") or {}) if isinstance(s_proj.get("crop"), dict) else {}
        # Handle pydantic model vs dict.
        if hasattr(c_proj, "model_dump"):
            c_proj = c_proj.model_dump()
        for key in ("x", "y", "w", "h"):
            if c_stored.get(key) != c_proj.get(key):
                return (False, None,
                        f"segment {i} crop {key} changed")

    # 5. Render geometry equivalence.
    stored_render = lineage.get("render") or {}
    proj_render = project.get("render") or {}
    if hasattr(proj_render, "model_dump"):
        proj_render = proj_render.model_dump()
    for key in ("width", "height", "fps"):
        if stored_render.get(key) != proj_render.get(key):
            return (False, None, f"render {key} changed")

    return (True, base_path, None)
