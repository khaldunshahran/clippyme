"""Reframed-base lineage for automatic framing preservation.

When the auto-pipeline emits a clip project, the original ``clip_final``
already contains the AI's tracked-speaker/split-layout framing. This module
provides pure helpers to:

- Build lineage at emission: reference to the reframed base + geometry/
  timeline baseline + file identity for staleness detection.
- Validate eligibility at render: semantic equivalence of source, timeline,
  and render geometry to the stored baseline.

Eligibility outcomes:
- ("eligible", base_path): reuse the reframed base.
- ("geometry_edit", None): genuine edit; use existing source renderer.
- ("invalid_asset", None): missing/stale/malformed base or source;
  FAIL CLEARLY, do not silently publish a center crop.

Segment IDs are metadata (not geometry). Ordered timing/crop/transition
comparisons retain reordering sensitivity.
"""

from __future__ import annotations

import os
from typing import Any


def _file_identity(path: str) -> dict[str, Any] | None:
    """File identity for staleness detection: name, size, mtime.

    Returns None if the file doesn't exist or can't be statted.
    This is a fingerprint, not a content hash; document limitations.
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
    source_path: str | None = None,
) -> dict[str, Any] | None:
    """Build reframed-base lineage for a newly emitted auto project.

    Args:
        clip_final: path to the original reframed clip.
        job_dir: job directory for safe reference resolution.
        source: source dict with file/width/height/duration.
        segments: segment list with id/start/end/crop/transition.
        render: render spec with width/height/fps/encoder.
        source_path: absolute path to source file for identity
            (optional; falls back to job_dir + source file).

    Returns lineage dict, or None if the base file is missing.
    """
    try:
        abs_final = os.path.abspath(clip_final)
        abs_job = os.path.abspath(job_dir)
        if not abs_final.startswith(abs_job + os.sep):
            return None
        rel_ref = os.path.relpath(abs_final, abs_job)
        if rel_ref.startswith("..") or os.path.isabs(rel_ref):
            return None
    except (OSError, ValueError):
        return None

    identity = _file_identity(abs_final)
    if identity is None:
        return None

    # Source identity: resolve actual source file for fingerprint.
    source_identity = None
    if source_path:
        source_identity = _file_identity(source_path)
    else:
        src_file = source.get("file")
        if src_file:
            candidate = os.path.join(abs_job, os.path.basename(src_file))
            source_identity = _file_identity(candidate)

    source_baseline = {
        "file": source.get("file"),
        "width": source.get("width"),
        "height": source.get("height"),
        "identity": source_identity,
    }
    segment_baseline = []
    for seg in segments:
        crop = seg.get("crop") or {}
        if hasattr(crop, "model_dump"):
            crop = crop.model_dump()
        if not isinstance(crop, dict):
            crop = {}
        segment_baseline.append({
            # ID is metadata, not geometry; stored for reference only.
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
    """Resolve a lineage reference safely inside job_dir."""
    if not isinstance(ref, str) or not ref:
        return None
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


def _shape_ok(value: Any, spec: dict[str, type]) -> bool:
    """Defensive shape check: value is dict with expected key types."""
    if not isinstance(value, dict):
        return False
    for key, typ in spec.items():
        if key in value and not isinstance(value[key], typ):
            # Allow int for float fields.
            if typ is float and isinstance(value[key], int):
                continue
            return False
    return True


def check_lineage_eligible(
    lineage: dict[str, Any],
    project: dict[str, Any],
    job_dir: str,
) -> tuple[str, str | None, str | None]:
    """Check lineage eligibility with asset vs geometry distinction.

    Returns (outcome, base_path, reason):
        - ("eligible", path, None): reuse the reframed base.
        - ("geometry_edit", None, reason): genuine edit; use source renderer.
        - ("invalid_asset", None, reason): missing/stale/malformed; FAIL CLEARLY.

    Defensive: malformed nested structures return invalid_asset, never crash.
    """
    if not isinstance(lineage, dict):
        return ("invalid_asset", None, "lineage malformed")
    if not isinstance(project, dict):
        return ("invalid_asset", None, "project malformed")

    # 1. Resolve base reference safely.
    base_ref = lineage.get("base_ref")
    base_path = _safe_resolve(base_ref, job_dir)
    if base_path is None:
        return ("invalid_asset", None,
                f"invalid base reference: {base_ref!r}")
    if not os.path.isfile(base_path):
        return ("invalid_asset", None,
                f"base file missing: {base_ref!r}")

    # 2. Base identity (stale/replaced).
    stored_identity = lineage.get("identity")
    if not _shape_ok(stored_identity, {"size": int, "mtime": float}):
        return ("invalid_asset", None, "lineage identity malformed")
    current_identity = _file_identity(base_path)
    if current_identity is None:
        return ("invalid_asset", None, "cannot stat base file")
    for key in ("size", "mtime"):
        if stored_identity.get(key) != current_identity.get(key):
            return ("invalid_asset", None,
                    f"base file {key} changed (stale/replaced)")

    # 3. Source identity (same-name replacement detection).
    stored_source = lineage.get("source")
    if not isinstance(stored_source, dict):
        return ("invalid_asset", None, "lineage source malformed")
    proj_source = project.get("source")
    if hasattr(proj_source, "model_dump"):
        proj_source = proj_source.model_dump()
    if not isinstance(proj_source, dict):
        return ("invalid_asset", None, "project source malformed")
    # File name and dimensions.
    for key in ("file", "width", "height"):
        if stored_source.get(key) != proj_source.get(key):
            return ("geometry_edit", None, f"source {key} changed")
    # Source file identity (fingerprint).
    stored_src_id = stored_source.get("identity")
    if isinstance(stored_src_id, dict) and stored_src_id.get("size") is not None:
        src_file = proj_source.get("file")
        if src_file:
            candidate = os.path.join(os.path.abspath(job_dir),
                                     os.path.basename(src_file))
            current_src_id = _file_identity(candidate)
            if current_src_id is None:
                return ("invalid_asset", None, "source file missing")
            for key in ("size", "mtime"):
                if stored_src_id.get(key) != current_src_id.get(key):
                    return ("invalid_asset", None,
                            f"source file {key} changed (replaced)")

    # 4. Segment equivalence (ordered; IDs are metadata).
    stored_segs = lineage.get("segments")
    proj_segs = project.get("segments")
    if hasattr(proj_segs, "__iter__") and not isinstance(proj_segs, (list, tuple)):
        proj_segs = list(proj_segs)
    if not isinstance(stored_segs, list) or not isinstance(proj_segs, list):
        return ("invalid_asset", None, "segments malformed")
    if len(stored_segs) != len(proj_segs):
        return ("geometry_edit", None,
                f"segment count changed ({len(stored_segs)} -> {len(proj_segs)})")
    for i, (s_stored, s_proj) in enumerate(zip(stored_segs, proj_segs)):
        if hasattr(s_proj, "model_dump"):
            s_proj = s_proj.model_dump()
        if not isinstance(s_stored, dict) or not isinstance(s_proj, dict):
            return ("invalid_asset", None, f"segment {i} malformed")
        # Geometry: start, end, transition. ID is metadata (ignored).
        for key in ("start", "end", "transition_in"):
            if s_stored.get(key) != s_proj.get(key):
                return ("geometry_edit", None,
                        f"segment {i} {key} changed")
        c_stored = s_stored.get("crop")
        c_proj = s_proj.get("crop")
        if hasattr(c_proj, "model_dump"):
            c_proj = c_proj.model_dump()
        if not isinstance(c_stored, dict) or not isinstance(c_proj, dict):
            return ("invalid_asset", None,
                    f"segment {i} crop malformed")
        for key in ("x", "y", "w", "h"):
            if c_stored.get(key) != c_proj.get(key):
                return ("geometry_edit", None,
                        f"segment {i} crop {key} changed")

    # 5. Render geometry.
    stored_render = lineage.get("render")
    proj_render = project.get("render")
    if hasattr(proj_render, "model_dump"):
        proj_render = proj_render.model_dump()
    if not isinstance(stored_render, dict) or not isinstance(proj_render, dict):
        return ("invalid_asset", None, "render spec malformed")
    for key in ("width", "height", "fps"):
        if stored_render.get(key) != proj_render.get(key):
            return ("geometry_edit", None, f"render {key} changed")

    return ("eligible", base_path, None)
