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
- ("framing_edit", None): the automatic framing trajectory was added,
  removed, or changed after the base was rendered; dynamic replay is not
  implemented -- FAIL CLEARLY with the stable
  "framing-trajectory-edit" reason, do not silently publish a center crop.
- ("invalid_asset", None): missing/stale/malformed base or source;
  FAIL CLEARLY, do not silently publish a center crop.

Order: geometry equivalence is established BEFORE base asset checks.
A genuine edit returns geometry_edit even if the old base is missing/stale.
Incomplete baselines are invalid_asset, not geometry_edit.

Segment IDs are metadata (not geometry). Ordered timing/crop/transition
comparisons retain reordering sensitivity.
"""

from __future__ import annotations

import math
import os
from typing import Any


def _file_identity(path: str) -> dict[str, Any] | None:
    """File identity for staleness detection: name, size, mtime.

    Returns None if the file doesn't exist or can't be statted.
    This is a fingerprint, not a content hash.
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
    framing: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Build reframed-base lineage for a newly emitted auto project.

    Returns None if the base file is missing or source identity cannot
    be captured. Source fingerprints are not optional when the lineage
    relationship is trusted.
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

    # Source identity is REQUIRED. Decline if it cannot be captured.
    source_identity = None
    if source_path:
        source_identity = _file_identity(source_path)
    else:
        src_file = source.get("file") if isinstance(source, dict) else None
        if src_file:
            candidate = os.path.join(abs_job, os.path.basename(src_file))
            source_identity = _file_identity(candidate)
    if source_identity is None:
        return None

    if not isinstance(source, dict):
        return None
    source_baseline = {
        "file": source.get("file"),
        "width": source.get("width"),
        "height": source.get("height"),
        "identity": source_identity,
    }
    # Require complete source baseline.
    if not source_baseline["file"]:
        return None

    segment_baseline = []
    for seg in segments:
        if not isinstance(seg, dict):
            return None
        crop = seg.get("crop")
        if hasattr(crop, "model_dump"):
            crop = crop.model_dump()
        if not isinstance(crop, dict):
            return None
        # Require complete crop baseline.
        for k in ("x", "y", "w", "h"):
            if crop.get(k) is None:
                return None
            if not isinstance(crop[k], (int, float)) or not math.isfinite(crop[k]):
                return None
        for k in ("start", "end"):
            if seg.get(k) is None:
                return None
            if not isinstance(seg[k], (int, float)) or not math.isfinite(seg[k]):
                return None
        segment_baseline.append({
            "id": seg.get("id"),  # metadata, not geometry
            "start": seg.get("start"),
            "end": seg.get("end"),
            "crop": {k: crop.get(k) for k in ("x", "y", "w", "h")},
            "transition_in": seg.get("transition_in", "cut"),
        })
    if not isinstance(render, dict):
        return None
    render_baseline = {
        "width": render.get("width"),
        "height": render.get("height"),
        "fps": render.get("fps"),
    }
    for k in ("width", "height", "fps"):
        if render_baseline[k] is None:
            return None

    # Task 13: framing baseline identity. The caller passes the already
    # validated project framing dict; only its immutable identity is stored
    # (the content hash catches same-size/same-mtime replacement). None for
    # old projects or when the domain helper is unavailable.
    framing_identity: dict[str, Any] | None = None
    if framing:
        try:
            from clippyme.domain.project_framing import framing_baseline_identity
        except ImportError:
            framing_baseline_identity = None  # type: ignore[assignment]
        if framing_baseline_identity is not None:
            try:
                framing_identity = framing_baseline_identity(framing)
            except Exception:
                framing_identity = None

    return {
        "base_ref": rel_ref,
        "identity": identity,
        "source": source_baseline,
        "segments": segment_baseline,
        "render": render_baseline,
        "framing": framing_identity,
    }


def _safe_resolve(ref: str, job_dir: str) -> str | None:
    """Resolve a lineage reference safely inside job_dir."""
    if not isinstance(ref, str) or not ref:
        return None
    if os.path.isabs(ref):
        return None
    # Check both separators for traversal.
    parts = ref.replace("\\", "/").split("/")
    if ".." in parts:
        return None
    try:
        abs_job = os.path.abspath(job_dir)
        abs_path = os.path.abspath(os.path.join(abs_job, ref))
        if not abs_path.startswith(abs_job + os.sep):
            return None
        return abs_path
    except (OSError, ValueError):
        return None


def _is_finite_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and math.isfinite(v)


def _validate_baseline(lineage: dict[str, Any]) -> str | None:
    """Validate baseline structure. Returns error reason or None if valid."""
    # Source baseline.
    src = lineage.get("source")
    if not isinstance(src, dict):
        return "lineage source malformed"
    if not src.get("file"):
        return "lineage source file missing"
    for k in ("width", "height"):
        if not _is_finite_number(src.get(k)):
            return f"lineage source {k} invalid"
    ident = src.get("identity")
    if not isinstance(ident, dict):
        return "lineage source identity missing/malformed"
    for k in ("size", "mtime"):
        if not _is_finite_number(ident.get(k)):
            return f"lineage source identity {k} invalid"

    # Segment baselines.
    segs = lineage.get("segments")
    if not isinstance(segs, list):
        return "lineage segments malformed"
    for i, s in enumerate(segs):
        if not isinstance(s, dict):
            return f"lineage segment {i} malformed"
        for k in ("start", "end"):
            if not _is_finite_number(s.get(k)):
                return f"lineage segment {i} {k} invalid"
        crop = s.get("crop")
        if not isinstance(crop, dict):
            return f"lineage segment {i} crop malformed"
        for k in ("x", "y", "w", "h"):
            if not _is_finite_number(crop.get(k)):
                return f"lineage segment {i} crop {k} invalid"

    # Render baseline.
    rnd = lineage.get("render")
    if not isinstance(rnd, dict):
        return "lineage render malformed"
    for k in ("width", "height", "fps"):
        if not _is_finite_number(rnd.get(k)):
            return f"lineage render {k} invalid"

    # Base identity.
    ident = lineage.get("identity")
    if not isinstance(ident, dict):
        return "lineage base identity missing/malformed"
    for k in ("size", "mtime"):
        if not _is_finite_number(ident.get(k)):
            return f"lineage base identity {k} invalid"

    # Task 13: framing baseline identity (optional; None for old projects).
    # Malformed -> invalid_asset (same class as every other incomplete
    # baseline), never a silent trust.
    _fr = lineage.get("framing")
    if _fr is not None:
        if not isinstance(_fr, dict):
            return "lineage framing malformed"
        # Digests are plain strings; tracker_digest may legitimately be ""
        # (see FramingPlanIdentity.tracker_digest default). Only plan_ref
        # must be non-empty to be resolvable.
        for _k in ("source_fingerprint", "config_digest",
                   "tracker_digest", "timeline_digest"):
            if not isinstance(_fr.get(_k), str):
                return f"lineage framing {_k} invalid"
        if not isinstance(_fr.get("plan_ref"), str) or not _fr.get("plan_ref"):
            return "lineage framing plan_ref invalid"
        _sha = _fr.get("sha256")
        if not (isinstance(_sha, str) and len(_sha) == 64):
            return "lineage framing sha256 invalid"
        try:
            int(_sha, 16)
        except (TypeError, ValueError):
            return "lineage framing sha256 invalid"
        if not _is_finite_number(_fr.get("output_aspect")) or not (
            _fr.get("output_aspect") > 0
        ):
            return "lineage framing output_aspect invalid"
        if not isinstance(_fr.get("analysis_complete"), bool):
            return "lineage framing analysis_complete invalid"

    return None


def check_lineage_eligible(
    lineage: dict[str, Any],
    project: dict[str, Any],
    job_dir: str,
) -> tuple[str, str | None, str | None]:
    """Check lineage eligibility with asset vs geometry distinction.

    Order:
    1. Validate baseline structure (incomplete -> invalid_asset).
    2. Compare geometry (source/segments/render). Differences ->
       geometry_edit, even if the old base is missing/stale.
    3. Resolve and validate base assets. Problems -> invalid_asset.
    4. Validate source file identity. Mismatch -> invalid_asset.

    Returns (outcome, base_path, reason).
    """
    if not isinstance(lineage, dict):
        return ("invalid_asset", None, "lineage malformed")
    if not isinstance(project, dict):
        return ("invalid_asset", None, "project malformed")

    # 1. Baseline structure first.
    baseline_err = _validate_baseline(lineage)
    if baseline_err:
        return ("invalid_asset", None, baseline_err)

    # Normalize project dicts (handle Pydantic models).
    def _norm(v):
        return v.model_dump() if hasattr(v, "model_dump") else v

    proj_source = _norm(project.get("source"))
    proj_segs = _norm(project.get("segments"))
    proj_render = _norm(project.get("render"))
    if not isinstance(proj_source, dict):
        return ("invalid_asset", None, "project source malformed")
    if not isinstance(proj_segs, list):
        return ("invalid_asset", None, "project segments malformed")
    if not isinstance(proj_render, dict):
        return ("invalid_asset", None, "project render malformed")

    stored_source = lineage["source"]
    stored_segs = lineage["segments"]
    stored_render = lineage["render"]

    # 2. Geometry comparison (before base asset checks).
    # Source file/dimensions.
    for key in ("file", "width", "height"):
        if stored_source.get(key) != proj_source.get(key):
            return ("geometry_edit", None, f"source {key} changed")

    # Segments (ordered; IDs are metadata).
    if len(stored_segs) != len(proj_segs):
        return ("geometry_edit", None,
                f"segment count changed ({len(stored_segs)} -> {len(proj_segs)})")
    for i, (s_stored, s_proj_raw) in enumerate(zip(stored_segs, proj_segs)):
        s_proj = _norm(s_proj_raw)
        if not isinstance(s_proj, dict):
            return ("invalid_asset", None, f"project segment {i} malformed")
        for key in ("start", "end", "transition_in"):
            if s_stored.get(key) != s_proj.get(key):
                return ("geometry_edit", None,
                        f"segment {i} {key} changed")
        c_proj = _norm(s_proj.get("crop"))
        # Baseline crop is validated; project crop missing -> geometry edit
        # (user removed crop) or invalid if malformed type.
        if not isinstance(c_proj, dict):
            return ("geometry_edit", None,
                    f"segment {i} crop changed")
        for key in ("x", "y", "w", "h"):
            if s_stored["crop"].get(key) != c_proj.get(key):
                return ("geometry_edit", None,
                        f"segment {i} crop {key} changed")

    # Render geometry.
    for key in ("width", "height", "fps"):
        if stored_render.get(key) != proj_render.get(key):
            return ("geometry_edit", None, f"render {key} changed")

    # Task 13: framing trajectory comparison (part of the geometry
    # baseline). Both None -> ok. A set-vs-None mismatch -> framing_edit.
    # Both set -> compare the identity keys; any difference -> framing_edit.
    # Never silently reuse the old base under a changed trajectory, and never
    # fall back to a static center crop.
    stored_framing = lineage.get("framing")
    proj_framing = _norm(project.get("framing"))
    if stored_framing is None and proj_framing is None:
        pass
    elif stored_framing is None:
        return ("framing_edit", None,
                "framing trajectory added (reason: framing-trajectory-edit)")
    elif proj_framing is None:
        return ("framing_edit", None,
                "framing trajectory removed (reason: framing-trajectory-edit)")
    else:
        if not isinstance(proj_framing, dict):
            return ("invalid_asset", None, "project framing malformed")
        proj_identity: dict[str, Any] | None = None
        try:
            from clippyme.domain.project_framing import (
                framing_baseline_identity as _fbi,
            )
        except ImportError:
            _fbi = None  # type: ignore[assignment]
        if _fbi is not None:
            try:
                proj_identity = _fbi(proj_framing)
            except Exception:
                proj_identity = None
            if not isinstance(proj_identity, dict):
                return ("invalid_asset", None, "project framing malformed")
        else:
            # Domain helper unavailable: compare the flat identity keys
            # directly (the baseline is stored flat).
            proj_identity = proj_framing
        for key in ("plan_ref", "sha256", "source_fingerprint",
                    "config_digest", "tracker_digest", "output_aspect",
                    "analysis_complete", "timeline_digest"):
            if stored_framing.get(key) != proj_identity.get(key):
                return ("framing_edit", None,
                        f"framing {key} changed "
                        f"(reason: framing-trajectory-edit)")

    # 3. Base asset checks (only when geometry matches).
    base_path = _safe_resolve(lineage.get("base_ref"), job_dir)
    if base_path is None:
        return ("invalid_asset", None,
                f"invalid base reference: {lineage.get('base_ref')!r}")
    if not os.path.isfile(base_path):
        return ("invalid_asset", None,
                f"base file missing: {lineage.get('base_ref')!r}")

    stored_identity = lineage["identity"]
    current_identity = _file_identity(base_path)
    if current_identity is None:
        return ("invalid_asset", None, "cannot stat base file")
    for key in ("size", "mtime"):
        if stored_identity.get(key) != current_identity.get(key):
            return ("invalid_asset", None,
                    f"base file {key} changed (stale/replaced)")

    # 4. Source file identity.
    stored_src_id = stored_source["identity"]
    src_file = proj_source.get("file")
    candidate = os.path.join(os.path.abspath(job_dir),
                             os.path.basename(src_file))
    current_src_id = _file_identity(candidate)
    if current_src_id is None:
        return ("invalid_asset", None, "source file missing")
    for key in ("size", "mtime"):
        if stored_src_id.get(key) != current_src_id.get(key):
            return ("invalid_asset", None,
                    f"source file {key} changed (replaced)")

    return ("eligible", base_path, None)
