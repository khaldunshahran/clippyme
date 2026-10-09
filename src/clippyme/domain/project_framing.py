"""Task 13: stable framing sidecar bridge (domain side).

Pure helpers that validate, normalize, identify and join a Task 12
turn-aware framing plan to a clip project. This module never detects
faces, decodes video, or calls a live provider: it consumes the persisted
``*.turn_framing_plan.json`` sidecar through the Task 12 production
parser and produces the additive, strict ``framing`` project field
defined by ``clippyme.domain.clip_project.ProjectFraming``.

Coordinate basis
----------------
Task 12 plans store ABSOLUTE frame indices in the render input, which in
production is the already-cut source slice (frame 0 = slice start =
selected clip start). :func:`normalize_timeline` performs EXACTLY ONE
rebase with integer arithmetic::

    clip_frame = plan_frame - slice_offset_frames

No float frame math is used anywhere, so 23.976/24/29.97/30/60 fps plans
cannot drift a boundary across a frame. Scene ``start_frame`` /
``end_frame`` stay INCLUSIVE ints (half-open ``[start, end + 1)`` in
frame space); only the origin moves.

Lifecycle (pipeline side promotes the artifact; this module reads it)
---------------------------------------------------------------------
1. Render produces a TEMPORARY sidecar next to the temp video.
2. After render validation the pipeline promotes it to the stable,
   job-relative name ``<clip_final stem>.turn_framing_plan.json`` via
   :func:`publish_attempt_sidecar` -- video first, then the sidecar, as
   one coordinated pair. Promotion VERIFIES the plan's Task 12
   ``source_fingerprint`` against the actual render input
   (:func:`verify_attempt_sidecar_source`); a plan naming a different
   source is REJECTED -- the attempt is treated as sidecar-less, never
   relabelled. The caller stages the previous accepted pair BEFORE
   publication (:func:`stage_previous_pair`) and restores BOTH the video
   and the sidecar on every publication/retirement failure
   (:func:`restore_staged_pair`); a retirement ``os.remove`` failure is
   never swallowed. A sidecar-less attempt retires the previous stable
   binding. A stable sidecar referenced by a preserved user project is
   never clobbered (:func:`stable_sidecar_protected`).
3. :func:`build_project_framing` resolves, validates and normalizes the
   stable sidecar into the project ``framing`` field. Any sidecar
   problem (missing, malformed, temp-named, escape, aspect mismatch,
   cross-source) yields ``None`` -- the project is emitted WITHOUT the
   field (the pre-Task-13 behavior), never with a lying field. When the
   caller passes ``expected_source_fingerprint`` (the actual current
   source slice), a structurally valid plan from a different source is
   rejected.
4. At render, the pipeline calls :func:`verify_framing_at_render` in the
   eligible-base branch: the sidecar must still exist and match the
   recorded sha256 + source_fingerprint + size. The verified plan is
   returned so the caller can recheck its source identity against the
   actual current source slice before reusing the base.

Identity notes
--------------
``framing_baseline_identity`` returns the comparable subset used for
lineage: ``plan_ref``, ``sha256``, ``source_fingerprint``,
``config_digest``, ``tracker_digest``, ``output_aspect``,
``analysis_complete``, ``timeline_digest``. ``size``/``mtime_ns`` are
deliberately EXCLUDED: re-promotion of byte-identical content must not
read as a trajectory change.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Stable reason strings (the FramingError message IS the reason, optionally
# followed by ": <detail>").
# ---------------------------------------------------------------------------

REASON_TEMP_NAME = "framing-sidecar-temp-name"
REASON_MISSING = "framing-sidecar-missing"
REASON_ESCAPE = "framing-sidecar-escape"
REASON_MALFORMED = "framing-plan-malformed"
REASON_MISMATCH = "framing-sidecar-mismatch"
REASON_INCOMPLETE = "analysis-incomplete"
REASON_TRAJECTORY_EDIT = "framing-trajectory-edit"
REASON_CROSS_SOURCE = "framing-cross-source"

#: Basename infixes that mark a temporary (unpromoted) render attempt.
#: A persisted project reference must never contain one of these.
TEMP_BASENAME_MARKERS = (".render.tmp", ".reframe.tmp")

#: Deterministic stable sidecar suffix promoted next to the accepted video.
STABLE_SIDECAR_SUFFIX = ".turn_framing_plan.json"

#: Keys compared for lineage eligibility (no size/mtime: re-promotion of
#: identical content must not read as a change).
_BASELINE_KEYS = (
    "plan_ref", "sha256", "source_fingerprint", "config_digest",
    "tracker_digest", "output_aspect", "analysis_complete", "timeline_digest",
)

_KEYFRAME_KINDS = ("lock", "ease")


class FramingError(Exception):
    """A framing problem. The message is the stable reason string."""


@dataclass(frozen=True)
class ResolvedSidecar:
    """A validated stable sidecar: absolute path, job-relative ref, identity."""

    abs_path: str
    ref: str  # job-relative posix-style path, as stored in the project
    sha256: str
    size: int
    mtime_ns: int


# --- lazy Task 12 production API -------------------------------------------
# framing_plan.py imports nothing from clippyme (verified: only stdlib), so
# there is no import cycle; the import stays lazy regardless so importing
# this domain module never pulls in the pipeline package.

_plan_api: dict[str, Any] | None = None


def _task12() -> dict[str, Any]:
    global _plan_api
    if _plan_api is None:
        from clippyme.pipeline.framing_plan import (  # noqa: E402
            FACE_STRATEGIES,
            TRACK_ID_RE,
            plan_from_dict,
            plan_to_dict,
        )
        _plan_api = {
            "FACE_STRATEGIES": FACE_STRATEGIES,
            "TRACK_ID_RE": TRACK_ID_RE,
            "plan_from_dict": plan_from_dict,
            "plan_to_dict": plan_to_dict,
        }
    return _plan_api


def _project_framing_model() -> Any:
    from clippyme.domain.clip_project import ProjectFraming  # noqa: E402
    return ProjectFraming


# --- small guards ------------------------------------------------------------


def _no_bool(name: str, v: Any) -> None:
    if isinstance(v, bool):
        raise FramingError(f"{REASON_MALFORMED}: {name} must not be a bool")


def _strict_int(name: str, v: Any) -> int:
    _no_bool(name, v)
    if not isinstance(v, int):
        raise FramingError(
            f"{REASON_MALFORMED}: {name} must be an int, got "
            f"{type(v).__name__}")
    return v


def _strict_num(name: str, v: Any) -> float:
    _no_bool(name, v)
    if not isinstance(v, (int, float)) or not math.isfinite(v):
        raise FramingError(f"{REASON_MALFORMED}: {name} must be a finite number")
    return float(v)


def _ref_basename(ref: str) -> str:
    return ref.replace("\\", "/").rsplit("/", 1)[-1]


# --- sidecar reference / resolution -------------------------------------------


def stable_sidecar_ref(clip_final: str) -> str:
    """Job-relative stable sidecar ref for a ``clip_final`` path.

    Returns ``<stem>.turn_framing_plan.json`` (basename only), the ref form
    stored in the project. Accepts a job-relative or absolute path.
    """
    if not isinstance(clip_final, str) or not clip_final:
        raise FramingError(f"{REASON_MISSING}: empty clip_final path")
    base = _ref_basename(clip_final)
    stem, _ext = os.path.splitext(base)
    if not stem:
        raise FramingError(f"{REASON_MISSING}: bad clip_final {clip_final!r}")
    return stem + STABLE_SIDECAR_SUFFIX


def resolve_sidecar(job_dir: str, ref: str) -> ResolvedSidecar:
    """Resolve a sidecar ref inside ``job_dir`` with real containment.

    Raises:
        FramingError: ``framing-sidecar-escape`` when the real path leaves
            the job dir (traversal or symlink escape);
            ``framing-sidecar-missing`` when the file does not exist;
            ``framing-sidecar-temp-name`` when the basename carries a
            temporary-attempt marker.
    """
    if not isinstance(ref, str) or not ref:
        raise FramingError(f"{REASON_MISSING}: empty sidecar ref")
    real_job = os.path.realpath(job_dir)
    abs_path = os.path.realpath(os.path.join(real_job, ref))
    if abs_path != real_job and not abs_path.startswith(real_job + os.sep):
        raise FramingError(f"{REASON_ESCAPE}: {ref!r} escapes the job dir")
    if not os.path.isfile(abs_path):
        raise FramingError(f"{REASON_MISSING}: {ref!r}")
    for marker in TEMP_BASENAME_MARKERS:
        if marker in _ref_basename(ref) or marker in os.path.basename(abs_path):
            raise FramingError(f"{REASON_TEMP_NAME}: {ref!r}")
    st = os.stat(abs_path)
    h = hashlib.sha256()
    with open(abs_path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return ResolvedSidecar(
        abs_path=abs_path,
        ref=ref,
        sha256=h.hexdigest(),
        size=st.st_size,
        mtime_ns=st.st_mtime_ns,
    )


def load_validated_plan(resolved: ResolvedSidecar) -> dict:
    """Read a resolved sidecar and validate it through the production parser.

    Returns the canonical ``plan_to_dict`` form. Anything unexpected --
    unreadable file, bad JSON, or a ``ValueError`` from
    ``plan_from_dict`` -- raises ``FramingError("framing-plan-malformed: …")``.
    """
    api = _task12()
    try:
        with open(resolved.abs_path, "r", encoding="utf-8") as fh:
            raw = fh.read()
    except OSError as e:
        raise FramingError(f"{REASON_MALFORMED}: unreadable sidecar: {e}")
    except UnicodeDecodeError as e:
        raise FramingError(f"{REASON_MALFORMED}: sidecar not valid UTF-8: {e}")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise FramingError(f"{REASON_MALFORMED}: invalid JSON ({e})")
    try:
        plan = api["plan_from_dict"](data)
    except ValueError as e:
        raise FramingError(f"{REASON_MALFORMED}: {e}")
    return api["plan_to_dict"](plan)


# --- timeline normalization -----------------------------------------------------


def _check_coords(ctx: str, k: dict, frame_w: int, frame_h: int) -> dict:
    x1 = _strict_int(f"{ctx}.x1", k["x1"])
    y1 = _strict_int(f"{ctx}.y1", k["y1"])
    x2 = _strict_int(f"{ctx}.x2", k["x2"])
    y2 = _strict_int(f"{ctx}.y2", k["y2"])
    if not 0 <= x1 < x2 <= frame_w:
        raise FramingError(f"{REASON_MALFORMED}: {ctx} x-coords out of bounds")
    if not 0 <= y1 < y2 <= frame_h:
        raise FramingError(f"{REASON_MALFORMED}: {ctx} y-coords out of bounds")
    kind = k["kind"]
    if kind not in _KEYFRAME_KINDS:
        raise FramingError(f"{REASON_MALFORMED}: {ctx} bad kind {kind!r}")
    return {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "kind": kind}


def normalize_timeline(
    *,
    plan: dict,
    slice_offset_frames: int,
    clip_frame_count: int,
    fps: float,
) -> list[dict]:
    """Normalize a validated Task 12 plan to the clip-relative timeline.

    Performs EXACTLY ONE rebase with integer arithmetic::

        clip_frame = plan_frame - slice_offset_frames

    (the plan is authored in the render input's absolute frames; the
    project source is the already-cut slice whose frame 0 is the clip
    start). Scene/keyframe ``start_frame``/``end_frame``/``frame`` stay
    INCLUSIVE ints -- half-open ``[start, end + 1)`` in frame space.

    Validation: booleans rejected for every numeric input; ``0 <=
    slice_offset_frames``; every scene ordered, finite, ``0 <= clip_start
    <= clip_end < clip_frame_count``; keyframes ordered with unique
    frames inside their scene; coords finite ints with ``0 <= x1 < x2 <=
    frame_w`` (same for y vs ``frame_h``); strategy strings preserved
    verbatim; non-face strategies (anything not TRACK/WIDE) keep NO
    keyframes (``[]``) and ``applies`` False -- keyframes are never
    manufactured for them; fallback reason/detail tokens preserved.

    Returns a list of scene dicts: ``{scene_id, strategy, start_frame,
    end_frame, applies, subject_ids, keyframes, fallback_reason,
    fallback_detail}`` with clip-relative inclusive frames.

    Raises:
        FramingError: on any malformed input (never silently coerces).
    """
    _no_bool("slice_offset_frames", slice_offset_frames)
    _no_bool("clip_frame_count", clip_frame_count)
    _no_bool("fps", fps)
    offset = _strict_int("slice_offset_frames", slice_offset_frames)
    total = _strict_int("clip_frame_count", clip_frame_count)
    rate = _strict_num("fps", fps)
    if offset < 0:
        raise FramingError(f"{REASON_MALFORMED}: slice_offset_frames < 0")
    if total <= 0:
        raise FramingError(f"{REASON_MALFORMED}: clip_frame_count must be > 0")
    if rate <= 0:
        raise FramingError(f"{REASON_MALFORMED}: fps must be > 0")
    if not isinstance(plan, dict):
        raise FramingError(f"{REASON_MALFORMED}: plan must be a dict")

    api = _task12()
    face_strategies = api["FACE_STRATEGIES"]
    track_id_re = api["TRACK_ID_RE"]

    frame_w = _strict_int("plan.frame_w", plan.get("frame_w"))
    frame_h = _strict_int("plan.frame_h", plan.get("frame_h"))
    if frame_w <= 0 or frame_h <= 0:
        raise FramingError(f"{REASON_MALFORMED}: plan dimensions invalid")

    raw_scenes = plan.get("scenes")
    if not isinstance(raw_scenes, list):
        raise FramingError(f"{REASON_MALFORMED}: plan scenes must be a list")

    out: list[dict] = []
    prev_end = -1
    prev_scene_id = -1
    for idx, s in enumerate(raw_scenes):
        ctx = f"scene[{idx}]"
        if not isinstance(s, dict):
            raise FramingError(f"{REASON_MALFORMED}: {ctx} must be a dict")
        scene_id = _strict_int(f"{ctx}.scene_id", s.get("scene_id"))
        if scene_id <= prev_scene_id:
            raise FramingError(
                f"{REASON_MALFORMED}: {ctx} scene ids must be ordered")
        prev_scene_id = scene_id
        strategy = s.get("strategy")
        if not isinstance(strategy, str) or not strategy:
            raise FramingError(f"{REASON_MALFORMED}: {ctx} bad strategy")
        p_start = _strict_int(f"{ctx}.start_frame", s.get("start_frame"))
        p_end = _strict_int(f"{ctx}.end_frame", s.get("end_frame"))
        start = p_start - offset
        end = p_end - offset
        if not 0 <= start <= end < total:
            raise FramingError(
                f"{REASON_MALFORMED}: {ctx} clip-relative frames "
                f"[{start}, {end}] outside [0, {total})")
        if start <= prev_end:
            raise FramingError(
                f"{REASON_MALFORMED}: {ctx} scenes must be ordered and "
                "non-overlapping")
        prev_end = end
        applies = s.get("applies")
        if not isinstance(applies, bool):
            raise FramingError(f"{REASON_MALFORMED}: {ctx} applies must be bool")

        is_face_strategy = strategy in face_strategies
        if not is_face_strategy:
            # Non-face strategies (SPLIT/SCREENCAST/GENERAL/DISABLED/…)
            # keep NO keyframes: their renderer is static/wide and we never
            # manufacture crop targets. A non-face scene claiming
            # applicability or carrying face content contradicts production
            # semantics -> untrusted.
            if applies:
                raise FramingError(
                    f"{REASON_MALFORMED}: {ctx} non-face scene claims "
                    "applies")
            for key in ("segments", "keyframes", "tracks"):
                if s.get(key):
                    raise FramingError(
                        f"{REASON_MALFORMED}: {ctx} non-face scene carries "
                        f"'{key}'")
            if s.get("fallback") or s.get("fallback_crop"):
                raise FramingError(
                    f"{REASON_MALFORMED}: {ctx} non-face scene carries a "
                    "fallback")
            out.append({
                "scene_id": scene_id,
                "strategy": strategy,
                "start_frame": start,
                "end_frame": end,
                "applies": False,
                "subject_ids": [],
                "keyframes": [],
                "fallback_reason": None,
                "fallback_detail": None,
            })
            continue
        if not applies:
            # Face strategy the plan did not apply (production never emits
            # this): no trajectory, same strict content ban.
            for key in ("segments", "keyframes", "tracks"):
                if s.get(key):
                    raise FramingError(
                        f"{REASON_MALFORMED}: {ctx} non-applicable scene "
                        f"carries '{key}'")
            if s.get("fallback") or s.get("fallback_crop"):
                raise FramingError(
                    f"{REASON_MALFORMED}: {ctx} non-applicable scene "
                    "carries a fallback")
            out.append({
                "scene_id": scene_id,
                "strategy": strategy,
                "start_frame": start,
                "end_frame": end,
                "applies": False,
                "subject_ids": [],
                "keyframes": [],
                "fallback_reason": None,
                "fallback_detail": None,
            })
            continue

        # Face scene (TRACK/WIDE, applies): anonymous subject ids from the
        # scene's segments, in first-appearance order.
        subject_ids: list[str] = []
        raw_segments = s.get("segments")
        if not isinstance(raw_segments, list):
            raise FramingError(f"{REASON_MALFORMED}: {ctx} segments bad")
        for g in raw_segments:
            if not isinstance(g, dict):
                raise FramingError(f"{REASON_MALFORMED}: {ctx} segment bad")
            for sid in g.get("subject_ids") or []:
                if not isinstance(sid, str) or not track_id_re.match(sid):
                    raise FramingError(
                        f"{REASON_MALFORMED}: {ctx} non-anonymous subject id")
                if sid not in subject_ids:
                    subject_ids.append(sid)

        raw_keyframes = s.get("keyframes")
        if not isinstance(raw_keyframes, list):
            raise FramingError(f"{REASON_MALFORMED}: {ctx} keyframes bad")
        keyframes: list[dict] = []
        prev_kf = start - 1
        for j, k in enumerate(raw_keyframes):
            kctx = f"{ctx}.keyframe[{j}]"
            if not isinstance(k, dict):
                raise FramingError(f"{REASON_MALFORMED}: {kctx} must be a dict")
            p_frame = _strict_int(f"{kctx}.frame", k.get("frame"))
            frame = p_frame - offset
            if frame <= prev_kf:
                raise FramingError(
                    f"{REASON_MALFORMED}: {kctx} keyframes must be ordered "
                    "with unique frames")
            if not start <= frame <= end:
                raise FramingError(
                    f"{REASON_MALFORMED}: {kctx} frame outside its scene")
            prev_kf = frame
            coords = _check_coords(kctx, k, frame_w, frame_h)
            keyframes.append({"frame": frame, **coords})

        fb = s.get("fallback")
        fallback_reason = None
        fallback_detail = None
        if fb is not None:
            if not isinstance(fb, dict):
                raise FramingError(f"{REASON_MALFORMED}: {ctx} fallback bad")
            fallback_reason = fb.get("reason")
            fallback_detail = fb.get("detail")
            if not isinstance(fallback_reason, str) or not fallback_reason:
                raise FramingError(
                    f"{REASON_MALFORMED}: {ctx} fallback reason bad")
            if fallback_detail is not None and not isinstance(
                    fallback_detail, str):
                raise FramingError(
                    f"{REASON_MALFORMED}: {ctx} fallback detail bad")

        out.append({
            "scene_id": scene_id,
            "strategy": strategy,
            "start_frame": start,
            "end_frame": end,
            "applies": True,
            "subject_ids": subject_ids,
            "keyframes": keyframes,
            "fallback_reason": fallback_reason,
            "fallback_detail": fallback_detail,
        })
    return out


def timeline_digest(scenes: list[dict]) -> str:
    """sha256 over the canonical JSON of normalized scenes."""
    payload = json.dumps(
        scenes, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# --- project bridge -----------------------------------------------------------


def _validate_or_none(framing: dict) -> Optional[dict]:
    """Validate against the ProjectFraming model; None when invalid."""
    try:
        _project_framing_model().model_validate(framing)
    except Exception:
        return None
    return framing


def _render_geometry(render_w: Any, render_h: Any,
                     render_fps: Any) -> tuple[int, int, float] | None:
    """Validated render geometry, or None when unusable."""
    try:
        w = _strict_int("render_w", render_w)
        h = _strict_int("render_h", render_h)
        fps = _strict_num("render_fps", render_fps)
    except FramingError:
        return None
    if w <= 0 or h <= 0 or fps <= 0:
        return None
    return w, h, fps


def build_project_framing(
    *,
    job_dir: str,
    clip_final: str,
    render_w: int,
    render_h: int,
    render_fps: float,
    slice_offset_frames: int = 0,
    clip_frame_count: int,
    expected_source_fingerprint: str | None = None,
) -> Optional[dict]:
    """Build the project ``framing`` field from the stable sidecar.

    Full bridge: stable ref -> resolve -> production-parse -> normalize ->
    identity/digest -> ProjectFraming-validated dict.

    When ``expected_source_fingerprint`` is given (the actual current
    source slice's Task 12 identity), a structurally valid plan naming a
    different source is rejected with ``None`` (``framing-cross-source``)
    -- cross-source plans are never trusted.

    Returns ``None`` (old behavior: project emitted WITHOUT the field)
    whenever the sidecar is missing, unresolvable, malformed, incomplete
    beyond audit, aspect-incompatible with this render, cross-source, or
    fails model validation. Never raises for sidecar problems and never
    emits a lying field.
    """
    try:
        ref = stable_sidecar_ref(clip_final)
        resolved = resolve_sidecar(job_dir, ref)
        plan = load_validated_plan(resolved)
    except FramingError:
        return None

    # Review-01: the sidecar's Task 12 source identity must match the
    # ACTUAL current source slice. A valid plan from a different source
    # is rejected, not trusted.
    if (expected_source_fingerprint is not None
            and plan["source_fingerprint"] != expected_source_fingerprint):
        return None

    geom = _render_geometry(render_w, render_h, render_fps)
    if geom is None:
        return None
    rw, rh, rfps = geom

    plan_identity = {
        "plan_ref": ref,
        "sha256": resolved.sha256,
        "size": resolved.size,
        "mtime_ns": resolved.mtime_ns,
        "schema_version": plan["schema_version"],
        "producer": plan["producer"],
        "source_fingerprint": plan["source_fingerprint"],
        "config_digest": plan["config_digest"],
        "tracker_digest": plan["tracker_digest"],
        "output_aspect": plan["output_aspect"],
        "analysis_complete": bool(plan["analysis_complete"]),
        "timeline_digest": timeline_digest([]),
    }
    source_block = {
        "fingerprint": plan["source_fingerprint"],
        "width": plan["frame_w"],
        "height": plan["frame_h"],
        "fps": plan["fps"],
        "frame_count": plan["frame_count"],
        "slice_offset_frames": slice_offset_frames,
    }
    render_block = {
        "width": rw,
        "height": rh,
        "fps": rfps,
        "aspect": rw / rh,
    }

    if not plan["analysis_complete"]:
        # Incomplete Task 12 plans are audit references only: eligible
        # False, no trajectory. The exhausted fallback must never be
        # presented as a normal camera path.
        return _validate_or_none({
            "version": 1,
            "eligible": False,
            "ineligibility_reason": REASON_INCOMPLETE,
            "plan": plan_identity,
            "source": source_block,
            "render": render_block,
            "scenes": [],
        })

    try:
        scenes = normalize_timeline(
            plan=plan,
            slice_offset_frames=slice_offset_frames,
            clip_frame_count=clip_frame_count,
            fps=plan["fps"],
        )
    except FramingError:
        return None

    # The plan's crops were built for its output aspect: an aspect
    # mismatch means this plan does not describe this render.
    render_aspect = rw / rh
    plan_aspect = plan["output_aspect"]
    if abs(plan_aspect - render_aspect) / render_aspect > 0.01:
        return None

    plan_identity["timeline_digest"] = timeline_digest(scenes)
    return _validate_or_none({
        "version": 1,
        "eligible": True,
        "plan": plan_identity,
        "source": source_block,
        "render": render_block,
        "scenes": scenes,
    })


def verify_framing_at_render(*, job_dir: str, framing: dict) -> dict:
    """Re-verify the stored framing against the live sidecar before render.

    Re-resolves ``framing["plan"]["plan_ref"]`` inside ``job_dir`` and
    compares sha256 AND source_fingerprint AND size against the recorded
    identity. Same-size/same-mtime content replacement is detected via
    the content hash.

    Returns the validated canonical plan dict, so the caller can recheck
    the plan's source identity against the actual current source slice
    before reusing a reframed base.

    Raises:
        FramingError: ``framing-sidecar-missing`` when the ref or file is
            gone; ``framing-sidecar-mismatch: …`` on any identity drift.
    """
    plan_id = framing.get("plan") if isinstance(framing, dict) else None
    ref = plan_id.get("plan_ref") if isinstance(plan_id, dict) else None
    if not isinstance(ref, str) or not ref:
        raise FramingError(f"{REASON_MISSING}: stored framing has no plan_ref")
    resolved = resolve_sidecar(job_dir, ref)
    plan = load_validated_plan(resolved)
    if plan_id.get("sha256") != resolved.sha256:
        raise FramingError(
            f"{REASON_MISMATCH}: sha256 changed for {ref!r}")
    if plan_id.get("size") != resolved.size:
        raise FramingError(f"{REASON_MISMATCH}: size changed for {ref!r}")
    if plan_id.get("source_fingerprint") != plan["source_fingerprint"]:
        raise FramingError(
            f"{REASON_MISMATCH}: source_fingerprint changed for {ref!r}")
    return plan


def framing_baseline_identity(framing: dict) -> dict:
    """Comparable framing identity subset for the lineage baseline.

    Returns ``{plan_ref, sha256, source_fingerprint, config_digest,
    tracker_digest, output_aspect, analysis_complete, timeline_digest}``.
    ``size``/``mtime_ns`` are excluded on purpose: re-promotion of
    byte-identical content must not read as a trajectory change.

    Raises:
        FramingError: ``framing-plan-malformed`` when the stored framing
            has no usable plan identity.
    """
    plan_id = framing.get("plan") if isinstance(framing, dict) else None
    if not isinstance(plan_id, dict):
        raise FramingError(f"{REASON_MALFORMED}: framing has no plan identity")
    missing = [k for k in _BASELINE_KEYS if k not in plan_id]
    if missing:
        raise FramingError(
            f"{REASON_MALFORMED}: plan identity missing {missing!r}")
    return {k: plan_id[k] for k in _BASELINE_KEYS}


# --- coordinated video + sidecar publication --------------------------------
# Review-01: the accepted video and its stable sidecar are published as ONE
# coordinated pair. The caller publishes the video first (os.replace temp
# -> final) and then calls publish_attempt_sidecar() exactly once.
#
# - Promotion failure is NEVER swallowed: the OSError propagates so the
#   caller can roll the just-published video back. An accepted attempt
#   must never pair a new video with a stale plan.
# - A sidecar-less attempt retires the previous stable binding, so the
#   new video cannot inherit a stale plan.
# - A stable sidecar referenced by a preserved user project (Task 01) is
#   never clobbered: promotion/retire skip it, keeping the preserved
#   project verifiable.
# - Promotion rebinds -> REMOVED (review-02). The attempt plan's Task 12
#   source_fingerprint is VERIFIED against the actual render input
#   (verify_attempt_sidecar_source); a plan naming a different source is
#   REJECTED and the attempt is treated as sidecar-less. Cross-source
#   evidence is never rewritten.

#: Outcomes of :func:`publish_attempt_sidecar`.
PUBLISH_PROMOTED = "promoted"
PUBLISH_RETIRED = "retired"
PUBLISH_NO_SIDECAR = "no-sidecar"
PUBLISH_SKIPPED_PRESERVED = "skipped-preserved"


def _task12_source_fingerprint(path: str) -> str:
    from clippyme.pipeline.framing_plan import (  # noqa: E402
        source_fingerprint,
    )
    return source_fingerprint(path)


def verify_attempt_sidecar_source(attempt_sidecar: str, *,
                                 source_path: str) -> bool:
    """Verify an attempt plan names the actual render input; reject mismatch.

    Parses the attempt sidecar through the production Task 12 parser,
    fingerprints ``source_path``, and compares the plan's
    ``source_fingerprint``. A mismatch raises
    ``FramingError("framing-cross-source: …")`` -- the attempt is treated
    as sidecar-less by the caller, never relabelled. Source identity is
    evidence; publication must not rewrite it.

    Returns True when the plan names the actual render input (no rewrite;
    the file bytes are untouched).

    Raises:
        FramingError: the sidecar is missing or not a production-valid
            plan, the source file cannot be fingerprinted, or the plan
            names a different source.
    """
    if not os.path.isfile(attempt_sidecar):
        raise FramingError(f"{REASON_MISSING}: {attempt_sidecar!r}")
    try:
        fingerprint = _task12_source_fingerprint(source_path)
    except OSError as exc:
        raise FramingError(
            f"{REASON_MISSING}: cannot fingerprint source: {exc}")
    api = _task12()
    try:
        with open(attempt_sidecar, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FramingError(f"{REASON_MALFORMED}: unreadable sidecar: {exc}")
    try:
        plan = api["plan_from_dict"](data)
    except ValueError as exc:
        raise FramingError(f"{REASON_MALFORMED}: {exc}")
    if plan.source_fingerprint != fingerprint:
        raise FramingError(
            f"{REASON_CROSS_SOURCE}: attempt plan names a different source; "
            "cross-source evidence is rejected, never rewritten")
    return True


def _sha256_file(path: str) -> str | None:
    h = hashlib.sha256()
    try:
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
    except OSError:
        return None
    return h.hexdigest()


def _same_bytes(a: str, b: str) -> bool:
    ha, hb = _sha256_file(a), _sha256_file(b)
    return ha is not None and ha == hb


def _preserved_project_refs_sidecar(job_dir: str, stable_basename: str,
                                    stable_sha256: str) -> bool:
    """True when a preserved user project references this exact sidecar.

    Scans ``clip-project-*.json`` (working + versioned) for a project the
    pipeline would preserve -- the FULL Task 01 preservation contract
    (:func:`project_has_user_edits`: ``origin == "user"``, ``version >
    1``, non-empty ``captions.edits``, or user-marked segments) -- whose
    ``framing.plan`` names this sidecar basename with a matching content
    hash. Unreadable files are ignored -- a reference that cannot be
    established cannot protect.

    The import is lazy: ``project_emit`` imports this module lazily
    inside functions, so a module-level import would cycle.
    """
    try:
        names = os.listdir(job_dir)
    except OSError:
        return False
    try:
        from clippyme.pipeline.project_emit import (  # noqa: E402
            project_has_user_edits,
        )
    except ImportError:
        return False
    for name in names:
        if not (name.startswith("clip-project-") and name.endswith(".json")):
            continue
        path = os.path.join(job_dir, name)
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, ValueError):
            continue
        if not isinstance(data, dict):
            continue
        try:
            preserved = project_has_user_edits(path)
        except Exception:
            continue
        if not preserved:
            continue
        framing = data.get("framing")
        plan_id = (framing.get("plan")
                   if isinstance(framing, dict) else None)
        if not isinstance(plan_id, dict):
            continue
        ref = plan_id.get("plan_ref")
        if not isinstance(ref, str) or _ref_basename(ref) != stable_basename:
            continue
        if plan_id.get("sha256") == stable_sha256:
            return True
    return False


def stable_sidecar_protected(*, job_dir: str, stable_path: str) -> bool:
    """True when the stable sidecar must not be clobbered.

    A preserved user project (Task 01 preservation) that references this
    exact sidecar (plan_ref basename + sha256) pins the artifact:
    promotion and retire must leave it in place so the preserved project
    stays verifiable via :func:`verify_framing_at_render`.
    """
    if not os.path.isfile(stable_path):
        return False
    digest = _sha256_file(stable_path)
    if digest is None:
        return False
    return _preserved_project_refs_sidecar(
        job_dir, os.path.basename(stable_path), digest)


def _retire_stable_sidecar(*, job_dir: str, stable_sidecar: str) -> str:
    """Retire a stale stable binding for a sidecar-less attempt.

    A removal failure is NEVER swallowed: the :exc:`OSError` propagates
    so the caller treats it like any publication failure -- rolling back
    to the staged pair -- instead of reporting ``"retired"`` while the
    stale file remains.
    """
    if not os.path.isfile(stable_sidecar):
        return PUBLISH_NO_SIDECAR
    if stable_sidecar_protected(job_dir=job_dir, stable_path=stable_sidecar):
        return PUBLISH_SKIPPED_PRESERVED
    os.remove(stable_sidecar)
    return PUBLISH_RETIRED


def publish_attempt_sidecar(*, attempt_sidecar: str, stable_sidecar: str,
                            source_path: str, job_dir: str) -> str:
    """Coordinate the sidecar half of video+sidecar publication.

    The caller publishes the accepted video FIRST (``os.replace`` temp ->
    final) and then calls this exactly once.

    - No usable attempt sidecar: the previous stable binding is retired
      (unless a preserved user project references it) so the new video
      never inherits a stale plan. Returns ``"retired"`` /
      ``"no-sidecar"`` / ``"skipped-preserved"``. A retirement
      ``os.remove`` failure propagates as :exc:`OSError` -- never reported
      as retired while the stale file remains.
    - Attempt sidecar present: its Task 12 source identity is VERIFIED
      against ``source_path`` (:func:`verify_attempt_sidecar_source`). A
      plan naming a different source is REJECTED -- the attempt is
      treated as sidecar-less (the temp file is removed), never
      relabelled. The verified plan is then promoted atomically. When the
      stable sidecar holds different bytes that a preserved user project
      references, promotion is SKIPPED (the attempt temp file is removed)
      and ``"skipped-preserved"`` is returned. Otherwise returns
      ``"promoted"``.

    A promotion ``os.replace`` failure is NEVER swallowed: the
    :exc:`OSError` propagates so the caller can restore the staged
    video+sidecar pair -- no success with a stale mixed pair.

    Raises:
        OSError: the promotion ``os.replace`` failed AFTER the video was
            published, or the stale-sidecar retirement ``os.remove``
            failed; the caller must restore the staged pair.
        FramingError: never -- a foreign or malformed attempt plan is
            treated as sidecar-less (retire path), not raised.
    """
    usable = os.path.isfile(attempt_sidecar)
    if usable:
        try:
            verify_attempt_sidecar_source(attempt_sidecar,
                                          source_path=source_path)
        except FramingError:
            # Unusable or FOREIGN attempt plan: treat the attempt as
            # sidecar-less and retire the stale binding rather than
            # promoting garbage. Cross-source plans are rejected, never
            # relabelled.
            try:
                os.remove(attempt_sidecar)
            except OSError:
                pass
            usable = False
    if not usable:
        return _retire_stable_sidecar(
            job_dir=job_dir, stable_sidecar=stable_sidecar)
    if os.path.isfile(stable_sidecar):
        if _same_bytes(attempt_sidecar, stable_sidecar):
            # Byte-identical re-promotion: nothing to protect, and the
            # content a preserved project references is unchanged.
            try:
                os.remove(attempt_sidecar)
            except OSError:
                pass
            return PUBLISH_PROMOTED
        if stable_sidecar_protected(job_dir=job_dir,
                                    stable_path=stable_sidecar):
            # A preserved user project pins the current artifact: the new
            # attempt's different plan must not overwrite it.
            try:
                os.remove(attempt_sidecar)
            except OSError:
                pass
            return PUBLISH_SKIPPED_PRESERVED
    os.replace(attempt_sidecar, stable_sidecar)
    return PUBLISH_PROMOTED


# --- accepted-pair staging and durable snapshot -------------------------------
# Review-02: the accepted video+sidecar pair is never destroyed by a
# failed publication.
#
# - stage_previous_pair(): the caller stages the CURRENT accepted pair
#   BEFORE replacing the final video. restore_staged_pair() puts BOTH
#   files back on any publication/retirement OSError; when no previous
#   video was staged, the just-published (failed) video is removed.
#   discard_staged_pair() cleans up after success. All best-effort, never
#   raising -- staging must not fail a good render.
# - snapshot_accepted_pair(): after a SUCCESSFUL coordinated publication
#   the accepted pair is snapshotted durably (copy2 preserves mtime, so
#   the lineage file identity survives). restore_accepted_snapshot()
#   brings the last accepted pair back; the pipeline uses it when it
#   DECLINES a rerender for a preserved Task 01 project, so the preserved
#   project's reframed_base keeps resolving to the preserved final video.


def stage_previous_pair(*, clip_final: str, stable_sidecar: str) -> dict:
    """Back up the current accepted pair before a new publication.

    Returns an opaque record for :func:`restore_staged_pair` /
    :func:`discard_staged_pair`. ``copy2`` preserves mtimes. Best-effort:
    never raises.
    """
    import shutil
    import tempfile

    record: dict[str, Any] = {"video": None, "sidecar": None, "dir": None}
    try:
        staging = tempfile.mkdtemp(prefix="task13-pair-")
        record["dir"] = staging
        if os.path.isfile(clip_final):
            dst = os.path.join(
                staging, "video" + os.path.splitext(clip_final)[1])
            shutil.copy2(clip_final, dst)
            record["video"] = dst
        if os.path.isfile(stable_sidecar):
            dst = os.path.join(staging, "sidecar.json")
            shutil.copy2(stable_sidecar, dst)
            record["sidecar"] = dst
    except OSError:
        pass
    return record


def restore_staged_pair(record: dict, *, clip_final: str,
                        stable_sidecar: str) -> None:
    """Restore the staged pair after a publication/retirement failure.

    Both the video and the sidecar are put back; when no previous video
    was staged, the just-published (failed) video is removed instead.
    Never leaves a broken pair. Best-effort: never raises.
    """
    import shutil

    try:
        if record.get("video"):
            shutil.copy2(record["video"], clip_final)
        else:
            try:
                if os.path.isfile(clip_final):
                    os.remove(clip_final)
            except OSError:
                pass
        if record.get("sidecar"):
            shutil.copy2(record["sidecar"], stable_sidecar)
        # No sidecar backup: the failed publish left the stable sidecar in
        # its pre-call state (os.replace/os.remove are atomic), so there
        # is nothing to repair.
    except OSError:
        pass
    finally:
        discard_staged_pair(record)


def discard_staged_pair(record: dict) -> None:
    """Clean up staging after a successful publication. Never raises."""
    import shutil

    staging = record.get("dir")
    if staging:
        try:
            shutil.rmtree(staging, ignore_errors=True)
        except OSError:
            pass
    record["video"] = None
    record["sidecar"] = None
    record["dir"] = None


#: Durable snapshot of the last accepted video+sidecar pair, inside the
#: job dir. Restored when a rerender is declined for a preserved project.
ACCEPTED_SNAPSHOT_DIRNAME = ".task13_accepted_pair"


def _snapshot_dir(job_dir: str) -> str:
    return os.path.join(job_dir, ACCEPTED_SNAPSHOT_DIRNAME)


def accepted_snapshot_exists(*, job_dir: str, clip_final: str) -> bool:
    """True when a durable accepted-pair video snapshot exists."""
    snap = os.path.join(
        _snapshot_dir(job_dir),
        "video" + os.path.splitext(clip_final)[1])
    return os.path.isfile(snap)


def snapshot_accepted_pair(*, job_dir: str, clip_final: str,
                           stable_sidecar: str) -> None:
    """Persist the last accepted pair after successful publication.

    ``copy2`` preserves mtimes so a later restore keeps the lineage file
    identity byte- and mtime-identical. A sidecar-less acceptance clears
    any stale sidecar snapshot. Best-effort: never raises.
    """
    import shutil

    try:
        snap = _snapshot_dir(job_dir)
        os.makedirs(snap, exist_ok=True)
        if os.path.isfile(clip_final):
            shutil.copy2(
                clip_final,
                os.path.join(
                    snap, "video" + os.path.splitext(clip_final)[1]))
        sidecar_snap = os.path.join(snap, "sidecar.json")
        if os.path.isfile(stable_sidecar):
            shutil.copy2(stable_sidecar, sidecar_snap)
        else:
            try:
                if os.path.isfile(sidecar_snap):
                    os.remove(sidecar_snap)
            except OSError:
                pass
    except OSError:
        pass


def restore_accepted_snapshot(*, job_dir: str, clip_final: str,
                              stable_sidecar: str) -> bool:
    """Restore the last accepted pair from the durable snapshot.

    Returns True when a video snapshot existed and was restored (mtime
    preserved via copy2, so lineage identity is intact). Best-effort:
    never raises.
    """
    import shutil

    restored = False
    try:
        snap = _snapshot_dir(job_dir)
        video_snap = os.path.join(
            snap, "video" + os.path.splitext(clip_final)[1])
        if os.path.isfile(video_snap):
            shutil.copy2(video_snap, clip_final)
            restored = True
        sidecar_snap = os.path.join(snap, "sidecar.json")
        if os.path.isfile(sidecar_snap):
            shutil.copy2(sidecar_snap, stable_sidecar)
    except OSError:
        pass
    return restored
