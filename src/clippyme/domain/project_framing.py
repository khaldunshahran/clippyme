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
import re
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


class FramingIntegrityError(FramingError, OSError):
    """A rollback/recovery integrity failure. Never collapsed into False.

    Raised when a transactional live-pair rollback could not be verified
    byte-identical to the pre-install state (review-05): the live pair is
    indeterminate and the verified staging backups are retained on disk.
    A distinct ``FramingError`` subtype so callers can single out the
    integrity case; also an ``OSError`` so generic OS-failure handlers
    still surface it.
    """


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
            scene_kind_of,
        )
        _plan_api = {
            "FACE_STRATEGIES": FACE_STRATEGIES,
            "TRACK_ID_RE": TRACK_ID_RE,
            "plan_from_dict": plan_from_dict,
            "plan_to_dict": plan_to_dict,
            "scene_kind_of": scene_kind_of,
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
    verbatim; the explicit ``scene_kind`` contract is honored — ``"none"``
    scenes (SPLIT/SCREENCAST/DISABLED/old GENERAL) keep NO keyframes
    (``[]``) and ``applies`` False, ``"face"`` scenes (TRACK/WIDE) and
    ``"action"`` scenes (GENERAL, Task 14) normalize their trajectories;
    keyframes are never manufactured for non-applicable scenes; fallback
    reason/detail tokens preserved.

    Returns a list of scene dicts: ``{scene_id, strategy, start_frame,
    end_frame, applies, scene_kind, subject_ids, keyframes,
    fallback_reason, fallback_detail}`` with clip-relative inclusive
    frames.

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
    scene_kind_of = api["scene_kind_of"]

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

        # Explicit versioned scene contract (Task 14): the parser
        # distinguishes face, action, and non-applicable scenes; the
        # bridge normalizes each contract on its own terms.
        kind = scene_kind_of(s)
        is_face_strategy = strategy in face_strategies
        if kind == "none":
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
                "scene_kind": "none",
                "subject_ids": [],
                "keyframes": [],
                "fallback_reason": None,
                "fallback_detail": None,
                "fit_ranges": [],
            })
            continue
        if kind == "face":
            if not is_face_strategy:
                raise FramingError(
                    f"{REASON_MALFORMED}: {ctx} face scene with non-face "
                    f"strategy {strategy!r}")
        elif kind == "action":
            # Defense in depth: the v4 parser already binds action scenes
            # to GENERAL; the bridge re-checks on the canonical form.
            if strategy != "GENERAL":
                raise FramingError(
                    f"{REASON_MALFORMED}: {ctx} action scene with "
                    f"non-GENERAL strategy {strategy!r}")
        else:  # pragma: no cover — scene_kind_of only returns the three
            raise FramingError(
                f"{REASON_MALFORMED}: {ctx} unknown scene kind {kind!r}")
        if not applies:
            # Applicable-kind scene the plan did not apply (production
            # never emits this): no trajectory, same strict content ban.
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
                "scene_kind": kind,
                "subject_ids": [],
                "keyframes": [],
                "fallback_reason": None,
                "fallback_detail": None,
                "fit": False,
                "fit_ranges": [],
            })
            continue

        # Applicable face (TRACK/WIDE) or action (GENERAL) scene:
        # anonymous subject ids from the scene's segments, in
        # first-appearance order.
        subject_ids: list[str] = []
        raw_segments = s.get("segments")
        if not isinstance(raw_segments, list):
            raise FramingError(f"{REASON_MALFORMED}: {ctx} segments bad")
        # Task 14 rev4: explicit conservative full-frame fit persisted on
        # action wide segments. Carried as lossless ordered clip-relative
        # fit intervals (rebased once, below) so plans whose fit boundaries
        # differ stay distinct through the bridge. Booleans,
        # coverage/order/bounds and consistency with crop ranges are all
        # validated; the single integer rebase is never applied twice.
        seg_fit = False
        fit_ranges: list[dict] = []
        prev_seg_end = start - 1
        for gi, g in enumerate(raw_segments):
            gctx = f"{ctx}.segment[{gi}]"
            if not isinstance(g, dict):
                raise FramingError(f"{REASON_MALFORMED}: {gctx} must be a dict")
            gs = _strict_int(f"{gctx}.start_frame", g.get("start_frame")) - offset
            ge = _strict_int(f"{gctx}.end_frame", g.get("end_frame")) - offset
            if not start <= gs <= ge <= end:
                raise FramingError(
                    f"{REASON_MALFORMED}: {gctx} frames [{gs}, {ge}] outside "
                    f"scene [{start}, {end}]")
            if gs <= prev_seg_end:
                raise FramingError(
                    f"{REASON_MALFORMED}: {gctx} segments must be ordered "
                    "and non-overlapping")
            prev_seg_end = ge
            _gf = g.get("fit", False)
            if not isinstance(_gf, bool):
                raise FramingError(
                    f"{REASON_MALFORMED}: {gctx} segment fit bad")
            if _gf:
                seg_fit = True
                fit_ranges.append({"start_frame": gs, "end_frame": ge})
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

        # A keyframe is a crop target; a fit range renders full-frame
        # letterbox (the renderer gives fit precedence). A keyframe inside
        # a fit range is contradictory data, never a valid plan.
        for kf in keyframes:
            for fr in fit_ranges:
                rs, re_ = fr["start_frame"], fr["end_frame"]
                if rs <= kf["frame"] <= re_:
                    raise FramingError(
                        f"{REASON_MALFORMED}: {ctx} keyframe at frame "
                        f"{kf['frame']} inside fit range [{rs}, {re_}]")

        out.append({
            "scene_id": scene_id,
            "strategy": strategy,
            "start_frame": start,
            "end_frame": end,
            "applies": True,
            "scene_kind": kind,
            "subject_ids": subject_ids,
            "keyframes": keyframes,
            "fallback_reason": fallback_reason,
            "fallback_detail": fallback_detail,
            "fit": seg_fit if kind == "action" else False,
            "fit_ranges": fit_ranges if kind == "action" else [],
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
# Review-03 corrections:
#  1. Staging is FAIL-CLOSED. stage_previous_pair() backs up the current
#     accepted video AND stable sidecar and VERIFIES both backups
#     byte-identical before returning. Any staging/verification failure
#     raises OSError -- callers must abort publication and leave the
#     accepted pair untouched. Staging lives in a destination-local
#     directory (alongside the final video, or an explicitly supplied
#     staging_dir); the global process temp is never used. A staging
#     failure therefore can never silently produce an incomplete record.
#     restore_staged_pair() propagates failures too and discards recovery
#     data only after verified restoration.
#  2. Durable snapshots are keyed PER CLIP:
#     <job>/.task13_accepted_pair/<safe-clip-key>/, with an identity.txt
#     recording the exact clip basename. Restore refuses data that does
#     not belong to the requested clip -- two clips never share snapshot
#     files.
#  3. Snapshot updates are atomic generations. The new pair is written to
#     a destination-local temp staging dir, verified, installed as the
#     inactive generation, and published with ONE atomic pointer flip
#     (``current``: "gen-a" | "gen-b"). A failed update never touches the
#     live generation: restore afterwards sees either the prior coherent
#     pair or the complete new pair -- never a mix.
# - snapshot_accepted_pair(): after a SUCCESSFUL coordinated publication
#   the accepted pair is snapshotted durably (copy2 preserves mtime, so
#   the lineage file identity survives). Returns True only when the new
#   generation is durably installed; False (never raising, never claiming
#   durable recovery) when the update cannot be completed.
# - restore_accepted_snapshot(): brings the last accepted pair back,
#   transactionally as a video+sidecar unit (review-04): the active
#   generation is strictly validated, the current live pair is staged
#   before either artifact is installed, and any install failure rolls
#   the live pair back exactly. True means the complete snapshot pair
#   was installed; False leaves the live pair untouched. The pipeline
#   uses it when it DECLINES a rerender for a preserved Task 01 project,
#   so the preserved project's reframed_base keeps resolving to the
#   preserved final video.


def _verify_backup(src: str, dst: str) -> None:
    """Fail closed: the backup must be byte-identical to the source.

    Raises OSError when either side cannot be hashed or the digests
    differ.
    """
    src_hash = _sha256_file(src)
    dst_hash = _sha256_file(dst)
    if src_hash is None or dst_hash is None or src_hash != dst_hash:
        raise OSError(
            "task13 staging verification failed: backup of "
            f"{os.path.basename(src)} is not byte-identical")


def _remove_tree_quietly(path: str) -> None:
    """Best-effort directory removal; never raises."""
    import shutil

    try:
        shutil.rmtree(path, ignore_errors=True)
    except OSError:
        pass


def stage_previous_pair(*, clip_final: str, stable_sidecar: str,
                        staging_dir: str | None = None) -> dict:
    """Back up the current accepted pair before a new publication.

    FAIL-CLOSED: the complete previous video+sidecar pair is backed up
    and VERIFIED byte-identical before the record is returned. Any
    staging or verification failure raises OSError -- the caller must
    abort publication and leave the accepted pair untouched. A staging
    failure is never converted into an incomplete record.

    The staging directory is destination-local: ``<final-dir>/
    .task13_staging/``, or the explicitly supplied ``staging_dir``. The
    global process temp is never used, so staging works regardless of
    where TEMP/TMP/TMPDIR point.

    Returns an opaque record for :func:`restore_staged_pair` /
    :func:`discard_staged_pair`. ``copy2`` preserves mtimes.

    Raises:
        OSError: the staging directory could not be created or a backup
            could not be created/verified. The accepted pair itself is
            untouched; the partial staging directory is removed.
    """
    import shutil
    import tempfile

    if staging_dir is None:
        parent = os.path.dirname(os.path.abspath(clip_final)) or os.getcwd()
        staging_parent = os.path.join(parent, ".task13_staging")
    else:
        staging_parent = staging_dir
    os.makedirs(staging_parent, exist_ok=True)
    staging = tempfile.mkdtemp(prefix="pair-", dir=staging_parent)
    record: dict[str, Any] = {"video": None, "sidecar": None, "dir": staging}
    try:
        if os.path.isfile(clip_final):
            dst = os.path.join(
                staging, "video" + os.path.splitext(clip_final)[1])
            shutil.copy2(clip_final, dst)
            _verify_backup(clip_final, dst)
            record["video"] = dst
        if os.path.isfile(stable_sidecar):
            dst = os.path.join(staging, "sidecar.json")
            shutil.copy2(stable_sidecar, dst)
            _verify_backup(stable_sidecar, dst)
            record["sidecar"] = dst
    except OSError:
        # Fail closed: never hand out a partial or unverified record.
        # The accepted pair was not touched; drop the staging dir.
        _remove_tree_quietly(staging)
        record["video"] = None
        record["sidecar"] = None
        record["dir"] = None
        raise
    return record


def restore_staged_pair(record: dict, *, clip_final: str,
                        stable_sidecar: str) -> None:
    """Restore the staged pair after a publication/retirement failure.

    Both the video and the sidecar are put back and VERIFIED
    byte-identical; when no previous video was staged, the just-published
    (failed) video is removed instead. Recovery data is discarded only
    after verified restoration -- on any failure the staging directory
    is KEPT (for forensics/retry) and the OSError PROPAGATES, never
    swallowed: the caller must treat the attempt as failed, loudly.

    Raises:
        OSError: a staged file could not be restored or verified.
    """
    import shutil

    if record.get("video"):
        shutil.copy2(record["video"], clip_final)
        _verify_backup(record["video"], clip_final)
    else:
        if os.path.isfile(clip_final):
            os.remove(clip_final)
    if record.get("sidecar"):
        shutil.copy2(record["sidecar"], stable_sidecar)
        _verify_backup(record["sidecar"], stable_sidecar)
    # No sidecar backup: the failed publish left the stable sidecar in
    # its pre-call state (os.replace/os.remove are atomic), so there is
    # nothing to repair.
    discard_staged_pair(record)


def discard_staged_pair(record: dict) -> None:
    """Clean up staging after a successful publication. Never raises."""
    staging = record.get("dir")
    if staging:
        _remove_tree_quietly(staging)
    record["video"] = None
    record["sidecar"] = None
    record["dir"] = None


#: Durable snapshot of the last accepted video+sidecar pair, inside the
#: job dir. Restored when a rerender is declined for a preserved project.
#: Per-clip layout (review-03)::
#:
#:     .task13_accepted_pair/<safe-clip-key>/
#:         identity.txt          # exact clip_final basename (ownership)
#:         current               # "gen-a" | "gen-b" (atomic pointer)
#:         gen-a/ video<ext> sidecar.json manifest.json
#:         gen-b/ ...
ACCEPTED_SNAPSHOT_DIRNAME = ".task13_accepted_pair"

_SNAPSHOT_GENERATIONS = ("gen-a", "gen-b")
_SNAPSHOT_IDENTITY_FILENAME = "identity.txt"
_SNAPSHOT_POINTER_FILENAME = "current"
_SNAPSHOT_MANIFEST_FILENAME = "manifest.json"
_SNAPSHOT_SIDECAR_FILENAME = "sidecar.json"
_SNAPSHOT_MANIFEST_KEYS = frozenset(
    ("clip_basename", "video_sha256", "sidecar_sha256"))
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")


def _clip_snapshot_key(clip_final: str) -> str:
    """Safe per-clip snapshot key from the clip final basename.

    Alphanumerics plus ``-_.`` survive; every other character becomes
    ``_``, so the key can never traverse directories. Leading dots are
    stripped (``..`` can never be a key).
    """
    base = os.path.basename(clip_final)
    safe = "".join(
        c if (c.isalnum() or c in "-_.") else "_" for c in base)
    safe = safe.lstrip(".")
    if not safe:
        safe = "clip-" + hashlib.sha256(
            base.encode("utf-8", "replace")).hexdigest()[:16]
    return safe


def _clip_snapshot_dir(job_dir: str, clip_final: str) -> str:
    """Per-clip snapshot directory: two clips never share snapshot files."""
    return os.path.join(
        job_dir, ACCEPTED_SNAPSHOT_DIRNAME, _clip_snapshot_key(clip_final))


def _snapshot_identity_ok(clip_dir: str, clip_final: str) -> bool:
    """True when the snapshot directory belongs to the requested clip."""
    ident = os.path.join(clip_dir, _SNAPSHOT_IDENTITY_FILENAME)
    try:
        with open(ident, "r", encoding="utf-8") as fh:
            return fh.read().strip() == os.path.basename(clip_final)
    except OSError:
        return False


def _read_snapshot_current(clip_dir: str) -> str | None:
    """The active generation, or None when no coherent generation exists."""
    pointer = os.path.join(clip_dir, _SNAPSHOT_POINTER_FILENAME)
    try:
        with open(pointer, "r", encoding="utf-8") as fh:
            gen = fh.read().strip()
    except OSError:
        return None
    if gen not in _SNAPSHOT_GENERATIONS:
        return None
    if not os.path.isdir(os.path.join(clip_dir, gen)):
        return None
    return gen


def _snapshot_video_name(clip_final: str) -> str:
    return "video" + os.path.splitext(clip_final)[1]


def _snapshot_manifest(gen_dir: str) -> dict | None:
    path = os.path.join(gen_dir, _SNAPSHOT_MANIFEST_FILENAME)
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _atomic_write_text(path: str, text: str) -> None:
    """Write a small file atomically (destination-local temp + os.replace).

    Never touches the global process temp. Raises OSError on failure.
    """
    import tempfile

    parent = os.path.dirname(os.path.abspath(path)) or os.getcwd()
    fd, tmp = tempfile.mkstemp(prefix=".tmp-", dir=parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        os.replace(tmp, path)
    except OSError:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise


def _validate_snapshot_generation(*, clip_dir: str,
                                  clip_final: str) -> dict | None:
    """Strict validation of the active snapshot generation (review-04).

    One validator shared by :func:`accepted_snapshot_exists` and
    :func:`restore_accepted_snapshot`. Returns a dict
    ``{"gen_dir", "video_sha256", "sidecar_sha256"}`` when EVERYTHING checks
    out, else None:

    - the clip directory owns the requested clip (ownership marker),
    - the ``current`` pointer names a coherent generation directory,
    - ``manifest.json`` parses as a dict with exactly the production
      schema keys, ``clip_basename`` equal to the requested clip
      basename, ``video_sha256`` a 64-char lowercase hex digest, and
      ``sidecar_sha256`` either None or a 64-char lowercase hex digest,
    - the video snapshot exists and hashes to the manifest digest,
    - sidecar semantics: when the manifest records a sidecar digest, the
      sidecar snapshot exists and hashes to it; a sidecar-less acceptance
      records None and requires no sidecar snapshot.
    """
    if not _snapshot_identity_ok(clip_dir, clip_final):
        return None
    gen = _read_snapshot_current(clip_dir)
    if gen is None:
        return None
    gen_dir = os.path.join(clip_dir, gen)
    manifest = _snapshot_manifest(gen_dir)
    if not manifest:
        return None
    if set(manifest.keys()) != _SNAPSHOT_MANIFEST_KEYS:
        return None
    if manifest.get("clip_basename") != os.path.basename(clip_final):
        return None
    video_hash = manifest.get("video_sha256")
    if not isinstance(video_hash, str) or not _HEX64.match(video_hash):
        return None
    sidecar_hash = manifest.get("sidecar_sha256")
    if not (sidecar_hash is None
            or (isinstance(sidecar_hash, str)
                and _HEX64.match(sidecar_hash))):
        return None
    video_snap = os.path.join(gen_dir, _snapshot_video_name(clip_final))
    if (not os.path.isfile(video_snap)
            or _sha256_file(video_snap) != video_hash):
        return None
    if sidecar_hash is not None:
        sidecar_snap = os.path.join(gen_dir, _SNAPSHOT_SIDECAR_FILENAME)
        if (not os.path.isfile(sidecar_snap)
                or _sha256_file(sidecar_snap) != sidecar_hash):
            return None
    return {"gen_dir": gen_dir, "video_sha256": video_hash,
            "sidecar_sha256": sidecar_hash}


def accepted_snapshot_exists(*, job_dir: str, clip_final: str) -> bool:
    """True only when a fully VALIDATED snapshot generation is available.

    Uses the strict generation validator (review-04): ownership, pointer,
    manifest schema and clip identity, video existence/hash, and sidecar
    presence/hash semantics. A corrupt manifest can never report available.
    """
    clip_dir = _clip_snapshot_dir(job_dir, clip_final)
    return _validate_snapshot_generation(
        clip_dir=clip_dir, clip_final=clip_final) is not None


def snapshot_recovery_expected(*, job_dir: str, clip_final: str) -> bool:
    """True when an accepted pair was RECORDED for this clip (review-04).

    The snapshot directory carries this clip's ownership marker, so
    recovery was expected even if the stored generation is now invalid
    or missing. Callers use this to distinguish "no snapshot was ever
    recorded" (the pipeline may render normally; the project file itself
    stays preserved) from "snapshot data exists but cannot be recovered"
    (fail clearly -- never a rerender over the preserved base and never
    a success report).
    """
    clip_dir = _clip_snapshot_dir(job_dir, clip_final)
    return _snapshot_identity_ok(clip_dir, clip_final)


def snapshot_accepted_pair(*, job_dir: str, clip_final: str,
                           stable_sidecar: str) -> bool:
    """Persist the last accepted pair after successful publication.

    The new generation is staged in a destination-local temp dir,
    VERIFIED (sha256 of both files against the live pair), installed as
    the inactive generation, and published with ONE atomic pointer flip.
    A failed update never touches the live generation: a later restore
    sees either the prior coherent pair or the complete new pair --
    never a mixed generation.

    Returns True only when the new generation is durably installed.
    Returns False -- never raises, never claims durable recovery -- when
    the update cannot be completed; the previously installed generation,
    if any, is left untouched. Callers must handle False (the pair
    itself is accepted; only durable recovery is unavailable).

    ``copy2`` preserves mtimes so a later restore keeps the lineage
    file identity byte- and mtime-identical. A sidecar-less acceptance
    records no sidecar snapshot (any stale one is retired on the next
    install).
    """
    import shutil
    import tempfile

    clip_dir = _clip_snapshot_dir(job_dir, clip_final)
    try:
        os.makedirs(clip_dir, exist_ok=True)
    except OSError:
        return False
    ident_path = os.path.join(clip_dir, _SNAPSHOT_IDENTITY_FILENAME)
    if os.path.isfile(ident_path) and not _snapshot_identity_ok(
            clip_dir, clip_final):
        # Another clip owns this key: never clobber its recovery data.
        return False
    current = _read_snapshot_current(clip_dir)
    target = "gen-b" if current == "gen-a" else "gen-a"
    staging = None
    try:
        # Ownership first: this clip's identity is recorded before the
        # new generation becomes visible.
        _atomic_write_text(ident_path, os.path.basename(clip_final))
        staging = tempfile.mkdtemp(prefix="gen-", dir=clip_dir)
        video_name = _snapshot_video_name(clip_final)
        staged_video = os.path.join(staging, video_name)
        staged_sidecar = os.path.join(staging, _SNAPSHOT_SIDECAR_FILENAME)
        if not os.path.isfile(clip_final):
            raise OSError("snapshot source video vanished")
        shutil.copy2(clip_final, staged_video)
        sidecar_src = stable_sidecar if os.path.isfile(stable_sidecar) else None
        if sidecar_src:
            shutil.copy2(sidecar_src, staged_sidecar)
        # Verify the staged generation against the live pair BEFORE it
        # can become visible.
        video_hash = _sha256_file(staged_video)
        if video_hash is None or video_hash != _sha256_file(clip_final):
            raise OSError("snapshot video verification failed")
        sidecar_hash = None
        if sidecar_src:
            sidecar_hash = _sha256_file(staged_sidecar)
            if (sidecar_hash is None
                    or sidecar_hash != _sha256_file(stable_sidecar)):
                raise OSError("snapshot sidecar verification failed")
        with open(os.path.join(staging, _SNAPSHOT_MANIFEST_FILENAME),
                  "w", encoding="utf-8") as fh:
            json.dump({
                "clip_basename": os.path.basename(clip_final),
                "video_sha256": video_hash,
                "sidecar_sha256": sidecar_hash,
            }, fh, sort_keys=True)
        # Install as the inactive generation, then flip the pointer with
        # one atomic rename. The live generation is untouched until the
        # flip, so an exception anywhere above leaves it coherent.
        target_dir = os.path.join(clip_dir, target)
        _remove_tree_quietly(target_dir)
        os.rename(staging, target_dir)
        staging = None  # now owned by target_dir
        _atomic_write_text(
            os.path.join(clip_dir, _SNAPSHOT_POINTER_FILENAME), target)
        # Retire the superseded generation (best-effort; the next
        # snapshot reuses the slot anyway).
        if current and current != target:
            _remove_tree_quietly(os.path.join(clip_dir, current))
        return True
    except OSError:
        if staging:
            _remove_tree_quietly(staging)
        return False


def _install_verified(src: str, dst: str) -> None:
    """Install ``src`` at ``dst`` atomically after verifying byte identity.

    Copies to a destination-local temp file, verifies sha256, then
    os.replace()s into place. Raises OSError on any failure; ``dst`` is
    never left half-written. Never touches the global process temp.
    """
    import shutil
    import tempfile

    parent = os.path.dirname(os.path.abspath(dst)) or os.getcwd()
    fd, tmp = tempfile.mkstemp(prefix=".task13-restore-", dir=parent)
    os.close(fd)
    try:
        shutil.copy2(src, tmp)
        _verify_backup(src, tmp)
        os.replace(tmp, dst)
    except OSError:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise


def _stage_live_pair(*, clip_final: str, stable_sidecar: str) -> dict | None:
    """Stage the CURRENT live pair before a snapshot install (review-04).

    Destination-local, verified ``copy2`` backups of whatever is live;
    each file is recorded with its existence flag and, when it existed,
    its exact pre-install sha256 (ground truth for post-rollback
    verification). Returns an opaque record for
    :func:`_restore_live_pair`, or None when staging cannot be completed
    -- fail-closed, the caller must not touch the live pair. Never
    touches the global process temp.
    """
    import shutil
    import tempfile

    parent = os.path.dirname(os.path.abspath(clip_final)) or os.getcwd()
    staging = None
    try:
        staging = tempfile.mkdtemp(prefix=".task13-live-backup-", dir=parent)
        paths = []
        for dst in (clip_final, stable_sidecar):
            entry = {"dst": dst, "existed": os.path.isfile(dst),
                     "backup": None, "sha256": None}
            if entry["existed"]:
                backup = os.path.join(
                    staging, os.path.basename(dst) + ".bak")
                shutil.copy2(dst, backup)
                _verify_backup(dst, backup)
                pre_hash = _sha256_file(dst)
                if pre_hash is None:
                    raise OSError(
                        "task13 staging failed: could not hash "
                        f"{os.path.basename(dst)}")
                entry["backup"] = backup
                entry["sha256"] = pre_hash
            paths.append(entry)
        return {"dir": staging, "paths": paths}
    except OSError:
        if staging:
            _remove_tree_quietly(staging)
        return None


def _install_direct(src: str, dst: str) -> None:
    """Copy bytes directly to ``dst`` (no atomic rename).

    Used only for best-effort rollback when the atomic install path is
    broken. Raises OSError on any failure.
    """
    import shutil

    shutil.copy2(src, dst)
    _verify_backup(src, dst)


def _restore_live_pair(record: dict) -> bool:
    """Restore the staged live pair; verified success or distinct failure.

    Each file that existed is reinstalled (atomically when possible,
    direct copy as fallback); each file that did not exist is removed
    when present. Snapshot (recovery) data is never touched.

    After the attempt, BOTH live paths are re-verified against the exact
    pre-install existence flag and sha256 recorded by
    :func:`_stage_live_pair`. When both verify, the staging directory is
    discarded and True is returned. When anything does not verify -- a
    restore error or a post-restore existence/hash mismatch -- the
    staging directory is RETAINED (never deleted) and
    :class:`FramingIntegrityError` is raised, so the failure can never be
    collapsed into ordinary False. Never returns False.
    """
    restore_errors: list[str] = []
    for entry in record["paths"]:
        dst = entry["dst"]
        try:
            if entry["existed"]:
                try:
                    _install_verified(entry["backup"], dst)
                except OSError:
                    _install_direct(entry["backup"], dst)
            elif os.path.isfile(dst) or os.path.islink(dst):
                os.remove(dst)
        except OSError as exc:
            restore_errors.append(f"{dst}: {exc}")
    verify_errors: list[str] = []
    for entry in record["paths"]:
        dst = entry["dst"]
        if entry["existed"]:
            actual = _sha256_file(dst)
            if actual is None:
                verify_errors.append(f"{dst}: missing after rollback")
            elif actual != entry.get("sha256"):
                verify_errors.append(f"{dst}: hash mismatch after rollback")
        elif os.path.isfile(dst) or os.path.islink(dst):
            verify_errors.append(f"{dst}: present after rollback")
    if restore_errors or verify_errors:
        raise FramingIntegrityError(
            "live-pair rollback failed integrity verification; verified "
            f"staging backups retained at {record['dir']}: "
            + "; ".join(restore_errors + verify_errors))
    _remove_tree_quietly(record["dir"])
    return True


def restore_accepted_snapshot(*, job_dir: str, clip_final: str,
                              stable_sidecar: str) -> bool:
    """Restore the last accepted pair from the durable snapshot, transactionally.

    The active generation is fully validated first (the same strict
    validator as :func:`accepted_snapshot_exists`). The current live pair
    is staged before either snapshot artifact is installed; if either
    install fails, the prior live state is restored as exactly as
    possible and the snapshot (recovery data) is left untouched.

    Returns True only when the complete snapshot pair was installed.
    Returns False when the generation is unavailable/invalid (no live
    mutation occurred) or when an install failed and the prior live pair
    was rolled back and re-verified byte-identical; False leaves the
    live pair exactly as it was before the call, never a mixed pair.
    When the rollback itself cannot be verified,
    :class:`FramingIntegrityError` is raised distinctly instead of
    returning False: the live pair is indeterminate and the verified
    staging backups are retained for manual recovery.
    """
    clip_dir = _clip_snapshot_dir(job_dir, clip_final)
    validated = _validate_snapshot_generation(
        clip_dir=clip_dir, clip_final=clip_final)
    if validated is None:
        return False
    gen_dir = validated["gen_dir"]
    expected_sidecar = validated["sidecar_sha256"]
    video_snap = os.path.join(gen_dir, _snapshot_video_name(clip_final))
    sidecar_snap = os.path.join(gen_dir, _SNAPSHOT_SIDECAR_FILENAME)
    live = _stage_live_pair(clip_final=clip_final,
                            stable_sidecar=stable_sidecar)
    if live is None:
        return False
    try:
        _install_verified(video_snap, clip_final)
        if expected_sidecar:
            _install_verified(sidecar_snap, stable_sidecar)
        elif os.path.isfile(stable_sidecar):
            # Sidecar-less acceptance: the stable binding must not linger.
            os.remove(stable_sidecar)
    except OSError:
        # Review-05: _restore_live_pair returns verified success only,
        # otherwise raises FramingIntegrityError. Verified rollback ->
        # False (the documented byte-identical case); the integrity
        # error propagates distinctly and is never converted to False.
        _restore_live_pair(live)
        return False
    _remove_tree_quietly(live["dir"])
    return True
