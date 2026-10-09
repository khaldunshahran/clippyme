"""Evidence-backed action-aware framing plans for automatic GENERAL scenes.

Pure, host-testable core of Task 14. It does NOT detect objects, decode
video, or call any provider: it consumes validated action observations
(absolute frame/time, scene id, anonymous local track id, finite in-bounds
box, bounded detector confidence, normalized category family, box motion,
local saliency support, visibility, provenance/reason) and produces
deterministic, versioned, auditable *action segments* — ordered
non-overlapping scene-local segments with fixed crop targets — which
``framing_plan.build_framing_plan`` persists as explicit
``scene_kind="action"`` scenes of the v4 turn-framing plan, rendered
verbatim by the two-pass renderer in ``reframe.py``.

Design rules (see docs/action-aware-framing.md for the full contract):

* Observations are strictly validated. Booleans, non-finite values, bad
  boxes, duplicate ``(frame, track_id)`` ids, regressing times, unsupported
  categories and cross-scene identity carryover are rejected with an
  ``ActionObservationRejected`` code. Track ids are opaque (``t-<int>``).
  No pixels, embeddings, names, OCR text or provider payloads are stored.
* A tight subject crop requires *sustained* detector evidence
  (``confidence * visibility`` over ``evidence_duration_s``) plus a relative
  ``switch_margin`` over the incumbent, after ``min_dwell_s`` and
  ``cooldown_s`` have elapsed. **Saliency or raw motion alone may bias a
  wide crop slightly but can never authorize a tight subject crop.**
* **Size alone never selects**: acquisition and switching rank tracks by
  sustained evidence, never by box area — a large irrelevant object cannot
  steal the crop from sustained action evidence.
* Multiple interacting subjects (top-two evidence within ``group_ratio``)
  share one group crop containing their union when it fits the output
  aspect; otherwise the scene widens. Fast or uncertain motion, boundary
  pressure, large camera motion, or evidence loss widen or fit the frame.
  No largest-detection chasing, no unrelated-object jumping, no zoom
  breathing, no oscillation.
* Tracks and decisions reset at hard cuts (a fresh ``ActionTracker`` per
  scene; the log rejects cross-scene identity carryover). The crop is fixed
  within each stable segment; only a short bounded ease interpolates
  between accepted decisions, with travel limited to ``max_travel_frac``
  of the frame dimension PER SECOND (the transition extends over as many
  frames as the speed cap needs, or the previous lock is held), zoom held
  unless widening, crops clamped in-bounds with padding and look-ahead
  room, and participants never cropped away.
* The single monotonic framing ``AnalysisBudget`` (``framing_plan``) is
  shared: decode, detector work, motion/saliency analysis and planning are
  charged before work starts. On exhaustion the plan is marked
  ``analysis_complete=false``, the first dropped frame/scene is recorded,
  no reusable partial plan is produced, and the accepted GENERAL fallback
  renders instead.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
from dataclasses import dataclass, field

from clippyme.pipeline.framing_plan import (
    ACTION_EVIDENCE_SPEC,
    ACTION_SCENE_KIND,
    ACTION_STRATEGIES,
    TRACK_ID_RE,
    CropKeyframe,
    FramingSegment,
)
from clippyme.pipeline.reframe_ops import saliency_biased_wide_crop


# Normalized category families. The detection seam maps provider class names
# onto this fixed vocabulary; anything else is rejected by the observation
# contract ("unsupported-category") so no free-form label can enter a plan.
ACTION_CATEGORY_FAMILIES = ("person", "animal", "vehicle", "object")

ACTION_PROVENANCE_VALUES = ("detector", "tracker", "injected-test",
                            "smoother-hold")
ACTION_REASON_MAX_LEN = 64

# Segment kinds for action scenes.
ACTION_SEGMENT_KINDS = ("single", "group", "wide")


class ActionObservationRejected(ValueError):
    """Raised when an action observation fails strict validation.

    The message is prefixed with a machine-readable code
    (``duplicate-observation``, ``time-regression``, ...).
    """


class ActionPlanError(Exception):
    """Raised when a GENERAL scene cannot be action-planned.

    Carries the machine-readable ``reason`` the caller stores as the
    scene's explicit fallback reason (never a silent default).
    """

    def __init__(self, reason: str, detail: str = ""):
        super().__init__(f"{reason}: {detail}")
        self.reason = reason
        self.detail = detail


@dataclass(frozen=True)
class ActionObservation:
    """One validated action observation.

    ``frame``/``time_s`` are absolute (source) coordinates, ``scene_id`` the
    scene index, ``track_id`` an opaque anonymous id (``t-<int>``), ``box`` a
    finite in-bounds ``(x, y, w, h)``, ``confidence`` the bounded detector
    confidence, ``category`` a normalized family, ``motion`` the per-track
    box motion in [0, 1], ``saliency`` the local saliency support in
    [0, 1], ``visibility`` in [0, 1], ``provenance`` one of
    ``ACTION_PROVENANCE_VALUES``, and ``reason`` a short machine-readable
    token.
    """

    frame: int
    time_s: float
    scene_id: int
    track_id: str
    box: tuple
    confidence: float
    category: str
    motion: float
    saliency: float
    visibility: float
    provenance: str
    reason: str


def _is_num(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


# --- flag + config ----------------------------------------------------------


def action_aware_enabled() -> bool:
    """``CLIPPYME_ACTION_AWARE_FRAMING`` — enabled by default on this branch.

    Only the explicit negatives ``0``/``false``/``no``/``off`` restore the
    accepted Task 13 GENERAL behavior (per-frame object/saliency helper or
    full-frame fallback, no persisted action plan). Unset, empty, or
    anything else means on. Documented in docs/action-aware-framing.md.
    """
    val = (os.getenv("CLIPPYME_ACTION_AWARE_FRAMING") or "").strip().lower()
    if not val:
        return True
    return val not in ("0", "false", "no", "off")


def _env_float(name: str, default: float, lo: float, hi: float) -> float:
    raw = (os.getenv(name) or "").strip()
    try:
        v = float(raw)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(v):
        return default
    return min(max(v, lo), hi)


def _env_int(name: str, default: int, lo: int) -> int:
    raw = (os.getenv(name) or "").strip()
    try:
        v = int(raw)
    except (TypeError, ValueError):
        return default
    return max(v, lo)


@dataclass(frozen=True)
class ActionFramingConfig:
    """Tunable action-segmentation / crop policy. All durations in seconds."""

    evidence_duration_s: float = 0.8   # sustained evidence window to acquire
    min_dwell_s: float = 1.0           # a decision must hold this long
    cooldown_s: float = 1.5            # no new switch within this of the last
    switch_margin: float = 1.25        # challenger evidence must beat
                                       # incumbent * margin
    merge_window_s: float = 0.5        # sub-window oscillation blips merged
    occlusion_hold_s: float = 1.0      # short occlusion holds the framing
    ease_frames: int = 6               # short bounded ease between decisions
    max_travel_frac: float = 0.5       # per-second cap on crop-centre travel
                                       # (fraction of the frame dimension / s)
    padding_frac: float = 0.30         # padding around the subject union box
    look_ahead_frac: float = 0.15      # lead room along the motion vector,
                                       # fraction of the crop dimension
    headroom_frac: float = 0.10        # headroom above the subject union
    action_evidence_threshold: float = 0.5  # sustained evidence floor
    target_subject_occupancy: float = 0.5   # subject height / crop height
    min_zoom: float = 1.0
    max_zoom: float = 2.0
    group_ratio: float = 0.88          # two-subject ambiguity threshold
    group_min_separation_frac: float = 0.05  # min centre separation
    fast_motion_widen: float = 0.55    # sustained box-motion above this widens
    fast_travel_frac: float = 0.15     # subject centre travel (fraction of
                                       # frame dimension) over the evidence
                                       # window above this widens: a static
                                       # lock cannot hold a fast crossing
    uncertain_motion_range: float = 0.75  # motion range above this widens
    camera_motion_widen: float = 0.35  # sustained camera-motion above widens
    boundary_pressure_frac: float = 0.5  # padding lost to clamping above
                                         # this widens instead
    saliency_bias_frac: float = 0.05   # max wide-crop bias toward saliency,
                                       # fraction of frame width (a wide crop
                                       # is NEVER tightened by saliency)
    sample_every: int = 2              # pass-1 analyzes every Nth frame

    def to_dict(self) -> dict:
        return {
            "evidence_duration_s": self.evidence_duration_s,
            "min_dwell_s": self.min_dwell_s,
            "cooldown_s": self.cooldown_s,
            "switch_margin": self.switch_margin,
            "merge_window_s": self.merge_window_s,
            "occlusion_hold_s": self.occlusion_hold_s,
            "ease_frames": self.ease_frames,
            "max_travel_frac": self.max_travel_frac,
            "padding_frac": self.padding_frac,
            "look_ahead_frac": self.look_ahead_frac,
            "headroom_frac": self.headroom_frac,
            "action_evidence_threshold": self.action_evidence_threshold,
            "target_subject_occupancy": self.target_subject_occupancy,
            "min_zoom": self.min_zoom,
            "max_zoom": self.max_zoom,
            "group_ratio": self.group_ratio,
            "group_min_separation_frac": self.group_min_separation_frac,
            "fast_motion_widen": self.fast_motion_widen,
            "fast_travel_frac": self.fast_travel_frac,
            "uncertain_motion_range": self.uncertain_motion_range,
            "camera_motion_widen": self.camera_motion_widen,
            "boundary_pressure_frac": self.boundary_pressure_frac,
            "saliency_bias_frac": self.saliency_bias_frac,
            "sample_every": self.sample_every,
        }

    def config_digest(self) -> str:
        """Stable identity of the config for plan reuse matching."""
        payload = json.dumps(self.to_dict(), sort_keys=True,
                             separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def action_config_from_env() -> ActionFramingConfig:
    """Build the config from ``CLIPPYME_ACTION_*`` overrides; garbage → default."""
    return ActionFramingConfig(
        evidence_duration_s=_env_float("CLIPPYME_ACTION_EVIDENCE_S", 0.8,
                                       0.05, 10.0),
        min_dwell_s=_env_float("CLIPPYME_ACTION_MIN_DWELL_S", 1.0, 0.0, 30.0),
        cooldown_s=_env_float("CLIPPYME_ACTION_COOLDOWN_S", 1.5, 0.0, 30.0),
        switch_margin=_env_float("CLIPPYME_ACTION_SWITCH_MARGIN", 1.25,
                                 1.0, 10.0),
        merge_window_s=_env_float("CLIPPYME_ACTION_MERGE_S", 0.5, 0.0, 10.0),
        occlusion_hold_s=_env_float("CLIPPYME_ACTION_OCCLUSION_S", 1.0,
                                    0.0, 10.0),
        ease_frames=_env_int("CLIPPYME_ACTION_EASE_FRAMES", 6, 0),
        padding_frac=_env_float("CLIPPYME_ACTION_PADDING", 0.30, 0.0, 2.0),
        look_ahead_frac=_env_float("CLIPPYME_ACTION_LOOK_AHEAD", 0.15,
                                   0.0, 1.0),
        action_evidence_threshold=_env_float(
            "CLIPPYME_ACTION_EVIDENCE_THRESHOLD", 0.5, 0.0, 1.0),
        max_travel_frac=_env_float("CLIPPYME_ACTION_MAX_TRAVEL", 0.5,
                                   0.01, 4.0),
        sample_every=_env_int("CLIPPYME_ACTION_SAMPLE_EVERY", 2, 1),
    )


def plan_config_digest(turn_digest: str,
                       action_cfg: ActionFramingConfig | None,
                       turn_on: bool = True) -> str:
    """Combined plan-build config identity for v4 action plans.

    Binds the turn config digest, the action config digest, and the
    face-planning flag, so any config or routing change invalidates plan
    reuse (``stale-config``). With no action config this is the turn
    digest unchanged (the Task 13 v3 path).
    """
    if action_cfg is None:
        return turn_digest
    h = hashlib.sha256()
    h.update(b"turn:")
    h.update(turn_digest.encode("utf-8"))
    h.update(b"|action:")
    h.update(action_cfg.config_digest().encode("utf-8"))
    h.update(b"|turn_on:1" if turn_on else b"|turn_on:0")
    return h.hexdigest()


# --- observation log --------------------------------------------------------


class ActionObservationLog:
    """Validating, order-preserving store of action observations.

    Rejects booleans, non-finite values, bad boxes, unsupported categories,
    duplicate ``(frame, track_id)`` ids, regressing times and cross-scene
    identity carryover. Rejections are counted in ``diagnostics`` by the
    caller; the log itself raises ``ActionObservationRejected``.
    """

    def __init__(self, frame_w: float, frame_h: float,
                 max_observations: int = 20000):
        self.frame_w = float(frame_w)
        self.frame_h = float(frame_h)
        self.max_observations = max(1, int(max_observations))
        self._obs: list[ActionObservation] = []
        self._seen: set[tuple[int, str]] = set()
        self._last_time: dict[str, float] = {}
        self._track_scene: dict[str, int] = {}
        self.diagnostics: list[str] = []
        self.dropped = 0
        # Structured budget-exhaustion state (mirrors Task 12): set once
        # via mark_exhaustion() when pass-1 analysis stops.
        self.exhaustion_reason: str | None = None
        self.exhaustion_frame: int | None = None
        self.exhaustion_scene: int | None = None

    def add(self, obs: ActionObservation) -> None:
        self._validate(obs)
        if len(self._obs) >= self.max_observations:
            self.dropped += 1
            if self.dropped == 1:
                self.mark_exhaustion("analysis-frame-budget-exhausted",
                                     frame=obs.frame, scene_id=obs.scene_id)
            return
        self._obs.append(obs)
        self._seen.add((obs.frame, obs.track_id))
        self._last_time[obs.track_id] = obs.time_s
        self._track_scene.setdefault(obs.track_id, obs.scene_id)

    def note(self, message: str) -> None:
        self.diagnostics.append(message)

    def mark_exhaustion(self, reason: str, frame: int | None = None,
                        scene_id: int | None = None) -> None:
        """Record budget exhaustion with structured frame/scene context.

        First call wins: pass 1 latches analysis off exactly once.
        """
        if self.exhaustion_reason is not None:
            return
        self.exhaustion_reason = str(reason)
        self.exhaustion_frame = frame
        self.exhaustion_scene = scene_id
        if reason not in self.diagnostics:
            self.diagnostics.append(reason)

    def observations_for_scene(self, scene_id: int) -> list[ActionObservation]:
        out = [o for o in self._obs if o.scene_id == scene_id]
        out.sort(key=lambda o: (o.frame, o.track_id))
        return out

    def track_summary(self, scene_id: int) -> list[dict]:
        first: dict[str, int] = {}
        last: dict[str, int] = {}
        count: dict[str, int] = {}
        for o in self._obs:
            if o.scene_id != scene_id:
                continue
            tid = o.track_id
            first.setdefault(tid, o.frame)
            last[tid] = o.frame
            count[tid] = count.get(tid, 0) + 1
        return [
            {"id": tid, "first_frame": first[tid], "last_frame": last[tid],
             "n_obs": count[tid]}
            for tid in sorted(first)
        ]

    # -- validation ------------------------------------------------------
    def _validate(self, obs: ActionObservation) -> None:
        def bad(code: str, detail: str = "") -> ActionObservationRejected:
            return ActionObservationRejected(
                f"{code}{(': ' + detail) if detail else ''}")

        if isinstance(obs.frame, bool) or not isinstance(obs.frame, int) \
                or obs.frame < 0:
            raise bad("bad-frame", repr(obs.frame))
        if (not _is_num(obs.time_s) or not math.isfinite(obs.time_s)
                or obs.time_s < 0):
            raise bad("bad-time", repr(obs.time_s))
        if (isinstance(obs.scene_id, bool) or not isinstance(obs.scene_id, int)
                or obs.scene_id < 0):
            raise bad("bad-scene-id", repr(obs.scene_id))
        if not isinstance(obs.track_id, str) \
                or not TRACK_ID_RE.match(obs.track_id):
            raise bad("bad-track-id", repr(obs.track_id))
        box = obs.box
        if (not isinstance(box, (tuple, list)) or len(box) != 4
                or not all(_is_num(v) and math.isfinite(v) for v in box)):
            raise bad("bad-box", repr(box))
        x, y, w, h = (float(v) for v in box)
        if w <= 0 or h <= 0 or x < 0 or y < 0:
            raise bad("bad-box", repr(box))
        if x + w > self.frame_w + 1e-6 or y + h > self.frame_h + 1e-6:
            raise bad("box-out-of-bounds", repr(box))
        if (not _is_num(obs.confidence) or not math.isfinite(obs.confidence)
                or not 0.0 <= obs.confidence <= 1.0):
            raise bad("bad-confidence", repr(obs.confidence))
        if obs.category not in ACTION_CATEGORY_FAMILIES:
            # Normalized families only: a free-form provider label can
            # never enter the plan.
            raise bad("unsupported-category", repr(obs.category))
        for attr, code in (("motion", "bad-motion"),
                           ("saliency", "bad-saliency"),
                           ("visibility", "bad-visibility")):
            v = getattr(obs, attr)
            if not _is_num(v) or not math.isfinite(v) \
                    or not 0.0 <= v <= 1.0:
                raise bad(code, repr(v))
        if obs.provenance not in ACTION_PROVENANCE_VALUES:
            raise bad("bad-provenance", repr(obs.provenance))
        if (not isinstance(obs.reason, str) or not obs.reason
                or len(obs.reason) > ACTION_REASON_MAX_LEN):
            raise bad("bad-reason", repr(obs.reason))
        if (obs.frame, obs.track_id) in self._seen:
            raise bad("duplicate-observation",
                      f"frame={obs.frame} track={obs.track_id}")
        prev_t = self._last_time.get(obs.track_id)
        if prev_t is not None and obs.time_s < prev_t - 1e-9:
            raise bad("time-regression",
                      f"track={obs.track_id} {prev_t} -> {obs.time_s}")
        prev_scene = self._track_scene.get(obs.track_id)
        if prev_scene is not None and prev_scene != obs.scene_id:
            raise bad("cross-scene-identity",
                      f"track={obs.track_id} scenes {prev_scene} -> "
                      f"{obs.scene_id}")


def validate_action_detection(det, frame_w, frame_h):
    """Validate one raw detector item before saliency/tracking.

    Returns a normalized ``{"box", "confidence", "category"}`` dict, or
    ``None`` when the item is malformed. Never raises: a malformed box
    must never reach ``ActionTracker.update()`` (or the saliency
    measurement) and abort pass 1. The caller drops malformed items
    individually and records a bounded scene fallback when nothing
    usable remains.

    Category families are NOT filtered here; the observation contract
    still rejects unsupported categories per observation (isolated).
    """
    if not isinstance(det, dict):
        return None
    box = det.get("box")
    if not isinstance(box, (tuple, list)) or len(box) != 4:
        return None
    if any(isinstance(v, bool) for v in box):
        # Booleans are never valid numerics (observation contract).
        return None
    try:
        x, y, w, h = (float(v) for v in box)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(v) for v in (x, y, w, h)):
        return None
    if w <= 0 or h <= 0 or x < 0 or y < 0:
        return None
    if x + w > frame_w + 1e-6 or y + h > frame_h + 1e-6:
        return None
    confidence_raw = det.get("confidence", 0.0)
    if isinstance(confidence_raw, bool):
        return None
    try:
        confidence = float(confidence_raw)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(confidence):
        return None
    category = det.get("category", "object")
    if not isinstance(category, str) or not category:
        return None
    return {"box": (x, y, w, h),
            "confidence": min(1.0, max(0.0, confidence)),
            "category": category}


# --- pure local tracker -----------------------------------------------------


class ActionTracker:
    """Greedy IoU track assignment for action detections (pure, no cv2).

    Assigns opaque ``t-<int>`` ids to detector boxes and reports
    per-track box motion (``1 - IoU`` against the track's previous box,
    in [0, 1]; 0.0 for a brand-new track). Ids are monotonic for the
    tracker's lifetime and NEVER reused: ``reset()`` drops the live
    tracks at hard cuts but keeps the counter, so the observation log's
    cross-scene identity guard can never collide with a legitimately
    new subject. No identity is carried across a cut.
    """

    def __init__(self, iou_match_thresh: float = 0.3):
        self.iou_match_thresh = float(iou_match_thresh)
        self._next = 0
        self._tracks: dict[str, tuple] = {}

    def reset(self) -> None:
        # Drop live tracks (no IoU continuity across the cut) but keep
        # the id counter monotonic: reusing t-0 for a new subject would
        # trip the log's cross-scene-identity rejection.
        self._tracks = {}

    @staticmethod
    def _iou(a: tuple, b: tuple) -> float:
        ax1, ay1, aw, ah = a
        bx1, by1, bw, bh = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax1 + aw, bx1 + bw), min(ay1 + ah, by1 + bh)
        iw, ih = ix2 - ix1, iy2 - iy1
        if iw <= 0 or ih <= 0:
            return 0.0
        inter = iw * ih
        union = aw * ah + bw * bh - inter
        return inter / union if union > 0 else 0.0

    def update(self, frame: int, detections: list[dict]) -> list[dict]:
        """Match detections to tracks; return matches in input order.

        Each detection is ``{"box": (x, y, w, h), "confidence": float,
        "category": str}``. Each returned match is ``{"track_id", "box",
        "confidence", "category", "motion"}``.
        """
        n = len(detections)
        assigned: list[dict | None] = [None] * n
        # Greedy by confidence: strongest detections claim tracks first,
        # so a flickering weak box can never steal an established id.
        order = sorted(range(n),
                       key=lambda i: float(detections[i].get("confidence",
                                                             0.0)),
                       reverse=True)
        free = set(self._tracks)
        for i in order:
            det = detections[i]
            box = tuple(float(v) for v in det["box"])
            best_tid, best_iou = None, self.iou_match_thresh
            for tid in free:
                iou = self._iou(box, self._tracks[tid])
                if iou >= best_iou:
                    best_iou, best_tid = iou, tid
            if best_tid is not None:
                free.discard(best_tid)
                motion = min(1.0, max(0.0, 1.0 - best_iou))
                tid = best_tid
            else:
                tid = f"t-{self._next}"
                self._next += 1
                motion = 0.0
            self._tracks[tid] = box
            assigned[i] = {
                "track_id": tid,
                "box": box,
                "confidence": float(det.get("confidence", 0.0)),
                "category": str(det.get("category", "object")),
                "motion": motion,
            }
        return [a for a in assigned if a is not None]


# --- action segmentation ----------------------------------------------------


def _clamp01(v: float) -> float:
    return min(1.0, max(0.0, v))


def segment_actions(log: ActionObservationLog, *, scene_id: int,
                    scene_start: int, scene_end: int,
                    config: ActionFramingConfig, fps: float,
                    frame_w: float, frame_h: float,
                    aspect: float = 9 / 16,
                    camera_motion: dict | None = None):
    """Segment one GENERAL scene's observations into ordered non-overlapping
    action segments.

    ``camera_motion`` maps analyzed frame numbers to a [0, 1] global-motion
    score (None/empty disables the camera-motion widen rule).

    Returns ``(segments, diagnostics)`` with ``FramingSegment`` entries
    (``kind`` in ``("single", "group", "wide")``). Raises
    ``ActionPlanError`` when the scene has no usable observations — the
    caller records the reason as an explicit fallback and keeps the
    accepted GENERAL behavior for that scene.
    """
    obs = [o for o in log.observations_for_scene(scene_id)
           if scene_start <= o.frame <= scene_end]
    if not obs:
        raise ActionPlanError(
            "no-observations",
            f"scene {scene_id}: no usable action observations")
    if fps <= 0:
        raise ActionPlanError("bad-fps", f"scene {scene_id}: fps={fps}")

    frames = sorted({o.frame for o in obs})
    by_frame: dict[int, dict[str, ActionObservation]] = {}
    for o in obs:
        by_frame.setdefault(o.frame, {})[o.track_id] = o
    # Iterate the FULL scene frame range, not just observed frames: an
    # empty stretch is evidence loss (occlusion / subject exit) and must
    # drive the hold/widen logic. Times for unobserved frames interpolate
    # from the first observation's clock.
    _anchor_frame = frames[0]
    _anchor_t = by_frame[_anchor_frame][
        next(iter(by_frame[_anchor_frame]))].time_s

    def t_of(f: int) -> float:
        fobs = by_frame.get(f)
        if fobs:
            return fobs[next(iter(fobs))].time_s
        return _anchor_t + (f - _anchor_frame) / fps
    track_obs: dict[str, list[ActionObservation]] = {}
    for o in obs:
        track_obs.setdefault(o.track_id, []).append(o)

    def evidence_vals(tid: str, t: float) -> list[float]:
        # Sustained detector evidence: confidence * visibility. Saliency
        # and motion are recorded as support signals but NEVER count as
        # evidence here — saliency alone must not authorize a tight crop.
        return [o.confidence * o.visibility for o in track_obs[tid]
                if t - config.evidence_duration_s - 1e-9 <= o.time_s
                <= t + 1e-9]

    def sustained(tid: str, t: float) -> float:
        vals = evidence_vals(tid, t)
        return sum(vals) / len(vals) if vals else 0.0

    def motion_stats(tid: str, t: float) -> tuple[float, float]:
        vals = [o.motion for o in track_obs[tid]
                if t - config.evidence_duration_s - 1e-9 <= o.time_s
                <= t + 1e-9]
        if not vals:
            return 0.0, 0.0
        return sum(vals) / len(vals), max(vals) - min(vals)

    def travel_speed(tid: str, t: float) -> float:
        # Centre displacement over the evidence window, as a fraction of
        # the frame dimension (framerate-independent physical speed).
        # A static lock cannot hold a subject that crosses a large part
        # of the frame during the window.
        pts = [(o.time_s, o.box[0] + o.box[2] / 2.0,
                o.box[1] + o.box[3] / 2.0)
               for o in track_obs[tid]
               if t - config.evidence_duration_s - 1e-9 <= o.time_s
               <= t + 1e-9]
        if len(pts) < 2:
            return 0.0
        pts.sort()
        (_, x0, y0), (_, x1, y1) = pts[0], pts[-1]
        return max(abs(x1 - x0) / frame_w, abs(y1 - y0) / frame_h)

    def motion_widen_reason(tid: str, t: float) -> str | None:
        # Acquisition gate: a subject whose motion would immediately
        # force a widen must never be acquired as a tight single/group
        # (it would oscillate single/wide every frame).
        mmean, mrange = motion_stats(tid, t)
        if mmean > config.fast_motion_widen:
            return "fast-motion-widen"
        if travel_speed(tid, t) > config.fast_travel_frac:
            return "fast-travel-widen"
        if mrange > config.uncertain_motion_range:
            return "uncertain-motion-widen"
        return None

    def saliency_mean(tid: str, t: float) -> float:
        vals = [o.saliency for o in track_obs[tid]
                if t - config.evidence_duration_s - 1e-9 <= o.time_s
                <= t + 1e-9]
        return sum(vals) / len(vals) if vals else 0.0

    def cam_sustained(t: float) -> float:
        if not camera_motion:
            return 0.0
        vals = [s for f, s in camera_motion.items()
                if t - config.evidence_duration_s - 1e-9 <= f / fps
                <= t + 1e-9]
        return sum(vals) / len(vals) if vals else 0.0

    def _centres(fobs: dict[str, ActionObservation], tid: str):
        b = fobs[tid].box
        return b[0] + b[2] / 2.0, b[1] + b[3] / 2.0

    def try_group(cands: dict[str, float],
                  fobs: dict[str, ActionObservation]):
        """Two strong, close, separable subjects sharing one crop.

        Returns the pair ``(a, b)`` or ``None``. Size alone never
        qualifies — both tracks need sustained detector evidence within
        ``group_ratio`` of each other, and the padded union must fit the
        widest available crop (the keyframe builder re-checks precisely;
        this is the fast path).
        """
        if len(cands) < 2:
            return None
        top = sorted(cands.items(), key=lambda kv: kv[1], reverse=True)[:2]
        (a, sa), (b, sb) = top
        if sb < config.action_evidence_threshold or sa <= 0:
            return None
        if sb / sa < config.group_ratio:
            return None
        cxa, _ = _centres(fobs, a)
        cxb, _ = _centres(fobs, b)
        if abs(cxa - cxb) / frame_w < config.group_min_separation_frac:
            return None
        ux1 = min(fobs[a].box[0], fobs[b].box[0])
        ux2 = max(fobs[a].box[0] + fobs[a].box[2],
                  fobs[b].box[0] + fobs[b].box[2])
        uy1 = min(fobs[a].box[1], fobs[b].box[1])
        uy2 = max(fobs[a].box[1] + fobs[a].box[3],
                  fobs[b].box[1] + fobs[b].box[3])
        pad = 1.0 + config.padding_frac
        crop_h = frame_h / config.min_zoom
        crop_w = crop_h * aspect
        if crop_w > frame_w:
            crop_w = float(frame_w)
            crop_h = crop_w / aspect
        if (ux2 - ux1) * pad > crop_w + 1e-6 \
                or (uy2 - uy1) * pad > crop_h + 1e-6:
            return None
        return (a, b)

    def pair_close(cands, fobs):
        """True when the top two candidates are ambiguous-close (evidence +
        separation) REGARDLESS of whether they fit one crop. Used to hold
        wide when a group crop is geometrically impossible."""
        if len(cands) < 2:
            return False
        top = sorted(cands.items(), key=lambda kv: kv[1], reverse=True)[:2]
        (a, sa), (b, sb) = top
        if sb < config.action_evidence_threshold or sa <= 0:
            return False
        if sb / sa < config.group_ratio:
            return False
        cxa, _ = _centres(fobs, a)
        cxb, _ = _centres(fobs, b)
        return abs(cxa - cxb) / frame_w >= config.group_min_separation_frac

    diagnostics: list[str] = []

    cur: dict | None = None
    closed: list[dict] = []
    last_switch_t = float("-inf")
    last_seen: dict[str, float] = {}

    def open_segment(kind, subject_ids, frame, t, reason):
        return {"kind": kind, "subject_ids": tuple(subject_ids),
                "start_frame": frame, "start_time": t, "reason": reason,
                "frames": [], "occluded_frames": 0}

    def close_segment(seg, end_frame):
        if seg is None:
            return
        if end_frame < seg["start_frame"]:
            diagnostics.append(f"dropped-empty-segment@{seg['start_frame']}")
            return
        seg["end_frame"] = end_frame
        closed.append(seg)

    def transition(kind, subject_ids, frame, t, reason):
        nonlocal cur
        close_segment(cur, frame - 1)
        cur = open_segment(kind, subject_ids, frame, t, reason)

    def accumulate(seg, cands, occluded):
        seg["frames"].append(cur_frame)
        if occluded:
            seg["occluded_frames"] += 1
        if seg["kind"] == "wide":
            return
        ids = seg["subject_ids"]
        vals = [sustained(i, t) for i in ids]
        seg.setdefault("evidence_vals", []).append(
            sum(vals) / len(vals) if vals else 0.0)
        mvals = [motion_stats(i, t)[0] for i in ids]
        seg.setdefault("motion_vals", []).append(
            sum(mvals) / len(mvals) if mvals else 0.0)
        svals = [saliency_mean(i, t) for i in ids]
        seg.setdefault("saliency_vals", []).append(
            sum(svals) / len(svals) if svals else 0.0)
        runner = max([s for tid, s in cands.items() if tid not in ids],
                     default=0.0)
        chosen = sum(vals) / len(vals) if vals else 0.0
        seg.setdefault("margins", []).append(
            min(chosen / max(runner, 1e-6), 100.0) if chosen > 0 else 0.0)

    for cur_frame in range(scene_start, scene_end + 1):
        t = t_of(cur_frame)
        fobs = by_frame.get(cur_frame, {})
        for tid in fobs:
            last_seen[tid] = t
        cam = cam_sustained(t)
        cam_block = cam > config.camera_motion_widen
        # At most one reprocess per frame (occlusion/motion-drop -> fresh
        # evaluation).
        for _ in range(2):
            sus = {tid: sustained(tid, t) for tid in fobs}
            cands = {tid: s for tid, s in sus.items()
                     if s >= config.action_evidence_threshold}
            if cur is None:
                # Large camera motion blocks acquisition: a tight crop on
                # a panning shot would fight the camera.
                pair = None if cam_block else try_group(cands, fobs)
                if pair:
                    mw = (motion_widen_reason(pair[0], t)
                          or motion_widen_reason(pair[1], t))
                    if mw:
                        cur = open_segment("wide", (), cur_frame, t, mw)
                    else:
                        cur = open_segment("group", pair, cur_frame, t,
                                           "interacting-subjects-group")
                        last_switch_t = t
                elif cands and not cam_block:
                    if pair_close(cands, fobs):
                        cur = open_segment("wide", (), cur_frame, t,
                                           "ambiguous-action-wide")
                    else:
                        best = max(cands, key=cands.get)
                        mw = motion_widen_reason(best, t)
                        if mw:
                            cur = open_segment("wide", (), cur_frame, t,
                                               mw)
                        else:
                            cur = open_segment(
                                "single", (best,), cur_frame, t,
                                "acquired-action-subject")
                            last_switch_t = t
                else:
                    cur = open_segment(
                        "wide", (), cur_frame, t,
                        "camera-motion-wide" if cam_block
                        else "no-action-wide")
                accumulate(cur, cands, False)
                break
            if cur["kind"] == "wide":
                pair = None if cam_block else try_group(cands, fobs)
                if pair:
                    mw = (motion_widen_reason(pair[0], t)
                          or motion_widen_reason(pair[1], t))
                    if not mw:
                        transition("group", pair, cur_frame, t,
                                   "interacting-subjects-group")
                        last_switch_t = t
                elif cands and not cam_block:
                    if pair_close(cands, fobs):
                        transition("wide", (), cur_frame, t,
                                   "ambiguous-action-wide")
                    else:
                        best = max(cands, key=cands.get)
                        if not motion_widen_reason(best, t):
                            transition("single", (best,), cur_frame, t,
                                       "acquired-action-subject")
                            last_switch_t = t
                accumulate(cur, cands, False)
                break
            # single / group: occlusion, motion, camera, then challengers.
            held = cur["subject_ids"]
            max_gap = max(t - last_seen.get(h, t) for h in held)
            held_visible = all(h in fobs for h in held)
            if not held_visible and max_gap > config.occlusion_hold_s:
                transition("wide", (), cur_frame, t, "action-lost-widen")
                cur = None
                continue
            occluded = not held_visible
            mmeans = [motion_stats(h, t)[0] for h in held]
            mranges = [motion_stats(h, t)[1] for h in held]
            if max(mmeans, default=0.0) > config.fast_motion_widen:
                # Fast motion: hold a wide frame rather than chase it.
                transition("wide", (), cur_frame, t, "fast-motion-widen")
                cur = None
                continue
            if max(mranges, default=0.0) > config.uncertain_motion_range:
                transition("wide", (), cur_frame, t,
                           "uncertain-motion-widen")
                cur = None
                continue
            if max((travel_speed(h, t) for h in held),
                   default=0.0) > config.fast_travel_frac:
                # Fast crossing: a static lock cannot hold it.
                transition("wide", (), cur_frame, t, "fast-travel-widen")
                cur = None
                continue
            if cam_block:
                transition("wide", (), cur_frame, t, "camera-motion-widen")
                cur = None
                continue
            others = {tid: s for tid, s in cands.items() if tid not in held}
            inc_sus = min([sus.get(h, 0.0) for h in held] or [0.0])
            switched = False
            if others and not occluded:
                best_id = max(others, key=others.get)
                best_sus = others[best_id]
                decisive = (best_sus >= max(
                    config.action_evidence_threshold,
                    inc_sus * config.switch_margin))
                cooled = (t - last_switch_t) >= config.cooldown_s - 1e-9
                dwelled = (t - cur["start_time"]) >= config.min_dwell_s - 1e-9
                if decisive and cooled and dwelled:
                    pair = None
                    if inc_sus >= config.action_evidence_threshold:
                        pair = try_group(
                            {**{h: sus.get(h, 0.0) for h in held},
                             best_id: best_sus}, fobs)
                    if pair and set(pair) != set(held):
                        transition("group", pair, cur_frame, t,
                                   "interacting-subjects-group")
                    else:
                        transition("single", (best_id,), cur_frame, t,
                                   "action-switch-sustained-evidence")
                    last_switch_t = t
                    switched = True
                elif (not decisive
                        and inc_sus >= config.action_evidence_threshold
                        and best_sus >= config.action_evidence_threshold):
                    pair = try_group(
                        {**{h: sus.get(h, 0.0) for h in held},
                         best_id: best_sus}, fobs)
                    if pair and set(pair) != set(held):
                        transition("group", pair, cur_frame, t,
                                   "interacting-subjects-group")
                        last_switch_t = t
                        switched = True
            if not switched:
                if cur["kind"] == "group":
                    a, b = held
                    if min(sus.get(a, 0.0), sus.get(b, 0.0)) < \
                            config.action_evidence_threshold * 0.5:
                        transition("wide", (), cur_frame, t,
                                   "action-pair-lost")
                        cur = None
                        continue
                accumulate(cur, cands, occluded)
            else:
                accumulate(cur, cands, False)
            break

    # Trailing prolonged-loss widening (mirrors the face path): when the
    # held subject(s) were last seen more than occlusion_hold_s before the
    # scene end, the framing must not freeze on empty space.
    t_end = scene_end / fps
    if cur is not None and cur["kind"] in ("single", "group"):
        held = cur["subject_ids"]
        last = max((last_seen.get(h, float("-inf")) for h in held),
                   default=float("-inf"))
        if last > float("-inf") \
                and t_end - last > config.occlusion_hold_s + 1e-9:
            last_frame = max(cur["frames"]) if cur["frames"] \
                else cur["start_frame"]
            close_segment(cur, last_frame)
            if last_frame + 1 <= scene_end:
                cur = open_segment("wide", (), last_frame + 1, t_end,
                                   "action-lost-widen")
            else:
                cur = None
    close_segment(cur, scene_end)

    # --- boundary normalization: ordered, non-overlapping, full coverage ---
    closed.sort(key=lambda s: s["start_frame"])
    merged: list[dict] = []
    for seg in closed:
        if merged and merged[-1]["kind"] == seg["kind"] \
                and merged[-1]["subject_ids"] == seg["subject_ids"]:
            prev = merged[-1]
            prev["frames"].extend(seg["frames"])
            prev["occluded_frames"] += seg["occluded_frames"]
            for key in ("evidence_vals", "motion_vals", "saliency_vals",
                        "margins"):
                prev.setdefault(key, []).extend(seg.get(key, []))
            diagnostics.append(f"merged-adjacent@{seg['start_frame']}")
        else:
            merged.append(seg)
    # Tiny-oscillation merging: a sub-merge-window blip between two stable
    # decisions is absorbed by its predecessor.
    final: list[dict] = []
    for seg in merged:
        dur_s = (seg["end_frame"] - seg["start_frame"] + 1) / fps
        if final and dur_s < config.merge_window_s and seg["kind"] != "wide":
            prev = final[-1]
            prev["frames"].extend(seg["frames"])
            prev["occluded_frames"] += seg["occluded_frames"]
            for key in ("evidence_vals", "motion_vals", "saliency_vals",
                        "margins"):
                prev.setdefault(key, []).extend(seg.get(key, []))
            diagnostics.append(
                f"merged-oscillation-blip@{seg['start_frame']}")
        else:
            final.append(seg)
    # Measurement-blip absorption: a brief non-wide segment adjacent to a
    # wide segment is a measurement artifact (e.g., acquiring a subject
    # just before its speed becomes measurable, then widening). It must
    # not surface as a sub-dwell decision — the conservative wide absorbs
    # it in either direction.
    absorbed: list[dict] = []
    for seg in final:
        dur_s = (seg["end_frame"] - seg["start_frame"] + 1) / fps
        if (dur_s <= config.merge_window_s and seg["kind"] != "wide"
                and absorbed and absorbed[-1]["kind"] == "wide"):
            prev = absorbed[-1]
            prev["end_frame"] = seg["end_frame"]
            prev["frames"].extend(seg["frames"])
            prev["occluded_frames"] += seg["occluded_frames"]
            for key in ("evidence_vals", "motion_vals", "saliency_vals",
                        "margins"):
                prev.setdefault(key, []).extend(seg.get(key, []))
            diagnostics.append(
                f"absorbed-blip-into-wide@{seg['start_frame']}")
        else:
            absorbed.append(seg)
    final = absorbed
    if (len(final) >= 2 and final[0]["kind"] != "wide"
            and final[1]["kind"] == "wide"
            and (final[0]["end_frame"] - final[0]["start_frame"] + 1)
            / fps <= config.merge_window_s):
        lead, nxt = final[0], final[1]
        nxt["start_frame"] = lead["start_frame"]
        nxt["frames"] = lead["frames"] + nxt["frames"]
        nxt["occluded_frames"] += lead["occluded_frames"]
        for key in ("evidence_vals", "motion_vals", "saliency_vals",
                    "margins"):
            nxt[key] = lead.get(key, []) + nxt.get(key, [])
        diagnostics.append(
            f"absorbed-leading-blip-into-wide@{lead['start_frame']}")
        final = final[1:]
    if not final:
        raise ActionPlanError("no-segments",
                              f"scene {scene_id}: segmentation produced "
                              "nothing")

    # Full scene coverage.
    final[0]["start_frame"] = min(final[0]["start_frame"], scene_start)
    for i in range(len(final) - 1):
        final[i]["end_frame"] = final[i + 1]["start_frame"] - 1
    final[-1]["end_frame"] = scene_end
    final = [s for s in final if s["end_frame"] >= s["start_frame"]]
    if not final:
        raise ActionPlanError("no-segments",
                              f"scene {scene_id}: no non-empty segments")

    # --- evidence / confidence -------------------------------------------
    out: list[FramingSegment] = []
    for seg in final:
        ev_vals = seg.get("evidence_vals", [])
        mo_vals = seg.get("motion_vals", [])
        sa_vals = seg.get("saliency_vals", [])
        margins = seg.get("margins", [])
        ev_mean = sum(ev_vals) / len(ev_vals) if ev_vals else 0.0
        mo_mean = sum(mo_vals) / len(mo_vals) if mo_vals else 0.0
        sa_mean = sum(sa_vals) / len(sa_vals) if sa_vals else 0.0
        margin_mean = sum(margins) / len(margins) if margins else 0.0
        if seg["kind"] == "wide":
            confidence = 0.2
        else:
            # 0.75 = strong sustained detector evidence.
            confidence = _clamp01(0.25 + 0.75 * min(1.0, ev_mean / 0.75))
        out.append(FramingSegment(
            scene_id=scene_id,
            start_frame=int(seg["start_frame"]),
            end_frame=int(seg["end_frame"]),
            subject_ids=tuple(seg["subject_ids"]),
            kind=seg["kind"],
            confidence=round(confidence, 4),
            evidence={
                "evidence_mean": round(ev_mean, 4),
                "motion_mean": round(mo_mean, 4),
                "saliency_mean": round(sa_mean, 4),
                "margin_mean": round(margin_mean, 4),
                "n_evidence_frames": len(ev_vals),
                "occluded_frames": int(seg["occluded_frames"]),
            },
            reason=seg["reason"],
        ))
    return out, diagnostics


# --- crop targets -----------------------------------------------------------


def action_crop_for_union(boxes: list[tuple], motion_vec: tuple[float, float],
                          frame_w: float, frame_h: float, aspect: float,
                          config: ActionFramingConfig,
                          use_look_ahead: bool = True) -> tuple[list, str] | \
        tuple[None, str]:
    """Fixed crop for an action subject union box.

    Zoom targets ``target_subject_occupancy`` of the crop height, clamped
    to ``[min_zoom, max_zoom]``. The padded union is a preference the crop
    widens toward (down to ``min_zoom``); only the raw union must fit.
    The centre sits on the union centre pushed by ``look_ahead_frac`` of
    the crop dimension along ``motion_vec`` (unit-ish direction) plus
    ``headroom_frac`` of headroom, then clamped in-bounds.

    Returns ``(crop, "")`` on success, or ``(None, reason)`` with
    ``reason`` in ``("action-union-too-wide", "boundary-pressure-widen",
    "participants-not-contained")`` — the caller widens instead of
    cropping a participant away.
    """
    if not boxes or aspect <= 0 or frame_w <= 0 or frame_h <= 0:
        return None, "action-union-too-wide"
    ux1 = min(b[0] for b in boxes)
    uy1 = min(b[1] for b in boxes)
    ux2 = max(b[0] + b[2] for b in boxes)
    uy2 = max(b[1] + b[3] for b in boxes)
    uw, uh = ux2 - ux1, uy2 - uy1
    if uh <= 0 or uw <= 0:
        return None, "action-union-too-wide"
    zoom = min(max(frame_h * config.target_subject_occupancy / uh,
                   config.min_zoom),
               config.max_zoom)
    padded_w = uw * (1.0 + config.padding_frac)
    padded_h = uh * (1.0 + config.padding_frac)
    crop_w_at = (frame_h / zoom) * aspect
    crop_h_at = frame_h / zoom
    if padded_w > crop_w_at + 1e-6 or padded_h > crop_h_at + 1e-6:
        z_fit = min((frame_h * aspect) / padded_w, frame_h / padded_h)
        if z_fit >= config.min_zoom - 1e-9:
            zoom = max(config.min_zoom, min(zoom, z_fit))
    crop_h = frame_h / zoom
    crop_w = crop_h * aspect
    if crop_w > frame_w:
        crop_w = float(frame_w)
        crop_h = crop_w / aspect
    if uw > crop_w + 1e-6 or uh > crop_h + 1e-6:
        return None, "action-union-too-wide"
    mdx, mdy = motion_vec
    mag = math.hypot(mdx, mdy)
    if use_look_ahead and mag > 1e-9:
        lead_x = (mdx / mag) * config.look_ahead_frac * crop_w
        lead_y = (mdy / mag) * config.look_ahead_frac * crop_h
    else:
        lead_x, lead_y = 0.0, 0.0
    cx = ux1 + uw / 2.0 + lead_x
    cy = uy1 + uh / 2.0 + lead_y + config.headroom_frac * crop_h
    # Containment is a rule, lead room a preference: clamp the desired
    # centre into the interval that keeps the union inside the crop.
    # (Non-empty because the raw union fits the crop, checked above.)
    cx = min(max(cx, ux2 - crop_w / 2.0), ux1 + crop_w / 2.0)
    cy = min(max(cy, uy2 - crop_h / 2.0), uy1 + crop_h / 2.0)
    # Boundary pressure: when clamping eats more than the allowed
    # fraction of the padding, the subject is hugging the frame edge and
    # a tight crop would pin it against the crop edge — widen instead.
    cx_c = min(max(cx, crop_w / 2.0), frame_w - crop_w / 2.0)
    cy_c = min(max(cy, crop_h / 2.0), frame_h - crop_h / 2.0)
    pad_px = config.padding_frac * min(crop_w, crop_h)
    if pad_px > 0 and max(abs(cx - cx_c), abs(cy - cy_c)) > \
            config.boundary_pressure_frac * pad_px + 1e-6:
        return None, "boundary-pressure-widen"
    x1 = int(round(cx_c - crop_w / 2.0))
    y1 = int(round(cy_c - crop_h / 2.0))
    x2 = int(round(cx_c + crop_w / 2.0))
    y2 = int(round(cy_c + crop_h / 2.0))
    x1 = max(0, min(int(frame_w) - 1, x1))
    y1 = max(0, min(int(frame_h) - 1, y1))
    x2 = max(x1 + 1, min(int(frame_w), x2))
    y2 = max(y1 + 1, min(int(frame_h), y2))
    # Never crop away a supported action participant.
    for b in boxes:
        if not (x1 <= b[0] + 1e-6 and y1 <= b[1] + 1e-6
                and b[0] + b[2] <= x2 + 1e-6
                and b[1] + b[3] <= y2 + 1e-6):
            return None, "participants-not-contained"
    return [x1, y1, x2, y2], ""


def _lerp_box(a: list, b: list, t: float, frame_w: float,
              frame_h: float) -> list:
    box = [int(round(av + (bv - av) * t)) for av, bv in zip(a, b)]
    box[0] = max(0, min(int(frame_w) - 1, box[0]))
    box[1] = max(0, min(int(frame_h) - 1, box[1]))
    box[2] = max(box[0] + 1, min(int(frame_w), box[2]))
    box[3] = max(box[1] + 1, min(int(frame_h), box[3]))
    return box


def build_action_keyframes(segments: list[FramingSegment],
                           log: ActionObservationLog, scene_id: int,
                           config: ActionFramingConfig,
                           frame_w: float, frame_h: float,
                           aspect: float, *,
                           fps: float) \
        -> tuple[list[FramingSegment], list[CropKeyframe], list[str]]:
    """Attach fixed lock crops to action segments; bounded ease keyframes.

    One stable subject (or group) keeps ONE crop for the whole segment.
    Between accepted decisions, ease keyframes interpolate linearly from
    the previous lock crop — the only camera motion the plan ever emits —
    with travel limited to ``max_travel_frac`` of the frame dimension PER
    SECOND: the transition is extended over as many frames as the
    per-frame speed cap needs, so EVERY consecutive rendered crop-centre
    step — including the final ease-to-lock step — respects the cap. When
    the segment cannot fit the transition, the previous lock is held (zero
    travel). Segments whose subject union cannot be cropped safely become
    ``wide`` with a reason. Returns ``(segments, keyframes, diagnostics)``.
    """
    diagnostics: list[str] = []
    keyframes: list[CropKeyframe] = []
    out_segments: list[FramingSegment] = []
    prev_lock: list | None = None

    for idx, seg in enumerate(segments):
        seg_obs = [o for o in log.observations_for_scene(scene_id)
                   if seg.start_frame <= o.frame <= seg.end_frame]
        lock: list | None = None
        kind = seg.kind
        subject_ids = seg.subject_ids
        reason = seg.reason
        if kind in ("single", "group"):
            boxes = []
            ok = True
            for tid in seg.subject_ids:
                tb = [o.box for o in seg_obs if o.track_id == tid]
                if not tb:
                    ok = False
                    break
                # Temporal union: the lock contains the subject's whole
                # travel within the segment (a median would crop the
                # subject away at the segment ends).
                ux1 = min(b[0] for b in tb)
                uy1 = min(b[1] for b in tb)
                ux2 = max(b[0] + b[2] for b in tb)
                uy2 = max(b[1] + b[3] for b in tb)
                boxes.append((ux1, uy1, ux2 - ux1, uy2 - uy1))
            motion_vec = (0.0, 0.0)
            if ok and boxes:
                # Motion direction from the primary track's observed
                # displacement across the segment (first -> last centre).
                prim = [o for o in seg_obs
                        if o.track_id == seg.subject_ids[0]]
                if len(prim) >= 2:
                    b0, b1 = prim[0].box, prim[-1].box
                    motion_vec = (
                        (b1[0] + b1[2] / 2.0) - (b0[0] + b0[2] / 2.0),
                        (b1[1] + b1[3] / 2.0) - (b0[1] + b0[3] / 2.0),
                    )
                lock, why = action_crop_for_union(
                    boxes, motion_vec, frame_w, frame_h, aspect, config,
                    use_look_ahead=True)
                if lock is None:
                    # Look-ahead is a preference, containment is a rule:
                    # retry without lead room before widening.
                    lock, why = action_crop_for_union(
                        boxes, (0.0, 0.0), frame_w, frame_h, aspect,
                        config, use_look_ahead=False)
                if lock is None:
                    reason = why
            if not ok or lock is None:
                kind = "wide"
                subject_ids = ()
                if lock is None and not reason:
                    reason = "action-union-too-wide"
                diagnostics.append(
                    f"scene {scene_id} segment @{seg.start_frame}: {reason}")
        if kind == "wide":
            # Saliency-weighted centroid of the segment's observations
            # biases the full-width window SLIGHTLY toward salient
            # content — it can never tighten the crop (the window stays
            # the widest available).
            tot = sum(o.saliency for o in seg_obs)
            if tot > 0:
                sal_cx = sum((o.box[0] + o.box[2] / 2.0) * o.saliency
                             for o in seg_obs) / tot
            else:
                sal_cx = frame_w / 2.0
            lock = saliency_biased_wide_crop(
                sal_cx, frame_w, frame_h, aspect, config.saliency_bias_frac)

        # Per-second travel limiting (mirrors the Task 12 mechanism):
        # extend the transition over enough frames that no consecutive
        # rendered step exceeds the speed cap, or hold the previous lock
        # when the segment cannot fit it.
        ease_n = 0
        if idx > 0 and prev_lock is not None and lock is not None \
                and config.ease_frames > 0 and fps > 0:
            capx = config.max_travel_frac * frame_w / fps
            capy = config.max_travel_frac * frame_h / fps
            travel = [abs(lock[i] - prev_lock[i]) for i in range(4)]
            need = 0
            if capx > 0:
                need = max(need, math.ceil(travel[0] / capx),
                           math.ceil(travel[2] / capx))
            if capy > 0:
                need = max(need, math.ceil(travel[1] / capy),
                           math.ceil(travel[3] / capy))
            # +1 frame of margin so int rounding of the keyframes can
            # never push a step over the cap.
            ease_n = max(int(config.ease_frames), need + 1)
            avail = seg.end_frame - seg.start_frame + 1
            if ease_n > avail:
                lock = list(prev_lock)
                ease_n = 0
                diagnostics.append(
                    f"scene {scene_id} segment @{seg.start_frame}: "
                    f"transition-held")

        if ease_n > 0:
            for i in range(ease_n):
                fr = seg.start_frame + i
                if fr > seg.end_frame:
                    break
                tt = (i + 1) / (ease_n + 1)
                eb = _lerp_box(prev_lock, lock, tt, frame_w, frame_h)
                keyframes.append(CropKeyframe(frame=fr, x1=eb[0], y1=eb[1],
                                              x2=eb[2], y2=eb[3], kind="ease"))
        # A lock keyframe pins the segment's stable crop in the plan so
        # the trajectory survives the project bridge (applicable scenes
        # need keyframes or a fallback reason). It sits at the first
        # stable frame (after any ease), clamped inside the segment.
        if lock is not None:
            lock_fr = min(seg.start_frame + ease_n, seg.end_frame)
            # Don't duplicate an ease keyframe on the same frame.
            if not any(k.frame == lock_fr for k in keyframes):
                keyframes.append(CropKeyframe(
                    frame=lock_fr, x1=int(lock[0]), y1=int(lock[1]),
                    x2=int(lock[2]), y2=int(lock[3]), kind="lock"))
        out_segments.append(FramingSegment(
            scene_id=seg.scene_id, start_frame=seg.start_frame,
            end_frame=seg.end_frame, subject_ids=subject_ids, kind=kind,
            confidence=seg.confidence, evidence=seg.evidence, reason=reason,
            crop=lock))
        prev_lock = lock

    keyframes.sort(key=lambda k: k.frame)
    return out_segments, keyframes, diagnostics
