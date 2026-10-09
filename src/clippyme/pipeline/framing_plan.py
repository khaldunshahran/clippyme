"""Evidence-backed turn-aware framing plans for automatic TRACK/WIDE scenes.

This module is the pure, host-testable core of Task 12. It does NOT detect
faces, decode video, or call any provider: it consumes validated face
observations (absolute frame/time, scene id, anonymous track id, box,
visibility, bounded mouth-motion evidence, provenance/reason) and produces a
deterministic, versioned, auditable *framing plan* — ordered non-overlapping
scene-local segments with fixed crop targets — which the two-pass renderer in
``reframe.py`` persists atomically and renders from verbatim.

Design rules (see docs/turn-aware-framing.md for the full contract):

* Observations are strictly validated. Booleans, non-finite values, bad boxes,
  duplicate ``(frame, track_id)`` ids, regressing times and cross-scene
  identity carryover are rejected with an ``ObservationRejected`` code. Track
  ids are opaque (``t-<int>``) — a name or diarization label can never validate.
* In multi-face scenes size alone cannot prove the active speaker: a switch
  requires *sustained* mouth-motion evidence over ``evidence_duration_s`` plus
  a relative ``switch_margin`` over the incumbent, after ``min_dwell_s`` and
  ``cooldown_s`` have elapsed. Weak evidence yields group/wide or held-safe
  framing, never a bounce.
* The crop is fixed within each stable segment. Only a short bounded ease
  (``ease_frames``) interpolates between accepted turns; travel is limited to
  ``max_travel_frac`` of the frame dimension PER SECOND (the transition
  extends over as many frames as the speed cap needs, or the previous lock is
  held), zoom is held unless widening, crops are clamped in-bounds with
  headroom/padding.
* Plans are versioned (``SCHEMA_VERSION``) and reused only when source
  fingerprint, dimensions/FPS/frame count, scene plan, schema version, config
  digest, output aspect, ordered scene strategies and the tracker decision
  digest all match. No pixels, embeddings, names, provider responses or raw
  mouth landmarks are ever persisted.
* Pass-1 analysis runs under one monotonic wall-time + frame budget
  (``AnalysisBudget``); when it exhausts, expensive analysis stops and the
  affected scenes keep the accepted static/wide fallback.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import struct
import time
from dataclasses import dataclass, field


SCHEMA_VERSION = 4
#: Schema versions the parser still accepts. Task 12/13 sidecars (v3)
#: remain valid: they parse, validate, normalize, publish and preserve
#: exactly as before — they are never silently treated as v4 action plans.
SUPPORTED_SCHEMA_VERSIONS = (3, 4)
PLAN_PRODUCER = "clippyme-turn-framing-plan"

# Bound for per-observation mouth-motion evidence. Mirrors the tracker's
# ``min(variance * 200.0, 3.0)`` scale so injected and tracker-supplied
# evidence share one range.
MOUTH_MOTION_MAX = 3.0

# Opaque anonymous track ids. Anything else (a name, a diarization label,
# a provider id) fails validation by construction.
TRACK_ID_RE = re.compile(r"^t-\d+$")

PROVENANCE_VALUES = ("detector", "tracker", "injected-test", "smoother-hold")
REASON_MAX_LEN = 64

# Scene strategies the plan governs. GENERAL/SPLIT/SCREENCAST/DISABLED keep
# their existing renderers and never reach the face plan builder.
FACE_STRATEGIES = ("TRACK", "WIDE")

#: Strategies eligible for Task 14 action planning. Only automatic GENERAL
#: scenes reach the action planner.
ACTION_STRATEGIES = ("GENERAL",)

#: Explicit versioned scene kinds (v4 schema). v3 plans carry no scene_kind
#: key: an applied face-strategy scene derives to "face", anything else to
#: "none" — old non-face scenes are never silently treated as face plans.
SCENE_KINDS = ("face", "action", "none")
ACTION_SCENE_KIND = "action"


# --- flag + config ----------------------------------------------------------

def turn_aware_enabled() -> bool:
    """``CLIPPYME_TURN_AWARE_FRAMING`` — enabled by default on this branch.

    Only the explicit negatives ``0``/``false``/``no``/``off`` restore the
    accepted static comfort behavior (whole-scene median collapse). Unset,
    empty, or anything else means on. Documented in docs/turn-aware-framing.md.
    """
    val = (os.getenv("CLIPPYME_TURN_AWARE_FRAMING") or "").strip().lower()
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
class TurnFramingConfig:
    """Tunable turn-segmentation / crop policy. All durations in seconds."""

    evidence_duration_s: float = 0.8   # sustained mouth-motion window to accept a turn
    min_dwell_s: float = 1.0           # a turn must hold this long before switching away
    cooldown_s: float = 1.5            # no new switch within this of the last accepted one
    switch_margin: float = 1.25        # challenger evidence must beat incumbent * margin
    merge_window_s: float = 0.5        # sub-window oscillation blips are merged away
    occlusion_hold_s: float = 1.0      # short occlusion holds the framing safely
    ease_frames: int = 6               # short bounded ease between accepted turns
    max_travel_frac: float = 0.5       # per-second cap on crop-centre travel
                                       # (fraction of the frame dimension / s)
    analysis_deadline_s: float = 120.0  # monotonic wall-time budget for pass-1
                                       # analysis (decode/detect/MAR/planner)
    headroom_frac: float = 0.15        # headroom above the subject, fraction of crop height
    padding_frac: float = 0.35         # padding around the subject union box
    hold_zoom: bool = True             # never zoom in mid-scene; only widen
    group_ratio: float = 0.88          # two-subject ambiguity threshold (runner-up / leader)
    group_min_separation_frac: float = 0.05  # min face-centre separation: guards against
                                             # near-duplicate detections of one face
    mouth_evidence_threshold: float = 0.5    # sustained evidence floor to acquire/switch
    target_face_occupancy: float = 0.4       # face height / crop height target
    min_zoom: float = 1.0
    max_zoom: float = 1.6
    max_analysis_frames: int = 20000   # hard cap on recorded observations

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
            "headroom_frac": self.headroom_frac,
            "padding_frac": self.padding_frac,
            "hold_zoom": self.hold_zoom,
            "group_ratio": self.group_ratio,
            "group_min_separation_frac": self.group_min_separation_frac,
            "mouth_evidence_threshold": self.mouth_evidence_threshold,
            "target_face_occupancy": self.target_face_occupancy,
            "min_zoom": self.min_zoom,
            "max_zoom": self.max_zoom,
            "max_analysis_frames": self.max_analysis_frames,
            "analysis_deadline_s": self.analysis_deadline_s,
        }

    def config_digest(self) -> str:
        """Stable identity of the config for plan reuse matching."""
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def config_from_env() -> TurnFramingConfig:
    """Build the config from ``CLIPPYME_TURN_*`` overrides; garbage → default."""
    return TurnFramingConfig(
        evidence_duration_s=_env_float("CLIPPYME_TURN_EVIDENCE_S", 0.8, 0.05, 10.0),
        min_dwell_s=_env_float("CLIPPYME_TURN_MIN_DWELL_S", 1.0, 0.0, 30.0),
        cooldown_s=_env_float("CLIPPYME_TURN_COOLDOWN_S", 1.5, 0.0, 30.0),
        switch_margin=_env_float("CLIPPYME_TURN_SWITCH_MARGIN", 1.25, 1.0, 10.0),
        merge_window_s=_env_float("CLIPPYME_TURN_MERGE_S", 0.5, 0.0, 10.0),
        occlusion_hold_s=_env_float("CLIPPYME_TURN_OCCLUSION_S", 1.0, 0.0, 10.0),
        ease_frames=_env_int("CLIPPYME_TURN_EASE_FRAMES", 6, 0),
        max_analysis_frames=_env_int("CLIPPYME_TURN_MAX_ANALYSIS_FRAMES", 20000, 1),
        analysis_deadline_s=_env_float("CLIPPYME_TURN_ANALYSIS_DEADLINE_S",
                                       120.0, 0.05, 3600.0),
    )


class AnalysisBudget:
    """Monotonic wall-time + frame budget for pass-1 turn analysis.

    One monotonic clock per framing invocation: the deadline is fixed at
    construction from ``time.monotonic()`` and never re-read from a wall
    clock that can jump. ``ok()`` latches -- once the deadline passes or the
    frame budget is spent it stays ``False`` and ``exhaust_reason`` names the
    cause:

    * ``"analysis_budget_exhausted"`` -- the wall-time deadline passed;
    * ``"analysis-frame-budget-exhausted"`` -- the frame/observation cap hit
      (the same token the ``ObservationLog`` records on its first drop).

    The caller checks ``ok()`` before every expensive step (frame
    decode/retrieve, face detection, mouth analysis, planner build) and stops
    new expensive analysis once it latches -- detector/MAR call counts must
    freeze, not merely have their results discarded.

    The clock is injectable for tests (``clock=time.monotonic`` by default).
    """

    def __init__(self, *, deadline_s: float, max_frames: int, clock=None):
        self._clock = clock if clock is not None else time.monotonic
        self._deadline = self._clock() + max(0.0, float(deadline_s))
        self.frames_left = max(0, int(max_frames))
        self.exhausted = False
        self.exhaust_reason: str | None = None

    def ok(self) -> bool:
        if self.exhausted:
            return False
        if self.frames_left <= 0:
            self.exhausted = True
            self.exhaust_reason = "analysis-frame-budget-exhausted"
            return False
        if self._clock() >= self._deadline:
            self.exhausted = True
            self.exhaust_reason = "analysis_budget_exhausted"
            return False
        return True

    def consume(self) -> None:
        """Spend one analyzed frame of the frame budget."""
        if self.frames_left > 0:
            self.frames_left -= 1


# --- observation record -----------------------------------------------------

class ObservationRejected(ValueError):
    """Raised when a face observation fails strict validation.

    The message is prefixed with a machine-readable code
    (``duplicate-observation``, ``time-regression``, ...).
    """


@dataclass(frozen=True)
class FaceObservation:
    """One validated face observation.

    ``frame``/``time_s`` are absolute (source) coordinates, ``scene_id`` the
    scene index, ``track_id`` an opaque anonymous id (``t-<int>`` — never a
    name or diarization id), ``box`` a finite in-bounds ``(x, y, w, h)``,
    ``visibility`` in [0, 1], ``mouth_motion`` bounded in [0, 3.0],
    ``provenance`` one of ``PROVENANCE_VALUES``, and ``reason`` a short
    machine-readable token.
    """

    frame: int
    time_s: float
    scene_id: int
    track_id: str
    box: tuple
    visibility: float
    mouth_motion: float
    provenance: str
    reason: str


def _is_num(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


class ObservationLog:
    """Validating, order-preserving store of face observations.

    Rejects booleans, non-finite values, bad boxes, duplicate
    ``(frame, track_id)`` ids, regressing times and cross-scene identity
    carryover. Rejections are counted in ``diagnostics`` by the caller; the
    log itself raises ``ObservationRejected``.
    """

    def __init__(self, frame_w: float, frame_h: float, max_observations: int = 20000):
        self.frame_w = float(frame_w)
        self.frame_h = float(frame_h)
        self.max_observations = max(1, int(max_observations))
        self._obs: list[FaceObservation] = []
        self._seen: set[tuple[int, str]] = set()
        self._last_time: dict[str, float] = {}
        self._track_scene: dict[str, int] = {}
        self.diagnostics: list[str] = []
        self.dropped = 0
        # Structured budget-exhaustion state (Task 12 review 02): set
        # once via mark_exhaustion() when pass-1 analysis stops.
        # exhaustion_frame is the first frame that was NOT analyzed;
        # scenes ending strictly before it were fully analyzed, the
        # containing scene is partial, later scenes are unobserved.
        # None when exhaustion was recorded without frame context.
        self.exhaustion_reason: str | None = None
        self.exhaustion_frame: int | None = None
        self.exhaustion_scene: int | None = None

    def add(self, obs: FaceObservation) -> None:
        self._validate(obs)
        if len(self._obs) >= self.max_observations:
            self.dropped += 1
            if self.dropped == 1:
                # Task 12 review 03: record exhaustion at the actual first
                # dropped observation. mark_exhaustion is first-call-wins,
                # so pass 1 can never replace this original context after
                # scene bookkeeping advances.
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

        First call wins: pass 1 latches analysis off exactly once, and
        the planner must see the ORIGINAL exhaustion point, not a later
        one. The reason is also appended to ``diagnostics`` (once) so
        older consumers that only read diagnostics keep working.
        """
        if self.exhaustion_reason is not None:
            return
        self.exhaustion_reason = str(reason)
        self.exhaustion_frame = frame
        self.exhaustion_scene = scene_id
        if reason not in self.diagnostics:
            self.diagnostics.append(reason)

    def observations_for_scene(self, scene_id: int) -> list[FaceObservation]:
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
    def _validate(self, obs: FaceObservation) -> None:
        def bad(code: str, detail: str = "") -> ObservationRejected:
            return ObservationRejected(f"{code}{(': ' + detail) if detail else ''}")

        if isinstance(obs.frame, bool) or not isinstance(obs.frame, int) or obs.frame < 0:
            raise bad("bad-frame", repr(obs.frame))
        if (not _is_num(obs.time_s) or not math.isfinite(obs.time_s)
                or obs.time_s < 0):
            raise bad("bad-time", repr(obs.time_s))
        if (isinstance(obs.scene_id, bool) or not isinstance(obs.scene_id, int)
                or obs.scene_id < 0):
            raise bad("bad-scene-id", repr(obs.scene_id))
        if not isinstance(obs.track_id, str) or not TRACK_ID_RE.match(obs.track_id):
            # Deliberately strict: only opaque generated ids validate, so a
            # name or diarization label can never enter the plan.
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
        if (not _is_num(obs.visibility) or not math.isfinite(obs.visibility)
                or not 0.0 <= obs.visibility <= 1.0):
            raise bad("bad-visibility", repr(obs.visibility))
        if (not _is_num(obs.mouth_motion) or not math.isfinite(obs.mouth_motion)
                or not 0.0 <= obs.mouth_motion <= MOUTH_MOTION_MAX):
            raise bad("bad-mouth-motion", repr(obs.mouth_motion))
        if obs.provenance not in PROVENANCE_VALUES:
            raise bad("bad-provenance", repr(obs.provenance))
        if (not isinstance(obs.reason, str) or not obs.reason
                or len(obs.reason) > REASON_MAX_LEN):
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
                      f"track={obs.track_id} scenes {prev_scene} -> {obs.scene_id}")


# --- turn segmentation ------------------------------------------------------

@dataclass
class FramingSegment:
    """One stable, scene-local framing segment.

    ``subject_ids`` holds one anonymous id, two for an ambiguous-dialogue
    pair crop, or ``()`` for a wide/fallback segment. ``crop`` is the segment's
    fixed lock crop ``[x1, y1, x2, y2]`` (filled in by the keyframe builder).
    """

    scene_id: int
    start_frame: int
    end_frame: int
    subject_ids: tuple
    kind: str  # "single" | "two-subject" | "wide"
    confidence: float
    evidence: dict = field(default_factory=dict)
    reason: str = ""
    crop: list = field(default_factory=list)


def _clamp01(v: float) -> float:
    return min(1.0, max(0.0, v))


def segment_turns(log: ObservationLog, *, scene_id: int, scene_start: int,
                  scene_end: int, config: TurnFramingConfig, fps: float,
                  frame_w: float, frame_h: float, aspect: float = 9 / 16):
    """Segment one scene's observations into ordered non-overlapping segments.

    Returns ``(segments, diagnostics)``. Raises ``_PlanBuildError`` when the
    scene has no usable observations. Never raises on data quirks beyond that:
    callers isolate per-scene failures.
    """
    obs = [o for o in log.observations_for_scene(scene_id)
           if scene_start <= o.frame <= scene_end]
    if not obs:
        raise _PlanBuildError("no-observations",
                              f"scene {scene_id}: no usable face observations")
    if fps <= 0:
        raise _PlanBuildError("bad-fps", f"scene {scene_id}: fps={fps}")

    frames = sorted({o.frame for o in obs})
    by_frame: dict[int, dict[str, FaceObservation]] = {}
    for o in obs:
        by_frame.setdefault(o.frame, {})[o.track_id] = o
    t_of = {f: by_frame[f][next(iter(by_frame[f]))].time_s for f in frames}
    track_obs: dict[str, list[FaceObservation]] = {}
    for o in obs:
        track_obs.setdefault(o.track_id, []).append(o)

    def sustained(tid: str, t: float) -> float:
        vals = [o.mouth_motion for o in track_obs[tid]
                if t - config.evidence_duration_s - 1e-9 <= o.time_s <= t + 1e-9]
        return sum(vals) / len(vals) if vals else 0.0

    def try_two_subject(cands: dict[str, float],
                        fobs: dict[str, FaceObservation]):
        """Ambiguous dialogue: two strong, close, separable subjects.

        Returns the pair ``(a, b)`` or ``None``. Size alone never qualifies —
        both tracks need sustained mouth-motion evidence within ``group_ratio``
        of each other, and the padded union must fit the widest 9:16 crop
        (``crop_for_subjects`` re-checks precisely; this is the fast path).
        """
        if len(cands) < 2:
            return None
        top = sorted(cands.items(), key=lambda kv: kv[1], reverse=True)[:2]
        (a, sa), (b, sb) = top
        if sb < config.mouth_evidence_threshold or sa <= 0:
            return None
        if sb / sa < config.group_ratio:
            return None
        boxa, boxb = fobs[a].box, fobs[b].box
        cxa = boxa[0] + boxa[2] / 2.0
        cxb = boxb[0] + boxb[2] / 2.0
        if abs(cxa - cxb) / frame_w < config.group_min_separation_frac:
            return None
        ux1 = min(boxa[0], boxb[0])
        ux2 = max(boxa[0] + boxa[2], boxb[0] + boxb[2])
        need_w = (ux2 - ux1) * (1.0 + config.padding_frac)
        if need_w > frame_w * aspect:
            return None
        return (a, b)

    def pair_close(cands, fobs):
        """True when the top two candidates are ambiguous-close (evidence +
        separation) REGARDLESS of whether they fit one 9:16 crop. Used to
        fall back to wide when a pair crop is geometrically impossible."""
        if len(cands) < 2:
            return False
        top = sorted(cands.items(), key=lambda kv: kv[1], reverse=True)[:2]
        (a, sa), (b, sb) = top
        if sb < config.mouth_evidence_threshold or sa <= 0:
            return False
        if sb / sa < config.group_ratio:
            return False
        boxa, boxb = fobs[a].box, fobs[b].box
        cxa = boxa[0] + boxa[2] / 2.0
        cxb = boxb[0] + boxb[2] / 2.0
        return abs(cxa - cxb) / frame_w >= config.group_min_separation_frac

    diagnostics: list[str] = []

    # Segment under construction: dicts until boundaries are final.
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

    def accumulate(seg, sus, cands, fobs, occluded):
        seg["frames"].append(cur_frame)
        if occluded:
            seg["occluded_frames"] += 1
        if seg["kind"] == "wide":
            return
        ids = seg["subject_ids"]
        vals = [sus.get(i, 0.0) for i in ids]
        seg.setdefault("mouth_vals", []).append(
            sum(vals) / len(vals) if vals else 0.0)
        runner = max([s for tid, s in cands.items() if tid not in ids],
                     default=0.0)
        chosen = sum(vals) / len(vals) if vals else 0.0
        seg.setdefault("margins", []).append(
            min(chosen / max(runner, 1e-6), 100.0) if chosen > 0 else 0.0)

    for cur_frame in frames:
        t = t_of[cur_frame]
        fobs = by_frame[cur_frame]
        for tid in fobs:
            last_seen[tid] = t
        # At most one reprocess per frame (occlusion-drop -> fresh evaluation).
        for _ in range(2):
            sus = {tid: sustained(tid, t) for tid in fobs}
            cands = {tid: s for tid, s in sus.items()
                     if s >= config.mouth_evidence_threshold}
            if cur is None:
                pair = try_two_subject(cands, fobs)
                if pair:
                    cur = open_segment("two-subject", pair, cur_frame, t,
                                       "ambiguous-dialogue-two-subject")
                    last_switch_t = t
                elif cands:
                    if pair_close(cands, fobs):
                        # Ambiguous dialogue that cannot share one 9:16
                        # crop: stay wide rather than crowning one speaker
                        # on ambiguity (mid-segment the incumbent holds).
                        cur = open_segment("wide", (), cur_frame, t,
                                           "ambiguous-dialogue-wide")
                    else:
                        best = max(cands, key=cands.get)
                        cur = open_segment("single", (best,), cur_frame, t,
                                           "acquired-speaker")
                        last_switch_t = t
                else:
                    cur = open_segment("wide", (), cur_frame, t,
                                       "no-speaker-wide")
                accumulate(cur, sus, cands, fobs, False)
                break
            if cur["kind"] == "wide":
                pair = try_two_subject(cands, fobs)
                if pair:
                    transition("two-subject", pair, cur_frame, t,
                               "ambiguous-dialogue-two-subject")
                    last_switch_t = t
                elif cands:
                    if pair_close(cands, fobs):
                        transition("wide", (), cur_frame, t,
                                   "ambiguous-dialogue-wide")
                    else:
                        best = max(cands, key=cands.get)
                        transition("single", (best,), cur_frame, t,
                                   "acquired-speaker")
                        last_switch_t = t
                accumulate(cur, sus, cands, fobs, False)
                break
            # single / two-subject: occlusion, then challenger evaluation.
            held = cur["subject_ids"]
            max_gap = max(t - last_seen.get(h, t) for h in held)
            held_visible = all(h in fobs for h in held)
            if not held_visible and max_gap > config.occlusion_hold_s:
                # Prolonged loss widens; the reprocess below re-evaluates.
                transition("wide", (), cur_frame, t, "speaker-lost-widen")
                cur = None
                continue
            occluded = not held_visible  # within the hold window: keep framing
            others = {tid: s for tid, s in cands.items() if tid not in held}
            inc_sus = min([sus.get(h, 0.0) for h in held] or [0.0])
            switched = False
            if others:
                best_id = max(others, key=others.get)
                best_sus = others[best_id]
                decisive = (best_sus >= max(config.mouth_evidence_threshold,
                                           inc_sus * config.switch_margin))
                cooled = (t - last_switch_t) >= config.cooldown_s - 1e-9
                dwelled = (t - cur["start_time"]) >= config.min_dwell_s - 1e-9
                if decisive and cooled and dwelled:
                    pair = None
                    if inc_sus >= config.mouth_evidence_threshold:
                        pair = try_two_subject(
                            {**{h: sus.get(h, 0.0) for h in held},
                             best_id: best_sus}, fobs)
                    if pair and set(pair) != set(held):
                        transition("two-subject", pair, cur_frame, t,
                                   "ambiguous-dialogue-two-subject")
                    else:
                        transition("single", (best_id,), cur_frame, t,
                                   "turn-switch-sustained-evidence")
                    last_switch_t = t
                    switched = True
                elif (not decisive
                        and inc_sus >= config.mouth_evidence_threshold
                        and best_sus >= config.mouth_evidence_threshold):
                    pair = try_two_subject(
                        {**{h: sus.get(h, 0.0) for h in held},
                         best_id: best_sus}, fobs)
                    if pair and set(pair) != set(held):
                        transition("two-subject", pair, cur_frame, t,
                                   "ambiguous-dialogue-two-subject")
                        last_switch_t = t
                        switched = True
            if not switched:
                if cur["kind"] == "two-subject":
                    a, b = held
                    if min(sus.get(a, 0.0), sus.get(b, 0.0)) < \
                            config.mouth_evidence_threshold * 0.5:
                        transition("wide", (), cur_frame, t,
                                   "dialogue-pair-lost")
                        cur = None
                        continue
                accumulate(cur, sus, cands, fobs, occluded)
            else:
                accumulate(cur, sus, cands, fobs, False)
            break

    # Trailing prolonged-loss widening: when the held subject(s) were last
    # seen more than occlusion_hold_s before the scene end, the framing must
    # not freeze on empty space — split off a trailing wide segment. (A short
    # trailing gap simply extends the hold, via the close below.)
    t_end = scene_end / fps
    if cur is not None and cur["kind"] in ("single", "two-subject"):
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
                                   "speaker-lost-widen")
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
            prev.setdefault("mouth_vals", []).extend(seg.get("mouth_vals", []))
            prev.setdefault("margins", []).extend(seg.get("margins", []))
            diagnostics.append(f"merged-adjacent@{seg['start_frame']}")
        else:
            merged.append(seg)
    # Tiny-oscillation merging: a sub-merge-window blip between two stable
    # framings is absorbed by its predecessor (it can never be a real turn:
    # min_dwell already exceeds the merge window by default).
    final: list[dict] = []
    for seg in merged:
        dur_s = (seg["end_frame"] - seg["start_frame"] + 1) / fps
        if final and dur_s < config.merge_window_s and seg["kind"] != "wide":
            prev = final[-1]
            prev["frames"].extend(seg["frames"])
            prev["occluded_frames"] += seg["occluded_frames"]
            prev.setdefault("mouth_vals", []).extend(seg.get("mouth_vals", []))
            prev.setdefault("margins", []).extend(seg.get("margins", []))
            diagnostics.append(
                f"merged-oscillation-blip@{seg['start_frame']}")
        else:
            final.append(seg)
    if not final:
        raise _PlanBuildError("no-segments",
                              f"scene {scene_id}: segmentation produced nothing")

    # Full scene coverage: first segment starts at scene_start, each segment
    # ends where the next begins, the last ends at scene_end.
    final[0]["start_frame"] = min(final[0]["start_frame"], scene_start)
    for i in range(len(final) - 1):
        final[i]["end_frame"] = final[i + 1]["start_frame"] - 1
    final[-1]["end_frame"] = scene_end
    final = [s for s in final if s["end_frame"] >= s["start_frame"]]
    if not final:
        raise _PlanBuildError("no-segments",
                              f"scene {scene_id}: no non-empty segments")

    # --- evidence / confidence -------------------------------------------
    out: list[FramingSegment] = []
    for seg in final:
        mouth_vals = seg.get("mouth_vals", [])
        margins = seg.get("margins", [])
        mouth_mean = sum(mouth_vals) / len(mouth_vals) if mouth_vals else 0.0
        margin_mean = sum(margins) / len(margins) if margins else 0.0
        if seg["kind"] == "wide":
            confidence = 0.2
        else:
            # 1.5 = strong sustained mouth-motion evidence (half the cap).
            confidence = _clamp01(0.25 + 0.75 * min(1.0, mouth_mean / 1.5))
        out.append(FramingSegment(
            scene_id=scene_id,
            start_frame=int(seg["start_frame"]),
            end_frame=int(seg["end_frame"]),
            subject_ids=tuple(seg["subject_ids"]),
            kind=seg["kind"],
            confidence=round(confidence, 4),
            evidence={
                "mouth_mean": round(mouth_mean, 4),
                "margin_mean": round(margin_mean, 4),
                "n_evidence_frames": len(mouth_vals),
                "occluded_frames": int(seg["occluded_frames"]),
            },
            reason=seg["reason"],
        ))
    return out, diagnostics


class _PlanBuildError(Exception):
    def __init__(self, reason: str, detail: str = ""):
        super().__init__(f"{reason}: {detail}")
        self.reason = reason
        self.detail = detail


# --- crop targets -----------------------------------------------------------

@dataclass
class CropKeyframe:
    frame: int
    x1: int
    y1: int
    x2: int
    y2: int
    kind: str  # "lock" | "ease"


def _median_box(boxes: list[tuple]) -> tuple:
    xs = sorted(b[0] for b in boxes)
    ys = sorted(b[1] for b in boxes)
    ws = sorted(b[2] for b in boxes)
    hs = sorted(b[3] for b in boxes)
    n = len(boxes)
    mid = n // 2
    if n % 2 == 1:
        return (xs[mid], ys[mid], ws[mid], hs[mid])
    return ((xs[mid - 1] + xs[mid]) / 2.0, (ys[mid - 1] + ys[mid]) / 2.0,
            (ws[mid - 1] + ws[mid]) / 2.0, (hs[mid - 1] + hs[mid]) / 2.0)


def crop_for_subjects(boxes: list[tuple], frame_w: float, frame_h: float,
                      aspect: float, config: TurnFramingConfig,
                      prev_crop_h: float | None = None) -> list | None:
    """Fixed crop for a subject union box.

    Zoom targets ``target_face_occupancy`` of the crop height (the same
    convention as the cameraman's ``zoom_for_face_height``), clamped to
    ``[min_zoom, max_zoom]``. The padded union width is a preference the
    crop widens toward (down to ``min_zoom``); only the raw union must
    fit, otherwise the caller falls back to wide. With ``hold_zoom``
    the crop never tightens
    relative to ``prev_crop_h`` — it holds or widens. The centre sits on the
    union centre pushed down by ``headroom_frac`` of the crop height so the
    subject keeps headroom; the box is clamped in-bounds. Returns ``None``
    when the raw subject union cannot fit the crop (caller falls back
    to wide).
    """
    if not boxes or aspect <= 0 or frame_w <= 0 or frame_h <= 0:
        return None
    ux1 = min(b[0] for b in boxes)
    uy1 = min(b[1] for b in boxes)
    ux2 = max(b[0] + b[2] for b in boxes)
    uy2 = max(b[1] + b[3] for b in boxes)
    uw, uh = ux2 - ux1, uy2 - uy1
    if uh <= 0 or uw <= 0:
        return None
    # Zoom from the height occupancy target. The padded union width is a
    # preference: widen (never tighten) toward fitting it, down to min_zoom.
    # Hard feasibility is the RAW union fitting the crop - a huge face gets a
    # tight-but-complete crop rather than a wide fallback.
    zoom = min(max(frame_h * config.target_face_occupancy / uh,
                   config.min_zoom),
               config.max_zoom)
    padded_w = uw * (1.0 + config.padding_frac)
    crop_w_at = (frame_h / zoom) * aspect
    if padded_w > crop_w_at + 1e-6:
        z_fit = (frame_h * aspect) / padded_w
        if z_fit >= config.min_zoom - 1e-9:
            zoom = max(config.min_zoom, min(zoom, z_fit))
    if config.hold_zoom and prev_crop_h:
        zoom = min(zoom, frame_h / prev_crop_h)  # hold unless widening
    crop_h = frame_h / zoom
    crop_w = crop_h * aspect
    if crop_w > frame_w:
        crop_w = float(frame_w)
        crop_h = crop_w / aspect
    if uw > crop_w + 1e-6 or uh > crop_h + 1e-6:
        return None  # the subject itself cannot fit: caller falls back wide
    cx = ux1 + uw / 2.0
    cy = uy1 + uh / 2.0 + config.headroom_frac * crop_h
    cx = min(max(cx, crop_w / 2.0), frame_w - crop_w / 2.0)
    cy = min(max(cy, crop_h / 2.0), frame_h - crop_h / 2.0)
    x1 = int(round(cx - crop_w / 2.0))
    y1 = int(round(cy - crop_h / 2.0))
    x2 = int(round(cx + crop_w / 2.0))
    y2 = int(round(cy + crop_h / 2.0))
    x1 = max(0, min(int(frame_w) - 1, x1))
    y1 = max(0, min(int(frame_h) - 1, y1))
    x2 = max(x1 + 1, min(int(frame_w), x2))
    y2 = max(y1 + 1, min(int(frame_h), y2))
    return [x1, y1, x2, y2]


def _wide_crop(median_cx: float | None, frame_w: float, frame_h: float,
               aspect: float) -> list:
    """Accepted wide behavior: full 9:16 window at zoom 1.0, centred on the
    segment's median subject centre (frame centre when unknown)."""
    crop_h = float(frame_h)
    crop_w = crop_h * aspect
    if crop_w > frame_w:
        crop_w = float(frame_w)
        crop_h = crop_w / aspect
    cx = frame_w / 2.0 if median_cx is None else median_cx
    cx = min(max(cx, crop_w / 2.0), frame_w - crop_w / 2.0)
    cy = frame_h / 2.0
    cy = min(max(cy, crop_h / 2.0), frame_h - crop_h / 2.0)
    x1 = max(0, int(round(cx - crop_w / 2.0)))
    y1 = max(0, int(round(cy - crop_h / 2.0)))
    x2 = min(int(frame_w), int(round(cx + crop_w / 2.0)))
    y2 = min(int(frame_h), int(round(cy + crop_h / 2.0)))
    return [x1, y1, max(x1 + 1, x2), y2]


def _lerp_box(a: list, b: list, t: float, frame_w: float,
              frame_h: float) -> list:
    box = [int(round(av + (bv - av) * t)) for av, bv in zip(a, b)]
    box[0] = max(0, min(int(frame_w) - 1, box[0]))
    box[1] = max(0, min(int(frame_h) - 1, box[1]))
    box[2] = max(box[0] + 1, min(int(frame_w), box[2]))
    box[3] = max(box[1] + 1, min(int(frame_h), box[3]))
    return box


def build_scene_keyframes(segments: list[FramingSegment], log: ObservationLog,
                          scene_id: int, config: TurnFramingConfig,
                          frame_w: float, frame_h: float,
                          aspect: float, *,
                          fps: float) -> tuple[list[FramingSegment], list[CropKeyframe], list[str]]:
    """Attach fixed lock crops to segments and build bounded ease keyframes.

    One stable subject (or pair) keeps ONE crop for the whole segment. Between
    accepted turns, ease keyframes interpolate linearly from the previous lock
    crop — the only camera motion the plan ever emits — with travel limited to
    ``max_travel_frac`` of the frame dimension PER SECOND: the transition is
    extended over as many frames as the per-frame speed cap
    (``max_travel_frac * dim / fps``) needs, so EVERY consecutive rendered
    crop-centre step — including the final ease-to-lock step — respects the
    cap. When the segment cannot fit the transition, the previous lock is held
    (zero travel — the safe fallback); excess travel is never hidden in the
    final step. Segments whose subjects cannot fit a crop become ``wide``
    with a reason. Returns ``(segments, keyframes, diagnostics)``.
    """
    diagnostics: list[str] = []
    keyframes: list[CropKeyframe] = []
    out_segments: list[FramingSegment] = []
    prev_lock: list | None = None
    prev_crop_h: float | None = None
    prev_kind: str | None = None

    for idx, seg in enumerate(segments):
        seg_obs = [o for o in log.observations_for_scene(scene_id)
                   if seg.start_frame <= o.frame <= seg.end_frame]
        lock: list | None = None
        kind = seg.kind
        subject_ids = seg.subject_ids
        reason = seg.reason
        if kind in ("single", "two-subject"):
            boxes = []
            ok = True
            for tid in seg.subject_ids:
                tb = [o.box for o in seg_obs if o.track_id == tid]
                if not tb:
                    ok = False
                    break
                boxes.append(_median_box(tb))
            if ok:
                # Hold-zoom ratchet applies only within continuous subject
                # framing. A transition OUT of wide is a deliberate reframe
                # (like a cut to a closer shot) and resets the baseline -
                # otherwise a wide opening would pin every later single.
                hold_h = prev_crop_h \
                    if prev_kind in ("single", "two-subject") else None
                lock = crop_for_subjects(boxes, frame_w, frame_h, aspect,
                                         config, hold_h)
            if lock is None:
                kind = "wide"
                subject_ids = ()
                reason = "subjects-too-wide-for-crop"
                diagnostics.append(
                    f"scene {scene_id} segment @{seg.start_frame}: {reason}")
        if kind == "wide":
            # Centre between the observed faces (mean of centres): the 9:16
            # window at zoom 1.0 is the widest accepted framing.
            cxs = [o.box[0] + o.box[2] / 2.0 for o in seg_obs]
            mean_cx = sum(cxs) / len(cxs) if cxs else None
            lock = _wide_crop(mean_cx, frame_w, frame_h, aspect)

        # Per-second travel limiting: extend the transition over enough frames
        # that no consecutive rendered step exceeds the speed cap, or hold the
        # previous lock when the segment cannot fit it.
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
            # +1 frame of margin so int rounding of the keyframes can never
            # push a step over the cap.
            ease_n = max(int(config.ease_frames), need + 1)
            avail = seg.end_frame - seg.start_frame + 1
            if ease_n > avail:
                # Safe fallback: hold the previous lock (zero travel) rather
                # than rushing the pan or hiding excess in the final step.
                lock = list(prev_lock)
                ease_n = 0
                diagnostics.append(
                    f"scene {scene_id} segment @{seg.start_frame}: "
                    f"transition-held")

        # Bounded ease: the only motion the plan emits. Uniform t spacing over
        # ease_n+1 steps keeps the final ease-to-lock step under the cap too.
        if ease_n > 0:
            for i in range(ease_n):
                fr = seg.start_frame + i
                if fr > seg.end_frame:
                    break
                tt = (i + 1) / (ease_n + 1)
                eb = _lerp_box(prev_lock, lock, tt, frame_w, frame_h)
                keyframes.append(CropKeyframe(frame=fr, x1=eb[0], y1=eb[1],
                                              x2=eb[2], y2=eb[3], kind="ease"))
        out_segments.append(FramingSegment(
            scene_id=seg.scene_id, start_frame=seg.start_frame,
            end_frame=seg.end_frame, subject_ids=subject_ids, kind=kind,
            confidence=seg.confidence, evidence=seg.evidence, reason=reason,
            crop=lock))
        prev_lock = lock
        prev_crop_h = float(lock[3] - lock[1])
        prev_kind = kind

    keyframes.sort(key=lambda k: k.frame)
    return out_segments, keyframes, diagnostics


def crop_at_frame(scene_plan: "ScenePlan", frame: int) -> list | None:
    """Deterministic crop lookup: exact ease keyframe hit wins, otherwise the
    containing segment's fixed lock crop. No decisions are (re)computed here —
    everything was decided when the plan was built.
    """
    if not scene_plan.applies or scene_plan.fallback is not None:
        return None
    for kf in scene_plan.keyframes:
        if kf.frame == frame:
            return [kf.x1, kf.y1, kf.x2, kf.y2]
    for seg in scene_plan.segments:
        if seg.start_frame <= frame <= seg.end_frame:
            return list(seg.crop) if seg.crop else None
    return None


# --- plan container / identity / serialization -------------------------------

@dataclass
class ScenePlan:
    scene_id: int
    strategy: str
    start_frame: int
    end_frame: int
    applies: bool
    scene_kind: str = "none"  # "face" | "action" | "none" (v4; v3 derives it)
    tracks: list = field(default_factory=list)
    segments: list = field(default_factory=list)   # FramingSegment
    keyframes: list = field(default_factory=list)  # CropKeyframe
    fallback: dict | None = None                   # {"reason", "detail"}
    fallback_crop: list | None = None              # [cx, cy, zoom] static crop

    @property
    def usable(self) -> bool:
        return self.applies and self.fallback is None


@dataclass
class FramingPlan:
    schema_version: int
    producer: str
    source_fingerprint: str
    frame_w: int
    frame_h: int
    fps: float
    frame_count: int
    scene_intervals: list
    config_digest: str
    output_aspect: float = 9 / 16  # actual render aspect the crops were built for
    tracker_digest: str = ""       # digest of the tracker decision config that
                                   # produced the plan's observations
    scenes: list = field(default_factory=list)  # ScenePlan
    created_at: str = ""
    # Whether pass-1 analysis ran to completion. An exhausted/incomplete
    # plan may be persisted for audit and safe rendering, but
    # plan_matches() refuses to reuse it -- the next invocation must
    # re-analyze.
    analysis_complete: bool = True


def _segment_to_dict(seg: FramingSegment) -> dict:
    return {
        "scene_id": seg.scene_id,
        "start_frame": seg.start_frame,
        "end_frame": seg.end_frame,
        "subject_ids": list(seg.subject_ids),
        "kind": seg.kind,
        "confidence": seg.confidence,
        "evidence": dict(seg.evidence),
        "reason": seg.reason,
        "crop": list(seg.crop),
    }


def _keyframe_to_dict(kf: CropKeyframe) -> dict:
    return {"frame": kf.frame, "x1": kf.x1, "y1": kf.y1, "x2": kf.x2,
            "y2": kf.y2, "kind": kf.kind}


def plan_to_dict(plan: FramingPlan) -> dict:
    # v4 persists the explicit versioned scene kind per scene. v3 omits it
    # (byte-identical to the Task 12/13 output when the action path is off).
    is_v4 = plan.schema_version == SCHEMA_VERSION
    scenes_out = []
    for sp in plan.scenes:
        sd = {
            "scene_id": sp.scene_id,
            "strategy": sp.strategy,
            "start_frame": sp.start_frame,
            "end_frame": sp.end_frame,
            "applies": sp.applies,
            "tracks": [dict(t) for t in sp.tracks],
            "segments": [_segment_to_dict(s) for s in sp.segments],
            "keyframes": [_keyframe_to_dict(k) for k in sp.keyframes],
            "fallback": dict(sp.fallback) if sp.fallback else None,
            "fallback_crop": list(sp.fallback_crop)
            if sp.fallback_crop else None,
        }
        if is_v4:
            sd["scene_kind"] = sp.scene_kind
        scenes_out.append(sd)
    return {
        "schema_version": plan.schema_version,
        "producer": plan.producer,
        "source_fingerprint": plan.source_fingerprint,
        "frame_w": plan.frame_w,
        "frame_h": plan.frame_h,
        "fps": plan.fps,
        "frame_count": plan.frame_count,
        "scene_intervals": [list(iv) for iv in plan.scene_intervals],
        "config_digest": plan.config_digest,
        "output_aspect": plan.output_aspect,
        "tracker_digest": plan.tracker_digest,
        "analysis_complete": plan.analysis_complete,
        "created_at": plan.created_at,
        "scenes": scenes_out,
    }


def scene_kind_of(scene: dict) -> str:
    """The scene's contract kind: explicit v4 ``scene_kind``, or the
    v3-derived contract (an applied face-strategy scene is ``"face"``,
    anything else is ``"none"``).

    Used by the domain bridge (``project_framing.normalize_timeline``) so
    face, action, and non-applicable scenes are distinguished on the
    canonical parsed form of both schema versions.
    """
    kind = scene.get("scene_kind")
    if kind in SCENE_KINDS:
        return kind
    if scene.get("applies") and scene.get("strategy") in FACE_STRATEGIES:
        return "face"
    return "none"


def _strict_int(v) -> bool:
    """int that is not a bool — JSON true/false must never validate as a number."""
    return isinstance(v, int) and not isinstance(v, bool)


def _strict_num(v) -> bool:
    """Finite non-boolean number."""
    return isinstance(v, (int, float)) and not isinstance(v, bool) \
        and math.isfinite(v)


def _req(d: dict, key: str, ctx: str):
    if not isinstance(d, dict) or key not in d:
        raise ValueError(f"malformed-plan: {ctx} missing '{key}'")
    return d[key]


def _no_unknown(d: dict, allowed: frozenset, ctx: str) -> None:
    unknown = set(d) - allowed
    if unknown:
        raise ValueError(
            f"malformed-plan: {ctx} has unknown keys {sorted(unknown)!r}")


# Exact key allow-lists: the parser rejects anything it did not write —
# pixels, landmarks, embeddings, names and any other nested smuggling.
_PLAN_TOP_KEYS = frozenset({
    "schema_version", "producer", "source_fingerprint", "frame_w", "frame_h",
    "fps", "frame_count", "scene_intervals", "config_digest", "output_aspect",
    "tracker_digest", "analysis_complete", "created_at", "scenes",
})
_SCENE_KEYS = frozenset({
    "scene_id", "strategy", "start_frame", "end_frame", "applies",
    "tracks", "segments", "keyframes", "fallback", "fallback_crop",
})
# v4 adds the explicit versioned scene kind; v3 dicts must NOT carry it
# (unknown keys are rejected, so a v3 sidecar can never smuggle a kind).
_SCENE_KEYS_V4 = _SCENE_KEYS | frozenset({"scene_kind"})
_SEGMENT_KEYS = frozenset({
    "scene_id", "start_frame", "end_frame", "subject_ids", "kind",
    "confidence", "evidence", "reason", "crop",
})
_KEYFRAME_KEYS = frozenset({"frame", "x1", "y1", "x2", "y2", "kind"})
_TRACK_KEYS = frozenset({"id", "first_frame", "last_frame", "n_obs"})
_FALLBACK_KEYS = frozenset({"reason", "detail"})
# Allowed segment evidence keys with bounded scalar values only.
_EVIDENCE_SPEC = {
    "mouth_mean": (0.0, MOUTH_MOTION_MAX),
    "margin_mean": (0.0, 100.0),
    "n_evidence_frames": "count",
    "occluded_frames": "count",
}

# Allowed ACTION segment evidence keys with bounded scalar values only.
# Distinct from the face contract's keys so parsing can tell face and
# action scenes apart (an action scene carrying mouth-motion keys — or a
# face scene carrying action keys — is malformed).
ACTION_EVIDENCE_SPEC = {
    "evidence_mean": (0.0, 1.0),
    "motion_mean": (0.0, 1.0),
    "saliency_mean": (0.0, 1.0),
    "margin_mean": (0.0, 100.0),
    "n_evidence_frames": "count",
    "occluded_frames": "count",
}


def _parse_track(t: dict, ctx: str, frame_count: int) -> dict:
    tctx = ctx + ".track"
    if not isinstance(t, dict):
        raise ValueError(f"malformed-plan: {tctx} must be a dict")
    _no_unknown(t, _TRACK_KEYS, tctx)
    tid = _req(t, "id", tctx)
    if not isinstance(tid, str) or not TRACK_ID_RE.match(tid):
        raise ValueError(f"malformed-plan: {tctx} non-anonymous track id")
    first = _req(t, "first_frame", tctx)
    last = _req(t, "last_frame", tctx)
    n_obs = _req(t, "n_obs", tctx)
    if (not _strict_int(first) or not _strict_int(last)
            or not _strict_int(n_obs)
            or not 0 <= first <= last < frame_count
            or not 0 <= n_obs <= frame_count):
        raise ValueError(f"malformed-plan: {tctx} bad frame/count fields")
    return {"id": tid, "first_frame": first, "last_frame": last,
            "n_obs": n_obs}


def _parse_evidence(ev: dict, ctx: str, spec: dict | None = None) -> dict:
    ectx = ctx + ".evidence"
    if spec is None:
        spec = _EVIDENCE_SPEC
    if not isinstance(ev, dict):
        raise ValueError(f"malformed-plan: {ectx} must be a dict")
    unknown = set(ev) - set(spec)
    if unknown:
        raise ValueError(
            f"malformed-plan: {ectx} unknown keys {sorted(unknown)!r}")
    out: dict = {}
    for key, key_spec in spec.items():
        if key not in ev:
            continue
        v = ev[key]
        if key_spec == "count":
            if not _strict_int(v) or v < 0:
                raise ValueError(f"malformed-plan: {ectx} bad '{key}'")
            out[key] = int(v)
        else:
            lo, hi = key_spec
            if not _strict_num(v) or not lo <= v <= hi:
                raise ValueError(f"malformed-plan: {ectx} bad '{key}'")
            out[key] = float(v)
    return out


def _parse_segment(g: dict, ctx: str, scene_id: int, start_frame: int,
                   end_frame: int, frame_w: int, frame_h: int,
                   scene_kind: str = "face") -> FramingSegment:
    gctx = ctx + ".segment"
    if not isinstance(g, dict):
        raise ValueError(f"malformed-plan: {gctx} must be a dict")
    _no_unknown(g, _SEGMENT_KEYS, gctx)
    seg_scene = _req(g, "scene_id", gctx)
    if not _strict_int(seg_scene) or seg_scene != scene_id:
        raise ValueError(f"malformed-plan: {gctx} scene_id mismatch")
    gs = _req(g, "start_frame", gctx)
    ge = _req(g, "end_frame", gctx)
    if (not _strict_int(gs) or not _strict_int(ge)
            or not start_frame <= gs <= ge <= end_frame):
        raise ValueError(f"malformed-plan: {gctx} frames out of range")
    kind = _req(g, "kind", gctx)
    # Face and action scenes have distinct segment-kind contracts; a
    # segment kind from the wrong contract is malformed.
    if scene_kind == "action":
        allowed_kinds = ("single", "group", "wide")
        want_n = {"single": 1, "group": 2, "wide": 0}
    else:
        allowed_kinds = ("single", "two-subject", "wide")
        want_n = {"single": 1, "two-subject": 2, "wide": 0}
    if kind not in allowed_kinds:
        raise ValueError(f"malformed-plan: {gctx} bad kind {kind!r} for "
                         f"{scene_kind} scene")
    sids = _req(g, "subject_ids", gctx)
    if (not isinstance(sids, list)
            or any(not isinstance(i, str) or not TRACK_ID_RE.match(i)
                   for i in sids)):
        raise ValueError(f"malformed-plan: {gctx} non-anonymous subject id")
    if len(sids) != want_n[kind]:
        raise ValueError(f"malformed-plan: {gctx} kind/subject-count mismatch")
    if kind in ("two-subject", "group") and len(set(sids)) != 2:
        raise ValueError(f"malformed-plan: {gctx} {kind} ids must be "
                         "distinct")
    conf = _req(g, "confidence", gctx)
    if not _strict_num(conf) or not 0.0 <= conf <= 1.0:
        raise ValueError(f"malformed-plan: {gctx} bad confidence")
    evidence = _parse_evidence(
        _req(g, "evidence", gctx), gctx,
        ACTION_EVIDENCE_SPEC if scene_kind == "action" else None)
    reason = _req(g, "reason", gctx)
    if (not isinstance(reason, str) or not reason
            or len(reason) > REASON_MAX_LEN):
        raise ValueError(f"malformed-plan: {gctx} bad reason token")
    crop = _req(g, "crop", gctx)
    if (not isinstance(crop, list) or len(crop) != 4
            or not all(_strict_int(v) for v in crop)):
        raise ValueError(f"malformed-plan: {gctx} bad crop")
    x1, y1, x2, y2 = crop
    if not (0 <= x1 < x2 <= frame_w and 0 <= y1 < y2 <= frame_h):
        raise ValueError(f"malformed-plan: {gctx} crop out of bounds")
    return FramingSegment(
        scene_id=scene_id, start_frame=gs, end_frame=ge,
        subject_ids=tuple(sids), kind=kind, confidence=float(conf),
        evidence=evidence, reason=reason, crop=list(crop))


def _parse_keyframe(k: dict, ctx: str, start_frame: int, end_frame: int,
                    frame_w: int, frame_h: int) -> CropKeyframe:
    kctx = ctx + ".keyframe"
    if not isinstance(k, dict):
        raise ValueError(f"malformed-plan: {kctx} must be a dict")
    _no_unknown(k, _KEYFRAME_KEYS, kctx)
    fr = _req(k, "frame", kctx)
    if not _strict_int(fr) or not start_frame <= fr <= end_frame:
        raise ValueError(f"malformed-plan: {kctx} frame outside scene")
    kk = _req(k, "kind", kctx)
    if kk not in ("lock", "ease"):
        raise ValueError(f"malformed-plan: {kctx} bad kind {kk!r}")
    coords = []
    for c in ("x1", "y1", "x2", "y2"):
        v = _req(k, c, kctx)
        if not _strict_int(v):
            raise ValueError(f"malformed-plan: {kctx} bad coord '{c}'")
        coords.append(v)
    if not (0 <= coords[0] < coords[2] <= frame_w
            and 0 <= coords[1] < coords[3] <= frame_h):
        raise ValueError(f"malformed-plan: {kctx} out of bounds")
    return CropKeyframe(frame=fr, x1=coords[0], y1=coords[1], x2=coords[2],
                        y2=coords[3], kind=kk)


def _parse_fallback(fb, ctx: str) -> dict | None:
    if fb is None:
        return None
    fctx = ctx + ".fallback"
    if not isinstance(fb, dict):
        raise ValueError(f"malformed-plan: {fctx} must be a dict or null")
    _no_unknown(fb, _FALLBACK_KEYS, fctx)
    reason = _req(fb, "reason", fctx)
    if (not isinstance(reason, str) or not reason
            or len(reason) > REASON_MAX_LEN):
        raise ValueError(f"malformed-plan: {fctx} reason must be a bounded "
                         f"token")
    detail = fb.get("detail")
    if detail is not None and (not isinstance(detail, str)
                               or len(detail) > 256):
        raise ValueError(f"malformed-plan: {fctx} detail must be a bounded "
                         f"string")
    out = {"reason": reason}
    if detail is not None:
        out["detail"] = detail
    return out


def _parse_fallback_crop(fc, ctx: str, frame_w: int,
                         frame_h: int) -> list | None:
    if fc is None:
        return None
    cctx = ctx + ".fallback_crop"
    if (not isinstance(fc, (list, tuple)) or len(fc) != 3
            or not all(_strict_num(v) for v in fc)):
        raise ValueError(f"malformed-plan: {cctx} must be [cx, cy, zoom] "
                         f"numbers")
    cx, cy, zoom = (float(v) for v in fc)
    if not (0.0 <= cx <= frame_w and 0.0 <= cy <= frame_h
            and 0.0 < zoom <= 16.0):
        raise ValueError(f"malformed-plan: {cctx} out of bounds")
    return [cx, cy, zoom]


def plan_from_dict(d: dict) -> FramingPlan:
    """Strict parse. Anything unexpected → ``ValueError`` (caller rebuilds).

    Every persisted field is validated recursively: strict non-boolean
    numerics, finite positive dimensions/FPS/frame count, ordered unique scene
    ids with intervals matching the plan header, anonymous track ids with
    bounded frame/count fields, ordered non-overlapping full-coverage
    segments, unique in-scene keyframes, allowed evidence keys with bounded
    scalar values only, bounded reason/detail tokens, and finite in-bounds
    fallback crops. Unknown keys — pixels, landmarks, embeddings, names —
    are rejected at every level.
    """
    if not isinstance(d, dict):
        raise ValueError("malformed-plan: top-level object must be a dict")
    _no_unknown(d, _PLAN_TOP_KEYS, "plan")
    schema_version = _req(d, "schema_version", "plan")
    if not _strict_int(schema_version) \
            or schema_version not in SUPPORTED_SCHEMA_VERSIONS:
        raise ValueError(
            f"malformed-plan: unsupported schema_version "
            f"{d.get('schema_version')!r} (expected one of "
            f"{SUPPORTED_SCHEMA_VERSIONS})")
    producer = _req(d, "producer", "plan")
    if not isinstance(producer, str) or not producer or len(producer) > 128:
        raise ValueError("malformed-plan: invalid 'producer'")
    source_fingerprint = _req(d, "source_fingerprint", "plan")
    if (not isinstance(source_fingerprint, str) or not source_fingerprint
            or len(source_fingerprint) > 256):
        raise ValueError("malformed-plan: invalid 'source_fingerprint'")
    frame_w = _req(d, "frame_w", "plan")
    frame_h = _req(d, "frame_h", "plan")
    frame_count = _req(d, "frame_count", "plan")
    for _name, _v in (("frame_w", frame_w), ("frame_h", frame_h),
                      ("frame_count", frame_count)):
        if not _strict_int(_v) or not 0 < _v <= 100000:
            raise ValueError(f"malformed-plan: invalid '{_name}'")
    fps = _req(d, "fps", "plan")
    if not _strict_num(fps) or not 0.0 < fps <= 1000.0:
        raise ValueError("malformed-plan: invalid 'fps'")
    output_aspect = _req(d, "output_aspect", "plan")
    if not _strict_num(output_aspect) or not 0.0 < output_aspect <= 16.0:
        raise ValueError("malformed-plan: invalid 'output_aspect'")
    tracker_digest = _req(d, "tracker_digest", "plan")
    if not isinstance(tracker_digest, str) or len(tracker_digest) > 256:
        raise ValueError("malformed-plan: invalid 'tracker_digest'")
    config_digest = _req(d, "config_digest", "plan")
    if (not isinstance(config_digest, str) or not config_digest
            or len(config_digest) > 256):
        raise ValueError("malformed-plan: invalid 'config_digest'")
    analysis_complete = _req(d, "analysis_complete", "plan")
    if not isinstance(analysis_complete, bool):
        raise ValueError("malformed-plan: 'analysis_complete' must be "
                         "a bool")
    created_at = _req(d, "created_at", "plan")
    if not isinstance(created_at, str) or len(created_at) > 64:
        raise ValueError("malformed-plan: invalid 'created_at'")
    intervals = _req(d, "scene_intervals", "plan")
    if not isinstance(intervals, list):
        raise ValueError("malformed-plan: invalid 'scene_intervals'")
    for iv in intervals:
        if (not isinstance(iv, (list, tuple)) or len(iv) != 2
                or not all(_strict_int(v) for v in iv)
                or not 0 <= iv[0] < iv[1] <= frame_count):
            raise ValueError(f"malformed-plan: bad scene interval {iv!r}")
    for idx in range(1, len(intervals)):
        if not intervals[idx - 1][1] < intervals[idx][0]:
            raise ValueError("malformed-plan: scene intervals must be "
                             "strictly ordered and non-overlapping")
    raw_scenes = _req(d, "scenes", "plan")
    if not isinstance(raw_scenes, list):
        raise ValueError("malformed-plan: invalid 'scenes'")
    if len(raw_scenes) != len(intervals):
        raise ValueError("malformed-plan: scenes/scene_intervals length "
                         "mismatch")
    scenes = []
    is_v4 = schema_version == SCHEMA_VERSION
    for idx, s in enumerate(raw_scenes):
        ctx = f"scene[{idx}]"
        if not isinstance(s, dict):
            raise ValueError(f"malformed-plan: {ctx} must be a dict")
        _no_unknown(s, _SCENE_KEYS_V4 if is_v4 else _SCENE_KEYS, ctx)
        scene_id = _req(s, "scene_id", ctx)
        if not _strict_int(scene_id) or scene_id != idx:
            raise ValueError(f"malformed-plan: {ctx} scene_id must be the "
                             f"ordered unique index {idx}")
        start_frame = _req(s, "start_frame", ctx)
        end_frame = _req(s, "end_frame", ctx)
        if (not _strict_int(start_frame) or not _strict_int(end_frame)
                or (start_frame, end_frame) != (intervals[idx][0],
                                                intervals[idx][1])):
            raise ValueError(f"malformed-plan: {ctx} frames must match "
                             f"scene_intervals[{idx}]")
        applies = _req(s, "applies", ctx)
        if not isinstance(applies, bool):
            raise ValueError(f"malformed-plan: {ctx} 'applies' must be a bool")
        strategy = _req(s, "strategy", ctx)
        if (not isinstance(strategy, str) or not strategy
                or len(strategy) > 32):
            raise ValueError(f"malformed-plan: {ctx} invalid 'strategy'")
        if is_v4:
            # Explicit versioned scene kind: parsing distinguishes the
            # face, action, and non-applicable contracts — an old
            # non-face scene is never silently treated as a face plan.
            scene_kind = _req(s, "scene_kind", ctx)
            if scene_kind not in SCENE_KINDS:
                raise ValueError(f"malformed-plan: {ctx} bad scene_kind "
                                 f"{scene_kind!r}")
            if scene_kind == "face":
                if strategy not in FACE_STRATEGIES or not applies:
                    raise ValueError(
                        f"malformed-plan: {ctx} face scene needs a face "
                        f"strategy and applies=true")
            elif scene_kind == "action":
                if strategy not in ACTION_STRATEGIES or not applies:
                    raise ValueError(
                        f"malformed-plan: {ctx} action scene needs a "
                        f"GENERAL strategy and applies=true")
            elif applies:
                raise ValueError(f"malformed-plan: {ctx} non-applicable "
                                 f"scene must not claim applies")
        else:
            # v3 legacy derivation: an applied face-strategy scene is a
            # face scene; anything else is non-applicable. The v3
            # validation below is unchanged (Task 12/13 sidecars parse
            # exactly as before).
            scene_kind = ("face" if applies and strategy in FACE_STRATEGIES
                          else "none")
        raw_tracks = _req(s, "tracks", ctx)
        if not isinstance(raw_tracks, list):
            raise ValueError(f"malformed-plan: {ctx} invalid 'tracks'")
        tracks = [_parse_track(t, ctx, frame_count) for t in raw_tracks]
        raw_segments = _req(s, "segments", ctx)
        if not isinstance(raw_segments, list):
            raise ValueError(f"malformed-plan: {ctx} invalid 'segments'")
        segments = []
        prev_end = start_frame - 1
        for g in raw_segments:
            seg = _parse_segment(g, ctx, scene_id, start_frame, end_frame,
                                 frame_w, frame_h, scene_kind)
            if seg.start_frame != prev_end + 1:
                raise ValueError(f"malformed-plan: {ctx} segments must "
                                 "be contiguous without gaps or "
                                 "overlaps")
            prev_end = seg.end_frame
            segments.append(seg)
        if segments and segments[-1].end_frame != end_frame:
            raise ValueError(f"malformed-plan: {ctx} segments must cover "
                             f"the scene")
        raw_keyframes = _req(s, "keyframes", ctx)
        if not isinstance(raw_keyframes, list):
            raise ValueError(f"malformed-plan: {ctx} invalid 'keyframes'")
        keyframes = []
        prev_kf = start_frame - 1
        for k in raw_keyframes:
            kf = _parse_keyframe(k, ctx, start_frame, end_frame, frame_w,
                                 frame_h)
            if kf.frame <= prev_kf:
                raise ValueError(f"malformed-plan: {ctx} keyframes must be "
                                 f"unique and ordered")
            prev_kf = kf.frame
            keyframes.append(kf)
        fallback = _parse_fallback(s.get("fallback"), ctx)
        fallback_crop = _parse_fallback_crop(s.get("fallback_crop"), ctx,
                                             frame_w, frame_h)
        # Relational invariants (Task 12 review 02): relationships
        # between tracks, segments and the scene — not just scalar
        # shapes.
        tids = [t["id"] for t in tracks]
        if len(set(tids)) != len(tids):
            raise ValueError(f"malformed-plan: {ctx} duplicate track "
                             "ids")
        for t in tracks:
            if not (start_frame <= t["first_frame"]
                    <= t["last_frame"] <= end_frame):
                raise ValueError(f"malformed-plan: {ctx} track "
                                 f"'{t["id"]}' frames outside its "
                                 "scene")
        track_by_id = {t["id"]: t for t in tracks}
        for seg in segments:
            for sid in seg.subject_ids:
                tr = track_by_id.get(sid)
                if tr is None:
                    raise ValueError(f"malformed-plan: {ctx} segment "
                                     f"subject '{sid}' missing from "
                                     "scene tracks")
                if not (tr["first_frame"] <= seg.end_frame
                        and tr["last_frame"] >= seg.start_frame):
                    raise ValueError(f"malformed-plan: {ctx} segment "
                                     f"subject '{sid}' track range "
                                     "inconsistent with segment")
        if fallback is not None and (segments or keyframes):
            raise ValueError(f"malformed-plan: {ctx} fallback scene "
                             "must not carry active segments/keyframes")
        if not applies and (segments or keyframes or tracks
                           or fallback is not None
                           or fallback_crop is not None):
            # v3 keeps the accepted message verbatim.
            content_word = "face" if not is_v4 else scene_kind
            raise ValueError(f"malformed-plan: {ctx} non-applicable "
                             "scene must not carry planned "
                             f"{content_word} content")
        scenes.append(ScenePlan(
            scene_id=scene_id, strategy=strategy, start_frame=start_frame,
            end_frame=end_frame, applies=applies, scene_kind=scene_kind,
            tracks=tracks,
            segments=segments, keyframes=keyframes, fallback=fallback,
            fallback_crop=fallback_crop))
    return FramingPlan(
        schema_version=int(schema_version),
        producer=producer,
        source_fingerprint=source_fingerprint,
        frame_w=frame_w, frame_h=frame_h, fps=float(fps),
        frame_count=frame_count,
        scene_intervals=[[int(iv[0]), int(iv[1])] for iv in intervals],
        config_digest=config_digest,
        output_aspect=float(output_aspect),
        tracker_digest=tracker_digest,
        analysis_complete=analysis_complete,
        scenes=scenes,
        created_at=created_at,
    )



def save_plan_atomic(plan: FramingPlan, path: str) -> str:
    """Persist the plan atomically (temp file + ``os.replace``).

    Readers never see a half-written plan. If the write, flush or replace
    fails, the temp file is removed in ``finally`` while an existing
    destination is left untouched. Returns ``path``.
    """
    payload = json.dumps(plan_to_dict(plan), sort_keys=True,
                         separators=(",", ":"))
    tmp = f"{path}.tmp-{os.getpid()}"
    failed = True
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(payload)
            fh.flush()
        os.replace(tmp, path)
        failed = False
    finally:
        if failed:
            try:
                os.remove(tmp)
            except OSError:
                pass
    return path


def load_plan(path: str) -> FramingPlan:
    """Load a persisted plan. ``FileNotFoundError`` when missing;
    ``ValueError`` when malformed — both mean "rebuild or safe fallback"."""
    with open(path, "r", encoding="utf-8") as fh:
        raw = fh.read()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"malformed-plan: invalid JSON ({e})") from e
    return plan_from_dict(data)


def plan_matches(plan: FramingPlan, *, source_fingerprint: str, frame_w: int,
                 frame_h: int, fps: float, frame_count: int,
                 scene_intervals: list, config_digest: str,
                 aspect: float | None = None,
                 strategies: list | None = None,
                 tracker_digest: str | None = None) -> tuple[bool, str]:
    """Reuse gate: every identity input must match, else rebuild.

    ``aspect`` is the requested output aspect (width/height), ``strategies``
    the current ordered per-scene strategies, ``tracker_digest`` the digest of
    the tracker decision config that produced the observations. Each is
    compared when given; ``None`` skips that check (callers that omit an
    input get a weaker gate — the production path always passes all three).

    Returns ``(True, "match")`` or ``(False, "<stale-field>")``.
    """
    # Both supported schemas (3 and 4) are current: a v3 sidecar parses
    # and may match when every identity input aligns. Cross-flag reuse is
    # still impossible — the v4 combined config digest never equals a v3
    # turn-only digest, so a stale flag state rebuilds via stale-config.
    if plan.schema_version not in SUPPORTED_SCHEMA_VERSIONS:
        return False, f"stale-schema:{plan.schema_version}"
    if not plan.analysis_complete:
        # An exhausted/incomplete sidecar may render safely, but it is
        # never a completed plan: force re-analysis on the next
        # invocation so a later run can retry the unobserved scenes.
        return False, "incomplete-analysis"
    if plan.source_fingerprint != source_fingerprint:
        return False, "stale-source"
    if plan.frame_w != int(frame_w) or plan.frame_h != int(frame_h):
        return False, "stale-dimensions"
    if aspect is not None and abs(plan.output_aspect - float(aspect)) > 1e-9:
        return False, (f"stale-aspect:{plan.output_aspect:.6f}-vs-"
                       f"{float(aspect):.6f}")
    if abs(plan.fps - float(fps)) > 1e-6:
        return False, "stale-fps"
    if plan.frame_count != int(frame_count):
        return False, "stale-frame-count"
    want = [[int(s), int(e)] for s, e in scene_intervals]
    if plan.scene_intervals != want:
        return False, "stale-scenes"
    if strategies is not None:
        want_strategies = [str(x) for x in strategies]
        plan_strategies = [sp.strategy for sp in plan.scenes]
        if plan_strategies != want_strategies:
            return False, "stale-strategies"
    if plan.config_digest != config_digest:
        return False, "stale-config"
    if tracker_digest is not None and plan.tracker_digest != tracker_digest:
        return False, "stale-tracker-config"
    return True, "match"


def source_fingerprint(path: str) -> str:
    """Sampled source identity for plan reuse matching.

    sha256 over ``(size, mtime_ns, first 64 KiB, last 64 KiB)``. This is an
    *identity* check so a plan is never reused for a different source — not a
    content proof and not a substitute for byte preservation (the source is
    never modified).
    """
    st = os.stat(path)
    h = hashlib.sha256()
    h.update(struct.pack(">QQ", st.st_size, st.st_mtime_ns))
    with open(path, "rb") as fh:
        h.update(fh.read(65536))
        if st.st_size > 65536:
            fh.seek(-65536, os.SEEK_END)
            h.update(fh.read(65536))
    return h.hexdigest()


# --- plan build --------------------------------------------------------------

def build_scene_plan(log: ObservationLog, *, scene_id: int, strategy: str,
                     scene_start: int, scene_end: int,
                     config: TurnFramingConfig, fps: float, frame_w: float,
                     frame_h: float, aspect: float) -> ScenePlan:
    """Segment one face scene and attach crops/keyframes.

    Raises ``_PlanBuildError`` when the scene cannot be planned (the caller
    records a fallback with the reason and keeps the accepted static/wide
    behavior for that scene).
    """
    segments, seg_diag = segment_turns(
        log, scene_id=scene_id, scene_start=scene_start, scene_end=scene_end,
        config=config, fps=fps, frame_w=frame_w, frame_h=frame_h,
        aspect=aspect)
    segments, keyframes, kf_diag = build_scene_keyframes(
        segments, log, scene_id, config, frame_w, frame_h, aspect, fps=fps)
    diagnostics = seg_diag + kf_diag
    return ScenePlan(
        scene_id=scene_id, strategy=strategy, start_frame=scene_start,
        end_frame=scene_end, applies=True, scene_kind="face",
        tracks=log.track_summary(scene_id), segments=segments,
        keyframes=keyframes, fallback=None)


def build_framing_plan(*, log: ObservationLog, scenes: list,
                       strategies: list, config: TurnFramingConfig, fps: float,
                       frame_w: float, frame_h: float, frame_count: int,
                       source_fingerprint: str,
                       aspect: float = 9 / 16,
                       tracker_digest: str = "",
                       action_log=None,
                       action_config=None,
                       action_camera_motion=None,
                       action_scene_fallbacks=None,
                       plan_face: bool = True,
                       config_digest: str | None = None) -> FramingPlan:
    """Build one versioned plan for the whole clip.

    Only ``TRACK``/``WIDE`` scenes are face-planned; every other strategy
    keeps its existing renderer (``applies=False``). Per-scene failures are
    isolated: an invalid face scene gets a fallback entry with the reason
    and the renderer keeps the accepted static/wide behavior for it.

    Task 14: when ``action_log`` is given, eligible automatic ``GENERAL``
    scenes are additionally action-planned (see ``action_framing``) and the
    plan is written with schema version 4 and an explicit per-scene
    ``scene_kind`` (``"face"`` / ``"action"`` / ``"none"``). With
    ``action_log=None`` (and ``plan_face=True``) the output is the Task 13
    v3 plan byte-for-byte.

    The plan's identity binds the actual render ``aspect``, the ordered scene
    ``strategies`` (persisted per scene) and the ``tracker_digest`` of the
    tracker decision config that produced the observations, so a sidecar is
    never reused across a different output shape, strategy assignment or
    tracker configuration. ``config_digest`` overrides the computed digest
    (the production path passes the combined turn+action digest so build and
    ``plan_matches`` use one value).
    """
    # Lazy: action_framing imports this module at ITS top level, so this
    # import must stay function-local (no cycle).
    from clippyme.pipeline import action_framing as af

    action_on = action_log is not None
    if action_on and action_config is None:
        action_config = af.ActionFramingConfig()
    # v4 whenever the action path (or a face-planning change) is involved;
    # otherwise the Task 13 v3 bytes are reproduced exactly.
    use_v4 = action_on or not plan_face
    schema_version = SCHEMA_VERSION if use_v4 else 3
    if config_digest is None:
        if action_on:
            config_digest = af.plan_config_digest(
                config.config_digest(), action_config, turn_on=plan_face)
        else:
            config_digest = config.config_digest()
    # Task 12 review 02: structured budget-exhaustion state. The pass-1
    # detector/MAR calls stop on AnalysisBudget expiry, but the planner
    # must not turn partial observations into a normal reusable plan.
    # Scenes the record proves fully analyzed may still be planned; the
    # affected partial scene and every later unobserved face scene get
    # an explicit static/wide fallback, and the plan is marked
    # incomplete so plan_matches() forces re-analysis next run.
    exhaustion_reason = log.exhaustion_reason
    if exhaustion_reason is None:
        for _d in log.diagnostics:
            if _d in ("analysis_budget_exhausted",
                      "analysis-frame-budget-exhausted"):
                exhaustion_reason = _d
                break
    exhaustion_frame = log.exhaustion_frame

    action_exhaustion_reason = None
    action_exhaustion_frame = None
    if action_on:
        action_exhaustion_reason = action_log.exhaustion_reason
        if action_exhaustion_reason is None:
            for _d in action_log.diagnostics:
                if _d in ("analysis_budget_exhausted",
                          "analysis-frame-budget-exhausted"):
                    action_exhaustion_reason = _d
                    break
        action_exhaustion_frame = action_log.exhaustion_frame

    analysis_complete = (exhaustion_reason is None
                         and action_exhaustion_reason is None)

    def _scene_fully_analyzed(s: int, e: int) -> bool:
        if exhaustion_reason is None:
            return True
        # exhaustion_frame is the first frame NOT analyzed: only scenes
        # ending strictly before it were fully analyzed. Without frame
        # context (a bare diagnostic note) nothing can be proven, so
        # every face scene falls back conservatively.
        return (exhaustion_frame is not None
                and int(e) < int(exhaustion_frame))

    def _action_scene_fully_analyzed(s: int, e: int) -> bool:
        if action_exhaustion_reason is None:
            return True
        return (action_exhaustion_frame is not None
                and int(e) < int(action_exhaustion_frame))

    def _exhaustion_fallback(idx: int, strategy: str, s: int, e: int,
                             reason: str) -> ScenePlan:
        return ScenePlan(
            scene_id=idx, strategy=strategy, start_frame=int(s),
            end_frame=int(e), applies=True, scene_kind="face",
            tracks=log.track_summary(idx), fallback={
                "reason": reason,
                "detail": ("pass-1 analysis stopped before this scene "
                           "was fully observed; re-analysis required")})

    def _action_exhaustion_fallback(idx: int, strategy: str, s: int, e: int,
                                    reason: str) -> ScenePlan:
        return ScenePlan(
            scene_id=idx, strategy=strategy, start_frame=int(s),
            end_frame=int(e), applies=True, scene_kind="action",
            tracks=action_log.track_summary(idx), fallback={
                "reason": reason,
                "detail": ("pass-1 action analysis stopped before this "
                           "scene was fully observed; re-analysis "
                           "required")})

    def _build_action_scene(idx: int, strategy: str, s: int, e: int
                            ) -> ScenePlan:
        forced = (action_scene_fallbacks or {}).get(idx)
        if forced is not None:
            reason, detail = forced
            return ScenePlan(
                scene_id=idx, strategy=strategy, start_frame=int(s),
                end_frame=int(e), applies=True, scene_kind="action",
                tracks=action_log.track_summary(idx),
                fallback={"reason": str(reason),
                          "detail": str(detail)[:256]})
        try:
            segments, _seg_diag = af.segment_actions(
                action_log, scene_id=idx, scene_start=int(s),
                scene_end=int(e), config=action_config, fps=fps,
                frame_w=frame_w, frame_h=frame_h, aspect=aspect,
                camera_motion=action_camera_motion)
            segments, keyframes, _kf_diag = af.build_action_keyframes(
                segments, action_log, idx, action_config, frame_w, frame_h,
                aspect, fps=fps)
        except af.ActionPlanError as exc:
            return ScenePlan(
                scene_id=idx, strategy=strategy, start_frame=int(s),
                end_frame=int(e), applies=True, scene_kind="action",
                tracks=action_log.track_summary(idx),
                fallback={"reason": exc.reason, "detail": exc.detail})
        except Exception as exc:  # isolate scene failures, never abort
            return ScenePlan(
                scene_id=idx, strategy=strategy, start_frame=int(s),
                end_frame=int(e), applies=True, scene_kind="action",
                tracks=action_log.track_summary(idx),
                fallback={"reason": "action-segmentation-failed",
                          "detail": type(exc).__name__})
        return ScenePlan(
            scene_id=idx, strategy=strategy, start_frame=int(s),
            end_frame=int(e), applies=True, scene_kind="action",
            tracks=action_log.track_summary(idx), segments=segments,
            keyframes=keyframes, fallback=None)

    scene_plans: list[ScenePlan] = []
    for idx, (start, end) in enumerate(scenes):
        strategy = strategies[idx] if idx < len(strategies) else "TRACK"
        if strategy in FACE_STRATEGIES:
            if not plan_face:
                # Turn-aware planning off: the scene keeps its accepted
                # renderer (the pass-2 smoothed trajectory); the plan
                # records it as a non-applicable face scene.
                scene_plans.append(ScenePlan(
                    scene_id=idx, strategy=strategy, start_frame=int(start),
                    end_frame=int(end), applies=False, scene_kind="face"))
                continue
            # Task 12 review 03: exhaustion fallbacks apply only to
            # TRACK/WIDE face scenes.
            if not _scene_fully_analyzed(int(start), int(end)):
                scene_plans.append(_exhaustion_fallback(
                    idx, strategy, start, end, exhaustion_reason))
                continue
            try:
                sp = build_scene_plan(
                    log, scene_id=idx, strategy=strategy,
                    scene_start=int(start), scene_end=int(end),
                    config=config, fps=fps, frame_w=frame_w,
                    frame_h=frame_h, aspect=aspect)
            except _PlanBuildError as e:
                sp = ScenePlan(
                    scene_id=idx, strategy=strategy, start_frame=int(start),
                    end_frame=int(end), applies=True, scene_kind="face",
                    tracks=log.track_summary(idx), fallback={
                        "reason": e.reason, "detail": e.detail})
            except Exception as e:  # isolate scene failures, never abort
                sp = ScenePlan(
                    scene_id=idx, strategy=strategy, start_frame=int(start),
                    end_frame=int(end), applies=True, scene_kind="face",
                    tracks=log.track_summary(idx), fallback={
                        "reason": "segmentation-failed",
                        "detail": type(e).__name__})
            scene_plans.append(sp)
            continue
        # Task 14: eligible automatic GENERAL scenes get action-planned.
        # Positive SPLIT/SCREENCAST decisions stay authoritative and every
        # other non-face strategy keeps applies=False.
        if action_on and strategy in af.ACTION_STRATEGIES:
            if not _action_scene_fully_analyzed(int(start), int(end)):
                scene_plans.append(_action_exhaustion_fallback(
                    idx, strategy, start, end, action_exhaustion_reason))
            else:
                scene_plans.append(_build_action_scene(idx, strategy,
                                                       start, end))
            continue
        scene_plans.append(ScenePlan(
            scene_id=idx, strategy=strategy, start_frame=int(start),
            end_frame=int(end), applies=False, scene_kind="none"))
    return FramingPlan(
        schema_version=schema_version,
        producer=PLAN_PRODUCER,
        source_fingerprint=source_fingerprint,
        frame_w=int(frame_w), frame_h=int(frame_h), fps=float(fps),
        frame_count=int(frame_count),
        scene_intervals=[[int(s), int(e)] for s, e in scenes],
        config_digest=config_digest,
        output_aspect=float(aspect),
        tracker_digest=str(tracker_digest),
        analysis_complete=analysis_complete,
        scenes=scene_plans,
        created_at=time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime()),
    )
