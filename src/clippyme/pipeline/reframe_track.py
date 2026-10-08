"""Pure tracking/camera classes for the reframe pipeline.

No cv2/torch/mediapipe imports: every decision can be host-unit-tested.  The
speaker tracker uses 2D geometry + IoU identity association, mouth-motion
confidence and relative switch hysteresis so the crop follows the actual speaker
without oscillating between nearby faces or inheriting IDs across crossings.
"""
import hashlib
import json
import math
import os
from collections import deque
from dataclasses import dataclass, field

from clippyme.pipeline.reframe_ops import (
    OneEuroFilter,
    advance_value_with_velocity,
    asymmetric_zoom_step,
    drift_to_center,
    limit_step,
    zoom_for_face_height,
)


class DetectionSmoother:
    """Rolling-average face boxes to suppress detector micro-jitter."""

    def __init__(self, window_size=5):
        self.window_size = window_size
        self.histories = {}
        self.last_seen_frame: dict[int, int] = {}

    def smooth(self, candidates, frame_number):
        smoothed = []
        claimed = set()
        for cand in candidates:
            x, y, w, h = cand["box"]
            cx, cy = x + w / 2, y + h / 2
            best_id = None
            best_cost = float("inf")
            for face_id, history in self.histories.items():
                if face_id in claimed or not history:
                    continue
                lx, ly, lw, lh = history[-1]
                lcx, lcy = lx + lw / 2, ly + lh / 2
                distance = math.hypot(cx - lcx, cy - lcy)
                scale = max(1.0, w, h, lw, lh)
                cost = distance / scale
                if cost < best_cost and cost < 2.0:
                    best_cost = cost
                    best_id = face_id
            if best_id is None:
                best_id = frame_number * 1000 + len(smoothed)
            claimed.add(best_id)
            history = self.histories.setdefault(best_id, deque(maxlen=self.window_size))
            history.append((x, y, w, h))
            self.last_seen_frame[best_id] = frame_number
            smoothed.append({
                **cand,
                "box": [
                    int(sum(box[0] for box in history) / len(history)),
                    int(sum(box[1] for box in history) / len(history)),
                    int(sum(box[2] for box in history) / len(history)),
                    int(sum(box[3] for box in history) / len(history)),
                ],
            })
        stale = [
            face_id for face_id, last in self.last_seen_frame.items()
            if frame_number - last > 60
        ]
        for face_id in stale:
            self.histories.pop(face_id, None)
            self.last_seen_frame.pop(face_id, None)
        return smoothed

    def reset(self):
        self.histories.clear()
        self.last_seen_frame.clear()


class SmoothedCameraman:
    """Adaptive, bounded virtual camera for a tracked face/person box."""

    SMOOTHING_SLOW = 0.08
    SMOOTHING_FAST = 0.30
    FAST_THRESHOLD_RATIO = 0.6
    ZOOM_RATE_IN = 0.05
    ZOOM_RATE_OUT = 0.12

    def __init__(self, output_width, output_height, video_width, video_height,
                 *, aspect_ratio: float = 9 / 16):
        self.output_width = output_width
        self.output_height = output_height
        self.video_width = video_width
        self.video_height = video_height
        self.aspect_ratio = float(aspect_ratio)

        self.max_crop_height = video_height
        self.max_crop_width = int(self.max_crop_height * self.aspect_ratio)
        if self.max_crop_width > video_width:
            self.max_crop_width = video_width
            self.max_crop_height = int(self.max_crop_width / self.aspect_ratio)
        self.min_crop_height = int(self.max_crop_height / 1.6)
        self.min_crop_width = int(self.min_crop_height * self.aspect_ratio)
        self.crop_width = self.max_crop_width
        self.crop_height = self.max_crop_height

        self.current_center_x = video_width / 2
        self.target_center_x = video_width / 2
        self.current_center_y = video_height / 2
        self.target_center_y = video_height / 2
        self.current_zoom = 1.0
        self.target_zoom = 1.0

        self.safe_zone_radius_x = self.max_crop_width * float(os.getenv("REFRAME_DEADZONE_X", "0.05"))
        self.safe_zone_radius_y = self.max_crop_height * float(os.getenv("REFRAME_DEADZONE_Y", "0.08"))
        self.frames_since_target = 0
        self.lost_hold_frames = int(os.getenv("REFRAME_LOST_HOLD", "90"))
        self.lost_drift_rate = float(os.getenv("REFRAME_LOST_DRIFT", "0.05"))

        smoother = os.getenv("REFRAME_SMOOTHER", "").strip().lower()
        self._use_euro = smoother == "euro"
        self._use_spring = smoother == "spring"
        if self._use_euro:
            minimum_cutoff = float(os.getenv("REFRAME_EURO_MINCUTOFF", "0.014"))
            beta = float(os.getenv("REFRAME_EURO_BETA", "0.0008"))
            self._euro_x = OneEuroFilter(min_cutoff=minimum_cutoff, beta=beta)
            self._euro_y = OneEuroFilter(min_cutoff=minimum_cutoff, beta=beta)
        if self._use_spring:
            self._spring_resp = float(os.getenv("REFRAME_SPRING_RESPONSE", "0.18"))
            self._spring_damp = float(os.getenv("REFRAME_SPRING_DAMPING", "0.82"))
            self._vx = 0.0
            self._vy = 0.0
        self._max_step_px = float(os.getenv("REFRAME_MAX_STEP_PX", "0"))
        self._spring_maxv = (
            self._max_step_px if self._max_step_px > 0 else self.max_crop_width * 0.05
        )

    def update_target(self, face_box, is_person_box: bool = False):
        if not face_box:
            return
        self.frames_since_target = 0
        x, y, width, height = face_box
        self.target_center_x = x + width / 2
        if is_person_box:
            self.target_center_y = y + height * 0.15
            self.target_zoom = 1.0
        else:
            self.target_center_y = y + height / 2
            # A wide union/group box must pull back rather than treating its
            # height as a small face and zooming into only the centre speaker.
            if width > height * 1.35:
                self.target_zoom = 1.0
            else:
                self.target_zoom = zoom_for_face_height(
                    height,
                    self.max_crop_height,
                    target_occupancy=0.4,
                    min_zoom=1.0,
                    max_zoom=1.6,
                )

    def _ease_axis(self, current: float, target: float, safe_radius: float, fast_ref: float) -> float:
        difference = target - current
        if abs(difference) <= safe_radius:
            return current
        rate = self.SMOOTHING_FAST if abs(difference) > fast_ref * self.FAST_THRESHOLD_RATIO else self.SMOOTHING_SLOW
        return current + difference * rate

    def get_crop_box(self, force_snap=False):
        if force_snap:
            self.current_center_x = self.target_center_x
            self.current_center_y = self.target_center_y
            self.current_zoom = self.target_zoom
            self.frames_since_target = 0
            if self._use_euro:
                self._euro_x.reset()
                self._euro_y.reset()
            if self._use_spring:
                self._vx = 0.0
                self._vy = 0.0
        else:
            self.frames_since_target += 1
            if self.frames_since_target > self.lost_hold_frames:
                self.target_center_x = drift_to_center(
                    self.target_center_x,
                    self.video_width / 2,
                    self.frames_since_target,
                    self.lost_hold_frames,
                    self.lost_drift_rate,
                )
                self.target_center_y = drift_to_center(
                    self.target_center_y,
                    self.video_height / 2,
                    self.frames_since_target,
                    self.lost_hold_frames,
                    self.lost_drift_rate,
                )
                if self.target_zoom > 1.0:
                    self.target_zoom = max(1.0, self.target_zoom - 0.01)

            previous_x, previous_y = self.current_center_x, self.current_center_y
            target_x = (
                self.target_center_x
                if abs(self.target_center_x - self.current_center_x) > self.safe_zone_radius_x
                else self.current_center_x
            )
            target_y = (
                self.target_center_y
                if abs(self.target_center_y - self.current_center_y) > self.safe_zone_radius_y
                else self.current_center_y
            )
            if self._use_euro:
                self.current_center_x = self._euro_x.filter(target_x, 1.0)
                self.current_center_y = self._euro_y.filter(target_y, 1.0)
            elif self._use_spring:
                self.current_center_x, self._vx = advance_value_with_velocity(
                    self.current_center_x,
                    target_x,
                    self._vx,
                    self._spring_resp,
                    self._spring_damp,
                    self._spring_maxv,
                )
                self.current_center_y, self._vy = advance_value_with_velocity(
                    self.current_center_y,
                    target_y,
                    self._vy,
                    self._spring_resp,
                    self._spring_damp,
                    self._spring_maxv,
                )
            else:
                self.current_center_x = self._ease_axis(
                    self.current_center_x,
                    self.target_center_x,
                    self.safe_zone_radius_x,
                    self.max_crop_width,
                )
                self.current_center_y = self._ease_axis(
                    self.current_center_y,
                    self.target_center_y,
                    self.safe_zone_radius_y,
                    self.max_crop_height,
                )
            if self._max_step_px > 0:
                self.current_center_x = limit_step(previous_x, self.current_center_x, self._max_step_px)
                self.current_center_y = limit_step(previous_y, self.current_center_y, self._max_step_px)
            if abs(self.target_zoom - self.current_zoom) > 0.01:
                self.current_zoom = asymmetric_zoom_step(
                    self.current_zoom,
                    self.target_zoom,
                    self.ZOOM_RATE_IN,
                    self.ZOOM_RATE_OUT,
                )

        self.crop_width = max(self.min_crop_width, int(self.max_crop_width / self.current_zoom))
        self.crop_height = max(self.min_crop_height, int(self.max_crop_height / self.current_zoom))
        half_width = self.crop_width / 2
        half_height = self.crop_height / 2
        center_x = max(half_width, min(self.video_width - half_width, self.current_center_x))
        center_y = max(half_height, min(self.video_height - half_height, self.current_center_y))
        self.current_center_x = center_x
        self.current_center_y = center_y
        return (
            max(0, int(center_x - half_width)),
            max(0, int(center_y - half_height)),
            min(self.video_width, int(center_x + half_width)),
            min(self.video_height, int(center_y + half_height)),
        )

    def crop_box_at(self, cx: float, cy: float, zoom: float):
        crop_width = max(self.min_crop_width, int(self.max_crop_width / zoom))
        crop_height = max(self.min_crop_height, int(self.max_crop_height / zoom))
        half_width = crop_width / 2
        half_height = crop_height / 2
        cx = max(half_width, min(self.video_width - half_width, cx))
        cy = max(half_height, min(self.video_height - half_height, cy))
        return (
            max(0, int(cx - half_width)),
            max(0, int(cy - half_height)),
            min(self.video_width, int(cx + half_width)),
            min(self.video_height, int(cy + half_height)),
        )


def _box_iou(first, second) -> float:
    ax, ay, aw, ah = first
    bx, by, bw, bh = second
    left = max(ax, bx)
    top = max(ay, by)
    right = min(ax + aw, bx + bw)
    bottom = min(ay + ah, by + bh)
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    union = max(0.0, aw * ah) + max(0.0, bw * bh) - intersection
    return intersection / union if union > 0 else 0.0


def _union_box(candidates):
    left = min(item["box"][0] for item in candidates)
    top = min(item["box"][1] for item in candidates)
    right = max(item["box"][0] + item["box"][2] for item in candidates)
    bottom = max(item["box"][1] + item["box"][3] for item in candidates)
    return [left, top, right - left, bottom - top]


@dataclass(frozen=True)
class TrackEvidence:
    """Per-candidate evidence for one frame, from the tracker's own state.

    ``track_id`` is the tracker's anonymous integer identity (the plan layer
    renders it as ``t-<id>``). ``size`` is the size score component,
    ``mouth`` the bounded mouth-motion evidence, ``sustained`` the smoothed
    speaker score. All finite floats.
    """

    track_id: int
    box: list
    size: float
    mouth: float
    sustained: float


@dataclass(frozen=True)
class SubjectDecision:
    """Additive per-frame decision result (Task 12).

    ``subject_ids`` holds the chosen anonymous track id(s) — two when the
    ambiguous-dialogue group crop wins. ``box`` is the chosen crop target box
    (``None`` when held-safe). ``evidence`` carries the named float components
    behind the choice (size/mouth/sustained/margin), ``confidence`` in
    [0, 1], ``ambiguity`` in [0, 1], and ``reason`` a machine-readable token.
    ``tracks`` exposes every matched candidate's evidence for observation
    recording; it is never a label — ids stay anonymous.
    """

    subject_ids: tuple
    box: object
    evidence: dict = field(default_factory=dict)
    confidence: float = 0.0
    ambiguity: float = 1.0
    reason: str = ""
    tracks: tuple = ()


class SpeakerTracker:
    """Track face identities and select a stable active-speaker target.

    Identity matching combines IoU and normalized 2D distance, which is robust to
    two people crossing horizontally.  A challenger must beat the current
    speaker by a configurable relative margin after the cooldown.  When two
    similarly-scored prominent faces are far apart, a temporary group box keeps
    the dialogue reaction in frame instead of snap-cutting between them.
    """

    MAR_WINDOW_SIZE = 25
    SIZE_WEIGHT = 0.30
    MOUTH_WEIGHT = 1.0
    # Task 12: below this mouth-motion evidence a single-subject claim is
    # "weak" — size alone must never crown the speaker.
    WEAK_MOUTH_FLOOR = 0.4

    def __init__(self, cooldown_frames=30):
        self.active_speaker_id = None
        self.speaker_scores = {}
        self.mar_history = {}
        self.last_seen = {}
        self.locked_counter = 0
        self.switch_cooldown = cooldown_frames
        self.last_switch_frame = -1000
        self.next_id = 0
        self.known_faces = []
        self.switch_margin = max(1.0, float(os.getenv("REFRAME_SPEAKER_SWITCH_MARGIN", "1.25")))
        self.min_face_ratio = max(0.0, min(1.0, float(os.getenv("REFRAME_MIN_FACE_RATIO", "0.10"))))
        self.group_enabled = os.getenv("REFRAME_DIALOGUE_GROUP", "1").lower() not in {"0", "false", "no"}
        self.group_score_ratio = max(0.5, min(1.0, float(os.getenv("REFRAME_DIALOGUE_SCORE_RATIO", "0.88"))))
        self.group_separation = max(0.05, min(0.9, float(os.getenv("REFRAME_DIALOGUE_SEPARATION", "0.28"))))

    def decision_config(self) -> dict:
        """The decision-affecting configuration that produced observations.

        Every value here can change which subject the tracker crowns, so the
        turn-aware plan binds its digest (``decision_digest``) into the plan
        identity: a sidecar built under different tracker settings is stale
        and must be rebuilt, never silently reused.
        """
        return {
            "cooldown_frames": self.switch_cooldown,
            "switch_margin": self.switch_margin,
            "min_face_ratio": self.min_face_ratio,
            "group_enabled": self.group_enabled,
            "group_score_ratio": self.group_score_ratio,
            "group_separation": self.group_separation,
            "mar_window_size": self.MAR_WINDOW_SIZE,
            "size_weight": self.SIZE_WEIGHT,
            "mouth_weight": self.MOUTH_WEIGHT,
            "weak_mouth_floor": self.WEAK_MOUTH_FLOOR,
        }

    def decision_digest(self) -> str:
        """Stable digest of ``decision_config`` for plan reuse matching."""
        payload = json.dumps(self.decision_config(), sort_keys=True,
                             separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def reset(self, frame_number=None):
        self.active_speaker_id = None
        self.speaker_scores = {}
        self.mar_history = {}
        self.last_seen = {}
        self.locked_counter = 0
        self.known_faces = []
        if frame_number is not None:
            self.last_switch_frame = frame_number - self.switch_cooldown

    def _match_identity(self, box, frame_number, frame_width, claimed):
        x, y, width, height = box
        center_x, center_y = x + width / 2, y + height / 2
        best_id = None
        best_cost = float("inf")
        for known in self.known_faces:
            if known["id"] in claimed or frame_number - known["last_frame"] > 30:
                continue
            previous = known["box"]
            px, py, pw, ph = previous
            previous_x, previous_y = px + pw / 2, py + ph / 2
            distance = math.hypot(center_x - previous_x, center_y - previous_y)
            scale = max(1.0, width, height, pw, ph, frame_width * 0.03)
            normalized = distance / scale
            iou = _box_iou(box, previous)
            cost = normalized - iou * 0.75
            if cost < best_cost and (iou >= 0.05 or normalized <= 1.8):
                best_cost = cost
                best_id = known["id"]
        if best_id is None:
            best_id = self.next_id
            self.next_id += 1
        return best_id

    def _mouth_motion(self, face_id) -> float:
        history = self.mar_history.get(face_id, [])
        if len(history) < 5:
            return 0.35
        mean = sum(history) / len(history)
        variance = sum((sample - mean) ** 2 for sample in history) / len(history)
        return min(variance * 200.0, 3.0)

    def _update_state(self, face_candidates, frame_number, width):
        """Evolve identity/MAR/score state for one frame (shared core).

        Returns ``None`` when there are no candidates, else a state tuple
        ``(current, ranked, challenger, challenger_score, active, active_score,
        components, frame_area_ref)`` where ``components`` maps track id to
        ``(size_component, mouth_component, sustained_score)``. The body is the
        exact former ``get_target`` state evolution, verbatim.
        """
        # Prune expired identities before matching. Otherwise long
        # streams accumulate an ever-growing candidate list.
        self.known_faces = [
            known
            for known in self.known_faces
            if frame_number - known["last_frame"] <= 30
        ]
        if not face_candidates:
            return None
        maximum_area = max(
            max(0.0, float(face.get("score") or face["box"][2] * face["box"][3]))
            for face in face_candidates
        )
        candidates = [
            face for face in face_candidates
            if float(face.get("score") or face["box"][2] * face["box"][3])
            >= maximum_area * self.min_face_ratio
        ]

        current = []
        claimed = set()
        for face in sorted(candidates, key=lambda item: float(item.get("score") or 0), reverse=True):
            face_id = self._match_identity(face["box"], frame_number, width, claimed)
            claimed.add(face_id)
            self.known_faces = [known for known in self.known_faces if known["id"] != face_id]
            self.known_faces.append({
                "id": face_id,
                "box": list(face["box"]),
                "last_frame": frame_number,
            })
            candidate = {
                "id": face_id,
                "box": face["box"],
                "score": float(face.get("score") or face["box"][2] * face["box"][3]),
                "mar": face.get("mar"),
            }
            current.append(candidate)
            self.last_seen[face_id] = frame_number
            if candidate["mar"] is not None:
                history = self.mar_history.setdefault(face_id, [])
                history.append(float(candidate["mar"]))
                if len(history) > self.MAR_WINDOW_SIZE:
                    del history[:-self.MAR_WINDOW_SIZE]

        visible_ids = {candidate["id"] for candidate in current}
        for face_id in list(self.speaker_scores):
            if face_id not in visible_ids:
                self.speaker_scores[face_id] *= 0.82
            if frame_number - self.last_seen.get(face_id, frame_number) > 60:
                self.speaker_scores.pop(face_id, None)
                self.mar_history.pop(face_id, None)
                self.last_seen.pop(face_id, None)

        frame_area_ref = max(1.0, width * width * 0.05)
        components = {}
        for candidate in current:
            face_id = candidate["id"]
            size_component = self.SIZE_WEIGHT * candidate["score"] / frame_area_ref
            mouth_component = self._mouth_motion(face_id)
            instant = size_component + self.MOUTH_WEIGHT * mouth_component
            previous = self.speaker_scores.get(face_id, instant)
            self.speaker_scores[face_id] = previous * 0.72 + instant * 0.28
            components[face_id] = (
                size_component, mouth_component, self.speaker_scores[face_id])

        ranked = sorted(current, key=lambda item: self.speaker_scores.get(item["id"], 0.0), reverse=True)
        challenger = ranked[0]
        challenger_score = self.speaker_scores.get(challenger["id"], 0.0)
        active = next((item for item in current if item["id"] == self.active_speaker_id), None)
        active_score = self.speaker_scores.get(self.active_speaker_id, 0.0)
        return (current, ranked, challenger, challenger_score, active,
                active_score, components, frame_area_ref)

    def _select(self, state, frame_number, width, weak_guard=False):
        """Shared decision on one frame's updated state.

        Returns ``(kind, subject_ids, box)``. ``kind`` is one of
        ``group``/``acquire``/``locked``/``switch``/``held``/``held-safe`` —
        plus ``weak-group``/``weak-hold`` when ``weak_guard`` is on. ``box``
        is exactly what the former ``get_target`` returned (``weak_guard``
        off). ``weak_guard`` (the ``decide()`` path only) enforces the Task 12
        rule *before* any state mutation: size alone cannot prove the active
        speaker, so claiming a NEW single subject requires its mouth-motion
        evidence to clear ``WEAK_MOUTH_FLOOR``; otherwise the decision
        degrades to a group crop or held-safe. ``locked`` is exempt — holding
        the incumbent is already safe framing.
        """
        (current, ranked, challenger, challenger_score, active,
         active_score, components, _ref) = state
        # Ambiguous dialogue/reaction: keep both prominent faces centred. It is
        # deliberately disabled while one mouth score decisively wins.
        if self.group_enabled and len(ranked) >= 2:
            second = ranked[1]
            second_score = self.speaker_scores.get(second["id"], 0.0)
            first_center = ranked[0]["box"][0] + ranked[0]["box"][2] / 2
            second_center = second["box"][0] + second["box"][2] / 2
            if (
                challenger_score > 0
                and second_score / challenger_score >= self.group_score_ratio
                and abs(first_center - second_center) >= width * self.group_separation
            ):
                return ("group", (ranked[0]["id"], ranked[1]["id"]),
                        _union_box(ranked[:2]))

        def _weak_downgrade(single_id):
            """Weak-evidence downgrade for a would-be single claim.

            Returns ``("weak-group", ids, box)`` / ``("weak-hold", (), None)``
            or ``None`` when the claim is allowed. Never mutates state.
            """
            if not weak_guard or len(current) < 2:
                return None
            if components[single_id][1] >= self.WEAK_MOUTH_FLOOR:
                return None
            order = sorted(current,
                           key=lambda c: components[c["id"]][2], reverse=True)
            first, second = order[0], order[1]
            first_cx = first["box"][0] + first["box"][2] / 2.0
            second_cx = second["box"][0] + second["box"][2] / 2.0
            if abs(first_cx - second_cx) >= width * self.group_separation:
                return ("weak-group", (first["id"], second["id"]),
                        _union_box([first, second]))
            return ("weak-hold", (), None)

        if active is None:
            if self.active_speaker_id is not None and frame_number - self.last_switch_frame < self.switch_cooldown:
                return ("held-safe", (), None)
            downgrade = _weak_downgrade(challenger["id"])
            if downgrade is not None:
                return downgrade
            self.active_speaker_id = challenger["id"]
            self.last_switch_frame = frame_number
            self.locked_counter = 0
            return ("acquire", (challenger["id"],), challenger["box"])

        if challenger["id"] == self.active_speaker_id:
            self.locked_counter += 1
            return ("locked", (challenger["id"],), active["box"])

        cooldown_done = frame_number - self.last_switch_frame >= self.switch_cooldown
        decisive = challenger_score >= max(0.05, active_score * self.switch_margin)
        if cooldown_done and decisive:
            downgrade = _weak_downgrade(challenger["id"])
            if downgrade is not None:
                return downgrade
            self.active_speaker_id = challenger["id"]
            self.last_switch_frame = frame_number
            self.locked_counter = 0
            return ("switch", (challenger["id"],), challenger["box"])

        return ("held", (active["id"],), active["box"])

    def get_target(self, face_candidates, frame_number, width):
        """Legacy box-only API — preserved exactly (delegates to the shared
        state update + decision)."""
        state = self._update_state(face_candidates, frame_number, width)
        if state is None:
            return None
        _kind, _ids, box = self._select(state, frame_number, width,
                                        weak_guard=False)
        return box

    @staticmethod
    def _clamp01(value: float) -> float:
        return min(1.0, max(0.0, value))

    def _group_decision(self, first_id, second_id, box, components, current,
                        reason) -> SubjectDecision:
        """Build the SubjectDecision for a two-subject group crop."""
        ids = (first_id, second_id)
        chosen_sus = sum(components[i][2] for i in ids) / len(ids)
        chosen_mouth = sum(components[i][1] for i in ids) / len(ids)
        chosen_size = sum(components[i][0] for i in ids) / len(ids)
        sa = components[first_id][2]
        sb = components[second_id][2]
        closeness = self._clamp01(min(sa, sb) / max(max(sa, sb), 1e-6))
        tracks = tuple(
            TrackEvidence(
                track_id=c["id"], box=list(c["box"]),
                size=components[c["id"]][0], mouth=components[c["id"]][1],
                sustained=components[c["id"]][2])
            for c in current)
        return SubjectDecision(
            subject_ids=tuple(ids),
            box=list(box) if box is not None else None,
            evidence={
                "size": chosen_size,
                "mouth": chosen_mouth,
                "sustained": chosen_sus,
                "runner_up_sustained": 0.0,
                "margin": 1.0,
                "pair_closeness": closeness,
            },
            confidence=self._clamp01(
                0.30 + 0.40 * closeness
                + 0.30 * self._clamp01(chosen_mouth / 1.5)),
            ambiguity=closeness,
            reason=reason,
            tracks=tracks,
        )

    def decide(self, face_candidates, frame_number, width) -> SubjectDecision:
        """Additive explainable decision for one frame (Task 12).

        Runs the same state update + decision as ``get_target`` (identical
        ``box`` for identical input/state) and additionally exposes the chosen
        subject id(s), per-candidate evidence, confidence, ambiguity and a
        machine-readable reason. Size alone can never prove the active speaker:
        a switch requires the challenger's *sustained* score — which blends
        mouth-motion evidence — to beat the incumbent by ``switch_margin``
        after the cooldown; weak evidence yields a group/wide or held-safe
        decision with high ambiguity.
        """
        state = self._update_state(face_candidates, frame_number, width)
        if state is None:
            return SubjectDecision(
                subject_ids=(), box=None, evidence={}, confidence=0.0,
                ambiguity=1.0, reason="no-candidates", tracks=())
        (current, ranked, _challenger, _challenger_score, _active,
         _active_score, components, _ref) = state
        # weak_guard=True: the Task 12 rule is enforced inside _select, before
        # any tracker state mutation, so a weak single claim never crowns.
        kind, ids, box = self._select(state, frame_number, width,
                                      weak_guard=True)

        tracks = tuple(
            TrackEvidence(
                track_id=c["id"], box=list(c["box"]),
                size=components[c["id"]][0], mouth=components[c["id"]][1],
                sustained=components[c["id"]][2])
            for c in current)

        if not ids:
            return SubjectDecision(
                subject_ids=(), box=None, evidence={}, confidence=0.0,
                ambiguity=1.0, reason="cooldown-hold-no-active", tracks=tracks)

        chosen_sus = sum(components[i][2] for i in ids) / len(ids)
        chosen_mouth = sum(components[i][1] for i in ids) / len(ids)
        chosen_size = sum(components[i][0] for i in ids) / len(ids)
        runner_sus = max(
            (components[c["id"]][2] for c in ranked if c["id"] not in ids),
            default=0.0)
        if kind == "weak-hold":
            return SubjectDecision(
                subject_ids=(), box=None, evidence={}, confidence=0.0,
                ambiguity=1.0, reason="weak-evidence-hold", tracks=tracks)
        margin = chosen_sus / max(runner_sus, 1e-6)
        if kind in ("group", "weak-group"):
            # A group crop IS the ambiguous outcome: ambiguity measures how
            # close the paired subjects' evidence is (the reason no single
            # speaker could be chosen), and confidence measures the call that
            # grouping — not picking — was right.
            return self._group_decision(
                ids[0], ids[1], box, components, current,
                "weak-evidence-group" if kind == "weak-group"
                else "ambiguous-dialogue-two-subject")
        decisiveness = self._clamp01(
            (chosen_sus - runner_sus) / max(chosen_sus, 1e-6))
        confidence = self._clamp01(
            0.35 + 0.40 * decisiveness + 0.25 * self._clamp01(chosen_mouth / 1.5))
        ambiguity = self._clamp01(1.0 - decisiveness)
        reasons = {
            "acquire": "initial-acquire",
            "locked": "speaker-locked",
            "switch": "turn-switch-sustained-evidence",
            "held": "incumbent-held-weak-challenger",
            "held-safe": "cooldown-hold-no-active",
        }
        return SubjectDecision(
            subject_ids=tuple(ids),
            box=list(box) if box is not None else None,
            evidence={
                "size": chosen_size,
                "mouth": chosen_mouth,
                "sustained": chosen_sus,
                "runner_up_sustained": runner_sus,
                "margin": margin,
            },
            confidence=confidence,
            ambiguity=ambiguity,
            reason=reasons.get(kind, kind),
            tracks=tracks,
        )
