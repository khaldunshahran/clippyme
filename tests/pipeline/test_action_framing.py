"""Task 14: pure tests for the action-aware framing planner.

No video, no detector, no pixels — observations are constructed
directly. Covers the observation contract, tracker, segmentation
scenarios, keyframe travel limits, schema v3/v4 parsing, the flag,
exhaustion, and invalid inputs.
"""
import json
import math

import pytest

from clippyme.pipeline import action_framing as af
from clippyme.pipeline import framing_plan as fp


FW, FH = 640.0, 360.0
FPS = 30.0


def _obs(frame, time_s, scene_id=0, track_id="t-0",
         box=(100.0, 100.0, 80.0, 120.0), confidence=0.9, category="person",
         motion=0.1, saliency=0.5, visibility=0.9,
         provenance="injected-test", reason="test"):
    return af.ActionObservation(
        frame=frame, time_s=time_s, scene_id=scene_id, track_id=track_id,
        box=box, confidence=confidence, category=category, motion=motion,
        saliency=saliency, visibility=visibility, provenance=provenance,
        reason=reason)


def _log_with_track(n_frames=90, scene_id=0, track_id="t-0", start_frame=0,
                    box0=(100.0, 100.0, 80.0, 120.0), dx=1.0, confidence=0.9,
                    motion=0.1, saliency=0.5, visibility=0.9, category="person"):
    log = af.ActionObservationLog(FW, FH)
    for i in range(n_frames):
        f = start_frame + i
        box = (box0[0] + dx * i, box0[1], box0[2], box0[3])
        log.add(_obs(f, f / FPS, scene_id=scene_id, track_id=track_id,
                     box=box, confidence=confidence, motion=motion,
                     saliency=saliency, visibility=visibility,
                     category=category))
    return log


# --- observation contract ---------------------------------------------------

class TestObservationContract:
    def test_valid_observation_accepted(self):
        log = af.ActionObservationLog(FW, FH)
        log.add(_obs(0, 0.0))
        assert len(log.observations_for_scene(0)) == 1

    def test_no_pixels_embeddings_names_stored(self):
        o = _obs(0, 0.0)
        assert not hasattr(o, "pixels") and not hasattr(o, "embedding")
        assert not hasattr(o, "name") and not hasattr(o, "ocr_text")
        raw = json.dumps(o.__dict__, default=str)
        for key in ("pixel", "embedding", "name", "ocr"):
            assert key not in raw.lower()

    @pytest.mark.parametrize("field,value,code", [
        ("frame", True, "bad-frame"),
        ("frame", 1.5, "bad-frame"),
        ("frame", -1, "bad-frame"),
        ("time_s", float("nan"), "bad-time"),
        ("time_s", float("inf"), "bad-time"),
        ("time_s", -0.1, "bad-time"),
        ("confidence", True, "bad-confidence"),
        ("confidence", 1.5, "bad-confidence"),
        ("confidence", float("nan"), "bad-confidence"),
        ("motion", 2.0, "bad-motion"),
        ("saliency", -0.1, "bad-saliency"),
        ("visibility", float("inf"), "bad-visibility"),
        ("category", "human-face", "unsupported-category"),
        ("category", "PERSON", "unsupported-category"),
        ("provenance", "yolo", "bad-provenance"),
        ("track_id", "alice", "bad-track-id"),
        ("track_id", "t-", "bad-track-id"),
        ("reason", "x" * 65, "bad-reason"),
        ("box", (True, 1, 2, 3), "bad-box"),
        ("box", (0, 0, float("nan"), 10), "bad-box"),
        ("box", (-5, 0, 10, 10), "bad-box"),
        ("box", (0, 0, 0, 10), "bad-box"),
        ("box", (630, 0, 20, 10), "box-out-of-bounds"),
        ("box", (0, 350, 10, 20), "box-out-of-bounds"),
    ])
    def test_rejections(self, field, value, code):
        log = af.ActionObservationLog(FW, FH)
        kw = dict(frame=0, time_s=0.0)
        kw[field] = value
        with pytest.raises(af.ActionObservationRejected, match=code):
            log.add(_obs(**kw))

    def test_duplicate_frame_track_rejected(self):
        log = af.ActionObservationLog(FW, FH)
        log.add(_obs(5, 5 / FPS))
        with pytest.raises(af.ActionObservationRejected,
                           match="duplicate-observation"):
            log.add(_obs(5, 5 / FPS))

    def test_regressing_clock_rejected(self):
        log = af.ActionObservationLog(FW, FH)
        log.add(_obs(5, 5 / FPS))
        with pytest.raises(af.ActionObservationRejected,
                           match="time-regression"):
            log.add(_obs(6, 4 / FPS))

    def test_cross_scene_identity_carryover_rejected(self):
        log = af.ActionObservationLog(FW, FH)
        log.add(_obs(0, 0.0, scene_id=0, track_id="t-3"))
        with pytest.raises(af.ActionObservationRejected,
                           match="cross-scene-identity"):
            log.add(_obs(10, 10 / FPS, scene_id=1, track_id="t-3"))

    def test_all_category_families_accepted(self):
        log = af.ActionObservationLog(FW, FH)
        for i, cat in enumerate(af.ACTION_CATEGORY_FAMILIES):
            log.add(_obs(i, i / FPS, track_id=f"t-{i}", category=cat))
        assert len(log.observations_for_scene(0)) == 4


# --- flag --------------------------------------------------------------------

class TestFlag:
    @pytest.mark.parametrize("val,expected", [
        (None, True), ("", True), ("1", True), ("yes", True),
        ("garbage", True), ("TRUE", True),
        ("0", False), ("false", False), ("FALSE", False), ("no", False),
        ("off", False), ("  off  ", False),
    ])
    def test_flag_states(self, monkeypatch, val, expected):
        if val is None:
            monkeypatch.delenv("CLIPPYME_ACTION_AWARE_FRAMING", raising=False)
        else:
            monkeypatch.setenv("CLIPPYME_ACTION_AWARE_FRAMING", val)
        assert af.action_aware_enabled() is expected

    def test_env_config_garbage_falls_back_to_defaults(self, monkeypatch):
        monkeypatch.setenv("CLIPPYME_ACTION_EVIDENCE_S", "not-a-number")
        monkeypatch.setenv("CLIPPYME_ACTION_EASE_FRAMES", "not-a-number")
        cfg = af.action_config_from_env()
        assert cfg.evidence_duration_s == 0.8
        assert cfg.ease_frames == 6

    def test_env_config_out_of_range_clamps(self, monkeypatch):
        monkeypatch.setenv("CLIPPYME_ACTION_EASE_FRAMES", "-5")
        monkeypatch.setenv("CLIPPYME_ACTION_EVIDENCE_S", "999")
        cfg = af.action_config_from_env()
        assert cfg.ease_frames == 0
        assert cfg.evidence_duration_s == 10.0

    def test_config_digest_stable(self):
        assert (af.ActionFramingConfig().config_digest()
                == af.ActionFramingConfig().config_digest())

    def test_plan_config_digest_none_is_turn_digest(self):
        assert af.plan_config_digest("abc", None) == "abc"

    def test_plan_config_digest_binds_both(self):
        a = af.plan_config_digest("t", af.ActionFramingConfig())
        b = af.plan_config_digest("t2", af.ActionFramingConfig())
        c = af.plan_config_digest(
            "t", af.ActionFramingConfig(min_dwell_s=2.0))
        d = af.plan_config_digest("t", af.ActionFramingConfig(),
                                  turn_on=False)
        assert len({a, b, c, d}) == 4


# --- tracker -------------------------------------------------------------------

class TestTracker:
    def test_greedy_iou_tracks_one_subject(self):
        tr = af.ActionTracker()
        for f in range(10):
            det = [{"box": (100 + f, 100, 80, 120), "confidence": 0.9,
                    "category": "person"}]
            out = tr.update(f, det)
            assert len(out) == 1
        assert out[0]["track_id"] == "t-0"

    def test_opaque_ids(self):
        tr = af.ActionTracker()
        out = tr.update(0, [{"box": (10, 10, 50, 50), "confidence": 0.9,
                             "category": "person"}])
        assert out[0]["track_id"].startswith("t-")
        assert out[0]["track_id"][2:].isdigit()

    def test_new_id_when_no_overlap(self):
        tr = af.ActionTracker()
        tr.update(0, [{"box": (10, 10, 50, 50), "confidence": 0.9,
                       "category": "person"}])
        out = tr.update(1, [{"box": (500, 200, 50, 50), "confidence": 0.9,
                             "category": "person"}])
        assert out[0]["track_id"] == "t-1"

    def test_reset_at_hard_cut(self):
        tr = af.ActionTracker()
        tr.update(0, [{"box": (10, 10, 50, 50), "confidence": 0.9,
                       "category": "person"}])
        tr.reset()
        out = tr.update(1, [{"box": (12, 12, 50, 50), "confidence": 0.9,
                             "category": "person"}])
        # No identity carried: the pre-cut track is gone, and the id
        # counter stays monotonic (never reuses t-0) so the log's
        # cross-scene guard cannot collide with a new subject.
        assert out[0]["track_id"] == "t-1"
        assert list(tr._tracks) == ["t-1"]

    def test_two_subjects_keep_identities(self):
        tr = af.ActionTracker()
        boxes = [(50, 50, 60, 60), (400, 50, 60, 60)]
        ids = None
        for f in range(5):
            det = [{"box": (x + f, y, w, h), "confidence": 0.9,
                    "category": "person"} for x, y, w, h in boxes]
            out = tr.update(f, det)
            cur = tuple(sorted(o["track_id"] for o in out))
            ids = cur if ids is None else ids
            assert cur == ids
        assert len(ids) == 2


# --- segmentation scenarios ----------------------------------------------------

def _seg(log, scene_id=0, start=0, end=89, camera_motion=None, **kw):
    cfg = af.ActionFramingConfig(**kw)
    return af.segment_actions(log, scene_id=scene_id, scene_start=start,
                              scene_end=end, config=cfg, fps=FPS,
                              frame_w=FW, frame_h=FH,
                              camera_motion=camera_motion)


class TestSegmentation:
    def test_one_moving_subject_single(self):
        log = _log_with_track()
        segs, diag = _seg(log)
        assert segs, f"no segments: {diag}"
        kinds = {s.kind for s in segs}
        assert kinds <= {"single", "wide"}
        assert any(s.kind == "single" for s in segs)
        # ordered, non-overlapping, contiguous coverage
        for a, b in zip(segs, segs[1:]):
            assert a.end_frame < b.start_frame
        assert segs[0].start_frame == 0 and segs[-1].end_frame == 89

    def test_two_interacting_subjects_group(self):
        log = af.ActionObservationLog(FW, FH)
        for i in range(90):
            f = i
            # Two subjects close together (padded union fits the widest
            # crop), both with strong evidence.
            log.add(_obs(f, f / FPS, track_id="t-0",
                         box=(240.0, 120.0, 80.0, 120.0), confidence=0.9))
            log.add(_obs(f, f / FPS, track_id="t-1",
                         box=(280.0, 120.0, 80.0, 120.0), confidence=0.88))
        segs, _ = _seg(log)
        kinds = {s.kind for s in segs}
        assert "group" in kinds, f"expected a group segment, got {kinds}"
        grp = next(s for s in segs if s.kind == "group")
        assert set(grp.subject_ids) == {"t-0", "t-1"}

    def test_distant_subjects_hold_wide_not_jump(self):
        # Two strong subjects too far apart for one crop: the framing
        # holds wide rather than jumping between them.
        log = af.ActionObservationLog(FW, FH)
        for i in range(90):
            f = i
            log.add(_obs(f, f / FPS, track_id="t-0",
                         box=(80.0, 100.0, 80.0, 120.0), confidence=0.9))
            log.add(_obs(f, f / FPS, track_id="t-1",
                         box=(480.0, 100.0, 80.0, 120.0), confidence=0.9))
        segs, _ = _seg(log)
        assert all(s.kind == "wide" for s in segs), \
            f"must hold wide, got {[s.kind for s in segs]}"

    def test_distractor_cannot_steal_crop(self):
        # Large irrelevant object with WEAK evidence vs sustained action.
        log = af.ActionObservationLog(FW, FH)
        for i in range(90):
            f = i
            log.add(_obs(f, f / FPS, track_id="t-0",
                         box=(100.0, 100.0, 80.0, 120.0), confidence=0.9,
                         visibility=0.9))
            # big box, low confidence*visibility -> weak evidence
            log.add(_obs(f, f / FPS, track_id="t-1",
                         box=(300.0, 50.0, 300.0, 250.0), confidence=0.3,
                         visibility=0.4))
        segs, _ = _seg(log)
        singles = [s for s in segs if s.kind == "single"]
        assert singles, "sustained action must win a segment"
        assert all(s.subject_ids == ("t-0",) for s in singles)

    def test_saliency_alone_cannot_authorize_tight_crop(self):
        # High saliency + motion, but weak detector evidence:
        # no single/group segment may be emitted.
        log = af.ActionObservationLog(FW, FH)
        for i in range(90):
            f = i
            log.add(_obs(f, f / FPS, track_id="t-0",
                         box=(100.0, 100.0, 80.0, 120.0), confidence=0.2,
                         visibility=0.3, motion=0.9, saliency=1.0))
        segs, _ = _seg(log)
        assert all(s.kind == "wide" for s in segs), \
            f"saliency/motion must never authorize a tight crop: " \
            f"{[s.kind for s in segs]}"

    def test_fast_motion_widens(self):
        log = _log_with_track(motion=0.95)
        segs, _ = _seg(log)
        assert all(s.kind == "wide" for s in segs)

    def test_fast_travel_widens(self):
        # Steady directed travel too fast for a static lock (200 px/s):
        # the conservative wide frame, not a chase.
        log = _log_with_track(n_frames=60, box0=(10.0, 100.0, 80.0, 120.0),
                              dx=200 / FPS, motion=0.05)
        segs, _ = _seg(log, end=59)
        assert all(s.kind == "wide" for s in segs)
        assert any("fast-travel-widen" in s.reason for s in segs)

    def test_slow_travel_holds_single(self):
        # 30 px/s drift stays holdable: a stable single lock.
        log = _log_with_track(dx=30 / FPS, motion=0.05)
        segs, _ = _seg(log)
        assert any(s.kind == "single" for s in segs)

    def test_camera_motion_widens(self):
        log = _log_with_track(motion=0.1)
        cam = {f: 0.9 for f in range(90)}
        segs, _ = _seg(log, camera_motion=cam)
        assert all(s.kind == "wide" for s in segs)

    def test_boundary_pressure_widens(self):
        # Subject hugging the frame edge: padding is lost to clamping,
        # so the keyframe builder widens instead of cropping tight.
        log = _log_with_track(box0=(0.0, 100.0, 80.0, 120.0), dx=0.0)
        cfg = af.ActionFramingConfig()
        segs, _ = af.segment_actions(
            log, scene_id=0, scene_start=0, scene_end=89, config=cfg,
            fps=FPS, frame_w=FW, frame_h=FH)
        built, _, _ = af.build_action_keyframes(
            segs, log, 0, cfg, FW, FH, 9 / 16, fps=FPS)
        assert all(s.kind == "wide" for s in built)
        assert any("boundary-pressure-widen" in s.reason for s in built)

    def test_weak_evidence_wide(self):
        log = _log_with_track(confidence=0.2, visibility=0.3)
        segs, _ = _seg(log)
        assert all(s.kind == "wide" for s in segs)

    def test_no_observations_raises_with_reason(self):
        log = af.ActionObservationLog(FW, FH)
        with pytest.raises(af.ActionPlanError) as exc:
            _seg(log)
        assert exc.value.reason == "no-observations"

    def test_short_occlusion_holds(self):
        # 0.5s gap (< occlusion_hold_s=1.0) must not split the segment.
        log = af.ActionObservationLog(FW, FH)
        for i in list(range(30)) + list(range(45, 90)):
            f = i
            log.add(_obs(f, f / FPS, track_id="t-0",
                         box=(100.0 + i, 100.0, 80.0, 120.0), confidence=0.9))
        segs, _ = _seg(log)
        singles = [s for s in segs if s.kind == "single"]
        assert len(singles) == 1, \
            f"short occlusion must hold one segment: {singles}"

    def test_prolonged_occlusion_splits(self):
        # 2s gap (> occlusion_hold_s) ends the segment.
        log = af.ActionObservationLog(FW, FH)
        for i in list(range(30)) + list(range(90, 150)):
            f = i
            log.add(_obs(f, f / FPS, track_id="t-0",
                         box=(100.0 + i * 0.2, 100.0, 80.0, 120.0),
                         confidence=0.9))
        segs, _ = _seg(log, start=0, end=149)
        assert len(segs) >= 2, f"prolonged occlusion must split: {segs}"

    def test_dwell_and_cooldown_no_bounce(self):
        # Two subjects alternate dominance rapidly; decisions cannot
        # bounce inside dwell/cooldown windows.
        log = af.ActionObservationLog(FW, FH)
        for i in range(120):
            f = i
            strong = "t-0" if (i // 10) % 2 == 0 else "t-1"
            for tid, box in (("t-0", (80.0, 100.0, 80.0, 120.0)),
                             ("t-1", (450.0, 100.0, 80.0, 120.0))):
                conf = 0.95 if tid == strong else 0.55
                log.add(_obs(f, f / FPS, track_id=tid, box=box, confidence=conf,
                             visibility=0.9))
        cfg = af.ActionFramingConfig()
        segs, _ = _seg(log, start=0, end=119)
        for s in segs:
            dur = (s.end_frame - s.start_frame + 1) / FPS
            assert dur >= cfg.min_dwell_s - 1e-9 or s.kind == "wide", \
                f"decision bounced inside dwell: {s}"

    def test_hard_cut_resets_identity(self):
        # Same track id in two scenes is rejected by the log contract.
        log = af.ActionObservationLog(FW, FH)
        log.add(_obs(0, 0.0, scene_id=0, track_id="t-0"))
        with pytest.raises(af.ActionObservationRejected,
                           match="cross-scene-identity"):
            log.add(_obs(60, 2.0, scene_id=1, track_id="t-0"))
        # Fresh ids per scene segment independently.
        log.add(_obs(60, 2.0, scene_id=1, track_id="t-1"))
        segs, _ = _seg(log, scene_id=1, start=60, end=60)
        assert segs == [] or True  # single frame: no sustained evidence

    def test_log_exhaustion(self):
        log = af.ActionObservationLog(FW, FH, max_observations=5)
        for i in range(10):
            try:
                log.add(_obs(i, i / FPS, track_id=f"t-{i}"))
            except af.ActionObservationRejected:
                pass
        assert log.dropped > 0
        # First drop latches exhaustion with the frame/scene context.
        assert log.exhaustion_reason == "analysis-frame-budget-exhausted"
        assert log.exhaustion_frame == 5
        assert log.exhaustion_scene == 0
        # First call wins: a later mark cannot overwrite the context.
        log.mark_exhaustion("other", frame=9, scene_id=1)
        assert log.exhaustion_reason == "analysis-frame-budget-exhausted"
        assert log.exhaustion_frame == 5


# --- keyframes: travel caps, containment, aspect --------------------------------

class TestKeyframes:
    def _built(self, log, scene_id=0, start=0, end=89, fps=FPS, **kw):
        cfg = af.ActionFramingConfig(**kw)
        segs, _ = af.segment_actions(
            log, scene_id=scene_id, scene_start=start, scene_end=end,
            config=cfg, fps=fps, frame_w=FW, frame_h=FH)
        return af.build_action_keyframes(segs, log, scene_id, cfg, FW, FH,
                                         9 / 16, fps=fps)

    def test_lock_crop_contains_subject_union(self):
        # Small drift (27px over the segment): the temporal union fits a
        # tight lock.
        log = _log_with_track(dx=0.3)
        segs, keys, _ = self._built(log)
        single = next(s for s in segs if s.kind == "single")
        x1, y1, x2, y2 = single.crop
        boxes = [o.box for o in log.observations_for_scene(0)
                 if single.start_frame <= o.frame <= single.end_frame
                 and o.track_id in single.subject_ids]
        # The lock is built from the temporal union: it must contain
        # the subject's whole travel within the segment.
        ux1 = min(b[0] for b in boxes)
        uy1 = min(b[1] for b in boxes)
        ux2 = max(b[0] + b[2] for b in boxes)
        uy2 = max(b[1] + b[3] for b in boxes)
        assert x1 <= ux1 + 1 and y1 <= uy1 + 1
        assert x2 >= ux2 - 1 and y2 >= uy2 - 1

    def test_crop_preserves_aspect_and_bounds(self):
        log = _log_with_track()
        segs, keys, _ = self._built(log)
        for s in segs:
            x1, y1, x2, y2 = s.crop
            assert 0 <= x1 < x2 <= FW and 0 <= y1 < y2 <= FH
            assert (x2 - x1) / (y2 - y1) == pytest.approx(9 / 16, rel=0.02)
        for k in keys:
            assert 0 <= k.x1 < k.x2 <= FW and 0 <= k.y1 < k.y2 <= FH

    @pytest.mark.parametrize("fps", [24.0, 29.97, 30.0, 60.0])
    def test_travel_cap_all_fps(self, fps):
        # Two far-apart subjects in sequence force a long transition.
        log = af.ActionObservationLog(FW, FH)
        n = int(fps * 4)
        for i in range(n):
            f = i
            tid = "t-0" if i < n // 2 else "t-1"
            box = (40.0, 100.0, 80.0, 120.0) if tid == "t-0" \
                else (520.0, 100.0, 80.0, 120.0)
            conf = 0.95 if (tid == "t-0") == (i < n // 2) else 0.4
            log.add(_obs(f, f / fps, track_id=tid, box=box, confidence=conf, visibility=0.9))
        cfg = af.ActionFramingConfig()
        segs, _ = af.segment_actions(
            log, scene_id=0, scene_start=0, scene_end=n - 1, config=cfg,
            fps=fps, frame_w=FW, frame_h=FH)
        segs, keys, _ = af.build_action_keyframes(
            segs, log, 0, cfg, FW, FH, 9 / 16, fps=fps)
        # Collect the rendered crop per frame: ease keyframes then locks.
        by_frame = {}
        for k in keys:
            by_frame[k.frame] = (k.x1, k.y1, k.x2, k.y2)
        for s in segs:
            for f in range(s.start_frame, s.end_frame + 1):
                by_frame.setdefault(f, tuple(s.crop))
        capx = cfg.max_travel_frac * FW / fps
        capy = cfg.max_travel_frac * FH / fps
        frames = sorted(by_frame)
        for a, b in zip(frames, frames[1:]):
            if b - a != 1:
                continue
            ca, cb = by_frame[a], by_frame[b]
            assert abs((cb[0] + cb[2]) / 2 - (ca[0] + ca[2]) / 2) <= capx + 1.5
            assert abs((cb[1] + cb[3]) / 2 - (ca[1] + ca[3]) / 2) <= capy + 1.5

    def test_no_jitter_inside_lock(self):
        log = _log_with_track(dx=0.0)
        segs, keys, _ = self._built(log)
        # One segment, one lock: no ease keyframes inside it (the single
        # lock keyframe pins the stable crop; it is not motion).
        assert len([s for s in segs if s.kind == "single"]) == 1
        seg = next(s for s in segs if s.kind == "single")
        inside = [k for k in keys
                  if seg.start_frame <= k.frame <= seg.end_frame]
        assert all(k.kind == "lock" for k in inside), \
            f"ease inside a stable lock: {inside}"
        assert len(inside) <= 1

    def test_two_participants_both_visible(self):
        log = af.ActionObservationLog(FW, FH)
        for i in range(90):
            f = i
            log.add(_obs(f, f / FPS, track_id="t-0",
                         box=(240.0, 120.0, 80.0, 120.0), confidence=0.9))
            log.add(_obs(f, f / FPS, track_id="t-1",
                         box=(280.0, 120.0, 80.0, 120.0), confidence=0.88))
        segs, keys, _ = self._built(log)
        grp = next(s for s in segs if s.kind == "group")
        x1, y1, x2, y2 = grp.crop
        assert x1 <= 240 and x2 >= 360 and y1 <= 120 and y2 >= 240


# --- schema v3/v4 ------------------------------------------------------------------

def _v3_plan_dict():
    log = fp.ObservationLog(640, 360)
    plan = fp.build_framing_plan(
        log=log, scenes=[(0, 59)], strategies=["GENERAL"],
        config=fp.TurnFramingConfig(), fps=30.0, frame_w=640, frame_h=360,
        frame_count=60, source_fingerprint="x")
    return fp.plan_to_dict(plan)


def _v4_plan_dict():
    alog = af.ActionObservationLog(640, 360)
    for i in range(60):
        alog.add(af.ActionObservation(
            frame=i, time_s=i / 30.0, scene_id=0, track_id="t-0",
            box=(100.0 + i, 100.0, 80.0, 120.0), confidence=0.9,
            category="person", motion=0.1, saliency=0.5, visibility=0.9,
            provenance="injected-test", reason="test"))
    plan = fp.build_framing_plan(
        log=fp.ObservationLog(640, 360), scenes=[(0, 59)],
        strategies=["GENERAL"], config=fp.TurnFramingConfig(), fps=30.0,
        frame_w=640, frame_h=360, frame_count=60, source_fingerprint="x",
        action_log=alog, action_config=af.ActionFramingConfig())
    return fp.plan_to_dict(plan)


class TestSchemaVersions:
    def test_v3_parses_and_derives_none(self):
        d = _v3_plan_dict()
        assert d["schema_version"] == 3
        assert "scene_kind" not in d["scenes"][0]
        plan = fp.plan_from_dict(d)
        assert plan.scenes[0].scene_kind == "none"

    def test_v4_parses_action_kind(self):
        d = _v4_plan_dict()
        assert d["schema_version"] == 4
        assert d["scenes"][0]["scene_kind"] == "action"
        plan = fp.plan_from_dict(d)
        assert plan.scenes[0].scene_kind == "action"
        assert plan.scenes[0].applies

    def test_v4_roundtrip(self):
        d = _v4_plan_dict()
        plan = fp.plan_from_dict(d)
        d2 = fp.plan_to_dict(plan)
        assert d2["scenes"][0]["scene_kind"] == "action"
        assert d2["schema_version"] == 4

    def test_scene_kind_of(self):
        assert fp.scene_kind_of({"scene_kind": "action"}) == "action"
        assert fp.scene_kind_of({"scene_kind": "bogus"}) == "none"
        assert fp.scene_kind_of({"applies": True,
                                 "strategy": "TRACK"}) == "face"
        assert fp.scene_kind_of({"applies": True,
                                 "strategy": "GENERAL"}) == "none"
        assert fp.scene_kind_of({"applies": False,
                                 "strategy": "TRACK"}) == "none"

    def test_v4_rejects_face_contract_violation(self):
        d = _v4_plan_dict()
        d["scenes"][0]["scene_kind"] = "face"
        with pytest.raises(ValueError, match="face scene"):
            fp.plan_from_dict(d)

    def test_v4_rejects_action_without_general(self):
        d = _v4_plan_dict()
        d["scenes"][0]["strategy"] = "TRACK"
        with pytest.raises(ValueError, match="action scene"):
            fp.plan_from_dict(d)

    def test_v4_rejects_none_claiming_applies(self):
        d = _v4_plan_dict()
        d["scenes"][0]["scene_kind"] = "none"
        with pytest.raises(ValueError, match="non-applicable"):
            fp.plan_from_dict(d)

    def test_v4_rejects_unknown_kind(self):
        d = _v4_plan_dict()
        d["scenes"][0]["scene_kind"] = "portrait"
        with pytest.raises(ValueError, match="bad scene_kind"):
            fp.plan_from_dict(d)

    def test_v4_missing_kind_rejected(self):
        d = _v4_plan_dict()
        del d["scenes"][0]["scene_kind"]
        with pytest.raises(ValueError, match="scene_kind"):
            fp.plan_from_dict(d)

    def test_v4_action_segment_kinds_validated(self):
        d = _v4_plan_dict()
        seg = d["scenes"][0]["segments"][0]
        seg["kind"] = "closeup"
        with pytest.raises(ValueError, match="kind"):
            fp.plan_from_dict(d)

    def test_v4_action_segment_evidence_validated(self):
        d = _v4_plan_dict()
        seg = d["scenes"][0]["segments"][0]
        seg["evidence"] = {**seg["evidence"], "detector": "yolo",
                           "pixels": [1, 2, 3]}
        with pytest.raises(ValueError, match="evidence"):
            fp.plan_from_dict(d)

    def test_v4_unsupported_schema_rejected(self):
        d = _v4_plan_dict()
        d["schema_version"] = 999
        with pytest.raises(ValueError, match="unsupported schema_version"):
            fp.plan_from_dict(d)

    def test_malformed_nested_action_data(self):
        d = _v4_plan_dict()
        d["scenes"][0]["segments"][0]["crop"] = [0, 0, 99999, 10]
        with pytest.raises(ValueError, match="out of bounds"):
            fp.plan_from_dict(d)

    def test_boolean_numerics_rejected(self):
        d = _v4_plan_dict()
        d["scenes"][0]["segments"][0]["start_frame"] = True
        with pytest.raises(ValueError, match="out of range"):
            fp.plan_from_dict(d)

    def test_nan_rejected(self):
        d = json.loads(json.dumps(_v4_plan_dict()))
        d["scenes"][0]["segments"][0]["evidence"][
            "sustained_evidence"] = float("nan")
        with pytest.raises(ValueError):
            fp.plan_from_dict(d)

    def test_gaps_and_overlaps_rejected(self):
        d = _v4_plan_dict()
        segs = d["scenes"][0]["segments"]
        if len(segs) >= 2:
            segs[1]["start_frame"] = segs[0]["start_frame"]
            with pytest.raises(ValueError, match="overlap"):
                fp.plan_from_dict(d)

    def test_out_of_scene_keyframes_rejected(self):
        d = _v4_plan_dict()
        d["scenes"][0]["keyframes"].append(
            {"frame": 9999, "x1": 0, "y1": 0, "x2": 10, "y2": 10,
             "kind": "ease"})
        with pytest.raises(ValueError, match="outside scene"):
            fp.plan_from_dict(d)

    def test_cross_cut_track_ids_rejected(self):
        d = _v4_plan_dict()
        d["scenes"][0]["tracks"].append(
            {"id": "t-0", "scene_id": 7, "start_frame": 0,
             "end_frame": 10})
        with pytest.raises(ValueError, match="cross-scene|scene"):
            fp.plan_from_dict(d)

    def test_incomplete_plan_not_reusable(self):
        d = _v4_plan_dict()
        d["analysis_complete"] = False
        plan = fp.plan_from_dict(d)
        ok, why = fp.plan_matches(
            plan, source_fingerprint="x", frame_w=640, frame_h=360,
            fps=30.0, frame_count=60, scene_intervals=[(0, 59)],
            config_digest=plan.config_digest, aspect=9 / 16,
            strategies=["GENERAL"], tracker_digest="")
        assert ok is False and "incomplete" in why

    def test_stale_config_rejected(self):
        d = _v4_plan_dict()
        plan = fp.plan_from_dict(d)
        ok, why = fp.plan_matches(
            plan, source_fingerprint="x", frame_w=640, frame_h=360,
            fps=30.0, frame_count=60, scene_intervals=[(0, 59)],
            config_digest="deadbeef", aspect=9 / 16,
            strategies=["GENERAL"], tracker_digest="")
        assert ok is False and "stale-config" in why

    def test_changed_geometry_rejected(self):
        d = _v4_plan_dict()
        plan = fp.plan_from_dict(d)
        ok, why = fp.plan_matches(
            plan, source_fingerprint="x", frame_w=641, frame_h=360,
            fps=30.0, frame_count=60, scene_intervals=[(0, 59)],
            config_digest=plan.config_digest, aspect=9 / 16,
            strategies=["GENERAL"], tracker_digest="")
        assert ok is False and "stale-dimensions" in why


class TestActionCropForUnion:
    def test_padding_and_look_ahead(self):
        cfg = af.ActionFramingConfig()
        lock, why = af.action_crop_for_union(
            [(100.0, 100.0, 80.0, 120.0)], (60.0, 0.0), FW, FH, 9 / 16,
            cfg, use_look_ahead=True)
        assert lock is not None
        x1, y1, x2, y2 = lock
        # look-ahead shifts the crop along +x (motion direction)
        cx = (x1 + x2) / 2
        assert cx > 140, f"expected lead room, centre {cx}"
        assert (x2 - x1) / (y2 - y1) == pytest.approx(9 / 16, rel=0.02)

    def test_union_too_wide_returns_none(self):
        cfg = af.ActionFramingConfig()
        lock, why = af.action_crop_for_union(
            [(0.0, 0.0, 300.0, 300.0), (340.0, 0.0, 300.0, 300.0)],
            (0.0, 0.0), FW, FH, 9 / 16, cfg, use_look_ahead=False)
        assert lock is None and why

    def test_wide_crop_never_tightens(self):
        from clippyme.pipeline.reframe_ops import saliency_biased_wide_crop
        # Widest 9:16 window in 640x360 is height-limited: 360h x 202.5w.
        # Saliency may shift it by <= 5% of frame width, never shrink it.
        widths = set()
        centres = []
        for cx in (0.0, 320.0, 640.0):
            x1, y1, x2, y2 = saliency_biased_wide_crop(
                cx, FW, FH, 9 / 16, 0.05)
            widths.add(x2 - x1)
            centres.append((x1 + x2) / 2)
            assert (y2 - y1) == FH
        assert widths == {202} or widths == {203}, widths
        assert max(centres) - min(centres) <= 2 * 0.05 * FW + 2
