"""Host tests for clippyme.pipeline.framing_plan (Task 12).

Pure module — no cv2/torch/mediapipe imports — so these run on the dev host.
Covers: strict observation validation, config/flag parsing, turn
segmentation (alternating speakers, dwell/cooldown boundaries, occlusion,
ambiguity, noisy evidence, invalid inputs, short scenes), hard-cut identity
isolation, crop math (containment/headroom/clamp/hold-zoom/travel-cap/ease),
plan identity/serialization/eligibility, and atomic persistence.
"""
import json
import math
import os

import pytest

from clippyme.pipeline import framing_plan as fp


# --- helpers ---------------------------------------------------------------

def _obs(frame, t, scene, tid, box=(60, 90, 90, 90), mouth=1.0,
         prov="injected-test", reason="r"):
    return fp.FaceObservation(
        frame=frame, time_s=t, scene_id=scene, track_id=tid, box=box,
        visibility=0.9, mouth_motion=mouth, provenance=prov, reason=reason)


def _scripted_log(scene_intervals=((0, 119),), fps=30.0, w=640, h=360,
                  speakers=("t-0", "t-1"), turn_at=60, boxes=None,
                  active_mouth=2.5, silent_mouth=0.0):
    """Two equal-size faces; `speakers[0]` talks before `turn_at`, the other
    after. mouth values emulate the tracker's bounded variance evidence."""
    log = fp.ObservationLog(w, h)
    boxes = boxes or {"t-0": (60, 90, 90, 90), "t-1": (490, 90, 90, 90)}
    for scene_id, (s, e) in enumerate(scene_intervals):
        for f in range(s, e + 1, 2):
            t = f / fps
            for i, tid in enumerate(speakers):
                active = (i == 0) == (f < turn_at)
                n = f // 2
                mouth = 0.35 if n < 5 else (active_mouth if active else silent_mouth)
                log.add(_obs(f, t, scene_id, tid, box=boxes[tid], mouth=mouth))
    return log


# --- observation validation -------------------------------------------------

class TestObservationValidation:
    def test_valid_observation_accepted(self):
        log = fp.ObservationLog(640, 360)
        log.add(_obs(4, 0.133, 0, "t-7"))
        assert len(log.observations_for_scene(0)) == 1

    def test_rejects_booleans(self):
        log = fp.ObservationLog(640, 360)
        with pytest.raises(fp.ObservationRejected, match="bad-frame"):
            log.add(_obs(True, 0.1, 0, "t-1"))
        with pytest.raises(fp.ObservationRejected, match="bad-scene-id"):
            log.add(_obs(2, 0.1, False, "t-1"))

    def test_rejects_nonfinite_time(self):
        log = fp.ObservationLog(640, 360)
        for bad in (float("nan"), float("inf"), -1.0):
            with pytest.raises(fp.ObservationRejected, match="bad-time"):
                log.add(_obs(2, bad, 0, "t-1"))

    def test_rejects_bad_boxes(self):
        log = fp.ObservationLog(640, 360)
        for box in ((10, 10, 0, 50), (10, 10, -5, 50), (-1, 10, 50, 50),
                    (600, 10, 50, 50), (10, 330, 50, 50), (10, 10, 50),
                    ("x", 10, 50, 50), (10, 10, float("nan"), 50)):
            with pytest.raises(fp.ObservationRejected, match="bad-box|box-out-of-bounds"):
                log.add(_obs(2, 0.1, 0, "t-1", box=box))

    def test_rejects_named_track_id(self):
        # A name or diarization label can never validate: only opaque t-<int>.
        log = fp.ObservationLog(640, 360)
        for bad in ("alice", "speaker_1", "t-", "t-1x", "", 7, None):
            with pytest.raises(fp.ObservationRejected, match="bad-track-id"):
                log.add(_obs(2, 0.1, 0, bad))

    def test_rejects_duplicate_frame_track(self):
        log = fp.ObservationLog(640, 360)
        log.add(_obs(4, 0.1, 0, "t-1"))
        with pytest.raises(fp.ObservationRejected, match="duplicate-observation"):
            log.add(_obs(4, 0.2, 0, "t-1"))

    def test_rejects_time_regression(self):
        log = fp.ObservationLog(640, 360)
        log.add(_obs(4, 0.2, 0, "t-1"))
        with pytest.raises(fp.ObservationRejected, match="time-regression"):
            log.add(_obs(6, 0.1, 0, "t-1"))

    def test_rejects_cross_scene_identity_carryover(self):
        log = fp.ObservationLog(640, 360)
        log.add(_obs(4, 0.1, 0, "t-1"))
        with pytest.raises(fp.ObservationRejected, match="cross-scene-identity"):
            log.add(_obs(130, 4.4, 1, "t-1"))

    def test_rejects_bad_evidence_and_provenance(self):
        log = fp.ObservationLog(640, 360)
        base = _obs(2, 0.1, 0, "t-1", mouth=1.0)
        with pytest.raises(fp.ObservationRejected, match="bad-visibility"):
            log.add(fp.FaceObservation(**{**base.__dict__, "visibility": 1.5}))
        with pytest.raises(fp.ObservationRejected, match="bad-mouth-motion"):
            log.add(_obs(2, 0.1, 0, "t-1", mouth=3.5))
        with pytest.raises(fp.ObservationRejected, match="bad-provenance"):
            log.add(_obs(2, 0.1, 0, "t-1", prov="whisper-diarizer"))
        with pytest.raises(fp.ObservationRejected, match="bad-reason"):
            log.add(_obs(2, 0.1, 0, "t-1", reason="x" * 65))


# --- flag + config parsing ---------------------------------------------------

class TestFlagAndConfig:
    def test_flag_enabled_by_default(self, monkeypatch):
        monkeypatch.delenv("CLIPPYME_TURN_AWARE_FRAMING", raising=False)
        assert fp.turn_aware_enabled() is True
        monkeypatch.setenv("CLIPPYME_TURN_AWARE_FRAMING", "")
        assert fp.turn_aware_enabled() is True

    def test_flag_explicit_negatives_disable(self, monkeypatch):
        for val in ("0", "false", "False", "no", "NO", "off", " 0 "):
            monkeypatch.setenv("CLIPPYME_TURN_AWARE_FRAMING", val)
            assert fp.turn_aware_enabled() is False, val

    def test_flag_other_values_enable(self, monkeypatch):
        for val in ("1", "true", "yes", "on", "maybe"):
            monkeypatch.setenv("CLIPPYME_TURN_AWARE_FRAMING", val)
            assert fp.turn_aware_enabled() is True, val

    def test_config_defaults(self, monkeypatch):
        for v in ("CLIPPYME_TURN_EVIDENCE_S", "CLIPPYME_TURN_MIN_DWELL_S",
                  "CLIPPYME_TURN_COOLDOWN_S", "CLIPPYME_TURN_SWITCH_MARGIN",
                  "CLIPPYME_TURN_EASE_FRAMES", "CLIPPYME_TURN_MAX_ANALYSIS_FRAMES"):
            monkeypatch.delenv(v, raising=False)
        cfg = fp.config_from_env()
        assert (cfg.evidence_duration_s, cfg.min_dwell_s, cfg.cooldown_s,
                cfg.switch_margin, cfg.ease_frames) == (0.8, 1.0, 1.5, 1.25, 6)

    def test_config_parses_env(self, monkeypatch):
        monkeypatch.setenv("CLIPPYME_TURN_EVIDENCE_S", "1.2")
        monkeypatch.setenv("CLIPPYME_TURN_MIN_DWELL_S", "0.5")
        monkeypatch.setenv("CLIPPYME_TURN_COOLDOWN_S", "2")
        monkeypatch.setenv("CLIPPYME_TURN_SWITCH_MARGIN", "2.0")
        monkeypatch.setenv("CLIPPYME_TURN_EASE_FRAMES", "3")
        cfg = fp.config_from_env()
        assert (cfg.evidence_duration_s, cfg.min_dwell_s, cfg.cooldown_s,
                cfg.switch_margin, cfg.ease_frames) == (1.2, 0.5, 2.0, 2.0, 3)

    def test_config_garbage_falls_back(self, monkeypatch):
        monkeypatch.setenv("CLIPPYME_TURN_EVIDENCE_S", "junk")
        monkeypatch.setenv("CLIPPYME_TURN_SWITCH_MARGIN", "nan")
        monkeypatch.setenv("CLIPPYME_TURN_EASE_FRAMES", "-4")
        cfg = fp.config_from_env()
        assert cfg.evidence_duration_s == 0.8
        assert cfg.switch_margin == 1.25
        # ease_frames clamps at 0, so -4 becomes 0 (a valid int, not the default).
        assert cfg.ease_frames == 0

    def test_config_digest_stable(self, monkeypatch):
        monkeypatch.delenv("CLIPPYME_TURN_EVIDENCE_S", raising=False)
        assert fp.config_from_env().config_digest() == fp.TurnFramingConfig().config_digest()
        monkeypatch.setenv("CLIPPYME_TURN_EVIDENCE_S", "9.9")
        assert fp.config_from_env().config_digest() != fp.TurnFramingConfig().config_digest()


# --- segmentation ------------------------------------------------------------

class TestSegmentation:
    def test_alternating_speakers_produce_turn(self):
        log = _scripted_log()
        segs, _ = fp.segment_turns(log, scene_id=0, scene_start=0,
                                   scene_end=119, config=fp.TurnFramingConfig(),
                                   fps=30.0, frame_w=640, frame_h=360)
        singles = [s for s in segs if s.kind == "single"]
        assert len(singles) >= 2
        assert singles[0].subject_ids == ("t-0",)
        # The accepted turn lands after sustained evidence, not instantly.
        turn = next(s for s in singles if s.subject_ids == ("t-1",))
        assert 60 <= turn.start_frame <= 100
        assert turn.reason == "turn-switch-sustained-evidence"

    def test_segments_ordered_non_overlapping_full_coverage(self):
        log = _scripted_log()
        segs, _ = fp.segment_turns(log, scene_id=0, scene_start=0,
                                   scene_end=119, config=fp.TurnFramingConfig(),
                                   fps=30.0, frame_w=640, frame_h=360)
        assert segs[0].start_frame == 0
        assert segs[-1].end_frame == 119
        for a, b in zip(segs, segs[1:]):
            assert a.end_frame + 1 == b.start_frame

    def test_size_alone_cannot_switch(self):
        # Equal mouth evidence on both faces, one much bigger, far apart: the
        # bigger face must NOT be crowned on size alone. Ambiguous and
        # unpairable -> the framing stays wide, never a lone size-winner.
        log = fp.ObservationLog(640, 360)
        for f in range(0, 120, 2):
            t = f / 30.0
            log.add(_obs(f, t, 0, "t-0", box=(60, 90, 90, 90), mouth=1.0))
            log.add(_obs(f, t, 0, "t-1", box=(400, 60, 180, 180), mouth=1.0))
        segs, _ = fp.segment_turns(log, scene_id=0, scene_start=0,
                                   scene_end=119, config=fp.TurnFramingConfig(),
                                   fps=30.0, frame_w=640, frame_h=360)
        singles = [s for s in segs if s.kind == "single"]
        assert not [s for s in singles if s.subject_ids == ("t-1",)], \
            "the bigger face must never win alone on size"
        assert any(s.kind == "wide" for s in segs)

    def test_dwell_blocks_early_switch(self):
        cfg = fp.TurnFramingConfig(min_dwell_s=100.0)  # effectively infinite
        log = _scripted_log()
        segs, _ = fp.segment_turns(log, scene_id=0, scene_start=0,
                                   scene_end=119, config=cfg,
                                   fps=30.0, frame_w=640, frame_h=360)
        singles = [s for s in segs if s.kind == "single"]
        assert {s.subject_ids for s in singles} == {("t-0",)}

    def test_cooldown_blocks_rapid_bounce(self):
        # A speaks, B takes a strong turn, A immediately tries to take it back:
        # the second switch inside the cooldown must be held.
        log = fp.ObservationLog(640, 360)
        sched = [(0, 59, "t-0"), (60, 65, "t-1"), (66, 119, "t-0")]
        boxes = {"t-0": (60, 90, 90, 90), "t-1": (490, 90, 90, 90)}
        for f in range(0, 120, 2):
            t = f / 30.0
            active = next(tid for s, e, tid in sched if s <= f <= e)
            for tid in ("t-0", "t-1"):
                n = f // 2
                mouth = 0.35 if n < 5 else (2.5 if tid == active else 0.0)
                log.add(_obs(f, t, 0, tid, box=boxes[tid], mouth=mouth))
        cfg = fp.TurnFramingConfig(cooldown_s=30.0)
        segs, _ = fp.segment_turns(log, scene_id=0, scene_start=0,
                                   scene_end=119, config=cfg,
                                   fps=30.0, frame_w=640, frame_h=360)
        # B's 6-frame turn can never be accepted (evidence window is 0.8s),
        # and A's "comeback" lands inside the cooldown of B's switch — no
        # bounce: at most one accepted turn total.
        switches = [s for s in segs if s.reason == "turn-switch-sustained-evidence"]
        assert len(switches) <= 1

    def test_cooldown_boundary(self):
        # Challenger crosses the margin just before vs just after the cooldown:
        # a long cooldown must DELAY the switch, a short one lets it through
        # as soon as evidence is sustained. Either way it cannot bounce.
        def switch_start(cooldown_s):
            log = fp.ObservationLog(640, 360)
            boxes = {"t-0": (60, 90, 90, 90), "t-1": (490, 90, 90, 90)}
            for f in range(0, 120, 2):
                t = f / 30.0
                active = "t-0" if f < 40 else "t-1"
                for tid in ("t-0", "t-1"):
                    n = f // 2
                    mouth = 0.35 if n < 5 else (2.5 if tid == active else 0.0)
                    log.add(_obs(f, t, 0, tid, box=boxes[tid], mouth=mouth))
            cfg = fp.TurnFramingConfig(cooldown_s=cooldown_s, min_dwell_s=0.0,
                                      evidence_duration_s=0.2)
            segs, _ = fp.segment_turns(log, scene_id=0, scene_start=0,
                                       scene_end=119, config=cfg, fps=30.0,
                                       frame_w=640, frame_h=360)
            turns = [s for s in segs if s.subject_ids == ("t-1",)]
            assert len(turns) == 1  # exactly one accepted turn, no bounce
            return turns[0].start_frame
        # A acquired ~frame 10 (t=0.33s); B sustained ~frame 46 (t=1.53s).
        assert 44 <= switch_start(0.1) <= 52   # evidence-gated, immediate
        assert switch_start(2.0) >= 70        # held through the cooldown

    def test_short_occlusion_holds_safely(self):
        log = fp.ObservationLog(640, 360)
        for f in range(0, 120, 2):
            t = f / 30.0
            # t-0 talks throughout but vanishes for frames 40-50 (0.33s).
            # A silent t-1 stays visible so the segmenter sees the gap.
            if not 40 <= f <= 50:
                log.add(_obs(f, t, 0, "t-0", mouth=2.5 if f > 10 else 0.35))
            log.add(_obs(f, t, 0, "t-1", box=(490, 90, 90, 90), mouth=0.0))
        cfg = fp.TurnFramingConfig(occlusion_hold_s=1.0)
        segs, _ = fp.segment_turns(log, scene_id=0, scene_start=0,
                                   scene_end=119, config=cfg,
                                   fps=30.0, frame_w=640, frame_h=360)
        singles = [s for s in segs if s.kind == "single"]
        assert singles and all(s.subject_ids == ("t-0",) for s in singles)
        assert any(s.evidence["occluded_frames"] > 0 for s in singles)

    def test_prolonged_loss_widens(self):
        log = fp.ObservationLog(640, 360)
        for f in range(0, 60, 2):
            log.add(_obs(f, f / 30.0, 0, "t-0", mouth=2.5))
        # t-0 gone for the rest of the scene.
        cfg = fp.TurnFramingConfig(occlusion_hold_s=0.5)
        segs, _ = fp.segment_turns(log, scene_id=0, scene_start=0,
                                   scene_end=119, config=cfg,
                                   fps=30.0, frame_w=640, frame_h=360)
        kinds = [s.kind for s in segs]
        assert "single" in kinds and "wide" in kinds
        wide = next(s for s in segs if s.kind == "wide" and s.start_frame > 30)
        assert wide.reason == "speaker-lost-widen"

    def test_ambiguous_dialogue_two_subject(self):
        log = fp.ObservationLog(640, 360)
        # Close pair: both faces fit one 9:16 crop (see crop_for_subjects).
        boxes = {"t-0": (300, 90, 60, 60), "t-1": (345, 90, 60, 60)}
        for f in range(0, 120, 2):
            # Both talk, near-equal evidence -> ambiguous dialogue.
            for tid in ("t-0", "t-1"):
                log.add(_obs(f, f / 30.0, 0, tid, box=boxes[tid], mouth=2.0))
        segs, _ = fp.segment_turns(log, scene_id=0, scene_start=0,
                                   scene_end=119, config=fp.TurnFramingConfig(),
                                   fps=30.0, frame_w=640, frame_h=360)
        pairs = [s for s in segs if s.kind == "two-subject"]
        assert pairs, [ (s.kind, s.subject_ids) for s in segs ]
        assert set(pairs[0].subject_ids) == {"t-0", "t-1"}
        # The pair crop (built with the keyframes) contains both faces.
        plan = fp.build_framing_plan(
            log=log, scenes=[(0, 119)], strategies=["WIDE"],
            config=fp.TurnFramingConfig(), fps=30.0, frame_w=640, frame_h=360,
            frame_count=120, source_fingerprint="x")
        pseg = next(s for s in plan.scenes[0].segments if s.kind == "two-subject")
        x1, y1, x2, y2 = pseg.crop
        assert x1 <= 300 and x2 >= 405 and y1 <= 90 and y2 >= 150

    def test_noisy_evidence_stays_stable(self):
        import random
        rng = random.Random(7)
        log = fp.ObservationLog(640, 360)
        for f in range(0, 120, 2):
            # t-0 talks; evidence jitters but stays above threshold.
            mouth = max(0.6, 2.0 + rng.uniform(-0.5, 0.5))
            log.add(_obs(f, f / 30.0, 0, "t-0", mouth=mouth))
            log.add(_obs(f, f / 30.0, 0, "t-1", box=(490, 90, 90, 90),
                         mouth=rng.uniform(0.0, 0.2)))
        segs, _ = fp.segment_turns(log, scene_id=0, scene_start=0,
                                   scene_end=119, config=fp.TurnFramingConfig(),
                                   fps=30.0, frame_w=640, frame_h=360)
        singles = [s for s in segs if s.kind == "single"]
        assert len(singles) == 1 and singles[0].subject_ids == ("t-0",)

    def test_stable_single_subject_stays_static(self):
        log = fp.ObservationLog(640, 360)
        for f in range(0, 120, 2):
            log.add(_obs(f, f / 30.0, 0, "t-0", mouth=2.0))
        segs, _ = fp.segment_turns(log, scene_id=0, scene_start=0,
                                   scene_end=119, config=fp.TurnFramingConfig(),
                                   fps=30.0, frame_w=640, frame_h=360)
        nonwide = [s for s in segs if s.kind != "wide"]
        assert len(nonwide) == 1 and nonwide[0].subject_ids == ("t-0",)

    def test_short_scene_no_crash(self):
        log = fp.ObservationLog(640, 360)
        for f in range(0, 6, 2):
            log.add(_obs(f, f / 30.0, 0, "t-0", mouth=2.0))
        segs, _ = fp.segment_turns(log, scene_id=0, scene_start=0,
                                   scene_end=5, config=fp.TurnFramingConfig(),
                                   fps=30.0, frame_w=640, frame_h=360)
        assert segs and segs[-1].end_frame == 5

    def test_no_observations_raises_plan_error(self):
        log = fp.ObservationLog(640, 360)
        with pytest.raises(fp._PlanBuildError, match="no-observations"):
            fp.segment_turns(log, scene_id=0, scene_start=0, scene_end=119,
                             config=fp.TurnFramingConfig(), fps=30.0,
                             frame_w=640, frame_h=360)

    def test_switch_margin_blocks_weak_challenger(self):
        # Challenger at 1.1x incumbent with margin 1.25: no switch.
        log = fp.ObservationLog(640, 360)
        boxes = {"t-0": (60, 90, 90, 90), "t-1": (490, 90, 90, 90)}
        for f in range(0, 120, 2):
            t = f / 30.0
            for tid, mouth in (("t-0", 2.0), ("t-1", 2.2)):
                log.add(_obs(f, t, 0, tid, box=boxes[tid], mouth=mouth))
        cfg = fp.TurnFramingConfig(switch_margin=1.25)
        segs, _ = fp.segment_turns(log, scene_id=0, scene_start=0,
                                   scene_end=119, config=cfg, fps=30.0,
                                   frame_w=640, frame_h=360)
        # 2.2/2.0 = 1.1 < 1.25: no decisive switch. The faces are
        # ambiguous-close but too far apart for a 9:16 pair crop, so the
        # framing stays wide instead of crowning on ambiguity - and nothing
        # bounces between singles.
        assert not [s for s in segs if s.reason == "turn-switch-sustained-evidence"]
        assert not [s for s in segs if s.kind == "single"], \
            "no single crowning on ambiguous evidence"
        assert any(s.reason == "ambiguous-dialogue-wide" for s in segs)


# --- crop math ----------------------------------------------------------------

class TestCropMath:
    def _planned_scene(self, log, scene=(0, 119), strategy="WIDE"):
        plan = fp.build_framing_plan(
            log=log, scenes=[scene], strategies=[strategy],
            config=fp.TurnFramingConfig(), fps=30.0, frame_w=640, frame_h=360,
            frame_count=scene[1] + 1, source_fingerprint="x")
        return plan.scenes[0]

    def test_single_subject_containment_and_headroom(self):
        log = fp.ObservationLog(640, 360)
        for f in range(0, 120, 2):
            log.add(_obs(f, f / 30.0, 0, "t-0", box=(60, 90, 90, 90), mouth=2.0))
        sp = self._planned_scene(log)
        seg = next(s for s in sp.segments if s.kind == "single")
        x1, y1, x2, y2 = seg.crop
        # Containment: the whole face box sits inside the crop...
        assert x1 <= 60 and y1 <= 90 and x2 >= 150 and y2 >= 180
        # ...with headroom above the face top...
        assert y1 < 90
        # ...and the crop stays in-bounds.
        assert 0 <= x1 < x2 <= 640 and 0 <= y1 < y2 <= 360

    def test_crop_fixed_within_stable_segment(self):
        log = fp.ObservationLog(640, 360)
        for f in range(0, 120, 2):
            log.add(_obs(f, f / 30.0, 0, "t-0", box=(60, 90, 90, 90), mouth=2.0))
        sp = self._planned_scene(log)
        seg = next(s for s in sp.segments if s.kind == "single")
        crops = {tuple(fp.crop_at_frame(sp, f)) for f in
                 range(seg.start_frame + 8, seg.end_frame + 1)}
        assert len(crops) == 1  # bit-identical crop: no flicker, no chase

    def test_ease_only_between_turns_bounded(self):
        log = _scripted_log()
        sp = self._planned_scene(log)
        ease = [k for k in sp.keyframes if k.kind == "ease"]
        assert ease, "expected ease keyframes around the accepted turn"
        # Ease frames cluster at segment starts only.
        starts = {s.start_frame for s in sp.segments}
        for k in ease:
            assert any(s <= k.frame < s + 6 for s in starts)
        # Monotonic travel from the left crop to the right crop.
        xs = [k.x1 for k in sorted(ease, key=lambda k: k.frame)
              if 60 <= k.frame <= 90]
        assert xs == sorted(xs), "ease must not oscillate"

    def test_travel_cap(self):
        cfg = fp.TurnFramingConfig(max_travel_frac=0.1)  # 64px on 640w
        log = _scripted_log()
        plan = fp.build_framing_plan(
            log=log, scenes=[(0, 119)], strategies=["WIDE"], config=cfg,
            fps=30.0, frame_w=640, frame_h=360, frame_count=120,
            source_fingerprint="x")
        sp = plan.scenes[0]
        singles = [s for s in sp.segments if s.kind == "single"]
        assert len(singles) >= 2
        c0 = (singles[0].crop[0] + singles[0].crop[2]) / 2
        c1 = (singles[1].crop[0] + singles[1].crop[2]) / 2
        assert abs(c1 - c0) <= 0.1 * 640 + 2

    def test_hold_zoom_never_tightens(self):
        # Second speaker has a much bigger face (would zoom in) — hold_zoom
        # keeps the crop at least as wide as the previous segment's.
        cfg = fp.TurnFramingConfig()
        log = fp.ObservationLog(640, 360)
        for f in range(0, 120, 2):
            t = f / 30.0
            active = "t-0" if f < 60 else "t-1"
            for tid, box in (("t-0", (60, 90, 90, 90)),
                             ("t-1", (400, 40, 200, 200))):
                n = f // 2
                mouth = 0.35 if n < 5 else (2.5 if tid == active else 0.0)
                log.add(_obs(f, t, 0, tid, box=box, mouth=mouth))
        plan = fp.build_framing_plan(
            log=log, scenes=[(0, 119)], strategies=["WIDE"], config=cfg,
            fps=30.0, frame_w=640, frame_h=360, frame_count=120,
            source_fingerprint="x")
        sp = plan.scenes[0]
        singles = [s for s in sp.segments if s.kind == "single"]
        assert len(singles) >= 2
        h0 = singles[0].crop[3] - singles[0].crop[1]
        h1 = singles[1].crop[3] - singles[1].crop[1]
        assert h1 >= h0 - 2, "zoom must hold or widen, never tighten"

    def test_subjects_too_wide_falls_back(self):
        log = fp.ObservationLog(640, 360)
        for f in range(0, 120, 2):
            log.add(_obs(f, f / 30.0, 0, "t-0", box=(0, 90, 90, 90), mouth=2.0))
            log.add(_obs(f, f / 30.0, 0, "t-1", box=(550, 90, 90, 90), mouth=2.0))
        sp = self._planned_scene(log)
        # Faces at opposite edges: no 9:16 crop fits both -> wide fallback.
        assert any(s.kind == "wide" for s in sp.segments)

    def test_crop_at_frame_never_recomputes_decisions(self):
        # crop_at_frame is a pure lookup over stored keyframes/segments: the
        # same frame always yields the same crop with no observation input.
        log = _scripted_log()
        sp = self._planned_scene(log)
        for f in (0, 10, 59, 60, 74, 75, 80, 119):
            assert fp.crop_at_frame(sp, f) == fp.crop_at_frame(sp, f)


# --- plan identity / serialization / eligibility ------------------------------

class TestPlanIdentity:
    def _plan(self, **kw):
        log = _scripted_log()
        args = dict(log=log, scenes=[(0, 119)], strategies=["WIDE"],
                    config=fp.TurnFramingConfig(), fps=30.0, frame_w=640,
                    frame_h=360, frame_count=120, source_fingerprint="fp123")
        args.update(kw)
        return fp.build_framing_plan(**args)

    def test_roundtrip(self):
        plan = self._plan()
        plan2 = fp.plan_from_dict(fp.plan_to_dict(plan))
        assert plan2.source_fingerprint == "fp123"
        assert plan2.config_digest == plan.config_digest
        assert [(s.start_frame, s.end_frame, s.subject_ids, s.kind, s.crop)
                for s in plan2.scenes[0].segments] == \
               [(s.start_frame, s.end_frame, s.subject_ids, s.kind, s.crop)
                for s in plan.scenes[0].segments]

    def test_no_pixels_or_names_persisted(self):
        plan = self._plan()
        raw = json.dumps(fp.plan_to_dict(plan))
        assert "alice" not in raw.lower()
        for key in ("embedding", "pixels", "landmark", "diarization"):
            assert key not in raw.lower()

    def test_matches(self):
        plan = self._plan()
        ok, why = fp.plan_matches(
            plan, source_fingerprint="fp123", frame_w=640, frame_h=360,
            fps=30.0, frame_count=120, scene_intervals=[(0, 119)],
            config_digest=fp.TurnFramingConfig().config_digest())
        assert (ok, why) == (True, "match")

    @pytest.mark.parametrize("field,value,reason", [
        ("source_fingerprint", "other", "stale-source"),
        ("frame_w", 641, "stale-dimensions"),
        ("frame_h", 361, "stale-dimensions"),
        ("fps", 29.97, "stale-fps"),
        ("frame_count", 121, "stale-frame-count"),
        ("scene_intervals", [(0, 100)], "stale-scenes"),
        ("config_digest", "deadbeef", "stale-config"),
    ])
    def test_stale_fields_rejected(self, field, value, reason):
        plan = self._plan()
        kw = dict(source_fingerprint="fp123", frame_w=640, frame_h=360,
                  fps=30.0, frame_count=120, scene_intervals=[(0, 119)],
                  config_digest=fp.TurnFramingConfig().config_digest())
        kw[field] = value
        ok, why = fp.plan_matches(plan, **kw)
        assert ok is False and why == reason

    def test_malformed_plans_rejected(self):
        plan = self._plan()
        good = fp.plan_to_dict(plan)
        bad_schema = dict(good, schema_version=999)
        with pytest.raises(ValueError, match="unsupported schema_version"):
            fp.plan_from_dict(bad_schema)
        bad_crop = json.loads(json.dumps(good))
        bad_crop["scenes"][0]["segments"][0]["crop"] = [0, 0, 9999, 10]
        with pytest.raises(ValueError, match="out of bounds"):
            fp.plan_from_dict(bad_crop)
        bad_overlap = json.loads(json.dumps(good))
        segs = bad_overlap["scenes"][0]["segments"]
        if len(segs) >= 2:
            segs[1]["start_frame"] = segs[0]["start_frame"]
            with pytest.raises(ValueError, match="overlap"):
                fp.plan_from_dict(bad_overlap)
        with pytest.raises(ValueError, match="top-level"):
            fp.plan_from_dict([1, 2, 3])
        with pytest.raises(ValueError, match="non-anonymous"):
            named = json.loads(json.dumps(good))
            named["scenes"][0]["segments"][0]["subject_ids"] = ["alice"]
            fp.plan_from_dict(named)

    def test_atomic_save_load(self, tmp_path):
        plan = self._plan()
        path = str(tmp_path / "plan.json")
        fp.save_plan_atomic(plan, path)
        assert not list(tmp_path.glob("*.tmp-*")), "no temp leftovers"
        loaded = fp.load_plan(path)
        assert loaded.config_digest == plan.config_digest
        assert len(loaded.scenes[0].segments) == len(plan.scenes[0].segments)

    def test_load_missing_and_corrupt(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            fp.load_plan(str(tmp_path / "nope.json"))
        bad = tmp_path / "bad.json"
        bad.write_text("{not json")
        with pytest.raises(ValueError, match="invalid JSON"):
            fp.load_plan(str(bad))

    def test_source_fingerprint(self, tmp_path):
        a = tmp_path / "a.bin"
        a.write_bytes(b"x" * 100000)
        f1 = fp.source_fingerprint(str(a))
        assert f1 == fp.source_fingerprint(str(a))
        a.write_bytes(b"y" + b"x" * 99999)
        assert fp.source_fingerprint(str(a)) != f1

    def test_non_face_strategies_not_planned(self):
        log = _scripted_log()
        plan = fp.build_framing_plan(
            log=log, scenes=[(0, 119)], strategies=["GENERAL"],
            config=fp.TurnFramingConfig(), fps=30.0, frame_w=640, frame_h=360,
            frame_count=120, source_fingerprint="x")
        sp = plan.scenes[0]
        assert sp.applies is False and sp.segments == [] and sp.fallback is None

    def test_invalid_scene_gets_fallback_not_crash(self):
        log = fp.ObservationLog(640, 360)  # no observations at all
        plan = fp.build_framing_plan(
            log=log, scenes=[(0, 119)], strategies=["TRACK"],
            config=fp.TurnFramingConfig(), fps=30.0, frame_w=640, frame_h=360,
            frame_count=120, source_fingerprint="x")
        sp = plan.scenes[0]
        assert sp.applies is True and sp.usable is False
        assert sp.fallback["reason"] == "no-observations"

    def test_analysis_frame_budget(self):
        cfg = fp.TurnFramingConfig(max_analysis_frames=10)
        log = fp.ObservationLog(640, 360, max_observations=cfg.max_analysis_frames)
        for f in range(0, 120, 2):
            try:
                log.add(_obs(f, f / 30.0, 0, "t-0", mouth=2.0))
            except fp.ObservationRejected:
                pass
        assert log.dropped > 0
        assert "analysis-frame-budget-exhausted" in log.diagnostics
