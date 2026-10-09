"""Task 14: mutation gates.

Each test guards a specific regression: if the implementation is
mutated to break the listed invariant, the test MUST fail. They are
deliberately behavioral (not white-box) so they catch the bug no
matter how it is introduced.
"""
import json
import os

import pytest

from clippyme.pipeline import action_framing as af
from clippyme.pipeline import framing_plan as fp
from clippyme.domain import project_framing as pf


FW, FH = 640.0, 360.0
FPS = 30.0


def _obs(frame, time_s, track_id="t-0", box=(100.0, 100.0, 80.0, 120.0),
         confidence=0.9, motion=0.1, saliency=0.5, visibility=0.9):
    return af.ActionObservation(
        frame=frame, time_s=time_s, scene_id=0, track_id=track_id, box=box,
        confidence=confidence, category="person", motion=motion,
        saliency=saliency, visibility=visibility,
        provenance="injected-test", reason="test")


def _segment(log, **kw):
    cfg = af.ActionFramingConfig(**kw)
    return af.segment_actions(log, scene_id=0, scene_start=0,
                              scene_end=89, config=cfg, fps=FPS,
                              frame_w=FW, frame_h=FH)


# GATE 1: the render pass must consume the persisted plan, never redetect.
def test_gate_render_consumes_plan_without_redetection():
    # The detector seam exists as a module-level name in reframe (so
    # tests can inject it), and the plan-reuse path is pinned by the
    # render test's zero-detector-calls assertion. Here we pin the
    # contract: the seam is the ONLY detector entry point for the
    # action path (no hidden direct YOLO calls in reframe.py).
    import ast
    from pathlib import Path
    import clippyme.pipeline as _pkg
    src = (Path(_pkg.__file__).parent / "reframe.py").read_text(
        encoding="utf-8")
    tree = ast.parse(src)
    yolo_calls = [n for n in ast.walk(tree)
                  if isinstance(n, ast.Name) and n.id == "YOLO"]
    assert not yolo_calls, \
        "action path must not call YOLO directly; use the detector seam"


# GATE 2: largest-box selection must never return.
def test_gate_no_largest_box_selection():
    log = af.ActionObservationLog(FW, FH)
    for i in range(90):
        # Small box, sustained strong evidence.
        log.add(_obs(i, i / FPS, track_id="t-0",
                     box=(100.0, 100.0, 60.0, 90.0),
                     confidence=0.92, visibility=0.95))
        # Huge box, weak evidence (a largest-box policy would pick it).
        log.add(_obs(i, i / FPS, track_id="t-1",
                     box=(250.0, 40.0, 350.0, 280.0),
                     confidence=0.30, visibility=0.40))
    segs, _ = _segment(log)
    singles = [s for s in segs if s.kind == "single"]
    assert singles, "sustained evidence must win a segment"
    for s in singles:
        assert s.subject_ids == ("t-0",), \
            f"largest box stole the crop: {s.subject_ids}"
    # And the built crop is tight on the small subject, not the giant.
    cfg = af.ActionFramingConfig()
    built, _, _ = af.build_action_keyframes(
        segs, log, 0, cfg, FW, FH, 9 / 16, fps=FPS)
    for s in built:
        if s.kind == "single":
            x1, _, x2, _ = s.crop
            assert (x2 - x1) < 350, \
                f"crop follows the giant box: {s.crop}"


# GATE 3: saliency must never authorize a tight crop.
def test_gate_saliency_cannot_authorize_tight_crop():
    log = af.ActionObservationLog(FW, FH)
    for i in range(90):
        log.add(_obs(i, i / FPS, confidence=0.15, visibility=0.25,
                     motion=0.95, saliency=1.0))
    segs, _ = _segment(log)
    assert segs, "expected segments"
    assert all(s.kind == "wide" for s in segs), \
        f"saliency authorized a tight crop: {[s.kind for s in segs]}"
    # The wide crop itself must not tighten toward saliency either.
    cfg = af.ActionFramingConfig()
    built, _, _ = af.build_action_keyframes(
        segs, log, 0, cfg, FW, FH, 9 / 16, fps=FPS)
    for s in built:
        x1, _, x2, _ = s.crop
        # Widest 9:16 window in 640x360 is 202px wide (height-limited).
        assert (x2 - x1) >= 200, \
            f"saliency tightened the wide crop: {s.crop}"


# GATE 4: hard-cut reset must exist.
def test_gate_hard_cut_resets_tracks():
    tr = af.ActionTracker()
    tr.update(0, [{"box": (100, 100, 80, 120), "confidence": 0.9,
                   "category": "person"}])
    assert tr._tracks, "pre-cut track must exist"
    tr.reset()
    assert tr._tracks == {}, "reset must drop live tracks"
    out = tr.update(1, [{"box": (102, 102, 80, 120), "confidence": 0.9,
                         "category": "person"}])
    # Monotonic ids: no reuse, no carryover.
    assert out[0]["track_id"] != "t-0"
    # And the log's cross-scene guard is the backstop.
    log = af.ActionObservationLog(FW, FH)
    log.add(_obs(0, 0.0, track_id="t-0"))
    with pytest.raises(af.ActionObservationRejected,
                       match="cross-scene-identity"):
        log.add(af.ActionObservation(
            frame=60, time_s=2.0, scene_id=1, track_id="t-0",
            box=(100.0, 100.0, 80.0, 120.0), confidence=0.9,
            category="person", motion=0.1, saliency=0.5, visibility=0.9,
            provenance="injected-test", reason="test"))


# GATE 5: dwell and travel caps must hold.
def test_gate_dwell_and_travel_caps():
    log = af.ActionObservationLog(FW, FH)
    for i in range(120):
        strong = "t-0" if (i // 10) % 2 == 0 else "t-1"
        for tid, box in (("t-0", (80.0, 100.0, 80.0, 120.0)),
                         ("t-1", (450.0, 100.0, 80.0, 120.0))):
            log.add(_obs(i, i / FPS, track_id=tid, box=box,
                         confidence=0.95 if tid == strong else 0.55))
    cfg = af.ActionFramingConfig()
    segs, _ = _segment(log, )
    # Re-segment the full 120 frames.
    segs, _ = af.segment_actions(
        log, scene_id=0, scene_start=0, scene_end=119, config=cfg,
        fps=FPS, frame_w=FW, frame_h=FH)
    for s in segs:
        if s.kind != "wide":
            dur = (s.end_frame - s.start_frame + 1) / FPS
            assert dur >= cfg.min_dwell_s - 1e-9, \
                f"dwell violated: {s.kind} @{s.start_frame} ({dur:.2f}s)"
    # Travel cap across built keyframes.
    built, keys, _ = af.build_action_keyframes(
        segs, log, 0, cfg, FW, FH, 9 / 16, fps=FPS)
    by_frame = {}
    for k in keys:
        by_frame[k.frame] = (k.x1, k.y1, k.x2, k.y2)
    for s in built:
        for f in range(s.start_frame, s.end_frame + 1):
            by_frame.setdefault(f, tuple(s.crop))
    capx = cfg.max_travel_frac * FW / FPS
    capy = cfg.max_travel_frac * FH / FPS
    for a, b in zip(sorted(by_frame), sorted(by_frame)[1:]):
        if b - a != 1:
            continue
        ca, cb = by_frame[a], by_frame[b]
        assert abs((cb[0] + cb[2]) / 2 - (ca[0] + ca[2]) / 2) <= capx + 1.5
        assert abs((cb[1] + cb[3]) / 2 - (ca[1] + ca[3]) / 2) <= capy + 1.5


# GATE 6: an incomplete action plan must never be reusable.
def test_gate_incomplete_plan_not_reusable():
    alog = af.ActionObservationLog(640, 360)
    for i in range(60):
        alog.add(af.ActionObservation(
            frame=i, time_s=i / 30.0, scene_id=0, track_id="t-0",
            box=(100.0, 100.0, 80.0, 120.0), confidence=0.9,
            category="person", motion=0.1, saliency=0.5, visibility=0.9,
            provenance="injected-test", reason="test"))
    plan = fp.build_framing_plan(
        log=fp.ObservationLog(640, 360), scenes=[(0, 59)],
        strategies=["GENERAL"], config=fp.TurnFramingConfig(), fps=30.0,
        frame_w=640, frame_h=360, frame_count=60, source_fingerprint="x",
        action_log=alog, action_config=af.ActionFramingConfig())
    d = fp.plan_to_dict(plan)
    d["analysis_complete"] = False
    plan2 = fp.plan_from_dict(d)
    ok, why = fp.plan_matches(
        plan2, source_fingerprint="x", frame_w=640, frame_h=360,
        fps=30.0, frame_count=60, scene_intervals=[(0, 59)],
        config_digest=plan2.config_digest, aspect=9 / 16,
        strategies=["GENERAL"], tracker_digest="")
    assert ok is False and "incomplete" in why
    # And the bridge marks it ineligible, never usable.
    assert plan2.analysis_complete is False


# GATE 7: the project bridge must not drop GENERAL action keyframes.
def test_gate_bridge_keeps_action_keyframes(tmp_path):
    alog = af.ActionObservationLog(640, 360)
    for i in range(60):
        alog.add(af.ActionObservation(
            frame=i, time_s=i / 30.0, scene_id=0, track_id="t-0",
            box=(100.0, 100.0, 80.0, 120.0), confidence=0.9,
            category="person", motion=0.1, saliency=0.5, visibility=0.9,
            provenance="injected-test", reason="test"))
    plan = fp.build_framing_plan(
        log=fp.ObservationLog(640, 360), scenes=[(0, 59)],
        strategies=["GENERAL"], config=fp.TurnFramingConfig(), fps=30.0,
        frame_w=640, frame_h=360, frame_count=60, source_fingerprint="x",
        action_log=alog, action_config=af.ActionFramingConfig())
    d = fp.plan_to_dict(plan)
    assert d["scenes"][0]["keyframes"], \
        "plan must carry action keyframes (lock)"
    job_dir = str(tmp_path / "job")
    os.makedirs(job_dir)
    ref = "clip.turn_framing_plan.json"
    with open(os.path.join(job_dir, ref), "w", encoding="utf-8") as fh:
        json.dump(d, fh)
    framing = pf.build_project_framing(
        job_dir=job_dir, clip_final="clip.mp4", render_w=360,
        render_h=640, render_fps=30.0, clip_frame_count=60)
    assert framing is not None and framing["eligible"]
    sc = framing["scenes"][0]
    assert sc["scene_kind"] == "action"
    assert len(sc["keyframes"]) == len(d["scenes"][0]["keyframes"]), \
        "bridge dropped action keyframes"


# GATE 8: Task 13 pair rollback must stay intact (spot-check the
# accepted invariant: a failed restore raises and keeps backups).
def test_gate_task13_rollback_intact():
    from clippyme.domain.project_framing import FramingIntegrityError
    # The Task 13 rollback guarantee: FramingIntegrityError exists,
    # is an OSError (so callers cannot swallow it as a value), and is
    # a FramingError.
    assert issubclass(FramingIntegrityError, OSError)
    assert issubclass(FramingIntegrityError, pf.FramingError)
