"""Task 14: load-bearing mutation gates for action-aware framing.

Each gate DISABLES one protection (via monkeypatch) and asserts the BAD
outcome actually appears. That proves the protection is load-bearing: if
the protection were removed from production code, the gate would catch
the regression.

Gate 1 is the tripwire variant: the pass-2 detection / recompute entries
are patched to RAISE, and the render must still succeed with the planned
crop -- proving pass 2 never redetects or recomputes.

All fixtures use injected detections only (no network, no real video),
mirroring tests/pipeline/test_action_framing.py; gate 1 reuses the
synthetic-video harness style of test_action_framing_render.py.
"""
import dataclasses
import json
import os
import shutil
import subprocess

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


def _segment(log, scene_id=0, start=0, end=89, **kw):
    cfg = af.ActionFramingConfig(**kw)
    return af.segment_actions(log, scene_id=scene_id, scene_start=start,
                              scene_end=end, config=cfg, fps=FPS,
                              frame_w=FW, frame_h=FH)


def _v4_action_plan_dict():
    """A production-built v4 plan with one action GENERAL scene."""
    alog = af.ActionObservationLog(640, 360)
    for i in range(60):
        alog.add(af.ActionObservation(
            frame=i, time_s=i / 30.0, scene_id=0, track_id="t-0",
            box=(100.0 + i * 0.5, 100.0, 80.0, 120.0), confidence=0.9,
            category="person", motion=0.05, saliency=0.5, visibility=0.9,
            provenance="injected-test", reason="test"))
    plan = fp.build_framing_plan(
        log=fp.ObservationLog(640, 360), scenes=[(0, 59)],
        strategies=["GENERAL"], config=fp.TurnFramingConfig(), fps=30.0,
        frame_w=640, frame_h=360, frame_count=60, source_fingerprint="x",
        action_log=alog, action_config=af.ActionFramingConfig())
    return fp.plan_to_dict(plan)


def _write_sidecar(job_dir, ref, plan_dict):
    p = os.path.join(job_dir, ref)
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(plan_dict, fh)
    return p


# ===========================================================================
# GATE 1: pass 2 must render the persisted plan -- never redetect/recompute.
# ===========================================================================

def _stub_heavy_modules():
    """Import-time stubs for the missing ML wheels (host has no
    mediapipe/ultralytics/torch); mirrors test_action_framing_render.py."""
    import sys
    import types
    if "mediapipe" not in sys.modules:
        mp = types.ModuleType("mediapipe")
        mp_solutions = types.ModuleType("mediapipe.solutions")
        mp_face = types.ModuleType("mediapipe.solutions.face_detection")
        mp_face.FaceDetection = object
        mp_solutions.face_detection = mp_face
        mp.solutions = mp_solutions
        sys.modules["mediapipe"] = mp
        sys.modules["mediapipe.solutions"] = mp_solutions
        sys.modules["mediapipe.solutions.face_detection"] = mp_face
    if "ultralytics" not in sys.modules:
        ultra = types.ModuleType("ultralytics")
        ultra.YOLO = object
        sys.modules["ultralytics"] = ultra
    if "torch" not in sys.modules:
        torch_mod = types.ModuleType("torch")
        cuda_mod = types.ModuleType("torch.cuda")
        cuda_mod.is_available = lambda: False
        torch_mod.cuda = cuda_mod
        torch_mod.device = lambda x: x
        sys.modules["torch"] = torch_mod
        sys.modules["torch.cuda"] = cuda_mod
    if "psutil" not in sys.modules:
        psutil_mod = types.ModuleType("psutil")

        class _VM:
            total = 16 * 1024 ** 3
            available = 8 * 1024 ** 3
            percent = 50.0

        psutil_mod.virtual_memory = lambda: _VM()
        psutil_mod.cpu_count = lambda logical=True: 8
        sys.modules["psutil"] = psutil_mod


def test_gate1_pass2_renders_plan_without_redetection(tmp_path, monkeypatch):
    """Tripwire: the pass-2 detection / recompute entries raise if called.

    The render must still succeed with the planned tight crop -- proving
    pass 2 renders the reloaded sidecar verbatim and never redetects or
    recomputes. A mutation that redetects in pass 2 would trip the
    tripwire (raise, or duplicated frames that fail the crop check).
    """
    cv2 = pytest.importorskip("cv2", reason="CV runtime unavailable")
    pytest.importorskip("scenedetect", reason="scenedetect unavailable")
    import numpy as np
    _stub_heavy_modules()
    from clippyme.pipeline import reframe

    W, H, FPSV, N = 640, 480, 30, 60

    def _frame(i):
        img = np.zeros((H, W, 3), dtype=np.uint8)
        # Green rect drifting SLOWLY left -> right (30 px/s): sustained
        # evidence, holdable -> single lock.
        x = int(200 + (60 * i / (N - 1)))
        cv2.rectangle(img, (x, 180), (x + 90, 300), (0, 255, 0), -1)
        return img

    src = str(tmp_path / "src.mp4")
    silent = str(tmp_path / "silent.mp4")
    vw = cv2.VideoWriter(silent, cv2.VideoWriter_fourcc(*"mp4v"), FPSV,
                         (W, H))
    for i in range(N):
        vw.write(_frame(i))
    vw.release()
    dur = N / FPSV
    subprocess.run(
        ["ffmpeg", "-y", "-i", silent, "-f", "lavfi", "-i",
         f"sine=frequency=880:duration={dur}", "-c:v", "libx264",
         "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", src],
        capture_output=True, check=True)

    from scenedetect import FrameTimecode
    monkeypatch.setattr(
        reframe, "detect_scenes",
        lambda path: ([(FrameTimecode(0, FPSV), FrameTimecode(N, FPSV))],
                      float(FPSV)))
    monkeypatch.setattr(reframe, "detect_face_candidates",
                        lambda frame: [])
    import clippyme.pipeline.reframe_detect as reframe_detect
    monkeypatch.setattr(reframe_detect, "detect_face_candidates",
                        lambda frame: [])
    from clippyme.pipeline.layouts import split_layout
    monkeypatch.setattr(split_layout, "detect_split_scenes",
                        lambda *a, **k: {})
    monkeypatch.delenv("CLIPPYME_ACTION_AWARE_FRAMING", raising=False)

    def _color_detect(frame):
        g = frame[:, :, 1].astype(int)
        b = frame[:, :, 0].astype(int)
        r = frame[:, :, 2].astype(int)
        green = (g > 110) & (g > r + 25) & (g > b + 25)
        if green.any():
            ys, xs = np.where(green)
            x1, x2 = float(xs.min()), float(xs.max())
            y1, y2 = float(ys.min()), float(ys.max())
            return [{"box": [x1, y1, x2 - x1 + 1, y2 - y1 + 1],
                     "confidence": 0.9, "category": "object"}]
        return []

    calls = {"n": 0}

    def _counting_detect(frame):
        calls["n"] += 1
        return _color_detect(frame)

    monkeypatch.setattr(reframe, "detect_action_candidates",
                        _counting_detect)

    out = str(tmp_path / "out.mp4")
    assert reframe.process_video_to_vertical(
        src, out, reframe_mode="auto") is True
    assert calls["n"] > 0, "action detector must run in pass 1"

    plan_path = os.path.splitext(out)[0] + ".turn_framing_plan.json"
    plan = fp.load_plan(plan_path)
    assert plan.schema_version == 4 and plan.analysis_complete is True
    assert len(plan.scenes) == 1
    sp = plan.scenes[0]
    assert sp.strategy == "GENERAL"
    assert sp.scene_kind == "action" and sp.usable
    singles = [s for s in sp.segments if s.kind == "single"]
    assert singles, "expected a planned single lock"
    px1, _, px2, _ = singles[0].crop
    assert (px2 - px1) < 200, \
        f"the planned crop must be tight: {singles[0].crop}"

    # --- arm the tripwires: any pass-2 redetection/recompute raises ---
    def _boom_detect(frame):
        calls["n"] += 1
        raise AssertionError("pass 2 must never redetect")

    def _boom_segment(*a, **k):
        raise AssertionError("pass 2 must never recompute the plan")

    monkeypatch.setattr(reframe, "detect_action_candidates", _boom_detect)
    monkeypatch.setattr(af, "segment_actions", _boom_segment)
    calls["n"] = 0

    # Second render to the SAME output reuses the stable sidecar.
    assert reframe.process_video_to_vertical(
        src, out, reframe_mode="auto") is True
    assert calls["n"] == 0, \
        "a reused plan must not redetect (detector called " \
        f"{calls['n']}x)"

    # The planned tight crop still renders: frame 30 shows the landmark
    # large (a duplicated-black / full-frame fallback would not).
    cap = cv2.VideoCapture(out)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 30)
    ok, fr = cap.read()
    cap.release()
    assert ok, "cannot decode rendered frame 30"
    g = fr[:, :, 1].astype(int)
    b = fr[:, :, 0].astype(int)
    r = fr[:, :, 2].astype(int)
    green_px = int((((g > 90) & (g > r + 20) & (g > b + 20)).sum()))
    # Tight lock renders the 90x120 landmark at ~180x240 px; the accepted
    # full-frame GENERAL fallback would show it at ~38x120 px.
    assert green_px > 8000, \
        f"the planned tight crop did not render (green px={green_px})"


# ===========================================================================
# GATE 2: union-based selection -- size alone must never select.
# ===========================================================================

def _distractor_log():
    """Small sustained action vs a large weak distractor."""
    log = af.ActionObservationLog(FW, FH)
    for i in range(90):
        log.add(_obs(i, i / FPS, track_id="t-0",
                     box=(100.0, 100.0, 60.0, 90.0),
                     confidence=0.92, visibility=0.95))
        log.add(_obs(i, i / FPS, track_id="t-1",
                     box=(250.0, 40.0, 350.0, 280.0),
                     confidence=0.30, visibility=0.40))
    return log


def test_gate2_largest_box_policy_steals_the_distractor(monkeypatch):
    """Replace the union crop builder with a largest-box-stealing variant.

    The bad outcome (the crop follows the large distractor) must appear --
    proving the production union-based crop target (size never selects) is
    what keeps the distractor from stealing the frame.
    """
    log = _distractor_log()
    cfg = af.ActionFramingConfig()
    segs, _ = _segment(log)
    singles = [s for s in segs if s.kind == "single"]
    assert singles and all(s.subject_ids == ("t-0",) for s in singles), \
        "evidence (not size) must select the small sustained subject"
    built, _, _ = af.build_action_keyframes(
        segs, log, 0, cfg, FW, FH, 9 / 16, fps=FPS)
    single = next(s for s in built if s.kind == "single")
    x1, y1, x2, y2 = single.crop
    # Unpatched sanity: tight on the small subject, nowhere near the
    # distractor centre (425, 180).
    assert (x2 - x1) < 200
    assert not (x1 <= 425 <= x2 and y1 <= 180 <= y2), \
        f"sanity: unpatched crop must not cover the distractor: " \
        f"{single.crop}"

    # The forbidden policy: crop the LARGEST box in the scene.
    scene_boxes = [o.box for o in log.observations_for_scene(0)]

    def _largest_box_stealing(boxes, motion_vec, frame_w, frame_h, aspect,
                              config, use_look_ahead=True):
        big = max(scene_boxes, key=lambda b: b[2] * b[3])
        bx, by, bw, bh = (float(v) for v in big)
        crop_h = min(frame_h, bh * 1.2)
        crop_w = crop_h * aspect
        cx, cy = bx + bw / 2.0, by + bh / 2.0
        sx1 = int(round(min(max(cx - crop_w / 2.0, 0.0),
                            frame_w - crop_w)))
        sy1 = int(round(min(max(cy - crop_h / 2.0, 0.0),
                            frame_h - crop_h)))
        return ([sx1, sy1, int(round(sx1 + crop_w)),
                 int(round(sy1 + crop_h))], "")

    monkeypatch.setattr(af, "action_crop_for_union",
                        _largest_box_stealing)
    built2, _, _ = af.build_action_keyframes(
        segs, log, 0, cfg, FW, FH, 9 / 16, fps=FPS)
    single2 = next(s for s in built2 if s.kind == "single")
    sx1, sy1, sx2, sy2 = single2.crop
    assert sx1 <= 425 <= sx2 and sy1 <= 180 <= sy2, \
        f"a largest-box policy must steal the distractor: {single2.crop}"


# ===========================================================================
# GATE 3: saliency must never authorize a tight crop.
# ===========================================================================

def test_gate3_saliency_as_evidence_authorizes_tight_crop(monkeypatch):
    """Map saliency into the detector-evidence slot (the protection's
    removal): a tight crop must then appear.

    Finding: segment_actions ignores provenance entirely -- the evidence
    mapping is ``confidence * visibility`` in ``evidence_vals``; saliency
    is only a recorded support signal. The patch below emulates a
    mutation that feeds saliency into that mapping.
    """
    log = af.ActionObservationLog(FW, FH)
    for i in range(90):
        log.add(_obs(i, i / FPS, confidence=0.15, visibility=0.25,
                     motion=0.1, saliency=1.0))
    cfg = af.ActionFramingConfig()
    segs, _ = _segment(log)
    assert segs and all(s.kind == "wide" for s in segs), \
        "sanity: saliency alone must not authorize a tight crop"

    _orig_segment = af.segment_actions

    def _saliency_counts_as_detector_evidence(log, **kw):
        ev_log = af.ActionObservationLog(FW, FH)
        for o in log.observations_for_scene(kw.get("scene_id", 0)):
            ev_log.add(dataclasses.replace(o, confidence=o.saliency,
                                           visibility=1.0))
        return _orig_segment(ev_log, **kw)

    monkeypatch.setattr(af, "segment_actions",
                        _saliency_counts_as_detector_evidence)
    segs2, _ = af.segment_actions(
        log, scene_id=0, scene_start=0, scene_end=89, config=cfg, fps=FPS,
        frame_w=FW, frame_h=FH)
    assert any(s.kind == "single" for s in segs2), \
        "saliency counted as detector evidence must authorize a tight crop"
    built, _, _ = af.build_action_keyframes(
        segs2, log, 0, cfg, FW, FH, 9 / 16, fps=FPS)
    single = next(s for s in built if s.kind == "single")
    x1, _, x2, _ = single.crop
    assert (x2 - x1) < 200, \
        f"without the gate, saliency tightens the crop: {single.crop}"


# ===========================================================================
# GATE 4: the hard-cut reset must exist.
# ===========================================================================

def test_gate4_disabled_reset_carries_identity_across_cut(monkeypatch):
    """Disable ActionTracker.reset (called at every hard cut in pass 1):
    the pre-cut identity must then carry across the cut.

    The bad outcome (t-0 continuing in the new scene) proves the reset is
    what prevents cross-cut identity carryover (the log's
    cross-scene-identity rejection is only the backstop).
    """
    det = [{"box": (100.0, 100.0, 80.0, 120.0), "confidence": 0.9,
            "category": "person"}]
    det2 = [{"box": (102.0, 102.0, 80.0, 120.0), "confidence": 0.9,
             "category": "person"}]

    tr = af.ActionTracker()
    assert tr.update(0, det)[0]["track_id"] == "t-0"
    tr.reset()
    out = tr.update(1, det2)
    assert out[0]["track_id"] == "t-1", \
        "sanity: the real reset drops the track and never reuses the id"

    tr2 = af.ActionTracker()
    assert tr2.update(0, det)[0]["track_id"] == "t-0"
    monkeypatch.setattr(af.ActionTracker, "reset", lambda self: None)
    tr2.reset()  # hard cut -- now a no-op
    carried = tr2.update(1, det2)
    assert carried[0]["track_id"] == "t-0", \
        "without the hard-cut reset the pre-cut identity carries across " \
        "the cut"


# ===========================================================================
# GATE 5: dwell and travel limits must hold.
# ===========================================================================

def _alternating_log(n_frames=180, every=18):
    """Two subjects alternate dominance; dwell is the binding constraint
    (cooldown 0, fast evidence window)."""
    log = af.ActionObservationLog(FW, FH)
    for i in range(n_frames):
        strong = "t-0" if (i // every) % 2 == 0 else "t-1"
        for tid, box in (("t-0", (80.0, 100.0, 80.0, 120.0)),
                         ("t-1", (450.0, 100.0, 80.0, 120.0))):
            log.add(_obs(i, i / FPS, track_id=tid, box=box,
                         confidence=0.95 if tid == strong else 0.55,
                         motion=0.05))
    return log


def _decision_switches(segs):
    return sum(1 for a, b in zip(segs, segs[1:])
               if (a.kind, a.subject_ids) != (b.kind, b.subject_ids))


def test_gate5a_no_dwell_limit_decisions_bounce():
    """min_dwell_s=0 (frozen dataclass -> replaced config): decisions must
    bounce -- strictly more switches than the dwell-limited run."""
    log = _alternating_log()
    # Temporal acquisition gate tuned to the 0.1s evidence window.
    base = dict(evidence_duration_s=0.1, cooldown_s=0.0,
                acquisition_min_span_s=0.05, acquisition_max_gap_s=0.05)
    cfg = af.ActionFramingConfig(min_dwell_s=1.0, **base)
    segs, _ = af.segment_actions(
        log, scene_id=0, scene_start=0, scene_end=179, config=cfg, fps=FPS,
        frame_w=FW, frame_h=FH)
    # Strip the leading acquisition-wide (temporal gate needs a few
    # frames); the dwell test is about the acquired singles.
    segs = segs[1:] if segs and segs[0].kind == "wide" else segs
    assert segs and all(s.kind == "single" for s in segs)
    n_limited = _decision_switches(segs)

    cfg_no_dwell = dataclasses.replace(cfg, min_dwell_s=0.0)
    segs2, _ = af.segment_actions(
        log, scene_id=0, scene_start=0, scene_end=179, config=cfg_no_dwell,
        fps=FPS, frame_w=FW, frame_h=FH)
    segs2 = segs2[1:] if segs2 and segs2[0].kind == "wide" else segs2
    assert segs2 and all(s.kind == "single" for s in segs2)
    n_free = _decision_switches(segs2)

    assert n_free > n_limited, \
        f"removing the dwell limit must let decisions bounce " \
        f"(limited={n_limited}, free={n_free})"


def _travel_log(n_frames=120, flip_at=30):
    """t-0 strong then t-1 strong, far apart: forces a long transition."""
    log = af.ActionObservationLog(FW, FH)
    for i in range(n_frames):
        tid = "t-0" if i < flip_at else "t-1"
        box = (40.0, 100.0, 80.0, 120.0) if tid == "t-0" \
            else (520.0, 100.0, 80.0, 120.0)
        conf = 0.95 if (tid == "t-0") == (i < flip_at) else 0.4
        log.add(_obs(i, i / FPS, track_id=tid, box=box, confidence=conf,
                     motion=0.05))
    return log


def _max_per_frame_travel(log, cfg, n_frames):
    segs, _ = af.segment_actions(
        log, scene_id=0, scene_start=0, scene_end=n_frames - 1, config=cfg,
        fps=FPS, frame_w=FW, frame_h=FH)
    built, keys, _ = af.build_action_keyframes(
        segs, log, 0, cfg, FW, FH, 9 / 16, fps=FPS)
    by_frame = {}
    for k in keys:
        by_frame[k.frame] = (k.x1, k.y1, k.x2, k.y2)
    for s in built:
        if s.fit:
            continue
        for f in range(s.start_frame, s.end_frame + 1):
            by_frame.setdefault(f, tuple(s.crop))
    mx = 0.0
    for a, b in zip(sorted(by_frame), sorted(by_frame)[1:]):
        if b - a != 1:
            continue
        ca, cb = by_frame[a], by_frame[b]
        mx = max(mx, abs((cb[0] + cb[2]) / 2 - (ca[0] + ca[2]) / 2))
    return mx


def test_gate5b_no_travel_limit_exceeds_the_cap():
    """max_travel_frac huge: per-frame crop travel must exceed the cap the
    unpatched limit enforces."""
    log = _travel_log()
    cfg = af.ActionFramingConfig()
    cap = cfg.max_travel_frac * FW / FPS  # px/frame at 30 fps
    limited = _max_per_frame_travel(log, cfg, 120)
    assert limited <= cap + 1.5, \
        f"sanity: the travel limit must hold (max {limited:.1f}px > " \
        f"cap {cap:.1f}px)"

    cfg_free = dataclasses.replace(cfg, max_travel_frac=4.0)
    free = _max_per_frame_travel(log, cfg_free, 120)
    assert free > cap + 1.5, \
        f"without the travel limit the transition must exceed the cap " \
        f"(max {free:.1f}px vs cap {cap:.1f}px)"


# ===========================================================================
# GATE 6: an incomplete action plan must never be reusable.
# ===========================================================================

def test_gate6_incomplete_plan_carried_when_eligibility_blind(
        tmp_path, monkeypatch):
    """Blind the bridge eligibility (incomplete treated as eligible): the
    incomplete plan's trajectory must then be carried into the project.

    Unpatched, the same sidecar is rejected/omitted -- proving the gate is
    what keeps a partially-analyzed plan out of the project.
    """
    d = _v4_action_plan_dict()
    assert d["scenes"][0]["keyframes"], "fixture must carry keyframes"
    d["analysis_complete"] = False
    job_dir = str(tmp_path / "job")
    os.makedirs(job_dir)
    ref = "clip.turn_framing_plan.json"
    _write_sidecar(job_dir, ref, d)
    kwargs = dict(job_dir=job_dir, clip_final="clip.mp4", render_w=360,
                  render_h=640, render_fps=30.0, clip_frame_count=60)

    framing = pf.build_project_framing(**kwargs)
    assert framing is not None
    assert framing["eligible"] is False
    assert framing["scenes"] == []
    assert framing["ineligibility_reason"] == pf.REASON_INCOMPLETE

    _orig_load = pf.load_validated_plan

    def _blind_eligibility(resolved):
        plan = _orig_load(resolved)
        plan = dict(plan)
        plan["analysis_complete"] = True
        return plan

    monkeypatch.setattr(pf, "load_validated_plan", _blind_eligibility)
    framing2 = pf.build_project_framing(**kwargs)
    assert framing2 is not None
    assert framing2["eligible"] is True, \
        "a blinded eligibility gate must carry the incomplete plan"
    assert framing2["scenes"], "the trajectory must be carried"
    assert framing2["scenes"][0]["scene_kind"] == "action"
    assert framing2["scenes"][0]["keyframes"], \
        "the incomplete plan's keyframes would reach the render"


# ===========================================================================
# GATE 7: the project bridge must carry GENERAL action keyframes.
# ===========================================================================

def test_gate7_dropped_action_keyframes_loses_trajectory(
        tmp_path, monkeypatch):
    """Drop action keyframes in the bridge normalization: the emitted
    project step must then FAIL (no framing field emitted for an
    applicable action scene with no trajectory), proving the bridge
    carrying them is load-bearing."""
    d = _v4_action_plan_dict()
    n_kf = len(d["scenes"][0]["keyframes"])
    assert n_kf > 0, "fixture must carry keyframes"
    job_dir = str(tmp_path / "job")
    os.makedirs(job_dir)
    ref = "clip.turn_framing_plan.json"
    _write_sidecar(job_dir, ref, d)
    kwargs = dict(job_dir=job_dir, clip_final="clip.mp4", render_w=360,
                  render_h=640, render_fps=30.0, clip_frame_count=60)

    framing = pf.build_project_framing(**kwargs)
    assert framing is not None and framing["eligible"] is True
    assert len(framing["scenes"][0]["keyframes"]) == n_kf, \
        "sanity: the bridge must carry every action keyframe"

    _orig_normalize = pf.normalize_timeline

    def _drop_action_keyframes(**kw):
        scenes = _orig_normalize(**kw)
        for sc in scenes:
            if sc.get("scene_kind") == "action":
                sc["keyframes"] = []
                # Also drop the conservative fit: without keyframes AND
                # without fit, the scene has no trajectory at all.
                sc["fit"] = False
        return scenes

    monkeypatch.setattr(pf, "normalize_timeline", _drop_action_keyframes)
    # The ProjectFraming model requires an applicable scene to carry
    # keyframes, a fallback reason, or fit: with the trajectory dropped,
    # the bridge refuses to emit a framing field at all -- the planned
    # action trajectory never reaches the project/render.
    framing2 = pf.build_project_framing(**kwargs)
    assert framing2 is None, \
        "dropping action keyframes must fail the emitted project: no " \
        "framing field may be emitted for an applicable action scene " \
        "with no trajectory"


# ===========================================================================
# GATE 8: Task 13 pair rollback -- raise/retain must stay load-bearing.
# ===========================================================================

def test_gate8_weakened_rollback_swallows_integrity_failure(
        tmp_path, monkeypatch):
    """Restore the pre-review-05 swallowing _restore_live_pair (catches
    errors, deletes staging, returns normally): with a sidecar-install
    failure injected, NO FramingIntegrityError may surface and the live
    pair must be left mixed -- proving the raise/retain behavior is what
    keeps a failed rollback loud and recoverable."""
    final = tmp_path / "clip.mp4"
    stable = tmp_path / "clip.turn_framing_plan.json"
    final.write_bytes(b"accepted-video-a")
    stable.write_bytes(b"accepted-sidecar-a")
    assert pf.snapshot_accepted_pair(
        job_dir=str(tmp_path), clip_final=str(final),
        stable_sidecar=str(stable))

    # This is the live pair that must survive a failed recovery attempt.
    final.write_bytes(b"live-video-b")
    stable.write_bytes(b"live-sidecar-b")

    real_replace = os.replace
    final_replaces = 0

    def fail_install_and_video_rollback(src, dst):
        nonlocal final_replaces
        dst_abs = os.path.abspath(str(dst))
        if dst_abs == os.path.abspath(str(final)):
            final_replaces += 1
            # First replace installs snapshot video A. The second is the
            # attempted atomic rollback to live video B.
            if final_replaces >= 2:
                raise OSError("injected video rollback replace failure")
        if dst_abs == os.path.abspath(str(stable)):
            # Snapshot sidecar installation fails, starting rollback.
            raise OSError("injected sidecar install failure")
        return real_replace(src, dst)

    real_copy2 = shutil.copy2

    def fail_direct_video_rollback(src, dst, *args, **kwargs):
        # Atomic rollback above failed; make the direct-copy fallback
        # fail too.
        if (".task13-live-backup-" in os.path.abspath(str(src))
                and os.path.abspath(str(dst)) == os.path.abspath(str(final))):
            raise OSError("injected direct video rollback failure")
        return real_copy2(src, dst, *args, **kwargs)

    monkeypatch.setattr(pf.os, "replace", fail_install_and_video_rollback)
    monkeypatch.setattr(shutil, "copy2", fail_direct_video_rollback)

    def _old_swallowing_restore(record):
        """The pre-review-05 implementation, inline."""
        for entry in record["paths"]:
            dst = entry["dst"]
            try:
                if entry["existed"]:
                    try:
                        pf._install_verified(entry["backup"], dst)
                    except OSError:
                        pf._install_direct(entry["backup"], dst)
                elif os.path.isfile(dst) or os.path.islink(dst):
                    os.remove(dst)
            except OSError:
                continue
        pf._remove_tree_quietly(record["dir"])

    monkeypatch.setattr(pf, "_restore_live_pair", _old_swallowing_restore)

    surfaced = False
    try:
        result = pf.restore_accepted_snapshot(
            job_dir=str(tmp_path), clip_final=str(final),
            stable_sidecar=str(stable))
    except OSError:
        surfaced = True
        result = False

    assert not surfaced, \
        "the weakened rollback must NOT surface the integrity failure"
    assert result is False, \
        "the failure collapses into an ordinary False"
    assert (final.read_bytes(), stable.read_bytes()) == (
        b"accepted-video-a", b"live-sidecar-b"), \
        "the live pair is left mixed: snapshot video with live sidecar"
    assert not list(tmp_path.glob(".task13-live-backup-*")), \
        "the verified staging backups were deleted"
