"""Task 14 review-02: load-bearing mutation gates for the two review fixes.

Gate 1 (temporal support): disabling the acquisition gate must let an
isolated detection authorize a tight crop -- proving the gate is what
keeps isolated detections conservative.

Gate 2 (conservative fit): replacing the fit decision with the pre-fix
center-wide crop must lose the edge landmarks -- proving the fit (not
just any wide crop) is what retains them.
"""
import pytest

from clippyme.pipeline import action_framing as af

FW, FH = 640.0, 360.0
FPS = 30.0


def _obs(frame, time_s, track_id="t-0", box=(100.0, 100.0, 80.0, 120.0),
         confidence=0.9, motion=0.1, saliency=0.5, visibility=0.9):
    return af.ActionObservation(
        frame=frame, time_s=time_s, scene_id=0, track_id=track_id, box=box,
        confidence=confidence, category="person", motion=motion,
        saliency=saliency, visibility=visibility,
        provenance="injected-test", reason="test")


def _segment(log, **cfg_kw):
    cfg = af.ActionFramingConfig(**cfg_kw)
    segs, _ = af.segment_actions(log, scene_id=0, scene_start=0,
                                 scene_end=59, config=cfg, fps=FPS,
                                 frame_w=FW, frame_h=FH)
    return segs


# ===========================================================================
# GATE 1: temporal acquisition support is load-bearing.
# ===========================================================================

def test_mutation_temporal_gate_disabled_allows_isolated_tight():
    """Disable the gate (min_frames=1, no span/gap) -> the isolated
    detection at frame 0 authorizes a tight single. This BAD outcome
    proves the gate is load-bearing: with the gate on, the same input
    stays wide (see test_task14_review2.py)."""
    log = af.ActionObservationLog(FW, FH)
    log.add(_obs(0, 0.0))
    segs = _segment(log, acquisition_min_frames=1,
                    acquisition_min_span_s=0.0,
                    acquisition_max_gap_s=999.0)
    kinds = [(s.kind, s.start_frame, s.end_frame) for s in segs]
    assert any(k == "single" for k, _, _ in kinds), \
        f"gate disabled: isolated detection must acquire tight, got {kinds}"


def test_mutation_temporal_gate_disabled_allows_two_frame_tight():
    """Two adjacent observations with the gate disabled -> tight."""
    log = af.ActionObservationLog(FW, FH)
    log.add(_obs(0, 0.0))
    log.add(_obs(1, 1 / FPS))
    segs = _segment(log, acquisition_min_frames=1,
                    acquisition_min_span_s=0.0,
                    acquisition_max_gap_s=999.0)
    kinds = [(s.kind, s.start_frame, s.end_frame) for s in segs]
    assert any(k == "single" for k, _, _ in kinds), \
        f"gate disabled: two adjacent obs must acquire tight, got {kinds}"


def test_temporal_gate_gap_breaks_support():
    """Observations split by a gap above acquisition_max_gap_s never
    authorize a tight crop, even though the total count is sufficient."""
    log = af.ActionObservationLog(FW, FH)
    for i in range(3):
        log.add(_obs(i, i / FPS))
    for i in range(30, 33):
        log.add(_obs(i, i / FPS))
    segs = _segment(log)
    kinds = [(s.kind, s.start_frame, s.end_frame) for s in segs]
    assert all(k == "wide" for k, _, _ in kinds), \
        f"gapped evidence must stay wide, got {kinds}"


def test_temporal_gate_sustained_acquires():
    """Sustained valid evidence (no large gaps) still acquires."""
    log = af.ActionObservationLog(FW, FH)
    for i in range(30):
        log.add(_obs(i, i / FPS))
    segs = _segment(log)
    kinds = [(s.kind, s.start_frame, s.end_frame) for s in segs]
    assert any(k == "single" for k, _, _ in kinds), \
        f"sustained evidence must acquire, got {kinds}"
    # Acquisition happens only after the support window fills, never
    # at frame 0.
    first_single = min(s.start_frame for s in segs if s.kind == "single")
    assert first_single > 0, "acquisition must not happen at frame 0"


# ===========================================================================
# GATE 2: conservative fit is load-bearing (vs the pre-fix center-wide crop).
# ===========================================================================

def test_mutation_center_wide_crop_loses_edge_landmarks():
    """Replacing the conservative fit with the pre-fix center-wide crop
    loses the edge landmarks. Uses the production
    saliency_biased_wide_crop (the exact pre-fix wide composition) on a
    640x480 source: the landmarks at x=20-80 and x=560-620 fall outside
    the aspect-shaped window. This BAD outcome proves the fit decision
    (full-frame letterbox, no crop) is what retains them."""
    pytest.importorskip("cv2", reason="CV runtime unavailable")
    import numpy as np
    from clippyme.pipeline.reframe_ops import saliency_biased_wide_crop

    W, H = 640, 480
    # Pre-fix behavior: saliency-biased wide crop of the 9:16 window.
    x1, y1, x2, y2 = saliency_biased_wide_crop(
        W / 2.0, float(W), float(H), 9 / 16, 0.05)
    # The window is aspect-shaped and interior: it cannot span x=20..620.
    assert (x2 - x1) < W, "wide crop must be narrower than the source"
    keeps_left = x1 <= 20
    keeps_right = x2 >= 620
    assert not (keeps_left and keeps_right), \
        f"center-wide crop [{x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}] must lose " \
        "at least one edge landmark (x=20-80 or x=560-620)"


def test_fit_segment_carries_no_crop_no_keyframes():
    """Production contract: a weak-evidence plan persists wide segments
    with fit=True, empty crop, and no keyframes."""
    from clippyme.pipeline import framing_plan as fp

    alog = af.ActionObservationLog(640, 480)
    for i in range(60):
        alog.add(af.ActionObservation(
            frame=i, time_s=i / 30.0, scene_id=0, track_id="t-0",
            box=(100.0, 100.0, 80.0, 120.0), confidence=0.1,
            category="person", motion=0.05, saliency=0.5, visibility=0.9,
            provenance="injected-test", reason="test"))
    plan = fp.build_framing_plan(
        log=fp.ObservationLog(640, 480), scenes=[(0, 59)],
        strategies=["GENERAL"], config=fp.TurnFramingConfig(), fps=30.0,
        frame_w=640, frame_h=480, frame_count=60, source_fingerprint="x",
        action_log=alog, action_config=af.ActionFramingConfig())
    sc = plan.scenes[0]
    assert sc.scene_kind == "action"
    wides = [s for s in sc.segments if s.kind == "wide"]
    assert wides, "weak evidence must produce wide segments"
    for s in wides:
        assert s.fit is True, "wide segment must be marked fit"
        assert s.crop == [], "fit segment must carry an empty crop"
    assert sc.keyframes == [], "fit ranges emit no keyframes"
    # The helper pass 2 uses:
    assert fp.segment_fit_at_frame(sc, 10) is True
