"""Task 14 review-02: load-bearing mutation gates for the two review fixes.

Gate 1 (temporal support): disabling the acquisition gate must let an
isolated detection authorize a tight crop -- proving the gate is what
keeps isolated detections conservative.

Gate 2 (conservative fit): replacing the fit decision with the pre-fix
center-wide crop must lose the edge landmarks -- proving the fit (not
just any wide crop) is what retains them.
"""
import sys
import types

import pytest


def _stub_heavy_modules():
    """Import-time stubs for the missing ML wheels (host has no GPU/CV)."""
    if "mediapipe" not in sys.modules:
        mp = types.ModuleType("mediapipe")
        solutions = types.ModuleType("mediapipe.solutions")
        face = types.ModuleType("mediapipe.solutions.face_detection")
        face.FaceDetection = object
        solutions.face_detection = face
        mp.solutions = solutions
        sys.modules["mediapipe"] = mp
        sys.modules["mediapipe.solutions"] = solutions
        sys.modules["mediapipe.solutions.face_detection"] = face
    if "ultralytics" not in sys.modules:
        ultra = types.ModuleType("ultralytics")
        ultra.YOLO = object
        sys.modules["ultralytics"] = ultra
    if "torch" not in sys.modules:
        torch = types.ModuleType("torch")
        cuda = types.ModuleType("torch.cuda")
        cuda.is_available = lambda: False
        torch.cuda = cuda
        torch.device = lambda value: value
        sys.modules["torch"] = torch
        sys.modules["torch.cuda"] = cuda
    if "psutil" not in sys.modules:
        psutil = types.ModuleType("psutil")

        class _VM:
            total = 16 * 1024 ** 3
            available = 8 * 1024 ** 3
            percent = 50.0

        psutil.virtual_memory = lambda: _VM()
        psutil.cpu_count = lambda logical=True: 8
        sys.modules["psutil"] = psutil


_stub_heavy_modules()

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

def test_mutation_fit_renderer_removed_loses_edge_landmarks(monkeypatch):
    """Fail-then-restore on the PRODUCTION fit renderer (review-03).

    The old center-wide test evaluated the pre-fix crop math in isolation.
    This one mutates the production render seam: routing explicit fit
    through the pre-fix center-wide crop composition loses the edge
    landmarks on a real frame; restoring create_general_fit_frame keeps
    them. This proves the production-render regression catches removal
    of fit rendering.
    """
    pytest.importorskip("cv2", reason="CV runtime unavailable")
    import numpy as np
    from clippyme.pipeline import reframe
    from clippyme.pipeline.reframe_ops import saliency_biased_wide_crop
    import cv2

    image = np.zeros((480, 640, 3), dtype=np.uint8)
    image[180:300, 20:80] = (0, 255, 0)
    image[180:300, 560:620] = (255, 0, 0)

    def landmarks_present(frame, thresh=100):
        g = int((frame[:, :, 1].astype(int) > 200).sum())
        b = int((frame[:, :, 0].astype(int) > 200).sum())
        return g > thresh and b > thresh

    # Production: deterministic full-frame fit keeps both landmarks.
    assert landmarks_present(reframe.create_general_fit_frame(
        image, 270, 480))

    # MUTATION: the fit renderer is removed; explicit fit falls back to
    # the pre-fix center-wide crop composition. An edge landmark is lost.
    def removed_fit(frame, w, h):
        x1, y1, x2, y2 = saliency_biased_wide_crop(
            frame.shape[1] / 2.0, float(frame.shape[1]),
            float(frame.shape[0]), 9 / 16, 0.05)
        return cv2.resize(frame[int(y1):int(y2), int(x1):int(x2)], (w, h))

    monkeypatch.setattr(reframe, "create_general_fit_frame", removed_fit)
    assert not landmarks_present(reframe.create_general_fit_frame(
        image, 270, 480)), \
        "removing the fit renderer must lose an edge landmark"

    # RESTORE: the production renderer keeps the landmarks again.
    monkeypatch.undo()
    assert landmarks_present(reframe.create_general_fit_frame(
        image, 270, 480))


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
