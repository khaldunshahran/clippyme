"""Task 12 review-03 durable tests: first-drop exhaustion attribution and
non-face strategy ordering.

Promotes coordinator-review3-remaining.py (2 probes) and adds mutation
gates that fail if first-drop attribution is delayed or if the
non-face strategy check moves behind exhaustion handling.
"""

from __future__ import annotations

from clippyme.pipeline import framing_plan as fp


def _obs(frame: int, track: str, scene: int = 0) -> fp.FaceObservation:
    return fp.FaceObservation(
        frame=frame,
        time_s=frame / 30.0,
        scene_id=scene,
        track_id=track,
        box=(40, 80, 100, 100),
        visibility=0.95,
        mouth_motion=2.0,
        provenance="injected-test",
        reason="review3",
    )


def _plan(log, strategies):
    scenes = [(0, 59), (60, 119), (120, 179), (180, 239)]
    return fp.build_framing_plan(
        log=log,
        scenes=scenes,
        strategies=strategies,
        config=fp.TurnFramingConfig(),
        fps=30.0,
        frame_w=640,
        frame_h=360,
        frame_count=240,
        source_fingerprint="review3-source",
        aspect=9 / 16,
        tracker_digest="tracker-digest",
    )


# --- Promoted probe 1 -------------------------------------------------------
def test_observation_cap_records_the_actual_first_dropped_frame_and_scene():
    """Context must be captured at drop time, before next-frame scene advance."""
    log = fp.ObservationLog(640, 360, max_observations=1)
    log.add(_obs(58, "t-0", scene=0))
    log.add(_obs(58, "t-1", scene=0))  # first dropped observation
    assert log.dropped == 1
    assert log.exhaustion_reason == "analysis-frame-budget-exhausted"
    assert log.exhaustion_frame == 58
    assert log.exhaustion_scene == 0


# --- Promoted probe 2 -------------------------------------------------------
def test_exhaustion_fallback_applies_only_to_face_strategies():
    """Later non-face layouts retain applies=False and their existing renderer."""
    log = fp.ObservationLog(640, 360)
    for frame in range(0, 10, 2):
        log.add(_obs(frame, "t-0", scene=0))
    log.mark_exhaustion("analysis_budget_exhausted", frame=10, scene_id=0)
    plan = _plan(log, ["TRACK", "SPLIT", "SCREENCAST", "GENERAL"])
    assert plan.scenes[0].applies is True
    assert plan.scenes[0].fallback["reason"] == "analysis_budget_exhausted"
    for scene in plan.scenes[1:]:
        assert scene.applies is False
        assert scene.fallback is None
        assert scene.tracks == []
        assert scene.segments == []
        assert scene.keyframes == []


# --- Mutation gate: delayed first-drop attribution --------------------------
def test_first_drop_attribution_survives_later_drops_and_bookkeeping():
    """A later drop or pass-1 mark must not replace the first drop's context.

    If attribution were delayed until the next budget check (after scene
    bookkeeping advanced), the exhaustion frame/scene would point at a
    later scene and the partially observed prior scene could be treated
    as fully analyzed and reused.
    """
    log = fp.ObservationLog(640, 360, max_observations=2)
    log.add(_obs(10, "t-0", scene=0))
    log.add(_obs(20, "t-1", scene=0))
    log.add(_obs(200, "t-2", scene=1))  # first drop: frame 200, scene 1
    log.add(_obs(210, "t-3", scene=1))  # second drop: must not win
    assert log.dropped == 2
    assert log.exhaustion_reason == "analysis-frame-budget-exhausted"
    assert log.exhaustion_frame == 200
    assert log.exhaustion_scene == 1
    assert log.diagnostics.count("analysis-frame-budget-exhausted") == 1
    # A later pass-1 mark cannot replace the original first-drop context.
    log.mark_exhaustion("analysis_budget_exhausted", frame=999, scene_id=9)
    assert log.exhaustion_reason == "analysis-frame-budget-exhausted"
    assert log.exhaustion_frame == 200
    assert log.exhaustion_scene == 1


def test_first_drop_context_drives_partial_scene_fallback():
    """The first drop's scene is the partially observed one, not a later one."""
    log = fp.ObservationLog(640, 360, max_observations=3)
    for frame in (0, 2, 4):
        log.add(_obs(frame, "t-0", scene=0))
    # First drop at frame 58 in scene 0: scene 0 is only partially observed.
    log.add(_obs(58, "t-1", scene=0))
    plan = _plan(log, ["TRACK", "TRACK", "TRACK", "TRACK"])
    assert plan.scenes[0].applies is True
    assert plan.scenes[0].fallback is not None
    assert plan.scenes[0].fallback["reason"] == "analysis-frame-budget-exhausted"
    assert plan.analysis_complete is False


# --- Mutation gate: non-face strategy ordering ------------------------------
def test_disabled_scene_never_gets_exhaustion_fallback():
    """If the strategy check moved behind exhaustion handling, a DISABLED
    scene with an unanalyzed interval would wrongly become applies=True."""
    log = fp.ObservationLog(640, 360)
    for frame in range(0, 10, 2):
        log.add(_obs(frame, "t-0", scene=0))
    log.mark_exhaustion("analysis_budget_exhausted", frame=10, scene_id=0)
    plan = _plan(log, ["TRACK", "DISABLED", "SPLIT", "SCREENCAST"])
    assert plan.scenes[0].applies is True
    assert plan.scenes[0].fallback["reason"] == "analysis_budget_exhausted"
    for scene in plan.scenes[1:]:
        assert scene.applies is False
        assert scene.fallback is None
        assert scene.tracks == []
        assert scene.segments == []
        assert scene.keyframes == []


def test_bare_diagnostic_exhaustion_keeps_nonface_scenes_inapplicable():
    """With no frame context every face scene falls back conservatively,
    but non-face scenes must still stay applies=False."""
    log = fp.ObservationLog(640, 360)
    for frame in range(0, 60, 2):
        log.add(_obs(frame, "t-0", scene=0))
    log.note("analysis-frame-budget-exhausted")
    plan = _plan(log, ["TRACK", "SPLIT", "GENERAL", "DISABLED"])
    assert plan.scenes[0].applies is True
    assert plan.scenes[0].fallback["reason"] == "analysis-frame-budget-exhausted"
    for scene in plan.scenes[1:]:
        assert scene.applies is False
        assert scene.fallback is None
        assert scene.tracks == []
        assert scene.segments == []
        assert scene.keyframes == []
