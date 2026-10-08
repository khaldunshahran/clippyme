"""Task 12 review-02: budget-exhaustion planning + relational sidecar validation.

Promotes ``work/muse-handoff/task-12/coordinator-review2-remaining.py``
(8 probes) into durable tests, plus mutation gates that fail if any of
the review-02 corrections is removed:

* exhaustion diagnostics ignored by the planner,
* incomplete sidecars reused as completed plans,
* ``analysis_deadline_s`` dropped from config identity,
* any relational parser invariant removed
  (subject/track membership, duplicate tracks, track/scene ranges,
  segment contiguity, scene ordering, two-subject distinctness,
  fallback/non-applicable consistency).
"""

from __future__ import annotations

import copy
import json

import pytest

from clippyme.pipeline import framing_plan as fp


def _obs(frame: int, track: str = "t-0") -> fp.FaceObservation:
    return fp.FaceObservation(
        frame=frame,
        time_s=frame / 30.0,
        scene_id=0,
        track_id=track,
        box=(40, 80, 100, 100),
        visibility=0.95,
        mouth_motion=2.0,
        provenance="injected-test",
        reason="review2",
    )


def _partial_exhausted_log() -> fp.ObservationLog:
    log = fp.ObservationLog(640, 360)
    # Only the opening fraction of a six-second scene was observed.
    for frame in range(0, 30, 2):
        log.add(_obs(frame))
    log.note("analysis_budget_exhausted")
    return log


def _build(log: fp.ObservationLog | None = None) -> fp.FramingPlan:
    return fp.build_framing_plan(
        log=log or _partial_exhausted_log(),
        scenes=[(0, 179)],
        strategies=["TRACK"],
        config=fp.TurnFramingConfig(),
        fps=30.0,
        frame_w=640,
        frame_h=360,
        frame_count=180,
        source_fingerprint="review2-source",
        aspect=9 / 16,
        tracker_digest="tracker-digest",
    )


# --- promoted review-02 probes ---------------------------------------------

def test_exhausted_partial_scene_is_safe_fallback_not_reusable_normal_plan():
    plan = _build()
    scene = plan.scenes[0]
    assert scene.usable is False
    assert scene.segments == []
    assert scene.keyframes == []
    assert scene.fallback is not None
    assert scene.fallback["reason"] in {
        "analysis_budget_exhausted",
        "analysis-frame-budget-exhausted",
    }


def test_exhausted_budget_skips_expensive_scene_planner(monkeypatch):
    calls = []

    def forbidden(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("scene planner ran after budget exhaustion")

    monkeypatch.setattr(fp, "build_scene_plan", forbidden)
    plan = _build()
    assert calls == []
    assert plan.scenes[0].fallback["reason"] == "analysis_budget_exhausted"


def test_analysis_deadline_changes_config_identity():
    short = fp.TurnFramingConfig(analysis_deadline_s=0.05)
    long = fp.TurnFramingConfig(analysis_deadline_s=120.0)
    assert short.config_digest() != long.config_digest()


def _normal_plan() -> fp.FramingPlan:
    log = fp.ObservationLog(640, 360)
    for frame in range(0, 180, 2):
        active = "t-0" if frame < 90 else "t-1"
        for tid, box in (
            ("t-0", (30, 80, 100, 100)),
            ("t-1", (510, 80, 100, 100)),
        ):
            log.add(fp.FaceObservation(
                frame=frame,
                time_s=frame / 30.0,
                scene_id=0,
                track_id=tid,
                box=box,
                visibility=0.95,
                mouth_motion=2.5 if tid == active and frame >= 10 else 0.1,
                provenance="injected-test",
                reason="review2",
            ))
    return _build(log)


def test_parser_rejects_segment_subject_missing_from_scene_tracks():
    raw = fp.plan_to_dict(_normal_plan())
    assert raw["scenes"][0]["segments"]
    raw["scenes"][0]["tracks"] = []
    with pytest.raises(ValueError, match="malformed-plan"):
        fp.plan_from_dict(raw)


def test_parser_rejects_duplicate_track_ids():
    raw = fp.plan_to_dict(_normal_plan())
    tracks = raw["scenes"][0]["tracks"]
    assert tracks
    tracks.append(copy.deepcopy(tracks[0]))
    with pytest.raises(ValueError, match="malformed-plan"):
        fp.plan_from_dict(raw)


def test_parser_rejects_track_frames_outside_its_scene():
    raw = fp.plan_to_dict(_normal_plan())
    raw["scenes"][0]["tracks"][0]["first_frame"] = 170
    raw["scenes"][0]["tracks"][0]["last_frame"] = 179
    # This remains inside the whole clip, but is inconsistent with the track's
    # segment evidence near the beginning. A strict parser must reject it.
    with pytest.raises(ValueError, match="malformed-plan"):
        fp.plan_from_dict(raw)


def test_parser_rejects_gap_between_segments():
    raw = fp.plan_to_dict(_normal_plan())
    segments = raw["scenes"][0]["segments"]
    assert len(segments) >= 2
    assert segments[1]["start_frame"] + 1 <= segments[1]["end_frame"]
    segments[1]["start_frame"] += 1
    with pytest.raises(ValueError, match="malformed-plan"):
        fp.plan_from_dict(raw)


def test_parser_rejects_overlapping_scene_intervals():
    empty = fp.ObservationLog(640, 360)
    plan = fp.build_framing_plan(
        log=empty,
        scenes=[(0, 59), (60, 119)],
        strategies=["GENERAL", "GENERAL"],
        config=fp.TurnFramingConfig(),
        fps=30.0,
        frame_w=640,
        frame_h=360,
        frame_count=120,
        source_fingerprint="review2-scenes",
        aspect=9 / 16,
        tracker_digest="tracker-digest",
    )
    raw = fp.plan_to_dict(plan)
    raw["scene_intervals"][1] = [50, 119]
    raw["scenes"][1]["start_frame"] = 50
    with pytest.raises(ValueError, match="malformed-plan"):
        fp.plan_from_dict(raw)


# --- mutation gates --------------------------------------------------------

def _roundtrip(plan: fp.FramingPlan) -> fp.FramingPlan:
    """JSON round-trip through the strict parser, as the reuse path does."""
    return fp.plan_from_dict(json.loads(json.dumps(fp.plan_to_dict(plan))))


def _match_kwargs() -> dict:
    return dict(
        source_fingerprint="review2-source",
        frame_w=640, frame_h=360, fps=30.0, frame_count=180,
        scene_intervals=[(0, 179)],
        config_digest=fp.TurnFramingConfig().config_digest(),
    )


def test_incomplete_plan_is_persisted_but_not_reusable():
    # The exhausted plan may be saved for audit and safe rendering, but the
    # next invocation must re-analyze instead of reusing it. Removing the
    # analysis_complete check in plan_matches() makes this pass wrongly.
    plan = _roundtrip(_build())
    assert plan.analysis_complete is False
    ok, why = fp.plan_matches(plan, **_match_kwargs())
    assert ok is False
    assert why == "incomplete-analysis"


def test_complete_plan_still_reusable_after_roundtrip():
    plan = _roundtrip(_normal_plan())
    assert plan.analysis_complete is True
    ok, why = fp.plan_matches(plan, **_match_kwargs())
    assert (ok, why) == (True, "match")


def test_analysis_deadline_serialized_in_config_dict():
    # Mutation gate: dropping analysis_deadline_s from to_dict() keeps the
    # digests equal and breaks the identity contract.
    d = fp.TurnFramingConfig().to_dict()
    assert d["analysis_deadline_s"] == 120.0
    assert "max_analysis_frames" in d


def test_fully_analyzed_earlier_scene_stays_planned():
    # Structured exhaustion at frame 100 of scene 1: scene 0 ([0, 59]) was
    # fully analyzed and may keep its plan; scene 1 falls back.
    log = fp.ObservationLog(640, 360)
    for frame in range(0, 60, 2):
        log.add(fp.FaceObservation(
            frame=frame, time_s=frame / 30.0, scene_id=0, track_id="t-0",
            box=(40, 80, 100, 100), visibility=0.95, mouth_motion=2.0,
            provenance="injected-test", reason="review2"))
    for frame in range(60, 80, 2):
        log.add(fp.FaceObservation(
            frame=frame, time_s=frame / 30.0, scene_id=1, track_id="t-2",
            box=(40, 80, 100, 100), visibility=0.95, mouth_motion=2.0,
            provenance="injected-test", reason="review2"))
    log.mark_exhaustion("analysis_budget_exhausted", frame=100, scene_id=1)
    plan = fp.build_framing_plan(
        log=log, scenes=[(0, 59), (60, 179)],
        strategies=["TRACK", "TRACK"], config=fp.TurnFramingConfig(),
        fps=30.0, frame_w=640, frame_h=360, frame_count=180,
        source_fingerprint="review2-multi", aspect=9 / 16,
        tracker_digest="tracker-digest")
    assert plan.scenes[0].usable is True
    assert plan.scenes[0].segments
    assert plan.scenes[1].usable is False
    assert plan.scenes[1].segments == []
    assert plan.scenes[1].fallback["reason"] == "analysis_budget_exhausted"
    assert plan.analysis_complete is False


def test_exhaustion_without_frame_context_falls_back_all_face_scenes():
    # A bare diagnostic note proves nothing about coverage: every face scene
    # must fall back conservatively.
    log = fp.ObservationLog(640, 360)
    for frame in range(0, 60, 2):
        log.add(fp.FaceObservation(
            frame=frame, time_s=frame / 30.0, scene_id=0, track_id="t-0",
            box=(40, 80, 100, 100), visibility=0.95, mouth_motion=2.0,
            provenance="injected-test", reason="review2"))
    log.note("analysis-frame-budget-exhausted")
    plan = fp.build_framing_plan(
        log=log, scenes=[(0, 59), (60, 179)],
        strategies=["TRACK", "TRACK"], config=fp.TurnFramingConfig(),
        fps=30.0, frame_w=640, frame_h=360, frame_count=180,
        source_fingerprint="review2-multi", aspect=9 / 16,
        tracker_digest="tracker-digest")
    assert all(s.usable is False for s in plan.scenes)
    assert all(s.fallback["reason"] == "analysis-frame-budget-exhausted"
               for s in plan.scenes)


def test_parser_rejects_two_subject_duplicate_ids():
    raw = fp.plan_to_dict(_normal_plan())
    seg = raw["scenes"][0]["segments"][0]
    seg["kind"] = "two-subject"
    seg["subject_ids"] = ["t-0", "t-0"]
    with pytest.raises(ValueError, match="malformed-plan"):
        fp.plan_from_dict(raw)


def test_parser_rejects_fallback_scene_carrying_segments():
    raw = fp.plan_to_dict(_normal_plan())
    raw["scenes"][0]["fallback"] = {"reason": "injected"}
    with pytest.raises(ValueError, match="malformed-plan"):
        fp.plan_from_dict(raw)


def test_parser_rejects_nonapplicable_scene_carrying_content():
    raw = fp.plan_to_dict(_normal_plan())
    raw["scenes"][0]["applies"] = False
    with pytest.raises(ValueError, match="malformed-plan"):
        fp.plan_from_dict(raw)


def test_parser_rejects_missing_analysis_complete():
    raw = fp.plan_to_dict(_normal_plan())
    del raw["analysis_complete"]
    with pytest.raises(ValueError, match="malformed-plan"):
        fp.plan_from_dict(raw)


def test_parser_rejects_nonbool_analysis_complete():
    raw = fp.plan_to_dict(_normal_plan())
    raw["analysis_complete"] = 1
    with pytest.raises(ValueError, match="malformed-plan"):
        fp.plan_from_dict(raw)


def test_mark_exhaustion_first_call_wins():
    log = fp.ObservationLog(640, 360)
    log.mark_exhaustion("analysis_budget_exhausted", frame=50, scene_id=0)
    log.mark_exhaustion("analysis-frame-budget-exhausted", frame=10, scene_id=0)
    assert log.exhaustion_reason == "analysis_budget_exhausted"
    assert log.exhaustion_frame == 50
    assert log.diagnostics.count("analysis_budget_exhausted") == 1
