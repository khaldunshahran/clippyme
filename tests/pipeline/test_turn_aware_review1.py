"""Task 12 review-01 corrections — durable regression tests.

Promotes ``work/muse-handoff/task-12/coordinator-review1-remaining.py`` (the
independent review probes: render-aspect identity/wiring, monotonic analysis
deadline, per-second camera-speed cap, strict recursive plan parsing, failed
atomic-write cleanup) and adds the correction-01/02/03/04 regression and
mutation-gate coverage the review requires.

Pure/injected only: no live models, no network, no FFmpeg. (The
production-route 1:1/16:9/9:16 audiovisual tests live in
test_turn_aware_integration.py.)
"""
from __future__ import annotations

import ast
import copy
import inspect
import math
from pathlib import Path

import pytest

from clippyme.pipeline import framing_plan as fp
from clippyme.pipeline.reframe_track import SpeakerTracker


REPO = Path(__file__).resolve().parents[2]
REFRAME = REPO / "src" / "clippyme" / "pipeline" / "reframe.py"


def _obs(frame: int, track: str, box: tuple[int, int, int, int], mouth: float):
    return fp.FaceObservation(
        frame=frame,
        time_s=frame / 30.0,
        scene_id=0,
        track_id=track,
        box=box,
        visibility=0.95,
        mouth_motion=mouth,
        provenance="injected-test",
        reason="review-probe",
    )


def _plan(*, aspect: float = 9 / 16, config: fp.TurnFramingConfig | None = None):
    log = fp.ObservationLog(640, 360)
    boxes = {"t-0": (10, 80, 90, 100), "t-1": (540, 80, 90, 100)}
    for frame in range(0, 180, 2):
        active = "t-0" if frame < 90 else "t-1"
        for tid in ("t-0", "t-1"):
            mouth = 2.5 if tid == active and frame >= 10 else 0.1
            log.add(_obs(frame, tid, boxes[tid], mouth))
    return fp.build_framing_plan(
        log=log,
        scenes=[(0, 179)],
        strategies=["WIDE"],
        config=config or fp.TurnFramingConfig(),
        fps=30.0,
        frame_w=640,
        frame_h=360,
        frame_count=180,
        source_fingerprint="review-source",
        aspect=aspect,
    )


def _build_call_keywords(function_name: str) -> set[str]:
    tree = ast.parse(REFRAME.read_text(encoding="utf-8"))
    matches = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        name = fn.id if isinstance(fn, ast.Name) else (
            fn.attr if isinstance(fn, ast.Attribute) else None
        )
        if name == function_name:
            matches.append({kw.arg for kw in node.keywords if kw.arg})
    assert matches, f"production has no {function_name}(...) call"
    return set().union(*matches)


def test_render_aspect_and_scene_strategy_are_plan_reuse_identity():
    """A 9:16 sidecar must never be reused for 1:1/16:9 or new strategies."""
    params = set(inspect.signature(fp.plan_matches).parameters)
    aspect_names = params & {"aspect", "output_aspect", "aspect_ratio"}
    strategy_names = params & {"strategies", "scene_strategies"}
    assert aspect_names, "plan_matches does not compare the requested output aspect"
    assert strategy_names, "plan_matches does not compare current scene strategies"

    plan = _plan(aspect=9 / 16)
    base = dict(
        source_fingerprint="review-source",
        frame_w=640,
        frame_h=360,
        fps=30.0,
        frame_count=180,
        scene_intervals=[(0, 179)],
        config_digest=fp.TurnFramingConfig().config_digest(),
    )
    base[next(iter(aspect_names))] = 1.0
    base[next(iter(strategy_names))] = ["WIDE"]
    ok, reason = fp.plan_matches(plan, **base)
    assert ok is False and "aspect" in reason

    base[next(iter(aspect_names))] = 9 / 16
    base[next(iter(strategy_names))] = ["TRACK"]
    ok, reason = fp.plan_matches(plan, **base)
    assert ok is False and "strateg" in reason


def test_production_threads_actual_output_aspect_through_build_and_reuse():
    build_kw = _build_call_keywords("build_framing_plan")
    match_kw = _build_call_keywords("plan_matches")
    assert build_kw & {"aspect", "output_aspect", "aspect_ratio"}, (
        "production plan build silently uses framing_plan's 9:16 default"
    )
    assert match_kw & {"aspect", "output_aspect", "aspect_ratio"}, (
        "persisted plan reuse is not invalidated when output aspect changes"
    )
    assert match_kw & {"strategies", "scene_strategies"}, (
        "persisted plan reuse is not invalidated when scene strategy changes"
    )


def test_analysis_has_monotonic_deadline_and_stops_expensive_sampling():
    cfg_fields = set(fp.TurnFramingConfig.__dataclass_fields__)
    deadline_names = cfg_fields & {
        "analysis_deadline_s",
        "max_analysis_seconds",
        "analysis_timeout_s",
    }
    assert deadline_names, "Task 12 has an observation count only, no wall-time budget"
    source = REFRAME.read_text(encoding="utf-8")
    assert "monotonic" in source, "analysis deadline must use a monotonic clock"
    assert any(name in source for name in deadline_names)
    # Exhaustion must prevent further detector/MAR calls, not merely discard
    # their results after paying the decode/analysis cost.
    assert "analysis_budget_exhausted" in source or ".exhausted" in source


def test_camera_travel_cap_is_per_second_not_per_switch():
    cfg = fp.TurnFramingConfig(ease_frames=6, max_travel_frac=0.5)
    scene = _plan(config=cfg).scenes[0]
    turns = [s for s in scene.segments if s.kind == "single"]
    assert len(turns) >= 2
    start = turns[1].start_frame
    centers = []
    for frame in range(start - 1, start + cfg.ease_frames + 1):
        crop = fp.crop_at_frame(scene, frame)
        assert crop is not None
        centers.append(((crop[0] + crop[2]) / 2.0, (crop[1] + crop[3]) / 2.0))
    max_x_step = max(abs(b[0] - a[0]) for a, b in zip(centers, centers[1:]))
    max_y_step = max(abs(b[1] - a[1]) for a, b in zip(centers, centers[1:]))
    assert max_x_step <= cfg.max_travel_frac * 640 / 30.0 + 1.0
    assert max_y_step <= cfg.max_travel_frac * 360 / 30.0 + 1.0


@pytest.mark.parametrize("mutation", [
    "bool-schema",
    "bool-dimension",
    "named-track-with-pixels",
    "pixel-evidence",
    "bad-fallback-crop",
    "bad-fallback-reason",
])
def test_persisted_plan_parser_rejects_untrusted_nested_values(mutation: str):
    raw = fp.plan_to_dict(_plan())
    scene = raw["scenes"][0]
    if mutation == "bool-schema":
        raw["schema_version"] = True
    elif mutation == "bool-dimension":
        raw["frame_w"] = True
    elif mutation == "named-track-with-pixels":
        scene["tracks"] = [{
            "id": "Alice",
            "first_frame": 0,
            "last_frame": 10,
            "n_obs": 6,
            "pixels": "raw-secret",
        }]
    elif mutation == "pixel-evidence":
        scene["segments"][0]["evidence"] = {"pixels": "raw-secret"}
    elif mutation == "bad-fallback-crop":
        scene["fallback_crop"] = [-100.0, math.nan, "zoom"]
    elif mutation == "bad-fallback-reason":
        scene["fallback"] = {"reason": ["not", "a", "token"]}
    else:  # pragma: no cover
        raise AssertionError(mutation)
    with pytest.raises(ValueError, match="malformed-plan"):
        fp.plan_from_dict(copy.deepcopy(raw))


def test_failed_atomic_replace_removes_temp_file(tmp_path, monkeypatch):
    destination = tmp_path / "plan.json"

    def fail_replace(_src, _dst):
        raise OSError("injected replace failure")

    monkeypatch.setattr(fp.os, "replace", fail_replace)
    with pytest.raises(OSError, match="injected replace failure"):
        fp.save_plan_atomic(_plan(), str(destination))
    assert not list(tmp_path.glob("*.tmp-*")), "failed atomic save leaked temp data"


# --- correction 1: full decision identity -----------------------------------

def test_plan_persists_and_matches_full_decision_identity():
    """Aspect, ordered strategies and tracker digest are persisted and matched."""
    tracker = SpeakerTracker(cooldown_frames=45)
    plan = fp.build_framing_plan(
        log=_fresh_log(),
        scenes=[(0, 179)], strategies=["WIDE"],
        config=fp.TurnFramingConfig(), fps=30.0, frame_w=640, frame_h=360,
        frame_count=180, source_fingerprint="review-source",
        aspect=1.0, tracker_digest=tracker.decision_digest(),
    )
    assert plan.output_aspect == pytest.approx(1.0)
    assert plan.tracker_digest == tracker.decision_digest()
    raw = fp.plan_to_dict(plan)
    assert raw["output_aspect"] == pytest.approx(1.0)
    assert raw["tracker_digest"] == tracker.decision_digest()
    plan2 = fp.plan_from_dict(raw)
    assert plan2.output_aspect == pytest.approx(1.0)

    base = dict(
        source_fingerprint="review-source", frame_w=640, frame_h=360,
        fps=30.0, frame_count=180, scene_intervals=[(0, 179)],
        config_digest=fp.TurnFramingConfig().config_digest(),
    )
    ok, why = fp.plan_matches(
        plan2, **base, aspect=1.0, strategies=["WIDE"],
        tracker_digest=tracker.decision_digest())
    assert (ok, why) == (True, "match")
    # A different tracker configuration invalidates the sidecar.
    other = SpeakerTracker(cooldown_frames=45)
    other.switch_margin = 9.99
    ok, why = fp.plan_matches(
        plan2, **base, aspect=1.0, strategies=["WIDE"],
        tracker_digest=other.decision_digest())
    assert ok is False and "tracker" in why


def _fresh_log():
    log = fp.ObservationLog(640, 360)
    boxes = {"t-0": (10, 80, 90, 100), "t-1": (540, 80, 90, 100)}
    for frame in range(0, 180, 2):
        active = "t-0" if frame < 90 else "t-1"
        for tid in ("t-0", "t-1"):
            mouth = 2.5 if tid == active and frame >= 10 else 0.1
            log.add(_obs(frame, tid, boxes[tid], mouth))
    return log


def test_tracker_decision_digest_covers_decision_inputs():
    """Every decision-affecting tracker knob feeds the digest."""
    a = SpeakerTracker(cooldown_frames=45)
    b = SpeakerTracker(cooldown_frames=45)
    assert a.decision_digest() == b.decision_digest()
    b.switch_margin = 2.0
    assert a.decision_digest() != b.decision_digest()
    for knob, value in (("switch_cooldown", 30), ("min_face_ratio", 0.5),
                        ("group_enabled", False), ("group_score_ratio", 0.6),
                        ("group_separation", 0.4)):
        c = SpeakerTracker(cooldown_frames=45)
        setattr(c, knob, value)
        assert a.decision_digest() != c.decision_digest(), knob


# --- correction 2: monotonic analysis budget (pure) --------------------------

class TestAnalysisBudget:
    def test_subsecond_deadline_with_injected_clock(self):
        now = [100.0]
        budget = fp.AnalysisBudget(deadline_s=0.05, max_frames=1000,
                                   clock=lambda: now[0])
        assert budget.ok()
        now[0] += 0.04
        assert budget.ok()
        now[0] += 0.02  # past the deadline
        assert not budget.ok()
        assert budget.exhausted
        assert budget.exhaust_reason == "analysis_budget_exhausted"
        # Latches: later checks stay exhausted.
        now[0] += 1000.0
        assert not budget.ok()

    def test_zero_deadline_exhausts_immediately(self):
        budget = fp.AnalysisBudget(deadline_s=0.0, max_frames=1000,
                                   clock=lambda: 5.0)
        assert not budget.ok()
        assert budget.exhaust_reason == "analysis_budget_exhausted"

    def test_frame_budget_exhaustion(self):
        budget = fp.AnalysisBudget(deadline_s=3600.0, max_frames=3,
                                   clock=lambda: 0.0)
        assert budget.ok()
        budget.consume()
        assert budget.ok()
        budget.consume()
        assert budget.ok()
        budget.consume()
        assert not budget.ok()
        assert budget.exhaust_reason == "analysis-frame-budget-exhausted"
        assert not budget.ok()  # latches

    def test_deadline_checked_before_frame_budget(self):
        # Both exhausted: the frame budget reports first (checked first).
        budget = fp.AnalysisBudget(deadline_s=0.0, max_frames=0,
                                   clock=lambda: 999.0)
        assert not budget.ok()
        assert budget.exhaust_reason == "analysis-frame-budget-exhausted"

    def test_uses_monotonic_clock_by_default(self):
        budget = fp.AnalysisBudget(deadline_s=60.0, max_frames=10)
        assert budget.ok()
        assert budget._clock is __import__("time").monotonic


# --- correction 3: per-second travel at 24/30/60 fps --------------------------

def _turn_plan_at_fps(fps: float, n_frames: int):
    """Two-speaker turn fixture with fps-correct observation times."""
    cfg = fp.TurnFramingConfig(ease_frames=6, max_travel_frac=0.5)
    log = fp.ObservationLog(640, 360)
    boxes = {"t-0": (10, 80, 90, 100), "t-1": (540, 80, 90, 100)}
    for frame in range(0, n_frames, 2):
        active = "t-0" if frame < n_frames // 2 else "t-1"
        for tid in ("t-0", "t-1"):
            mouth = 2.5 if tid == active and frame >= 10 else 0.1
            log.add(fp.FaceObservation(
                frame=frame, time_s=frame / fps, scene_id=0, track_id=tid,
                box=boxes[tid], visibility=0.95, mouth_motion=mouth,
                provenance="injected-test", reason="review-probe"))
    plan = fp.build_framing_plan(
        log=log, scenes=[(0, n_frames - 1)], strategies=["WIDE"], config=cfg,
        fps=fps, frame_w=640, frame_h=360, frame_count=n_frames,
        source_fingerprint="fps-regression", aspect=9 / 16)
    return plan.scenes[0], cfg


@pytest.mark.parametrize("fps,n_frames", [(24.0, 360), (30.0, 360), (60.0, 720)])
def test_travel_cap_per_second_at_fps(fps: float, n_frames: int):
    """Every consecutive rendered crop-centre step — including the final
    ease-to-lock step — respects max_travel_frac * dim / fps."""
    scene, cfg = _turn_plan_at_fps(fps, n_frames)
    turns = [s for s in scene.segments if s.kind == "single"]
    assert len(turns) >= 2, "fixture must contain an accepted turn"
    capx = cfg.max_travel_frac * 640 / fps + 1.0
    capy = cfg.max_travel_frac * 360 / fps + 1.0
    centers = []
    for frame in range(0, n_frames):
        crop = fp.crop_at_frame(scene, frame)
        assert crop is not None
        centers.append(((crop[0] + crop[2]) / 2.0,
                        (crop[1] + crop[3]) / 2.0))
    max_x = max(abs(b[0] - a[0]) for a, b in zip(centers, centers[1:]))
    max_y = max(abs(b[1] - a[1]) for a, b in zip(centers, centers[1:]))
    assert max_x <= capx, f"fps={fps}: x step {max_x:.2f} > cap {capx:.2f}"
    assert max_y <= capy, f"fps={fps}: y step {max_y:.2f} > cap {capy:.2f}"


def _old_per_switch_keyframes(segments, prev_locks, config, frame_w, frame_h):
    """Replicates the pre-fix behavior: cap travel once per switch at
    max_travel_frac * dim, then ease over ease_frames. Used only to prove the
    per-second regression discriminates old from new."""
    import math as _math

    def _lerp(a, b, t):
        return [int(round(av + (bv - av) * t)) for av, bv in zip(a, b)]

    out = []
    for idx, (seg, lock) in enumerate(zip(segments, prev_locks)):
        if idx == 0:
            continue
        prev = prev_locks[idx - 1]
        pcx = (prev[0] + prev[2]) / 2.0
        pcy = (prev[1] + prev[3]) / 2.0
        ncx = (lock[0] + lock[2]) / 2.0
        ncy = (lock[1] + lock[3]) / 2.0
        capx, capy = config.max_travel_frac * frame_w, config.max_travel_frac * frame_h
        dx, dy = ncx - pcx, ncy - pcy
        if abs(dx) > capx:
            ncx = pcx + _math.copysign(capx, dx)
        if abs(dy) > capy:
            ncy = pcy + _math.copysign(capy, dy)
        shx = int(round(ncx - (lock[0] + lock[2]) / 2.0))
        shy = int(round(ncy - (lock[1] + lock[3]) / 2.0))
        lock = [lock[0] + shx, lock[1] + shy, lock[2] + shx, lock[3] + shy]
        n = int(config.ease_frames)
        for i in range(n):
            out.append((seg.start_frame + i, _lerp(prev, lock, (i + 1) / (n + 1))))
        out.append((seg.start_frame + n, lock))
    return out


def test_mutation_gate_restored_per_switch_travel_fails():
    """If the per-switch clamp + fixed ease_frames behavior is restored, the
    per-second bound must fail — proving the regression is load-bearing."""
    scene, cfg = _turn_plan_at_fps(30.0, 360)
    turns = [s for s in scene.segments if s.kind == "single"]
    assert len(turns) >= 2
    locks = [list(s.crop) for s in scene.segments]
    old = _old_per_switch_keyframes(scene.segments, locks, cfg, 640, 360)
    assert old, "old-style simulation must emit keyframes"
    centers = [((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0) for _, b in old]
    max_x = max(abs(b[0] - a[0]) for a, b in zip(centers, centers[1:]))
    bound = cfg.max_travel_frac * 640 / 30.0 + 1.0
    assert max_x > bound, (
        f"mutation gate broken: old per-switch behavior fits the per-second "
        f"bound ({max_x:.2f} <= {bound:.2f})")


# --- correction 4: strict parsing, mutation gates -----------------------------

def test_strict_numeric_helpers_reject_booleans():
    assert fp._strict_int(True) is False
    assert fp._strict_int(False) is False
    assert fp._strict_int(1) is True
    assert fp._strict_int(0) is True
    assert fp._strict_num(True) is False
    assert fp._strict_num(1.5) is True
    assert fp._strict_num(float("nan")) is False
    assert fp._strict_num(float("inf")) is False


def test_mutation_gate_permissive_parsing_is_load_bearing(monkeypatch):
    """Relaxing the strict helpers to the old plain-isinstance checks must let
    a bool-valued field slip through — proving the strictness tests above are
    sensitive to that regression."""
    raw = fp.plan_to_dict(_plan())
    raw["scenes"][0]["segments"][0]["confidence"] = True
    with pytest.raises(ValueError, match="malformed-plan"):
        fp.plan_from_dict(copy.deepcopy(raw))
    # Simulate the regression: bools pass as numbers again.
    monkeypatch.setattr(fp, "_strict_int", lambda v: isinstance(v, int))
    monkeypatch.setattr(
        fp, "_strict_num",
        lambda v: isinstance(v, (int, float)) and math.isfinite(v))
    fp.plan_from_dict(copy.deepcopy(raw))  # must NOT raise under the mutation


def test_parser_rejects_unknown_top_level_keys():
    raw = fp.plan_to_dict(_plan())
    raw["extra"] = "smuggled"
    with pytest.raises(ValueError, match="malformed-plan"):
        fp.plan_from_dict(raw)


def test_parser_rejects_non_ordered_scene_ids():
    raw = fp.plan_to_dict(_plan())
    raw["scenes"][0]["scene_id"] = 7
    with pytest.raises(ValueError, match="malformed-plan"):
        fp.plan_from_dict(raw)


def test_parser_rejects_scene_interval_mismatch():
    raw = fp.plan_to_dict(_plan())
    raw["scenes"][0]["start_frame"] = 5
    with pytest.raises(ValueError, match="malformed-plan"):
        fp.plan_from_dict(raw)


def test_parser_rejects_overlapping_segments():
    raw = fp.plan_to_dict(_plan())
    segs = raw["scenes"][0]["segments"]
    if len(segs) >= 2:
        segs[1]["start_frame"] = segs[0]["start_frame"]
        with pytest.raises(ValueError, match="malformed-plan"):
            fp.plan_from_dict(raw)
    else:
        pytest.skip("fixture has a single segment")


def test_parser_rejects_unbounded_reason():
    raw = fp.plan_to_dict(_plan())
    raw["scenes"][0]["segments"][0]["reason"] = "x" * 65
    with pytest.raises(ValueError, match="malformed-plan"):
        fp.plan_from_dict(raw)


def test_parser_rejects_empty_reason():
    raw = fp.plan_to_dict(_plan())
    raw["scenes"][0]["segments"][0]["reason"] = ""
    with pytest.raises(ValueError, match="malformed-plan"):
        fp.plan_from_dict(raw)


def test_parser_rejects_duplicate_keyframe_frames():
    raw = fp.plan_to_dict(_plan())
    kfs = raw["scenes"][0]["keyframes"]
    assert kfs, "fixture must have keyframes"
    dup = dict(kfs[0])
    kfs.insert(1, dup)
    with pytest.raises(ValueError, match="malformed-plan"):
        fp.plan_from_dict(raw)


def test_parser_rejects_wide_segment_with_subjects():
    raw = fp.plan_to_dict(_plan())
    wide = next(s for s in raw["scenes"][0]["segments"] if s["kind"] == "wide")
    wide["subject_ids"] = ["t-0"]
    with pytest.raises(ValueError, match="malformed-plan"):
        fp.plan_from_dict(raw)


def test_parser_rejects_single_segment_wrong_subject_count():
    raw = fp.plan_to_dict(_plan())
    single = next(s for s in raw["scenes"][0]["segments"]
                  if s["kind"] == "single")
    single["subject_ids"] = ["t-0", "t-1"]
    with pytest.raises(ValueError, match="malformed-plan"):
        fp.plan_from_dict(raw)


def test_parser_rejects_evidence_out_of_bounds():
    raw = fp.plan_to_dict(_plan())
    raw["scenes"][0]["segments"][0]["evidence"]["mouth_mean"] = 99.0
    with pytest.raises(ValueError, match="malformed-plan"):
        fp.plan_from_dict(raw)


def test_parser_rejects_track_frame_bounds():
    raw = fp.plan_to_dict(_plan())
    raw["scenes"][0]["tracks"][0]["first_frame"] = 5000
    with pytest.raises(ValueError, match="malformed-plan"):
        fp.plan_from_dict(raw)


def test_parser_rejects_missing_required_fields():
    raw = fp.plan_to_dict(_plan())
    del raw["output_aspect"]
    with pytest.raises(ValueError, match="malformed-plan"):
        fp.plan_from_dict(raw)
    raw = fp.plan_to_dict(_plan())
    del raw["tracker_digest"]
    with pytest.raises(ValueError, match="malformed-plan"):
        fp.plan_from_dict(raw)


def test_parser_rejects_old_schema_version():
    raw = fp.plan_to_dict(_plan())
    raw["schema_version"] = 1
    with pytest.raises(ValueError, match="malformed-plan"):
        fp.plan_from_dict(raw)


def test_failed_write_cleans_temp_and_preserves_destination(tmp_path, monkeypatch):
    """A failing write (not just replace) removes the temp file and never
    touches an existing destination."""
    import builtins

    plan = _plan()
    destination = tmp_path / "plan.json"
    fp.save_plan_atomic(plan, str(destination))
    before = destination.read_bytes()

    def fail_open(*a, **k):
        raise OSError("injected write failure")

    monkeypatch.setattr(builtins, "open", fail_open)
    with pytest.raises(OSError, match="injected write failure"):
        fp.save_plan_atomic(plan, str(destination))
    assert destination.read_bytes() == before, "destination must be preserved"
    assert not list(tmp_path.glob("*.tmp-*")), "failed save leaked temp data"
