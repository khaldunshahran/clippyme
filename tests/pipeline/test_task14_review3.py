"""Task 14 review-03: explicit fit is authoritative; timed fit ranges preserved.

Promotes coordinator-review3-remaining.py (both probes) into the durable
suite, extended per review-03:

Correction 1 -- explicit fit is authoritative during rendering:
- the action fit branch renders through a deterministic full-frame fit
  renderer that bypasses the optional object/saliency analysis in
  create_general_frame (the legacy GENERAL helper keeps its opt-in
  behavior for callers WITHOUT such a plan);
- verified with BOTH optional modes enabled, on first render AND sidecar
  reuse: zero optional analysis calls, edge containment, geometry, audio;
- the fit renderer is byte-identical to the legacy GENERAL letterbox
  when the opt-ins are disabled.

Correction 2 -- timed fit ranges survive ClipProject:
- normalize_timeline carries lossless ordered clip-relative fit intervals
  through the additive project model, schema twin, normalization,
  timeline digest, and lineage geometry identity (one integer rebase);
- booleans, coverage/order/bounds, and consistency with crop ranges are
  validated; old projects validate unchanged;
- differing fit boundaries stay distinct through real emission,
  save/load/version/undo, and unchanged-base reuse; edited/malformed
  timing cannot silently reuse stale pixels.
"""

import copy
import importlib.util
import json
import os
from pathlib import Path

import pytest

from clippyme.pipeline import action_framing as af, framing_plan as fp
from clippyme.domain import project_framing as pf
from clippyme.domain.clip_project import (
    FramingSceneTimeline,
    validate_project,
)


def _fixture():
    repo = Path(__file__).resolve().parents[2]  # repo root (clippyme/)
    spec = importlib.util.spec_from_file_location(
        "_r3_render_fixture",
        repo / "tests/pipeline/test_action_framing_render.py")
    f = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(f)
    return f


def _edge_frame(np_mod):
    def edge_frame(i):
        image = np_mod.zeros((480, 640, 3), dtype=np_mod.uint8)
        image[180:300, 20:80] = (0, 255, 0)
        image[180:300, 560:620] = (255, 0, 0)
        return image
    return edge_frame


def _install_all_fit(f, monkeypatch, tmp_path):
    """Render setup: all segments fit, both optional modes armed."""
    monkeypatch.setattr(f, "_frame", _edge_frame(f.np))
    src = f._build_src(tmp_path / "src.mp4", tmp_path)
    f._install(monkeypatch, detect_fn=lambda frame: [
        {"box": [20., 180., 60., 120.], "confidence": .1,
         "category": "object"}],
        observe_fn=lambda *args: ([.1], [.5], 0.))
    # BOTH optional modes enabled: the persisted all-fit plan must invoke
    # neither.
    monkeypatch.setenv("REFRAME_OBJECT_WEIGHTS", "person:1")
    monkeypatch.setenv("REFRAME_SALIENT_GENERAL", "1")
    calls = {"object": 0, "salient": 0}

    def optional_object_crop(frame, w, h, weights=None):
        calls["object"] += 1
        return f.cv2.resize(frame[:, 185:455], (w, h))

    def salient_crop(frame, w, h):
        calls["salient"] += 1
        return f.cv2.resize(frame[:, 185:455], (w, h))

    monkeypatch.setattr(f.reframe, "_weighted_object_general_crop",
                        optional_object_crop)
    monkeypatch.setattr(f.reframe, "_salient_general_crop", salient_crop)
    return src, calls


def _dims(f, path):
    st = f._probe(path)["streams"][0]
    return int(st["width"]), int(st["height"])


def _assert_fit_render(f, out):
    plan = fp.load_plan(os.path.splitext(out)[0] + ".turn_framing_plan.json")
    assert all(s.fit for scene in plan.scenes for s in scene.segments)
    for frame in f._read_frames(out, [30, 90]).values():
        assert f._green_present(frame, thresh=100)
        assert f._blue_present(frame, thresh=100)
    assert _dims(f, out) == (270, 480)
    assert f._tone_hz(out) == pytest.approx(880, abs=40)


# --- Correction 1: explicit fit is authoritative ----------------------------


def test_project_bridge_preserves_fit_interval_boundaries():
    log = af.ActionObservationLog(640, 480)
    for i in range(90):
        log.add(af.ActionObservation(i, i / 30., 0, 't-0',
                                    (240., 180., 60., 120.), .9,
                                    'object', .1, .5, 1., 'injected-test', 'test'))
    plan = fp.build_framing_plan(log=fp.ObservationLog(640, 480),
                                scenes=[(0, 89)], strategies=['GENERAL'],
                                config=fp.TurnFramingConfig(), fps=30.,
                                frame_w=640, frame_h=480, frame_count=90,
                                source_fingerprint='x', action_log=log,
                                action_config=af.ActionFramingConfig())
    first = fp.plan_to_dict(plan)
    scene = first['scenes'][0]
    assert [s['kind'] for s in scene['segments']] == ['wide', 'single']
    # Both independently valid plans carry the same crop lock at frame30,
    # but fit ends at frame14 in A and frame19 in B.
    scene['segments'][0]['end_frame'] = 14
    scene['segments'][1]['start_frame'] = 15
    scene['keyframes'][0]['frame'] = 30
    second = copy.deepcopy(first)
    second['scenes'][0]['segments'][0]['end_frame'] = 19
    second['scenes'][0]['segments'][1]['start_frame'] = 20
    a, b = fp.plan_from_dict(first), fp.plan_from_dict(second)
    assert not fp.segment_fit_at_frame(a.scenes[0], 17)
    assert fp.segment_fit_at_frame(b.scenes[0], 17)
    na = pf.normalize_timeline(plan=first, slice_offset_frames=0, clip_frame_count=90, fps=30.)
    nb = pf.normalize_timeline(plan=second, slice_offset_frames=0, clip_frame_count=90, fps=30.)
    assert na != nb, 'different timed fit decisions collapsed to the same project timeline'
    assert pf.timeline_digest(na) != pf.timeline_digest(nb)


def test_explicit_fit_never_invokes_optional_analysis(tmp_path, monkeypatch):
    """Both optional modes on, first render AND sidecar reuse.

    A persisted all-fit plan must render full-frame letterbox on all 120
    frames: zero optional object/saliency analysis calls on either render,
    edge containment, 270x480 geometry, and the 880 Hz audio tone.
    """
    f = _fixture()
    src, calls = _install_all_fit(f, monkeypatch, tmp_path)
    out = str(tmp_path / "out.mp4")
    assert f.reframe.process_video_to_vertical(
        src, out, reframe_mode="auto") is True
    assert calls == {"object": 0, "salient": 0}, \
        f"first render invoked optional analysis: {calls}"
    _assert_fit_render(f, out)
    # Sidecar reuse: second render to the same path must not recompute
    # framing and must not invoke optional analysis either.
    assert f.reframe.process_video_to_vertical(
        src, out, reframe_mode="auto") is True
    assert calls == {"object": 0, "salient": 0}, \
        f"reuse render invoked optional analysis: {calls}"
    _assert_fit_render(f, out)


def test_fit_renderer_byte_identical_to_legacy_general():
    """With the opt-ins disabled the fit renderer is the legacy letterbox."""
    import numpy as np
    import os as _os
    _os.environ.pop("REFRAME_OBJECT_WEIGHTS", None)
    _os.environ.pop("REFRAME_SALIENT_GENERAL", None)
    from clippyme.pipeline import reframe
    rng = np.random.RandomState(7)
    frame = rng.randint(0, 255, (480, 640, 3)).astype("uint8")
    legacy = reframe.create_general_frame(frame, 270, 480)
    fit = reframe.create_general_fit_frame(frame, 270, 480)
    assert legacy.shape == fit.shape == (480, 270, 3)
    assert bool((legacy == fit).all())


def test_mutation_fit_branch_removed_invokes_optional_crop(tmp_path,
                                                           monkeypatch):
    """Fail-then-restore: the test catches removal of the fit renderer.

    Simulating the review-03 bug (fit branch routed back through
    create_general_frame) must invoke the optional crop and drop the edge
    participants; restoring the production branch must pass clean.
    """
    f = _fixture()
    from clippyme.pipeline import reframe
    src, calls = _install_all_fit(f, monkeypatch, tmp_path)

    real_fit = reframe.create_general_fit_frame

    def buggy_fit(frame, w, h):
        # The review-03 bug: route explicit fit through the GENERAL
        # helper, which honors the user's optional crop opt-ins.
        return reframe.create_general_frame(frame, w, h)

    # FAIL: with the bug in place, the optional crop runs and the edges
    # are removed.
    monkeypatch.setattr(reframe, "create_general_fit_frame", buggy_fit)
    out = str(tmp_path / "buggy.mp4")
    assert f.reframe.process_video_to_vertical(
        src, out, reframe_mode="auto") is True
    assert calls["object"] > 0, \
        "mutation must invoke the optional object crop"
    frames = f._read_frames(out, [30])
    assert not f._green_present(frames[30], thresh=100) or \
        not f._blue_present(frames[30], thresh=100), \
        "mutation must remove at least one edge participant"

    # RESTORE: the production fit renderer passes the same gate.
    monkeypatch.setattr(reframe, "create_general_fit_frame", real_fit)
    calls["object"] = 0
    calls["salient"] = 0
    out2 = str(tmp_path / "fixed.mp4")
    assert f.reframe.process_video_to_vertical(
        src, out2, reframe_mode="auto") is True
    assert calls == {"object": 0, "salient": 0}
    _assert_fit_render(f, out2)


# --- Correction 2: timed fit ranges survive ClipProject ---------------------


def _two_boundary_plans():
    """Two parser-valid plans differing only in fit boundary (14 vs 19)."""
    log = af.ActionObservationLog(640, 480)
    for i in range(90):
        log.add(af.ActionObservation(i, i / 30., 0, "t-0",
                                    (240., 180., 60., 120.), .9,
                                    "object", .1, .5, 1.,
                                    "injected-test", "test"))
    plan = fp.build_framing_plan(log=fp.ObservationLog(640, 480),
                                scenes=[(0, 89)], strategies=["GENERAL"],
                                config=fp.TurnFramingConfig(), fps=30.,
                                frame_w=640, frame_h=480, frame_count=90,
                                source_fingerprint="x", action_log=log,
                                action_config=af.ActionFramingConfig())
    first = fp.plan_to_dict(plan)
    scene = first["scenes"][0]
    assert [s["kind"] for s in scene["segments"]] == ["wide", "single"]
    scene["segments"][0]["end_frame"] = 14
    scene["segments"][1]["start_frame"] = 15
    scene["keyframes"][0]["frame"] = 30
    second = copy.deepcopy(first)
    second["scenes"][0]["segments"][0]["end_frame"] = 19
    second["scenes"][0]["segments"][1]["start_frame"] = 20
    return first, second


def test_fit_ranges_survive_clip_project_round_trip():
    """normalize -> model -> JSON -> model keeps the intervals lossless."""
    first, second = _two_boundary_plans()
    for raw, end in ((first, 14), (second, 19)):
        tl = pf.normalize_timeline(plan=raw, slice_offset_frames=0,
                                   clip_frame_count=90, fps=30.)
        assert len(tl) == 1
        scene = tl[0]
        assert scene["fit"] is True
        assert scene["fit_ranges"] == [
            {"start_frame": 0, "end_frame": end}]
        # Model validation (save/load shape).
        model = FramingSceneTimeline(**scene)
        assert [(r.start_frame, r.end_frame)
                for r in model.fit_ranges] == [(0, end)]
        dumped = json.loads(model.model_dump_json())
        assert dumped["fit_ranges"] == [
            {"start_frame": 0, "end_frame": end}]
        assert FramingSceneTimeline(**dumped).fit_ranges == \
            model.fit_ranges
        # Consistency with the render decision at every frame.
        parsed = fp.plan_from_dict(raw)
        for frame_no in range(90):
            in_range = any(r["start_frame"] <= frame_no <= r["end_frame"]
                           for r in scene["fit_ranges"])
            assert fp.segment_fit_at_frame(
                parsed.scenes[0], frame_no) == in_range


def test_old_project_without_fit_ranges_stays_valid():
    """Additive: projects saved before rev4 validate unchanged."""
    tl = pf.normalize_timeline(plan=_two_boundary_plans()[0],
                               slice_offset_frames=0,
                               clip_frame_count=90, fps=30.)
    scene = copy.deepcopy(tl[0])
    del scene["fit_ranges"]
    del scene["fit"]
    model = FramingSceneTimeline(**scene)
    assert model.fit is False
    assert model.fit_ranges == []


def test_malformed_fit_intervals_rejected():
    """Booleans, coverage/order/bounds, and keyframe consistency enforced."""
    first, _ = _two_boundary_plans()
    tl = pf.normalize_timeline(plan=first, slice_offset_frames=0,
                               clip_frame_count=90, fps=30.)
    base = tl[0]

    def with_ranges(ranges, fit=True):
        s = copy.deepcopy(base)
        s["fit"] = fit
        s["fit_ranges"] = ranges
        return s

    # Bridge-level rejection (normalize_timeline on hand-built plans).
    raw = copy.deepcopy(first)
    raw["scenes"][0]["segments"][0]["end_frame"] = True  # bool, not int
    with pytest.raises(pf.FramingError):
        pf.normalize_timeline(plan=raw, slice_offset_frames=0,
                              clip_frame_count=90, fps=30.)

    # Model-level rejection.
    with pytest.raises(Exception):
        FramingSceneTimeline(**with_ranges(
            [{"start_frame": 0, "end_frame": 14},
             {"start_frame": 10, "end_frame": 19}]))  # overlapping
    with pytest.raises(Exception):
        FramingSceneTimeline(**with_ranges(
            [{"start_frame": 0, "end_frame": 200}]))  # out of scene bounds
    with pytest.raises(Exception):
        FramingSceneTimeline(**with_ranges(
            [{"start_frame": 25, "end_frame": 35}]))  # covers keyframe@30
    with pytest.raises(Exception):
        FramingSceneTimeline(**with_ranges(
            [{"start_frame": 0, "end_frame": 14}], fit=False))  # ranges w/o fit


def test_fit_ranges_rebase_exactly_once():
    """The single integer rebase is applied once, never twice."""
    first, _ = _two_boundary_plans()
    offset = 100
    # Shift the whole plan by the slice offset, then normalize with that
    # offset: the bridge must subtract it exactly once.
    raw = copy.deepcopy(first)
    for seg in raw["scenes"][0]["segments"]:
        seg["start_frame"] += offset
        seg["end_frame"] += offset
    raw["scenes"][0]["start_frame"] += offset
    raw["scenes"][0]["end_frame"] += offset
    raw["scenes"][0]["keyframes"][0]["frame"] += offset
    tl = pf.normalize_timeline(plan=raw, slice_offset_frames=offset,
                               clip_frame_count=90, fps=30.)
    assert tl[0]["fit_ranges"] == [{"start_frame": 0, "end_frame": 14}]
    assert tl[0]["start_frame"] == 0 and tl[0]["end_frame"] == 89


def test_mutation_dropped_fit_ranges_collapses_boundaries(monkeypatch):
    """Fail-then-restore: the gate catches removal of interval serialization.

    If normalize_timeline folds fit back to the old scene-wide any() flag
    (dropping fit_ranges), the two boundary plans normalize and digest
    identically; restoring the production code keeps them distinct.
    """
    first, second = _two_boundary_plans()
    real = pf.normalize_timeline

    def folded(*args, **kwargs):
        out = real(*args, **kwargs)
        for scene in out:
            scene.pop("fit_ranges", None)
        return out

    # FAIL: the old fold collapses the boundary distinction.
    monkeypatch.setattr(pf, "normalize_timeline", folded)
    na = pf.normalize_timeline(plan=first, slice_offset_frames=0,
                               clip_frame_count=90, fps=30.)
    nb = pf.normalize_timeline(plan=second, slice_offset_frames=0,
                               clip_frame_count=90, fps=30.)
    assert na == nb, "mutation must reproduce the old collapse"
    assert pf.timeline_digest(na) == pf.timeline_digest(nb)

    # RESTORE: production intervals keep the boundaries distinct.
    monkeypatch.setattr(pf, "normalize_timeline", real)
    na = pf.normalize_timeline(plan=first, slice_offset_frames=0,
                               clip_frame_count=90, fps=30.)
    nb = pf.normalize_timeline(plan=second, slice_offset_frames=0,
                               clip_frame_count=90, fps=30.)
    assert na != nb
    assert pf.timeline_digest(na) != pf.timeline_digest(nb)


def test_edited_timing_cannot_reuse_stale_base(tmp_path):
    """Lineage identity carries the intervals: edited timing is not reusable.

    Real emission for both boundary plans; a project saved with fit [0,14]
    round-trips through save/load/version/undo with the intervals intact,
    and its lineage identity differs from the [0,19] plan's -- the stale
    base cannot be silently reused after a timing edit.
    """
    first, second = _two_boundary_plans()

    def emit(plan_dict, name):
        job = tmp_path / name
        job.mkdir()
        ref = "clip.turn_framing_plan.json"
        (job / ref).write_text(json.dumps(plan_dict), encoding="utf-8")
        return pf.build_project_framing(
            job_dir=str(job), clip_final="clip.mp4", render_w=270,
            render_h=480, render_fps=30.0, clip_frame_count=90)

    fa, fb = emit(first, "job_a"), emit(second, "job_b")
    assert fa is not None and fb is not None
    ia = pf.framing_baseline_identity(fa)
    ib = pf.framing_baseline_identity(fb)
    assert ia != ib, "edited fit timing must change the lineage identity"
    assert ia["timeline_digest"] != ib["timeline_digest"]

    def project_dict(framing, version):
        return {
            "schema": "nugget.clip-project/1",
            "version": version,
            "job_id": "job-1",
            "clip_index": 0,
            "source": {"file": "clip.mp4", "width": 640, "height": 480,
                       "duration": 3.0},
            "segments": [{"id": "seg0", "start": 0.0, "end": 3.0,
                          "crop": {"x": 0.0, "y": 0.0, "w": 202.0,
                                   "h": 360.0}}],
            "render": {"width": 270, "height": 480, "fps": 30.0},
            "framing": framing,
        }

    # Save v1 with fit [0,14]; save/load round-trip keeps the intervals.
    v1 = validate_project(project_dict(fa, 1))
    assert v1.framing.scenes[0].fit_ranges[0].end_frame == 14
    saved_v1 = v1.model_dump_json()
    assert validate_project(
        json.loads(saved_v1)).framing.scenes[0].fit_ranges[0].end_frame == 14
    # Edit to the [0,19] plan -> v2; the intervals follow the edit.
    v2 = validate_project(project_dict(fb, 2))
    assert v2.version == 2
    assert v2.framing.scenes[0].fit_ranges[0].end_frame == 19
    # Undo: reloading the saved v1 restores [0,14] -- no stale pixels.
    undone = validate_project(json.loads(saved_v1))
    assert undone.version == 1
    assert undone.framing.scenes[0].fit_ranges[0].end_frame == 14
