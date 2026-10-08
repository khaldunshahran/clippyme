"""Task 13 (domain side): stable framing sidecar bridge tests.

Covers resolve_sidecar (containment, symlink escape where supported,
missing, temp names), load_validated_plan (malformed nested, booleans,
NaN/inf), normalize_timeline (nonzero slice offset single rebase,
24/29.97/30/60 fps, square/9:16/16:9, no rounding drift, non-face scenes
get no keyframes, incomplete plans), build_project_framing (missing
sidecar -> None, aspect mismatch -> None), verify_framing_at_render
(same-size/same-mtime content replacement detected), and
framing_baseline_identity (excludes size/mtime).

Mutation gates: double-rebase, accepting same-size replacement,
manufacturing non-face keyframes, trusting an incomplete plan, dropping
the field during save/load round-trip.

Fixtures are production-valid Task 12 plans: every raw plan dict is run
through the real plan_from_dict/plan_to_dict before use.
"""

import copy
import hashlib
import json
import math
import os

import pytest

from clippyme.domain import project_framing as pf
from clippyme.domain.clip_project import (
    FramingKeyframe,
    FramingPlanIdentity,
    FramingRender,
    FramingSceneTimeline,
    FramingSource,
    ProjectFraming,
    project_json_schema,
    validate_project,
)
from clippyme.pipeline.framing_plan import plan_from_dict, plan_to_dict


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _raw_plan(**over):
    """A production-valid Task 12 plan dict (two scenes)."""
    plan = {
        "schema_version": 3,
        "producer": "clippyme-turn-framing-plan",
        "source_fingerprint": "fp-" + "a" * 16,
        "frame_w": 1080,
        "frame_h": 1920,
        "fps": 30.0,
        "frame_count": 300,
        "scene_intervals": [[0, 149], [150, 299]],
        "config_digest": "cfg-digest",
        "tracker_digest": "",
        "output_aspect": 9 / 16,
        "analysis_complete": True,
        "created_at": "2026-10-09T00:00:00+0000",
        "scenes": [
            {
                "scene_id": 0,
                "strategy": "TRACK",
                "start_frame": 0,
                "end_frame": 149,
                "applies": True,
                "tracks": [
                    {"id": "t-0", "first_frame": 0, "last_frame": 149,
                     "n_obs": 120},
                ],
                "segments": [
                    {
                        "scene_id": 0,
                        "start_frame": 0,
                        "end_frame": 149,
                        "subject_ids": ["t-0"],
                        "kind": "single",
                        "confidence": 0.9,
                        "evidence": {},
                        "reason": "turn-accepted",
                        "crop": [200, 400, 500, 900],
                    },
                ],
                "keyframes": [
                    {"frame": 0, "x1": 200, "y1": 400, "x2": 500, "y2": 900,
                     "kind": "lock"},
                    {"frame": 149, "x1": 210, "y1": 410, "x2": 510,
                     "y2": 910, "kind": "ease"},
                ],
                "fallback": None,
                "fallback_crop": None,
            },
            {
                "scene_id": 1,
                "strategy": "SPLIT",
                "start_frame": 150,
                "end_frame": 299,
                "applies": False,
                "tracks": [],
                "segments": [],
                "keyframes": [],
                "fallback": None,
                "fallback_crop": None,
            },
        ],
    }
    plan.update(over)
    return plan


def _valid_plan(**over):
    """Canonical plan_to_dict form of a production-valid plan."""
    return plan_to_dict(plan_from_dict(_raw_plan(**over)))


@pytest.fixture
def job_dir(tmp_path):
    d = tmp_path / "job"
    d.mkdir()
    return str(d)


def _write_sidecar(job_dir, ref, plan_dict):
    p = os.path.join(job_dir, ref)
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(plan_dict, fh)
    return p


def _mini_project(**over):
    d = {
        "schema": "nugget.clip-project/1",
        "job_id": "job-1",
        "clip_index": 0,
        "source": {"file": "source_0.mp4", "width": 1920, "height": 1080},
        "segments": [
            {"id": "seg0", "start": 0.0, "end": 10.0,
             "crop": {"x": 0.0, "y": 0.0, "w": 608.0, "h": 1080.0}},
        ],
    }
    d.update(over)
    return d


# ---------------------------------------------------------------------------
# stable_sidecar_ref
# ---------------------------------------------------------------------------

class TestStableSidecarRef:
    def test_relative(self):
        assert pf.stable_sidecar_ref("clip_final_0.mp4") == \
            "clip_final_0.turn_framing_plan.json"

    def test_nested_relative(self):
        assert pf.stable_sidecar_ref("clips/clip_final_0.mp4") == \
            "clip_final_0.turn_framing_plan.json"

    def test_absolute_windows(self):
        assert pf.stable_sidecar_ref(
            "C:\\jobs\\j1\\clip_final_0.mp4") == \
            "clip_final_0.turn_framing_plan.json"

    def test_no_extension(self):
        assert pf.stable_sidecar_ref("clipfinal") == \
            "clipfinal.turn_framing_plan.json"

    def test_empty_raises(self):
        with pytest.raises(pf.FramingError) as ei:
            pf.stable_sidecar_ref("")
        assert str(ei.value).startswith(pf.REASON_MISSING)


# ---------------------------------------------------------------------------
# resolve_sidecar
# ---------------------------------------------------------------------------

class TestResolveSidecar:
    def test_ok(self, job_dir):
        ref = "clip_final_0.turn_framing_plan.json"
        p = _write_sidecar(job_dir, ref, _raw_plan())
        r = pf.resolve_sidecar(job_dir, ref)
        assert r.ref == ref
        assert r.abs_path == os.path.realpath(p)
        assert r.sha256 == hashlib.sha256(
            open(p, "rb").read()).hexdigest()
        assert r.size == os.path.getsize(p)
        assert r.mtime_ns == os.stat(p).st_mtime_ns

    def test_missing(self, job_dir):
        with pytest.raises(pf.FramingError) as ei:
            pf.resolve_sidecar(job_dir, "nope.turn_framing_plan.json")
        assert str(ei.value).startswith(pf.REASON_MISSING)

    def test_traversal(self, job_dir):
        with pytest.raises(pf.FramingError) as ei:
            pf.resolve_sidecar(job_dir, "../evil.turn_framing_plan.json")
        assert str(ei.value).startswith(pf.REASON_ESCAPE)

    def test_absolute_ref_outside(self, tmp_path):
        outside = tmp_path / "outside.json"
        outside.write_text("{}")
        job = tmp_path / "job"
        job.mkdir()
        with pytest.raises(pf.FramingError) as ei:
            pf.resolve_sidecar(str(job), os.path.abspath(str(outside)))
        assert str(ei.value).startswith(pf.REASON_ESCAPE)

    def test_dotdot_inside_ok(self, job_dir):
        # ".." that stays inside the job resolves fine (no false escape).
        sub = os.path.join(job_dir, "sub")
        os.mkdir(sub)
        ref = "clip_final_0.turn_framing_plan.json"
        _write_sidecar(job_dir, ref, _raw_plan())
        r = pf.resolve_sidecar(job_dir, "sub/../" + ref)
        assert r.sha256

    @pytest.mark.parametrize("marker", [".render.tmp", ".reframe.tmp"])
    def test_temp_basename_rejected(self, job_dir, marker):
        ref = f"clip{marker}.turn_framing_plan.json"
        _write_sidecar(job_dir, ref, _raw_plan())
        with pytest.raises(pf.FramingError) as ei:
            pf.resolve_sidecar(job_dir, ref)
        assert str(ei.value).startswith(pf.REASON_TEMP_NAME)

    def test_symlink_escape(self, job_dir, tmp_path):
        outside = tmp_path / "outside.json"
        outside.write_text("{}")
        link = os.path.join(job_dir, "link.turn_framing_plan.json")
        try:
            os.symlink(str(outside), link)
        except OSError:
            pytest.skip("symlinks not permitted here")
        with pytest.raises(pf.FramingError) as ei:
            pf.resolve_sidecar(job_dir, "link.turn_framing_plan.json")
        assert str(ei.value).startswith(pf.REASON_ESCAPE)

    def test_symlink_inside_ok(self, job_dir):
        target = _write_sidecar(
            job_dir, "real.turn_framing_plan.json", _raw_plan())
        link = os.path.join(job_dir, "link.turn_framing_plan.json")
        try:
            os.symlink(target, link)
        except OSError:
            pytest.skip("symlinks not permitted here")
        r = pf.resolve_sidecar(job_dir, "link.turn_framing_plan.json")
        assert r.abs_path == os.path.realpath(target)


# ---------------------------------------------------------------------------
# load_validated_plan
# ---------------------------------------------------------------------------

class TestLoadValidatedPlan:
    def test_round_trip(self, job_dir):
        ref = "clip_final_0.turn_framing_plan.json"
        _write_sidecar(job_dir, ref, _raw_plan())
        plan = pf.load_validated_plan(pf.resolve_sidecar(job_dir, ref))
        assert plan["schema_version"] == 3
        assert plan["producer"] == "clippyme-turn-framing-plan"
        assert len(plan["scenes"]) == 2

    def test_bad_json(self, job_dir):
        ref = "clip_final_0.turn_framing_plan.json"
        with open(os.path.join(job_dir, ref), "w") as fh:
            fh.write("{not json")
        with pytest.raises(pf.FramingError) as ei:
            pf.load_validated_plan(pf.resolve_sidecar(job_dir, ref))
        assert str(ei.value).startswith(pf.REASON_MALFORMED)

    def test_bool_rejected(self, job_dir):
        raw = _raw_plan()
        raw["frame_w"] = True
        ref = "clip_final_0.turn_framing_plan.json"
        _write_sidecar(job_dir, ref, raw)
        with pytest.raises(pf.FramingError) as ei:
            pf.load_validated_plan(pf.resolve_sidecar(job_dir, ref))
        assert str(ei.value).startswith(pf.REASON_MALFORMED)

    def test_nan_rejected(self, job_dir):
        ref = "clip_final_0.turn_framing_plan.json"
        with open(os.path.join(job_dir, ref), "w") as fh:
            raw = _raw_plan()
            raw["fps"] = float("nan")
            fh.write(json.dumps(raw))  # emits NaN token; json accepts on read
        with pytest.raises(pf.FramingError) as ei:
            pf.load_validated_plan(pf.resolve_sidecar(job_dir, ref))
        assert str(ei.value).startswith(pf.REASON_MALFORMED)

    def test_inf_rejected(self, job_dir):
        ref = "clip_final_0.turn_framing_plan.json"
        with open(os.path.join(job_dir, ref), "w") as fh:
            raw = _raw_plan()
            raw["output_aspect"] = float("inf")
            fh.write(json.dumps(raw))
        with pytest.raises(pf.FramingError) as ei:
            pf.load_validated_plan(pf.resolve_sidecar(job_dir, ref))
        assert str(ei.value).startswith(pf.REASON_MALFORMED)

    def test_nested_malformed_keyframe(self, job_dir):
        raw = _raw_plan()
        raw["scenes"][0]["keyframes"][0]["x2"] = 50  # x2 < x1
        ref = "clip_final_0.turn_framing_plan.json"
        _write_sidecar(job_dir, ref, raw)
        with pytest.raises(pf.FramingError) as ei:
            pf.load_validated_plan(pf.resolve_sidecar(job_dir, ref))
        assert str(ei.value).startswith(pf.REASON_MALFORMED)

    def test_non_utf8_rejected(self, job_dir):
        ref = "clip_final_0.turn_framing_plan.json"
        with open(os.path.join(job_dir, ref), "wb") as fh:
            fh.write(b'{"schema_version": 3, \xff\xfe not utf8}')
        with pytest.raises(pf.FramingError) as ei:
            pf.load_validated_plan(pf.resolve_sidecar(job_dir, ref))
        assert str(ei.value).startswith(pf.REASON_MALFORMED)

    def test_smuggled_key_rejected(self, job_dir):
        raw = _raw_plan()
        raw["scenes"][0]["keyframes"][0]["mouth_landmarks"] = [1, 2, 3]
        ref = "clip_final_0.turn_framing_plan.json"
        _write_sidecar(job_dir, ref, raw)
        with pytest.raises(pf.FramingError) as ei:
            pf.load_validated_plan(pf.resolve_sidecar(job_dir, ref))
        assert str(ei.value).startswith(pf.REASON_MALFORMED)


# ---------------------------------------------------------------------------
# normalize_timeline
# ---------------------------------------------------------------------------

class TestNormalizeTimeline:
    def test_zero_offset_exact(self):
        scenes = pf.normalize_timeline(
            plan=_valid_plan(), slice_offset_frames=0,
            clip_frame_count=300, fps=30.0)
        assert scenes[0]["scene_id"] == 0
        assert scenes[0]["strategy"] == "TRACK"
        assert (scenes[0]["start_frame"], scenes[0]["end_frame"]) == (0, 149)
        assert scenes[0]["applies"] is True
        assert scenes[0]["subject_ids"] == ["t-0"]
        assert [k["frame"] for k in scenes[0]["keyframes"]] == [0, 149]
        assert scenes[0]["keyframes"][0]["x1"] == 200
        assert scenes[0]["keyframes"][0]["kind"] == "lock"
        assert scenes[0]["fallback_reason"] is None
        # Non-face scene: no keyframes manufactured.
        assert scenes[1]["strategy"] == "SPLIT"
        assert scenes[1]["applies"] is False
        assert scenes[1]["keyframes"] == []
        assert scenes[1]["subject_ids"] == []

    def test_nonzero_offset_single_rebase(self):
        # Plan authored in absolute frames of a slice starting at 45.
        raw = _raw_plan()
        for s in raw["scenes"]:
            s["start_frame"] += 45
            s["end_frame"] += 45
            for t in s["tracks"]:
                t["first_frame"] += 45
                t["last_frame"] += 45
            for g in s["segments"]:
                g["start_frame"] += 45
                g["end_frame"] += 45
            for k in s["keyframes"]:
                k["frame"] += 45
        raw["scene_intervals"] = [[45, 194], [195, 344]]
        raw["frame_count"] = 345
        plan = plan_to_dict(plan_from_dict(raw))
        scenes = pf.normalize_timeline(
            plan=plan, slice_offset_frames=45,
            clip_frame_count=300, fps=30.0)
        # Plan frame F -> clip frame F-45, exactly once.
        assert (scenes[0]["start_frame"], scenes[0]["end_frame"]) == (0, 149)
        assert [k["frame"] for k in scenes[0]["keyframes"]] == [0, 149]
        assert (scenes[1]["start_frame"], scenes[1]["end_frame"]) == (150, 299)

    @pytest.mark.parametrize("fps", [24.0, 30000 / 1001, 30.0, 60.0])
    @pytest.mark.parametrize("w,h", [(1080, 1080), (608, 1080), (1920, 1080)])
    def test_fps_and_geometry_no_drift(self, fps, w, h):
        # Frame math is integer-only: fps/geometry never move a boundary.
        raw = _raw_plan()
        raw["fps"] = fps
        plan = plan_to_dict(plan_from_dict(raw))
        scenes = pf.normalize_timeline(
            plan=plan, slice_offset_frames=0,
            clip_frame_count=300, fps=fps)
        assert (scenes[0]["start_frame"], scenes[0]["end_frame"]) == (0, 149)
        assert [k["frame"] for k in scenes[0]["keyframes"]] == [0, 149]
        assert all(isinstance(k["frame"], int)
                   for k in scenes[0]["keyframes"])

    def test_bool_inputs_rejected(self):
        plan = _valid_plan()
        with pytest.raises(pf.FramingError):
            pf.normalize_timeline(plan=plan, slice_offset_frames=True,
                                  clip_frame_count=300, fps=30.0)
        with pytest.raises(pf.FramingError):
            pf.normalize_timeline(plan=plan, slice_offset_frames=0,
                                  clip_frame_count=False, fps=30.0)
        with pytest.raises(pf.FramingError):
            pf.normalize_timeline(plan=plan, slice_offset_frames=0,
                                  clip_frame_count=300, fps=True)

    def test_scene_out_of_clip_rejected(self):
        plan = _valid_plan()
        with pytest.raises(pf.FramingError) as ei:
            pf.normalize_timeline(plan=plan, slice_offset_frames=0,
                                  clip_frame_count=100, fps=30.0)
        assert str(ei.value).startswith(pf.REASON_MALFORMED)

    def test_double_rebase_fails(self):
        # Mutation gate: a caller that passes a nonzero offset for a plan
        # whose frames are already clip-relative must fail validation
        # instead of silently double-rebasing into negative frames.
        plan = _valid_plan()
        with pytest.raises(pf.FramingError):
            pf.normalize_timeline(plan=plan, slice_offset_frames=45,
                                  clip_frame_count=300, fps=30.0)

    def test_negative_offset_rejected(self):
        with pytest.raises(pf.FramingError):
            pf.normalize_timeline(plan=_valid_plan(), slice_offset_frames=-1,
                                  clip_frame_count=300, fps=30.0)

    @pytest.mark.parametrize("strategy",
                             ["SPLIT", "SCREENCAST", "GENERAL", "DISABLED",
                              "SOMETHING_NEW"])
    def test_non_face_strategies_no_keyframes(self, strategy):
        raw = _raw_plan()
        raw["scenes"][1]["strategy"] = strategy
        plan = plan_to_dict(plan_from_dict(raw))
        scenes = pf.normalize_timeline(
            plan=plan, slice_offset_frames=0,
            clip_frame_count=300, fps=30.0)
        assert scenes[1]["strategy"] == strategy  # verbatim
        assert scenes[1]["keyframes"] == []
        assert scenes[1]["applies"] is False

    def test_non_face_with_keyframes_rejected(self):
        # Manufacturing face content on a non-face scene is untrusted.
        raw = _raw_plan()
        raw["scenes"][1]["keyframes"] = [
            {"frame": 150, "x1": 10, "y1": 10, "x2": 100, "y2": 100,
             "kind": "lock"},
        ]
        # Bypass the production parser (it would allow this shape); feed
        # the raw dict straight to normalize.
        with pytest.raises(pf.FramingError) as ei:
            pf.normalize_timeline(
                plan=raw, slice_offset_frames=0,
                clip_frame_count=300, fps=30.0)
        assert str(ei.value).startswith(pf.REASON_MALFORMED)

    def test_fallback_preserved(self):
        raw = _raw_plan()
        raw["scenes"][0]["segments"] = []
        raw["scenes"][0]["keyframes"] = []
        raw["scenes"][0]["fallback"] = {
            "reason": "analysis-frame-budget-exhausted",
            "detail": "scene 0",
        }
        plan = plan_to_dict(plan_from_dict(raw))
        scenes = pf.normalize_timeline(
            plan=plan, slice_offset_frames=0,
            clip_frame_count=300, fps=30.0)
        assert scenes[0]["applies"] is True
        assert scenes[0]["keyframes"] == []
        assert scenes[0]["fallback_reason"] == \
            "analysis-frame-budget-exhausted"
        assert scenes[0]["fallback_detail"] == "scene 0"

    def test_keyframe_outside_scene_rejected(self):
        plan = _valid_plan()
        # Corrupt post-parse: move a keyframe outside its scene.
        plan["scenes"][0]["keyframes"][1]["frame"] = 200
        with pytest.raises(pf.FramingError):
            pf.normalize_timeline(plan=plan, slice_offset_frames=0,
                                  clip_frame_count=300, fps=30.0)

    def test_duplicate_keyframes_rejected(self):
        plan = _valid_plan()
        plan["scenes"][0]["keyframes"][1]["frame"] = \
            plan["scenes"][0]["keyframes"][0]["frame"]
        with pytest.raises(pf.FramingError):
            pf.normalize_timeline(plan=plan, slice_offset_frames=0,
                                  clip_frame_count=300, fps=30.0)

    def test_timeline_digest(self):
        a = pf.normalize_timeline(plan=_valid_plan(), slice_offset_frames=0,
                                  clip_frame_count=300, fps=30.0)
        b = pf.normalize_timeline(plan=_valid_plan(), slice_offset_frames=0,
                                  clip_frame_count=300, fps=30.0)
        assert pf.timeline_digest(a) == pf.timeline_digest(b)
        assert len(pf.timeline_digest(a)) == 64
        c = copy.deepcopy(a)
        c[0]["keyframes"][0]["x1"] += 1
        assert pf.timeline_digest(c) != pf.timeline_digest(a)


# ---------------------------------------------------------------------------
# build_project_framing
# ---------------------------------------------------------------------------

class TestBuildProjectFraming:
    def _build(self, job_dir, **kw):
        ref = "clip_final_0.turn_framing_plan.json"
        _write_sidecar(job_dir, ref, _raw_plan())
        params = dict(job_dir=job_dir, clip_final="clip_final_0.mp4",
                      render_w=608, render_h=1080, render_fps=30.0,
                      clip_frame_count=300)
        params.update(kw)
        return pf.build_project_framing(**params)

    def test_full_field(self, job_dir):
        framing = self._build(job_dir)
        assert framing is not None
        assert framing["version"] == 1
        assert framing["eligible"] is True
        plan_id = framing["plan"]
        assert plan_id["plan_ref"] == "clip_final_0.turn_framing_plan.json"
        assert len(plan_id["sha256"]) == 64
        assert plan_id["schema_version"] == 3
        assert plan_id["producer"] == "clippyme-turn-framing-plan"
        assert plan_id["analysis_complete"] is True
        assert len(plan_id["timeline_digest"]) == 64
        assert framing["source"]["slice_offset_frames"] == 0
        assert framing["source"]["frame_count"] == 300
        assert framing["render"]["width"] == 608
        assert framing["render"]["aspect"] == pytest.approx(608 / 1080)
        assert len(framing["scenes"]) == 2
        # Validates against the production model.
        ProjectFraming.model_validate(framing)
        # ...and inside a full project.
        validate_project(_mini_project(framing=framing))

    def test_missing_sidecar_returns_none(self, job_dir):
        assert pf.build_project_framing(
            job_dir=job_dir, clip_final="clip_final_0.mp4",
            render_w=608, render_h=1080, render_fps=30.0,
            clip_frame_count=300) is None

    def test_malformed_sidecar_returns_none(self, job_dir):
        ref = "clip_final_0.turn_framing_plan.json"
        with open(os.path.join(job_dir, ref), "w") as fh:
            fh.write("garbage")
        assert pf.build_project_framing(
            job_dir=job_dir, clip_final="clip_final_0.mp4",
            render_w=608, render_h=1080, render_fps=30.0,
            clip_frame_count=300) is None

    def test_temp_sidecar_not_trusted(self, job_dir):
        # Only a temp-named sidecar exists: no stable ref -> None, and the
        # temp file itself must never be picked up.
        _write_sidecar(job_dir, "clip_final_0.render.tmp.turn_framing_plan.json",
                       _raw_plan())
        assert pf.build_project_framing(
            job_dir=job_dir, clip_final="clip_final_0.mp4",
            render_w=608, render_h=1080, render_fps=30.0,
            clip_frame_count=300) is None

    def test_aspect_mismatch_returns_none(self, job_dir):
        # Plan built for 9:16; a square render must not get the field.
        assert self._build(job_dir, render_w=1080, render_h=1080) is None

    def test_aspect_within_tolerance_ok(self, job_dir):
        # 608x1080 vs 9:16 plan: 0.08% off -> accepted.
        assert self._build(job_dir) is not None

    def test_incomplete_plan_audit(self, job_dir):
        raw = _raw_plan(analysis_complete=False)
        ref = "clip_final_0.turn_framing_plan.json"
        _write_sidecar(job_dir, ref, raw)
        framing = pf.build_project_framing(
            job_dir=job_dir, clip_final="clip_final_0.mp4",
            render_w=608, render_h=1080, render_fps=30.0,
            clip_frame_count=300)
        assert framing is not None
        assert framing["eligible"] is False
        assert framing["ineligibility_reason"] == pf.REASON_INCOMPLETE
        assert framing["scenes"] == []
        assert framing["plan"]["analysis_complete"] is False
        ProjectFraming.model_validate(framing)

    def test_nonzero_slice_offset(self, job_dir):
        raw = _raw_plan()
        for s in raw["scenes"]:
            s["start_frame"] += 45
            s["end_frame"] += 45
            for t in s["tracks"]:
                t["first_frame"] += 45
                t["last_frame"] += 45
            for g in s["segments"]:
                g["start_frame"] += 45
                g["end_frame"] += 45
            for k in s["keyframes"]:
                k["frame"] += 45
        raw["scene_intervals"] = [[45, 194], [195, 344]]
        raw["frame_count"] = 345
        ref = "clip_final_0.turn_framing_plan.json"
        _write_sidecar(job_dir, ref, raw)
        framing = pf.build_project_framing(
            job_dir=job_dir, clip_final="clip_final_0.mp4",
            render_w=608, render_h=1080, render_fps=30.0,
            slice_offset_frames=45, clip_frame_count=300)
        assert framing is not None
        assert framing["source"]["slice_offset_frames"] == 45
        assert (framing["scenes"][0]["start_frame"],
                framing["scenes"][0]["end_frame"]) == (0, 149)
        ProjectFraming.model_validate(framing)


# ---------------------------------------------------------------------------
# verify_framing_at_render
# ---------------------------------------------------------------------------

class TestVerifyFramingAtRender:
    def _framing(self, job_dir):
        ref = "clip_final_0.turn_framing_plan.json"
        _write_sidecar(job_dir, ref, _raw_plan())
        framing = pf.build_project_framing(
            job_dir=job_dir, clip_final="clip_final_0.mp4",
            render_w=608, render_h=1080, render_fps=30.0,
            clip_frame_count=300)
        assert framing is not None
        return framing

    def test_ok(self, job_dir):
        framing = self._framing(job_dir)
        pf.verify_framing_at_render(job_dir=job_dir, framing=framing)  # no raise

    def test_missing_sidecar(self, job_dir):
        framing = self._framing(job_dir)
        os.remove(os.path.join(job_dir, "clip_final_0.turn_framing_plan.json"))
        with pytest.raises(pf.FramingError) as ei:
            pf.verify_framing_at_render(job_dir=job_dir, framing=framing)
        assert str(ei.value).startswith(pf.REASON_MISSING)

    def test_same_size_replacement_detected(self, job_dir):
        # Mutation gate: same size AND same mtime, different bytes ->
        # the content hash must still catch it.
        framing = self._framing(job_dir)
        p = os.path.join(job_dir, "clip_final_0.turn_framing_plan.json")
        st = os.stat(p)
        raw = open(p, "rb").read()
        tampered = bytearray(raw)
        # Flip one ASCII byte inside the config digest (stays valid
        # UTF-8/JSON, length unchanged).
        i = tampered.find(b"cfg-digest")
        assert i != -1
        tampered[i] ^= 0x01
        with open(p, "wb") as fh:
            fh.write(tampered)
        os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns))  # restore mtime
        assert os.path.getsize(p) == framing["plan"]["size"]
        with pytest.raises(pf.FramingError) as ei:
            pf.verify_framing_at_render(job_dir=job_dir, framing=framing)
        assert str(ei.value).startswith(pf.REASON_MISMATCH)

    def test_tampered_source_fingerprint(self, job_dir):
        framing = self._framing(job_dir)
        p = os.path.join(job_dir, "clip_final_0.turn_framing_plan.json")
        raw = _raw_plan()
        raw["source_fingerprint"] = "fp-" + "b" * 16
        with open(p, "w", encoding="utf-8") as fh:
            json.dump(raw, fh)
        with pytest.raises(pf.FramingError) as ei:
            pf.verify_framing_at_render(job_dir=job_dir, framing=framing)
        assert str(ei.value).startswith(pf.REASON_MISMATCH)

    def test_no_plan_ref(self, job_dir):
        with pytest.raises(pf.FramingError) as ei:
            pf.verify_framing_at_render(job_dir=job_dir, framing={})
        assert str(ei.value).startswith(pf.REASON_MISSING)


# ---------------------------------------------------------------------------
# framing_baseline_identity
# ---------------------------------------------------------------------------

class TestFramingBaselineIdentity:
    def test_subset(self, job_dir):
        ref = "clip_final_0.turn_framing_plan.json"
        _write_sidecar(job_dir, ref, _raw_plan())
        framing = pf.build_project_framing(
            job_dir=job_dir, clip_final="clip_final_0.mp4",
            render_w=608, render_h=1080, render_fps=30.0,
            clip_frame_count=300)
        ident = pf.framing_baseline_identity(framing)
        assert set(ident) == {
            "plan_ref", "sha256", "source_fingerprint", "config_digest",
            "tracker_digest", "output_aspect", "analysis_complete",
            "timeline_digest",
        }
        assert "size" not in ident and "mtime_ns" not in ident

    def test_malformed_raises(self):
        with pytest.raises(pf.FramingError) as ei:
            pf.framing_baseline_identity({"plan": {"plan_ref": "x"}})
        assert str(ei.value).startswith(pf.REASON_MALFORMED)


# ---------------------------------------------------------------------------
# Reason constants
# ---------------------------------------------------------------------------

def test_reason_constants():
    assert pf.REASON_TEMP_NAME == "framing-sidecar-temp-name"
    assert pf.REASON_MISSING == "framing-sidecar-missing"
    assert pf.REASON_ESCAPE == "framing-sidecar-escape"
    assert pf.REASON_MALFORMED == "framing-plan-malformed"
    assert pf.REASON_MISMATCH == "framing-sidecar-mismatch"
    assert pf.REASON_INCOMPLETE == "analysis-incomplete"
    assert pf.REASON_TRAJECTORY_EDIT == "framing-trajectory-edit"
    assert pf.TEMP_BASENAME_MARKERS == (".render.tmp", ".reframe.tmp")


# ---------------------------------------------------------------------------
# Pydantic model strictness
# ---------------------------------------------------------------------------

class TestModelStrictness:
    def _identity(self, **over):
        d = {
            "plan_ref": "clip_final_0.turn_framing_plan.json",
            "sha256": "a" * 64,
            "size": 123,
            "mtime_ns": 456,
            "schema_version": 3,
            "producer": "clippyme-turn-framing-plan",
            "source_fingerprint": "fp",
            "config_digest": "cfg",
            "output_aspect": 0.5625,
            "analysis_complete": True,
            "timeline_digest": "b" * 64,
        }
        d.update(over)
        return d

    def _scene(self, **over):
        d = {
            "scene_id": 0,
            "strategy": "TRACK",
            "start_frame": 0,
            "end_frame": 149,
            "applies": True,
            "subject_ids": ["t-0"],
            "keyframes": [
                {"frame": 0, "x1": 1, "y1": 1, "x2": 10, "y2": 10,
                 "kind": "lock"},
            ],
        }
        d.update(over)
        return d

    def _framing(self, **over):
        d = {
            "plan": self._identity(),
            "source": {"fingerprint": "fp", "width": 1080, "height": 1920,
                       "fps": 30.0, "frame_count": 300,
                       "slice_offset_frames": 0},
            "render": {"width": 608, "height": 1080, "fps": 30.0,
                       "aspect": 608 / 1080},
            "scenes": [self._scene()],
        }
        d.update(over)
        return d

    def test_keyframe_bool_rejected(self):
        with pytest.raises(Exception):
            FramingKeyframe(frame=True, x1=0, y1=0, x2=1, y2=1, kind="lock")

    def test_keyframe_box_order(self):
        with pytest.raises(Exception):
            FramingKeyframe(frame=0, x1=5, y1=0, x2=5, y2=1, kind="lock")

    def test_keyframe_bad_kind(self):
        with pytest.raises(Exception):
            FramingKeyframe(frame=0, x1=0, y1=0, x2=1, y2=1, kind="zoom")

    def test_scene_subject_id_must_be_anonymous(self):
        with pytest.raises(Exception):
            FramingSceneTimeline(**self._scene(subject_ids=["sehreen"]))

    def test_scene_keyframes_ordered_unique(self):
        kf = {"frame": 5, "x1": 0, "y1": 0, "x2": 1, "y2": 1, "kind": "lock"}
        with pytest.raises(Exception):
            FramingSceneTimeline(
                **self._scene(keyframes=[kf, dict(kf)]))

    def test_scene_start_after_end(self):
        with pytest.raises(Exception):
            FramingSceneTimeline(**self._scene(start_frame=10, end_frame=5))

    def test_identity_sha_pattern(self):
        with pytest.raises(Exception):
            FramingPlanIdentity(**self._identity(sha256="xyz"))

    def test_identity_inf_rejected(self):
        with pytest.raises(Exception):
            FramingPlanIdentity(**self._identity(output_aspect=float("inf")))

    def test_identity_nan_rejected(self):
        with pytest.raises(Exception):
            FramingPlanIdentity(**self._identity(output_aspect=float("nan")))

    def test_source_bool_rejected(self):
        with pytest.raises(Exception):
            FramingSource(fingerprint="fp", width=True, height=1080,
                          fps=30.0, frame_count=300, slice_offset_frames=0)

    def test_render_inf_rejected(self):
        with pytest.raises(Exception):
            FramingRender(width=608, height=1080, fps=float("inf"), aspect=1.0)

    def test_eligible_requires_scenes(self):
        with pytest.raises(Exception):
            ProjectFraming(**self._framing(scenes=[]))

    def test_eligible_applies_scene_needs_keyframes_or_fallback(self):
        with pytest.raises(Exception):
            ProjectFraming(**self._framing(
                scenes=[self._scene(keyframes=[])]))

    def test_eligible_applies_scene_with_fallback_ok(self):
        f = self._framing(scenes=[self._scene(
            keyframes=[], fallback_reason="analysis-frame-budget-exhausted")])
        ProjectFraming.model_validate(f)

    def test_ineligible_requires_reason(self):
        with pytest.raises(Exception):
            ProjectFraming(**self._framing(eligible=False, scenes=[]))

    def test_ineligible_audit_ok(self):
        f = self._framing(eligible=False,
                          ineligibility_reason="analysis-incomplete",
                          scenes=[])
        f["plan"]["analysis_complete"] = False
        ProjectFraming.model_validate(f)

    def test_version_literal(self):
        with pytest.raises(Exception):
            ProjectFraming(**self._framing(version=2))


# ---------------------------------------------------------------------------
# Project save/load round-trip preserves the field (mutation gate)
# ---------------------------------------------------------------------------

class TestProjectRoundTrip:
    def test_framing_preserved(self, job_dir):
        ref = "clip_final_0.turn_framing_plan.json"
        _write_sidecar(job_dir, ref, _raw_plan())
        framing = pf.build_project_framing(
            job_dir=job_dir, clip_final="clip_final_0.mp4",
            render_w=608, render_h=1080, render_fps=30.0,
            clip_frame_count=300)
        p = os.path.join(job_dir, "clip-project-0.json")
        with open(p, "w", encoding="utf-8") as fh:
            fh.write(validate_project(
                _mini_project(framing=framing)).model_dump_json(indent=2))
        with open(p, encoding="utf-8") as fh:
            loaded = validate_project(json.load(fh))
        assert loaded.framing is not None
        assert loaded.framing.plan.sha256 == framing["plan"]["sha256"]
        assert loaded.framing.plan.timeline_digest == \
            framing["plan"]["timeline_digest"]
        assert len(loaded.framing.scenes) == 2
        assert loaded.framing.scenes[0].keyframes[0].frame == 0

    def test_old_project_without_framing(self):
        proj = validate_project(_mini_project())
        assert proj.framing is None
        # Serializes without the field defaulting to something truthy.
        data = json.loads(proj.model_dump_json())
        assert data["framing"] is None


def test_schema_twin_has_framing():
    """Generated schema twin includes the additive framing field."""
    from pathlib import Path
    schema_path = (Path(__file__).parent.parent.parent /
                   "src/clippyme/domain/clip_project.schema.json")
    if not schema_path.exists():
        import clippyme.domain.clip_project as cp
        schema_path = Path(cp.__file__).parent / "clip_project.schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    assert "framing" in schema["properties"]
    assert "ProjectFraming" in schema.get("$defs", {})


def test_schema_twin_matches_model():
    """The twin on disk equals what project_json_schema() produces."""
    from pathlib import Path
    import clippyme.domain.clip_project as cp
    schema_path = Path(cp.__file__).parent / "clip_project.schema.json"
    on_disk = json.loads(schema_path.read_text(encoding="utf-8"))
    assert on_disk == project_json_schema()
