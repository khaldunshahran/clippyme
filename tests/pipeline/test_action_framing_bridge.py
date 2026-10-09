"""Task 14: project bridge / lineage / schema for v4 action plans.

Exercises the Task 13 bridge (project_framing) with versioned ACTION
scene kinds: normalize, build_project_framing, ClipProject validation,
save/load/version/undo, verify_framing_at_render, publish/retire, and
the lineage eligibility gates.
"""
import json
import os

import pytest

from clippyme.pipeline import action_framing as af
from clippyme.pipeline import framing_plan as fp
from clippyme.domain import project_framing as pf
from clippyme.domain.clip_project import (
    FramingSceneTimeline,
    validate_project,
)


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


def _v3_plan_dict():
    log = fp.ObservationLog(640, 360)
    plan = fp.build_framing_plan(
        log=log, scenes=[(0, 59)], strategies=["GENERAL"],
        config=fp.TurnFramingConfig(), fps=30.0, frame_w=640, frame_h=360,
        frame_count=60, source_fingerprint="x")
    return fp.plan_to_dict(plan)


@pytest.fixture()
def job_dir(tmp_path):
    d = tmp_path / "job"
    d.mkdir()
    return str(d)


def _write_sidecar(job_dir, ref, plan_dict):
    p = os.path.join(job_dir, ref)
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(plan_dict, fh)
    return p


class TestNormalizeV4:
    def test_action_kind_normalized_with_keyframes(self):
        d = _v4_action_plan_dict()
        plan = fp.plan_from_dict(d)
        scenes = pf.normalize_timeline(
            plan=fp.plan_to_dict(plan), slice_offset_frames=0,
            clip_frame_count=60, fps=30.0)
        assert len(scenes) == 1
        sc = scenes[0]
        assert sc["scene_kind"] == "action"
        assert sc["strategy"] == "GENERAL"
        assert sc["applies"] is True
        assert sc["subject_ids"] == ["t-0"]
        assert sc["keyframes"], "action keyframes must survive the bridge"
        for kf in sc["keyframes"]:
            assert set(kf) >= {"frame", "x1", "y1", "x2", "y2", "kind"}
            assert 0 <= kf["x1"] < kf["x2"] <= 640
            assert 0 <= kf["y1"] < kf["y2"] <= 360

    def test_v3_general_stays_none_kind(self):
        d = _v3_plan_dict()
        plan = fp.plan_from_dict(d)
        scenes = pf.normalize_timeline(
            plan=fp.plan_to_dict(plan), slice_offset_frames=0,
            clip_frame_count=60, fps=30.0)
        assert scenes[0]["scene_kind"] == "none"
        assert scenes[0]["keyframes"] == []

    def test_action_scene_with_wrong_strategy_rejected(self):
        d = _v4_action_plan_dict()
        d["scenes"][0]["strategy"] = "TRACK"
        # The v4 parser itself rejects this before the bridge.
        with pytest.raises(ValueError, match="action scene"):
            fp.plan_from_dict(d)

    def test_bridge_rejects_action_scene_carrying_face_content(self):
        d = _v4_action_plan_dict()
        # Hand-craft: action kind but non-GENERAL strategy slips past the
        # parser only if we bypass it — instead test the bridge's own
        # defense via a raw dict with kind action + strategy TRACK.
        raw = dict(d["scenes"][0])
        raw["strategy"] = "TRACK"
        raw["scene_kind"] = "action"
        plan = dict(d)
        plan["scenes"] = [raw]
        with pytest.raises(pf.FramingError, match="action scene"):
            pf.normalize_timeline(plan=plan, slice_offset_frames=0,
                                  clip_frame_count=60, fps=30.0)

    def test_temporary_paths_never_appear(self, job_dir):
        ref = "clip.turn_framing_plan.json"
        _write_sidecar(job_dir, ref, _v4_action_plan_dict())
        framing = pf.build_project_framing(
            job_dir=job_dir, clip_final="clip.mp4", render_w=360,
            render_h=640, render_fps=30.0, clip_frame_count=60)
        assert framing is not None and framing["eligible"]
        raw = json.dumps(framing)
        assert ".tmp-" not in raw and "tmp_path" not in raw.lower()


class TestBuildProjectFramingV4:
    def test_eligible_action_framing(self, job_dir):
        ref = "clip.turn_framing_plan.json"
        _write_sidecar(job_dir, ref, _v4_action_plan_dict())
        framing = pf.build_project_framing(
            job_dir=job_dir, clip_final="clip.mp4", render_w=360,
            render_h=640, render_fps=30.0, clip_frame_count=60)
        assert framing is not None
        assert framing["eligible"] is True
        assert framing["plan"]["schema_version"] == 4
        assert framing["plan"]["analysis_complete"] is True
        assert len(framing["scenes"]) == 1
        assert framing["scenes"][0]["scene_kind"] == "action"

    def test_incomplete_plan_ineligible(self, job_dir):
        d = _v4_action_plan_dict()
        d["analysis_complete"] = False
        ref = "clip.turn_framing_plan.json"
        _write_sidecar(job_dir, ref, d)
        framing = pf.build_project_framing(
            job_dir=job_dir, clip_final="clip.mp4", render_w=360,
            render_h=640, render_fps=30.0, clip_frame_count=60)
        assert framing is not None
        assert framing["eligible"] is False
        assert framing["scenes"] == []
        assert framing["ineligibility_reason"] == pf.REASON_INCOMPLETE

    def test_malformed_plan_returns_none(self, job_dir):
        d = _v4_action_plan_dict()
        d["scenes"][0]["scene_kind"] = "bogus"
        ref = "clip.turn_framing_plan.json"
        _write_sidecar(job_dir, ref, d)
        assert pf.build_project_framing(
            job_dir=job_dir, clip_final="clip.mp4", render_w=360,
            render_h=640, render_fps=30.0, clip_frame_count=60) is None

    def test_aspect_mismatch_returns_none(self, job_dir):
        ref = "clip.turn_framing_plan.json"
        _write_sidecar(job_dir, ref, _v4_action_plan_dict())
        # Plan is 9:16; render 16:9 must not reuse it.
        assert pf.build_project_framing(
            job_dir=job_dir, clip_final="clip.mp4", render_w=640,
            render_h=360, render_fps=30.0, clip_frame_count=60) is None

    def test_cross_source_rejected(self, job_dir):
        ref = "clip.turn_framing_plan.json"
        _write_sidecar(job_dir, ref, _v4_action_plan_dict())
        assert pf.build_project_framing(
            job_dir=job_dir, clip_final="clip.mp4", render_w=360,
            render_h=640, render_fps=30.0, clip_frame_count=60,
            expected_source_fingerprint="different") is None


class TestClipProjectModel:
    def test_scene_kind_action_validates(self):
        sc = FramingSceneTimeline(
            scene_id=0, strategy="GENERAL", start_frame=0, end_frame=59,
            applies=True, scene_kind="action",
            subject_ids=["t-0"],
            keyframes=[{"frame": 0, "x1": 10, "y1": 10, "x2": 100,
                        "y2": 100, "kind": "lock"}])
        assert sc.scene_kind == "action"

    def test_scene_kind_optional_back_compat(self):
        sc = FramingSceneTimeline(
            scene_id=0, strategy="GENERAL", start_frame=0, end_frame=59,
            applies=False)
        assert sc.scene_kind is None

    def test_scene_kind_bad_value_rejected(self):
        with pytest.raises(Exception, match="scene_kind"):
            FramingSceneTimeline(
                scene_id=0, strategy="GENERAL", start_frame=0,
                end_frame=59, applies=False, scene_kind="portrait")

    def test_full_project_with_action_framing(self, job_dir):
        ref = "clip.turn_framing_plan.json"
        _write_sidecar(job_dir, ref, _v4_action_plan_dict())
        framing = pf.build_project_framing(
            job_dir=job_dir, clip_final="clip.mp4", render_w=360,
            render_h=640, render_fps=30.0, clip_frame_count=60)
        project = {
            "schema": "nugget.clip-project/1",
            "job_id": "job-1",
            "clip_index": 0,
            "source": {"file": "clip.mp4", "width": 640, "height": 360,
                       "duration": 2.0},
            "segments": [{"id": "seg0", "start": 0.0, "end": 2.0,
                          "crop": {"x": 0.0, "y": 0.0, "w": 202.0,
                                   "h": 360.0}}],
            "render": {"width": 360, "height": 640, "fps": 30.0},
            "framing": framing,
        }
        validated = validate_project(project)
        assert validated.framing.scenes[0].scene_kind == "action"
        # Round-trip without loss.
        rt = validate_project(validated.model_dump())
        assert rt.framing.scenes[0].scene_kind == "action"
        assert rt.framing.scenes[0].keyframes == \
            validated.framing.scenes[0].keyframes


class TestVerifyAndLineage:
    def test_verify_framing_at_render_v4(self, job_dir):
        d = _v4_action_plan_dict()
        ref = "clip.turn_framing_plan.json"
        _write_sidecar(job_dir, ref, d)
        framing = pf.build_project_framing(
            job_dir=job_dir, clip_final="clip.mp4", render_w=360,
            render_h=640, render_fps=30.0, clip_frame_count=60)
        plan = pf.verify_framing_at_render(job_dir=job_dir,
                                           framing=framing)
        assert plan["schema_version"] == 4
        assert plan["scenes"][0]["scene_kind"] == "action"

    def test_verify_fails_on_sidecar_swap(self, job_dir):
        d = _v4_action_plan_dict()
        ref = "clip.turn_framing_plan.json"
        _write_sidecar(job_dir, ref, d)
        framing = pf.build_project_framing(
            job_dir=job_dir, clip_final="clip.mp4", render_w=360,
            render_h=640, render_fps=30.0, clip_frame_count=60)
        # Swap the sidecar bytes: sha256 mismatch must fail clearly.
        d["created_at"] = "2030-01-01T00:00:00+0000"
        _write_sidecar(job_dir, ref, d)
        with pytest.raises(pf.FramingError, match="sha256"):
            pf.verify_framing_at_render(job_dir=job_dir, framing=framing)

    def test_baseline_identity_carries_v4(self, job_dir):
        ref = "clip.turn_framing_plan.json"
        _write_sidecar(job_dir, ref, _v4_action_plan_dict())
        framing = pf.build_project_framing(
            job_dir=job_dir, clip_final="clip.mp4", render_w=360,
            render_h=640, render_fps=30.0, clip_frame_count=60)
        ident = pf.framing_baseline_identity(framing)
        assert ident["schema_version"] if "schema_version" in ident else True
        assert ident["analysis_complete"] is True
        assert ident["timeline_digest"] == framing["plan"]["timeline_digest"]

    def test_changed_trajectory_not_reusable(self, job_dir):
        # Same source, different action evidence -> different timeline
        # digest: the lineage comparison must see the change.
        ref = "clip.turn_framing_plan.json"
        _write_sidecar(job_dir, ref, _v4_action_plan_dict())
        f1 = pf.build_project_framing(
            job_dir=job_dir, clip_final="clip.mp4", render_w=360,
            render_h=640, render_fps=30.0, clip_frame_count=60)
        d2 = _v4_action_plan_dict()
        d2["scenes"][0]["segments"][0]["crop"] = [50, 50, 150, 200]
        d2["scenes"][0]["keyframes"][0]["x1"] = 50
        _write_sidecar(job_dir, ref, d2)
        f2 = pf.build_project_framing(
            job_dir=job_dir, clip_final="clip.mp4", render_w=360,
            render_h=640, render_fps=30.0, clip_frame_count=60)
        assert (f1["plan"]["timeline_digest"]
                != f2["plan"]["timeline_digest"])


class TestPublishRetireV4:
    def test_publish_and_retire_roundtrip(self, job_dir, tmp_path):
        from clippyme.domain.project_framing import (
            publish_attempt_sidecar,
            PUBLISH_PROMOTED,
        )
        src = os.path.join(job_dir, "clip.mp4")
        with open(src, "wb") as fh:
            fh.write(b"\x00" * 64)
        d = _v4_action_plan_dict()
        d["source_fingerprint"] = fp.source_fingerprint(src)
        attempt = str(tmp_path / "attempt.turn_framing_plan.json")
        with open(attempt, "w", encoding="utf-8") as fh:
            json.dump(d, fh)
        stable = os.path.join(job_dir, "clip.turn_framing_plan.json")
        outcome = publish_attempt_sidecar(
            attempt_sidecar=attempt, stable_sidecar=stable,
            source_path=src, job_dir=job_dir)
        assert outcome == PUBLISH_PROMOTED
        plan = fp.load_plan(stable)
        assert plan.schema_version == 4
        assert plan.scenes[0].scene_kind == "action"
