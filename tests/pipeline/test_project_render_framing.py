"""Task 13: ClipProject framing bridge -- lineage eligibility + render dispatch.

- ``build_lineage`` stores the framing baseline identity; ``_validate_baseline``
  rejects malformed framing as invalid_asset.
- ``check_lineage_eligible`` returns the NEW "framing_edit" outcome (with the
  stable "framing-trajectory-edit" reason) for added/removed/changed
  trajectories; malformed project framing is invalid_asset.
- ``render_project``: unchanged project reuses the Task 05 base (framing
  verified via ``verify_framing_at_render``); a mutated framing field raises
  ProjectRenderError (never a silent center crop, never _materialize_timeline);
  an explicit static crop edit keeps the accepted source-render fallback;
  old projects (no framing) keep their accepted behavior.

Mutation gates:
- ignoring framing changes during base eligibility (mutated framing field +
  eligible baseline) must NOT reuse the base;
- dropping the framing field during save/load must not silently validate.
"""

import asyncio
import copy
import hashlib
import json
import os
from pathlib import Path

import pytest

from clippyme.domain import project_framing as pf
from clippyme.domain import project_lineage as pl
from clippyme.domain import project_render as pr
from clippyme.domain.clip_project import validate_project
from tests.pipeline.task13_helpers import sidecar_for, write_task12_sidecar


BASE_BYTES = b"reframed base pixels" * 2000
SOURCE_BYTES = b"source slice pixels" * 2000
STABLE_SIDECAR = "clip_final_0.turn_framing_plan.json"


def _write_source(job_dir):
    src = os.path.join(job_dir, "source_0.mp4")
    # Write-once: the Task 12 source fingerprint samples mtime_ns, so a
    # rewrite between bind and check would invalidate the binding.
    if not os.path.isfile(src):
        Path(src).write_bytes(SOURCE_BYTES)
    return src


def _real_framing(job_dir, base_clip, marker="plan-1"):
    """Build a real framing record the way pipeline/project_emit does."""
    src = _write_source(job_dir)
    # Mirror production reframe: the plan is stamped with the actual render
    # input's Task 12 source identity at write time (review-02: publication
    # verifies rather than relabels).
    from tests.pipeline.task13_helpers import write_task12_sidecar_for_source
    sidecar, _ = write_task12_sidecar_for_source(base_clip, marker, src)
    from clippyme.pipeline.framing_plan import source_fingerprint as _fp12
    framing = pf.build_project_framing(
        job_dir=job_dir, clip_final=base_clip,
        render_w=608, render_h=1080, render_fps=30, clip_frame_count=60,
        expected_source_fingerprint=_fp12(src))
    assert framing is not None, "test plan must bridge to a framing record"
    assert framing["eligible"] is True
    return framing


def _segments():
    return [{
        "id": "seg-1", "start": 0.0, "end": 2.0,
        "crop": {"x": 660.0, "y": 0.0, "w": 600.0, "h": 1080.0},
        "transition_in": "cut",
    }]


def _project_dict(base_clip, job_dir, framing):
    lin = pl.build_lineage(
        clip_final=base_clip, job_dir=job_dir,
        source={"file": "source_0.mp4", "width": 1920, "height": 1080},
        segments=_segments(),
        render={"width": 608, "height": 1080, "fps": 30},
        source_path=_write_source(job_dir),
        framing=framing,
    )
    assert lin is not None
    proj = {
        "schema": "nugget.clip-project/1",
        "job_id": "job1",
        "clip_index": 0,
        "version": 1,
        "origin": "auto",
        "source": {"file": "source_0.mp4", "width": 1920, "height": 1080,
                   "duration": 2.0},
        "segments": _segments(),
        "captions": {"style": "classic_white", "position": "bottom",
                     "words": [], "edits": {}},
        "overlays": [],
        "audio": [{"file": None, "volume": 1.0, "offset": 0.0}],
        "grade": {"preset": "default"},
        "render": {"width": 608, "height": 1080, "fps": 30,
                   "encoder": "libx264"},
        "reframed_base": lin,
    }
    if framing is not None:
        proj["framing"] = framing
    return proj


@pytest.fixture
def job_dir(tmp_path):
    d = tmp_path / "job"
    d.mkdir()
    return str(d)


@pytest.fixture
def base_clip(job_dir):
    p = os.path.join(job_dir, "clip_final_0.mp4")
    Path(p).write_bytes(BASE_BYTES)
    return p


@pytest.fixture
def bridged(job_dir, base_clip):
    """(project, framing) with a real bridged framing record."""
    framing = _real_framing(job_dir, base_clip)
    project = validate_project(_project_dict(base_clip, job_dir, framing))
    assert project.framing is not None, "model must retain the framing field"
    assert project.framing.plan.plan_ref == STABLE_SIDECAR
    return project, framing


# --- Lineage: framing baseline identity ---


def test_build_lineage_stores_framing_identity(bridged):
    project, framing = bridged
    lin = project.reframed_base.model_dump()
    assert lin["framing"] == pf.framing_baseline_identity(framing)


def test_build_lineage_without_framing_stores_none(job_dir, base_clip):
    lin = pl.build_lineage(
        clip_final=base_clip, job_dir=job_dir,
        source={"file": "source_0.mp4", "width": 1920, "height": 1080},
        segments=_segments(), render={"width": 608, "height": 1080, "fps": 30},
        source_path=_write_source(job_dir), framing=None)
    assert lin is not None
    assert lin["framing"] is None, "old emissions degrade to no framing baseline"


def test_lineage_malformed_framing_is_invalid_asset(bridged, job_dir):
    project, _ = bridged
    lineage = copy.deepcopy(project.reframed_base.model_dump())
    lineage["framing"] = {"sha256": "not-hex"}
    proj = project.model_dump()
    outcome, _, reason = pl.check_lineage_eligible(lineage, proj, job_dir)
    assert outcome == "invalid_asset"
    assert "framing" in reason


def test_framing_added_is_framing_edit(job_dir, base_clip):
    proj = _project_dict(base_clip, job_dir, framing=None)
    proj["reframed_base"]["framing"] = None
    proj["framing"] = _real_framing(job_dir, base_clip)
    outcome, _, reason = pl.check_lineage_eligible(
        proj["reframed_base"], proj, job_dir)
    assert outcome == "framing_edit"
    assert "framing-trajectory-edit" in reason


def test_framing_removed_is_framing_edit(bridged, job_dir):
    project, _ = bridged
    proj = project.model_dump()
    proj["framing"] = None
    outcome, _, reason = pl.check_lineage_eligible(
        proj["reframed_base"], proj, job_dir)
    assert outcome == "framing_edit"
    assert "framing-trajectory-edit" in reason


def test_framing_changed_is_framing_edit(bridged, job_dir):
    project, framing = bridged
    proj = project.model_dump()
    proj["framing"]["plan"]["timeline_digest"] = "00" * 64
    outcome, _, reason = pl.check_lineage_eligible(
        proj["reframed_base"], proj, job_dir)
    assert outcome == "framing_edit"
    assert "timeline_digest" in reason
    assert "framing-trajectory-edit" in reason


def test_framing_malformed_project_is_invalid_asset(bridged, job_dir):
    project, _ = bridged
    proj = project.model_dump()
    proj["framing"] = "not-a-dict"
    outcome, _, reason = pl.check_lineage_eligible(
        proj["reframed_base"], proj, job_dir)
    assert outcome == "invalid_asset"
    assert "framing" in reason


def test_framing_unchanged_stays_eligible(bridged, job_dir, base_clip):
    project, _ = bridged
    proj = project.model_dump()
    outcome, path, reason = pl.check_lineage_eligible(
        proj["reframed_base"], proj, job_dir)
    assert outcome == "eligible"
    assert path == base_clip
    assert reason is None


# --- render_project dispatch ---


def _render_stubs(monkeypatch, calls):
    """Stub everything after base selection; record the base decision."""
    async def fake_materialize(project, job_dir_, clip_index):
        calls.append("materialize")
        p = os.path.join(job_dir_, "timeline_test.mp4")
        Path(p).write_bytes(b"timeline")
        return p, 2.0

    async def fake_compose(**kwargs):
        calls.append("compose")
        p = os.path.join(kwargs["job_dir"], "composed_test.mp4")
        Path(p).write_bytes(b"composed")
        return "composed_test.mp4"

    async def fake_mix(path, project, job_dir_, clip_index):
        return path

    def fake_derive(project, timeline_duration, metadata, clip_info):
        return ({}, {}, {}, {}, {}, {}, metadata, clip_info)

    def fake_probe(path):
        return {"has_video": True, "width": 608, "height": 1080,
                "fps": 30, "duration": 2.0}

    def fake_inspect(path, **kwargs):
        return {"critical": False, "warnings": []}

    monkeypatch.setattr(pr, "_materialize_timeline", fake_materialize)
    monkeypatch.setattr(pr, "compose_layers", fake_compose)
    monkeypatch.setattr(pr, "_mix_music", fake_mix)
    monkeypatch.setattr(pr, "_derive_compose_inputs", fake_derive)
    monkeypatch.setattr(pr, "probe_media", fake_probe)
    monkeypatch.setattr(pr, "inspect_clip", fake_inspect)


def _run(project, job_dir):
    return asyncio.run(pr.render_project(
        job_id="job1", clip_index=0, project=project, job_dir=job_dir,
        metadata={}, clip_info={"title": "t"}))


def test_render_unchanged_reuses_base_and_verifies_framing(
    bridged, job_dir, monkeypatch
):
    project, framing = bridged
    calls = []
    _render_stubs(monkeypatch, calls)
    verified = []
    real_verify = pf.verify_framing_at_render

    def spy_verify(*, job_dir, framing):
        verified.append(framing["plan"]["plan_ref"])
        return real_verify(job_dir=job_dir, framing=framing)

    monkeypatch.setattr(pf, "verify_framing_at_render", spy_verify)
    result = _run(project, job_dir)
    assert "materialize" not in calls, "eligible base must be reused"
    assert verified == [STABLE_SIDECAR]
    assert os.path.isfile(result.output_path), "composed output must exist"


def test_render_mutated_framing_fails_clearly(bridged, job_dir, monkeypatch):
    """Mutation gate: a changed framing record must NOT reuse the old base."""
    project, framing = bridged
    bad = copy.deepcopy(framing)
    bad["plan"]["sha256"] = "ab" * 32  # valid shape, wrong value: same-size
    # replacement must be detected
    project = validate_project({**project.model_dump(), "framing": bad})
    calls = []
    _render_stubs(monkeypatch, calls)

    with pytest.raises(pr.ProjectRenderError) as exc:
        _run(project, job_dir)
    assert "framing-trajectory-edit" in str(exc.value)
    assert "materialize" not in calls, (
        "a trajectory edit must never fall through to the source renderer"
    )


def test_render_stale_sidecar_fails_like_invalid_asset(
    bridged, job_dir, monkeypatch
):
    project, framing = bridged
    # Replace the sidecar bytes after emission: same name, new content.
    write_task12_sidecar(os.path.join(job_dir, "clip_final_0.mp4"), "replaced")
    calls = []
    _render_stubs(monkeypatch, calls)

    with pytest.raises(pr.ProjectRenderError) as exc:
        _run(project, job_dir)
    assert "framing sidecar unusable" in str(exc.value)
    assert "materialize" not in calls


def test_render_static_crop_edit_keeps_source_fallback(
    bridged, job_dir, monkeypatch
):
    project, _ = bridged
    calls = []
    _render_stubs(monkeypatch, calls)
    monkeypatch.setattr(
        pl, "check_lineage_eligible",
        lambda lineage, proj, jd: ("geometry_edit", None,
                                   "segment 0 crop x changed"),
    )
    result = _run(project, job_dir)
    assert "materialize" in calls, "static crop edit keeps the source fallback"
    assert os.path.isfile(result.output_path)


def test_render_framing_edit_never_materializes(bridged, job_dir, monkeypatch):
    project, _ = bridged
    calls = []
    _render_stubs(monkeypatch, calls)
    reason = "framing timeline_digest changed (reason: framing-trajectory-edit)"
    monkeypatch.setattr(
        pl, "check_lineage_eligible",
        lambda lineage, proj, jd: ("framing_edit", None, reason),
    )
    with pytest.raises(pr.ProjectRenderError) as exc:
        _run(project, job_dir)
    assert "framing-trajectory-edit" in str(exc.value)
    assert "dynamic replay is not implemented" in str(exc.value)
    assert "materialize" not in calls


def test_render_old_project_without_framing_keeps_working(
    job_dir, base_clip, monkeypatch
):
    # Old emission: reframed_base with no framing baseline, project with no
    # framing field at all.
    proj_dict = _project_dict(base_clip, job_dir, framing=None)
    proj_dict["reframed_base"]["framing"] = None
    assert "framing" not in proj_dict
    project = validate_project(proj_dict)
    assert project.framing is None
    calls = []
    _render_stubs(monkeypatch, calls)
    result = _run(project, job_dir)
    assert "materialize" not in calls, "old eligible base must still be reused"
    assert os.path.isfile(result.output_path)
