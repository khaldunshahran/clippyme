"""Task 13 review-01 corrections: coordinated publication, source binding,
preserved artifacts, and the framing-without-lineage render guard.

Part 1 promotes the coordinator's four independent production-path probes
verbatim (``coordinator-review1-remaining.py``): promotion failure, a
sidecar-less later attempt, source identity at emission, and preservation
of a user project's immutable sidecar.

Part 2 adds the missing render-guard regression (framing set but
``reframed_base`` absent must fail clearly BEFORE static
materialization), covers all three entry points (queued orchestrator
``_render_one_clip``, direct CLI main.py clip loop, reframe-only path),
and adds load-bearing mutation gates: each correction is temporarily
disabled and the corresponding probe is shown to fail.
"""

from __future__ import annotations

import argparse
import ast
import asyncio
import copy
import hashlib
import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from clippyme.domain import project_framing as pf
from clippyme.domain import project_render as pr
from clippyme.pipeline import framing_plan
from clippyme.pipeline import orchestrator as orch
from clippyme.pipeline import project_emit
from clippyme.domain.clip_project import validate_project
from tests.pipeline.conftest import framing_impl
from tests.pipeline.test_sidecar_promotion import (
    promo_job,
    SOURCE_BYTES,
    SLICE_BYTES,
    VIDEO_BYTES,
    _direct_env,
)
from tests.pipeline.task13_helpers import (
    sidecar_for,
    write_task12_sidecar,
    write_task12_sidecar_for_source,
)


# ===========================================================================
# Part 1: the four coordinator probes, promoted verbatim.
# ===========================================================================

def _stable_plan(job) -> dict:
    return json.loads(Path(sidecar_for(str(job.final))).read_text(encoding="utf-8"))


def test_sidecar_promotion_failure_cannot_publish_mixed_pair(promo_job, monkeypatch):
    """A good video cannot be accepted beside the previous attempt's plan."""
    job = promo_job
    assert orch._render_one_clip(**job.kwargs)
    assert _stable_plan(job)["config_digest"] == "cfg-attempt-1"

    # Force another render while retaining the previously accepted sidecar.
    job.final.unlink()
    real_replace = os.replace

    def fail_attempt_sidecar_replace(src, dst):
        if (str(src).endswith(".turn_framing_plan.json")
                and ".render.tmp" in str(src)):
            raise OSError("injected sidecar promotion failure")
        return real_replace(src, dst)

    monkeypatch.setattr(orch.os, "replace", fail_attempt_sidecar_replace)
    accepted = orch._render_one_clip(**job.kwargs)

    # Either publication fails/rolls back, or the accepted pair must belong to
    # the same attempt. Returning success with attempt-1's sidecar next to the
    # attempt-2 video violates the stable-pair contract.
    if accepted and job.final.exists():
        assert _stable_plan(job)["config_digest"] == "cfg-attempt-2"


def test_sidecarless_new_attempt_cannot_inherit_old_stable_plan(promo_job):
    """Static/disabled or otherwise sidecar-less rerenders clear stale plans."""
    job = promo_job
    assert orch._render_one_clip(**job.kwargs)
    assert Path(sidecar_for(str(job.final))).is_file()

    job.final.unlink()
    job.control.write_sidecar = False
    assert orch._render_one_clip(**job.kwargs)

    stable = Path(sidecar_for(str(job.final)))
    assert not stable.exists(), "new sidecar-less output must not inherit old framing"
    emitted = json.loads((job.root / "clip-project-0.json").read_text(encoding="utf-8"))
    assert "framing" not in emitted


def test_emission_rejects_valid_plan_from_a_different_source(promo_job):
    """The sidecar's Task 12 source identity must match the current slice."""
    job = promo_job
    assert orch._render_one_clip(**job.kwargs)
    emitted = json.loads((job.root / "clip-project-0.json").read_text(encoding="utf-8"))
    assert emitted.get("framing") is not None

    actual_slice_fp = framing_plan.source_fingerprint(str(job.sl))
    stored_fp = emitted["framing"]["plan"]["source_fingerprint"]
    assert stored_fp == actual_slice_fp, (
        "emission trusted a structurally valid sidecar from another source"
    )


def test_preserved_user_project_keeps_its_referenced_sidecar(promo_job):
    """A rerender cannot overwrite the immutable plan of a preserved project."""
    job = promo_job
    assert orch._render_one_clip(**job.kwargs)
    project_path = job.root / "clip-project-0.json"
    project = json.loads(project_path.read_text(encoding="utf-8"))
    project["origin"] = "user"
    project["version"] = 2
    project_path.write_text(json.dumps(project, sort_keys=True), encoding="utf-8")
    preserved_bytes = project_path.read_bytes()
    preserved_framing = project["framing"]
    old_hash = preserved_framing["plan"]["sha256"]
    assert old_hash == hashlib.sha256(
        Path(sidecar_for(str(job.final))).read_bytes()
    ).hexdigest()

    # Force regeneration. The project is intentionally preserved by Task 01,
    # so the plan it references must remain resolvable too.
    job.final.unlink()
    assert orch._render_one_clip(**job.kwargs)
    assert project_path.read_bytes() == preserved_bytes
    pf.verify_framing_at_render(
        job_dir=str(job.root), framing=preserved_framing
    )


# ===========================================================================
# Part 2a: coordinated-publication unit tests.
# ===========================================================================

def _write_attempt(tmp_path, name="clip.render.tmp.mp4", marker="x",
                 source_path=None):
    video = tmp_path / name
    video.write_bytes(VIDEO_BYTES)
    if source_path is None:
        sidecar, _ = write_task12_sidecar(str(video), marker)
    else:
        # Production-shaped: the plan names the actual render input.
        sidecar, _ = write_task12_sidecar_for_source(
            str(video), marker, str(source_path))
    return str(video), sidecar


def test_verify_accepts_plan_matching_source_and_rewrites_nothing(tmp_path):
    # Review-02: source identity is verified, never rewritten. A plan that
    # already names the render input is accepted byte-identical.
    src = tmp_path / "slice.mp4"
    src.write_bytes(SLICE_BYTES)
    video, sidecar = _write_attempt(tmp_path, source_path=src)
    assert pf.verify_attempt_sidecar_source(
        sidecar, source_path=str(src)) is True
    before = Path(sidecar).read_bytes()
    assert pf.verify_attempt_sidecar_source(
        sidecar, source_path=str(src)) is True
    assert Path(sidecar).read_bytes() == before, (
        "verification must not rewrite the plan")


def test_verify_rejects_malformed_sidecar(tmp_path):
    bad = tmp_path / "clip.render.tmp.turn_framing_plan.json"
    bad.write_text("{not json", encoding="utf-8")
    src = tmp_path / "slice.mp4"
    src.write_bytes(SLICE_BYTES)
    with pytest.raises(pf.FramingError):
        pf.verify_attempt_sidecar_source(str(bad), source_path=str(src))


def test_verify_rejects_foreign_plan_without_relabelling(tmp_path):
    # A production-valid plan from source A is rejected for source B --
    # never relabelled to B.
    source_a = tmp_path / "source-a.mp4"
    source_b = tmp_path / "source-b.mp4"
    source_a.write_bytes(b"A" * 20000)
    source_b.write_bytes(b"B" * 20000)
    _, attempt = _write_attempt(tmp_path, marker="foreign",
                                source_path=source_a)
    before = Path(attempt).read_bytes()
    with pytest.raises(pf.FramingError) as exc:
        pf.verify_attempt_sidecar_source(
            attempt, source_path=str(source_b))
    assert "framing-cross-source" in str(exc.value)
    assert Path(attempt).read_bytes() == before, (
        "a rejected plan must not be relabelled")


def test_publish_promotes_source_verified_plan(tmp_path):
    src = tmp_path / "slice.mp4"
    src.write_bytes(SLICE_BYTES)
    video, attempt = _write_attempt(tmp_path, marker="a1", source_path=src)
    stable = sidecar_for(str(tmp_path / "clip.mp4"))
    assert pf.publish_attempt_sidecar(
        attempt_sidecar=attempt, stable_sidecar=stable,
        source_path=str(src), job_dir=str(tmp_path)) == pf.PUBLISH_PROMOTED
    assert not os.path.exists(attempt)
    plan = json.loads(Path(stable).read_text(encoding="utf-8"))
    assert plan["config_digest"] == "cfg-a1"
    assert plan["source_fingerprint"] == framing_plan.source_fingerprint(
        str(src))


def test_publish_rejects_foreign_attempt_plan_as_sidecar_less(tmp_path):
    # Review-02: a foreign attempt plan is rejected -- the attempt is
    # treated as sidecar-less (stale binding retired), never promoted.
    source_a = tmp_path / "source-a.mp4"
    source_b = tmp_path / "source-b.mp4"
    source_a.write_bytes(b"A" * 20000)
    source_b.write_bytes(b"B" * 20000)
    _, attempt = _write_attempt(tmp_path, marker="foreign",
                                source_path=source_a)
    stable = sidecar_for(str(tmp_path / "clip.mp4"))
    write_task12_sidecar(str(tmp_path / "clip.mp4"), "stale")
    out = pf.publish_attempt_sidecar(
        attempt_sidecar=attempt, stable_sidecar=stable,
        source_path=str(source_b), job_dir=str(tmp_path))
    assert out != pf.PUBLISH_PROMOTED
    assert not os.path.exists(stable), (
        "a foreign plan must not be laundered; the stale binding is retired")
    assert not os.path.exists(attempt), "rejected attempt must not leak"


def test_publish_retires_stale_sidecar(tmp_path):
    stable = sidecar_for(str(tmp_path / "clip.mp4"))
    write_task12_sidecar(str(tmp_path / "clip.mp4"), "stale")
    assert pf.publish_attempt_sidecar(
        attempt_sidecar=str(tmp_path / "missing.render.tmp.turn_framing_plan.json"),
        stable_sidecar=stable, source_path=str(tmp_path / "slice.mp4"),
        job_dir=str(tmp_path)) == pf.PUBLISH_RETIRED
    assert not os.path.exists(stable)


def test_publish_no_sidecar_no_stable(tmp_path):
    out = pf.publish_attempt_sidecar(
        attempt_sidecar=str(tmp_path / "missing.render.tmp.turn_framing_plan.json"),
        stable_sidecar=sidecar_for(str(tmp_path / "clip.mp4")),
        source_path=str(tmp_path / "slice.mp4"), job_dir=str(tmp_path))
    assert out == pf.PUBLISH_NO_SIDECAR


def test_publish_skips_clobber_of_preserved_reference(tmp_path):
    # Stable sidecar pinned by a preserved user project; the new attempt's
    # different plan must not overwrite it.
    stable = sidecar_for(str(tmp_path / "clip.mp4"))
    _, stable_plan = write_task12_sidecar(str(tmp_path / "clip.mp4"), "old")
    stable_bytes = Path(stable).read_bytes()
    stable_sha = hashlib.sha256(stable_bytes).hexdigest()
    project = {
        "origin": "user", "version": 2,
        "framing": {"plan": {
            "plan_ref": os.path.basename(stable), "sha256": stable_sha}},
    }
    (tmp_path / "clip-project-0.json").write_text(
        json.dumps(project), encoding="utf-8")
    src = tmp_path / "slice.mp4"
    src.write_bytes(SLICE_BYTES)
    _, attempt = _write_attempt(tmp_path, marker="new", source_path=src)
    out = pf.publish_attempt_sidecar(
        attempt_sidecar=attempt, stable_sidecar=stable,
        source_path=str(src), job_dir=str(tmp_path))
    assert out == pf.PUBLISH_SKIPPED_PRESERVED
    assert Path(stable).read_bytes() == stable_bytes
    assert not os.path.exists(attempt), "attempt temp file must not leak"


def test_build_project_framing_rejects_cross_source(tmp_path):
    base = tmp_path / "clip.mp4"
    base.write_bytes(VIDEO_BYTES)
    sidecar, _ = write_task12_sidecar(str(base), "foreign")
    src = tmp_path / "slice.mp4"
    src.write_bytes(SLICE_BYTES)
    # Positive control: the sidecar's own (foreign) identity is accepted when
    # it is the expected one.
    ok = pf.build_project_framing(
        job_dir=str(tmp_path), clip_final=str(base),
        render_w=608, render_h=1080, render_fps=30, clip_frame_count=60,
        expected_source_fingerprint="fp-" + "b" * 16)
    assert ok is not None
    # The actual slice has a different identity: rejected.
    assert pf.build_project_framing(
        job_dir=str(tmp_path), clip_final=str(base),
        render_w=608, render_h=1080, render_fps=30, clip_frame_count=60,
        expected_source_fingerprint=framing_plan.source_fingerprint(str(src)),
    ) is None


# ===========================================================================
# Part 2b: the render guard -- framing without a lineage base must fail
# BEFORE static materialization. Old projects keep working.
# ===========================================================================

def _render_stubs_local(monkeypatch, calls):
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


def _guard_project_dict(job_dir, framing):
    src = os.path.join(job_dir, "source_0.mp4")
    Path(src).write_bytes(SOURCE_BYTES)
    proj = {
        "schema": "nugget.clip-project/1",
        "job_id": "job1",
        "clip_index": 0,
        "version": 1,
        "origin": "auto",
        "source": {"file": "source_0.mp4", "width": 1920, "height": 1080,
                   "duration": 2.0},
        "segments": [{
            "id": "seg-1", "start": 0.0, "end": 2.0,
            "crop": {"x": 660.0, "y": 0.0, "w": 600.0, "h": 1080.0},
            "transition_in": "cut",
        }],
        "captions": {"style": "classic_white", "position": "bottom",
                     "words": [], "edits": {}},
        "overlays": [],
        "audio": [{"file": None, "volume": 1.0, "offset": 0.0}],
        "grade": {"preset": "default"},
        "render": {"width": 608, "height": 1080, "fps": 30,
                   "encoder": "libx264"},
    }
    if framing is not None:
        proj["framing"] = framing
    return proj


def _framing_record(job_dir):
    base = os.path.join(job_dir, "clip_final_0.mp4")
    Path(base).write_bytes(b"base" * 2000)
    write_task12_sidecar(base, "guard")
    framing = pf.build_project_framing(
        job_dir=job_dir, clip_final=base,
        render_w=608, render_h=1080, render_fps=30, clip_frame_count=60)
    assert framing is not None
    return framing


def test_render_framing_without_lineage_base_fails_before_materialization(
    tmp_path, monkeypatch
):
    """Regression: framing set but reframed_base absent must fail clearly."""
    job_dir = str(tmp_path / "job")
    os.makedirs(job_dir)
    framing = _framing_record(job_dir)
    project = validate_project(_guard_project_dict(job_dir, framing))
    assert project.framing is not None
    assert project.reframed_base is None
    calls = []
    _render_stubs_local(monkeypatch, calls)
    with pytest.raises(pr.ProjectRenderError) as exc:
        asyncio.run(pr.render_project(
            job_id="job1", clip_index=0, project=project, job_dir=job_dir,
            metadata={}, clip_info={"title": "t"}))
    assert "framing-trajectory-edit" in str(exc.value)
    assert "materialize" not in calls, (
        "the guard must fire BEFORE static materialization")


def test_render_old_project_without_framing_or_base_still_materializes(
    tmp_path, monkeypatch
):
    """Compatibility: genuinely old projects keep the static path."""
    job_dir = str(tmp_path / "job")
    os.makedirs(job_dir)
    project = validate_project(_guard_project_dict(job_dir, None))
    assert project.framing is None
    assert project.reframed_base is None
    calls = []
    _render_stubs_local(monkeypatch, calls)
    result = asyncio.run(pr.render_project(
        job_id="job1", clip_index=0, project=project, job_dir=job_dir,
        metadata={}, clip_info={"title": "t"}))
    assert "materialize" in calls
    assert os.path.isfile(result.output_path)


# ===========================================================================
# Part 2c: all three entry points.
# ===========================================================================

def _compile_main_block(kind):
    main_path = Path(orch.__file__).with_name("main.py")
    tree = ast.parse(main_path.read_text(encoding="utf8"))
    if kind == "loop":
        nodes = [
            node for node in ast.walk(tree) if isinstance(node, ast.For)
            and ast.unparse(node.iter) == "enumerate(clips_data['shorts'])"
        ]
    else:
        nodes = [
            node for node in ast.walk(tree) if isinstance(node, ast.If)
            and ast.unparse(node.test) == "args.reframe_only"
        ]
    assert len(nodes) == 1, f"locate the actual main.py {kind}"
    return compile(ast.Module(body=[nodes[0]], type_ignores=[]),
                   str(main_path), "exec")


def _fail_sidecar_replace(monkeypatch, marker):
    real_replace = os.replace

    def failing(src, dst):
        if (str(src).endswith(".turn_framing_plan.json")
                and marker in str(src)):
            raise OSError("injected sidecar promotion failure")
        return real_replace(src, dst)

    monkeypatch.setattr(os, "replace", failing)


def test_direct_cli_promotion_failure_restores_previous_pair(
    tmp_path, monkeypatch
):
    """Direct CLI loop: promotion failure restores the accepted pair.

    Review-02: rollback means restoring the previous video+sidecar, not
    deleting the video.
    """
    ctx = _direct_env(tmp_path, "success")
    old_video = ctx.final.read_bytes()
    _fail_sidecar_replace(monkeypatch, ".render.tmp")
    exec(_compile_main_block("loop"), ctx.env)
    assert ctx.final.read_bytes() == old_video, (
        "failed promotion must restore the previous video")
    assert not os.path.exists(sidecar_for(str(ctx.final))), (
        "no stable sidecar may be left behind")
    temp_sidecar = sidecar_for(str(ctx.final) + ".render.tmp.mp4")
    assert not os.path.exists(temp_sidecar), "attempt temp sidecar must not leak"


def test_direct_cli_sidecarless_render_retires_stale_plan(tmp_path):
    """Direct CLI loop: a sidecar-less render retires the previous binding."""
    ctx = _direct_env(tmp_path, "success", write_sidecar=False)
    stale = sidecar_for(str(ctx.final))
    write_task12_sidecar(str(ctx.final), "stale")
    assert os.path.isfile(stale)
    exec(_compile_main_block("loop"), ctx.env)
    assert ctx.final.read_bytes() == VIDEO_BYTES, "the clip itself is accepted"
    assert not os.path.exists(stale), "stale plan must be retired"


def _reframe_only_env(tmp_path):
    from clippyme.domain import project_framing as _pf_module
    src = tmp_path / "slice.mp4"
    src.write_bytes(SLICE_BYTES)
    final = tmp_path / "out.mp4"

    def render(source_path, output_path, **kwargs):
        Path(output_path).write_bytes(VIDEO_BYTES)
        write_task12_sidecar_for_source(
            output_path, "reframe-only", source_path)
        return True

    class _Sys:
        def exit(self, code=0):
            raise SystemExit(code)

    import time as _time
    env = dict(
        os=os, sys=_Sys(), time=_time,
        args=argparse.Namespace(
            reframe_only=True, input=str(src), output=str(final),
            reframe_mode="auto", no_zoom=True),
        process_video_to_vertical=render,
        normalize_audio=lambda p: None, select_cover_frame=lambda p: None,
        turn_framing_sidecar_for=sidecar_for,
        aspect_ratio=9 / 16, letterbox_zoom=0,
        _project_framing=_pf_module,
    )
    return SimpleNamespace(env=env, src=src, final=final)


def test_reframe_only_promotion_failure_rolls_back(tmp_path, monkeypatch):
    """Reframe-only path: promotion failure exits non-zero with no mixed pair."""
    ctx = _reframe_only_env(tmp_path)
    _fail_sidecar_replace(monkeypatch, ".reframe.tmp")
    with pytest.raises(SystemExit) as exc:
        exec(_compile_main_block("reframe"), ctx.env)
    assert exc.value.code == 1
    assert not ctx.final.exists(), "failed promotion must roll back the video"
    assert not os.path.exists(sidecar_for(str(ctx.final)))


def test_reframe_only_success_still_promotes(tmp_path):
    ctx = _reframe_only_env(tmp_path)
    with pytest.raises(SystemExit) as exc:
        exec(_compile_main_block("reframe"), ctx.env)
    assert exc.value.code == 0
    assert ctx.final.read_bytes() == VIDEO_BYTES
    plan = json.loads(
        Path(sidecar_for(str(ctx.final))).read_text(encoding="utf-8"))
    assert plan["config_digest"] == "cfg-reframe-only"
    assert plan["source_fingerprint"] == framing_plan.source_fingerprint(
        str(ctx.src))


# ===========================================================================
# Part 2d: load-bearing mutation gates. Each disables one correction and
# proves the corresponding probe then fails.
# ===========================================================================

def _legacy_best_effort_publish(*, attempt_sidecar, stable_sidecar, **kwargs):
    """Pre-review-01 behavior: best-effort promotion, failure swallowed."""
    try:
        if os.path.isfile(attempt_sidecar):
            os.replace(attempt_sidecar, stable_sidecar)
            return pf.PUBLISH_PROMOTED
    except OSError:
        pass
    try:
        if os.path.isfile(attempt_sidecar):
            os.remove(attempt_sidecar)
    except OSError:
        pass
    return pf.PUBLISH_NO_SIDECAR


def test_mutation_no_rollback_leaves_mixed_pair(promo_job, monkeypatch):
    """Disabling coordinated publication reintroduces the probe-1 mixed pair."""
    job = promo_job
    assert orch._render_one_clip(**job.kwargs)
    job.final.unlink()
    real_replace = os.replace

    def fail_attempt_sidecar_replace(src, dst):
        if (str(src).endswith(".turn_framing_plan.json")
                and ".render.tmp" in str(src)):
            raise OSError("injected sidecar promotion failure")
        return real_replace(src, dst)

    monkeypatch.setattr(orch.os, "replace", fail_attempt_sidecar_replace)
    monkeypatch.setattr(pf, "publish_attempt_sidecar",
                        _legacy_best_effort_publish)
    accepted = orch._render_one_clip(**job.kwargs)
    # The bug, reproduced: success with attempt-1's plan beside the
    # attempt-2 video.
    assert accepted and job.final.exists()
    assert _stable_plan(job)["config_digest"] == "cfg-attempt-1"


def _no_retire_publish(*, attempt_sidecar, stable_sidecar, **kwargs):
    """Promotion without the retire rule: stale bindings linger."""
    if os.path.isfile(attempt_sidecar):
        os.replace(attempt_sidecar, stable_sidecar)
        return pf.PUBLISH_PROMOTED
    return pf.PUBLISH_NO_SIDECAR


def test_mutation_no_retire_keeps_stale_plan(promo_job, monkeypatch):
    """Disabling the retire rule reintroduces the probe-2 stale inheritance."""
    job = promo_job
    assert orch._render_one_clip(**job.kwargs)
    job.final.unlink()
    job.control.write_sidecar = False
    monkeypatch.setattr(pf, "publish_attempt_sidecar", _no_retire_publish)
    assert orch._render_one_clip(**job.kwargs)
    # The bug, reproduced: the stale stable plan survives and is emitted.
    assert Path(sidecar_for(str(job.final))).is_file()
    emitted = json.loads((job.root / "clip-project-0.json").read_text(encoding="utf-8"))
    assert "framing" in emitted


def _install_foreign_renderer(job):
    """Renderer writing plans with the default foreign fingerprint.

    Simulates a renderer that does NOT stamp the true render input
    (pre-production shape); publication must reject these plans.
    """
    def foreign_render(source_path, dest, **kwargs):
        job.control.attempt += 1
        job.control.render_calls.append(dest)
        Path(dest).write_bytes(VIDEO_BYTES)
        write_task12_sidecar(dest, f"attempt-{job.control.attempt}")
        return True

    job.kwargs["legacy"].process_video_to_vertical = foreign_render


def test_mutation_no_source_verification_promotes_foreign_plan(
    promo_job, monkeypatch
):
    """Disabling mismatch rejection promotes a foreign plan (stale framing)."""
    job = promo_job
    monkeypatch.setattr(pf, "verify_attempt_sidecar_source",
                        lambda *a, **k: True)
    _install_foreign_renderer(job)
    assert orch._render_one_clip(**job.kwargs)
    plan = _stable_plan(job)
    # The bug, reproduced: the foreign plan is promoted; its stale source
    # identity does not match the actual slice.
    assert plan["source_fingerprint"] == "fp-" + "b" * 16
    assert plan["source_fingerprint"] != framing_plan.source_fingerprint(
        str(job.sl))


def test_mutation_no_emission_check_trusts_foreign_plan(promo_job, monkeypatch):
    """Disabling the emission-side source check trusts the foreign plan."""
    real_build = pf.build_project_framing

    def no_check(**kwargs):
        kwargs["expected_source_fingerprint"] = None
        return real_build(**kwargs)

    monkeypatch.setattr(pf, "verify_attempt_sidecar_source",
                        lambda *a, **k: True)
    monkeypatch.setattr(pf, "build_project_framing", no_check)
    job = promo_job
    _install_foreign_renderer(job)
    assert orch._render_one_clip(**job.kwargs)
    emitted = json.loads((job.root / "clip-project-0.json").read_text(encoding="utf-8"))
    # The bug, reproduced: without the emission-side comparison the
    # structurally valid foreign fingerprint is trusted.
    assert emitted["framing"]["plan"]["source_fingerprint"] == "fp-" + "b" * 16
    assert emitted["framing"]["plan"]["source_fingerprint"] != (
        framing_plan.source_fingerprint(str(job.sl)))


def _no_preserve_publish(*, attempt_sidecar, stable_sidecar, **kwargs):
    """Promotion without the preserved-project guard clobbers the artifact."""
    os.replace(attempt_sidecar, stable_sidecar)
    return pf.PUBLISH_PROMOTED


def test_mutation_no_preserve_guard_breaks_preserved_project(
    promo_job, monkeypatch
):
    """Disabling the preserved-project guard reintroduces the probe-4 break."""
    job = promo_job
    assert orch._render_one_clip(**job.kwargs)
    project_path = job.root / "clip-project-0.json"
    project = json.loads(project_path.read_text(encoding="utf-8"))
    project["origin"] = "user"
    project["version"] = 2
    project_path.write_text(json.dumps(project, sort_keys=True), encoding="utf-8")
    preserved_framing = project["framing"]
    job.final.unlink()
    # Bypass the review-02 rerender decline so the publish-time guard is
    # exercised in isolation.
    monkeypatch.setattr(orch, "project_has_user_edits", lambda p: False)
    monkeypatch.setattr(pf, "publish_attempt_sidecar", _no_preserve_publish)
    assert orch._render_one_clip(**job.kwargs)
    # The bug, reproduced: the preserved project's referenced sidecar was
    # overwritten, so verification fails.
    with pytest.raises(pf.FramingError):
        pf.verify_framing_at_render(
            job_dir=str(job.root), framing=preserved_framing)


def test_mutation_no_render_guard_silently_materializes(tmp_path, monkeypatch):
    """Disabling the render guard reintroduces silent static materialization."""
    job_dir = str(tmp_path / "job")
    os.makedirs(job_dir)
    framing = _framing_record(job_dir)
    project = validate_project(_guard_project_dict(job_dir, framing))
    monkeypatch.setattr(pr, "_require_framing_lineage_base",
                        lambda project: None)
    calls = []
    _render_stubs_local(monkeypatch, calls)
    result = asyncio.run(pr.render_project(
        job_id="job1", clip_index=0, project=project, job_dir=job_dir,
        metadata={}, clip_info={"title": "t"}))
    # The bug, reproduced: a static center crop is silently materialized
    # for a project whose recorded trajectory has no lineage base.
    assert "materialize" in calls
    assert os.path.isfile(result.output_path)


# ===========================================================================
# Part 2e: review-02 load-bearing mutation gates. Each disables one
# review-02 correction and proves the corresponding probe then fails.
# ===========================================================================

def test_mutation_no_staging_destroys_previous_pair(promo_job, monkeypatch):
    """Disabling backup staging destroys the old accepted pair on failure."""
    job = promo_job
    assert orch._render_one_clip(**job.kwargs)
    old_video = job.final.read_bytes()
    stable = Path(sidecar_for(str(job.final)))
    old_sidecar = stable.read_bytes()

    real_replace = os.replace

    def fail_attempt_sidecar_replace(src, dst):
        if (str(src).endswith(".turn_framing_plan.json")
                and ".render.tmp" in str(src)):
            raise OSError("injected sidecar promotion failure")
        return real_replace(src, dst)

    monkeypatch.setattr(orch.os, "replace", fail_attempt_sidecar_replace)
    # Disable staging: no backup is taken, and the failure path falls back
    # to deleting the just-published video (the old pair was already
    # replaced by os.replace).
    monkeypatch.setattr(
        pf, "stage_previous_pair",
        lambda **k: {"video": None, "sidecar": None, "dir": None})

    def no_restore(record, *, clip_final, stable_sidecar):
        try:
            os.remove(clip_final)
        except OSError:
            pass

    monkeypatch.setattr(pf, "restore_staged_pair", no_restore)

    # Force a re-render; the injected sidecar failure then destroys the
    # previously accepted pair.
    job.final.unlink()
    accepted = orch._render_one_clip(**job.kwargs)
    # The bug, reproduced: the previous accepted video is destroyed -- the
    # replacement was deleted and the old bytes are unrecoverable.
    assert not accepted
    assert not job.final.exists(), (
        "without staging, a failed publish destroys the old accepted video")


def test_mutation_swallowed_retire_claims_success_with_stale_file(
    tmp_path, monkeypatch
):
    """Swallowing the retirement removal reintroduces the false 'retired'."""
    stable = Path(sidecar_for(str(tmp_path / "clip.mp4")))
    write_task12_sidecar(str(tmp_path / "clip.mp4"), "stale")
    real_remove = os.remove

    def fail_stable_remove(path):
        if os.path.abspath(str(path)) == os.path.abspath(str(stable)):
            raise OSError("injected retirement failure")
        return real_remove(path)

    monkeypatch.setattr(pf.os, "remove", fail_stable_remove)

    def _swallowing_retire(*, job_dir, stable_sidecar):
        if not os.path.isfile(stable_sidecar):
            return pf.PUBLISH_NO_SIDECAR
        try:
            os.remove(stable_sidecar)
        except OSError:
            pass
        return pf.PUBLISH_RETIRED

    monkeypatch.setattr(pf, "_retire_stable_sidecar", _swallowing_retire)
    outcome = pf.publish_attempt_sidecar(
        attempt_sidecar=str(tmp_path / "missing.render.tmp.turn_framing_plan.json"),
        stable_sidecar=str(stable), source_path=str(tmp_path / "slice.mp4"),
        job_dir=str(tmp_path))
    # The bug, reproduced: reported "retired" while the stale file remains.
    assert outcome == pf.PUBLISH_RETIRED
    assert stable.exists(), "stale sidecar was not retired"


def _origin_version_only_refs(job_dir, stable_basename, stable_sha256):
    """The pre-review-02 narrow check: origin/version only, no edits."""
    try:
        names = os.listdir(job_dir)
    except OSError:
        return False
    for name in names:
        if not (name.startswith("clip-project-") and name.endswith(".json")):
            continue
        path = os.path.join(job_dir, name)
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, ValueError):
            continue
        if not isinstance(data, dict):
            continue
        try:
            preserved = (data.get("origin") == "user"
                         or int(data.get("version", 1)) > 1)
        except (TypeError, ValueError):
            preserved = True
        if not preserved:
            continue
        framing = data.get("framing")
        plan_id = (framing.get("plan")
                   if isinstance(framing, dict) else None)
        if not isinstance(plan_id, dict):
            continue
        ref = plan_id.get("plan_ref")
        if not isinstance(ref, str) or ref.rsplit("/", 1)[-1] != stable_basename:
            continue
        if plan_id.get("sha256") == stable_sha256:
            return True
    return False


def test_mutation_origin_version_only_protection_loses_caption_edited_plan(
    promo_job, monkeypatch
):
    """Narrowing protection to origin/version loses caption-edited plans."""
    job = promo_job
    assert orch._render_one_clip(**job.kwargs)
    project_path = job.root / "clip-project-0.json"
    project = json.loads(project_path.read_text(encoding="utf-8"))
    project["captions"]["edits"] = {"0": "corrected"}
    project_path.write_text(json.dumps(project, sort_keys=True), encoding="utf-8")
    preserved = json.loads(project_path.read_text(encoding="utf-8"))
    assert project_emit.project_has_user_edits(str(project_path)) is True

    # Bypass the rerender decline so the publish-time protector is
    # exercised in isolation, then narrow it to origin/version only.
    monkeypatch.setattr(orch, "project_has_user_edits", lambda p: False)
    monkeypatch.setattr(pf, "_preserved_project_refs_sidecar",
                        _origin_version_only_refs)
    job.final.unlink()
    assert orch._render_one_clip(**job.kwargs)
    # The bug, reproduced: the caption-edited project's referenced sidecar
    # was overwritten, so verification fails.
    with pytest.raises(pf.FramingError):
        pf.verify_framing_at_render(
            job_dir=str(job.root), framing=preserved["framing"])


def test_mutation_no_decline_replaces_preserved_pair(promo_job, monkeypatch):
    """Disabling the rerender decline lets a rerender replace the pair."""
    job = promo_job
    assert orch._render_one_clip(**job.kwargs)
    project_path = job.root / "clip-project-0.json"
    project = json.loads(project_path.read_text(encoding="utf-8"))
    project["origin"] = "user"
    project["version"] = 2
    project_path.write_text(json.dumps(project, sort_keys=True), encoding="utf-8")
    preserved_bytes = project_path.read_bytes()

    # Disable the decline (orchestrator's module-level binding).
    monkeypatch.setattr(orch, "project_has_user_edits", lambda p: False)
    job.final.unlink()
    assert orch._render_one_clip(**job.kwargs)
    from clippyme.domain import project_lineage as _pl
    preserved = json.loads(preserved_bytes.decode("utf-8"))
    outcome, _, reason = _pl.check_lineage_eligible(
        preserved["reframed_base"], preserved, str(job.root))
    assert outcome != "eligible", (
        "without the decline, the rerender invalidates the preserved lineage"
        f" (got {outcome}: {reason})")
