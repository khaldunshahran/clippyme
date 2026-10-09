"""Task 13 review-03 corrections: fail-closed staging, per-clip snapshots,
atomic snapshot generations, and one working/archived project resolver.

Part 1 promotes the coordinator's four review-03 integrity probes
(``coordinator-review3-remaining.py``, left untouched in the handoff dir):
a failed backup must abort mutation, snapshots are isolated per clip,
snapshot updates are atomic as a video+sidecar generation, and an
archived-only user project declines the rerender before its pixel base
changes.

Part 2 adds load-bearing mutation gates: staging failure aborts in the
queued and reframe-only callers (not just the direct loop), disabling
per-clip keying installs another clip's bytes, disabling atomic
generations leaves a mixed generation, and a working-only pre-render
guard lets an archived project rerender into invalid lineage.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pytest

from clippyme.domain import project_framing as pf
from clippyme.domain import project_lineage as pl
from clippyme.domain.clip_project import validate_project
from clippyme.domain.project_render import save_project_version
from clippyme.pipeline import orchestrator as orch
from tests.pipeline.conftest import framing_impl  # noqa: F401 (fixture)
from tests.pipeline.task13_helpers import sidecar_for, write_task12_sidecar
from tests.pipeline.test_sidecar_promotion import (
    VIDEO_BYTES,
    _direct_env,
    promo_job,
)
from tests.pipeline.test_task13_review1 import (
    _compile_main_block,
    _fail_sidecar_replace,
    _reframe_only_env,
)
from tests.pipeline.test_task13_review2 import _install_changed_renderer


# ===========================================================================
# Part 1: the four coordinator review-03 probes, promoted verbatim.
# ===========================================================================

def test_direct_stage_failure_preserves_previous_pair(tmp_path, monkeypatch):
    """A failed backup must abort mutation, not delete the accepted video."""
    ctx = _direct_env(tmp_path, "success")
    old_video = ctx.final.read_bytes()
    stable = Path(sidecar_for(str(ctx.final)))
    write_task12_sidecar(str(ctx.final), "accepted-old")
    old_sidecar = stable.read_bytes()

    real_copy2 = shutil.copy2

    def fail_video_backup(src, dst, *args, **kwargs):
        if os.path.abspath(str(src)) == os.path.abspath(str(ctx.final)):
            raise OSError("injected staging failure")
        return real_copy2(src, dst, *args, **kwargs)

    monkeypatch.setattr(shutil, "copy2", fail_video_backup)
    _fail_sidecar_replace(monkeypatch, ".render.tmp")
    exec(_compile_main_block("loop"), ctx.env)

    assert ctx.final.read_bytes() == old_video
    assert stable.read_bytes() == old_sidecar


def test_accepted_snapshots_are_isolated_per_clip(tmp_path):
    """One clip's durable recovery data must never restore another clip."""
    clip0 = tmp_path / "clip_0.mp4"
    side0 = tmp_path / "clip_0.turn_framing_plan.json"
    clip1 = tmp_path / "clip_1.mp4"
    side1 = tmp_path / "clip_1.turn_framing_plan.json"
    clip0.write_bytes(b"video-zero")
    side0.write_bytes(b"sidecar-zero")
    pf.snapshot_accepted_pair(
        job_dir=str(tmp_path), clip_final=str(clip0), stable_sidecar=str(side0))
    clip1.write_bytes(b"video-one")
    side1.write_bytes(b"sidecar-one")
    pf.snapshot_accepted_pair(
        job_dir=str(tmp_path), clip_final=str(clip1), stable_sidecar=str(side1))

    clip0.write_bytes(b"damaged-zero")
    side0.write_bytes(b"damaged-sidecar-zero")
    assert pf.restore_accepted_snapshot(
        job_dir=str(tmp_path), clip_final=str(clip0), stable_sidecar=str(side0))
    assert clip0.read_bytes() == b"video-zero"
    assert side0.read_bytes() == b"sidecar-zero"


def test_snapshot_update_is_atomic_for_video_and_sidecar(tmp_path, monkeypatch):
    """A partial snapshot update must retain the prior coherent pair."""
    final = tmp_path / "clip.mp4"
    stable = tmp_path / "clip.turn_framing_plan.json"
    final.write_bytes(b"accepted-video-a")
    stable.write_bytes(b"accepted-sidecar-a")
    pf.snapshot_accepted_pair(
        job_dir=str(tmp_path), clip_final=str(final), stable_sidecar=str(stable))

    final.write_bytes(b"accepted-video-b")
    stable.write_bytes(b"accepted-sidecar-b")
    real_copy2 = shutil.copy2

    def fail_sidecar_snapshot(src, dst, *args, **kwargs):
        if (os.path.abspath(str(src)) == os.path.abspath(str(stable))
                and Path(dst).name == "sidecar.json"):
            raise OSError("injected snapshot sidecar failure")
        return real_copy2(src, dst, *args, **kwargs)

    monkeypatch.setattr(shutil, "copy2", fail_sidecar_snapshot)
    pf.snapshot_accepted_pair(
        job_dir=str(tmp_path), clip_final=str(final), stable_sidecar=str(stable))

    final.write_bytes(b"damaged")
    stable.write_bytes(b"damaged")
    assert pf.restore_accepted_snapshot(
        job_dir=str(tmp_path), clip_final=str(final), stable_sidecar=str(stable))
    restored = (final.read_bytes(), stable.read_bytes())
    assert restored in {
        (b"accepted-video-a", b"accepted-sidecar-a"),
        (b"accepted-video-b", b"accepted-sidecar-b"),
    }


def test_archived_only_user_project_declines_rerender_and_keeps_lineage(promo_job):
    """Archived Task-01 projects pin their accepted pair before rendering."""
    job = promo_job
    assert orch._render_one_clip(**job.kwargs)
    working = job.root / "clip-project-0.json"
    project = json.loads(working.read_text(encoding="utf-8"))
    project["origin"] = "user"
    version, _ = save_project_version(
        job_dir=str(job.root), clip_index=0,
        project=validate_project(project), expected_version=1)
    archived = job.root / f"clip-project-0.v{version}.json"
    working.unlink()

    _install_changed_renderer(job)
    job.final.unlink()
    assert orch._render_one_clip(**job.kwargs)

    preserved = json.loads(archived.read_text(encoding="utf-8"))
    outcome, _, reason = pl.check_lineage_eligible(
        preserved["reframed_base"], preserved, str(job.root))
    assert outcome == "eligible", reason


# ===========================================================================
# Part 2: load-bearing mutation gates. Each disables one review-03
# correction and proves the corresponding probe then fails.
# ===========================================================================

def test_orchestrator_stage_failure_aborts_and_preserves_pair(
    promo_job, monkeypatch
):
    """Staging failure in the queued path aborts before touching finals."""
    job = promo_job
    assert orch._render_one_clip(**job.kwargs)
    stable = Path(sidecar_for(str(job.final)))
    old_sidecar = stable.read_bytes()
    # Force a re-render with no accepted video on disk; the stable sidecar
    # is the surviving half of the old pair.
    job.final.unlink()

    real_copy2 = shutil.copy2

    def fail_sidecar_backup(src, dst, *args, **kwargs):
        if os.path.abspath(str(src)) == os.path.abspath(str(stable)):
            raise OSError("injected staging failure")
        return real_copy2(src, dst, *args, **kwargs)

    monkeypatch.setattr(shutil, "copy2", fail_sidecar_backup)
    assert not orch._render_one_clip(**job.kwargs)
    assert not job.final.exists(), "no new video may be published"
    assert stable.read_bytes() == old_sidecar, "old sidecar must survive"


def test_reframe_only_stage_failure_aborts_and_preserves_pair(
    tmp_path, monkeypatch
):
    """Staging failure in the reframe-only path exits non-zero, pair kept."""
    ctx = _reframe_only_env(tmp_path)
    old_video = b"old-accepted-video" * 1000
    ctx.final.write_bytes(old_video)
    stable = Path(sidecar_for(str(ctx.final)))
    write_task12_sidecar(str(ctx.final), "accepted-old")
    old_sidecar = stable.read_bytes()

    real_copy2 = shutil.copy2

    def fail_video_backup(src, dst, *args, **kwargs):
        if os.path.abspath(str(src)) == os.path.abspath(str(ctx.final)):
            raise OSError("injected staging failure")
        return real_copy2(src, dst, *args, **kwargs)

    monkeypatch.setattr(shutil, "copy2", fail_video_backup)
    with pytest.raises(SystemExit) as exc:
        exec(_compile_main_block("reframe"), ctx.env)
    assert exc.value.code == 1
    assert ctx.final.read_bytes() == old_video
    assert stable.read_bytes() == old_sidecar


def test_mutation_shared_snapshot_dir_installs_wrong_clip(
    tmp_path, monkeypatch
):
    """Disabling per-clip keying lets one clip restore another's bytes."""
    monkeypatch.setattr(pf, "_clip_snapshot_key",
                        lambda clip_final: "shared")
    monkeypatch.setattr(pf, "_snapshot_identity_ok",
                        lambda clip_dir, clip_final: True)
    clip0 = tmp_path / "clip_0.mp4"
    side0 = tmp_path / "clip_0.turn_framing_plan.json"
    clip1 = tmp_path / "clip_1.mp4"
    side1 = tmp_path / "clip_1.turn_framing_plan.json"
    clip0.write_bytes(b"video-zero")
    side0.write_bytes(b"sidecar-zero")
    assert pf.snapshot_accepted_pair(
        job_dir=str(tmp_path), clip_final=str(clip0), stable_sidecar=str(side0))
    clip1.write_bytes(b"video-one")
    side1.write_bytes(b"sidecar-one")
    assert pf.snapshot_accepted_pair(
        job_dir=str(tmp_path), clip_final=str(clip1), stable_sidecar=str(side1))

    clip0.write_bytes(b"damaged-zero")
    side0.write_bytes(b"damaged-sidecar-zero")
    assert pf.restore_accepted_snapshot(
        job_dir=str(tmp_path), clip_final=str(clip0), stable_sidecar=str(side0))
    # The bug, reproduced: clip 0's restore installed clip 1's bytes.
    assert clip0.read_bytes() == b"video-one"
    assert side0.read_bytes() == b"sidecar-one"


def _legacy_inplace_snapshot_update(*, job_dir, clip_final, stable_sidecar):
    """Pre-review-03 behavior: overwrite the live generation in place.

    Errors swallowed (the review-02 shape); the manifest is rewritten to
    match whatever landed, so the restore-time verification does not mask
    the mixed generation this mutation is meant to demonstrate.
    """
    clip_dir = pf._clip_snapshot_dir(job_dir, clip_final)
    gen = pf._read_snapshot_current(clip_dir) or "gen-a"
    gen_dir = os.path.join(clip_dir, gen)
    os.makedirs(gen_dir, exist_ok=True)
    video_name = "video" + os.path.splitext(clip_final)[1]
    try:
        if os.path.isfile(clip_final):
            shutil.copy2(clip_final, os.path.join(gen_dir, video_name))
        if os.path.isfile(stable_sidecar):
            shutil.copy2(stable_sidecar,
                         os.path.join(gen_dir, "sidecar.json"))
    except OSError:
        pass
    manifest = {
        "clip_basename": os.path.basename(clip_final),
        "video_sha256": pf._sha256_file(os.path.join(gen_dir, video_name)),
        "sidecar_sha256": pf._sha256_file(
            os.path.join(gen_dir, "sidecar.json")),
    }
    with open(os.path.join(gen_dir, "manifest.json"),
              "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, sort_keys=True)


def test_mutation_non_atomic_snapshot_update_leaves_mixed_generation(
    tmp_path, monkeypatch
):
    """Disabling atomic generations reintroduces the mixed-pair restore."""
    final = tmp_path / "clip.mp4"
    stable = tmp_path / "clip.turn_framing_plan.json"
    final.write_bytes(b"accepted-video-a")
    stable.write_bytes(b"accepted-sidecar-a")
    assert pf.snapshot_accepted_pair(
        job_dir=str(tmp_path), clip_final=str(final),
        stable_sidecar=str(stable))

    final.write_bytes(b"accepted-video-b")
    stable.write_bytes(b"accepted-sidecar-b")
    real_copy2 = shutil.copy2

    def fail_sidecar_snapshot(src, dst, *args, **kwargs):
        if (os.path.abspath(str(src)) == os.path.abspath(str(stable))
                and Path(dst).name == "sidecar.json"):
            raise OSError("injected snapshot sidecar failure")
        return real_copy2(src, dst, *args, **kwargs)

    monkeypatch.setattr(shutil, "copy2", fail_sidecar_snapshot)
    _legacy_inplace_snapshot_update(
        job_dir=str(tmp_path), clip_final=str(final),
        stable_sidecar=str(stable))

    final.write_bytes(b"damaged")
    stable.write_bytes(b"damaged")
    assert pf.restore_accepted_snapshot(
        job_dir=str(tmp_path), clip_final=str(final),
        stable_sidecar=str(stable))
    # The bug, reproduced: the restore installs a pair that never existed.
    assert (final.read_bytes(), stable.read_bytes()) == (
        b"accepted-video-b", b"accepted-sidecar-a")


def test_mutation_working_only_guard_allows_archived_rerender(
    promo_job, monkeypatch
):
    """A working-only pre-render guard lets an archived project rerender."""
    job = promo_job
    assert orch._render_one_clip(**job.kwargs)
    working = job.root / "clip-project-0.json"
    project = json.loads(working.read_text(encoding="utf-8"))
    project["origin"] = "user"
    version, _ = save_project_version(
        job_dir=str(job.root), clip_index=0,
        project=validate_project(project), expected_version=1)
    archived = job.root / f"clip-project-0.v{version}.json"
    working.unlink()

    def _working_only_resolver(output_dir, index):
        # Pre-review-03 shape: the pre-render guard saw working files only.
        from clippyme.pipeline.project_emit import PROJECT_FILENAME_TEMPLATE

        only = os.path.join(
            output_dir, PROJECT_FILENAME_TEMPLATE.format(index=index))
        return only if os.path.isfile(only) else None

    monkeypatch.setattr(
        orch, "_resolve_project_for_preservation", _working_only_resolver)
    _install_changed_renderer(job)
    job.final.unlink()
    assert orch._render_one_clip(**job.kwargs)

    # The bug, reproduced: the rerender replaced the pixel base while the
    # sidecar publish reported skipped-preserved; the archived project's
    # lineage is now invalid.
    assert job.final.read_bytes() == VIDEO_BYTES + b"second-attempt"
    preserved = json.loads(archived.read_text(encoding="utf-8"))
    outcome, _, _ = pl.check_lineage_eligible(
        preserved["reframed_base"], preserved, str(job.root))
    assert outcome == "invalid_asset"
