"""Task 13 review-04 corrections: strict snapshot validation, transactional
live restoration, and a preserved-project guard that checks the restore result.

Part 1 promotes the coordinator's three review-04 recovery-integrity probes
(``coordinator-review4-remaining.py``, left untouched in the handoff dir):
a corrupt manifest must not report recovery available, a failed restore
must not mark a preserved clip complete, and live restoration must never
expose a mixed video+sidecar pair.

Part 2 adds load-bearing mutation gates: the pre-review-04 weak predicate
accepts a corrupt manifest (finding 1), discarding the restore result
reproduces the false success (finding 2), and disabling live-pair rollback
reintroduces the mixed pair (finding 3).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from clippyme.domain import project_framing as pf
from clippyme.pipeline import orchestrator as orch
from tests.pipeline.conftest import framing_impl  # noqa: F401 (fixture)
from tests.pipeline.task13_helpers import sidecar_for
from tests.pipeline.test_sidecar_promotion import promo_job


def _active_generation(job_dir: Path, clip_final: Path) -> Path:
    clip_dir = Path(pf._clip_snapshot_dir(str(job_dir), str(clip_final)))
    current = pf._read_snapshot_current(str(clip_dir))
    assert current is not None
    return clip_dir / current


# ===========================================================================
# Part 1: the three coordinator review-04 probes, promoted verbatim.
# ===========================================================================

def test_snapshot_exists_requires_a_verified_generation(tmp_path):
    """The pre-render predicate must reject a corrupt manifest."""
    final = tmp_path / "clip.mp4"
    stable = tmp_path / "clip.turn_framing_plan.json"
    final.write_bytes(b"accepted-video")
    stable.write_bytes(b"accepted-sidecar")
    assert pf.snapshot_accepted_pair(
        job_dir=str(tmp_path), clip_final=str(final),
        stable_sidecar=str(stable))
    generation = _active_generation(tmp_path, final)
    (generation / "manifest.json").write_text("{}", encoding="utf-8")

    assert not pf.accepted_snapshot_exists(
        job_dir=str(tmp_path), clip_final=str(final))


def test_corrupt_snapshot_cannot_report_preserved_clip_complete(promo_job):
    """A failed restore must not mark a missing preserved base complete."""
    job = promo_job
    assert orch._render_one_clip(**job.kwargs)
    project_path = job.root / "clip-project-0.json"
    project = json.loads(project_path.read_text(encoding="utf-8"))
    project["origin"] = "user"
    project["version"] = 2
    project_path.write_text(json.dumps(project, sort_keys=True), encoding="utf-8")

    generation = _active_generation(job.root, job.final)
    (generation / "manifest.json").write_text("{}", encoding="utf-8")
    job.final.unlink()

    accepted = orch._render_one_clip(**job.kwargs)
    assert not accepted, "invalid recovery data cannot produce a successful clip"
    assert not job.final.exists(), "no verified preserved base was restored"


def test_restore_failure_never_leaves_a_mixed_live_pair(tmp_path, monkeypatch):
    """Video and sidecar recovery must commit or roll back together."""
    final = tmp_path / "clip.mp4"
    stable = tmp_path / "clip.turn_framing_plan.json"
    final.write_bytes(b"accepted-video-a")
    stable.write_bytes(b"accepted-sidecar-a")
    assert pf.snapshot_accepted_pair(
        job_dir=str(tmp_path), clip_final=str(final),
        stable_sidecar=str(stable))

    final.write_bytes(b"live-video-b")
    stable.write_bytes(b"live-sidecar-b")
    real_replace = os.replace

    def fail_sidecar_install(src, dst):
        if os.path.abspath(str(dst)) == os.path.abspath(str(stable)):
            raise OSError("injected live sidecar install failure")
        return real_replace(src, dst)

    monkeypatch.setattr(pf.os, "replace", fail_sidecar_install)
    assert not pf.restore_accepted_snapshot(
        job_dir=str(tmp_path), clip_final=str(final),
        stable_sidecar=str(stable))

    pair = (final.read_bytes(), stable.read_bytes())
    assert pair in {
        (b"accepted-video-a", b"accepted-sidecar-a"),
        (b"live-video-b", b"live-sidecar-b"),
    }, f"restore exposed a mixed pair: {pair!r}"


# ===========================================================================
# Part 2: load-bearing mutation gates.
# ===========================================================================

def test_mutation_weak_predicate_accepts_corrupt_manifest(tmp_path):
    """Disabling strict generation validation reintroduces finding 1."""
    final = tmp_path / "clip.mp4"
    stable = tmp_path / "clip.turn_framing_plan.json"
    final.write_bytes(b"accepted-video")
    stable.write_bytes(b"accepted-sidecar")
    assert pf.snapshot_accepted_pair(
        job_dir=str(tmp_path), clip_final=str(final),
        stable_sidecar=str(stable))
    generation = _active_generation(tmp_path, final)
    (generation / "manifest.json").write_text("{}", encoding="utf-8")

    def _legacy_weak_exists(*, job_dir, clip_final):
        # Pre-review-04 shape: ownership + pointer + video file only --
        # no manifest schema or hash validation.
        clip_dir = pf._clip_snapshot_dir(job_dir, clip_final)
        if not pf._snapshot_identity_ok(clip_dir, clip_final):
            return False
        gen = pf._read_snapshot_current(clip_dir)
        if gen is None:
            return False
        video_snap = os.path.join(
            clip_dir, gen, pf._snapshot_video_name(clip_final))
        return os.path.isfile(video_snap)

    # The bug, reproduced: a corrupt manifest reports recovery available.
    assert _legacy_weak_exists(
        job_dir=str(tmp_path), clip_final=str(final))
    # With the fix, the same corrupt manifest is rejected.
    assert not pf.accepted_snapshot_exists(
        job_dir=str(tmp_path), clip_final=str(final))
    assert pf._validate_snapshot_generation(
        clip_dir=str(_active_generation(tmp_path, final).parent),
        clip_final=str(final)) is None


def test_mutation_ignored_restore_failure_must_not_report_success(promo_job,
                                                                  monkeypatch):
    """Discarding the restore result reproduces the false success of finding 2."""
    job = promo_job
    assert orch._render_one_clip(**job.kwargs)
    project_path = job.root / "clip-project-0.json"
    project = json.loads(project_path.read_text(encoding="utf-8"))
    project["origin"] = "user"
    project["version"] = 2
    project_path.write_text(json.dumps(project, sort_keys=True), encoding="utf-8")

    # Recovery is broken (no valid snapshot can restore); the guard must
    # check the result instead of reporting success. The pre-review-04
    # caller shape discarded the boolean and returned True here.
    monkeypatch.setattr(pf, "restore_accepted_snapshot",
                        lambda **kwargs: False)
    # Unlink the final so checkpoint resume cannot short-circuit past
    # the preserved-project guard (as in the promoted probe above).
    job.final.unlink()
    assert not orch._render_one_clip(**job.kwargs), \
        "the preserved-project guard must check the restore result"
    assert not job.final.exists(), "no verified preserved base was restored"


def test_mutation_no_rollback_leaves_mixed_live_pair(tmp_path, monkeypatch):
    """Disabling live-pair rollback reintroduces the mixed pair of finding 3."""
    final = tmp_path / "clip.mp4"
    stable = tmp_path / "clip.turn_framing_plan.json"
    final.write_bytes(b"accepted-video-a")
    stable.write_bytes(b"accepted-sidecar-a")
    assert pf.snapshot_accepted_pair(
        job_dir=str(tmp_path), clip_final=str(final),
        stable_sidecar=str(stable))

    final.write_bytes(b"live-video-b")
    stable.write_bytes(b"live-sidecar-b")
    real_replace = os.replace

    def fail_sidecar_install(src, dst):
        if os.path.abspath(str(dst)) == os.path.abspath(str(stable)):
            raise OSError("injected live sidecar install failure")
        return real_replace(src, dst)

    monkeypatch.setattr(pf.os, "replace", fail_sidecar_install)
    # Disable the transactional rollback: the mixed pair reappears.
    monkeypatch.setattr(pf, "_restore_live_pair", lambda record: None)
    assert not pf.restore_accepted_snapshot(
        job_dir=str(tmp_path), clip_final=str(final),
        stable_sidecar=str(stable))
    # The bug, reproduced: snapshot video A beside live sidecar B.
    assert (final.read_bytes(), stable.read_bytes()) == (
        b"accepted-video-a", b"live-sidecar-b")
