"""Task 13 review-05: rollback-failure integrity gate.

A failed live-pair rollback can never be collapsed into ordinary
``False``, and its verified staging backups must survive the failure.

Part 1 promotes the coordinator's review-05 probe
(``coordinator-review5-remaining.py``, kept verbatim in the handoff dir):
inject a sidecar-install failure, then make both the atomic and the
direct-copy video rollback paths fail. The call must surface an
``OSError`` (``FramingIntegrityError``), must not restore the live pair
to something unverified, and must keep the ``.task13-live-backup-*``
staging directory.

Part 2 adds load-bearing gates: the pre-review-05 swallowing
``_restore_live_pair`` (catch-all + unconditional staging delete)
reproduces the probe failure; the new contract returns verified ``True``
only after re-checking pre-install existence and sha256 for both live
paths, and raises ``FramingIntegrityError`` -- retaining staging --
otherwise. The preserved-project route records the integrity error as
the critical outcome ``framing-recovery-integrity-failed`` and never
rerenders over the preserved base.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pytest

from clippyme.domain import project_framing as pf
from clippyme.pipeline import orchestrator as orch
from tests.pipeline.conftest import framing_impl  # noqa: F401 (fixture)
from tests.pipeline.test_sidecar_promotion import promo_job


# ===========================================================================
# Part 1: the coordinator review-05 probe, promoted verbatim.
# ===========================================================================

def test_live_restore_rollback_failure_is_surfaced_and_recovery_kept(
    tmp_path, monkeypatch
):
    """A failed rollback cannot return ordinary False and erase its backup."""
    final = tmp_path / "clip.mp4"
    stable = tmp_path / "clip.turn_framing_plan.json"
    final.write_bytes(b"accepted-video-a")
    stable.write_bytes(b"accepted-sidecar-a")
    assert pf.snapshot_accepted_pair(
        job_dir=str(tmp_path), clip_final=str(final),
        stable_sidecar=str(stable))

    # This is the live pair that must survive a failed recovery attempt.
    final.write_bytes(b"live-video-b")
    stable.write_bytes(b"live-sidecar-b")

    real_replace = os.replace
    final_replaces = 0

    def fail_install_and_video_rollback(src, dst):
        nonlocal final_replaces
        dst_abs = os.path.abspath(str(dst))
        if dst_abs == os.path.abspath(str(final)):
            final_replaces += 1
            # First replace installs snapshot video A. The second is the
            # attempted atomic rollback to live video B.
            if final_replaces >= 2:
                raise OSError("injected video rollback replace failure")
        if dst_abs == os.path.abspath(str(stable)):
            # Snapshot sidecar installation fails, starting rollback.
            raise OSError("injected sidecar install failure")
        return real_replace(src, dst)

    real_copy2 = shutil.copy2

    def fail_direct_video_rollback(src, dst, *args, **kwargs):
        # Atomic rollback above failed; make the direct-copy fallback fail too.
        if (".task13-live-backup-" in os.path.abspath(str(src))
                and os.path.abspath(str(dst)) == os.path.abspath(str(final))):
            raise OSError("injected direct video rollback failure")
        return real_copy2(src, dst, *args, **kwargs)

    monkeypatch.setattr(pf.os, "replace", fail_install_and_video_rollback)
    monkeypatch.setattr(shutil, "copy2", fail_direct_video_rollback)

    surfaced = False
    try:
        result = pf.restore_accepted_snapshot(
            job_dir=str(tmp_path), clip_final=str(final),
            stable_sidecar=str(stable))
    except OSError:
        surfaced = True
        result = False

    assert not result
    pair = (final.read_bytes(), stable.read_bytes())
    if pair != (b"live-video-b", b"live-sidecar-b"):
        assert surfaced, "rollback failure was silently collapsed into ordinary False"
        backups = list(tmp_path.glob(".task13-live-backup-*"))
        assert backups, "the only verified live-pair backup was deleted"


# ===========================================================================
# Part 2: load-bearing gates for the review-05 contract.
# ===========================================================================

def test_mutation_swallowing_rollback_fails_the_gate(tmp_path, monkeypatch):
    """The pre-review-05 _restore_live_pair reproduces the probe failure.

    Monkeypatching in the OLD swallowing behavior (catch every restore
    error, unconditionally delete the staging directory) must break the
    review-05 gate: no OSError surfaces, the live pair is mixed, and the
    staging backups are gone.
    """
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
    final_replaces = 0

    def fail_install_and_video_rollback(src, dst):
        nonlocal final_replaces
        dst_abs = os.path.abspath(str(dst))
        if dst_abs == os.path.abspath(str(final)):
            final_replaces += 1
            if final_replaces >= 2:
                raise OSError("injected video rollback replace failure")
        if dst_abs == os.path.abspath(str(stable)):
            raise OSError("injected sidecar install failure")
        return real_replace(src, dst)

    real_copy2 = shutil.copy2

    def fail_direct_video_rollback(src, dst, *args, **kwargs):
        if (".task13-live-backup-" in os.path.abspath(str(src))
                and os.path.abspath(str(dst)) == os.path.abspath(str(final))):
            raise OSError("injected direct video rollback failure")
        return real_copy2(src, dst, *args, **kwargs)

    monkeypatch.setattr(pf.os, "replace", fail_install_and_video_rollback)
    monkeypatch.setattr(shutil, "copy2", fail_direct_video_rollback)

    def _old_swallowing_restore(record):
        """The pre-review-05 implementation, inline."""
        for entry in record["paths"]:
            dst = entry["dst"]
            try:
                if entry["existed"]:
                    try:
                        pf._install_verified(entry["backup"], dst)
                    except OSError:
                        pf._install_direct(entry["backup"], dst)
                elif os.path.isfile(dst) or os.path.islink(dst):
                    os.remove(dst)
            except OSError:
                continue
        pf._remove_tree_quietly(record["dir"])

    monkeypatch.setattr(pf, "_restore_live_pair", _old_swallowing_restore)

    surfaced = False
    try:
        result = pf.restore_accepted_snapshot(
            job_dir=str(tmp_path), clip_final=str(final),
            stable_sidecar=str(stable))
    except OSError:
        surfaced = True
        result = False

    assert not result
    pair = (final.read_bytes(), stable.read_bytes())
    backups = list(tmp_path.glob(".task13-live-backup-*"))

    gate_failed = (
        pair != (b"live-video-b", b"live-sidecar-b")
        and (not surfaced or not backups)
    )
    assert gate_failed, (
        "the swallowing mutation did not reproduce the probe failure: "
        f"surfaced={surfaced} pair={pair!r} backups={backups}"
    )


def test_restore_live_pair_verified_success_returns_true_and_cleans_up(
    tmp_path,
):
    """A clean rollback returns True and discards the staging directory."""
    final = tmp_path / "clip.mp4"
    stable = tmp_path / "clip.turn_framing_plan.json"
    final.write_bytes(b"live-video-b")
    stable.write_bytes(b"live-sidecar-b")
    record = pf._stage_live_pair(
        clip_final=str(final), stable_sidecar=str(stable))
    assert record is not None
    staging = record["dir"]
    assert os.path.isdir(staging)

    # Simulate the clobbered live pair a failed install leaves behind.
    final.write_bytes(b"snapshot-video-a")
    stable.write_bytes(b"snapshot-sidecar-a")

    assert pf._restore_live_pair(record) is True
    assert (final.read_bytes(), stable.read_bytes()) == (
        b"live-video-b", b"live-sidecar-b")
    assert not os.path.exists(staging), "verified rollback must discard staging"


def test_restore_live_pair_raises_and_keeps_staging_on_verify_mismatch(
    tmp_path,
):
    """An unverifiable rollback raises FramingIntegrityError, staging kept."""
    final = tmp_path / "clip.mp4"
    stable = tmp_path / "clip.turn_framing_plan.json"
    final.write_bytes(b"live-video-b")
    stable.write_bytes(b"live-sidecar-b")
    record = pf._stage_live_pair(
        clip_final=str(final), stable_sidecar=str(stable))
    assert record is not None

    # Corrupt the verified backups so the rollback cannot be trusted.
    for entry in record["paths"]:
        if entry["existed"]:
            Path(entry["backup"]).write_bytes(b"corrupted-backup")
    final.write_bytes(b"snapshot-video-a")

    with pytest.raises(pf.FramingIntegrityError):
        pf._restore_live_pair(record)
    assert os.path.isdir(record["dir"]), \
        "staging backups must be retained on integrity failure"


def test_restore_live_pair_unverified_is_also_an_oserror(tmp_path):
    """The integrity error is a distinct FramingError AND an OSError."""
    assert issubclass(pf.FramingIntegrityError, pf.FramingError)
    assert issubclass(pf.FramingIntegrityError, OSError)

    final = tmp_path / "clip.mp4"
    stable = tmp_path / "clip.turn_framing_plan.json"
    final.write_bytes(b"live-video-b")
    stable.write_bytes(b"live-sidecar-b")
    record = pf._stage_live_pair(
        clip_final=str(final), stable_sidecar=str(stable))
    assert record is not None
    # Remove a backup entirely: the reinstall raises, so verification of
    # the recorded pre-install state cannot succeed.
    for entry in record["paths"]:
        if entry["existed"] and entry["backup"] is not None:
            os.remove(entry["backup"])
    final.write_bytes(b"snapshot-video-a")
    with pytest.raises(OSError):
        pf._restore_live_pair(record)
    assert os.path.isdir(record["dir"])


def test_integrity_error_is_critical_and_never_rerenders(promo_job,
                                                         monkeypatch):
    """The preserved-project guard records FramingIntegrityError critically.

    No success, no rerender over the preserved base, and the distinct
    ``framing-recovery-integrity-failed`` reason is recorded.
    """
    job = promo_job
    assert orch._render_one_clip(**job.kwargs)
    project_path = job.root / "clip-project-0.json"
    project = json.loads(project_path.read_text(encoding="utf-8"))
    project["origin"] = "user"
    project["version"] = 2
    project_path.write_text(json.dumps(project, sort_keys=True),
                            encoding="utf-8")

    def boom(**kwargs):
        raise pf.FramingIntegrityError("injected rollback integrity failure")

    monkeypatch.setattr(pf, "restore_accepted_snapshot", boom)
    # Unlink the final so checkpoint resume cannot short-circuit past
    # the preserved-project guard (as in the review-04 mutation tests).
    job.final.unlink()

    assert not orch._render_one_clip(**job.kwargs), \
        "an integrity failure must not report success"
    assert not job.final.exists(), "no rerender over the preserved base"

    metadata = json.loads(
        (job.root / "metadata.json").read_text(encoding="utf-8"))
    qa = metadata["shorts"][0]["qa"]
    assert qa["ok"] is False and qa["critical"] is True
    assert "framing-recovery-integrity-failed" in qa["issues"]
