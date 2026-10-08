"""Task 13: framing in the Task 05 lineage baseline (contract tests).

CONTRACT NOTE: these tests target the AGREED interface for the PIPELINE
worker's ``check_lineage_eligible`` edit, which has not landed yet:

- ``check_lineage_eligible(lineage, project, job_dir)`` keeps returning
  ``(outcome, base_path, reason)``;
- the lineage dict may carry ``"framing"``: the
  ``framing_baseline_identity`` subset recorded at emission;
- the project dict may carry ``"framing"``: the full ProjectFraming
  field built by ``build_project_framing``;
- unchanged framing baseline -> ``"eligible"`` (base reuse);
- changed framing baseline -> ``"framing_edit"`` (fail-clear, never a
  stale-base reuse and never a silent static crop);
- malformed project framing identity -> ``"invalid_asset"``.

These tests FAIL until the pipeline worker lands that edit; that is
expected and recorded in the Task 13 handoff.
"""

import copy
import json
import os

import pytest

from clippyme.domain import project_framing as pf
from clippyme.domain import project_lineage as pl
from clippyme.pipeline.framing_plan import plan_from_dict, plan_to_dict


def _raw_plan(**over):
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


@pytest.fixture
def case(tmp_path):
    """A job dir with base clip, source, stable sidecar, lineage, project."""
    job = tmp_path / "job"
    job.mkdir()
    job_dir = str(job)

    base = os.path.join(job_dir, "clip_final_0.mp4")
    with open(base, "wb") as fh:
        fh.write(b"\x00" * 4096)
    source = os.path.join(job_dir, "source_0.mp4")
    with open(source, "wb") as fh:
        fh.write(b"\x01" * 4096)

    ref = "clip_final_0.turn_framing_plan.json"
    with open(os.path.join(job_dir, ref), "w", encoding="utf-8") as fh:
        json.dump(_raw_plan(), fh)
    framing = pf.build_project_framing(
        job_dir=job_dir, clip_final="clip_final_0.mp4",
        render_w=608, render_h=1080, render_fps=30.0, clip_frame_count=300)
    assert framing is not None

    project = {
        "source": {"file": "source_0.mp4", "width": 1920, "height": 1080},
        "segments": [
            {"id": "seg0", "start": 0.0, "end": 10.0,
             "crop": {"x": 0.0, "y": 0.0, "w": 608.0, "h": 1080.0},
             "transition_in": "cut"},
        ],
        "render": {"width": 608, "height": 1080, "fps": 30},
        "framing": framing,
    }
    lineage = pl.build_lineage(
        clip_final=base,
        job_dir=job_dir,
        source=project["source"],
        segments=project["segments"],
        render=project["render"],
        source_path=source,
    )
    assert lineage is not None
    # The pipeline worker's emission records the framing baseline here.
    lineage["framing"] = pf.framing_baseline_identity(framing)
    return job_dir, lineage, project


def test_unchanged_framing_stays_eligible(case):
    job_dir, lineage, project = case
    outcome, base_path, reason = pl.check_lineage_eligible(
        lineage, project, job_dir)
    assert outcome == "eligible"
    assert base_path is not None


def test_changed_framing_identity_is_framing_edit(case):
    job_dir, lineage, project = case
    project = copy.deepcopy(project)
    # Simulate a trajectory edit: same sidecar file, different timeline.
    project["framing"]["plan"]["timeline_digest"] = "0" * 64
    outcome, base_path, reason = pl.check_lineage_eligible(
        lineage, project, job_dir)
    assert outcome == "framing_edit"
    assert base_path is None
    assert reason and pf.REASON_TRAJECTORY_EDIT in reason


def test_malformed_project_framing_is_invalid_asset(case):
    job_dir, lineage, project = case
    project = copy.deepcopy(project)
    project["framing"] = {"plan": {"plan_ref": "x"}}  # identity malformed
    outcome, base_path, reason = pl.check_lineage_eligible(
        lineage, project, job_dir)
    assert outcome == "invalid_asset"
    assert base_path is None


def test_no_framing_baseline_old_behavior(case):
    # Lineage without a framing baseline (pre-Task-13 emission) keeps the
    # old outcome: eligibility ignores framing entirely.
    job_dir, lineage, project = case
    del lineage["framing"]
    del project["framing"]
    outcome, base_path, reason = pl.check_lineage_eligible(
        lineage, project, job_dir)
    assert outcome == "eligible"
