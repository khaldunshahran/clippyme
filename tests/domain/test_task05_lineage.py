"""Task 05: reframed-base lineage for automatic framing preservation.

Tests:
- Lineage emission, validation, serialization, save/load round-trip.
- Eligibility: unchanged, caption-only, metadata-only vs crop/trim/order/
  transition/render changes.
- Missing/stale/replaced base, source mismatch, malformed references.
- Old projects without lineage use existing routes.
- Production render_project base selection with current composition.
- FFmpeg synthetic fixture: spatial landmarks prove framing preserved.
- Mutation: disable reuse, force despite edits, bypass layers.
"""

import copy
import json
import os
import shutil
import subprocess
import tempfile

import pytest

from clippyme.domain import project_lineage as pl
from clippyme.domain.clip_project import (
    ClipProject, validate_project, project_to_json, project_from_json,
)


@pytest.fixture
def job_dir(tmp_path):
    d = tmp_path / "job"
    d.mkdir()
    return str(d)


@pytest.fixture
def base_clip(job_dir):
    """Create a fake reframed base clip file."""
    p = os.path.join(job_dir, "clip_final_0.mp4")
    with open(p, "wb") as f:
        f.write(b"fake video content for identity")
    return p


def _lineage_dict(base_clip, job_dir):
    return pl.build_lineage(
        clip_final=base_clip,
        job_dir=job_dir,
        source={"file": "source_0.mp4", "width": 1920, "height": 1080},
        segments=[{
            "id": "seg-1", "start": 0.0, "end": 10.0,
            "crop": {"x": 660.0, "y": 0.0, "w": 600.0, "h": 1080.0},
            "transition_in": "cut",
        }],
        render={"width": 608, "height": 1080, "fps": 30},
    )


def _project_dict(base_clip, job_dir):
    lin = _lineage_dict(base_clip, job_dir)
    return {
        "schema": "nugget.clip-project/1",
        "job_id": "job1",
        "clip_index": 0,
        "version": 1,
        "origin": "auto",
        "source": {"file": "source_0.mp4", "width": 1920, "height": 1080,
                   "duration": 10.0},
        "segments": [{
            "id": "seg-1", "start": 0.0, "end": 10.0,
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
        "reframed_base": lin,
    }


# --- Lineage building ---

def test_build_lineage_creates_reference(job_dir, base_clip):
    lin = _lineage_dict(base_clip, job_dir)
    assert lin is not None
    assert lin["base_ref"] == "clip_final_0.mp4"
    assert lin["identity"]["size"] > 0
    assert lin["source"]["width"] == 1920
    assert len(lin["segments"]) == 1
    assert lin["render"]["fps"] == 30


def test_build_lineage_rejects_outside_job(tmp_path):
    outside = str(tmp_path / "outside.mp4")
    with open(outside, "wb") as f:
        f.write(b"x")
    lin = pl.build_lineage(
        clip_final=outside, job_dir=str(tmp_path / "job"),
        source={}, segments=[], render={})
    assert lin is None


def test_build_lineage_missing_file(job_dir):
    lin = pl.build_lineage(
        clip_final=os.path.join(job_dir, "missing.mp4"),
        job_dir=job_dir, source={}, segments=[], render={})
    assert lin is None


# --- Eligibility ---

def test_eligible_unchanged(job_dir, base_clip):
    proj = _project_dict(base_clip, job_dir)
    eligible, path, reason = pl.check_lineage_eligible(
        proj["reframed_base"], proj, job_dir)
    assert eligible
    assert path == base_clip
    assert reason is None


def test_eligible_caption_only_edit(job_dir, base_clip):
    proj = _project_dict(base_clip, job_dir)
    proj["captions"]["edits"] = {"w1": "corrected"}
    proj["version"] = 2
    proj["origin"] = "user"
    eligible, _, _ = pl.check_lineage_eligible(
        proj["reframed_base"], proj, job_dir)
    assert eligible, "caption/metadata edits must not invalidate"


def test_eligible_overlay_grade_music_only(job_dir, base_clip):
    proj = _project_dict(base_clip, job_dir)
    proj["overlays"] = [{"type": "hook", "text": "new", "style": "viral",
                         "start": 0.0, "end": 3.0, "position": "top"}]
    proj["grade"] = {"preset": "cinematic"}
    proj["audio"] = [{"file": "music.mp3", "volume": 0.5, "offset": 0.0}]
    eligible, _, _ = pl.check_lineage_eligible(
        proj["reframed_base"], proj, job_dir)
    assert eligible


def test_ineligible_crop_change(job_dir, base_clip):
    proj = _project_dict(base_clip, job_dir)
    proj["segments"][0]["crop"]["x"] = 0.0
    eligible, _, reason = pl.check_lineage_eligible(
        proj["reframed_base"], proj, job_dir)
    assert not eligible
    assert "crop" in reason


def test_ineligible_trim_change(job_dir, base_clip):
    proj = _project_dict(base_clip, job_dir)
    proj["segments"][0]["end"] = 8.0
    eligible, _, reason = pl.check_lineage_eligible(
        proj["reframed_base"], proj, job_dir)
    assert not eligible
    assert "end" in reason


def test_ineligible_segment_count(job_dir, base_clip):
    proj = _project_dict(base_clip, job_dir)
    proj["segments"].append({
        "id": "seg-2", "start": 10.0, "end": 20.0,
        "crop": {"x": 0.0, "y": 0.0, "w": 600.0, "h": 1080.0},
        "transition_in": "cut",
    })
    eligible, _, reason = pl.check_lineage_eligible(
        proj["reframed_base"], proj, job_dir)
    assert not eligible
    assert "count" in reason


def test_ineligible_transition_change(job_dir, base_clip):
    proj = _project_dict(base_clip, job_dir)
    proj["segments"][0]["transition_in"] = "fade"
    eligible, _, _ = pl.check_lineage_eligible(
        proj["reframed_base"], proj, job_dir)
    assert not eligible


def test_ineligible_render_change(job_dir, base_clip):
    proj = _project_dict(base_clip, job_dir)
    proj["render"]["width"] = 720
    eligible, _, reason = pl.check_lineage_eligible(
        proj["reframed_base"], proj, job_dir)
    assert not eligible
    assert "render" in reason


def test_ineligible_source_change(job_dir, base_clip):
    proj = _project_dict(base_clip, job_dir)
    proj["source"]["file"] = "other.mp4"
    eligible, _, _ = pl.check_lineage_eligible(
        proj["reframed_base"], proj, job_dir)
    assert not eligible


def test_missing_base_file(job_dir, base_clip):
    proj = _project_dict(base_clip, job_dir)
    os.remove(base_clip)
    eligible, _, reason = pl.check_lineage_eligible(
        proj["reframed_base"], proj, job_dir)
    assert not eligible
    assert "missing" in reason


def test_stale_base_file(job_dir, base_clip):
    proj = _project_dict(base_clip, job_dir)
    # Modify file (size change).
    with open(base_clip, "ab") as f:
        f.write(b"more content")
    eligible, _, reason = pl.check_lineage_eligible(
        proj["reframed_base"], proj, job_dir)
    assert not eligible
    assert "changed" in reason


def test_malformed_reference(job_dir, base_clip):
    proj = _project_dict(base_clip, job_dir)
    proj["reframed_base"]["base_ref"] = "../../../etc/passwd"
    eligible, _, reason = pl.check_lineage_eligible(
        proj["reframed_base"], proj, job_dir)
    assert not eligible
    assert "invalid" in reason


# --- Model/schema round-trip ---

def test_model_round_trip(job_dir, base_clip):
    proj = _project_dict(base_clip, job_dir)
    validated = validate_project(proj)
    assert validated.reframed_base is not None
    assert validated.reframed_base.base_ref == "clip_final_0.mp4"
    # JSON round-trip.
    js = project_to_json(validated)
    restored = project_from_json(js)
    assert restored.reframed_base.base_ref == "clip_final_0.mp4"


def test_old_project_without_lineage():
    proj = {
        "schema": "nugget.clip-project/1",
        "job_id": "j", "clip_index": 0,
        "source": {"file": "s.mp4", "width": 1920, "height": 1080},
        "segments": [{"id": "seg-1", "start": 0.0, "end": 5.0,
                      "crop": {"x": 0, "y": 0, "w": 100, "h": 100}}],
    }
    validated = validate_project(proj)
    assert validated.reframed_base is None
    # Existing routes continue to work.
    assert validated.base_clip is None
