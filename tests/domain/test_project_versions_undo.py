"""Copilot-edit -> undo version-token flow (follow-up 7).

Simulates what the editor's serialized mutation queue does at the domain
level: apply a copilot-style patch through the versioned save path with
``expected_version``, then undo by saving the prior snapshot; assert the
version token advances and a stale token is rejected with ConflictError
(409 at the API layer).
"""
import json
import os

import pytest

from clippyme.domain.clip_project import validate_project
from clippyme.domain.project_render import (
    latest_project_version,
    save_project_version,
)
from clippyme.domain.errors import ConflictError


def _project_dict(**over):
    d = {
        "schema": "nugget.clip-project/1",
        "version": 1,
        "origin": "auto",
        "job_id": "job-1",
        "clip_index": 0,
        "source": {"file": "source_x.mp4", "width": 1920, "height": 1080},
        "segments": [{
            "id": "seg-1", "start": 0.0, "end": 10.0,
            "crop": {"x": 0.0, "y": 0.0, "w": 608.0, "h": 1080.0},
        }],
        "captions": {"words": [
            {"id": "w0", "w": "hello", "start": 0.5, "end": 0.9},
        ]},
    }
    d.update(over)
    return d


def _seed_working_copy(job_dir, project_dict):
    # Pipeline-created working copy (no version map yet).
    path = os.path.join(job_dir, "clip-project-0.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(project_dict, f)
    return path


def test_copilot_edit_then_undo_version_token(tmp_path):
    job_dir = str(tmp_path)
    v1 = _project_dict()
    _seed_working_copy(job_dir, v1)
    assert latest_project_version(job_dir, 0) == 1  # working-copy fallback

    # 1. Copilot-style edit through the versioned save path.
    edited = validate_project(_project_dict())
    edited.captions.words[0].w = "goodbye"
    ver2, saved2 = save_project_version(
        job_dir=job_dir, clip_index=0, project=edited, expected_version=1)
    assert ver2 == 2
    assert saved2.captions.words[0].w == "goodbye"
    assert latest_project_version(job_dir, 0) == 2

    # 2. Undo: save the v1 snapshot with the current token.
    undo_proj = validate_project(v1)
    ver3, saved3 = save_project_version(
        job_dir=job_dir, clip_index=0, project=undo_proj, expected_version=2)
    assert ver3 == 3
    assert saved3.captions.words[0].w == "hello"  # restored
    assert latest_project_version(job_dir, 0) == 3

    # 3. Stale token rejected.
    with pytest.raises(ConflictError):
        save_project_version(
            job_dir=job_dir, clip_index=0,
            project=validate_project(_project_dict()),
            expected_version=1)
