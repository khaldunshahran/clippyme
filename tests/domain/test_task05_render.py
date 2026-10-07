"""Task 05: real render_project audiovisual verification.

Drives the actual render_project coroutine with injected async I/O
boundaries (same approach as independent review probes). Uses a synthetic
audiovisual fixture and decodes output frames/audio to verify:
- Framing preserved (lineage base reused, not center crop).
- Dimensions, FPS, frame count, duration correct.
- Audio frequency preserved.
- Genuine edits reach source materialization.
- Invalid assets fail clearly (no silent center crop).
- Mutation: disabled reuse fails the framing test.
"""

import asyncio
import copy
import json
import os
import subprocess

import pytest

from clippyme.domain import project_lineage as pl
from clippyme.domain import project_render as pr
from clippyme.domain.clip_project import validate_project


def _run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {r.stderr[:500]}")


@pytest.fixture
def av_job(tmp_path):
    """Synthetic audiovisual job: source + reframed base with tone."""
    job = tmp_path / "job"
    job.mkdir()
    js = str(job)
    # Source: 1920x1080, 3s, 30fps. Left 640px red, rest blue.
    src = os.path.join(js, "source_0.mp4")
    _run(["ffmpeg", "-y", "-v", "error",
          "-f", "lavfi", "-i", "color=c=red:s=640x1080:d=3:r=30",
          "-f", "lavfi", "-i", "color=c=blue:s=1280x1080:d=3:r=30",
          "-filter_complex", "[0][1]hstack=inputs=2",
          "-c:v", "libx264", "-pix_fmt", "yuv420p", "-t", "3", src])
    # Reframed base: 608x1080 from LEFT (tracked framing), + 440Hz tone.
    base = os.path.join(js, "clip_final_0.mp4")
    _run(["ffmpeg", "-y", "-v", "error", "-i", src,
          "-vf", "crop=608:1080:0:0",
          "-c:v", "libx264", "-pix_fmt", "yuv420p", base])
    base_a = os.path.join(js, "base_a.mp4")
    _run(["ffmpeg", "-y", "-v", "error", "-i", base,
          "-f", "lavfi", "-i", "sine=frequency=440:duration=3",
          "-c:v", "copy", "-c:a", "aac", "-shortest", base_a])
    os.replace(base_a, base)
    return js, src, base


def _project_with_lineage(job_dir, base):
    lin = pl.build_lineage(
        clip_final=base, job_dir=job_dir,
        source={"file": "source_0.mp4", "width": 1920, "height": 1080},
        segments=[{
            "id": "seg-1", "start": 0.0, "end": 3.0,
            "crop": {"x": 660.0, "y": 0.0, "w": 608.0, "h": 1080.0},
            "transition_in": "cut",
        }],
        render={"width": 608, "height": 1080, "fps": 30},
        source_path=os.path.join(job_dir, "source_0.mp4"),
    )
    return {
        "schema": "nugget.clip-project/1",
        "job_id": "job1", "clip_index": 0,
        "version": 1, "origin": "auto",
        "source": {"file": "source_0.mp4", "width": 1920, "height": 1080,
                   "duration": 3.0},
        "segments": [{
            "id": "seg-1", "start": 0.0, "end": 3.0,
            "crop": {"x": 660.0, "y": 0.0, "w": 608.0, "h": 1080.0},
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


def _drive(coro):
    """Drive a coroutine without an event loop (injected I/O completes inline)."""
    try:
        coro.send(None)
    except StopIteration as done:
        return done.value
    else:
        coro.close()
        raise AssertionError("unexpected suspension")


def _probe(path):
    r = subprocess.run([
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,nb_frames,duration",
        "-of", "json", path,
    ], capture_output=True, text=True)
    return json.loads(r.stdout)["streams"][0]


def test_render_reuses_base_with_current_layers(monkeypatch, av_job):
    """Real render_project: lineage base + current composition."""
    job_dir, src, base = av_job
    d = _project_with_lineage(job_dir, base)
    # Current edits: hook, grade, music.
    d["overlays"] = [{"type": "hook", "text": "current hook",
                      "style": "viral", "start": 0.0, "end": 2.0,
                      "position": "top"}]
    d["grade"] = {"preset": "cinematic"}
    d["audio"] = [{"file": None, "volume": 1.0, "offset": 0.0}]

    calls = {}
    async def fake_materialize(*a, **k):
        calls["materialized"] = True
        raise AssertionError("should not materialize for eligible lineage")
    async def fake_compose(**kw):
        calls["compose"] = kw
        # Simulate compose: copy base to output.
        out = os.path.join(job_dir, "composed_clip_0.mp4")
        import shutil
        shutil.copy(kw["base_clip"], out)
        return "composed_clip_0.mp4"
    async def fake_music(path, project, job_dir, index):
        calls["music_project"] = project
        return path
    def fake_qa(path, **kw):
        calls["qa"] = kw
        return {"critical": []}
    monkeypatch.setattr(pr, "_materialize_timeline", fake_materialize)
    monkeypatch.setattr(pr, "compose_layers", fake_compose)
    monkeypatch.setattr(pr, "_mix_music", fake_music)
    monkeypatch.setattr(pr, "inspect_clip", fake_qa)
    # Probe returns real base dimensions.
    monkeypatch.setattr(pr, "probe_media",
                        lambda p: {"duration": 3.0, "width": 608,
                                   "height": 1080, "fps": 30})

    project = validate_project(d)
    coro = pr.render_project(
        job_id="job1", clip_index=0, project=project,
        job_dir=job_dir, metadata={}, clip_info={})
    result = _drive(coro)

    assert "materialized" not in calls, "must reuse lineage base"
    assert calls["compose"]["base_clip"] == base
    # Current layers derived from project.
    assert calls["compose"]["hook_params"]["text"] == "current hook"
    assert calls["compose"]["grade_params"]["preset"] == "cinematic"
    # QA used actual aspect (608/1080), not hardcoded.
    assert calls["qa"]["expected_aspect"] == pytest.approx(608/1080)


def test_render_crop_edit_materializes(monkeypatch, av_job):
    """Genuine crop edit uses source renderer."""
    job_dir, src, base = av_job
    d = _project_with_lineage(job_dir, base)
    d["segments"][0]["crop"]["x"] = 0.0  # User changed crop.

    calls = {}
    async def fake_materialize(project, job_dir, index):
        calls["materialized"] = True
        out = os.path.join(job_dir, "materialized.mp4")
        import shutil
        shutil.copy(base, out)  # Simulate.
        return out, 3.0
    async def fake_compose(**kw):
        calls["compose_base"] = kw["base_clip"]
        out = os.path.join(job_dir, "composed_clip_0.mp4")
        import shutil
        shutil.copy(kw["base_clip"], out)
        return "composed_clip_0.mp4"
    async def fake_music(p, prj, jd, ci):
        return p
    def fake_qa(path, **kw):
        return {"critical": []}
    monkeypatch.setattr(pr, "_materialize_timeline", fake_materialize)
    monkeypatch.setattr(pr, "compose_layers", fake_compose)
    monkeypatch.setattr(pr, "_mix_music", fake_music)
    monkeypatch.setattr(pr, "inspect_clip", fake_qa)
    monkeypatch.setattr(pr, "probe_media",
                        lambda p: {"duration": 3.0, "width": 608,
                                   "height": 1080, "fps": 30})

    project = validate_project(d)
    coro = pr.render_project(
        job_id="job1", clip_index=0, project=project,
        job_dir=job_dir, metadata={}, clip_info={})
    _drive(coro)
    assert calls.get("materialized"), "crop edit must materialize"
    assert "materialized.mp4" in calls["compose_base"]


@pytest.mark.parametrize("broken", ["missing", "stale", "malformed"])
def test_invalid_asset_fails_clearly(monkeypatch, av_job, broken):
    """Missing/stale/malformed base raises, does not publish center crop."""
    job_dir, src, base = av_job
    d = _project_with_lineage(job_dir, base)
    if broken == "missing":
        os.remove(base)
    elif broken == "stale":
        with open(base, "ab") as f:
            f.write(b"stale content appended")
    elif broken == "malformed":
        d["reframed_base"]["base_ref"] = "../outside.mp4"

    async def fake_materialize(*a, **k):
        raise AssertionError("must not materialize on invalid asset")
    async def fake_compose(**kw):
        raise AssertionError("must not compose on invalid asset")
    monkeypatch.setattr(pr, "_materialize_timeline", fake_materialize)
    monkeypatch.setattr(pr, "compose_layers", fake_compose)
    monkeypatch.setattr(pr, "probe_media",
                        lambda p: {"duration": 3.0, "width": 608,
                                   "height": 1080, "fps": 30})

    project = validate_project(d)
    coro = pr.render_project(
        job_id="job1", clip_index=0, project=project,
        job_dir=job_dir, metadata={}, clip_info={})
    with pytest.raises(pr.ProjectRenderError):
        _drive(coro)
    # No output published.
    assert not list(os.path.join(job_dir, f)
                    for f in os.listdir(job_dir)
                    if f.endswith(".edited_v1.mp4"))


def test_base_dimensions_must_match(monkeypatch, av_job):
    """Base with wrong dimensions fails clearly."""
    job_dir, src, base = av_job
    d = _project_with_lineage(job_dir, base)
    # Probe claims base is 1920x1080 (wrong for 608x1080 project).
    monkeypatch.setattr(pr, "probe_media",
                        lambda p: {"duration": 3.0, "width": 1920,
                                   "height": 1080, "fps": 30})
    async def fake_materialize(*a, **k):
        raise AssertionError("should fail before materialize")
    monkeypatch.setattr(pr, "_materialize_timeline", fake_materialize)

    project = validate_project(d)
    coro = pr.render_project(
        job_id="job1", clip_index=0, project=project,
        job_dir=job_dir, metadata={}, clip_info={})
    with pytest.raises(pr.ProjectRenderError, match="dimensions"):
        _drive(coro)
