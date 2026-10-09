"""Task 01 reference regressions for production caller control flow.

Media boundary/content tests live separately. These tests inject the media
probe, encoder, and CV renderer while retaining the real provenance, metadata,
project emission, and publication decisions. The direct CLI's exact per-clip
loop is compiled from its AST so the unavailable CV stack is not imported.
"""
import argparse
import ast
import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from clippyme.domain.project_render import get_project_path, latest_project_version
from clippyme.pipeline import orchestrator as orch
from clippyme.pipeline import project_emit
from clippyme.domain import project_framing as _task13_project_framing
from clippyme.pipeline import run_ops as ops


SOURCE_BYTES = b"source media" * 2000
OLD_SLICE_BYTES = b"previous source slice" * 2000
OLD_FINAL_BYTES = b"previous finished clip" * 2000
NEW_SLICE_BYTES = b"newly encoded source slice" * 2000
NEW_FINAL_BYTES = b"newly rendered finished clip" * 2000


def _project(**kwargs):
    """A valid project document, matching the actual emitter's inputs/schema."""
    return {
        "schema": "nugget.clip-project/1", "version": 1, "origin": "auto",
        "job_id": kwargs.get("job_id", "test"), "clip_index": kwargs.get("index", 0),
        "source": {"file": os.path.basename(kwargs.get("clip_source", "source_clip.mp4")),
                   "width": 1920, "height": 1080, "duration": 2.0},
        "segments": [{"id": "seg-1", "start": 0.0, "end": 2.0,
                      "crop": {"x": 656.0, "y": 0.0, "w": 608.0, "h": 1080.0},
                      "transition_in": "cut"}],
        "captions": {"style": "classic_white", "position": "bottom",
                     "words": [{"id": "w0", "w": "original", "start": 0.1, "end": 0.3}],
                     "edits": {}},
        "overlays": [], "audio": [{"file": None, "volume": 1.0, "offset": 0.0}],
        "grade": {"preset": "default"},
        "render": {"width": 608, "height": 1080, "fps": 30, "encoder": "libx264"},
    }


def _write_json(path, value):
    Path(path).write_text(json.dumps(value), encoding="utf8")


def _downgrade_to_v3(slice_path):
    sidecar = Path(ops.slice_provenance_path(str(slice_path)))
    record = json.loads(sidecar.read_text(encoding="utf8"))
    record["version"] = 3
    _write_json(sidecar, record)


class _State:
    def mark_clip(self, *args):
        pass

    def start(self, *args, **kwargs):
        pass


@pytest.fixture
def queued_job(tmp_path, monkeypatch):
    root = tmp_path / "job"
    root.mkdir()
    source = root / "input.mp4"
    source.write_bytes(SOURCE_BYTES)
    sl = root / "source_clip.mp4"
    sl.write_bytes(OLD_SLICE_BYTES)
    ops.write_slice_provenance(str(sl), start=0.0, end=2.0,
                               source_fp=ops.source_fingerprint(str(source)))
    control = SimpleNamespace(cut_calls=[], render_calls=[], mutate_source=False)

    def encode(command, **kwargs):
        control.cut_calls.append(command)
        Path(command[-1]).write_bytes(NEW_SLICE_BYTES)
        if control.mutate_source:
            source.write_bytes(SOURCE_BYTES + b" changed during encoder run")
        return SimpleNamespace(returncode=0, stderr=b"")

    def render(source_path, dest, **kwargs):
        control.render_calls.append((source_path, dest))
        Path(dest).write_bytes(NEW_FINAL_BYTES)
        return True

    def valid_media(path, minimum=1):
        return bool(path and Path(path).is_file() and Path(path).stat().st_size >= minimum)

    # Keep the actual reuse/publish/project decisions. Only impure outcomes
    # are injected; using the real emitter means bypassing its guard clobbers
    # project bytes and makes the preservation assertions fail.
    monkeypatch.setattr(orch, "_valid_media_file", valid_media)
    monkeypatch.setattr(orch.subprocess, "run", encode)
    monkeypatch.setattr(orch, "inspect_clip", lambda *args, **kw: {"critical": False, "warnings": []})
    monkeypatch.setattr(orch, "validate_cut_output", lambda *args, **kw: (True, ""))
    monkeypatch.setattr(project_emit, "_build_project", lambda **kw: _project(**kw))
    legacy = SimpleNamespace(process_video_to_vertical=render,
                             normalize_audio=lambda *a: None,
                             select_cover_frame=lambda *a: None)
    clip = {"start": 0.0, "end": 2.0, "clip_filename": "clip.mp4"}
    data = {"shorts": [clip], "transcript": {"segments": []}}
    args = argparse.Namespace(reframe_mode="auto", no_zoom=True, letterbox_zoom=0.0)
    kwargs = dict(index=0, total=1, clip=clip, input_video=str(source),
                  video_title="clip", output_dir=str(root),
                  metadata_file=str(root / "metadata.json"), clips_data=data,
                  args=args, aspect_ratio=9 / 16, state=_State(), legacy=legacy)
    return SimpleNamespace(root=root, source=source, sl=sl, final=root / "clip.mp4",
                           stamp=Path(ops.slice_provenance_path(str(sl))),
                           control=control, kwargs=kwargs)


def test_queued_rejects_v3_regenerates_v4_then_reuses_v4(queued_job):
    job = queued_job
    assert ops.slice_provenance_valid(str(job.sl), start=0, end=2, source_path=str(job.source))
    _downgrade_to_v3(job.sl)
    assert not ops.slice_provenance_valid(str(job.sl), start=0, end=2, source_path=str(job.source))

    assert orch._render_one_clip(**job.kwargs)
    assert len(job.control.cut_calls) == 1, "queued caller must regenerate the old contract"
    assert job.sl.read_bytes() == NEW_SLICE_BYTES
    assert ops.read_slice_provenance(str(job.sl))["version"] == 4
    assert ops.slice_provenance_valid(str(job.sl), start=0, end=2, source_path=str(job.source))

    # Force the source-slice reuse path, not the existing-final early return.
    job.final.unlink()
    assert orch._render_one_clip(**job.kwargs)
    assert len(job.control.cut_calls) == 1, "current v4 must be reused, not regenerated"
    assert len(job.control.render_calls) == 2


@pytest.mark.parametrize("completed_final", [False, True], ids=["successful-render", "completed-resume"])
@pytest.mark.parametrize("project_kind", ["working-v8", "working-origin-user", "working-caption-edit",
                                         "working-malformed", "archived-v8"])
def test_queued_preserves_active_project_on_both_paths(queued_job, completed_final, project_kind):
    job = queued_job
    if completed_final:
        job.final.write_bytes(OLD_FINAL_BYTES)
    working = job.root / "clip-project-0.json"
    project = _project()
    if project_kind == "archived-v8":
        active = job.root / "clip-project-0.v8.json"
        project.update(version=8, origin="user")
        _write_json(job.root / "project-versions.json", {"0": {"latest": 8, "versions": {
            "8": {"version": 8, "project_file": active.name, "origin": "user",
                  "output_file": "clip.edited_v8.mp4"}}}})
    else:
        active = working
        if project_kind == "working-v8":
            project.update(version=8, origin="user")
        elif project_kind == "working-origin-user":
            project["origin"] = "user"
        elif project_kind == "working-caption-edit":
            project["captions"]["edits"] = {"w0": "user correction"}
    if project_kind == "working-malformed":
        active.write_text("malformed existing user file{", encoding="utf8")
    else:
        _write_json(active, project)
    before = active.read_bytes()
    assert get_project_path(str(job.root), 0) == str(active)

    assert orch._render_one_clip(**job.kwargs)
    assert active.read_bytes() == before
    assert get_project_path(str(job.root), 0) == str(active)
    assert len(job.control.cut_calls) == 0
    assert len(job.control.render_calls) == (0 if completed_final else 1)
    if project_kind == "archived-v8":
        assert not working.exists(), "a new default would hide the active archived edit"
        assert latest_project_version(str(job.root), 0) == 8
        assert json.loads(active.read_text(encoding="utf8"))["version"] == 8


def test_queued_source_mutation_rejects_before_publication(queued_job):
    job = queued_job
    _downgrade_to_v3(job.sl)  # Require a cut while retaining the old valid media.
    old_slice, old_stamp = job.sl.read_bytes(), job.stamp.read_bytes()
    job.control.mutate_source = True
    with pytest.raises(RuntimeError, match="source changed during cut"):
        orch._render_one_clip(**job.kwargs)
    assert len(job.control.cut_calls) == 1
    assert len(job.control.render_calls) == 0
    assert job.sl.read_bytes() == old_slice
    assert job.stamp.read_bytes() == old_stamp
    assert not Path(str(job.sl) + ".tmp.mp4").exists()
    assert not job.final.exists()


@pytest.fixture(scope="module")
def direct_clip_loop():
    main_path = Path(orch.__file__).with_name("main.py")
    tree = ast.parse(main_path.read_text(encoding="utf8"))
    loops = [node for node in ast.walk(tree) if isinstance(node, ast.For)
             and ast.unparse(node.iter) == "enumerate(clips_data['shorts'])"]
    assert len(loops) == 1, "locate the actual legacy CLI loop, not a copied implementation"
    return compile(ast.Module(body=[loops[0]], type_ignores=[]), str(main_path), "exec")


@pytest.mark.parametrize("mode", ["source-mutation", "render-failure", "render-invalid", "render-success"])
def test_direct_cli_publication_preserves_previous_artifacts(tmp_path, direct_clip_loop, mode):
    source = tmp_path / "input.mp4"
    sl = tmp_path / "source_clip_clip_1.mp4"
    final = tmp_path / "clip_clip_1.mp4"
    source.write_bytes(SOURCE_BYTES)
    sl.write_bytes(OLD_SLICE_BYTES)
    final.write_bytes(OLD_FINAL_BYTES)
    ops.write_slice_provenance(str(sl), start=0, end=2,
                               source_fp=ops.source_fingerprint(str(source)))
    stamp = Path(ops.slice_provenance_path(str(sl)))
    old_stamp = stamp.read_bytes()
    renders, normalizations, covers = [], [], []

    def encode(command, **kwargs):
        Path(command[-1]).write_bytes(NEW_SLICE_BYTES)
        if mode == "source-mutation":
            source.write_bytes(SOURCE_BYTES + b" changed during encoder run")
        return SimpleNamespace(returncode=0, stderr=b"")

    def render(source_path, output_path, **kwargs):
        renders.append(output_path)
        Path(output_path).write_bytes(NEW_FINAL_BYTES)
        return mode != "render-failure"

    def validate(path, duration):
        if mode == "render-invalid" and path.endswith(".render.tmp.mp4"):
            return False, "injected invalid successful render"
        return True, ""

    # This is the exact direct caller AST with small injected dependencies.
    # Removing its fingerprint check, stable-to-temp routing, validation, or
    # replace-on-success gate changes the observed artifacts below.
    import subprocess
    subprocess_stub = SimpleNamespace(run=encode, DEVNULL=subprocess.DEVNULL,
                                      PIPE=subprocess.PIPE, TimeoutExpired=subprocess.TimeoutExpired)
    clip = {"start": 0.0, "end": 2.0}
    env = dict(os=os, json=json, subprocess=subprocess_stub, clips_data={"shorts": [clip]},
               video_title="clip", output_dir=str(tmp_path), input_video=str(source),
               metadata_tmp=str(tmp_path / "metadata.json.tmp"),
               metadata_file=str(tmp_path / "metadata.json"),
               args=argparse.Namespace(reframe_mode="auto", no_zoom=True),
               aspect_ratio=9 / 16, letterbox_zoom=0,
               clip_output_basename=ops.clip_output_basename,
               source_fingerprint=ops.source_fingerprint,
               build_cut_command=ops.build_cut_command,
               write_slice_provenance=ops.write_slice_provenance,
               validate_cut_output=validate, process_video_to_vertical=render,
               _project_framing=_task13_project_framing,
               normalize_audio=lambda path: normalizations.append(path),
               select_cover_frame=lambda path: covers.append(path))
    exec(direct_clip_loop, env)

    if mode == "source-mutation":
        assert renders == []
        assert sl.read_bytes() == OLD_SLICE_BYTES
        assert stamp.read_bytes() == old_stamp
        assert not Path(str(sl) + ".tmp.mp4").exists()
    else:
        assert renders == [str(final) + ".render.tmp.mp4"]
        assert sl.read_bytes() == NEW_SLICE_BYTES
    if mode == "render-success":
        assert final.read_bytes() == NEW_FINAL_BYTES
        assert normalizations == [str(final)]
        assert covers == [str(final)]
    else:
        assert final.read_bytes() == OLD_FINAL_BYTES
        assert normalizations == covers == []
    if mode == "render-invalid":
        assert not Path(str(final) + ".render.tmp.mp4").exists()
