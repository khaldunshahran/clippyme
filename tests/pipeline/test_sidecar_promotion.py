"""Task 13: stable turn-framing sidecar promotion and lifecycle.

Drives the REAL queued (``orchestrator._render_one_clip``) and direct
(``main.py`` clip loop, AST-compiled like
``test_caller_regressions.py``) render/promotion paths with injected fakes.
The fake renderer writes production-valid Task 12 plans (via the real
``plan_from_dict``/``plan_to_dict``) next to the attempt video, exactly
like production ``reframe`` does.

- accepted video + sidecar share a stable final identity (sidecar at
  ``<clip_final stem>.turn_framing_plan.json``; content hash matches the
  project framing ``plan.sha256``);
- project JSON never references a temp attempt path;
- failed QA/retry removes the temp sidecar and preserves a previously
  accepted final video+sidecar pair;
- a successful later attempt promotes only its own plan;
- completed resume emits/retains the same project framing record without
  rerendering (render function not called; framing field identical).

Mutation gates:
- omitting promotion (no sidecar written by the render) -> the project
  simply has no framing field (emission degrades, never fails);
- temp-path binding: fails if ".render.tmp" appears anywhere in the
  emitted project JSON.
"""

import argparse
import ast
import hashlib
import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from clippyme.pipeline import orchestrator as orch
from clippyme.pipeline import project_emit
from clippyme.pipeline import run_ops as ops
from tests.pipeline.task13_helpers import sidecar_for, write_task12_sidecar


SOURCE_BYTES = b"source media" * 2000
SLICE_BYTES = b"source slice" * 2000
VIDEO_BYTES = b"rendered vertical clip" * 2000  # >= 10_000 bytes


class _State:
    def mark_clip(self, *args):
        pass

    def start(self, *args, **kwargs):
        pass


@pytest.fixture
def promo_job(tmp_path, monkeypatch, framing_impl):
    """Queued render job with a fake renderer that writes a per-attempt plan.

    ``control.qa_reports``: queue of inspect_clip reports; each render
    attempt pops one (default: non-critical). ``control.write_sidecar``:
    when False, the fake render writes no sidecar (promotion-omission
    simulation).
    """
    root = tmp_path / "job"
    root.mkdir()
    source = root / "input.mp4"
    source.write_bytes(SOURCE_BYTES)
    sl = root / "source_clip.mp4"
    sl.write_bytes(SLICE_BYTES)
    ops.write_slice_provenance(
        str(sl), start=0.0, end=2.0, source_fp=ops.source_fingerprint(str(source))
    )
    control = SimpleNamespace(
        render_calls=[], qa_reports=[], write_sidecar=True, attempt=0
    )

    def render(source_path, dest, **kwargs):
        control.attempt += 1
        control.render_calls.append(dest)
        Path(dest).write_bytes(VIDEO_BYTES)
        if control.write_sidecar:
            # Production-valid Task 12 plan next to the attempt video, with
            # a per-attempt digest so plans are distinguishable.
            write_task12_sidecar(dest, f"attempt-{control.attempt}")
        return True

    def inspect(path, **kwargs):
        if control.qa_reports:
            return control.qa_reports.pop(0)
        return {"critical": False, "warnings": []}

    def valid_media(path, minimum=1):
        return bool(
            path and Path(path).is_file() and Path(path).stat().st_size >= minimum
        )

    def fake_probe(path):
        return {"width": 608, "height": 1080, "fps": 30, "duration": 2.0}

    monkeypatch.setattr(orch, "_valid_media_file", valid_media)
    monkeypatch.setattr(orch, "inspect_clip", inspect)
    monkeypatch.setattr(orch, "validate_cut_output", lambda *a, **k: (True, ""))
    monkeypatch.setattr(project_emit, "probe_media", fake_probe)
    legacy = SimpleNamespace(
        process_video_to_vertical=render,
        normalize_audio=lambda *a: None,
        select_cover_frame=lambda *a: None,
        # NOTE: deliberately no turn_framing_sidecar_for -- production
        # passes the real reframe module; the orchestrator must tolerate
        # fakes without it (see _sidecar_for).
    )
    clip = {"start": 0.0, "end": 2.0, "clip_filename": "clip.mp4"}
    data = {"shorts": [clip], "transcript": {"segments": []}}
    args = argparse.Namespace(reframe_mode="auto", no_zoom=True, letterbox_zoom=0.0)
    kwargs = dict(
        index=0, total=1, clip=clip, input_video=str(source),
        video_title="clip", output_dir=str(root),
        metadata_file=str(root / "metadata.json"), clips_data=data,
        args=args, aspect_ratio=9 / 16, state=_State(), legacy=legacy,
    )
    return SimpleNamespace(
        root=root, source=source, sl=sl, final=root / "clip.mp4",
        control=control, kwargs=kwargs,
    )


def _build_kwargs(job):
    return dict(
        job_id="job1", index=0, start=0.0, end=2.0,
        clip=job.kwargs["clip"], clips_data=job.kwargs["clips_data"],
        clip_source=str(job.sl), clip_final=str(job.final),
        job_dir=str(job.root),
    )


def test_queued_promotes_sidecar_to_stable_name(promo_job):
    job = promo_job
    assert orch._render_one_clip(**job.kwargs)

    temp_sidecar = sidecar_for(str(job.final) + ".render.tmp.mp4")
    stable_sidecar = sidecar_for(str(job.final))
    assert not os.path.exists(temp_sidecar), "temp sidecar must not leak"
    assert os.path.isfile(stable_sidecar), "stable sidecar must be promoted"
    assert job.final.read_bytes() == VIDEO_BYTES

    plan = json.loads(Path(stable_sidecar).read_text(encoding="utf-8"))
    assert plan["config_digest"] == "cfg-attempt-1"

    # Accepted video + sidecar share a stable final identity: the project
    # framing content hash matches the promoted sidecar bytes.
    data = project_emit._build_project(**_build_kwargs(job))
    assert "framing" in data, "usable sidecar must produce a framing field"
    plan_id = data["framing"]["plan"]
    assert plan_id["plan_ref"] == "clip.turn_framing_plan.json"
    assert plan_id["sha256"] == hashlib.sha256(
        Path(stable_sidecar).read_bytes()
    ).hexdigest()

    # Project JSON never references the temporary attempt path.
    project_path = job.root / "clip-project-0.json"
    assert project_path.is_file()
    project_text = project_path.read_text(encoding="utf-8")
    assert ".render.tmp" not in project_text
    assert ".reframe.tmp" not in project_text
    assert ".render.tmp" not in json.dumps(data)


def test_queued_failed_qa_removes_temp_sidecar_preserves_previous_pair(promo_job):
    job = promo_job
    assert orch._render_one_clip(**job.kwargs)
    stable_sidecar = sidecar_for(str(job.final))
    first_bytes = Path(stable_sidecar).read_bytes()

    # Drop the final video (keep the accepted sidecar); the re-render fails
    # QA on every attempt (retries=1 -> 2 attempts).
    job.final.unlink()
    job.control.qa_reports = [
        {"critical": True, "issues": ["bad"]},
        {"critical": True, "issues": ["bad"]},
    ]
    assert orch._render_one_clip(**job.kwargs) is False

    temp_sidecar = sidecar_for(str(job.final) + ".render.tmp.mp4")
    assert not os.path.exists(temp_sidecar), "failed attempt must not leak"
    assert not job.final.exists(), "failed render must not publish"
    assert Path(stable_sidecar).read_bytes() == first_bytes, (
        "previously accepted sidecar must be preserved"
    )


def test_queued_retry_promotes_only_own_plan(promo_job):
    job = promo_job
    assert orch._render_one_clip(**job.kwargs)

    job.final.unlink()
    job.control.qa_reports = [
        {"critical": True, "issues": ["bad"]},
        {"critical": False, "warnings": []},
    ]
    assert orch._render_one_clip(**job.kwargs) is True

    stable_sidecar = sidecar_for(str(job.final))
    plan = json.loads(Path(stable_sidecar).read_text(encoding="utf-8"))
    assert plan["config_digest"] == "cfg-attempt-3", (
        "attempts: 1 (first render) + 2 (failed) + 3 (accepted); "
        "only the accepted attempt's plan may be promoted"
    )
    temp_sidecar = sidecar_for(str(job.final) + ".render.tmp.mp4")
    assert not os.path.exists(temp_sidecar)


def test_queued_resume_keeps_framing_without_rerender(promo_job):
    job = promo_job
    assert orch._render_one_clip(**job.kwargs)
    before = project_emit._build_project(**_build_kwargs(job))["framing"]
    stable_sidecar = sidecar_for(str(job.final))
    sidecar_bytes = Path(stable_sidecar).read_bytes()
    render_calls = len(job.control.render_calls)

    assert orch._render_one_clip(**job.kwargs) is True  # resume path
    assert len(job.control.render_calls) == render_calls, (
        "resume must not rerender"
    )
    assert Path(stable_sidecar).read_bytes() == sidecar_bytes
    after = project_emit._build_project(**_build_kwargs(job))["framing"]
    assert after == before, "resume must retain the same framing record"


def test_queued_omitted_sidecar_degrades_without_framing(promo_job):
    """Mutation gate: no promotion -> project simply has no framing field."""
    job = promo_job
    job.control.write_sidecar = False
    assert orch._render_one_clip(**job.kwargs) is True

    stable_sidecar = sidecar_for(str(job.final))
    assert not os.path.exists(stable_sidecar)
    data = project_emit._build_project(**_build_kwargs(job))
    assert "framing" not in data, "missing sidecar must degrade gracefully"
    project_path = job.root / "clip-project-0.json"
    assert project_path.is_file(), "emission must still succeed"


# --- Direct CLI loop (AST-compiled, like test_caller_regressions.py) ---


@pytest.fixture(scope="module")
def direct_clip_loop():
    main_path = Path(orch.__file__).with_name("main.py")
    tree = ast.parse(main_path.read_text(encoding="utf8"))
    loops = [
        node for node in ast.walk(tree) if isinstance(node, ast.For)
        and ast.unparse(node.iter) == "enumerate(clips_data['shorts'])"
    ]
    assert len(loops) == 1, "locate the actual legacy CLI loop"
    return compile(ast.Module(body=[loops[0]], type_ignores=[]), str(main_path), "exec")


@pytest.fixture(scope="module")
def direct_reframe_only_block():
    main_path = Path(orch.__file__).with_name("main.py")
    tree = ast.parse(main_path.read_text(encoding="utf8"))
    blocks = [
        node for node in ast.walk(tree) if isinstance(node, ast.If)
        and ast.unparse(node.test) == "args.reframe_only"
    ]
    assert len(blocks) == 1, "locate the actual reframe-only block"
    return compile(ast.Module(body=[blocks[0]], type_ignores=[]), str(main_path), "exec")


def _direct_env(tmp_path, monkey_mode="success"):
    """Injected env for the AST-compiled direct loop (mirrors the real names).

    monkey_mode: "success" | "render-invalid".
    """
    source = tmp_path / "input.mp4"
    sl = tmp_path / "source_clip_clip_1.mp4"
    final = tmp_path / "clip_clip_1.mp4"
    source.write_bytes(SOURCE_BYTES)
    sl.write_bytes(SLICE_BYTES)
    final.write_bytes(SOURCE_BYTES)  # previously accepted clip
    ops.write_slice_provenance(
        str(sl), start=0.0, end=2.0, source_fp=ops.source_fingerprint(str(source)))

    def encode(command, **kwargs):
        Path(command[-1]).write_bytes(SLICE_BYTES)
        return SimpleNamespace(returncode=0, stderr=b"")

    def render(source_path, output_path, **kwargs):
        Path(output_path).write_bytes(VIDEO_BYTES)
        write_task12_sidecar(output_path, "direct")
        return True

    def validate(path, duration):
        if monkey_mode == "render-invalid" and path.endswith(".render.tmp.mp4"):
            return False, "injected invalid successful render"
        return True, ""

    subprocess_stub = SimpleNamespace(
        run=encode, DEVNULL=subprocess.DEVNULL,
        PIPE=subprocess.PIPE, TimeoutExpired=subprocess.TimeoutExpired)
    clip = {"start": 0.0, "end": 2.0}
    env = dict(
        os=os, json=json, subprocess=subprocess_stub,
        clips_data={"shorts": [clip]}, video_title="clip",
        output_dir=str(tmp_path), input_video=str(source),
        metadata_tmp=str(tmp_path / "metadata.json.tmp"),
        metadata_file=str(tmp_path / "metadata.json"),
        args=argparse.Namespace(reframe_mode="auto", no_zoom=True),
        aspect_ratio=9 / 16, letterbox_zoom=0,
        clip_output_basename=ops.clip_output_basename,
        source_fingerprint=ops.source_fingerprint,
        build_cut_command=ops.build_cut_command,
        write_slice_provenance=ops.write_slice_provenance,
        validate_cut_output=validate, process_video_to_vertical=render,
        normalize_audio=lambda path: None, select_cover_frame=lambda path: None,
    )
    return SimpleNamespace(
        env=env, source=source, sl=sl, final=final, render=render)


def test_direct_cli_promotes_sidecar(tmp_path, direct_clip_loop):
    ctx = _direct_env(tmp_path, "success")
    exec(direct_clip_loop, ctx.env)

    temp_sidecar = sidecar_for(str(ctx.final) + ".render.tmp.mp4")
    stable_sidecar = sidecar_for(str(ctx.final))
    assert ctx.final.read_bytes() == VIDEO_BYTES
    assert not os.path.exists(temp_sidecar)
    assert os.path.isfile(stable_sidecar)
    plan = json.loads(Path(stable_sidecar).read_text(encoding="utf-8"))
    assert plan["config_digest"] == "cfg-direct"


def test_direct_cli_invalid_render_removes_temp_sidecar(tmp_path, direct_clip_loop):
    ctx = _direct_env(tmp_path, "render-invalid")
    old_bytes = ctx.final.read_bytes()
    exec(direct_clip_loop, ctx.env)

    temp_sidecar = sidecar_for(str(ctx.final) + ".render.tmp.mp4")
    assert ctx.final.read_bytes() == old_bytes, "invalid render must not publish"
    assert not os.path.exists(temp_sidecar), "invalid render must not leak a sidecar"
    assert not os.path.exists(str(ctx.final) + ".render.tmp.mp4")


def test_direct_reframe_only_promotes_sidecar(tmp_path, direct_reframe_only_block):
    src = tmp_path / "slice.mp4"
    src.write_bytes(SLICE_BYTES)
    final = tmp_path / "out.mp4"

    def render(source_path, output_path, **kwargs):
        Path(output_path).write_bytes(VIDEO_BYTES)
        write_task12_sidecar(output_path, "reframe-only")
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
    )
    with pytest.raises(SystemExit) as exc:
        exec(direct_reframe_only_block, env)
    assert exc.value.code == 0

    temp_sidecar = sidecar_for(str(final) + ".reframe.tmp.mp4")
    stable_sidecar = sidecar_for(str(final))
    assert final.read_bytes() == VIDEO_BYTES
    assert not os.path.exists(temp_sidecar)
    plan = json.loads(Path(stable_sidecar).read_text(encoding="utf-8"))
    assert plan["config_digest"] == "cfg-reframe-only"
