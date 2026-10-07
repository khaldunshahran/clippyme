"""Independent Task 05 review contracts, retained by the coordinator."""

import asyncio
import json
from pathlib import Path
import pytest
from clippyme.domain import project_lineage as pl, project_render as pr
from clippyme.domain.clip_project import validate_project, project_to_json

@pytest.fixture
def case(tmp_path):
    source = tmp_path / "source.mp4"
    base = tmp_path / "base.mp4"
    source.write_bytes(b"source original")
    base.write_bytes(b"base original")
    d = {"schema": "nugget.clip-project/1", "job_id": "job", "clip_index": 0,
         "source": {"file": source.name, "width": 1920, "height": 1080, "duration": 10.0},
         "segments": [{"id": "seg-1", "start": 0., "end": 10.,
                      "crop": {"x": 660., "y": 0., "w": 608., "h": 1080.},
                      "transition_in": "cut"}],
         "render": {"width": 608, "height": 1080, "fps": 30, "encoder": "libx264"}}
    d["reframed_base"] = pl.build_lineage(clip_final=str(base), job_dir=str(tmp_path),
        source=d["source"], segments=d["segments"], render=d["render"])
    return tmp_path, source, base, d

def run_render(monkeypatch, case, *, dimensions=None):
    job, source, base, d = case
    calls = {"materialize": [], "compose": [], "music": [], "qa": []}
    async def materialize(project, job_dir, index):
        calls["materialize"].append(project)
        path = job / "materialized.mp4"; path.write_bytes(b"center crop")
        return str(path), 10.
    async def compose(**kw):
        calls["compose"].append(kw)
        (job / "composed_clip_0.mp4").write_bytes(Path(kw["base_clip"]).read_bytes())
        return "composed_clip_0.mp4"
    async def music(path, project, job_dir, index):
        calls["music"].append(project)
        return path
    def qa(path, **kw):
        calls["qa"].append(kw)
        return {"critical": []}
    monkeypatch.setattr(pr, "_materialize_timeline", materialize)
    monkeypatch.setattr(pr, "compose_layers", compose)
    monkeypatch.setattr(pr, "_mix_music", music)
    monkeypatch.setattr(pr, "inspect_clip", qa)
    width, height = dimensions or (608, 1080)
    monkeypatch.setattr(pr, "probe_media", lambda path: {"duration": 10., "width": width, "height": height, "fps": 30., "has_video": True})
    coroutine = pr.render_project(job_id="job", clip_index=0,
        project=validate_project(d), job_dir=str(job), metadata={},
        clip_info={"start": 0., "end": 10., "clip_filename": "base.mp4"})
    # Every injected async I/O boundary completes inline. Drive the real
    # coroutine without an event loop: Windows loopback sockets are blocked
    # in this sandbox. No production code or eligibility decision is replaced.
    try:
        coroutine.send(None)
    except StopIteration as completed:
        result = completed.value
    else:
        coroutine.close()
        raise AssertionError("unexpected suspension in injected control-flow test")
    return calls, result

@pytest.mark.parametrize("broken", ["missing", "stale", "malformed"])
def test_invalid_asset_does_not_publish_center_crop(monkeypatch, case, broken):
    job, source, base, d = case
    if broken == "missing": base.unlink()
    if broken == "stale": base.write_bytes(b"replaced content of different length")
    if broken == "malformed": d["reframed_base"]["base_ref"] = "../outside.mp4"
    with pytest.raises(pr.ProjectRenderError):
        run_render(monkeypatch, case)
    assert not list(job.glob("*.edited_v*.mp4"))

def test_replaced_source_not_eligible(case):
    job, source, base, d = case
    source.write_bytes(b"a completely different source")
    outcome, _, reason = pl.check_lineage_eligible(d["reframed_base"], d, str(job))
    assert outcome == "invalid_asset", "same source filename does not prove source identity"

def test_segment_id_is_metadata(case):
    job, source, base, d = case
    d["segments"][0]["id"] = "renamed-only"
    assert pl.check_lineage_eligible(d["reframed_base"], d, str(job))[0] == "eligible"

def test_nonvertical_project_qa_uses_actual_aspect(monkeypatch, case):
    job, source, base, d = case
    d["render"]["width"] = 1080
    d["reframed_base"]["render"]["width"] = 1080
    calls, _ = run_render(monkeypatch, case, dimensions=(1080, 1080))
    assert calls["qa"][0]["expected_aspect"] == 1.0

def test_base_media_geometry_must_match(monkeypatch, case):
    with pytest.raises(pr.ProjectRenderError):
        run_render(monkeypatch, case, dimensions=(1920, 1080))

def test_schema_twin_has_additive_lineage():
    schema = json.loads(Path("src/clippyme/domain/clip_project.schema.json").read_text(encoding="utf-8"))
    assert "reframed_base" in schema["properties"]

def test_malformed_baseline_fails_cleanly(case):
    job, source, base, d = case
    d["reframed_base"]["segments"][0]["crop"] = "invalid"
    # This shape survives the production ReframedBase list[dict] model.
    d = validate_project(d).model_dump()
    try:
        outcome, _, reason = pl.check_lineage_eligible(d["reframed_base"], d, str(job))
    except (AttributeError, TypeError) as exc:
        pytest.fail(f"unvalidated nested lineage crashes: {exc}")
    assert outcome == "invalid_asset"

def test_real_route_reuses_base_and_current_layers(monkeypatch, case):
    job, source, base, d = case
    d["captions"] = {"words": [{"id": "w0", "w": "old", "start": 1., "end": 2.}], "edits": {"w0": "fixed"}}
    d["overlays"] = [{"type": "hook", "text": "current hook", "start": 0., "end": 2.}]
    d["grade"] = {"preset": "cinematic"}
    d["audio"] = [{"file": "music.mp3", "volume": 0.4}]
    calls, result = run_render(monkeypatch, case)
    assert not calls["materialize"]
    comp = calls["compose"][0]
    assert comp["base_clip"] == str(base)
    assert comp["hook_params"]["text"] == "current hook"
    assert comp["grade_params"]["preset"] == "cinematic"
    assert comp["metadata"]["transcript"]["segments"][0]["words"][0]["word"] == "fixed"
    assert calls["music"][0].audio[0].file == "music.mp3"

def test_real_route_crop_edit_materializes(monkeypatch, case):
    case[3]["segments"][0]["crop"]["x"] = 0.
    calls, _ = run_render(monkeypatch, case)
    assert calls["materialize"]
    assert calls["compose"][0]["base_clip"].endswith("materialized.mp4")

def test_committed_framing_regression_detects_disabled_production_reuse(monkeypatch, tmp_path):
    import runpy
    suite = runpy.run_path("tests/domain/test_task05_ffmpeg.py")
    fixture = suite["synth_job"].__wrapped__(tmp_path)
    # Patch the actual lookup (project_lineage), not the stale alias.
    monkeypatch.setattr(pl, "check_lineage_eligible",
                        lambda *args, **kwargs: ("geometry_edit", None, "disabled by review mutation"))
    # The committed production-framing gate must fail when the actual
    # render selector is disabled. It currently never reaches that selector.
    with pytest.raises(AssertionError):
        suite["test_lineage_base_preserves_framing"](fixture)

import json
import runpy
import sys
from pathlib import Path
import pytest
from clippyme.domain import project_lineage as pl, project_render as pr
from clippyme.domain.clip_project import validate_project

def render(monkeypatch, case, probe):
    job, source, base, d = case
    async def compose(**kw):
        (job / "composed_clip_0.mp4").write_bytes(b"composed")
        return "composed_clip_0.mp4"
    async def music(path, *args): return path
    monkeypatch.setattr(pr, "compose_layers", compose)
    monkeypatch.setattr(pr, "_mix_music", music)
    monkeypatch.setattr(pr, "inspect_clip", lambda *a, **k: {"critical": []})
    monkeypatch.setattr(pr, "probe_media", lambda p: probe)
    coroutine = pr.render_project(job_id="job", clip_index=0,
        project=validate_project(d), job_dir=str(job), metadata={}, clip_info={})
    try: coroutine.send(None)
    except StopIteration as done: return done.value
    else:
        coroutine.close()
        raise AssertionError("unexpected suspension")

@pytest.mark.parametrize("update", [
    {"duration": 1.0},
    {"duration": float("inf")},
    {"fps": None},
    {"fps": float("nan")},
    {"fps": float("inf")},
    {"has_video": False},
])
def test_reuse_rejects_incompatible_or_invalid_media(monkeypatch, case, update):
    probe = {"duration": 10., "width": 608, "height": 1080, "fps": 30., "has_video": True}
    probe.update(update)
    with pytest.raises(pr.ProjectRenderError):
        render(monkeypatch, case, probe)

@pytest.mark.parametrize("broken", ["missing", "stale"])
def test_geometry_edit_can_use_available_source_without_original_base(case, broken):
    job, source, base, d = case
    d["segments"][0]["crop"]["x"] = 0.
    if broken == "missing": base.unlink()
    else: base.write_bytes(b"different-size-base")
    outcome, _, reason = pl.check_lineage_eligible(d["reframed_base"], d, str(job))
    assert outcome == "geometry_edit", reason

@pytest.mark.parametrize("identity", [None, {}, "invalid"])
def test_source_identity_is_required_for_lineage_reuse(case, identity):
    job, source, base, d = case
    d["reframed_base"]["source"]["identity"] = identity
    outcome, _, reason = pl.check_lineage_eligible(d["reframed_base"], d, str(job))
    assert outcome == "invalid_asset", "missing/malformed identity must not bypass validation"

def test_missing_baseline_crop_field_is_invalid_asset(case):
    job, source, base, d = case
    del d["reframed_base"]["segments"][0]["crop"]["x"]
    outcome, _, reason = pl.check_lineage_eligible(d["reframed_base"], d, str(job))
    assert outcome == "invalid_asset", "incomplete baseline is not a real geometry edit"

def test_committed_av_gate_detects_corrupt_export(monkeypatch, tmp_path):
    import shutil
    suite = runpy.run_path("tests/domain/test_task05_render.py")
    fixture = suite["av_job"].__wrapped__(tmp_path)
    def corrupt_copy(src, dst, *args, **kwargs):
        Path(dst).write_bytes(b"not a video; no frames or audio")
        return dst
    monkeypatch.setattr(shutil, "copy", corrupt_copy)
    # A claimed audiovisual export gate must reject a corrupt exported file.
    with pytest.raises((AssertionError, RuntimeError, ValueError, KeyError)):
        suite["test_render_reuses_base_with_current_layers"](monkeypatch, fixture)

def test_version_save_regression_calls_real_save(monkeypatch, tmp_path):
    suite = runpy.run_path("tests/domain/test_task05_lineage.py")
    def broken_save(**kwargs):
        raise AssertionError("review mutation: actual save path broken")
    monkeypatch.setattr(pr, "save_project_version", broken_save)
    # The promised versioned save/load test must exercise the real primitive.
    with pytest.raises(AssertionError):
        suite["test_versioned_save_load_preserves_lineage"](tmp_path)

import runpy
import shutil
import subprocess
import sys
from pathlib import Path
import pytest
from clippyme.domain import project_render as pr

def test_video_evidence_must_be_explicit(monkeypatch, case):
    probe = {"duration": 10., "width": 608, "height": 1080, "fps": 30.}
    with pytest.raises(pr.ProjectRenderError):
        render(monkeypatch, case, probe)

@pytest.mark.parametrize("mutation", ["silent", "white", "one_frame"])
def test_av_regression_rejects_media_mutations(monkeypatch, tmp_path, mutation):
    suite = runpy.run_path("tests/domain/test_task05_render.py")
    fixture = suite["av_job"].__wrapped__(tmp_path)
    def bad_copy(src, dst, *a, **kw):
        if mutation == "silent":
            cmd = ["ffmpeg", "-y", "-v", "error", "-i", str(src), "-an", "-c:v", "copy", str(dst)]
        elif mutation == "white":
            cmd = ["ffmpeg", "-y", "-v", "error", "-i", str(src),
                   "-vf", "drawbox=color=white:t=fill", "-c:v", "libx264", "-c:a", "copy", str(dst)]
        else:
            cmd = ["ffmpeg", "-y", "-v", "error", "-i", str(src),
                   "-frames:v", "1", "-an", "-c:v", "libx264", str(dst)]
        proc = subprocess.run(cmd, capture_output=True)
        assert proc.returncode == 0, proc.stderr
        return dst
    monkeypatch.setattr(shutil, "copy", bad_copy)
    with pytest.raises((AssertionError, RuntimeError, ValueError, KeyError)):
        suite["test_render_reuses_base_with_current_layers"](monkeypatch, fixture)

def test_render_media_gate_calls_existing_compositor(monkeypatch, tmp_path):
    # The durable real-export regression uses the existing compositor.
    suite = runpy.run_path("tests/domain/test_task05_real_export.py")
    calls = []
    original = pr.compose_layers
    async def spy(**kw):
        calls.append(kw)
        return await original(**kw)
    monkeypatch.setattr(pr, "compose_layers", spy)
    suite["test_real_export_framing_and_audio"](monkeypatch, tmp_path, edited=False)
    assert calls, "real export must invoke the existing compositor"

def test_direct_mutation_gate_restores_patches(monkeypatch, tmp_path):
    suite = runpy.run_path("tests/domain/test_task05_ffmpeg.py")
    fixture = suite["synth_job"].__wrapped__(tmp_path)
    original = {key: getattr(pr, key) for key in
                ["compose_layers", "_materialize_timeline", "_mix_music", "probe_media", "inspect_clip"]}
    # Ensure this regression is isolated even though the test under review leaks.
    try:
        suite["test_lineage_base_preserves_framing"](fixture)
        assert all(getattr(pr, key) is value for key, value in original.items()), "direct helper leaks production monkeypatches"
    finally:
        for key, value in original.items():
            setattr(pr, key, value)

def test_actual_music_mix_and_render_idempotency(monkeypatch, tmp_path):
    import asyncio
    import hashlib
    import json
    import numpy as np
    import subprocess
    suite = runpy.run_path("tests/domain/test_task05_real_export.py")
    job, source, base = suite["_make_fixture"](tmp_path)
    data = suite["_project_dict"](job, source, base)
    music = Path(job) / "music.wav"
    suite["_run"](["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i",
                   "sine=frequency=440:sample_rate=48000:duration=2", str(music)])
    data["audio"] = [{"file": music.name, "volume": 0.4, "offset": 0.0}]
    assets = {p: hashlib.sha256(Path(p).read_bytes()).digest() for p in [source, base]}
    async def inline_thread(func, *args, **kwargs):
        return func(*args, **kwargs)
    monkeypatch.setattr(asyncio, "to_thread", inline_thread)
    monkeypatch.setenv("CLIPPYME_VIDEO_ENCODER", "libx264")
    result = suite["_drive_real"](pr.render_project(job_id="fixture", clip_index=0,
        project=validate_project(data), job_dir=job, metadata={},
        clip_info={"clip_filename": "clip_final_0.mp4"}))
    pcm = suite["_run"](["ffmpeg", "-v", "error", "-i", result.output_path,
        "-vn", "-ac", "1", "-ar", "48000", "-f", "f32le", "-"])
    samples = np.frombuffer(pcm, dtype=np.float32)[4800:24000]
    spectrum = np.abs(np.fft.rfft(samples * np.hanning(len(samples))))
    frequencies = np.fft.rfftfreq(len(samples), 1 / 48000)
    for tone in [440, 880]:
        band = spectrum[np.abs(frequencies - tone) < 10]
        assert band.max() > spectrum.max() * 0.05, f"missing {tone}Hz music/speech"
    for path, fingerprint in assets.items():
        assert hashlib.sha256(Path(path).read_bytes()).digest() == fingerprint
    rendered_project = validate_project(json.loads(Path(
        pr.get_project_path(job, 0, version=result.version)).read_text(encoding="utf-8")))
    async def must_not_recompose(**kwargs):
        raise AssertionError("same render key must deduplicate before composition")
    monkeypatch.setattr(pr, "compose_layers", must_not_recompose)
    duplicate = suite["_drive_real"](pr.render_project(job_id="fixture", clip_index=0,
        project=rendered_project, job_dir=job, metadata={},
        clip_info={"clip_filename": "clip_final_0.mp4"}))
    assert duplicate.deduped is True
    assert duplicate.output_path == result.output_path

def test_archived_undo_saves_original_lineage(case):
    import json
    import uuid
    job, source, base, data = case
    version1, saved1 = pr.save_project_version(job_dir=str(job), clip_index=0,
                                             project=validate_project(data))
    edited = saved1.model_dump()
    edited["captions"]["edits"] = {"w0": "edited"}
    edited["origin"] = "user"
    edited["idempotency_key"] = str(uuid.uuid4())
    version2, _ = pr.save_project_version(job_dir=str(job), clip_index=0,
        project=validate_project(edited), expected_version=version1)
    archive = json.loads(Path(pr.get_project_path(str(job), 0, version=version1)).read_text(encoding="utf-8"))
    archive["origin"] = "user"
    archive["idempotency_key"] = str(uuid.uuid4())
    version3, undone = pr.save_project_version(job_dir=str(job), clip_index=0,
        project=validate_project(archive), expected_version=version2)
    assert version3 == version2 + 1
    assert undone.captions.edits == {}
    assert undone.reframed_base.model_dump() == saved1.reframed_base.model_dump()
    latest = validate_project(json.loads(Path(pr.get_project_path(str(job), 0)).read_text(encoding="utf-8")))
    assert latest.captions.edits == {}
    assert latest.version == version3
