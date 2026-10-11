"""Task 18 — finishing compiler gates (host-runnable, no heavy deps).

Covers contracts 2/3/4/5 at the compiler level:
- default-OFF rollout (FinishingDisabled),
- template/param allowlists and bounded typed values,
- text escaping (no markup/code from chat reaches the composition),
- production time-mapping reuse (word IDs, retained-word edits, removed
  words dropped, out-of-segment words dropped),
- half-open hook intervals,
- empty captions handled honestly,
- framing-trajectory projects fail clearly (no invented trajectory),
- deterministic manifest digest (preview/export identity),
- no remote asset fetches in generated compositions.
"""
import asyncio
import json
import os

import pytest

from clippyme.domain.clip_project import validate_project
from clippyme.finishing import templates
from clippyme.finishing.compile import (
    CompileError,
    FinishingDisabled,
    FinishingParams,
    compile_hyperframes_composition,
    project_digest,
)


def _project(**over):
    data = {
        "schema": "nugget.clip-project/1",
        "job_id": "job1",
        "clip_index": 0,
        "source": {"file": "source_0.mp4", "width": 608, "height": 1080},
        "segments": [
            {"id": "seg0", "start": 2.0, "end": 8.0,
             "crop": {"x": 0, "y": 0, "w": 608, "h": 1080}},
            {"id": "seg1", "start": 10.0, "end": 14.0,
             "crop": {"x": 0, "y": 0, "w": 608, "h": 1080}},
        ],
        "captions": {
            "style": "hormozi",
            "position": "lower-third",
            "words": [
                {"id": "w0", "w": "hello", "start": 2.0, "end": 2.5},
                {"id": "w1", "w": "world", "start": 2.5, "end": 3.0},
                {"id": "w2", "w": "again", "start": 10.5, "end": 11.0},
                {"id": "w3", "w": "dropped", "start": 50.0, "end": 51.0},
            ],
            "edits": {"w1": "WORLD"},
            "next_word_id": 4,
        },
        "overlays": [
            {"type": "hook", "text": "watch this", "style": "viral",
             "start": 0.0, "end": 2.5, "position": "top"},
        ],
        "render": {"width": 608, "height": 1080, "fps": 30},
        "finishing": {"renderer": "hyperframes", "template": "clean-captions"},
        "version": 3,
    }
    data.update(over)
    return validate_project(data)


@pytest.fixture()
def fake_base(monkeypatch, tmp_path):
    """Stub resolve_clean_base: no ffmpeg, deterministic fake base."""
    import clippyme.finishing.compile as comp

    async def _fake(project, job_dir, clip_index):
        fake = tmp_path / "base.mp4"
        fake.write_bytes(b"fake-base-bytes")
        return {
            "path": str(fake),
            "sha256": "f" * 64,
            "provenance": "test-fake",
            "duration": 10.0,  # seg0 (6s) + seg1 (4s)
            "width": 608,
            "height": 1080,
            "fps": 30.0,
        }

    monkeypatch.setattr(comp, "resolve_clean_base", _fake)
    return _fake


def _compile(project, job_dir, **kw):
    kw.setdefault("template_id", "clean-captions")
    kw.setdefault("params", {})
    kw.setdefault("clip_index", 0)
    kw.setdefault("acknowledged_version", project.version)
    kw.setdefault("renderer_enabled", True)
    return asyncio.run(compile_hyperframes_composition(project, job_dir=job_dir, **kw))


# --- template / param gates -------------------------------------------------

def test_unknown_template_rejected():
    with pytest.raises(CompileError, match="unknown template"):
        FinishingParams.from_dict("cinematic-zoom", {})


def test_unknown_param_rejected():
    with pytest.raises(CompileError, match="unknown finishing params"):
        FinishingParams.from_dict("clean-captions", {"zoom": 2})


def test_out_of_enum_param_rejected():
    with pytest.raises(CompileError, match="invalid caption_size"):
        FinishingParams.from_dict("clean-captions", {"caption_size": "huge"})


def test_template_ids_are_the_three_reviewed():
    assert set(templates.TEMPLATE_IDS) == {
        "clean-captions", "emphasized-word", "timed-opening"}
    for tid in templates.TEMPLATE_IDS:
        assert templates.get_template(tid).version == 1


# --- default-OFF rollout -----------------------------------------------------

def test_renderer_off_by_default(fake_base, tmp_path):
    p = _project(finishing=None)
    with pytest.raises(FinishingDisabled):
        _compile(p, str(tmp_path))


def test_renderer_legacy_refused(fake_base, tmp_path):
    p = _project(finishing={"renderer": "legacy", "template": "clean-captions"})
    with pytest.raises(FinishingDisabled):
        _compile(p, str(tmp_path))


def test_renderer_enabled_flag_off_refused(fake_base, tmp_path):
    p = _project()
    with pytest.raises(FinishingDisabled):
        _compile(p, str(tmp_path), renderer_enabled=False)


# --- version identity --------------------------------------------------------

def test_stale_version_rejected(fake_base, tmp_path):
    p = _project()
    with pytest.raises(CompileError, match="stale project version"):
        _compile(p, str(tmp_path), acknowledged_version=p.version - 1)


# --- production time mapping --------------------------------------------------

def test_word_edits_and_removed_words_honored(fake_base, tmp_path):
    p = _project()
    # mark w0 removed via a retained-word edit on the validated project
    p.captions.words[0].removed = True
    compiled = _compile(p, str(tmp_path))
    html = open(compiled.html_path, encoding="utf-8").read()
    # edit applied (production _map_words_to_timeline applies edits by word ID)
    assert "WORLD" in html
    assert ">hello<" not in html  # removed word never rendered
    # out-of-segment word (50s, no segment covers it) is dropped
    assert "dropped" not in html
    # multi-segment mapping: w2 at source 10.5 -> timeline 6.5
    assert 'data-start="6.500"' in html


def test_nonzero_source_offsets_mapped(fake_base, tmp_path):
    p = _project()
    compiled = _compile(p, str(tmp_path))
    html = open(compiled.html_path, encoding="utf-8").read()
    # seg0 starts at source 2.0 -> w0 (source 2.0) maps to timeline 0.0
    assert 'data-start="0.000"' in html
    assert 'data-end="0.500"' in html


# --- hook intervals -----------------------------------------------------------

def test_hook_half_open_interval(fake_base, tmp_path):
    p = _project()
    compiled = _compile(p, str(tmp_path), template_id="timed-opening")
    assert compiled.hook == {"text": "watch this", "start": 0.0, "end": 2.5}
    html = open(compiled.html_path, encoding="utf-8").read()
    assert 'id="hf-hook" data-start="0.000" data-end="2.500"' in html
    assert "watch this" in html


def test_timed_opening_without_hook_fails(fake_base, tmp_path):
    p = _project(overlays=[])
    with pytest.raises(CompileError, match="needs a hook"):
        _compile(p, str(tmp_path), template_id="timed-opening")


def test_timed_opening_prefers_grounded_opening(fake_base, tmp_path):
    p = _project(opening={
        "status": "selected", "option_id": "op1", "text": "grounded hook here",
        "display_start": 0.5, "display_end": 3.0})
    compiled = _compile(p, str(tmp_path), template_id="timed-opening")
    assert compiled.hook["text"] == "grounded hook here"
    assert compiled.hook["start"] == 0.5
    assert compiled.hook["end"] == 3.0


# --- escaping -----------------------------------------------------------------

def test_caption_text_is_escaped(fake_base, tmp_path):
    p = _project()
    p.captions.words[1].w = '<script>alert(1)</script>'
    p.captions.edits = {}
    compiled = _compile(p, str(tmp_path))
    html = open(compiled.html_path, encoding="utf-8").read()
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html
    # the only <script> tags are the reviewed ones (data block + seek impl +
    # preview postMessage bridge)
    assert html.count("<script") == 3
    assert "<script>alert(1)</script>" not in html


def test_hook_text_is_escaped(fake_base, tmp_path):
    p = _project(overlays=[
        {"type": "hook", "text": 'a"b <i>hi</i>', "style": "viral",
         "start": 0.0, "end": 1.0, "position": "top"}])
    compiled = _compile(p, str(tmp_path), template_id="timed-opening")
    html = open(compiled.html_path, encoding="utf-8").read()
    assert "a&quot;b &lt;i&gt;hi&lt;/i&gt;" in html


def test_no_remote_fetches_in_composition(fake_base, tmp_path):
    p = _project()
    compiled = _compile(p, str(tmp_path))
    html = open(compiled.html_path, encoding="utf-8").read()
    assert "http://" not in html
    assert "https://" not in html
    assert "url(http" not in html
    # only local asset references
    assert 'src="base.mp4"' in html
    assert "Anton-Regular.ttf" in html


def test_empty_captions_honest(fake_base, tmp_path):
    data_captions = {"style": "hormozi", "position": "lower-third",
                     "words": [], "edits": {}, "next_word_id": 0}
    p = _project(captions=data_captions, overlays=[])
    compiled = _compile(p, str(tmp_path))
    assert compiled.manifest["captions"] == {"phrases": 0, "words": 0}
    assert compiled.hook is None
    assert os.path.isfile(compiled.html_path)


def test_framing_trajectory_fails_clearly(tmp_path):
    # A project carrying a framing trajectory has no verified clean base:
    # the real clean-base gate must fail clearly, never invent a static crop.
    # (No ffmpeg involved: the gate fires before any materialization.)
    import asyncio
    from types import SimpleNamespace
    from clippyme.finishing.cleanbase import (
        CleanBaseUnavailable, resolve_clean_base)
    p = _project()
    p.framing = SimpleNamespace()  # non-None marker: trajectory present
    with pytest.raises(CleanBaseUnavailable, match="framing trajectory"):
        asyncio.run(resolve_clean_base(p, str(tmp_path), 0))

# --- review-01 group 5: clean base + music honesty ----------------------------------

def test_reframed_base_lineage_refused(tmp_path):
    # Any reframed_base lineage (no framing field) is an honest refusal.
    # The old code silently rematerialized the static segment crop, quietly
    # discarding accepted Task05 lineage framing (tracked/split/screencast
    # pixels).
    import asyncio
    from types import SimpleNamespace
    from clippyme.finishing.cleanbase import (
        CleanBaseUnavailable, resolve_clean_base)
    p = _project()
    p.reframed_base = SimpleNamespace(base_ref="clip_final.mp4")
    with pytest.raises(CleanBaseUnavailable, match="reframed_base"):
        asyncio.run(resolve_clean_base(p, str(tmp_path), 0))


def test_lineage_base_not_silently_replaced_by_static_crop(tmp_path):
    # Landmark regression: the accepted (lineage) base deliberately differs
    # from the static segment crop — the pilot must not render a quiet
    # substitution of the static crop for the accepted base.
    import asyncio
    from clippyme.finishing.cleanbase import (
        CleanBaseUnavailable, resolve_clean_base)
    p = _project()
    p.reframed_base = {
        "base_ref": "clip_final.mp4",
        "segments": [{"id": "seg0",
                      "crop": {"x": 40, "y": 80, "w": 400, "h": 711}}],
    }
    with pytest.raises(CleanBaseUnavailable, match="legacy render path"):
        asyncio.run(resolve_clean_base(p, str(tmp_path), 0))


def test_configured_music_refused(fake_base, tmp_path):
    # A project with configured music fails clearly: the pilot renders
    # source audio only and would silently drop the music mix.
    from clippyme.finishing.cleanbase import CleanBaseUnavailable
    p = _project(audio=[{"file": "music/song.mp3"}])
    with pytest.raises(CleanBaseUnavailable, match="configured music"):
        _compile(p, str(tmp_path))


def test_source_audio_project_unchanged(fake_base, tmp_path):
    # music=None (source-audio) projects are unaffected by the music gate.
    p = _project()
    compiled = _compile(p, str(tmp_path))
    assert os.path.isfile(compiled.html_path)


def test_source_audio_preserved_in_clean_base(tmp_path, monkeypatch):
    # The materialized clean base keeps source audio for music=None projects.
    import asyncio
    import shutil
    import subprocess
    from clippyme.domain import project_render as pr
    from clippyme.finishing.cleanbase import resolve_clean_base
    if not (shutil.which("ffmpeg") and shutil.which("ffprobe")):
        pytest.skip("ffmpeg/ffprobe not on PATH")
    src = tmp_path / "timeline.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error",
         "-f", "lavfi", "-i", "testsrc=size=608x1080:rate=30:duration=2",
         "-f", "lavfi", "-i", "sine=frequency=440:duration=2",
         "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-shortest", str(src)],
        check=True)

    async def _fake_timeline(project, job_dir, clip_index):
        return (str(src), 2.0)

    monkeypatch.setattr(pr, "_materialize_timeline", _fake_timeline)
    p = _project()  # default audio: [AudioTrack()] -> file=None (no music)
    base = asyncio.run(resolve_clean_base(p, str(tmp_path), 0))
    assert base["has_audio"] is True
    assert pr.probe_media(base["path"])["has_audio"] is True


def test_preview_bridge_postmessage_protocol(fake_base, tmp_path):
    # The generated composition carries the shape-validated postMessage
    # bridge (contract 2 backend); the worker engine's __hf object is kept.
    p = _project()
    compiled = _compile(p, str(tmp_path))
    html = open(compiled.html_path, encoding="utf-8").read()
    assert "window.__hf = {" in html  # engine object untouched
    assert "postMessage" in html
    assert "{ hf: 1, type: 'ready' }" in html
    assert "type: 'timeupdate'" in html
    assert "__hf.seek" in html
    assert "video.play()" in html
    assert "video.pause()" in html
    assert "m.hf !== 1" in html  # shape validation
