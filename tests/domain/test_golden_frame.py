"""Golden-frame tests: preview/render parity for caption-affecting edits.

Renders ONE frame CPU-only through the real render path::

    project_render._map_words_to_timeline
        -> subtitles.generate_ass_karaoke
        -> ffmpeg subtitles burn on a blank clip

and compares it against a stored reference PNG.

Comparison is NOT pixel-perfect: a per-channel tolerance
(``PER_CHANNEL_TOL``) plus a max differing-pixel percentage
(``MAX_DIFF_PCT``) absorbs encoder/anti-aliasing noise while still
catching wrong text, wrong timing, or missing captions.

References are generated IN THE CI ENVIRONMENT -- never hand-made::

    pytest tests/domain/test_golden_frame.py --generate-golden

The test pins ``tests/fixtures/fonts/DejaVuSans.ttf`` via the ffmpeg
``subtitles`` filter ``fontsdir`` option so CI and laptop render
identical glyphs.
"""

import os
import shutil
import subprocess

import pytest

from clippyme.domain.clip_project import validate_project
from clippyme.domain.project_render import _map_words_to_timeline
from clippyme.domain.subtitles import generate_ass_karaoke

FIXTURES = os.path.join(os.path.dirname(__file__), "..", "fixtures")
FONTS_DIR = os.path.join(FIXTURES, "fonts")
GOLDEN_DIR = os.path.join(FIXTURES, "golden")
FONT_NAME = "DejaVu Sans"
PER_CHANNEL_TOL = 16
# Tight: the burn pipeline is bit-deterministic (same-input re-render =
# 0.0000% diff), so even a single changed word (0.31% in the negative
# control) fails. References are generated in the same CI env that runs
# the test, so no cross-machine AA variance budget is needed.
MAX_DIFF_PCT = 0.1
W, H = 608, 1080

ffmpeg = shutil.which("ffmpeg")


def _project(words, edits=None, segments=None):
    d = {
        "schema": "nugget.clip-project/1",
        "job_id": "golden",
        "clip_index": 0,
        "source": {"file": "s.mp4", "width": 1920, "height": 1080,
                   "duration": 60.0},
        "segments": segments or [
            {"id": "seg0", "start": 0.0, "end": 60.0,
             "crop": {"x": 0, "y": 0, "w": 1920, "h": 1080}}],
        "captions": {"words": words, "edits": edits or {}},
    }
    return validate_project(d)


def _words(n, start=1.0, step=0.6, dur=0.5, texts=None):
    return [
        {"id": f"w{i}",
         "w": texts[i] if texts else f"word{i}",
         "start": round(start + i * step, 3),
         "end": round(start + i * step + dur, 3)}
        for i in range(n)
    ]


def _seg(s, e, i=0):
    return {"id": f"seg{i}", "start": float(s), "end": float(e),
            "crop": {"x": 0, "y": 0, "w": 1920, "h": 1080}}


def _run(cmd, cwd):
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True,
                       timeout=120)
    assert r.returncode == 0, f"{' '.join(cmd[:3])} failed: {r.stderr[-500:]}"


def _render_frame(project, at, tmp_path):
    """Render one captioned frame at output-timeline time ``at``."""
    mapped, timeline_dur = _map_words_to_timeline(project)
    transcript = {"segments": [{"words": [
        {"word": w["word"], "start": w["start"], "end": w["end"]}
        for w in mapped]}]}
    # Relative paths + cwd dodge ffmpeg's Windows drive-colon escaping in
    # filter args; the pinned font rides alongside for a relative fontsdir.
    generate_ass_karaoke(transcript, 0.0, timeline_dur,
                         os.path.join(str(tmp_path), "t.ass"),
                         preset="classic_white", font_name=FONT_NAME)
    fonts_tmp = os.path.join(str(tmp_path), "fonts")
    os.makedirs(fonts_tmp, exist_ok=True)
    for f in os.listdir(FONTS_DIR):
        shutil.copy(os.path.join(FONTS_DIR, f),
                    os.path.join(fonts_tmp, f))
    _run([ffmpeg, "-y", "-f", "lavfi", "-i",
          f"color=c=black:s={W}x{H}:d={at + 2}:r=30",
          "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
          "blank.mp4"], cwd=tmp_path)
    # NOTE: -ss goes AFTER -i (output seeking). Input seeking snaps to the
    # nearest keyframe, which silently renders the wrong instant.
    _run([ffmpeg, "-y", "-i", "blank.mp4", "-ss", str(at),
          "-vf", "subtitles=t.ass:fontsdir=fonts",
          "-frames:v", "1", "frame.png"], cwd=tmp_path)
    ass_path = os.path.join(str(tmp_path), "t.ass")
    with open(ass_path, encoding="utf-8") as f:
        ass_text = f.read()
    return os.path.join(str(tmp_path), "frame.png"), ass_text


def _diff_pct(frame_path, ref_path):
    from PIL import Image, ImageChops
    a = Image.open(frame_path).convert("RGB")
    b = Image.open(ref_path).convert("RGB")
    assert a.size == b.size, f"size drift: {a.size} vs {b.size}"
    bad = sum(1 for px in ImageChops.difference(a, b).getdata()
              if max(px) > PER_CHANNEL_TOL)
    return 100.0 * bad / (a.size[0] * a.size[1])


def _check_or_generate(name, project, at, tmp_path, generate_golden,
                       expect_text=None):
    ref = os.path.join(GOLDEN_DIR, name + ".png")
    if generate_golden:
        os.makedirs(GOLDEN_DIR, exist_ok=True)
        frame, ass_text = _render_frame(project, at, tmp_path)
        shutil.copy(frame, ref)
        pytest.skip(f"generated golden reference {ref}")
    if not os.path.isfile(ref):
        pytest.skip(f"missing golden reference {ref} "
                    "(run with --generate-golden in CI)")
    frame, ass_text = _render_frame(project, at, tmp_path)
    if expect_text:
        # The edit must reach the burned caption track, not just the pixels.
        assert expect_text in ass_text, (
            f"{name}: {expect_text!r} not in generated ASS")
    pct = _diff_pct(frame, ref)
    assert pct <= MAX_DIFF_PCT, (
        f"{name}: {pct:.2f}% of pixels differ from the golden frame "
        f"(max {MAX_DIFF_PCT}%) -- preview/render drift?")


@pytest.mark.skipif(not ffmpeg, reason="ffmpeg not on PATH")
def test_edited_word_frame(tmp_path, generate_golden):
    """A word edit changes the burned caption (slice 1)."""
    project = _project(_words(4, texts=["hello", "brave", "new", "world"]),
                       edits={"w1": "edited"})
    _check_or_generate("edited_word", project, at=1.7,
                       tmp_path=tmp_path, generate_golden=generate_golden,
                       expect_text="EDITED")  # classic_white uppercases


@pytest.mark.skipif(not ffmpeg, reason="ffmpeg not on PATH")
def test_dropped_span_frame(tmp_path, generate_golden):
    """After a drop, the frame at output t shows the source-shifted word.

    Segments [0,10] + [20,30] (drop [10,20] in SOURCE time): output t=14.5
    is source t=24.5, so the visible caption must be the source-24.5 word.
    """
    words = [
        {"id": "w0", "w": "dropped", "start": 12.0, "end": 12.5},
        {"id": "w1", "w": "keptword", "start": 24.5, "end": 25.0},
    ]
    project = _project(words, segments=[_seg(0, 10), _seg(20, 30, 1)])
    _check_or_generate("dropped_span", project, at=14.5,
                       tmp_path=tmp_path, generate_golden=generate_golden,
                       expect_text="KEPTWORD")
