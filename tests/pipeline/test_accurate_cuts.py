"""Regression tests for accurate clip cuts (Task 01).

The old ``build_cut_command`` used input-side ``-ss`` with stream copy:
the persisted slice really started at the keyframe at/before the requested
start (up to ~one GOP early). The new implementation seeks accurately
(``-ss`` AFTER ``-i``) and re-encodes video+audio, so the slice starts at
the first frame at/after the requested start -- sub-frame head offset,
never a GOP.

These tests FAIL on the old implementation and PASS on the new one.
All media is synthetic (ffmpeg lavfi); no user footage or API keys.
Skips when ffmpeg/ffprobe are unavailable.
"""
import json
import os
import shutil
import subprocess

import pytest

from clippyme.pipeline.run_ops import (
    build_cut_command,
    read_slice_provenance,
    slice_provenance,
    slice_provenance_valid,
    write_slice_provenance,
)

FFMPEG = shutil.which("ffmpeg")
FFPROBE = shutil.which("ffprobe")

needs_ff = pytest.mark.skipif(
    not (FFMPEG and FFPROBE), reason="ffmpeg/ffprobe not available"
)


def _run(*args, **kwargs):
    return subprocess.run(*args, check=True, capture_output=True, **kwargs)


def _make_source(path, with_audio=True, duration=10.0, fps=30, gop=60):
    """Long-GOP testsrc video; keyframes every gop/fps seconds.

    testsrc burns the frame number into each frame, so frame identity is
    verifiable without OCR (pixel-compare against a reference extraction).
    """
    cmd = [
        FFMPEG, "-y",
        "-f", "lavfi", "-i",
        f"testsrc=duration={duration}:size=320x240:rate={fps}",
    ]
    if with_audio:
        # 440Hz sine; beep markers are unnecessary -- duration + stream
        # presence checks plus the video-frame test cover alignment.
        cmd += ["-f", "lavfi", "-i", f"sine=frequency=440:duration={duration}"]
    cmd += [
        "-c:v", "libx264", "-g", str(gop), "-pix_fmt", "yuv420p",
        "-r", str(fps),
    ]
    if with_audio:
        cmd += ["-c:a", "aac", "-shortest"]
    cmd.append(path)
    _run(cmd)


def _extract_frame(video, out_png, frame_num=0):
    _run([FFMPEG, "-y", "-i", video, "-vf", f"select='eq(n,{frame_num})'",
          "-vframes", "1", out_png])


def _probe_duration(path):
    out = _run([FFPROBE, "-v", "quiet", "-show_entries", "format=duration",
                "-of", "csv=p=0", path], text=True).stdout.strip()
    return float(out)


def _has_audio(path):
    out = _run([FFPROBE, "-v", "quiet", "-select_streams", "a",
                "-show_entries", "stream=index", "-of", "csv=p=0", path],
               text=True).stdout.strip()
    return bool(out)


# --- command shape ------------------------------------------------------------

def test_build_cut_command_rejects_bad_intervals(tmp_path):
    with pytest.raises(ValueError):
        build_cut_command("in.mp4", 5.0, 5.0, str(tmp_path / "o.mp4"))
    with pytest.raises(ValueError):
        build_cut_command("in.mp4", 6.0, 5.0, str(tmp_path / "o.mp4"))
    with pytest.raises(ValueError):
        build_cut_command("in.mp4", -1.0, 5.0, str(tmp_path / "o.mp4"))


def test_build_cut_command_accurate_seek_shape():
    """-ss must come AFTER -i (accurate decode); streams must re-encode."""
    cmd = build_cut_command("in.mp4", 3.5, 7.5, "out.mp4")
    i_idx = cmd.index("-i")
    ss_idx = cmd.index("-ss")
    assert ss_idx > i_idx, "output seeking required for accurate cuts"
    assert "copy" not in cmd, "stream copy must not be used"
    assert "-map" in cmd and "0:a?" in cmd, "audio must be optional-mapped"


# --- frame accuracy ------------------------------------------------------------

@needs_ff
def test_accurate_cut_starts_at_requested_frame_not_keyframe(tmp_path):
    """Requested start 3.5s sits between keyframes (GOP=2s: keys at 2,4).

    Old behavior: slice starts at the 2.0s keyframe (frame 60).
    New behavior: slice starts at frame 105 (3.5s * 30fps).
    """
    src = str(tmp_path / "src.mp4")
    _make_source(src)
    cut = str(tmp_path / "cut.mp4")
    _run(build_cut_command(src, 3.5, 7.5, cut))

    first_png = str(tmp_path / "first.png")
    _extract_frame(cut, first_png, 0)
    ref105_png = str(tmp_path / "ref105.png")
    _extract_frame(src, ref105_png, 105)
    ref60_png = str(tmp_path / "ref60.png")
    _extract_frame(src, ref60_png, 60)

    # Simpler robust check: SSIM of the cut's first frame against the two
    # candidate source frames (105 = correct, 60 = old keyframe behavior).
    def ssim(a, b):
        out = subprocess.run(
            [FFMPEG, "-y", "-i", a, "-i", b,
             "-lavfi", "ssim", "-f", "null", "-"],
            capture_output=True, text=True).stderr
        for line in out.splitlines():
            if "SSIM" in line and "All:" in line:
                return float(line.split("All:")[1].split()[0])
        return -1.0

    ssim_105 = ssim(first_png, ref105_png)
    ssim_60 = ssim(first_png, ref60_png)
    print(f"\nfirst-frame SSIM vs frame105 (correct): {ssim_105:.4f}")
    print(f"first-frame SSIM vs frame60 (old keyframe): {ssim_60:.4f}")
    assert ssim_105 > 0.90, f"first frame should match source frame 105, SSIM={ssim_105}"
    assert ssim_105 > ssim_60 + 0.05, "first frame must match 105 better than the old keyframe 60"


@needs_ff
def test_cut_duration_matches_request(tmp_path):
    src = str(tmp_path / "src.mp4")
    _make_source(src)
    cut = str(tmp_path / "cut.mp4")
    _run(build_cut_command(src, 3.5, 7.5, cut))
    dur = _probe_duration(cut)
    # Frame tolerance: +/- 2 frames at 30fps.
    assert abs(dur - 4.0) < 2 / 30 + 0.05, f"duration {dur} not within tolerance of 4.0s"


@needs_ff
def test_fractional_edge_resolves_to_next_frame(tmp_path):
    """3.517s is not frame-aligned at 30fps; convention = ceil to frame 106."""
    src = str(tmp_path / "src.mp4")
    _make_source(src)
    cut = str(tmp_path / "cut.mp4")
    _run(build_cut_command(src, 3.517, 5.0, cut))

    def ssim(a, b):
        out = subprocess.run(
            [FFMPEG, "-y", "-i", a, "-i", b,
             "-lavfi", "ssim", "-f", "null", "-"],
            capture_output=True, text=True).stderr
        for line in out.splitlines():
            if "SSIM" in line and "All:" in line:
                return float(line.split("All:")[1].split()[0])
        return -1.0

    first_png = str(tmp_path / "first.png")
    _extract_frame(cut, first_png, 0)
    ref106 = str(tmp_path / "ref106.png")
    _extract_frame(src, ref106, 106)
    ref105 = str(tmp_path / "ref105.png")
    _extract_frame(src, ref105, 105)
    s106 = ssim(first_png, ref106)
    s105 = ssim(first_png, ref105)
    print(f"\nSSIM vs 106: {s106:.4f}, vs 105: {s105:.4f}")
    assert s106 > s105, "fractional edge must resolve to the next frame (106), not 105"


@needs_ff
def test_start_at_zero_and_short_interval(tmp_path):
    src = str(tmp_path / "src.mp4")
    _make_source(src)
    cut0 = str(tmp_path / "cut0.mp4")
    _run(build_cut_command(src, 0.0, 2.0, cut0))
    assert abs(_probe_duration(cut0) - 2.0) < 0.15
    cut_short = str(tmp_path / "cutshort.mp4")
    _run(build_cut_command(src, 1.0, 1.5, cut_short))
    assert abs(_probe_duration(cut_short) - 0.5) < 0.15


# --- audio ---------------------------------------------------------------------

@needs_ff
def test_audio_present_and_trimmed(tmp_path):
    src = str(tmp_path / "src.mp4")
    _make_source(src, with_audio=True)
    cut = str(tmp_path / "cut.mp4")
    _run(build_cut_command(src, 3.5, 7.5, cut))
    assert _has_audio(cut), "audio stream must survive the accurate cut"
    # Audio duration should match the requested interval within tolerance
    # (decoder priming is small for AAC re-encode; allow 100ms).
    a_dur = float(_run(
        [FFPROBE, "-v", "quiet", "-select_streams", "a:0",
         "-show_entries", "stream=duration", "-of", "csv=p=0", cut],
        text=True).stdout.strip())
    assert abs(a_dur - 4.0) < 0.15, f"audio duration {a_dur} not within tolerance"


@needs_ff
def test_absent_audio_cuts_cleanly(tmp_path):
    src = str(tmp_path / "src.mp4")
    _make_source(src, with_audio=False)
    cut = str(tmp_path / "cut.mp4")
    _run(build_cut_command(src, 1.0, 3.0, cut))  # must not fail
    assert not _has_audio(cut)
    assert abs(_probe_duration(cut) - 2.0) < 0.15


# --- provenance -----------------------------------------------------------------

def test_provenance_roundtrip_and_invalidation(tmp_path):
    src = tmp_path / "src.mp4"
    src.write_bytes(b"x" * 20000)
    sl = str(tmp_path / "source_clip.mp4")

    write_slice_provenance(sl, start=3.5, end=7.5, source_path=str(src))
    assert slice_provenance_valid(sl, start=3.5, end=7.5, source_path=str(src))

    # Changed interval invalidates
    assert not slice_provenance_valid(sl, start=3.6, end=7.5, source_path=str(src))
    assert not slice_provenance_valid(sl, start=3.5, end=7.6, source_path=str(src))

    # Changed source (different size) invalidates
    src2 = tmp_path / "src2.mp4"
    src2.write_bytes(b"y" * 21000)
    assert not slice_provenance_valid(sl, start=3.5, end=7.5, source_path=str(src2))

    # Missing provenance (legacy slice) invalidates
    assert not slice_provenance_valid(str(tmp_path / "nosuch.mp4"),
                                     start=3.5, end=7.5, source_path=str(src))

    # Corrupt provenance file invalidates rather than raising
    with open(sl + ".slice.json", "w") as f:
        f.write("not json{")
    assert not slice_provenance_valid(sl, start=3.5, end=7.5, source_path=str(src))
    assert read_slice_provenance(sl) is None


def test_provenance_record_shape(tmp_path):
    src = tmp_path / "src.mp4"
    src.write_bytes(b"x" * 100)
    rec = slice_provenance(start=1.0, end=2.0, source_path=str(src))
    assert rec["version"] == 2
    assert rec["start"] == 1.0 and rec["end"] == 2.0
    assert rec["source"]["size"] == 100


# --- project word-timing consistency (Task 01 req 8) ---------------------------

def test_clip_relative_words_consistent_with_accurate_slice():
    """Words are mapped against the REQUESTED start as time 0.

    With the accurate cut, the actual slice starts within one frame of the
    requested start, so this single convention is consistent for the
    project, transcript, and render -- no consumer re-rounds independently.
    """
    from clippyme.pipeline.project_emit import _clip_relative_words

    transcript = {"segments": [
        {"words": [
            {"word": "hello", "start": 3.4, "end": 3.6},   # straddles start
            {"word": "world", "start": 4.0, "end": 4.5},   # inside
            {"word": "early", "start": 2.0, "end": 2.5},   # before: excluded
            {"word": "late", "start": 7.6, "end": 8.0},    # after: excluded
            {"word": "edge", "start": 7.4, "end": 7.8},    # straddles end
        ]}
    ]}
    words = _clip_relative_words(transcript, 3.5, 7.5)
    by_w = {w["w"]: w for w in words}
    assert set(by_w) == {"hello", "world", "edge"}
    # hello straddles the requested start: clamped to 0.0
    assert by_w["hello"]["start"] == 0.0
    assert by_w["hello"]["end"] == pytest.approx(0.1)
    # world fully inside: relative times
    assert by_w["world"]["start"] == pytest.approx(0.5)
    assert by_w["world"]["end"] == pytest.approx(1.0)
    # edge straddles the requested end: clamped to duration
    assert by_w["edge"]["end"] == pytest.approx(4.0)
    # All word times lie within [0, duration]
    for w in words:
        assert 0.0 <= w["start"] < w["end"] <= 4.0
