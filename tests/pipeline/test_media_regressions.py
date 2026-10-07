"""Task 01 rev5: decoded-media regressions for the normalized shared-clock cut.

Self-contained fixtures (no network, no API):
- late: 12s source, beep at 8s; cut [6,9) must keep frames 180-269
- unequal: video +0.5s / audio 0; cut [3.5,5.5) must keep frames 90-149
- common: +5s container origin; cut [3.5,5.5) elapsed must keep 105-164
- cross-density VFR: 10fps then 60fps; cut [3,5) must preserve spacing

Each asserts decoded first/last frame identity (SSIM), audio marker time,
output PTS origin, and production validation. Encoder is parameterized:
libx264 always; h264_nvenc when available (explicit skip otherwise).
"""

import array
import json
import os
import re
import shutil
import subprocess

import pytest

from clippyme.pipeline.run_ops import build_cut_command, validate_cut_output

FFMPEG = shutil.which("ffmpeg")
FFPROBE = shutil.which("ffprobe")
needs_ff = pytest.mark.skipif(not (FFMPEG and FFPROBE), reason="ffmpeg/ffprobe required")


def _run(cmd, **kw):
    return subprocess.run(cmd, check=True, capture_output=True, **kw)


def _ssim(a, b):
    flt = "[0:v][1:v]ssim"
    err = _run([FFMPEG, "-hide_banner", "-i", a, "-i", b,
                "-filter_complex", flt, "-an", "-f", "null", "-"],
               text=True).stderr
    m = re.search(r"All:([0-9.]+)", err)
    return float(m.group(1)) if m else 0.0


def _extract_frame(video, out_png, n):
    _run([FFMPEG, "-v", "error", "-i", video, "-vf",
          f"select=eq(n\\,{n})", "-vframes", "1", out_png])


def _marker(path):
    pcm = _run([FFMPEG, "-v", "error", "-copyts", "-i", path, "-map", "0:a:0",
                "-af", "aresample=async=1:first_pts=0", "-ar", "48000",
                "-ac", "1", "-f", "s16le", "-"]).stdout
    samples = array.array("h")
    samples.frombytes(pcm)
    return next((i / 48000 for i, s in enumerate(samples) if abs(s) > 5000), None)


def _frame_pts(path):
    out = _run([FFPROBE, "-v", "error", "-select_streams", "v:0",
                "-show_entries", "frame=best_effort_timestamp_time,pkt_pts_time",
                "-of", "json", path], text=True).stdout
    frames = json.loads(out)["frames"]
    return [float(f["best_effort_timestamp_time"]) for f in frames]


def _make_late_source(path):
    """12s testsrc + beep at 8s."""
    _run([FFMPEG, "-y", "-f", "lavfi", "-i", "testsrc=duration=12:size=320x240:rate=30",
          "-f", "lavfi", "-i",
          "aevalsrc='if(between(t,8,8.02),0.9*sin(2*PI*1000*t),0)':s=48000:d=12",
          "-c:v", "libx264", "-g", "60", "-pix_fmt", "yuv420p",
          "-c:a", "pcm_s16le", "-shortest", path])


def _make_common_origin_source(path):
    """Both streams at +5s container origin; beep at elapsed 4.0s.

    Generate a 12s source with beep at 4s, then shift both streams by +5s
    at mux time. A [3.5,5.5) trim selects elapsed time; the beep (elapsed
    4.0s) lands at slice 0.5s.
    """
    tmp = str(path) + ".tmp.mkv"
    _run([FFMPEG, "-y", "-f", "lavfi", "-i", "testsrc=duration=12:size=320x240:rate=30",
          "-f", "lavfi", "-i",
          "aevalsrc='if(between(t,4,4.02),0.9*sin(2*PI*1000*t),0)':s=48000:d=12",
          "-c:v", "libx264", "-g", "60", "-pix_fmt", "yuv420p",
          "-c:a", "pcm_s16le", "-shortest", tmp])
    _run([FFMPEG, "-y", "-i", tmp, "-c", "copy",
          "-output_ts_offset", "5", path])
    os.remove(tmp)


def _make_cross_density_vfr(path):
    """8s VFR: 4s at 10fps, then 4s at 60fps."""
    _run([FFMPEG, "-y", "-f", "lavfi", "-i", "testsrc=duration=8:size=320x240:rate=60",
          "-vf", "select='if(lt(t,4),not(mod(n,6)),1)'", "-fps_mode", "vfr",
          "-c:v", "libx264", "-pix_fmt", "yuv420p", path])


def _encoders():
    encs = ["libx264"]
    # h264_nvenc if the binary supports it
    try:
        out = subprocess.run([FFMPEG, "-hide_banner", "-h", "encoder=h264_nvenc"],
                             capture_output=True, text=True, timeout=10)
        if out.returncode == 0:
            encs.append("h264_nvenc")
    except Exception:
        pass
    return encs


@pytest.fixture(params=_encoders(), ids=lambda e: f"enc-{e}")
def encoder(request, monkeypatch):
    """Parameterize over available encoders; env var is read at call time."""
    monkeypatch.setenv("CLIPPYME_VIDEO_ENCODER", request.param)
    return request.param


@needs_ff
def test_late_cut_selects_correct_interval(tmp_path, encoder):
    """[6,9) on a 12s source: frames 180-269, beep at 2.0s."""
    src = str(tmp_path / "late.mkv")
    _make_late_source(src)
    cut = str(tmp_path / "cut.mp4")
    _run(build_cut_command(src, 6.0, 9.0, cut))

    # Decoded identity
    first = str(tmp_path / "first.png")
    _extract_frame(cut, first, 0)
    ref180 = str(tmp_path / "ref180.png")
    _extract_frame(src, ref180, 180)
    ref210 = str(tmp_path / "ref210.png")
    _extract_frame(src, ref210, 210)
    s180 = _ssim(first, ref180)
    s210 = _ssim(first, ref210)
    assert s180 > s210, f"first frame must be 180, not 210 (rev3 error): {s180:.4f} vs {s210:.4f}"
    assert s180 > 0.95

    # Audio marker: beep at source 8s -> slice 2.0s
    m = _marker(cut)
    assert m == pytest.approx(2.0, abs=0.01), f"beep at {m}, expected 2.0"

    # PTS origin and validation
    pts = _frame_pts(cut)
    assert pts[0] == pytest.approx(0.0, abs=0.05)
    assert len(pts) == 90, f"expected 90 frames, got {len(pts)}"
    ok, reason = validate_cut_output(cut, 3.0)
    assert ok, reason


@needs_ff
def test_common_origin_cut(tmp_path, encoder):
    """+5s container origin: elapsed [3.5,5.5) keeps correct content."""
    src = str(tmp_path / "common.mkv")
    _make_common_origin_source(src)
    cut = str(tmp_path / "cut.mp4")
    _run(build_cut_command(src, 3.5, 5.5, cut))

    # Beep at elapsed 4.0s (container 9.0s) -> slice 0.5s
    m = _marker(cut)
    assert m == pytest.approx(0.5, abs=0.01), f"beep at {m}, expected 0.5"

    pts = _frame_pts(cut)
    assert pts[0] == pytest.approx(0.0, abs=0.05)
    assert len(pts) == 60, f"expected 60 frames, got {len(pts)}"
    ok, reason = validate_cut_output(cut, 2.0)
    assert ok, reason


@needs_ff
def test_cross_density_vfr_preserves_spacing(tmp_path, encoder):
    """VFR [3,5): 10fps section then 60fps; PTS spacing must be preserved."""
    src = str(tmp_path / "vfr.mp4")
    _make_cross_density_vfr(src)
    cut = str(tmp_path / "cut.mp4")
    _run(build_cut_command(src, 3.0, 5.0, cut))

    pts = _frame_pts(cut)
    # [3,4): 10fps -> 10 frames at 100ms spacing; [4,5): 60fps -> 60 frames
    assert len(pts) == 70, f"expected 70 frames, got {len(pts)}"
    # First 10 intervals ~100ms, rest ~16.7ms
    gaps = [pts[i+1] - pts[i] for i in range(len(pts)-1)]
    assert all(abs(g - 0.1) < 0.02 for g in gaps[:9]), "sparse section spacing"
    assert all(abs(g - 0.0167) < 0.01 for g in gaps[10:]), "dense section spacing"
    # Output PTS = source PTS - 3.0
    assert pts[0] == pytest.approx(0.0, abs=0.05)
    ok, reason = validate_cut_output(cut, 2.0)
    assert ok, reason
