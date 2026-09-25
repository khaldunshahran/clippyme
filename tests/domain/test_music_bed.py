"""Phase 3B: optional music bed + sidechain ducking infrastructure.

Unit tests for the filtergraph builder and placeholder-bed synthesis; one
integration test that renders a synthetic dialogue clip through the ducking
mix and proves music dips during speech and recovers in gaps (skipped when
ffmpeg is unavailable). The placeholder bed is TEST ONLY — production stays
silent by default.
"""
import shutil
import struct
import subprocess
import wave

import pytest

from clippyme.domain.music_bed import (
    DUCK_FILTERGRAPH_NAME,
    PLACEHOLDER_PREFIX,
    build_duck_filtergraph,
    measure_loudness_lufs,
    mix_music_ducked,
    normalize_final_loudness,
    synthesize_placeholder_bed,
    verify_ducking,
)

FFMPEG = shutil.which("ffmpeg")


def test_filtergraph_shape():
    fg = build_duck_filtergraph()
    assert "[1:a]" in fg and "[0:a]" in fg
    assert "sidechaincompress" in fg
    assert "[mixout]" in fg
    assert "amix" in fg
    assert DUCK_FILTERGRAPH_NAME == "music_duck"  # stable id for logs/metrics
    assert "release=400" in fg  # gentle recovery, no pumping


def test_filtergraph_debug_mode():
    fg = build_duck_filtergraph(debug=True)
    assert "[ducked_dbg]" in fg
    assert "anullsink" not in fg
    fg = build_duck_filtergraph(debug=False)
    assert "[ducked_dbg]anullsink" in fg  # unused tap terminated: no dangling output
    assert fg.count("[ducked_dbg]") == 2  # asplit label + anullsink consumer


def test_placeholder_synthesis_shape(tmp_path):
    out = str(tmp_path / "bed.wav")
    path = synthesize_placeholder_bed(out, duration_s=3.0, sample_rate=22050)
    assert PLACEHOLDER_PREFIX in path
    with wave.open(path, "rb") as wf:
        assert wf.getnchannels() == 2
        assert wf.getframerate() == 22050
        frames = wf.getnframes()
        assert abs(frames / 22050 - 3.0) < 0.1
        # fade-in: first 100ms stays quiet — no DC-offset click at start
        raw = wf.readframes(2205)
        first = struct.unpack("<%dh" % (len(raw) // 2), raw)
        assert max(abs(s) for s in first) < 1500


def _make_dialogue_clip(path, duration_s=6.0):
    """Synthetic 'dialogue': 1.5s speech-ish tone + 1.5s silence, x2."""
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error",
         "-f", "lavfi", "-i", "sine=frequency=220:duration=1.5",
         "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo:d=1.5",
         "-filter_complex",
         "[0:a]aformat=channel_layouts=stereo[t];"
         "[t][1:a]concat=n=2:v=0:a=1[half];"
         "[half]asplit[a][b];[a][b]concat=n=2:v=0:a=1[out]",
         "-map", "[out]", "-t", str(duration_s), "-ar", "48000",
         "-c:a", "pcm_s16le", path],
        check=True, timeout=60)


def _music_rms(path, start, dur):
    import re
    out = subprocess.run(
        ["ffmpeg", "-v", "info", "-ss", str(start), "-t", str(dur),
         "-i", path, "-af", "astats=reset=1",
         "-f", "null", "-"],
        capture_output=True, text=True, timeout=60)
    levels = [float(m) for m in re.findall(r"RMS level dB:\s*([-.\d]+)", out.stderr)]
    return levels[-1] if levels else float("nan")  # Overall printed last


@pytest.mark.skipif(FFMPEG is None, reason="ffmpeg not available")
def test_ducking_mix_measurable(tmp_path):
    """Music RMS during speech gaps vs speech must differ (ducking engaged)."""
    dialogue = str(tmp_path / "dialogue.wav")
    bed = synthesize_placeholder_bed(str(tmp_path / "placeholder_bed.wav"),
                                     duration_s=6.0)
    out = str(tmp_path / "mixed.wav")
    _make_dialogue_clip(dialogue)
    # debug=True taps the ducked bed stem: measuring the full mix would let
    # the dialogue mask the bed dip
    mix_music_ducked(dialogue, bed, out, bed_gain_db=-10.0, debug=True)
    stem = out + ".ducked_dbg.wav"
    # dialogue = [1.5s tone][1.5s silence] x2: speech 0-1.5 -> ducked,
    # gaps 1.5-3.0 / 4.5-6.0 -> bed audible (release=400ms recovered)
    rms_speech = _music_rms(stem, 0.5, 0.8)
    rms_gap = _music_rms(stem, 2.0, 0.8)
    assert rms_gap > rms_speech + 2.0, (rms_speech, rms_gap)  # >=2dB ducking
    assert rms_gap > -60.0  # music still audible in the gaps
    final = normalize_final_loudness(out, target_lufs=-14.0)
    assert final is True
    loud = measure_loudness_lufs(out)
    assert loud is not None and abs(loud - (-14.0)) <= 1.0


@pytest.mark.skipif(FFMPEG is None, reason="ffmpeg not available")
def test_verify_ducking_stats(tmp_path):
    dialogue = str(tmp_path / "dialogue.wav")
    bed = synthesize_placeholder_bed(str(tmp_path / "placeholder_bed.wav"),
                                     duration_s=6.0)
    out = str(tmp_path / "mixed.wav")
    _make_dialogue_clip(dialogue)
    mix_music_ducked(dialogue, bed, out, debug=True)
    report = verify_ducking(out + ".ducked_dbg.wav",
                            speech_windows=[(0.2, 1.3)],
                            gap_windows=[(2.0, 2.8), (4.7, 5.7)])
    assert report["ducking_db"] >= 2.0
    assert report["gap_rms_db"] > -60.0
