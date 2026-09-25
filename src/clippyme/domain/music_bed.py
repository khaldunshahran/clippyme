"""Phase 3B: optional music-bed infrastructure with sidechain ducking.

OpusClip-level channels mix a subtle music bed under dialogue with the music
ducked whenever speech is present. This module provides the full mechanism —
ffmpeg ``sidechaincompress`` ducking, post-mix -14 LUFS verification — while
keeping production behaviour unchanged: no music is ever mixed unless the
caller explicitly passes a bed, and the only bed this module can synthesize
is a clearly-marked PLACEHOLDER for tests/verification.

    TEST-ONLY: ``synthesize_placeholder_bed`` generates a neutral synth chord
    pad. It is NOT licensed production music. Real beds come from the
    (still open) music catalog/licensor decision; wiring one in = passing its
    path to ``mix_music_ducked``.
"""
from __future__ import annotations

import logging
import math
import os
import re
import shutil
import struct
import subprocess
import wave
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

PLACEHOLDER_PREFIX = "placeholder_"
DUCK_FILTERGRAPH_NAME = "music_duck"

TARGET_LUFS = -14.0
FINAL_LOUDNESS_TOLERANCE = 1.0

# Ducking character: speech gate opens on modest dialogue energy, gentle
# 2:1-ish compression, medium attack, slow release so the bed breathes back
# instead of pumping.
_DUCK_THRESHOLD = "0.02"   # sidechain detection threshold
_DUCK_RATIO = "4"
_DUCK_ATTACK = "20"        # ms
_DUCK_RELEASE = "400"      # ms — gentle recovery, no pumping
_SPEECH_GATE_NOTE = "athreshold=0.008"


def build_duck_filtergraph(debug: bool = False) -> str:
    """ffmpeg -filter_complex graph: sidechain-ducked music + dialogue mix.

    Inputs: [0:a] dialogue (sidechain key), [1:a] music bed.
    Output: [mixout]. In debug mode a second output [ducked_dbg] taps the
    ducked music pre-mix; otherwise the tap is terminated with anullsink so
    ffmpeg never fails on an unconnected filtergraph output.
    """
    # debug tap: a dangling [ducked_dbg] output label the caller can -map.
    # non-debug: the tap gets its own filterchain into anullsink so ffmpeg
    # never fails on an unconnected filtergraph output.
    tap_chain = "" if debug else ";[ducked_dbg]anullsink"
    return (
        # music bed -> gain staging -> sidechain compressor keyed by dialogue
        f"[1:a]volume=1.0[bed];"
        f"[bed][0:a]sidechaincompress=threshold={_DUCK_THRESHOLD}"
        f":ratio={_DUCK_RATIO}:attack={_DUCK_ATTACK}:release={_DUCK_RELEASE}"
        f"[ducked];"
        # tap for measurement/debugging (terminated unless debug=True)
        f"[ducked]asplit=2[ducked_mix][ducked_dbg]{tap_chain};"
        # final mix: dialogue untouched, ducked bed underneath
        f"[0:a][ducked_mix]amix=inputs=2:normalize=0[mixout]"
    )


def synthesize_placeholder_bed(
    out_path: str,
    duration_s: float = 10.0,
    sample_rate: int = 44100,
) -> str:
    """Synthesize a neutral chord-pad placeholder bed (TEST ONLY).

    NOT licensed production music — a soft synth pad (root+fifth+octave,
    slow attack, fade in/out) meant only for ducking-mix verification and
    unit tests. The filename always carries the ``placeholder_`` prefix so a
    placeholder can never be mistaken for a real licensed bed.
    """
    logger.warning("synthesize_placeholder_bed: TEST-ONLY placeholder, "
                   "not licensed production music")
    directory = os.path.dirname(out_path)
    base = os.path.basename(out_path)
    if not base.startswith(PLACEHOLDER_PREFIX):
        base = PLACEHOLDER_PREFIX + base
    final_path = os.path.join(directory, base) if directory else base

    n = int(duration_s * sample_rate)
    # Am - F - C - G-ish pad: gentle, no strong melody to distract
    chords = [(220.0, 261.63, 329.63), (174.61, 220.0, 261.63),
              (130.81, 164.81, 196.0), (196.0, 246.94, 293.66)]
    chord_s = duration_s / len(chords)
    amp = 0.18
    frames = bytearray()
    for i in range(n):
        t = i / sample_rate
        chord = chords[int(t / chord_s) % len(chords)]
        sample = sum(math.sin(2 * math.pi * f * t) for f in chord) / len(chords[0])
        # slow attack per chord + global fade in/out to avoid clicks
        local = (t % chord_s) / chord_s
        env = min(1.0, local * 4.0) * min(1.0, (1.0 - local) * 8.0 + 0.2)
        edge = min(1.0, t / 0.5, (duration_s - t) / 0.5)
        v = int(max(-1.0, min(1.0, sample * amp * env * edge)) * 32767)
        frames += struct.pack("<hh", v, v)
    with wave.open(final_path, "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(bytes(frames))
    return final_path


def _run(cmd: List[str], timeout: int = 300) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def mix_music_ducked(
    dialogue_path: str,
    bed_path: str,
    out_path: str,
    bed_gain_db: float = -12.0,
    debug: bool = False,
) -> str:
    """Mix ``bed_path`` under ``dialogue_path`` with sidechain ducking.

    Dialogue passes through untouched; the bed is ducked whenever speech
    energy crosses the gate. Returns ``out_path``.
    """
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg not available for music-bed mixing")
    fg = build_duck_filtergraph(debug=debug)
    if bed_gain_db != 0.0:
        gain = 10.0 ** (bed_gain_db / 20.0)
        fg = fg.replace("[1:a]volume=1.0[bed];",
                        f"[1:a]volume={gain:.4f}[bed];")
    cmd = [ffmpeg, "-y", "-v", "error",
           "-i", dialogue_path,
           "-i", bed_path,
           "-filter_complex", fg]
    if debug:
        # debug tap becomes a second output; each output gets its own -map
        # (ffmpeg applies -map to the NEXT output file, in order)
        cmd += ["-map", "[mixout]", "-c:a", "pcm_s16le", out_path,
                "-map", "[ducked_dbg]", "-c:a", "pcm_s16le",
                out_path + ".ducked_dbg.wav"]
    else:
        cmd += ["-map", "[mixout]", "-c:a", "pcm_s16le", out_path]
    result = _run(cmd)
    if result.returncode != 0 or not os.path.exists(out_path):
        raise RuntimeError(f"music ducking mix failed: {result.stderr[-500:]}")
    return out_path


def measure_loudness_lufs(path: str) -> Optional[float]:
    """Integrated loudness via ffmpeg ebur128, or None when unavailable."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return None
    result = _run([ffmpeg, "-v", "info", "-i", path,
                   "-map", "0:a", "-af", "ebur128=peak=true",
                   "-f", "null", "-"])
    text = result.stderr or ""
    matches = re.findall(r"I:\s*(-?\d+(?:\.\d+)?)\s*LUFS", text)
    if not matches:
        return None
    try:
        return float(matches[-1])
    except ValueError:
        return None


def normalize_final_loudness(path: str, target_lufs: float = TARGET_LUFS) -> bool:
    """Post-mix EBU R128 normalization reusing the existing postprocess helper.

    Runs AFTER the visual compose + music mix so the shipped file — not an
    intermediate — is what hits the loudness target. Returns True only when
    the measured integrated loudness lands within ±1 LUFS of the target.

    NOTE: the real ``postprocess.normalize_audio`` is fixed at -14 LUFS
    (social standard), takes no target argument, and returns None — the
    local test stub did take/return. ``TARGET_LUFS`` is -14.0, so the
    default call is exact; a non-default target logs a warning.
    """
    from clippyme.pipeline.postprocess import normalize_audio
    if target_lufs != TARGET_LUFS:
        logger.warning(
            "normalize_final_loudness: postprocess.normalize_audio is fixed "
            "at -14 LUFS; requested target %s ignored", target_lufs)
    normalize_audio(path)
    measured = measure_loudness_lufs(path)
    if measured is None:
        logger.warning(
            "normalize_final_loudness: could not measure loudness of %s", path)
        return False
    ok = abs(measured - TARGET_LUFS) <= FINAL_LOUDNESS_TOLERANCE
    if not ok:
        logger.warning(
            "normalize_final_loudness: %s measures %.1f LUFS, target %.1f",
            path, measured, TARGET_LUFS)
    return ok


def _rms_db(path: str, start: float, dur: float) -> float:
    # astats prints its summary at info level; the "Overall" RMS line is
    # printed last (after the per-channel lines), so take the last match.
    result = _run(["ffmpeg", "-v", "info",
                   "-ss", str(start), "-t", str(dur), "-i", path,
                   "-af", "astats=reset=1", "-f", "null", "-"])
    levels = [float(m) for m in re.findall(r"RMS level dB:\s*(-?[\d.]+)",
                                          result.stderr or "")]
    return levels[-1] if levels else float("nan")


def verify_ducking(
    bed_audio_path: str,
    speech_windows: List[Tuple[float, float]],
    gap_windows: List[Tuple[float, float]],
) -> Dict[str, float]:
    """Quantify the duck: mean RMS in speech windows vs gap windows.

    ``bed_audio_path`` is the ducked BED stem (the ``[ducked_dbg]`` tap),
    not the full mix — measuring the full mix would let the dialogue mask
    the bed dip. Returns ``ducking_db`` (positive = music dips during
    speech) and ``gap_rms_db`` (must stay comfortably above silence: music
    audible).
    """
    def _mean(windows):
        vals = [_rms_db(bed_audio_path, s, e - s) for s, e in windows]
        vals = [v for v in vals if v == v]  # drop NaN
        return sum(vals) / len(vals) if vals else float("nan")

    speech = _mean(speech_windows)
    gap = _mean(gap_windows)
    return {
        "speech_rms_db": speech,
        "gap_rms_db": gap,
        "ducking_db": (gap - speech) if gap == gap and speech == speech else float("nan"),
        "speech_gate_note": _SPEECH_GATE_NOTE,
    }
