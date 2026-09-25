"""Phase 2 (2D): punch-in peak detection + zoom expression helpers.

Pure tests — no ffmpeg needed (decode failures are covered by the
never-raise contract of detect_energy_peaks).
"""
import pytest

from clippyme.pipeline.media_probe import (
    PUNCH_IN_MAX,
    detect_energy_peaks,
    pick_energy_peaks,
    punch_zoom_suffix,
)


def _series(dbs, step=0.25):
    return list(dbs), [i * step for i in range(len(dbs))]


def test_pick_energy_peaks_spacing_and_cap():
    levels, times = _series(
        [-20, -19, -8, -19, -20, -7, -19, -20, -19, -18, -20, -19.5, -6])
    assert pick_energy_peaks(levels, times) == [0.5, 3.0]
    # -7@1.25 suppressed: only 1.75s from the taller -6@3.0 peak.
    assert pick_energy_peaks(levels, times, max_peaks=1) == [3.0]


def test_pick_energy_peaks_flat_or_junk():
    assert pick_energy_peaks([-20] * 8, [i * 0.25 for i in range(8)]) == []
    assert pick_energy_peaks([], []) == []
    assert pick_energy_peaks(None, None) == []
    assert pick_energy_peaks("nope", "nope") == []


def test_pick_energy_peaks_threshold_relative_to_median():
    # Max only 3 dB above the median -> below the default 6 dB floor.
    levels, times = _series([-20, -19, -17, -19, -20, -18, -20])
    assert pick_energy_peaks(levels, times) == []
    assert pick_energy_peaks(levels, times, threshold_db=2.0) == [0.5]


def test_detect_energy_peaks_never_raises():
    assert detect_energy_peaks("/nonexistent/path.mp4") == []
    assert detect_energy_peaks("") == []
    assert detect_energy_peaks(None) == []


def test_punch_zoom_suffix_format():
    s = punch_zoom_suffix(30, [1.5, 5.0])
    assert s.count("exp(-pow(") == 2
    assert "(on/30.0-1.50)" in s
    assert "(on/30.0-5.00)" in s


def test_punch_zoom_suffix_empty_and_caps():
    assert punch_zoom_suffix(30, None) == ""
    assert punch_zoom_suffix(30, []) == ""
    assert punch_zoom_suffix(0, [1.0]) == ""
    assert punch_zoom_suffix(30, [1, 2, 3, 4, 5]).count("exp(") == PUNCH_IN_MAX
    # Invalid entries are skipped, not fatal.
    assert punch_zoom_suffix(30, ["x", -2.0, 1.0]).count("exp(") == 1


def test_punch_in_amplitude_keeps_zoom_within_cap():
    # Drift (<=1.05) + pulse amplitude must peak at <= 1.12.
    from clippyme.pipeline.media_probe import PUNCH_IN_AMPLITUDE
    assert 1.05 + PUNCH_IN_AMPLITUDE <= 1.12 + 1e-9
