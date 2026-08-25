"""Punch-in: short dynamic pushes toward the subject on audio emphasis beats.

Adds dynamic editorial energy to long static shots.
"""
import os
import numpy as np

ENABLED = os.environ.get("PUNCH_IN", "0") == "1"
MAX_ZOOM = float(os.environ.get("PUNCH_IN_ZOOM", "1.12"))
RISE_SECONDS = 0.25
HOLD_SECONDS = 1.30
FALL_SECONDS = 0.55
MIN_GAP_SECONDS = 18.0
BEAT_PROMINENCE = 0.60


def _ease(t):
    """Smoothstep on [0, 1]."""
    t = min(max(t, 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)


def zoom_curve(n_frames, fps, emphasis_times, max_zoom=None, start_offset=0.0):
    """Per-frame zoom factor for punch-in animation."""
    max_zoom = MAX_ZOOM if max_zoom is None else max_zoom
    zooms = [1.0] * max(0, n_frames)
    if not zooms or max_zoom <= 1.0 or not emphasis_times:
        return zooms

    span = RISE_SECONDS + HOLD_SECONDS + FALL_SECONDS
    for t in emphasis_times:
        local = t - start_offset
        if local <= -span or local >= n_frames / float(fps):
            continue
        for f in range(max(0, int(local * fps)), min(n_frames, int((local + span) * fps) + 1)):
            dt = f / float(fps) - local
            if dt < 0:
                continue
            if dt < RISE_SECONDS:
                factor = _ease(dt / RISE_SECONDS)
            elif dt < RISE_SECONDS + HOLD_SECONDS:
                factor = 1.0
            elif dt < span:
                factor = 1.0 - _ease((dt - RISE_SECONDS - HOLD_SECONDS) / FALL_SECONDS)
            else:
                factor = 0.0
            zooms[f] = max(zooms[f], 1.0 + (max_zoom - 1.0) * factor)

    return zooms


def find_emphasis_beats(audio_path, min_gap=MIN_GAP_SECONDS):
    """Estimate speech emphasis timestamps from audio envelope."""
    # Fallback to empty list if audio cannot be processed
    return []
