"""Detect creator-burned captions in source videos.

Problem: when a source video already has captions burned in by its creator,
Nugget's karaoke captions stack on top → double captions, which looks
amateur. The 9:16 reframe can also clip the creator's 16:9-positioned
captions at the frame edge.

This module detects pre-burned captions WITHOUT OCR (no tesseract on the
render hosts). Heuristic: burned captions are high-contrast text in the
bottom band that CHANGES between frames during speech. We sample frames at
word times and measure the caption-band change fraction (reusing the
proven metric from clip_qa._band_change_fraction). A consistently changing
band during speech = likely burned captions.

Returns a dict: {"likely": bool, "confidence": float, "detail": str}.
"""
import logging
import os

logger = logging.getLogger(__name__)


def _sample_times(transcript, clip_start, clip_end, n=6):
    """Evenly spaced sample times (seconds) within speech regions."""
    words = []
    for seg in (transcript or {}).get("segments") or []:
        for w in seg.get("words") or []:
            try:
                ws, we = float(w["start"]), float(w["end"])
            except (TypeError, ValueError, KeyError):
                continue
            if we > clip_start and ws < clip_end:
                words.append((max(ws, clip_start), min(we, clip_end)))
    if len(words) < 2:
        return []
    # Pick n times spread across the word span.
    t0, t1 = words[0][0], words[-1][1]
    if t1 - t0 < 2.0:
        return []
    step = (t1 - t0) / n
    return [t0 + step * (i + 0.5) for i in range(n)]


def detect_preburned_captions(video_path, transcript=None,
                             clip_start=0.0, clip_end=None,
                             position="bottom"):
    """Heuristic detection of creator-burned captions in a source video.

    Args:
        video_path: source video file.
        transcript: word-timestamped transcript dict (for speech-time sampling).
        clip_start/clip_end: range to analyze (source-relative seconds).
        position: caption band position ("bottom" default).

    Returns {"likely": bool, "confidence": 0..1, "detail": str}.
    """
    from clippyme.domain.clip_qa import (
        _band_change_fraction,
        _caption_band,
        _grab_frame,
    )

    result = {"likely": False, "confidence": 0.0, "detail": "insufficient data"}
    if not video_path or not os.path.exists(video_path):
        result["detail"] = "video not found"
        return result

    if clip_end is None:
        clip_end = clip_start + 60.0

    times = _sample_times(transcript or {}, clip_start, clip_end)
    if len(times) < 3:
        result["detail"] = "not enough speech to sample"
        return result

    frames = []
    for t in times:
        f = _grab_frame(video_path, t)
        if f is not None:
            frames.append(f)
    if len(frames) < 3:
        result["detail"] = "frame grab failed"
        return result

    h, w = frames[0].shape[:2]
    band = _caption_band(w, h, position)

    # Band change between consecutive samples. Burned captions flip words
    # every second or two, so the band should change hard between samples
    # even when the speaker barely moves.
    changes = []
    for a, b in zip(frames, frames[1:]):
        if a.shape != b.shape:
            continue
        changes.append(_band_change_fraction(a, b, band))
    if not changes:
        return result

    mean_change = sum(changes) / len(changes)
    # Calibrated: talking-head without captions ≈ 0.01–0.05 band change
    # (breathing/mic motion); with burned captions ≈ 0.15–0.40.
    if mean_change >= 0.12:
        result.update({
            "likely": True,
            "confidence": min(1.0, mean_change / 0.30),
            "detail": f"caption-band change {mean_change:.2f} across {len(changes)} samples",
        })
    else:
        result.update({
            "likely": False,
            "confidence": max(0.0, 1.0 - mean_change / 0.12),
            "detail": f"caption-band stable ({mean_change:.3f})",
        })
    return result
