"""2B: transcript confidence quality gate — pure helpers (host-testable).

A transcript whose word confidences average below ``TRANSCRIPT_QUALITY_FLOOR``
is unreliable for word-snapped cuts and karaoke timing: the pipeline retries
once with a secondary provider and, if still low, flags the transcript so
downstream stages (cut padding, QA) can compensate instead of trusting it.

Words that omit a probability are *unknown*, never zero — they are excluded
from the mean rather than dragging it down.
"""
from __future__ import annotations

# Below this known-mean word confidence the transcript is "low" quality.
TRANSCRIPT_QUALITY_FLOOR = 0.55


def iter_transcript_words(transcript):
    """Yield word dicts from a transcript's segments (tolerates both shapes)."""
    if not isinstance(transcript, dict):
        return
    for seg in transcript.get("segments", []) or []:
        if not isinstance(seg, dict):
            continue
        for w in seg.get("words", []) or []:
            if isinstance(w, dict):
                yield w


def _word_confidence(w):
    try:
        p = float(w.get("probability"))
    except (TypeError, ValueError):
        return None
    if p != p or p == float("inf") or p == float("-inf"):  # NaN / inf guard
        return None
    return p


def assess_transcript_quality(transcript) -> dict:
    """Mean word confidence over words that report one (pure).

    Returns ``{"quality", "mean_confidence", "known_words", "total_words"}``
    where quality is "ok" | "low" | "unknown" (no word reported confidence).
    """
    total = 0
    known = 0
    acc = 0.0
    for w in iter_transcript_words(transcript):
        total += 1
        p = _word_confidence(w)
        if p is not None:
            known += 1
            acc += p
    mean = (acc / known) if known else None
    if mean is None:
        quality = "unknown"
    elif mean < TRANSCRIPT_QUALITY_FLOOR:
        quality = "low"
    else:
        quality = "ok"
    return {
        "quality": quality,
        "mean_confidence": mean,
        "known_words": known,
        "total_words": total,
    }


def annotate_transcript_quality(transcript, quality=None) -> dict:
    """Stamp confidence metadata onto a transcript dict (mutates + returns)."""
    if not isinstance(transcript, dict):
        return transcript
    q = quality or assess_transcript_quality(transcript)
    transcript["transcript_quality"] = q["quality"]
    transcript["transcript_confidence"] = q["mean_confidence"]
    transcript["transcript_confidence_words"] = q["known_words"]
    return transcript


def pick_better_transcript(primary, primary_q, alt, alt_q):
    """Return ``(transcript, quality)`` — the better of two attempts (pure).

    Known confidence always beats unknown; between two known means the
    higher wins; ties (or both unknown) keep the primary.
    """
    pm = primary_q.get("mean_confidence")
    am = alt_q.get("mean_confidence")
    if am is not None and (pm is None or am > pm):
        return alt, alt_q
    return primary, primary_q


def secondary_provider_for(primary: str):
    """One-shot fallback provider for the quality-gate retry (pure).

    Prefers a *different* cloud provider when its key is configured;
    Faster-Whisper is always available as the last resort. Returns None
    when there is nothing sensible to retry with.
    """
    import os

    primary = (primary or "").strip().lower()
    has_dg = bool(os.getenv("DEEPGRAM_API_KEY"))
    has_el = bool(os.getenv("ELEVENLABS_API_KEY"))
    if primary == "deepgram":
        return "elevenlabs" if has_el else "whisper"
    if primary == "elevenlabs":
        return "deepgram" if has_dg else "whisper"
    if has_dg:
        return "deepgram"
    if has_el:
        return "elevenlabs"
    # Last resort: local Faster-Whisper always works — except when it *was*
    # the primary (retrying the same engine is pointless).
    return "whisper" if primary != "whisper" else None
