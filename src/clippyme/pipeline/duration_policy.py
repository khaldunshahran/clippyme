"""Shared duration policy for clip selection and edge repair.

This module resolves the effective min/max duration bounds once, used by:
- ``build_viral_prompt`` (selection prompt generation)
- ``snap_clips_to_transcript`` via both production callers:
  - queued orchestrator (``orchestrator.py``)
  - direct CLI (``main.py``)

Precedence (highest to lowest):
1. Explicit ``max_duration`` > 0 (hard limit, never silently increased).
2. Explicit ``min_duration`` > 0 (if min > max, max wins; see below).
3. Duration mode: ``all``/``auto``/``mixed``/``portfolio`` -> multi-tier.
4. Defaults: min 15s, max 120s (or video duration if shorter).

Contradictory/invalid inputs have deterministic outcomes:
- ``min_duration > max_duration``: max is the hard limit; min is clamped
  to max (documented, not silent growth).
- Nonfinite (NaN/Inf), negative, zero, or bool inputs: treated as absent.
- ``max_duration`` exceeding source duration: clamped to source duration.
- Known short sources: bounds never exceed source length.

The resolved max is a HARD LIMIT for optional repair expansion (sentence,
silence, waveform). Mandatory word-boundary repair is distinct: if an
already-selected interval cannot preserve words AND meet duration, the
caller receives an honest conflict signal (not silent truncation).
"""

import math
from typing import NamedTuple, Optional


class DurationPolicy(NamedTuple):
    """Resolved duration bounds for a job."""
    min_duration: float
    max_duration: float
    mode: str  # 'fixed' or 'multi-tier'
    source_duration: Optional[float]
    # Diagnostic flags.
    min_clamped: bool = False
    max_clamped_to_source: bool = False


def _clean_bound(value) -> Optional[float]:
    """Return a valid positive finite float, or None if absent/invalid.

    Rejects bool, None, non-numeric, NaN, Inf, zero, and negative.
    """
    if value is None or isinstance(value, bool):
        return None
    try:
        fv = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(fv) or fv <= 0:
        return None
    return fv


def resolve_duration_policy(
    *,
    min_duration=None,
    max_duration=None,
    duration_mode=None,
    clip_type=None,
    video_duration=None,
    source_duration=None,
) -> DurationPolicy:
    """Resolve effective duration bounds from job options.

    All callers (prompt generation, queued repair, direct CLI repair)
    MUST use this function to ensure consistent bounds.
    """
    src_dur = _clean_bound(video_duration if video_duration is not None
                           else source_duration)

    dmode = (duration_mode or "").strip().lower()
    explicit_max = _clean_bound(max_duration)
    explicit_min = _clean_bound(min_duration)
    # User provided a bound (even if invalid) vs omitted entirely.
    # Invalid values fall back to fixed defaults, NOT multi-tier.
    user_set_max = max_duration is not None
    user_set_min = min_duration is not None
    # Multi-tier only when user omitted BOTH bounds (and mode allows).
    is_multi_tier = (
        not user_set_max and not user_set_min
        and (dmode in ("all", "auto", "mixed", "portfolio")
             or dmode == "")
    )

    if is_multi_tier:
        mode = "multi-tier"
        max_clamped = False
        if src_dur is not None:
            # Short source: bounds must satisfy 0 < min <= max <= source.
            # Use source length directly; don't demand unavailable tiers.
            if src_dur < 15.0:
                min_dur = min(5.0, src_dur / 2.0)
                max_dur = src_dur
            elif src_dur < 30.0:
                min_dur = 10.0
                max_dur = src_dur
            else:
                min_dur = 15.0
                max_dur = min(src_dur, 300.0)
                max_clamped = max_dur < 300.0
            # Ensure min <= max.
            if min_dur > max_dur:
                min_dur = max_dur / 2.0
        else:
            min_dur = 15.0
            max_dur = 300.0
    else:
        min_dur = explicit_min or 15.0
        max_dur = explicit_max or 120.0
        mode = "fixed"
        max_clamped = False
        min_clamped = False
        # min > max: max is hard limit; clamp min to max.
        if min_dur > max_dur:
            min_dur = max_dur
            min_clamped = True
        # Never exceed source duration.
        if src_dur is not None and max_dur > src_dur:
            max_dur = src_dur
            max_clamped = True
            if min_dur > max_dur:
                min_dur = max_dur
                min_clamped = True

    return DurationPolicy(
        min_duration=min_dur,
        max_duration=max_dur,
        mode=mode,
        source_duration=src_dur,
        min_clamped=min_clamped if not is_multi_tier else False,
        max_clamped_to_source=max_clamped,
    )
