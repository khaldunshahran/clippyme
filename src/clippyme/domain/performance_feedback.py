"""Honest performance feedback for AI clip generation.

Reads stored analytics (views, likes, shares, comments, retention) and
returns validated descriptive observations. It does NOT invent causal
rules: the current registry lacks comparable cohorts (creator, genre,
post-age, experiment/version, per-platform attribution), so no duration
winner, hook archetype, completion rate, or comment-velocity claim is
produced. get_learned_patterns_prompt() returns an empty string for
current legacy records; a future comparable-evidence system is not
implemented here.

has_data means "valid recorded measurements exist". It does NOT mean a
causal or learned pattern exists.
"""
from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

from clippyme.domain.analytics_service import load_analytics_data

logger = logging.getLogger(__name__)


def _is_valid_count(value: Any) -> bool:
    """Finite non-negative number, not a boolean."""
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return math.isfinite(value) and value >= 0
    return False


def _is_valid_retention(value: Any) -> bool:
    """Finite fraction in [0, 1], not a boolean."""
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return math.isfinite(value) and 0.0 <= value <= 1.0
    return False


def _validate_metrics(raw: Any) -> Optional[Dict[str, Any]]:
    """Return validated metrics dict or None. Never mutates input."""
    if not isinstance(raw, dict):
        return None
    out: Dict[str, Any] = {}
    for key in ("views", "likes", "shares", "comments"):
        v = raw.get(key)
        if v is None:
            out[key] = 0
        elif _is_valid_count(v):
            out[key] = v
        else:
            return None
    r = raw.get("retention_rate")
    if r is None:
        out["retention_rate"] = None  # unknown, distinct from recorded zero
        out["retention_recorded"] = bool(raw.get("retention_recorded", False))
    elif _is_valid_retention(r):
        out["retention_rate"] = float(r)
        # Presence: only from an explicit stored marker. A bare value
        # without the marker is unknown, not a recorded zero.
        out["retention_recorded"] = bool(raw.get("retention_recorded", False))
    else:
        return None
    return out


def _validate_clip(clip_id: Any, raw: Any) -> Optional[Tuple[str, Dict[str, Any], Optional[str], Optional[str]]]:
    """Return (clip_id, metrics, title, hook_text) or None. No mutation."""
    if not isinstance(raw, dict):
        return None
    metrics = _validate_metrics(raw.get("metrics"))
    if metrics is None:
        return None
    title = raw.get("title")
    hook = raw.get("hook_text")
    return (
        str(clip_id),
        metrics,
        title if isinstance(title, str) and title else None,
        hook if isinstance(hook, str) and hook else None,
    )


def _empty_insights(reason: str, sample_size: int = 0) -> Dict[str, Any]:
    """Honest empty result: null/empty insight keys, explicit status."""
    return {
        "has_data": False,
        "sample_size": sample_size,
        "status": "no_measurements",
        "reason": reason,
        # Legacy comparison fields: null/empty, never fabricated.
        "story_duration_multiplier": None,
        "top_duration_tier": None,
        "avg_completion_rate": None,
        "top_hook_patterns": [],
        "recommendations": [],
        "top_titles": [],
        "observations": [],
    }


def analyze_performance_patterns() -> Dict[str, Any]:
    """Validated descriptive observations from stored analytics.

    Returns insight keys with null/empty values where evidence is absent.
    Useful statistics identify their metric, count, and denominator.
    Never mutates the loaded registry.
    """
    try:
        data = load_analytics_data()
    except Exception as e:
        logger.warning("performance_feedback: cannot load analytics: %s", e)
        return _empty_insights("analytics registry unreadable")

    if not isinstance(data, dict):
        return _empty_insights("analytics registry malformed")
    raw_clips = data.get("clips", {})
    if not isinstance(raw_clips, dict):
        return _empty_insights("analytics clips registry malformed")

    valid: List[Tuple[str, Dict[str, Any], Optional[str], Optional[str]]] = []
    for clip_id, raw in raw_clips.items():
        v = _validate_clip(clip_id, raw)
        if v is not None:
            valid.append(v)

    if not valid:
        return _empty_insights(
            "no valid clip measurements recorded",
            sample_size=len(raw_clips),
        )

    measured = [v for v in valid if v[1]["views"] > 0]
    if not measured:
        return {
            **_empty_insights(
                "clips tracked but no views recorded yet",
                sample_size=len(valid),
            ),
            "has_data": False,
        }

    n = len(measured)
    total_views = sum(v[1]["views"] for v in measured)
    total_likes = sum(v[1]["likes"] for v in measured)
    total_shares = sum(v[1]["shares"] for v in measured)
    total_comments = sum(v[1]["comments"] for v in measured)

    observations: List[Dict[str, Any]] = [
        {
            "metric": "views",
            "count": total_views,
            "denominator": f"{n} clips with recorded views",
            "note": "Total recorded views across measured clips.",
        },
        {
            "metric": "shares",
            "count": total_shares,
            "denominator": f"{total_views} recorded views" if total_views else "0 recorded views",
            "note": "Total recorded shares; share rate = shares / recorded views.",
        },
    ]

    # Retention: only from explicitly recorded values; unknown != zero.
    recorded = [v[1]["retention_rate"] for v in measured
                if v[1].get("retention_recorded") and v[1]["retention_rate"] is not None]
    avg_retention: Optional[float] = None
    if recorded:
        avg_retention = round(sum(recorded) / len(recorded), 4)
        observations.append({
            "metric": "retention_rate",
            "count": len(recorded),
            "denominator": f"{n} measured clips",
            "note": (
                f"Average of {len(recorded)} explicitly recorded retention "
                f"values: {avg_retention}. Unknown retention is not counted "
                f"as zero."
            ),
        })

    # Top titles as recorded examples, not proven archetypes.
    by_views = sorted(measured, key=lambda v: v[1]["views"], reverse=True)
    top_titles = [v[2] for v in by_views[:3] if v[2]]

    return {
        "has_data": True,
        "sample_size": n,
        "status": "descriptive_observations",
        "reason": (
            "Valid recorded measurements exist. No comparable cohorts "
            "(creator, genre, post-age, experiment/version, per-platform "
            "attribution) are available, so no causal duration, hook, "
            "completion, or comment-velocity rule is inferred."
        ),
        # Legacy comparison fields: null/empty, never fabricated.
        "story_duration_multiplier": None,
        "top_duration_tier": None,
        "avg_completion_rate": None,
        "top_hook_patterns": [],
        "recommendations": [],
        "top_titles": top_titles,
        "observations": observations,
        "totals": {
            "views": total_views,
            "likes": total_likes,
            "shares": total_shares,
            "comments": total_comments,
        },
        "retention": {
            "average_recorded": avg_retention,
            "recorded_count": len(recorded),
            "measured_count": n,
        },
    }


def get_learned_patterns_prompt() -> str:
    """Automatic empirical prompt block for selection/rescoring.

    Current legacy records cannot establish causal duration/hook rules,
    so this returns an empty string: no empirical duration, hook-theme,
    completion, or comment-velocity directive enters default
    selection/rescoring. Explicit caller overrides via build_prompt's
    learned_patterns parameter are preserved by the caller.
    """
    return ""
