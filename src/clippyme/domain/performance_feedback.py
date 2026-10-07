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


def _safe_float(value: Any) -> Optional[float]:
    """Return float for actual int/float input; None otherwise.

    Strings are rejected even when float-convertible: numeric evidence
    must be numeric. Never raises (incl. OverflowError on huge ints).
    """
    if isinstance(value, bool):
        return None
    if not isinstance(value, (int, float)):
        return None
    try:
        fv = float(value)
    except (OverflowError, ValueError):
        return None
    try:
        if not math.isfinite(fv):
            return None
    except OverflowError:
        return None
    return fv


def _is_valid_count(value: Any) -> bool:
    """Finite non-negative number, not a boolean. Never raises."""
    fv = _safe_float(value)
    return fv is not None and fv >= 0


def _is_valid_retention(value: Any) -> bool:
    """Finite fraction in [0, 1], not a boolean. Never raises."""
    fv = _safe_float(value)
    return fv is not None and 0.0 <= fv <= 1.0


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
    # Presence requires boolean True. Malformed markers ("false", 1, [],
    # dicts) never create evidence.
    marked = raw.get("retention_recorded") is True
    if r is None:
        out["retention_rate"] = None
        out["retention_recorded"] = False
        out["retention_legacy_positive"] = False
    elif _is_valid_retention(r):
        fv = _safe_float(r)
        out["retention_rate"] = fv
        out["retention_recorded"] = marked
        # Legacy compatibility: positive finite fractions remain usable
        # observations even without a marker; explicit zero requires it.
        out["retention_legacy_positive"] = (not marked) and fv > 0
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
    # Honest totals policy:
    # - Integers sum with exact integer arithmetic (no float rounding).
    # - Floats sum separately; mixed int+float uses float.
    # - Overflow is DISCLOSED per metric in the API output, not hidden.
    #   A total that cannot be represented exactly is marked
    #   "overflow": True with a reason, and the value is either capped
    #   (clearly flagged) or omitted. Never presented as an exact
    #   "Recorded Total".
    def _safe_total(values):
        """Return (value, overflow, inexact) for a metric.

        Exact integer arithmetic when all inputs are ints. Floats are
        summed as floats. On overflow, value is None and overflow=True.
        """
        int_total = 0
        float_total = 0.0
        has_int = False
        has_float = False
        for v in values:
            if isinstance(v, bool):
                continue
            if isinstance(v, int):
                # Python ints are arbitrary precision; check float range
                # only if we later need float conversion.
                int_total += v
                has_int = True
            elif isinstance(v, float):
                if not math.isfinite(v):
                    continue
                float_total += v
                has_float = True
                if not math.isfinite(float_total):
                    return None, True, False
            # Other types already filtered by _safe_float validation.
        if has_float:
            # Mixed: convert int part to float (may lose precision;
            # flag inexact if int part exceeds float precision).
            if has_int:
                if abs(int_total) > 2**53:
                    # Integer part cannot be represented exactly as float.
                    return None, True, True
                float_total += float(int_total)
                if not math.isfinite(float_total):
                    return None, True, False
            return float_total, False, False
        if has_int:
            return int_total, False, False
        return 0, False, False

    _totals = {}
    _overflow_flags = {}
    for _metric in ("views", "likes", "shares", "comments"):
        _val, _overflow, _inexact = _safe_total(
            v[1].get(_metric) for v in measured)
        _totals[_metric] = _val
        _overflow_flags[_metric] = {
            "overflow": _overflow,
            "inexact": _inexact,
        }
    total_views = _totals["views"]
    total_likes = _totals["likes"]
    total_shares = _totals["shares"]
    total_comments = _totals["comments"]

    observations: List[Dict[str, Any]] = [
        {
            "metric": "views",
            "count": total_views,
            "denominator": f"{n} clips with recorded views",
            "note": (
                "Total recorded views across measured clips."
                if total_views is not None
                else "Total views unavailable: sum exceeds representable "
                     "range (overflow). Not shown as zero."
            ),
        },
        {
            "metric": "shares",
            "count": total_shares,
            # Unavailable (None) is NOT zero. Genuine zero stays zero.
            "denominator": (
                "total views unavailable (overflow; exceeds representable range)"
                if total_views is None
                else f"{total_views} recorded views"
            ),
            "note": (
                "Total recorded shares."
                + (" Share rate cannot be calculated: total views unavailable."
                   if total_views is None or total_shares is None
                   else " Share rate = shares / recorded views.")
            ),
        },
    ]

    # Retention: explicitly recorded values (zero counts) plus positive
    # finite legacy fractions. Unknown stays unknown. Unweighted average.
    # Provenance tracked separately: explicit vs legacy-positive.
    explicit = [v[1]["retention_rate"] for v in measured
                if v[1].get("retention_recorded")
                and v[1]["retention_rate"] is not None]
    legacy_pos = [v[1]["retention_rate"] for v in measured
                  if v[1].get("retention_legacy_positive")
                  and v[1]["retention_rate"] is not None]
    recorded = explicit + legacy_pos
    avg_retention: Optional[float] = None
    if recorded:
        avg_retention = round(sum(recorded) / len(recorded), 4)
        # Honest provenance: distinguish explicit from legacy-positive.
        if explicit and legacy_pos:
            _prov = (f"{len(explicit)} explicitly recorded + "
                     f"{len(legacy_pos)} legacy positive observations")
        elif explicit:
            _prov = f"{len(explicit)} explicitly recorded values"
        else:
            _prov = f"{len(legacy_pos)} legacy positive observations"
        observations.append({
            "metric": "retention_rate",
            "count": len(recorded),
            "denominator": f"{n} measured clips (unweighted average)",
            "note": (
                f"Observed average retention: {avg_retention} from {_prov}. "
                f"Unknown retention is not counted as zero."
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
            # Per-metric integrity disclosure. A metric with
            # overflow=True is NOT an exact recorded total; the value
            # is None (unavailable) with a reason, not zero.
            "integrity": {
                _m: {
                    "overflow": _overflow_flags[_m]["overflow"],
                    "inexact": _overflow_flags[_m]["inexact"],
                    "status": ("overflow_unavailable"
                               if _overflow_flags[_m]["overflow"]
                               else "exact"),
                }
                for _m in ("views", "likes", "shares", "comments")
            },
        },
        "retention": {
            "average_recorded": avg_retention,
            "recorded_count": len(recorded),
            "explicit_count": len(explicit),
            "legacy_positive_count": len(legacy_pos),
            "measured_count": n,
            "averaging": "unweighted",
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
