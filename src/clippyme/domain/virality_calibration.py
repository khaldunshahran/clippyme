"""Phase 3A: virality feedback-loop calibration scaffolding.

Closes the open loop between real post performance and the deterministic
virality features (Phase 2). Correlates deterministic features against
observed outcomes across published clips and proposes bounded weight
multipliers — but NEVER applies them automatically.

Design rules:
- Needs >= ``MIN_CALIBRATION_CLIPS`` (default 20) clips WITH real view data.
  Analytics currently hold 19 clips, 0 with views -> the loop reports
  "needs real view data to calibrate" until real performance arrives.
- ``recalibrate_weights`` REFUSES unless called with ``approved=True`` and a
  named operator: recalibration is a manual, quarterly, human-approved
  operation. No cron, no auto-tune, no silent drift.
- Proposed multipliers are bounded to [0.5, 2.0] so a noisy quarter cannot
  flip the scorer upside down.
- Approved overrides are consumed by ``gemini_parser.deterministic_score``
  (and therefore by highlight_service's multi-tier highlight scoring, which
  ranks candidates through it) via ``get_active_weight_overrides()``.
"""
from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

MIN_CALIBRATION_CLIPS = 20
MULTIPLIER_FLOOR = 0.5
MULTIPLIER_CEIL = 2.0
OVERRIDES_FILE_PATH = Path("data/virality_weight_overrides.json")

# Deterministic features eligible for calibration (Phase 2 feature names).
CALIBRATION_FEATURES = [
    "speech_rate_wpm",
    "question_density",
    "exclamation_density",
    "audio_energy_variance",
    "duration",
]

_OUTCOMES = ("engagement", "reach", "retention")

_override_cache: Optional[Dict[str, float]] = None


# --- math -------------------------------------------------------------------

def pearson(xs: List[float], ys: List[float]) -> Optional[float]:
    """Pearson r, or None when undefined (too few points / zero variance)."""
    n = len(xs)
    if n != len(ys) or n < 2:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = math.sqrt(sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys))
    if den == 0:
        return None
    return num / den


def _engagement_score(metrics: Dict[str, Any]) -> float:
    return (float(metrics.get("views", 0) or 0)
            + 3.0 * float(metrics.get("shares", 0) or 0)
            + 2.0 * float(metrics.get("comments", 0) or 0)
            + 5.0 * float(metrics.get("likes", 0) or 0))


def _reach_score(metrics: Dict[str, Any]) -> float:
    # log-scaled: a 10x view jump should not dominate 10x harder than 2x
    return math.log1p(float(metrics.get("views", 0) or 0))


def _retention_score(metrics: Dict[str, Any]) -> float:
    return float(metrics.get("retention_rate", 0.0) or 0.0)


def _outcome_value(outcome: str, metrics: Dict[str, Any]) -> float:
    if outcome == "engagement":
        return _engagement_score(metrics)
    if outcome == "reach":
        return _reach_score(metrics)
    return _retention_score(metrics)


# --- data access --------------------------------------------------------------

def _load_published() -> Dict[str, Any]:
    from clippyme.domain.analytics_service import load_analytics_data
    return load_analytics_data() or {}


def _usable_clips(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    clips = (data or {}).get("clips", {}) or {}
    usable = []
    for clip in clips.values():
        metrics = clip.get("metrics") or {}
        feats = clip.get("deterministic_features") or {}
        if not feats:
            continue
        if float(metrics.get("views", 0) or 0) <= 0:
            continue  # no real view data -> cannot calibrate from this clip
        usable.append({"features": feats, "metrics": metrics})
    return usable


# --- correlation ---------------------------------------------------------------

def compute_feature_correlations(
    min_clips: int = MIN_CALIBRATION_CLIPS,
    data: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Correlate each deterministic feature against engagement/reach/retention.

    Returns a report dict with ``has_data`` False and a plain-language
    ``reason`` ("needs real view data to calibrate") until enough real
    performance exists — the open loop stays visibly open instead of
    silently fitting noise.
    """
    clips = _usable_clips(data if data is not None else _load_published())
    report: Dict[str, Any] = {
        "has_data": False,
        "sample_size": len(clips),
        "min_clips": min_clips,
        "reason": "",
        "correlations": {},
    }
    if len(clips) < min_clips:
        report["reason"] = (
            f"needs real view data to calibrate: {len(clips)}/{min_clips} "
            "published clips have view metrics")
        return report

    correlations: Dict[str, Dict[str, Optional[float]]] = {}
    for feat in CALIBRATION_FEATURES:
        xs = [float(c["features"].get(feat) or 0.0) for c in clips]
        per_outcome = {}
        for outcome in _OUTCOMES:
            ys = [_outcome_value(outcome, c["metrics"]) for c in clips]
            per_outcome[f"vs_{outcome}"] = pearson(xs, ys)
        correlations[feat] = per_outcome
    report["has_data"] = True
    report["correlations"] = correlations
    report["reason"] = "ok"
    return report


# --- proposals ------------------------------------------------------------------

def propose_weight_adjustments(
    report: Dict[str, Any],
    outcome: str = "engagement",
) -> Dict[str, Any]:
    """Turn correlations into bounded multiplier proposals.

    Positive correlation with the outcome -> multiplier > 1 (amplify the
    feature's voice in the score); negative -> < 1. Magnitude scales with
    |r| and is hard-bounded to [0.5, 2.0]. Returns a PROPOSED payload that
    still needs explicit operator approval via ``recalibrate_weights``.
    """
    proposal: Dict[str, Any] = {
        "has_data": bool(report.get("has_data")),
        "outcome": outcome,
        "status": "PROPOSED - awaiting operator approval",
        "weight_overrides": {},
        "correlations_used": {},
    }
    if not report.get("has_data"):
        proposal["status"] = "no-data"
        proposal["reason"] = report.get("reason", "")
        return proposal

    overrides: Dict[str, float] = {}
    used: Dict[str, Optional[float]] = {}
    for feat in CALIBRATION_FEATURES:
        r = (report.get("correlations", {}).get(feat, {})
             .get(f"vs_{outcome}"))
        used[feat] = r
        if r is None:
            mult = 1.0
        else:
            mult = 1.0 + 0.5 * max(-1.0, min(1.0, r))
        overrides[feat] = max(MULTIPLIER_FLOOR, min(MULTIPLIER_CEIL, mult))
    proposal["weight_overrides"] = overrides
    proposal["correlations_used"] = used
    return proposal


# --- manual recalibration ---------------------------------------------------------

def recalibrate_weights(
    approved: bool = False,
    operator: Optional[str] = None,
    min_clips: int = MIN_CALIBRATION_CLIPS,
    outcome: str = "engagement",
    data: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Manually trigger a virality weight recalibration.

    This is the ONLY writer of the overrides file, and it refuses to run
    without explicit human approval (``approved=True`` + operator name).
    Intended cadence: quarterly, operator-reviewed. There is deliberately no
    automatic caller anywhere in the codebase.
    """
    if not approved or not operator:
        return {
            "status": "refused",
            "reason": ("recalibration requires explicit operator approval: "
                       "call recalibrate_weights(approved=True, operator=<name>)"),
        }
    report = compute_feature_correlations(min_clips=min_clips, data=data)
    if not report.get("has_data"):
        return {"status": "insufficient_data", "reason": report.get("reason", "")}

    proposal = propose_weight_adjustments(report, outcome=outcome)
    payload = {
        "schema": "virality_weight_overrides/v1",
        "approved_by": operator,
        "outcome": outcome,
        "sample_size": report["sample_size"],
        "weight_overrides": proposal["weight_overrides"],
        "correlations": proposal["correlations_used"],
        "note": ("Manual quarterly recalibration. Multipliers bounded to "
                 "[0.5, 2.0]. Delete this file to restore Phase-2 defaults."),
    }
    OVERRIDES_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
    OVERRIDES_FILE_PATH.write_text(json.dumps(payload, indent=2))
    clear_override_cache()
    logger.info("virality recalibration applied by %s on %d clips",
                operator, report["sample_size"])
    return {"status": "applied", **payload}


# --- consumption -------------------------------------------------------------------

def get_active_weight_overrides() -> Dict[str, float]:
    """Approved overrides, or {} when none exist (Phase-2 defaults rule)."""
    global _override_cache
    if _override_cache is not None:
        return _override_cache
    try:
        payload = json.loads(OVERRIDES_FILE_PATH.read_text())
        overrides = {
            k: float(v) for k, v in
            (payload.get("weight_overrides") or {}).items()
            if k in CALIBRATION_FEATURES
        }
    except (OSError, ValueError, AttributeError):
        overrides = {}
    _override_cache = overrides
    return overrides


def clear_override_cache() -> None:
    global _override_cache
    _override_cache = None
