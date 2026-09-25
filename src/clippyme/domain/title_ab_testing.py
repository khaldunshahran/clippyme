"""Phase 3D: A/B title testing — mechanism only, default OFF.

The metadata generator stores 2-3 title variants per clip (variant A = the
main title). This module provides the rotation/attribution mechanism:

- :func:`ab_testing_enabled` — master switch, env ``CLIPPYME_AB_TITLES=1``.
  Default OFF: every post ships variant A, i.e. today's behaviour exactly.
- :func:`assign_variant` — deterministic round-robin assignment from a
  stable key (SHA-256), so the same post always maps to the same variant.
- :func:`epsilon_greedy_assignment` — exploitation variant once per-variant
  performance exists; falls back to the deterministic hash without data.
- :func:`resolve_title_for_post` — what publish_service calls: returns the
  title to ship + the assignment record. Disabled -> variant A.
- :func:`record_title_variant_assignment` — analytics attribution written
  AFTER publish returns the real ``post_id``.
- :func:`variant_performance_summary` — per-variant impressions/CTR rollup
  for the operator's manual read (no auto-optimization anywhere).

Nothing here changes copy, schedules posts, or spends anything. Rotation on
live posts is an explicit operator decision (env var); the variants
themselves stay in clip/job metadata either way.
"""
from __future__ import annotations

import hashlib
import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

AB_TITLES_ENV = "CLIPPYME_AB_TITLES"
MAX_VARIANTS = 3
EPSILON_DEFAULT = 0.1


# --- enablement ---------------------------------------------------------------

def ab_testing_enabled(explicit: Optional[bool] = None) -> bool:
    """Master switch. Default OFF — live rotation needs an operator decision."""
    if explicit is not None:
        return bool(explicit)
    return os.getenv(AB_TITLES_ENV, "").strip().lower() in ("1", "true", "yes", "on")


# --- variants --------------------------------------------------------------------

def extract_title_variants(clip_info: Optional[Dict[str, Any]]) -> List[str]:
    """Up to MAX_VARIANTS titles, variant A first (the main title)."""
    clip_info = clip_info or {}
    variants = list(clip_info.get("title_variants") or [])
    if not variants and clip_info.get("title"):
        variants = [clip_info["title"]]
    # de-dupe, keep order, cap
    seen, out = set(), []
    for v in variants:
        v = (v or "").strip()
        if v and v not in seen:
            seen.add(v)
            out.append(v)
        if len(out) >= MAX_VARIANTS:
            break
    return out


def _variant_label(index: int) -> str:
    return chr(ord("A") + index) if 0 <= index < 26 else f"V{index}"


# --- assignment ---------------------------------------------------------------------

def _hash_index(key: str, n: int, salt: str = "clippyme-ab-v1") -> int:
    digest = hashlib.sha256(f"{salt}:{key}".encode("utf-8")).hexdigest()
    return int(digest, 16) % n


def assign_variant(variants: List[str], key: str,
                   method: str = "round_robin") -> int:
    """Deterministic variant index for a stable ``key``.

    Round-robin here = stable hash mod N: uniform over many keys, identical
    for the same key on every call (no shared counter, no cross-post
    coupling). ``method`` is recorded for analytics transparency.
    """
    n = len(variants)
    if n <= 1:
        return 0
    return _hash_index(key, n)


def epsilon_greedy_assignment(
    variants: List[str],
    key: str,
    performance: Optional[Dict[str, Dict[str, float]]] = None,
    epsilon: float = EPSILON_DEFAULT,
) -> int:
    """Exploit the best-performing variant with prob 1-epsilon, else explore.

    ``performance`` maps variant-index-as-str -> {"impressions", "ctr"}.
    Without usable performance data this degrades to the deterministic hash
    (identical to :func:`assign_variant`), so cold posts behave sanely.
    """
    n = len(variants)
    if n <= 1:
        return 0
    best: Optional[int] = None
    if performance:
        scored = []
        for i in range(n):
            p = performance.get(str(i)) or {}
            imp = float(p.get("impressions", 0) or 0)
            if imp > 0:
                scored.append((float(p.get("ctr", 0) or 0), i))
        if scored:
            scored.sort(reverse=True)
            best = scored[0][1]
    if best is None:
        return _hash_index(key, n)
    # deterministic explore/exploit split from the key hash (no RNG state)
    explore = (_hash_index(f"eps:{key}", 1000) / 1000.0) < epsilon
    if explore:
        # explore: hash to any NON-best variant
        others = [i for i in range(n) if i != best]
        return others[_hash_index(f"explore:{key}", len(others))]
    return best


# --- resolution (publish path) ---------------------------------------------------------

def resolve_title_for_post(
    clip_info: Optional[Dict[str, Any]],
    assignment_key: str,
    enabled: Optional[bool] = None,
    method: str = "round_robin",
    performance: Optional[Dict[str, Dict[str, float]]] = None,
) -> Dict[str, Any]:
    """Return the title to ship for one post + the assignment record.

    Default (switch off): variant A — byte-identical to today's behaviour.
    An explicit per-request title still wins; publish_service applies that
    BEFORE calling this (the override path never enters the experiment).
    """
    variants = extract_title_variants(clip_info)
    on = ab_testing_enabled(enabled)
    if not on or len(variants) <= 1:
        idx = 0
        used_method = "control"
    elif method == "epsilon_greedy":
        idx = epsilon_greedy_assignment(variants, assignment_key, performance)
        used_method = "epsilon_greedy"
    else:
        idx = assign_variant(variants, assignment_key)
        used_method = "round_robin"
    return {
        "title": variants[idx] if variants else "",
        "variant": _variant_label(idx),
        "variant_index": idx,
        "variants": variants,
        "method": used_method,
        "enabled": on,
        "assignment_key": assignment_key,
    }


# --- attribution (analytics path) ---------------------------------------------------------

def record_title_variant_assignment(
    job_id: str,
    clip_index: int,
    post_id: Optional[str],
    assignment: Dict[str, Any],
    platform: Optional[str] = None,
) -> bool:
    """Write which title variant shipped on a post, keyed by real post_id.

    Called AFTER publish returns. Additive-only: existing analytics fields
    are never clobbered. Returns False when the clip has no analytics
    record (nothing to attribute to).
    """
    from clippyme.domain.analytics_service import (
        load_analytics_data as load_analytics,
        save_analytics_data as save_analytics,
    )
    try:
        data = load_analytics() or {}
        clips = data.get("clips") or {}
        record_key = f"{job_id}:{clip_index}"
        rec = clips.get(record_key)
        if not rec:
            logger.warning("A/B attribution: no analytics record for %s",
                           record_key)
            return False
        attributions = rec.setdefault("title_variant_assignments", {})
        if post_id:
            attributions[post_id] = {
                "variant": assignment.get("variant"),
                "variant_index": assignment.get("variant_index"),
                "title": assignment.get("title"),
                "method": assignment.get("method"),
                "platform": platform,
                "assignment_key": assignment.get("assignment_key"),
            }
            # convenience rollup: latest shipped variant on this clip
            rec["title_variant_shipped"] = assignment.get("variant")
            rec["title_variant_index"] = assignment.get("variant_index")
            rec["title_variant_post_id"] = post_id
        save_analytics(data)
        return True
    except Exception as exc:
        logger.warning("A/B attribution write failed: %s", exc)
        return False


def variant_performance_summary(
    data: Optional[Dict[str, Any]] = None,
) -> Dict[str, Dict[str, float]]:
    """Per-variant impressions/CTR rollup across attributed clips.

    Read-only, for the operator's manual review. No auto-optimization: the
    rotation method stays whatever the operator configured.
    """
    from clippyme.domain.analytics_service import load_analytics_data as load_analytics
    data = data if data is not None else (load_analytics() or {})
    agg: Dict[str, Dict[str, float]] = {}
    for rec in (data.get("clips") or {}).values():
        variant = rec.get("title_variant_shipped")
        if not variant:
            continue
        metrics = rec.get("metrics") or {}
        slot = agg.setdefault(variant, {"impressions": 0.0, "engagements": 0.0,
                                        "clips": 0})
        views = float(metrics.get("views", 0) or 0)
        slot["impressions"] += views
        slot["engagements"] += views * float(metrics.get("engagement_rate", 0) or 0)
        slot["clips"] += 1
    summary = {}
    for variant, slot in sorted(agg.items()):
        imp = slot["impressions"]
        summary[variant] = {
            "impressions": imp,
            "clips": slot["clips"],
            "ctr": (slot["engagements"] / imp) if imp > 0 else 0.0,
        }
    return summary
