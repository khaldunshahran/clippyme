"""Published clip performance analytics tracking and Zernio synchronization.

Stores performance records (views, likes, shares, comments, retention) for
clips published across TikTok, Instagram Reels, and YouTube Shorts.
Provides data to performance_feedback.py to power the automated continuous
learning loop in clip generation.
"""
from __future__ import annotations

import json
import logging
import os
import re
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_LOCK = threading.Lock()
ANALYTICS_FILE_PATH = Path("data/published_analytics.json")


def _ensure_data_dir() -> Path:
    target = ANALYTICS_FILE_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def _keep(new_val, old_val):
    """Prefer the new value, but never let a ``None`` update clobber stored data."""
    return old_val if new_val is None else new_val


def load_analytics_data() -> Dict[str, Any]:
    """Load the analytics registry with fallback to default empty shape."""
    path = ANALYTICS_FILE_PATH
    if not path.exists():
        return {"version": 1, "updated_at": None, "clips": {}}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict) and "clips" in data:
                return data
            return {"version": 1, "updated_at": None, "clips": {}}
    except Exception as exc:
        logger.warning("analytics_service: failed to read %s: %s", path, exc)
        return {"version": 1, "updated_at": None, "clips": {}}


def save_analytics_data(data: Dict[str, Any]) -> bool:
    """Atomic write of analytics registry."""
    path = _ensure_data_dir()
    with _LOCK:
        data["updated_at"] = datetime.now(timezone.utc).isoformat()
        try:
            tmp_fd, tmp_path = tempfile.mkstemp(dir=path.parent, prefix="analytics_", suffix=".tmp")
            with open(tmp_fd, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            # Atomic rename on Windows/POSIX
            os.replace(tmp_path, path)
            return True
        except Exception as exc:
            logger.error("analytics_service: atomic write failed: %s", exc)
            return False


def record_published_clip(
    job_id: str,
    clip_index: int,
    clip_info: Dict[str, Any],
    publish_result: Dict[str, Any],
    platforms: Optional[List[Any]] = None,
) -> Dict[str, Any]:
    """Record a clip after it is published/scheduled to social platforms.

    Phase 3A: retains ``deterministic_features``, ``deterministic_score``,
    ``viral_score`` and ``title_variants`` from ``clip_info`` so the
    virality calibration loop can correlate features with real outcomes.
    Updates carrying ``None`` for these fields never clobber values already
    stored (``_keep`` pattern).
    """
    data = load_analytics_data()
    clip_id = f"{job_id}:{clip_index}"
    existing = data["clips"].get(clip_id, {})

    start = float(clip_info.get("start") or 0.0)
    end = float(clip_info.get("end") or 0.0)
    duration = max(0.0, end - start)
    tier = (
        clip_info.get("duration_tier")
        or ("short" if duration <= 60.0 else ("mid" if duration <= 120.0 else "extended"))
    )

    plat_names = []
    if platforms:
        for p in platforms:
            if isinstance(p, str):
                plat_names.append(p.lower())
            elif isinstance(p, dict):
                name = p.get("platform") or p.get("id")
                if name:
                    plat_names.append(str(name).lower())

    record = {
        "clip_id": clip_id,
        "job_id": job_id,
        "clip_index": clip_index,
        "title": clip_info.get("video_title_for_youtube_short") or f"Clip {clip_index + 1}",
        "hook_text": clip_info.get("viral_hook_text") or clip_info.get("hook") or "",
        "viral_reason": clip_info.get("viral_reason") or "",
        "speaker_name": clip_info.get("speaker_name") or "",
        "duration": round(duration, 2),
        "duration_tier": tier,
        "platforms": list(dict.fromkeys(plat_names)),
        "post_id": publish_result.get("post_id"),
        "scheduled_for": publish_result.get("scheduled_for"),
        "published_at": datetime.now(timezone.utc).isoformat(),
        # Phase 3A: virality feedback inputs — retained from clip_info;
        # None-valued updates never clobber already-stored values.
        "deterministic_features": _keep(
            clip_info.get("deterministic_features"),
            existing.get("deterministic_features")),
        "deterministic_score": _keep(
            clip_info.get("deterministic_score"),
            existing.get("deterministic_score")),
        "viral_score": _keep(
            clip_info.get("viral_score"), existing.get("viral_score")),
        "title_variants": _keep(
            clip_info.get("title_variants"), existing.get("title_variants")),
        "metrics": {
            "views": 0,
            "likes": 0,
            "shares": 0,
            "comments": 0,
            "retention_rate": 0.0,
            "engagement_rate": 0.0,
        },
        "last_synced": None,
    }

    # If record already exists, preserve its current metrics
    if clip_id in data["clips"]:
        existing_metrics = data["clips"][clip_id].get("metrics")
        if existing_metrics:
            record["metrics"] = existing_metrics
            record["last_synced"] = data["clips"][clip_id].get("last_synced")

    data["clips"][clip_id] = record
    save_analytics_data(data)
    logger.info("analytics_service: recorded published clip %s (tier=%s, dur=%.1fs)", clip_id, tier, duration)
    return record


def update_clip_metrics(clip_id: str, metrics: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Update views, likes, shares, comments, retention for a tracked clip."""
    data = load_analytics_data()
    if clip_id not in data["clips"]:
        logger.warning("analytics_service: clip %s not found in registry", clip_id)
        return None

    clip = data["clips"][clip_id]
    current = clip.setdefault("metrics", {})

    views = int(metrics.get("views", current.get("views", 0)))
    likes = int(metrics.get("likes", current.get("likes", 0)))
    shares = int(metrics.get("shares", current.get("shares", 0)))
    comments = int(metrics.get("comments", current.get("comments", 0)))
    retention = float(metrics.get("retention_rate", current.get("retention_rate", 0.0)))

    # Engagement rate: (likes + comments + shares) / max(views, 1)
    eng_rate = round((likes + comments + shares) / max(views, 1), 4) if views > 0 else 0.0

    current["views"] = views
    current["likes"] = likes
    current["shares"] = shares
    current["comments"] = comments
    current["retention_rate"] = round(retention, 4)
    current["engagement_rate"] = eng_rate

    clip["last_synced"] = datetime.now(timezone.utc).isoformat()
    save_analytics_data(data)
    return clip


def sync_zernio_analytics(api_key: Optional[str] = None) -> Dict[str, Any]:
    """Fetch live post performance from Zernio Analytics API for all tracked posts."""
    from clippyme.storage.config_store import load_zernio_config

    cfg = load_zernio_config()
    key = api_key or cfg.get("api_key")
    if not key:
        logger.info("analytics_service: no Zernio API key configured; skipping live network sync")
        return {"synced": 0, "status": "no_api_key"}

    data = load_analytics_data()
    posts_to_sync = [
        (cid, c.get("post_id"))
        for cid, c in data.get("clips", {}).items()
        if c.get("post_id")
    ]

    if not posts_to_sync:
        return {"synced": 0, "status": "no_tracked_posts"}

    from clippyme.integrations.social_publisher import ZernioClient, ZernioError

    client = ZernioClient(key)
    synced_count = 0

    for clip_id, post_id in posts_to_sync:
        try:
            # Query Zernio analytics for this post
            # Endpoint per Zernio docs: GET /analytics/posts with postId
            resp = client._request("GET", f"/analytics/posts", params={"postId": post_id})
            if not resp or not isinstance(resp, dict):
                # Fallback to single post GET
                resp = client._request("GET", f"/posts/{post_id}")

            if isinstance(resp, dict):
                metrics = resp.get("analytics") or resp.get("metrics") or resp
                update_clip_metrics(
                    clip_id,
                    {
                        "views": metrics.get("views") or metrics.get("impressions") or 0,
                        "likes": metrics.get("likes") or metrics.get("reactions") or 0,
                        "shares": metrics.get("shares") or metrics.get("reposts") or 0,
                        "comments": metrics.get("comments") or 0,
                        "retention_rate": metrics.get("retentionRate") or metrics.get("completionRate") or 0.0,
                    },
                )
                synced_count += 1
        except ZernioError as ze:
            logger.debug("analytics_service: Zernio analytics call for post %s failed: %s", post_id, ze)
        except Exception as exc:
            logger.debug("analytics_service: error syncing post %s: %s", post_id, exc)

    return {
        "synced": synced_count,
        "synced_count": synced_count,
        "total": len(posts_to_sync),
        "status": "ok",
        "message": f"Successfully synced {synced_count} clips from Zernio.",
    }


def get_analytics_summary() -> Dict[str, Any]:
    """Aggregate high-level stats, top-performing clips, and platform breakdown."""
    data = load_analytics_data()
    clips = list(data.get("clips", {}).values())

    total_published = len(clips)
    total_views = sum(c.get("metrics", {}).get("views", 0) for c in clips)
    total_likes = sum(c.get("metrics", {}).get("likes", 0) for c in clips)
    total_shares = sum(c.get("metrics", {}).get("shares", 0) for c in clips)
    total_comments = sum(c.get("metrics", {}).get("comments", 0) for c in clips)

    clips_with_retention = [c for c in clips if c.get("metrics", {}).get("retention_rate", 0) > 0]
    avg_retention = (
        round(sum(c["metrics"]["retention_rate"] for c in clips_with_retention) / len(clips_with_retention), 4)
        if clips_with_retention
        else 0.0
    )

    # Sort top clips by viral impact (weighted: shares * 5 + comments * 3 + likes)
    def _viral_impact(c):
        m = c.get("metrics", {})
        return (m.get("shares", 0) * 5.0) + (m.get("comments", 0) * 3.0) + m.get("likes", 0)

    top_clips = sorted(clips, key=_viral_impact, reverse=True)[:10]

    # Platform breakdown
    platforms_stat: Dict[str, Dict[str, int]] = {}
    for c in clips:
        m = c.get("metrics", {})
        for p in c.get("platforms", ["tiktok"]):
            st = platforms_stat.setdefault(p, {"views": 0, "shares": 0, "likes": 0, "posts": 0})
            st["views"] += m.get("views", 0)
            st["shares"] += m.get("shares", 0)
            st["likes"] += m.get("likes", 0)
            st["posts"] += 1

    # Enrich clips with top-level fields for frontend views
    enriched_clips = []
    for c in sorted(clips, key=lambda x: x.get("published_at") or "", reverse=True):
        m = c.get("metrics", {})
        item = dict(c)
        item["views"] = m.get("views", 0)
        item["likes"] = m.get("likes", 0)
        item["shares"] = m.get("shares", 0)
        item["comments"] = m.get("comments", 0)
        item["retention_rate"] = m.get("retention_rate", 0.0)
        dur = float(c.get("duration", 0.0))
        item["narrative_tier"] = c.get("duration_tier") or ("mid" if dur > 60 else "short")
        enriched_clips.append(item)

    return {
        "total_published": total_published,
        "total_views": total_views,
        "total_likes": total_likes,
        "total_shares": total_shares,
        "total_comments": total_comments,
        "avg_retention": avg_retention,
        "platform_breakdown": platforms_stat,
        "top_clips": top_clips,
        "all_clips": enriched_clips,
        "clips": enriched_clips,
        "last_updated": data.get("updated_at"),
    }
