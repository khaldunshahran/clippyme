"""2E: audience-aware scheduling helpers (pure, host-testable).

Derives per-hour engagement weights from post analytics history so the
SmartScheduler picks slots when *this account's* audience is actually
watching, instead of the one-size-fits-all prime-time table.

History record shapes understood (all fields optional; junk records skipped):

  Native ``analytics_service`` records::

      {"platforms": ["tiktok"], "account_id": ...,        # account_id rare
       "scheduled_for": "<iso>", "published_at": "<iso>",
       "metrics": {"views":.., "likes":.., "shares":.., "comments":..}}

  Generic caller-supplied records::

      {"platform": "tiktok", "account_id" | "accountId": "..",
       "hour": 18,                       # local post hour, 0-23
       "published_at" | "posted_at": "<iso>",
       "views":.., "likes":.., "comments":.., "shares":..,
       "engagement": <float>}            # explicit score wins when present

Matching order: exact account match first (when the caller supplies account
ids); platform-level aggregate as the fallback. Fewer than
``MIN_HISTORY_RECORDS`` *useful* records (hour + engagement signal present)
-> the history is "thin" and these helpers return None, in which case the
scheduler falls back to the documented conservative per-platform local
windows below.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

logger = logging.getLogger("clippyme")

# 2E: conservative local-time fallback windows per platform, used when
# analytics history is too thin to learn from. Deliberately wide rather
# than clever — posting inside these beats posting at 4am.
PLATFORM_FALLBACK_WINDOWS: dict[str, list[tuple[int, int]]] = {
    "tiktok": [(18, 23)],               # TikTok: evening prime time
    "instagram": [(12, 13), (18, 21)],  # Reels: lunch break + evening
    "youtube": [(16, 19)],              # Shorts: late afternoon
}

# "Thin history" threshold: fewer useful records than this -> don't learn.
MIN_HISTORY_RECORDS = 5

_PLATFORM_ALIASES = {
    "tiktok": "tiktok",
    "instagram": "instagram", "ig": "instagram", "reels": "instagram",
    "youtube": "youtube", "yt": "youtube", "shorts": "youtube",
}


def normalize_platform(platform) -> str:
    """Lowercase platform name with common aliases resolved."""
    p = str(platform or "").strip().lower()
    if p in _PLATFORM_ALIASES:
        return _PLATFORM_ALIASES[p]
    first = p.split()[0] if p.split() else ""
    return _PLATFORM_ALIASES.get(first, first)


def _record_platforms(rec: dict) -> list[str]:
    plats = rec.get("platforms")
    if isinstance(plats, (list, tuple)):
        return [normalize_platform(p) for p in plats if p]
    p = rec.get("platform")
    return [normalize_platform(p)] if p else []


def _record_account(rec: dict):
    for key in ("account_id", "accountId", "account"):
        val = rec.get(key)
        if val:
            return str(val)
    return None


def _record_hour(rec: dict, tz=None):
    """Local post hour (0-23): explicit ``hour`` field, else ISO timestamp."""
    hour = rec.get("hour")
    if hour is not None:
        try:
            hi = int(hour)
            return hi if 0 <= hi <= 23 else None
        except (TypeError, ValueError):
            pass
    for key in ("scheduled_for", "published_at", "posted_at", "timestamp"):
        ts = rec.get(key)
        if not ts or not isinstance(ts, str):
            continue
        try:
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except ValueError:
            continue
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        if tz is not None:
            dt = dt.astimezone(tz)
        return dt.hour
    return None


def _record_engagement(rec: dict):
    """Numeric engagement signal; None when the record carries no metrics.

    Explicit ``engagement``/``score`` wins; otherwise a documented heuristic
    over views/likes/comments/shares (shares weigh most — they drive reach).
    All-zero metrics still count as a *useful* record (weight 0.0).
    """
    metrics = rec.get("metrics")
    src = metrics if isinstance(metrics, dict) else rec
    for key in ("engagement", "score"):
        val = src.get(key)
        if isinstance(val, (int, float)) and val == val:  # not NaN
            return float(val)

    def _num(key: str) -> float:
        try:
            return float(src.get(key) or 0)
        except (TypeError, ValueError):
            return 0.0

    if not any(k in src for k in ("views", "likes", "shares", "comments")):
        return None
    return _num("views") + 10.0 * _num("likes") + 20.0 * _num("comments") + 30.0 * _num("shares")


def _useful_pairs(records, tz=None) -> list[tuple[int, float]]:
    """(hour, engagement) for every record carrying both signals."""
    out = []
    for rec in records or []:
        if not isinstance(rec, dict):
            continue
        hour = _record_hour(rec, tz)
        eng = _record_engagement(rec)
        if hour is None or eng is None:
            continue
        out.append((rec, hour, eng))
    return out


def hourly_weights(history, platform=None, account=None,
                   min_records: int = MIN_HISTORY_RECORDS, tz=None):
    """Aggregate engagement per local hour -> ``{hour: weight}``.

    Exact account match first; platform-level aggregate as fallback.
    Returns None when fewer than ``min_records`` useful records exist
    ("thin history") — the caller then uses the platform fallback windows.
    """
    pairs = _useful_pairs(history, tz)
    if not pairs:
        return None
    key = normalize_platform(platform)
    pool = None
    if account:
        exact = [(h, e) for r, h, e in pairs if _record_account(r) == str(account)]
        if len(exact) >= min_records:
            pool = exact
    if pool is None and key:
        plat = [(h, e) for r, h, e in pairs if key in _record_platforms(r)]
        if len(plat) >= min_records:
            pool = plat
    if pool is None:
        return None
    weights: dict[int, float] = {}
    for hour, eng in pool:
        weights[hour] = weights.get(hour, 0.0) + eng
    return weights


def windows_from_history(history, platform=None, account=None,
                         min_records: int = MIN_HISTORY_RECORDS, tz=None):
    """Merge hot hours into ``[(start, end)]`` windows; None when thin/flat.

    "Hot" = hours with at least half the peak hour's engagement; contiguous
    hot hours merge into one window. A zero peak (no engagement anywhere)
    also yields None so the scheduler falls back instead of learning noise.
    """
    weights = hourly_weights(history, platform=platform, account=account,
                             min_records=min_records, tz=tz)
    if not weights:
        return None
    peak = max(weights.values())
    if peak <= 0:
        return None
    hot = sorted(h for h, w in weights.items() if w >= 0.5 * peak)
    windows = []
    start = prev = hot[0]
    for hour in hot[1:]:
        if hour == prev + 1:
            prev = hour
            continue
        windows.append((start, prev + 1))
        start = prev = hour
    windows.append((start, prev + 1))
    return windows


def slot_windows_for(platform=None, account=None, history=None, tz=None):
    """2E: ``{weekday: [(start, end)]}`` for the scheduler, or None.

    Learned windows when history is sufficient, else the conservative
    per-platform fallback windows when the platform is known. None means
    "keep the caller's default" (preserves legacy behavior exactly).
    """
    if history:
        learned = windows_from_history(history, platform=platform,
                                       account=account, tz=tz)
        if learned:
            return {d: list(learned) for d in range(7)}
    key = normalize_platform(platform)
    if key in PLATFORM_FALLBACK_WINDOWS:
        return {d: list(PLATFORM_FALLBACK_WINDOWS[key]) for d in range(7)}
    return None


def load_analytics_history() -> list[dict]:
    """Best-effort load of native analytics records for 2E. Never raises."""
    try:
        from clippyme.domain.analytics_service import load_analytics_data
        data = load_analytics_data() or {}
        clips = data.get("clips", {})
        recs = list(clips.values()) if isinstance(clips, dict) else []
        return [r for r in recs if isinstance(r, dict)]
    except Exception as exc:  # analytics store missing/corrupt -> no learning
        logger.warning("audience_schedule: analytics history unavailable: %s", exc)
        return []
