"""Phase 2 (2E): audience-aware scheduling — pure helper tests.

No network, no Zernio: hourly weights, window derivation, platform
fallbacks, and SmartScheduler wiring (90-min gap + collision rule untouched).
"""
import random
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from clippyme.integrations.audience_schedule import (
    MIN_HISTORY_RECORDS,
    PLATFORM_FALLBACK_WINDOWS,
    hourly_weights,
    normalize_platform,
    slot_windows_for,
    windows_from_history,
)
from clippyme.integrations.social_publisher import (
    DEFAULT_SLOT_WINDOWS,
    MIN_GAP_BETWEEN_POSTS_SECONDS,
    SmartScheduler,
)


def _rec(platform, hour, views, account=None, likes=0):
    r = {"platform": platform, "hour": hour, "views": views, "likes": likes}
    if account:
        r["account_id"] = account
    return r


HIST = (
    [_rec("tiktok", 19, 1000, "acc1") for _ in range(4)]
    + [_rec("tiktok", 20, 800, "acc1") for _ in range(3)]
    + [_rec("tiktok", 3, 5, "acc1") for _ in range(3)]
    + [_rec("instagram", 12, 500, "acc2") for _ in range(5)]
    + [_rec("tiktok", 19, 10, "other") for _ in range(6)]
)


def test_min_history_threshold_documented():
    assert MIN_HISTORY_RECORDS == 5


def test_normalize_platform_aliases():
    assert normalize_platform("TikTok") == "tiktok"
    assert normalize_platform("instagram reels") == "instagram"
    assert normalize_platform("IG") == "instagram"
    assert normalize_platform("YouTube Shorts") == "youtube"


def test_exact_account_match_beats_platform_pool():
    w = hourly_weights(HIST, platform="tiktok", account="acc1")
    assert w[19] == 4000.0 and w[20] == 2400.0
    assert 12 not in w  # acc2's instagram rows excluded


def test_platform_fallback_when_account_thin():
    w = hourly_weights(HIST, platform="tiktok", account="nope")
    assert w[19] == 4060.0  # platform-level pool includes every account


def test_thin_history_returns_none():
    thin = [_rec("tiktok", 19, 100) for _ in range(MIN_HISTORY_RECORDS - 1)]
    assert hourly_weights(thin, platform="tiktok") is None
    assert windows_from_history(thin, platform="tiktok") is None


def test_windows_merge_contiguous_hot_hours():
    assert windows_from_history(HIST, platform="tiktok", account="acc1") == [(19, 21)]


def test_windows_none_when_no_engagement():
    flat = [_rec("tiktok", h, 0) for h in (9, 10, 11, 12, 13)]
    assert windows_from_history(flat, platform="tiktok") is None


def test_slot_windows_for_learned_and_fallbacks():
    sw = slot_windows_for(platform="tiktok", account="acc1", history=HIST)
    assert sw[0] == [(19, 21)] and len(sw) == 7
    assert slot_windows_for(platform="tiktok")[0] == PLATFORM_FALLBACK_WINDOWS["tiktok"]
    assert slot_windows_for(platform="instagram")[0] == [(12, 13), (18, 21)]
    assert slot_windows_for(platform="youtube")[0] == [(16, 19)]
    assert slot_windows_for(platform="vimeo") is None
    assert slot_windows_for() is None


def test_native_analytics_record_shape():
    native = [
        {"platforms": ["tiktok"], "scheduled_for": "2026-09-20T19:30:00+02:00",
         "metrics": {"views": 900, "likes": 40, "shares": 5, "comments": 8}}
        for _ in range(5)
    ]
    w = hourly_weights(native, platform="tiktok")
    assert w == {19: 5 * (900 + 400 + 160 + 150)}


def test_scheduler_legacy_default_untouched():
    s = SmartScheduler(rng=random.Random(1))
    assert s.slot_windows == dict(DEFAULT_SLOT_WINDOWS)
    assert s.min_gap_seconds == MIN_GAP_BETWEEN_POSTS_SECONDS == 5400


def test_scheduler_platform_fallback_windows():
    s = SmartScheduler(platform="tiktok", rng=random.Random(1))
    assert s.slot_windows[0] == [(18, 23)]
    assert s.min_gap_seconds == 5400  # collision rule preserved


def test_scheduler_learned_windows():
    s = SmartScheduler(platform="tiktok", account_id="acc1",
                       analytics_history=HIST, rng=random.Random(1))
    assert s.slot_windows[3] == [(19, 21)]


def test_scheduler_explicit_windows_win():
    custom = {d: [(10, 11)] for d in range(7)}
    s = SmartScheduler(slot_windows=custom, platform="tiktok",
                       analytics_history=HIST, rng=random.Random(1))
    assert s.slot_windows == custom


def test_find_slot_uses_learned_window_and_gap():
    tz = ZoneInfo("Europe/Rome")
    now = datetime.now(tz)
    day = (now + timedelta(days=1)).date()
    s = SmartScheduler(platform="tiktok", account_id="acc1",
                       analytics_history=HIST, rng=random.Random(7))
    slot = s.find_slot(day, [], now=now)
    assert slot.hour in (19, 20)
    slot2 = s.find_slot(day, [slot], now=now)
    assert abs((slot2 - slot).total_seconds()) >= 5400
