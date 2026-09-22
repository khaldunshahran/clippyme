"""Channel Profile Management & Routing Service.

Allows users to manage multiple social channels/accounts (e.g. US Politics,
Breaking World News, Pop & Entertainment) directly within ClippyMe.

Each channel profile configures:
- Niche and topic category associations
- Default styling presets (captions, fonts, colors, aspect ratio)
- Attribution banner platform and handle (e.g. @USPoliticsDaily)
- Connected publishing destination account IDs
"""
from __future__ import annotations

import contextlib
import json
import logging
import os
import re
import tempfile
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger("clippyme")

DATA_DIR = "data"
CHANNELS_FILE = os.path.join(DATA_DIR, "channels.json")
_CHANNELS_LOCK = threading.RLock()

DEFAULT_CHANNELS = [
    {
        "id": "ch_us_politics",
        "name": "US Politics Daily",
        "description": "High-stakes congressional debates, election updates, and political commentary for US audiences.",
        "niches": ["politics", "nation"],
        "default_preset": "viral",
        "banner_platform": "youtube",
        "banner_handle": "@USPoliticsDaily",
        "banner_y_pct": 0.85,
        "reframe_mode": "auto",
        "sub_preset": "hormozi_bold",
        "sub_font": "Montserrat-Black",
        "sub_color": "#FFFFFF",
        "publishing_targets": {"youtube": True, "tiktok": True, "instagram": False},
        "created_at": "2026-09-16T00:00:00Z",
        "updated_at": "2026-09-16T00:00:00Z",
    },
    {
        "id": "ch_breaking_world",
        "name": "Global Breaking & Crisis",
        "description": "Rapid response coverage on breaking world events, disasters, and international press briefings.",
        "niches": ["breaking_world", "world"],
        "default_preset": "viral",
        "banner_platform": "tiktok",
        "banner_handle": "@GlobalBreakingNow",
        "banner_y_pct": 0.85,
        "reframe_mode": "auto",
        "sub_preset": "classic",
        "sub_font": "Inter-Bold",
        "sub_color": "#FFFFFF",
        "publishing_targets": {"youtube": True, "tiktok": True, "instagram": True},
        "created_at": "2026-09-16T00:00:00Z",
        "updated_at": "2026-09-16T00:00:00Z",
    },
    {
        "id": "ch_pop_entertainment",
        "name": "Pop Culture & Entertainment",
        "description": "Trending celebrity news, red carpets, creator drama, and viral pop moments.",
        "niches": ["entertainment", "culture"],
        "default_preset": "viral",
        "banner_platform": "instagram",
        "banner_handle": "@PopCultureBuzz",
        "banner_y_pct": 0.85,
        "reframe_mode": "auto",
        "sub_preset": "hormozi_bold",
        "sub_font": "Montserrat-Black",
        "sub_color": "#FFE500",
        "publishing_targets": {"youtube": False, "tiktok": True, "instagram": True},
        "created_at": "2026-09-16T00:00:00Z",
        "updated_at": "2026-09-16T00:00:00Z",
    },
]


def load_channels(filepath: str = CHANNELS_FILE) -> List[Dict[str, Any]]:
    """Load channel profiles from persistent store. Seeds default channels if empty."""
    with _CHANNELS_LOCK:
        if not os.path.exists(filepath):
            save_channels(DEFAULT_CHANNELS, filepath)
            return list(DEFAULT_CHANNELS)
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    return data
                if isinstance(data, dict) and "channels" in data:
                    return data["channels"]
                return list(DEFAULT_CHANNELS)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Error reading channels file %s: %s", filepath, exc)
            return list(DEFAULT_CHANNELS)


def save_channels(channels: List[Dict[str, Any]], filepath: str = CHANNELS_FILE) -> bool:
    """Atomically save channel profiles to disk with safe permissions."""
    with _CHANNELS_LOCK:
        tmp_path = None
        try:
            dir_path = os.path.dirname(filepath) or "."
            os.makedirs(dir_path, mode=0o700, exist_ok=True)
            fd, tmp_path = tempfile.mkstemp(prefix=".channels-", suffix=".tmp", dir=dir_path)
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(channels, f, indent=2, ensure_ascii=False)
            os.replace(tmp_path, filepath)
            return True
        except Exception as exc:
            logger.error("Failed to save channels: %s", exc)
            if tmp_path and os.path.exists(tmp_path):
                with contextlib.suppress(OSError):
                    os.remove(tmp_path)
            return False


def get_channel(channel_id: str, filepath: str = CHANNELS_FILE) -> Optional[Dict[str, Any]]:
    """Retrieve a single channel by ID."""
    channels = load_channels(filepath)
    for ch in channels:
        if ch.get("id") == channel_id:
            return ch
    return None


def create_channel(data: Dict[str, Any], filepath: str = CHANNELS_FILE) -> Dict[str, Any]:
    """Create a new channel profile and persist."""
    channels = load_channels(filepath)
    raw_name = str(data.get("name") or "New Channel").strip()
    slug = re.sub(r"[^a-zA-Z0-9_]+", "_", raw_name.lower()).strip("_") or "channel"
    channel_id = f"ch_{slug}"

    # Ensure unique ID
    existing_ids = {ch["id"] for ch in channels}
    suffix = 1
    base_id = channel_id
    while channel_id in existing_ids:
        channel_id = f"{base_id}_{suffix}"
        suffix += 1

    now_iso = datetime.now(timezone.utc).isoformat()
    new_channel = {
        "id": channel_id,
        "name": raw_name,
        "description": str(data.get("description") or ""),
        "niches": [n.lower().strip() for n in data.get("niches") or ["general"]],
        "default_preset": data.get("default_preset") or "viral",
        "banner_platform": data.get("banner_platform") or "youtube",
        "banner_handle": data.get("banner_handle") or f"@{slug}",
        "banner_y_pct": float(data.get("banner_y_pct") or 0.85),
        "reframe_mode": data.get("reframe_mode") or "auto",
        "sub_preset": data.get("sub_preset") or "hormozi_bold",
        "sub_font": data.get("sub_font") or "Montserrat-Black",
        "sub_color": data.get("sub_color") or "#FFFFFF",
        "publishing_targets": data.get("publishing_targets") or {"youtube": True, "tiktok": True, "instagram": False},
        "created_at": now_iso,
        "updated_at": now_iso,
    }

    channels.append(new_channel)
    save_channels(channels, filepath)
    return new_channel


def update_channel(channel_id: str, patch: Dict[str, Any], filepath: str = CHANNELS_FILE) -> Optional[Dict[str, Any]]:
    """Update an existing channel profile."""
    channels = load_channels(filepath)
    target = None
    for ch in channels:
        if ch.get("id") == channel_id:
            target = ch
            break
    if not target:
        return None

    # Updatable fields
    updatable = (
        "name", "description", "niches", "default_preset", "banner_platform",
        "banner_handle", "banner_y_pct", "reframe_mode", "sub_preset",
        "sub_font", "sub_color", "publishing_targets"
    )
    for field in updatable:
        if field in patch and patch[field] is not None:
            target[field] = patch[field]

    target["updated_at"] = datetime.now(timezone.utc).isoformat()
    save_channels(channels, filepath)
    return target


def delete_channel(channel_id: str, filepath: str = CHANNELS_FILE) -> bool:
    """Delete a channel profile by ID."""
    channels = load_channels(filepath)
    initial_len = len(channels)
    channels = [ch for ch in channels if ch.get("id") != channel_id]
    if len(channels) == initial_len:
        return False
    return save_channels(channels, filepath)


def match_channel_for_category(category: Optional[str], filepath: str = CHANNELS_FILE) -> Optional[Dict[str, Any]]:
    """Find the best-fitting channel profile for a given topic category."""
    if not category:
        channels = load_channels(filepath)
        return channels[0] if channels else None

    cat_lower = category.lower().strip()
    channels = load_channels(filepath)

    # 1. Exact match in niches
    for ch in channels:
        niches = [n.lower().strip() for n in ch.get("niches", [])]
        if cat_lower in niches:
            return ch

    # 2. Substring match (e.g. "breaking_world" matching "world" or "breaking")
    for ch in channels:
        niches = [n.lower().strip() for n in ch.get("niches", [])]
        if any(n in cat_lower or cat_lower in n for n in niches):
            return ch

    # Fallback to first channel
    return channels[0] if channels else None
