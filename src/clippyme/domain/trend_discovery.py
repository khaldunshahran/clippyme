"""Trend Discovery & AI Sourcing Engine — domain logic.

Aggregates raw US trending topics (Google News US, Google Trends US),
prompts Gemini to identify viral hooks and long-form YouTube search queries,
searches YouTube via yt-dlp for primary source footage (> 5 min),
and maintains the curated Trend Radar state with atomic persistence.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import re
import subprocess
import tempfile
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import json_repair

from clippyme.integrations.trend_sources import (
    TrendSourceItem,
    collect_raw_us_trends,
)

logger = logging.getLogger("clippyme")

DATA_DIR = "data"
TREND_RADAR_FILE = os.path.join(DATA_DIR, "trend_radar.json")
_STATE_LOCK = threading.RLock()

# Retention: keep discovered items for 48 hours
RETENTION_HOURS = 48

DEFAULT_CATEGORIES = ["politics", "world", "entertainment", "nation"]


# ---------------------------------------------------------------------------
# Pure Helpers (Host-testable)
# ---------------------------------------------------------------------------


def clean_json_markdown(text: str) -> str:
    """Strip markdown json code blocks if present."""
    s = (text or "").strip()
    if s.startswith("```"):
        lines = s.split("\n")
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        s = "\n".join(lines).strip()
    return s


def format_duration(seconds: Optional[int]) -> str:
    """Format duration seconds to MM:SS or H:MM:SS."""
    if seconds is None or seconds < 0:
        return "0:00"
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def make_topic_id(title: str) -> str:
    """Generate a stable, path-safe hash for a topic title."""
    clean = re.sub(r"[^a-zA-Z0-9]+", "", (title or "").lower().strip())
    digest = hashlib.sha256(clean.encode("utf-8")).hexdigest()[:16]
    return f"tr_{digest}"


def build_gemini_analysis_prompt(items: List[TrendSourceItem], target_count: int = 10) -> str:
    """Build the prompt instructing Gemini to analyze US virality and output search queries."""
    summaries = []
    for idx, item in enumerate(items, 1):
        line = f"{idx}. [{item.category.upper()}] {item.title}"
        if item.approx_traffic:
            line += f" (Search volume: {item.approx_traffic})"
        if item.snippet:
            line += f" - {item.snippet[:140]}"
        summaries.append(line)

    bullet_list = "\n".join(summaries)

    return f"""You are an elite short-form video strategist creating viral TikToks, YouTube Shorts, and IG Reels for a UNITED STATES target audience.
Analyze the following live trending headlines and search spikes currently dominating US internet discussions:

{bullet_list}

Select up to {target_count} topics with the HIGHEST VIRALITY POTENTIAL in the United States.
Focus on:
1. US & Global Politics (intense debates, high-stakes hearings, polarizing clashes, major policy reveals)
2. Sudden Breaking News & Global Crises (disasters, unexpected dramatic incidents, rescue operations, major press briefings)
3. Entertainment & Pop Culture (celebrity bombshells, major announcements, trending cultural debates)

For each selected topic, return a JSON array of objects with the exact schema:
[
  {{
    "topic_title": "Concise, punchy topic headline",
    "category": "politics" | "breaking_world" | "entertainment" | "culture",
    "virality_score": 1-100 integer (score based on emotional trigger, curiosity, polarization, shock value),
    "viral_hook": "1-2 sentences explaining why this will explode on US social feeds (what triggers viewers to comment, share, or watch to the end)",
    "youtube_query": "Specific, optimized search query to find the PRIMARY LONG-FORM SOURCE video (5 to 60 minutes long, such as full press conference, congressional hearing, podcast interview, or full news report). DO NOT include words like 'shorts', 'tiktok', or 'clip' in the query.",
    "suggested_preset": "viral" | "hormozi" | "commentary"
  }}
]

Respond ONLY with valid JSON. Do not include markdown preamble.
"""


def parse_gemini_analysis_response(raw_text: str) -> List[Dict[str, Any]]:
    """Parse Gemini's response using json_repair."""
    clean = clean_json_markdown(raw_text)
    if not clean:
        return []
    try:
        data = json_repair.loads(clean)
        if isinstance(data, list):
            valid = []
            for entry in data:
                if isinstance(entry, dict) and entry.get("topic_title"):
                    valid.append(
                        {
                            "topic_title": str(entry.get("topic_title", "")).strip(),
                            "category": str(entry.get("category", "politics")).strip().lower(),
                            "virality_score": max(1, min(100, int(entry.get("virality_score", 75)))),
                            "viral_hook": str(entry.get("viral_hook", "")).strip(),
                            "youtube_query": str(entry.get("youtube_query", "")).strip(),
                            "suggested_preset": str(entry.get("suggested_preset", "viral")).strip(),
                        }
                    )
            return valid
        if isinstance(data, dict) and "topics" in data and isinstance(data["topics"], list):
            return parse_gemini_analysis_response(json.dumps(data["topics"]))
    except Exception as exc:
        logger.warning("Failed to parse Gemini trend analysis response: %s", exc)
    return []


def filter_and_rank_yt_candidates(
    entries: List[Dict[str, Any]],
    min_duration: int = 300,  # 5 minutes minimum
    max_duration: int = 7200,  # 2 hours maximum
) -> Optional[Dict[str, Any]]:
    """Filter yt-dlp flat-playlist entries and return the best long-form source."""
    if not entries:
        return None

    long_candidates = []
    fallback_candidates = []

    for item in entries:
        if not isinstance(item, dict):
            continue
        duration = item.get("duration")
        video_id = item.get("id")
        title = item.get("title") or ""
        if not video_id or not title:
            continue

        url = item.get("url")
        if not url or not url.startswith("http"):
            url = f"https://www.youtube.com/watch?v={video_id}"

        # Thumbnail extraction
        thumbnails = item.get("thumbnails") or []
        thumb_url = ""
        if thumbnails and isinstance(thumbnails, list):
            thumb_url = thumbnails[-1].get("url") or ""
        if not thumb_url:
            thumb_url = f"https://i.ytimg.com/vi/{video_id}/hqdefault.jpg"

        candidate = {
            "video_id": str(video_id),
            "video_url": url,
            "title": str(title),
            "channel": str(item.get("channel") or item.get("uploader") or "YouTube"),
            "duration": int(duration) if duration is not None else 0,
            "duration_string": format_duration(duration),
            "thumbnail_url": thumb_url,
            "view_count": int(item.get("view_count") or 0),
        }

        if duration and min_duration <= duration <= max_duration:
            long_candidates.append(candidate)
        elif duration and duration >= 120:  # 2+ minutes fallback
            fallback_candidates.append(candidate)

    if long_candidates:
        # Sort by view_count if available, otherwise preserve search relevance order
        return long_candidates[0]
    if fallback_candidates:
        return fallback_candidates[0]

    return None


def heuristic_trend_analysis(raw_items: List[TrendSourceItem], limit: int = 10) -> List[Dict[str, Any]]:
    """Heuristic fallback if Gemini API is unavailable."""
    results = []
    seen = set()

    for item in raw_items:
        clean_title = item.title.strip()
        if not clean_title or clean_title in seen:
            continue
        seen.add(clean_title)

        cat = item.category
        if cat == "trending_spike":
            cat = "breaking_world" if any(w in clean_title.lower() for w in ("disaster", "earthquake", "crash", "war", "crisis")) else "culture"

        score = 80
        if item.approx_traffic:
            score = 90 if "M" in item.approx_traffic or "500K" in item.approx_traffic else 85

        results.append(
            {
                "topic_title": clean_title,
                "category": cat,
                "virality_score": score,
                "viral_hook": f"Trending story with high US audience interest from {item.source}. Strong engagement potential for commentary and reactions.",
                "youtube_query": f"{clean_title} news discussion full interview",
                "suggested_preset": "viral",
            }
        )
        if len(results) >= limit:
            break

    return results


# ---------------------------------------------------------------------------
# Storage & Persistence
# ---------------------------------------------------------------------------


def load_trend_radar(filepath: str = TREND_RADAR_FILE) -> Dict[str, Any]:
    """Load persisted trend radar state."""
    with _STATE_LOCK:
        if not os.path.exists(filepath):
            return {"last_scanned": None, "topics": []}
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data if isinstance(data, dict) else {"last_scanned": None, "topics": []}
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Error reading %s: %s", filepath, exc)
            return {"last_scanned": None, "topics": []}


def save_trend_radar(data: Dict[str, Any], filepath: str = TREND_RADAR_FILE) -> bool:
    """Atomically save trend radar state with owner-only permissions."""
    with _STATE_LOCK:
        tmp_path = None
        try:
            dir_path = os.path.dirname(filepath) or "."
            os.makedirs(dir_path, mode=0o700, exist_ok=True)
            fd, tmp_path = tempfile.mkstemp(prefix=".trend_radar-", suffix=".tmp", dir=dir_path)
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            os.replace(tmp_path, filepath)
            return True
        except Exception as exc:
            logger.error("Failed to save trend radar: %s", exc)
            if tmp_path and os.path.exists(tmp_path):
                with contextlib.suppress(OSError):
                    os.remove(tmp_path)
            return False


def mark_trend_clipped(topic_id: str, job_id: str, filepath: str = TREND_RADAR_FILE) -> bool:
    """Mark a topic as clipped with its associated job_id."""
    with _STATE_LOCK:
        data = load_trend_radar(filepath)
        updated = False
        for t in data.get("topics", []):
            if t.get("id") == topic_id:
                t["clipped"] = True
                t["job_id"] = job_id
                t["clipped_at"] = datetime.now(timezone.utc).isoformat()
                updated = True
                break
        if updated:
            return save_trend_radar(data, filepath)
        return False


# ---------------------------------------------------------------------------
# YouTube Search (yt-dlp) & Execution
# ---------------------------------------------------------------------------


def search_youtube_source(
    query: str,
    min_duration: int = 300,
    max_duration: int = 7200,
    timeout: float = 12.0,
) -> Optional[Dict[str, Any]]:
    """Use yt-dlp flat-playlist search to find long-form YouTube source videos."""
    if not query:
        return None

    clean_query = query.replace('"', "").replace("'", "").strip()
    cmd = [
        "yt-dlp",
        "--dump-json",
        "--flat-playlist",
        "--playlist-end",
        "4",
        f"ytsearch4:{clean_query}",
    ]

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        if proc.returncode != 0 and not proc.stdout:
            logger.warning("yt-dlp search returned code %d for query %r: %s", proc.returncode, query, proc.stderr[:200])
            return None

        entries = []
        for line in proc.stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue

        return filter_and_rank_yt_candidates(entries, min_duration=min_duration, max_duration=max_duration)
    except subprocess.TimeoutExpired:
        logger.warning("yt-dlp search timed out for query: %s", query)
    except Exception as exc:
        logger.warning("yt-dlp search error for query %r: %s", query, exc)

    return None


# ---------------------------------------------------------------------------
# Core Discovery Pipeline
# ---------------------------------------------------------------------------


def run_trend_discovery(
    api_key: Optional[str] = None,
    model: str = "gemini-3.6-flash",
    categories: Optional[List[str]] = None,
    max_topics: int = 10,
    filepath: str = TREND_RADAR_FILE,
) -> Dict[str, Any]:
    """Execute complete trend discovery workflow and persist results.

    1. Ingest raw RSS trends (Google News US + Google Trends US).
    2. Analyze with Gemini (or heuristic fallback).
    3. Pair with long-form YouTube source videos via yt-dlp.
    4. Merge into persistent trend radar state.
    """
    logger.info("Starting Trend Discovery scan...")
    raw_items = collect_raw_us_trends(categories=categories or DEFAULT_CATEGORIES, items_per_category=8)
    logger.info("Collected %d raw trend items across US feeds", len(raw_items))

    analyzed_topics: List[Dict[str, Any]] = []

    if api_key:
        try:
            candidate_models = [model, "gemini-3.6-flash", "gemini-3.5-flash-lite", "gemini-3.5-flash"]
            # Remove duplicates preserving order
            seen = set()
            models_to_try = [m for m in candidate_models if not (m in seen or seen.add(m))]
            for m in models_to_try:
                try:
                    from google import genai

                    client = genai.Client(api_key=api_key)
                    prompt = build_gemini_analysis_prompt(raw_items, target_count=max_topics)
                    response = client.models.generate_content(model=m, contents=prompt)
                    resp_text = getattr(response, "text", "") or ""
                    analyzed_topics = parse_gemini_analysis_response(resp_text)
                    if analyzed_topics:
                        logger.info("Gemini trend analysis succeeded using model: %s", m)
                        break
                except Exception as exc:
                    logger.warning("Gemini trend analysis call failed with model %s: %s", m, exc)
            if not analyzed_topics:
                raise RuntimeError("All candidate models failed to return analyzed topics")
            logger.info("Gemini returned %d ranked topics", len(analyzed_topics))
        except Exception as exc:
            from clippyme.pipeline.gemini_service import _redact_key

            logger.warning("Gemini trend analysis call failed, using heuristic: %s", _redact_key(str(exc)))
            analyzed_topics = heuristic_trend_analysis(raw_items, limit=max_topics)
    else:
        logger.info("No Gemini API key provided, using heuristic trend analysis")
        analyzed_topics = heuristic_trend_analysis(raw_items, limit=max_topics)

    # Resolve YouTube sources
    existing_radar = load_trend_radar(filepath)
    existing_by_id = {t["id"]: t for t in existing_radar.get("topics", [])}

    discovered_list: List[Dict[str, Any]] = []
    now_iso = datetime.now(timezone.utc).isoformat()

    for item in analyzed_topics:
        topic_title = item.get("topic_title") or ""
        topic_id = make_topic_id(topic_title)

        # Check if already present and preserve its status
        prev = existing_by_id.get(topic_id)
        if prev and prev.get("matched_video"):
            # Update score / viral hook if newly scanned, retain video and clipped flag
            prev["virality_score"] = item.get("virality_score", prev.get("virality_score"))
            prev["viral_hook"] = item.get("viral_hook", prev.get("viral_hook"))
            discovered_list.append(prev)
            continue

        query = item.get("youtube_query") or f"{topic_title} news full coverage"
        video_match = search_youtube_source(query)

        radar_item = {
            "id": topic_id,
            "topic_title": topic_title,
            "category": item.get("category", "politics"),
            "virality_score": item.get("virality_score", 75),
            "viral_hook": item.get("viral_hook", ""),
            "youtube_query": query,
            "suggested_preset": item.get("suggested_preset", "viral"),
            "matched_video": video_match,
            "discovered_at": now_iso,
            "clipped": False,
            "job_id": None,
        }
        discovered_list.append(radar_item)

    # Sort topics by virality score descending
    discovered_list.sort(key=lambda x: x.get("virality_score", 0), reverse=True)

    result_state = {
        "last_scanned": now_iso,
        "topics": discovered_list,
    }
    save_trend_radar(result_state, filepath)
    logger.info("Trend Discovery complete: %d topics saved to %s", len(discovered_list), filepath)
    return result_state
