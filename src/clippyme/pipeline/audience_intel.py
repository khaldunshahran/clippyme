"""Audience & Community Intelligence for viral clip selection.

Extracts and structures social signals from yt-dlp metadata (top comments,
comment likes, viewer-voted timestamps, descriptions, channel bio).
Pure logic: host-testable, no network or heavy CV dependencies.
"""
import json
import os
import re
from typing import Any, Dict, List, Optional


AUDIENCE_INTEL_FILENAME = "audience_intel.json"

# Matches timestamps like 1:23, 01:23, 1:02:34, 12:45
_TIMESTAMP_REGEX = re.compile(r"\b(?:(\d{1,2}):)?([0-5]?\d):([0-5]\d)\b")


def parse_timestamp_to_seconds(ts_match: re.Match) -> float:
    """Convert regex match (hours, minutes, seconds) to total seconds."""
    hours_str, mins_str, secs_str = ts_match.groups()
    hours = int(hours_str) if hours_str else 0
    mins = int(mins_str) if mins_str else 0
    secs = int(secs_str) if secs_str else 0
    return float(hours * 3600 + mins * 60 + secs)


def extract_timestamps_from_text(text: str) -> List[Dict[str, Any]]:
    """Extract all timestamps and surrounding text snippets from a comment or description."""
    if not text:
        return []
    results = []
    for match in _TIMESTAMP_REGEX.finditer(text):
        seconds = parse_timestamp_to_seconds(match)
        # Capture surrounding context (up to 60 chars around timestamp)
        start_idx = max(0, match.start() - 30)
        end_idx = min(len(text), match.end() + 50)
        snippet = text[start_idx:end_idx].replace("\n", " ").strip()
        results.append({
            "timestamp": match.group(0),
            "seconds": seconds,
            "snippet": snippet,
        })
    return results


def parse_audience_intel(info: dict) -> Dict[str, Any]:
    """Extract structured audience intelligence from yt-dlp info dictionary."""
    if not isinstance(info, dict):
        return {}

    title = (info.get("title") or "").strip()
    description = (info.get("description") or "").strip()
    channel = (info.get("channel") or info.get("uploader") or "").strip()
    channel_id = (info.get("channel_id") or info.get("uploader_id") or "").strip()
    tags = [t for t in info.get("tags") or [] if isinstance(t, str) and len(t) < 50][:15]
    view_count = info.get("view_count")
    like_count = info.get("like_count")
    comment_count = info.get("comment_count")

    # Raw comments from yt-dlp
    raw_comments = info.get("comments") or []
    cleaned_comments = []
    timestamp_map: Dict[int, Dict[str, Any]] = {}

    for c in raw_comments:
        if not isinstance(c, dict):
            continue
        text = (c.get("text") or "").strip()
        if not text or len(text) < 4:
            continue
        # Skip obvious spam or URL-only comments
        if text.startswith("http://") or text.startswith("https://") or "bit.ly" in text:
            continue

        c_likes = int(c.get("like_count") or 0)
        author = (c.get("author") or "").strip()
        cleaned_comments.append({
            "text": text[:300],  # Cap single comment length
            "like_count": c_likes,
            "author": author[:50],
        })

        # Extract timestamps in comment
        for ts_entry in extract_timestamps_from_text(text):
            sec_bucket = int(round(ts_entry["seconds"]))
            if sec_bucket not in timestamp_map:
                timestamp_map[sec_bucket] = {
                    "timestamp": ts_entry["timestamp"],
                    "seconds": float(sec_bucket),
                    "count": 0,
                    "max_likes": c_likes,
                    "sample_comment": text[:150],
                }
            timestamp_map[sec_bucket]["count"] += 1
            if c_likes > timestamp_map[sec_bucket]["max_likes"]:
                timestamp_map[sec_bucket]["max_likes"] = c_likes
                timestamp_map[sec_bucket]["sample_comment"] = text[:150]

    # Also check description for timestamps (creator chapter marks or highlights)
    for ts_entry in extract_timestamps_from_text(description):
        sec_bucket = int(round(ts_entry["seconds"]))
        if sec_bucket not in timestamp_map:
            timestamp_map[sec_bucket] = {
                "timestamp": ts_entry["timestamp"],
                "seconds": float(sec_bucket),
                "count": 1,
                "max_likes": 999999,  # High priority because creator marked it
                "sample_comment": f"Chapter/Highlight: {ts_entry['snippet'][:150]}",
            }

    # Sort comments by likes descending, pick top 25
    cleaned_comments.sort(key=lambda x: x["like_count"], reverse=True)
    top_comments = cleaned_comments[:25]

    # Sort audience timestamps by max_likes and frequency
    audience_timestamps = sorted(
        timestamp_map.values(),
        key=lambda x: (x["max_likes"], x["count"]),
        reverse=True,
    )[:15]
    # Re-sort chronologically for easier prompt scanning
    audience_timestamps.sort(key=lambda x: x["seconds"])

    return {
        "title": title,
        "description": description[:1000],  # Concise summary
        "channel": channel,
        "channel_id": channel_id,
        "tags": tags,
        "view_count": view_count,
        "like_count": like_count,
        "comment_count": comment_count,
        "top_comments": top_comments,
        "audience_timestamps": audience_timestamps,
    }


def format_audience_intel_prompt(intel: Optional[dict]) -> str:
    """Render audience intelligence block for Gemini prompt injection."""
    if not intel or not isinstance(intel, dict):
        return ""

    title = intel.get("title")
    channel = intel.get("channel")
    description = intel.get("description")
    top_comments = intel.get("top_comments") or []
    timestamps = intel.get("audience_timestamps") or []

    if not title and not top_comments and not timestamps:
        return ""

    lines = ["\n## AUDIENCE INTELLIGENCE & SOCIAL SIGNALS (PROVEN ENGAGEMENT)"]
    lines.append(
        "The following metadata and community reactions show what real human viewers "
        "already resonated with, debated, or timestamped in this source video."
    )

    if title:
        lines.append(f"- Source Video Title: \"{title}\"")
    if channel:
        lines.append(f"- Channel / Creator: {channel}")
    if description:
        # First 300 chars of description
        clean_desc = description.replace("\n", " ")[:300].strip()
        lines.append(f"- Creator Description Snippet: \"{clean_desc}\"")

    if top_comments:
        lines.append("\nTop-Liked Audience Comments & Debates (ranked by real engagement):")
        for i, c in enumerate(top_comments[:10], 1):
            likes_str = f"[{c['like_count']} likes] " if c.get("like_count") else ""
            clean_text = c['text'].replace('\n', ' ').strip()
            lines.append(f"  {i}. {likes_str}\"{clean_text}\"")

    if timestamps:
        lines.append("\nAudience & Chapter Highlight Timestamps:")
        for ts in timestamps[:10]:
            sec = int(ts['seconds'])
            snippet = ts.get('sample_comment', '').replace('\n', ' ').strip()
            lines.append(f"  * {ts['timestamp']} ({sec}s): {snippet}")

    lines.append(
        "\nCRITICAL VIRAL RULE: Pay SPECIAL attention to moments corresponding to these "
        "audience timestamps, quotes, or polarizing debates. They have already proven to "
        "stop viewers from scrolling and drive high comment counts."
    )
    return "\n".join(lines)


def save_audience_intel(output_dir: str, intel: dict) -> str:
    """Atomically save audience_intel.json to output directory."""
    os.makedirs(output_dir, exist_ok=True)
    target_path = os.path.join(output_dir, AUDIENCE_INTEL_FILENAME)
    tmp_path = target_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(intel, f, indent=2, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, target_path)
    return target_path


def load_audience_intel(output_dir: str) -> Optional[dict]:
    """Load audience_intel.json from output directory if present."""
    target_path = os.path.join(output_dir, AUDIENCE_INTEL_FILENAME)
    if not os.path.isfile(target_path):
        return None
    try:
        with open(target_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None
