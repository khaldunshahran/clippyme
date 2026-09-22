"""Tests for clippyme.domain.trend_discovery pure helpers and persistence."""
import os
import tempfile
import pytest

from clippyme.integrations.trend_sources import TrendSourceItem
from clippyme.domain.trend_discovery import (
    clean_json_markdown,
    format_duration,
    make_topic_id,
    build_gemini_analysis_prompt,
    parse_gemini_analysis_response,
    filter_and_rank_yt_candidates,
    heuristic_trend_analysis,
    load_trend_radar,
    save_trend_radar,
    mark_trend_clipped,
    run_trend_discovery,
)


def test_clean_json_markdown():
    assert clean_json_markdown('```json\n{"a": 1}\n```') == '{"a": 1}'
    assert clean_json_markdown('{"a": 1}') == '{"a": 1}'
    assert clean_json_markdown(None) == ""


def test_format_duration():
    assert format_duration(45) == "0:45"
    assert format_duration(920) == "15:20"
    assert format_duration(3665) == "1:01:05"
    assert format_duration(None) == "0:00"


def test_make_topic_id():
    id1 = make_topic_id("Nepal Earthquake Hits Region!")
    id2 = make_topic_id("nepal earthquake hits region")
    assert id1 == id2
    assert id1.startswith("tr_")
    assert len(id1) == 19


def test_build_gemini_analysis_prompt():
    items = [
        TrendSourceItem(title="Senate AI debate", source="WP", category="politics", approx_traffic="100K+", snippet="Senators clash"),
    ]
    prompt = build_gemini_analysis_prompt(items, target_count=5)
    assert "UNITED STATES" in prompt
    assert "Senate AI debate" in prompt
    assert "100K+" in prompt


def test_parse_gemini_analysis_response():
    valid_json = """
    [
      {
        "topic_title": "Senate Clash on AI Oversight",
        "category": "politics",
        "virality_score": 92,
        "viral_hook": "Heated exchanges trigger debate on free speech.",
        "youtube_query": "Senate judiciary hearing AI oversight 2026",
        "suggested_preset": "viral"
      }
    ]
    """
    parsed = parse_gemini_analysis_response(valid_json)
    assert len(parsed) == 1
    assert parsed[0]["topic_title"] == "Senate Clash on AI Oversight"
    assert parsed[0]["virality_score"] == 92
    assert parsed[0]["category"] == "politics"

    # Test with markdown fences and wrapper dict
    wrapped_json = '```json\n{"topics": [{"topic_title": "Disaster Response", "virality_score": 88}]}\n```'
    parsed2 = parse_gemini_analysis_response(wrapped_json)
    assert len(parsed2) == 1
    assert parsed2[0]["topic_title"] == "Disaster Response"
    assert parsed2[0]["virality_score"] == 88


def test_filter_and_rank_yt_candidates():
    entries = [
        {"id": "short1", "title": "Quick snippet", "duration": 45},
        {"id": "long1", "title": "Full Committee Hearing", "duration": 3600, "channel": "C-SPAN", "view_count": 150000},
        {"id": "long2", "title": "Press Conference", "duration": 1800, "channel": "News Network", "view_count": 50000},
    ]
    best = filter_and_rank_yt_candidates(entries, min_duration=300)
    assert best is not None
    assert best["video_id"] == "long1"
    assert best["title"] == "Full Committee Hearing"
    assert best["channel"] == "C-SPAN"
    assert best["duration"] == 3600
    assert best["duration_string"] == "1:00:00"
    assert best["video_url"] == "https://www.youtube.com/watch?v=long1"


def test_heuristic_trend_analysis():
    raw = [
        TrendSourceItem(title="Major Earthquake in Nepal", source="Google Trends", category="trending_spike", approx_traffic="500K+"),
        TrendSourceItem(title="Budget Talks Stall", source="Reuters", category="politics"),
    ]
    analyzed = heuristic_trend_analysis(raw, limit=5)
    assert len(analyzed) == 2
    assert analyzed[0]["category"] == "breaking_world"
    assert analyzed[0]["virality_score"] >= 85
    assert "Earthquake in Nepal" in analyzed[0]["topic_title"]


def test_radar_persistence_and_clip_marking():
    with tempfile.TemporaryDirectory() as td:
        file_path = os.path.join(td, "radar.json")
        data = {
            "last_scanned": "2026-09-16T00:00:00Z",
            "topics": [
                {
                    "id": "tr_12345",
                    "topic_title": "Test Topic",
                    "virality_score": 85,
                    "clipped": False,
                    "job_id": None,
                }
            ],
        }
        assert save_trend_radar(data, file_path) is True
        loaded = load_trend_radar(file_path)
        assert len(loaded["topics"]) == 1
        assert loaded["topics"][0]["topic_title"] == "Test Topic"

        # Mark clipped
        marked = mark_trend_clipped("tr_12345", "job_abc789", file_path)
        assert marked is True
        reloaded = load_trend_radar(file_path)
        assert reloaded["topics"][0]["clipped"] is True
        assert reloaded["topics"][0]["job_id"] == "job_abc789"


def test_run_trend_discovery_mocked(monkeypatch):
    with tempfile.TemporaryDirectory() as td:
        file_path = os.path.join(td, "radar.json")

        def fake_raw(categories=None, items_per_category=8):
            return [
                TrendSourceItem(title="Disaster in Nepal", source="News", category="world"),
                TrendSourceItem(title="Senate AI Hearing", source="News", category="politics"),
            ]

        def fake_yt(query, min_duration=300, max_duration=7200, timeout=12.0):
            return {
                "video_id": "vid_xyz",
                "video_url": "https://www.youtube.com/watch?v=vid_xyz",
                "title": f"Footage: {query}",
                "channel": "Live Coverage",
                "duration": 600,
                "duration_string": "10:00",
                "thumbnail_url": "https://i.ytimg.com/vi/vid_xyz/hqdefault.jpg",
                "view_count": 10000,
            }

        monkeypatch.setattr("clippyme.domain.trend_discovery.collect_raw_us_trends", fake_raw)
        monkeypatch.setattr("clippyme.domain.trend_discovery.search_youtube_source", fake_yt)

        res = run_trend_discovery(api_key=None, max_topics=2, filepath=file_path)
        assert len(res["topics"]) == 2
        assert res["topics"][0]["matched_video"] is not None
        assert res["topics"][0]["matched_video"]["video_id"] == "vid_xyz"

        # Verify saved to file
        loaded = load_trend_radar(file_path)
        assert len(loaded["topics"]) == 2
