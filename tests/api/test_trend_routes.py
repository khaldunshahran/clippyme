"""Tests for clippyme.api.trend_routes."""
import json
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from clippyme.api.app import app

client = TestClient(app)

MOCK_RADAR_DATA = {
    "last_scanned": "2026-09-16T01:00:00Z",
    "topics": [
        {
            "id": "tr_1",
            "topic_title": "Senate Clash on AI Oversight",
            "category": "politics",
            "virality_score": 95,
            "viral_hook": "Heated exchanges trigger widespread commentary.",
            "youtube_query": "Senate judiciary hearing AI 2026",
            "matched_video": {
                "video_id": "vid1",
                "video_url": "https://www.youtube.com/watch?v=vid1",
                "title": "Full Committee Hearing on AI",
                "channel": "C-SPAN",
                "duration": 3600,
                "duration_string": "1:00:00",
                "thumbnail_url": "https://i.ytimg.com/vi/vid1/hqdefault.jpg",
                "view_count": 80000,
            },
            "clipped": False,
            "job_id": None,
        },
        {
            "id": "tr_2",
            "topic_title": "Nepal Earthquake Relief Operations",
            "category": "breaking_world",
            "virality_score": 90,
            "viral_hook": "Dramatic rescue footage drawing global attention.",
            "youtube_query": "Nepal earthquake update live briefing",
            "matched_video": {
                "video_id": "vid2",
                "video_url": "https://www.youtube.com/watch?v=vid2",
                "title": "Nepal Disaster Briefing",
                "channel": "Global News",
                "duration": 900,
                "duration_string": "15:00",
                "thumbnail_url": "https://i.ytimg.com/vi/vid2/hqdefault.jpg",
                "view_count": 120000,
            },
            "clipped": True,
            "job_id": "job_clipped_123",
        },
    ],
}


def test_get_trends():
    with patch("clippyme.api.trend_routes.load_trend_radar", return_value=MOCK_RADAR_DATA):
        res = client.get("/api/trends")
        assert res.status_code == 200
        data = res.json()
        assert data["total"] == 2
        assert len(data["topics"]) == 2
        assert data["topics"][0]["topic_title"] == "Senate Clash on AI Oversight"


def test_get_trends_category_filter():
    with patch("clippyme.api.trend_routes.load_trend_radar", return_value=MOCK_RADAR_DATA):
        res = client.get("/api/trends?category=politics")
        assert res.status_code == 200
        data = res.json()
        assert data["total"] == 1
        assert data["topics"][0]["category"] == "politics"

        res_world = client.get("/api/trends?category=breaking_world")
        assert res_world.status_code == 200
        assert res_world.json()["total"] == 1


def test_get_trends_exclude_clipped():
    with patch("clippyme.api.trend_routes.load_trend_radar", return_value=MOCK_RADAR_DATA):
        res = client.get("/api/trends?include_clipped=false")
        assert res.status_code == 200
        data = res.json()
        assert data["total"] == 1
        assert data["topics"][0]["id"] == "tr_1"


def test_trend_scan_trigger():
    fake_result = {
        "last_scanned": "2026-09-16T02:00:00Z",
        "topics": [MOCK_RADAR_DATA["topics"][0]],
    }
    with patch("clippyme.api.trend_routes.run_trend_discovery", return_value=fake_result):
        res = client.post(
            "/api/trends/scan",
            headers={"Origin": "http://localhost:5175", "X-Gemini-Key": "fake-key"},
        )
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "ok"
        assert data["topics_count"] == 1


def test_trend_clip_submission():
    payload = {
        "video_url": "https://www.youtube.com/watch?v=vid1",
        "topic_id": "tr_1",
        "instructions": "Focus on heated moments",
        "preset_id": "viral",
    }
    with patch("clippyme.api.trend_routes.mark_trend_clipped") as mock_mark:
        res = client.post(
            "/api/trends/clip",
            json=payload,
            headers={"Origin": "http://localhost:5175", "X-Gemini-Key": "fake-key"},
        )
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "queued"
        assert "job_id" in data
        assert data["topic_id"] == "tr_1"
        mock_mark.assert_called_once()


def test_trend_config_get_and_update():
    res = client.get("/api/trends/config")
    assert res.status_code == 200
    cfg = res.json()
    assert "interval_hours" in cfg

    update_res = client.post(
        "/api/trends/config",
        json={"interval_hours": 4, "categories": ["politics", "entertainment"]},
        headers={"Origin": "http://localhost:5175"},
    )
    assert update_res.status_code == 200
    updated = update_res.json()["config"]
    assert updated["interval_hours"] == 4
    assert "politics" in updated["categories"]
