"""Tests for clippyme.api.channel_routes."""
import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient

from clippyme.api.app import app

client = TestClient(app)

MOCK_CHANNELS = [
    {
        "id": "ch_us_politics",
        "name": "US Politics Daily",
        "description": "Politics channel",
        "niches": ["politics", "nation"],
        "default_preset": "viral",
        "banner_handle": "@USPoliticsDaily",
        "banner_platform": "youtube",
        "banner_y_pct": 0.85,
        "reframe_mode": "auto",
        "sub_preset": "hormozi_bold",
    },
    {
        "id": "ch_breaking_world",
        "name": "Global Breaking",
        "description": "World news channel",
        "niches": ["breaking_world", "world"],
        "default_preset": "viral",
        "banner_handle": "@GlobalBreaking",
        "banner_platform": "tiktok",
        "banner_y_pct": 0.85,
        "reframe_mode": "auto",
        "sub_preset": "classic",
    },
]


def test_list_channels():
    with patch("clippyme.api.channel_routes.load_channels", return_value=MOCK_CHANNELS):
        res = client.get("/api/channels")
        assert res.status_code == 200
        data = res.json()
        assert data["total"] == 2
        assert data["channels"][0]["id"] == "ch_us_politics"


def test_match_channel():
    with patch("clippyme.api.channel_routes.match_channel_for_category", return_value=MOCK_CHANNELS[0]):
        res = client.get("/api/channels/match?category=politics")
        assert res.status_code == 200
        assert res.json()["matched"]["id"] == "ch_us_politics"


def test_create_update_delete_channel():
    new_ch = {
        "id": "ch_tech",
        "name": "Tech Channel",
        "niches": ["tech"],
        "banner_handle": "@Tech",
    }
    with patch("clippyme.api.channel_routes.create_channel", return_value=new_ch):
        res = client.post(
            "/api/channels",
            json={"name": "Tech Channel", "niches": ["tech"], "banner_handle": "@Tech"},
            headers={"Origin": "http://localhost:5175"},
        )
        assert res.status_code == 200
        assert res.json()["channel"]["name"] == "Tech Channel"

    with patch("clippyme.api.channel_routes.update_channel", return_value={**new_ch, "name": "Updated Tech"}):
        res_put = client.put(
            "/api/channels/ch_tech",
            json={"name": "Updated Tech"},
            headers={"Origin": "http://localhost:5175"},
        )
        assert res_put.status_code == 200
        assert res_put.json()["channel"]["name"] == "Updated Tech"

    with patch("clippyme.api.channel_routes.delete_channel", return_value=True):
        res_del = client.delete(
            "/api/channels/ch_tech",
            headers={"Origin": "http://localhost:5175"},
        )
        assert res_del.status_code == 200
        assert res_del.json()["status"] == "deleted"


def test_channel_routes_auth_enforced_when_enabled(monkeypatch):
    monkeypatch.setenv("AUTH_ENABLED", "1")
    monkeypatch.setenv("SUPABASE_JWT_SECRET", "test-secret")

    # Unauthenticated request receives 401
    res = client.get("/api/channels")
    assert res.status_code == 401

    res_get = client.get("/api/channels/ch_us_politics")
    assert res_get.status_code == 401
