"""Unit tests for analytics_service and performance_feedback loop."""
import os
import pytest
from pathlib import Path
from clippyme.domain import analytics_service as ans
from clippyme.domain import performance_feedback as pf


@pytest.fixture(autouse=True)
def clean_analytics_file(tmp_path, monkeypatch):
    test_file = tmp_path / "test_analytics.json"
    monkeypatch.setattr(ans, "ANALYTICS_FILE_PATH", test_file)
    yield
    if test_file.exists():
        test_file.unlink()


def test_record_and_update_clip_metrics():
    clip_info = {
        "start": 10.0,
        "end": 85.0,
        "video_title_for_youtube_short": "Rudy Giuliani $2M Shakedown",
        "viral_hook_text": "The $2,000,000 Shakedown",
        "viral_reason": "Confrontation and revelation.",
        "speaker_name": "John Kiriakou",
        "duration_tier": "mid",
    }
    publish_result = {
        "post_id": "zernio_post_999",
        "scheduled_for": None,
    }

    # 1. Record publish
    rec = ans.record_published_clip(
        "job_abc", 0, clip_info, publish_result, platforms=["tiktok", "instagram"]
    )
    assert rec["clip_id"] == "job_abc:0"
    assert rec["duration"] == 75.0
    assert rec["duration_tier"] == "mid"
    assert rec["platforms"] == ["tiktok", "instagram"]

    # 2. Update metrics
    updated = ans.update_clip_metrics(
        "job_abc:0",
        {
            "views": 25000,
            "likes": 2100,
            "shares": 850,
            "comments": 340,
            "retention_rate": 0.82,
        },
    )
    assert updated is not None
    assert updated["metrics"]["views"] == 25000
    assert updated["metrics"]["shares"] == 850
    assert updated["metrics"]["retention_rate"] == 0.82
    assert updated["metrics"]["engagement_rate"] > 0.1

    # 3. Summary check
    summary = ans.get_analytics_summary()
    assert summary["total_published"] == 1
    assert summary["total_views"] == 25000
    assert summary["total_shares"] == 850
    assert summary["avg_retention"] == 0.82
    assert "tiktok" in summary["platform_breakdown"]


def test_performance_feedback_extracts_learned_patterns():
    # Record two clips: one top performer, one mediocre
    ans.record_published_clip(
        "job_1", 0,
        {"start": 0, "end": 85, "video_title_for_youtube_short": "Giuliani $2M Shakedown", "viral_hook_text": "The $2,000,000 Shakedown"},
        {"post_id": "p1"},
    )
    ans.update_clip_metrics("job_1:0", {"views": 100000, "likes": 9000, "shares": 3500, "comments": 1200, "retention_rate": 0.88})

    ans.record_published_clip(
        "job_2", 0,
        {"start": 0, "end": 25, "video_title_for_youtube_short": "Short chat", "viral_hook_text": "Did you know"},
        {"post_id": "p2"},
    )
    ans.update_clip_metrics("job_2:0", {"views": 5000, "likes": 100, "shares": 20, "comments": 15, "retention_rate": 0.35})

    patterns = pf.analyze_performance_patterns()
    assert patterns["has_data"] is True
    assert patterns["sample_size"] == 2
    assert patterns["story_duration_multiplier"] >= 1.0

    prompt_snippet = pf.get_learned_patterns_prompt()
    assert "LEARNED AUDIENCE PERFORMANCE PATTERNS" in prompt_snippet
    assert "WHOLE-STORY ARCS DOMINATE" in prompt_snippet
