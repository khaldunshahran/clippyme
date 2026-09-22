"""Tests for audience_intel module: timestamp parsing, comment filtering, prompt formatting."""
import json
import os
import tempfile
from clippyme.pipeline.audience_intel import (
    extract_timestamps_from_text,
    parse_audience_intel,
    format_audience_intel_prompt,
    save_audience_intel,
    load_audience_intel,
)


def test_extract_timestamps_from_text():
    text = "The breakdown at 14:20 is insane! Also 1:02:15 had me dying, and check 0:45."
    extracted = extract_timestamps_from_text(text)
    assert len(extracted) == 3

    assert extracted[0]["timestamp"] == "14:20"
    assert extracted[0]["seconds"] == 860.0

    assert extracted[1]["timestamp"] == "1:02:15"
    assert extracted[1]["seconds"] == 3735.0

    assert extracted[2]["timestamp"] == "0:45"
    assert extracted[2]["seconds"] == 45.0


def test_parse_audience_intel_with_mock_info():
    mock_info = {
        "title": "How to Build an AI Startup",
        "description": "Full guide.\n02:15 Why most founders fail\n08:30 The marketing hack",
        "channel": "Tech Lead",
        "uploader_id": "techlead",
        "view_count": 150000,
        "like_count": 12000,
        "comment_count": 450,
        "tags": ["ai", "startups", "tech"],
        "comments": [
            {"text": "14:20 is literally the best advice on the internet", "like_count": 3400, "author": "Alice"},
            {"text": "I disagree completely with his point at 08:30", "like_count": 1800, "author": "Bob"},
            {"text": "Great video man keep it up", "like_count": 50, "author": "Charlie"},
            {"text": "https://spamlink.com buy crypto", "like_count": 1, "author": "Spammer"},
        ],
    }

    intel = parse_audience_intel(mock_info)
    assert intel["title"] == "How to Build an AI Startup"
    assert intel["channel"] == "Tech Lead"
    assert len(intel["top_comments"]) == 3  # Spammer filtered
    assert intel["top_comments"][0]["like_count"] == 3400

    # Timestamp extraction should have captured 02:15, 08:30 from description and 14:20, 08:30 from comments
    ts_seconds = [t["seconds"] for t in intel["audience_timestamps"]]
    assert 860.0 in ts_seconds  # 14:20
    assert 510.0 in ts_seconds  # 08:30
    assert 135.0 in ts_seconds  # 02:15


def test_format_audience_intel_prompt():
    intel = {
        "title": "The Secrets of High Retention",
        "channel": "Viral Masters",
        "description": "Learn the retention strategies used by top creators.",
        "top_comments": [
            {"text": "The controversy at 05:10 had everyone talking", "like_count": 2500, "author": "Viewer1"}
        ],
        "audience_timestamps": [
            {"timestamp": "05:10", "seconds": 310.0, "sample_comment": "The controversy at 05:10"}
        ],
    }
    prompt_block = format_audience_intel_prompt(intel)
    assert "AUDIENCE INTELLIGENCE & SOCIAL SIGNALS" in prompt_block
    assert "The Secrets of High Retention" in prompt_block
    assert "2500 likes" in prompt_block
    assert "05:10 (310s)" in prompt_block


def test_save_and_load_audience_intel():
    with tempfile.TemporaryDirectory() as tmpdir:
        data = {"title": "Test Title", "top_comments": [{"text": "Nice", "like_count": 10}]}
        saved_file = save_audience_intel(tmpdir, data)
        assert os.path.isfile(saved_file)

        loaded = load_audience_intel(tmpdir)
        assert loaded == data
