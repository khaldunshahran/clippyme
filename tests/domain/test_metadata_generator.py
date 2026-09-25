"""Tests for AI platform metadata generator and video brain intelligence."""
import json
import os
import pytest
from unittest.mock import MagicMock, patch

from clippyme.domain.metadata_generator import (
    _clean_tags,
    _normalize_platform_dict,
    generate_clip_metadata,
    generate_all_clips_metadata,
)
from clippyme.domain.errors import ValidationError, NotFoundError


def test_clean_tags():
    assert _clean_tags(["shorts", "#trending", "  viral  "]) == ["#shorts", "#trending", "#viral"]
    assert _clean_tags([], default_tag="#fyp") == ["#fyp", "#trending", "#viral"]
    assert _clean_tags(None) == ["#shorts", "#trending", "#viral"]


def test_normalize_platform_dict():
    raw_response = {
        "speaker_name": "John Kiriakou",
        "title": "Secret Warning Revealed",
        "hashtags": ["#intel", "#cia"],
        "platforms": {
            "tiktok": {
                "caption": "Did you know this happened? #fyp",
                "hashtags": ["#fyp", "#intel"],
            },
            "instagram": {
                "caption": "A deep dive into foreign intelligence. Drop thoughts below 👇",
                "hashtags": ["#reels", "#intel"],
            },
            "youtube": {
                "title": "Explosive Intel Report",
                "description": "Full breakdown of warnings before the event. #shorts",
                "hashtags": ["#shorts", "#intel"],
            },
        },
    }

    norm = _normalize_platform_dict(raw_response, fallback_title="Fallback Title")
    assert norm["speaker_name"] == "John Kiriakou"
    assert norm["title"] == "Secret Warning Revealed"
    assert "tiktok" in norm["platforms"]
    assert "instagram" in norm["platforms"]
    assert "youtube" in norm["platforms"]
    assert "Did you know this happened?" in norm["platforms"]["tiktok"]["caption"]
    assert "#reels" in norm["platforms"]["instagram"]["caption"]
    assert norm["platforms"]["youtube"]["title"] == "Explosive Intel Report"


def test_normalize_platform_dict_fallbacks():
    norm = _normalize_platform_dict({}, fallback_title="Fallback Title", fallback_speaker="Guest")
    assert norm["title"] == "Fallback Title"
    assert norm["speaker_name"] == "Guest"
    assert "platforms" in norm
    assert "tiktok" in norm["platforms"]
    assert "instagram" in norm["platforms"]
    assert "youtube" in norm["platforms"]
    assert len(norm["hashtags"]) > 0


def test_generate_clip_metadata_missing_key():
    with pytest.raises(ValidationError):
        generate_clip_metadata(
            api_key="",
            clip_transcript="Test dialogue",
            start=0.0,
            end=10.0,
        )


def test_generate_all_clips_metadata_missing_job(tmp_path):
    with pytest.raises(NotFoundError):
        generate_all_clips_metadata("non-existent-job-id", str(tmp_path))


# --- Phase 3D: title_variants normalization (appended) -------------------------

def test_normalize_platform_dict_stores_three_title_variants():
    from clippyme.domain.metadata_generator import _normalize_platform_dict
    payload = {
        "title": "Main title",
        "alternate_titles": ["Alt one", "Alt two", "Alt one"],  # dup + 3
    }
    out = _normalize_platform_dict(payload, "Main title", {}, 0, 30)
    assert out["title"] == "Main title"
    assert out["title_variants"] == ["Main title", "Alt one", "Alt two"]
    assert out["title_variant_source"] == "llm"


def test_normalize_platform_dict_variants_fallback():
    from clippyme.domain.metadata_generator import _normalize_platform_dict
    out = _normalize_platform_dict({"title": "Solo"}, "Solo", {}, 0, 30)
    assert out["title_variants"] == ["Solo"]
    assert out["title_variant_source"] == "fallback"


def test_normalize_platform_dict_variant_a_is_main_title():
    from clippyme.domain.metadata_generator import _normalize_platform_dict
    out = _normalize_platform_dict(
        {"title": "Other", "alternate_titles": ["X"]}, "Main title", {}, 0, 30)
    assert out["title_variants"][0] == "Main title"
