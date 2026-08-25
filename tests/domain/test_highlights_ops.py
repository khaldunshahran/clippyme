import json
import pytest
from clippyme.domain.highlights_ops import (
    build_supercut_prompt,
    parse_supercut_response,
    build_multi_tier_highlights_prompt,
    parse_multi_tier_response,
    snap_supercut_timeline,
    fuse_supercut_subtitles,
    _parse_safe_score,
    _deduplicate_and_trim_overlaps,
    _enforce_tier_duration,
    _enforce_duration_bounds,
)


def test_build_supercut_prompt():
    words = [
        {"word": "Hello", "start": 0.0, "end": 0.5},
        {"word": "world", "start": 0.5, "end": 1.0},
        {"word": "this", "start": 1.0, "end": 1.5},
        {"word": "is", "start": 1.5, "end": 2.0},
        {"word": "a", "start": 2.0, "end": 2.2},
        {"word": "test.", "start": 2.2, "end": 2.8},
    ]
    prompt = build_supercut_prompt(words, target_duration_sec=60, video_duration_sec=600.0, video_title="Sample Video")
    assert "Sample Video" in prompt
    assert "~60 seconds" in prompt
    assert "50s to 70s" in prompt
    assert "[0.00s - 2.80s] Hello world this is a test." in prompt


def test_parse_supercut_response_valid():
    sample_json = """
    {
      "title": "Epic Tech Highlights",
      "viral_score": 94,
      "summary": "Full overview",
      "cuts": [
        {"role": "setup", "start": 2.0, "end": 10.0, "summary": "Intro problem"},
        {"role": "hook", "start": 50.0, "end": 55.0, "summary": "Shocking reveal"},
        {"role": "climax", "start": 120.0, "end": 130.0, "summary": "Final takeaway"}
      ]
    }
    """
    res = parse_supercut_response(sample_json, target_duration_sec=60)
    assert res["title"] == "Epic Tech Highlights"
    assert res["viral_score"] == 94
    assert len(res["cuts"]) == 3
    assert res["cuts"][0]["role"] == "setup"
    assert res["cuts"][0]["duration"] == 8.0
    assert res["cuts"][1]["duration"] == 5.0
    assert res["cuts"][2]["duration"] == 10.0
    assert res["total_duration"] == 23.0


def test_parse_supercut_response_nested_reels_fallback():
    sample_nested_json = """
    {
      "reels": [
        {
          "tier": "recap",
          "title": "Nested Reel Title",
          "viral_score": "95%",
          "summary": "Nested digest",
          "cuts": [
            {"role": "setup", "start": 0.0, "end": 15.0, "summary": "Setup"},
            {"role": "beat", "start": 40.0, "end": 60.0, "summary": "Beat"}
          ]
        }
      ]
    }
    """
    res = parse_supercut_response(sample_nested_json, target_duration_sec=60)
    assert res["title"] == "Nested Reel Title"
    assert res["viral_score"] == 95
    assert len(res["cuts"]) == 2
    assert res["total_duration"] == 35.0


def test_parse_supercut_response_markdown_fence():
    sample_json = """```json
    {
      "title": "Clean Title",
      "cuts": [
        {"role": "beat", "start": 10.0, "end": 20.0, "summary": "Fun part"}
      ]
    }
    ```"""
    res = parse_supercut_response(sample_json)
    assert res["title"] == "Clean Title"
    assert len(res["cuts"]) == 1


def test_parse_supercut_response_invalid():
    with pytest.raises(ValueError):
        parse_supercut_response("not json")

    with pytest.raises(ValueError):
        parse_supercut_response('{"cuts": []}')

    with pytest.raises(ValueError):
        parse_supercut_response('{"cuts": [{"start": 10.0, "end": 5.0}]}')


def test_parse_safe_score():
    assert _parse_safe_score(95) == 95
    assert _parse_safe_score("96") == 96
    assert _parse_safe_score("92.8%") == 92
    assert _parse_safe_score("high", default=80) == 80
    assert _parse_safe_score(None, default=85) == 85
    assert _parse_safe_score(999) == 100
    assert _parse_safe_score(-5) == 0


def test_deduplicate_and_trim_overlaps():
    raw_cuts = [
        {"role": "beat", "start": 30.0, "end": 60.0},
        {"role": "setup", "start": 10.0, "end": 40.0},  # Overlaps 30-40s with next
        {"role": "redundant", "start": 15.0, "end": 25.0},  # Engulfed inside 10-40
    ]
    cleaned = _deduplicate_and_trim_overlaps(raw_cuts)
    assert len(cleaned) == 2
    assert cleaned[0]["start"] == 10.0
    assert cleaned[0]["end"] == 40.0
    # Next cut start is pushed past prev_end
    assert cleaned[1]["start"] == 40.05
    assert cleaned[1]["end"] == 60.0


def test_enforce_tier_duration_anchor_protection():
    # Setup (15s) + Beat 1 (25s) + Beat 2 (30s) + Outro (15s) = 85s
    # micro max is 58s. Anchor total = 15 + 15 = 30s. Middle budget = 28s.
    # Middle beats should be trimmed/dropped while keeping both Setup and Outro!
    cuts = [
        {"role": "setup", "start": 0.0, "end": 15.0, "duration": 15.0},
        {"role": "beat", "start": 30.0, "end": 55.0, "duration": 25.0},
        {"role": "beat", "start": 60.0, "end": 90.0, "duration": 30.0},
        {"role": "outro", "start": 120.0, "end": 135.0, "duration": 15.0},
    ]
    bounded = _enforce_tier_duration(cuts, "micro")
    total = sum(c["duration"] for c in bounded)
    assert total <= 58.0
    # Setup (first cut) is preserved
    assert bounded[0]["role"] == "setup"
    assert bounded[0]["start"] == 0.0
    # Outro (last cut) is preserved at the very end
    assert bounded[-1]["role"] == "outro"
    assert bounded[-1]["start"] == 120.0
    assert bounded[-1]["end"] == 135.0


def test_enforce_duration_bounds_single_large_cut():
    # Single 90s cut should be hard trimmed down to 58s, not left as 90s!
    cuts = [
        {"role": "monologue", "start": 10.0, "end": 100.0, "duration": 90.0}
    ]
    bounded = _enforce_duration_bounds(cuts, min_d=25.0, max_d=58.0, tier_name="micro")
    assert len(bounded) == 1
    assert bounded[0]["duration"] == 58.0
    assert bounded[0]["start"] == 10.0
    assert bounded[0]["end"] == 68.0


def test_snap_supercut_timeline():
    words = [
        {"word": "First", "start": 10.0, "end": 10.5},
        {"word": "sentence", "start": 10.5, "end": 11.0},
        {"word": "here.", "start": 11.0, "end": 11.8},
        {"word": "Second", "start": 20.0, "end": 20.5},
        {"word": "sentence.", "start": 20.5, "end": 21.5},
    ]
    raw_cuts = [
        {"role": "hook", "start": 9.8, "end": 12.0, "summary": "H"},
        {"role": "beat", "start": 19.9, "end": 21.8, "summary": "B"},
    ]
    snapped = snap_supercut_timeline(raw_cuts, words, source_video_duration_sec=30.0)
    assert len(snapped) == 2
    assert snapped[0]["start"] == 9.95
    assert snapped[0]["end"] == 11.88
    assert snapped[1]["start"] == 19.95
    assert snapped[1]["end"] == 21.58


def test_fuse_supercut_subtitles():
    cuts = [
        {"start": 10.0, "end": 15.0},  # duration 5s
        {"start": 30.0, "end": 35.0},  # duration 5s
    ]
    words = [
        # In cut 1 (10-15s)
        {"word": "Clip1Word1", "start": 10.0, "end": 11.0},
        {"word": "Clip1Word2", "start": 12.0, "end": 13.0},
        # Outside cuts (20-25s)
        {"word": "IgnoredWord", "start": 20.0, "end": 21.0},
        # In cut 2 (30-35s)
        {"word": "Clip2Word1", "start": 30.5, "end": 31.5},
        {"word": "Clip2Word2", "start": 33.0, "end": 34.0},
    ]
    fused = fuse_supercut_subtitles(cuts, words)
    assert len(fused) == 4

    # Cut 1 words should start at 0.0s and 2.0s
    assert fused[0]["word"] == "Clip1Word1"
    assert fused[0]["start"] == 0.0
    assert fused[0]["end"] == 1.0

    assert fused[1]["word"] == "Clip1Word2"
    assert fused[1]["start"] == 2.0
    assert fused[1]["end"] == 3.0

    # Cut 2 words should be offset by Cut 1 duration (5.0s)
    assert fused[2]["word"] == "Clip2Word1"
    assert fused[2]["start"] == 5.5  # 5.0 + (30.5 - 30.0)
    assert fused[2]["end"] == 6.5

    assert fused[3]["word"] == "Clip2Word2"
    assert fused[3]["start"] == 8.0  # 5.0 + (33.0 - 30.0)
    assert fused[3]["end"] == 9.0


def test_build_multi_tier_highlights_prompt():
    words = [
        {"word": "Let's", "start": 0.0, "end": 0.5},
        {"word": "talk", "start": 0.5, "end": 1.0},
        {"word": "about", "start": 1.0, "end": 1.5},
        {"word": "AI", "start": 1.5, "end": 2.0},
        {"word": "supercuts.", "start": 2.0, "end": 2.8},
    ]
    prompt = build_multi_tier_highlights_prompt(words, video_duration_sec=600.0, video_title="AI Talk")
    assert "AI Talk" in prompt
    assert '"micro"' in prompt
    assert '"recap"' in prompt
    assert '"extended"' in prompt


def test_parse_multi_tier_response_with_resilience():
    # Includes malformed score ("96.5%"), out of order cuts, and one broken reel that should not kill the batch
    sample_json = """
    {
      "reels": [
        {
          "tier": "micro",
          "title": "Ultra-Short Teaser",
          "viral_score": "96.5%",
          "summary": "Fast 45s supercut",
          "cuts": [
            {"role": "beat", "start": 80.0, "end": 90.0, "summary": "Punchline"},
            {"role": "hook", "start": 40.0, "end": 45.0, "summary": "Wild hook"}
          ]
        },
        {
          "tier": "corrupt_reel",
          "cuts": "invalid cuts type"
        },
        {
          "tier": "recap",
          "title": "Story Recap",
          "viral_score": 92,
          "summary": "Full 2min digest",
          "cuts": [
            {"role": "setup", "start": 0.0, "end": 10.0, "summary": "Setup"},
            {"role": "beat", "start": 80.0, "end": 100.0, "summary": "Climax"}
          ]
        }
      ]
    }
    """
    reels = parse_multi_tier_response(sample_json)
    assert len(reels) == 2
    assert reels[0]["tier"] == "micro"
    assert reels[0]["viral_score"] == 96
    # Verified chronological sorting of cuts
    assert reels[0]["cuts"][0]["start"] == 40.0
    assert reels[0]["cuts"][1]["start"] == 80.0
    assert reels[0]["total_duration"] == 15.0

    assert reels[1]["tier"] == "recap"
    assert reels[1]["viral_score"] == 92
