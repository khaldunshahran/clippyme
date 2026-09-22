"""Unit tests for narrative completeness and story arc consolidation."""
import pytest
from clippyme.pipeline.gemini_parser import consolidate_narrative_continuations, validate_and_dedupe
from clippyme.pipeline.gemini_request import build_viral_prompt


def test_giuliani_story_fragments_are_consolidated():
    """Verify that split story fragments (like Rudy Giuliani $2M shakedown + photo proof)
    are automatically merged into a single complete narrative arc.
    """
    clip1 = {
        "start": 2019.65,
        "end": 2049.77,
        "viral_score": 96,
        "viral_reason": "Opens with a shocking story of Rudy Giuliani trying to shake down Kiriakou for $2 million.",
        "video_title_for_youtube_short": "Rudy Giuliani asked him for 2 MILLION dollars and he laughed in his face",
        "viral_hook_text": "The $2,000,000 Shakedown",
        "speaker_name": "John Kiriakou",
    }
    clip2 = {
        "start": 2063.55,
        "end": 2106.48,
        "viral_score": 95,
        "viral_reason": "Giuliani responded by saying he never met him, then speaker sent photo to New York Times.",
        "video_title_for_youtube_short": "Rudy Giuliani lied about meeting him, then he sent THIS photo to the New York Times",
        "viral_hook_text": "He Forgot About the Photo",
        "speaker_name": "John Kiriakou",
    }

    result = consolidate_narrative_continuations([clip1, clip2])
    assert len(result) == 1, "The two story fragments should be fused into 1 complete narrative clip"
    merged = result[0]
    assert merged["start"] == 2019.65
    assert merged["end"] == 2106.48
    assert merged["duration_tier"] == "mid"
    assert "Giuliani" in merged["video_title_for_youtube_short"]
    assert "Continues with" in merged["viral_reason"]


def test_unrelated_story_fragments_not_consolidated():
    """Clips with different entities and no continuation cues should remain separate."""
    clip1 = {
        "start": 100.0,
        "end": 140.0,
        "viral_score": 85,
        "viral_reason": "Deep explanation of quantum computing algorithms and matrix multiplication.",
        "video_title_for_youtube_short": "How Quantum Computing Actually Works",
        "viral_hook_text": "The quantum secret",
    }
    clip2 = {
        "start": 150.0,
        "end": 190.0,
        "viral_score": 88,
        "viral_reason": "Funny blooper where the microphone fell off the desk during interview.",
        "video_title_for_youtube_short": "Live Stream Disaster Caught on Camera",
        "viral_hook_text": "He knocked the mic off",
    }

    result = consolidate_narrative_continuations([clip1, clip2])
    assert len(result) == 2, "Unrelated clips must NOT be merged"


def test_build_viral_prompt_narrative_completeness_rules():
    """Prompt must contain the mandatory whole-story narrative integrity rules."""
    transcript = {
        "text": "Sample video transcript",
        "segments": [{"words": [{"word": "Sample", "start": 0.0, "end": 1.0}]}]
    }
    prompt, _ = build_viral_prompt(transcript, video_duration=300.0)
    assert "NARRATIVE COMPLETENESS & WHOLE-STORY INTEGRITY" in prompt
    assert "NEVER SPLIT A CONTINUOUS ANECDOTE" in prompt
    assert "DURATION SERVES THE STORY" in prompt
    assert "RESOLUTION & PUNCHLINE" in prompt


def test_three_part_story_chain_is_consolidated():
    """Verify that a 3-part continuous narrative story is completely fused into one clip."""
    part1 = {
        "start": 10.0,
        "end": 35.0,
        "viral_score": 90,
        "viral_reason": "Setup of the confrontation with the CEO.",
        "video_title_for_youtube_short": "Confronting the CEO about hidden fees",
        "viral_hook_text": "The hidden fee setup",
        "speaker_name": "Investigator",
    }
    part2 = {
        "start": 40.0,
        "end": 75.0,
        "viral_score": 92,
        "viral_reason": "CEO responded by denying all allegations and getting angry.",
        "video_title_for_youtube_short": "CEO lied about the audit report",
        "viral_hook_text": "He denied everything",
        "speaker_name": "Investigator",
    }
    part3 = {
        "start": 80.0,
        "end": 115.0,
        "viral_score": 95,
        "viral_reason": "Turns out we had the audio recording all along and played it.",
        "video_title_for_youtube_short": "The tape that caught him on the spot",
        "viral_hook_text": "We played the tape",
        "speaker_name": "Investigator",
    }

    result = consolidate_narrative_continuations([part1, part2, part3])
    assert len(result) == 1, "All 3 story fragments should chain-merge into a single unified clip"
    merged = result[0]
    assert merged["start"] == 10.0
    assert merged["end"] == 115.0
    assert merged["viral_score"] == 95
    assert merged["duration_tier"] == "mid"
