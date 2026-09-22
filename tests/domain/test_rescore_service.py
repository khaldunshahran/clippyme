"""Unit tests for rescore_service with whole-story narrative and performance feedback."""
import json
import os
import pytest
from unittest.mock import MagicMock, patch

from clippyme.domain import rescore_service as rs


def test_rescore_job_incorporates_narrative_rules_and_updates_clips(tmp_path):
    job_id = "test_rescore_narrative"
    job_dir = tmp_path / job_id
    job_dir.mkdir(parents=True)

    metadata_file = job_dir / f"{job_id}_metadata.json"
    initial_metadata = {
        "title": "Interview with John",
        "shorts": [
            {
                "start": 10.0,
                "end": 85.0,
                "viral_score": 0,
                "viral_reason": "",
                "video_title_for_youtube_short": "",
                "hook": "",
            }
        ],
        "transcript": {
            "segments": [
                {
                    "start": 10.0,
                    "end": 85.0,
                    "text": "Rudy Giuliani asked him for 2 million dollars and he laughed in his face.",
                }
            ]
        },
    }
    metadata_file.write_text(json.dumps(initial_metadata), encoding="utf-8")

    captured_prompt = {}

    def fake_fallback(client, prompt, model_chain, max_attempts=2, log_fn=None):
        captured_prompt["prompt"] = prompt
        resp = MagicMock()
        resp.text = json.dumps({
            "clips": [
                {
                    "clip_index": 1,
                    "viral_score": 94,
                    "viral_reason": "Complete beginning-to-end narrative with confrontation setup and payoff.",
                    "video_title_for_youtube_short": "The $2M Giuliani Shakedown",
                    "hook": "He wanted two million dollars",
                    "duration_tier": "mid",
                    "target_platforms": ["tiktok", "youtube_shorts"],
                }
            ]
        })
        return resp, "gemini-3.5-flash-lite"

    with patch("clippyme.domain.rescore_service.generate_with_model_fallback", side_effect=fake_fallback), \
         patch("google.genai.Client"):
        result = rs.rescore_job(
            job_id=job_id,
            output_dir=str(tmp_path),
            api_key="fake_key",
        )

    # 1. Verify prompt contains whole story narrative rules and performance feedback
    assert "NARRATIVE COMPLETENESS & WHOLE-STORY INTEGRITY" in captured_prompt["prompt"]
    assert "SETUP & CONTEXT" in captured_prompt["prompt"]
    assert "RESOLUTION & PUNCHLINE" in captured_prompt["prompt"]
    assert "duration_tier" in captured_prompt["prompt"]

    # 2. Verify returned result
    assert result["success"] is True
    assert result["rescored_clips"] == 1
    c = result["clips"][0]
    assert c["viral_score"] == 94
    assert c["duration_tier"] == "mid"
    assert c["target_platforms"] == ["tiktok", "youtube_shorts"]
    assert c["viral_hook_text"] == "He wanted two million dollars"

    # 3. Verify disk metadata updated
    with open(metadata_file, "r", encoding="utf-8") as f:
        saved = json.load(f)
    saved_clip = saved["shorts"][0]
    assert saved_clip["viral_score"] == 94
    assert saved_clip["duration_tier"] == "mid"
    assert saved_clip["viral_hook_text"] == "He wanted two million dollars"
