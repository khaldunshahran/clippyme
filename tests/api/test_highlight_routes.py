import os
import json
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from clippyme.api.app import app

client = TestClient(app)


def test_get_highlights_job_not_found(tmp_path):
    with patch("clippyme.api.highlight_routes.OUTPUT_DIR", str(tmp_path)):
        res = client.get("/api/highlights/nonexistent_job_123")
        assert res.status_code == 404
        assert "not found" in res.json()["detail"].lower()


def test_get_highlights_success(tmp_path):
    job_id = "test_job_highlights"
    job_dir = tmp_path / job_id
    job_dir.mkdir(parents=True)
    metadata_file = job_dir / f"{job_id}_metadata.json"
    metadata_file.write_text(json.dumps({
        "title": "My Podcast",
        "highlights": [
            {
                "id": "abc123",
                "title": "Epic 60s Reel",
                "target_duration": 60,
                "actual_duration": 58.5,
                "aspect": "9:16",
                "filename": "highlight_reel_60s_9x16_abc123.mp4",
                "video_url": f"/videos/{job_id}/highlight_reel_60s_9x16_abc123.mp4",
                "cuts": []
            }
        ]
    }), encoding="utf-8")

    with patch("clippyme.api.highlight_routes.OUTPUT_DIR", str(tmp_path)):
        res = client.get(f"/api/highlights/{job_id}")
        assert res.status_code == 200
        data = res.json()
        assert "highlights" in data
        assert len(data["highlights"]) == 1
        assert data["highlights"][0]["title"] == "Epic 60s Reel"


def test_get_highlights_without_prior_metadata_creates_default(tmp_path):
    job_id = "job_without_meta"
    job_dir = tmp_path / job_id
    job_dir.mkdir(parents=True)
    # Only source_info.json exists
    src_file = job_dir / "source_info.json"
    src_file.write_text(json.dumps({"title": "Direct Upload Video", "duration": 180.0}), encoding="utf-8")

    with patch("clippyme.api.highlight_routes.OUTPUT_DIR", str(tmp_path)):
        res = client.get(f"/api/highlights/{job_id}")
        assert res.status_code == 200
        data = res.json()
        assert "highlights" in data
        assert data["highlights"] == []
        assert (job_dir / "metadata.json").exists()


def test_plan_highlights_success(tmp_path):
    job_id = "job_plan_test"
    job_dir = tmp_path / job_id
    job_dir.mkdir(parents=True)

    metadata_file = job_dir / f"{job_id}_metadata.json"
    metadata_file.write_text(json.dumps({
        "title": "Podcast Episode",
        "full_transcript": {
            "segments": [
                {
                    "words": [
                        {"word": "Intro", "start": 0.0, "end": 1.0},
                        {"word": "sentence", "start": 1.0, "end": 2.0},
                        {"word": "Punchline", "start": 50.0, "end": 55.0},
                    ]
                }
            ]
        }
    }), encoding="utf-8")

    mock_gemini_response = MagicMock()
    mock_gemini_response.text = json.dumps({
        "title": "Top Highlights",
        "cuts": [
            {"role": "hook", "start": 50.0, "end": 55.0, "summary": "Punchline teaser"},
            {"role": "setup", "start": 0.0, "end": 2.0, "summary": "Intro"}
        ]
    })
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = mock_gemini_response

    with patch("clippyme.api.highlight_routes.OUTPUT_DIR", str(tmp_path)), \
         patch("clippyme.domain.highlight_service.load_persistent_config", return_value={"GEMINI_API_KEY": "fake_key"}), \
         patch("google.genai.Client", return_value=mock_client):

        res = client.post(
            f"/api/highlights/{job_id}/plan",
            json={"target_duration": 60},
            headers={"x-gemini-key": "fake_key"}
        )
        assert res.status_code == 200
        data = res.json()
        assert data["title"] == "Top Highlights"
        assert len(data["cuts"]) == 2


def test_render_highlight_custom_cuts_and_aspect(tmp_path):
    job_id = "job_render_aspect"
    job_dir = tmp_path / job_id
    job_dir.mkdir(parents=True)

    source_file = job_dir / "source_video.mp4"
    source_file.write_bytes(b"dummy mp4 video bytes")

    metadata_file = job_dir / f"{job_id}_metadata.json"
    metadata_file.write_text(json.dumps({
        "title": "Aspect Test",
        "full_transcript": {
            "segments": [
                {
                    "words": [
                        {"word": "First", "start": 5.0, "end": 10.0},
                        {"word": "Second", "start": 20.0, "end": 25.0},
                    ]
                }
            ]
        }
    }), encoding="utf-8")

    mock_subp_res = MagicMock()
    mock_subp_res.returncode = 0

    with patch("clippyme.api.highlight_routes.OUTPUT_DIR", str(tmp_path)), \
         patch("subprocess.run", return_value=mock_subp_res), \
         patch("clippyme.domain.subtitles.burn_subtitles", return_value=True), \
         patch("clippyme.domain.grade.apply_grade", return_value=True):

        res = client.post(
            f"/api/highlights/{job_id}/render",
            json={
                "target_duration": 60,
                "aspect": "1:1",
                "reframe_mode": "blur_pad",
                "cuts": [
                    {"role": "hook", "start": 5.0, "end": 10.0, "summary": "Cut 1"},
                    {"role": "beat", "start": 20.0, "end": 25.0, "summary": "Cut 2"}
                ],
                "subtitles": {
                    "enabled": True,
                    "preset": "neon_green",
                },
                "grade_preset": "vibrant"
            }
        )
        assert res.status_code == 200
        data = res.json()
        assert data["success"] is True
        assert data["highlight"]["aspect"] == "1:1"
        assert len(data["highlight"]["cuts"]) == 2


def test_delete_highlight(tmp_path):
    job_id = "job_del_test"
    job_dir = tmp_path / job_id
    job_dir.mkdir(parents=True)

    filename = "highlight_reel_60s_9x16_abc123.mp4"
    file_path = job_dir / filename
    file_path.write_bytes(b"video bytes")

    metadata_file = job_dir / f"{job_id}_metadata.json"
    metadata_file.write_text(json.dumps({
        "highlights": [
            {"filename": filename, "title": "To Delete"}
        ]
    }), encoding="utf-8")

    with patch("clippyme.api.highlight_routes.OUTPUT_DIR", str(tmp_path)):
        res = client.delete(f"/api/highlights/{job_id}/{filename}")
        assert res.status_code == 200
        assert res.json()["success"] is True
        assert not file_path.exists()


def test_generate_all_highlights(tmp_path):
    job_id = "job_gen_all"
    job_dir = tmp_path / job_id
    job_dir.mkdir(parents=True)

    source_file = job_dir / "source_video.mp4"
    source_file.write_bytes(b"dummy mp4 video bytes")

    metadata_file = job_dir / f"{job_id}_metadata.json"
    metadata_file.write_text(json.dumps({
        "title": "Podcast Episode",
        "duration": 600.0,
        "full_transcript": {
            "segments": [
                {
                    "words": [
                        {"word": "Intro", "start": 0.0, "end": 5.0},
                        {"word": "Middle", "start": 50.0, "end": 60.0},
                    ]
                }
            ]
        }
    }), encoding="utf-8")

    mock_gemini_response = MagicMock()
    mock_gemini_response.text = json.dumps({
        "reels": [
            {
                "tier": "micro",
                "title": "Micro Teaser",
                "viral_score": 96,
                "summary": "Teaser",
                "cuts": [{"role": "hook", "start": 50.0, "end": 60.0, "summary": "Middle"}]
            },
            {
                "tier": "recap",
                "title": "Full Story",
                "viral_score": 90,
                "summary": "Recap",
                "cuts": [
                    {"role": "setup", "start": 0.0, "end": 5.0, "summary": "Intro"},
                    {"role": "beat", "start": 50.0, "end": 60.0, "summary": "Middle"}
                ]
            }
        ]
    })
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = mock_gemini_response

    mock_subp_res = MagicMock()
    mock_subp_res.returncode = 0

    with patch("clippyme.api.highlight_routes.OUTPUT_DIR", str(tmp_path)), \
         patch("clippyme.domain.highlight_service.load_persistent_config", return_value={"GEMINI_API_KEY": "fake_key"}), \
         patch("google.genai.Client", return_value=mock_client), \
         patch("subprocess.run", return_value=mock_subp_res), \
         patch("clippyme.domain.subtitles.burn_subtitles", return_value=True), \
         patch("clippyme.domain.grade.apply_grade", return_value=True):

        res = client.post(
            f"/api/highlights/{job_id}/generate-all",
            json={"aspect": "9:16", "reframe_mode": "blur_pad"},
            headers={"x-gemini-key": "fake_key"}
        )
        assert res.status_code == 200
        data = res.json()
        assert data["success"] is True
        assert len(data["highlights"]) == 2
        assert data["highlights"][0]["tier"] == "micro"
        assert data["highlights"][1]["tier"] == "recap"


def test_apply_edit_highlight(tmp_path):
    job_id = "job_apply_edit"
    job_dir = tmp_path / job_id
    job_dir.mkdir(parents=True)

    source_file = job_dir / "source_video.mp4"
    source_file.write_bytes(b"dummy mp4 video bytes")

    hl_id = "hl_edit_123"
    metadata_file = job_dir / f"{job_id}_metadata.json"
    metadata_file.write_text(json.dumps({
        "title": "Apply Edit Test",
        "highlights": [
            {
                "id": hl_id,
                "title": "Original Reel",
                "target_duration": 60,
                "aspect": "9:16",
                "filename": f"highlight_{hl_id}.mp4",
                "cuts": [{"role": "beat", "start": 5.0, "end": 15.0, "summary": "Beat"}]
            }
        ]
    }), encoding="utf-8")

    mock_subp_res = MagicMock()
    mock_subp_res.returncode = 0

    with patch("clippyme.api.highlight_routes.OUTPUT_DIR", str(tmp_path)), \
         patch("subprocess.run", return_value=mock_subp_res), \
         patch("clippyme.domain.subtitles.burn_subtitles", return_value=True), \
         patch("clippyme.domain.grade.apply_grade", return_value=True):

        res = client.post(
            f"/api/highlights/{job_id}/{hl_id}/apply-edit",
            json={
                "aspect": "16:9",
                "title": "Updated Widescreen Reel",
                "cuts": [{"role": "beat", "start": 6.0, "end": 14.0, "summary": "Tighter beat"}],
            }
        )
        assert res.status_code == 200
        data = res.json()
        assert data["success"] is True
        assert data["highlight"]["title"] == "Updated Widescreen Reel"
        assert data["highlight"]["aspect"] == "16:9"


def test_process_endpoint_with_highlights_flag(tmp_path):
    with patch("clippyme.api.app.OUTPUT_DIR", str(tmp_path)), \
         patch("clippyme.api.app.submit_job") as mock_submit:
        res = client.post(
            "/api/process",
            json={
                "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                "highlights": True,
            },
            headers={"X-Gemini-Key": "test_key", "Origin": "http://localhost:5175"}
        )
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "queued"
        assert "job_id" in data
        assert mock_submit.called
        call_kwargs = mock_submit.call_args.kwargs
        assert "--highlights" in call_kwargs["cmd"]


def test_render_highlights_with_hook_and_logo(tmp_path):
    job_id = "job_hook_logo_test"
    job_dir = tmp_path / job_id
    job_dir.mkdir(parents=True)

    source_file = job_dir / "source_video.mp4"
    source_file.write_bytes(b"dummy mp4 video bytes")

    logo_file = tmp_path / "test_logo.png"
    logo_file.write_bytes(b"dummy png bytes")

    metadata_file = job_dir / f"{job_id}_metadata.json"
    metadata_file.write_text(json.dumps({
        "title": "Hook & Logo Test",
        "duration": 60.0,
        "full_transcript": {"segments": [{"words": [{"word": "Hi", "start": 0.0, "end": 2.0}]}]}
    }), encoding="utf-8")

    mock_subp_res = MagicMock()
    mock_subp_res.returncode = 0

    with patch("clippyme.api.highlight_routes.OUTPUT_DIR", str(tmp_path)), \
         patch("subprocess.run", return_value=mock_subp_res), \
         patch("clippyme.domain.subtitles.burn_subtitles", return_value=True), \
         patch("clippyme.domain.hooks.add_hook_to_video", return_value=True) as mock_add_hook, \
         patch("clippyme.domain.logo.add_logo_to_video", return_value=True) as mock_add_logo:

        res = client.post(
            f"/api/highlights/{job_id}/render",
            json={
                "target_duration": 30,
                "aspect": "9:16",
                "cuts": [{"role": "setup", "start": 0.0, "end": 10.0}],
                "hook": {
                    "enabled": True,
                    "text": "Watch this recap!",
                    "position": "top",
                    "size": "L"
                },
                "logo": {
                    "enabled": True,
                    "path": str(logo_file),
                    "position": "top-right"
                }
            }
        )
        assert res.status_code == 200
        assert mock_add_hook.called
        assert mock_add_logo.called
        data = res.json()
        assert data["highlight"]["hook"]["text"] == "Watch this recap!"


