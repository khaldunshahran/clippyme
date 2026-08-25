import json
from unittest.mock import MagicMock, patch
from clippyme.studio.youtube_studio import (
    generate_viral_titles,
    refine_viral_titles,
    generate_youtube_chapters,
)


def test_generate_viral_titles_mocked():
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = json.dumps({
        "titles": ["10 Tips For High Productivity", "How I Built My System"],
        "summary": "A video about productivity systems.",
        "recommended": [{"index": 0, "reason": "High CTR hook"}]
    })
    mock_client.models.generate_content.return_value = mock_response

    with patch("google.genai.Client", return_value=mock_client):
        res = generate_viral_titles(
            transcript_text="Today we talk about productivity systems.",
            api_key="fake_key",
        )
        assert len(res["titles"]) == 2
        assert res["titles"][0] == "10 Tips For High Productivity"
        assert len(res["recommended"]) == 1


def test_refine_viral_titles_mocked():
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = json.dumps({
        "titles": ["Make It More Extreme", "Unbelievable Results"]
    })
    mock_client.models.generate_content.return_value = mock_response

    with patch("google.genai.Client", return_value=mock_client):
        res = refine_viral_titles(
            context="Productivity video",
            user_instruction="Make it sound more urgent",
            api_key="fake_key",
        )
        assert len(res["titles"]) == 2


def test_generate_youtube_chapters_mocked():
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = json.dumps({
        "description": "Full description text",
        "chapters": [
            {"timestamp": "00:00", "title": "Intro"},
            {"timestamp": "01:30", "title": "Core Method"}
        ],
        "hashtags": ["#productivity", "#tips"]
    })
    mock_client.models.generate_content.return_value = mock_response

    with patch("google.genai.Client", return_value=mock_client):
        res = generate_youtube_chapters(
            transcript_segments=[
                {"start": 0.0, "text": "Welcome to the video"},
                {"start": 90.0, "text": "Here is the core method"}
            ],
            api_key="fake_key",
        )
        assert len(res["chapters"]) == 2
        assert res["chapters"][0]["timestamp"] == "00:00"


def test_generate_youtube_thumbnail_gemini_mocked(tmp_path):
    import base64
    from clippyme.studio.youtube_studio import generate_youtube_thumbnail

    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_candidate = MagicMock()
    mock_part = MagicMock()
    mock_part.inline_data.data = base64.b64encode(b"fake_image_bytes_png")
    mock_candidate.content.parts = [mock_part]
    mock_response.candidates = [mock_candidate]
    mock_client.models.generate_content.return_value = mock_response

    with patch("google.genai.Client", return_value=mock_client):
        for ratio in ["16:9", "9:16", "1:1"]:
            out_file = generate_youtube_thumbnail(
                title="Amazing Video Title",
                output_dir=str(tmp_path),
                api_key="fake_key",
                aspect_ratio=ratio,
                model="gemini-2.5-flash",
            )
            assert out_file.endswith(".png")
            with open(out_file, "rb") as f:
                assert f.read() == b"fake_image_bytes_png"


def test_generate_youtube_thumbnail_imagen_mocked(tmp_path):
    from clippyme.studio.youtube_studio import generate_youtube_thumbnail

    mock_client = MagicMock()
    mock_result = MagicMock()
    mock_img_obj = MagicMock()
    mock_img_obj.image.image_bytes = b"imagen_rendered_bytes"
    mock_result.generated_images = [mock_img_obj]
    mock_client.models.generate_images.return_value = mock_result

    with patch("google.genai.Client", return_value=mock_client):
        out_file = generate_youtube_thumbnail(
            title="Nano Banana Cover",
            output_dir=str(tmp_path),
            api_key="fake_key",
            aspect_ratio="9:16",
            model="imagen-3.0-generate-002",
        )
        assert out_file.endswith(".png")
        with open(out_file, "rb") as f:
            assert f.read() == b"imagen_rendered_bytes"
        assert mock_client.models.generate_images.called

