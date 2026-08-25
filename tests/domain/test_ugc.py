import json
from unittest.mock import MagicMock, patch
from clippyme.ugc.research import research_product_online
from clippyme.ugc.scripting import generate_ugc_scripts


def test_research_product_mocked():
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = json.dumps({
        "product_name": "ClippyMe",
        "tagline": "Turn long videos into viral shorts",
        "target_audience": "Content Creators",
        "core_problem": "Video editing takes hours",
        "unique_solution": "Fully automated AI clipping pipeline",
        "key_features": ["Active speaker tracking", "Auto captions", "Smart cut"],
        "viral_angles": ["Stop wasting 5 hours editing shorts"]
    })
    mock_client.models.generate_content.return_value = mock_response

    with patch("google.genai.Client", return_value=mock_client):
        res = research_product_online("https://clippyme.app", "fake_key")
        assert res["product_name"] == "ClippyMe"
        assert len(res["key_features"]) == 3


def test_generate_ugc_scripts_mocked():
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = json.dumps({
        "scripts": [
            {
                "title": "The 5-Hour Mistake",
                "hook_text_overlay": "STOP EDITING MANUALLY 🛑",
                "full_script_text": "Are you still spending 5 hours editing shorts? Look at this...",
                "estimated_duration_sec": 30,
                "segments": [
                    {"segment_type": "talking_head", "voiceover": "Are you still spending 5 hours editing?"}
                ]
            }
        ]
    })
    mock_client.models.generate_content.return_value = mock_response

    with patch("google.genai.Client", return_value=mock_client):
        scripts = generate_ugc_scripts(
            research_data={"product_name": "ClippyMe"},
            gemini_key="fake_key",
        )
        assert len(scripts) == 1
        assert scripts[0]["title"] == "The 5-Hour Mistake"


def test_product_html_parser():
    from clippyme.ugc.research import _ProductHTMLParser

    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Awesome Tool - AI Video Platform</title>
        <meta name="description" content="Turn videos into viral vertical shorts automatically.">
        <style>body { color: red; }</style>
        <script>console.log('secret');</script>
    </head>
    <body>
        <header><nav>Home About</nav></header>
        <h1>Main Heading</h1>
        <h2>Feature 1</h2>
        <p>This is the description paragraph of the product.</p>
        <footer>Copyright 2026</footer>
    </body>
    </html>
    """
    parser = _ProductHTMLParser()
    parser.feed(html)

    assert parser.title == "Awesome Tool - AI Video Platform"
    assert parser.meta_desc == "Turn videos into viral vertical shorts automatically."
    assert parser.headings == ["Main Heading", "Feature 1"]
    assert "secret" not in " ".join(parser.text_chunks)
    assert "description paragraph" in " ".join(parser.text_chunks)

