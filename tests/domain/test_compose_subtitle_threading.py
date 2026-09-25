"""Phase 2 (2A + 2D): compose threads subtitle animate/keywords into ASS.

_async _apply_subtitles_ is driven with asyncio.run(); generate_ass_karaoke,
burn_subtitles and the QA probe are stubbed so the test asserts purely on
the forwarded kwargs.
"""
import asyncio

from clippyme.domain import compose
from clippyme.domain.compose import _resolve_subtitle_keywords


def test_resolve_subtitle_keywords_precedence():
    params = {"keywords": ["a"]}
    assert _resolve_subtitle_keywords(params, {}, {}) == ["a"]
    assert _resolve_subtitle_keywords({}, {"keywords": ["b"]}, {}) == ["b"]
    assert _resolve_subtitle_keywords(
        {}, {}, {"emphasis_keywords": ["c"]}) == ["c"]
    assert _resolve_subtitle_keywords(
        {}, {"highlight_words": ["d"]}, {}) == ["d"]
    assert _resolve_subtitle_keywords({}, {}, {}) is None


def _run_apply(monkeypatch, tmp_path, subtitle_params, clip_info, metadata):
    captured = {}

    def fake_ass(*args, **kwargs):
        captured.update(kwargs)
        return True

    async def fake_verify(*args, **kwargs):
        return True

    monkeypatch.setattr(compose, "generate_ass_karaoke", fake_ass)
    monkeypatch.setattr(compose, "burn_subtitles", lambda *a, **k: None)
    monkeypatch.setattr(compose, "_verify_subtitles_burned", fake_verify)
    monkeypatch.setattr(compose, "_letterbox_caption_band_top",
                        lambda *a, **k: None)
    out = asyncio.run(compose._apply_subtitles(
        "in.mp4", str(tmp_path), 0, metadata, clip_info,
        subtitle_params, [], pre_vf=None))
    return out, captured


def test_animate_and_keywords_forwarded(monkeypatch, tmp_path):
    params = {"animate": "pop", "keywords": ["money"]}
    clip_info = {"start": 0, "end": 8}
    metadata = {"transcript": {"segments": []}}
    out, captured = _run_apply(monkeypatch, tmp_path, params, clip_info, metadata)
    assert captured["animate"] == "pop"
    assert captured["keywords"] == ["money"]
    assert out.endswith("composed_sub_0.mp4")


def test_metadata_keywords_used_when_params_empty(monkeypatch, tmp_path):
    params = {}
    clip_info = {"start": 0, "end": 8, "keywords": ["fame"]}
    metadata = {"transcript": {"segments": []}}
    _, captured = _run_apply(monkeypatch, tmp_path, params, clip_info, metadata)
    assert captured["animate"] is None
    assert captured["keywords"] == ["fame"]
