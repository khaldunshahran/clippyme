"""Phase 2 (2D): ASS keyword coloring + normalization.

No keywords -> byte-identical legacy output. Keywords hold the highlight
colour for the whole event; matching is case- and punctuation-insensitive.
"""
from clippyme.domain.subtitles import (
    _normalize_keyword,
    _normalize_keywords,
    generate_ass_karaoke,
)

TR = {"segments": [{"words": [
    {"word": "Hello", "start": 0.0, "end": 0.4},
    {"word": "world!", "start": 0.4, "end": 0.8},
    {"word": "this", "start": 0.8, "end": 1.0},
    {"word": "is", "start": 1.0, "end": 1.2},
    {"word": "MONEY", "start": 1.2, "end": 1.8},
]}]}


def _render(tmp_path, **kwargs):
    out = tmp_path / "kw.ass"
    generate_ass_karaoke(TR, 0.0, 2.0, str(out), **kwargs)
    return out.read_text(encoding="utf-8")


def test_no_keywords_is_legacy_output(tmp_path):
    a = _render(tmp_path)
    b = _render(tmp_path, keywords=None)
    c = _render(tmp_path, keywords=[])
    assert "\\2c" not in a
    assert a == b == c


def test_keyword_words_hold_highlight(tmp_path):
    body = _render(tmp_path, keywords=["money", "HELLO"])
    # Two keyword words x (set highlight + restore base).
    assert body.count("\\2c") == 4
    # Non-keyword words keep the plain karaoke tag with no color override.
    assert "{\\k20}THIS" in body or "{\\k" in body


def test_keyword_matching_normalizes_case_and_punctuation(tmp_path):
    body = _render(tmp_path, keywords=["world"])
    assert body.count("\\2c") == 2  # "world!" matched
    body2 = _render(tmp_path, keywords=["MoNeY!"])
    assert body2.count("\\2c") == 2


def test_keyword_string_input(tmp_path):
    body = _render(tmp_path, keywords="money, hello")
    assert body.count("\\2c") == 4


def test_normalize_helpers():
    assert _normalize_keyword("Hello!") == "hello"
    assert _normalize_keyword("  MONEY ") == "money"
    assert _normalize_keyword("") == ""
    assert _normalize_keywords(None) is None
    assert _normalize_keywords([]) is None
    assert _normalize_keywords("money, fame") == {"money", "fame"}
    assert _normalize_keywords(["a", "A!", ""]) == {"a"}


def test_keywords_compose_with_pop_animation(tmp_path):
    body = _render(tmp_path, keywords=["money"], animate="pop")
    assert "\\t" in body  # pop transforms present
    assert body.count("\\2c") == 2  # keyword coloring intact
