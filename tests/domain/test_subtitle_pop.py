"""Phase 2 (2A): animated caption pop — explicit kwarg wins, legacy untouched."""
from clippyme.domain.subtitles import generate_ass_karaoke

TR = {"segments": [{"words": [
    {"word": "hello", "start": 0.0, "end": 0.5},
    {"word": "world", "start": 0.5, "end": 1.0},
]}]}


def _render(tmp_path, **kwargs):
    out = tmp_path / "pop.ass"
    generate_ass_karaoke(TR, 0.0, 1.5, str(out), **kwargs)
    return out.read_text(encoding="utf-8")


def test_pop_emits_scale_transforms(tmp_path):
    body = _render(tmp_path, animate="pop")
    assert "\\t" in body
    assert "\\fscx112" in body or "fscx112" in body


def test_no_animate_has_no_transforms(tmp_path):
    body = _render(tmp_path)
    assert "\\t" not in body


def test_unknown_animate_value_behaves_like_none(tmp_path):
    assert _render(tmp_path, animate="wiggle") == _render(tmp_path)


def test_pop_does_not_change_event_count(tmp_path):
    plain = _render(tmp_path)
    pop = _render(tmp_path, animate="pop")
    assert plain.count("Dialogue:") == pop.count("Dialogue:")
