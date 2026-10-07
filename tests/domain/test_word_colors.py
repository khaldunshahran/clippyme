"""Slice 2: word_colors -- validation, palette resolution, render mapping.

Covers:
- Captions.word_colors model validation (strict int 0/1/2, known word IDs)
- subtitles.resolve_word_color (per preset / per index, unknown preset)
- project_render._map_words_to_timeline (color index carried through)
- copilot_patch word_colors (accepted / rejected)
- parity: resolve_word_color output == what generate_ass_karaoke burns
"""
import pytest
from pydantic import ValidationError

from clippyme.domain.clip_project import validate_project
from clippyme.domain.copilot_patch import apply_copilot_patch
from clippyme.domain.project_render import _map_words_to_timeline
from clippyme.domain.subtitles import (
    SUBTITLE_PRESETS,
    generate_ass_karaoke,
    hex_to_ass_color,
    resolve_word_color,
)

_WORDS = ["hello", "brave", "world"]


def _project_dict(word_colors=None):
    words = [
        {"id": f"w{i}", "w": t, "start": 1.0 + i, "end": 1.5 + i}
        for i, t in enumerate(_WORDS)
    ]
    caps = {"words": words}
    if word_colors is not None:
        caps["word_colors"] = word_colors
    return {
        "schema": "nugget.clip-project/1",
        "job_id": "word-colors",
        "clip_index": 0,
        "source": {"file": "s.mp4", "width": 1920, "height": 1080,
                   "duration": 60.0},
        "segments": [{"id": "seg0", "start": 0.0, "end": 60.0,
                      "crop": {"x": 0, "y": 0, "w": 1920, "h": 1080}}],
        "captions": caps,
    }


def _project(word_colors=None):
    return validate_project(_project_dict(word_colors))


# --- model validation -----------------------------------------------------

class TestWordColorsValidation:
    def test_absent_defaults_to_empty(self):
        p = _project()
        assert p.captions.word_colors == {}

    def test_valid_0_1_2_accepted(self):
        p = _project({"w0": 0, "w1": 1, "w2": 2})
        assert p.captions.word_colors == {"w0": 0, "w1": 1, "w2": 2}

    def test_string_value_rejected(self):
        with pytest.raises(ValidationError) as ei:
            _project({"w0": "1"})
        assert "must be 0, 1, or 2" in str(ei.value)

    def test_float_value_rejected(self):
        with pytest.raises(ValidationError) as ei:
            _project({"w0": 1.0})
        assert "must be 0, 1, or 2" in str(ei.value)

    def test_bool_value_rejected(self):
        with pytest.raises(ValidationError) as ei:
            _project({"w0": True})
        assert "must be 0, 1, or 2" in str(ei.value)

    def test_out_of_range_rejected(self):
        for bad in (3, -1, 99):
            with pytest.raises(ValidationError) as ei:
                _project({"w1": bad})
            assert (f"word_colors['w1'] must be 0, 1, or 2, got {bad}"
                    in str(ei.value))

    def test_unknown_word_id_rejected(self):
        with pytest.raises(ValidationError) as ei:
            _project({"w99": 1})
        assert "word_colors['w99'] is not a known word ID" in str(ei.value)

    def test_empty_key_rejected(self):
        with pytest.raises(ValidationError) as ei:
            _project({"": 1})
        assert "non-empty string" in str(ei.value)

    def test_non_dict_rejected(self):
        # pydantic's own dict check fires before the field validator here;
        # the helper's message is covered directly below.
        with pytest.raises(ValidationError):
            _project(["w0"])

    def test_helper_rejects_non_dict(self):
        from clippyme.domain.clip_project import validate_word_colors_mapping
        with pytest.raises(ValueError) as ei:
            validate_word_colors_mapping(["w0"], {"w0"})
        assert "must be an object" in str(ei.value)


# --- palette resolution ---------------------------------------------------

class TestResolveWordColor:
    def test_index_0_is_text_color(self):
        assert resolve_word_color("classic_white", 0) == "#FFFFFF"

    def test_index_1_is_highlight_color(self):
        assert resolve_word_color("classic_white", 1) == "#FFFF00"
        assert resolve_word_color("hormozi_bold", 1) == "#00FF00"

    def test_index_2_is_accent_color(self):
        assert resolve_word_color("classic_white", 2) == "#00FF66"
        assert resolve_word_color("fire_impact", 2) == "#FFAA00"

    def test_every_preset_has_all_three(self):
        for preset, style in SUBTITLE_PRESETS.items():
            assert resolve_word_color(preset, 0) == style["text_color"]
            assert resolve_word_color(preset, 1) == style["highlight_color"]
            assert resolve_word_color(preset, 2) == style["accent_color"]

    def test_unknown_preset_falls_back_to_classic_white(self):
        assert (resolve_word_color("nope", 0)
                == resolve_word_color("classic_white", 0))
        assert resolve_word_color("nope", 1) == "#FFFF00"
        assert resolve_word_color("nope", 2) == "#00FF66"

    def test_bad_index_rejected(self):
        for bad in (3, -1, "1", True, None):
            with pytest.raises(ValueError):
                resolve_word_color("classic_white", bad)


# --- render mapping -------------------------------------------------------

class TestMapWordsToTimeline:
    def test_color_index_carried_through(self):
        project = _project({"w1": 2})
        mapped, _ = _map_words_to_timeline(project)
        by_word = {w["word"]: w for w in mapped}
        assert by_word["brave"]["color"] == 2
        assert by_word["hello"]["color"] == 0
        assert by_word["world"]["color"] == 0

    def test_absent_colors_default_to_zero(self):
        project = _project()
        mapped, _ = _map_words_to_timeline(project)
        assert all(w["color"] == 0 for w in mapped)


# --- copilot patch --------------------------------------------------------

class TestCopilotPatchWordColors:
    def test_accepted(self):
        project = _project()
        apply_copilot_patch(project, {"word_colors": {"w2": 1, "w0": 2}})
        assert project.captions.word_colors == {"w2": 1, "w0": 2}

    def test_merges_with_existing(self):
        project = _project({"w0": 1})
        apply_copilot_patch(project, {"word_colors": {"w1": 2}})
        assert project.captions.word_colors == {"w0": 1, "w1": 2}

    def test_bad_id_rejected(self):
        with pytest.raises(ValueError) as ei:
            apply_copilot_patch(_project(), {"word_colors": {"w99": 1}})
        assert "word_colors['w99'] is not a known word ID" in str(ei.value)

    def test_bad_value_rejected(self):
        with pytest.raises(ValueError) as ei:
            apply_copilot_patch(_project(), {"word_colors": {"w0": 5}})
        assert "word_colors['w0'] must be 0, 1, or 2, got 5" in str(ei.value)

    def test_empty_dict_rejected(self):
        with pytest.raises(ValueError):
            apply_copilot_patch(_project(), {"word_colors": {}})

    def test_non_dict_rejected(self):
        with pytest.raises(ValueError):
            apply_copilot_patch(_project(), {"word_colors": "w0"})


# --- ASS parity -----------------------------------------------------------

def _ass_for(words, tmp_path, preset="classic_white"):
    transcript = {"segments": [{"words": words}]}
    out = str(tmp_path / "t.ass")
    assert generate_ass_karaoke(transcript, 0.0, 5.0, out, preset=preset)
    with open(out, encoding="utf-8") as f:
        return f.read()


class TestAssParity:
    def test_colored_word_gets_override(self, tmp_path):
        ass = _ass_for(
            [{"word": "hello", "start": 0.5, "end": 1.0, "color": 1}],
            tmp_path)
        expected = hex_to_ass_color(resolve_word_color("classic_white", 1),
                                    1.0)
        assert f"\\1c{expected}\\2c{expected}" in ass
        assert "HELLO" in ass  # classic_white uppercases

    def test_color2_resolves_accent(self, tmp_path):
        ass = _ass_for(
            [{"word": "hello", "start": 0.5, "end": 1.0, "color": 2}],
            tmp_path)
        expected = hex_to_ass_color(resolve_word_color("classic_white", 2),
                                    1.0)
        assert f"\\1c{expected}\\2c{expected}" in ass

    def test_default_word_has_no_override(self, tmp_path):
        ass = _ass_for(
            [{"word": "hello", "start": 0.5, "end": 1.0, "color": 0}],
            tmp_path)
        assert "\\1c" not in ass
        assert "\\2c" not in ass

    def test_color0_byte_identical_to_no_color_key(self, tmp_path):
        words = [{"word": "hello", "start": 0.5, "end": 1.0}]
        a = _ass_for(words, tmp_path)
        b = _ass_for([{**words[0], "color": 0}], tmp_path)
        assert a == b

    def test_only_colored_words_carry_overrides(self, tmp_path):
        ass = _ass_for([
            {"word": "hello", "start": 0.5, "end": 1.0, "color": 0},
            {"word": "world", "start": 1.0, "end": 1.5, "color": 1},
        ], tmp_path)
        assert ass.count("\\1c") == 1
        assert ass.count("\\2c") == 1
