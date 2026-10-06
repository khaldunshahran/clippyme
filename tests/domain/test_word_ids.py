"""Stable word IDs: migration, counter, duplicates, ID-keyed word_edits."""

import pytest

from clippyme.domain.clip_project import (
    Captions,
    CaptionWord,
    ClipProject,
    _backfill_word_ids,
    validate_project,
)
from clippyme.domain.copilot_patch import apply_copilot_patch


def _project_dict(**over):
    base = {
        "schema": "nugget.clip-project/1",
        "job_id": "j1",
        "clip_index": 0,
        "source": {"file": "s.mp4", "width": 1920, "height": 1080,
                   "duration": 60.0},
        "segments": [{"id": "seg0", "start": 0.0, "end": 60.0,
                      "crop": {"x": 0, "y": 0, "w": 1920, "h": 1080}}],
        "captions": {"words": [], "edits": {}},
    }
    base.update(over)
    return base


def _words(n, with_ids=True):
    return [
        {"id": f"w{i}" if with_ids else "", "w": f"word{i}",
         "start": float(i), "end": float(i) + 0.5}
        for i in range(n)
    ]


def test_backfill_assigns_ids_from_counter():
    d = _project_dict()
    d["captions"] = {"words": _words(3, with_ids=False), "next_word_id": 10}
    p = validate_project(d)
    assert [w.id for w in p.captions.words] == ["w10", "w11", "w12"]
    assert p.captions.next_word_id == 13


def test_backfill_advances_past_existing_ids():
    d = _project_dict()
    words = _words(2, with_ids=False)
    words[0]["id"] = "w41"
    d["captions"] = {"words": words, "next_word_id": 0}
    p = validate_project(d)
    assert p.captions.words[0].id == "w41"
    assert p.captions.words[1].id == "w42"
    assert p.captions.next_word_id == 43


def test_backfill_is_idempotent():
    d = _project_dict()
    d["captions"] = {"words": _words(2), "next_word_id": 2}
    once = validate_project(d).model_dump()
    twice = validate_project(once)
    assert [w.id for w in twice.captions.words] == ["w0", "w1"]
    assert twice.captions.next_word_id == 2


def test_duplicate_word_ids_rejected():
    d = _project_dict()
    words = _words(2)
    words[1]["id"] = "w0"
    d["captions"] = {"words": words}
    with pytest.raises(ValueError, match="duplicate word id"):
        validate_project(d)


def test_word_edits_accept_word_ids():
    p = validate_project(_project_dict(captions={"words": _words(3)}))
    apply_copilot_patch(p, {"word_edits": {"w1": "corrected"}})
    assert p.captions.edits == {"w1": "corrected"}


def test_word_edits_legacy_index_keys_still_work():
    p = validate_project(_project_dict(captions={"words": _words(3)}))
    apply_copilot_patch(p, {"word_edits": {"2": "fixed"}})
    # Stored under the stable ID, not the transient index.
    assert p.captions.edits == {"w2": "fixed"}


def test_word_edits_rejects_unknown_keys():
    p = validate_project(_project_dict(captions={"words": _words(2)}))
    with pytest.raises(ValueError, match="word id or a word index"):
        apply_copilot_patch(p, {"word_edits": {"w99": "x"}})
    with pytest.raises(ValueError, match="word id or a word index"):
        apply_copilot_patch(p, {"word_edits": {"7": "x"}})
    with pytest.raises(ValueError, match="word id or a word index"):
        apply_copilot_patch(p, {"word_edits": {"nope": "x"}})


def test_word_edits_rejects_bad_values():
    p = validate_project(_project_dict(captions={"words": _words(2)}))
    with pytest.raises(ValueError, match="non-empty string"):
        apply_copilot_patch(p, {"word_edits": {"w0": "  "}})
    with pytest.raises(ValueError, match="non-empty object"):
        apply_copilot_patch(p, {"word_edits": {}})


# --- Phase A: removed flag + read-path duplicate repair --------------------

def test_removed_flag_defaults_false_and_roundtrips():
    d = _project_dict(captions={"words": _words(2)})
    p = validate_project(d)
    assert all(w.removed is False for w in p.captions.words)
    rt = validate_project(p.model_dump())
    assert [w.id for w in rt.captions.words] == ["w0", "w1"]
    assert all(w.removed is False for w in rt.captions.words)


def test_removed_words_keep_ids_and_edits():
    d = _project_dict(captions={"words": _words(3)})
    d["captions"]["words"][1]["removed"] = True
    p = validate_project(d)
    assert p.captions.words[1].id == "w1"
    assert p.captions.words[1].removed is True
    # ID-keyed edits still resolve to the removed word.
    apply_copilot_patch(p, {"word_edits": {"w1": "kept"}})
    assert p.captions.edits == {"w1": "kept"}


def test_map_words_skips_removed():
    from clippyme.domain.project_render import _map_words_to_timeline
    d = _project_dict(captions={"words": _words(3)})
    d["captions"]["words"][1]["removed"] = True
    p = validate_project(d)
    mapped, _dur = _map_words_to_timeline(p)
    assert [w["word"] for w in mapped] == ["word0", "word2"]


def test_read_repair_reassigns_duplicate_ids(caplog):
    d = _project_dict()
    words = _words(3)
    words[2]["id"] = "w1"  # duplicate of words[1]
    d["captions"] = {"words": words, "next_word_id": 3}
    out = _backfill_word_ids(d, repair_duplicates=True)  # must not raise
    assert [w["id"] for w in out["captions"]["words"]] == ["w0", "w1", "w3"]
    assert out["captions"]["next_word_id"] == 4
    assert "duplicate word id" in caplog.text


def test_read_repair_still_rejects_on_write():
    d = _project_dict()
    words = _words(2)
    words[1]["id"] = "w0"
    d["captions"] = {"words": words}
    with pytest.raises(ValueError, match="duplicate word id"):
        _backfill_word_ids(d)  # strict: no repair flag
    with pytest.raises(ValueError, match="duplicate word id"):
        validate_project(d)


def test_read_repair_backfills_missing_ids_too():
    d = _project_dict()
    d["captions"] = {"words": _words(3, with_ids=False)}
    out = _backfill_word_ids(d, repair_duplicates=True)
    assert [w["id"] for w in out["captions"]["words"]] == ["w0", "w1", "w2"]
    assert out["captions"]["next_word_id"] == 3
