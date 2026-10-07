"""Task 04: speaker evidence and shared duration policy tests."""
import ast
import sys

import pytest

from clippyme.pipeline import gemini_request as gr
from clippyme.pipeline.duration_policy import resolve_duration_policy
from clippyme.pipeline import cut_ops


# --- Speaker contract ---

def _transcript(words_with_speakers):
    """Build a transcript_result from (word, start, end, speaker) tuples."""
    words = []
    for w, s, e, sp in words_with_speakers:
        entry = {"word": w, "start": s, "end": e}
        if sp != "missing":
            entry["speaker"] = sp
        words.append(entry)
    return {"segments": [{"words": words}]}


def test_speaker_zero_preserved():
    """Speaker ID 0 is falsy but valid; must be preserved."""
    t = _transcript([("hello", 0.0, 0.5, 0), ("world", 0.5, 1.0, 0)])
    words = gr.extract_prompt_words(t)
    assert words[0]["sp"] == 0
    assert words[1]["sp"] == 0


def test_two_speaker_turn():
    """Two-speaker turn: IDs preserved, order and timestamps exact."""
    t = _transcript([
        ("hello", 0.0, 0.5, 0),
        ("hi", 0.6, 1.0, 1),
        ("bye", 1.1, 1.5, 0),
    ])
    words = gr.extract_prompt_words(t)
    assert [w["sp"] for w in words] == [0, 1, 0]
    assert words[0]["s"] == 0.0 and words[2]["e"] == 1.5


def test_unknown_speaker_stays_unknown():
    """Missing speaker evidence -> no 'sp' key (not 0)."""
    t = _transcript([("hello", 0.0, 0.5, "missing")])
    words = gr.extract_prompt_words(t)
    assert "sp" not in words[0]


def test_invalid_speaker_rejected():
    """Bool, negative, non-integer speakers rejected."""
    for bad in (True, False, -1, -5, 1.5, "1", None, [], {}):
        t = _transcript([("w", 0.0, 0.5, bad if bad != "missing" else "missing")])
        # For None, we need to actually set it (not missing).
        if bad is None:
            t = {"segments": [{"words": [{"word": "w", "start": 0.0, "end": 0.5, "speaker": None}]}]}
        words = gr.extract_prompt_words(t)
        assert "sp" not in words[0], f"bad={bad}"


def test_toon_speaker_column():
    """TOON includes sp column when evidence exists; '-' for unknown."""
    words = [
        {"w": "hello", "s": 0.0, "e": 0.5, "sp": 0},
        {"w": "world", "s": 0.5, "e": 1.0},  # unknown
        {"w": "hi", "s": 1.0, "e": 1.5, "sp": 1},
    ]
    toon = gr.encode_words_toon(words)
    assert "words[3]{w,s,e,sp}:" in toon
    assert ",0" in toon  # speaker 0 preserved
    assert ",-" in toon  # unknown as '-', not 0
    assert ",1" in toon


def test_toon_no_speaker_compact():
    """No speaker evidence -> compact three-column form."""
    words = [{"w": "hello", "s": 0.0, "e": 0.5}]
    toon = gr.encode_words_toon(words)
    assert "words[1]{w,s,e}:" in toon
    assert "sp" not in toon


def test_segment_speaker_not_copied_to_words():
    """Segment majority label NOT copied to words without evidence."""
    t = {
        "segments": [{
            "speaker": 1,  # segment majority
            "words": [
                {"word": "a", "start": 0.0, "end": 0.5},  # no word speaker
                {"word": "b", "start": 0.5, "end": 1.0, "speaker": 0},  # conflicting
            ],
        }]
    }
    words = gr.extract_prompt_words(t)
    assert "sp" not in words[0]  # not copied from segment
    assert words[1]["sp"] == 0  # word evidence wins
    # Segment summary available separately.
    segs = gr.extract_segment_speakers(t)
    assert len(segs) == 1 and segs[0]["speaker"] == 1
    assert segs[0]["scope"] == "segment-summary"


def test_build_viral_prompt_serializes_speakers():
    """Actual build_viral_prompt includes speaker evidence in TOON."""
    t = _transcript([("hello", 0.0, 0.5, 0), ("world", 0.5, 1.0, 1)])
    t["language"] = "en"
    prompt, words = gr.build_viral_prompt(t, video_duration=300)
    # Words carry speakers.
    assert words[0]["sp"] == 0 and words[1]["sp"] == 1
    # TOON in prompt has speaker column.
    assert "{w,s,e,sp}" in prompt
    # Prompt wording is honest about anonymous labels.
    assert "anonymous voice labels" in prompt.lower() or "NOT real names" in prompt


# --- Duration policy ---

def test_policy_fixed_max30():
    p = resolve_duration_policy(min_duration=15, max_duration=30)
    assert p.min_duration == 15.0 and p.max_duration == 30.0
    assert p.mode == "fixed"


def test_policy_fixed_max60():
    p = resolve_duration_policy(max_duration=60)
    assert p.max_duration == 60.0


def test_policy_long_clips():
    for max_d in (90, 150, 240):
        p = resolve_duration_policy(max_duration=max_d, video_duration=600)
        assert p.max_duration == float(max_d), f"max={max_d}"


def test_policy_multi_tier_default():
    p = resolve_duration_policy(video_duration=600)
    assert p.mode == "multi-tier"
    assert p.min_duration == 15.0
    assert p.max_duration <= 300.0


def test_policy_explicit_max_hard_limit():
    """Explicit max is hard limit, even with mixed mode."""
    p = resolve_duration_policy(max_duration=90, duration_mode="mixed",
                                video_duration=600)
    assert p.max_duration == 90.0


def test_policy_contradictory_bounds():
    """min > max: max wins, min clamped (deterministic)."""
    p = resolve_duration_policy(min_duration=100, max_duration=30)
    assert p.max_duration == 30.0
    assert p.min_duration == 30.0
    assert p.min_clamped is True


def test_policy_invalid_inputs():
    """NaN, Inf, bool, negative, zero treated as absent."""
    import math
    for bad in (float("nan"), float("inf"), True, False, -10, 0):
        p = resolve_duration_policy(max_duration=bad, video_duration=600)
        # Falls back to default 120 (not NaN/Inf).
        assert math.isfinite(p.max_duration)
        assert p.max_duration == 120.0


def test_policy_short_source():
    """Bounds never exceed source duration."""
    p = resolve_duration_policy(max_duration=120, video_duration=45)
    assert p.max_duration == 45.0
    assert p.max_clamped_to_source is True


def test_prompt_and_repair_share_policy():
    """build_viral_prompt bounds match resolve_duration_policy."""
    t = _transcript([("w", 0.0, 0.5, "missing")])
    t["language"] = "en"
    prompt, _ = gr.build_viral_prompt(
        t, video_duration=600, min_duration=20, max_duration=90)
    p = resolve_duration_policy(min_duration=20, max_duration=90,
                                video_duration=600)
    # Prompt contains the resolved bounds.
    assert "20" in prompt and "90" in prompt
    assert p.max_duration == 90.0


# --- Repair respects policy ---

def _words_fixture():
    """Words with sentence boundaries for repair tests."""
    words = []
    t = 0.0
    for i, w in enumerate(["Hello", "world.", "This", "is", "a", "test."]):
        words.append({"word": w, "start": t, "end": t + 0.4})
        t += 0.5
    return words


def test_long_clip_sentence_extension_succeeds():
    """90s max: sentence extension within limit succeeds."""
    words = _words_fixture()
    shorts = [{"start": 0.0, "end": 2.0}]  # raw clip
    # With max 90, extension to sentence boundary (~3.0) is allowed.
    events = cut_ops.snap_clips_to_transcript(
        shorts, words, source_duration=600, max_duration=90)
    # Should succeed without mid-word truncation.
    assert shorts[0]["end"] <= 90.0
    # Edges align to word boundaries (not mid-word).
    for w in words:
        # end should not be inside a word.
        assert not (w["start"] < shorts[0]["end"] < w["end"])


def test_short_clip_extension_blocked():
    """30s max: optional extension beyond max fails."""
    words = _words_fixture()
    # Create a clip already at 29s; sentence would extend beyond 30.
    shorts = [{"start": 0.0, "end": 29.0}]
    # Add words to extend the timeline.
    long_words = [{"word": f"w{i}", "start": float(i), "end": float(i) + 0.4}
                  for i in range(35)]
    events = cut_ops.snap_clips_to_transcript(
        shorts, long_words, source_duration=600, max_duration=30)
    assert shorts[0]["end"] <= 30.0, "must not exceed explicit max"


def test_both_callers_pass_policy():
    """AST check: orchestrator and main pass max_duration to repair."""
    import pathlib
    import clippyme.pipeline.orchestrator as _orch
    base = pathlib.Path(_orch.__file__).parent
    for fname in ("orchestrator.py", "main.py"):
        src = (base / fname).read_text(encoding="utf-8")
        tree = ast.parse(src)
        found = False
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                name = ""
                if isinstance(func, ast.Attribute):
                    name = func.attr
                elif isinstance(func, ast.Name):
                    name = func.id
                if name == "snap_clips_to_transcript":
                    kw_names = [kw.arg for kw in node.keywords]
                    if "max_duration" in kw_names:
                        found = True
        assert found, f"{fname} must pass max_duration to snap_clips_to_transcript"
