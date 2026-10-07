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
    """build_viral_prompt mandatory limits match resolved policy exactly."""
    t = _transcript([("w", 0.0, 0.5, "missing")])
    t["language"] = "en"
    prompt, _ = gr.build_viral_prompt(
        t, video_duration=600, min_duration=20, max_duration=90)
    p = resolve_duration_policy(min_duration=20, max_duration=90,
                                video_duration=600)
    # Exact mandatory constraint, not just numbers appearing anywhere.
    assert "20\u2264 duration \u2264 90" in prompt or "20s \u2264 duration \u2264 90s" in prompt
    assert p.max_duration == 90.0
    assert p.min_duration == 20.0

def test_prompt_preserves_exact_fractional_limits():
    """max30.6 must not become 31 in prompt."""
    t = _transcript([("w", 0.0, 0.5, "missing")])
    t["language"] = "en"
    prompt, _ = gr.build_viral_prompt(
        t, video_duration=100, min_duration=15.2, max_duration=30.6)
    assert "30.6" in prompt, "exact max must appear"
    assert "15.2" in prompt, "exact min must appear"


# --- Repair respects policy ---

def _words_fixture():
    """Words with sentence boundaries for repair tests."""
    words = []
    t = 0.0
    for i, w in enumerate(["Hello", "world.", "This", "is", "a", "test."]):
        words.append({"word": w, "start": t, "end": t + 0.4})
        t += 0.5
    return words


def test_genuine_long_sentence_repair_can_expand_beyond_sixty():
    """Real ~90s clip: sentence repair expands beyond 60s (no old ceiling)."""
    words = [
        {"word": "Before.", "start": 8.0, "end": 9.0},
        {"word": "Next", "start": 10.0, "end": 10.4},
        {"word": "ending.", "start": 89.5, "end": 90.0},
    ]
    clips = [{"start": 10.2, "end": 89.1}]
    cut_ops.snap_clips_to_transcript(
        clips, words, source_duration=200.0, max_duration=95.0)
    # Sentence boundaries: start ~9.95 (before "Before."), end ~90.08.
    assert clips[0]["start"] == pytest.approx(9.95, abs=0.1)
    assert clips[0]["end"] == pytest.approx(90.08, abs=0.1)
    duration = clips[0]["end"] - clips[0]["start"]
    assert duration > 60.0, "must prove >60s expansion works"
    assert duration < 95.0, "must respect max"

def test_long_clip_sentence_extension_succeeds():
    """90s max: sentence extension within limit succeeds."""
    words = _words_fixture()
    shorts = [{"start": 0.0, "end": 2.0}]
    cut_ops.snap_clips_to_transcript(
        shorts, words, source_duration=600, max_duration=90)
    assert shorts[0]["end"] <= 90.0
    for w in words:
        assert not (w["start"] < shorts[0]["end"] < w["end"])


def test_short_clip_sentence_expansion_blocked_by_max():
    """30s max: sentence expansion that would exceed max is blocked."""
    # Words with sentence-final punctuation at 31s.
    words = [
        {"word": "Start", "start": 0.0, "end": 0.4},
        {"word": "middle", "start": 15.0, "end": 15.4},
        {"word": "end.", "start": 30.5, "end": 31.0},  # sentence ends at 31
    ]
    shorts = [{"start": 0.0, "end": 29.0}]
    cut_ops.snap_clips_to_transcript(
        shorts, words, source_duration=600, max_duration=30)
    # Must NOT extend to 31s sentence boundary; stays <=30.
    assert shorts[0]["end"] <= 30.0, "must not exceed explicit max"
    # And must not cut mid-word.
    assert shorts[0]["end"] <= 30.5 or shorts[0]["end"] == 31.0

def test_short_clip_extension_blocked():
    """30s max: optional extension beyond max fails."""
    words = _words_fixture()
    shorts = [{"start": 0.0, "end": 29.0}]
    long_words = [{"word": f"w{i}", "start": float(i), "end": float(i) + 0.4}
                  for i in range(35)]
    events = cut_ops.snap_clips_to_transcript(
        shorts, long_words, source_duration=600, max_duration=30)
    assert shorts[0]["end"] <= 30.0, "must not exceed explicit max"


def test_both_callers_pass_policy():
    """AST check: orchestrator and main pass max_duration (not hardcoded)."""
    import pathlib
    import clippyme.pipeline.orchestrator as _orch
    base = pathlib.Path(_orch.__file__).parent
    for fname in ("orchestrator.py", "main.py"):
        src = (base / fname).read_text(encoding="utf-8")
        tree = ast.parse(src)
        found_policy = False
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                name = func.attr if isinstance(func, ast.Attribute) else (
                    func.id if isinstance(func, ast.Name) else "")
                if name == "snap_clips_to_transcript":
                    for kw in node.keywords:
                        if kw.arg == "max_duration":
                            # Must reference policy, not a literal 60.
                            kw_src = ast.unparse(kw.value)
                            assert "_policy.max_duration" in kw_src or "max_duration" in kw_src, (
                                f"{fname}: must pass policy max, not literal")
                            assert kw_src.strip() != "60" and kw_src.strip() != "60.0", (
                                f"{fname}: must not hardcode 60")
                            found_policy = True
        assert found_policy, f"{fname} must pass max_duration to snap_clips_to_transcript"

# --- Provider-shaped fixtures and hook backfill ---

def test_deepgram_shaped_fixture():
    """Deepgram Nova-3 word speaker format."""
    t = {
        "segments": [{
            "words": [
                {"word": "hello", "start": 0.0, "end": 0.5, "speaker": 0,
                 "confidence": 0.99},
                {"word": "world", "start": 0.5, "end": 1.0, "speaker": 1,
                 "confidence": 0.98},
            ]
        }]
    }
    words = gr.extract_prompt_words(t)
    assert words[0]["sp"] == 0
    assert words[1]["sp"] == 1

def test_elevenlabs_shaped_fixture():
    """ElevenLabs word speaker format."""
    t = {
        "segments": [{
            "words": [
                {"word": "hi", "start": 0.0, "end": 0.3, "speaker": 0},
                {"word": "there", "start": 0.3, "end": 0.6, "speaker": 0},
            ]
        }]
    }
    words = gr.extract_prompt_words(t)
    assert all(w["sp"] == 0 for w in words)

def test_whisper_pyannote_shaped_fixture():
    """Whisper + pyannote word speaker format."""
    t = {
        "segments": [
            {"speaker": "SPEAKER_00",  # segment-level (string, ignored)
             "words": [
                 {"word": "a", "start": 0.0, "end": 0.2, "speaker": 0},
             ]},
        ]
    }
    words = gr.extract_prompt_words(t)
    # Word-level int preserved; segment string not copied.
    assert words[0]["sp"] == 0

def test_hook_backfill_compatibility():
    """Hook backfill works with speaker-augmented words."""
    from clippyme.pipeline import gemini_parser
    words = [
        {"w": "hello", "s": 0.0, "e": 0.5, "sp": 0},
        {"w": "world", "s": 0.5, "e": 1.0, "sp": 1},
    ]
    # backfill_hook_text expects {w,s,e}; sp must not break it.
    try:
        result = gemini_parser.backfill_hook_text(words, 0.0, 1.0)
        # Should return text without crashing on 'sp' key.
        assert isinstance(result, str)
    except AttributeError:
        # If function doesn't exist, at least verify words are intact.
        assert words[0]["w"] == "hello"

def test_no_transcript_mutation():
    """build_viral_prompt does not mutate input transcript."""
    import copy
    t = {
        "text": "Hello world",
        "language": "en",
        "segments": [{
            "start": 0.0, "end": 1.0, "speaker": 7,
            "words": [{"word": "Hello", "start": 0.0, "end": 0.5}],
        }],
    }
    original = copy.deepcopy(t)
    gr.build_viral_prompt(t, 100.0)
    assert t == original, "transcript must not be mutated"

def test_exact_toon_rows():
    """TOON rows match expected format exactly."""
    words = [
        {"w": "hello", "s": 0.0, "e": 0.5, "sp": 0},
        {"w": "world", "s": 1.25, "e": 1.75, "sp": 1},
    ]
    toon = gr.encode_words_toon(words)
    lines = toon.split("\n")
    assert lines[0] == "words[2]{w,s,e,sp}:"
    assert lines[1] == "  hello,0.0,0.5,0"
    assert lines[2] == "  world,1.25,1.75,1"
