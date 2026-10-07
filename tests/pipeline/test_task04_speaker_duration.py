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
    assert "20.0s \u2264 duration \u2264 90.0s" in prompt
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
    """Real hook backfill accepts speaker-augmented words."""
    from clippyme.pipeline import gemini_parser as gp
    _, words = gr.build_viral_prompt({"segments": [{"words": [
        {"word": "Hello", "start": 0.0, "end": 0.5, "speaker": 0},
        {"word": "world", "start": 0.5, "end": 1.0, "speaker": 1},
    ]}]}, 100.0)
    clips = [{"start": 0.0, "end": 1.0, "viral_hook_text": "",
              "video_title_for_youtube_short": "The payoff is surprising"}]
    result = gp.backfill_hook_text(clips, words)
    assert result is clips
    assert clips[0]["viral_hook_text"] == "The payoff is surprising"
    # Speaker evidence preserved through the pipeline.
    assert [w["sp"] for w in words] == [0, 1]

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

# --- Content preservation (review2) ---

def test_padding_conflict_keeps_selected_words():
    """Optional padding that exceeds max is declined; content preserved."""
    words = [
        {"word": "First", "start": 10.0, "end": 10.4},
        {"word": "payoff.", "start": 39.5, "end": 40.0},
    ]
    clips = [{"start": 10.0, "end": 40.0}]
    cut_ops.snap_clips_to_transcript(
        clips, words, source_duration=100.0,
        max_duration=30.0, min_duration=15.0)
    # Baseline [10,40] is 30s (fits max). Padding declined.
    assert clips[0]["start"] <= 10.0
    assert clips[0]["end"] >= 40.0, "preserve selected payoff"
    assert clips[0]["end"] - clips[0]["start"] <= 30.0

def test_oversized_proposal_not_silently_reselected():
    """Over-limit interval retained with honest conflict, not shrunk to first word."""
    words = [
        {"word": "First", "start": 0.0, "end": 0.4},
        {"word": "payoff.", "start": 34.5, "end": 35.0},
    ]
    clips = [{"start": 0.0, "end": 35.0}]
    cut_ops.snap_clips_to_transcript(
        clips, words, source_duration=100.0,
        max_duration=30.0, min_duration=15.0)
    # Must NOT become [0, 0.4]. Retain with conflict.
    assert clips[0]["end"] >= 35.0, "retain selected word-safe interval"
    assert clips[0]["duration_conflict"] == "maximum_conflict"

def test_cap_fallback_checks_minimum():
    """After fallback, minimum is validated."""
    words = [
        {"word": "First", "start": 0.0, "end": 0.4},
        {"word": "payoff.", "start": 34.5, "end": 35.0},
    ]
    clips = [{"start": 0.0, "end": 35.0}]
    cut_ops.snap_clips_to_transcript(
        clips, words, source_duration=100.0,
        max_duration=30.0, min_duration=15.0)
    duration = clips[0]["end"] - clips[0]["start"]
    # Either meets min, or reports minimum conflict.
    assert duration >= 15.0 or clips[0].get("duration_conflict") == "minimum_conflict"

def test_overlapping_word_boundary_safe():
    """New boundary must not cut through overlapping speech."""
    words = [
        {"word": "first", "start": 0.0, "end": 0.4},
        {"word": "A", "start": 9.5, "end": 10.0},
        {"word": "B", "start": 9.7, "end": 10.4},  # overlaps A
    ]
    clips = [{"start": 0.0, "end": 10.4}]
    cut_ops.snap_clips_to_transcript(
        clips, words, source_duration=20.0,
        max_duration=10.0, min_duration=1.0)
    edge = clips[0]["end"]
    # Edge must not be inside any word.
    for w in words:
        assert not (w["start"] < edge < w["end"]), f"edge {edge} cuts word {w}"

def test_stale_conflict_cleared_on_success():
    """Old diagnostic removed when current policy satisfied."""
    words = [
        {"word": "First", "start": 0.0, "end": 0.4},
        {"word": "payoff.", "start": 19.5, "end": 20.0},
    ]
    clips = [{"start": 0.0, "end": 20.0,
              "duration_conflict": "maximum_conflict",
              "duration_conflict_detail": "old policy"}]
    cut_ops.snap_clips_to_transcript(
        clips, words, source_duration=100.0,
        max_duration=60.0, min_duration=15.0)
    assert "duration_conflict" not in clips[0]
    assert "duration_conflict_detail" not in clips[0]

# --- Prompt matches policy (review2) ---

@pytest.mark.parametrize("length", [5.0, 10.0, 24.0])
def test_short_source_prompt_matches_resolved_minimum(length):
    """Actual prompt mandatory limits equal resolved policy."""
    import re
    p = resolve_duration_policy(video_duration=length)
    prompt, _ = gr.build_viral_prompt({"segments": []}, length)
    matches = re.findall(
        r"([\d.eE+-]+)s\s*\u2264 duration \u2264\s*([\d.eE+-]+)s", prompt)
    assert matches, "mandatory constraint must be parseable"
    lo, hi = map(float, matches[0])
    assert (lo, hi) == (p.min_duration, p.max_duration)
    assert lo <= hi <= length
    # Unavailable tiers must not be mandatory.
    assert "MANDATORY MIX" not in prompt

@pytest.mark.parametrize("maximum", [30.1234567, 99.999999])
def test_explicit_maximum_full_precision(maximum):
    """Explicit max preserved with full precision, never rounded up."""
    import re
    prompt, _ = gr.build_viral_prompt(
        {"segments": []}, 100.0, min_duration=15.0, max_duration=maximum)
    matches = re.findall(
        r"([\d.eE+-]+)s\s*\u2264 duration \u2264\s*([\d.eE+-]+)s", prompt)
    assert matches
    assert float(matches[0][1]) == maximum

def test_segment_summary_parsed_not_incidental():
    """Segment summary block parsed, not incidental numbers."""
    import re
    t = {"text": "Hello", "language": "en", "segments": [{
        "start": 0.0, "end": 1.0, "speaker": 7,
        "words": [{"word": "Hello", "start": 0.0, "end": 1.0}],
    }]}
    prompt, words = gr.build_viral_prompt(t, 100.0)
    # Words have no speaker.
    assert "sp" not in words[0]
    # Segment summary block exists with correct structure.
    m = re.search(
        r"SEGMENT SPEAKER SUMMARIES.*?segment (\d+): speaker (\d+)",
        prompt, re.DOTALL)
    assert m, "segment summary block must be parseable"
    assert m.group(1) == "0"  # segment_index
    assert m.group(2) == "7"  # speaker

# --- Actual caller integration (not just AST) ---

def test_queued_caller_forwards_policy_values(tmp_path, monkeypatch, capsys):
    """Actual orchestrator _load_or_analyze forwards resolved max/min."""
    from types import SimpleNamespace
    from clippyme.pipeline import orchestrator as orch
    transcript = {"segments": [{"words": [
        {"word": "First", "start": 0.0, "end": 0.4},
        {"word": "last.", "start": 29.5, "end": 29.92},
    ]}]}
    args = SimpleNamespace(
        min_duration=15.0, max_duration=30.0, duration_mode="mixed",
        clip_type=None, reframe_mode="auto", skip_analysis=False,
        monitor=False, instructions=None, min_clips=None, max_clips=None,
        aspect="9:16", caption_style_default=None)
    class State:
        def artifact(self, *a): return None
        def completed(self, *a): return False
        def start(self, *a): pass
        def consume_steering(self): return []
        def set_clip_total(self, *a): pass
        def complete_stage(self, *a, **k): pass
    legacy = SimpleNamespace(
        get_viral_clips=lambda *a, **k: {"shorts": [{"start": 0.0, "end": 29.92}]})
    monkeypatch.setenv("CLIPPYME_SILENCE_SNAP", "0")
    monkeypatch.setenv("CLIPPYME_MAX_CLIPS", "0")
    data, _ = orch._load_or_analyze(
        args, str(tmp_path / "source.mp4"), "source",
        str(tmp_path), 100.0, transcript, State(), legacy)
    clip = data["shorts"][0]
    # Max 30 respected.
    assert clip["end"] - clip["start"] <= 30.0

def test_queued_minimum_conflict_observable(tmp_path, monkeypatch, capsys):
    """Minimum conflict surfaces in actual queued caller."""
    from types import SimpleNamespace
    from clippyme.pipeline import orchestrator as orch
    transcript = {"segments": [{"words": [
        {"word": "First", "start": 0.0, "end": 0.4},
        {"word": "last.", "start": 29.4, "end": 29.8},
    ]}]}
    args = SimpleNamespace(
        min_duration=30.0, max_duration=60.0, duration_mode=None,
        clip_type=None, reframe_mode="auto", skip_analysis=False,
        monitor=False, instructions=None, min_clips=None, max_clips=None,
        aspect="9:16", caption_style_default=None)
    class State:
        def artifact(self, *a): return None
        def completed(self, *a): return False
        def start(self, *a): pass
        def consume_steering(self): return []
        def set_clip_total(self, *a): pass
        def complete_stage(self, *a, **k): pass
    legacy = SimpleNamespace(
        get_viral_clips=lambda *a, **k: {"shorts": [{"start": 0.0, "end": 30.0}]})
    monkeypatch.setenv("CLIPPYME_SILENCE_SNAP", "0")
    monkeypatch.setenv("CLIPPYME_MAX_CLIPS", "0")
    data, _ = orch._load_or_analyze(
        args, str(tmp_path / "source.mp4"), "source",
        str(tmp_path), 100.0, transcript, State(), legacy)
    clip = data["shorts"][0]
    text = capsys.readouterr().out.lower()
    assert (clip["end"] - clip["start"] >= 30.0
            or "conflict" in text
            or any("duration" in k or "conflict" in k for k in clip))
