"""Phase 2 (2B): transcript quality gate — pure helper tests.

No network, no models: assess/annotate/pick/secondary-provider plus the
cut_ops integration (probability preservation, low-confidence edge pads).
"""
import pytest

from clippyme.pipeline.transcript_quality import (
    TRANSCRIPT_QUALITY_FLOOR,
    annotate_transcript_quality,
    assess_transcript_quality,
    pick_better_transcript,
    secondary_provider_for,
)
from clippyme.pipeline.cut_ops import (
    DEFAULT_POST_PAD,
    DEFAULT_PRE_PAD,
    LOW_CONFIDENCE_POST_PAD,
    LOW_CONFIDENCE_PRE_PAD,
    flatten_words,
    snap_clip_to_words,
)


def _words(probs):
    return {"segments": [{"words": [
        {"word": f"w{i}", "start": float(i), "end": float(i) + 0.5, "probability": p}
        for i, p in enumerate(probs)
    ]}]}


def test_floor_value():
    assert TRANSCRIPT_QUALITY_FLOOR == 0.55


def test_assess_low_high():
    low = assess_transcript_quality(_words([0.3, 0.4, 0.5, 0.2]))
    assert low["quality"] == "low"
    assert low["mean_confidence"] == pytest.approx(0.35)
    high = assess_transcript_quality(_words([0.9, 0.95, 0.85]))
    assert high["quality"] == "ok"
    assert high["known_words"] == 3


def test_assess_unknown_ignores_missing_probability():
    # Words without probability must be ignored, not treated as zero.
    t = {"segments": [{"words": [
        {"word": "a", "start": 0, "end": 1, "probability": 0.9},
        {"word": "b", "start": 1, "end": 2},
        {"word": "c", "start": 2, "end": 3, "probability": 0.9},
    ]}]}
    q = assess_transcript_quality(t)
    assert q["quality"] == "ok"
    assert q["mean_confidence"] == pytest.approx(0.9)
    assert q["known_words"] == 2 and q["total_words"] == 3


def test_assess_unknown_when_no_probabilities():
    t = {"segments": [{"words": [{"word": "a", "start": 0, "end": 1}]}]}
    q = assess_transcript_quality(t)
    assert q["quality"] == "unknown"
    assert q["mean_confidence"] is None


def test_assess_empty_transcript():
    assert assess_transcript_quality({})["quality"] == "unknown"
    assert assess_transcript_quality(None)["quality"] == "unknown"


def test_annotate_adds_fields():
    t = annotate_transcript_quality(_words([0.3]))
    assert t["transcript_quality"] == "low"
    assert t["transcript_confidence"] == pytest.approx(0.3)
    assert t["transcript_confidence_words"] == 1


def test_pick_better_transcript():
    primary, secondary = {"segments": []}, {"segments": []}
    low = {"quality": "low", "mean_confidence": 0.4}
    ok = {"quality": "ok", "mean_confidence": 0.8}
    unk = {"quality": "unknown", "mean_confidence": None}
    # Better result wins regardless of order.
    assert pick_better_transcript(primary, low, secondary, ok)[0] is secondary
    assert pick_better_transcript(primary, ok, secondary, low)[0] is primary
    # Unknown never beats known; both unknown -> keep primary.
    assert pick_better_transcript(primary, unk, secondary, ok)[0] is secondary
    assert pick_better_transcript(primary, unk, secondary, unk)[0] is primary


def test_secondary_provider_for(monkeypatch):
    monkeypatch.delenv("DEEPGRAM_API_KEY", raising=False)
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    assert secondary_provider_for("deepgram") == "whisper"
    assert secondary_provider_for("whisper") is None
    monkeypatch.setenv("DEEPGRAM_API_KEY", "x")
    assert secondary_provider_for("elevenlabs") == "deepgram"
    # Unknown primary prefers another configured cloud provider over local.
    assert secondary_provider_for("mystery") == "deepgram"
    monkeypatch.delenv("DEEPGRAM_API_KEY")
    assert secondary_provider_for("mystery") == "whisper"


def test_flatten_words_preserves_probability_optionally():
    fw = flatten_words(_words([0.42, 0.77]))
    assert fw[0]["probability"] == pytest.approx(0.42)
    fw2 = flatten_words({"segments": [{"words": [{"word": "x", "start": 0, "end": 1}]}]})
    assert "probability" not in fw2[0]


def test_snap_widens_pads_for_low_confidence_anchors():
    words = [
        {"start": 10.0, "end": 10.4, "word": "a", "probability": 0.3},
        {"start": 10.5, "end": 11.0, "word": "b", "probability": 0.95},
    ]
    s, e = snap_clip_to_words(10.02, 10.98, words)
    assert s == pytest.approx(10.0 - LOW_CONFIDENCE_PRE_PAD)
    assert e == pytest.approx(11.0 + DEFAULT_POST_PAD)

    words_end_low = [
        {"start": 10.0, "end": 10.4, "word": "a", "probability": 0.95},
        {"start": 10.5, "end": 11.0, "word": "b", "probability": 0.2},
    ]
    s2, e2 = snap_clip_to_words(10.02, 10.98, words_end_low)
    assert s2 == pytest.approx(10.0 - DEFAULT_PRE_PAD)
    assert e2 == pytest.approx(11.0 + LOW_CONFIDENCE_POST_PAD)


def test_snap_unknown_confidence_keeps_default_pads():
    words = [
        {"start": 10.0, "end": 10.4, "word": "a"},
        {"start": 10.5, "end": 11.0, "word": "b"},
    ]
    s, e = snap_clip_to_words(10.02, 10.98, words)
    assert s == pytest.approx(10.0 - DEFAULT_PRE_PAD)
    assert e == pytest.approx(11.0 + DEFAULT_POST_PAD)


def test_snap_explicit_pad_never_shrinks():
    words = [{"start": 10.0, "end": 10.4, "word": "a", "probability": 0.2}]
    s, _ = snap_clip_to_words(10.02, 10.5, words, pre_pad=0.30)
    assert s == pytest.approx(10.0 - 0.30)
