"""Phase 2 (2F): deterministic virality cross-check.

Pure-logic tests: no media, no LLM calls. ``energy_fn`` is injected so
``cross_check_scores`` never touches ffmpeg.
"""
import pytest

from clippyme.pipeline.gemini_parser import (
    SCORE_DISAGREEMENT_THRESHOLD,
    SCORE_REVIEW_RANK_PENALTY,
    compute_deterministic_features,
    cross_check_scores,
    deterministic_score,
)
from clippyme.pipeline.media_probe import compute_audio_energy_variance


def _words(n, step=0.4, q=0, excl=0):
    words = []
    for i in range(n):
        w = "word"
        if i < q:
            w = "really?"
        elif i < q + excl:
            w = "wow!"
        words.append({"word": w, "start": i * step, "end": i * step + 0.3})
    return words


def test_features_word_count_wpm_and_densities():
    words = _words(75, q=4, excl=4)
    feats = compute_deterministic_features(
        {"start": 0, "end": 30}, words)
    assert feats["word_count"] == 75
    assert feats["duration"] == 30
    assert feats["speech_rate_wpm"] == 150.0
    assert feats["question_density"] == pytest.approx(4 / 75, abs=1e-4)
    assert feats["exclamation_density"] == pytest.approx(4 / 75, abs=1e-4)
    assert feats["audio_energy_variance"] is None


def test_features_audio_variance_rounded():
    feats = compute_deterministic_features(
        {"start": 0, "end": 10}, _words(10), audio_energy_variance=53.5781371)
    assert feats["audio_energy_variance"] == 53.578


def test_deterministic_score_bounds_and_sweet_spot():
    hot = {"speech_rate_wpm": 160, "question_density": 0.05,
           "exclamation_density": 0.05, "audio_energy_variance": 30,
           "duration": 30}
    assert deterministic_score(hot) == 100  # 50+15+8+8+10+10 clamped
    flat = {"speech_rate_wpm": 60, "question_density": 0.0,
            "exclamation_density": 0.0, "audio_energy_variance": 2,
            "duration": 5}
    # 50-10-6-8 = 26, inside 1..100
    assert deterministic_score(flat) == 26
    for feats in (hot, flat, {}):
        assert 1 <= deterministic_score(feats) <= 100


def test_cross_check_flags_big_disagreement_and_downranks():
    words_a = _words(40)                      # 80 wpm, flat -> det ~= 50
    words_b = _words(75, q=4, excl=4)         # 150 wpm, hooks -> det ~= 91
    cands = [
        {"start": 0, "end": 30, "viral_score": 95},
        {"start": 40, "end": 70, "viral_score": 88},
    ]
    out = cross_check_scores(
        cands, transcript_words=words_a + [
            {**w, "start": w["start"] + 40, "end": w["end"] + 40}
            for w in words_b],
        energy_fn=lambda *a, **k: None)
    by_score = {c["viral_score"]: c for c in out}
    hot = by_score[95]
    assert hot["score_review_required"] is True
    assert hot["score_disagreement"] == 45 > SCORE_DISAGREEMENT_THRESHOLD
    assert hot["viral_score"] == 95  # LLM score never replaced
    calm = by_score[88]
    assert calm["score_review_required"] is False
    assert calm["score_disagreement"] == 3
    # 10-point ranking penalty: unflagged 88 sorts above flagged 95
    assert out[0]["viral_score"] == 88
    assert out[1]["viral_score"] == 95
    assert SCORE_REVIEW_RANK_PENALTY == 10


def test_disagreement_at_threshold_not_flagged():
    # det for 75-word/30s hooky clip is 91; LLM 116 would exceed scale, so
    # build det==70 via features and check the >25 boundary directly.
    cands = [{"start": 0, "end": 30, "viral_score": 95}]
    out = cross_check_scores(
        cands, transcript_words=_words(40),  # det == 50
        energy_fn=lambda *a, **k: None)
    # disagreement 45 > 25 -> flagged (sanity)
    assert out[0]["score_review_required"] is True
    cands2 = [{"start": 0, "end": 30, "viral_score": 75}]
    out2 = cross_check_scores(
        cands2, transcript_words=_words(40),
        energy_fn=lambda *a, **k: None)
    assert out2[0]["score_disagreement"] == 25
    assert out2[0]["score_review_required"] is False  # strictly greater


def test_cross_check_survives_failing_energy_probe():
    def boom(*a, **k):
        raise RuntimeError("no ffmpeg")
    out = cross_check_scores(
        [{"start": 0, "end": 30, "viral_score": 80}],
        transcript_words=_words(75, q=4, excl=4), energy_fn=boom)
    assert out[0]["deterministic_features"]["audio_energy_variance"] is None
    assert out[0]["deterministic_score"] == 91


def test_energy_variance_missing_file_returns_none():
    assert compute_audio_energy_variance(
        "/nonexistent/clip.mp4", 0.0, 8.0) is None
