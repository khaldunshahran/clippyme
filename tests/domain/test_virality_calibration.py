"""Phase 3A: virality feedback-loop calibration scaffolding.

Pure-logic + file-IO tests against synthetic analytics data. No network, no
LLM. The overrides file is written under a tmp cwd so the real repo data dir
is never touched.
"""
import json

import pytest

from clippyme.domain import virality_calibration as vc
from clippyme.pipeline.gemini_parser import deterministic_score


def _clip(clip_id, feats, views, shares=0, comments=0, likes=0, retention=0.0):
    return {
        "clip_id": clip_id,
        "deterministic_features": dict(feats),
        "metrics": {
            "views": views, "shares": shares, "comments": comments,
            "likes": likes, "retention_rate": retention, "engagement_rate": 0.0,
        },
    }


def _synthetic_data(n=24):
    """speech_rate_wpm rises with engagement -> positive correlation expected."""
    clips = {}
    for i in range(n):
        wpm = 100 + i * 5  # 100 .. 215
        engagement = max(0, (wpm - 100)) * 10  # monotonic in wpm
        feats = {
            "word_count": 60, "duration": 30.0,
            "speech_rate_wpm": float(wpm),
            "question_density": 0.05 if i % 2 == 0 else 0.0,
            "exclamation_density": 0.0,
            "audio_energy_variance": 20.0,
        }
        clips[f"job0:{i}"] = _clip(
            f"job0:{i}", feats, views=100 + i * 50,
            likes=engagement, retention=0.7)
    return {"version": 1, "clips": clips}


@pytest.fixture(autouse=True)
def _clean_cache():
    vc.clear_override_cache()
    yield
    vc.clear_override_cache()


def test_no_data_reports_open_loop():
    report = vc.compute_feature_correlations(data={"version": 1, "clips": {}})
    assert report["has_data"] is False
    assert report["sample_size"] == 0
    assert "needs real view data" in report["reason"]


def test_insufficient_sample_reports_open_loop():
    data = _synthetic_data(n=5)
    report = vc.compute_feature_correlations(min_clips=20, data=data)
    assert report["has_data"] is False
    assert report["sample_size"] == 5
    assert "needs real view data" in report["reason"]


def test_correlation_detects_predictive_feature():
    report = vc.compute_feature_correlations(min_clips=10, data=_synthetic_data(24))
    assert report["has_data"] is True
    assert report["sample_size"] == 24
    r = report["correlations"]["speech_rate_wpm"]["vs_engagement"]
    assert r is not None and r > 0.8  # constructed to be strongly positive


def test_pearson_edge_cases():
    assert vc.pearson([1.0], [2.0]) is None
    assert vc.pearson([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]) is None  # zero variance
    assert vc.pearson([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0)
    assert vc.pearson([1.0, 2.0, 3.0], [3.0, 2.0, 1.0]) == pytest.approx(-1.0)


def test_propose_weights_without_data_is_empty():
    proposal = vc.propose_weight_adjustments({"has_data": False, "reason": "x"})
    assert proposal["has_data"] is False
    assert proposal["weight_overrides"] == {}


def test_propose_weights_shape_and_bounds():
    report = vc.compute_feature_correlations(min_clips=10, data=_synthetic_data(24))
    proposal = vc.propose_weight_adjustments(report, outcome="engagement")
    assert proposal["has_data"] is True
    ov = proposal["weight_overrides"]
    assert set(ov) == set(vc.CALIBRATION_FEATURES)
    for mult in ov.values():
        assert vc.MULTIPLIER_FLOOR <= mult <= vc.MULTIPLIER_CEIL
    # strongly predictive feature gets its weight increased
    assert ov["speech_rate_wpm"] > 1.0
    assert "PROPOSED" in proposal["status"]


def test_recalibrate_refuses_without_approval(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = vc.recalibrate_weights(data=_synthetic_data(24))
    assert result["status"] == "refused"
    assert "explicit operator approval" in result["reason"]
    assert not (tmp_path / "data" / "virality_weight_overrides.json").exists()
    # scoring untouched
    assert vc.get_active_weight_overrides() == {}


def test_recalibrate_approved_writes_and_consumes(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = vc.recalibrate_weights(
        approved=True, operator="test", min_clips=10, data=_synthetic_data(24))
    assert result["status"] == "applied"
    path = tmp_path / "data" / "virality_weight_overrides.json"
    assert path.exists()
    payload = json.loads(path.read_text())
    assert payload["approved_by"] == "test"
    assert payload["sample_size"] == 24

    active = vc.get_active_weight_overrides()
    assert active["speech_rate_wpm"] > 1.0

    # highlight-service consumption path: deterministic_score honours overrides
    feats = {"speech_rate_wpm": 160, "question_density": 0.0,
             "exclamation_density": 0.0, "audio_energy_variance": None,
             "duration": 30}
    default = deterministic_score(feats, weight_overrides={})
    tuned = deterministic_score(feats)  # loads the approved file
    assert tuned > default  # wpm delta (6) scaled up


def test_recalibrate_insufficient_data_even_when_approved(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = vc.recalibrate_weights(approved=True, operator="t",
                                    data=_synthetic_data(5))
    assert result["status"] == "insufficient_data"
    assert not (tmp_path / "data" / "virality_weight_overrides.json").exists()


def test_deterministic_score_default_unchanged_without_file():
    # Hand-checked against the REAL Phase-2 formula (named-delta refactor
    # preserves it exactly; see tests/pipeline/test_deterministic_scoring.py):
    # hot = 50 + wpm(15) + q(8) + excl(8) + energy(10) + dur(10) = 101 -> 100
    hot = {"speech_rate_wpm": 160, "question_density": 0.05,
           "exclamation_density": 0.05, "audio_energy_variance": 30,
           "duration": 30}
    assert deterministic_score(hot, weight_overrides={}) == 100
    # explicit overrides scale the wpm delta (15 -> 30); clamp still holds
    assert deterministic_score(hot, weight_overrides={"speech_rate_wpm": 2.0}) == 100
    # scaling is visible below the clamp: 50 + 15 + 10 = 75 -> 50 + 30 + 10 = 90
    mid = {"speech_rate_wpm": 160, "duration": 30}
    assert deterministic_score(mid, weight_overrides={}) == 75
    assert deterministic_score(mid, weight_overrides={"speech_rate_wpm": 2.0}) == 90
    # 50 + wpm(-10) + dur(10) = 50; override 0.5 -> 50 - 5 + 10 = 55
    mild = {"speech_rate_wpm": 100, "duration": 30}
    assert deterministic_score(mild, weight_overrides={}) == 50
    assert deterministic_score(mild, weight_overrides={"speech_rate_wpm": 0.5}) == 55
