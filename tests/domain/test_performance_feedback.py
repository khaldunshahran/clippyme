"""Task 03: honest performance feedback.

No invented multipliers, retention, hook archetypes, or comment-velocity
claims. Validates shapes/values, distinguishes unknown from recorded zero,
never mutates the registry. Automatic prompt is empty; explicit overrides
preserved.
"""
import copy
import json

import pytest

from clippyme.domain import performance_feedback as pf
from clippyme.domain import analytics_service as analytics


@pytest.fixture()
def registry(tmp_path, monkeypatch):
    path = tmp_path / "analytics.json"
    monkeypatch.setattr(analytics, "ANALYTICS_FILE_PATH", path)
    # performance_feedback imports load_analytics_data at module import;
    # patch the reference it holds.
    monkeypatch.setattr(pf, "load_analytics_data", analytics.load_analytics_data)
    return path


def _write(path, clips):
    path.write_text(json.dumps({"clips": clips}))


def _clip(clip_id, views=0, likes=0, shares=0, comments=0,
          retention=None, recorded=False, title=None, hook=None, duration=60):
    m = {"views": views, "likes": likes, "shares": shares,
         "comments": comments, "retention_rate": retention if retention is not None else 0.0}
    if recorded:
        m["retention_recorded"] = True
    c = {"metrics": m, "duration": duration}
    if title:
        c["title"] = title
    if hook:
        c["hook_text"] = hook
    return {clip_id: c}


def test_missing_registry_empty(registry):
    out = pf.analyze_performance_patterns()
    assert out["has_data"] is False
    assert out["story_duration_multiplier"] is None
    assert out["top_duration_tier"] is None
    assert out["avg_completion_rate"] is None
    assert out["top_hook_patterns"] == []
    assert out["recommendations"] == []


def test_empty_clips_empty(registry):
    _write(registry, {})
    out = pf.analyze_performance_patterns()
    assert out["has_data"] is False
    assert out["status"] == "no_measurements"


def test_zero_view_records_no_data(registry):
    _write(registry, _clip("a", views=0))
    out = pf.analyze_performance_patterns()
    assert out["has_data"] is False
    assert "no views" in out["reason"]


def test_one_valid_record_observations(registry):
    clips = _clip("a", views=1000, likes=50, shares=20, comments=10,
                  retention=0.65, recorded=True, title="Great clip")
    _write(registry, clips)
    out = pf.analyze_performance_patterns()
    assert out["has_data"] is True
    assert out["sample_size"] == 1
    # No invented causal fields.
    assert out["story_duration_multiplier"] is None
    assert out["top_hook_patterns"] == []
    # Honest observations with metric/count/denominator.
    assert out["totals"]["views"] == 1000
    assert out["totals"]["shares"] == 20
    assert out["retention"]["average_recorded"] == 0.65
    assert out["retention"]["recorded_count"] == 1
    assert out["top_titles"] == ["Great clip"]
    obs_metrics = {o["metric"] for o in out["observations"]}
    assert "views" in obs_metrics and "shares" in obs_metrics


def test_unknown_retention_not_counted_as_zero(registry):
    # retention 0.0 without marker = unknown, not a recorded zero.
    clips = {}
    clips.update(_clip("a", views=100, retention=0.0, recorded=False))
    clips.update(_clip("b", views=200, retention=0.5, recorded=True))
    _write(registry, clips)
    out = pf.analyze_performance_patterns()
    # Only the recorded 0.5 counts.
    assert out["retention"]["average_recorded"] == 0.5
    assert out["retention"]["recorded_count"] == 1


def test_explicit_zero_retention_recorded(registry):
    clips = _clip("a", views=100, retention=0.0, recorded=True)
    _write(registry, clips)
    out = pf.analyze_performance_patterns()
    assert out["retention"]["average_recorded"] == 0.0
    assert out["retention"]["recorded_count"] == 1


def test_invalid_records_isolated(registry):
    valid = _clip("good", views=500, likes=25)
    bad = {
        "bad1": {"metrics": {"views": float("nan"), "likes": 1}},
        "bad2": {"metrics": {"views": -5, "likes": 1}},
        "bad3": {"metrics": {"views": True, "likes": 1}},  # bool rejected
        "bad4": {"metrics": {"views": 10, "retention_rate": 1.5}},  # >1
        "bad5": "not a dict",
        "bad6": {"metrics": "not a dict"},
    }
    clips = {}
    clips.update(valid)
    clips.update(bad)
    _write(registry, clips)
    before = json.loads(registry.read_text())
    out = pf.analyze_performance_patterns()
    assert out["has_data"] is True
    assert out["sample_size"] == 1
    assert out["totals"]["views"] == 500
    # No mutation.
    after = json.loads(registry.read_text())
    assert before == after


def test_mixed_valid_invalid(registry):
    clips = {}
    clips.update(_clip("v1", views=300, shares=15))
    clips.update(_clip("v2", views=700, shares=35))
    clips["bad"] = {"metrics": {"views": float("inf")}}
    _write(registry, clips)
    out = pf.analyze_performance_patterns()
    assert out["sample_size"] == 2
    assert out["totals"]["views"] == 1000
    assert out["totals"]["shares"] == 50


def test_short_outperforming_long_no_duration_winner(registry):
    # Short clips beat long ones; still no duration multiplier invented.
    clips = {}
    clips.update(_clip("short1", views=10000, shares=500, duration=30))
    clips.update(_clip("long1", views=100, shares=2, duration=150))
    _write(registry, clips)
    out = pf.analyze_performance_patterns()
    assert out["has_data"] is True
    assert out["story_duration_multiplier"] is None
    assert out["top_duration_tier"] is None


def test_auto_prompt_empty(registry):
    _write(registry, _clip("a", views=5000, shares=200, retention=0.8,
                           recorded=True, title="Viral", hook="Wow"))
    assert pf.get_learned_patterns_prompt() == ""


def test_auto_prompt_empty_no_data(registry):
    assert pf.get_learned_patterns_prompt() == ""


def test_no_registry_mutation(registry):
    clips = _clip("a", views=100, likes=10)
    _write(registry, clips)
    before = registry.read_text()
    pf.analyze_performance_patterns()
    pf.get_learned_patterns_prompt()
    assert registry.read_text() == before


# --- build_prompt / rescore consumer tests ---

def test_build_viral_prompt_no_empirical_block(registry):
    """Real build_viral_prompt path: no WHOLE-STORY block without override."""
    _write(registry, _clip("a", views=9000, shares=400, retention=0.85,
                           recorded=True))
    from clippyme.pipeline import gemini_request as gr
    prompt, _words = gr.build_viral_prompt(
        transcript_result={"segments": []},
        video_duration=300.0,
        learned_patterns=None,
    )
    assert "WHOLE-STORY ARCS DOMINATE" not in prompt
    assert "2.8x" not in prompt
    assert "LEARNED AUDIENCE PERFORMANCE PATTERNS" not in prompt


def test_build_viral_prompt_explicit_override_preserved(registry):
    from clippyme.pipeline import gemini_request as gr
    prompt, _words = gr.build_viral_prompt(
        transcript_result={"segments": []},
        video_duration=300.0,
        learned_patterns="CUSTOM OVERRIDE BLOCK",
    )
    assert "CUSTOM OVERRIDE BLOCK" in prompt


def test_rescore_no_empirical_block(registry, monkeypatch, tmp_path):
    """Actual rescore_job with injected SDK: no fabricated block, updates score/hook."""
    import importlib
    import sys
    import types
    import json as _json
    _write(registry, _clip("a", views=9000, shares=400))
    # Inject fake google.genai (no installed SDK needed).
    fake_genai = types.ModuleType("google.genai")
    fake_genai.Client = lambda **kwargs: object()
    monkeypatch.setitem(sys.modules, "google.genai", fake_genai)
    import google
    monkeypatch.setattr(google, "genai", fake_genai, raising=False)
    rs = importlib.import_module("clippyme.domain.rescore_service")
    metadata = {"video_title": "Fixture",
                "shorts": [{"start": 0, "end": 10}],
                "transcript": {"segments": [{"start": 0, "end": 10, "text": "Fixture words."}]}}
    monkeypatch.setattr(rs, "load_job_metadata",
                        lambda *a: (str(tmp_path / "metadata.json"), metadata))
    monkeypatch.setattr(rs, "load_persistent_config", lambda: {})
    captured = []
    def generate(client, prompt, *a, **k):
        captured.append(prompt)
        response = {"clips": [{"clip_index": 1, "viral_score": 91,
                               "viral_reason": "Fixture",
                               "video_title_for_youtube_short": "New title",
                               "hook": "New hook"}]}
        return types.SimpleNamespace(text=_json.dumps(response)), "injected-model"
    monkeypatch.setattr(rs, "generate_with_model_fallback", generate)
    saved = []
    monkeypatch.setattr(rs, "save_job_metadata", lambda *a: saved.append(a))
    fake_metadata = types.ModuleType("clippyme.domain.metadata_generator")
    fake_metadata.generate_all_clips_metadata = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, fake_metadata.__name__, fake_metadata)
    result = rs.rescore_job("fixture", output_dir=str(tmp_path),
                            api_key="injected-unused-key")
    assert len(captured) == 1
    assert "Fixture words." in captured[0]
    for fabricated in ("LEARNED AUDIENCE PERFORMANCE PATTERNS",
                       "WHOLE-STORY ARCS DOMINATE", "2.8x", "3.2x"):
        assert fabricated not in captured[0]
    assert result["success"] is True
    assert result["clips"][0]["viral_score"] == 91
    assert saved and metadata["shorts"][0]["viral_hook_text"] == "New hook"





def test_invalid_retention_preserves_value_and_marker(registry):
    """Invalid update preserves BOTH value and marker (not just marker)."""
    _write(registry, {"id": {"metrics": {"views": 100, "retention_rate": 0.65,
                                         "retention_recorded": True}}})
    # Invalid 9.9 must not overwrite.
    analytics.update_clip_metrics("id", {"retention_rate": 9.9})
    stored = analytics.load_analytics_data()["clips"]["id"]["metrics"]
    assert stored["retention_rate"] == 0.65, f"value overwritten: {stored}"
    assert stored["retention_recorded"] is True
    # Boolean True must not become 1.0.
    analytics.update_clip_metrics("id", {"retention_rate": True})
    stored = analytics.load_analytics_data()["clips"]["id"]["metrics"]
    assert stored["retention_rate"] == 0.65
    assert stored["retention_recorded"] is True


def test_views_only_update_preserves_retention(registry):
    """A views-only sync never manufactures or overwrites retention."""
    _write(registry, {"id": {"metrics": {"views": 50, "retention_rate": 0.65,
                                         "retention_recorded": True}}})
    analytics.update_clip_metrics("id", {"views": 200, "likes": 10})
    stored = analytics.load_analytics_data()["clips"]["id"]["metrics"]
    assert stored["views"] == 200
    assert stored["retention_rate"] == 0.65
    assert stored["retention_recorded"] is True


def test_sync_without_retention_preserves(registry, monkeypatch):
    """Zernio sync without retentionRate preserves prior value/marker."""
    import sys, types
    _write(registry, {"id": {"metrics": {"views": 50, "retention_rate": 0.65,
                                         "retention_recorded": True},
                             "post_id": "p1"}})
    fake_module = types.ModuleType("clippyme.integrations.social_publisher")
    class FakeClient:
        def __init__(self, key): pass
        def _request(self, *a, **k):
            return {"analytics": {"views": 200, "likes": 10}}  # no retention
    fake_module.ZernioClient = FakeClient
    fake_module.ZernioError = type("ZernioError", (Exception,), {})
    monkeypatch.setitem(sys.modules, fake_module.__name__, fake_module)
    from clippyme.storage import config_store
    monkeypatch.setattr(config_store, "load_zernio_config", lambda: {})
    analytics.sync_zernio_analytics(api_key="x")
    stored = analytics.load_analytics_data()["clips"]["id"]["metrics"]
    assert stored["views"] == 200
    assert stored["retention_rate"] == 0.65
    assert stored["retention_recorded"] is True


def test_republish_preserves_marker(registry):
    """Republish preserves retention value and marker."""
    analytics.record_published_clip(
        job_id="j9", clip_index=0, clip_info={}, publish_result={}, platforms=[])
    analytics.update_clip_metrics("j9:0", {"retention_rate": 0.0})  # explicit zero
    # Republish same clip.
    analytics.record_published_clip(
        job_id="j9", clip_index=0, clip_info={}, publish_result={}, platforms=[])
    stored = analytics.load_analytics_data()["clips"]["j9:0"]["metrics"]
    assert stored["retention_rate"] == 0.0
    assert stored["retention_recorded"] is True


def test_oversized_int_isolated(registry):
    """Oversized JSON int doesn't kill valid neighbors."""
    _write(registry, {
        "good": {"metrics": {"views": 100, "retention_rate": 0.5,
                             "retention_recorded": True}},
        "bad": {"metrics": {"views": 10**400, "likes": 1}},
    })
    out = pf.analyze_performance_patterns()
    assert out["has_data"] is True
    assert out["sample_size"] == 1
    assert out["totals"]["views"] == 100
    assert out["retention"]["average_recorded"] == 0.5

# --- analytics_service presence marker ---

def test_retention_marker_only_on_explicit_valid(registry):
    analytics.record_published_clip(
        job_id="j1", clip_index=0, clip_info={"video_title_for_youtube_short": "T"},
        publish_result={"post_id": "p1"}, platforms=["tiktok"])
    clip_id = "j1:0"  # clip_id = f"{job_id}:{clip_index}"
    # Update without retention: no marker.
    analytics.update_clip_metrics(clip_id, {"views": 100})
    data = analytics.load_analytics_data()
    assert data["clips"][clip_id]["metrics"].get("retention_recorded") is not True
    # Explicit valid zero: marker set.
    analytics.update_clip_metrics(clip_id, {"retention_rate": 0.0})
    data = analytics.load_analytics_data()
    assert data["clips"][clip_id]["metrics"]["retention_recorded"] is True
    # Invalid value: marker unchanged (still True from before), no crash.
    analytics.update_clip_metrics(clip_id, {"retention_rate": 9.9})
    data = analytics.load_analytics_data()
    assert data["clips"][clip_id]["metrics"]["retention_recorded"] is True


def test_legacy_file_without_marker_usable(registry):
    # Legacy file: no marker, retention 0.0 -> treated as unknown.
    _write(registry, _clip("legacy", views=50, retention=0.0, recorded=False))
    out = pf.analyze_performance_patterns()
    assert out["has_data"] is True
    assert out["retention"]["recorded_count"] == 0
    assert out["retention"]["average_recorded"] is None
