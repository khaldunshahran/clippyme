"""Phase 3D: A/B title testing mechanism (default OFF).

Covers deterministic assignment, epsilon-greedy exploitation, the
default-off path, and analytics attribution. No network, no LLM.
"""
import pytest

from clippyme.domain import title_ab_testing as ab


def _variants():
    return ["main title", "alt one", "alt two"]


# --- enablement --------------------------------------------------------------

def test_disabled_by_default(monkeypatch):
    monkeypatch.delenv("CLIPPYME_AB_TITLES", raising=False)
    assert ab.ab_testing_enabled() is False


def test_env_enables(monkeypatch):
    monkeypatch.setenv("CLIPPYME_AB_TITLES", "1")
    assert ab.ab_testing_enabled() is True


# --- assignment --------------------------------------------------------------

def test_assign_deterministic():
    v = _variants()
    assert ab.assign_variant(v, "post-1") == ab.assign_variant(v, "post-1")


def test_assign_index_in_range():
    v = _variants()
    for i in range(50):
        idx = ab.assign_variant(v, f"key-{i}")
        assert 0 <= idx < len(v)


def test_round_robin_covers_all_variants():
    v = _variants()
    seen = {ab.assign_variant(v, f"rr-{i}", method="round_robin") for i in range(300)}
    assert seen == {0, 1, 2}


def test_single_variant_always_zero():
    assert ab.assign_variant(["only"], "anything") == 0


def test_epsilon_greedy_exploits_best():
    v = _variants()
    perf = {"0": {"impressions": 100, "ctr": 0.01},
            "1": {"impressions": 100, "ctr": 0.09},   # best
            "2": {"impressions": 100, "ctr": 0.02}}
    assert ab.epsilon_greedy_assignment(v, "k", performance=perf, epsilon=0.0) == 1


def test_epsilon_greedy_falls_back_to_hash_without_performance():
    v = _variants()
    a = ab.epsilon_greedy_assignment(v, "k9", performance=None, epsilon=0.0)
    b = ab.assign_variant(v, "k9", method="round_robin")
    assert a == b


# --- resolution --------------------------------------------------------------

def test_default_off_returns_variant_a(monkeypatch):
    monkeypatch.delenv("CLIPPYME_AB_TITLES", raising=False)
    clip = {"title_variants": _variants()}
    res = ab.resolve_title_for_post(clip, "post-x")
    assert res["variant"] == "A"
    assert res["variant_index"] == 0
    assert res["title"] == "main title"
    assert res["enabled"] is False


def test_explicit_enabled_resolves_deterministically(monkeypatch):
    monkeypatch.delenv("CLIPPYME_AB_TITLES", raising=False)
    clip = {"title_variants": _variants()}
    r1 = ab.resolve_title_for_post(clip, "post-x", enabled=True)
    r2 = ab.resolve_title_for_post(clip, "post-x", enabled=True)
    assert r1 == r2
    assert r1["enabled"] is True
    assert r1["title"] == _variants()[r1["variant_index"]]


def test_clip_without_variants_resolves_a():
    res = ab.resolve_title_for_post({"title": "solo"}, "k", enabled=True)
    assert res["variant"] == "A" and res["title"] == "solo"


def test_extract_variants():
    assert ab.extract_title_variants({"title_variants": _variants()}) == _variants()
    assert ab.extract_title_variants({"title": "solo"}) == ["solo"]
    assert ab.extract_title_variants({}) == []


# --- attribution -------------------------------------------------------------

def _make_record(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from clippyme.domain.analytics_service import record_published_clip
    clip_info = {"clip_id": "c0", "title": "main title",
                 "title_variants": _variants()}
    result = {"post_id": "post-123", "published_at": "2026-09-24T00:00:00Z"}
    record_published_clip("job9", 0, clip_info, result, ["tiktok"])


def test_attribution_written(tmp_path, monkeypatch):
    _make_record(tmp_path, monkeypatch)
    from clippyme.domain.analytics_service import (
        load_analytics_data as load_analytics)
    assignment = {"variant": "B", "variant_index": 1, "title": "alt one",
                  "method": "round_robin"}
    ok = ab.record_title_variant_assignment(
        "job9", 0, "post-123", assignment, platform="tiktok")
    assert ok is True
    data = load_analytics()
    rec = data["clips"]["job9:0"]
    assert rec["title_variant_shipped"] == "B"
    assert rec["title_variant_index"] == 1
    assert rec["title_variant_post_id"] == "post-123"
    assert rec["title_variant_assignments"]["post-123"]["platform"] == "tiktok"


def test_attribution_missing_record_returns_false(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ok = ab.record_title_variant_assignment(
        "nope", 0, "post-1", {"variant": "A", "variant_index": 0})
    assert ok is False


def test_variant_summary(tmp_path, monkeypatch):
    _make_record(tmp_path, monkeypatch)
    from clippyme.domain.analytics_service import (
        load_analytics_data as load_analytics,
        save_analytics_data as save_analytics)
    data = load_analytics()
    rec = data["clips"]["job9:0"]
    rec["title_variant_shipped"] = "A"
    rec["metrics"] = {"views": 1000, "likes": 100, "shares": 10,
                      "comments": 5, "retention_rate": 0.8,
                      "engagement_rate": 0.1}
    save_analytics(data)
    summary = ab.variant_performance_summary()
    assert "A" in summary
    assert summary["A"]["impressions"] == 1000
    assert summary["A"]["ctr"] == pytest.approx(0.1)
