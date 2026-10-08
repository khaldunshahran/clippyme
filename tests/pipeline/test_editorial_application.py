"""Tests for applying validated editorial plans (Task 10).

Covers the pure application contract (fail-safe base semantics,
strict identity join, continuity groups, honest funnel, rollback),
the production wiring (run_windowed_discovery applies plans exactly
once before returning shorts), the direct-CLI and queued-orchestrator
clip loops (spies on the established source-slice call), fallback
metadata handling, and mutation checks.

All provider interactions use injected deterministic fakes; no live
SDK calls.
"""
import copy
import json
import os
import re
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from clippyme.pipeline import candidate_discovery as cd
from clippyme.pipeline import editorial_application as ea
from clippyme.pipeline import editorial_refinement as er


# --- Fixtures ----------------------------------------------------------

def make_short(cid, start, end, **kw):
    d = {
        "candidate_id": cid,
        "start": float(start),
        "end": float(end),
        "viral_score": 80,
        "viral_reason": "a sufficiently long viral reason for validation",
        "viral_hook_text": f"hook for {cid}",
        "video_description": f"description for {cid}",
        "video_title_for_youtube_short": f"Title {cid}",
        "hashtags": ["#test"],
        "speaker_name": "spk",
        "event_type": "story",
        "custom_additive": "must-survive",
    }
    d.update(kw)
    return d


def make_plan(cid, verdict, start_s, end_s, fp="fp-1", status="refined",
              version="1", **kw):
    d = {
        "plan_version": version,
        "candidate_id": cid,
        "verdict": verdict,
        "status": status,
        "start_s": start_s,
        "end_s": end_s,
        "quotes": {},
        "quality": {},
        "recommendation_score": 75.0,
        "reason": "test reason",
        "continuity_links": [],
        "bridge_evidence": None,
        "conflict": None,
        "moment_fingerprint": fp,
        "continuity_group": None,
        "components": [],
    }
    d.update(kw)
    return d


def make_store_rec(cid, start_s, end_s, fp="fp-1"):
    return {"candidate_id": cid, "moment_fingerprint": fp,
            "start_s": start_s, "end_s": end_s, "event_key": "ek"}


def base_case():
    """Two shorts with a full-accept store; caller supplies plans."""
    shorts = [make_short("c1", 10.0, 30.0), make_short("c2", 40.0, 60.0)]
    store = {"c1": make_store_rec("c1", 10.0, 30.0),
             "c2": make_store_rec("c2", 40.0, 60.0)}
    return shorts, store


def apply(shorts, store, plans, **kw):
    return ea.apply_editorial_plans(shorts, store, plans,
                                    200.0, 15.0, 120.0, **kw)


# --- Rollback flag -----------------------------------------------------

def test_flag_default_on(monkeypatch):
    monkeypatch.delenv("CLIPPYME_APPLY_EDITORIAL_PLANS", raising=False)
    assert ea.application_enabled() is True


def test_flag_disable_forms(monkeypatch):
    for v in ("0", "false", "no", "off", "FALSE", "No"):
        monkeypatch.setenv("CLIPPYME_APPLY_EDITORIAL_PLANS", v)
        assert ea.application_enabled() is False


def test_flag_enable_forms(monkeypatch):
    for v in ("1", "true", "yes", "on"):
        monkeypatch.setenv("CLIPPYME_APPLY_EDITORIAL_PLANS", v)
        assert ea.application_enabled() is True


# --- Fail-safe base semantics ------------------------------------------

def test_all_verdicts_exact_lists_and_counts():
    shorts = [make_short("c1", 10.0, 30.0), make_short("c2", 40.0, 60.0),
              make_short("c3", 70.0, 90.0), make_short("c4", 100.0, 120.0)]
    store = {c: make_store_rec(c, s, e) for c, s, e in
             [("c1", 10.0, 30.0), ("c2", 40.0, 60.0),
              ("c3", 70.0, 90.0), ("c4", 100.0, 120.0)]}
    plans = [make_plan("c1", "accept", 10.0, 30.0),
             make_plan("c2", "adjust", 42.0, 62.0),
             make_plan("c3", "reject", 70.0, 90.0),
             make_plan("c4", "unrefined", 100.0, 120.0,
                       status="unrefined")]
    out, rec = apply(shorts, store, plans)
    assert [s["candidate_id"] for s in out] == ["c1", "c2", "c4"]
    assert (out[0]["start"], out[0]["end"]) == (10.0, 30.0)
    assert (out[1]["start"], out[1]["end"]) == (42.0, 62.0)
    assert (out[2]["start"], out[2]["end"]) == (100.0, 120.0)
    assert rec["accepts_retained"] == 1
    assert rec["adjustments_applied"] == 1
    assert rec["rejects_removed"] == 1
    assert rec["unrefined_retained"] == 1
    assert rec["original_selected_count"] == 4
    assert rec["final_selected_count"] == 3
    assert rec["counts_reconcile"] is True


def test_adjust_preserves_every_non_boundary_field():
    shorts, store = base_case()
    before = copy.deepcopy(shorts[1])
    plans = [make_plan("c1", "accept", 10.0, 30.0),
             make_plan("c2", "adjust", 41.5, 58.25)]
    out, _ = apply(shorts, store, plans)
    adj = [s for s in out if s["candidate_id"] == "c2"][0]
    assert (adj["start"], adj["end"]) == (41.5, 58.25)
    for k, v in before.items():
        if k in ("start", "end"):
            continue
        assert adj[k] == v, f"field dropped/changed: {k}"
    # Additive annotation is the only allowed new key.
    assert set(adj) - set(before) == {"editorial_application"}


def test_inputs_never_mutated_deep():
    shorts, store = base_case()
    plans = [make_plan("c1", "adjust", 11.0, 29.0),
             make_plan("c2", "reject", 40.0, 60.0),
             make_plan("group:c1+c2", "adjust", 11.0, 29.0,
                       components=["c1", "c2"])]
    shorts_snap = copy.deepcopy(shorts)
    store_snap = copy.deepcopy(store)
    plans_snap = copy.deepcopy(plans)
    out, _ = apply(shorts, store, plans)
    assert shorts == shorts_snap
    assert store == store_snap
    assert plans == plans_snap
    # Mutating the output must not alias the inputs.
    out[0]["hashtags"].append("#mutated")
    out[0]["editorial_application"]["verdict"] = "mutated"
    assert shorts == shorts_snap


def test_reject_records_original_interval():
    shorts, store = base_case()
    plans = [make_plan("c1", "accept", 10.0, 30.0),
             make_plan("c2", "reject", 40.0, 60.0)]
    _, rec = apply(shorts, store, plans)
    removed = [c for c in rec["interval_changes"]
               if c["action"] == "removed"]
    assert len(removed) == 1
    assert removed[0]["candidate_id"] == "c2"
    assert (removed[0]["original_start_s"],
            removed[0]["original_end_s"]) == (40.0, 60.0)
    assert removed[0]["applied_start_s"] is None


def test_no_quota_padding_when_all_rejected():
    shorts, store = base_case()
    plans = [make_plan("c1", "reject", 10.0, 30.0),
             make_plan("c2", "reject", 40.0, 60.0)]
    out, rec = apply(shorts, store, plans)
    assert out == []
    assert rec["final_selected_count"] == 0
    assert rec["counts_reconcile"] is True


# --- Strict identity join: isolation with exact reasons ----------------

def _isolation_case(plan=None, store_rec=None, shorts=None, plans=None):
    """Single-candidate isolation harness.

    Returns (output_short, record, input_snapshot).
    """
    s = shorts if shorts is not None else [make_short("cx", 10.0, 30.0)]
    st = (store_rec if store_rec is not None
          else {"cx": make_store_rec("cx", 10.0, 30.0)})
    pl = plans if plans is not None else (
        [plan] if plan is not None else [])
    snap = copy.deepcopy(s)
    out, rec = apply(s, st, pl)
    return out, rec, snap


def _assert_isolated(out, rec, snap, reason):
    assert len(out) == 1
    kept = out[0]
    assert kept["application_skipped"] == reason
    # Original short retained: every pre-existing field unchanged.
    for k, v in snap[0].items():
        assert kept[k] == v, f"field changed on isolated short: {k}"
    assert rec["application_skips"] == {reason: 1}
    assert rec["skipped"] == [{"candidate_id": "cx", "reason": reason}]
    assert rec["final_selected_count"] == 1
    assert rec["counts_reconcile"] is True


def test_skip_no_matching_plan():
    out, rec, snap = _isolation_case(plan=None)
    _assert_isolated(out, rec, snap, "no_matching_plan")


def test_skip_missing_candidate_store_entry():
    out, rec, snap = _isolation_case(
        plan=make_plan("cx", "accept", 10.0, 30.0), store_rec={})
    _assert_isolated(out, rec, snap, "missing_candidate_store_entry")


def test_skip_duplicate_candidate_id():
    s = [make_short("cx", 10.0, 30.0), make_short("cx", 10.0, 30.0)]
    st = {"cx": make_store_rec("cx", 10.0, 30.0)}
    pl = [make_plan("cx", "accept", 10.0, 30.0)]
    snap = copy.deepcopy(s)
    out, rec = apply(s, st, pl)
    assert [x.get("application_skipped") for x in out] == [
        "duplicate_candidate_id", "duplicate_candidate_id"]
    assert out[0]["start"] == snap[0]["start"]
    assert out[1]["start"] == snap[1]["start"]
    assert rec["application_skips"] == {"duplicate_candidate_id": 2}
    assert rec["counts_reconcile"] is True


def test_skip_duplicate_matching_plans():
    out, rec, snap = _isolation_case(plans=[
        make_plan("cx", "accept", 10.0, 30.0),
        make_plan("cx", "accept", 10.0, 30.0)])
    _assert_isolated(out, rec, snap, "duplicate_matching_plans")


def test_skip_moment_fingerprint_mismatch():
    out, rec, snap = _isolation_case(
        plan=make_plan("cx", "accept", 10.0, 30.0, fp="fp-other"))
    _assert_isolated(out, rec, snap, "moment_fingerprint_mismatch")


def test_skip_unknown_plan_version():
    out, rec, snap = _isolation_case(
        plan=make_plan("cx", "accept", 10.0, 30.0, version="2"))
    _assert_isolated(out, rec, snap, "unknown_plan_version")
    assert ea.APPLICATION_VERSION == er.EDITORIAL_VERSION == "1"


def test_skip_unknown_verdict():
    out, rec, snap = _isolation_case(
        plan=make_plan("cx", "maybe", 10.0, 30.0))
    _assert_isolated(out, rec, snap, "unknown_verdict")


def test_skip_unknown_status():
    out, rec, snap = _isolation_case(
        plan=make_plan("cx", "accept", 10.0, 30.0, status="pending"))
    _assert_isolated(out, rec, snap, "unknown_status")


def test_skip_contradictory_plan_state():
    # refined status with an unrefined verdict is contradictory.
    out, rec, snap = _isolation_case(
        plan=make_plan("cx", "unrefined", 10.0, 30.0, status="refined"))
    _assert_isolated(out, rec, snap, "contradictory_plan_state")
    # unrefined status with a refined verdict is contradictory too.
    out, rec, snap = _isolation_case(
        plan=make_plan("cx", "accept", 10.0, 30.0, status="unrefined"))
    _assert_isolated(out, rec, snap, "contradictory_plan_state")


def test_skip_nan_boundaries():
    out, rec, snap = _isolation_case(
        plan=make_plan("cx", "adjust", float("nan"), 30.0))
    _assert_isolated(out, rec, snap, "non_finite_plan_boundaries")


def test_skip_infinite_boundaries():
    out, rec, snap = _isolation_case(
        plan=make_plan("cx", "adjust", 10.0, float("inf")))
    _assert_isolated(out, rec, snap, "non_finite_plan_boundaries")


def test_skip_bool_boundaries():
    out, rec, snap = _isolation_case(
        plan=make_plan("cx", "adjust", True, 30.0))
    _assert_isolated(out, rec, snap, "non_finite_plan_boundaries")


def test_skip_reversed_boundaries():
    out, rec, snap = _isolation_case(
        plan=make_plan("cx", "adjust", 30.0, 10.0))
    _assert_isolated(out, rec, snap, "reversed_plan_boundaries")


def test_skip_out_of_source_boundaries():
    out, rec, snap = _isolation_case(
        plan=make_plan("cx", "adjust", 10.0, 250.0))
    _assert_isolated(out, rec, snap, "plan_boundaries_out_of_source")


def test_skip_negative_start_boundaries():
    out, rec, snap = _isolation_case(
        plan=make_plan("cx", "adjust", -5.0, 30.0))
    _assert_isolated(out, rec, snap, "plan_boundaries_out_of_source")


def test_skip_out_of_policy_boundaries():
    # 2s clip against a 15s minimum.
    out, rec, snap = _isolation_case(
        plan=make_plan("cx", "adjust", 10.0, 12.0))
    _assert_isolated(out, rec, snap, "plan_boundaries_out_of_policy")


def test_isolation_does_not_affect_sibling():
    shorts = [make_short("good", 10.0, 30.0), make_short("bad", 40.0, 60.0)]
    store = {"good": make_store_rec("good", 10.0, 30.0),
             "bad": make_store_rec("bad", 40.0, 60.0)}
    plans = [make_plan("good", "adjust", 12.0, 32.0),
             make_plan("bad", "adjust", float("nan"), 60.0)]
    out, rec = apply(shorts, store, plans)
    by_id = {s["candidate_id"]: s for s in out}
    assert (by_id["good"]["start"], by_id["good"]["end"]) == (12.0, 32.0)
    assert by_id["bad"]["application_skipped"] == "non_finite_plan_boundaries"
    assert (by_id["bad"]["start"], by_id["bad"]["end"]) == (40.0, 60.0)
    assert rec["counts_reconcile"] is True


# --- Continuity groups -------------------------------------------------

def _group_case(**kw):
    """c1 accept [10,30], c2 adjust [40,60]->[42,62], c3 accept [70,90]."""
    shorts = [make_short("c1", 10.0, 30.0), make_short("c2", 40.0, 60.0),
              make_short("c3", 70.0, 90.0)]
    store = {"c1": make_store_rec("c1", 10.0, 30.0),
             "c2": make_store_rec("c2", 40.0, 60.0),
             "c3": make_store_rec("c3", 70.0, 90.0)}
    plans = [make_plan("c1", "accept", 10.0, 30.0),
             make_plan("c2", "adjust", 42.0, 62.0),
             make_plan("c3", "accept", 70.0, 90.0)]
    group = make_plan("group:c1+c2", "adjust", 10.0, 62.0, fp="fp-1",
                      components=["c1", "c2"])
    group.update(kw.pop("group_kw", {}))
    plans.append(group)
    kw.setdefault("plans", plans)
    return apply(shorts, store, **kw)


def test_valid_group_replaces_components_once_at_earliest_position():
    out, rec = _group_case()
    assert [(s["candidate_id"], s["start"], s["end"]) for s in out] == [
        ("group:c1+c2", 10.0, 62.0), ("c3", 70.0, 90.0)]
    grp = out[0]
    assert grp["component_candidate_ids"] == ["c1", "c2"]
    assert rec["groups_proposed"] == 1
    assert rec["groups_applied"] == 1
    assert rec["groups_rejected"] == 0
    assert rec["counts_reconcile"] is True
    grouped = [c for c in rec["interval_changes"]
               if c["action"] == "grouped"]
    assert len(grouped) == 1
    assert grouped[0]["candidate_id"] == "group:c1+c2"
    assert grouped[0]["component_intervals"] == {
        "c1": [10.0, 30.0], "c2": [42.0, 62.0]}
    assert (grouped[0]["applied_start_s"],
            grouped[0]["applied_end_s"]) == (10.0, 62.0)


def test_group_copies_earliest_component_display_fields():
    out, _ = _group_case()
    grp = out[0]
    # No invented copy: hook/title/description/viral score come from c1.
    assert grp["viral_hook_text"] == "hook for c1"
    assert grp["video_title_for_youtube_short"] == "Title c1"
    assert grp["video_description"] == "description for c1"
    assert grp["viral_score"] == 80
    assert grp["custom_additive"] == "must-survive"


def test_group_keeps_editorial_recommendation_separate():
    out, _ = _group_case()
    grp = out[0]
    # The plan's editorial score is NOT relabeled as virality.
    assert grp["editorial_recommendation_score"] == 75.0
    assert grp["viral_score"] == 80  # earliest component's virality kept
    assert grp["viral_reason"] == (
        "a sufficiently long viral reason for validation")


def test_group_rejected_when_component_missing():
    out, rec = _group_case(
        group_kw={"components": ["c1", "cX"]})
    assert [s["candidate_id"] for s in out] == ["c1", "c2", "c3"]
    assert rec["groups_rejected"] == 1
    assert rec["group_rejections"][0]["reason"] == (
        "group_missing_component:cX")
    assert rec["counts_reconcile"] is True


def test_group_rejected_when_component_rejected():
    shorts = [make_short("c1", 10.0, 30.0), make_short("c2", 40.0, 60.0)]
    store = {"c1": make_store_rec("c1", 10.0, 30.0),
             "c2": make_store_rec("c2", 40.0, 60.0)}
    plans = [make_plan("c1", "accept", 10.0, 30.0),
             make_plan("c2", "reject", 40.0, 60.0),
             make_plan("group:c1+c2", "adjust", 10.0, 60.0,
                       components=["c1", "c2"])]
    out, rec = apply(shorts, store, plans)
    assert [s["candidate_id"] for s in out] == ["c1"]
    assert rec["groups_rejected"] == 1
    assert rec["group_rejections"][0]["reason"] == (
        "group_component_not_eligible:c2:rejected")


def test_group_rejected_when_component_unrefined():
    shorts = [make_short("c1", 10.0, 30.0), make_short("c2", 40.0, 60.0)]
    store = {"c1": make_store_rec("c1", 10.0, 30.0),
             "c2": make_store_rec("c2", 40.0, 60.0)}
    plans = [make_plan("c1", "accept", 10.0, 30.0),
             make_plan("c2", "unrefined", 40.0, 60.0, status="unrefined"),
             make_plan("group:c1+c2", "adjust", 10.0, 60.0,
                       components=["c1", "c2"])]
    out, rec = apply(shorts, store, plans)
    assert [s["candidate_id"] for s in out] == ["c1", "c2"]
    assert rec["group_rejections"][0]["reason"] == (
        "group_component_not_eligible:c2:unrefined")


def test_group_rejected_when_component_skipped_stale():
    shorts = [make_short("c1", 10.0, 30.0), make_short("c2", 40.0, 60.0)]
    store = {"c1": make_store_rec("c1", 10.0, 30.0),
             "c2": make_store_rec("c2", 40.0, 60.0)}
    plans = [make_plan("c1", "accept", 10.0, 30.0),
             make_plan("c2", "accept", 40.0, 60.0, fp="stale-fp"),
             make_plan("group:c1+c2", "adjust", 10.0, 60.0,
                       components=["c1", "c2"])]
    out, rec = apply(shorts, store, plans)
    assert [s["candidate_id"] for s in out] == ["c1", "c2"]
    assert rec["group_rejections"][0]["reason"] == (
        "group_component_not_eligible:c2:skipped")
    assert out[1]["application_skipped"] == "moment_fingerprint_mismatch"


def test_group_rejected_on_duplicate_components():
    out, rec = _group_case(group_kw={"components": ["c1", "c1"]})
    assert [s["candidate_id"] for s in out] == ["c1", "c2", "c3"]
    assert rec["group_rejections"][0]["reason"] == (
        "group_duplicate_components")


def test_group_rejected_on_empty_components():
    out, rec = _group_case(group_kw={"components": []})
    assert [s["candidate_id"] for s in out] == ["c1", "c2", "c3"]
    assert rec["group_rejections"][0]["reason"] == "group_empty_components"


def test_group_rejected_when_not_refined():
    out, rec = _group_case(
        group_kw={"status": "unrefined", "verdict": "unrefined"})
    assert [s["candidate_id"] for s in out] == ["c1", "c2", "c3"]
    assert rec["group_rejections"][0]["reason"] == "group_not_refined"


def test_group_rejected_when_interval_out_of_policy():
    # 200s merged interval against a 120s maximum.
    out, rec = _group_case(group_kw={"start_s": 0.0, "end_s": 200.0})
    assert [s["candidate_id"] for s in out] == ["c1", "c2", "c3"]
    assert rec["group_rejections"][0]["reason"] == (
        "group_boundaries_out_of_policy")


def test_group_rejected_when_interval_out_of_source():
    out, rec = _group_case(group_kw={"start_s": 10.0, "end_s": 250.0})
    assert [s["candidate_id"] for s in out] == ["c1", "c2", "c3"]
    assert rec["group_rejections"][0]["reason"] == (
        "group_boundaries_out_of_source")


def test_group_rejected_when_interval_does_not_cover_components():
    # c2's applied interval ends at 62; the group ends at 50.
    out, rec = _group_case(group_kw={"start_s": 10.0, "end_s": 50.0})
    assert [s["candidate_id"] for s in out] == ["c1", "c2", "c3"]
    assert rec["group_rejections"][0]["reason"] == (
        "group_interval_does_not_cover_components")


def test_group_rejected_on_reversed_boundaries():
    out, rec = _group_case(group_kw={"start_s": 62.0, "end_s": 10.0})
    assert [s["candidate_id"] for s in out] == ["c1", "c2", "c3"]
    assert rec["group_rejections"][0]["reason"] == (
        "group_reversed_boundaries")


def test_overlapping_groups_second_rejected_first_wins():
    shorts = [make_short("c1", 10.0, 30.0), make_short("c2", 40.0, 60.0),
              make_short("c3", 70.0, 90.0)]
    store = {"c1": make_store_rec("c1", 10.0, 30.0),
             "c2": make_store_rec("c2", 40.0, 60.0),
             "c3": make_store_rec("c3", 70.0, 90.0)}
    plans = [make_plan("c1", "accept", 10.0, 30.0),
             make_plan("c2", "accept", 40.0, 60.0),
             make_plan("c3", "accept", 70.0, 90.0),
             make_plan("group:c1+c2", "adjust", 10.0, 60.0,
                       components=["c1", "c2"]),
             make_plan("group:c2+c3", "adjust", 40.0, 90.0,
                       components=["c2", "c3"])]
    out, rec = apply(shorts, store, plans)
    assert [s["candidate_id"] for s in out] == ["group:c1+c2", "c3"]
    assert rec["groups_applied"] == 1
    assert rec["groups_rejected"] == 1
    assert rec["group_rejections"][0]["group_id"] == "group:c2+c3"
    assert rec["counts_reconcile"] is True


def test_duplicate_group_second_rejected_components_consumed_once():
    shorts = [make_short("c1", 10.0, 30.0), make_short("c2", 40.0, 60.0)]
    store = {"c1": make_store_rec("c1", 10.0, 30.0),
             "c2": make_store_rec("c2", 40.0, 60.0)}
    plans = [make_plan("c1", "accept", 10.0, 30.0),
             make_plan("c2", "accept", 40.0, 60.0),
             make_plan("group:c1+c2", "adjust", 10.0, 60.0,
                       components=["c1", "c2"]),
             make_plan("group:c1+c2", "adjust", 10.0, 60.0,
                       components=["c1", "c2"])]
    out, rec = apply(shorts, store, plans)
    assert [s["candidate_id"] for s in out] == ["group:c1+c2"]
    assert rec["groups_applied"] == 1
    assert rec["groups_rejected"] == 1
    assert rec["group_rejections"][0]["reason"] == (
        "group_component_not_eligible:c1:grouped")
    assert rec["counts_reconcile"] is True


def test_group_id_conflicting_with_existing_short_rejected():
    # A short already carries the group id: the group must not collide.
    shorts = [make_short("c1", 10.0, 30.0), make_short("c2", 40.0, 60.0),
              make_short("group:c1+c2", 10.0, 60.0)]
    store = {"c1": make_store_rec("c1", 10.0, 30.0),
             "c2": make_store_rec("c2", 40.0, 60.0),
             "group:c1+c2": make_store_rec("group:c1+c2", 10.0, 60.0)}
    plans = [make_plan("c1", "accept", 10.0, 30.0),
             make_plan("c2", "accept", 40.0, 60.0),
             make_plan("group:c1+c2", "adjust", 10.0, 60.0,
                       components=["c1", "c2"])]
    out, rec = apply(shorts, store, plans)
    assert [s["candidate_id"] for s in out] == [
        "c1", "c2", "group:c1+c2"]
    assert rec["groups_applied"] == 0
    assert rec["groups_rejected"] == 1
    assert rec["group_rejections"][0]["reason"] == (
        "group_id_conflicts_with_base_id")
    assert rec["counts_reconcile"] is True


def test_group_order_deterministic_across_input_orders():
    def run(order):
        shorts = [make_short("c1", 10.0, 30.0),
                  make_short("c2", 40.0, 60.0),
                  make_short("c3", 70.0, 90.0),
                  make_short("c4", 100.0, 120.0)]
        store = {f"c{i}": make_store_rec(f"c{i}", s, e) for i, (s, e) in
                 enumerate([(10.0, 30.0), (40.0, 60.0),
                            (70.0, 90.0), (100.0, 120.0)], start=1)}
        plans = [make_plan("c1", "accept", 10.0, 30.0),
                 make_plan("c2", "accept", 40.0, 60.0),
                 make_plan("c3", "accept", 70.0, 90.0),
                 make_plan("c4", "accept", 100.0, 120.0)]
        g1 = make_plan("group:c3+c4", "adjust", 70.0, 120.0,
                       components=["c3", "c4"])
        g2 = make_plan("group:c1+c2", "adjust", 10.0, 60.0,
                       components=["c1", "c2"])
        plans.extend(order(g1, g2))
        out, _ = apply(shorts, store, plans)
        return [s["candidate_id"] for s in out]
    a = run(lambda g1, g2: [g1, g2])
    b = run(lambda g1, g2: [g2, g1])
    assert a == b == ["group:c1+c2", "group:c3+c4"]


# --- Determinism / promotion / rollback / empty ------------------------

def test_deterministic_identical_inputs_identical_outputs():
    shorts = [make_short("c1", 10.0, 30.0), make_short("c2", 40.0, 60.0),
              make_short("c3", 70.0, 90.0)]
    store = {"c1": make_store_rec("c1", 10.0, 30.0),
             "c2": make_store_rec("c2", 40.0, 60.0),
             "c3": make_store_rec("c3", 70.0, 90.0)}
    plans = [make_plan("c1", "accept", 10.0, 30.0),
             make_plan("c2", "adjust", 42.0, 62.0),
             make_plan("c3", "reject", 70.0, 90.0),
             make_plan("group:c1+c2", "adjust", 10.0, 62.0,
                       components=["c1", "c2"])]
    out1, rec1 = apply(copy.deepcopy(shorts), copy.deepcopy(store),
                       copy.deepcopy(plans))
    out2, rec2 = apply(copy.deepcopy(shorts), copy.deepcopy(store),
                       copy.deepcopy(plans))
    assert (json.dumps(out1, sort_keys=True)
            == json.dumps(out2, sort_keys=True))
    assert (json.dumps(rec1, sort_keys=True)
            == json.dumps(rec2, sort_keys=True))


def test_no_promotion_of_non_selected_store_items():
    shorts = [make_short("c1", 10.0, 30.0)]
    store = {"c1": make_store_rec("c1", 10.0, 30.0),
             "cX": make_store_rec("cX", 130.0, 150.0)}
    plans = [make_plan("c1", "accept", 10.0, 30.0),
             make_plan("cX", "accept", 130.0, 150.0)]
    out, rec = apply(shorts, store, plans)
    assert [s["candidate_id"] for s in out] == ["c1"]
    assert rec["plans_ignored_not_selected"] == 1
    assert rec["plans_supplied"] == 2
    assert rec["counts_reconcile"] is True


def test_disabled_flag_preserves_shorts_byte_for_byte(monkeypatch):
    monkeypatch.setenv("CLIPPYME_APPLY_EDITORIAL_PLANS", "0")
    shorts = [make_short("c1", 10.0, 30.0), make_short("c2", 40.0, 60.0)]
    store = {"c1": make_store_rec("c1", 10.0, 30.0),
             "c2": make_store_rec("c2", 40.0, 60.0)}
    plans = [make_plan("c1", "adjust", 12.0, 32.0),
             make_plan("c2", "reject", 40.0, 60.0)]
    before = json.dumps(shorts, sort_keys=True)
    out, rec = ea.apply_editorial_plans(shorts, store, plans,
                                        200.0, 15.0, 120.0)
    assert json.dumps(out, sort_keys=True) == before
    assert rec["enabled"] is False
    assert rec["disabled_reason"] == "flag_disabled"
    assert rec["original_selected_count"] == 2
    assert rec["final_selected_count"] == 2
    assert rec["adjustments_applied"] == 0
    assert rec["rejects_removed"] == 0
    assert rec["counts_reconcile"] is True


def test_explicit_enabled_false_overrides(monkeypatch):
    monkeypatch.delenv("CLIPPYME_APPLY_EDITORIAL_PLANS", raising=False)
    shorts, store = base_case()
    plans = [make_plan("c1", "reject", 10.0, 30.0),
             make_plan("c2", "reject", 40.0, 60.0)]
    out, rec = ea.apply_editorial_plans(shorts, store, plans,
                                        200.0, 15.0, 120.0,
                                        enabled=False)
    assert len(out) == 2
    assert rec["enabled"] is False


def test_editorial_not_run_disables_application():
    shorts, store = base_case()
    plans = [make_plan("c1", "adjust", 12.0, 32.0),
             make_plan("c2", "accept", 40.0, 60.0)]
    before = copy.deepcopy(shorts)
    out, rec = ea.apply_editorial_plans(shorts, store, plans,
                                        200.0, 15.0, 120.0,
                                        editorial_ran=False)
    assert out == before
    assert rec["enabled"] is False
    assert rec["disabled_reason"] == "editorial_refinement_not_run"


def test_legacy_shorts_without_candidate_id_unchanged():
    shorts = [{"start": 5.0, "end": 25.0, "viral_score": 70,
               "viral_reason": "a sufficiently long viral reason here"},
              make_short("c1", 10.0, 30.0)]
    store = {"c1": make_store_rec("c1", 10.0, 30.0)}
    plans = [make_plan("c1", "adjust", 12.0, 32.0)]
    before = copy.deepcopy(shorts)
    out, rec = apply(shorts, store, plans)
    assert out[0] == before[0]
    assert "editorial_application" not in out[0]
    assert "application_skipped" not in out[0]
    assert rec["legacy_no_id_retained"] == 1
    assert (out[1]["start"], out[1]["end"]) == (12.0, 32.0)
    assert rec["counts_reconcile"] is True


def test_empty_stays_empty():
    out, rec = ea.apply_editorial_plans([], {}, [], 200.0, 15.0, 120.0)
    assert out == []
    assert rec["original_selected_count"] == 0
    assert rec["final_selected_count"] == 0
    assert rec["counts_reconcile"] is True
    assert rec["enabled"] is True


def test_invalid_duration_bounds_skips_everything():
    shorts, store = base_case()
    plans = [make_plan("c1", "accept", 10.0, 30.0),
             make_plan("c2", "accept", 40.0, 60.0)]
    out, rec = ea.apply_editorial_plans(shorts, store, plans,
                                        200.0, float("nan"), 120.0)
    assert [(s["start"], s["end"]) for s in out] == [
        (10.0, 30.0), (40.0, 60.0)]
    assert all(s["application_skipped"] == "invalid_duration_bounds"
               for s in out)
    assert rec["application_skips"] == {"invalid_duration_bounds": 2}
    assert rec["counts_reconcile"] is True


def test_unknown_source_duration_skips_only_source_check():
    shorts, store = base_case()
    plans = [make_plan("c1", "adjust", 12.0, 32.0),
             make_plan("c2", "accept", 40.0, 60.0)]
    out, rec = ea.apply_editorial_plans(shorts, store, plans,
                                        0.0, 15.0, 120.0)
    # Review 02: an invalid source duration cannot approve or change
    # any applicable Task 10 clip; each original short is retained.
    assert [s["application_skipped"] for s in out] == [
        "invalid_source_duration", "invalid_source_duration"]
    assert rec["application_skips"] == {"invalid_source_duration": 2}
    assert rec["accepts_retained"] == 0
    assert rec["adjustments_applied"] == 0
    assert rec["source_bounds_checked"] is False
    assert rec["counts_reconcile"] is True


# --- Funnel reconciliation ---------------------------------------------

def test_funnel_reconciliation_mixed_case():
    shorts = [make_short("c1", 10.0, 30.0), make_short("c2", 40.0, 60.0),
              make_short("c3", 70.0, 90.0), make_short("c4", 100.0, 120.0),
              {"start": 1.0, "end": 2.0}]  # legacy, no candidate_id
    store = {"c1": make_store_rec("c1", 10.0, 30.0),
             "c2": make_store_rec("c2", 40.0, 60.0),
             "c3": make_store_rec("c3", 70.0, 90.0),
             "c4": make_store_rec("c4", 100.0, 120.0)}
    plans = [make_plan("c1", "accept", 10.0, 30.0),
             make_plan("c2", "adjust", 42.0, 62.0),
             make_plan("c3", "reject", 70.0, 90.0),
             # c4: no plan -> skip
             make_plan("cX", "accept", 130.0, 150.0)]  # not selected
    out, rec = apply(shorts, store, plans)
    assert rec["original_selected_count"] == 5
    assert rec["accepts_retained"] == 1
    assert rec["adjustments_applied"] == 1
    assert rec["rejects_removed"] == 1
    assert rec["unrefined_retained"] == 0
    assert rec["legacy_no_id_retained"] == 1
    assert rec["application_skips"] == {"no_matching_plan": 1}
    assert rec["plans_ignored_not_selected"] == 1
    assert rec["final_selected_count"] == 4
    # 5 == 1+1+1+0+1+1 ; 4 == 1+1+0+1+1+0-0
    assert rec["counts_reconcile"] is True
    assert len(out) == 4


def test_funnel_reconciliation_group_case():
    out, rec = _group_case()
    assert rec["original_selected_count"] == 3
    assert rec["groups_proposed"] == 1
    assert rec["groups_applied"] == 1
    assert rec["final_selected_count"] == 2
    # 3 == 2+1+0+0+0+0 ; 2 == 2+1+0+0+0+1-2
    assert rec["counts_reconcile"] is True


def test_funnel_reconciliation_all_skipped():
    shorts = [make_short("c1", 10.0, 30.0)]
    store = {"c1": make_store_rec("c1", 10.0, 30.0)}
    out, rec = apply(shorts, store, [])  # no plans at all
    assert rec["application_skips"] == {"no_matching_plan": 1}
    assert rec["final_selected_count"] == 1
    assert rec["counts_reconcile"] is True


def test_funnel_carries_version_and_bounds():
    _, rec = apply(*base_case(),
                   [make_plan("c1", "accept", 10.0, 30.0),
                    make_plan("c2", "accept", 40.0, 60.0)])
    assert rec["version"] == "1"
    assert rec["flag"] == "CLIPPYME_APPLY_EDITORIAL_PLANS"
    assert rec["editorial_refinement_ran"] is True
    assert rec["duration_min_s"] == 15.0
    assert rec["duration_max_s"] == 120.0
    assert rec["source_duration_s"] == 200.0
    assert rec["plans_supplied"] == 2


def test_no_quotes_quality_or_secrets_in_record():
    _, rec = _group_case()
    blob = json.dumps(rec)
    for key in ("api_key", "secret", "token", "response_text",
                "\"quotes\"", "\"quality\""):
        assert key not in blob


# --- Production wiring: run_windowed_discovery -------------------------

class FakeClient:
    def __init__(self):
        self.closed = 0

    def close(self):
        self.closed += 1


def _wiring_words(n=200):
    return [{"text": f"word{i}", "start": i * 0.5, "end": (i + 1) * 0.5,
             "speaker": 0} for i in range(n)]


def _wiring_disc_gen(client, prompt, config, budget):
    cands = [
        {"start_s": 10.0, "end_s": 30.0, "event_type": "story",
         "evidence_text": "word20 word21 word22",
         "context": "the setup needed to understand the moment",
         "expected_payoff": "the claim resolves with a clear payoff",
         "confidence": 0.8, "recommendation_score": 90},
        {"start_s": 40.0, "end_s": 60.0, "event_type": "story",
         "evidence_text": "word80 word81 word82",
         "context": "the setup needed to understand the moment",
         "expected_payoff": "the claim resolves with a clear payoff",
         "confidence": 0.8, "recommendation_score": 85},
        {"start_s": 70.0, "end_s": 90.0, "event_type": "story",
         "evidence_text": "word140 word141 word142",
         "context": "the setup needed to understand the moment",
         "expected_payoff": "the claim resolves with a clear payoff",
         "confidence": 0.8, "recommendation_score": 80},
    ]
    return json.dumps({"candidates": cands})


def _wiring_quality():
    return {"opening_strength": 0.9, "context_sufficiency": 0.9,
            "payoff_delivery": 0.9, "useful_value": 0.8,
            "entertaining_value": 0.7, "transcript_support": 0.9,
            "editability": 0.9, "confidence": 0.9}


def _wiring_ed_prompt(packet, instructions, **prefs):
    return f"Candidate: {packet['candidate_id']}"


def _wiring_ed_gen(client, prompt, config, budget):
    _wiring_ed_gen.calls += 1
    cid = re.search(r"Candidate: (\S+)", prompt).group(1)
    if _wiring_ed_gen.calls == 1:
        d = {"candidate_id": cid, "verdict": "adjust",
             "proposed_start_s": 12.0, "proposed_end_s": 32.0,
             "opening_evidence": "word24 word25",
             "premise_evidence": "word30 word31",
             "payoff_evidence": "word58 word59",
             "ending_evidence": "word62 word63",
             "evidence_basis": "transcript",
             "decision_reason": "trim the lead-in for a stronger opening",
             "quality": _wiring_quality(), "continuity_links": []}
    elif _wiring_ed_gen.calls == 2:
        d = {"candidate_id": cid, "verdict": "reject",
             "evidence_basis": "transcript",
             "decision_reason": "the moment never pays off",
             "quality": _wiring_quality(), "continuity_links": []}
    else:
        d = {"candidate_id": cid, "verdict": "accept",
             "proposed_start_s": 70.0, "proposed_end_s": 90.0,
             "opening_evidence": "word140 word141",
             "premise_evidence": "word150 word151",
             "payoff_evidence": "word170 word171",
             "ending_evidence": "word178 word179",
             "evidence_basis": "transcript",
             "decision_reason": "complete story with a clean payoff",
             "quality": _wiring_quality(), "continuity_links": []}
    return json.dumps({"decision": d})


_wiring_ed_gen.calls = 0


def _wiring_config():
    return cd.DiscoveryConfig(
        window_target_s=240.0, window_overlap_s=45.0,
        boundary_tolerance_s=12.0, max_windows=32,
        max_raw_candidates=200, max_provider_attempts=24,
        deadline_s=600.0, sdk_timeout_ms=120000,
        model_name="test-model", fallback_models=())


def run_wired(**kw):
    _wiring_ed_gen.calls = 0
    return cd.run_windowed_discovery(
        {"words": _wiring_words()}, 100.0,
        config=_wiring_config(),
        client_factory=FakeClient,
        generate_fn=_wiring_disc_gen,
        editorial_build_prompt_fn=_wiring_ed_prompt,
        editorial_generate_fn=_wiring_ed_gen,
        **kw)


def test_wiring_applies_adjusted_boundaries_and_removes_rejects():
    result = run_wired()
    shorts = result["shorts"]
    assert [(s["start"], s["end"]) for s in shorts] == [
        (12.0, 32.0), (70.0, 90.0)]
    # The rejected candidate's interval is gone.
    assert all(not (s["start"] == 40.0 and s["end"] == 60.0)
               for s in shorts)
    assert len(shorts) == 2


def test_wiring_returns_application_record_beside_other_records():
    result = run_wired()
    assert set(("shorts", "candidate_discovery", "candidate_store",
                "editorial_refinement", "editorial_plans",
                "editorial_application")) <= set(result)
    app = result["editorial_application"]
    assert app["version"] == "1"
    assert app["enabled"] is True
    assert app["accepts_retained"] == 1
    assert app["adjustments_applied"] == 1
    assert app["rejects_removed"] == 1
    assert app["final_selected_count"] == 2
    assert app["counts_reconcile"] is True
    # The adjusted short kept its discovery copy fields.
    adj = result["shorts"][0]
    assert adj["viral_hook_text"] == ""
    assert adj["candidate_id"]
    assert adj["editorial_application"]["action"] == "adjusted"


def test_wiring_application_runs_exactly_once(monkeypatch):
    calls = {"n": 0}
    real = ea.apply_editorial_plans

    def counting(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(ea, "apply_editorial_plans", counting)
    run_wired()
    assert calls["n"] == 1


def test_mutation_bypassed_application_step_is_detected(monkeypatch):
    # If _finalize stopped calling the application step, the wired
    # shorts would keep discovery boundaries and the reject.
    monkeypatch.setattr(
        ea, "apply_editorial_plans",
        lambda shorts, *a, **k: (shorts, {"bypassed": True}))
    result = run_wired()
    bounds = [(s["start"], s["end"]) for s in result["shorts"]]
    assert (12.0, 32.0) not in bounds  # adjustment lost under bypass
    assert len(result["shorts"]) == 3  # reject renders under bypass


def test_wiring_disabled_flag_keeps_discovery_shorts(monkeypatch):
    monkeypatch.setenv("CLIPPYME_APPLY_EDITORIAL_PLANS", "0")
    result = run_wired()
    bounds = [(s["start"], s["end"]) for s in result["shorts"]]
    assert bounds == [(10.0, 30.0), (40.0, 60.0), (70.0, 90.0)]
    app = result["editorial_application"]
    assert app["enabled"] is False
    assert app["disabled_reason"] == "flag_disabled"


# --- Direct CLI ----------------------------------------------------------

def _stub_heavy_modules(monkeypatch):
    """Stub the heavy CV/ML/genai stack absent from the host runner."""
    for name in ["torch", "mediapipe", "ultralytics"]:
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    torch_cuda = types.ModuleType("torch.cuda")
    torch_cuda.is_available = lambda: False
    sys.modules["torch"].cuda = torch_cuda
    sys.modules["ultralytics"].YOLO = object
    psutil_mod = types.ModuleType("psutil")
    vmem = types.SimpleNamespace(total=16 * 1024 ** 3,
                                 available=8 * 1024 ** 3)
    psutil_mod.virtual_memory = lambda: vmem
    monkeypatch.setitem(sys.modules, "psutil", psutil_mod)
    google_pkg = types.ModuleType("google")
    genai_mod = types.ModuleType("google.genai")

    class FakeGenaiClient:
        def __init__(self, api_key=None):
            pass

        def close(self):
            pass

    genai_mod.Client = FakeGenaiClient
    google_pkg.genai = genai_mod
    monkeypatch.setitem(sys.modules, "google", google_pkg)
    monkeypatch.setitem(sys.modules, "google.genai", genai_mod)


@pytest.fixture
def main_mod(monkeypatch):
    _stub_heavy_modules(monkeypatch)
    sys.modules.pop("clippyme.pipeline.main", None)
    import clippyme.pipeline.main as main
    return main


def test_direct_cli_analysis_returns_applied_shorts(main_mod, monkeypatch):
    """The direct CLI's analysis entry delegates to run_windowed_discovery.

    Fakes are injected at the candidate_discovery seam (the CLI builds
    its own config/factory); the returned shorts must carry the
    applied editorial boundaries.
    """
    real_rwd = cd.run_windowed_discovery

    def fake_rwd(transcript_result, video_duration, **kw):
        _wiring_ed_gen.calls = 0
        kw["config"] = _wiring_config()
        kw["client_factory"] = FakeClient
        kw["generate_fn"] = _wiring_disc_gen
        kw["editorial_build_prompt_fn"] = _wiring_ed_prompt
        kw["editorial_generate_fn"] = _wiring_ed_gen
        return real_rwd(transcript_result, video_duration, **kw)

    monkeypatch.setattr(cd, "run_windowed_discovery", fake_rwd)
    result = main_mod._get_viral_clips_windowed(
        {"words": _wiring_words()}, 100.0, "fake-key")
    assert [(s["start"], s["end"]) for s in result["shorts"]] == [
        (12.0, 32.0), (70.0, 90.0)]
    app = result["editorial_application"]
    assert app["adjustments_applied"] == 1
    assert app["rejects_removed"] == 1
    assert app["counts_reconcile"] is True


def test_direct_cli_slice_seam_receives_adjusted_bounds(main_mod):
    """The __main__ clip loop is inline (not importable); this drives
    its exact slice-call sequence per clip against the real
    build_cut_command: start = clip['start']; end = clip['end'];
    cut_command = build_cut_command(input_video, start, end, dest).
    """
    from clippyme.pipeline.run_ops import (
        build_cut_command as real_build)
    result = run_wired()
    seen = []

    def spy_build(input_video, start, end, dest):
        seen.append((start, end))
        return real_build(input_video, start, end, dest)

    for clip in result["shorts"]:
        start = clip["start"]
        end = clip["end"]
        cmd = spy_build("input.mp4", start, end, "tmp.mp4")
        assert (f"trim=start={start:.3f}:end={end:.3f}"
                in " ".join(cmd))
    assert (12.0, 32.0) in seen
    assert (70.0, 90.0) in seen
    # The rejected candidate's discovery interval never reaches cutting.
    assert not any(s == 40.0 and e == 60.0 for s, e in seen)


# --- Queued orchestrator -------------------------------------------------

@pytest.fixture
def orch_mod():
    import clippyme.pipeline.orchestrator as orch
    return orch


def _drive_render_one_clip(orch_mod, monkeypatch, tmp_path, clip,
                           clips_data, index):
    calls = []
    real_build = orch_mod.build_cut_command

    def spy_build(input_video, start, end, dest):
        calls.append((start, end))
        return real_build(input_video, start, end, dest)

    monkeypatch.setattr(orch_mod, "build_cut_command", spy_build)
    monkeypatch.setattr(
        orch_mod, "_valid_media_file",
        lambda path, minimum=1: bool(
            path and os.path.isfile(path)
            and os.path.getsize(path) >= minimum))
    monkeypatch.setattr(orch_mod, "slice_provenance_valid",
                        lambda *a, **k: False)
    monkeypatch.setattr(orch_mod, "source_fingerprint", lambda p: "fp")
    monkeypatch.setattr(orch_mod, "validate_cut_output",
                        lambda p, d: (True, ""))
    monkeypatch.setattr(orch_mod, "write_slice_provenance",
                        lambda *a, **k: None)
    monkeypatch.setattr(orch_mod, "inspect_clip",
                        lambda *a, **k: {"critical": False,
                                         "warnings": [], "ok": True})
    monkeypatch.setattr(orch_mod, "emit_clip_project",
                        lambda **k: "fake")

    def fake_run(cmd, **kw):
        out = cmd[-1]
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_bytes(b"x" * 20000)

        class Done:
            returncode = 0
            stderr = b""
        return Done()

    monkeypatch.setattr(orch_mod.subprocess, "run", fake_run)

    class Legacy:
        def process_video_to_vertical(self, src, tmp, **kw):
            Path(tmp).write_bytes(b"y" * 20000)
            return True

        def normalize_audio(self, p):
            pass

        def select_cover_frame(self, p):
            pass

    class FakeState:
        def start(self, *a, **k):
            pass

        def mark_clip(self, *a, **k):
            pass

    input_video = str(tmp_path / "input.mp4")
    Path(input_video).write_bytes(b"z" * 20000)
    metadata_file = str(tmp_path / "meta.json")
    args = types.SimpleNamespace(reframe_mode="auto", no_zoom=True,
                                 letterbox_zoom=1.0)
    ok = orch_mod._render_one_clip(
        index=index, total=2, clip=clip, input_video=input_video,
        video_title="video", output_dir=str(tmp_path),
        metadata_file=metadata_file, clips_data=clips_data,
        args=args, aspect_ratio=9 / 16, state=FakeState(),
        legacy=Legacy())
    return ok, calls, metadata_file


def test_queued_render_receives_adjusted_bounds(orch_mod, monkeypatch,
                                                tmp_path):
    result = run_wired()
    clips_data = copy.deepcopy(result)
    all_calls = []
    for i, clip in enumerate(clips_data["shorts"]):
        ok, calls, _ = _drive_render_one_clip(
            orch_mod, monkeypatch, tmp_path, clip, clips_data, i)
        assert ok is True
        all_calls.extend(calls)
    assert (12.0, 32.0) in all_calls
    assert (70.0, 90.0) in all_calls
    # The rejected candidate never reaches the source-slice call.
    assert not any(s == 40.0 and e == 60.0 for s, e in all_calls)


def test_queued_metadata_atomically_preserves_application(
        orch_mod, monkeypatch, tmp_path):
    result = run_wired()
    clips_data = copy.deepcopy(result)
    clip = copy.deepcopy(clips_data["shorts"][0])
    ok, _, metadata_file = _drive_render_one_clip(
        orch_mod, monkeypatch, tmp_path, clip, clips_data, 0)
    assert ok is True
    assert os.path.isfile(metadata_file)
    # No torn writes: the file parses and carries the record.
    saved = json.loads(Path(metadata_file).read_text(encoding="utf-8"))
    app = saved["editorial_application"]
    assert app["version"] == "1"
    assert app["adjustments_applied"] == 1
    assert app["rejects_removed"] == 1
    assert app["counts_reconcile"] is True
    assert saved["shorts"][0]["start"] == 12.0


# --- Fallback metadata ---------------------------------------------------

def test_fallback_record_marks_plans_not_applied():
    shorts, store = base_case()
    plans = [make_plan("c1", "adjust", 12.0, 32.0),
             make_plan("c2", "accept", 40.0, 60.0)]
    _, app_record = apply(shorts, store, plans)
    fallback_shorts = [{"start": 0.0, "end": 45.0},
                       {"start": 45.0, "end": 90.0}]
    data = {"shorts": fallback_shorts}
    ea.preserve_application_through_fallback(
        data, app_record, "texttiling")
    rec = data["editorial_application"]
    assert rec["version"] == "1"
    assert rec["enabled"] is False
    assert rec["disabled_reason"] == (
        "fallback_shorts_have_no_editorial_plans")
    assert rec["fallback_strategy"] == "texttiling"
    # The earlier record survives for audit, verbatim.
    assert rec["superseded_record"] == app_record
    # Fallback shorts are untouched: no stale plan boundaries leak in.
    assert fallback_shorts == [{"start": 0.0, "end": 45.0},
                               {"start": 45.0, "end": 90.0}]
    assert all("application_skipped" not in s for s in fallback_shorts)
    assert all("editorial_application" not in s for s in fallback_shorts)


def test_mutation_fallback_inheriting_stale_plans_fails():
    """A fallback record must never claim plans were applied."""
    data = {"shorts": [{"start": 0.0, "end": 45.0}]}
    ea.preserve_application_through_fallback(
        data, {"version": "1", "enabled": True}, "whole_video")
    rec = data["editorial_application"]
    assert rec["enabled"] is False
    assert rec["fallback_strategy"] == "whole_video"
    assert rec["counts_reconcile"] is True


def test_fallback_record_without_superseded():
    rec = ea.fallback_application_record(None, "whole_video", 0)
    assert rec["superseded_record"] is None
    assert rec["final_selected_count"] == 0
    assert rec["version"] == "1"


# --- Mutation checks -----------------------------------------------------

def test_mutation_ignored_adjusted_boundaries_fails():
    """If the render path used discovery bounds instead of applied
    bounds, the source-slice spy would see the stale interval."""
    result = run_wired()
    adj = result["shorts"][0]
    assert (adj["start"], adj["end"]) == (12.0, 32.0)
    assert adj["editorial_application"]["original_start_s"] == 10.0


def test_mutation_rendered_reject_fails():
    result = run_wired()
    ids = [s["candidate_id"] for s in result["shorts"]]
    assert len(ids) == len(set(ids)) == 2
    assert result["editorial_application"]["rejects_removed"] == 1


def test_mutation_dropped_fields_fails():
    shorts = [make_short("c1", 10.0, 30.0)]
    store = {"c1": make_store_rec("c1", 10.0, 30.0)}
    plans = [make_plan("c1", "adjust", 12.0, 32.0)]
    before = copy.deepcopy(shorts[0])
    out, _ = apply(shorts, store, plans)
    adj = out[0]
    dropped = [k for k in before if k not in ("start", "end")
               and adj.get(k) != before[k]]
    assert dropped == []


def test_mutation_invalid_group_removing_components_fails():
    out, rec = _group_case(group_kw={"components": ["c1", "cX"]})
    assert [s["candidate_id"] for s in out] == ["c1", "c2", "c3"]
    assert rec["groups_applied"] == 0
    assert rec["groups_rejected"] == 1


# --- ViralClip compatibility ---------------------------------------------

def test_output_shorts_remain_viralclip_compatible():
    from clippyme.schemas import ViralClip
    out, _ = _group_case()
    for s in out:
        ViralClip.model_validate(s)
    # Adjusted-only case too.
    shorts, store = base_case()
    plans = [make_plan("c1", "adjust", 12.0, 32.0),
             make_plan("c2", "accept", 40.0, 60.0)]
    out2, _ = apply(shorts, store, plans)
    for s in out2:
        ViralClip.model_validate(s)


def test_wired_shorts_remain_viralclip_compatible():
    from clippyme.schemas import ViralClip
    result = run_wired()
    for s in result["shorts"]:
        ViralClip.model_validate(s)

# --- Review-01 promoted probes + mutation coverage ----------------------

from types import SimpleNamespace as _R1SimpleNamespace


def _r1_short(cid="c1", start=10.0, end=30.0):
    return {
        "candidate_id": cid,
        "start": start,
        "end": end,
        "viral_score": 80,
        "viral_reason": "grounded candidate reason",
        "unknown_field": {"kept": True},
    }


def _r1_store(cid="c1", start_s=10.0, end_s=30.0, fp="fp-1"):
    return {
        cid: {
            "candidate_id": cid,
            "start_s": start_s,
            "end_s": end_s,
            "moment_fingerprint": fp,
            "recommendation_score": 90.0,
            "quality": {},
        }
    }


def _r1_plan(cid="c1", verdict="accept", start=10.0, end=30.0, fp="fp-1",
             **kw):
    plan = {
        "candidate_id": cid,
        "verdict": verdict,
        "start_s": start,
        "end_s": end,
        "plan_version": er.EDITORIAL_VERSION,
        "status": "refined",
        "moment_fingerprint": fp,
    }
    plan.update(kw)
    return plan


def _r1_apply(shorts, records, plans, source_duration=200.0):
    return ea.apply_editorial_plans(
        shorts, records, plans, source_duration, 15.0, 120.0,
        enabled=True, editorial_ran=True,
    )


def _r1_assert_skipped(out, rec, original, reason):
    assert len(out) == 1
    for key, value in original.items():
        assert out[0][key] == value
    assert out[0]["application_skipped"] == reason
    assert rec["application_skips"] == {reason: 1}
    assert rec["counts_reconcile"] is True


def test_r1_store_record_identity_must_match_lookup_key():
    records = _r1_store()
    records["c1"]["candidate_id"] = "different-id"
    original = copy.deepcopy(_r1_short())
    out, rec = _r1_apply([_r1_short()], records, [_r1_plan()])
    _r1_assert_skipped(out, rec, original, "candidate_store_id_mismatch")


def test_r1_adjusted_plan_must_overlap_current_candidate_interval():
    original = copy.deepcopy(_r1_short())
    out, rec = _r1_apply(
        [_r1_short()], _r1_store(),
        [_r1_plan(verdict="adjust", start=80.0, end=100.0)],
    )
    _r1_assert_skipped(
        out, rec, original, "plan_interval_disjoint_from_candidate")


def test_r1_unknown_source_duration_cannot_authorize_negative_cut():
    original = copy.deepcopy(_r1_short())
    out, rec = _r1_apply(
        [_r1_short()], _r1_store(),
        [_r1_plan(verdict="adjust", start=-5.0, end=15.0)],
        source_duration=0.0,
    )
    _r1_assert_skipped(out, rec, original, "invalid_source_duration")
    assert rec["source_bounds_checked"] is False


def test_r1_accept_does_not_approve_malformed_original_short_interval():
    short = _r1_short()
    short["start"] = True
    original = copy.deepcopy(short)
    out, rec = _r1_apply([short], _r1_store(), [_r1_plan()])
    _r1_assert_skipped(out, rec, original, "invalid_short_boundaries")
    assert out[0]["start"] is True
    assert rec["accepts_retained"] == 0
    # The caller's short is untouched: annotation landed on the copy.
    assert short == original


def test_r1_non_adjust_group_verdict_cannot_replace_components():
    shorts = [_r1_short("c1", 10.0, 30.0), _r1_short("c2", 40.0, 60.0)]
    records = _r1_store("c1", 10.0, 30.0, "fp-1")
    records.update(_r1_store("c2", 40.0, 60.0, "fp-2"))
    plans = [
        _r1_plan("c1", "accept", 10.0, 30.0, fp="fp-1"),
        _r1_plan("c2", "accept", 40.0, 60.0, fp="fp-2"),
        {
            "candidate_id": "group:c1+c2",
            "verdict": "reject",
            "start_s": 10.0,
            "end_s": 60.0,
            "plan_version": er.EDITORIAL_VERSION,
            "status": "refined",
            "moment_fingerprint": "",
            "components": ["c1", "c2"],
        },
    ]
    out, rec = _r1_apply(shorts, records, plans)
    assert [s["candidate_id"] for s in out] == ["c1", "c2"]
    assert rec["groups_applied"] == 0
    assert rec["groups_rejected"] == 1
    assert rec["group_rejections"][0]["reason"] == "group_invalid_verdict"
    assert rec["counts_reconcile"] is True


def test_r1_all_editorial_rejections_do_not_render_whole_video_fallback(
    orch_mod, tmp_path, monkeypatch,
):
    orch = orch_mod

    class _R1State:
        def __init__(self):
            self.total = None

        def artifact(self, name):
            return None

        def completed(self, stage):
            return False

        def consume_steering(self):
            return []

        def start(self, *a, **k):
            pass

        def complete_stage(self, *a, **k):
            pass

        def set_clip_total(self, n):
            self.total = n

    class _R1Legacy:
        def get_viral_clips(self, *a, **k):
            return {
                "shorts": [],
                "editorial_application": {
                    "version": ea.APPLICATION_VERSION,
                    "enabled": True,
                    "status": "applied",
                    "ran": True,
                    "fail_soft": False,
                    "original_selected_count": 1,
                    "accepts_retained": 0,
                    "adjustments_applied": 0,
                    "rejects_removed": 1,
                    "unrefined_retained": 0,
                    "legacy_no_id_retained": 0,
                    "groups_applied": 0,
                    "groups_rejected": 0,
                    "application_skips": {},
                    "final_selected_count": 0,
                    "counts_reconcile": True,
                    "source_bounds_checked": True,
                },
            }

        def build_texttiling_fallback(self, transcript, video_title):
            raise AssertionError(
                "fallback must not run on intentional empty")

    monkeypatch.setenv("CLIPPYME_SILENCE_SNAP", "0")

    args = _R1SimpleNamespace(
        monitor=False, aspect="9:16", skip_analysis=False,
        instructions=None, min_duration=None, max_duration=None,
        min_clips=None, max_clips=None, clip_type=None,
    )
    state = _R1State()
    result, _metadata_file = orch._load_or_analyze(
        args, str(tmp_path / "source.mp4"), "source", str(tmp_path),
        120.0, {"segments": []}, state, _R1Legacy(),
    )
    assert result["shorts"] == []
    assert result["editorial_application"]["enabled"] is True
    assert result["editorial_application"]["rejects_removed"] == 1
    assert state.total == 0


def _r1_editorial_empty_record(**kw):
    rec = {
        "version": ea.APPLICATION_VERSION,
        "enabled": True,
        "counts_reconcile": True,
        "original_selected_count": 2,
        "final_selected_count": 0,
        "accepts_retained": 0,
        "adjustments_applied": 0,
        "rejects_removed": 2,
        "unrefined_retained": 0,
        "legacy_no_id_retained": 0,
        "groups_applied": 0,
    }
    rec.update(kw)
    return {"shorts": [], "editorial_application": rec}


def test_r1_is_intentional_editorial_empty_shapes():
    assert ea.is_intentional_editorial_empty(
        _r1_editorial_empty_record()) is True
    assert ea.is_intentional_editorial_empty({"shorts": []}) is False
    assert ea.is_intentional_editorial_empty(None) is False
    assert ea.is_intentional_editorial_empty(
        _r1_editorial_empty_record(enabled=False)) is False
    assert ea.is_intentional_editorial_empty(
        _r1_editorial_empty_record(counts_reconcile=False)) is False
    assert ea.is_intentional_editorial_empty(
        _r1_editorial_empty_record(original_selected_count=0)) is False
    assert ea.is_intentional_editorial_empty(
        _r1_editorial_empty_record(final_selected_count=1)) is False
    assert ea.is_intentional_editorial_empty(
        {"editorial_application": "nope"}) is False


def test_r1_direct_empty_action_intentional_empty(main_mod):
    assert main_mod._resolve_empty_shorts_action(
        _r1_editorial_empty_record(), False) == "intentional_empty"


def test_r1_direct_empty_action_whole_video_on_genuine_failure(main_mod):
    resolve = main_mod._resolve_empty_shorts_action
    assert resolve({"shorts": []}, False) == "whole_video"
    assert resolve(None, False) == "whole_video"
    assert resolve(
        _r1_editorial_empty_record(enabled=False), False) == "whole_video"
    assert resolve(
        _r1_editorial_empty_record(counts_reconcile=False),
        False) == "whole_video"
    assert resolve(
        _r1_editorial_empty_record(original_selected_count=0),
        False) == "whole_video"


def test_r1_direct_empty_action_monitor_empty(main_mod):
    assert main_mod._resolve_empty_shorts_action(
        {"shorts": []}, True) == "monitor_empty"


def test_r1_reversed_store_interval_is_invalid_store_boundaries():
    records = _r1_store()
    records["c1"]["start_s"], records["c1"]["end_s"] = 30.0, 10.0
    original = copy.deepcopy(_r1_short())
    out, rec = _r1_apply([_r1_short()], records, [_r1_plan()])
    _r1_assert_skipped(out, rec, original, "invalid_store_boundaries")


def test_r1_boolean_store_interval_is_invalid_store_boundaries():
    records = _r1_store()
    records["c1"]["start_s"] = True
    original = copy.deepcopy(_r1_short())
    out, rec = _r1_apply([_r1_short()], records, [_r1_plan()])
    _r1_assert_skipped(out, rec, original, "invalid_store_boundaries")


def test_r1_short_store_interval_disagreement_isolated():
    records = _r1_store()
    records["c1"]["start_s"] = 11.0  # short says [10, 30]
    original = copy.deepcopy(_r1_short())
    out, rec = _r1_apply([_r1_short()], records, [_r1_plan()])
    _r1_assert_skipped(out, rec, original, "short_store_interval_mismatch")


def test_r1_accept_plan_must_retain_exact_candidate_interval():
    original = copy.deepcopy(_r1_short())
    out, rec = _r1_apply(
        [_r1_short()], _r1_store(),
        [_r1_plan("c1", "accept", 10.0, 30.000001)],
    )
    _r1_assert_skipped(out, rec, original, "plan_interval_mismatch")
    assert rec["accepts_retained"] == 0


def test_r1_reject_plan_must_retain_exact_candidate_interval():
    original = copy.deepcopy(_r1_short())
    out, rec = _r1_apply(
        [_r1_short()], _r1_store(),
        [_r1_plan("c1", "reject", 12.0, 28.0)],
    )
    _r1_assert_skipped(out, rec, original, "plan_interval_mismatch")
    assert rec["rejects_removed"] == 0


# --- review-02: promoted probes (strict source + strict empty) ---

def _r2_short(cid="c1", start=10.0, end=30.0):
    return {
        "candidate_id": cid,
        "start": start,
        "end": end,
        "viral_score": 80,
        "viral_reason": "grounded candidate reason",
    }


def _r2_store(cid="c1", start=10.0, end=30.0, fp="fp-1"):
    return {
        cid: {
            "candidate_id": cid,
            "moment_fingerprint": fp,
            "start_s": start,
            "end_s": end,
        }
    }


def _r2_plan(cid="c1", verdict="accept", start=10.0, end=30.0,
             fp="fp-1", **extra):
    item = {
        "plan_version": er.EDITORIAL_VERSION,
        "candidate_id": cid,
        "verdict": verdict,
        "status": "refined" if verdict != "unrefined" else "unrefined",
        "start_s": start,
        "end_s": end,
        "moment_fingerprint": fp,
        "reason": "validated editorial decision",
        "components": [],
    }
    item.update(extra)
    return item


def _r2_apply(shorts, store, plans, source_duration):
    return ea.apply_editorial_plans(
        shorts, store, plans, source_duration, 15.0, 120.0,
        enabled=True, editorial_ran=True,
    )


def _r2_assert_one_skip(output, record, reason):
    assert len(output) == 1
    assert output[0]["application_skipped"] == reason
    assert record["application_skips"] == {reason: 1}
    assert record["accepts_retained"] == 0
    assert record["adjustments_applied"] == 0
    assert record["rejects_removed"] == 0
    assert record["counts_reconcile"] is True


def test_r2_invalid_source_duration_skips_even_nonnegative_accept():
    output, record = _r2_apply(
        [_r2_short()], _r2_store(), [_r2_plan()], source_duration=0.0)
    _r2_assert_one_skip(output, record, "invalid_source_duration")


def test_r2_negative_current_candidate_interval_is_not_authoritative():
    output, record = _r2_apply(
        [_r2_short(start=-5.0, end=15.0)],
        _r2_store(start=-5.0, end=15.0),
        [_r2_plan(verdict="adjust", start=0.0, end=20.0)],
        source_duration=100.0,
    )
    _r2_assert_one_skip(output, record, "candidate_boundaries_out_of_source")


def test_r2_current_candidate_interval_past_source_is_not_authoritative():
    output, record = _r2_apply(
        [_r2_short(start=90.0, end=110.0)],
        _r2_store(start=90.0, end=110.0),
        [_r2_plan(verdict="adjust", start=80.0, end=100.0)],
        source_duration=100.0,
    )
    _r2_assert_one_skip(output, record, "candidate_boundaries_out_of_source")


def test_r2_invalid_source_duration_cannot_authorize_group_application():
    shorts = [_r2_short("c1", 10.0, 30.0), _r2_short("c2", 40.0, 60.0)]
    store = {}
    store.update(_r2_store("c1", 10.0, 30.0, "fp-1"))
    store.update(_r2_store("c2", 40.0, 60.0, "fp-2"))
    plans = [
        _r2_plan("c1", "accept", 10.0, 30.0, "fp-1"),
        _r2_plan("c2", "accept", 40.0, 60.0, "fp-2"),
        _r2_plan(
            "group:c1+c2", "adjust", -5.0, 60.0, "",
            components=["c1", "c2"],
        ),
    ]
    output, record = _r2_apply(shorts, store, plans, source_duration=0.0)
    assert [item["candidate_id"] for item in output] == ["c1", "c2"]
    assert all(item["application_skipped"] == "invalid_source_duration"
               for item in output)
    assert record["groups_applied"] == 0
    assert record["counts_reconcile"] is True


def _r2_empty_record(**overrides):
    record = {
        "version": ea.APPLICATION_VERSION,
        "enabled": True,
        "counts_reconcile": True,
        "original_selected_count": 1,
        "final_selected_count": 0,
        "accepts_retained": 0,
        "adjustments_applied": 0,
        "rejects_removed": 1,
        "unrefined_retained": 0,
        "legacy_no_id_retained": 0,
        "application_skips": {},
        "groups_applied": 0,
    }
    record.update(overrides)
    return {"shorts": [], "editorial_application": record}


def test_r2_only_current_strictly_reconciled_record_suppresses_fallback():
    assert ea.is_intentional_editorial_empty(_r2_empty_record()) is True
    assert ea.is_intentional_editorial_empty(
        _r2_empty_record(version="stale")) is False
    assert ea.is_intentional_editorial_empty(
        _r2_empty_record(original_selected_count=True)) is False
    assert ea.is_intentional_editorial_empty(
        _r2_empty_record(original_selected_count="1")) is False
    assert ea.is_intentional_editorial_empty(
        _r2_empty_record(rejects_removed=0)) is False


# --- review-02 mutation coverage ---

@pytest.mark.parametrize("bad_duration", [None, True, False, 0.0, -5.0,
                                          float("nan"), float("inf")])
def test_r2_missing_or_bad_source_duration_blocks_all_verdicts(bad_duration):
    for verdict in ("accept", "adjust", "reject"):
        start, end = (10.0, 30.0) if verdict != "adjust" else (12.0, 28.0)
        output, record = _r2_apply(
            [_r2_short()], _r2_store(),
            [_r2_plan(verdict=verdict, start=start, end=end)],
            source_duration=bad_duration)
        assert output[0]["application_skipped"] == \
            "invalid_source_duration"
        assert record["accepts_retained"] == 0
        assert record["adjustments_applied"] == 0
        assert record["rejects_removed"] == 0


def test_r2_source_duration_zero_blocks_group_even_positive_bounds():
    shorts = [_r2_short("c1", 10.0, 30.0), _r2_short("c2", 40.0, 60.0)]
    store = {}
    store.update(_r2_store("c1", 10.0, 30.0, "fp-1"))
    store.update(_r2_store("c2", 40.0, 60.0, "fp-2"))
    plans = [
        _r2_plan("c1", "accept", 10.0, 30.0, "fp-1"),
        _r2_plan("c2", "accept", 40.0, 60.0, "fp-2"),
        _r2_plan("group:c1+c2", "adjust", 10.0, 60.0, "",
                 components=["c1", "c2"]),
    ]
    output, record = _r2_apply(shorts, store, plans, source_duration=0.0)
    assert record["groups_applied"] == 0
    assert record["groups_rejected"] == 1
    assert (record["group_rejections"][0]["reason"]
            == "invalid_source_duration")


def test_r2_store_interval_out_of_source_is_not_authoritative():
    store = _r2_store(start=90.0, end=130.0)
    output, record = _r2_apply(
        [_r2_short(start=90.0, end=130.0)], store, [_r2_plan()],
        source_duration=120.0)
    _r2_assert_one_skip(output, record, "candidate_boundaries_out_of_source")


def test_r2_plan_interval_out_of_source_still_rejected():
    output, record = _r2_apply(
        [_r2_short()], _r2_store(),
        [_r2_plan(verdict="adjust", start=90.0, end=150.0)],
        source_duration=120.0)
    _r2_assert_one_skip(output, record, "plan_boundaries_out_of_source")


def test_r2_group_interval_out_of_source_still_rejected():
    shorts = [_r2_short("c1", 10.0, 30.0), _r2_short("c2", 40.0, 60.0)]
    store = {}
    store.update(_r2_store("c1", 10.0, 30.0, "fp-1"))
    store.update(_r2_store("c2", 40.0, 60.0, "fp-2"))
    plans = [
        _r2_plan("c1", "accept", 10.0, 30.0, "fp-1"),
        _r2_plan("c2", "accept", 40.0, 60.0, "fp-2"),
        _r2_plan("group:c1+c2", "adjust", 10.0, 130.0, "",
                 components=["c1", "c2"]),
    ]
    output, record = _r2_apply(shorts, store, plans, source_duration=120.0)
    assert record["groups_applied"] == 0
    assert record["groups_rejected"] == 1
    assert (record["group_rejections"][0]["reason"]
            == "group_boundaries_out_of_source")
    # Components stay individual.
    assert [s["candidate_id"] for s in output] == ["c1", "c2"]


@pytest.mark.parametrize("bad_flag", [1, "yes", None])
def test_r2_non_boolean_enabled_flag_never_suppresses_fallback(bad_flag):
    assert ea.is_intentional_editorial_empty(
        _r2_empty_record(enabled=bad_flag)) is False


@pytest.mark.parametrize("bad_count", [True, "2", 2.0, -1, None])
def test_r2_coerced_or_negative_counts_never_suppress_fallback(bad_count):
    assert ea.is_intentional_editorial_empty(
        _r2_empty_record(original_selected_count=bad_count)) is False
    assert ea.is_intentional_editorial_empty(
        _r2_empty_record(rejects_removed=bad_count)) is False


def test_r2_inconsistent_zero_output_arithmetic_keeps_fallback():
    # Claims 2 rejects for 1 original selection.
    assert ea.is_intentional_editorial_empty(
        _r2_empty_record(rejects_removed=2)) is False
    # Retained output claimed with zero final count.
    assert ea.is_intentional_editorial_empty(
        _r2_empty_record(accepts_retained=1)) is False
    # Grouped output present.
    assert ea.is_intentional_editorial_empty(
        _r2_empty_record(groups_applied=1)) is False
    # Non-empty shorts list despite an empty-claiming funnel.
    bad = _r2_empty_record()
    bad["shorts"] = [_r2_short()]
    assert ea.is_intentional_editorial_empty(bad) is False


def test_r2_real_all_rejected_record_still_suppresses_fallback():
    output, record = _r2_apply(
        [_r2_short("c1", 10.0, 30.0), _r2_short("c2", 40.0, 60.0)],
        {**_r2_store("c1", 10.0, 30.0, "fp-1"),
         **_r2_store("c2", 40.0, 60.0, "fp-2")},
        [_r2_plan("c1", "reject", 10.0, 30.0, "fp-1"),
         _r2_plan("c2", "reject", 40.0, 60.0, "fp-2")],
        source_duration=200.0)
    assert output == []
    clips_data = {"shorts": output, "editorial_application": record}
    assert ea.is_intentional_editorial_empty(clips_data) is True
