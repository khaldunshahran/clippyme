"""Task 19 -- bulk_edit domain: typed intents, planning, ledger, apply/undo.

Host-runnable (no heavy deps, no app import). Fixtures use dissimilar clips
with different durations, styles and versions; no full project is ever copied
across clips -- each clip gets its own built dict.
"""
import json
import os

import pytest

from clippyme.domain.bulk_edit import (
    OUTCOME_APPLIED,
    OUTCOME_CONFLICT,
    OUTCOME_EXCLUDED,
    OUTCOME_FAILED,
    OUTCOME_PENDING,
    OUTCOME_PLANNED,
    OUTCOME_PRESERVED,
    OUTCOME_UNCHANGED,
    OUTCOME_UNSUPPORTED,
    apply_operation,
    bulk_item_key,
    create_operation,
    find_version_by_item_key,
    load_bulk_ledger,
    plan_bulk_operation,
    resolve_scope_indices,
    retry_operation,
    undo_operation,
    validate_bulk_intent,
)
from clippyme.domain.project_render import (
    latest_project_version,
    save_project_version,
)
from clippyme.domain.clip_project import validate_project


def _project_dict(clip_index, version=1, **over):
    d = {
        "schema": "nugget.clip-project/1",
        "version": version,
        "origin": "auto",
        "job_id": "job-1",
        "clip_index": clip_index,
        "idempotency_key": f"seed-{clip_index}",
        "source": {"file": f"source_{clip_index}.mp4", "width": 1920,
                   "height": 1080, "duration": 20.0},
        "segments": [{
            "id": f"seg-{clip_index}", "start": 0.0, "end": 20.0,
            "crop": {"x": 0.0, "y": 0.0, "w": 608.0, "h": 1080.0},
        }],
        "captions": {
            "style": "classic_white",
            "position": "bottom",
            "words": [{"id": "w0", "w": "hello", "start": 0.5, "end": 0.9}],
        },
        "overlays": [{
            "type": "hook", "text": f"hook {clip_index}", "style": "viral",
            "start": 0.0, "end": 3.0, "position": "top",
        }],
    }
    d.update(over)
    return d


def _seed_clip(job_dir, clip_index, project_dict):
    path = os.path.join(job_dir, f"clip-project-{clip_index}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(project_dict, f)
    return path


def _seed_job(tmp_path, clips):
    job_dir = str(tmp_path / "job-1")
    os.makedirs(job_dir, exist_ok=True)
    for ci, proj in clips.items():
        _seed_clip(job_dir, ci, proj)
    return job_dir


# --- intent validation -------------------------------------------------------

def test_validate_intent_rejects_unknown_op():
    with pytest.raises(ValueError, match="unknown bulk operation"):
        validate_bulk_intent({"op": "delete_everything", "params": {"x": 1}})


def test_validate_intent_rejects_unknown_keys_and_bad_enums():
    with pytest.raises(ValueError, match="unknown caption_appearance params"):
        validate_bulk_intent({"op": "caption_appearance",
                              "params": {"caption_style": "classic_white",
                                         "font_size": 99}})
    with pytest.raises(ValueError, match="unknown caption style"):
        validate_bulk_intent({"op": "caption_appearance",
                              "params": {"caption_style": "comic_sans"}})
    with pytest.raises(ValueError, match="unknown intent keys"):
        validate_bulk_intent({"op": "caption_appearance",
                              "params": {"caption_style": "classic_white"},
                              "sneaky": True})


def test_validate_intent_rejects_bool_numbers():
    with pytest.raises(ValueError, match="must not be a boolean"):
        validate_bulk_intent({"op": "hook_timing",
                              "params": {"start": True, "end": 2.0}})


def test_validate_intent_hook_timing_end_before_start():
    with pytest.raises(ValueError, match="end must be after hook start"):
        validate_bulk_intent({"op": "hook_timing",
                              "params": {"start": 2.0, "end": 1.0}})


def test_validate_intent_hyperframes_unknown_template():
    with pytest.raises(ValueError, match="unsupported hyperframes template"):
        validate_bulk_intent({"op": "hyperframes_finishing",
                              "params": {"template": "explosions"}})


def test_validate_intent_hyperframes_rejects_on_override():
    with pytest.raises(ValueError, match="on_override does not apply"):
        validate_bulk_intent({"op": "hyperframes_finishing",
                              "params": {"template": "clean-captions"},
                              "on_override": "preserve"})


def test_validate_intent_normalizes():
    norm = validate_bulk_intent({"op": "caption_appearance",
                                 "params": {"caption_style": "hormozi_bold"}})
    assert norm == {"op": "caption_appearance",
                    "params": {"caption_style": "hormozi_bold"},
                    "on_override": "preserve"}


# --- scope -------------------------------------------------------------------

def test_resolve_scope_selected_orders_and_dedupes(tmp_path):
    job_dir = _seed_job(tmp_path, {0: _project_dict(0), 1: _project_dict(1)})
    assert resolve_scope_indices(
        job_dir, {"mode": "selected", "clip_indices": [1, 0, 1]}) == [0, 1]


def test_resolve_scope_all_enumerates_snapshot(tmp_path):
    job_dir = _seed_job(tmp_path, {0: _project_dict(0), 2: _project_dict(2)})
    assert resolve_scope_indices(job_dir, {"mode": "all"}) == [0, 2]
    # A clip added later is not part of the frozen snapshot.
    _seed_clip(job_dir, 5, _project_dict(5))
    assert resolve_scope_indices(job_dir, {"mode": "all"}) == [0, 2, 5]


def test_resolve_scope_rejects_bad_mode():
    with pytest.raises(ValueError, match="unknown scope mode"):
        resolve_scope_indices("/nonexistent", {"mode": "everything"})


# --- planning: overrides -----------------------------------------------------

def _three_clips(tmp_path):
    # Clip 0/1 share the common baseline; clip 2 has an individual override.
    clips = {
        0: _project_dict(0),
        1: _project_dict(1),
        2: _project_dict(2, captions={
            "style": "neon_glow", "position": "bottom",
            "words": [{"id": "w0", "w": "yo", "start": 0.5, "end": 0.9}]}),
    }
    return _seed_job(tmp_path, clips)


def test_plan_preserves_individual_overrides_by_default(tmp_path):
    job_dir = _three_clips(tmp_path)
    plan = plan_bulk_operation(
        job_dir=job_dir, clip_indices=[0, 1, 2],
        intent={"op": "caption_appearance",
                "params": {"caption_style": "hormozi_bold"}})
    assert plan["baselines"]["caption_style"] == "classic_white"
    by_ci = {i["clip_index"]: i for i in plan["items"]}
    assert by_ci[0]["changes"] == [
        {"field": "caption_style", "from": "classic_white",
         "to": "hormozi_bold", "replaced_override": False}]
    assert by_ci[1]["changes"][0]["to"] == "hormozi_bold"
    # Clip 2 differs from both baseline and target: preserved, disclosed.
    assert by_ci[2]["outcome"] == OUTCOME_PRESERVED
    assert by_ci[2]["changes"] == []
    assert by_ci[2]["preserved"] == [
        {"field": "caption_style", "current": "neon_glow",
         "baseline": "classic_white",
         "reason": "individual override preserved"}]
    assert by_ci[2]["patch"] == {}


def test_plan_replace_mode_identifies_affected_fields(tmp_path):
    job_dir = _three_clips(tmp_path)
    plan = plan_bulk_operation(
        job_dir=job_dir, clip_indices=[0, 1, 2],
        intent={"op": "caption_appearance",
                "params": {"caption_style": "hormozi_bold"},
                "on_override": "replace"})
    by_ci = {i["clip_index"]: i for i in plan["items"]}
    ch = by_ci[2]["changes"][0]
    assert ch["field"] == "caption_style"
    assert ch["from"] == "neon_glow" and ch["to"] == "hormozi_bold"
    assert ch["replaced_override"] is True
    assert by_ci[2]["patch"] == {"captions.style": "hormozi_bold"}


def test_plan_unchanged_when_already_target(tmp_path):
    clips = {0: _project_dict(0, captions={
        "style": "hormozi_bold", "position": "bottom",
        "words": [{"id": "w0", "w": "hi", "start": 0.5, "end": 0.9}]})}
    job_dir = _seed_job(tmp_path, clips)
    plan = plan_bulk_operation(
        job_dir=job_dir, clip_indices=[0],
        intent={"op": "caption_appearance",
                "params": {"caption_style": "hormozi_bold"}})
    item = plan["items"][0]
    assert item["patch"] == {}
    assert "already" in item["summary"]


def test_shared_style_never_touches_transcript_or_hook_text(tmp_path):
    job_dir = _three_clips(tmp_path)
    op = create_operation(
        job_dir=job_dir, job_id="job-1",
        scope={"mode": "selected", "clip_indices": [0, 1, 2]},
        intent={"op": "caption_appearance",
                "params": {"caption_style": "hormozi_bold",
                           "caption_position": "top"}},
        idempotency_key="k-style-1")
    op = apply_operation(job_dir=job_dir, operation_id=op["operation_id"])
    assert op["items"]["0"]["outcome"] == OUTCOME_APPLIED
    assert op["items"]["1"]["outcome"] == OUTCOME_APPLIED
    # Clip 2: its style override was preserved while its position changed.
    assert op["items"]["2"]["outcome"] == OUTCOME_APPLIED
    assert [p["field"] for p in op["items"]["2"]["preserved"]] == [
        "caption_style"]
    assert [c["field"] for c in op["items"]["2"]["changes"]] == [
        "caption_position"]
    for ci_str in ("0", "1", "2"):
        with open(os.path.join(
                job_dir, f"clip-project-{ci_str}.json"), encoding="utf-8") as f:
            proj = json.load(f)
        # Words, hook text and segments are byte-identical to the seed.
        assert proj["captions"]["words"][0]["w"] in ("hello", "yo")
        assert proj["overlays"][0]["text"] == f"hook {ci_str}"
        assert proj["segments"][0]["start"] == 0.0
        assert proj["source"]["file"] == f"source_{ci_str}.mp4"
    # The preserved clip kept its individual style override.
    with open(os.path.join(job_dir, "clip-project-2.json"),
              encoding="utf-8") as f:
        assert json.load(f)["captions"]["style"] == "neon_glow"


# --- planning: hook appearance / timing --------------------------------------

def test_plan_hook_appearance_no_hook_excluded(tmp_path):
    clips = {0: _project_dict(0, overlays=[])}
    job_dir = _seed_job(tmp_path, clips)
    plan = plan_bulk_operation(
        job_dir=job_dir, clip_indices=[0],
        intent={"op": "hook_appearance", "params": {"position": "middle"}})
    item = plan["items"][0]
    assert item["outcome"] == OUTCOME_EXCLUDED
    assert "never invents hook content" in item["excluded"]["reason"]


def test_plan_hook_timing_adapts_to_each_clip_duration(tmp_path):
    clips = {
        0: _project_dict(0),  # 20s duration
        1: _project_dict(1, source={
            "file": "source_1.mp4", "width": 1920, "height": 1080,
            "duration": 2.0}),
    }
    job_dir = _seed_job(tmp_path, clips)
    plan = plan_bulk_operation(
        job_dir=job_dir, clip_indices=[0, 1],
        intent={"op": "hook_timing", "params": {"end": 5.0}})
    by_ci = {i["clip_index"]: i for i in plan["items"]}
    assert by_ci[0]["patch"] == {"hook.end": 5.0}
    assert by_ci[0]["adapted"] == []
    # Clip 1: end clamped to its 2s duration, disclosed.
    assert by_ci[1]["patch"] == {"hook.end": 2.0}
    assert by_ci[1]["adapted"][0]["requested"] == 5.0
    assert by_ci[1]["adapted"][0]["applied"] == 2.0


def test_plan_hook_timing_collapse_excluded(tmp_path):
    clips = {0: _project_dict(0, source={
        "file": "source_0.mp4", "width": 1920, "height": 1080,
        "duration": 1.0})}
    job_dir = _seed_job(tmp_path, clips)
    plan = plan_bulk_operation(
        job_dir=job_dir, clip_indices=[0],
        intent={"op": "hook_timing", "params": {"start": 2.0, "end": 5.0}})
    item = plan["items"][0]
    assert item["outcome"] == OUTCOME_EXCLUDED
    assert "collapses" in item["excluded"]["reason"]


def test_plan_hook_timing_grounded_opening_excluded(tmp_path):
    clips = {0: _project_dict(0, opening={
        "status": "selected", "option_id": "opt-1", "text": "grounded hook",
        "display_start": 0.0, "display_end": 2.5})}
    job_dir = _seed_job(tmp_path, clips)
    plan = plan_bulk_operation(
        job_dir=job_dir, clip_indices=[0],
        intent={"op": "hook_timing", "params": {"end": 4.0}})
    item = plan["items"][0]
    assert item["outcome"] == OUTCOME_EXCLUDED
    assert "evidence-bound" in item["excluded"]["reason"]


# --- planning: hyperframes ---------------------------------------------------

def test_plan_hyperframes_worker_down_is_pending_not_silent(tmp_path):
    job_dir = _seed_job(tmp_path, {0: _project_dict(0)})
    plan = plan_bulk_operation(
        job_dir=job_dir, clip_indices=[0],
        intent={"op": "hyperframes_finishing",
                "params": {"template": "clean-captions"}},
        worker_available={"available": False, "reason": "node not found"})
    item = plan["items"][0]
    assert item["outcome"] == OUTCOME_PENDING
    assert "node not found" in item["excluded"]["reason"]
    assert item["patch"] == {}


def test_check_hyperframes_clip_support_framing_and_lineage():
    # The Task-18 refusal conditions, unit-tested without a full project.
    from types import SimpleNamespace
    from clippyme.domain.bulk_edit import check_hyperframes_clip_support
    assert "framing trajectory" in check_hyperframes_clip_support(
        SimpleNamespace(framing=object(), reframed_base=None, audio=[],
                        music=None))
    assert "reframed_base lineage" in check_hyperframes_clip_support(
        SimpleNamespace(framing=None, reframed_base=object(), audio=[],
                        music=None))
    assert check_hyperframes_clip_support(
        SimpleNamespace(framing=None, reframed_base=None, audio=[],
                        music=None)) is None


def test_plan_hyperframes_reframed_lineage_unsupported(tmp_path):
    clips = {0: _project_dict(0, reframed_base={"base_ref": "clip_0.mp4"})}
    job_dir = _seed_job(tmp_path, clips)
    plan = plan_bulk_operation(
        job_dir=job_dir, clip_indices=[0],
        intent={"op": "hyperframes_finishing",
                "params": {"template": "clean-captions"}},
        worker_available={"available": True})
    item = plan["items"][0]
    assert item["outcome"] == OUTCOME_UNSUPPORTED
    assert "reframed_base lineage" in item["excluded"]["reason"]


def test_plan_hyperframes_configured_music_unsupported(tmp_path):
    clips = {0: _project_dict(0, audio=[
        {"file": "audio/track.mp3", "volume": 1.0, "offset": 0.0}])}
    job_dir = _seed_job(tmp_path, clips)
    plan = plan_bulk_operation(
        job_dir=job_dir, clip_indices=[0],
        intent={"op": "hyperframes_finishing",
                "params": {"template": "timed-opening"}},
        worker_available={"available": True})
    item = plan["items"][0]
    assert item["outcome"] == OUTCOME_UNSUPPORTED
    assert "configured music" in item["excluded"]["reason"]


# --- ledger + idempotency ----------------------------------------------------

def test_prepare_idempotent_same_key_and_payload(tmp_path):
    job_dir = _seed_job(tmp_path, {0: _project_dict(0)})
    kw = dict(job_dir=job_dir, job_id="job-1",
              scope={"mode": "selected", "clip_indices": [0]},
              intent={"op": "caption_appearance",
                      "params": {"caption_style": "hormozi_bold"}},
              idempotency_key="key-abc")
    op1 = create_operation(**kw)
    op2 = create_operation(**kw)
    assert op1["operation_id"] == op2["operation_id"]
    assert len(load_bulk_ledger(job_dir)["operations"]) == 1


def test_prepare_rejects_key_reuse_with_different_payload(tmp_path):
    job_dir = _seed_job(tmp_path, {0: _project_dict(0)})
    create_operation(
        job_dir=job_dir, job_id="job-1",
        scope={"mode": "selected", "clip_indices": [0]},
        intent={"op": "caption_appearance",
                "params": {"caption_style": "hormozi_bold"}},
        idempotency_key="key-abc")
    with pytest.raises(KeyError, match="idempotency_key_reused"):
        create_operation(
            job_dir=job_dir, job_id="job-1",
            scope={"mode": "selected", "clip_indices": [0]},
            intent={"op": "caption_appearance",
                    "params": {"caption_style": "neon_glow"}},
            idempotency_key="key-abc")


def test_prepare_freezes_base_versions(tmp_path):
    clips = {0: _project_dict(0, version=3), 1: _project_dict(1, version=7)}
    job_dir = _seed_job(tmp_path, clips)
    op = create_operation(
        job_dir=job_dir, job_id="job-1",
        scope={"mode": "selected", "clip_indices": [0, 1]},
        intent={"op": "caption_appearance",
                "params": {"caption_style": "hormozi_bold"}},
        idempotency_key="k-freeze")
    assert op["base_versions"] == {"0": 3, "1": 7}
    assert op["scope"]["clip_indices"] == [0, 1]
    assert op["status"] == "proposed"


# --- apply -------------------------------------------------------------------

def test_apply_bumps_versions_under_frozen_guards(tmp_path):
    clips = {0: _project_dict(0, version=2), 1: _project_dict(1, version=4)}
    job_dir = _seed_job(tmp_path, clips)
    op = create_operation(
        job_dir=job_dir, job_id="job-1",
        scope={"mode": "selected", "clip_indices": [0, 1]},
        intent={"op": "caption_appearance",
                "params": {"caption_style": "hormozi_bold"}},
        idempotency_key="k-apply-1")
    op = apply_operation(job_dir=job_dir, operation_id=op["operation_id"])
    assert op["status"] == "applied"
    assert op["items"]["0"]["outcome"] == OUTCOME_APPLIED
    assert op["items"]["0"]["result_version"] == 3
    assert op["items"]["1"]["result_version"] == 5
    assert latest_project_version(job_dir, 0) == 3
    # The applied project carries the deterministic item key.
    with open(os.path.join(job_dir, "clip-project-0.json"),
              encoding="utf-8") as f:
        assert json.load(f)["idempotency_key"] == bulk_item_key(
            op["operation_id"], 0)


def test_apply_is_idempotent_no_duplicate_versions(tmp_path):
    job_dir = _seed_job(tmp_path, {0: _project_dict(0, version=2)})
    op = create_operation(
        job_dir=job_dir, job_id="job-1",
        scope={"mode": "selected", "clip_indices": [0]},
        intent={"op": "caption_appearance",
                "params": {"caption_style": "hormozi_bold"}},
        idempotency_key="k-apply-2")
    op1 = apply_operation(job_dir=job_dir, operation_id=op["operation_id"])
    v1 = op1["items"]["0"]["result_version"]
    op2 = apply_operation(job_dir=job_dir, operation_id=op["operation_id"])
    assert op2["items"]["0"]["result_version"] == v1
    assert latest_project_version(job_dir, 0) == v1


def test_apply_partial_on_stale_base_version(tmp_path):
    clips = {0: _project_dict(0, version=2), 1: _project_dict(1, version=2)}
    job_dir = _seed_job(tmp_path, clips)
    op = create_operation(
        job_dir=job_dir, job_id="job-1",
        scope={"mode": "selected", "clip_indices": [0, 1]},
        intent={"op": "caption_appearance",
                "params": {"caption_style": "hormozi_bold"}},
        idempotency_key="k-partial")
    # Someone edits clip 1 after the proposal froze v2.
    edited = validate_project(_project_dict(1, version=2))
    edited.captions.style = "fire_impact"
    save_project_version(job_dir=job_dir, clip_index=1, project=edited,
                         expected_version=2)
    op = apply_operation(job_dir=job_dir, operation_id=op["operation_id"])
    assert op["status"] == "partial"
    assert op["items"]["0"]["outcome"] == OUTCOME_APPLIED
    assert op["items"]["1"]["outcome"] == OUTCOME_CONFLICT
    assert op["items"]["1"]["current_version"] == 3
    # The conflicting clip was NOT overwritten.
    with open(os.path.join(job_dir, "clip-project-1.json"),
              encoding="utf-8") as f:
        assert json.load(f)["captions"]["style"] == "fire_impact"


def test_crash_between_save_and_ledger_adopted_not_duplicated(tmp_path):
    """A save that landed before its ledger write is adopted on retry."""
    job_dir = _seed_job(tmp_path, {0: _project_dict(0, version=2),
                                   1: _project_dict(1, version=2)})
    op = create_operation(
        job_dir=job_dir, job_id="job-1",
        scope={"mode": "selected", "clip_indices": [0, 1]},
        intent={"op": "caption_appearance",
                "params": {"caption_style": "hormozi_bold"}},
        idempotency_key="k-crash")
    op_id = op["operation_id"]
    # Simulate the crashed apply: clip 0's save landed with the item key,
    # but the ledger was never updated (still "planned").
    key0 = bulk_item_key(op_id, 0)
    proj0 = validate_project(_project_dict(0, version=2))
    proj0.captions.style = "hormozi_bold"
    proj0.origin = "user"
    proj0.idempotency_key = key0
    v_landed, _ = save_project_version(job_dir=job_dir, clip_index=0,
                                       project=proj0, expected_version=2)
    assert find_version_by_item_key(job_dir, 0, key0) == v_landed
    # Now the "retry" (or duplicate request) must adopt, not duplicate.
    op = apply_operation(job_dir=job_dir, operation_id=op_id)
    item0 = op["items"]["0"]
    assert item0["outcome"] == OUTCOME_APPLIED
    assert item0["result_version"] == v_landed
    assert item0["recovered"] is True
    assert latest_project_version(job_dir, 0) == v_landed
    assert op["items"]["1"]["outcome"] == OUTCOME_APPLIED


# --- retry -------------------------------------------------------------------

def test_retry_only_eligible_items_never_reapplies(tmp_path, monkeypatch):
    import clippyme.domain.bulk_edit as be
    job_dir = _seed_job(tmp_path, {0: _project_dict(0, version=2),
                                   1: _project_dict(1, version=2)})
    op = create_operation(
        job_dir=job_dir, job_id="job-1",
        scope={"mode": "selected", "clip_indices": [0, 1]},
        intent={"op": "caption_appearance",
                "params": {"caption_style": "hormozi_bold"}},
        idempotency_key="k-retry")
    op_id = op["operation_id"]
    calls = {"n": 0}
    real_apply_patch = be.apply_plan_patch

    def flaky(project, patch, op_name):
        if project.clip_index == 1 and calls["n"] == 0:
            calls["n"] += 1
            raise RuntimeError("boom")
        return real_apply_patch(project, patch, op_name)

    monkeypatch.setattr(be, "apply_plan_patch", flaky)
    op = apply_operation(job_dir=job_dir, operation_id=op_id)
    assert op["items"]["0"]["outcome"] == OUTCOME_APPLIED
    assert op["items"]["1"]["outcome"] == OUTCOME_FAILED
    v0 = op["items"]["0"]["result_version"]

    op = retry_operation(job_dir=job_dir, operation_id=op_id)
    assert op["items"]["1"]["outcome"] == OUTCOME_APPLIED
    # Clip 0 was never reapplied: its version is untouched.
    assert op["items"]["0"]["result_version"] == v0
    assert latest_project_version(job_dir, 0) == v0


def test_retry_conflict_when_version_moved_not_silently_rebased(tmp_path):
    job_dir = _seed_job(tmp_path, {0: _project_dict(0, version=2)})
    op = create_operation(
        job_dir=job_dir, job_id="job-1",
        scope={"mode": "selected", "clip_indices": [0]},
        intent={"op": "caption_appearance",
                "params": {"caption_style": "hormozi_bold"}},
        idempotency_key="k-retry-conflict")
    op_id = op["operation_id"]
    # Fail the first apply by removing the project file mid-flight.
    os.remove(os.path.join(job_dir, "clip-project-0.json"))
    op = apply_operation(job_dir=job_dir, operation_id=op_id)
    assert op["items"]["0"]["outcome"] == OUTCOME_FAILED
    # Restore the project, then move it past the frozen base version.
    _seed_clip(job_dir, 0, _project_dict(0, version=2))
    edited = validate_project(_project_dict(0, version=2))
    edited.captions.style = "minimal_clean"
    save_project_version(job_dir=job_dir, clip_index=0, project=edited,
                         expected_version=2)
    op = retry_operation(job_dir=job_dir, operation_id=op_id)
    assert op["items"]["0"]["outcome"] == OUTCOME_CONFLICT
    with open(os.path.join(job_dir, "clip-project-0.json"),
              encoding="utf-8") as f:
        assert json.load(f)["captions"]["style"] == "minimal_clean"


# --- undo --------------------------------------------------------------------

def test_undo_restores_base_as_new_revision(tmp_path):
    job_dir = _seed_job(tmp_path, {0: _project_dict(0, version=2)})
    op = create_operation(
        job_dir=job_dir, job_id="job-1",
        scope={"mode": "selected", "clip_indices": [0]},
        intent={"op": "caption_appearance",
                "params": {"caption_style": "hormozi_bold"}},
        idempotency_key="k-undo")
    op_id = op["operation_id"]
    op = apply_operation(job_dir=job_dir, operation_id=op_id)
    assert op["items"]["0"]["result_version"] == 3
    op = undo_operation(job_dir=job_dir, operation_id=op_id)
    undo = op["undo"]
    assert undo["status"] == "undone"
    assert undo["items"]["0"]["outcome"] == "undone"
    assert undo["items"]["0"]["version"] == 4
    with open(os.path.join(job_dir, "clip-project-0.json"),
              encoding="utf-8") as f:
        proj = json.load(f)
    assert proj["captions"]["style"] == "classic_white"  # base restored
    assert proj["version"] == 4


def test_undo_conflict_when_later_edit_exists(tmp_path):
    job_dir = _seed_job(tmp_path, {0: _project_dict(0, version=2)})
    op = create_operation(
        job_dir=job_dir, job_id="job-1",
        scope={"mode": "selected", "clip_indices": [0]},
        intent={"op": "caption_appearance",
                "params": {"caption_style": "hormozi_bold"}},
        idempotency_key="k-undo-conflict")
    op_id = op["operation_id"]
    apply_operation(job_dir=job_dir, operation_id=op_id)
    # A later edit moves the clip past the operation's result version.
    later = validate_project(_project_dict(0, version=3))
    later.captions.style = "fire_impact"
    save_project_version(job_dir=job_dir, clip_index=0, project=later,
                         expected_version=3)
    op = undo_operation(job_dir=job_dir, operation_id=op_id)
    undo = op["undo"]
    assert undo["status"] == "undo_failed"
    assert undo["items"]["0"]["outcome"] == "conflict"
    # The later edit is untouched.
    with open(os.path.join(job_dir, "clip-project-0.json"),
              encoding="utf-8") as f:
        assert json.load(f)["captions"]["style"] == "fire_impact"


def test_undo_is_idempotent(tmp_path):
    job_dir = _seed_job(tmp_path, {0: _project_dict(0, version=2)})
    op = create_operation(
        job_dir=job_dir, job_id="job-1",
        scope={"mode": "selected", "clip_indices": [0]},
        intent={"op": "caption_appearance",
                "params": {"caption_style": "hormozi_bold"}},
        idempotency_key="k-undo-idem")
    op_id = op["operation_id"]
    apply_operation(job_dir=job_dir, operation_id=op_id)
    op = undo_operation(job_dir=job_dir, operation_id=op_id)
    v_undo = op["undo"]["items"]["0"]["version"]
    op = undo_operation(job_dir=job_dir, operation_id=op_id)
    assert op["undo"]["items"]["0"]["outcome"] == "undone"
    assert op["undo"]["items"]["0"]["version"] == v_undo
    assert latest_project_version(job_dir, 0) == v_undo


def test_undo_skips_non_applied_items(tmp_path):
    job_dir = _seed_job(tmp_path, {0: _project_dict(0, version=2)})
    op = create_operation(
        job_dir=job_dir, job_id="job-1",
        scope={"mode": "selected", "clip_indices": [0]},
        intent={"op": "caption_appearance",
                "params": {"caption_style": "classic_white"}},
        idempotency_key="k-undo-skip")
    op_id = op["operation_id"]
    apply_operation(job_dir=job_dir, operation_id=op_id)
    assert op["items"]["0"]["outcome"] == OUTCOME_UNCHANGED
    op = undo_operation(job_dir=job_dir, operation_id=op_id)
    assert op["undo"]["status"] == "nothing_to_undo"
    assert op["undo"]["items"] == {}
