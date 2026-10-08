"""Apply validated Task 09 editorial plans to the clips that render (Task 10).

Pure, deterministic post-processing for the Task 08/09 quality path.
``run_windowed_discovery`` calls :func:`apply_editorial_plans` exactly
once -- after Task 09 returns validated editorial plans and before the
final ``shorts`` list reaches the render loops.

Contract:

- Inputs are never mutated. The returned shorts list is built from
  deep copies; the caller's ``shorts``, ``candidate_store`` and
  ``editorial_plans`` are left byte-for-byte untouched.
- Strict identity join per short: a short is applicable only with a
  unique non-empty ``candidate_id``, a candidate-store record carrying
  the exact same ``candidate_id``, and exactly one matching non-group
  plan carrying the current plan version and a matching moment
  fingerprint. The store interval and the selected short interval must
  each be finite, non-boolean, and ordered, and must agree with each
  other. The plan interval must describe the current candidate:
  accept/reject/unrefined retain it exactly, adjust must overlap it.
  The source duration must be finite and positive for the upper bound
  check; an unknown duration never authorizes a negative cut. Every
  numeric value must be a finite non-boolean; source and resolved
  duration bounds are revalidated. Duplicates, missing/malformed
  plans, fingerprint mismatches, invalid intervals, identity and
  interval mismatches, and contradictory state are isolated per item
  with an exact reason -- they never crash the run and never erase
  unrelated clips.
- Fail-safe base semantics: refined ``accept`` retains the original
  short and interval; refined ``adjust`` copies the short and replaces
  only ``start``/``end`` (every other field -- titles, descriptions,
  hook, hashtags, score, reason, speaker metadata, and unknown additive
  fields -- is preserved); refined ``reject`` removes the short with an
  auditable reason; ``unrefined`` keeps the original short and records
  that the discovery interval was retained; invalid/missing application
  data keeps the original short, marks it ``application_skipped`` with
  the exact reason, and never silently approves or rejects it.
  Candidate-store items absent from the final shorts are never promoted,
  and no quota padding happens here.
- Continuity groups: a group plan replaces its component shorts only
  when the group is a refined adjust plan, has a non-empty unique
  component list,
  every component is present in the current final shorts, every
  component survived as refined accept/adjust, the group interval is
  finite, in source bounds, policy-compliant, and covers all applied
  component intervals, and the group identity does not conflict with a
  base ID. The group is inserted at the earliest component position,
  components are removed once, ordering stays deterministic, and
  ``component_candidate_ids`` is exposed. Display fields are copied
  from the earliest component -- no hook/title/description is invented.
  The group's editorial recommendation is kept separately and never
  relabeled as measured virality. Any failed group check retains the
  individual component shorts and records the group rejection.
- Honest funnel: the returned ``editorial_application`` record carries
  the version/flag, original selected count, plans supplied, accepts
  retained, adjustments applied, rejects removed, unrefined retained,
  application skips by reason, groups proposed/applied/rejected with
  reasons, the final selected count, exact reconciliation, and the
  original + applied intervals for every changed/removed/grouped ID.
  No provider responses and no secrets are ever recorded.

Rollback: ``CLIPPYME_APPLY_EDITORIAL_PLANS`` (default ON) gates the
step. The caller additionally passes ``editorial_ran`` so application
stays off whenever Task 09 did not run on the quality path. When the
step is off, the original shorts are returned unchanged and an explicit
disabled record is produced.

No provider calls are made; identical inputs always produce identical
outputs and funnels.
"""
from __future__ import annotations

import copy
from typing import Any, Dict, List, Mapping, Optional, Tuple

from clippyme.pipeline.candidate_discovery import (
    _env_flag,
    _is_finite_number,
)
from clippyme.pipeline.editorial_refinement import EDITORIAL_VERSION

APPLICATION_VERSION = "1"

APPLICATION_FLAG = "CLIPPYME_APPLY_EDITORIAL_PLANS"

_BASE_VERDICTS = ("accept", "adjust", "reject", "unrefined")

# Outcomes that keep a component eligible for continuity grouping.
_GROUP_ELIGIBLE = ("accepted", "adjusted")


def application_enabled() -> bool:
    """Rollback flag for the Task 10 application step; default ON."""
    return _env_flag(APPLICATION_FLAG, True)


def _is_group_plan(plan: Mapping[str, Any]) -> bool:
    cid = plan.get("candidate_id")
    return isinstance(cid, str) and cid.startswith("group:")


def _as_float(value: Any) -> Optional[float]:
    if _is_finite_number(value):
        return float(value)
    return None


def _ordered_interval(start_v: Any, end_v: Any) -> Optional[Tuple[float, float]]:
    """Finite non-boolean ordered interval, or None when malformed."""
    ss = _as_float(start_v)
    se = _as_float(end_v)
    if ss is None or se is None:
        return None
    if not ss < se:
        return None
    return (ss, se)


def _strict_count(rec: Mapping[str, Any], key: str) -> Optional[int]:
    """Non-boolean nonnegative integer count, or None when malformed.

    Booleans, strings, floats, and missing values never coerce: only an
    actual integer count is trusted.
    """
    value = rec.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    if value < 0:
        return None
    return value


def is_intentional_editorial_empty(clips_data: Any) -> bool:
    """Whether enabled editorial application honestly reduced a non-empty
    selection to zero shorts.

    Such results must be persisted as zero shorts with no fallback
    rendering: the empty set is the validated editorial outcome, not a
    discovery/provider failure.

    Strict contract: only a current APPLICATION_VERSION record with
    exact boolean flags, non-boolean nonnegative integer counts, an
    actual empty shorts list, empty application_skips and skipped
    fields, and independently verified zero-output arithmetic may
    suppress fallback. Every original selection must have been actually
    rejected; nothing may be retained, adjusted, grouped, or skipped --
    every skipped short is retained, so any skip contradicts zero
    output. Stale versions, type-coerced counts, wrong-typed or missing
    skip fields, or inconsistent metadata never suppress fallback.
    """
    if not isinstance(clips_data, dict):
        return False
    shorts = clips_data.get("shorts")
    if not isinstance(shorts, list) or len(shorts) != 0:
        return False
    rec = clips_data.get("editorial_application")
    if not isinstance(rec, Mapping):
        return False
    if rec.get("version") != APPLICATION_VERSION:
        return False
    if rec.get("enabled") is not True:
        return False
    if rec.get("counts_reconcile") is not True:
        return False
    orig = _strict_count(rec, "original_selected_count")
    final = _strict_count(rec, "final_selected_count")
    accepts = _strict_count(rec, "accepts_retained")
    adjustments = _strict_count(rec, "adjustments_applied")
    rejects = _strict_count(rec, "rejects_removed")
    unrefined = _strict_count(rec, "unrefined_retained")
    legacy = _strict_count(rec, "legacy_no_id_retained")
    grouped = _strict_count(rec, "groups_applied")
    counts = (orig, final, accepts, adjustments, rejects, unrefined,
              legacy, grouped)
    if any(c is None for c in counts):
        return False
    if not orig > 0:
        return False
    if final != 0:
        return False
    # No skipped item may coexist with an intentionally empty result:
    # every skipped short is retained, so any skip contradicts zero
    # output. application_skips must be an actual empty mapping and
    # skipped an actual empty list; missing fields, wrong types, or
    # non-empty values preserve the genuine-failure fallback path.
    skips = rec.get("application_skips")
    if not isinstance(skips, Mapping) or len(skips) != 0:
        return False
    skipped = rec.get("skipped")
    if not isinstance(skipped, list) or len(skipped) != 0:
        return False
    # Independently verified zero-output arithmetic: every original
    # selection was actually rejected and nothing was retained,
    # adjusted, or grouped. counts_reconcile alone is not trusted.
    if rejects != orig:
        return False
    return (accepts == 0 and adjustments == 0 and unrefined == 0
            and legacy == 0 and grouped == 0)


def disabled_application_record(reason: str,
                                original_count: int = 0) -> Dict[str, Any]:
    """Reconciled record for a disabled / not-run application step."""
    original_count = int(original_count)
    return {
        "version": APPLICATION_VERSION,
        "enabled": False,
        "disabled_reason": reason,
        "flag": APPLICATION_FLAG,
        "original_selected_count": original_count,
        "plans_supplied": 0,
        "base_plans_supplied": 0,
        "group_plans_supplied": 0,
        "plans_ignored_not_selected": 0,
        "accepts_retained": 0,
        "adjustments_applied": 0,
        "rejects_removed": 0,
        "unrefined_retained": 0,
        "legacy_no_id_retained": 0,
        "application_skips": {},
        "skipped": [],
        "groups_proposed": 0,
        "groups_applied": 0,
        "groups_rejected": 0,
        "group_rejections": [],
        "final_selected_count": original_count,
        "interval_changes": [],
        "counts_reconcile": True,
    }


def fallback_application_record(superseded_record: Any,
                                strategy: str,
                                final_shorts_count: int = 0) -> Dict[str, Any]:
    """Record for fallback shortlists (TextTiling / whole-video).

    Fallback shorts were not selected by windowed discovery, so Task 09
    plans must NOT be applied to them. The earlier application record is
    preserved for audit inside ``superseded_record``; this record states
    explicitly that no plan was applied to the fallback shorts.
    """
    return {
        "version": APPLICATION_VERSION,
        "enabled": False,
        "disabled_reason": "fallback_shorts_have_no_editorial_plans",
        "fallback_strategy": strategy,
        "fallback_note": (
            "Fallback shorts were not selected by windowed discovery; "
            "Task 09 editorial plans were NOT applied to them."
        ),
        "superseded_record": (superseded_record
                              if isinstance(superseded_record, Mapping)
                              else None),
        "original_selected_count": 0,
        "final_selected_count": int(final_shorts_count),
        "counts_reconcile": True,
    }


def preserve_application_through_fallback(data: Any,
                                          app_record: Any,
                                          strategy: str) -> Any:
    """Attach fallback-marked application metadata to a fallback result.

    ``data`` is a fresh fallback result dict owned by the caller (it is
    mutated and returned). Fallback shorts never inherit stale plans:
    the record states the plans were not applied, while the earlier
    record survives for audit under ``superseded_record``.
    """
    if not isinstance(data, dict):
        return data
    shorts = data.get("shorts")
    count = len(shorts) if isinstance(shorts, list) else 0
    data["editorial_application"] = fallback_application_record(
        app_record, strategy, count)
    return data


def _validate_plan_boundaries(plan: Mapping[str, Any],
                              source_duration: Optional[float],
                              duration_min: float,
                              duration_max: float,
                              source_checked: bool) -> Tuple[
                                  Optional[Tuple[float, float]],
                                  Optional[str]]:
    """Validate a plan's start_s/end_s; return ((ss, se), None) or reason."""
    ss = _as_float(plan.get("start_s"))
    se = _as_float(plan.get("end_s"))
    if ss is None or se is None:
        return None, "non_finite_plan_boundaries"
    if not ss < se:
        return None, "reversed_plan_boundaries"
    if ss < 0.0:
        # A negative cut is never authorized. When the source duration
        # is invalid the upper bound cannot be verified either; report
        # the missing duration instead of silently skipping the check.
        return None, ("plan_boundaries_out_of_source" if source_checked
                      else "invalid_source_duration")
    if source_checked and se > source_duration:
        return None, "plan_boundaries_out_of_source"
    duration = se - ss
    if not (duration_min <= duration <= duration_max):
        return None, "plan_boundaries_out_of_policy"
    return (ss, se), None


def _validate_base_plan(plan: Mapping[str, Any],
                        store_rec: Mapping[str, Any],
                        source_duration: Optional[float],
                        duration_min: float,
                        duration_max: float,
                        source_checked: bool) -> Tuple[
                            Optional[Tuple[float, float]], Optional[str]]:
    """Strict identity + state + interval join for one base plan."""
    if plan.get("plan_version") != EDITORIAL_VERSION:
        return None, "unknown_plan_version"
    verdict = plan.get("verdict")
    if verdict not in _BASE_VERDICTS:
        return None, "unknown_verdict"
    status = plan.get("status")
    if status not in ("refined", "unrefined"):
        return None, "unknown_status"
    if (status == "refined") != (verdict in ("accept", "adjust", "reject")):
        return None, "contradictory_plan_state"
    if (plan.get("moment_fingerprint") or "") != (
            store_rec.get("moment_fingerprint") or ""):
        return None, "moment_fingerprint_mismatch"
    return _validate_plan_boundaries(
        plan, source_duration, duration_min, duration_max, source_checked)


def _validate_group_plan(group: Mapping[str, Any],
                         pos_of: Mapping[str, int],
                         outcome_of: Mapping[str, str],
                         applied_interval: Mapping[str, Any],
                         source_duration: Optional[float],
                         duration_min: float,
                         duration_max: float,
                         source_checked: bool) -> Tuple[
                             Optional[Tuple[List[str], float, float]],
                             Optional[str]]:
    """Validate one continuity group plan against current shorts."""
    gid = group.get("candidate_id")
    if not source_checked:
        # No group may apply without a valid source duration.
        return None, "invalid_source_duration"
    if group.get("status") != "refined":
        return None, "group_not_refined"
    if group.get("verdict") != "adjust":
        # Only a genuine refined adjustment plan may replace components;
        # a reject/accept/unrefined group plan never does.
        return None, "group_invalid_verdict"
    if group.get("plan_version") != EDITORIAL_VERSION:
        return None, "group_unknown_plan_version"
    comps = group.get("components")
    if not isinstance(comps, list) or not comps:
        return None, "group_empty_components"
    if any(not isinstance(c, str) or not c for c in comps):
        return None, "group_invalid_component_id"
    if len(set(comps)) != len(comps):
        return None, "group_duplicate_components"
    for c in comps:
        oc = outcome_of.get(c)
        if oc is None:
            return None, f"group_missing_component:{c}"
        if oc not in _GROUP_ELIGIBLE:
            return None, f"group_component_not_eligible:{c}:{oc}"
    for c in comps:
        if c not in pos_of:
            return None, f"group_missing_component:{c}"
    for c in comps:
        iv = applied_interval.get(c)
        if (not isinstance(iv, (list, tuple)) or len(iv) != 2
                or not _is_finite_number(iv[0])
                or not _is_finite_number(iv[1])):
            return None, f"group_component_interval_unknown:{c}"
    gss = _as_float(group.get("start_s"))
    gse = _as_float(group.get("end_s"))
    if gss is None or gse is None:
        return None, "group_non_finite_boundaries"
    if not gss < gse:
        return None, "group_reversed_boundaries"
    if source_checked and not (gss >= 0.0 and gse <= source_duration):
        return None, "group_boundaries_out_of_source"
    if not (duration_min <= gse - gss <= duration_max):
        return None, "group_boundaries_out_of_policy"
    lo = min(float(applied_interval[c][0]) for c in comps)
    hi = max(float(applied_interval[c][1]) for c in comps)
    if not (gss <= lo + 1e-9 and gse >= hi - 1e-9):
        return None, "group_interval_does_not_cover_components"
    if gid in pos_of:
        return None, "group_id_conflicts_with_base_id"
    return (list(comps), gss, gse), None


def apply_editorial_plans(
    shorts: Any,
    candidate_store: Any,
    editorial_plans: Any,
    source_duration: Any,
    duration_min: Any,
    duration_max: Any,
    *,
    enabled: Optional[bool] = None,
    editorial_ran: bool = True,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Apply validated editorial plans to discovery shorts.

    Returns ``(new_shorts, editorial_application_record)``. Never
    mutates the inputs. Deterministic: identical inputs always yield
    identical outputs and funnels.
    """
    # Deep-copy first: every output short derives from these copies, so
    # the caller's objects are never touched.
    work: List[Any] = (copy.deepcopy(list(shorts))
                       if isinstance(shorts, (list, tuple)) else [])
    n_original = len(work)

    flag_on = application_enabled() if enabled is None else bool(enabled)
    if not flag_on:
        return work, disabled_application_record("flag_disabled",
                                                 n_original)
    if not editorial_ran:
        return work, disabled_application_record(
            "editorial_refinement_not_run", n_original)

    dmin = _as_float(duration_min)
    dmax = _as_float(duration_max)
    bounds_ok = (dmin is not None and dmax is not None
                 and dmin > 0.0 and dmin <= dmax)
    source_checked = _is_finite_number(source_duration) and float(
        source_duration) > 0.0
    src_dur = float(source_duration) if source_checked else None

    store = (candidate_store if isinstance(candidate_store, Mapping)
             else {})
    plans: List[Any] = (list(editorial_plans)
                        if isinstance(editorial_plans, (list, tuple))
                        else [])

    # Index non-group plans by candidate_id; collect group plans.
    base_by_id: Dict[str, List[Mapping[str, Any]]] = {}
    group_plans: List[Mapping[str, Any]] = []
    for p in plans:
        if not isinstance(p, Mapping):
            continue
        if _is_group_plan(p):
            group_plans.append(p)
            continue
        pid = p.get("candidate_id")
        if isinstance(pid, str) and pid:
            base_by_id.setdefault(pid, []).append(p)

    # --- Degraded path: resolved duration bounds are unusable. ---
    # Fail-safe: retain every short unchanged, mark each skipped with
    # the exact reason, reconcile honestly. Never crash, never erase.
    if not bounds_ok:
        skips: Dict[str, int] = {}
        skipped: List[Dict[str, str]] = []
        for s in work:
            cid = (s.get("candidate_id") if isinstance(s, Mapping)
                   else None)
            if isinstance(s, Mapping):
                s["application_skipped"] = "invalid_duration_bounds"
            skips["invalid_duration_bounds"] = skips.get(
                "invalid_duration_bounds", 0) + 1
            skipped.append({"candidate_id": cid if isinstance(
                cid, str) else "", "reason": "invalid_duration_bounds"})
        record = disabled_application_record(
            "invalid_duration_bounds", n_original)
        record["enabled"] = True
        record["editorial_refinement_ran"] = bool(editorial_ran)
        record["plans_supplied"] = len(plans)
        record["application_skips"] = dict(sorted(skips.items()))
        record["skipped"] = sorted(
            skipped, key=lambda d: d["candidate_id"])
        record["counts_reconcile"] = (
            n_original == sum(skips.values()) == len(work))
        return work, record

    # --- Base pass (input order; deterministic). ---
    id_counts: Dict[str, int] = {}
    for s in work:
        cid = s.get("candidate_id") if isinstance(s, Mapping) else None
        if isinstance(cid, str) and cid.strip():
            id_counts[cid] = id_counts.get(cid, 0) + 1

    out: List[Any] = []
    outcome_of: Dict[str, str] = {}
    applied_interval: Dict[str, Any] = {}
    accepts = adjustments = rejects = unrefined = legacy = 0
    skips: Dict[str, int] = {}
    skipped: List[Dict[str, str]] = []
    interval_changes: List[Dict[str, Any]] = []

    def _skip(short: Any, cid: str, reason: str) -> None:
        short["application_skipped"] = reason
        out.append(short)
        skips[reason] = skips.get(reason, 0) + 1
        skipped.append({"candidate_id": cid, "reason": reason})
        outcome_of[cid] = "skipped"

    for s in work:
        cid = s.get("candidate_id") if isinstance(s, Mapping) else None
        if not isinstance(s, Mapping) or not (
                isinstance(cid, str) and cid.strip()):
            # Legacy / non-windowed short without a usable candidate ID:
            # not applicable, retained byte-identical (no annotation).
            out.append(s)
            legacy += 1
            continue
        if not source_checked:
            # A missing, boolean, non-finite, zero, or negative source
            # duration cannot approve or change any Task 10 clip: retain
            # each original short with an exact reason.
            _skip(s, cid, "invalid_source_duration")
            continue
        if id_counts[cid] > 1:
            _skip(s, cid, "duplicate_candidate_id")
            continue
        rec = store.get(cid)
        if not isinstance(rec, Mapping):
            _skip(s, cid, "missing_candidate_store_entry")
            continue
        # Strict base join: the store record must carry the exact same
        # candidate_id, and the store and short intervals must each be
        # finite, non-boolean, and ordered, and must agree with each
        # other. Anything else keeps the original short with an exact
        # application_skipped reason and never touches the counts.
        if rec.get("candidate_id") != cid:
            _skip(s, cid, "candidate_store_id_mismatch")
            continue
        store_iv = _ordered_interval(rec.get("start_s"), rec.get("end_s"))
        if store_iv is None:
            _skip(s, cid, "invalid_store_boundaries")
            continue
        short_iv = _ordered_interval(s.get("start"), s.get("end"))
        if short_iv is None:
            _skip(s, cid, "invalid_short_boundaries")
            continue
        if short_iv != store_iv:
            _skip(s, cid, "short_store_interval_mismatch")
            continue
        # Current evidence must sit inside the source: a candidate whose
        # selected short or store interval escapes [0, source_duration]
        # is not authoritative. (source_checked is True here, so src_dur
        # is a finite positive number.)
        if (store_iv[0] < 0.0 or store_iv[1] > src_dur
                or short_iv[0] < 0.0 or short_iv[1] > src_dur):
            _skip(s, cid, "candidate_boundaries_out_of_source")
            continue
        matches = base_by_id.get(cid, [])
        if not matches:
            _skip(s, cid, "no_matching_plan")
            continue
        if len(matches) > 1:
            _skip(s, cid, "duplicate_matching_plans")
            continue
        plan = matches[0]
        (interval, reason) = _validate_base_plan(
            plan, rec, src_dur, dmin, dmax, source_checked)
        if reason is not None:
            _skip(s, cid, reason)
            continue
        pss, pse = interval
        verdict = plan.get("verdict")
        # The plan must describe the current candidate: accept/reject/
        # unrefined retain its exact interval; adjust must overlap it.
        if verdict == "adjust":
            if not (pss < store_iv[1] and pse > store_iv[0]):
                _skip(s, cid, "plan_interval_disjoint_from_candidate")
                continue
        elif not (abs(pss - store_iv[0]) <= 1e-9
                  and abs(pse - store_iv[1]) <= 1e-9):
            _skip(s, cid, "plan_interval_mismatch")
            continue
        orig_start, orig_end = short_iv
        if verdict == "accept":
            s["editorial_application"] = {
                "verdict": "accept",
                "action": "retained",
                "start_s": orig_start,
                "end_s": orig_end,
            }
            out.append(s)
            accepts += 1
            outcome_of[cid] = "accepted"
            applied_interval[cid] = ((orig_start, orig_end)
                                    if orig_start is not None
                                    and orig_end is not None else None)
        elif verdict == "adjust":
            s["start"] = pss
            s["end"] = pse
            s["editorial_application"] = {
                "verdict": "adjust",
                "action": "adjusted",
                "original_start_s": orig_start,
                "original_end_s": orig_end,
                "applied_start_s": pss,
                "applied_end_s": pse,
            }
            out.append(s)
            adjustments += 1
            outcome_of[cid] = "adjusted"
            applied_interval[cid] = (pss, pse)
            interval_changes.append({
                "candidate_id": cid,
                "action": "adjusted",
                "original_start_s": orig_start,
                "original_end_s": orig_end,
                "applied_start_s": pss,
                "applied_end_s": pse,
            })
        elif verdict == "reject":
            # Removed with an auditable reason; never reaches cutting.
            rejects += 1
            outcome_of[cid] = "rejected"
            interval_changes.append({
                "candidate_id": cid,
                "action": "removed",
                "original_start_s": orig_start,
                "original_end_s": orig_end,
                "applied_start_s": None,
                "applied_end_s": None,
                "reason": plan.get("reason") or "",
            })
        else:  # unrefined: discovery interval honestly retained.
            s["editorial_application"] = {
                "verdict": "unrefined",
                "action": "retained",
                "note": "discovery_interval_retained",
                "start_s": orig_start,
                "end_s": orig_end,
            }
            out.append(s)
            unrefined += 1
            outcome_of[cid] = "unrefined"
            applied_interval[cid] = ((orig_start, orig_end)
                                    if orig_start is not None
                                    and orig_end is not None else None)

    # --- Continuity-group pass (deterministic proposal order). ---
    def _pos() -> Dict[str, int]:
        return {s.get("candidate_id"): i for i, s in enumerate(out)
                if isinstance(s, Mapping)
                and isinstance(s.get("candidate_id"), str)}

    pos_of = _pos()

    def _group_key(g: Mapping[str, Any]) -> Tuple[float, str]:
        comps = g.get("components")
        best = None
        if isinstance(comps, list):
            for c in comps:
                if c in pos_of and (best is None or pos_of[c] < best):
                    best = pos_of[c]
        return (float(best) if best is not None else float("inf"),
                str(g.get("candidate_id") or ""))

    groups_proposed = groups_applied = groups_rejected = 0
    group_rejections: List[Dict[str, Any]] = []
    components_consumed = 0
    for g in sorted(group_plans, key=_group_key):
        groups_proposed += 1
        gid = g.get("candidate_id")
        gid_s = gid if isinstance(gid, str) else ""
        comps_for_log = (list(g.get("components"))
                         if isinstance(g.get("components"), list) else [])
        (valid, reason) = _validate_group_plan(
            g, pos_of, outcome_of, applied_interval,
            src_dur, dmin, dmax, source_checked)
        if reason is not None:
            groups_rejected += 1
            group_rejections.append({
                "group_id": gid_s,
                "components": comps_for_log,
                "reason": reason,
            })
            continue
        comps, gss, gse = valid
        # Components ordered by current position; earliest donates the
        # display fields (no invented hook/title/description).
        ordered = [c for _, c in sorted((pos_of[c], c) for c in comps)]
        insert_at = pos_of[ordered[0]]
        gs = copy.deepcopy(out[insert_at])
        gs["start"] = gss
        gs["end"] = gse
        gs["candidate_id"] = gid_s
        gs["component_candidate_ids"] = ordered
        gs.pop("application_skipped", None)
        # The group's editorial recommendation stays separate; the
        # copied viral_score/viral_reason describe the source moment.
        if g.get("recommendation_score") is not None:
            gs["editorial_recommendation_score"] = g.get(
                "recommendation_score")
        gs["editorial_application"] = {
            "verdict": g.get("verdict"),
            "action": "grouped",
            "group_id": gid_s,
            "component_candidate_ids": ordered,
            "applied_start_s": gss,
            "applied_end_s": gse,
            "reason": g.get("reason") or "",
        }
        for ix in sorted((pos_of[c] for c in comps), reverse=True):
            del out[ix]
        out.insert(insert_at, gs)
        for c in comps:
            outcome_of[c] = "grouped"
        components_consumed += len(comps)
        groups_applied += 1
        pos_of = _pos()
        interval_changes.append({
            "candidate_id": gid_s,
            "action": "grouped",
            "component_intervals": {
                c: [float(applied_interval[c][0]),
                    float(applied_interval[c][1])] for c in ordered},
            "applied_start_s": gss,
            "applied_end_s": gse,
        })

    # --- Honest funnel. ---
    shorts_cids = set(id_counts)
    plans_ignored = 0
    for cid, plist in base_by_id.items():
        if cid not in shorts_cids:
            plans_ignored += len(plist)
    skips_total = sum(skips.values())
    c1 = (n_original == accepts + adjustments + rejects + unrefined
          + skips_total + legacy)
    c2 = (len(out) == accepts + adjustments + unrefined + skips_total
          + legacy + groups_applied - components_consumed)
    c3 = (groups_proposed == groups_applied + groups_rejected)
    record = {
        "version": APPLICATION_VERSION,
        "enabled": True,
        "flag": APPLICATION_FLAG,
        "editorial_refinement_ran": bool(editorial_ran),
        "source_duration_s": src_dur,
        "duration_min_s": dmin,
        "duration_max_s": dmax,
        "source_bounds_checked": bool(source_checked),
        "original_selected_count": n_original,
        "plans_supplied": len(plans),
        "base_plans_supplied": sum(len(v)
                                   for v in base_by_id.values()),
        "group_plans_supplied": len(group_plans),
        "plans_ignored_not_selected": plans_ignored,
        "accepts_retained": accepts,
        "adjustments_applied": adjustments,
        "rejects_removed": rejects,
        "unrefined_retained": unrefined,
        "legacy_no_id_retained": legacy,
        "application_skips": dict(sorted(skips.items())),
        "skipped": sorted(skipped,
                          key=lambda d: d["candidate_id"]),
        "groups_proposed": groups_proposed,
        "groups_applied": groups_applied,
        "groups_rejected": groups_rejected,
        "group_rejections": group_rejections,
        "final_selected_count": len(out),
        "interval_changes": sorted(
            interval_changes,
            key=lambda d: str(d.get("candidate_id"))),
        "counts_reconcile": bool(c1 and c2 and c3),
    }
    return out, record
