"""Task 19 -- version-aware bulk editing: typed operations + operation ledger.

A bulk operation proposes ONE shared caption/hook style change across several
clips of a job, freezes the exact scope (ordered unique clip IDs + the
acknowledged base version of every clip), applies it per clip under the
existing optimistic-concurrency contract, and records everything in a
versioned, idempotent operation ledger.

Design rules (TASK.md scope 1-7):
- The proposal freezes job_id, ordered unique clip indices and each clip's
  acknowledged base version. "All" means the enumerated current-job snapshot
  at prepare time, never clips added later.
- Typed operations only touch appearance fields that already exist on the
  project: captions.style / captions.position, the hook overlay's style /
  position / start / end, and the optional Hyperframes finishing selection.
  Transcript words, hook text, cuts, source, crop, framing, music and asset
  references are never copied or invented by a shared style.
- Individual overrides are preserved by default: for each changed field the
  planner computes the scope's common baseline (mode of current values); a
  clip that differs from BOTH the baseline and the target keeps its value
  and is disclosed as preserved. Explicit ``on_override="replace"`` replaces
  them, and the proposal identifies every affected field.
- Hook timing is adapted per clip (end clamped to the clip duration) using
  the existing Overlay validators; adaptations, exclusions and conflicts are
  disclosed per clip, never silent.
- The optional Hyperframes bulk setting uses only Task 18's supported
  templates and capability checks. Tracked framing, reframed lineage,
  configured music or a disabled worker produce an explicit per-clip
  unsupported/pending outcome -- footage is never silently rematerialized.
- Apply goes through the ONE save authority,
  ``project_render.save_project_version``, with each clip's frozen expected
  version. No blind reload-and-overwrite on 409.
- The ledger (``bulk-operations.json`` next to ``project-versions.json``)
  records operation identity, frozen scope, typed intent identity,
  base/result versions and per-clip outcomes. It is written atomically
  (unique tmp + os.replace, strict finite JSON) after every per-clip step.
  EVERY whole-ledger read/modify/write transaction -- create, apply
  initial/final, retry reset, undo init/final, per-item updates -- runs
  inside the shared per-job serialization boundary, loading the ledger
  fresh inside the lock and merging only its own operation's fields, so a
  concurrent operation can neither be erased nor erase another (in-process
  only; the API serves these calls from one process, no cross-process
  guarantee is claimed). Lock order is always operation_lock -> job lock;
  no ledger snapshot loaded outside the boundary is held across another
  job transaction. A crash between a project
  save and the ledger write is recoverable: each applied clip's project
  carries a deterministic item idempotency key
  (``bulk:<operation_id>:<clip_index>``), and apply detects an already-landed
  save by scanning the version map instead of saving again. The exact
  pre-apply base snapshot is persisted BEFORE the save, so a recovered
  apply still has undo provenance; a landed undo save is recovered by its
  own deterministic key (``bulk:<operation_id>:undo:<clip_index>``) before
  any later-edit conflict is reported.
- An idempotency key binds to the exact operation payload (sha256 over the
  canonical scope+intent). Reusing a key with a different payload is
  rejected; replaying the same key+payload returns the recorded outcome and
  never creates a duplicate version.
- Partial success stays partial: failures are isolated, retry only touches
  eligible unresolved items under the ORIGINAL version guards, applied items
  are never reapplied.
- Undo only targets successfully applied clips whose current version still
  equals the operation's acknowledged result version. It restores the frozen
  base snapshot through the versioned save path (a new acknowledged
  revision); later edits surface as explicit undo conflicts. Repeated undo
  is idempotent via the recorded undo outcomes.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import threading
import time
import uuid

from clippyme.domain.clip_project import (
    CAPTION_POSITIONS,
    CAPTION_STYLES,
    HyperframesFinishing,
    project_from_json,
    validate_project,
)
from clippyme.domain.errors import ConflictError
from clippyme.domain.project_render import (
    _load_version_map,
    get_project_path,
    latest_project_version,
    save_project_version,
)

BULK_LEDGER_FILENAME = "bulk-operations.json"
"""Per-job operation ledger, next to project-versions.json."""

# ---------------------------------------------------------------------------
# Typed operations
# ---------------------------------------------------------------------------

OP_CAPTION_APPEARANCE = "caption_appearance"
OP_HOOK_APPEARANCE = "hook_appearance"
OP_HOOK_TIMING = "hook_timing"
OP_HYPERFRAMES_FINISHING = "hyperframes_finishing"

BULK_OPERATIONS = frozenset({
    OP_CAPTION_APPEARANCE,
    OP_HOOK_APPEARANCE,
    OP_HOOK_TIMING,
    OP_HYPERFRAMES_FINISHING,
})

ON_OVERRIDE_PRESERVE = "preserve"
ON_OVERRIDE_REPLACE = "replace"
ON_OVERRIDE_MODES = frozenset({ON_OVERRIDE_PRESERVE, ON_OVERRIDE_REPLACE})

HOOK_OVERLAY_STYLES = frozenset({"viral"})
"""Closed hook-overlay style set: the only style the pipeline emits."""

HOOK_POSITIONS = frozenset({"top", "middle", "bottom"})

HF_CAPTION_SIZES = frozenset({"small", "medium", "large"})
HF_CAPTION_POSITIONS = frozenset({"lower-third", "middle"})

MAX_BULK_CLIPS = 50

# Per-clip outcomes.
OUTCOME_PLANNED = "planned"        # frozen proposal, not yet attempted
OUTCOME_APPLIED = "applied"        # saved; result_version acknowledged
OUTCOME_PENDING = "pending"        # environmental block (e.g. worker down); retryable
OUTCOME_CONFLICT = "conflict"      # base version moved; needs a fresh proposal
OUTCOME_FAILED = "failed"          # save/validation error; retryable
OUTCOME_UNSUPPORTED = "unsupported"  # project-inherent block; terminal
OUTCOME_PRESERVED = "preserved"    # individual override kept; terminal
OUTCOME_UNCHANGED = "unchanged"    # already had the target value; terminal
OUTCOME_EXCLUDED = "excluded"      # clip cannot take this op; terminal

TERMINAL_OUTCOMES = frozenset({
    OUTCOME_APPLIED, OUTCOME_CONFLICT, OUTCOME_FAILED,
    OUTCOME_UNSUPPORTED, OUTCOME_PRESERVED, OUTCOME_UNCHANGED,
    OUTCOME_EXCLUDED,
})
RETRYABLE_OUTCOMES = frozenset({OUTCOME_FAILED, OUTCOME_PENDING})


def _reject_bool(v, what):
    if isinstance(v, bool):
        raise ValueError(f"{what} must not be a boolean")
    return v


def validate_bulk_intent(intent: dict) -> dict:
    """Strictly validate a bulk intent; return the normalized intent.

    Normalized form: {"op": ..., "params": {...}, "on_override": ...}.
    Raises ValueError on anything unknown, missing or out of the closed
    sets -- never silently narrow or invent an operation.
    """
    if not isinstance(intent, dict):
        raise ValueError("intent must be a dict")
    op = intent.get("op")
    if op not in BULK_OPERATIONS:
        raise ValueError(
            f"unknown bulk operation {op!r}; expected one of {sorted(BULK_OPERATIONS)}")
    params = intent.get("params")
    if not isinstance(params, dict) or not params:
        raise ValueError("intent.params must be a non-empty dict")
    on_override = intent.get("on_override", ON_OVERRIDE_PRESERVE)
    if on_override not in ON_OVERRIDE_MODES:
        raise ValueError(
            f"unknown on_override {on_override!r}; expected 'preserve' or 'replace'")
    unknown_top = set(intent) - {"op", "params", "on_override"}
    if unknown_top:
        raise ValueError(f"unknown intent keys: {sorted(unknown_top)}")

    norm_params: dict = {}
    if op == OP_CAPTION_APPEARANCE:
        unknown = set(params) - {"caption_style", "caption_position"}
        if unknown:
            raise ValueError(f"unknown caption_appearance params: {sorted(unknown)}")
        if "caption_style" in params:
            s = params["caption_style"]
            if s not in CAPTION_STYLES:
                raise ValueError(f"unknown caption style: {s!r}")
            norm_params["caption_style"] = s
        if "caption_position" in params:
            p = params["caption_position"]
            if p not in CAPTION_POSITIONS:
                raise ValueError(f"unknown caption position: {p!r}")
            norm_params["caption_position"] = p
    elif op == OP_HOOK_APPEARANCE:
        unknown = set(params) - {"style", "position"}
        if unknown:
            raise ValueError(f"unknown hook_appearance params: {sorted(unknown)}")
        if "style" in params:
            s = params["style"]
            if s not in HOOK_OVERLAY_STYLES:
                raise ValueError(f"unknown hook overlay style: {s!r}")
            norm_params["style"] = s
        if "position" in params:
            p = params["position"]
            if p not in HOOK_POSITIONS:
                raise ValueError(f"unknown hook position: {p!r}")
            norm_params["position"] = p
    elif op == OP_HOOK_TIMING:
        unknown = set(params) - {"start", "end"}
        if unknown:
            raise ValueError(f"unknown hook_timing params: {sorted(unknown)}")
        if "start" in params:
            s = _reject_bool(params["start"], "hook start")
            if (not isinstance(s, (int, float)) or not math.isfinite(s)
                    or not (s >= 0)):
                raise ValueError(
                    f"hook start must be a finite number >= 0, got {s!r}")
            norm_params["start"] = float(s)
        if "end" in params:
            e = _reject_bool(params["end"], "hook end")
            if (not isinstance(e, (int, float)) or not math.isfinite(e)
                    or not (e > 0)):
                raise ValueError(
                    f"hook end must be a finite number > 0, got {e!r}")
            norm_params["end"] = float(e)
        if ("start" in norm_params and "end" in norm_params
                and not norm_params["end"] > norm_params["start"]):
            raise ValueError("hook end must be after hook start")
    elif op == OP_HYPERFRAMES_FINISHING:
        # The normalized form carries the canonical on_override="replace"
        # (the finishing selection always replaces the finishing field).
        # Revalidation must accept that canonical value -- plan and retry
        # re-validate the already-normalized intent -- but nothing else.
        if ("on_override" in intent
                and intent["on_override"] != ON_OVERRIDE_REPLACE):
            raise ValueError(
                "on_override does not apply to hyperframes_finishing: the "
                "finishing selection always replaces the finishing field")
        unknown = set(params) - {"template", "caption_size", "caption_position"}
        if unknown:
            raise ValueError(
                f"unknown hyperframes_finishing params: {sorted(unknown)}")
        template = params.get("template")
        if not isinstance(template, str) or not template:
            raise ValueError("hyperframes_finishing requires a template id")
        supported = _supported_hyperframes_templates()
        if template not in supported:
            raise ValueError(
                f"unsupported hyperframes template {template!r}; "
                f"supported: {sorted(supported)}")
        norm_params["template"] = template
        size = params.get("caption_size", "medium")
        if size not in HF_CAPTION_SIZES:
            raise ValueError(f"unknown caption_size: {size!r}")
        norm_params["caption_size"] = size
        pos = params.get("caption_position", "lower-third")
        if pos not in HF_CAPTION_POSITIONS:
            raise ValueError(f"unknown caption_position: {pos!r}")
        norm_params["caption_position"] = pos
        on_override = ON_OVERRIDE_REPLACE  # explicit whole-field selection
    if not norm_params:
        raise ValueError(f"intent.params for {op} carries no usable fields")
    return {"op": op, "params": norm_params, "on_override": on_override}


def _supported_hyperframes_templates() -> set[str]:
    """Task 18's supported template ids (closed set, no expansion here)."""
    from clippyme.finishing import templates as _t
    return set(_t.TEMPLATES.keys())


def _check_hyperframes_worker() -> dict:
    """Task 18 capability check (cheap PATH probe, no rendering)."""
    from clippyme.finishing import worker_client as _wc
    return _wc.check_available()


def _configured_music_file(project) -> str | None:
    from clippyme.finishing.compile import _configured_music_file as _cmf
    return _cmf(project)


def check_hyperframes_clip_support(project) -> str | None:
    """Per-clip Task-18 support gate for the bulk finishing option.

    Returns None when the clip can take the finishing selection, else the
    explicit reason it cannot. Read-only: no rendering, no mutation. Mirrors
    the pilot's refusal conditions (framing trajectory / reframed lineage /
    configured music) so bulk never silently rematerializes footage.
    """
    if getattr(project, "framing", None) is not None:
        return ("project carries a framing trajectory (tracked/split/screencast); "
                "the Hyperframes pilot cannot replay it -- use the legacy render path")
    if getattr(project, "reframed_base", None) is not None:
        return ("project carries reframed_base lineage (accepted framing); "
                "the Hyperframes pilot cannot safely replay it -- use the legacy render path")
    music = _configured_music_file(project)
    if music:
        return (f"project has configured music ({music}); the Hyperframes pilot "
                "renders source audio only")
    return None


# ---------------------------------------------------------------------------
# Scope resolution
# ---------------------------------------------------------------------------

def _validate_clip_index_list(raw) -> list[int]:
    """Ordered unique clip indices from a client-supplied list."""
    if (not isinstance(raw, list) or not raw
            or any(not isinstance(x, int) or isinstance(x, bool) or x < 0
                   for x in raw)):
        raise ValueError(
            "scope.clip_indices must be a non-empty list of clip indices")
    indices = sorted(set(raw))
    if len(indices) > MAX_BULK_CLIPS:
        raise ValueError(f"too many clips (max {MAX_BULK_CLIPS})")
    return indices


def resolve_scope_indices(job_dir: str, scope: dict) -> list[int]:
    """Resolve and freeze the ordered unique clip indices for a scope.

    scope: {"mode": "selected"|"all", "clip_indices": [...]}. "all" with no
    explicit list enumerates the current job snapshot (version map +
    working-copy files); clips added later are never included. An explicit
    clip_indices list (either mode) is the client's acknowledged frozen
    enumeration: every index must exist in the current job, so no
    undisclosed extra clip can enter the operation.
    """
    if not isinstance(scope, dict):
        raise ValueError("scope must be a dict")
    mode = scope.get("mode")
    if mode == "all":
        current = enumerate_job_clips(job_dir)
        if not current:
            raise ValueError("the job has no clip projects to select")
        raw = scope.get("clip_indices")
        if raw is None:
            return current
        indices = _validate_clip_index_list(raw)
        unknown = [i for i in indices if i not in set(current)]
        if unknown:
            raise ValueError(
                f"scope references unknown clips for this job: {unknown}")
        return indices
    if mode == "selected":
        return _validate_clip_index_list(scope.get("clip_indices"))
    raise ValueError(f"unknown scope mode {mode!r}; expected 'selected' or 'all'")


def enumerate_job_clips(job_dir: str) -> list[int]:
    """Sorted clip indices with a project in this job (version map + files)."""
    indices: set[int] = set()
    for key in _load_version_map(job_dir):
        try:
            indices.add(int(key))
        except (TypeError, ValueError):
            continue
    try:
        names = os.listdir(job_dir)
    except OSError:
        names = []
    for name in names:
        m = re.fullmatch(r"clip-project-(\d+)\.json", name)
        if m:
            indices.add(int(m.group(1)))
    return sorted(indices)


# ---------------------------------------------------------------------------
# Planning (read-only; no persistence, no mutation of stored projects)
# ---------------------------------------------------------------------------

def _hook_overlay(project):
    return next((o for o in project.overlays if o.type == "hook"), None)


def _clip_duration(project) -> float:
    dur = getattr(project.source, "duration", None)
    if isinstance(dur, (int, float)) and dur and dur > 0:
        return float(dur)
    ends = [s.end for s in project.segments
            if isinstance(getattr(s, "end", None), (int, float))]
    if ends:
        return float(max(ends))
    return 10.0


def _mode(values: list) -> object:
    """Most common value; ties break towards first-seen (deterministic)."""
    counts: dict = {}
    order: list = []
    for v in values:
        key = json.dumps(v, sort_keys=True, default=str)
        if key not in counts:
            counts[key] = [0, v]
            order.append(key)
        counts[key][0] += 1
    best = max(order, key=lambda k: (counts[k][0], -order.index(k)))
    return counts[best][1]


def _classify_field(*, current, target, baseline, on_override):
    """Shared override rule: preserve individual overrides by default.

    - current == target -> "unchanged"
    - current != target and on_override == preserve and current != baseline
      -> "preserved" (individual override kept, disclosed)
    - else -> "change" (replaced_override flags an overridden value)
    """
    if current == target:
        return OUTCOME_UNCHANGED, None
    if on_override == ON_OVERRIDE_PRESERVE and current != baseline:
        return OUTCOME_PRESERVED, {
            "current": current, "baseline": baseline,
            "reason": "individual override preserved",
        }
    return "change", {
        "from": current, "to": target,
        "replaced_override": current != baseline,
    }


def plan_bulk_operation(*, job_dir: str, clip_indices: list[int],
                        intent: dict, worker_available: dict | None = None) -> dict:
    """Freeze the per-clip plan for a validated intent (read-only).

    Returns {"baselines": {...}, "items": [per-clip plan, ...],
    "worker": {...}|None}. Raises ValueError (bad intent) or
    FileNotFoundError (a selected clip has no project file).
    """
    norm = validate_bulk_intent(intent)
    op = norm["op"]
    params = norm["params"]
    on_override = norm["on_override"]

    if op == OP_HYPERFRAMES_FINISHING and worker_available is None:
        worker_available = _check_hyperframes_worker()

    # First pass: load projects + collect current values for baselines.
    loaded: list[dict] = []
    for ci in clip_indices:
        path = get_project_path(job_dir, ci)
        if not path:
            raise FileNotFoundError(
                f"no project file for clip {ci} in this job")
        with open(path, "r", encoding="utf-8") as f:
            project = project_from_json(f.read())
        loaded.append({
            "clip_index": ci,
            "project": project,
            "expected_version": latest_project_version(job_dir, ci),
        })

    field_currents: dict[str, list] = {}
    hook_clips: list[dict] = []
    if op in (OP_HOOK_APPEARANCE, OP_HOOK_TIMING):
        for entry in loaded:
            hook = _hook_overlay(entry["project"])
            if hook is not None:
                hook_clips.append(entry)
                entry["hook"] = hook
    if op == OP_CAPTION_APPEARANCE:
        for entry in loaded:
            p = entry["project"]
            field_currents.setdefault("caption_style", []).append(p.captions.style)
            field_currents.setdefault("caption_position", []).append(
                p.captions.position)
    elif op == OP_HOOK_APPEARANCE:
        for entry in hook_clips:
            h = entry["hook"]
            field_currents.setdefault("style", []).append(h.style)
            field_currents.setdefault("position", []).append(h.position)
    baselines = {f: _mode(vals) for f, vals in field_currents.items() if vals}

    items: list[dict] = []
    for entry in loaded:
        ci = entry["clip_index"]
        project = entry["project"]
        item = {
            "clip_index": ci,
            "expected_version": entry["expected_version"],
            "outcome": OUTCOME_PLANNED,
            "patch": {},
            "changes": [],
            "preserved": [],
            "adapted": [],
            "excluded": None,
            "summary": "",
        }
        if op == OP_CAPTION_APPEARANCE:
            _plan_caption_appearance(item, project, params, baselines, on_override)
        elif op == OP_HOOK_APPEARANCE:
            _plan_hook_appearance(item, project, params, baselines, on_override)
        elif op == OP_HOOK_TIMING:
            _plan_hook_timing(item, project, params)
        elif op == OP_HYPERFRAMES_FINISHING:
            _plan_hyperframes(item, project, params, worker_available)
        # Terminal classification for items with nothing to save: an
        # item is only "planned" (save-attempted) when its patch is
        # non-empty, so apply can never mint a spurious empty version.
        # Preserved overrides keep their disclosed terminal outcome.
        if item["outcome"] == OUTCOME_PLANNED and not item["patch"]:
            if item["preserved"]:
                item["outcome"] = OUTCOME_PRESERVED
                item["summary"] = (
                    "individual overrides preserved; nothing to change")
            elif not item["excluded"]:
                item["outcome"] = OUTCOME_UNCHANGED
                if not item["summary"]:
                    item["summary"] = "already matches the requested change"
        items.append(item)
    plan = {"baselines": baselines, "items": items}
    if op == OP_HYPERFRAMES_FINISHING:
        plan["worker"] = {
            "available": bool(worker_available.get("available")),
            "reason": worker_available.get("reason", ""),
        }
    return plan


def _field_plan(item: dict, field: str, current, target, baseline, on_override,
               patch_key: str):
    outcome, detail = _classify_field(
        current=current, target=target, baseline=baseline,
        on_override=on_override)
    if outcome == OUTCOME_UNCHANGED:
        item["summary"] += f"{field}: already {target!r}; "
    elif outcome == OUTCOME_PRESERVED:
        item["preserved"].append({"field": field, **detail})
        item["summary"] += f"{field}: kept override {current!r}; "
    else:
        item["patch"][patch_key] = target
        item["changes"].append({"field": field, **detail})
        flag = " (override replaced)" if detail["replaced_override"] else ""
        item["summary"] += f"{field}: {current!r} -> {target!r}{flag}; "


def _plan_caption_appearance(item, project, params, baselines, on_override):
    if "caption_style" in params:
        _field_plan(item, "caption_style", project.captions.style,
                    params["caption_style"], baselines.get("caption_style"),
                    on_override, "captions.style")
    if "caption_position" in params:
        _field_plan(item, "caption_position", project.captions.position,
                    params["caption_position"], baselines.get("caption_position"),
                    on_override, "captions.position")
    item["summary"] = item["summary"].rstrip("; ") or "no caption fields requested"


def _plan_hook_appearance(item, project, params, baselines, on_override):
    hook = _hook_overlay(project)
    if hook is None:
        item["outcome"] = OUTCOME_EXCLUDED
        item["excluded"] = {
            "reason": "clip has no hook overlay; bulk never invents hook content",
        }
        item["summary"] = "excluded: no hook overlay"
        return
    if "style" in params:
        _field_plan(item, "style", hook.style, params["style"],
                    baselines.get("style"), on_override, "hook.style")
    if "position" in params:
        _field_plan(item, "position", hook.position, params["position"],
                    baselines.get("position"), on_override, "hook.position")
    item["summary"] = item["summary"].rstrip("; ") or "no hook fields requested"


def _plan_hook_timing(item, project, params):
    hook = _hook_overlay(project)
    if hook is None:
        item["outcome"] = OUTCOME_EXCLUDED
        item["excluded"] = {
            "reason": "clip has no hook overlay; bulk never invents hook content",
        }
        item["summary"] = "excluded: no hook overlay"
        return
    opening = getattr(project, "opening", None)
    if opening is not None and getattr(opening, "status", None) in (
            "selected", "user_authored"):
        item["outcome"] = OUTCOME_EXCLUDED
        item["excluded"] = {
            "reason": (
                "clip has a grounded/user-authored opening; its display timing "
                "is evidence-bound -- retime it per clip"),
        }
        item["summary"] = "excluded: grounded/user-authored opening timing"
        return
    duration = _clip_duration(project)
    start = params.get("start", hook.start)
    end = params.get("end", hook.end)
    applied_end = min(end, round(duration, 3))
    if applied_end < end:
        item["adapted"].append({
            "field": "end", "requested": end, "applied": applied_end,
            "reason": f"clamped to the clip duration ({duration:.2f}s)",
        })
    if not applied_end > start:
        item["outcome"] = OUTCOME_EXCLUDED
        item["excluded"] = {
            "reason": (
                f"requested timing collapses on this clip "
                f"(start={start}, end={end}, duration={duration:.2f}s)"),
        }
        item["summary"] = "excluded: timing collapses for this clip"
        return
    # Timing is inherently per-clip: it is always set (adapted), never an
    # "override" -- the hook's text/content is never touched.
    if "start" in params and start != hook.start:
        item["patch"]["hook.start"] = start
        item["changes"].append({"field": "start", "from": hook.start, "to": start,
                               "replaced_override": False})
    if "end" in params and applied_end != hook.end:
        item["patch"]["hook.end"] = applied_end
        item["changes"].append({"field": "end", "from": hook.end, "to": applied_end,
                               "replaced_override": False})
    if not item["patch"]:
        item["outcome"] = OUTCOME_UNCHANGED
        item["summary"] = "timing already matches"
    else:
        item["summary"] = (
            f"hook timing -> start={item['patch'].get('hook.start', hook.start)}, "
            f"end={item['patch'].get('hook.end', hook.end)}"
            + (" (adapted)" if item["adapted"] else ""))


def _plan_hyperframes(item, project, params, worker_available):
    if not worker_available.get("available"):
        item["outcome"] = OUTCOME_PENDING
        item["excluded"] = {
            "reason": ("hyperframes worker unavailable: "
                       f"{worker_available.get('reason', 'unknown')}; "
                       "ordinary project edits are unaffected"),
        }
        item["summary"] = "pending: worker unavailable"
        return
    reason = check_hyperframes_clip_support(project)
    if reason:
        item["outcome"] = OUTCOME_UNSUPPORTED
        item["excluded"] = {"reason": reason}
        item["summary"] = f"unsupported: {reason}"
        return
    finishing = {
        "renderer": "hyperframes",
        "template": params["template"],
        "caption_size": params["caption_size"],
        "caption_position": params["caption_position"],
    }
    current = getattr(project, "finishing", None)
    current_dict = (current.model_dump(exclude_none=True) if current else None)
    if current_dict == finishing:
        item["outcome"] = OUTCOME_UNCHANGED
        item["summary"] = "finishing already matches"
        return
    if current_dict:
        item["changes"].append({"field": "finishing", "from": current_dict,
                               "to": finishing, "replaced_override": True})
    else:
        item["changes"].append({"field": "finishing", "from": None,
                               "to": finishing, "replaced_override": False})
    item["patch"]["finishing"] = finishing
    item["summary"] = f"finishing -> {params['template']}"


# ---------------------------------------------------------------------------
# Patch application (per clip, through the ONE save authority)
# ---------------------------------------------------------------------------

def apply_plan_patch(project, patch: dict, op: str):
    """Apply a frozen per-clip patch to a project (mutates + returns it).

    Only the planner produces these patches; the keys are closed per op.
    """
    for key, value in (patch or {}).items():
        if op == OP_CAPTION_APPEARANCE:
            if key == "captions.style":
                project.captions.style = value
            elif key == "captions.position":
                project.captions.position = value
            else:
                raise ValueError(f"unknown caption patch key: {key!r}")
        elif op == OP_HOOK_APPEARANCE:
            hook = _hook_overlay(project)
            if hook is None:
                raise ValueError("hook overlay disappeared before apply")
            if key == "hook.style":
                hook.style = value
            elif key == "hook.position":
                hook.position = value
            else:
                raise ValueError(f"unknown hook patch key: {key!r}")
        elif op == OP_HOOK_TIMING:
            hook = _hook_overlay(project)
            if hook is None:
                raise ValueError("hook overlay disappeared before apply")
            if key == "hook.start":
                hook.start = value
            elif key == "hook.end":
                hook.end = value
            else:
                raise ValueError(f"unknown hook timing patch key: {key!r}")
        elif op == OP_HYPERFRAMES_FINISHING:
            if key != "finishing":
                raise ValueError(f"unknown finishing patch key: {key!r}")
            project.finishing = HyperframesFinishing(**value)
        else:
            raise ValueError(f"unknown bulk operation: {op!r}")
    return project


# ---------------------------------------------------------------------------
# Operation ledger (atomic JSON, next to project-versions.json)
# ---------------------------------------------------------------------------

def _ledger_path(job_dir: str) -> str:
    return os.path.join(job_dir, BULK_LEDGER_FILENAME)


def load_bulk_ledger(job_dir: str) -> dict:
    path = _ledger_path(job_dir)
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and isinstance(data.get("operations"), dict):
            return data
    except (OSError, ValueError):
        pass
    return {"operations": {}}


def save_bulk_ledger(job_dir: str, ledger: dict) -> None:
    """Atomic ledger write (unique tmp + os.replace), mirroring the version map.

    The staging name is unique per process+thread+call, so concurrent
    operations in one job never collide on the same tmp file. Strict finite
    JSON: a nonfinite float raises here instead of being persisted.
    """
    path = _ledger_path(job_dir)
    tmp = (f"{path}.tmp-{os.getpid()}-{threading.get_ident()}"
           f"-{uuid.uuid4().hex}")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(ledger, f, indent=2, allow_nan=False)
    os.replace(tmp, path)


def get_operation(job_dir: str, operation_id: str) -> dict | None:
    return load_bulk_ledger(job_dir)["operations"].get(operation_id)


def canonical_payload(*, job_id: str, clip_indices: list[int],
                      intent: dict) -> dict:
    return {
        "job_id": job_id,
        "clip_indices": sorted(set(clip_indices)),
        "intent": {
            "op": intent["op"],
            "params": intent["params"],
            "on_override": intent["on_override"],
        },
    }


def payload_sha(payload: dict) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                           default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def find_operation_by_key(job_dir: str, idempotency_key: str) -> dict | None:
    for op in load_bulk_ledger(job_dir)["operations"].values():
        if op.get("idempotency_key") == idempotency_key:
            return op
    return None


def bulk_item_key(operation_id: str, clip_index: int) -> str:
    """Deterministic per-item idempotency key (crash-recovery anchor)."""
    return f"bulk:{operation_id}:{int(clip_index)}"


def find_version_by_item_key(job_dir: str, clip_index: int,
                             item_key: str) -> int | None:
    """Highest recorded version stamped with this item key (or None).

    Crash recovery: a save that landed before the ledger write is detected
    here, so apply never saves twice for one item.
    """
    entry = _load_version_map(job_dir).get(str(clip_index), {})
    best = None
    for ver, rec in (entry.get("versions") or {}).items():
        if rec.get("idempotency_key") == item_key:
            try:
                v = int(ver)
            except (TypeError, ValueError):
                continue
            best = v if best is None or v > best else best
    return best


# One lock per operation id: concurrent apply/retry/undo requests for the
# same operation serialize in-process, so two racing applies can never mint
# two versions for one item. (Cross-process races keep the version-guard +
# item-key recovery as the backstop, same as the existing save path.)
_op_locks: dict[str, threading.Lock] = {}
_op_locks_guard = threading.Lock()


def operation_lock(operation_id: str) -> threading.Lock:
    with _op_locks_guard:
        return _op_locks.setdefault(operation_id, threading.Lock())


# One lock per job directory: the shared bulk-operations.json is read,
# decided on and rewritten as one serialized transaction, so concurrent
# create/apply/retry/undo calls -- including concurrent preparations with
# different keys or the same key -- can neither lose a record nor mint
# duplicate operations. In-process only: the API runs these through
# asyncio.to_thread inside a single process. This lock makes NO
# cross-process guarantee; across processes the optimistic version guards
# and the deterministic item-key recovery remain the backstop.
_job_locks: dict[str, threading.Lock] = {}
_job_locks_guard = threading.Lock()


def job_ledger_lock(job_dir: str) -> threading.Lock:
    """The process-wide lock for one job's bulk-operations.json."""
    key = os.path.abspath(job_dir)
    with _job_locks_guard:
        return _job_locks.setdefault(key, threading.Lock())


def _transact_ledger(job_dir: str, mutate):
    """Run one whole-ledger read/modify/write as a single job transaction.

    The ledger is loaded fresh INSIDE the shared per-job serialization
    boundary, the caller's ``mutate`` touches only its own operation's
    fields on that fresh object, and the write lands before the lock
    releases. A concurrent create/apply/retry/undo in the same job can
    therefore neither be erased by this transaction nor erase it -- unique
    tmp names alone cannot prevent that lost update. Callers must never
    hold a ledger snapshot loaded outside this boundary across another
    job transaction.

    Lock order is always operation_lock (outer) -> job_ledger_lock
    (inner); _record_item/_save_undo_state take only the job lock, so no
    inversion (and no deadlock) is possible.
    """
    with job_ledger_lock(job_dir):
        ledger = load_bulk_ledger(job_dir)
        result = mutate(ledger)
        save_bulk_ledger(job_dir, ledger)
        return result


# ---------------------------------------------------------------------------
# Apply / retry / undo orchestration
# ---------------------------------------------------------------------------

def _record_item(job_dir: str, operation_id: str, clip_index: int,
                 update: dict) -> dict:
    """Merge one item's outcome into the ledger (atomic) and return the op.

    The whole load/modify/write runs under the per-job ledger lock, so two
    different operations in one job can never overwrite each other's item
    updates.
    """
    with job_ledger_lock(job_dir):
        ledger = load_bulk_ledger(job_dir)
        op = ledger["operations"].get(operation_id)
        if op is None:
            raise KeyError(f"unknown bulk operation {operation_id}")
        item = op["items"][str(clip_index)]
        item.update(update)
        op["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        save_bulk_ledger(job_dir, ledger)
        return op


def _finalize_status(op: dict) -> str:
    outcomes = [i.get("outcome") for i in op["items"].values()]
    applied = any(o == OUTCOME_APPLIED for o in outcomes)
    bad = any(o in (OUTCOME_FAILED, OUTCOME_CONFLICT) for o in outcomes)
    open_ = any(o not in TERMINAL_OUTCOMES for o in outcomes)
    if applied and (bad or open_):
        return "partial"
    if applied:
        return "applied"
    if open_:
        # Nothing applied yet, but retryable items remain.
        return "partial"
    if bad:
        return "failed"
    # All benign-terminal (preserved/unchanged/excluded/unsupported).
    return "applied"


def create_operation(*, job_dir: str, job_id: str, scope: dict,
                     intent: dict, idempotency_key: str) -> dict:
    """Prepare (and persist) a bulk operation. Idempotent on (key, payload).

    A repeated prepare with the same key AND payload returns the existing
    operation; the same key with a different payload is rejected with
    KeyError("idempotency_key_reused"). Raises ValueError/FileNotFoundError
    for bad scope/intent.

    The entire read -> idempotency decision -> plan -> write runs under the
    per-job ledger lock, so concurrent preparations (same key or different
    keys) serialize: the same key+payload returns one operation, different
    keys each get their own record, and no record is lost.
    """
    norm_intent = validate_bulk_intent(intent)
    clip_indices = resolve_scope_indices(job_dir, scope)
    payload = canonical_payload(job_id=job_id, clip_indices=clip_indices,
                                intent=norm_intent)
    sha = payload_sha(payload)

    with job_ledger_lock(job_dir):
        ledger = load_bulk_ledger(job_dir)
        for op in ledger["operations"].values():
            if op.get("idempotency_key") == idempotency_key:
                if op.get("payload_sha") != sha:
                    raise KeyError(
                        "idempotency_key_reused: this key is bound to a different "
                        "operation payload")
                return op

        plan = plan_bulk_operation(job_dir=job_dir, clip_indices=clip_indices,
                                   intent=norm_intent)
        now = time.strftime("%Y-%m-%dT%H:%M:%S")
        operation_id = uuid.uuid4().hex
        op = {
            "operation_id": operation_id,
            "idempotency_key": idempotency_key,
            "payload_sha": sha,
            "created_at": now,
            "updated_at": now,
            "job_id": job_id,
            "scope": {"mode": scope.get("mode"), "clip_indices": clip_indices,
                      "count": len(clip_indices)},
            "intent": norm_intent,
            "baselines": plan["baselines"],
            "base_versions": {str(i["clip_index"]): i["expected_version"]
                              for i in plan["items"]},
            "status": "proposed",
            "items": {},
            "undo": None,
        }
        if "worker" in plan:
            op["worker"] = plan["worker"]
        for item in plan["items"]:
            ci = str(item["clip_index"])
            op["items"][ci] = {
                "clip_index": item["clip_index"],
                "expected_version": item["expected_version"],
                "outcome": item["outcome"],
                "patch": item["patch"],
                "changes": item["changes"],
                "preserved": item["preserved"],
                "adapted": item["adapted"],
                "excluded": item["excluded"],
                "summary": item["summary"],
                "result_version": None,
                "base_snapshot": None,
                "error": None,
                "recovered": False,
            }
        ledger["operations"][operation_id] = op
        save_bulk_ledger(job_dir, ledger)
        return op


def apply_operation(*, job_dir: str, operation_id: str) -> dict:
    """Apply a prepared operation. Idempotent; crash-resumable.

    Items already terminal are never retouched. For each remaining item the
    version map is scanned for the deterministic item key first (a save that
    landed before its ledger write is adopted, never duplicated), then the
    frozen expected version is enforced through the single save authority.
    The ledger is rewritten after every item.
    """
    with operation_lock(operation_id):
        # Initial status write runs as one job transaction: the ledger is
        # loaded fresh inside the shared boundary, so a proposal committed
        # by a concurrent prepare cannot be erased by this write.
        def _mark_applying(ledger):
            op = ledger["operations"].get(operation_id)
            if op is None:
                raise KeyError(f"unknown bulk operation {operation_id}")
            op["status"] = "applying"
            op["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            return op

        op = _transact_ledger(job_dir, _mark_applying)

        intent_op = op["intent"]["op"]
        for ci_str, item in sorted(op["items"].items(),
                                   key=lambda kv: int(kv[0])):
            ci = int(ci_str)
            # Only planned items are save-attempted. Terminal items (applied,
            # conflict, failed, unsupported, preserved, unchanged, excluded)
            # are never retouched here; retry resets eligible items to
            # planned first. A pending item carries no patch -- attempting it
            # would mint a spurious empty version.
            if item["outcome"] != OUTCOME_PLANNED:
                continue
            key = bulk_item_key(operation_id, ci)
            # Crash recovery: the save may have landed without its ledger
            # write. Adopt it; never save twice for one item.
            landed = find_version_by_item_key(job_dir, ci, key)
            if landed is not None and landed > item["expected_version"]:
                op = _record_item(job_dir, operation_id, ci, {
                    "outcome": OUTCOME_APPLIED,
                    "result_version": landed,
                    "recovered": True,
                    "error": None,
                })
                continue
            try:
                path = get_project_path(job_dir, ci)
                if not path:
                    op = _record_item(job_dir, operation_id, ci, {
                        "outcome": OUTCOME_FAILED,
                        "error": "project file disappeared before apply",
                    })
                    continue
                with open(path, "r", encoding="utf-8") as f:
                    project = project_from_json(f.read())
                latest = latest_project_version(job_dir, ci)
                if latest != item["expected_version"]:
                    op = _record_item(job_dir, operation_id, ci, {
                        "outcome": OUTCOME_CONFLICT,
                        "error": (
                            f"clip is at v{latest}, proposal froze "
                            f"v{item['expected_version']} -- refresh and "
                            "prepare a new proposal"),
                        "current_version": latest,
                    })
                    continue
                base_snapshot = project.model_dump(mode="json")
                # Undo provenance is persisted BEFORE the save: if the
                # applied acknowledgment below is lost (crash, injected
                # failure), recovery adopts the landed version but the
                # exact pre-apply baseline is already durable. It is never
                # inferred from the already-modified project afterwards.
                op = _record_item(job_dir, operation_id, ci, {
                    "base_snapshot": base_snapshot,
                })
                patched = apply_plan_patch(project, item["patch"], intent_op)
                patched.origin = "user"
                patched.idempotency_key = key
                # Re-validate the mutated model: never silently corrupt.
                patched = validate_project(patched.model_dump(mode="json"))
                version, _ = save_project_version(
                    job_dir=job_dir, clip_index=ci, project=patched,
                    expected_version=item["expected_version"])
                op = _record_item(job_dir, operation_id, ci, {
                    "outcome": OUTCOME_APPLIED,
                    "result_version": version,
                    "error": None,
                })
            except ConflictError as exc:
                op = _record_item(job_dir, operation_id, ci, {
                    "outcome": OUTCOME_CONFLICT,
                    "error": str(exc)[:300],
                    "current_version": latest_project_version(job_dir, ci),
                })
            except Exception as exc:  # per-clip isolation
                op = _record_item(job_dir, operation_id, ci, {
                    "outcome": OUTCOME_FAILED,
                    "error": str(exc)[:300],
                })

        # Final status write re-reads the ledger inside the job
        # transaction: concurrent operations committed during the per-clip
        # saves are merged, never clobbered.
        def _finalize(ledger):
            op = ledger["operations"][operation_id]
            op["status"] = _finalize_status(op)
            op["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            return op

        return _transact_ledger(job_dir, _finalize)


def retry_operation(*, job_dir: str, operation_id: str) -> dict:
    """Retry eligible unresolved items under the ORIGINAL version guards.

    Eligible: failed / pending. Applied / conflict / unsupported / preserved /
    unchanged / excluded items are never retouched. A retryable item whose
    clip moved past the frozen base version becomes a conflict, never a
    silent rebase.
    """
    with operation_lock(operation_id):
        # The reset decision reads the ledger fresh inside the shared job
        # transaction boundary.
        def _load_for_reset(ledger):
            op = ledger["operations"].get(operation_id)
            if op is None:
                raise KeyError(f"unknown bulk operation {operation_id}")
            return op

        op = _transact_ledger(job_dir, _load_for_reset)
        eligible = [ci for ci, it in op["items"].items()
                    if it.get("outcome") in RETRYABLE_OUTCOMES]
        if not eligible:
            return op
        # Compute the reset + re-plan in memory, then commit it as ONE job
        # transaction below: the ledger is re-read fresh inside the
        # boundary, so concurrent operations committed meanwhile are
        # merged, never clobbered.
        #
        # A pending hyperframes item carries no patch by design (the worker
        # was down at plan time), so its stored patch is never trusted.
        # Re-plan each eligible clip from the frozen intent under the
        # ORIGINAL version/scope guards: only the plan fields (patch,
        # changes, excluded, ...) are refreshed, the frozen
        # expected_version is kept, and nothing is saved unless the
        # rebuilt patch is non-empty and validated.
        refresh = {}
        worker_info = None
        if op["intent"]["op"] == OP_HYPERFRAMES_FINISHING:
            worker = _check_hyperframes_worker()
            worker_info = {"available": bool(worker.get("available")),
                           "reason": worker.get("reason", "")}
            for ci in eligible:
                path = get_project_path(job_dir, int(ci))
                if not path:
                    refresh[ci] = {
                        "outcome": OUTCOME_FAILED,
                        "error": "project file disappeared before retry",
                    }
                    continue
                plan = plan_bulk_operation(
                    job_dir=job_dir, clip_indices=[int(ci)],
                    intent=op["intent"], worker_available=worker)
                fresh = plan["items"][0]
                refresh[ci] = {
                    field: fresh[field]
                    for field in ("outcome", "patch", "changes",
                                  "preserved", "adapted", "excluded",
                                  "summary")
                }
                refresh[ci]["error"] = None

        def _mark_retrying(ledger):
            target = ledger["operations"][operation_id]
            for ci in eligible:
                it = target["items"][ci]
                it["outcome"] = OUTCOME_PLANNED
                it["error"] = None
                for field, value in refresh.get(ci, {}).items():
                    it[field] = value
            if worker_info is not None:
                target["worker"] = worker_info
            target["status"] = "applying"
            target["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")

        _transact_ledger(job_dir, _mark_retrying)
    return apply_operation(job_dir=job_dir, operation_id=operation_id)


def undo_operation(*, job_dir: str, operation_id: str) -> dict:
    """Undo successfully applied clips without overwriting later edits.

    Only items with outcome "applied" are candidates, and only while the
    clip's current version still equals the operation's acknowledged result
    version. The frozen base snapshot is restored through the versioned save
    path (a new acknowledged revision). Later edits surface as explicit undo
    conflicts. Recorded undo outcomes make repeated undo idempotent.
    """
    with operation_lock(operation_id):
        # The undo-init write runs as one job transaction: the ledger is
        # loaded fresh inside the shared boundary, so a proposal committed
        # by a concurrent prepare cannot be erased by this write.
        def _init_undo(ledger):
            op = ledger["operations"].get(operation_id)
            if op is None:
                raise KeyError(f"unknown bulk operation {operation_id}")
            undo = op.get("undo") or {"undone_at": None, "status": "pending",
                                      "items": {}}
            op["undo"] = undo
            return op, undo

        op, undo = _transact_ledger(job_dir, _init_undo)

        for ci_str, item in sorted(op["items"].items(),
                                   key=lambda kv: int(kv[0])):
            ci = int(ci_str)
            if item.get("outcome") != OUTCOME_APPLIED:
                continue
            prev = undo["items"].get(ci_str)
            if prev and prev.get("outcome") == "undone":
                continue  # idempotent: already undone, keep the record
            undo_key = f"bulk:{operation_id}:undo:{ci}"
            landed_undo = find_version_by_item_key(job_dir, ci, undo_key)
            latest = latest_project_version(job_dir, ci)
            if landed_undo is not None and landed_undo == latest:
                # This operation's own undo save already landed; only its
                # ledger acknowledgment was lost. Recover the exact undo --
                # this is NOT a later edit, so it must not be reported as
                # a conflict.
                undo["items"][ci_str] = {
                    "clip_index": ci,
                    "outcome": "undone",
                    "version": landed_undo,
                    "error": None,
                    "recovered": True,
                }
                _save_undo_state(job_dir, operation_id, op)
                continue
            if latest != item["result_version"]:
                undo["items"][ci_str] = {
                    "clip_index": ci,
                    "outcome": "conflict",
                    "error": (
                        f"clip is at v{latest}, this operation's result was "
                        f"v{item['result_version']} -- a later edit exists; "
                        "undo refuses to overwrite it"),
                    "current_version": latest,
                }
                _save_undo_state(job_dir, operation_id, op)
                continue
            snapshot = item.get("base_snapshot")
            if not isinstance(snapshot, dict):
                undo["items"][ci_str] = {
                    "clip_index": ci,
                    "outcome": "failed",
                    "error": "no base snapshot recorded for this clip",
                }
                _save_undo_state(job_dir, operation_id, op)
                continue
            try:
                restore = validate_project(snapshot)
                restore.origin = "user"
                restore.idempotency_key = f"bulk:{operation_id}:undo:{ci}"
                version, _ = save_project_version(
                    job_dir=job_dir, clip_index=ci, project=restore,
                    expected_version=item["result_version"])
                undo["items"][ci_str] = {
                    "clip_index": ci,
                    "outcome": "undone",
                    "version": version,
                    "error": None,
                }
            except ConflictError as exc:
                undo["items"][ci_str] = {
                    "clip_index": ci,
                    "outcome": "conflict",
                    "error": str(exc)[:300],
                    "current_version": latest_project_version(job_dir, ci),
                }
            except Exception as exc:
                undo["items"][ci_str] = {
                    "clip_index": ci,
                    "outcome": "failed",
                    "error": str(exc)[:300],
                }
            _save_undo_state(job_dir, operation_id, op)

        # Finalization re-reads the ledger inside the job transaction:
        # concurrent operations committed during the per-clip undos are
        # merged, never clobbered. The undo outcomes themselves were
        # persisted per item via _save_undo_state, so the status is derived
        # from the freshly loaded record.
        def _finalize_undo(ledger):
            op = ledger["operations"][operation_id]
            undo = op["undo"]
            outcomes = [r.get("outcome") for r in undo["items"].values()]
            applied_items = [i for i in op["items"].values()
                             if i.get("outcome") == OUTCOME_APPLIED]
            if outcomes and all(o == "undone" for o in outcomes) and len(
                    outcomes) == len(applied_items):
                undo["status"] = "undone"
            elif any(o == "undone" for o in outcomes):
                undo["status"] = "partially_undone"
            elif outcomes:
                undo["status"] = "undo_failed"
            else:
                undo["status"] = "nothing_to_undo"
            undo["undone_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            return op

        return _transact_ledger(job_dir, _finalize_undo)


def _save_undo_state(job_dir: str, operation_id: str, op: dict) -> None:
    with job_ledger_lock(job_dir):
        ledger = load_bulk_ledger(job_dir)
        target = ledger["operations"].get(operation_id)
        if target is not None:
            target["undo"] = op["undo"]
            target["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            save_bulk_ledger(job_dir, ledger)
