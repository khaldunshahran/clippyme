# Duration Policy

Shared duration bounds for clip selection and edge repair.

## Resolution

`resolve_duration_policy()` in `src/clippyme/pipeline/duration_policy.py`
is the single source of truth, used by:
- `build_viral_prompt()` (selection prompt)
- `snap_clips_to_transcript()` via both production callers:
  - Queued orchestrator (`orchestrator.py`)
  - Direct CLI (`main.py`)

## Precedence

1. **Explicit `max_duration` > 0**: Hard limit. Never silently increased,
   even in `mixed`/`portfolio` mode.
2. **Explicit `min_duration` > 0**: If `min > max`, max wins; min is
   clamped to max (flagged via `min_clamped`).
3. **Duration mode**: `all`/`auto`/`mixed`/`portfolio` with NO explicit
   bounds → multi-tier (15s–300s, bounded by source).
4. **Defaults**: min 15s, max 120s.

## Invalid inputs

Bool, None, NaN, Inf, zero, negative, and non-numeric values are treated
as absent. If user provided a bound (even invalid), falls back to fixed
defaults (not multi-tier).

## Source bounds

`max_duration` never exceeds source duration. Known short sources clamp
accordingly (flagged via `max_clamped_to_source`).

## Repair enforcement

The resolved `max_duration` is a HARD LIMIT for optional expansion.
Enforced after ALL stages via `_enforce_duration_cap()`:
- Word snap (padding)
- Sentence snap (forward extension)
- Silence refinement
- Waveform polish

Content-preserving fallback (not greedy shrink):
- Baseline = word-safe interval from RAW proposal, before any optional
  stages, with overlap-chain closure and source/neighbor guards.
- If optional stages exceed max: DECLINE them, revert to baseline.
  Selected words are never removed to manufacture an in-policy clip.
- If baseline itself exceeds max: retain baseline, report
  `maximum_conflict` honestly for selection validation.
- Never truncates arbitrarily at `start+max` (could cut mid-word).

Overlap safety:
- `_word_safe_interval()` uses fixed-point iteration through overlapping
  word chains (e.g. A[9,11], B[10.5,12], C[11.5,13]: end=10 → 13).
- Every final proposal validated, even when within budget.
- Source/neighbor guards applied to all baselines, proposals, fallbacks.
- If guards prevent word completion: clamp and report `source_conflict`
  or `neighbor_conflict`.

Conflicts surfaced (not hidden):
- `maximum_conflict`: interval exceeds max; retained with diagnostic.
- `minimum_conflict`: interval below min after repair (checked after
  every branch, including baseline fallback).
- `source_conflict` / `neighbor_conflict`: guards prevented completion.
- Stored as `clip["duration_conflict"]` + `clip["duration_conflict_detail"]`,
  emitted as events even when edges don't move. Stale diagnostics cleared
  each invocation. Both callers surface these in logs/metadata.

## Short sources

Multi-tier mode respects source length: `0 < min <= max <= source`.
- Source <15s: min=5s (or half source), max=source.
- Source <30s: min=10s, max=source.
- Otherwise: min=15s, max=min(source, 300s).

Exact fractional limits preserved: max30.6 stays 30.6 (not 31).
