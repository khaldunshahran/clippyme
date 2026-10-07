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

Word-safe: when over max, shrink to latest word boundary at or before
`start+max` (never mid-word). Already over-limit intervals are not grown
for optional reasons.

Conflicts are surfaced, not hidden:
- `maximum_conflict`: duration exceeds max and no word-safe shrink exists.
- `minimum_conflict`: duration below min after repair.
- Diagnostics stored as `clip["duration_conflict"]` and emitted as events
  even when edges don't move. Both callers surface these.

Mandatory word-boundary repair is distinct from optional expansion:
if an interval cannot preserve words AND meet duration, the conflict is
signaled honestly (not silent truncation through a word).

## Short sources

Multi-tier mode respects source length: `0 < min <= max <= source`.
- Source <15s: min=5s (or half source), max=source.
- Source <30s: min=10s, max=source.
- Otherwise: min=15s, max=min(source, 300s).

Exact fractional limits preserved: max30.6 stays 30.6 (not 31).
