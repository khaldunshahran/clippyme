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

The resolved `max_duration` is a HARD LIMIT for optional expansion:
- Sentence snap (forward extension)
- Silence refinement
- Waveform polish

Enforced after ALL stages. Mandatory word-boundary repair is distinct:
if an interval cannot preserve words AND meet duration, the conflict is
signaled honestly (not silent truncation through a word).
