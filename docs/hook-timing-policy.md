# Hook Timing Policy (Task 06)

Project hook overlays carry explicit `start`/`end` (clip-relative seconds).
Exports honor the stored interval; legacy callers without timing keep defaults.

## Timing contract

- **Explicit presence**: `hook_start`/`hook_end` keys in `hook_params`.
  `start=0` is explicit, not missing. Both must be present; half-specified
  does not activate timed mode.
- **Validation** (`domain/hook_timing.py`): finite numeric, no bool/string
  coercion. Negative, NaN/inf, or `end < start` raises `HookTimingError`
  -> `ProjectRenderError` before publishing.
- **Effective interval**: intersection of `[start, end)` with
  `[0, timeline_duration)`. End beyond duration is clipped.
  Zero-length or wholly-outside intervals are **inactive** (no hook),
  never permission for legacy default duration.
- **Output clock**: project overlay times are already timeline-relative.
  No shifting by source `clip.start` or segment source start.
  Smart Cut is disabled for project composition; the interval refers to
  the materialized output clock.

## Render path

1. `_derive_compose_inputs` copies `Overlay.start`/`end` to
   `hook_params["hook_start"]`/`hook_params["hook_end"]`.
2. `compose_layers(..., timeline_duration=...)` threads the duration through
   `_compose_layers_impl` to `_apply_hook`.
3. `_apply_hook`: if explicit timing, validate and compute effective interval.
   Inactive -> skip hook (logo still applied if present).
   Otherwise legacy: 4s, or full clip for `reframe_mode="disabled"`.
4. `add_hook_to_video(..., hook_start, hook_end, hook_inactive)`:
   builds FFmpeg filter with `enable='between(t,start,end)'`.
   Animation (`fade`, slide) uses `st=hook_start` and `(t-hook_start)`
   clock, so delayed animation starts on appearance.

## Fused Hook+Logo

Timing applies ONLY to the hook. The logo filter has no `enable` suffix
and remains visible before/during/after the hook interval. Single-pass
fusion preserved; z-order unchanged (hook below, logo topmost).

## Legacy compatibility

- Callers without `hook_start`/`hook_end` keep 0-4s (or full-clip for
  letterbox). `add_hook_to_video` positional args unchanged; new kwargs
  are optional and default to legacy behavior.
- `build_hook_overlay_filter` / `build_hook_logo_filter` keep old signatures
  working; `enable_start`/`anim_start` default to legacy (start=0).
- No schema changes: `Overlay.start`/`end` already exist.

## Limits

- First-supported-hook only; no multi-hook system.
- Explicit timing + time-changing legacy ops (e.g. Smart Cut) is
  unsupported; project composition disables Smart Cut.
- Frame quantization tolerance: 1e-6s (explicit).
