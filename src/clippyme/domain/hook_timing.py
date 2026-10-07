"""Hook timing validation and interval computation (Task 06).

Pure helpers for project hook start/end timing on export.

- Times are clip-relative seconds on the project's final timeline.
- Explicit presence: start=0 is valid and not treated as missing.
- Validation rejects bool/string coercion, NaN/inf, negative, reversed.
- Effective interval is intersection with [0, timeline_duration).
  Zero-length or wholly-outside intervals are inactive (not an error,
  not permission for legacy default duration).
"""

from __future__ import annotations

import math
from typing import Any


class HookTimingError(ValueError):
    """Invalid hook timing (not a renderable interval)."""


def _is_valid_time(v: Any) -> bool:
    """Finite numeric, excluding bool (no coercion)."""
    if isinstance(v, bool):
        return False
    return isinstance(v, (int, float)) and math.isfinite(v)


def validate_hook_interval(start: Any, end: Any) -> tuple[float, float]:
    """Validate explicit hook start/end.

    Returns (float(start), float(end)).
    Raises HookTimingError for: non-numeric, bool, NaN/inf, negative,
    or end < start.
    """
    if not _is_valid_time(start):
        raise HookTimingError(f"hook start must be finite numeric, got {start!r}")
    if not _is_valid_time(end):
        raise HookTimingError(f"hook end must be finite numeric, got {end!r}")
    s, e = float(start), float(end)
    if s < 0:
        raise HookTimingError(f"hook start must be >= 0, got {s}")
    if e < 0:
        raise HookTimingError(f"hook end must be >= 0, got {e}")
    if e < s:
        raise HookTimingError(f"hook end ({e}) < start ({s})")
    return (s, e)


def effective_hook_interval(
    start: float,
    end: float,
    timeline_duration: float,
    *,
    quant_tol: float = 1e-6,
) -> tuple[float, float] | None:
    """Intersect [start, end) with [0, timeline_duration).

    Returns (eff_start, eff_end), or None if the interval is inactive
    (zero-length or wholly outside the timeline). End beyond duration
    is clipped. Frame quantization tolerance is explicit.
    """
    if not _is_valid_time(timeline_duration) or timeline_duration <= 0:
        raise HookTimingError(
            f"timeline duration must be positive, got {timeline_duration!r}")
    s = max(0.0, float(start))
    e = min(float(end), float(timeline_duration))
    # Inactive if zero-length (within tolerance) or outside.
    if e - s <= quant_tol:
        return None
    if s >= timeline_duration - quant_tol:
        return None
    return (s, e)


def has_explicit_timing(hook_params: dict[str, Any]) -> bool:
    """Check for explicit start/end presence (not truthiness).

    start=0 is explicit. Both must be present; half-specified does not
    activate timed mode (documented, not silent full-clip).
    """
    if not isinstance(hook_params, dict):
        return False
    return "hook_start" in hook_params and "hook_end" in hook_params
