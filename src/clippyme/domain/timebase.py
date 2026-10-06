"""Source-time <-> output-timeline conversion (pure, host-testable).

GLOBAL RULE: every span stored on a clip project (segment boundaries,
``drop_ranges``, and later slices' ``censor_spans`` / B-roll anchors) is in
SOURCE time -- seconds in the source media. The render pipeline
(``project_render._map_words_to_timeline`` / ``compose.py``) and the editor
preview convert source -> output by subtracting dropped spans. Never store
output-time spans on the project; convert at the render/preview boundary.

Segments are the source of truth: the kept ranges in source time. Drops are
derived as the gaps between consecutive segments (plus head/tail outside the
first/last segment, bounded by the source range).
"""

from __future__ import annotations


def merge_spans(spans: list) -> list[list[float]]:
    """Sort spans and merge overlapping or adjacent ones.

    Adjacent spans (end == next start, within rounding) become one span --
    this is what "remove caption & video" on a shift-click range relies on.
    """
    norm: list[list[float]] = []
    for s in spans or []:
        try:
            a, b = float(s[0]), float(s[1])
        except (TypeError, ValueError, IndexError):
            continue
        if b > a:
            norm.append([a, b])
    norm.sort(key=lambda x: (x[0], x[1]))
    merged: list[list[float]] = []
    for a, b in norm:
        if merged and a <= merged[-1][1] + 1e-6:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return merged


def invert_drops(drops: list, source_start: float, source_end: float) -> list[dict]:
    """Drops (source time) -> kept segments (source time)."""
    kept: list[dict] = []
    cur = float(source_start)
    for a, b in merge_spans(drops):
        if a > cur:
            kept.append({"start": round(cur, 3), "end": round(min(a, source_end), 3)})
        cur = max(cur, b)
    if cur < float(source_end):
        kept.append({"start": round(cur, 3), "end": round(float(source_end), 3)})
    return [s for s in kept if s["end"] > s["start"]]


def derive_drops(segments: list, source_start: float, source_end: float) -> list[list[float]]:
    """Segments (kept, source time) -> drops (source time): the gaps."""
    segs = sorted(
        ({"start": float(s["start"]), "end": float(s["end"])} for s in segments or []),
        key=lambda s: s["start"],
    )
    drops: list[list[float]] = []
    cur = float(source_start)
    for s in segs:
        if s["start"] > cur:
            drops.append([round(cur, 3), round(s["start"], 3)])
        cur = max(cur, s["end"])
    if cur < float(source_end):
        drops.append([round(cur, 3), round(float(source_end), 3)])
    return merge_spans(drops)


def source_to_output(t: float, segments: list) -> float | None:
    """Source time -> output-timeline time. None when t falls in a dropped span.

    Assumes cut transitions (segments butt up); crossfaded boundaries overlap
    by the fade length (see project_render._timeline_layout) -- callers that
    need fade-exact mapping should use the render layout, not this helper.
    """
    t = float(t)
    off = 0.0
    for s in segments or []:
        a, b = float(s["start"]), float(s["end"])
        if a <= t < b:
            return round(off + (t - a), 3)
        off += b - a
    return None


def output_to_source(t: float, segments: list) -> float | None:
    """Output-timeline time -> source time. None when out of range."""
    t = float(t)
    off = 0.0
    for s in segments or []:
        a, b = float(s["start"]), float(s["end"])
        dur = b - a
        if off <= t < off + dur:
            return round(a + (t - off), 3)
        off += dur
    return None


def kept_duration(segments: list) -> float:
    """Total kept source seconds across segments (= output duration for cuts)."""
    return round(sum(float(s["end"]) - float(s["start"]) for s in segments or []), 3)
