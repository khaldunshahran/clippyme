/**
 * Source-time <-> output-timeline conversion.
 *
 * GLOBAL RULE (mirrors src/clippyme/domain/timebase.py): every span stored
 * on a clip project is in SOURCE time (seconds in the source media). The
 * preview converts source -> output by subtracting dropped spans; the
 * render pipeline (compose.py) does the same. Never store output-time
 * spans on the project.
 *
 * This module is the JS twin of the Python timebase. Both are driven by
 * the same fixture (tests/fixtures/drop_mappings.json) so they cannot drift.
 */

/** Sort spans and merge overlapping or adjacent ones (adjacent => one span). */
export function mergeSpans(spans) {
  const norm = [];
  for (const s of spans || []) {
    const a = Number(s?.[0]);
    const b = Number(s?.[1]);
    if (Number.isFinite(a) && Number.isFinite(b) && b > a) norm.push([a, b]);
  }
  norm.sort((x, y) => x[0] - y[0] || x[1] - y[1]);
  const merged = [];
  for (const [a, b] of norm) {
    if (merged.length && a <= merged[merged.length - 1][1] + 1e-6) {
      merged[merged.length - 1][1] = Math.max(merged[merged.length - 1][1], b);
    } else {
      merged.push([a, b]);
    }
  }
  return merged;
}

/** Drops (source time) -> kept segments (source time). */
export function invertDrops(drops, sourceStart, sourceEnd) {
  const kept = [];
  let cur = Number(sourceStart);
  const end = Number(sourceEnd);
  for (const [a, b] of mergeSpans(drops)) {
    if (a > cur) kept.push({ start: r3(cur), end: r3(Math.min(a, end)) });
    cur = Math.max(cur, b);
  }
  if (cur < end) kept.push({ start: r3(cur), end: r3(end) });
  return kept.filter((s) => s.end > s.start);
}

/** Segments (kept, source time) -> drops (source time): the gaps. */
export function deriveDrops(segments, sourceStart, sourceEnd) {
  const segs = [...(segments || [])]
    .map((s) => ({ start: Number(s.start), end: Number(s.end) }))
    .sort((a, b) => a.start - b.start);
  const drops = [];
  let cur = Number(sourceStart);
  const end = Number(sourceEnd);
  for (const s of segs) {
    if (s.start > cur) drops.push([r3(cur), r3(s.start)]);
    cur = Math.max(cur, s.end);
  }
  if (cur < end) drops.push([r3(cur), r3(end)]);
  return mergeSpans(drops);
}

/**
 * Source time -> output-timeline time. Returns null when t falls in a
 * dropped span. Assumes cut transitions (segments butt up).
 */
export function sourceToOutput(t, segments) {
  t = Number(t);
  let off = 0;
  for (const s of segments || []) {
    const a = Number(s.start);
    const b = Number(s.end);
    if (a <= t && t < b) return r3(off + (t - a));
    off += b - a;
  }
  return null;
}

/** Output-timeline time -> source time. Returns null when out of range. */
export function outputToSource(t, segments) {
  t = Number(t);
  let off = 0;
  for (const s of segments || []) {
    const a = Number(s.start);
    const b = Number(s.end);
    const dur = b - a;
    if (off <= t && t < off + dur) return r3(a + (t - off));
    off += dur;
  }
  return null;
}

/** Total kept source seconds (= output duration for cut transitions). */
export function keptDuration(segments) {
  return r3(
    (segments || []).reduce((sum, s) => sum + (Number(s.end) - Number(s.start)), 0)
  );
}

/**
 * Split the segment containing source-time t into two at t.
 * Returns a new segments array (sorted); no-op when t is not strictly
 * inside a segment. New segment gets id `${oldId}b`, old keeps its id
 * with the earlier range (IDs are opaque and never reused).
 */
export function splitSegmentAt(segments, t, minDur = 0.4) {
  t = Number(t);
  const out = [];
  let changed = false;
  for (const s of segments || []) {
    if (!changed && s.start + minDur <= t && t <= s.end - minDur) {
      out.push({ ...s, end: r3(t) });
      out.push({ ...s, id: `${s.id}b`, start: r3(t) });
      changed = true;
    } else {
      out.push({ ...s });
    }
  }
  return changed ? out : (segments || []).map((s) => ({ ...s }));
}

/**
 * Remove a source-time span [s, e] from the segments (carve-out).
 * Overlapping segments are split into kept pieces; pieces shorter than
 * minKeep are dropped. Returns a new segments array (sorted).
 */
export function removeSpanFromSegments(segments, s, e, minKeep = 0.3) {
  s = Number(s);
  e = Number(e);
  if (!(e > s)) return (segments || []).map((x) => ({ ...x }));
  const out = [];
  let n = 0;
  for (const seg of segments || []) {
    const a = Number(seg.start);
    const b = Number(seg.end);
    if (b <= s || a >= e) {
      out.push({ ...seg }); // untouched
      continue;
    }
    if (a < s && r3(s - a) >= minKeep) {
      out.push({ ...seg, id: n === 0 ? seg.id : `${seg.id}b`, end: r3(s) });
      n++;
    }
    if (b > e && r3(b - e) >= minKeep) {
      out.push({ ...seg, id: `${seg.id}b`, start: r3(e) });
      n++;
    }
    // The middle (fully covered) part is dropped; IDs are never reused.
  }
  out.sort((x, y) => x.start - y.start);
  return out;
}

function r3(x) {
  return Math.round(x * 1000) / 1000;
}
