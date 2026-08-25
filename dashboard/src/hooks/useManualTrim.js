// Manual-trim state for EditClipModal (flycut-style tap-to-cut + AI trim).
// Extracted from the modal so the Trim tab could become a dumb component:
// this hook owns the lazy transcript load, the dropped-segment set and the
// conversational (Gemini) trim. State stays LIFTED in the modal via this
// hook — tabs are conditionally rendered, so a tab component owning the
// segments would lose them on every tab switch, and apply() needs dropRanges.
import { useEffect, useRef, useState } from 'react';
import { getClipTranscript, editClipAI } from '../redesign/realApi';
import { dropSetFromRanges, segmentIndicesHit, rangesFromDropSet } from '../lib/trimSelection';

export function useManualTrim({ jobId, idx, active, initialDropRanges }) {
  const [segments, setSegments] = useState(null); // null = not loaded
  const [segErr, setSegErr] = useState(false);
  const [dropped, setDropped] = useState(() => new Set());
  // Tracks whether we've already seeded `dropped` from initialDropRanges on the
  // first transcript load. Prevents a parent re-render (toast, tab switch) from
  // resetting the user's in-progress tap-to-cut selections while segments load.
  const seededRef = useRef(false);

  // Lazy-load transcript segments the first time the Trim tab is opened
  // (`active` is false in bulk mode — manual trim is per-clip). Cheap GET;
  // backend reads metadata.json. Failure → hide the trim list silently.
  useEffect(() => {
    if (!active || segments !== null || !jobId) return;
    let alive = true;
    getClipTranscript(jobId, idx)
      .then((d) => { if (!alive) return;
        const segs = d.segments || [];
        setSegments(segs);
        // Seed dropped segments only on the first successful load — never on a
        // re-render-triggered re-run. initialDropRanges is intentionally excluded
        // from the dep array below to avoid resetting user edits.
        if (!seededRef.current) {
          seededRef.current = true;
          setDropped(dropSetFromRanges(segs, initialDropRanges));
        }
      })
      .catch(() => { if (alive) setSegErr(true); });
    return () => { alive = false; };
  // initialDropRanges intentionally excluded: it's an array reference that
  // changes identity on every parent render; including it would wipe in-progress
  // trim selections. The seededRef gate ensures it's applied exactly once.
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [active, segments, jobId, idx]);

  const toggleDrop = (i) => setDropped((prev) => {
    const next = new Set(prev);
    next.has(i) ? next.delete(i) : next.add(i);
    return next;
  });

  // Conversational trim: ask Gemini which spans to cut, then mark every
  // segment overlapping a returned span as dropped (reuses the tap-to-cut
  // state).
  const [text, setText] = useState('');
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState('');
  const ask = async () => {
    const instr = text.trim();
    if (!instr || busy || !jobId) return;
    setBusy(true); setMsg('');
    try {
      const { drop_ranges = [], explanation = '' } = await editClipAI(jobId, idx, instr);
      const hit = segmentIndicesHit(segments || [], drop_ranges);
      if (hit.size) {
        setDropped((prev) => new Set([...prev, ...hit]));
        setMsg(explanation || `Cut ${hit.size} segment${hit.size === 1 ? '' : 's'}.`);
      } else {
        setMsg(explanation || 'Nothing to cut for that instruction.');
      }
    } catch (e) {
      setMsg(e.message || 'AI trim failed.');
    } finally {
      setBusy(false);
    }
  };

  const dropRanges = rangesFromDropSet(segments || [], dropped);
  return {
    segments, segErr, dropped, toggleDrop,
    dropRanges, hasDrops: dropRanges.length > 0,
    ai: { text, setText, busy, msg, ask },
  };
}
