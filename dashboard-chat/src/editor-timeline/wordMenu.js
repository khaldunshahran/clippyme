/**
 * Slice-1 word actions: pure project transforms.
 *
 * Each takes a project dict (already cloned by the edit queue) and returns
 * the mutated project. All times are SOURCE time -- segments store source
 * seconds, and drops are derived as segment gaps (see timebase.js).
 */
import { removeSpanFromSegments, splitSegmentAt } from './timebase.js';

export function wordById(project, id) {
  return (project?.captions?.words || []).find((w) => w.id === id) || null;
}

export function displayWord(project, word, index) {
  const edits = project?.captions?.edits || {};
  if (word.id && edits[word.id] != null) return edits[word.id];
  if (edits[String(index)] != null) return edits[String(index)]; // legacy
  return word.w;
}

/** Slice 1: edit word -> captions.edits keyed by stable word ID. */
export function applyWordEdit(project, wordId, text) {
  const words = project.captions.words || [];
  const word = words.find((w) => w.id === wordId);
  if (!word) throw new Error(`unknown word id ${wordId}`);
  const clean = String(text || "").trim();
  if (!clean) throw new Error("word text must not be empty");
  project.captions.edits = { ...(project.captions.edits || {}) };
  project.captions.edits[wordId] = clean.slice(0, 80);
  return project;
}

/** Slice 1: split & trim -> split the containing segment at the word start. */
export function applySplitAtWord(project, word) {
  const before = (project.segments || []).length;
  project.segments = splitSegmentAt(project.segments || [], word.start);
  if (project.segments.length === before) {
    throw new Error("could not split: word is not inside a segment");
  }
  project.origin = "user";
  return project;
}

/**
 * Slice 1: remove caption & video -> carve ONE merged source-time span
 * [min start, max end] out of the segments. Adjacent words merge via
 * removeSpanFromSegments (see timebase.mergeSpans).
 */
export function applyRemoveWords(project, words) {
  if (!words.length) throw new Error("no words selected");
  const s = Math.min(...words.map((w) => w.start));
  const e = Math.max(...words.map((w) => w.end));
  const keptBefore = keptTotal(project.segments);
  project.segments = removeSpanFromSegments(project.segments || [], s, e);
  if (!(project.segments || []).length) {
    throw new Error("refusing to remove the entire clip");
  }
  if (keptTotal(project.segments) >= keptBefore) {
    throw new Error("nothing to remove: span is already dropped");
  }
  project.origin = "user";
  return project;
}

function keptTotal(segments) {
  return (segments || []).reduce((n, s) => n + (s.end - s.start), 0);
}

/** Menu descriptor for the transcript panel (slice 1 only). */
export function slice1Menu({ hasRange }) {
  return [
    { id: "edit", label: "Edit words" },
    { id: "split", label: "Split & trim" },
    { id: "remove", label: "Remove caption & video", danger: true,
      hint: hasRange ? "selected range" : undefined },
  ];
}
