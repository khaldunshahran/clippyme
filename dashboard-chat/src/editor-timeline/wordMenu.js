/**
 * Slice-1 word actions: pure project transforms.
 *
 * Each takes a project dict (already cloned by the edit queue) and returns
 * the mutated project. All times are SOURCE time -- segments store source
 * seconds, and drops are derived as segment gaps (see timebase.js).
 */
import {
  removeSpanFromSegments,
  restoreSpanToSegments,
  splitSegmentAt,
} from './timebase.js';

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

/**
 * Slice 2: highlight words -> captions.word_colors keyed by stable word ID.
 * colorIndex 0 = template default (deletes the key -- 0 is stored as absent),
 * 1 = Color 1, 2 = Color 2. Applies to every id in wordIds.
 */
export function applyWordColors(project, wordIds, colorIndex) {
  if (!Number.isInteger(colorIndex) || colorIndex < 0 || colorIndex > 2) {
    throw new Error(`color index must be 0, 1, or 2, got ${colorIndex}`);
  }
  const ids = [...new Set(wordIds || [])];
  if (!ids.length) throw new Error("no words selected");
  const known = new Set((project?.captions?.words || []).map((w) => w.id));
  for (const id of ids) {
    if (!known.has(id)) throw new Error(`unknown word id ${id}`);
  }
  if (!project.captions) project.captions = {};
  const wc = { ...(project.captions.word_colors || {}) };
  for (const id of ids) {
    if (colorIndex === 0) delete wc[id];
    else wc[id] = colorIndex;
  }
  project.captions.word_colors = wc;
  project.origin = "user";
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
  // Flag the words: they stay in captions.words with their stable IDs so
  // the transcript can strike them through, they can be restored, and
  // ID-keyed data (colors/emoji) survives. compose.py skips removed words.
  for (const w of words) w.removed = true;
  project.origin = "user";
  return project;
}

/**
 * Slice 1: restore removed words -> clear the flag and merge the
 * [min start, max end] span back into the segments (union).
 */
export function applyRestoreWords(project, words) {
  const targets = (words || []).filter((w) => w && w.removed);
  if (!targets.length) throw new Error("no removed words selected");
  const s = Math.min(...targets.map((w) => w.start));
  const e = Math.max(...targets.map((w) => w.end));
  project.segments = restoreSpanToSegments(project.segments || [], s, e);
  for (const w of targets) w.removed = false;
  project.origin = "user";
  return project;
}

function keptTotal(segments) {
  return (segments || []).reduce((n, s) => n + (s.end - s.start), 0);
}

/** Menu descriptor for the transcript panel (slice 1 only). */
export function slice1Menu({ hasRange, hasRemoved, allRemoved }) {
  if (allRemoved) return [{ id: "restore", label: "Restore" }];
  const items = [
    { id: "edit", label: "Edit words" },
    { id: "split", label: "Split & trim" },
  ];
  if (hasRemoved) items.push({ id: "restore", label: "Restore" });
  items.push({
    id: "remove", label: "Remove caption & video", danger: true,
    hint: hasRange ? "selected range" : undefined,
  });
  return items;
}

/** Menu descriptor for the transcript panel (slice 2: slice 1 + Highlight). */
export function slice2Menu(opts) {
  return [
    ...slice1Menu(opts),
    {
      id: "highlight",
      label: "Highlight",
      submenu: [
        { id: "color-0", label: "Default" },
        { id: "color-1", label: "Color 1" },
        { id: "color-2", label: "Color 2" },
      ],
    },
  ];
}
