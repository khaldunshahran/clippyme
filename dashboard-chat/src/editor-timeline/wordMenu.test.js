/**
 * wordMenu: removed flag + restore (Phase A item 1).
 */
import { describe, expect, it } from 'vitest';
import { applyRemoveWords, applyRestoreWords, slice1Menu } from './wordMenu.js';

const crop = { x: 0, y: 0, w: 1080, h: 1920 };
function project() {
  return {
    origin: 'auto',
    version: 1,
    segments: [{ id: 'seg0', start: 0, end: 30, crop: { ...crop } }],
    captions: {
      words: [
        { id: 'w0', w: 'hello', start: 1, end: 1.5 },
        { id: 'w1', w: 'world', start: 12, end: 12.5 },
        { id: 'w2', w: 'again', start: 25, end: 25.5 },
      ],
      edits: {},
    },
  };
}

describe('applyRemoveWords', () => {
  it('flags words removed and keeps their IDs', () => {
    const p = project();
    applyRemoveWords(p, [p.captions.words[1]]);
    expect(p.captions.words[1].removed).toBe(true);
    expect(p.captions.words[1].id).toBe('w1');
    expect(p.captions.words[0].removed).not.toBe(true);
  });

  it('carves the span from segments', () => {
    const p = project();
    applyRemoveWords(p, [p.captions.words[1]]);
    const kept = p.segments.map((s) => [s.start, s.end]);
    expect(kept).toEqual([[0, 12], [12.5, 30]]);
  });
});

describe('applyRestoreWords', () => {
  it('clears the flag and merges the span back', () => {
    const p = project();
    applyRemoveWords(p, [p.captions.words[1]]);
    applyRestoreWords(p, [p.captions.words[1]]);
    expect(p.captions.words[1].removed).toBe(false);
    expect(p.segments.map((s) => [s.start, s.end])).toEqual([[0, 30]]);
  });

  it('refuses when nothing removed is selected', () => {
    expect(() => applyRestoreWords(project(), [project().captions.words[0]]))
      .toThrow('no removed words selected');
  });

  it('restored words keep ID-keyed edits', () => {
    const p = project();
    p.captions.edits = { w1: 'WORLD' };
    applyRemoveWords(p, [p.captions.words[1]]);
    applyRestoreWords(p, [p.captions.words[1]]);
    expect(p.captions.edits).toEqual({ w1: 'WORLD' });
  });
});

describe('slice1Menu', () => {
  it('shows only Restore when everything selected is removed', () => {
    expect(slice1Menu({ hasRange: false, hasRemoved: true, allRemoved: true }))
      .toEqual([{ id: 'restore', label: 'Restore' }]);
  });

  it('adds Restore alongside remove for mixed selection', () => {
    const ids = slice1Menu({ hasRange: true, hasRemoved: true, allRemoved: false })
      .map((m) => m.id);
    expect(ids).toEqual(['edit', 'split', 'restore', 'remove']);
  });

  it('is unchanged for a normal selection', () => {
    const ids = slice1Menu({ hasRange: false, hasRemoved: false, allRemoved: false })
      .map((m) => m.id);
    expect(ids).toEqual(['edit', 'split', 'remove']);
  });
});
