/**
 * wordMenu: removed flag + restore (Phase A item 1).
 */
import { describe, expect, it } from 'vitest';
import {
  applyRemoveWords,
  applyRestoreWords,
  applyWordColors,
  slice1Menu,
  slice2Menu,
} from './wordMenu.js';

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

describe('applyWordColors', () => {
  it('sets color indexes by word id', () => {
    const p = project();
    applyWordColors(p, ['w0', 'w2'], 1);
    expect(p.captions.word_colors).toEqual({ w0: 1, w2: 1 });
    expect(p.origin).toBe('user');
  });

  it('applies to multiple ids at once', () => {
    const p = project();
    applyWordColors(p, ['w0', 'w1', 'w2'], 2);
    expect(p.captions.word_colors).toEqual({ w0: 2, w1: 2, w2: 2 });
  });

  it('deletes the key for color 0 (default = absent)', () => {
    const p = project();
    p.captions.word_colors = { w0: 1, w1: 2 };
    applyWordColors(p, ['w0'], 0);
    expect(p.captions.word_colors).toEqual({ w1: 2 });
  });

  it('rejects a bad color index', () => {
    expect(() => applyWordColors(project(), ['w0'], 3))
      .toThrow('color index must be 0, 1, or 2');
    expect(() => applyWordColors(project(), ['w0'], -1))
      .toThrow('color index must be 0, 1, or 2');
    expect(() => applyWordColors(project(), ['w0'], '1'))
      .toThrow('color index must be 0, 1, or 2');
  });

  it('rejects unknown word ids', () => {
    expect(() => applyWordColors(project(), ['w99'], 1))
      .toThrow('unknown word id w99');
  });

  it('rejects an empty selection', () => {
    expect(() => applyWordColors(project(), [], 1))
      .toThrow('no words selected');
  });

  it('dedupes repeated ids', () => {
    const p = project();
    applyWordColors(p, ['w1', 'w1'], 2);
    expect(p.captions.word_colors).toEqual({ w1: 2 });
  });
});

describe('slice2Menu', () => {
  it('keeps all slice-1 items and adds the Highlight submenu', () => {
    const items = slice2Menu({ hasRange: true, hasRemoved: true, allRemoved: false });
    expect(items.map((m) => m.id)).toEqual(['edit', 'split', 'restore', 'remove', 'highlight']);
    const hl = items.find((m) => m.id === 'highlight');
    expect(hl.label).toBe('Highlight');
    expect(hl.submenu.map((s) => s.id)).toEqual(['color-0', 'color-1', 'color-2']);
    expect(hl.submenu.map((s) => s.label)).toEqual(['Default', 'Color 1', 'Color 2']);
  });

  it('still shows only restore + highlight when everything is removed', () => {
    const ids = slice2Menu({ hasRange: false, hasRemoved: true, allRemoved: true })
      .map((m) => m.id);
    expect(ids).toEqual(['restore', 'highlight']);
  });
});
