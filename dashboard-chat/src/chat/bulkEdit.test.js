/**
 * Task 19 — bulkEdit.js pure helper tests.
 *
 * Scope freezing (ordered unique; all = enumerated snapshot), intent
 * summaries, operation inspection (retryable/applied/undoable items) and
 * card-state lifetime classification. No React, no network.
 */
import { describe, expect, it } from 'vitest';
import {
  freezeScope,
  summarizeIntent,
  bulkItemsArray,
  retryableItems,
  appliedItems,
  undoableItems,
  isBulkOpenState,
  bulkOutcomeLabel,
} from './bulkEdit.js';

const clips = [{ index: 2 }, { index: 0 }, { index: 1 }];

describe('freezeScope', () => {
  it('all mode enumerates the current snapshot, ordered unique', () => {
    expect(freezeScope(clips, 'all')).toEqual([0, 1, 2]);
  });
  it('selected mode dedupes and sorts explicit indices', () => {
    expect(freezeScope(clips, 'selected', [2, 0, 2])).toEqual([0, 2]);
  });
  it('selected mode drops non-integer/negative indices', () => {
    expect(freezeScope(clips, 'selected', [1, -1, 'x', 1.5, null])).toEqual([1]);
  });
  it('rejects unknown modes', () => {
    expect(() => freezeScope(clips, 'everything')).toThrow(/unknown bulk scope mode/);
  });
  it('falls back to positional indices when clip.index is missing', () => {
    expect(freezeScope([{}, {}], 'all')).toEqual([0, 1]);
  });
});

describe('summarizeIntent', () => {
  it('summarizes a typed intent in one line', () => {
    expect(summarizeIntent({
      op: 'caption_appearance',
      params: { caption_style: 'hormozi_bold' },
      on_override: 'preserve',
    })).toBe('Caption style: caption_style=hormozi_bold');
  });
  it('flags explicit replacement', () => {
    expect(summarizeIntent({
      op: 'hook_timing', params: { end: 2 }, on_override: 'replace',
    })).toMatch(/\(replace overrides\)/);
  });
  it('handles unknown input gracefully', () => {
    expect(summarizeIntent(null)).toBe('');
  });
});

const op = (items, undo) => ({
  items: Object.fromEntries(items.map((i) => [String(i.clip_index), i])),
  ...(undo ? { undo } : {}),
});
const item = (clip_index, outcome, extra = {}) => ({ clip_index, outcome, ...extra });

describe('operation inspection', () => {
  it('bulkItemsArray returns a clip-ordered array from the dict form', () => {
    const o = op([item(2, 'applied'), item(0, 'planned')]);
    expect(bulkItemsArray(o).map((i) => i.clip_index)).toEqual([0, 2]);
  });
  it('retryableItems picks failed and pending only', () => {
    const o = op([
      item(0, 'applied'), item(1, 'failed'), item(2, 'pending'),
      item(3, 'conflict'), item(4, 'excluded'),
    ]);
    expect(retryableItems(o).map((i) => i.clip_index)).toEqual([1, 2]);
  });
  it('undoableItems excludes already-undone clips', () => {
    const o = op([item(0, 'applied'), item(1, 'applied'), item(2, 'failed')], {
      items: { 0: { outcome: 'undone', version: 9 } },
    });
    expect(undoableItems(o).map((i) => i.clip_index)).toEqual([1]);
    expect(appliedItems(o).map((i) => i.clip_index)).toEqual([0, 1]);
  });
});

describe('isBulkOpenState', () => {
  it('unapplied proposal states are open', () => {
    for (const s of ['idle', 'preparing', 'proposal', 'applying', 'retrying', 'undoing']) {
      expect(isBulkOpenState(s)).toBe(true);
    }
  });
  it('terminal states are not open', () => {
    for (const s of ['results', 'error', 'stale', 'done', 'whatever']) {
      expect(isBulkOpenState(s)).toBe(false);
    }
  });
});

describe('bulkOutcomeLabel', () => {
  it('labels known outcomes and passes through unknowns', () => {
    expect(bulkOutcomeLabel('preserved')).toBe('preserved override');
    expect(bulkOutcomeLabel('conflict')).toBe('conflict — clip moved');
    expect(bulkOutcomeLabel('mystery')).toBe('mystery');
  });
});
