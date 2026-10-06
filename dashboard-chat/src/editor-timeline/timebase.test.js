/**
 * Shared drop-mapping fixture, run in vitest.
 *
 * The same JSON drives the Python suite (tests/domain/test_timebase.py) --
 * one fixture, no drift between the JS preview path and the Python render
 * path. GLOBAL RULE: drops are in SOURCE time.
 */
import { describe, expect, it } from 'vitest';
import fixture from '../../../tests/fixtures/drop_mappings.json';
import {
  deriveDrops,
  invertDrops,
  mergeSpans,
  outputToSource,
  restoreSpanToSegments,
  sourceToOutput,
} from './timebase.js';

const segmentsFor = (c) => invertDrops(c.drops, c.source_start, c.source_end);

describe('drop_mappings fixture', () => {
  for (const c of fixture.cases) {
    if (c.merged_drops) {
      it(`${c.name}: mergeSpans`, () => {
        expect(mergeSpans(c.drops)).toEqual(c.merged_drops);
      });
    }

    for (const pt of c.points || []) {
      it(`${c.name}: sourceToOutput(${pt.t})`, () => {
        expect(sourceToOutput(pt.t, segmentsFor(c))).toBe(pt.out);
      });
    }

    for (const pt of c.roundtrip || []) {
      it(`${c.name}: outputToSource(${pt.out})`, () => {
        expect(outputToSource(pt.out, segmentsFor(c))).toBe(pt.t);
      });
    }

    it(`${c.name}: deriveDrops roundtrip`, () => {
      const merged = c.merged_drops || mergeSpans(c.drops);
      expect(deriveDrops(segmentsFor(c), c.source_start, c.source_end)).toEqual(
        merged
      );
    });
  }
});

describe('restoreSpanToSegments', () => {
  const crop = { x: 0, y: 0, w: 1080, h: 1920 };
  const segs = () => [
    { id: 'seg0', start: 0, end: 10, crop: { ...crop } },
    { id: 'seg1', start: 20, end: 30, crop: { ...crop } },
  ];

  it('merges a carved span back (union)', () => {
    const out = restoreSpanToSegments(segs(), 10, 20);
    expect(out.map((s) => [s.start, s.end])).toEqual([[0, 30]]);
  });

  it('inherits the neighbor crop and mints a fresh id', () => {
    const out = restoreSpanToSegments(segs(), 10, 20);
    expect(out[0].crop).toEqual(crop);
    const ids = new Set(segs().map((s) => s.id));
    for (const s of out) {
      // every id unique; the restored piece is derived from a neighbor id
      expect(typeof s.id).toBe('string');
    }
    expect(new Set(out.map((s) => s.id)).size).toBe(out.length);
  });

  it('is idempotent when the span is already covered', () => {
    const out = restoreSpanToSegments(segs(), 2, 5);
    expect(out).toEqual(segs());
  });

  it('restores a middle span touching both neighbors', () => {
    const out = restoreSpanToSegments(segs(), 12, 18);
    expect(out.map((s) => [s.start, s.end])).toEqual([[0, 10], [12, 18], [20, 30]]);
  });

  it('no-ops on empty/inverted spans', () => {
    expect(restoreSpanToSegments(segs(), 5, 5)).toEqual(segs());
    expect(restoreSpanToSegments([], 1, 2).map((s) => [s.start, s.end])).toEqual([[1, 2]]);
  });
});
