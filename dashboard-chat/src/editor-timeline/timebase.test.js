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
