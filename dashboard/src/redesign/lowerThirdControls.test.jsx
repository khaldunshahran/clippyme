import { describe, expect, test, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import {
  LowerThirdControls, LowerThirdPreview,
  buildLowerThirdSpec, LOWER_THIRD_PRESETS, LOWER_THIRD_ACCENTS, LOWER_THIRD_DEFAULT,
} from './lowerThirdControls';

describe('buildLowerThirdSpec', () => {
  test('fills defaults for an empty value', () => {
    const s = buildLowerThirdSpec(null);
    expect(s).toEqual(LOWER_THIRD_DEFAULT);
  });

  test('clamps invalid preset / accent / timing', () => {
    const s = buildLowerThirdSpec({ preset: 'nope', style: { accent: 'nope' }, startSec: -2, endSec: 1 });
    expect(s.preset).toBe('name_slide');
    expect(s.style.accent).toBe('rust');
    expect(s.startSec).toBe(0);
    // end must stay after start
    const s2 = buildLowerThirdSpec({ startSec: 5, endSec: 3 });
    expect(s2.endSec).toBeGreaterThan(s2.startSec);
  });

  test('keeps valid custom values', () => {
    const s = buildLowerThirdSpec({
      preset: 'bold_block',
      lines: { name: 'Jane', title: 'Berlin' },
      style: { font: 'Anton-Regular', accent: 'gold', align: 'center' },
      startSec: 1, endSec: 6,
    });
    expect(s.preset).toBe('bold_block');
    expect(s.lines.name).toBe('Jane');
    expect(s.style.font).toBe('Anton-Regular');
    expect(s.style.align).toBe('center');
  });
});

describe('LowerThirdControls', () => {
  test('offers 4–6 broadcast presets with color variants', () => {
    expect(LOWER_THIRD_PRESETS.length).toBeGreaterThanOrEqual(4);
    expect(LOWER_THIRD_PRESETS.length).toBeLessThanOrEqual(6);
    expect(LOWER_THIRD_ACCENTS.length).toBeGreaterThanOrEqual(3);
  });

  test('preset picker, text fields, font, timing and preview all render', () => {
    const onChange = vi.fn();
    render(<LowerThirdControls value={LOWER_THIRD_DEFAULT} onChange={onChange} />);
    for (const p of LOWER_THIRD_PRESETS) {
      expect(screen.getByTitle(p.desc)).toBeTruthy();
    }
    expect(screen.getByLabelText('Lower third name')).toBeTruthy();
    expect(screen.getByLabelText('Lower third title')).toBeTruthy();
    expect(screen.getByLabelText('Lower third font')).toBeTruthy();
    expect(screen.getByLabelText('Lower third start seconds')).toBeTruthy();
    expect(screen.getByLabelText('Lower third end seconds')).toBeTruthy();
    // live preview reflects the typed name (also echoed in the input value)
    expect(screen.getAllByText('Alex Rivera').length).toBeGreaterThanOrEqual(1);
  });

  test('typing a name emits a normalized spec object', () => {
    const onChange = vi.fn();
    render(<LowerThirdControls value={LOWER_THIRD_DEFAULT} onChange={onChange} />);
    fireEvent.change(screen.getByLabelText('Lower third name'), { target: { value: 'Sam Lee' } });
    expect(onChange).toHaveBeenCalledTimes(1);
    const spec = onChange.mock.calls[0][0];
    expect(spec).toMatchObject({
      preset: 'name_slide',
      lines: { name: 'Sam Lee', title: 'Founder, Brightline Studio' },
      startSec: 0.5, endSec: 4.5,
    });
    expect(spec.style.font).toBeTruthy();
  });

  test('honest note: backend burn-in not supported yet', () => {
    render(<LowerThirdControls value={LOWER_THIRD_DEFAULT} onChange={() => {}} />);
    expect(screen.getByText(/backend can.t burn lower thirds/i)).toBeTruthy();
  });

  test('preview renders every preset without crashing', () => {
    for (const p of LOWER_THIRD_PRESETS) {
      const { unmount } = render(
        <LowerThirdPreview spec={{ ...LOWER_THIRD_DEFAULT, preset: p.id }} />
      );
      unmount();
    }
  });
});
