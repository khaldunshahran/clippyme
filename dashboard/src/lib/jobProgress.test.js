import { describe, expect, test } from 'vitest';
import { computeJobProgress, JOB_STEP_BASELINE } from './jobProgress';

describe('computeJobProgress', () => {
  test('baseline percent follows the pipeline step', () => {
    const { pct, phase } = computeJobProgress({ logs: [], step: 'transcribing', clipsCount: 0, status: 'processing' });
    expect(pct).toBe(JOB_STEP_BASELINE.transcribing);
    expect(phase).toBe('transcribing');
  });

  test('verified clips nudge the percent upward (capped)', () => {
    const base = computeJobProgress({ logs: [], step: 'cutting', clipsCount: 0, status: 'processing' }).pct;
    const withClips = computeJobProgress({ logs: [], step: 'cutting', clipsCount: 4, status: 'processing' }).pct;
    expect(withClips).toBeGreaterThan(base);
    expect(withClips).toBeLessThanOrEqual(96);
  });

  test('reported [runtime] progress wins over the baseline', () => {
    const { pct } = computeJobProgress({
      logs: ['[runtime] stage=analyzing progress=72'],
      step: 'analyzing', clipsCount: 0, status: 'processing',
    });
    expect(pct).toBe(72);
  });

  test('download percent maps into the early range while acquiring', () => {
    const { pct, phase } = computeJobProgress({
      logs: ['[runtime] stage=downloading download_percent=50'],
      step: 'downloading', clipsCount: 0, status: 'processing',
    });
    expect(pct).toBeGreaterThanOrEqual(2);
    expect(pct).toBeLessThanOrEqual(18);
    expect(phase).toBe('downloading 50%');
  });

  test('error status reports 100% failed', () => {
    const { pct, phase } = computeJobProgress({ logs: [], step: 'cutting', status: 'error' });
    expect(pct).toBe(100);
    expect(phase).toBe('failed');
  });

  test('paused flag surfaces in the phase', () => {
    const { phase } = computeJobProgress({ logs: [], step: 'cutting', status: 'processing', paused: true });
    expect(phase).toBe('paused');
  });

  test('unknown step falls back to a sane default', () => {
    const { pct, phase } = computeJobProgress({ logs: [], step: 'mystery', clipsCount: 0, status: 'processing' });
    expect(pct).toBe(JOB_STEP_BASELINE.queued);
    expect(phase).toBe('working');
  });
});
