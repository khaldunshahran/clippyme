// Nugget — shared live job-progress computation.
//
// The percent/phase formula used to live inline in ProcessingView
// (redesign/processing.jsx). Extracted here so the sidebar's live
// generation-progress block reuses the exact same numbers (no drift
// between the big view and the sidebar chip).
import { latestRuntimeTelemetry } from './runtimeTelemetry';

// Baseline percent per pipeline stage — mirrors STEP_INFO in
// redesign/processing.jsx. Keep the two in sync if stages change.
export const JOB_STEP_BASELINE = {
  queued: 5,
  acquiring: 12,
  downloading: 18,
  preflight: 22,
  transcribing: 38,
  analyzing: 58,
  cutting: 68,
  reframing: 82,
  quality: 93,
  finalizing: 97,
  processing: 80,
};

const PHASE_WORDS = {
  queued: 'queued',
  acquiring: 'fetching',
  downloading: 'fetching',
  preflight: 'checking capacity',
  transcribing: 'transcribing',
  analyzing: 'scoring',
  cutting: 'cutting',
  reframing: 'rendering',
  quality: 'verifying',
  finalizing: 'finalizing',
  processing: 'rendering',
  completed: 'complete',
};

/**
 * Compute a live { pct, phase, effectiveStep } for an in-flight job.
 *
 * @param {object} args
 * @param {string[]} args.logs     pipeline log lines (carries [runtime] telemetry)
 * @param {string}   args.step     current step id (e.g. 'transcribing')
 * @param {number}   args.clipsCount number of verified clips so far
 * @param {string}   args.status   job status ('processing' | 'error' | ...)
 * @param {boolean}  args.paused   pipeline paused flag
 */
export function computeJobProgress({ logs = [], step, clipsCount = 0, status, paused = false } = {}) {
  const { runtime } = latestRuntimeTelemetry(logs);
  const effectiveStep = runtime?.stage || step;
  const baseline = JOB_STEP_BASELINE[effectiveStep] ?? JOB_STEP_BASELINE.queued;
  const reported = Number(runtime?.progress);
  const dlPct = Number(runtime?.download_percent);
  const isAcquiring = effectiveStep === 'acquiring' || effectiveStep === 'downloading';
  const failed = status === 'error';

  const pct = failed
    ? 100
    : isAcquiring && Number.isFinite(dlPct) && dlPct > 0
      ? Math.min(18, Math.max(2, Math.round(dlPct * 0.18)))
      : Number.isFinite(reported)
        ? Math.min(100, Math.max(0, reported))
        : Math.min(96, baseline + Math.min(18, clipsCount * 3));

  const phase = failed
    ? 'failed'
    : paused
      ? 'paused'
      : isAcquiring && Number.isFinite(dlPct) && dlPct > 0
        ? `downloading ${dlPct}%`
        : (PHASE_WORDS[effectiveStep] || (clipsCount > 0 ? 'rendering' : 'working'));

  return { pct: Math.round(pct), phase, effectiveStep };
}
