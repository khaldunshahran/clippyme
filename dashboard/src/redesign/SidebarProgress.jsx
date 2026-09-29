// Nugget — live generation progress in the sidebar.
//
// Renders ONLY while a job is genuinely active (status === 'processing').
// Follows the same telemetry the ProcessingView uses (lib/jobProgress), so
// the percent matches the big view exactly. Hidden completely when idle or
// when the backend is unreachable (no fake "0%" placeholder).
import { useMemo } from 'react';
import { Icon } from './icon';
import { computeJobProgress } from '../lib/jobProgress';

function jobName(media) {
  if (!media) return 'Rendering clips';
  const p = media.payload;
  if (typeof p === 'string') {
    const s = p.trim();
    if (!s) return 'Rendering clips';
    if (/^https?:\/\//i.test(s)) {
      try {
        return new URL(s).hostname.replace(/^www\./, '');
      } catch {
        return s.length > 34 ? `${s.slice(0, 34)}…` : s;
      }
    }
    return s.length > 34 ? `${s.slice(0, 34)}…` : s;
  }
  if (p && typeof p === 'object' && typeof p.name === 'string' && p.name) return p.name;
  return 'Rendering clips';
}

export function SidebarProgress({ status, step, logs = [], clipsCount = 0, media, paused, onOpen }) {
  const active = status === 'processing';
  const { pct, phase } = useMemo(
    () => computeJobProgress({ logs, step, clipsCount, status, paused }),
    [logs, step, clipsCount, status, paused]
  );
  if (!active) return null;
  return (
    <button
      type="button"
      className="sb-progress fade-in"
      onClick={onOpen}
      aria-label={`Rendering ${jobName(media)} — ${pct} percent, ${phase}. Open the job.`}
      title="Open the running job"
    >
      <div className="sb-progress-top">
        <span className="sb-progress-ic" aria-hidden="true">
          <Icon n="loader" />
        </span>
        <span className="sb-progress-name">{jobName(media)}</span>
        <span className="sb-progress-pct">{pct}%</span>
      </div>
      <div className="sb-progress-bar" role="progressbar" aria-valuenow={pct} aria-valuemin={0} aria-valuemax={100}>
        <span className="sb-progress-fill" style={{ width: `${pct}%` }} />
      </div>
      <div className="sb-progress-phase">{paused ? 'paused' : phase}</div>
    </button>
  );
}
