import { Loader2 } from 'lucide-react';
import { fmtEta } from '../api/chatApi.js';

/**
 * Bare progress body: no card wrapper, no thumbnail/title of its own.
 * Shimmer + "Starting your clip job…" while no data yet (the 4bd36f9 fix:
 * render the real pill as soon as progress data exists), then bar + % + ETA
 * + status indicator. Used inside the validation card so ONE card morphs
 * from "link looks good" into live progress — no duplicate thumbnail.
 */
export function ProgressBody({ progress, status }) {
  if (!progress) {
    return (
      <div>
        <div style={{ height: 6, borderRadius: 99, background: 'rgba(255,255,255,0.08)', overflow: 'hidden', position: 'relative' }}>
          <div style={{
            position: 'absolute', inset: 0, width: '40%',
            background: 'var(--nc-accent)', borderRadius: 99,
            animation: 'nc-bar-slide 1.2s ease-in-out infinite',
          }} />
        </div>
        <div style={{ marginTop: 8, fontSize: 12.5, color: 'var(--nc-text-dim)', display: 'flex', alignItems: 'center', gap: 6 }}>
          <Loader2 size={13} style={{ animation: 'nc-spin 1.2s linear infinite' }} />
          Starting your clip job…
        </div>
      </div>
    );
  }

  const pct = Math.max(0, Math.min(100, Math.round(progress.progress ?? 0)));
  const eta = fmtEta(progress.eta_seconds);
  const done = status === 'done';

  return (
    <div>
      <div style={{ height: 6, borderRadius: 99, background: 'rgba(255,255,255,0.08)', overflow: 'hidden' }}>
        <div style={{
          height: '100%', width: `${pct}%`, borderRadius: 99,
          background: 'linear-gradient(90deg, var(--nc-accent), #4da3e8)',
          transition: 'width 0.6s cubic-bezier(0.22,1,0.36,1)',
        }} />
      </div>
      <div style={{ marginTop: 8, fontSize: 12.5, color: 'var(--nc-text-dim)', display: 'flex', alignItems: 'center', gap: 8 }}>
        <span style={{ fontVariantNumeric: 'tabular-nums', fontWeight: 600, color: 'var(--nc-text)' }}>{pct}%</span>
        {eta && <span>~ {eta}</span>}
        <span style={{ display: 'inline-flex', alignItems: 'center', gap: 4 }}>
          {!done && <span style={{ width: 6, height: 6, borderRadius: '50%', background: 'var(--nc-accent)', animation: 'nc-pulse-dot 1.6s ease-in-out infinite' }} />}
          {done ? 'done' : 'clipping'}
        </span>
      </div>
    </div>
  );
}

/** Minimal progress card: thumbnail + full title + % + ETA. No pipeline internals. */
export default function ProgressPill({ progress, title, thumbnail, state }) {
  return (
    <div className="nc-card nc-anim-fade-up" style={{ padding: 14, display: 'flex', gap: 12, alignItems: 'center', maxWidth: 560 }}>
      {thumbnail
        ? <img src={thumbnail} alt="" style={{ width: 96, height: 54, objectFit: 'cover', borderRadius: 8, flexShrink: 0, background: '#000' }} draggable={false} />
        : <div className="nc-skeleton" style={{ width: 96, height: 54, flexShrink: 0, borderRadius: 8 }} />}
      <div style={{ flex: 1, minWidth: 0 }}>
        {!progress ? (
          <div className="nc-skeleton" style={{ height: 14, width: '70%', marginBottom: 10 }} />
        ) : (
          <div style={{ fontSize: 13.5, fontWeight: 600, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', marginBottom: 10 }}>
            {title || 'Clipping your video…'}
          </div>
        )}
        <ProgressBody progress={progress} status={state} />
      </div>
    </div>
  );
}
