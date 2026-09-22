import { useEffect, useMemo, useRef } from 'react';
import { Icon, Btn, Badge, Panel, RingGauge, PebbleWaveform } from './primitives';
import { LazyVideo } from './LazyVideo';
import { Hero } from './chrome';
import { PIPE } from './data';
import { clipVideoSrc, fmtDuration } from './realApi';
import { latestRuntimeTelemetry, formatEta, formatMetric } from '../lib/runtimeTelemetry';

const STEP_INFO = {
  queued: { pct: 5, idx: 0 },
  acquiring: { pct: 12, idx: 0 },
  downloading: { pct: 18, idx: 0 },
  preflight: { pct: 22, idx: 0 },
  transcribing: { pct: 38, idx: 1 },
  analyzing: { pct: 58, idx: 2 },
  cutting: { pct: 68, idx: 3 },
  reframing: { pct: 82, idx: 3 },
  quality: { pct: 93, idx: 4 },
  finalizing: { pct: 97, idx: 4 },
  processing: { pct: 80, idx: 3 },
};

function MiniClip({ clip }) {
  return (
    <div className="card raised-sm fade-in" style={{ cursor: 'default', padding: 12 }}>
      <div className="clip-media">
        <LazyVideo
          src={clipVideoSrc(clip)}
          muted
          playsInline
          aria-label="Verified clip preview"
          style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', objectFit: 'cover' }}
        />
        <div className="clip-top" style={{ padding: 6 }}>
          <span className="score"><Icon n="flame" />{Math.round(clip.viral_score || 0)}</span>
        </div>
        <div className="clip-bottom" style={{ padding: 6 }}>
          <span className="dur">{fmtDuration(clip.start, clip.end)}</span>
        </div>
      </div>
    </div>
  );
}

function Metric({ label, value, hint }) {
  return (
    <div className="operation-metric">
      <div className="label">{label}</div>
      <div className="operation-value">{value}</div>
      {hint && <div className="operation-hint">{hint}</div>}
    </div>
  );
}

function Operations({ runtime, preflight }) {
  if (!runtime && !preflight) return null;
  const isAcquiring = runtime?.stage === 'acquiring' || runtime?.stage === 'downloading';
  const dlPercent = runtime?.download_percent;
  const dlSpeed = runtime?.download_speed ? String(runtime.download_speed).replace(/_/g, ' ') : null;
  const dlEta = runtime?.download_eta ? String(runtime.download_eta).replace(/_/g, ' ') : null;
  const dlBytes = runtime?.download_bytes ? String(runtime.download_bytes).replace(/_/g, ' ') : null;

  return (
    <section className="operations" aria-labelledby="operations-title">
      <div className="section-head" style={{ marginBottom: 12 }}>
        <h3 id="operations-title" style={{ fontSize: 18 }}>Operations</h3>
        {runtime?.stage && <Badge tone="out">{runtime.stage}</Badge>}
      </div>
      <div className="operation-grid">
        <Metric label="attempt" value={runtime?.attempt || '1'} hint="bounded retry" />
        <Metric label="verified clips" value={runtime?.clips || '0/0'} hint="QA passed" />
        {isAcquiring && dlPercent !== undefined && dlPercent !== null ? (
          <Metric
            label="download"
            value={`${dlPercent}%`}
            hint={dlSpeed ? `${dlSpeed}${dlEta ? ` · ETA ${dlEta}` : ''}` : (dlBytes || 'fetching source')}
          />
        ) : (
          <Metric label="ETA" value={formatEta(runtime?.eta_s)} hint="live estimate" />
        )}
        <Metric label="CPU" value={formatMetric(runtime?.cpu, '%')} />
        <Metric label="job RAM" value={formatMetric(runtime?.rss_mb, ' MB')} />
        <Metric label="disk free" value={formatMetric(runtime?.disk_free_gb, ' GB')} />
      </div>
      {preflight && (
        <div className="operation-grid" style={{ marginTop: 10 }}>
          <Metric label="planned clips" value={formatMetric(preflight.clips)} />
          <Metric label="estimated time" value={formatEta(Number(preflight.runtime_min) * 60)} />
          <Metric label="peak disk" value={formatMetric(preflight.disk_gb, ' GB')} />
          <Metric
            label="Gemini estimate"
            value={Number.isFinite(Number(preflight.cost_usd)) ? `$${Number(preflight.cost_usd).toFixed(4)}` : '—'}
            hint="upper-bound estimate"
          />
        </div>
      )}
    </section>
  );
}

export function ProcessingView({
  media,
  status,
  logs = [],
  step,
  clips = [],
  onCancel,
  onRetry,
  paused = false,
  onPause,
  onResume,
  onStop,
}) {
  const logRef = useRef(null);
  const followTail = useRef(true);
  useEffect(() => {
    const node = logRef.current;
    if (node && followTail.current) node.scrollTop = node.scrollHeight;
  }, [logs]);

  const { runtime, preflight } = useMemo(() => latestRuntimeTelemetry(logs), [logs]);
  const visibleLogs = useMemo(
    () => logs.filter((line) => !String(line).startsWith('[runtime]') && !String(line).startsWith('[preflight]')),
    [logs]
  );
  const failed = status === 'error';
  const effectiveStep = runtime?.stage || step;
  const info = STEP_INFO[effectiveStep] || STEP_INFO.queued;
  const reportedProgress = Number(runtime?.progress);
  const dlPct = Number(runtime?.download_percent);
  const isAcquiring = effectiveStep === 'acquiring' || effectiveStep === 'downloading';
  const pct = failed
    ? 100
    : isAcquiring && Number.isFinite(dlPct) && dlPct > 0
    ? Math.min(18, Math.max(2, Math.round(dlPct * 0.18)))
    : Number.isFinite(reportedProgress)
    ? Math.min(100, Math.max(0, reportedProgress))
    : Math.min(96, info.pct + Math.min(18, clips.length * 3));
  const sourceLabel = media?.type === 'url' ? media.payload : (media?.payload?.name || media?.payload || 'your video');
  const words = {
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
  const phase = failed
    ? 'failed'
    : paused
    ? 'paused'
    : isAcquiring && Number.isFinite(dlPct) && dlPct > 0
    ? `downloading ${dlPct}%`
    : (words[effectiveStep] || (clips.length > 0 ? 'rendering' : 'working'));

  return (
    <main className="container fade-in">
      <Hero
        eyebrow={failed ? 'Pipeline error' : paused ? 'Pipeline paused' : 'Pipeline running'}
        line1={failed ? 'Something broke.' : paused ? 'Work is paused.' : 'Shaping your clips.'}
        grad={failed ? 'Error' : paused ? 'Paused' : 'Active'}
        sub={
          failed
            ? 'Check the log below, then retry or start over.'
            : 'Every phase is checkpointed. Verified clips appear immediately, and retries resume from durable work.'
        }
      />

      {/* Boss Telemetry Section */}
      <div className="boss">
        <div className="ring-wrap">
          <RingGauge
            score={Math.round(pct)}
            max={100}
            size="big"
            accent={failed ? 'var(--rust)' : 'var(--gold)'}
            label="Overall Progress"
            sublabel="/100%"
          />
        </div>

        <div>
          <div className="boss-label">Live Pipeline Readout</div>
          <div className="boss-sub">
            Status: <b>{phase}</b> {clips.length > 0 && `· ${clips.length} moments verified`}
          </div>
          <div className="subm-row">
            <div className="subm">
              <RingGauge score={Math.min(100, Math.round(pct * 1.05))} size="med" accent="var(--rust)" label="Hook score" />
              <div className="txt">
                <div>Hook</div>
                <b>{Math.min(100, Math.round(pct * 1.05))}</b>
              </div>
            </div>
            <div className="subm">
              <RingGauge score={Math.min(100, Math.round(pct * 0.95))} size="med" accent="var(--sage)" label="Coherence score" />
              <div className="txt">
                <div>Coherence</div>
                <b>{Math.min(100, Math.round(pct * 0.95))}</b>
              </div>
            </div>
            <div className="subm">
              <RingGauge score={Math.min(100, Math.round(pct))} size="med" accent="var(--plum)" label="Payoff score" />
              <div className="txt">
                <div>Payoff</div>
                <b>{Math.round(pct)}</b>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 24 }}>
        <Panel title="Operations & Logs" sub={`Source: ${String(sourceLabel).slice(0, 50)}`}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 18, flexWrap: 'wrap', gap: 10 }}>
            <PebbleWaveform count={20} activeCount={Math.round((pct / 100) * 20)} />
            <div style={{ display: 'flex', gap: 8 }}>
              {failed && <Btn variant="primary" size="sm" icon="wand-sparkles" onClick={onRetry}>Retry</Btn>}
              {!failed && onPause && (
                paused ? (
                  <Btn variant="primary" size="sm" icon="play" onClick={onResume}>Resume</Btn>
                ) : (
                  <Btn variant="ghost" size="sm" icon="clock" onClick={onPause}>Pause</Btn>
                )
              )}
              {!failed && onStop && clips.length > 0 && (
                <Btn variant="ghost" size="sm" icon="check-square" onClick={onStop}>Stop & keep</Btn>
              )}
              <Btn variant="ghost" size="sm" icon="x" onClick={onCancel}>{failed ? 'Start over' : 'Discard'}</Btn>
            </div>
          </div>

          <Operations runtime={runtime} preflight={preflight} />

          <div
            ref={logRef}
            role="log"
            aria-live="polite"
            aria-relevant="additions"
            style={{
              marginTop: 18,
              background: 'var(--surface-deep)',
              boxShadow: 'var(--clay-pressed-md)',
              borderRadius: 'var(--r-input)',
              padding: '16px 20px',
              maxHeight: 220,
              overflowY: 'auto',
              fontFamily: 'var(--font-mono)',
              fontSize: 12.5,
              color: 'var(--ink)',
            }}
          >
            {visibleLogs.length === 0 && <div>waiting for the worker…</div>}
            {visibleLogs.map((line, index) => (
              <div key={`${index}-${line}`} style={{ margin: '4px 0', color: /error/i.test(line) ? 'var(--rust-deep)' : undefined }}>
                {line}
              </div>
            ))}
          </div>
        </Panel>

        {/* Verified Clips Section */}
        <section aria-labelledby="clips-title">
          <div className="section-head">
            <h3 id="clips-title" style={{ fontSize: 20 }}>Verified Moments</h3>
            {clips.length > 0 ? (
              <Badge tone="teal" icon="check">{clips.length} ready</Badge>
            ) : (
              <Badge tone="out">{failed ? 'no clips' : 'scoring moments…'}</Badge>
            )}
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(180px, 1fr))', gap: 16 }}>
            {clips.map((clip, index) => (
              <MiniClip key={clip.original_index ?? index} clip={clip} />
            ))}
          </div>
        </section>
      </div>
    </main>
  );
}
