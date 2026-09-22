// ClippyMe redesign — AI Highlights & Supercut Studio:
// exact Create flow architecture (Hero + Source + Presets + Options Recipe + Summary Bar)
// with multi-tier generation (<60s, ~120s, 180s-720s) and full post-generation video editing suite.
import { useState, useRef, useEffect, useMemo } from 'react';
import { Icon, Btn, Badge, Panel, Segmented, Switch, Stepper } from './primitives';
import { Hero } from './chrome';
import { LANGUAGES, GEMINI_MODELS, HOOK_STYLE_DEFAULT } from './data';
import { HookStyleControls, HookPreview } from './hookStyle';
import { SubtitleControls } from './subtitleControls';
import { LogoControls, GradeControls } from './layerControls';
import { BannerControls } from './bannerControls';
import { EditClipModal } from './captions';
import { LazyVideo } from './LazyVideo';
import {
  generateAllHighlights,
  applyEditHighlight,
  getHighlights,
  deleteHighlight,
  retryJobApi,
  fmtDuration,
} from './realApi';
import { submitProcessJob, pollJob } from '../lib/api';
import { latestRuntimeTelemetry, formatEta, formatMetric } from '../lib/runtimeTelemetry';
import { readStoredJson, writeStoredJson, removeStoredValue } from '../lib/storage';

const HIGHLIGHTS_SESSION_KEY = 'clippyme_highlights_session';

export function loadPersistedHighlightsSession() {
  const value = readStoredJson(HIGHLIGHTS_SESSION_KEY, null);
  if (!value || typeof value !== 'object' || !value.jobId) return null;
  // Expire after 7 days
  if (Date.now() - (value.timestamp || 0) > 7 * 24 * 60 * 60 * 1000) {
    removeStoredValue(HIGHLIGHTS_SESSION_KEY);
    return null;
  }
  return value;
}

export function savePersistedHighlightsSession(session) {
  writeStoredJson(HIGHLIGHTS_SESSION_KEY, { ...session, timestamp: Date.now() });
}

export function clearPersistedHighlightsSession() {
  removeStoredValue(HIGHLIGHTS_SESSION_KEY);
}

const HIGHLIGHTS_PIPE = [
  { id: 'acquire', name: 'Acquire', meta: 'Fetching source video', icon: 'download' },
  { id: 'transcribe', name: 'Transcribe', meta: 'Speech-to-text with timestamps', icon: 'mic' },
  { id: 'analyze', name: 'Narrative AI', meta: 'Scoring viral moments with Gemini', icon: 'sparkles' },
  { id: 'compose', name: 'Compose & Subtitles', meta: 'Multi-cut sizing & karaoke', icon: 'layers' },
  { id: 'deliver', name: 'Deliver Reels', meta: 'Multi-tier packages (<60s, ~120s, 180s+)', icon: 'film' },
];

function HighlightsMetric({ label, value, hint }) {
  return (
    <div className="operation-metric">
      <div className="label">{label}</div>
      <div className="operation-value">{value}</div>
      {hint && <div className="operation-hint">{hint}</div>}
    </div>
  );
}

function HighlightsOperations({ runtime, preflight }) {
  if (!runtime && !preflight) return null;
  const isAcquiring = runtime?.stage === 'acquiring' || runtime?.stage === 'downloading';
  const dlPercent = runtime?.download_percent;
  const dlSpeed = runtime?.download_speed ? String(runtime.download_speed).replace(/_/g, ' ') : null;
  const dlEta = runtime?.download_eta ? String(runtime.download_eta).replace(/_/g, ' ') : null;
  const dlBytes = runtime?.download_bytes ? String(runtime.download_bytes).replace(/_/g, ' ') : null;

  return (
    <section className="operations" aria-labelledby="hl-operations-title">
      <div className="stream-head">
        <h3 id="hl-operations-title" aria-level="2">Operations</h3>
        {runtime?.stage && <Badge tone="out">{runtime.stage}</Badge>}
      </div>
      <div className="operation-grid">
        <HighlightsMetric label="attempt" value={runtime?.attempt || '1'} hint="bounded retry" />
        <HighlightsMetric label="packages" value="3 tiers" hint="micro / story / extended" />
        {isAcquiring && dlPercent !== undefined && dlPercent !== null ? (
          <HighlightsMetric
            label="download"
            value={`${dlPercent}%`}
            hint={dlSpeed ? `${dlSpeed}${dlEta ? ` · ETA ${dlEta}` : ''}` : (dlBytes || 'fetching source')}
          />
        ) : (
          <HighlightsMetric label="ETA" value={formatEta(runtime?.eta_s)} hint="live estimate" />
        )}
        <HighlightsMetric label="CPU" value={formatMetric(runtime?.cpu, '%')} />
        <HighlightsMetric label="job RAM" value={formatMetric(runtime?.rss_mb, ' MB')} />
        <HighlightsMetric label="disk free" value={formatMetric(runtime?.disk_free_gb, ' GB')} />
      </div>
      {preflight && (
        <div className="operation-grid operation-grid-secondary">
          <HighlightsMetric label="source duration" value={formatEta(preflight.duration_seconds)} />
          <HighlightsMetric label="estimated time" value={formatEta(Number(preflight.runtime_min) * 60)} />
          <HighlightsMetric label="peak disk" value={formatMetric(preflight.disk_gb, ' GB')} />
          <HighlightsMetric label="Gemini estimate" value={Number.isFinite(Number(preflight.cost_usd)) ? `$${Number(preflight.cost_usd).toFixed(4)}` : '—'} hint="narrative analysis" />
        </div>
      )}
    </section>
  );
}

function HighlightsProcessingView({
  status,
  statusText,
  logs = [],
  onCancel,
  onRetry,
}) {
  const logRef = useRef(null);
  const followTail = useRef(true);

  useEffect(() => {
    const node = logRef.current;
    if (node && followTail.current) node.scrollTop = node.scrollHeight;
  }, [logs]);

  const { runtime, preflight } = useMemo(() => latestRuntimeTelemetry(logs), [logs]);
  const visibleLogs = useMemo(() => logs.filter((line) => !String(line).startsWith('[runtime]') && !String(line).startsWith('[preflight]')), [logs]);
  const failed = status === 'error';
  const effectiveStep = runtime?.stage || (status === 'synthesizing' ? 'compose' : 'transcribing');

  // Compute active pipeline step index (0-4)
  let activeIdx = 0;
  if (effectiveStep === 'transcribing' || effectiveStep === 'preflight') activeIdx = 1;
  else if (effectiveStep === 'analyzing' || status === 'synthesizing') activeIdx = 2;
  else if (effectiveStep === 'compose' || effectiveStep === 'reframing' || effectiveStep === 'cutting') activeIdx = 3;
  else if (status === 'complete') activeIdx = 4;

  const dlPct = Number(runtime?.download_percent);
  const isAcquiring = (effectiveStep === 'acquiring' || effectiveStep === 'downloading');
  const reportedProgress = Number(runtime?.progress);
  const pct = failed
    ? 100
    : isAcquiring && Number.isFinite(dlPct) && dlPct > 0
    ? Math.min(20, Math.max(5, Math.round(dlPct * 0.2)))
    : status === 'synthesizing'
    ? 85
    : Number.isFinite(reportedProgress)
    ? Math.min(100, Math.max(0, reportedProgress))
    : (activeIdx * 22 + 10);

  const phase = failed
    ? 'failed'
    : status === 'synthesizing'
    ? 'synthesizing reels'
    : isAcquiring && Number.isFinite(dlPct) && dlPct > 0
    ? `downloading ${dlPct}%`
    : effectiveStep || 'processing';

  return (
    <main className="container fade-in">
      <Hero
        eyebrow={failed ? 'Pipeline error' : status === 'synthesizing' ? 'Synthesizing highlights' : 'Pipeline running'}
        line1={failed ? 'Something broke.' : 'Creating highlight reels.'}
        grad={status === 'synthesizing' ? 'AI supercuts in progress.' : undefined}
        sub={statusText || 'Downloading, transcribing, and scoring viral narrative moments...'}
      />
      <div className="proc">
        <aside className="proc-aside">
          <Panel>
            <div className="pipe">
              {HIGHLIGHTS_PIPE.map((pipelineStep, index) => {
                const done = !failed && index < activeIdx;
                const active = !failed && index === activeIdx;
                return (
                  <div key={pipelineStep.id} className={`pstep${done ? ' done' : active ? ' active' : ''}`} aria-current={active ? 'step' : undefined}>
                    <div className="rail">
                      <div className="pdot"><Icon n={done ? 'check' : pipelineStep.icon} /></div>
                      {index < HIGHLIGHTS_PIPE.length - 1 && <div className="pseg-v" />}
                    </div>
                    <div className="pbody">
                      <div className="pname">{pipelineStep.name}</div>
                      <div className="pmeta">{active ? (statusText || `${pipelineStep.meta} …`) : done ? 'done' : pipelineStep.meta}</div>
                    </div>
                  </div>
                );
              })}
            </div>
          </Panel>
        </aside>

        <div>
          <Panel>
            <div className="pbar-wrap" role="progressbar" aria-label={`Pipeline ${phase}`} aria-valuemin="0" aria-valuemax="100" aria-valuenow={Math.round(pct)}>
              <div className="pbar"><i style={{ width: `${pct}%`, background: failed ? 'var(--danger)' : undefined }} /></div>
              <div className="pbar-pct">{phase}</div>
            </div>

            <HighlightsOperations runtime={runtime} preflight={preflight} stage={effectiveStep} />

            <div className="terminal">
              <div className="term-head">
                <span className="dot" /><span className="dot" /><span className="dot" />
                <span className="term-title">highlights · console</span>
                <div style={{ marginLeft: 'auto' }}>
                  <button type="button" className="action-btn" title="Copy console output" aria-label="Copy console output"
                    onClick={() => {
                      try { navigator.clipboard?.writeText?.(visibleLogs.join('\n')); } catch { /* ignore */ }
                    }}>
                    <Icon n="clipboard" />
                  </button>
                </div>
              </div>
              <div ref={logRef} className="term-body" tabIndex={0} aria-label="Terminal log output"
                onScroll={(e) => {
                  const el = e.currentTarget;
                  followTail.current = el.scrollHeight - el.scrollTop - el.clientHeight < 32;
                }}>
                {visibleLogs.length === 0 ? (
                  <div className="empty" style={{ padding: 24, textAlign: 'center' }}>Connecting to processing worker...</div>
                ) : (
                  visibleLogs.map((l, i) => <div key={i} className="line">{l}</div>)
                )}
              </div>
            </div>

            <div style={{ display: 'flex', gap: 12, marginTop: 16, justifyContent: 'flex-end', alignItems: 'center' }}>
              {failed && onRetry && (
                <Btn variant="grad" icon="wand-sparkles" onClick={onRetry}>
                  Retry from Checkpoint
                </Btn>
              )}
              <Btn variant="secondary" icon="x" onClick={onCancel}>
                Cancel & Start Over
              </Btn>
            </div>
          </Panel>
        </div>
      </div>
    </main>
  );
}

const HIGHLIGHT_PRESETS = [
  {
    id: 'all_tiers',
    title: 'All Tiers (Auto)',
    desc: 'Generates Micro (<60s), Story (~2m), & Extended compilations',
    icon: 'sparkles',
  },
  {
    id: 'micro_only',
    title: 'Micro-Shorts (<60s)',
    desc: 'Ultra-fast pacing & viral hooks for TikTok, Shorts, and Reels',
    icon: 'smartphone',
  },
  {
    id: 'story_digest',
    title: 'Story Digest (~2m)',
    desc: 'Cohesive narrative arc (Hook → Setup → Core Beats → Outro)',
    icon: 'book-open',
  },
  {
    id: 'extended_supercut',
    title: 'Extended Supercut (5m+)',
    desc: 'Comprehensive highlight compilation for YouTube long-form',
    icon: 'film',
  },
];

function PresetCards({ presets, active, onPick }) {
  return (
    <div className="preset-row">
      {presets.map((p) => (
        <div
          key={p.id}
          role="button"
          tabIndex={0}
          aria-pressed={active === p.id}
          className={'preset' + (active === p.id ? ' on' : '')}
          onClick={() => onPick(p.id)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' || e.key === ' ') {
              e.preventDefault();
              onPick(p.id);
            }
          }}
        >
          <span className="pcheck"><Icon n="check" /></span>
          <span className="pico"><Icon n={p.icon} /></span>
          <span className="pt">{p.title}</span>
          <span className="pd">{p.desc}</span>
        </div>
      ))}
    </div>
  );
}

function SourcePanel({ opts, set, disabled }) {
  const [drag, setDrag] = useState(false);
  const fileInput = useRef(null);
  const pickFile = (f) => f && set({ file: f, fileName: f.name });

  return (
    <Panel title="Source" sub="Paste a link or drop a file" icon="link">
      <div>
        <Segmented
          full
          value={opts.source}
          onChange={(id) => set({ source: id })}
          options={[
            { id: 'url', label: 'URL', icon: 'globe' },
            { id: 'file', label: 'Upload', icon: 'file-up' },
          ]}
        />
        <div style={{ height: 14 }} />

        {opts.source === 'url' ? (
          <div className="input">
            <Icon n="link" />
            <input
              value={opts.url}
              placeholder="Paste a video link (YouTube, Twitch, or Kick)"
              disabled={disabled}
              onChange={(e) => set({ url: e.target.value })}
            />
            <button
              type="button"
              className="paste"
              onClick={async () => {
                try {
                  const text = await navigator.clipboard.readText();
                  if (text) set({ url: text.trim() });
                } catch {
                  /* clipboard blocked */
                }
              }}
            >
              <Icon n="clipboard" />Paste
            </button>
          </div>
        ) : (
          <div
            className={'dropzone' + (opts.file ? ' has' : drag ? ' drag' : '')}
            role="button"
            tabIndex={0}
            aria-label={opts.file ? 'Remove selected video' : 'Choose a video file'}
            onDragOver={(e) => { e.preventDefault(); setDrag(true); }}
            onDragLeave={() => setDrag(false)}
            onDrop={(e) => { e.preventDefault(); setDrag(false); pickFile(e.dataTransfer.files?.[0]); }}
            onClick={() => { if (opts.file) { set({ file: null, fileName: '' }); } else { fileInput.current?.click(); } }}
            onKeyDown={(e) => {
              if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); e.currentTarget.click(); }
            }}
          >
            <input
              ref={fileInput}
              type="file"
              accept="video/*,.mp4,.mov,.webm,.mkv,.m4v,.avi"
              hidden
              onChange={(e) => pickFile(e.target.files?.[0])}
            />
            <div className="dz-ico"><Icon n={opts.file ? 'file-video' : 'upload'} /></div>
            {opts.file ? (
              <div>
                <b>{opts.fileName}</b>
                <div className="label" style={{ marginTop: 6 }}>Ready · click to remove</div>
              </div>
            ) : (
              <div>
                Drop a video or <b style={{ color: 'var(--brand-blue)' }}>browse</b>
                <div className="label" style={{ marginTop: 6, textTransform: 'none', letterSpacing: 0 }}>
                  MP4 · MOV · WEBM · up to 16&nbsp;GB
                </div>
              </div>
            )}
          </div>
        )}

        <div style={{ marginTop: 18, paddingTop: 18, borderTop: '1px solid var(--line-1)' }}>
          <div className="field" style={{ marginBottom: 0 }}>
            <span className="field-label">
              <Icon n="sparkles" style={{ color: 'var(--brand-blue)' }} /> AI instructions · optional
            </span>
            <textarea
              className="ta"
              rows="2"
              value={opts.instructions || ''}
              placeholder="e.g. “Find the funniest moments” or “Focus on the main interview debate”"
              onChange={(e) => set({ instructions: e.target.value })}
            />
          </div>
        </div>
      </div>
    </Panel>
  );
}

function OptRow({ icon, label, desc, on, set, onConfig, configActive }) {
  return (
    <div className={'row opt' + (on ? ' on' : '')}>
      <div className="row-left">
        <div className="ricon">
          <Icon n={icon} />
        </div>
        <div className="row-text">
          <div className="title">{label}</div>
          <div className="desc">{desc}</div>
        </div>
      </div>
      <div className="row-control">
        {onConfig && on && (
          <button
            type="button"
            className={'cfg-btn' + (configActive ? ' active' : '')}
            onClick={onConfig}
            aria-label={'Configure ' + label}
          >
            <Icon n="sliders-horizontal" />
          </button>
        )}
        <Switch on={on} onChange={set} />
      </div>
    </div>
  );
}

const SUB_KEYMAP = {
  mode: 'subMode', preset: 'subPreset', font: 'subFont', font_color: 'subColor',
  outline_color: 'subStroke', font_size: 'subFontSize', border_width: 'subOutlineW',
  bg: 'subBg', position: 'subPosition', align: 'subAlign', offset_y: 'subOffsetY',
};

function SubConfig({ opts, set }) {
  const value = {
    mode: opts.subMode || 'karaoke',
    preset: opts.subPreset || 'hormozi_bold',
    font: opts.subFont || 'Montserrat-Black',
    font_color: opts.subColor || '#FFFFFF',
    outline_color: opts.subStroke || '#000000',
    font_size: opts.subFontSize || 0,
    border_width: opts.subOutlineW ?? 2,
    bg: !!opts.subBg,
    position: opts.subPosition || 'bottom',
    align: opts.subAlign || 'left',
    offset_y: opts.subOffsetY || 0,
  };
  const onChange = (partial) => {
    const patch = {};
    for (const [k, v] of Object.entries(partial)) patch[SUB_KEYMAP[k]] = v;
    set(patch);
  };
  return <SubtitleControls variant="create" value={value} onChange={onChange} />;
}

function HookConfig({ opts, set }) {
  const hs = opts.hookStyle || HOOK_STYLE_DEFAULT;
  const setStyle = (partial) => set({ hookStyle: { ...HOOK_STYLE_DEFAULT, ...hs, ...partial } });
  return (
    <div className="cfg-drawer fade-in">
      <HookPreview text="Highlight Opener" style={hs} />
      <div className="cf-row" style={{ marginTop: 12 }}>
        <span className="field-label" style={{ marginBottom: 9, display: 'flex' }}>Position</span>
        <Segmented
          full
          value={opts.hookPos || 'top'}
          onChange={(id) => set({ hookPos: id })}
          options={[{ id: 'top', label: 'Top' }, { id: 'center', label: 'Center' }, { id: 'bottom', label: 'Bottom' }]}
        />
      </div>
      <div className="cf-row">
        <span className="field-label" style={{ marginBottom: 9, display: 'flex' }}>Size</span>
        <Segmented
          full
          value={opts.hookSize || 'M'}
          onChange={(id) => set({ hookSize: id })}
          options={[{ id: 'S', label: 'Small' }, { id: 'M', label: 'Medium' }, { id: 'L', label: 'Large' }]}
        />
      </div>
      <HookStyleControls style={hs} set={setStyle} />
    </div>
  );
}

function LogoConfig({ opts, set }) {
  return (
    <div className="cfg-drawer fade-in">
      <LogoControls
        position={opts.logoPos || 'top-right'}
        size={opts.logoSize || 'M'}
        onChange={(p) => set(p.position !== undefined ? { logoPos: p.position } : { logoSize: p.size })}
      />
      <div className="od" style={{ marginTop: 2 }}>Upload your logo PNG in Settings → Brand logo.</div>
    </div>
  );
}

function BannerConfig({ opts, set }) {
  const value = {
    platform: opts.bannerPlatform || 'kick',
    handle: opts.bannerHandle || '',
    y_pct: opts.bannerYPct ?? 0.85,
  };
  const onChange = (partial) => {
    const patch = {};
    if (partial.platform !== undefined) patch.bannerPlatform = partial.platform;
    if (partial.handle !== undefined) patch.bannerHandle = partial.handle;
    if (partial.y_pct !== undefined) patch.bannerYPct = partial.y_pct;
    set(patch);
  };
  return (
    <div className="cfg-drawer fade-in">
      <BannerControls value={value} onChange={onChange} />
    </div>
  );
}

function OptionsPanel({ opts, set, ready, onGenerate, processing, statusText }) {
  const [subCfg, setSubCfg] = useState(false);
  const [hookCfg, setHookCfg] = useState(false);
  const [logoCfg, setLogoCfg] = useState(false);
  const [bannerCfg, setBannerCfg] = useState(false);

  return (
    <div className="recipe-panel raised">
      <div className="recipe-head">
        <div className="ricon">
          <Icon n="sliders-horizontal" />
        </div>
        <div className="recipe-title-block">
          <h3>Recipe</h3>
          <span className="sub">Highlight synthesis &amp; styling controls</span>
        </div>
      </div>

      <div className="group">
        <div className="group-label">Highlight strategy &amp; duration</div>

        {/* Content Mode */}
        <div className="row opt">
          <div className="row-left">
            <div className="ricon">
              <Icon n="mic" />
            </div>
            <div className="row-text">
              <div className="title">Content mode</div>
              <div className="desc">
                {opts.contentMode === 'podcast' && 'Multi-speaker conversational dialogue & interview debates'}
                {opts.contentMode === 'lecture' && 'Structured tutorials, presentations & educational lessons'}
                {opts.contentMode === 'meeting' && 'Business demos, product reviews & strategic discussions'}
                {opts.contentMode === 'vlog' && 'Personal experiences, storytelling & daily vlogs'}
                {(!opts.contentMode || opts.contentMode === 'social') && 'High-energy commentary & fast talking-head clips'}
              </div>
            </div>
          </div>
          <div className="row-control" style={{ minWidth: 220 }}>
            <select className="sel dropdown" value={opts.contentMode || 'podcast'} onChange={(e) => set({ contentMode: e.target.value })}>
              <option value="podcast">🎙️ Podcast / Interview</option>
              <option value="lecture">💡 Tutorial / Lecture</option>
              <option value="meeting">💼 Meeting / Demo</option>
              <option value="vlog">📹 Vlog / Storytelling</option>
              <option value="social">⚡ Social / Talking Head</option>
            </select>
          </div>
        </div>

        {/* Output Style */}
        <div className="row opt">
          <div className="row-left">
            <div className="ricon">
              <Icon n="film" />
            </div>
            <div className="row-text">
              <div className="title">Output style</div>
              <div className="desc">
                {(!opts.outputStyle || opts.outputStyle === 'recap') && 'Cohesive chronological digest from setup to conclusion'}
                {opts.outputStyle === 'trailer' && 'High-suspense teaser with hooks withholding the ending'}
                {opts.outputStyle === 'educational' && 'Milestone summary of key concepts & actionable takeaways'}
                {opts.outputStyle === 'best_moments' && 'Peak excitement, funniest banter & climax highlights'}
                {opts.outputStyle === 'decision_log' && 'Decisions, action items, verdicts & strategic outcomes'}
              </div>
            </div>
          </div>
          <div className="row-control" style={{ minWidth: 220 }}>
            <select className="sel dropdown" value={opts.outputStyle || 'recap'} onChange={(e) => set({ outputStyle: e.target.value })}>
              <option value="recap">📖 Story Recap (Full Arc)</option>
              <option value="trailer">🎬 Curiosity Trailer (Hooks)</option>
              <option value="educational">🧠 Educational Digest</option>
              <option value="best_moments">🔥 Peak Moments</option>
              <option value="decision_log">📋 Decision / Action Log</option>
            </select>
          </div>
        </div>

        {/* Focus Theme / Topic */}
        <div className="row opt">
          <div className="row-left">
            <div className="ricon">
              <Icon n="target" />
            </div>
            <div className="row-text">
              <div className="title">Focus theme / topic</div>
              <div className="desc">Optional topic filter steering AI to specific arguments or moments</div>
            </div>
          </div>
          <div className="row-control" style={{ minWidth: 220, maxWidth: 320, width: '100%' }}>
            <input
              type="text"
              aria-label="Focus theme or topic"
              style={{
                width: '100%',
                padding: '7px 12px',
                borderRadius: '8px',
                border: '1px solid var(--line-2, #ddd)',
                background: 'var(--surface, #fff)',
                color: 'var(--fg, #111)',
                fontSize: '13px',
              }}
              value={opts.theme || ''}
              placeholder="e.g. Focus on pricing model or competitor debate"
              onChange={(e) => set({ theme: e.target.value })}
            />
          </div>
        </div>

        {/* Smooth Gap Merging */}
        <OptRow
          icon="git-merge"
          label="Adjacent gap merging"
          desc="Merge cuts separated by <= 1.5s pauses to eliminate jarring micro-cuts"
          on={opts.mergeGap !== false}
          set={(v) => set({ mergeGap: v })}
        />

        {/* Content Focus */}
        <div className="row opt">
          <div className="row-left">
            <div className="ricon">
              <Icon n="sparkles" />
            </div>
            <div className="row-text">
              <div className="title">Content focus</div>
              <div className="desc">
                {opts.clipType === 'educational' && 'Key takeaways, framework breakdowns & tutorials'}
                {opts.clipType === 'humor' && 'Banter, comedic reactions & funny bloopers'}
                {opts.clipType === 'storytelling' && 'Complete narrative arcs & emotional moments'}
                {opts.clipType === 'all' && 'Balanced mix across all story beats'}
                {(!opts.clipType || opts.clipType === 'viral') && 'High energy, viral hooks & punchy takeaways'}
              </div>
            </div>
          </div>
          <div className="row-control" style={{ minWidth: 220 }}>
            <select className="sel dropdown" value={opts.clipType || 'viral'} onChange={(e) => set({ clipType: e.target.value })}>
              <option value="viral">🔥 Viral Hooks &amp; Hot Takes</option>
              <option value="educational">💡 Educational &amp; Insights</option>
              <option value="humor">😂 Humor &amp; Banter</option>
              <option value="storytelling">🎙️ Storytelling &amp; Deep Dives</option>
              <option value="all">🌐 Balanced Mix</option>
            </select>
          </div>
        </div>

        {/* Target Highlights Tier */}
        <div className="row opt">
          <div className="row-left">
            <div className="ricon">
              <Icon n="clock" />
            </div>
            <div className="row-text">
              <div className="title">Highlights generation tiers</div>
              <div className="desc">
                {opts.tierMode === 'micro' && 'Under 60s micro supercuts for Shorts, TikTok & Reels'}
                {opts.tierMode === 'story' && '1–3 min cohesive story digests for LinkedIn & Feed'}
                {opts.tierMode === 'extended' && '3–10 min extended highlight compilations for YouTube'}
                {(!opts.tierMode || opts.tierMode === 'all') && 'All tiers: Micro (<60s), Story (~2m), & Extended (3-10m)'}
              </div>
            </div>
          </div>
          <div className="row-control">
            <Segmented
              value={opts.tierMode || 'all'}
              onChange={(id) => set({ tierMode: id })}
              options={[
                { id: 'all', label: 'All Tiers' },
                { id: 'micro', label: '<60s' },
                { id: 'story', label: '~2m' },
                { id: 'extended', label: '3-10m' },
              ]}
            />
          </div>
        </div>

        {/* Aspect Ratio */}
        <div className="row opt">
          <div className="row-left">
            <div className="ricon">
              <Icon n="crop" />
            </div>
            <div className="row-text">
              <div className="title">Aspect ratio</div>
              <div className="desc">9:16 vertical (Shorts/Reels) · 1:1 square · 16:9 horizontal · 4:5 portrait</div>
            </div>
          </div>
          <div className="row-control">
            <Segmented
              value={opts.aspect || '9:16'}
              onChange={(id) => set({ aspect: id })}
              options={[
                { id: '9:16', label: '9:16' },
                { id: '1:1', label: '1:1' },
                { id: '16:9', label: '16:9' },
                { id: '4:5', label: '4:5' },
              ]}
            />
          </div>
        </div>
      </div>

      <div className="group">
        <div className="group-label">AI &amp; reframe</div>

        <OptRow
          icon="sparkles"
          label="Find viral moments"
          desc="Gemini scores the transcript · off = whole video"
          on={opts.detect !== false}
          set={(v) => set({ detect: v })}
        />

        {opts.detect !== false && (
          <div className="row opt">
            <div className="row-left">
              <div className="ricon">
                <Icon n="sparkles" />
              </div>
              <div className="row-text">
                <div className="title">Gemini model</div>
                <div className="desc">Override for this job · blank uses the Settings default</div>
              </div>
            </div>
            <div className="row-control" style={{ minWidth: 200 }}>
              <select className="sel dropdown" value={opts.model || ''} onChange={(e) => set({ model: e.target.value })}>
                {GEMINI_MODELS.map(([v, l]) => <option key={v || 'default'} value={v}>{l}</option>)}
              </select>
            </div>
          </div>
        )}

        {/* Reframe Mode */}
        <div className="row opt">
          <div className="row-left">
            <div className="ricon">
              <Icon n="scan-face" />
            </div>
            <div className="row-text">
              <div className="title">Reframe</div>
              <div className="desc">Auto face-track · Subject FrameShift crop · Off letterbox bands</div>
            </div>
          </div>
          <div className="row-control">
            <Segmented
              value={(opts.reframeMode === 'object' ? 'subject' : opts.reframeMode) || (opts.reframe === false ? 'disabled' : 'auto')}
              onChange={(id) => set({ reframeMode: id })}
              options={[
                { id: 'auto', label: 'Auto' },
                { id: 'subject', label: 'Subject' },
                { id: 'blur_pad', label: 'Blur BG' },
                { id: 'disabled', label: 'Off' },
              ]}
            />
          </div>
        </div>

        {((opts.reframeMode === 'disabled' || opts.reframeMode === 'letterbox') || (!opts.reframeMode && opts.reframe === false)) && (
          <div className="row opt">
            <div className="row-left">
              <div className="ricon">
                <Icon n="zoom-in" />
              </div>
              <div className="row-text">
                <div className="title">Letterbox zoom</div>
                <div className="desc">Crop sides for a bigger picture and smaller bars</div>
              </div>
            </div>
            <div className="row-control">
              <Segmented
                value={String(opts.letterboxZoom || 0)}
                onChange={(id) => set({ letterboxZoom: Number(id) })}
                options={[
                  { id: '0', label: 'Off' },
                  { id: '5', label: '5%' },
                  { id: '10', label: '10%' },
                  { id: '15', label: '15%' },
                ]}
              />
            </div>
          </div>
        )}

        <OptRow
          icon="scissors"
          label="Smart cut"
          desc="Remove silence & filler words between cuts"
          on={opts.smartcut !== false}
          set={(v) => set({ smartcut: v })}
        />

        <OptRow
          icon="zoom-in"
          label="Subtle zoom"
          desc="Gentle Ken Burns motion (1.0→1.05x)"
          on={!!opts.zoom}
          set={(v) => set({ zoom: v })}
        />

        <div className="row opt">
          <div className="row-left">
            <div className="ricon">
              <Icon n="languages" />
            </div>
            <div className="row-text">
              <div className="title">Spoken language</div>
              <div className="desc">Single language boosts transcription accuracy</div>
            </div>
          </div>
          <div className="row-control" style={{ minWidth: 180 }}>
            <select className="sel dropdown" value={opts.language || 'multi'} onChange={(e) => set({ language: e.target.value })}>
              {LANGUAGES.map(([v, l]) => <option key={v} value={v}>{l}</option>)}
            </select>
          </div>
        </div>
      </div>

      <div className="group">
        <div className="group-label">Captions, overlays &amp; branding</div>

        <OptRow
          icon="captions"
          label="Subtitles &amp; Karaoke"
          desc="Burn synchronized karaoke or classic captions"
          on={opts.subtitles !== false}
          set={(v) => set({ subtitles: v })}
          onConfig={() => setSubCfg(!subCfg)}
          configActive={subCfg}
        />
        {opts.subtitles !== false && subCfg && <SubConfig opts={opts} set={set} />}

        <OptRow
          icon="type"
          label="Text hooks"
          desc="Add a scroll-stopping headline banner"
          on={opts.hooks}
          set={(v) => set({ hooks: v })}
          onConfig={() => setHookCfg(!hookCfg)}
          configActive={hookCfg}
        />
        {opts.hooks && hookCfg && <HookConfig opts={opts} set={set} />}

        <OptRow
          icon="stamp"
          label="Brand logo"
          desc="Burn your brand logo onto the highlight reels"
          on={opts.logo}
          set={(v) => set({ logo: v })}
          onConfig={() => setLogoCfg(!logoCfg)}
          configActive={logoCfg}
        />
        {opts.logo && logoCfg && <LogoConfig opts={opts} set={set} />}

        <OptRow
          icon="rss"
          label="Attribution banner"
          desc="Platform logo + social handle overlay"
          on={opts.banner}
          set={(v) => set({ banner: v })}
          onConfig={() => setBannerCfg(!bannerCfg)}
          configActive={bannerCfg}
        />
        {opts.banner && bannerCfg && <BannerConfig opts={opts} set={set} />}

        <div className="row opt">
          <div className="row-left">
            <div className="ricon">
              <Icon n="palette" />
            </div>
            <div className="row-text">
              <div className="title">Colour grade</div>
              <div className="desc">Cinematic color grading filter on every highlight reel</div>
            </div>
          </div>
          <div className="row-control">
            <GradeControls
              withOff
              full={false}
              preset={opts.gradePreset || 'none'}
              onChange={(p) => set({ gradePreset: p.preset })}
            />
          </div>
        </div>
      </div>

      {/* Integrated Synthesis Action Footer */}
      <SummaryBar opts={opts} ready={ready} onGenerate={onGenerate} processing={processing} statusText={statusText} />
    </div>
  );
}

function SummaryBar({ opts, ready, onGenerate, processing, statusText }) {
  const chips = [
    opts.aspect || '9:16',
    opts.tierMode === 'micro' ? '<60s micro' : opts.tierMode === 'story' ? '~2m story' : opts.tierMode === 'extended' ? '3-10m supercut' : 'all tiers',
    opts.contentMode && `${opts.contentMode} mode`,
    opts.outputStyle && `${opts.outputStyle} style`,
    opts.theme && `theme: "${opts.theme.slice(0, 14)}${opts.theme.length > 14 ? '…' : ''}"`,
    opts.mergeGap !== false && 'gap merge',
    opts.reframeMode === 'blur_pad' ? 'blur bg' : opts.reframeMode === 'letterbox' ? 'fit' : 'crop',
    opts.smartcut && 'smart-cut',
    opts.subtitles !== false && ((opts.subMode || 'karaoke') + ' subs'),
    opts.hooks && 'hooks',
    opts.gradePreset && opts.gradePreset !== 'none' && opts.gradePreset,
  ].filter(Boolean);

  return (
    <div className="recipe-footer summary">
      <div className="rf-left">
        <div className="rf-status s-main">
          {processing
            ? (statusText || 'Synthesizing AI Highlight Reels...')
            : ready
            ? 'Ready to synthesize · Multi-tier highlight reels'
            : 'Paste a link or drop a video above to begin'}
        </div>
        <div className="rf-chips s-sub">
          {chips.map((c) => (
            <span key={c} className="chip">
              {c}
            </span>
          ))}
        </div>
      </div>
      <div className="rf-right s-right">
        <Btn
          variant="grad"
          size="lg"
          icon="wand-sparkles"
          onClick={onGenerate}
          disabled={!ready || processing}
          loading={processing}
        >
          {processing ? 'Synthesizing...' : 'Generate AI Highlights'}
        </Btn>
      </div>
    </div>
  );
}

// --------------------------------------------------------------------------
// MAIN COMPONENT: HighlightsStudioView
// --------------------------------------------------------------------------
export function HighlightsStudioView({ apiKey = '', onToast }) {
  const restored = useMemo(() => loadPersistedHighlightsSession(), []);

  const [opts, setOpts] = useState(() => ({
    source: 'url',
    url: '',
    file: null,
    fileName: '',
    mode: 'single',
    preset: 'all_tiers',
    tierMode: 'all',
    aspect: '9:16',
    contentMode: 'podcast',
    outputStyle: 'recap',
    theme: '',
    mergeGap: true,
    model: '',
    reframeMode: 'auto',
    letterboxZoom: 0,
    zoom: false,
    smartcut: true,
    language: 'multi',
    subtitles: true,
    subMode: 'karaoke',
    subPreset: 'hormozi_bold',
    hooks: false,
    logo: false,
    banner: false,
    gradePreset: 'none',
    ...(restored?.opts || {}),
  }));

  // Update opts with preset-staleness check: if recipe fields change directly, clear preset
  const updateOpts = (patch) => {
    setOpts((prev) => {
      const next = { ...prev, ...patch };
      if (!patch.preset && (patch.tierMode !== undefined || patch.aspect !== undefined || patch.reframeMode !== undefined)) {
        next.preset = null;
      }
      return next;
    });
  };

  const [activeJobId, setActiveJobId] = useState(restored?.jobId || '');
  const [status, setStatus] = useState(restored?.status || 'idle'); // 'idle' | 'processing' | 'synthesizing' | 'complete' | 'error'
  const [statusText, setStatusText] = useState(restored?.statusText || '');
  const [logs, setLogs] = useState(restored?.logs || []);
  const [highlights, setHighlights] = useState(restored?.highlights || []);
  const [editingIndex, setEditingIndex] = useState(null);
  const pollIntervalRef = useRef(null);

  const stopPolling = () => {
    if (pollIntervalRef.current) {
      clearInterval(pollIntervalRef.current);
      pollIntervalRef.current = null;
    }
  };

  // Sync to durable session storage whenever active state changes
  useEffect(() => {
    if (status === 'idle' || !activeJobId) {
      clearPersistedHighlightsSession();
    } else {
      savePersistedHighlightsSession({
        jobId: activeJobId,
        status,
        statusText,
        logs: logs.slice(-300),
        highlights,
        opts,
      });
    }
  }, [activeJobId, status, statusText, logs, highlights, opts]);

  const getEffectiveKey = () => (apiKey || (typeof localStorage !== 'undefined' ? localStorage.getItem('gemini_key') : '') || '').trim();

  // Core polling worker
  const startPolling = (jobId, currentOpts, effectiveKey) => {
    stopPolling();
    let pollCount = 0;
    let consecutiveErrors = 0;

    pollIntervalRef.current = setInterval(async () => {
      pollCount += 1;

      // Hard timeout (600 ticks * 1.5s = 900s = 15 minutes)
      if (pollCount > 600) {
        stopPolling();
        setStatus('error');
        onToast?.({ type: 'error', message: 'Highlights processing timed out after 15 minutes' });
        return;
      }

      try {
        const statusRes = await pollJob(jobId);
        consecutiveErrors = 0;

        if (statusRes.logs && statusRes.logs.length > 0) {
          setLogs(statusRes.logs);
          setStatusText(statusRes.logs[statusRes.logs.length - 1]);
        } else if (statusRes.result?.operations?.detail) {
          setStatusText(statusRes.result.operations.detail);
        }

        if (statusRes.status === 'complete' || statusRes.status === 'completed') {
          stopPolling();
          setStatus('synthesizing');
          setStatusText('⚡ Loading synthesized highlight reels...');

          try {
            let fetchedReels = statusRes.result?.highlights || [];
            if (!fetchedReels || fetchedReels.length === 0) {
              try {
                const hlRes = await getHighlights(jobId, effectiveKey);
                fetchedReels = hlRes.highlights || [];
              } catch {
                fetchedReels = [];
              }
            }
            if (!fetchedReels || fetchedReels.length === 0) {
              const genRes = await generateAllHighlights(jobId, {
                aspect: currentOpts.aspect,
                reframe_mode: currentOpts.reframeMode,
                content_mode: currentOpts.contentMode || 'podcast',
                output_style: currentOpts.outputStyle || 'recap',
                theme: currentOpts.theme || undefined,
                merge_gap_seconds: currentOpts.mergeGap !== false ? 1.5 : 0.0,
                subtitles: {
                  enabled: currentOpts.subtitles !== false,
                  preset: currentOpts.subPreset,
                  position: currentOpts.subPosition || 'bottom',
                },
                grade_preset: currentOpts.gradePreset,
              }, effectiveKey);
              fetchedReels = genRes.highlights || [];
            }

            setHighlights(fetchedReels);
            setStatus('complete');
            onToast?.({ type: 'success', message: `Synthesized ${fetchedReels.length} highlight reels!` });
          } catch (ge) {
            setStatus('error');
            onToast?.({ type: 'error', message: ge.message || 'Highlights synthesis failed' });
          }
        } else if (statusRes.status === 'failed') {
          stopPolling();
          setStatus('error');
          onToast?.({ type: 'error', message: statusRes.error || 'Transcription failed' });
        }
      } catch (pollErr) {
        consecutiveErrors += 1;
        if (consecutiveErrors >= 6) {
          stopPolling();
          setStatus('error');
          onToast?.({ type: 'error', message: pollErr.message || 'Lost connection to processing worker' });
        }
      }
    }, 1500);
  };

  // Resume polling on mount if an active job was in-flight when tab was switched
  useEffect(() => {
    if (activeJobId && (status === 'processing' || status === 'synthesizing')) {
      startPolling(activeJobId, opts, getEffectiveKey());
    }
    return () => stopPolling();
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // Pick a preset
  const handlePickPreset = (presetId) => {
    if (presetId === 'micro_only') updateOpts({ preset: presetId, tierMode: 'micro', aspect: '9:16', reframeMode: 'auto' });
    else if (presetId === 'story_digest') updateOpts({ preset: presetId, tierMode: 'story', aspect: '9:16', reframeMode: 'auto' });
    else if (presetId === 'extended_supercut') updateOpts({ preset: presetId, tierMode: 'extended', aspect: '16:9', reframeMode: 'letterbox' });
    else updateOpts({ preset: presetId, tierMode: 'all', aspect: '9:16', reframeMode: 'auto' });
  };

  const isReady = opts.source === 'url' ? !!opts.url.trim() : !!opts.file;

  // Main Submit Pipeline
  const handleGenerate = async () => {
    if (!isReady) return;
    stopPolling();
    setStatus('processing');
    setLogs([]);
    setStatusText(opts.source === 'url' ? 'Downloading & transcribing video...' : 'Uploading & processing video...');

    try {
      const effectiveKey = getEffectiveKey();
      const effectiveInstructions = [
        opts.instructions?.trim(),
        opts.theme?.trim() ? `Focus Theme: ${opts.theme.trim()}` : null,
      ].filter(Boolean).join(' · ') || undefined;

      const dataPayload = opts.source === 'url'
        ? {
            type: 'url',
            payload: opts.url.trim(),
            instructions: effectiveInstructions,
            highlights: true,
            aspect: opts.aspect,
            reframe_mode: opts.reframeMode,
            preselections: {
              language: opts.language,
              model: opts.model || undefined,
            },
          }
        : {
            type: 'file',
            payload: opts.file,
            instructions: effectiveInstructions,
            highlights: true,
            aspect: opts.aspect,
            reframe_mode: opts.reframeMode,
            preselections: {
              language: opts.language,
              model: opts.model || undefined,
            },
          };

      const submitRes = await submitProcessJob(dataPayload, effectiveKey);
      const newJobId = submitRes.job_id || submitRes.id;

      if (!newJobId) {
        throw new Error('Failed to create processing job');
      }
      setActiveJobId(newJobId);
      startPolling(newJobId, opts, effectiveKey);
    } catch (err) {
      stopPolling();
      setStatus('error');
      onToast?.({ type: 'error', message: err.message || 'Failed to submit video' });
    }
  };

  // Reset / Cancel action
  const handleReset = () => {
    stopPolling();
    clearPersistedHighlightsSession();
    setActiveJobId('');
    setStatus('idle');
    setStatusText('');
    setLogs([]);
    setHighlights([]);
  };

  // Edit Reel Apply
  // Edit Reel Apply
  const handleApplyEdit = async (staged) => {
    if (editingIndex === null || !highlights[editingIndex]) return;
    const targetReel = highlights[editingIndex];
    setEditingIndex(null);
    onToast?.({ type: 'info', message: 'Reprocessing highlight reel in background...' });

    try {
      const payload = {
        title: (staged.toggles?.hook && staged.hookParams?.text) ? staged.hookParams.text : targetReel.title,
        aspect: staged.aspect || targetReel.aspect || '9:16',
        reframe_mode: staged.reframeMode || targetReel.reframe_mode || 'auto',
        subtitles: staged.toggles?.subtitles ? staged.subtitleParams : { enabled: false },
        hook: staged.toggles?.hook ? staged.hookParams : { enabled: false },
        grade_preset: staged.toggles?.grade ? staged.gradeParams?.preset : 'none',
      };

      const res = await applyEditHighlight(activeJobId, targetReel.id || targetReel.filename, payload, getEffectiveKey());
      if (res.highlight) {
        setHighlights((prev) => prev.map((h, i) => (i === editingIndex ? res.highlight : h)));
        onToast?.({ type: 'success', message: 'Highlight reel reprocessed!' });
      }
    } catch (err) {
      onToast?.({ type: 'error', message: err.message || 'Failed to reprocess highlight reel' });
    }
  };

  // Delete Reel
  const handleDeleteReel = async (filename) => {
    try {
      await deleteHighlight(activeJobId, filename);
      setHighlights((prev) => prev.filter((h) => h.filename !== filename));
      onToast?.({ type: 'info', message: 'Highlight reel deleted' });
    } catch {
      onToast?.({ type: 'error', message: 'Failed to delete reel' });
    }
  };

  // Retry action: resume failed/interrupted highlight job from its checkpoints
  const handleRetry = async () => {
    if (!activeJobId) {
      handleGenerate();
      return;
    }
    const effectiveKey = getEffectiveKey();
    stopPolling();
    setStatus('processing');
    setStatusText('♻️ Resuming highlights pipeline from checkpoint...');
    setLogs((prev) => [...prev, '♻️ Resuming highlights pipeline from checkpoint...']);
    onToast?.({ type: 'info', message: 'Resuming highlights from checkpoint...' });

    try {
      await retryJobApi(activeJobId, effectiveKey);
      startPolling(activeJobId, opts, effectiveKey);
    } catch {
      // If retry endpoint returns (e.g. state already on disk or busy), resume polling directly
      startPolling(activeJobId, opts, effectiveKey);
    }
  };

  // Render processing view if job is active
  if (status === 'processing' || status === 'synthesizing' || status === 'error') {
    return (
      <HighlightsProcessingView
        status={status}
        statusText={statusText}
        logs={logs}
        opts={opts}
        onCancel={handleReset}
        onRetry={handleRetry}
      />
    );
  }

  return (
    <div className="container fade-in">
      {/* Hero Header */}
      <Hero
        eyebrow="Drop a link · get viral highlight reels"
        line1="Long videos in."
        grad="Supercut highlights out."
        sub="Drop a link from YouTube, Twitch, or Kick (or upload a file) and Nugget synthesizes multi-tier highlight packages (<60s, ~120s, 180s–720s) with complete video editing controls."
      />

      <div className="create-workbench">
        {/* Source Panel */}
        <SourcePanel opts={opts} set={updateOpts} disabled={status === 'processing'} />

        {/* Preset Cards */}
        <div>
          <div className="label" style={{ marginBottom: 10, fontWeight: 700 }}>
            Start from a highlight preset, or fine-tune the recipe below
          </div>
          <PresetCards
            presets={HIGHLIGHT_PRESETS}
            active={opts.preset}
            onPick={handlePickPreset}
          />
        </div>

        {/* Options Panel (Recipe) with Integrated Synthesis Action Deck */}
        <OptionsPanel
          opts={opts}
          set={updateOpts}
          ready={isReady}
          onGenerate={handleGenerate}
          processing={status === 'processing' || status === 'synthesizing'}
          statusText={statusText}
        />
      </div>

      {/* Results View: Clipping-Style Cards Grid Aligned with Workbench */}
      {highlights.length > 0 && (
        <div className="create-workbench" style={{ marginTop: 36, paddingTop: 28, borderTop: '1px solid var(--line-1)' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 20 }}>
            <div>
              <h2 style={{ fontSize: '20px', fontWeight: 800, margin: 0, display: 'flex', alignItems: 'center', gap: 8 }}>
                <Icon n="film" style={{ color: 'var(--brand-teal, #02C5BF)' }} />
                Generated Highlight Packages ({highlights.length})
              </h2>
              <span style={{ fontSize: '13px', color: 'var(--fg-3)' }}>
                Multi-tier highlight reels ready for preview, editing, and export.
              </span>
            </div>

            <Btn variant="secondary" size="sm" icon="refresh-cw" onClick={handleReset}>
              Start new highlight
            </Btn>
          </div>

          <div className="results-grid" role="list" aria-label="Generated highlight packages">
            {highlights.map((reel, idx) => {
              const hasScore = typeof reel.viral_score === 'number' && !isNaN(reel.viral_score);
              const score = hasScore ? Math.round(reel.viral_score) : null;
              const tierBadge = reel.tier === 'micro' ? '<60s' : reel.tier === 'extended' ? 'Supercut' : 'Digest';
              const tierLabel = reel.tier === 'micro' ? 'Micro-Short (<60s)' : reel.tier === 'extended' ? 'Extended Supercut' : 'Story Digest (~2m)';
              const durText = fmtDuration(0, reel.actual_duration || reel.target_duration);

              const isLandscape = reel.aspect === '16:9' || reel.tier === 'extended' || opts.aspect === '16:9';

              return (
                <article key={reel.id || idx} className={`clip${score !== null && score >= 90 ? ' top' : ''}`}>
                  <div className={`clip-media${isLandscape ? ' landscape' : ''}`} style={{ padding: 0, background: '#000' }}>
                    <LazyVideo
                      src={reel.video_url}
                      controls
                      playsInline
                      aria-label={`Preview ${reel.title}`}
                      style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', objectFit: 'cover', zIndex: 0 }}
                    />
                    <div className="clip-top" style={{ padding: 10 }}>
                      {hasScore && (
                        <span className="score"><Icon n="flame" />{score}</span>
                      )}
                      <span className="rf-badge" title={`Tier: ${tierLabel}`}>
                        <Icon n="film" />{tierBadge}
                      </span>
                    </div>
                    <div className="clip-bottom" style={{ padding: 10 }}>
                      <span className="dur">{durText}</span>
                    </div>
                  </div>

                  <button
                    type="button"
                    className="clip-edit"
                    onClick={(e) => { e.stopPropagation(); setEditingIndex(idx); }}
                    aria-label={`Edit and reprocess ${reel.title}`}
                  >
                    <Icon n="sliders-horizontal" />
                    Edit & reprocess
                  </button>

                  <div className="clip-foot">
                    <span className="ttl" title={reel.title}>{reel.title}</span>
                    <a
                      href={reel.video_url}
                      download={reel.filename}
                      className="mini"
                      title="Download reel"
                      aria-label={`Download ${reel.title}`}
                      style={{ textDecoration: 'none' }}
                    >
                      <Icon n="download" />
                    </a>
                    <button
                      type="button"
                      className="mini"
                      title="Delete reel"
                      aria-label={`Delete ${reel.title}`}
                      onClick={(e) => {
                        e.stopPropagation();
                        if (window.confirm('Delete this highlight reel?')) {
                          handleDeleteReel(reel.filename);
                        }
                      }}
                    >
                      <Icon n="trash-2" />
                    </button>
                  </div>
                </article>
              );
            })}
          </div>
        </div>
      )}

      {/* Edit Modal from Clipping side */}
      {editingIndex !== null && highlights[editingIndex] && (
        <EditClipModal
          clip={{
            ...highlights[editingIndex],
            video_url: highlights[editingIndex].video_url || highlights[editingIndex].url,
            video_title_for_youtube_short: highlights[editingIndex].title,
            viral_score: highlights[editingIndex].viral_score,
            start: 0,
            end: highlights[editingIndex].actual_duration || highlights[editingIndex].target_duration || 60,
          }}
          idx={editingIndex}
          jobId={activeJobId}
          appliedMode={highlights[editingIndex].reframe_mode || 'auto'}
          initial={{
            reframeMode: highlights[editingIndex].reframe_mode || 'auto',
            subtitleParams: highlights[editingIndex].subtitles,
            hookParams: highlights[editingIndex].hook,
            gradeParams: { preset: highlights[editingIndex].grade_preset || 'none' },
            toggles: {
              subtitles: highlights[editingIndex].subtitles?.enabled !== false && !!highlights[editingIndex].subtitles,
              hook: highlights[editingIndex].hook?.enabled !== false && !!highlights[editingIndex].hook,
              grade: !!highlights[editingIndex].grade_preset && highlights[editingIndex].grade_preset !== 'none',
              smartcut: true,
            },
          }}
          preselections={{
            subtitles: { preset: opts.subPreset, mode: opts.subMode },
            hook: opts.hookStyle,
            grade: { preset: opts.gradePreset },
          }}
          onClose={() => setEditingIndex(null)}
          onApply={handleApplyEdit}
          pushToast={(tone, msg) => onToast?.({ type: tone, message: msg })}
        />
      )}
    </div>
  );
}
