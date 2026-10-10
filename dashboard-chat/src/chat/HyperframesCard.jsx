import { useEffect, useRef, useState } from 'react';
import { getClipProject, saveClipProject, composeProject } from '../api/realApi.js';
import {
  getHyperframesCapability,
  compileHyperframes,
  renderHyperframes,
  hyperframesRenderStatus,
  cancelHyperframesRender,
  hyperframesCompositionUrl,
} from '../api/hyperframesApi.js';

// Task 18: the /hyperframes slash-command capability card. Contract 1:
// single-clip identity + version check, template proposal, Apply via the
// existing controller/version guard, and distinct pending vs applied vs
// rendering vs rendered states. Ordinary edit path stays usable elsewhere.
//
// Contract 8: delayed async responses can NEVER replace another selection's
// preview — every async completion re-checks the card's own identity
// (scopeKey) against the app's current activeSelectionKey and refuses to
// commit stale data.
//
// Contract 5: preview and export use the SAME compiled composition and clock.
// The preview is a sandboxed iframe running the exact index.html the export
// renders, driven through the same window.__hf.seek(t) protocol the worker's
// engine uses for frame capture. It is marked as a lightweight preview; the
// decoded export is the reference. No vendored player: the player
// web-component cannot drive the engine's __hf seek protocol, and its CDN
// runtime fallback would be a remote fetch from our composition.
const PARAM_OPTIONS = {
  caption_size: ['small', 'medium', 'large'],
  caption_position: ['lower-third', 'middle'],
};

function identityKey(jobId, index) {
  return jobId != null && index != null ? `${jobId}:${index}` : null;
}

// Sandboxed preview of one compiled composition. Drives the SAME
// window.__hf.seek(t) protocol the export worker's engine uses, so the
// caption/hook boundary appearance agrees with the decoded export by
// construction. The base <video> is scrubbed alongside for a faithful
// lightweight preview. Marked as preview, never as the export reference.
function HfPreview({ src, duration }) {
  const frameRef = useRef(null);
  const [t, setT] = useState(0);
  const [ready, setReady] = useState(false);
  const [playing, setPlaying] = useState(false);
  const playTimer = useRef(null);

  const seekTo = (time) => {
    const w = frameRef.current && frameRef.current.contentWindow;
    if (!w || !w.__hf || typeof w.__hf.seek !== 'function') return false;
    const clamped = Math.max(0, Math.min(duration, time));
    w.__hf.seek(clamped);
    try {
      const v = w.document.getElementById('hf-base');
      if (v && Number.isFinite(v.duration) && v.duration > 0) {
        v.currentTime = Math.min(clamped, v.duration - 0.04);
      } else if (v) {
        v.currentTime = clamped;
      }
    } catch { /* video seek is best-effort; captions are the contract */ }
    setT(clamped);
    return true;
  };

  useEffect(() => {
    setReady(false); setT(0); setPlaying(false);
    let cancelled = false;
    let tries = 0;
    const poll = () => {
      if (cancelled) return;
      const w = frameRef.current && frameRef.current.contentWindow;
      if (w && w.__hf && typeof w.__hf.seek === 'function') {
        if (!cancelled) { setReady(true); seekTo(0); }
        return;
      }
      if (++tries < 100) setTimeout(poll, 100); // 10s max, then stays "loading"
    };
    poll();
    return () => { cancelled = true; clearInterval(playTimer.current); };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [src]);

  useEffect(() => () => clearInterval(playTimer.current), []);

  const togglePlay = () => {
    if (playing) { clearInterval(playTimer.current); setPlaying(false); return; }
    if (!ready) return;
    if (t >= duration - 0.05) seekTo(0);
    setPlaying(true);
    const startAt = Date.now();
    const startT = t >= duration - 0.05 ? 0 : t;
    playTimer.current = setInterval(() => {
      const nt = startT + (Date.now() - startAt) / 1000;
      if (nt >= duration) { clearInterval(playTimer.current); setPlaying(false); seekTo(duration); return; }
      seekTo(nt);
    }, 66);
  };

  return (
    <div data-testid="hf-preview-frame">
      <div style={{ fontSize: 12, opacity: 0.75, marginBottom: 6 }}>
        Lightweight preview — the decoded export is the reference. Both use the same compiled composition.
      </div>
      <iframe
        ref={frameRef}
        data-testid="hf-preview-iframe"
        title="Hyperframes composition preview"
        src={src}
        sandbox="allow-scripts allow-same-origin"
        style={{ width: '100%', aspectRatio: '9/16', maxHeight: 420, background: '#000', border: '1px solid rgba(255,255,255,0.15)', borderRadius: 6 }}
      />
      {!ready && <div style={{ fontSize: 12, opacity: 0.7, marginTop: 4 }}>Loading preview…</div>}
      {ready && (
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginTop: 6 }}>
          <button data-testid="hf-preview-play" onClick={togglePlay}
            style={{ padding: '4px 12px', borderRadius: 6, border: '1px solid rgba(255,255,255,0.25)', background: 'rgba(255,255,255,0.08)', color: 'inherit', cursor: 'pointer' }}>
            {playing ? 'Pause' : 'Play'}
          </button>
          <input data-testid="hf-preview-scrub" type="range" min={0} max={duration} step={0.033}
            value={t} onChange={(e) => { clearInterval(playTimer.current); setPlaying(false); seekTo(parseFloat(e.target.value)); }}
            style={{ flex: 1 }} aria-label="Preview scrub" />
          <span style={{ fontSize: 11, opacity: 0.7, minWidth: 88 }}>{t.toFixed(2)}s / {duration.toFixed(2)}s</span>
        </div>
      )}
    </div>
  );
}

export default function HyperframesCard({ data = {}, activeSelectionKey = null }) {
  const { jobId, index, scope } = data;
  const scopeKey = identityKey(jobId, index);
  const isLive = activeSelectionKey == null || scopeKey === activeSelectionKey;

  const [state, setState] = useState('checking');
  const [capability, setCapability] = useState(null);
  const [error, setError] = useState(null);
  const [templateId, setTemplateId] = useState('');
  const [params, setParams] = useState({ caption_size: 'medium', caption_position: 'lower-third' });
  const [preview, setPreview] = useState(null); // { compile_id, preview_identity, composition_url, manifest }
  const [render, setRender] = useState(null); // { render_id, state, output_url, metrics, error }
  const [legacy, setLegacy] = useState(null); // { composed_url, version }
  const pollRef = useRef(null);
  const aliveRef = useRef(true);

  useEffect(() => () => { aliveRef.current = false; clearInterval(pollRef.current); }, []);

  const commit = (fn) => {
    if (aliveRef.current && isLive) fn();
    else if (aliveRef.current) setError('This panel was opened for a different clip — select that clip again to continue here.');
  };

  // --- capability fetch on mount / identity change ---
  useEffect(() => {
    if (scopeKey == null) { setState('no-selection'); return; }
    let cancelled = false;
    (async () => {
      try {
        const cap = await getHyperframesCapability();
        if (cancelled) return;
        if (activeSelectionKey != null && scopeKey !== activeSelectionKey) {
          setState('stale');
          return;
        }
        setCapability(cap);
        if (!cap.available) { setState('unavailable'); setError(cap.reason || 'Unavailable'); return; }
        const first = (cap.templates || [])[0];
        setTemplateId(first ? first.id : '');
        setState('proposal');
      } catch (e) {
        if (!cancelled) { setError(e.message); setState('unavailable'); }
      }
    })();
    return () => { cancelled = true; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [scopeKey]);

  // --- apply: typed param edit through the Task16/17 controller authority ---
  const apply = async () => {
    setState('applying'); setError(null);
    try {
      const current = await getClipProject(jobId, index);
      const project = current.project || current;
      const version = current.version ?? project.version ?? 1;
      project.finishing = {
        renderer: 'hyperframes',
        template: templateId,
        caption_size: params.caption_size,
        caption_position: params.caption_position,
      };
      const saved = await saveClipProject(jobId, index, project, { expectedVersion: version });
      commit(() => { setState('applied'); setAppliedVersion(saved.version ?? version + 1); });
    } catch (e) {
      commit(() => { setState(e.status === 409 ? 'proposal' : 'error'); setError(`Apply failed: ${e.message}`); });
    }
  };
  const [appliedVersion, setAppliedVersion] = useState(null);

  // --- compile -> preview (same composition identity as export) ---
  const startPreview = async () => {
    setState('compiling'); setError(null);
    try {
      const res = await compileHyperframes(jobId, index, {
        template_id: templateId,
        params,
        expected_version: appliedVersion,
      });
      commit(() => {
        setPreview({
          compile_id: res.compile_id,
          preview_identity: res.preview_identity, // manifest digest string
          composition_url: hyperframesCompositionUrl(jobId, res.compile_id),
          manifest: res.manifest || null,
        });
        setState('preview');
      });
    } catch (e) {
      commit(() => { setState('applied'); setError(`Preview failed: ${e.message}`); });
    }
  };

  // --- render -> decoded export (uses the SAVED finishing; no overrides) ---
  // Backend render states: rendering | rendered | failed | cancelled.
  const startRender = async () => {
    setState('rendering'); setError(null); setLegacy(null);
    try {
      const res = await renderHyperframes(jobId, index, { expected_version: appliedVersion });
      const initialState = res.state || 'rendering';
      commit(() => setRender({ render_id: res.render_id, state: initialState }));
      if (initialState !== 'rendering') {
        // Cache hit (or instant terminal): resolve without polling.
        commit(() => {
          setRender({ render_id: res.render_id, state: initialState, output_url: res.output_url });
          setState(initialState === 'rendered' ? 'rendered' : 'failed');
          if (initialState !== 'rendered') setError(res.error || 'Render failed');
        });
        return;
      }
      clearInterval(pollRef.current);
      pollRef.current = setInterval(async () => {
        try {
          const st = await hyperframesRenderStatus(res.render_id);
          const rs = st.state;
          if (rs !== 'rendering') {
            clearInterval(pollRef.current);
            if (aliveRef.current && isLive) {
              setRender({ render_id: res.render_id, state: rs, output_url: st.output_url, metrics: st.metrics, error: st.error });
              setState(rs === 'rendered' ? 'rendered' : rs === 'cancelled' ? 'cancelled' : 'failed');
              if (rs !== 'rendered' && rs !== 'cancelled') setError(st.error || 'Render failed');
            }
          } else if (aliveRef.current && isLive) {
            setRender({ render_id: res.render_id, state: rs });
          }
        } catch (e) {
          clearInterval(pollRef.current);
          if (aliveRef.current && isLive) { setRender({ render_id: res.render_id, state: 'failed', error: e.message }); setState('failed'); setError(e.message); }
        }
      }, 1000);
    } catch (e) {
      commit(() => { setState('preview'); setError(`Render failed: ${e.message}`); });
    }
  };

  const cancelRender = async () => {
    if (render?.render_id) {
      try { await cancelHyperframesRender(render.render_id); } catch { /* worker already done */ }
    }
    clearInterval(pollRef.current);
    setRender((r) => ({ ...r, state: 'cancelled' }));
    setState('cancelled');
  };

  // --- explicit recoverable legacy choice (contract 2) ---
  const legacyRender = async () => {
    setError(null);
    try {
      const current = await getClipProject(jobId, index);
      const project = current.project || current;
      const res = await composeProject(jobId, index, project);
      commit(() => { setLegacy({ composed_url: res.composed_url, version: res.version }); setState('legacy'); });
    } catch (e) {
      commit(() => setError(`Legacy render failed: ${e.message}`));
    }
  };

  const boxStyle = { background: 'rgba(255,255,255,0.04)', border: '1px solid rgba(255,255,255,0.12)', borderRadius: 8, padding: 10, marginTop: 6, fontSize: 13 };
  const btnStyle = { marginRight: 8, padding: '5px 12px', borderRadius: 6, border: '1px solid rgba(255,255,255,0.25)', background: 'rgba(255,255,255,0.08)', color: 'inherit', cursor: 'pointer' };

  return (
    <div data-testid="hyperframes-card">
      <div style={{ fontSize: 12, opacity: 0.75 }}>hyperframes — finishing pilot</div>

      {state === 'checking' && <div style={boxStyle}>Checking capability…</div>}

      {state === 'no-selection' && (
        <div style={boxStyle}>
          Select a clip first — Hyperframes finishing works on one selected clip at a time (scope: {String(scope || 'none')}).
        </div>
      )}

      {state === 'stale' && (
        <div style={boxStyle}>
          This panel belongs to a different clip — delayed responses can’t replace another selection’s preview. Select that clip again to continue here.
        </div>
      )}

      {state === 'unavailable' && (
        <div style={boxStyle}>
          Hyperframes finishing isn’t available here{error ? `: ${error}` : ''}. The accepted legacy renderer is unchanged — your project keeps its timings, audio, and framing.
        </div>
      )}

      {['proposal', 'applying', 'applied', 'compiling', 'preview', 'rendering', 'cancelled'].includes(state) && capability?.available && (
        <div style={boxStyle}>
          <div style={{ marginBottom: 8, opacity: 0.8 }}>
            Renderer: <strong>Hyperframes</strong> (optional pilot — legacy renderer stays the default).
            {' '}{state === 'applied' ? `Applied to project version ${appliedVersion ?? '?'}.` : ''}
          </div>
          <label style={{ display: 'block', marginBottom: 6 }}>
            Template:{' '}
            <select data-testid="hf-template" value={templateId} onChange={(e) => setTemplateId(e.target.value)}
              disabled={['applying', 'compiling', 'rendering'].includes(state)} style={{ color: '#000' }}>
              {(capability.templates || []).map((t) => (
                <option key={t.id} value={t.id}>{t.title || t.id}</option>
              ))}
            </select>
          </label>
          <div style={{ display: 'flex', gap: 12, marginBottom: 8 }}>
            {Object.keys(PARAM_OPTIONS).map((p) => (
              <label key={p} style={{ display: 'block' }}>
                {p.replace('_', ' ')}:{' '}
                <select data-testid={`hf-${p}`} value={params[p]} onChange={(e) => setParams({ ...params, [p]: e.target.value })}
                  disabled={['applying', 'compiling', 'rendering'].includes(state)} style={{ color: '#000' }}>
                  {PARAM_OPTIONS[p].map((v) => <option key={v} value={v}>{v}</option>)}
                </select>
              </label>
            ))}
          </div>
          <div>
            <button data-testid="hf-apply" style={btnStyle} onClick={apply} disabled={state === 'applying'}>
              {state === 'applying' ? 'Applying…' : 'Apply template'}
            </button>
            {['applied', 'preview', 'cancelled'].includes(state) && (
              <button data-testid="hf-preview" style={btnStyle} onClick={startPreview} disabled={state === 'compiling'}>
                {state === 'compiling' ? 'Compiling…' : 'Preview captions'}
              </button>
            )}
          </div>
          {error && <div style={{ color: '#f88', marginTop: 8 }} data-testid="hf-error">{error}</div>}
        </div>
      )}

      {state === 'preview' && preview && (
        <div style={boxStyle}>
          <HfPreview src={preview.composition_url} duration={preview.manifest?.duration || 10} />
          <div style={{ fontSize: 11, opacity: 0.65, marginTop: 6, wordBreak: 'break-all' }} data-testid="hf-preview-identity">
            preview identity: v{preview.manifest?.project_version ?? '?'} · digest {String(preview.preview_identity || '').slice(0, 12)}… · {preview.manifest?.template?.id ?? ''} · engine {preview.manifest?.renderer?.engine ?? ''}
          </div>
          <div style={{ marginTop: 8 }}>
            <button data-testid="hf-render" style={btnStyle} onClick={startRender}>Render export</button>
          </div>
        </div>
      )}

      {state === 'rendering' && (
        <div style={boxStyle}>
          <div>Rendering… ({render?.state || 'rendering'})</div>
          <button data-testid="hf-cancel" style={{ ...btnStyle, marginTop: 6 }} onClick={cancelRender}>Cancel</button>
        </div>
      )}

      {state === 'rendered' && render?.output_url && (
        <div style={boxStyle}>
          <div style={{ color: '#8f8' }}>Export rendered.</div>
          <a data-testid="hf-output" href={render.output_url} target="_blank" rel="noreferrer">Download export</a>
          {render.metrics && (
            <div style={{ fontSize: 11, opacity: 0.65, marginTop: 4 }}>
              runtime {render.metrics.render_seconds ?? '?'}s · frames {render.metrics.frames ?? '?'} · peak RSS {render.metrics.peak_memory_mb ?? '?'} MB · cache {render.metrics.cache ?? 'miss'}
            </div>
          )}
        </div>
      )}

      {state === 'failed' && (
        <div style={boxStyle}>
          <div style={{ color: '#f88' }}>Hyperframes render failed{error ? `: ${error}` : ''}.</div>
          <div style={{ marginTop: 8 }}>
            <button data-testid="hf-legacy" style={btnStyle} onClick={legacyRender}>
              Render with the accepted legacy renderer instead
            </button>
          </div>
        </div>
      )}

      {state === 'cancelled' && (
        <div style={boxStyle}>Render cancelled — nothing was published. You can preview or render again.</div>
      )}

      {state === 'legacy' && legacy && (
        <div style={boxStyle}>
          <div style={{ color: '#fc6' }}>Rendered with the accepted legacy renderer (project version {legacy.version}).</div>
          <a data-testid="hf-legacy-output" href={legacy.composed_url} target="_blank" rel="noreferrer">Download</a>
        </div>
      )}

      {state === 'error' && (
        <div style={boxStyle}><span style={{ color: '#f88' }}>{error || 'Something went wrong.'}</span></div>
      )}
    </div>
  );
}
