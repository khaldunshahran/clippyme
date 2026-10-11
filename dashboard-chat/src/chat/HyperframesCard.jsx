import { useEffect, useRef, useState } from 'react';
import { getClipProject, saveClipProject, composeProject } from '../api/realApi.js';
import {
  getHyperframesCapability,
  compileHyperframes,
  renderHyperframes,
  hyperframesRenderStatus,
  cancelHyperframesRender,
  getHyperframesPreviewToken,
  hyperframesCompositionUrl,
} from '../api/hyperframesApi.js';

// Task 18: the /hyperframes slash-command capability card. Contract 1:
// single-clip identity + version-checked Apply, template proposal, and
// distinct pending vs applied vs rendering vs rendered states. Ordinary edit
// path stays usable elsewhere.
//
// Contract 8: delayed async responses can NEVER replace another selection's
// preview — every async completion re-checks the card's own identity
// (scopeKey) against the app's current activeSelectionKey and the host's
// lifetimeKey, and refuses to commit stale data.
//
// Contract 5: preview and export use the SAME compiled composition. The
// preview is a sandboxed iframe running the exact index.html the export
// renders. No vendored player: the player web-component cannot drive the
// engine's seek protocol, and its CDN runtime fallback would be a remote
// fetch from our composition.
//
// Review 01 corrections:
//  - Single project authority: when the host passes the selected workspace
//    `controller`, Apply routes through controller.applyEdit(label, mutate)
//    and the legacy fallback drains via controller.exportProject(). The
//    controller owns expectedVersion/ackVersion and surfaces
//    SAVE_INTERRUPTED honestly. Without a controller the card keeps the
//    explicit standalone direct-API path (standalone compatibility).
//  - Lifetime gates: `lifetimeKey` (host-owned; changes on session reset,
//    selection close/switch/reselection, controller disposal) is captured
//    at every async start and re-checked after every await against a ref of
//    the current props. Mismatch drops the continuation: no state commit,
//    no save. isLive is false when activeSelectionKey is null — a cleared
//    selection cannot authorize an old card to save.
//  - Preview auth: the iframe cannot attach apiFetch's Bearer <redacted>
//    (plain navigation), so the card mints a short-lived preview token via
//    authed POST and puts it in the iframe src as ?token=. No Supabase JWT
//    ever appears in a URL.
//  - Cross-origin bridge: the composition may be served from a different
//    origin than the dashboard, so contentWindow.__hf / document access is
//    gone. The card drives the composition through the validated postMessage
//    protocol ({hf:1,...} both directions): ready/timeupdate in,
//    seek/play/pause out. Play actually starts playback; the playing state
//    is confirmed by timeupdate messages.
//  - Export binds to the acknowledged preview: startRender sends the
//    preview's compile_id + the applied version. Changing template/params
//    after Apply invalidates the preview (unapplied edits invalidate
//    preview/export) and re-preview is required before render.
//  - Cancel honesty: cancelRender awaits the cancel API, then polls the
//    render status once and displays the backend-acknowledged terminal
//    state — never "nothing was published" when the backend already
//    rendered, never "cancelled" when the backend says otherwise.
const PARAM_OPTIONS = {
  caption_size: ['small', 'medium', 'large'],
  caption_position: ['lower-third', 'middle'],
};

function identityKey(jobId, index) {
  return jobId != null && index != null ? `${jobId}:${index}` : null;
}

// Sandboxed preview of one compiled composition. Drives the composition
// through the validated postMessage protocol (see header): the iframe is
// sandboxed to an opaque origin and may live on a different origin than
// the dashboard, so contentWindow.__hf / document access is impossible and
// must never be attempted. The base <video> inside the composition follows
// the same seek/play/pause messages, so captions stay in sync by
// construction. Marked as preview, never as the export reference.
function HfPreview({ src, hfiid, duration, renderWidth, renderHeight }) {
  const frameRef = useRef(null);
  const [t, setT] = useState(0);
  const [ready, setReady] = useState(false);
  const [playing, setPlaying] = useState(false);
  const playReqRef = useRef(false);

  // Cross-origin-safe send. targetOrigin is '*' because the sandboxed frame
  // has an opaque origin (no narrower origin exists); the {hf:1,...} shape
  // is validated on both sides instead.
  const post = (msg) => {
    const w = frameRef.current && frameRef.current.contentWindow;
    if (!w || typeof w.postMessage !== 'function') return false;
    w.postMessage({ hf: 1, ...msg }, '*');
    return true;
  };

  const seekTo = (time) => {
    const clamped = Math.max(0, Math.min(duration || 0, time));
    post({ type: 'seek', t: clamped });
    setT(clamped);
  };

  // Remembers the src this effect last initialized for. Real React only
  // re-runs this effect when src changes (deps below); the naive test-harness
  // useEffect mock ignores deps and re-runs on every render, so the reset
  // block is guarded to keep the declared [src] semantics everywhere
  // (including StrictMode, where mount -> cleanup -> mount must leave the
  // listener registered).
  const initSrcRef = useRef(null);
  useEffect(() => {
    const srcChanged = initSrcRef.current !== src;
    initSrcRef.current = src;
    if (srcChanged) {
      setReady(false); setT(0); setPlaying(false);
      playReqRef.current = false;
    }
    if (typeof window === 'undefined' || typeof window.addEventListener !== 'function') return undefined;
    const onMessage = (event) => {
      const d = event && event.data;
      // Validate the shape on our side; ignore everything else.
      if (!d || typeof d !== 'object' || d.hf !== 1) return;
      // Bind to the OWNED iframe only (review-02): shape-only checks accept
      // messages from unrelated windows, and concurrently mounted previews
      // could otherwise cross-update each other.
      const owned = frameRef.current && frameRef.current.contentWindow;
      if (!owned || event.source !== owned) return;
      // Per-instance binding: a stale/reloaded frame carries a different
      // hfiid (messages without one are still accepted for harness compat).
      if (hfiid && d.hfiid && d.hfiid !== hfiid) return;
      if (d.type === 'ready') {
        setReady(true);
        seekTo(0);
      } else if (d.type === 'timeupdate' && Number.isFinite(d.t)) {
        // Bounded values only: ignore out-of-range timestamps.
        if (d.t < 0 || d.t > (duration || 0) + 1) return;
        setT(d.t);
        // The playing indicator is confirmed by actual composition
        // timeupdates, not just by our play request: if playback never
        // starts, the button honestly stays on "Play".
        if (playReqRef.current) setPlaying(true);
      }
    };
    window.addEventListener('message', onMessage);
    return () => { try { window.removeEventListener('message', onMessage); } catch { /* noop */ } };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [src]);

  const togglePlay = () => {
    if (!ready) return;
    if (playing) {
      playReqRef.current = false;
      post({ type: 'pause' });
      setPlaying(false);
      return;
    }
    if (t >= (duration || 0) - 0.05) post({ type: 'seek', t: 0 });
    playReqRef.current = true;
    post({ type: 'play' });
  };

  const scrub = (value) => {
    playReqRef.current = false;
    setPlaying(false);
    post({ type: 'pause' });
    seekTo(parseFloat(value));
  };

  // Real rendered aspect from the compile manifest; the 9/16 fallback only
  // applies when the manifest carries no render dimensions.
  const rw = Number(renderWidth);
  const rh = Number(renderHeight);
  const aspect = Number.isFinite(rw) && Number.isFinite(rh) && rw > 0 && rh > 0
    ? `${rw}/${rh}`
    : '9/16';

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
        sandbox="allow-scripts"
        style={{ width: '100%', aspectRatio: aspect, maxHeight: 420, background: '#000', border: '1px solid rgba(255,255,255,0.15)', borderRadius: 6 }}
      />
      {!ready && <div style={{ fontSize: 12, opacity: 0.7, marginTop: 4 }}>Loading preview…</div>}
      {ready && (
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginTop: 6 }}>
          <button data-testid="hf-preview-play" onClick={togglePlay}
            style={{ padding: '4px 12px', borderRadius: 6, border: '1px solid rgba(255,255,255,0.25)', background: 'rgba(255,255,255,0.08)', color: 'inherit', cursor: 'pointer' }}>
            {playing ? 'Pause' : 'Play'}
          </button>
          <input data-testid="hf-preview-scrub" type="range" min={0} max={duration} step={0.033}
            value={t} onChange={(e) => scrub(e.target.value)}
            style={{ flex: 1 }} aria-label="Preview scrub" />
          <span style={{ fontSize: 11, opacity: 0.7, minWidth: 88 }}>{t.toFixed(2)}s / {duration.toFixed(2)}s</span>
        </div>
      )}
    </div>
  );
}

export default function HyperframesCard({ data = {}, activeSelectionKey = null, controller = null, lifetimeKey = null }) {
  const { jobId, index, scope } = data;
  const scopeKey = identityKey(jobId, index);
  // Review 01: a null selection authorizes NOTHING. isLive is true only
  // when the app's current selection is non-null and matches this card's
  // own clip identity. Apply is disabled and refuses otherwise.
  const isLive = activeSelectionKey != null && scopeKey != null && scopeKey === activeSelectionKey;

  const [state, setState] = useState('checking');
  const [capability, setCapability] = useState(null);
  const [error, setError] = useState(null);
  const [templateId, setTemplateId] = useState('');
  const [params, setParams] = useState({ caption_size: 'medium', caption_position: 'lower-third' });
  const [preview, setPreview] = useState(null); // { compile_id, preview_identity, composition_url, manifest }
  const [render, setRender] = useState(null); // { render_id, state, output_url, metrics, error }
  const [legacy, setLegacy] = useState(null); // { composed_url, version }
  const pollRef = useRef(null);
  // Serializes status polls so requests never overlap (review-02). The
  // chain is reset each time polling (re)starts.
  const pollChainRef = useRef(Promise.resolve());

  // Poll the render status until a terminal state. The interval is cleared
  // ONLY on a terminal state, lifetime end, or status error — never after a
  // successful in-progress poll (review-02: clearing after EVERY poll
  // stopped polling on the first in-progress response).
  const startPolling = (renderId, tok) => {
    clearInterval(pollRef.current);
    pollChainRef.current = Promise.resolve();
    pollRef.current = setInterval(() => {
      pollChainRef.current = pollChainRef.current.then(async () => {
        try {
          const st = await hyperframesRenderStatus(renderId);
          if (!lifetimeAlive(tok)) { clearInterval(pollRef.current); return; }
          const rs = st.state;
          if (rs !== 'rendering') {
            clearInterval(pollRef.current);
            commit(tok, () => {
              setRender({ render_id: renderId, state: rs, output_url: st.output_url, metrics: st.metrics, error: st.error });
              setState(rs === 'rendered' ? 'rendered' : rs === 'cancelled' ? 'cancelled' : 'failed');
              if (rs !== 'rendered' && rs !== 'cancelled') setError(st.error || 'Render failed');
            });
          } else {
            commit(tok, () => setRender({ render_id: renderId, state: rs }));
          }
        } catch (e) {
          clearInterval(pollRef.current);
          commit(tok, () => { setRender({ render_id: renderId, state: 'failed', error: e.message }); setState('failed'); setError(e.message); });
        }
      });
    }, 1000);
  };
  const aliveRef = useRef(true);
  const [appliedVersion, setAppliedVersion] = useState(null);

  // Lifetime gate (review 01): liveRef always carries the CURRENT props
  // (re-assigned on every render), so a delayed async continuation can
  // compare against reality instead of trusting its closure. Capture the
  // key at each async start; after every await, lifetimeAlive(tok) must
  // hold or the continuation is dropped: no state commit, no save.
  const liveRef = useRef(null);
  liveRef.current = { activeSelectionKey, lifetimeKey, scopeKey, jobId, index };
  const captureLifetime = () => ({
    activeSelectionKey: liveRef.current.activeSelectionKey,
    lifetimeKey: liveRef.current.lifetimeKey,
  });
  const lifetimeAlive = (tok) => {
    const cur = liveRef.current;
    return cur.lifetimeKey === tok.lifetimeKey
      && cur.activeSelectionKey != null
      && cur.activeSelectionKey === tok.activeSelectionKey
      && cur.scopeKey === cur.activeSelectionKey;
  };

  useEffect(() => () => { aliveRef.current = false; clearInterval(pollRef.current); }, []);

  const commit = (tok, fn) => {
    if (!aliveRef.current) return;
    if (!lifetimeAlive(tok)) {
      setError('This panel was opened for a different clip — select that clip again to continue here.');
      return;
    }
    fn();
  };

  // --- capability fetch on mount / identity change ---
  useEffect(() => {
    if (scopeKey == null) { setState('no-selection'); return; }
    let cancelled = false;
    (async () => {
      try {
        const cap = await getHyperframesCapability();
        if (cancelled) return;
        const cur = liveRef.current;
        if (cur.activeSelectionKey != null && cur.scopeKey !== cur.activeSelectionKey) {
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

  const finishingSelection = () => ({
    renderer: 'hyperframes',
    template: templateId,
    caption_size: params.caption_size,
    caption_position: params.caption_position,
  });

  // --- apply: typed finishing selection through the single project authority ---
  const apply = async () => {
    if (!isLive) {
      setError('Select the clip this panel belongs to before applying — a cleared or changed selection cannot authorize a save.');
      return;
    }
    const tok = captureLifetime();
    setState('applying'); setError(null);
    try {
      if (controller) {
        // Single project authority (review 01): the selected workspace
        // controller's serialized queue owns the mutation. The controller
        // carries expectedVersion = the acknowledged version and surfaces
        // SAVE_INTERRUPTED honestly; the mutate callback only sets the
        // finishing selection on the clone it receives.
        const saved = await controller.applyEdit('Hyperframes finishing', (project) => {
          project.finishing = finishingSelection();
          return project;
        });
        commit(tok, () => { setState('applied'); setAppliedVersion(saved?.version ?? null); });
        return;
      }
      // Standalone compatibility (no controller host): the explicit
      // direct-API path with the version guard. Production hosts always
      // pass the controller.
      const current = await getClipProject(jobId, index);
      if (!lifetimeAlive(tok)) return;
      const project = current.project || current;
      const version = current.version ?? project.version ?? 1;
      project.finishing = finishingSelection();
      const saved = await saveClipProject(jobId, index, project, { expectedVersion: version });
      commit(tok, () => { setState('applied'); setAppliedVersion(saved.version ?? version + 1); });
    } catch (e) {
      if (!aliveRef.current) return;
      if (e?.code === 'SAVE_INTERRUPTED') {
        // The save request was already sent; the lifetime ended before the
        // ack. Never claim it failed, never retry blindly.
        setState('error');
        setError('The apply request was already sent but its confirmation was interrupted (session reset or selection change). It may already be saved — reload the project to verify before retrying.');
        return;
      }
      commit(tok, () => {
        setState(e?.status === 409 ? 'proposal' : 'error');
        setError(e?.code === 'DISPOSED'
          ? 'Apply rejected: the project controller was disposed (the selection changed). Select the clip again to continue.'
          : `Apply failed: ${e.message}`);
      });
    }
  };

  // Unapplied template/param edits invalidate the acknowledged preview:
  // export must never render a compile_id from a different selection.
  // Review-02: a change after Apply with NO acknowledged preview yet
  // returns to 'proposal' — changed settings must be re-applied /
  // acknowledged before they can be previewed or exported. A change after
  // an acknowledged preview keeps the applied settings ('applied') and
  // allows re-preview of the variant; the preview itself is invalidated.
  const invalidatePreview = () => {
    setPreview(null);
    setRender(null);
    setState((s) => (s === 'preview' ? 'applied' : 'proposal'));
  };
  const onTemplateChange = (e) => { setTemplateId(e.target.value); invalidatePreview(); };
  const onParamChange = (p) => (e) => { setParams((prev) => ({ ...prev, [p]: e.target.value })); invalidatePreview(); };

  // --- compile -> preview (same composition identity as export) ---
  const startPreview = async () => {
    if (!isLive) return;
    const tok = captureLifetime();
    setState('compiling'); setError(null);
    try {
      const res = await compileHyperframes(jobId, index, {
        template_id: templateId,
        params,
        expected_version: appliedVersion,
      });
      if (!lifetimeAlive(tok)) return;
      // The sandboxed iframe performs plain navigation and cannot attach
      // apiFetch's Bearer <redacted>; mint a short-lived preview token via
      // the authed POST and authorize the composition URL with it. No
      // Supabase JWT ever goes into a URL.
      // Mint a short-lived preview token via the authed POST and authorize
      // the composition URL with it. A failed mint is an honest preview
      // error (never silently load unauthenticated). No JWT in URLs.
      const tokenRes = await getHyperframesPreviewToken(jobId, res.compile_id);
      if (!lifetimeAlive(tok)) return;
      const token = tokenRes && tokenRes.token;
      if (!token) throw new Error('preview token missing from server response');
      commit(tok, () => {
        // Per-instance binding (review-02): the preview iframe URL carries
        // a unique hfiid that the composition echoes in its messages, so a
        // stale/reloaded frame cannot update a concurrently mounted one.
        const hfiid = `hf-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`;
        setPreview({
          compile_id: res.compile_id,
          preview_identity: res.preview_identity, // manifest digest string
          composition_url: `${hyperframesCompositionUrl(jobId, res.compile_id, token)}&hfiid=${encodeURIComponent(hfiid)}`,
          hfiid,
          manifest: res.manifest || null,
        });
        setState('preview');
      });
    } catch (e) {
      commit(tok, () => { setState('applied'); setError(`Preview failed: ${e.message}`); });
    }
  };

  // --- render -> decoded export (bound to the acknowledged preview) ---
  // Backend render states: rendering | rendered | failed | cancelled.
  const startRender = async () => {
    if (!isLive) return;
    // Export consumes the exact acknowledged preview composition: the
    // preview's compile_id plus the applied project version. Unapplied
    // edits clear the preview (invalidatePreview), so a stale compile_id
    // from a different template selection can never be rendered.
    if (!preview?.compile_id) {
      setError('Preview the composition first — render always exports the previewed compile.');
      return;
    }
    const tok = captureLifetime();
    const compileId = preview.compile_id;
    const version = appliedVersion;
    setState('rendering'); setError(null); setLegacy(null);
    try {
      // Serialize through the single project authority (review-02): drain
      // any pending saves so this export composes the acknowledged
      // snapshot, not a stale one. compile_id forwarding alone is not
      // queue integration.
      if (controller && typeof controller.drain === 'function') {
        await controller.drain();
        if (!lifetimeAlive(tok)) return;
      }
      const res = await renderHyperframes(jobId, index, { compile_id: compileId, expected_version: version });
      if (!lifetimeAlive(tok)) return;
      const initialState = res.state || 'rendering';
      commit(tok, () => setRender({ render_id: res.render_id, state: initialState }));
      if (initialState !== 'rendering') {
        // Cache hit (or instant terminal): resolve without polling.
        commit(tok, () => {
          setRender({ render_id: res.render_id, state: initialState, output_url: res.output_url });
          setState(initialState === 'rendered' ? 'rendered' : 'failed');
          if (initialState !== 'rendered') setError(res.error || 'Render failed');
        });
        return;
      }
      startPolling(res.render_id, tok);
    } catch (e) {
      commit(tok, () => { setState('preview'); setError(`Render failed: ${e.message}`); });
    }
  };

  // Cancel honesty (review 01): await the cancel API, then poll the render
  // status ONCE and display the backend-acknowledged terminal state. Never
  // claim "nothing was published" when the backend already rendered, and
  // never claim "cancelled" when the backend says otherwise or the cancel
  // itself failed.
  const cancelRender = async () => {
    const renderId = render?.render_id;
    if (!renderId) return;
    const tok = captureLifetime();
    setError(null);
    clearInterval(pollRef.current);
    try {
      await cancelHyperframesRender(renderId);
      if (!lifetimeAlive(tok)) return;
      const st = await hyperframesRenderStatus(renderId);
      if (!lifetimeAlive(tok)) return;
      const rs = st && st.state;
      commit(tok, () => {
        setRender({ render_id: renderId, state: rs, output_url: st.output_url, metrics: st.metrics, error: st.error });
        if (rs === 'rendered') {
          setState('rendered');
        } else if (rs === 'cancelled') {
          setState('cancelled');
        } else {
          setState('failed');
          setError(st.error || `Cancel acknowledged, but the backend reports the render as "${rs || 'unknown'}".`);
        }
      });
    } catch (e) {
      commit(tok, () => {
        // The cancel did not go through: the render may still be running or
        // may already have finished. Restart polling so the card keeps an
        // honest status path instead of stranding the user — do not claim
        // cancelled.
        setError(`Cancel failed: ${e.message}. The render may still complete — resuming status checks.`);
        if (lifetimeAlive(tok)) startPolling(renderId, tok);
      });
    }
  };

  // --- explicit recoverable legacy choice (contract 2) ---
  const legacyRender = async () => {
    if (!isLive) return;
    const tok = captureLifetime();
    setError(null);
    try {
      if (controller) {
        // Single project authority: drain the serialized queue and compose
        // exactly the acknowledged snapshot — never direct
        // getClipProject/composeProject.
        const out = await controller.exportProject();
        commit(tok, () => {
          setLegacy({ composed_url: out.composed_url, version: out.identity?.version ?? null });
          setState('legacy');
        });
        return;
      }
      // Standalone compatibility: the explicit direct-API path.
      const current = await getClipProject(jobId, index);
      if (!lifetimeAlive(tok)) return;
      const project = current.project || current;
      const res = await composeProject(jobId, index, project);
      commit(tok, () => { setLegacy({ composed_url: res.composed_url, version: res.version }); setState('legacy'); });
    } catch (e) {
      commit(tok, () => setError(`Legacy render failed: ${e.message}`));
    }
  };

  const boxStyle = { background: 'rgba(255,255,255,0.04)', border: '1px solid rgba(255,255,255,0.12)', borderRadius: 8, padding: 10, marginTop: 6, fontSize: 13 };
  const btnStyle = { marginRight: 8, padding: '5px 12px', borderRadius: 6, border: '1px solid rgba(255,255,255,0.25)', background: 'rgba(255,255,255,0.08)', color: 'inherit', cursor: 'pointer' };
  const applyDisabled = state === 'applying' || !isLive;

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
          {!isLive && (
            <div style={{ fontSize: 12, opacity: 0.7, marginBottom: 8 }}>
              The selection changed or was cleared — Apply is disabled until you select this clip again.
            </div>
          )}
          <label style={{ display: 'block', marginBottom: 6 }}>
            Template:{' '}
            <select data-testid="hf-template" value={templateId} onChange={onTemplateChange}
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
                <select data-testid={`hf-${p}`} value={params[p]} onChange={onParamChange(p)}
                  disabled={['applying', 'compiling', 'rendering'].includes(state)} style={{ color: '#000' }}>
                  {PARAM_OPTIONS[p].map((v) => <option key={v} value={v}>{v}</option>)}
                </select>
              </label>
            ))}
          </div>
          <div>
            <button data-testid="hf-apply" style={btnStyle} onClick={apply} disabled={applyDisabled}>
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
          <HfPreview
            src={preview.composition_url}
            hfiid={preview.hfiid}
            duration={preview.manifest?.duration || 10}
            renderWidth={preview.manifest?.render?.width}
            renderHeight={preview.manifest?.render?.height}
          />
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
          {error && <div style={{ color: '#f88', marginTop: 8 }} data-testid="hf-error">{error}</div>}
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
