/**
 * Task 18 — Hyperframes capability card tests.
 *
 * House pattern (task17Workspace.test.js): minimal react hook harness with
 * persistent useState/useRef and immediate useEffect (node: no
 * window/document), mocked api transports, and direct component calls with
 * a manual element-tree walk.
 *
 * Contracts covered:
 *  1. single-clip identity + version-checked Apply (typed finishing via the
 *     selected controller authority; expectedVersion concurrency guard in
 *     the standalone path).
 *  2. default-OFF honesty: legacy renderer named as the unchanged default.
 *  8. delayed async responses can never replace another selection's
 *     preview (stale selection -> 'stale' state, no commit).
 *  5. preview and export share one compiled composition identity.
 *
 * Review 01 additions (promoted coordinator probes, adapted to the
 * controller-prop API):
 *  R1. Apply routes through the selected workspace controller
 *      (controller.applyEdit), never direct saves; legacy export drains via
 *      controller.exportProject(). Without a controller the explicit
 *      standalone direct-API path is kept.
 *  R2. Lifetime gates: lifetimeKey captured at async start, re-checked
 *      after every await (switch/close/reselect/session-reset drops the
 *      continuation); isLive is false when activeSelectionKey is null, so
 *      a cleared selection cannot authorize an old card to save.
 *  R3. Preview auth: authed POST mints a short-lived preview token; the
 *      iframe src carries ?token= (never a Supabase JWT). The composition
 *      is driven through the validated postMessage protocol ({hf:1,...}),
 *      never contentWindow.__hf / document. Play actually starts playback
 *      (timeupdate confirms); the iframe aspect comes from the manifest's
 *      render width/height.
 *  R4. Export binds to the acknowledged preview: startRender sends
 *      { compile_id, expected_version }; unapplied template/param edits
 *      invalidate the preview and require re-preview before render.
 *  R5. Cancel honesty: cancelRender awaits the cancel API, polls the render
 *      status once, and displays the backend-acknowledged terminal state.
 *
 * Load-bearing mutations (run manually, each must fail, then re-run green):
 *  M1: apply ignores the controller prop (direct saveClipProject) ->
 *      'Apply must use the selected workspace controller' fails.
 *  M2: lifetime check removed after the getClipProject await ->
 *      'switching selection during a project fetch prevents the subsequent
 *      save' fails.
 *  M3: startRender drops compile_id -> 'export must send the exact preview
 *      composition identity' fails.
 *  M4: iframe src built without the preview token -> 'preview fetches an
 *      authed token and puts it in the iframe src' fails.
 *  M5: isLive allows null activeSelectionKey -> 'clearing selection cannot
 *      authorize an old card to save' fails.
 */
import { describe, expect, it, vi, beforeEach, afterEach } from 'vitest';
// This repo's JSX transform uses the classic runtime (React.createElement);
// the components don't import React themselves, and the react-module mock
// below spreads the real module, so expose it as the React global.
import React from 'react';
vi.stubGlobal('React', React);

const hooks = vi.hoisted(() => ({ values: [], index: 0 }));
const hfApi = vi.hoisted(() => ({
  getHyperframesCapability: vi.fn(),
  compileHyperframes: vi.fn(),
  renderHyperframes: vi.fn(),
  hyperframesRenderStatus: vi.fn(),
  cancelHyperframesRender: vi.fn(),
  getHyperframesPreviewToken: vi.fn(),
  hyperframesCompositionUrl: vi.fn((j, c, t) => `/api/hyperframes/composition/${j}/${c}/index.html${t ? `?token=${t}` : ''}`),
}));
const realApi = vi.hoisted(() => ({
  getClipProject: vi.fn(),
  saveClipProject: vi.fn(),
  composeProject: vi.fn(),
}));

vi.mock('react', async (original) => {
  const actual = await original();
  return {
    ...actual,
    useState: (initial) => {
      const i = hooks.index++;
      if (!(i in hooks.values)) hooks.values[i] = typeof initial === 'function' ? initial() : initial;
      return [
        hooks.values[i],
        (v) => { hooks.values[i] = typeof v === 'function' ? v(hooks.values[i]) : v; },
      ];
    },
    useRef: (initial) => {
      const i = hooks.index++;
      if (!(i in hooks.values)) hooks.values[i] = { current: initial };
      return hooks.values[i];
    },
    useMemo: (fn) => fn(),
    useCallback: (fn) => fn,
    useEffect: (fn) => {
      try {
        const r = fn();
        if (r && typeof r.catch === 'function') r.catch(() => {});
      } catch { /* node env: best-effort */ }
    },
  };
});
vi.mock('../api/hyperframesApi.js', () => ({ ...hfApi }));
vi.mock('../api/realApi.js', () => ({ ...realApi }));

import HyperframesCard from './HyperframesCard.jsx';
import ChatThread from './ChatThread.jsx';

// --- element tree helpers (real react/jsx-runtime elements) ---
function walk(node, out = []) {
  if (Array.isArray(node)) { node.forEach((k) => walk(k, out)); return out; }
  if (!node || typeof node !== 'object') return out;
  out.push(node);
  const kids = node.props?.children;
  if (Array.isArray(kids)) kids.forEach((k) => walk(k, out));
  else if (kids) walk(kids, out);
  return out;
}
function findByTestId(root, id) {
  return walk(root).find((n) => n.props && n.props['data-testid'] === id) || null;
}
function textOf(node) {
  const parts = [];
  const collect = (n) => {
    if (typeof n === 'string') parts.push(n);
    else if (Array.isArray(n)) n.forEach(collect);
    else if (n && typeof n === 'object') collect(n.props?.children);
  };
  collect(node);
  return parts.join(' ');
}
function optionValues(root) {
  return walk(root).filter((n) => n.type === 'option').map((n) => n.props?.value);
}
const flush = async (n = 5) => { for (let i = 0; i < n; i++) await Promise.resolve(); };

const TEMPLATES = [
  { id: 'clean-captions', title: 'Clean captions' },
  { id: 'emphasized-word', title: 'Emphasized word' },
  { id: 'timed-opening', title: 'Timed opening' },
];
const card = (props) => { hooks.index = 0; return HyperframesCard(props); };
const data = (jobId, index, scope = 'single') => ({ command: '/hyperframes', jobId, index, scope });

beforeEach(() => {
  hooks.values = [];
  hooks.index = 0;
  vi.clearAllMocks();
  hfApi.getHyperframesCapability.mockResolvedValue({ available: true, reason: null, templates: TEMPLATES });
  hfApi.getHyperframesPreviewToken.mockResolvedValue({ token: 'tok-123', expires_in: 300 });
  realApi.getClipProject.mockResolvedValue({ project: { version: 1 }, version: 1 });
  realApi.saveClipProject.mockResolvedValue({ version: 2, project: {} });
});

afterEach(() => {
  // The postMessage tests stub a window; restore the module-top React stub.
  vi.unstubAllGlobals();
  vi.stubGlobal('React', React);
});

describe('HyperframesCard capability states', () => {
  it('shows the no-selection state when no clip is selected', async () => {
    const tree = card({ data: data(null, null, 'global'), activeSelectionKey: null });
    await flush();
    const t = textOf(card({ data: data(null, null, 'global'), activeSelectionKey: null }));
    expect(t).toMatch(/Select a clip first/);
    expect(hfApi.getHyperframesCapability).not.toHaveBeenCalled();
    expect(tree).toBeTruthy();
  });

  it('offers the three reviewed templates with closed params on capability', async () => {
    card({ data: data('j1', 0), activeSelectionKey: 'j1:0' });
    await flush();
    const tree = card({ data: data('j1', 0), activeSelectionKey: 'j1:0' });
    expect(optionValues(tree)).toEqual(expect.arrayContaining(['clean-captions', 'emphasized-word', 'timed-opening']));
    expect(optionValues(tree)).not.toContain('cinematic-zoom');
    // caption_position enum must be the backend's closed enum exactly
    expect(optionValues(tree)).toEqual(
      expect.arrayContaining(['lower-third', 'middle']));
    expect(optionValues(tree)).not.toContain('bottom');
    const t = textOf(tree);
    expect(t).toMatch(/legacy renderer stays the default/);
    expect(findByTestId(tree, 'hf-apply')).toBeTruthy();
  });

  it('shows honest pending state when the backend is unavailable (contract 2)', async () => {
    hfApi.getHyperframesCapability.mockResolvedValue({ available: false, reason: 'worker not installed' });
    card({ data: data('j1', 0), activeSelectionKey: 'j1:0' });
    await flush();
    const tree = card({ data: data('j1', 0), activeSelectionKey: 'j1:0' });
    const t = textOf(tree);
    expect(t).toMatch(/isn.t available here/);
    expect(t).toMatch(/accepted legacy renderer is unchanged/);
    expect(findByTestId(tree, 'hf-apply')).toBeNull();
  });

  it('refuses to commit a delayed capability response for a stale selection (contract 8)', async () => {
    let resolveCap;
    hfApi.getHyperframesCapability.mockReturnValue(new Promise((r) => { resolveCap = r; }));
    card({ data: data('j1', 0), activeSelectionKey: 'j1:0' });
    // selection moves away BEFORE the capability resolves
    const mid = card({ data: data('j1', 0), activeSelectionKey: 'j2:0' });
    resolveCap({ available: true, reason: null, templates: TEMPLATES });
    await flush();
    const tree = card({ data: data('j1', 0), activeSelectionKey: 'j2:0' });
    const t = textOf(tree);
    expect(t).toMatch(/different clip/);
    expect(optionValues(tree)).not.toContain('clean-captions');
    expect(findByTestId(tree, 'hf-apply')).toBeNull();
    expect(mid).toBeTruthy();
  });
});

describe('HyperframesCard apply (contract 1)', () => {
  it('saves a typed finishing selection through the version guard', async () => {
    card({ data: data('j1', 0), activeSelectionKey: 'j1:0' });
    await flush();
    let tree = card({ data: data('j1', 0), activeSelectionKey: 'j1:0' });
    const btn = findByTestId(tree, 'hf-apply');
    expect(btn).toBeTruthy();
    await btn.props.onClick();
    await flush();
    tree = card({ data: data('j1', 0), activeSelectionKey: 'j1:0' });

    expect(realApi.saveClipProject).toHaveBeenCalledTimes(1);
    const [, , project, opts] = realApi.saveClipProject.mock.calls[0];
    expect(opts.expectedVersion).toBe(1);
    expect(project.finishing).toEqual({
      renderer: 'hyperframes',
      template: 'clean-captions',
      caption_size: 'medium',
      caption_position: 'lower-third',
    });
    expect(textOf(tree)).toMatch(/Applied to project version 2/);
    expect(findByTestId(tree, 'hf-preview')).toBeTruthy();
  });

  it('surfaces a 409 as an honest apply error without committing', async () => {
    realApi.saveClipProject.mockRejectedValue(Object.assign(new Error('stale project version'), { status: 409 }));
    card({ data: data('j1', 0), activeSelectionKey: 'j1:0' });
    await flush();
    let tree = card({ data: data('j1', 0), activeSelectionKey: 'j1:0' });
    await findByTestId(tree, 'hf-apply').props.onClick();
    await flush();
    tree = card({ data: data('j1', 0), activeSelectionKey: 'j1:0' });
    expect(textOf(tree)).toMatch(/Apply failed/);
    expect(textOf(tree)).not.toMatch(/Applied to project version/);
  });
});

describe('ChatThread /hyperframes wiring', () => {
  it('maps the message scope to jobId/index and passes activeSelectionKey', async () => {
    const thread = {
      id: 't1',
      messages: [{
        id: 'm1', role: 'ai', kind: 'hyperframes',
        data: { command: '/hyperframes', scope: { threadId: 't1', jobId: 'j9', clipIndex: 2 } },
      }],
    };
    const renderThread = () => { hooks.index = 0; return ChatThread({ thread, activeSelectionKey: 'j9:2' }); };
    const tree = renderThread();
    // HyperframesMsg is the wrapper: invoke it to get the HyperframesCard
    // element, then drive the card directly (no DOM renderer here).
    const msgEl = walk(tree).find((n) => n.type && n.type.name === 'HyperframesMsg');
    expect(msgEl).toBeTruthy();
    const cardEl = msgEl.type(msgEl.props);
    expect(cardEl).toBeTruthy();
    // HyperframesMsg maps the message scope -> jobId/index (identity for
    // contract 1) and threads activeSelectionKey through (contract 8).
    expect(cardEl.props.data).toMatchObject({ jobId: 'j9', index: 2 });
    expect(cardEl.props.activeSelectionKey).toBe('j9:2');
    // Drive the card itself (no DOM renderer in this harness).
    hooks.values = [];
    hooks.index = 0;
    cardEl.type(cardEl.props);
    await flush();
    hooks.index = 0;
    const cardTree = cardEl.type(cardEl.props);
    expect(hfApi.getHyperframesCapability).toHaveBeenCalledWith();
    expect(findByTestId(cardTree, 'hyperframes-card')).toBeTruthy();
  });

  it('threads the controller and lifetimeKey to the matching card only (R1)', async () => {
    const controller = { applyEdit: vi.fn() };
    const thread = {
      id: 't1',
      messages: [
        { id: 'm1', role: 'ai', kind: 'hyperframes', data: { command: '/hyperframes', scope: { threadId: 't1', jobId: 'j9', clipIndex: 2 } } },
        { id: 'm2', role: 'ai', kind: 'hyperframes', data: { command: '/hyperframes', scope: { threadId: 't1', jobId: 'j7', clipIndex: 0 } } },
      ],
    };
    const renderThread = () => {
      hooks.index = 0;
      return ChatThread({ thread, activeSelectionKey: 'j9:2', controller, controllerKey: 'j9:2', lifetimeKey: 'life-9' });
    };
    const tree = renderThread();
    const msgs = walk(tree).filter((n) => n.type && n.type.name === 'HyperframesMsg');
    expect(msgs).toHaveLength(2);
    const cardEls = msgs.map((m) => m.type(m.props));
    // The card for the currently selected clip gets the shared controller.
    expect(cardEls[0].props.controller).toBe(controller);
    expect(cardEls[0].props.lifetimeKey).toBe('life-9');
    // A card for any other clip must NOT receive it (wrong project).
    expect(cardEls[1].props.controller).toBeNull();
    expect(cardEls[1].props.lifetimeKey).toBe('life-9');
  });
});

describe('HyperframesCard preview/export identity (contract 5)', () => {
  const MANIFEST = {
    project_version: 2,
    template: { id: 'clean-captions', version: 1 },
    renderer: { engine: '0.8.145' },
    render: { width: 608, height: 1080, fps: 30 },
    duration: 10.0,
  };
  async function toPreview() {
    hfApi.compileHyperframes.mockResolvedValue({
      compile_id: 'abcdef0123456789',
      preview_identity: 'digest-abc-123',
      manifest: MANIFEST,
    });
    card({ data: data('j1', 0), activeSelectionKey: 'j1:0' });
    await flush();
    let tree = card({ data: data('j1', 0), activeSelectionKey: 'j1:0' });
    await findByTestId(tree, 'hf-apply').props.onClick();
    await flush();
    tree = card({ data: data('j1', 0), activeSelectionKey: 'j1:0' });
    await findByTestId(tree, 'hf-preview').props.onClick();
    await flush();
    return card({ data: data('j1', 0), activeSelectionKey: 'j1:0' });
  }

  it('renders a sandboxed iframe preview (not the player element) on the composition URL', async () => {
    const tree = await toPreview();
    // HfPreview is a nested function component: invoke it like the
    // ChatThread test invokes HyperframesMsg (no DOM renderer here).
    const hfPreviewEl = walk(tree).find((n) => n.type && n.type.name === 'HfPreview');
    expect(hfPreviewEl).toBeTruthy();
    hooks.index = 0;
    const previewTree = hfPreviewEl.type(hfPreviewEl.props);
    const iframe = findByTestId(previewTree, 'hf-preview-iframe');
    expect(iframe).toBeTruthy();
    expect(iframe.props.src).toMatch(/^\/api\/hyperframes\/composition\/j1\/abcdef0123456789\/index\.html\?token=tok-123&hfiid=hf-[A-Za-z0-9-]+$/);
    expect(iframe.props.sandbox).toMatch(/allow-scripts/);
    // no vendored player element: the player cannot drive the seek protocol
    expect(walk(tree).filter((n) => n.type === 'hyperframes-player')).toEqual([]);
    const id = findByTestId(tree, 'hf-preview-identity');
    expect(textOf(id)).toMatch(/digest-abc-1/); // 12-char digest prefix
    expect(textOf(id)).toMatch(/clean-captions/);
    // (review-02) the iframe URL carries the per-instance hfiid; the
    // composition echoes it so a stale frame cannot cross-update.
  });

  it('template options show titles from the capability (not blank)', async () => {
    card({ data: data('j1', 0), activeSelectionKey: 'j1:0' });
    await flush();
    const tree = card({ data: data('j1', 0), activeSelectionKey: 'j1:0' });
    const options = walk(tree).filter((n) => n.type === 'option');
    const labels = options.map((n) => textOf(n).trim()).filter(Boolean);
    expect(labels).toEqual(expect.arrayContaining(['Clean captions', 'Emphasized word', 'Timed opening']));
  });

  it('maps backend render states (rendering/rendered) instead of invented statuses', async () => {
    const tree0 = await toPreview();
    hfApi.renderHyperframes.mockResolvedValue({ render_id: 'rid-1', state: 'rendering' });
    await findByTestId(tree0, 'hf-render').props.onClick();
    await flush();
    const tree1 = card({ data: data('j1', 0), activeSelectionKey: 'j1:0' });
    expect(textOf(tree1)).toMatch(/Rendering/);
    expect(textOf(tree1)).not.toMatch(/queued/);
  });
});

describe('review-01 R1/R2: single project authority + lifetime gates', () => {
  it('Apply must use the selected workspace controller, never direct saves', async () => {
    const controller = { applyEdit: vi.fn().mockResolvedValue({ version: 2 }) };
    const props = { data: data('j1', 0), activeSelectionKey: 'j1:0', controller };
    card(props); await flush();
    await findByTestId(card(props), 'hf-apply').props.onClick();
    await flush();
    expect(controller.applyEdit).toHaveBeenCalledTimes(1);
    expect(realApi.saveClipProject).not.toHaveBeenCalled();
    const tree = card(props);
    expect(textOf(tree)).toMatch(/Applied to project version 2/);
  });

  it('switching selection during a project fetch prevents the subsequent save', async () => {
    let resolveGet;
    realApi.getClipProject.mockReturnValue(new Promise((r) => { resolveGet = r; }));
    const p = { data: data('j1', 0), activeSelectionKey: 'j1:0' };
    card(p); await flush();
    const pending = findByTestId(card(p), 'hf-apply').props.onClick();
    card({ ...p, activeSelectionKey: 'j2:0' });
    resolveGet({ project: { version: 1 }, version: 1 }); await pending;
    expect(realApi.saveClipProject).not.toHaveBeenCalled();
  });

  it('clearing selection cannot authorize an old card to save', async () => {
    const p = { data: data('j1', 0), activeSelectionKey: 'j1:0' };
    card(p); await flush();
    const tree = card({ ...p, activeSelectionKey: null });
    const button = findByTestId(tree, 'hf-apply');
    expect(button).toBeTruthy();
    expect(button.props.disabled).toBe(true);
    if (button && !button.props.disabled) await button.props.onClick();
    expect(realApi.saveClipProject).not.toHaveBeenCalled();
  });

  it('a lifetimeKey change during a project fetch drops the continuation', async () => {
    let resolveGet;
    realApi.getClipProject.mockReturnValue(new Promise((r) => { resolveGet = r; }));
    const p = { data: data('j1', 0), activeSelectionKey: 'j1:0', lifetimeKey: 'life-1' };
    card(p); await flush();
    const pending = findByTestId(card(p), 'hf-apply').props.onClick();
    // session reset / reselection bumps the lifetime key mid-flight
    card({ ...p, lifetimeKey: 'life-2' });
    resolveGet({ project: { version: 1 }, version: 1 });
    await pending;
    expect(realApi.saveClipProject).not.toHaveBeenCalled();
  });

  it('a controller save interrupted after dispatch stays honest and never retries', async () => {
    const err = Object.assign(
      new Error('interrupted: apply (save request issued; confirmation cancelled)'),
      { code: 'SAVE_INTERRUPTED' },
    );
    const controller = { applyEdit: vi.fn().mockRejectedValue(err) };
    const p = { data: data('j1', 0), activeSelectionKey: 'j1:0', controller };
    card(p); await flush();
    await findByTestId(card(p), 'hf-apply').props.onClick();
    await flush();
    const tree = card(p);
    expect(controller.applyEdit).toHaveBeenCalledTimes(1);
    expect(realApi.saveClipProject).not.toHaveBeenCalled();
    expect(textOf(tree)).toMatch(/already been sent|interrupted/i);
    expect(textOf(tree)).not.toMatch(/Applied to project version/);
  });

  it('legacy export drains through the controller, never direct compose', async () => {
    const controller = {
      applyEdit: vi.fn().mockResolvedValue({ version: 2 }),
      exportProject: vi.fn().mockResolvedValue({ composed_url: '/legacy/x.mp4', identity: { version: 3 } }),
    };
    hfApi.compileHyperframes.mockResolvedValue({ compile_id: 'c1', preview_identity: 'd', manifest: { duration: 5 } });
    hfApi.renderHyperframes.mockResolvedValue({ render_id: 'r', state: 'failed', error: 'worker blew up' });
    const p = { data: data('j1', 0), activeSelectionKey: 'j1:0', controller };
    card(p); await flush();
    let tree = card(p);
    await findByTestId(tree, 'hf-apply').props.onClick();
    await flush();
    tree = card(p);
    await findByTestId(tree, 'hf-preview').props.onClick();
    await flush();
    tree = card(p);
    await findByTestId(tree, 'hf-render').props.onClick();
    await flush();
    tree = card(p);
    expect(findByTestId(tree, 'hf-legacy')).toBeTruthy();
    await findByTestId(tree, 'hf-legacy').props.onClick();
    await flush();
    tree = card(p);
    expect(controller.exportProject).toHaveBeenCalledTimes(1);
    expect(realApi.composeProject).not.toHaveBeenCalled();
    expect(realApi.getClipProject).not.toHaveBeenCalled();
    expect(textOf(tree)).toMatch(/accepted legacy renderer/);
    expect(findByTestId(tree, 'hf-legacy-output')).toBeTruthy();
  });
});

describe('review-01 R3: preview auth + postMessage + aspect + playback', () => {
  const MANIFEST = {
    project_version: 2,
    template: { id: 'clean-captions', version: 1 },
    renderer: { engine: '0.8.145' },
    render: { width: 608, height: 1080, fps: 30 },
    duration: 10.0,
  };
  async function toPreview() {
    hfApi.compileHyperframes.mockResolvedValue({
      compile_id: 'abcdef0123456789',
      preview_identity: 'digest-abc-123',
      manifest: MANIFEST,
    });
    card({ data: data('j1', 0), activeSelectionKey: 'j1:0' });
    await flush();
    let tree = card({ data: data('j1', 0), activeSelectionKey: 'j1:0' });
    await findByTestId(tree, 'hf-apply').props.onClick();
    await flush();
    tree = card({ data: data('j1', 0), activeSelectionKey: 'j1:0' });
    await findByTestId(tree, 'hf-preview').props.onClick();
    await flush();
    return card({ data: data('j1', 0), activeSelectionKey: 'j1:0' });
  }
  // Drive the nested HfPreview with a stubbed window (message bridge) and
  // a fake contentWindow carrying ONLY postMessage: no __hf, no document,
  // the cross-origin reality. HfPreview hook order: frameRef(0), t(1),
  // ready(2), playing(3), playReqRef(4).
  function drivePreview(hfPreviewEl) {
    const listeners = {};
    vi.stubGlobal('window', {
      addEventListener: vi.fn((type, fn) => { listeners[type] = fn; }),
      removeEventListener: vi.fn(),
    });
    const postMessage = vi.fn();
    // Fresh hook slots: the card's slots are occupied (index 0 is the card's
    // state string), so the nested component gets its own array.
    hooks.values = [];
    hooks.index = 0;
    hfPreviewEl.type(hfPreviewEl.props);
    hooks.values[0].current = { contentWindow: { postMessage } };
    const rerender = () => { hooks.index = 0; return hfPreviewEl.type(hfPreviewEl.props); };
    return { listeners, postMessage, rerender };
  }

  it('preview fetches an authed token and puts it in the iframe src (never a JWT)', async () => {
    const tree = await toPreview();
    expect(hfApi.getHyperframesPreviewToken).toHaveBeenCalledTimes(1);
    expect(hfApi.getHyperframesPreviewToken).toHaveBeenCalledWith('j1', 'abcdef0123456789');
    const hfPreviewEl = walk(tree).find((n) => n.type && n.type.name === 'HfPreview');
    hooks.index = 0;
    const previewTree = hfPreviewEl.type(hfPreviewEl.props);
    const iframe = findByTestId(previewTree, 'hf-preview-iframe');
    expect(iframe.props.src).toMatch(/^\/api\/hyperframes\/composition\/j1\/abcdef0123456789\/index\.html\?token=tok-123&hfiid=hf-[A-Za-z0-9-]+$/);
    expect(iframe.props.src).not.toMatch(/eyJ/); // no Supabase JWT in the URL
  });

  it('a missing preview token fails honestly instead of loading unauthenticated', async () => {
    hfApi.getHyperframesPreviewToken.mockResolvedValue({});
    hfApi.compileHyperframes.mockResolvedValue({
      compile_id: 'c9', preview_identity: 'd', manifest: { duration: 5 },
    });
    card({ data: data('j1', 0), activeSelectionKey: 'j1:0' });
    await flush();
    let tree = card({ data: data('j1', 0), activeSelectionKey: 'j1:0' });
    await findByTestId(tree, 'hf-apply').props.onClick();
    await flush();
    tree = card({ data: data('j1', 0), activeSelectionKey: 'j1:0' });
    await findByTestId(tree, 'hf-preview').props.onClick();
    await flush();
    tree = card({ data: data('j1', 0), activeSelectionKey: 'j1:0' });
    expect(textOf(tree)).toMatch(/Preview failed/);
    expect(findByTestId(tree, 'hf-preview-frame')).toBeNull();
  });

  it('sizes the preview iframe from the manifest render dimensions', async () => {
    const tree = await toPreview();
    const hfPreviewEl = walk(tree).find((n) => n.type && n.type.name === 'HfPreview');
    hooks.index = 0;
    const previewTree = hfPreviewEl.type(hfPreviewEl.props);
    const iframe = findByTestId(previewTree, 'hf-preview-iframe');
    expect(iframe.props.style.aspectRatio).toBe('608/1080');
  });

  it('drives the composition through the postMessage protocol, never contentWindow.__hf', async () => {
    const tree = await toPreview();
    const hfPreviewEl = walk(tree).find((n) => n.type && n.type.name === 'HfPreview');
    const { listeners, postMessage, rerender } = drivePreview(hfPreviewEl);
    expect(listeners.message).toBeTruthy();
    // (review-02) messages are bound to event.source === the owned iframe;
    // the harness provides the owned window as the source.
    const owned = hooks.values[0].current.contentWindow;
    // The composition announces readiness; junk messages are ignored.
    listeners.message({ source: owned, data: { nope: true } });
    listeners.message({ source: owned, data: { hf: 1, type: 'ready' } });
    await flush();
    expect(hooks.values[2]).toBe(true); // ready
    // Play ACTUALLY starts playback via the protocol (targetOrigin '*').
    const playBtn = findByTestId(rerender(), 'hf-preview-play');
    expect(playBtn).toBeTruthy();
    await playBtn.props.onClick();
    expect(postMessage).toHaveBeenCalledWith({ hf: 1, type: 'play' }, '*');
    // Playing state is confirmed by the composition's timeupdate messages.
    listeners.message({ source: owned, data: { hf: 1, type: 'timeupdate', t: 3.25 } });
    expect(hooks.values[1]).toBe(3.25);
    const pauseBtn = findByTestId(rerender(), 'hf-preview-play');
    expect(textOf(pauseBtn)).toMatch(/Pause/);
    await pauseBtn.props.onClick();
    expect(postMessage).toHaveBeenCalledWith({ hf: 1, type: 'pause' }, '*');
    // Scrub sends pause + seek through the protocol.
    const scrub = findByTestId(rerender(), 'hf-preview-scrub');
    scrub.props.onChange({ target: { value: '2.5' } });
    expect(postMessage).toHaveBeenCalledWith({ hf: 1, type: 'seek', t: 2.5 }, '*');
    // Every cross-origin post uses '*' (sandboxed opaque origin); the fake
    // contentWindow never had __hf/document and nothing threw.
    expect(postMessage.mock.calls.length).toBeGreaterThan(0);
    expect(postMessage.mock.calls.every((c) => c[1] === '*')).toBe(true);
    expect('__hf' in (hooks.values[0].current || {})).toBe(false);
  });
});

describe('review-01 R4: export bound to the acknowledged preview', () => {
  const MANIFEST = { project_version: 2, duration: 5 };
  it('export must send the exact preview composition identity', async () => {
    const p = { data: data('j1', 0), activeSelectionKey: 'j1:0' };
    hfApi.compileHyperframes.mockResolvedValue({ compile_id: 'abcdef0123456789', preview_identity: 'same', manifest: { duration: 5, project_version: 2 } });
    hfApi.renderHyperframes.mockResolvedValue({ render_id: 'r', state: 'rendered', output_url: '/out' });
    card(p); await flush();
    await findByTestId(card(p), 'hf-apply').props.onClick();
    await flush();
    const applied = card(p);
    findByTestId(applied, 'hf-template').props.onChange({ target: { value: 'emphasized-word' } });
    // (review-02) a template change after Apply requires re-acknowledgment:
    // the card returns to 'proposal'; re-Apply, then preview and render.
    await findByTestId(card(p), 'hf-apply').props.onClick();
    await flush();
    await findByTestId(card(p), 'hf-preview').props.onClick();
    await flush();
    await findByTestId(card(p), 'hf-render').props.onClick();
    await flush();
    expect(hfApi.renderHyperframes).toHaveBeenCalledWith('j1', 0, expect.objectContaining({ compile_id: 'abcdef0123456789' }));
    const body = hfApi.renderHyperframes.mock.calls[0][2];
    expect(body.expected_version).toBe(2);
  });

  it('changing the template after preview invalidates it and requires re-preview', async () => {
    hfApi.compileHyperframes
      .mockResolvedValueOnce({ compile_id: 'compile-aaa', preview_identity: 'd1', manifest: MANIFEST })
      .mockResolvedValueOnce({ compile_id: 'compile-bbb', preview_identity: 'd2', manifest: MANIFEST });
    hfApi.renderHyperframes.mockResolvedValue({ render_id: 'r2', state: 'rendered', output_url: '/o2' });
    const p = { data: data('j1', 0), activeSelectionKey: 'j1:0' };
    card(p); await flush();
    let tree = card(p);
    await findByTestId(tree, 'hf-apply').props.onClick();
    await flush();
    tree = card(p);
    await findByTestId(tree, 'hf-preview').props.onClick();
    await flush();
    tree = card(p);
    expect(findByTestId(tree, 'hf-render')).toBeTruthy();
    // Unapplied edit AFTER the preview was acknowledged: the old compile
    // must not be renderable anymore.
    findByTestId(tree, 'hf-template').props.onChange({ target: { value: 'emphasized-word' } });
    tree = card(p);
    expect(findByTestId(tree, 'hf-render')).toBeNull();
    expect(findByTestId(tree, 'hf-preview')).toBeTruthy();
    expect(textOf(tree)).not.toMatch(/compile-aaa/);
    // Re-preview compiles the NEW selection; render binds to the new id.
    await findByTestId(tree, 'hf-preview').props.onClick();
    await flush();
    tree = card(p);
    expect(hfApi.compileHyperframes).toHaveBeenLastCalledWith(
      'j1', 0, expect.objectContaining({ template_id: 'emphasized-word' }));
    await findByTestId(tree, 'hf-render').props.onClick();
    await flush();
    expect(hfApi.renderHyperframes).toHaveBeenCalledWith(
      'j1', 0, expect.objectContaining({ compile_id: 'compile-bbb' }));
  });
});

describe('review-01 R5: cancel reflects the backend-acknowledged state', () => {
  const MANIFEST = { project_version: 2, duration: 5 };
  // House pattern: the harness useEffect ignores deps and re-runs on every
  // card() call, scheduling a stale 'proposal' reset. So (like toPreview)
  // this helper ends with the preview tree and the tests click hf-render on
  // that same tree with no card() in between; assertions after a card()
  // read the tree synchronously.
  async function toPreviewTree() {
    hfApi.compileHyperframes.mockResolvedValue({ compile_id: 'abcdef0123456789', preview_identity: 'd', manifest: MANIFEST });
    hfApi.renderHyperframes.mockResolvedValue({ render_id: 'rid-9', state: 'rendering' });
    card({ data: data('j1', 0), activeSelectionKey: 'j1:0' });
    await flush();
    let tree = card({ data: data('j1', 0), activeSelectionKey: 'j1:0' });
    await findByTestId(tree, 'hf-apply').props.onClick();
    await flush();
    tree = card({ data: data('j1', 0), activeSelectionKey: 'j1:0' });
    await findByTestId(tree, 'hf-preview').props.onClick();
    await flush();
    return card({ data: data('j1', 0), activeSelectionKey: 'j1:0' });
  }
  async function toRendering() {
    const previewTree = await toPreviewTree();
    await findByTestId(previewTree, 'hf-render').props.onClick();
    await flush();
    return card({ data: data('j1', 0), activeSelectionKey: 'j1:0' });
  }

  it('cancel shows rendered when the backend already rendered', async () => {
    const tree = await toRendering();
    expect(findByTestId(tree, 'hf-cancel')).toBeTruthy();
    hfApi.cancelHyperframesRender.mockResolvedValue({ ok: true });
    hfApi.hyperframesRenderStatus.mockResolvedValue({ state: 'rendered', output_url: '/exports/r9.mp4' });
    await findByTestId(tree, 'hf-cancel').props.onClick();
    await flush();
    const after = card({ data: data('j1', 0), activeSelectionKey: 'j1:0' });
    const t = textOf(after);
    expect(hfApi.cancelHyperframesRender).toHaveBeenCalledWith('rid-9');
    expect(hfApi.hyperframesRenderStatus).toHaveBeenCalledWith('rid-9');
    expect(t).toMatch(/Export rendered/);
    expect(findByTestId(after, 'hf-output')).toBeTruthy();
    expect(t).not.toMatch(/nothing was published/);
  });

  it('cancel shows cancelled when the backend acknowledges cancellation', async () => {
    const tree = await toRendering();
    hfApi.cancelHyperframesRender.mockResolvedValue({ ok: true });
    hfApi.hyperframesRenderStatus.mockResolvedValue({ state: 'cancelled' });
    await findByTestId(tree, 'hf-cancel').props.onClick();
    await flush();
    const after = card({ data: data('j1', 0), activeSelectionKey: 'j1:0' });
    expect(textOf(after)).toMatch(/Render cancelled/);
    expect(textOf(after)).not.toMatch(/Export rendered/);
  });

  it('a failed cancel stays honest instead of claiming cancelled', async () => {
    const tree = await toRendering();
    hfApi.cancelHyperframesRender.mockRejectedValue(new Error('worker unreachable'));
    await findByTestId(tree, 'hf-cancel').props.onClick();
    await flush();
    const after = card({ data: data('j1', 0), activeSelectionKey: 'j1:0' });
    const t = textOf(after);
    expect(t).toMatch(/Cancel failed/);
    expect(t).not.toMatch(/nothing was published/);
    expect(t).not.toMatch(/Render cancelled/); // never claims cancelled: the cancel did not go through
    expect(hfApi.cancelHyperframesRender).toHaveBeenCalledWith('rid-9');
  });
});

describe('coordinator review02 remaining production contracts', () => {
  async function previewCard() {
    const p={data:data('j1',0),activeSelectionKey:'j1:0'};
    hfApi.compileHyperframes.mockResolvedValue({compile_id:'abcdef0123456789',preview_identity:'same',manifest:{duration:5,render:{width:90,height:160},project_version:2}});
    card(p); await flush();
    await findByTestId(card(p),'hf-apply').props.onClick();
    await findByTestId(card(p),'hf-preview').props.onClick();
    return {p,tree:card(p)};
  }
  it('keeps polling through rendering until the backend reaches rendered', async () => {
    vi.useFakeTimers();
    try {
      const {p,tree}=await previewCard();
      hfApi.renderHyperframes.mockResolvedValue({render_id:'r',state:'rendering'});
      hfApi.hyperframesRenderStatus.mockResolvedValueOnce({state:'rendering'}).mockResolvedValueOnce({state:'rendered',output_url:'/final'});
      await findByTestId(tree,'hf-render').props.onClick();
      await vi.advanceTimersByTimeAsync(2200);
      expect(hfApi.hyperframesRenderStatus).toHaveBeenCalledTimes(2);
      expect(textOf(card(p))).toMatch(/Export rendered/);
    } finally { vi.clearAllTimers(); vi.useRealTimers(); }
  });
  it('ignores preview messages from a different iframe/window', async () => {
    const {tree}=await previewCard();
    const element=walk(tree).find(n=>n.type?.name==='HfPreview');
    hooks.values=[]; hooks.index=0;
    const listeners=[];
    vi.stubGlobal('window',{addEventListener:(name,fn)=>{if(name==='message') listeners.push(fn);},removeEventListener:()=>{}});
    try {
      element.type(element.props);
      const ownedWindow={postMessage:vi.fn()};
      hooks.values[0].current={contentWindow:ownedWindow};
      // Wrong window: ignored, play button never appears.
      for(const fn of listeners) fn({source:{},origin:'null',data:{hf:1,type:'ready'}});
      hooks.index=0;
      const updated=element.type(element.props);
      expect(findByTestId(updated,'hf-preview-play')).toBeNull();
      // Owned window: accepted, play button appears. (load-bearing: the
      // source check must not reject the real iframe.)
      for(const fn of listeners) fn({source:ownedWindow,origin:'null',data:{hf:1,type:'ready'}});
      await flush();
      hooks.index=0;
      const updated2=element.type(element.props);
      expect(findByTestId(updated2,'hf-preview-play')).not.toBeNull();
      // Stale instance id: a message from the owned window but with a
      // different hfiid must not drive this preview.
      const hfiid = element.props.hfiid;
      expect(hfiid).toMatch(/^hf-/);
      for(const fn of listeners) fn({source:ownedWindow,origin:'null',data:{hf:1,hfiid:'hf-stale',type:'timeupdate',t:1}});
      await flush();
      // t stays 0 (the stale timeupdate was ignored).
      expect(hooks.values[1]).toBe(0);
    } finally { vi.unstubAllGlobals(); vi.stubGlobal('React',React); }
  });
});

describe('coordinator acknowledged finishing contract', () => {
  it('cannot compile/export a changed template without acknowledging its save', async () => {
    const p={data:data('j1',0),activeSelectionKey:'j1:0'};
    card(p); await flush();
    await findByTestId(card(p),'hf-apply').props.onClick();
    const applied=card(p);
    findByTestId(applied,'hf-template').props.onChange({target:{value:'emphasized-word'}});
    const changed=card(p),preview=findByTestId(changed,'hf-preview');
    if(preview && !preview.props.disabled) await preview.props.onClick();
    expect(hfApi.compileHyperframes).not.toHaveBeenCalled();
  });
});
