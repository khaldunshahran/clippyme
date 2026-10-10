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
 *     existing controller authority; expectedVersion concurrency guard).
 *  2. default-OFF honesty: legacy renderer named as the unchanged default.
 *  8. delayed async responses can never replace another selection's
 *     preview (stale selection -> 'stale' state, no commit).
 *
 * Load-bearing mutations (run manually, each must fail, then re-run green):
 *  M1: HyperframesCard drops the isLive guard in commit (stale commit)
 *  M2: apply drops expectedVersion (stale apply)
 *  M3: HyperframesMsg stops mapping scope -> jobId/index (identity loss)
 */
import { describe, expect, it, vi, beforeEach } from 'vitest';
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
  hyperframesCompositionUrl: vi.fn((j, c) => `/api/hyperframes/composition/${j}/${c}/index.html`),
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
  realApi.getClipProject.mockResolvedValue({ project: { version: 1 }, version: 1 });
  realApi.saveClipProject.mockResolvedValue({ version: 2, project: {} });
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
});

describe('HyperframesCard preview/export identity (contract 5)', () => {
  const MANIFEST = {
    project_version: 2,
    template: { id: 'clean-captions', version: 1 },
    renderer: { engine: '0.8.145' },
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
    expect(iframe.props.src).toBe('/api/hyperframes/composition/j1/abcdef0123456789/index.html');
    expect(iframe.props.sandbox).toMatch(/allow-scripts/);
    // no vendored player element: the player cannot drive the __hf protocol
    expect(walk(tree).filter((n) => n.type === 'hyperframes-player')).toEqual([]);
    const id = findByTestId(tree, 'hf-preview-identity');
    expect(textOf(id)).toMatch(/digest-abc-1/); // 12-char digest prefix
    expect(textOf(id)).toMatch(/clean-captions/);
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
