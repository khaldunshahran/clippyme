/**
 * UI-01 — EditorView mounted host tests.
 *
 * Coordinator-probe style: the minimal hook harness drives the REAL
 * EditorView (and its nested ClipEditorView) with no DOM. A tiny
 * element-tree expander "mounts" function components so state, props and
 * handlers are exercised for real:
 *
 *  1. Hosted vs standalone chrome — hosted (controller prop) renders the
 *     single workspace header + "Ask Nugget to edit" and NO floating
 *     copilot; standalone (no controller prop) keeps the floating copilot.
 *  2. Transcript rail toggle collapses/expands; the editor is untouched.
 *  3. "Ask Nugget to edit" is an accessible button wired to onAskNugget.
 *  4. The single workspace header shows the actual pending/saved state and
 *     the draft-vs-exact preview distinction; Export still forwards to the
 *     controller (proposal/apply/export path intact).
 *
 * Load-bearing mutations (each must fail, then the suite re-runs green):
 *  M1: EditorView renders CopilotPanel when hosted (second AI composer)
 *  M2: transcript toggle stops collapsing the rail
 *  M3: header Export stops calling controller.exportProject
 */
import { describe, expect, it, vi, beforeEach } from 'vitest';
import * as React from 'react';

// Vitest compiles JSX with the classic runtime (React.createElement) and
// the components under test never import React (the app relies on the
// bundler). Expose it globally so executed JSX resolves.
vi.stubGlobal('React', React);

// --- minimal hook harness (persistent useState/useRef, identity
// useMemo/useCallback, immediate best-effort useEffect) ---
const hooks = vi.hoisted(() => ({ values: [], index: 0 }));
const wsMock = vi.hoisted(() => ({ current: null }));

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
      } catch { /* node env: window/document absent; effects are best-effort */ }
    },
  };
});

// API transport mocks: components under test must never call the network.
vi.mock('../api/realApi.js', () => ({
  getClipProject: vi.fn(),
  saveClipProject: vi.fn(),
  composeProject: vi.fn(),
  backfillClipProjects: vi.fn(),
  listClipProjectVersions: vi.fn(),
  editClipAI: vi.fn(),
  uploadAudio: vi.fn(),
  safeResolveUrl: (u) => u,
  clipPreviewSrc: () => '',
  clipVideoSrc: () => '',
}));
vi.mock('../api/chatApi.js', () => ({
  editAiPatch: vi.fn(),
}));
vi.mock('../api/sessionReset.js', () => ({
  onSessionReset: () => () => {},
}));
vi.mock('./useWorkspaceController.js', () => ({
  useWorkspaceController: (opts) =>
    wsMock.current && !opts?.disabled
      ? wsMock.current
      : { controller: null, snap: null, resetKey: 0, toast: null, pushToast: () => {} },
}));

import EditorView from './EditorView.jsx';

const clone = (p) => JSON.parse(JSON.stringify(p));

function makeProject(version = 3) {
  return {
    origin: 'auto',
    version,
    idempotency_key: `key-v${version}`,
    source: { duration: 30, width: 1920, height: 1080 },
    segments: [{ id: 'seg0', start: 0, end: 30, crop: { x: 0, y: 0, w: 608, h: 1080 } }],
    captions: {
      style: 'classic_white',
      position: 'bottom',
      words: [
        { id: 'w0', w: 'hello', start: 1, end: 1.5 },
        { id: 'w1', w: 'world', start: 1.6, end: 2.1 },
      ],
      edits: {},
    },
    overlays: [],
    audio: [],
  };
}

function mockController(snapOverrides = {}) {
  const calls = { exportProject: 0, saveDraft: 0, applyEdit: 0 };
  const snap = {
    status: 'ready',
    loadError: '',
    project: makeProject(3),
    dirty: false,
    saving: false,
    exporting: false,
    versions: [{ version: 1 }, { version: 2 }, { version: 3 }],
    versionsLoading: false,
    ...snapOverrides,
  };
  const controller = {
    _calls: calls,
    getState: () => clone(snap),
    subscribe: () => () => {},
    applyEdit: async () => { calls.applyEdit++; return null; },
    saveDraft: async () => { calls.saveDraft++; return { version: 4 }; },
    exportProject: async () => {
      calls.exportProject++;
      return { version: 3, composed_url: '/c.mp4', deduped: false };
    },
    restoreVersion: async () => null,
    resetToAiCut: async () => null,
    listVersions: () => {},
    aiTrim: async () => ({ drop_ranges: [], explanation: '' }),
    uploadAudio: async () => ({ file: 'x.mp3' }),
    undo: async () => {},
    redo: async () => {},
    dispose: () => {},
    load: async () => {},
  };
  return controller;
}

// --- tiny element-tree expander: "mounts" function components so hooks run.
function expand(el, depth = 0) {
  if (depth > 40 || el == null || typeof el !== 'object') return el;
  if (Array.isArray(el)) return el.map((e) => expand(e, depth + 1));
  const { type, props } = el;
  if (typeof type === 'function') {
    return expand(type(props || {}), depth + 1);
  }
  if (props && 'children' in props && props.children !== undefined) {
    return { ...el, props: { ...props, children: expand(props.children, depth + 1) } };
  }
  return el;
}

function findAll(el, pred, acc = []) {
  if (el == null || typeof el !== 'object') return acc;
  if (Array.isArray(el)) {
    el.forEach((e) => findAll(e, pred, acc));
    return acc;
  }
  if (typeof el.type === 'string' && pred(el)) acc.push(el);
  const children = el.props?.children;
  if (children !== undefined) findAll(children, pred, acc);
  return acc;
}

const byClass = (substr) => (el) =>
  typeof el.props?.className === 'string' && el.props.className.includes(substr);
const buttons = (tree) => findAll(tree, (el) => el.type === 'button');
// Btn renders <button><Icon/><span>{children}</span></button>; match text at
// any depth inside the button.
const buttonText = (b) => JSON.stringify(b.props.children);
const buttonByText = (tree, text) =>
  buttons(tree).find((b) => buttonText(b).includes(`"${text}"`));
const textOf = (el) => {
  const c = el.props.children;
  if (typeof c === 'string') return c;
  if (Array.isArray(c)) return c.map((x) => (typeof x === 'string' ? x : '')).join('');
  return '';
};

function renderEditor(props) {
  hooks.index = 0;
  return expand(EditorView(props));
}

beforeEach(() => {
  hooks.values = [];
  hooks.index = 0;
  wsMock.current = null;
});

describe('UI-01 hosted vs standalone chrome', () => {
  const clip = { title: 'Clip 1', raw: null };

  it('hosted: single workspace header, Ask Nugget action, no floating copilot', () => {
    const controller = mockController();
    const onAskNugget = vi.fn();
    const tree = renderEditor({
      jobId: 'j1', clipIndex: 0, clip, onBack: () => {},
      controller, snap: controller.getState(), resetKey: 0, toast: null,
      onAskNugget,
    });
    // The old EditorView header ("Editing — …") is gone: exactly one Back.
    const backs = buttons(tree).filter((b) => textOf(b).includes('Back') || buttonText(b).includes('Back'));
    expect(backs.length).toBe(1);
    // Contextual AI entry into the main chat composer.
    const ask = buttons(tree).find((b) => b.props['aria-label'] === 'Ask Nugget to edit — focuses the chat composer');
    expect(ask).toBeTruthy();
    expect(ask.type).toBe('button');
    ask.props.onClick();
    expect(onAskNugget).toHaveBeenCalledTimes(1);
    // No floating second AI composer in hosted mode.
    const copilot = buttons(tree).find((b) => b.props['aria-label'] === 'Open AI copilot');
    expect(copilot).toBeFalsy();
    expect(buttonText({ props: { children: tree } }).includes('AI copilot')).toBe(false);
    // Transcript rail toggle present and pressed (open by default in node).
    const toggle = buttons(tree).find((b) => b.props['aria-controls'] === 'nc-transcript-panel');
    expect(toggle).toBeTruthy();
    expect(toggle.props['aria-pressed']).toBe(true);
    expect(findAll(tree, byClass('nc-ws-transcript')).length).toBe(1);
  });

  it('standalone (no controller prop): floating copilot is preserved', () => {
    const controller = mockController();
    wsMock.current = {
      controller,
      snap: controller.getState(),
      resetKey: 0,
      toast: null,
      pushToast: () => {},
    };
    const tree = renderEditor({ jobId: 'j1', clipIndex: 0, clip: { title: 'Clip 1', raw: null }, onBack: () => {} });
    // No "Ask Nugget" (no host composer to focus).
    const ask = buttons(tree).find((b) => (b.props['aria-label'] || '').startsWith('Ask Nugget to edit'));
    expect(ask).toBeFalsy();
    // Floating copilot kept.
    const html = buttonText({ props: { children: tree } });
    expect(html.includes('AI copilot')).toBe(true);
  });

  it('transcript rail toggle collapses and re-expands without losing the editor', () => {
    const controller = mockController();
    const props = {
      jobId: 'j1', clipIndex: 0, clip: { title: 'Clip 1', raw: null }, onBack: () => {},
      controller, snap: controller.getState(), resetKey: 0, toast: null, onAskNugget: () => {},
    };
    let tree = renderEditor(props);
    expect(findAll(tree, byClass('nc-ws-transcript')).length).toBe(1);
    const toggle = buttons(tree).find((b) => b.props['aria-controls'] === 'nc-transcript-panel');
    toggle.props.onClick(); // collapse
    tree = renderEditor(props);
    expect(findAll(tree, byClass('nc-ws-transcript')).length).toBe(0);
    expect(buttons(tree).find((b) => b.props['aria-controls'] === 'nc-transcript-panel').props['aria-pressed']).toBe(false);
    // The editor (single header + export) is untouched by the toggle.
    expect(buttons(tree).filter((b) => buttonText(b).includes('Back')).length).toBe(1);
    expect(buttonByText(tree, 'Export')).toBeTruthy();
    const toggle2 = buttons(tree).find((b) => b.props['aria-controls'] === 'nc-transcript-panel');
    toggle2.props.onClick(); // re-expand
    tree = renderEditor(props);
    expect(findAll(tree, byClass('nc-ws-transcript')).length).toBe(1);
  });
});

describe('UI-01 single workspace header state', () => {
  const clip = { title: 'Clip 1', raw: null };
  const base = () => {
    const controller = mockController();
    return {
      controller,
      props: {
        jobId: 'j1', clipIndex: 0, clip, onBack: () => {},
        controller, snap: controller.getState(), resetKey: 0, toast: null, onAskNugget: () => {},
      },
    };
  };

  it('shows acknowledged version and draft preview when clean', () => {
    const { props } = base();
    const tree = renderEditor(props);
    const html = buttonText({ props: { children: tree } });
    // Version badge: "v" + 3 render as separate children.
    expect(html).toContain('Acknowledged project version');
    expect(html).toContain('Draft preview');
    expect(html).toContain('Saved');
  });

  it('shows pending save state; export stays guarded while saving', async () => {
    const controller = mockController({ saving: true });
    const tree = renderEditor({
      jobId: 'j1', clipIndex: 0, clip, onBack: () => {},
      controller, snap: controller.getState(), resetKey: 0, toast: null, onAskNugget: () => {},
    });
    const html = buttonText({ props: { children: tree } });
    expect(html).toContain('Saving…');
    // Export action still present, but correctly guarded while a save is in
    // flight — no second export is issued.
    const exportBtn = buttonByText(tree, 'Export');
    expect(exportBtn).toBeTruthy();
    await exportBtn.props.onClick();
    expect(controller._calls.exportProject).toBe(0);
  });

  it('header Export forwards through the controller (apply/export path intact)', async () => {
    const { controller, props } = base();
    const tree = renderEditor(props);
    const exportBtn = buttonByText(tree, 'Export');
    expect(exportBtn).toBeTruthy();
    await exportBtn.props.onClick();
    expect(controller._calls.exportProject).toBe(1);
    // After export the preview badge names the exact render.
    const tree2 = renderEditor(props);
    expect(buttonText({ props: { children: tree2 } })).toContain('Exact v3 render');
  });
});
