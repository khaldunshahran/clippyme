/**
 * UI-01 — App workspace layout mounted host tests.
 *
 * Coordinator-probe style: the minimal hook harness drives the REAL
 * AppShell layout with the REAL Composer and a stubbed EditorView (its
 * internals are covered in ui01Workspace.test.js). Heavy chat children are
 * stubbed; what matters here is the layout contract:
 *
 *  1. Narrow-view (Chat/Editor) switching flips CSS visibility only — both
 *     columns stay mounted, unsent composer text and the controller
 *     identity survive the switch.
 *  2. Chat-rail collapse is CSS-only; the workspace keeps its controller.
 *  3. The workspace's "Ask Nugget to edit" handler focuses the REAL
 *     composer and returns narrow screens to the chat view.
 *
 * Load-bearing mutation:
 *  M4: chat column unmounts on narrow switch (state preservation broken)
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
const editorPropsCap = vi.hoisted(() => ({ current: null }));
const wsMock = vi.hoisted(() => ({ current: null }));
const chatMock = vi.hoisted(() => ({ current: null }));
const focusMock = vi.hoisted(() => ({ fn: null }));

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
  scheduleClipPost: vi.fn(),
  getZernioAccounts: vi.fn(),
  zernioTargets: vi.fn(),
  getJobClips: vi.fn(),
  normClip: (c) => c,
}));
vi.mock('../api/sessionReset.js', () => ({
  onSessionReset: () => () => {},
}));
vi.mock('./useWorkspaceController.js', () => ({
  useWorkspaceController: () => wsMock.current,
}));
vi.mock('./useChat.js', () => ({
  useChat: () => chatMock.current,
}));

// Heavy children become stubs; Composer stays REAL; EditorView is stubbed
// here (its internals are covered in ui01Workspace.test.js) but captures
// the props the App passes it.
vi.mock('./Sidebar.jsx', () => ({ default: () => null }));
vi.mock('./WelcomeView.jsx', () => ({ default: () => null }));
vi.mock('./ChatThread.jsx', () => ({ default: () => null }));
vi.mock('./SettingsPopup.jsx', () => ({ default: () => null }));
vi.mock('./ClipGrid.jsx', () => ({ default: () => null }));
vi.mock('./ClipPopup.jsx', () => ({ default: () => null }));
vi.mock('./SelectionBar.jsx', () => ({ default: () => null }));
vi.mock('./EditorView.jsx', () => ({
  default: (props) => {
    editorPropsCap.current = props;
    return null;
  },
}));

import { AppShell } from '../App.jsx';
import Composer from './Composer.jsx';

function mockController() {
  return {
    getState: () => ({
      status: 'ready',
      project: { version: 3 },
      dirty: false, saving: false, exporting: false,
      versions: [], versionsLoading: false,
    }),
    subscribe: () => () => {},
    applyEdit: async () => null,
    undo: async () => {},
    redo: async () => {},
    dispose: () => {},
    load: async () => {},
  };
}

// --- expander that also records mounted function components.
function expand(el, depth = 0, seen = []) {
  if (depth > 40 || el == null || typeof el !== 'object') return el;
  if (Array.isArray(el)) return el.map((e) => expand(e, depth + 1, seen));
  const { type, props } = el;
  if (typeof type === 'function') {
    seen.push(el);
    return expand(type(props || {}), depth + 1, seen);
  }
  if (props && 'children' in props && props.children !== undefined) {
    return { ...el, props: { ...props, children: expand(props.children, depth + 1, seen) } };
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
const buttonText = (b) => JSON.stringify(b.props.children);
const buttonByText = (tree, text) =>
  buttons(tree).find((b) => buttonText(b).includes(`"${text}"`));

function render() {
  hooks.index = 0;
  const seen = [];
  const tree = expand(AppShell({}), 0, seen);
  return { tree, seen };
}
const seenComponent = (seen, comp) => seen.find((el) => el.type === comp);

function stubChat() {
  const thread = {
    id: 't1',
    title: 'Test job',
    jobId: 'j1',
    phase: 'idle',
    messages: [{ id: 'm1', role: 'user', text: 'hi' }],
    clips: [{ index: 0, title: 'Clip 1', raw: null }],
    settings: {},
  };
  chatMock.current = {
    threads: [thread],
    active: thread,
    activeId: 't1',
    settingsOpen: false,
    setSettingsOpen: () => {},
    gridOpen: false,
    setGridOpen: () => {},
    popupClip: null,
    setPopupClip: () => {},
    selectedClip: { threadId: 't1', jobId: 'j1', clipIndex: 0 },
    selectClip: () => {},
    clearSelection: () => {},
    workspaceRef: { current: null },
    createThread: () => 't2',
    selectThread: () => {},
    deleteThread: () => {},
    sendMessage: async () => {},
    applyProposal: () => {},
    discardProposal: () => {},
    confirmSettings: () => {},
    proceedToSettings: () => {},
    patchThread: () => {},
    startBulkEdit: () => {},
    DEFAULT_SETTINGS: {},
  };
}

beforeEach(() => {
  hooks.values = [];
  hooks.index = 0;
  editorPropsCap.current = null;
  wsMock.current = null;
  focusMock.fn = vi.fn();
  stubChat();
  const controller = mockController();
  wsMock.current = { controller, snap: controller.getState(), resetKey: 0, toast: null };
});

describe('UI-01 narrow-view switching preserves mounted state', () => {
  it('Chat/Editor toggle flips visibility only; composer text + controller identity survive', () => {
    const { tree, seen } = render();
    const controller = wsMock.current.controller;

    const row = findAll(tree, byClass('nc-workspace-row'))[0];
    expect(row.props.className).toContain('has-selection');
    expect(row.props['data-mview']).toBe('chat');
    expect(findAll(tree, byClass('nc-chat-col')).length).toBe(1);
    expect(findAll(tree, byClass('nc-workspace-panel')).length).toBe(1);

    // Type unsent text into the real composer.
    const ta = findAll(tree, (el) => el.type === 'textarea')[0];
    expect(ta).toBeTruthy();
    ta.props.onChange({ target: { value: 'make the hook punchier' } });
    expect(seenComponent(seen, Composer)).toBeTruthy();
    const firstController = editorPropsCap.current.controller;
    expect(firstController).toBe(controller);

    // Switch to the editor view (narrow toggle).
    buttonByText(tree, 'Editor').props.onClick();

    const { tree: tree2, seen: seen2 } = render();
    expect(findAll(tree2, byClass('nc-workspace-row'))[0].props['data-mview']).toBe('editor');
    // Still mounted — nothing unmounted, nothing reset.
    expect(findAll(tree2, byClass('nc-chat-col')).length).toBe(1);
    expect(findAll(tree2, byClass('nc-workspace-panel')).length).toBe(1);
    expect(findAll(tree2, (el) => el.type === 'textarea')[0].props.value).toBe('make the hook punchier');
    expect(seenComponent(seen2, Composer)).toBeTruthy();
    expect(editorPropsCap.current.controller).toBe(controller);
    expect(editorPropsCap.current.controller).toBe(firstController);
  });

  it('chat-rail collapse is CSS-only; the workspace keeps its controller', () => {
    const { tree } = render();
    const controller = wsMock.current.controller;
    const toggle = editorPropsCap.current.onToggleChat;
    expect(typeof toggle).toBe('function');
    toggle();

    const { tree: tree2 } = render();
    const row = findAll(tree2, byClass('nc-workspace-row'))[0];
    expect(row.props.className).toContain('chat-collapsed');
    // CSS hides the rail; React keeps it mounted.
    expect(findAll(tree2, byClass('nc-chat-col')).length).toBe(1);
    expect(findAll(tree2, byClass('nc-workspace-panel')).length).toBe(1);
    expect(editorPropsCap.current.controller).toBe(controller);
    expect(editorPropsCap.current.chatCollapsed).toBe(true);
  });

  it('"Ask Nugget to edit" focuses the real composer and returns to chat view', async () => {
    let { tree, seen } = render();
    // Go to editor view first (narrow).
    buttonByText(tree, 'Editor').props.onClick();
    ({ tree, seen } = render());
    expect(findAll(tree, byClass('nc-workspace-row'))[0].props['data-mview']).toBe('editor');

    // The composer ref object travels on the real Composer's inputRef prop.
    const composerEl = seenComponent(seen, Composer);
    expect(composerEl).toBeTruthy();
    composerEl.props.inputRef.current = { focus: focusMock.fn };

    // Invoke the handler the App passes to the workspace.
    const onAskNugget = editorPropsCap.current.onAskNugget;
    expect(typeof onAskNugget).toBe('function');
    onAskNugget();
    await new Promise((r) => setTimeout(r, 20));
    expect(focusMock.fn).toHaveBeenCalledTimes(1);

    ({ tree } = render());
    expect(findAll(tree, byClass('nc-workspace-row'))[0].props['data-mview']).toBe('chat');
  });
});
