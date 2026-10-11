/**
 * UI-01 review-02 — durable regressions for preview identity, timebase
 * isolation, and draft-source selection.
 *
 * Coordinator-probe style (same minimal hook harness as
 * ui01Workspace.test.js / coordinatorUi01Review02.test.js): the REAL
 * ClipEditorView is driven with no DOM; state, classification and
 * handlers are exercised for real.
 *
 * Contracts pinned here:
 *  1. Exact-render classification binds to the server-ACKNOWLEDGED
 *     version + dirty state + content identity (idempotency key) —
 *     never to the draft's content version (a restore can roll the
 *     draft back while the server sits newer).
 *  2. Rendered playback (exact or historical) is never looped through
 *     the current draft's segment bounds, and the rendered playhead
 *     cannot authorize a source-coordinate split. Draft looping/split
 *     still work in draft mode (positive controls — including a
 *     multi-segment project with a nonzero source offset).
 *  3. The draft player selects the clip's RAW source (video_url).
 *     The production preview helper prefers composed_video_url; the
 *     mock below implements exactly that production policy, so if the
 *     editor ever regresses to it, test 3 fails. A composed-only clip
 *     is presented honestly as a composed file.
 *
 * Load-bearing mutations (each must fail, then the suite re-runs green):
 *  MA: classification compares previewKind.version to the draft content
 *      version (savedVersion) again -> classification tests fail.
 *  MB: the artifactPlayback early-return is removed from onTimeUpdate
 *      -> timebase isolation tests fail.
 *  MD: draftPreviewSource prefers composed_video_url -> source tests fail.
 */
import { describe, expect, it, vi, beforeEach } from 'vitest';
import * as React from 'react';

vi.stubGlobal('React', React);

const hooks = vi.hoisted(() => ({ values: [], index: 0 }));

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
      } catch { /* node env: effects are best-effort */ }
    },
  };
});

// API transport mocks: no network. NOTE clipPreviewSrc implements the
// PRODUCTION selection policy (previewUrl -> composed -> raw) on
// purpose: the editor must not use it for the draft player.
vi.mock('../api/realApi.js', () => ({
  getClipProject: vi.fn(),
  saveClipProject: vi.fn(),
  composeProject: vi.fn(),
  backfillClipProjects: vi.fn(),
  listClipProjectVersions: vi.fn(),
  editClipAI: vi.fn(),
  uploadAudio: vi.fn(),
  safeResolveUrl: (u) => u,
  clipPreviewSrc: (clip) => clip?.composed_video_url || clip?.video_url || '',
  clipVideoSrc: (clip) => clip?.video_url || '',
}));
vi.mock('../api/chatApi.js', () => ({
  editAiPatch: vi.fn(),
}));
vi.mock('../api/sessionReset.js', () => ({
  onSessionReset: () => () => {},
}));

import { ClipEditorView, draftPreviewSource } from '../editor/clipEditor.jsx';

const clone = (p) => JSON.parse(JSON.stringify(p));

function makeProject(version = 3, segments) {
  return {
    origin: 'auto',
    version,
    idempotency_key: `key-v${version}`,
    source: { duration: 30, width: 1920, height: 1080 },
    segments: segments || [{ id: 'seg0', start: 0, end: 30, crop: { x: 0, y: 0, w: 608, h: 1080 } }],
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
    ackVersion: 3,
    versions: [{ version: 1 }, { version: 2 }, { version: 3 }],
    versionsLoading: false,
    ...snapOverrides,
  };
  const controller = {
    _calls: calls,
    _snap: snap,
    getState: () => clone(snap),
    subscribe: () => () => {},
    applyEdit: async () => { calls.applyEdit++; return null; },
    saveDraft: async () => { calls.saveDraft++; return { version: 4 }; },
    exportProject: async () => {
      calls.exportProject++;
      return {
        version: 3, composed_url: '/c.mp4', deduped: false,
        identity: { version: 3, idempotency_key: 'key-v3' },
      };
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

// --- tiny element-tree expander (same as the sibling suites) ----------
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

const buttons = (tree) => findAll(tree, (el) => el.type === 'button');
const buttonText = (b) => JSON.stringify(b.props.children);
const buttonByText = (tree, text) =>
  buttons(tree).find((b) => buttonText(b).includes(`"${text}"`));
const textOf = (el) => {
  const c = el.props.children;
  if (typeof c === 'string') return c;
  if (Array.isArray(c)) return c.map((x) => (typeof x === 'string' ? x : '')).join('');
  return '';
};
// The split button's label is dynamic ("Split at 0:10.0"): match the stem.
const splitButton = (tree) =>
  buttons(tree).find((b) => buttonText(b).includes('Split at'));
const previewBadge = (tree) =>
  findAll(tree, (e) => e.props.title === 'What the preview player is showing')[0];

function renderDirect(controller, clip) {
  hooks.index = 0;
  return expand(ClipEditorView({
    jobId: 'j1', clipIndex: 0, clip: clip || { title: 'Clip 1' },
    controller, onBack: () => {},
  }));
}

async function exportAndRender(controller, clip) {
  let tree = renderDirect(controller, clip);
  await buttonByText(tree, 'Export').props.onClick();
  tree = renderDirect(controller, clip);
  return tree;
}

beforeEach(() => {
  hooks.values = [];
  hooks.index = 0;
});

describe('review-02: draft source selection', () => {
  it('draftPreviewSource prefers the raw video_url over the composed file', () => {
    expect(draftPreviewSource({ video_url: 'raw.mp4', composed_video_url: 'comp.mp4' }))
      .toEqual({ src: 'raw.mp4', composed: false });
  });

  it('draftPreviewSource marks a composed-only clip as composed (honest fallback)', () => {
    expect(draftPreviewSource({ video_url: null, composed_video_url: 'comp.mp4' }))
      .toEqual({ src: 'comp.mp4', composed: true });
    expect(draftPreviewSource({ composed_video_url: 'comp.mp4' }))
      .toEqual({ src: 'comp.mp4', composed: true });
  });

  it('draftPreviewSource returns empty for a clip with no media', () => {
    expect(draftPreviewSource({})).toEqual({ src: '', composed: false });
    expect(draftPreviewSource(null)).toEqual({ src: '', composed: false });
  });

  it('mounted editor: draft player src is the raw source when both URLs exist', () => {
    const controller = mockController();
    const clip = { title: 'Clip 1', video_url: 'raw.mp4', composed_video_url: 'comp.mp4' };
    renderDirect(controller, clip);
    const tree = renderDirect(controller, clip);
    const video = findAll(tree, (e) => e.type === 'video')[0];
    expect(video.props.src).toBe('raw.mp4');
    expect(textOf(previewBadge(tree))).toBe('Draft preview');
  });

  it('mounted editor: composed-only clip is labeled a composed file, not a clean draft', () => {
    const controller = mockController();
    const clip = { title: 'Clip 1', composed_video_url: 'comp.mp4' };
    renderDirect(controller, clip);
    const tree = renderDirect(controller, clip);
    const video = findAll(tree, (e) => e.type === 'video')[0];
    expect(video.props.src).toBe('comp.mp4');
    expect(textOf(previewBadge(tree))).toContain('Composed file');
  });
});

describe('review-02: exact-render classification binds to ack + dirty + content identity', () => {
  it('clean export at the acknowledged version is Exact', async () => {
    const controller = mockController();
    const tree = await exportAndRender(controller);
    expect(textOf(previewBadge(tree))).toBe('Exact v3 render');
  });

  it('a dirty draft makes the same render Historical even at the same version', async () => {
    const controller = mockController();
    await exportAndRender(controller);
    controller._snap.dirty = true;
    const tree = renderDirect(controller);
    expect(textOf(previewBadge(tree))).toBe('Historical v3 render');
  });

  it('restored draft v3 with server ack v7 is Historical (the review-02 probe)', async () => {
    const controller = mockController();
    await exportAndRender(controller);
    controller._snap.ackVersion = 7;
    controller._snap.project = makeProject(3);
    controller._snap.dirty = true;
    const tree = renderDirect(controller);
    expect(textOf(previewBadge(tree))).toBe('Historical v3 render');
  });

  it('a changed content identity (idempotency key) makes the render Historical', async () => {
    const controller = mockController();
    await exportAndRender(controller);
    controller._snap.project = { ...makeProject(3), idempotency_key: 'key-other' };
    const tree = renderDirect(controller);
    expect(textOf(previewBadge(tree))).toBe('Historical v3 render');
  });
});

describe('review-02: rendered playback timebase isolation', () => {
  const offsetSegments = [
    { id: 'segA', start: 5, end: 8, crop: { x: 0, y: 0, w: 608, h: 1080 } },
    { id: 'segB', start: 10, end: 14, crop: { x: 0, y: 0, w: 608, h: 1080 } },
  ];

  it('draft mode control: the selected-segment loop still rewinds the draft player', () => {
    const controller = mockController({ project: makeProject(3, offsetSegments) });
    const tree = renderDirect(controller);
    const video = findAll(tree, (e) => e.type === 'video')[0];
    const player = { currentTime: 8.4, paused: false };
    video.ref.current = player;
    video.props.onTimeUpdate();
    expect(player.currentTime).toBe(5);
  });

  it('render mode: playback past the draft segment end is NOT rewound (nonzero offset, multi-segment)', async () => {
    const controller = mockController({ project: makeProject(3, offsetSegments) });
    const tree = await exportAndRender(controller);
    expect(textOf(previewBadge(tree))).toBe('Exact v3 render');
    const video = findAll(tree, (e) => e.type === 'video')[0];
    const player = { currentTime: 8.4, paused: false };
    video.ref.current = player;
    video.props.onTimeUpdate();
    expect(player.currentTime).toBe(8.4);
    player.currentTime = 12;
    video.props.onTimeUpdate();
    expect(player.currentTime).toBe(12);
  });

  it('render mode: the rendered playhead cannot authorize a split', async () => {
    const controller = mockController();
    const tree = await exportAndRender(controller);
    const video = findAll(tree, (e) => e.type === 'video')[0];
    const player = { currentTime: 10, paused: true };
    video.ref.current = player;
    video.props.onTimeUpdate();
    const tree2 = renderDirect(controller);
    await splitButton(tree2).props.onClick();
    expect(controller._calls.applyEdit).toBe(0);
  });

  it('draft mode control: split at the playhead still applies', async () => {
    const controller = mockController();
    const tree = renderDirect(controller);
    const video = findAll(tree, (e) => e.type === 'video')[0];
    const player = { currentTime: 10, paused: true };
    video.ref.current = player;
    video.props.onTimeUpdate();
    const tree2 = renderDirect(controller);
    await splitButton(tree2).props.onClick();
    expect(controller._calls.applyEdit).toBe(1);
  });
});
