// ui01Review06.test.js — UI-01 review-06 contracts (durable suite).
//
// Contract (review-06) — the relative-URL exception is removed.
// resolveDraftInput classifies by recorded provenance only: the
// recorded project.source.file is the sole authority for a raw
// ('source') input. With NO authoritative source identity
// (missing/empty/whitespace/non-string source.file), EVERY
// clip-level pick classifies honestly — 'composed' for the composed
// artifact, 'result' for any other pick whatever its shape
// (media-route /videos path, absolute/scheme-relative URL, or
// scheme-less relative reference such as 'example_clip_1.mp4' or
// '/example_clip_1.mp4'), and 'none' when there is no media at all.
// Reviews 04/05 established this for the route and cloud shapes;
// the scheme-less relative exception they retained is gone: a URL
// shape today's pipeline does not emit still cannot prove rawness
// for a legacy caller. The fallback file itself stays playable —
// it is simply labeled and treated as the delivered result it is,
// with draft overlays and source-coordinate operations suppressed.
// Genuine draft fixtures record their actual source.file and keep
// classifying 'source'.
//
// The promoted coordinator contract is kept verbatim at the bottom
// of this file. The coordinator's verbatim probe lives in
// coordinatorUi01Review06.test.js.

import { describe, expect, it, vi, beforeEach } from 'vitest';

import * as React from 'react';
vi.stubGlobal('React', React);

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

import {
  ClipEditorView,
  isAbsoluteMediaUrl,
  projectSourceFileUrl,
  resolveDraftInput,
  SEEK_LANDING_TOLERANCE_S,
} from '../editor/clipEditor.jsx';

const clone = (p) => JSON.parse(JSON.stringify(p));

// Production-shaped project: source records the raw slice basename
// (pipeline/project_emit.py), duration 30 in SOURCE coordinates.
function makeProject(version = 3) {
  return {
    origin: 'auto',
    version,
    idempotency_key: `key-v${version}`,
    source: { file: 'source_example_clip_1.mp4', duration: 30, width: 1920, height: 1080 },
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

// The real controller API surface (same shape as the coordinator
// probe's mock): saveDraft / exportProject / restoreVersion /
// resetToAiCut, versions carried on the snapshot.
function mockController(project, snapOverrides = {}) {
  const snap = {
    status: 'ready',
    loadError: '',
    project: clone(project),
    ackVersion: project.version,
    dirty: false,
    saving: false,
    exporting: false,
    versions: [{ version: 3 }, { version: 2 }],
    versionsLoading: false,
    ...snapOverrides,
  };
  const controller = {
    getState: () => clone(snap),
    subscribe: () => () => {},
    applyEdit: async (label, recipe) => { if (recipe) await recipe(clone(snap.project)); return null; },
    saveDraft: async () => ({ version: snap.project.version }),
    exportProject: async () => ({ version: 3, composed_url: 'render-v3.mp4', deduped: false }),
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
  return { controller, snap };
}

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
const badge = (tree) => textOf(
  findAll(tree, (el) => el.props?.title === 'What the preview player is showing')[0] || { props: {} },
);
const videoEl = (tree) => findAll(tree, (el) => el.type === 'video')[0];
const scrubber = (tree) => findAll(tree, (el) => el.props?.['aria-label'] === 'Timeline scrubber')[0];
const splitLabel = (tree) => {
  const b = buttons(tree).find((x) => buttonText(x).includes('Split at'));
  return b ? buttonText(b) : '';
};

// Install a plain-object media player behind the video element's ref
// (rendered elements are frozen; the component reads the player
// through videoRef.current — the coordinator probe's pattern).
function installPlayer(tree, currentTime) {
  const player = { currentTime, paused: true, duration: 30 };
  videoEl(tree).ref.current = player;
  return player;
}

function clickScrubberAt(tree, clientX) {
  const s = scrubber(tree);
  s.ref.current = { getBoundingClientRect: () => ({ left: 0, width: 300 }) };
  s.props.onClick({ clientX });
}

function renderEditor(props) {
  hooks.index = 0;
  return expand(ClipEditorView(props));
}

// Mount = render twice: the first pass runs the (immediately executed)
// effects that seed the draft input; the second pass reads the seeded
// state — the same convention as ui01Review02.test.js.
function mount(props) {
  renderEditor(props);
  return renderEditor(props);
}

// Mount the editor onto the exported v3 render, then diverge the
// draft (v4, dirty, nonzero multi-segment [10,14) + [18,24) on the
// 30s source) so the render classifies Historical — the review-03
// reproducer shape. NOTE: the mock controller's getState() returns a
// fresh clone per call, so the divergence is applied by replacing
// what getState returns (the probe's pattern), not by mutating snap.
async function mountOnRender() {
  const toasts = [];
  const { controller, snap } = mockController(makeProject(3));
  const clip = {
    video_url: '/videos/j1/example_clip_1.mp4',
    composed_video_url: '/videos/j1/composed_example_clip_1.mp4',
  };
  const props = {
    jobId: 'j1', clipIndex: 0, clip, onBack: () => {}, controller,
    pushToast: (kind, msg) => toasts.push({ kind, msg }),
  };
  let tree = mount(props);
  await buttonByText(tree, 'Export').props.onClick();
  tree = renderEditor(props);
  const diverged = clone(snap);
  diverged.project = makeProject(4);
  diverged.project.segments = [
    { id: 'seg0', start: 10, end: 14, crop: { x: 0, y: 0, w: 608, h: 1080 } },
    { id: 'seg1', start: 18, end: 24, crop: { x: 0, y: 0, w: 608, h: 1080 } },
  ];
  diverged.dirty = true;
  controller.getState = () => clone(diverged);
  tree = renderEditor(props);
  return { tree, props, controller, toasts, clip };
}

beforeEach(() => {
  hooks.values = [];
  hooks.index = 0;
  wsMock.current = null;
});



describe('UI-01 review-06 — relative references without source identity (unit)', () => {
  const legacy = { source: { duration: 30, width: 1920, height: 1080 } };

  it('a relative pick with a composed sibling is still a result, not a draft', () => {
    expect(resolveDraftInput({ video_url: 'example_clip_1.mp4', composed_video_url: 'comp.mp4' }, legacy, 'j1'))
      .toEqual({ src: 'example_clip_1.mp4', kind: 'result' });
  });

  it('missing or malformed source identity never mints clean-draft provenance for a relative pick', () => {
    expect(resolveDraftInput({ video_url: 'example_clip_1.mp4' }, null, 'j1').kind).toBe('result');
    expect(resolveDraftInput({ video_url: 'example_clip_1.mp4' }, undefined, 'j1').kind).toBe('result');
    expect(resolveDraftInput({ video_url: 'example_clip_1.mp4' }, { source: { file: '' } }, 'j1').kind).toBe('result');
    expect(resolveDraftInput({ video_url: 'example_clip_1.mp4' }, { source: { file: '   ' } }, 'j1').kind).toBe('result');
    expect(resolveDraftInput({ video_url: 'example_clip_1.mp4' }, { source: { file: 42 } }, 'j1').kind).toBe('result');
  });

  it('composed-only and empty clips stay composed / none without a source identity', () => {
    expect(resolveDraftInput({ composed_video_url: 'comp.mp4' }, legacy, 'j1'))
      .toEqual({ src: 'comp.mp4', kind: 'composed' });
    expect(resolveDraftInput({}, legacy, 'j1')).toEqual({ src: '', kind: 'none' });
  });

  it('a genuine draft fixture records its source.file and still classifies as the raw source', () => {
    // The correction removes the URL-shape exception, not draft
    // editing: a project that records its actual source.file keeps
    // the genuine 'source' input even when the clip also carries a
    // relative result reference.
    const clip = { video_url: 'example_clip_1.mp4', composed_video_url: 'comp.mp4' };
    expect(resolveDraftInput(clip, makeProject(3), 'j1'))
      .toEqual({ src: '/videos/j1/source_example_clip_1.mp4', kind: 'source' });
  });
});

describe('UI-01 review-06 — mounted provenance for source-less relative projects', () => {
  it('a source-less project on a relative URL shows the result badge and disables source seeking', () => {
    const project = makeProject(3);
    project.source = { duration: 30, width: 1920, height: 1080 }; // no file recorded
    const { controller } = mockController(project);
    const clip = { video_url: 'example_clip_1.mp4', composed_video_url: null };
    const props = { jobId: 'j1', clipIndex: 0, clip, onBack: () => {}, controller };
    const tree = mount(props);
    expect(videoEl(tree).props.src).toBe('example_clip_1.mp4');
    expect(badge(tree)).toBe('Result file — not the raw source');
    // The media stays playable, but it is an artifact state: a
    // source-timeline gesture must not write source coordinates in.
    const player = installPlayer(tree, 1);
    clickScrubberAt(tree, (20 / 30) * 300);
    expect(player.currentTime).toBe(1);
  });

  it('a source-less project on a root-relative URL shows the result badge too', () => {
    const project = makeProject(3);
    project.source = { duration: 30, width: 1920, height: 1080 }; // no file recorded
    const { controller } = mockController(project);
    const clip = { video_url: '/example_clip_1.mp4', composed_video_url: null };
    const props = { jobId: 'j1', clipIndex: 0, clip, onBack: () => {}, controller };
    const tree = mount(props);
    expect(videoEl(tree).props.src).toBe('/example_clip_1.mp4');
    expect(badge(tree)).toBe('Result file — not the raw source');
  });
});

// --- Promoted coordinator contract (verbatim) ------------------------
it('missing source identity cannot authorize a relative result as raw', () => {
  const project={source:{duration:30,width:1920,height:1080}};
  expect(resolveDraftInput({video_url:'example_clip_1.mp4'},project,'j1').kind).toBe('result');
  expect(resolveDraftInput({video_url:'/example_clip_1.mp4'},project,'j1').kind).toBe('result');
});
