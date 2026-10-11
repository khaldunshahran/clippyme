// ui01Review04.test.js — UI-01 review-04 contracts (durable suite).
//
// Contract 1 — a deferred source seek must never leak into a fallback
// artifact. While an exported render shows, a source-timeline gesture
// returns the player to the draft input and queues the seek for the
// draft's loadedmetadata. That pending seek is BOUND to the exact
// input + preview session it was queued for: if the source load then
// FAILS (the production media mount refuses source_* files), the
// player falls back to the result file — and the result's own
// loadedmetadata must not consume the source seek. An export or a new
// preview session must not consume an older source seek either.
// Successful pointer/keyboard draft transitions (review-03) are
// preserved.
//
// Contract 2 — missing source identity must not invent clean-draft
// provenance. resolveDraftInput returns kind='draft' only for legacy
// clip-level references that are NOT backend result URLs. A
// clip.video_url served through the /videos media route is the
// pipeline's reframed RESULT file (domain/job_results.py), however
// old the project is: missing/empty/non-string project.source.file
// classifies it 'result' — honest badge, draft overlays and
// source-coordinate operations suppressed.
//
// The coordinator's verbatim probe lives in
// coordinatorUi01Review04.test.js (the review-03 durable suite plus
// the two review-04 contracts); the two promoted contracts are kept
// verbatim at the bottom of this file, alongside the delayed-metadata
// controls and classification units the probe does not reach. Test
// kinds are labelled accurately: "mounted" tests use the hook-harness
// tree expander below (rendered element objects are frozen, so media
// players are plain objects installed through element refs, exactly
// as the coordinator probe does); unit tests call the exported pure
// helpers.

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
  projectSourceFileUrl,
  resolveDraftInput,
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


describe('UI-01 review-04 — missing source identity (unit)', () => {
  const legacy = { source: { duration: 30, width: 1920, height: 1080 } };

  it('a backend media-route video_url is a result, not a clean draft', () => {
    expect(resolveDraftInput({ video_url: '/videos/j1/example_clip_1.mp4' }, legacy, 'j1'))
      .toEqual({ src: '/videos/j1/example_clip_1.mp4', kind: 'result' });
    // The absolute backend-URL form (safeResolveUrl output in
    // production) classifies the same way.
    expect(resolveDraftInput({ video_url: 'https://api.example/videos/j1/example_clip_1.mp4' }, legacy, 'j1').kind)
      .toBe('result');
  });

  it('empty / whitespace / non-string source.file mints no draft provenance either', () => {
    const clip = { video_url: '/videos/j1/example_clip_1.mp4' };
    expect(resolveDraftInput(clip, { source: { file: '' } }, 'j1').kind).toBe('result');
    expect(resolveDraftInput(clip, { source: { file: '   ' } }, 'j1').kind).toBe('result');
    expect(resolveDraftInput(clip, { source: { file: 42 } }, 'j1').kind).toBe('result');
    expect(resolveDraftInput(clip, { source: { file: null } }, 'j1').kind).toBe('result');
    expect(resolveDraftInput(clip, { source: null }, 'j1').kind).toBe('result');
    expect(resolveDraftInput(clip, {}, 'j1').kind).toBe('result');
    expect(resolveDraftInput(clip, null, 'j1').kind).toBe('result');
    expect(resolveDraftInput(clip, undefined, 'j1').kind).toBe('result');
  });

  it('composed-only and empty clips stay composed / none without a source identity', () => {
    expect(resolveDraftInput({ composed_video_url: '/videos/j1/c.mp4' }, legacy, 'j1'))
      .toEqual({ src: '/videos/j1/c.mp4', kind: 'composed' });
    expect(resolveDraftInput({}, legacy, 'j1')).toEqual({ src: '', kind: 'none' });
  });

  it('a non-route legacy clip reference keeps the legacy draft treatment', () => {
    expect(resolveDraftInput({ video_url: 'raw.mp4' }, legacy, 'j1'))
      .toEqual({ src: 'raw.mp4', kind: 'draft' });
    expect(resolveDraftInput({ video_url: 'raw.mp4', composed_video_url: 'comp.mp4' }, legacy, 'j1'))
      .toEqual({ src: 'raw.mp4', kind: 'draft' });
  });

  it('a recorded source still wins; an unsafe recorded source still falls back to result', () => {
    const clip = { video_url: '/videos/j1/example_clip_1.mp4' };
    expect(resolveDraftInput(clip, makeProject(3), 'j1'))
      .toEqual({ src: '/videos/j1/source_example_clip_1.mp4', kind: 'source' });
    const evil = { source: { file: '../escape.mp4', duration: 30 } };
    expect(resolveDraftInput(clip, evil, 'j1'))
      .toEqual({ src: '/videos/j1/example_clip_1.mp4', kind: 'result' });
  });
});

describe('UI-01 review-04 — mounted provenance for source-less projects', () => {
  it('a source-less project on a route URL shows the result badge and disables source seeking', () => {
    const project = makeProject(3);
    project.source = { duration: 30, width: 1920, height: 1080 }; // no file recorded
    const { controller } = mockController(project);
    const clip = { video_url: '/videos/j1/example_clip_1.mp4', composed_video_url: null };
    const props = { jobId: 'j1', clipIndex: 0, clip, onBack: () => {}, controller };
    const tree = mount(props);
    expect(videoEl(tree).props.src).toBe('/videos/j1/example_clip_1.mp4');
    expect(badge(tree)).toBe('Result file — not the raw source');
    // The media stays playable, but it is an artifact state: a
    // source-timeline gesture must not write source coordinates in.
    const player = installPlayer(tree, 1);
    clickScrubberAt(tree, (20 / 30) * 300);
    expect(player.currentTime).toBe(1);
  });
});

describe('UI-01 review-04 — delayed metadata controls (mounted)', () => {
  it('a pending source seek is not consumed by an export', async () => {
    const { tree, props } = await mountOnRender();
    const artifact = installPlayer(tree, 1);
    clickScrubberAt(tree, (20 / 30) * 300); // queue a source seek for t=20
    expect(artifact.currentTime).toBe(1);
    let next = renderEditor(props);
    expect(badge(next)).toBe('Draft preview'); // returned to the draft input
    // Export before the draft metadata ever arrives: the render that
    // replaces the player must not inherit the queued source seek.
    await buttonByText(next, 'Export').props.onClick();
    next = renderEditor(props);
    expect(badge(next)).toContain('Historical');
    const render = installPlayer(next, 0);
    videoEl(next).props.onLoadedMetadata();
    expect(render.currentTime).toBe(0);
  });

  it('a pending source seek is not consumed after a preview-session switch', async () => {
    const { tree, props } = await mountOnRender();
    const artifact = installPlayer(tree, 1);
    clickScrubberAt(tree, (20 / 30) * 300); // queue a source seek for session j1:0
    expect(artifact.currentTime).toBe(1);
    // Switch to another clip (a new preview session) before the draft
    // metadata arrives; the new session's late metadata must not
    // consume the older session's seek.
    const b = mockController(makeProject(3));
    const propsB = {
      jobId: 'j1', clipIndex: 1,
      clip: { video_url: '/videos/j1/example_clip_2.mp4', composed_video_url: null },
      onBack: () => {}, controller: b.controller,
    };
    let treeB = renderEditor(propsB);
    treeB = renderEditor(propsB);
    expect(badge(treeB)).toBe('Draft preview');
    const playerB = installPlayer(treeB, 0);
    videoEl(treeB).props.onLoadedMetadata();
    expect(playerB.currentTime).toBe(0);
  });

  it('a late artifact metadata event does not destroy a pending source seek', async () => {
    const { tree, props } = await mountOnRender();
    const artifact = installPlayer(tree, 1);
    // The artifact element's metadata handler, captured while the
    // render is showing: a superseded element can still deliver a
    // late loadedmetadata (from an earlier load()) after the gesture
    // has already returned the player to the draft input.
    const strayMeta = videoEl(tree).props.onLoadedMetadata;
    clickScrubberAt(tree, (20 / 30) * 300); // queue a source seek for t=20
    expect(artifact.currentTime).toBe(1);
    const next = renderEditor(props);
    expect(badge(next)).toBe('Draft preview');
    const draft = installPlayer(next, 0);
    strayMeta(); // late, non-matching: must neither consume nor destroy
    expect(draft.currentTime).toBe(0);
    videoEl(next).props.onLoadedMetadata(); // the draft's own metadata
    expect(draft.currentTime).toBe(20);
  });

  it('a seek discarded at loadedmetadata is retried on canplay', async () => {
    const { tree, props } = await mountOnRender();
    const artifact = installPlayer(tree, 1);
    clickScrubberAt(tree, (20 / 30) * 300); // queue a source seek for t=20
    expect(artifact.currentTime).toBe(1);
    const next = renderEditor(props);
    expect(badge(next)).toBe('Draft preview');
    const draft = installPlayer(next, 0);
    // Model an element at readyState 1 whose first currentTime write
    // is silently discarded (target beyond the buffered range): the
    // seek must stay queued, not be consumed as if it had landed.
    let backing = 0;
    let writes = 0;
    Object.defineProperty(draft, 'currentTime', {
      get: () => backing,
      set: (val) => { writes += 1; if (writes > 1) backing = val; },
    });
    videoEl(next).props.onLoadedMetadata();
    expect(draft.currentTime).toBe(0);
    videoEl(next).props.onCanPlay();
    expect(draft.currentTime).toBe(20);
  });

  it('positive control: a successful deferred source seek still lands on the draft', async () => {
    const { tree, props } = await mountOnRender();
    const artifact = installPlayer(tree, 1);
    clickScrubberAt(tree, (20 / 30) * 300);
    expect(artifact.currentTime).toBe(1);
    const next = renderEditor(props);
    expect(badge(next)).toBe('Draft preview');
    const draft = installPlayer(next, 0);
    videoEl(next).props.onLoadedMetadata();
    expect(draft.currentTime).toBe(20);
  });
});

// --- Promoted coordinator contracts (verbatim) ------------------------

it('a failed deferred source load must not seek the result artifact on fallback metadata', async () => {
  const {tree,props} = await mountOnRender();
  const artifact = installPlayer(tree,1);
  clickScrubberAt(tree,200); // requests source time 20
  expect(artifact.currentTime).toBe(1);
  let next = renderEditor(props);
  expect(videoEl(next).props.src).toContain('source_example_clip_1.mp4');
  videoEl(next).props.onError(); // source is refused by production media mount
  next = renderEditor(props);
  const fallback = installPlayer(next,0);
  videoEl(next).props.onLoadedMetadata();
  expect(fallback.currentTime).toBe(0);
});

it('a missing source identity cannot classify reframed video_url as a clean draft', () => {
  const legacy={source:{duration:30,width:1920,height:1080}};
  const input=resolveDraftInput({video_url:'/videos/j1/example_clip_1.mp4'},legacy,'j1');
  expect(input.kind).toBe('result');
});

it('remote result URLs do not prove clean raw footage when source identity is missing', () => {
  const p={source:{width:1920,height:1080,duration:30}};
  const input=resolveDraftInput({video_url:'https://media.example.net/job/example_clip_1.mp4'},p,'j1');
  expect(input.kind).toBe('result');
});

it('a discarded source seek at a positive old position must remain pending', async () => {
  const {tree,props}=await mountOnRender();
  installPlayer(tree,1);
  clickScrubberAt(tree,200);
  const next=renderEditor(props);
  const draft=installPlayer(next,5);
  let pos=5;
  let writes=0;
  Object.defineProperty(draft,'currentTime',{
    get:()=>pos,
    set:(t)=>{writes++; if(writes>1)pos=t;},
  });
  videoEl(next).props.onLoadedMetadata();
  expect(draft.currentTime).toBe(5);
  videoEl(next).props.onCanPlay();
  expect(draft.currentTime).toBe(20);
});
