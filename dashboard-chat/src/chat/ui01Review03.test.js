// ui01Review03.test.js — UI-01 review-03 contracts (durable suite).
//
// Group 1 — source timeline / artifact seek isolation: a source-
// coordinate seek (Timeline pointer click OR arrow keys, both routed
// through ClipEditorView's onSeek) must never be written into a
// rendered artifact's playback element. While an exported render is
// showing and a genuine draft input exists, the gesture returns the
// player to the draft input and the seek lands on the DRAFT (deferred
// to loadedmetadata — the artifact element is never touched). While
// the composed/result fallback shows (no usable draft input), source
// seeking is disabled outright. Artifact playback time is tracked on
// a separate playhead: it never moves the draft playhead (observed
// through the TrimPanel "Split at …" label, which renders the draft
// playhead).
//
// Group 2 — the authoritative draft input is the loaded project's
// recorded source file (project.source.file), resolved through the
// existing /videos/{job_id}/{file} media route — NOT clip.video_url,
// which the backend sets to the pipeline's reframed RESULT file
// (domain/job_results.py + pipeline/orchestrator.py), while the raw
// slice basename is recorded separately by pipeline/project_emit.py.
// Unsafe/unresolvable source refs fall back to an honestly labeled
// artifact state; a source that fails to LOAD falls back the same way
// (video error → result badge + warning toast). Projects that record
// no source file keep the legacy clip-level selection (compat).
//
// The coordinator's verbatim probe lives in
// coordinatorUi01Review03.test.js; this suite pins the same contracts
// plus the branches the probe does not reach (keyboard seek, composed
// state, load-error fallback, traversal validation, playhead
// separation). Test kinds are labelled accurately: "mounted" tests use
// the hook-harness tree expander below (the same harness family as
// ui01Review02.test.js — rendered element objects are frozen, so media
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

describe('UI-01 review-03 — authoritative source resolution (unit)', () => {
  it('projectSourceFileUrl resolves a recorded basename through the media route', () => {
    expect(projectSourceFileUrl(makeProject(3), 'j1')).toBe('/videos/j1/source_example_clip_1.mp4');
    expect(projectSourceFileUrl({ source: { file: 'source_a.mov' } }, 'job-9_x')).toBe('/videos/job-9_x/source_a.mov');
  });

  it('projectSourceFileUrl rejects traversal, separators, schemes, dotfiles and non-media', () => {
    const bad = [
      '../secret.mp4', 'a/b.mp4', 'a\\b.mp4', '/abs/path.mp4',
      'https://evil.example/x.mp4', '.hidden.mp4', 'source_clip_1.json',
      'source clip.mp4', '', '   ', '..mp4',
    ];
    for (const file of bad) {
      expect(projectSourceFileUrl({ source: { file } }, 'j1')).toBe('');
    }
    expect(projectSourceFileUrl({}, 'j1')).toBe('');
    expect(projectSourceFileUrl(null, 'j1')).toBe('');
    expect(projectSourceFileUrl({ source: { file: 'source_a.mp4' } }, '')).toBe('');
    expect(projectSourceFileUrl({ source: { file: 'source_a.mp4' } }, '../j1')).toBe('');
  });

  it('resolveDraftInput: recorded source wins over the result file', () => {
    const clip = { video_url: '/videos/j1/example_clip_1.mp4', composed_video_url: '/videos/j1/c.mp4' };
    expect(resolveDraftInput(clip, makeProject(3), 'j1'))
      .toEqual({ src: '/videos/j1/source_example_clip_1.mp4', kind: 'source' });
  });

  it('resolveDraftInput: unsafe recorded source falls back to an honest artifact kind', () => {
    const clip = { video_url: '/videos/j1/example_clip_1.mp4', composed_video_url: '/videos/j1/c.mp4' };
    const evil = { source: { file: '../../etc/passwd.mp4', duration: 30 } };
    expect(resolveDraftInput(clip, evil, 'j1'))
      .toEqual({ src: '/videos/j1/example_clip_1.mp4', kind: 'result' });
    const composedOnly = { video_url: null, composed_video_url: '/videos/j1/c.mp4' };
    expect(resolveDraftInput(composedOnly, evil, 'j1'))
      .toEqual({ src: '/videos/j1/c.mp4', kind: 'composed' });
    expect(resolveDraftInput({}, evil, 'j1')).toEqual({ src: '', kind: 'none' });
  });

  it('resolveDraftInput: no recorded source keeps the legacy clip-level selection', () => {
    const legacy = { source: { duration: 30, width: 1920, height: 1080 } };
    expect(resolveDraftInput({ video_url: 'raw.mp4', composed_video_url: 'comp.mp4' }, legacy, 'j1'))
      .toEqual({ src: 'raw.mp4', kind: 'draft' });
    expect(resolveDraftInput({ composed_video_url: 'comp.mp4' }, legacy, 'j1'))
      .toEqual({ src: 'comp.mp4', kind: 'composed' });
    expect(resolveDraftInput({}, legacy, 'j1')).toEqual({ src: '', kind: 'none' });
    expect(resolveDraftInput({ video_url: 'raw.mp4' }, null, 'j1'))
      .toEqual({ src: 'raw.mp4', kind: 'draft' });
  });
});

describe('UI-01 review-03 — mounted draft input selection', () => {
  it('draft player loads the project source slice, not the reframed result', () => {
    const { controller } = mockController(makeProject(3));
    const clip = {
      video_url: '/videos/j1/example_clip_1.mp4',
      composed_video_url: '/videos/j1/composed_example_clip_1.mp4',
    };
    const tree = mount({ jobId: 'j1', clipIndex: 0, clip, onBack: () => {}, controller });
    expect(videoEl(tree).props.src).toBe('/videos/j1/source_example_clip_1.mp4');
    expect(videoEl(tree).props.src).not.toBe(clip.video_url);
    expect(badge(tree)).toBe('Draft preview');
  });

  it('a source that fails to load falls back to the honestly labeled result', () => {
    const toasts = [];
    const { controller } = mockController(makeProject(3));
    const clip = {
      video_url: '/videos/j1/example_clip_1.mp4',
      composed_video_url: '/videos/j1/composed_example_clip_1.mp4',
    };
    const props = {
      jobId: 'j1', clipIndex: 0, clip, onBack: () => {}, controller,
      pushToast: (kind, msg) => toasts.push({ kind, msg }),
    };
    let tree = mount(props);
    expect(videoEl(tree).props.src).toContain('source_example_clip_1.mp4');
    videoEl(tree).props.onError();
    tree = renderEditor(props);
    expect(videoEl(tree).props.src).toBe('/videos/j1/example_clip_1.mp4');
    expect(badge(tree)).toBe('Result file — not the raw source');
    expect(toasts.some((t) => t.kind === 'warn' && /raw source/i.test(t.msg))).toBe(true);
    // The fallback is an artifact: overlays stay suppressed.
    expect(findAll(tree, (el) => el.props?.['data-ui01'] === 'caption-overlay').length).toBe(0);
  });
});

describe('UI-01 review-03 — source timeline seek isolation (mounted)', () => {
  it('pointer seek on an exported artifact returns to the draft input; the artifact element is untouched', async () => {
    const { tree, props } = await mountOnRender();
    expect(badge(tree)).toContain('Historical');
    const player = installPlayer(tree, 1);
    clickScrubberAt(tree, (20 / 30) * 300);
    // The artifact element was never seeked in source coordinates.
    expect(player.currentTime).toBe(1);
    const after = renderEditor(props);
    expect(badge(after)).toBe('Draft preview');
    expect(videoEl(after).props.src).toContain('source_example_clip_1.mp4');
    // The pending source seek is applied to the DRAFT input on load.
    const draftPlayer = installPlayer(after, 0);
    videoEl(after).props.onLoadedMetadata();
    expect(draftPlayer.currentTime).toBe(20);
  });

  it('keyboard seek on an exported artifact is isolated the same way', async () => {
    const { tree, props } = await mountOnRender();
    // Park the artifact at 3s; the keyboard seek target is playhead+1 =
    // 1s, so a coupled write would visibly move the artifact to 1.
    const player = installPlayer(tree, 3);
    scrubber(tree).props.onKeyDown({ key: 'ArrowRight', preventDefault: () => {} });
    expect(player.currentTime).toBe(3);
    const after = renderEditor(props);
    expect(badge(after)).toBe('Draft preview');
    expect(videoEl(after).props.src).toContain('source_example_clip_1.mp4');
  });

  it('artifact playback time never moves the draft playhead', async () => {
    const { tree, props } = await mountOnRender();
    expect(splitLabel(tree)).toContain('0:00.0');
    const player = installPlayer(tree, 2);
    player.paused = false;
    videoEl(tree).props.onTimeUpdate();
    // Draft playhead (rendered by the TrimPanel split button) is
    // unmoved by the artifact's own playback time.
    expect(splitLabel(renderEditor(props))).toContain('0:00.0');
  });

  it('source seeking stays disabled while the composed fallback is showing', () => {
    // The project records a source, but it cannot be loaded and the
    // clip has no result file — the composed artifact is what shows.
    const { controller } = mockController(makeProject(3));
    const clip = { video_url: null, composed_video_url: '/videos/j1/composed_example_clip_1.mp4' };
    const props = { jobId: 'j1', clipIndex: 0, clip, onBack: () => {}, controller };
    let tree = mount(props);
    expect(videoEl(tree).props.src).toContain('source_example_clip_1.mp4');
    videoEl(tree).props.onError();
    tree = renderEditor(props);
    expect(badge(tree)).toBe('Composed file — burned-in graphics');
    const player = installPlayer(tree, 1);
    clickScrubberAt(tree, (20 / 30) * 300);
    expect(player.currentTime).toBe(1);
  });

  it('source seeking stays disabled while the result fallback is showing', () => {
    const evilProject = makeProject(3);
    evilProject.source = { file: '../escape.mp4', duration: 30, width: 1920, height: 1080 };
    const { controller } = mockController(evilProject);
    const clip = { video_url: '/videos/j1/example_clip_1.mp4', composed_video_url: null };
    const props = { jobId: 'j1', clipIndex: 0, clip, onBack: () => {}, controller };
    const tree = mount(props);
    expect(badge(tree)).toBe('Result file — not the raw source');
    const player = installPlayer(tree, 1);
    clickScrubberAt(tree, (20 / 30) * 300);
    expect(player.currentTime).toBe(1);
  });

  it('positive control: draft-mode seek still writes the draft element and playhead', () => {
    const { controller } = mockController(makeProject(3));
    const clip = { video_url: '/videos/j1/example_clip_1.mp4', composed_video_url: null };
    const props = { jobId: 'j1', clipIndex: 0, clip, onBack: () => {}, controller };
    const tree = mount(props);
    expect(badge(tree)).toBe('Draft preview');
    const player = installPlayer(tree, 0);
    clickScrubberAt(tree, (12 / 30) * 300);
    expect(player.currentTime).toBe(12);
    expect(splitLabel(renderEditor(props))).toContain('0:12.0');
  });
});
