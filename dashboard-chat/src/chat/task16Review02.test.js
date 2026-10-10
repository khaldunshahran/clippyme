/**
 * Task 16 review-02 — lifecycle contracts (durable regression tests).
 *
 * 1. DISPOSAL IS TERMINAL. A disposed controller rejects every new operation
 *    before any transform or API call, so a delayed upload/AI callback can
 *    never save through it. load() is the explicit lifecycle restart (the
 *    host calls it after a StrictMode remount whose cleanup disposed the
 *    shared controller). Removing the disposed guard must fail these tests.
 *
 * 2. RESTORE/EXPORT SEQUENCING. An ordinary queued restore must not cancel a
 *    valid export ahead of it: the export finishes with its frozen identity,
 *    then the restore executes. resetSession()/dispose() still truly cancel
 *    in-flight work. Re-adding the generation bump to restore must fail.
 *
 * 3. COPILOT IDENTITY BINDING. The real CopilotPanel binds a generated patch
 *    to the (jobId, clipIndex) that produced it; after a clip switch the
 *    stale result is neither shown nor appliable, and in-flight results are
 *    suppressed. The real host (EditorView) remounts the panel on clip
 *    switches and session resets. Removing the identity binding or the host
 *    key must fail these tests.
 */
import { describe, expect, it, vi, beforeEach } from 'vitest';

// --- minimal hook harness: persistent useState (coordinator-probe style),
// plus identity useMemo/useCallback and a safe useEffect for the host test.
const hooks = vi.hoisted(() => ({ values: [], index: 0 }));
const patchApi = vi.hoisted(() => ({ editAiPatch: vi.fn() }));
const sessionResetCbs = vi.hoisted(() => []);
vi.mock('react', async (original) => {
  const actual = await original();
  return {
    ...actual,
    useState: (initial) => {
      const i = hooks.index++;
      if (!(i in hooks.values)) hooks.values[i] = initial;
      return [
        hooks.values[i],
        (v) => { hooks.values[i] = typeof v === 'function' ? v(hooks.values[i]) : v; },
      ];
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
  getClipProject: vi.fn(async () => null),
  saveClipProject: vi.fn(),
  composeProject: vi.fn(),
  backfillClipProjects: vi.fn(async () => {}),
  listClipProjectVersions: vi.fn(async () => ({ versions: [] })),
  editClipAI: vi.fn(),
  uploadAudio: vi.fn(),
  clipVideoSrc: () => '',
  safeResolveUrl: (u) => u,
  clipPreviewSrc: () => '',
}));
vi.mock('../api/chatApi.js', () => patchApi);
vi.mock('../api/sessionReset.js', () => ({
  onSessionReset: (cb) => { sessionResetCbs.push(cb); return () => {}; },
}));

import { createProjectController } from './projectController.js';
import CopilotPanel from './CopilotPanel.jsx';
import EditorView from './EditorView.jsx';
import * as React from 'react';
vi.stubGlobal('React', React);

const clone = (p) => JSON.parse(JSON.stringify(p));
const project = (v, tag = 'A') => ({
  version: v,
  idempotency_key: `${tag}${v}`,
  origin: 'auto',
  source: { tag },
  segments: [{ start: 0, end: 30 }],
  captions: { edits: {} },
  overlays: [],
});

function makeFixture(tag = 'A', clipIndex = 0) {
  let server = project(1, tag);
  const saves = [];
  const composes = [];
  let composeGate = null;
  let getGate = null;
  let getCalls = 0;
  const api = {
    getClipProject: async (j, i, v) => {
      getCalls++;
      if (getGate) await getGate;
      return v == null ? clone(server) : { ...clone(server), version: v };
    },
    saveClipProject: async (j, i, p, o) => {
      saves.push(clone(p));
      if (o.expectedVersion !== server.version) {
        throw Object.assign(new Error('stale'), { status: 409 });
      }
      server = { ...clone(p), version: server.version + 1 };
      return { project: clone(server), version: server.version };
    },
    composeProject: async (j, i, p) => {
      composes.push(clone(p));
      if (composeGate) await composeGate;
      return { composed_url: '/a.mp4' };
    },
    listClipProjectVersions: async () => ({ versions: [] }),
    backfillClipProjects: async () => {},
    editClipAI: async () => { throw new Error('no AI in fixture'); },
    uploadAudio: async () => ({ ok: true }),
  };
  const c = createProjectController({ jobId: 'j', clipIndex, api });
  return {
    c, saves, composes,
    getCalls: () => getCalls,
    setComposeGate: (g) => { composeGate = g; },
    setGetGate: (g) => { getGate = g; },
  };
}

// --- element-tree helpers for the real components ---
function nodes(tree, predicate) {
  const out = [];
  const walk = (n) => {
    if (!n || typeof n !== 'object') return;
    if (predicate(n)) out.push(n);
    const kids = n.props && n.props.children;
    (Array.isArray(kids) ? kids : [kids]).forEach(walk);
  };
  walk(tree);
  return out;
}
function renderCopilot(props) {
  hooks.index = 0;
  return CopilotPanel(props);
}
function renderHost(props) {
  hooks.index = 0;
  return EditorView(props);
}
const findCopilot = (tree) => nodes(tree, (n) => n.type === CopilotPanel)[0];

beforeEach(() => {
  hooks.values = [];
  hooks.index = 0;
  sessionResetCbs.length = 0;
  patchApi.editAiPatch.mockReset();
});

describe('1. disposal is terminal', () => {
  it('a late callback that starts after disposal saves nothing', async () => {
    const f = makeFixture();
    await f.c.load();
    f.c.dispose();
    // A delayed upload/AI callback invoking applyEdit AFTER disposal:
    await expect(
      f.c.applyEdit('late upload/AI callback', (p) => p)
    ).rejects.toMatchObject({ code: 'DISPOSED' });
    expect(f.saves).toHaveLength(0);
  });

  it('every operation rejects before any transform or API call once disposed', async () => {
    const f = makeFixture();
    await f.c.load();
    f.c.dispose();
    await expect(f.c.applyEdit('x', (p) => p)).rejects.toMatchObject({ code: 'DISPOSED' });
    await expect(f.c.undo()).rejects.toMatchObject({ code: 'DISPOSED' });
    await expect(f.c.redo()).rejects.toMatchObject({ code: 'DISPOSED' });
    await expect(f.c.saveDraft()).rejects.toMatchObject({ code: 'DISPOSED' });
    await expect(f.c.exportProject()).rejects.toMatchObject({ code: 'DISPOSED' });
    await expect(f.c.restoreVersion(1)).rejects.toMatchObject({ code: 'DISPOSED' });
    await expect(f.c.resetToAiCut()).rejects.toMatchObject({ code: 'DISPOSED' });
    await expect(f.c.listVersions()).rejects.toMatchObject({ code: 'DISPOSED' });
    await expect(f.c.aiTrim('trim')).rejects.toMatchObject({ code: 'DISPOSED' });
    await expect(f.c.uploadAudio({})).rejects.toMatchObject({ code: 'DISPOSED' });
    expect(f.saves).toHaveLength(0);
    expect(f.composes).toHaveLength(0);
    expect(f.getCalls()).toBe(1); // only the initial load ran
  });

  it('in-flight work is still cancelled by dispose (response cancellation preserved)', async () => {
    const f = makeFixture();
    await f.c.load();
    let release;
    f.setComposeGate(new Promise((r) => { release = r; }));
    const ep = f.c.exportProject();
    const guarded = ep.catch(() => {});
    await vi.waitFor(() => expect(f.composes).toHaveLength(1));
    f.c.dispose();
    release();
    await expect(ep).rejects.toMatchObject({ code: 'QUEUE_CANCELLED' });
    await guarded;
    // The disposed controller is terminal; no state is resurrected and no
    // further operation is accepted.
    await expect(f.c.exportProject()).rejects.toMatchObject({ code: 'DISPOSED' });
  });

  it('load() is the explicit lifecycle restart: it revives a disposed controller', async () => {
    const f = makeFixture();
    await f.c.load();
    f.c.dispose();
    await f.c.load(); // e.g. StrictMode remount after cleanup disposed it
    await f.c.applyEdit('after revive', (p) => p);
    expect(f.saves).toHaveLength(1);
  });
});

describe('2. restore/export sequencing', () => {
  it('a queued restore does not cancel the valid export ahead of it', async () => {
    const f = makeFixture();
    await f.c.load();
    await f.c.applyEdit('edit', (p) => p);
    let release;
    f.setComposeGate(new Promise((r) => { release = r; }));
    const exportP = f.c.exportProject();
    await vi.waitFor(() => expect(f.composes).toHaveLength(1));
    const restoreP = f.c.restoreVersion(1);
    release();
    const st = await exportP.then(
      () => 'fulfilled',
      () => 'rejected'
    );
    expect(st).toBe('fulfilled');
    await restoreP;
    expect(f.c.getState().exporting).toBe(false);
    expect(f.c.getState().dirty).toBe(true);
  });

  it('an ordinary queued restore still works with no export in flight', async () => {
    const f = makeFixture();
    await f.c.load();
    await f.c.applyEdit('edit', (p) => p); // server at v2
    const restored = await f.c.restoreVersion(1);
    expect(restored.version).toBe(1);
    expect(f.c.getState().dirty).toBe(true);
    expect(f.c.getState().project.version).toBe(1);
  });

  it('reset during a restore still cancels it (true cancellation preserved)', async () => {
    const f = makeFixture();
    await f.c.load();
    let release;
    f.setGetGate(new Promise((r) => { release = r; }));
    const rp = f.c.restoreVersion(1);
    await vi.waitFor(() => expect(f.getCalls()).toBeGreaterThan(1));
    f.c.resetSession();
    release();
    const res = await rp;
    expect(res).toBe(null);
    expect(f.c.getState().dirty).toBe(false);
    expect(f.c.getState().project).toBe(null);
  });
});

describe('3. copilot identity binding', () => {
  it('the actual Copilot component must not apply clip A result to clip B', async () => {
    const a = makeFixture('A', 0);
    await a.c.load();
    patchApi.editAiPatch.mockResolvedValue({
      project: project(1, 'A'),
      patch: { ops: [] },
      explanation: 'A edit',
    });
    hooks.values = [];
    const propsA = { jobId: 'j', clipIndex: 0, applyEdit: a.c.applyEdit };
    let tree = renderCopilot(propsA);
    nodes(tree, (n) => n.type === 'input')[0].props.onChange({ target: { value: 'edit A' } });
    tree = renderCopilot(propsA);
    const edit = nodes(tree, (n) => n.type === 'button' && n.props.children === 'Edit')[0];
    await edit.props.onClick();

    const b = makeFixture('B', 1);
    await b.c.load();
    const propsB = { jobId: 'j', clipIndex: 1, applyEdit: b.c.applyEdit };
    tree = renderCopilot(propsB);
    const apply = nodes(
      tree,
      (n) =>
        n.type === 'button' &&
        Array.isArray(n.props.children) &&
        n.props.children.includes('Apply to project')
    )[0];
    if (apply) await apply.props.onClick();
    expect(b.saves).toHaveLength(0);
    expect(b.c.getState().project.source.tag).toBe('B');
  });

  it('an in-flight Copilot result is suppressed after a clip switch', async () => {
    const a = makeFixture('A', 0);
    await a.c.load();
    let release;
    patchApi.editAiPatch.mockImplementation(() => new Promise((r) => { release = r; }));
    hooks.values = [];
    const propsA = { jobId: 'j', clipIndex: 0, applyEdit: a.c.applyEdit };
    let tree = renderCopilot(propsA);
    nodes(tree, (n) => n.type === 'input')[0].props.onChange({ target: { value: 'edit A' } });
    tree = renderCopilot(propsA);
    const edit = nodes(tree, (n) => n.type === 'button' && n.props.children === 'Edit')[0];
    const runP = edit.props.onClick();
    // Switch clips while the AI response is still in flight.
    const propsB = { jobId: 'j', clipIndex: 1, applyEdit: a.c.applyEdit };
    release({ project: project(1, 'A'), patch: { ops: [] }, explanation: 'late A' });
    await runP;
    tree = renderCopilot(propsB);
    const apply = nodes(
      tree,
      (n) =>
        n.type === 'button' &&
        Array.isArray(n.props.children) &&
        n.props.children.includes('Apply to project')
    )[0];
    expect(apply).toBeUndefined();
  });

  it('host remounts CopilotPanel on clip switch (real host lifecycle)', () => {
    const base = { jobId: 'j', clip: {}, onBack: () => {} };
    hooks.values = [];
    const keyA = findCopilot(renderHost({ ...base, clipIndex: 0 })).key;
    hooks.values = [];
    const keyB = findCopilot(renderHost({ ...base, clipIndex: 1 })).key;
    expect(keyA).toBe('j:0:0');
    expect(keyB).toBe('j:1:0');
    expect(keyA).not.toBe(keyB);
  });

  it('host remounts CopilotPanel on session reset (real host lifecycle)', () => {
    const props = { jobId: 'j', clipIndex: 0, clip: {}, onBack: () => {} };
    hooks.values = [];
    const keyA = findCopilot(renderHost(props)).key;
    expect(sessionResetCbs.length).toBeGreaterThan(0);
    sessionResetCbs.forEach((cb) => cb()); // controller.resetSession() + resetKey bump
    const keyB = findCopilot(renderHost(props)).key;
    expect(keyA).toBe('j:0:0');
    expect(keyB).toBe('j:0:1');
    expect(keyA).not.toBe(keyB);
  });
});
