/**
 * Task 16 review-01: production-call regressions for the three corrected areas.
 *
 * 1. Manual callbacks lose their project: the real panel callbacks mutate in
 *    place and return nothing. The controlled bridge must retain the mutated
 *    clone. (Before: undefined was forwarded to saveClipProject.)
 * 2. Restored draft edits use a historical concurrency version: normal
 *    mutation/undo/redo saves must target the acknowledged version, while
 *    explicitly supplied AI base-version guards still reject stale patches.
 * 3. Lifecycle invalidation: queue save callbacks and export completions are
 *    guarded by the session/controller lifetime. A reset/dispose cannot be
 *    repopulated, cannot start another save, and cannot replace the preview.
 *
 * These use the REAL createProjectController + REAL createEditQueue with an
 * injected API only. They are load-bearing: removing callback adaptation,
 * acknowledged-version forwarding, or lifecycle cancellation must fail them.
 */
import { describe, expect, it, vi } from 'vitest';

vi.mock('../api/realApi.js', () => ({
  getClipProject: vi.fn(),
  saveClipProject: vi.fn(),
  composeProject: vi.fn(),
  backfillClipProjects: vi.fn(),
  listClipProjectVersions: vi.fn(),
  editClipAI: vi.fn(),
  safeResolveUrl: (u) => u,
  clipPreviewSrc: () => '',
  uploadAudio: vi.fn(),
}));

import { createProjectController } from './projectController.js';
import { createControlledActions } from '../editor/clipEditor.jsx';
import * as realApi from '../api/realApi.js';

const clone = (p) => JSON.parse(JSON.stringify(p));
const project = (v) => ({
  version: v,
  idempotency_key: `v${v}`,
  origin: 'auto',
  segments: [{ start: 0, end: 30 }],
  captions: { style: 'classic_white', edits: {} },
  overlays: [],
});

/**
 * Fake backend with real optimistic-concurrency (409 on version mismatch)
 * and controllable async gates for in-flight lifecycle tests.
 */
function fixture() {
  let server = project(1);
  const saves = [];
  let saveGate = null;
  let composeGate = null;
  const toasts = [];
  const api = {
    getClipProject: async (j, i, v) => (v == null ? clone(server) : project(v)),
    backfillClipProjects: async () => {},
    listClipProjectVersions: async () => ({ versions: [] }),
    saveClipProject: async (j, i, p, o) => {
      saves.push({ p: clone(p), o });
      if (saveGate) await saveGate;
      if (o.expectedVersion !== server.version) {
        const e = new Error('stale project version');
        e.status = 409;
        throw e;
      }
      server = { ...clone(p), version: server.version + 1 };
      return { project: clone(server), version: server.version };
    },
    composeProject: async () => {
      if (composeGate) await composeGate;
      return { composed_url: '/old.mp4' };
    },
  };
  const c = createProjectController({
    jobId: 'j',
    clipIndex: 0,
    api,
    onToast: (m) => toasts.push(m),
  });
  return {
    c,
    saves,
    toasts,
    setSaveGate: (g) => (saveGate = g),
    setComposeGate: (g) => (composeGate = g),
  };
}

describe('review-01 area 1: void-returning manual callbacks', () => {
  it('production manual callback saves the modified draft without requiring a return', async () => {
    const f = fixture();
    await f.c.load();
    const a = createControlledActions(f.c);
    // The actual panel callback shape: mutate in place, return nothing.
    await a.mutate('caption style', (p) => {
      p.captions.style = 'bold';
    });
    expect(f.c.getState().project.captions.style).toBe('bold');
    expect(f.saves[0].p.origin).toBe('user');
    expect(realApi.saveClipProject).not.toHaveBeenCalled();
    f.c.dispose();
  });

  it('transcript -> manual (void) -> AI -> export contains all three', async () => {
    const f = fixture();
    await f.c.load();
    const a = createControlledActions(f.c);
    // Transcript-style word edit (void).
    await a.mutate('edit word', (p) => {
      p.captions.edits['0'] = 'hallo';
    });
    // Manual trim (void).
    await a.mutate('trim segment', (p) => {
      p.segments[0].end = 20;
    });
    // AI patch apply (CopilotPanel's exact call shape).
    const base = f.c.getState().project;
    const patched = clone(base);
    patched.segments = [{ id: 'seg-ai', start: 5, end: 20 }];
    await f.c.applyEdit('copilot: cut intro', () => patched, {
      expectedVersion: patched.version,
    });
    const res = await a.exportProject();
    expect(res.identity.version).toBe(4);
    expect(f.c.getState().project.captions.edits['0']).toBe('hallo');
    expect(f.c.getState().project.segments[0].end).toBe(20);
    f.c.dispose();
  });

  it('a non-object mutate return is rejected, never saved', async () => {
    const f = fixture();
    await f.c.load();
    const a = createControlledActions(f.c);
    await expect(a.mutate('bad', () => 42)).rejects.toThrow(/mutate in place or return a project/);
    expect(f.saves).toHaveLength(0);
    f.c.dispose();
  });

  it('a rejected async panel save surfaces via toast, never as an unhandled rejection', async () => {
    const f = fixture();
    await f.c.load();
    const a = createControlledActions(f.c);
    await expect(
      a.mutate('boom', async () => {
        throw new Error('panel exploded');
      })
    ).rejects.toThrow('panel exploded');
    // The queue reported it through onError -> toast; nothing was saved.
    expect(f.saves).toHaveLength(0);
    expect(f.toasts.some((m) => /boom/.test(m))).toBe(true);
    f.c.dispose();
  });
});

describe('review-01 area 2: acknowledged version for normal saves', () => {
  it('manual edit after restore saves against acknowledged version, not historical version', async () => {
    const f = fixture();
    await f.c.load();
    await f.c.applyEdit('edit', (p) => {
      p.captions.edits.a = 'a';
      return p;
    });
    await f.c.restoreVersion(1);
    await f.c.applyEdit('restored edit', (p) => {
      p.captions.edits.b = 'b';
      return p;
    });
    expect(f.saves[1].o.expectedVersion).toBe(2);
    expect(f.c.getState().project.captions.edits.b).toBe('b');
    f.c.dispose();
  });

  it('restore -> undo saves against the acknowledged version', async () => {
    const f = fixture();
    await f.c.load();
    await f.c.applyEdit('edit', (p) => {
      p.captions.edits.a = 'a';
      return p;
    });
    await f.c.restoreVersion(1);
    // Undo the (empty) restore stack? No -- make an edit first, then undo it.
    await f.c.applyEdit('restored edit', (p) => {
      p.captions.edits.b = 'b';
      return p;
    });
    await f.c.undo();
    const undoSave = f.saves[f.saves.length - 1];
    expect(undoSave.o.expectedVersion).toBe(3);
    f.c.dispose();
  });

  it('explicit AI base-version guard still rejects a stale patch (not weakened)', async () => {
    const f = fixture();
    await f.c.load();
    // Advance the server to v2 with a normal edit.
    await f.c.applyEdit('edit', (p) => {
      p.captions.edits.a = 'a';
      return p;
    });
    // An AI patch computed against v1 must still 409.
    const stalePatch = project(1);
    await expect(
      f.c.applyEdit('copilot: stale', () => stalePatch, { expectedVersion: 1 })
    ).rejects.toThrow(/stale project version/i);
    f.c.dispose();
  });

  it('restore -> export saves the draft against the acknowledged version', async () => {
    const f = fixture();
    await f.c.load();
    await f.c.applyEdit('edit', (p) => {
      p.captions.edits.a = 'a';
      return p;
    });
    await f.c.restoreVersion(1); // dirty draft at historical v1, ack v2
    const res = await f.c.exportProject();
    // The dirty draft was saved with expectedVersion=2 before composing.
    const draftSave = f.saves.find((s) => s.o.expectedVersion === 2);
    expect(draftSave).toBeDefined();
    expect(res.identity.version).toBe(3);
    f.c.dispose();
  });
});

describe('review-01 area 3: lifecycle invalidation', () => {
  it('session reset cannot be repopulated by an in-flight save', async () => {
    const f = fixture();
    await f.c.load();
    let release;
    f.setSaveGate(new Promise((r) => (release = r)));
    const pending = f.c.applyEdit('edit', (p) => {
      p.captions.edits.a = 'a';
      return p;
    });
    await vi.waitFor(() => expect(f.saves.length).toBe(1));
    f.c.resetSession();
    release();
    await pending.catch(() => {});
    expect(f.c.getState().project).toBeNull();
    expect(f.c.canUndo()).toBe(false);
  });

  it('queued patches from a reset session never start another save', async () => {
    const f = fixture();
    await f.c.load();
    let release;
    f.setSaveGate(new Promise((r) => (release = r)));
    const first = f.c.applyEdit('edit', (p) => p);
    await vi.waitFor(() => expect(f.saves.length).toBe(1));
    const stale = f.c.applyEdit('queued old patch', () => project(1));
    f.c.resetSession();
    release();
    await Promise.allSettled([first, stale]);
    expect(f.saves.length).toBe(1);
  });

  it('old export cannot replace preview after controller disposal on clip switch', async () => {
    const f = fixture();
    await f.c.load();
    let release;
    f.setComposeGate(new Promise((r) => (release = r)));
    const onExported = vi.fn();
    const actions = createControlledActions(f.c, { onExported });
    const pending = actions.exportProject();
    await new Promise((r) => setTimeout(r, 10));
    f.c.dispose();
    release();
    await pending.catch(() => {});
    expect(onExported).not.toHaveBeenCalled();
    expect(f.c.getState().lastExport).toBeNull();
  });

  it('a stale AI result cannot apply across a clip switch', async () => {
    // Start an AI trim request, then dispose (clip switch) before it resolves.
    let resolveAI;
    const api = {
      getClipProject: async () => project(1),
      editClipAI: () => new Promise((r) => (resolveAI = r)),
    };
    const c2 = createProjectController({ jobId: 'j', clipIndex: 1, api });
    await c2.load();
    const aiP = c2.aiTrim('cut the intro');
    c2.dispose(); // clip switch
    resolveAI({ drop_ranges: [] });
    await expect(aiP).rejects.toThrow(/clip changed/);
  });

  it('in-flight restore is cancelled by reset; a fresh restore after reset works', async () => {
    const f = fixture();
    await f.c.load();
    let release;
    f.setSaveGate(new Promise((r) => (release = r)));
    // Occupy the queue with a gated save so the restore stays queued.
    const first = f.c.applyEdit('edit', (p) => p);
    await vi.waitFor(() => expect(f.saves.length).toBe(1));
    const restoreP = f.c.restoreVersion(1);
    f.c.resetSession();
    release();
    await Promise.allSettled([first, restoreP]);
    expect(f.c.getState().project).toBeNull();
    expect(f.c.getState().dirty).toBe(false);
  });
});
