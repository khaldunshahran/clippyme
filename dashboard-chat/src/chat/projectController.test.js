/**
 * Task 16: project-controller integration tests.
 *
 * These exercise the REAL controller (projectController.js) together with
 * the REAL serialized edit queue (editor-timeline/undo.js) against a
 * stubbed API -- no duplicated helper expressions. They fail if:
 *  - edits bypass the queue (export would compose a non-acknowledged project),
 *  - expectedVersion is dropped from any save,
 *  - a stale patch overwrites newer edits,
 *  - export does not drain the queue / its identity shifts mid-flight,
 *  - late responses leak into another clip's state.
 */
import { describe, expect, it, vi } from 'vitest';
import { createProjectController } from './projectController.js';

function makeProject(version) {
  return {
    origin: 'auto',
    version,
    idempotency_key: `key-v${version}`,
    source: { duration: 30, width: 1920, height: 1080 },
    segments: [{ id: 'seg0', start: 0, end: 30, crop: { x: 0, y: 0, w: 1080, h: 1920 } }],
    captions: {
      style: 'classic_white',
      position: 'bottom',
      words: [{ id: 'w0', w: 'hello', start: 1, end: 1.5 }],
      edits: {},
    },
    overlays: [],
  };
}

const clone = (p) => JSON.parse(JSON.stringify(p));
const tick = () => new Promise((r) => setTimeout(r, 0));
// Compare project content ignoring the version/key bookkeeping.
const strip = (p) => {
  const { version, idempotency_key, ...rest } = p;
  return rest;
};

/**
 * Fake backend: versioned store with real optimistic-concurrency checks
 * (409 when expectedVersion != stored), version history, and controllable
 * async gates for in-flight tests.
 */
function fakeBackend() {
  let server = makeProject(1);
  const history = { 1: clone(server) };
  const saves = []; // { version, expectedVersion }
  const composeCalls = []; // payload at call time (pre-gate)
  const composes = []; // payloads that finished composing
  const gets = [];
  const backfills = [];
  const gates = {};
  const api = {
    getClipProject: async (jobId, index, version = null) => {
      gets.push(version ?? 'latest');
      if (gates.get) await gates.get;
      if (version != null && history[version]) return clone(history[version]);
      return clone(server);
    },
    saveClipProject: async (jobId, index, project, { expectedVersion } = {}) => {
      if (gates.save) await gates.save;
      saves.push({ version: project.version, expectedVersion });
      if (expectedVersion != null && expectedVersion !== server.version) {
        const e = new Error(
          `stale project version (expected ${expectedVersion}, stored ${server.version})`
        );
        e.status = 409;
        throw e;
      }
      const stored = server.version;
      server = clone(project);
      server.version = stored + 1;
      history[server.version] = clone(server);
      return { version: server.version, project: clone(server) };
    },
    composeProject: async (jobId, index, project) => {
      composeCalls.push(clone(project));
      if (gates.compose) await gates.compose;
      composes.push(clone(project));
      return { composed_url: '/composed/v.mp4', version: project.version, deduped: false };
    },
    backfillClipProjects: async () => {
      backfills.push(true);
      return { results: [] };
    },
    listClipProjectVersions: async () => ({
      versions: [1, 2, 3].map((v) => ({
        version: v,
        origin: v === 1 ? 'auto' : 'user',
        created_at: '2026-01-01T00:00:00',
        output_file: null,
        output_url: null,
      })),
      latest: 3,
    }),
    editClipAI: async (jobId, index, instruction) => ({
      drop_ranges: [[0, 5]],
      explanation: `cut per: ${instruction}`,
    }),
    uploadAudio: async () => ({ file: 'audio/x.mp3', size: 1 }),
  };
  return {
    api,
    saves,
    gets,
    composes,
    composeCalls,
    backfills,
    gates,
    serverVersion: () => server.version,
    serverProject: () => clone(server),
    bumpServer: (mut) => {
      const s = clone(server);
      mut(s);
      s.version = server.version + 1;
      server = s;
      history[server.version] = clone(server);
    },
  };
}

function makeController(backend, toasts = []) {
  return createProjectController({
    jobId: 'job1',
    clipIndex: 0,
    api: backend.api,
    onToast: (m) => toasts.push(m),
  });
}

describe('projectController: unified edit -> export flow', () => {
  it('transcript edit + manual edit + AI patch apply all land in the export payload', async () => {
    const backend = fakeBackend();
    const c = makeController(backend);
    await c.load();
    expect(c.getState().project.version).toBe(1);

    // 1. transcript-style word edit (TranscriptPanel's applyEdit call shape)
    await c.applyEdit('edit word', (p) => {
      p.captions.edits['0'] = 'hallo';
      return p;
    });
    // 2. manual trim (nested editor trim handler shape)
    await c.applyEdit('trim segment', (p) => {
      p.segments[0].end = 20;
      return p;
    });
    // 3. AI patch apply (CopilotPanel's exact call shape: whole patched
    //    project + the version the patch was computed against)
    const base = c.getState().project;
    const patched = clone(base);
    patched.segments = [{ id: 'seg-ai', start: 5, end: 20 }];
    await c.applyEdit('copilot: cut intro', () => patched, {
      expectedVersion: patched.version,
    });

    const res = await c.exportProject();
    expect(backend.composes).toHaveLength(1);
    const payload = backend.composes[0];
    expect(payload.captions.edits['0']).toBe('hallo');
    expect(payload.segments).toEqual([{ id: 'seg-ai', start: 5, end: 20 }]);
    // Export renders the exact acknowledged project/version.
    expect(payload.version).toBe(res.identity.version);
    expect(payload.idempotency_key).toBe(res.identity.idempotency_key);
    // User edits minted a fresh key (never the seeded one).
    expect(payload.idempotency_key).not.toBe('key-v1');
    expect(payload.origin).toBe('user');
    // Every save carried optimistic concurrency.
    expect(backend.saves.length).toBeGreaterThan(0);
    for (const s of backend.saves) {
      expect(s.expectedVersion).not.toBeNull();
      expect(s.expectedVersion).not.toBeUndefined();
    }
    c.dispose();
  });

  it('undo then redo through the controller restores exact content', async () => {
    const backend = fakeBackend();
    const c = makeController(backend);
    await c.load();
    const original = clone(c.getState().project);

    await c.applyEdit('edit word', (p) => {
      p.captions.edits['0'] = 'hallo';
      return p;
    });
    const edited = clone(c.getState().project);
    expect(edited.captions.edits['0']).toBe('hallo');

    await c.undo();
    expect(strip(c.getState().project)).toEqual(strip(original));
    await c.redo();
    expect(strip(c.getState().project)).toEqual(strip(edited));
    expect(c.getState().project.captions.edits['0']).toBe('hallo');
    c.dispose();
  });

  it('a failed save leaves the undo stack and saved state honest', async () => {
    const backend = fakeBackend();
    backend.api.saveClipProject = async () => {
      throw new Error('network down');
    };
    const toasts = [];
    const c = makeController(backend, toasts);
    await c.load();

    await expect(
      c.applyEdit('edit word', (p) => {
        p.captions.edits['0'] = 'x';
        return p;
      })
    ).rejects.toThrow('network down');

    expect(c.canUndo()).toBe(false); // failed call never touched the stack
    expect(c.getState().project.captions.edits['0']).toBeUndefined();
    expect(c.getState().dirty).toBe(false);
    expect(toasts.some((m) => /couldn't save/i.test(m))).toBe(true);
    c.dispose();
  });
});

describe('projectController: optimistic concurrency', () => {
  it('a patch computed against an old version is rejected (409), triggers reload, never overwrites', async () => {
    const backend = fakeBackend();
    const toasts = [];
    const c = makeController(backend, toasts);
    await c.load(); // v1

    // Another client saves twice behind our back.
    backend.bumpServer((s) => {
      s.captions.edits['9'] = 'theirs';
    });
    backend.bumpServer((s) => {
      s.captions.edits['9'] = 'theirs2';
    });
    expect(backend.serverVersion()).toBe(3);

    // Stale copilot patch computed against v1.
    const stalePatch = makeProject(1);
    stalePatch.captions.edits['0'] = 'STALE';
    await expect(
      c.applyEdit('copilot: stale', () => stalePatch, { expectedVersion: 1 })
    ).rejects.toMatchObject({ status: 409 });

    // Reload happened: the latest server state is adopted.
    await vi.waitFor(() => expect(c.getState().project?.version).toBe(3));
    expect(c.getState().project.captions.edits['9']).toBe('theirs2');
    // The server was never overwritten with the stale content.
    expect(backend.serverProject().captions.edits['0']).not.toBe('STALE');
    expect(backend.serverVersion()).toBe(3);
    expect(toasts.some((m) => /someone else saved first/i.test(m))).toBe(true);
    c.dispose();
  });
});

describe('projectController: export drain + identity', () => {
  it('export waits for queued edits: an edit queued just before export is included', async () => {
    const backend = fakeBackend();
    let releaseSave;
    backend.gates.save = new Promise((res) => {
      releaseSave = res;
    });
    const c = makeController(backend);
    await c.load();

    const editP = c.applyEdit('edit word', (p) => {
      p.captions.edits['0'] = 'hallo';
      return p;
    });
    const exportP = c.exportProject(); // queued behind the edit
    await tick();
    await tick();
    expect(backend.composes).toHaveLength(0); // export has not run yet

    releaseSave();
    backend.gates.save = null;
    await editP;
    const res = await exportP;
    expect(backend.composes).toHaveLength(1);
    expect(backend.composes[0].captions.edits['0']).toBe('hallo');
    expect(res.identity.version).toBe(backend.composes[0].version);
    c.dispose();
  });

  it('an edit made while export is in flight does not change the export identity', async () => {
    const backend = fakeBackend();
    let releaseCompose;
    backend.gates.compose = new Promise((res) => {
      releaseCompose = res;
    });
    const c = makeController(backend);
    await c.load();
    await c.applyEdit('edit word', (p) => {
      p.captions.edits['0'] = 'hallo';
      return p;
    });

    const exportP = c.exportProject();
    // Wait until the export is actually inside the compose call.
    await vi.waitFor(() => expect(backend.composeCalls).toHaveLength(1));

    // A late edit lands while compose is in flight: it must serialize
    // behind the export, never into it.
    const lateEditP = c.applyEdit('late edit', (p) => {
      p.captions.edits['1'] = 'LATE';
      return p;
    });
    await tick();
    await tick();
    expect(backend.saves).toHaveLength(1); // the late edit's save has NOT run

    releaseCompose();
    backend.gates.compose = null;
    const res = await exportP;

    // The in-flight export composed the pre-edit project...
    expect(backend.composes).toHaveLength(1);
    expect(backend.composes[0].captions.edits['0']).toBe('hallo');
    expect(backend.composes[0].captions.edits['1']).toBeUndefined();
    // ...with its identity frozen at snapshot time.
    expect(res.identity.version).toBe(backend.composes[0].version);
    expect(res.identity.idempotency_key).toBe(backend.composes[0].idempotency_key);

    // The late edit is applied afterwards -- not lost, not mixed in.
    await lateEditP;
    expect(backend.saves).toHaveLength(2);
    expect(c.getState().project.captions.edits['1']).toBe('LATE');
    c.dispose();
  });
});

describe('projectController: version restore', () => {
  it('restore loads the chosen version, marks dirty, keeps its idempotency key', async () => {
    const backend = fakeBackend();
    const c = makeController(backend);
    await c.load();
    await c.applyEdit('edit word', (p) => {
      p.captions.edits['0'] = 'hallo';
      return p;
    });
    await c.applyEdit('trim', (p) => {
      p.segments[0].end = 20;
      return p;
    });
    expect(backend.serverVersion()).toBe(3);

    const restored = await c.restoreVersion(1);
    expect(restored.version).toBe(1);
    const st = c.getState();
    expect(st.project.version).toBe(1);
    expect(st.dirty).toBe(true);
    // The restored version's key is kept: an unchanged export dedups.
    expect(st.project.idempotency_key).toBe('key-v1');

    // Export saves the restored draft first (with a VALID expectedVersion --
    // the last acknowledged one, not the stale restored one), then composes
    // exactly the restored content.
    const res = await c.exportProject();
    expect(backend.composes).toHaveLength(1);
    expect(backend.composes[0].idempotency_key).toBe('key-v1');
    expect(backend.composes[0].version).toBe(4);
    expect(res.identity.idempotency_key).toBe('key-v1');
    expect(c.getState().dirty).toBe(false);
    c.dispose();
  });

  it('resetToAiCut loads the earliest auto version and marks dirty', async () => {
    const backend = fakeBackend();
    const c = makeController(backend);
    await c.load();
    const p = await c.resetToAiCut();
    expect(p.version).toBe(1); // earliest origin=auto in the fake list
    expect(c.getState().dirty).toBe(true);
    c.dispose();
  });
});

describe('projectController: load / backfill / errors', () => {
  it('404 on load triggers one backfill then retry', async () => {
    const backend = fakeBackend();
    let calls = 0;
    backend.api.getClipProject = async () => {
      calls += 1;
      if (calls === 1) {
        const e = new Error('not found');
        e.status = 404;
        throw e;
      }
      return makeProject(1);
    };
    const c = makeController(backend);
    const p = await c.load();
    expect(backend.backfills).toHaveLength(1);
    expect(p.version).toBe(1);
    expect(c.getState().status).toBe('ready');
    expect(c.getState().loadError).toBe('');
    c.dispose();
  });

  it('non-404 load errors stay visible', async () => {
    const backend = fakeBackend();
    backend.api.getClipProject = async () => {
      const e = new Error('boom');
      e.status = 500;
      throw e;
    };
    const c = makeController(backend);
    await expect(c.load()).rejects.toThrow('boom');
    expect(c.getState().status).toBe('error');
    expect(c.getState().loadError).toMatch(/boom/);
    c.dispose();
  });
});

describe('projectController: generation tokens / session reset', () => {
  it('a late load response for a previous clip never touches the new clip state', async () => {
    let releaseA;
    const gateA = new Promise((res) => {
      releaseA = res;
    });
    const apiFor = (tag) => ({
      getClipProject: async () => {
        if (tag === 'A') await gateA; // clip A hangs
        const p = makeProject(1);
        p.clipTag = tag;
        return p;
      },
      saveClipProject: async () => {
        throw new Error('not needed');
      },
      composeProject: async () => {
        throw new Error('not needed');
      },
      backfillClipProjects: async () => ({}),
      listClipProjectVersions: async () => ({ versions: [] }),
      editClipAI: async () => ({}),
      uploadAudio: async () => ({}),
    });
    const cA = createProjectController({ jobId: 'j', clipIndex: 0, api: apiFor('A'), onToast: () => {} });
    const cB = createProjectController({ jobId: 'j', clipIndex: 1, api: apiFor('B'), onToast: () => {} });
    const seenB = [];
    cB.subscribe((s) => seenB.push(s.project?.clipTag ?? null));

    const loadA = cA.load(); // in flight, hanging
    await cB.load(); // completes immediately
    expect(cB.getState().project.clipTag).toBe('B');

    releaseA(); // A's late response lands
    await loadA;
    await tick();
    // B untouched: every observed B state still belongs to clip B (or null).
    expect(seenB.every((t) => t === null || t === 'B')).toBe(true);
    expect(cB.getState().project.clipTag).toBe('B');
    // A got its own response.
    expect(cA.getState().project.clipTag).toBe('A');
    cA.dispose();
    cB.dispose();
  });

  it('dispose invalidates in-flight work: unsubscribed views get no late state', async () => {
    let release;
    const gate = new Promise((res) => {
      release = res;
    });
    const backend = fakeBackend();
    backend.api.getClipProject = async () => {
      await gate;
      return makeProject(1);
    };
    const c = makeController(backend);
    const seen = [];
    const unsub = c.subscribe((s) => seen.push(s.status));
    const loadP = c.load();
    // Clip switch: the view unsubscribes and disposes the old controller.
    unsub();
    c.dispose();
    release();
    await loadP; // resolves null -- superseded
    await tick();
    // No state was pushed after unsubscribe; the late response was dropped.
    expect(seen).not.toContain('ready');
    expect(c.getState().project).toBeNull();
    c.dispose();
  });

  it('resetSession clears queue history, project, and blocks late responses', async () => {
    let release;
    const gate = new Promise((res) => {
      release = res;
    });
    const backend = fakeBackend();
    backend.api.getClipProject = async () => {
      await gate;
      return makeProject(1);
    };
    const c = makeController(backend);
    const loadP = c.load();
    c.resetSession(); // sign-out while loading
    release();
    await loadP; // resolves null -- superseded
    await tick();
    expect(c.getState().project).toBeNull();
    expect(c.getState().status).toBe('loading');
    expect(c.canUndo()).toBe(false);
    c.dispose();
  });

  it('resetSession drops undo history after real edits', async () => {
    const backend = fakeBackend();
    const c = makeController(backend);
    await c.load();
    await c.applyEdit('edit word', (p) => {
      p.captions.edits['0'] = 'x';
      return p;
    });
    expect(c.canUndo()).toBe(true);
    c.resetSession();
    expect(c.canUndo()).toBe(false);
    expect(c.canRedo()).toBe(false);
    expect(c.getState().project).toBeNull();
    c.dispose();
  });
});
