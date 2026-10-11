/**
 * UI-01 review-01 — real controller lifecycle contracts (no DOM).
 *
 * Exercises the REAL createProjectController + the REAL serialized edit
 * queue (editor-timeline/undo.js) against an injected stub API. These pin
 * the identity semantics the UI-01 header badges bind to:
 *
 *  1. A deferred save through the queue updates the acknowledged version
 *     and clears dirty (the header's "Saved" / ack badge source).
 *  2. A version restore keeps the server-acknowledged version while the
 *     draft holds historical content, and marks dirty (the header must
 *     show the ACK version, with the draft version distinguished
 *     separately).
 *  3. Dispose is terminal: new operations reject, so a clip switch can
 *     never save through the old controller.
 *  4. Export records the composed snapshot's identity (what the
 *     "Exact vN render" badge binds to).
 *
 * Load-bearing mutations (each must fail, then the suite re-runs green):
 *  M-a: onUpdate stops advancing ackVersion (badge would go stale)
 *  M-b: restoreVersion overwrites ackVersion with the restored version
 *  M-c: dispose stops rejecting new operations
 */
import { describe, expect, it, vi } from 'vitest';
import { createProjectController } from './projectController.js';

function makeProject(version) {
  return {
    origin: 'auto',
    version,
    idempotency_key: `key-v${version}`,
    source: { duration: 30, width: 1920, height: 1080 },
    segments: [{ id: 'seg0', start: 0, end: 30, crop: { x: 0, y: 0, w: 608, h: 1080 } }],
    captions: {
      style: 'classic_white',
      position: 'bottom',
      words: [{ id: 'w0', w: 'hello', start: 1, end: 1.5 }],
      edits: {},
    },
    overlays: [],
    audio: [],
  };
}

const clone = (p) => JSON.parse(JSON.stringify(p));
const tick = () => new Promise((r) => setTimeout(r, 0));

/** Fake backend: versioned store, history, controllable save gate. */
function fakeBackend() {
  let server = makeProject(3);
  const history = { 1: makeProject(1), 2: makeProject(2), 3: clone(server) };
  const saves = [];
  let saveGate = null;
  const api = {
    getClipProject: async (jobId, index, version = null) => {
      if (version != null && history[version]) return clone(history[version]);
      return clone(server);
    },
    saveClipProject: async (jobId, index, project, { expectedVersion } = {}) => {
      if (saveGate) await saveGate;
      saves.push({ version: project.version, expectedVersion });
      if (expectedVersion != null && expectedVersion !== server.version) {
        const e = new Error(`stale project version (expected ${expectedVersion}, stored ${server.version})`);
        e.status = 409;
        throw e;
      }
      server = clone(project);
      server.version = history[3].version + saves.length; // 4, 5, ...
      history[server.version] = clone(server);
      return { version: server.version, project: clone(server) };
    },
    composeProject: async (jobId, index, project) => ({
      version: project.version,
      composed_url: `/render-v${project.version}.mp4`,
      deduped: false,
    }),
    backfillClipProjects: async () => {},
    listClipProjectVersions: async () => ({
      versions: [1, 2, 3].map((v) => ({ version: v, origin: v === 1 ? 'auto' : 'user' })),
    }),
    editClipAI: async () => ({ drop_ranges: [], explanation: '' }),
    uploadAudio: async () => ({ file: 'x.mp3' }),
  };
  return { api, saves, setSaveGate: (p) => { saveGate = p; }, get server() { return server; } };
}

function makeController(backend) {
  const c = createProjectController({
    jobId: 'j1',
    clipIndex: 0,
    api: backend.api,
    onToast: () => {},
  });
  return c;
}

describe('UI-01 real controller lifecycle (injected API)', () => {
  it('deferred save through the queue updates ackVersion and clears dirty', async () => {
    const backend = fakeBackend();
    const c = makeController(backend);
    await c.load();
    expect(c.getState().ackVersion).toBe(3);

    // A chat/transcript-side save: restore an old version (dirty), then
    // save with the save deferred behind a gate.
    await c.restoreVersion(1);
    expect(c.getState().dirty).toBe(true);

    let release;
    backend.setSaveGate(new Promise((r) => { release = r; }));
    const saving = c.saveDraft();
    await tick();
    await tick();
    // Still in flight: the header would show "Saving…".
    expect(c.getState().saving).toBe(true);
    release();
    const saved = await saving;
    expect(saved.version).toBe(4);
    const snap = c.getState();
    expect(snap.saving).toBe(false);
    expect(snap.dirty).toBe(false);
    expect(snap.ackVersion).toBe(4);
    expect(snap.project.version).toBe(4);
    // The save carried the last acknowledged version for concurrency.
    expect(backend.saves[0].expectedVersion).toBe(3);
    c.dispose();
  });

  it('restore keeps the acknowledged version while the draft holds history', async () => {
    const backend = fakeBackend();
    const c = makeController(backend);
    await c.load();
    await c.restoreVersion(2);
    const dirtySnap = c.getState();
    expect(dirtySnap.dirty).toBe(true);
    expect(dirtySnap.project.version).toBe(2);
    // ackVersion stays at the last acknowledged version (3 here), never
    // the restored draft's version.
    expect(dirtySnap.ackVersion).toBe(3);
    c.dispose();
  });

  it('dispose is terminal: new operations reject with DISPOSED', async () => {
    const backend = fakeBackend();
    const c = makeController(backend);
    await c.load();
    c.dispose();
    await expect(c.applyEdit('x', (p) => p)).rejects.toMatchObject({ code: 'DISPOSED' });
    await expect(c.saveDraft()).rejects.toMatchObject({ code: 'DISPOSED' });
    await expect(c.exportProject()).rejects.toMatchObject({ code: 'DISPOSED' });
    await expect(c.restoreVersion(1)).rejects.toMatchObject({ code: 'DISPOSED' });
  });

  it('export records the composed snapshot identity', async () => {
    const backend = fakeBackend();
    const c = makeController(backend);
    await c.load();
    await c.applyEdit('trim', (p) => {
      p.segments[0].start = 1;
      return p;
    });
    const res = await c.exportProject();
    // The export composes the acknowledged snapshot and records whose
    // render it is — the preview badge's identity source.
    expect(res.identity.version).toBe(4);
    expect(res.identity.idempotency_key).toBe(c.getState().project.idempotency_key);
    expect(res.composed_url).toBe('/render-v4.mp4');
    expect(c.getState().lastExport.identity.version).toBe(4);
    c.dispose();
  });
});
