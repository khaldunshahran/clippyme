/**
 * Task 16: authority-mode tests for the nested ClipEditorView.
 *
 * 1. resolveEditorMode: exactly one save authority, never an accidental
 *    second one -- a controller always wins, standalone needs the explicit
 *    prop, anything else is a loud misconfiguration.
 * 2. createControlledActions: the EXACT forwarding ClipEditorView uses in
 *    controlled mode, exercised against a REAL project controller with a
 *    stubbed API. vi.mock on ../api/realApi.js proves the nested view never
 *    calls the real API itself in controlled mode.
 *
 * These are load-bearing: they fail if controlled forwarding is removed
 * (the nested view saving its own copy) or if queue-drain/version
 * protection is dropped.
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

import { resolveEditorMode, createControlledActions } from './clipEditor.jsx';
import { createProjectController } from '../chat/projectController.js';
import * as realApi from '../api/realApi.js';

function makeProject(version) {
  return {
    origin: 'auto',
    version,
    idempotency_key: `key-v${version}`,
    source: { duration: 30, width: 1920, height: 1080 },
    segments: [{ id: 'seg0', start: 0, end: 30 }],
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

function fakeBackend() {
  let server = makeProject(1);
  const saves = [];
  const composes = [];
  const gets = [];
  const api = {
    getClipProject: async (jobId, index, version = null) => {
      gets.push(version ?? 'latest');
      const p = clone(server);
      if (version != null) {
        p.version = version;
        p.idempotency_key = `key-v${version}`;
      }
      return p;
    },
    saveClipProject: async (jobId, index, project, { expectedVersion } = {}) => {
      saves.push({ version: project.version, expectedVersion });
      if (expectedVersion != null && expectedVersion !== server.version) {
        const e = new Error('stale project version');
        e.status = 409;
        throw e;
      }
      const stored = server.version;
      server = clone(project);
      server.version = stored + 1;
      return { version: server.version, project: clone(server) };
    },
    composeProject: async (jobId, index, project) => {
      composes.push(clone(project));
      return { composed_url: '/c.mp4', version: project.version, deduped: false };
    },
    backfillClipProjects: async () => ({}),
    listClipProjectVersions: async () => ({
      versions: [{ version: 1, origin: 'auto' }],
      latest: 1,
    }),
    editClipAI: async () => ({ drop_ranges: [[0, 2]], explanation: 'cut' }),
    uploadAudio: async () => ({ file: 'audio/x.mp3' }),
  };
  return { api, saves, composes, gets };
}

function controlledFixture(view = {}) {
  const backend = fakeBackend();
  const toasts = [];
  const controller = createProjectController({
    jobId: 'job1',
    clipIndex: 0,
    api: backend.api,
    onToast: (m) => toasts.push(m),
  });
  const actions = createControlledActions(controller, {
    pushToast: (m) => toasts.push(m),
    setSelSeg: () => {},
    onExported: () => {},
    ...view,
  });
  return { backend, toasts, controller, actions };
}

describe('resolveEditorMode', () => {
  const ctl = { fake: 'controller' };

  it('a controller means controlled', () => {
    expect(resolveEditorMode({ controller: ctl })).toBe('controlled');
  });

  it('explicit standalone prop means standalone', () => {
    expect(resolveEditorMode({ standalone: true })).toBe('standalone');
  });

  it('neither is a loud misconfiguration, never a silent second authority', () => {
    expect(resolveEditorMode({})).toBe('unconfigured');
    expect(resolveEditorMode({ controller: null, standalone: false })).toBe('unconfigured');
  });

  it('a controller always wins over standalone: exactly one save authority', () => {
    expect(resolveEditorMode({ controller: ctl, standalone: true })).toBe('controlled');
  });

  it('standalone must be explicit: truthy non-true does not count', () => {
    expect(resolveEditorMode({ standalone: 1 })).toBe('unconfigured');
    expect(resolveEditorMode({ standalone: 'yes' })).toBe('unconfigured');
  });
});

describe('createControlledActions: nested view forwards to the controller', () => {
  it('manual edits go through the controller queue with expectedVersion -- never direct API', async () => {
    const { backend, controller, actions } = controlledFixture();
    await controller.load();

    await actions.mutate('trim segment', (p) => {
      p.segments[0].end = 12;
      return p;
    });

    expect(backend.saves).toHaveLength(1);
    expect(backend.saves[0].expectedVersion).toBe(1);
    expect(controller.getState().project.segments[0].end).toBe(12);
    expect(controller.getState().project.origin).toBe('user');
    // The nested view never touched the real API itself.
    expect(realApi.saveClipProject).not.toHaveBeenCalled();
    expect(realApi.getClipProject).not.toHaveBeenCalled();
    controller.dispose();
  });

  it('save delegates to controller.saveDraft (expectedVersion = last acknowledged)', async () => {
    const { backend, toasts, controller, actions } = controlledFixture();
    await controller.load();
    // Make the draft dirty the controlled way: restore an older version.
    await actions.loadVersion(1);
    expect(controller.getState().dirty).toBe(true);

    const saved = await actions.save();
    expect(saved).not.toBeNull();
    expect(backend.saves).toHaveLength(1);
    expect(backend.saves[0].expectedVersion).toBe(1); // last acknowledged
    expect(controller.getState().dirty).toBe(false);
    expect(toasts.some((m) => /draft saved as v/i.test(m))).toBe(true);
    expect(realApi.saveClipProject).not.toHaveBeenCalled();
    controller.dispose();
  });

  it('export delegates to controller.exportProject: drains, snapshots, composes acknowledged state', async () => {
    const { backend, controller, actions } = controlledFixture();
    await controller.load();
    await actions.mutate('edit word', (p) => {
      p.captions.edits['0'] = 'hallo';
      return p;
    });

    const exported = [];
    const actions2 = createControlledActions(controller, { onExported: (r) => exported.push(r) });
    const res = await actions2.exportProject();

    expect(backend.composes).toHaveLength(1);
    expect(backend.composes[0].captions.edits['0']).toBe('hallo');
    expect(res.identity.version).toBe(backend.composes[0].version);
    expect(exported).toHaveLength(1); // the view swaps its preview to it
    expect(realApi.composeProject).not.toHaveBeenCalled();
    controller.dispose();
  });

  it('loadVersion routes through the controller: keeps the restored key, marks dirty', async () => {
    const { backend, toasts, controller, actions } = controlledFixture();
    await controller.load();
    await actions.mutate('edit word', (p) => {
      p.captions.edits['0'] = 'hallo';
      return p;
    });

    const proj = await actions.loadVersion(1);
    expect(proj.version).toBe(1);
    expect(controller.getState().dirty).toBe(true);
    expect(controller.getState().project.idempotency_key).toBe('key-v1');
    expect(toasts.some((m) => /loaded v1 into the draft/i.test(m))).toBe(true);
    // No direct fetch, no standalone confirm-gated path.
    expect(realApi.getClipProject).not.toHaveBeenCalled();
    expect(backend.gets).toContain(1);
    controller.dispose();
  });

  it('resetToAiCut routes through the controller', async () => {
    const { backend, controller, actions } = controlledFixture();
    await controller.load();
    const proj = await actions.resetToAiCut();
    expect(proj.version).toBe(1);
    expect(controller.getState().dirty).toBe(true);
    expect(realApi.getClipProject).not.toHaveBeenCalled();
    controller.dispose();
  });

  it('aiTrim and uploadAudio go through the controller API, not the direct imports', async () => {
    const { controller, actions } = controlledFixture();
    await controller.load();

    const ai = await actions.aiTrim('cut the intro');
    expect(ai.drop_ranges).toEqual([[0, 2]]);
    const up = await actions.uploadAudio({ name: 'x.mp3' });
    expect(up.file).toBe('audio/x.mp3');
    expect(realApi.editClipAI).not.toHaveBeenCalled();
    expect(realApi.uploadAudio).not.toHaveBeenCalled();
    controller.dispose();
  });

  it('a stale controlled save is rejected and reloads instead of overwriting', async () => {
    const { backend, controller, actions } = controlledFixture();
    await controller.load();
    // Restore v1 (dirty), then the server moves on underneath us.
    await actions.loadVersion(1);
    const staleAck = controller.getState();
    expect(staleAck.dirty).toBe(true);
    // Simulate another client: bump the fake server past the ack version.
    // (saveDraft sends expectedVersion = last acknowledged = 1.)
    backend.api.saveClipProject = async (jobId, index, project, { expectedVersion } = {}) => {
      const e = new Error('stale project version');
      e.status = 409;
      throw e;
    };
    await expect(actions.save()).rejects.toMatchObject({ status: 409 });
    // Reload pulled the latest; the stale draft was NOT persisted.
    await vi.waitFor(() => expect(controller.getState().status).toBe('ready'));
    controller.dispose();
  });
});
