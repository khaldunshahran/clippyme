/**
 * Task 17 — single-clip chat editing workspace regressions.
 *
 * Coordinator-probe style: the pure modules (messageRouter, proposals,
 * selection) are tested directly with no mocks; apply paths run against
 * the REAL createProjectController with an injected fake backend (true
 * 409 semantics, following the task16 fixture pattern); and a minimal
 * hook harness drives the REAL useChat.sendMessage for the
 * hyperframes/bulk/need-selection/edit branches, asserting zero provider
 * calls where the contract forbids them.
 *
 * Load-bearing mutations (each must fail, then the suite re-runs green):
 *  M1: selection+URL routes to 'new-source'            (router order)
 *  M2: applyChatProposal drops the identity check      (STALE_SELECTION)
 *  M3: applyChatProposal drops expectedVersion         (409 guard)
 *  M4: /hyperframes branch calls editAiPatch           (zero provider calls)
 */
import { describe, expect, it, vi, beforeEach } from 'vitest';

// --- minimal hook harness: persistent useState/useRef, identity
// useMemo/useCallback, immediate useEffect (node: no window/document) ---
const hooks = vi.hoisted(() => ({ values: [], index: 0 }));
const sessionResetCbs = vi.hoisted(() => []);
const apiMocks = vi.hoisted(() => ({
  editAiPatch: vi.fn(),
  validateUrl: vi.fn(),
  submitJob: vi.fn(),
  getProgress: vi.fn(),
  steerJob: vi.fn(),
  getHistory: vi.fn(async () => []),
  getJobClips: vi.fn(async () => []),
}));
const storage = vi.hoisted(() => {
  let store = {};
  return {
    getItem: (k) => (k in store ? store[k] : null),
    setItem: (k, v) => { store[k] = String(v); },
    removeItem: (k) => { delete store[k]; },
    _reset: () => { store = {}; },
    _dump: () => ({ ...store }),
  };
});
vi.stubGlobal('localStorage', storage);

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
// chatApi: keep the REAL pure helpers (extractUrl, normClip); mock transport.
vi.mock('../api/chatApi.js', async (importOriginal) => {
  const actual = await importOriginal();
  return { ...actual, ...apiMocks };
});
vi.mock('../api/sessionReset.js', () => ({
  onSessionReset: (cb) => { sessionResetCbs.push(cb); return () => {}; },
}));

import { routeChatMessage } from './messageRouter.js';
import { requestChatProposal, applyChatProposal } from './proposals.js';
import {
  loadSelection, saveSelection, clearStoredSelection, isValidSelection,
  SELECTION_LS_KEY,
} from './selection.js';
import { createProjectController } from './projectController.js';
import { useChat } from './useChat.js';

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

/**
 * Fake backend with real optimistic-concurrency (409 on expectedVersion
 * mismatch) and call counters, following the task16 fixture pattern.
 */
function makeFixture(tag = 'A', clipIndex = 0, jobId = 'j') {
  let server = project(1, tag);
  const saves = [];
  const composes = [];
  let getCalls = 0;
  const api = {
    getClipProject: async (j, i, v) => {
      getCalls += 1;
      return v == null ? clone(server) : { ...clone(server), version: v };
    },
    saveClipProject: async (j, i, p, o) => {
      saves.push({ p: clone(p), expectedVersion: o?.expectedVersion, jobId: j, clipIndex: i });
      if (o?.expectedVersion !== server.version) {
        throw Object.assign(new Error('stale project version'), { status: 409 });
      }
      server = { ...clone(p), version: server.version + 1 };
      return { project: clone(server), version: server.version };
    },
    composeProject: async (j, i, p) => {
      composes.push(clone(p));
      return { composed_url: '/x.mp4' };
    },
    listClipProjectVersions: async () => ({ versions: [] }),
    backfillClipProjects: async () => {},
    editClipAI: async () => { throw new Error('no AI in fixture'); },
    uploadAudio: async () => ({ ok: true }),
  };
  const c = createProjectController({ jobId, clipIndex, api });
  return {
    c, api, saves, composes,
    getCalls: () => getCalls,
    getServer: () => clone(server),
  };
}

const SEL = (clipIndex = 0) => ({ threadId: 't1', jobId: 'job1', clipIndex });

beforeEach(() => {
  hooks.values = [];
  hooks.index = 0;
  sessionResetCbs.length = 0;
  storage._reset();
  Object.values(apiMocks).forEach((m) => m.mockReset?.());
  apiMocks.getHistory.mockResolvedValue([]);
  apiMocks.getJobClips.mockResolvedValue([]);
});

// =====================================================================
describe('messageRouter — the real dispatcher', () => {
  it('steers a running job (clipping/submitting + jobId)', () => {
    expect(routeChatMessage('focus on funny', { phase: 'clipping', jobId: 'j' }).type).toBe('steer');
    expect(routeChatMessage('focus on funny', { phase: 'submitting', jobId: 'j' }).type).toBe('steer');
    // steering wins over everything else, even /hyperframes
    expect(routeChatMessage('/hyperframes x', { phase: 'clipping', jobId: 'j' }).type).toBe('steer');
  });

  it('does not steer without a job id', () => {
    expect(routeChatMessage('https://example.com/v', { phase: 'clipping', jobId: null }).type)
      .toBe('new-source');
  });

  it('recognizes the exact /hyperframes command', () => {
    expect(routeChatMessage('/hyperframes make it cinematic', { phase: 'done' }).type).toBe('hyperframes');
    expect(routeChatMessage('  /hyperframes', { phase: 'idle' }).type).toBe('hyperframes');
    expect(routeChatMessage('/HYPERFRAMES x', { phase: 'done', selection: SEL() }).type).toBe('hyperframes');
    // not a bare word-boundary match: no false positive
    expect(routeChatMessage('/hyperframesx', { phase: 'idle' }).type).toBe('idle-hint');
  });

  it('routes bulk scope to bulk (routing alone never mutates)', () => {
    for (const t of [
      'make all clips punchier',
      'apply this to every clip please',
      'bulk edit these clips',
      'each clip needs louder captions',
      'these selected clips are too dark',
    ]) {
      expect(routeChatMessage(t, { phase: 'done', selection: SEL() }).type).toBe('bulk');
      expect(routeChatMessage(t, { phase: 'done' }).type).toBe('bulk');
    }
    // "all" without a clip scope is not bulk
    expect(routeChatMessage('make it all punchier', { phase: 'done', selection: SEL() }).type).toBe('edit');
  });

  it('edits with a selection — even when the text contains a URL', () => {
    const r = routeChatMessage('make the hook punchier', { phase: 'done', selection: SEL(2) });
    expect(r.type).toBe('edit');
    expect(r.selection).toEqual(SEL(2));
    // M1 load-bearing: an embedded link must NOT become a new source
    const r2 = routeChatMessage('use this style https://example.com/v', { phase: 'done', selection: SEL(0) });
    expect(r2.type).toBe('edit');
  });

  it('routes a URL without selection to new-source', () => {
    const r = routeChatMessage('try https://youtube.com/watch?v=x now', { phase: 'done' });
    expect(r.type).toBe('new-source');
    expect(r.url).toBe('https://youtube.com/watch?v=x');
  });

  it('asks for a selection after completion instead of guessing clip0', () => {
    expect(routeChatMessage('make it punchier', { phase: 'done' }).type).toBe('need-selection');
  });

  it('keeps the idle hint on a fresh thread', () => {
    expect(routeChatMessage('hello', { phase: 'idle' }).type).toBe('idle-hint');
  });

  it('stays silent during validating/settings/caption-pick/error', () => {
    for (const phase of ['validating', 'settings', 'caption-pick', 'error']) {
      expect(routeChatMessage('hello', { phase }).type).toBe('none');
    }
  });
});

// =====================================================================
describe('selection persistence — safe metadata only', () => {
  it('round-trips the three ids', () => {
    expect(saveSelection({ threadId: 't', jobId: 'j', clipIndex: 3 })).toBe(true);
    expect(loadSelection()).toEqual({ threadId: 't', jobId: 'j', clipIndex: 3 });
  });

  it('persists ONLY the three ids — never controller/queues/tokens', () => {
    saveSelection({ threadId: 't', jobId: 'j', clipIndex: 0, controller: { x: 1 }, token: 'secret' });
    const raw = storage.getItem(SELECTION_LS_KEY);
    expect(Object.keys(JSON.parse(raw)).sort()).toEqual(['clipIndex', 'jobId', 'threadId']);
  });

  it('rejects invalid shapes', () => {
    expect(isValidSelection(null)).toBe(false);
    expect(isValidSelection({})).toBe(false);
    expect(isValidSelection({ threadId: 't', jobId: 'j', clipIndex: -1 })).toBe(false);
    expect(isValidSelection({ threadId: 't', jobId: 'j', clipIndex: 1.5 })).toBe(false);
    expect(isValidSelection({ threadId: 't', jobId: 'j' })).toBe(false);
    expect(saveSelection({ threadId: 't', jobId: 'j', clipIndex: -1 })).toBe(false);
    expect(storage.getItem(SELECTION_LS_KEY)).toBeNull();
  });

  it('loadSelection returns null for malformed or extra-field payloads', () => {
    storage.setItem(SELECTION_LS_KEY, 'not-json{');
    expect(loadSelection()).toBeNull();
    storage.setItem(SELECTION_LS_KEY, JSON.stringify({ threadId: 't', jobId: 'j', clipIndex: 0, extra: 1 }));
    // valid triple still loads (only the three ids are ever read)
    expect(loadSelection()).toEqual({ threadId: 't', jobId: 'j', clipIndex: 0 });
  });

  it('clearStoredSelection removes the key', () => {
    saveSelection({ threadId: 't', jobId: 'j', clipIndex: 0 });
    clearStoredSelection();
    expect(storage.getItem(SELECTION_LS_KEY)).toBeNull();
    expect(loadSelection()).toBeNull();
  });
});

// =====================================================================
describe('requestChatProposal — proposal generation never saves', () => {
  it('returns the bound identity, base version, explanation and project', async () => {
    const editAiPatchFn = vi.fn(async () => ({
      explanation: 'Made the hook punchier.',
      patch: { hook: { text: 'WOW' } },
      project: project(4, 'P'),
    }));
    const saveClipProject = vi.fn();
    const res = await requestChatProposal({ editAiPatchFn, jobId: 'j', clipIndex: 2, instruction: 'punchier hook' });
    expect(editAiPatchFn).toHaveBeenCalledTimes(1);
    expect(editAiPatchFn).toHaveBeenCalledWith('j', 2, 'punchier hook');
    expect(res.identity).toEqual({ jobId: 'j', clipIndex: 2 });
    expect(res.baseVersion).toBe(4);
    expect(res.explanation).toBe('Made the hook punchier.');
    expect(res.project.version).toBe(4);
    expect(saveClipProject).not.toHaveBeenCalled();
  });

  it('throws on an unusable patch (no project / no version)', async () => {
    await expect(requestChatProposal({
      editAiPatchFn: async () => ({ explanation: 'oops', patch: {} }),
      jobId: 'j', clipIndex: 0, instruction: 'x',
    })).rejects.toThrow(/unusable/i);
    await expect(requestChatProposal({
      editAiPatchFn: async () => ({ project: { noversion: true } }),
      jobId: 'j', clipIndex: 0, instruction: 'x',
    })).rejects.toThrow(/unusable/i);
  });

  it('propagates provider errors honestly', async () => {
    await expect(requestChatProposal({
      editAiPatchFn: async () => { throw new Error('model exploded'); },
      jobId: 'j', clipIndex: 0, instruction: 'x',
    })).rejects.toThrow('model exploded');
  });
});

// =====================================================================
describe('applyChatProposal — real controller, true 409 semantics', () => {
  it('applies against the captured identity and advances the version', async () => {
    const f = makeFixture('A', 0);
    await f.c.load();
    const patched = { ...project(1, 'A'), hook: { text: 'WOW' } };
    await applyChatProposal({
      controller: f.c,
      identity: { jobId: 'j', clipIndex: 0 },
      baseVersion: 1,
      project: patched,
      label: 'chat: punchier hook',
    });
    expect(f.saves).toHaveLength(1);
    expect(f.saves[0].expectedVersion).toBe(1);
    expect(f.c.getState().ackVersion).toBe(2);
    expect(f.getServer().hook.text).toBe('WOW');
  });

  it('refuses a stale base version with 409 and never overwrites (M3 load-bearing)', async () => {
    const f = makeFixture('A', 0);
    await f.c.load();
    // A manual save moves the server to v2 while the proposal was based on v1.
    await f.c.applyEdit('manual caption fix', (p) => ({ ...p, captions: { edits: { w1: 'hi' } } }));
    expect(f.c.getState().ackVersion).toBe(2);
    const stalePatch = { ...project(1, 'A'), hook: { text: 'STALE' } };
    await expect(applyChatProposal({
      controller: f.c,
      identity: { jobId: 'j', clipIndex: 0 },
      baseVersion: 1,
      project: stalePatch,
      label: 'chat: stale',
    })).rejects.toMatchObject({ status: 409 });
    // The stale project was NOT written: server still holds the manual edit.
    expect(f.getServer().hook).toBeUndefined();
    expect(f.getServer().captions.edits).toEqual({ w1: 'hi' });
    expect(f.c.getState().ackVersion).toBe(2);
  });

  it('refuses a wrong-identity controller with STALE_SELECTION and ZERO api calls (M2 load-bearing)', async () => {
    const a = makeFixture('A', 0);
    const b = makeFixture('B', 1);
    await a.c.load();
    await b.c.load();
    const bSavesBefore = b.saves.length;
    const bGetsBefore = b.getCalls();
    const aSavesBefore = a.saves.length;
    const err = await applyChatProposal({
      controller: b.c, // live controller is clip 1…
      identity: { jobId: 'j', clipIndex: 0 }, // …proposal was made for clip 0
      baseVersion: 1,
      project: project(1, 'A'),
      label: 'chat: wrong target',
    }).then(() => null, (e) => e);
    expect(err).toBeTruthy();
    expect(err.code).toBe('STALE_SELECTION');
    // The refused apply issued ZERO new api calls on either controller.
    expect(b.saves.length).toBe(bSavesBefore);
    expect(b.getCalls()).toBe(bGetsBefore);
    expect(a.saves.length).toBe(aSavesBefore);
  });

  it('refuses when there is no live controller at all', async () => {
    const err = await applyChatProposal({
      controller: null,
      identity: { jobId: 'j', clipIndex: 0 },
      baseVersion: 1,
      project: project(1),
    }).then(() => null, (e) => e);
    expect(err?.code).toBe('STALE_SELECTION');
  });
});

// =====================================================================
describe('full agreement flow — proposal → apply → manual → undo → redo → export', () => {
  it('exported payload and version agree across every surface', async () => {
    const f = makeFixture('A', 0);
    await f.c.load();

    // 1. proposal (never saved by generation itself)
    const savesBefore = f.saves.length;
    const res = await requestChatProposal({
      editAiPatchFn: async () => ({
        explanation: 'Punchier hook.',
        patch: { hook: { text: 'WOW' } },
        project: { ...project(1, 'A'), hook: { text: 'WOW' } },
      }),
      jobId: 'j', clipIndex: 0, instruction: 'punchier hook',
    });
    expect(f.saves.length).toBe(savesBefore);

    // 2. apply the proposal through the shared authority
    await applyChatProposal({
      controller: f.c, identity: res.identity, baseVersion: res.baseVersion,
      project: res.project, label: 'chat: punchier hook',
    });
    const afterApply = f.c.getState().ackVersion;
    expect(f.getServer().hook.text).toBe('WOW');

    // 3. manual edit on top
    await f.c.applyEdit('manual: caption nudge', (p) => ({ ...p, manual: true }));
    const afterManual = f.c.getState().ackVersion;
    expect(afterManual).toBeGreaterThan(afterApply);
    expect(f.getServer().manual).toBe(true);

    // 4. undo restores the proposal-only content; 5. redo restores the manual edit
    await f.c.undo();
    expect(f.c.getState().project.manual).toBeUndefined();
    expect(f.c.getState().project.hook.text).toBe('WOW');
    await f.c.redo();
    expect(f.c.getState().project.manual).toBe(true);

    // 6. export drains the queue and composes EXACTLY the acknowledged project
    const out = await f.c.exportProject();
    const ack = f.c.getState().ackVersion;
    expect(f.composes).toHaveLength(1);
    expect(f.composes[0].version).toBe(ack);
    expect(out.identity.version).toBe(ack);
    expect(out.composed_url).toBe('/x.mp4');
    const lastExport = f.c.getState().lastExport;
    expect(lastExport.identity.version).toBe(ack);
  });
});

// =====================================================================
describe('switch / disposal / reset — no wrong-target apply, no stale preview', () => {
  it('a completed proposal at clip switch is refused at apply (identity-bound)', async () => {
    const a = makeFixture('A', 0);
    const b = makeFixture('B', 1);
    await a.c.load();
    await b.c.load();
    const res = await requestChatProposal({
      editAiPatchFn: async () => ({ project: project(1, 'A'), patch: {}, explanation: 'A edit' }),
      jobId: 'j', clipIndex: 0, instruction: 'edit A',
    });
    // User switched to clip 1 before applying: the live controller refuses.
    const err = await applyChatProposal({
      controller: b.c, identity: res.identity, baseVersion: res.baseVersion,
      project: res.project, label: 'chat: edit A',
    }).then(() => null, (e) => e);
    expect(err?.code).toBe('STALE_SELECTION');
    expect(b.saves).toHaveLength(0);
    expect(b.getServer().source.tag).toBe('B');
  });

  it('a disposed controller rejects the apply (terminal disposal)', async () => {
    const f = makeFixture('A', 0);
    await f.c.load();
    f.c.dispose();
    const err = await applyChatProposal({
      controller: f.c,
      identity: { jobId: 'j', clipIndex: 0 },
      baseVersion: 1,
      project: project(1, 'A'),
      label: 'chat: late',
    }).then(() => null, (e) => e);
    expect(err?.code).toBe('DISPOSED');
    expect(f.saves).toHaveLength(0);
  });

  it('resetSession then load works again (restart)', async () => {
    const f = makeFixture('A', 0);
    await f.c.load();
    f.c.resetSession();
    expect(f.c.getState().project).toBeNull();
    await f.c.load();
    await f.c.applyEdit('after reset', (p) => ({ ...p, ok: true }));
    expect(f.c.getState().ackVersion).toBe(2);
  });

  it('an export in flight at reset cannot record lastExport (pending export isolation)', async () => {
    const f = makeFixture('A', 0);
    await f.c.load();
    let release;
    const gate = new Promise((r) => { release = r; });
    const origCompose = f.api.composeProject;
    f.api.composeProject = async (j, i, p) => { await gate; return origCompose(j, i, p); };
    const exportP = f.c.exportProject();
    await vi.waitFor(() => expect(f.composes.length).toBe(0));
    f.c.resetSession();
    release();
    await expect(exportP).rejects.toMatchObject({ code: 'QUEUE_CANCELLED' });
    expect(f.c.getState().lastExport).toBeNull();
  });
});

// =====================================================================
describe('useChat host wiring — the real sendMessage', () => {
  function renderChat() {
    hooks.index = 0;
    return useChat();
  }
  // One thread, job done, two clips. Returns a fresh chat + the thread id.
  function setupDoneThread() {
    let chat = renderChat();
    const threadId = chat.createThread();
    chat = renderChat();
    chat.patchThread(threadId, {
      phase: 'done',
      jobId: 'job1',
      clips: [{ index: 0, title: 'Clip 1' }, { index: 1, title: 'Clip 2' }],
    });
    chat = renderChat();
    return { chat, threadId };
  }
  const threadOf = (chat, threadId) => chat.threads.find((t) => t.id === threadId);
  const lastAi = (chat, threadId) => {
    const msgs = threadOf(chat, threadId).messages;
    return msgs[msgs.length - 1];
  };

  it('hyperframes pushes an honest message with zero provider calls', async () => {
    const { threadId } = setupDoneThread();
    let chat = renderChat();
    await chat.sendMessage('/hyperframes make it cinematic');
    chat = renderChat();
    const m = lastAi(chat, threadId);
    expect(m.kind).toBe('hyperframes');
    expect(m.data.command).toBe('/hyperframes make it cinematic');
    expect(m.data.scope).toBeNull();
    expect(apiMocks.editAiPatch).not.toHaveBeenCalled();
    expect(apiMocks.steerJob).not.toHaveBeenCalled();
  });

  it('hyperframes with a selection reports the scope, still zero calls', async () => {
    let { chat, threadId } = setupDoneThread();
    chat.selectClip(threadId, 1);
    chat = renderChat();
    chat.workspaceRef.current = { controller: { jobId: 'job1', clipIndex: 1 } };
    await chat.sendMessage('/hyperframes slow-mo zoom');
    chat = renderChat();
    const m = lastAi(chat, threadId);
    expect(m.kind).toBe('hyperframes');
    expect(m.data.scope).toEqual({ threadId, jobId: 'job1', clipIndex: 1 });
    expect(apiMocks.editAiPatch).not.toHaveBeenCalled();
  });

  it('bulk scope opens a bulk card with zero provider calls (nothing saved until Apply)', async () => {
    const { threadId } = setupDoneThread();
    let chat = renderChat();
    await chat.sendMessage('make all clips punchier');
    chat = renderChat();
    const m = lastAi(chat, threadId);
    expect(m.kind).toBe('bulk');
    expect(m.data.scope).toEqual({ mode: 'all' });
    expect(m.data.jobId).toBe('job1');
    expect(apiMocks.editAiPatch).not.toHaveBeenCalled();
  });

  it('post-completion text without selection asks for a selection (never guesses clip0)', async () => {
    const { threadId } = setupDoneThread();
    let chat = renderChat();
    await chat.sendMessage('make it punchier');
    chat = renderChat();
    const m = lastAi(chat, threadId);
    expect(m.kind).toBe('notice');
    expect(m.text).toMatch(/select a clip/i);
    expect(apiMocks.editAiPatch).not.toHaveBeenCalled();
  });

  it('edit with selection calls editAiPatch once and fills a ready proposal', async () => {
    let { chat, threadId } = setupDoneThread();
    chat.selectClip(threadId, 0);
    chat = renderChat();
    expect(chat.selectedClip).toEqual({ threadId, jobId: 'job1', clipIndex: 0 });
    chat.workspaceRef.current = { controller: { jobId: 'job1', clipIndex: 0 } };
    apiMocks.editAiPatch.mockResolvedValue({
      explanation: 'Hook is punchier now.',
      patch: { hook: { text: 'WOW' } },
      project: { version: 3, hook: { text: 'WOW' } },
    });
    await chat.sendMessage('make the hook punchier');
    expect(apiMocks.editAiPatch).toHaveBeenCalledTimes(1);
    expect(apiMocks.editAiPatch).toHaveBeenCalledWith('job1', 0, 'make the hook punchier');
    chat = renderChat();
    const m = lastAi(chat, threadId);
    expect(m.kind).toBe('proposal');
    expect(m.data.state).toBe('ready');
    expect(m.data.baseVersion).toBe(3);
    expect(m.data.identity).toEqual({ jobId: 'job1', clipIndex: 0 });
    expect(m.data.explanation).toBe('Hook is punchier now.');
  });

  it('edit with selection keeps an embedded URL as an edit (M1 at host level)', async () => {
    let { chat, threadId } = setupDoneThread();
    chat.selectClip(threadId, 0);
    chat = renderChat();
    chat.workspaceRef.current = { controller: { jobId: 'job1', clipIndex: 0 } };
    apiMocks.editAiPatch.mockResolvedValue({ project: { version: 1 }, patch: {} });
    apiMocks.validateUrl.mockResolvedValue({ valid: true, downloadable: true, title: 'X' });
    await chat.sendMessage('like this https://example.com/v but punchier');
    expect(apiMocks.editAiPatch).toHaveBeenCalledTimes(1);
    expect(apiMocks.validateUrl).not.toHaveBeenCalled();
    chat = renderChat();
    expect(chat.selectedClip).toEqual({ threadId, jobId: 'job1', clipIndex: 0 });
  });

  it('new-source flow clears the selection and validates (existing flow preserved)', async () => {
    let { chat, threadId } = setupDoneThread();
    chat.selectClip(threadId, 0);
    chat = renderChat();
    expect(chat.selectedClip).not.toBeNull();
    chat.clearSelection(); // the explicit New-source action
    chat = renderChat();
    apiMocks.validateUrl.mockResolvedValue({ valid: false, downloadable: false, reason: 'nope' });
    await chat.sendMessage('https://example.com/new-video');
    expect(apiMocks.validateUrl).toHaveBeenCalledWith('https://example.com/new-video');
    expect(apiMocks.editAiPatch).not.toHaveBeenCalled();
    chat = renderChat();
    expect(chat.selectedClip).toBeNull();
    expect(storage.getItem(SELECTION_LS_KEY)).toBeNull();
  });

  it('running-job steering is preserved (route-level)', async () => {
    let chat = renderChat();
    const threadId = chat.createThread();
    chat = renderChat();
    chat.patchThread(threadId, { phase: 'clipping', jobId: 'job9' });
    chat = renderChat();
    apiMocks.steerJob.mockResolvedValue({ accepted: true });
    await chat.sendMessage('focus on the funny moments');
    expect(apiMocks.steerJob).toHaveBeenCalledWith('job9', 'focus on the funny moments');
    expect(apiMocks.editAiPatch).not.toHaveBeenCalled();
    chat = renderChat();
    expect(lastAi(chat, threadId).text).toMatch(/steer the clip selection/i);
  });

  it('applyProposal through the host marks applied on the same controller', async () => {
    let { chat, threadId } = setupDoneThread();
    chat.selectClip(threadId, 0);
    const f = makeFixture('A', 0, 'job1');
    await f.c.load();
    chat = renderChat();
    chat.workspaceRef.current = { controller: f.c };
    apiMocks.editAiPatch.mockResolvedValue({
      explanation: 'Done.',
      patch: {},
      project: { ...project(1, 'A'), hook: { text: 'WOW' } },
    });
    await chat.sendMessage('punchier hook');
    chat = renderChat();
    const proposal = lastAi(chat, threadId);
    expect(proposal.data.state).toBe('ready');
    await chat.applyProposal(threadId, proposal.id);
    chat = renderChat();
    const after = threadOf(chat, threadId).messages.find((m) => m.id === proposal.id);
    expect(after.data.state).toBe('applied');
    expect(f.getServer().hook.text).toBe('WOW');
  });

  it('apply after a clip switch is refused as stale with zero saves', async () => {
    let { chat, threadId } = setupDoneThread();
    chat.selectClip(threadId, 0);
    const a = makeFixture('A', 0, 'job1');
    const b = makeFixture('B', 1, 'job1');
    await a.c.load();
    await b.c.load();
    chat = renderChat();
    chat.workspaceRef.current = { controller: a.c };
    apiMocks.editAiPatch.mockResolvedValue({ project: project(1, 'A'), patch: {}, explanation: 'A' });
    await chat.sendMessage('edit A');
    chat = renderChat();
    const proposal = lastAi(chat, threadId);
    expect(proposal.data.state).toBe('ready');
    // User switches to clip 1: the workspace now hosts controller b.
    chat.selectClip(threadId, 1);
    chat = renderChat();
    chat.workspaceRef.current = { controller: b.c };
    await chat.applyProposal(threadId, proposal.id);
    chat = renderChat();
    const after = threadOf(chat, threadId).messages.find((m) => m.id === proposal.id);
    expect(after.data.state).toBe('stale');
    expect(after.data.staleReason).toBe('selection-changed');
    expect(a.saves).toHaveLength(0);
    expect(b.saves).toHaveLength(0);
    expect(b.getServer().source.tag).toBe('B');
  });

  it('session reset clears the selection (registered in useChat)', async () => {
    let { chat, threadId } = setupDoneThread();
    chat.selectClip(threadId, 0);
    chat = renderChat();
    expect(chat.selectedClip).not.toBeNull();
    expect(sessionResetCbs.length).toBeGreaterThan(0);
    sessionResetCbs.forEach((cb) => cb());
    chat = renderChat();
    expect(chat.selectedClip).toBeNull();
    expect(storage.getItem(SELECTION_LS_KEY)).toBeNull();
  });

  it('thread switch drops a foreign selection; new thread starts clean', async () => {
    let { chat, threadId } = setupDoneThread();
    chat.selectClip(threadId, 0);
    chat = renderChat();
    const other = chat.createThread();
    chat = renderChat();
    chat.selectThread(other);
    chat = renderChat();
    expect(chat.selectedClip).toBeNull();
    expect(storage.getItem(SELECTION_LS_KEY)).toBeNull();
  });
});

// =====================================================================
// review-01 corrections — refresh coherence, proposal lifetime, bulk scope
// =====================================================================
describe('review-01: refresh restores one coherent identity', () => {
  const render = () => { hooks.index = 0; return useChat(); };
  const T = (id, jobId) => ({ id, title: id, phase: 'done', jobId, clips: [{ title: 'one' }], messages: [], settings: {} });
  const seed = (threads, selection) => {
    storage.setItem('nugget-chat-threads-v1', JSON.stringify(threads));
    if (selection) storage.setItem(SELECTION_LS_KEY, JSON.stringify(selection));
    else storage.removeItem(SELECTION_LS_KEY);
  };

  it('activates the persisted selection thread instead of the newest thread', () => {
    seed([T('newest', 'job2'), T('selected', 'job1')], { threadId: 'selected', jobId: 'job1', clipIndex: 0 });
    const chat = render();
    expect(chat.selectedClip).toEqual({ threadId: 'selected', jobId: 'job1', clipIndex: 0 });
    expect(chat.activeId).toBe('selected');
  });

  it('keeps a normal valid refresh intact (selection thread is newest)', () => {
    seed([T('t1', 'j')], { threadId: 't1', jobId: 'j', clipIndex: 0 });
    const chat = render();
    expect(chat.selectedClip).toEqual({ threadId: 't1', jobId: 'j', clipIndex: 0 });
    expect(chat.activeId).toBe('t1');
  });

  it('drops a stale persisted selection (job changed) and falls back to the newest thread', () => {
    seed([T('newest', 'job2'), T('selected', 'job1')], { threadId: 'selected', jobId: 'CHANGED', clipIndex: 0 });
    const chat = render();
    expect(chat.selectedClip).toBeNull();
    expect(storage.getItem(SELECTION_LS_KEY)).toBeNull();
    expect(chat.activeId).toBe('newest');
  });

  it('drops a persisted selection whose clip is gone', () => {
    seed([T('newest', 'job2'), T('selected', 'job1')], { threadId: 'selected', jobId: 'job1', clipIndex: 5 });
    const chat = render();
    expect(chat.selectedClip).toBeNull();
    expect(chat.activeId).toBe('newest');
  });

  it('drops a persisted selection whose thread is gone', () => {
    seed([T('newest', 'job2')], { threadId: 'ghost', jobId: 'job1', clipIndex: 0 });
    const chat = render();
    expect(chat.selectedClip).toBeNull();
    expect(chat.activeId).toBe('newest');
  });

  it('with no persisted selection the newest thread stays active', () => {
    seed([T('newest', 'job2'), T('older', 'job1')]);
    const chat = render();
    expect(chat.selectedClip).toBeNull();
    expect(chat.activeId).toBe('newest');
  });
});

// =====================================================================
describe('review-01: proposal selection/session lifetime', () => {
  const render = () => { hooks.index = 0; return useChat(); };
  const seed = () => {
    storage.setItem('nugget-chat-threads-v1', JSON.stringify([
      { id: 't1', title: 't1', phase: 'done', jobId: 'j', clips: [{ title: 'one' }], messages: [], settings: {} },
    ]));
    storage.setItem(SELECTION_LS_KEY, JSON.stringify({ threadId: 't1', jobId: 'j', clipIndex: 0 }));
    return render();
  };
  const threadOf = (chat, id) => chat.threads.find((t) => t.id === id);
  const proposalOf = (chat, id) => threadOf(chat, id).messages.find((m) => m.kind === 'proposal');
  async function wired() {
    const chat = seed();
    const f = makeFixture('A', 0, 'j');
    await f.c.load();
    chat.workspaceRef.current = { controller: f.c };
    return { chat, f };
  }

  it('a pending proposal cannot become ready after a session reset', async () => {
    let release;
    apiMocks.editAiPatch.mockImplementation(() => new Promise((r) => { release = r; }));
    let { chat } = await wired();
    const pending = chat.sendMessage('change hook');
    await vi.waitFor(() => expect(release).toBeTypeOf('function'));
    sessionResetCbs.forEach((cb) => cb());
    release({ project: project(1, 'A'), patch: {} });
    await pending;
    chat = render();
    const p = proposalOf(chat, 't1');
    expect(p?.data.state).not.toBe('ready');
    expect(p?.data.state).toBe('stale');
  });

  it('a ready proposal cannot be applied after reset + reselecting identical ids', async () => {
    apiMocks.editAiPatch.mockResolvedValue({ project: project(1, 'A'), patch: {} });
    let { chat } = await wired();
    await chat.sendMessage('change hook');
    chat = render();
    const id = proposalOf(chat, 't1').id;
    expect(proposalOf(chat, 't1').data.state).toBe('ready');
    sessionResetCbs.forEach((cb) => cb());
    chat = render();
    expect(proposalOf(chat, 't1').data.state).toBe('stale');
    chat.selectClip('t1', 0); // identical ids — must not revive
    chat = render();
    const fresh = makeFixture('B', 0, 'j');
    await fresh.c.load();
    chat.workspaceRef.current = { controller: fresh.c };
    await chat.applyProposal('t1', id);
    expect(fresh.saves).toHaveLength(0);
  });

  it('a pending proposal invalidated by clearSelection cannot revive on reselect', async () => {
    let release;
    apiMocks.editAiPatch.mockImplementation(() => new Promise((r) => { release = r; }));
    let { chat } = await wired();
    const pending = chat.sendMessage('change hook');
    await vi.waitFor(() => expect(release).toBeTypeOf('function'));
    chat.clearSelection();
    chat = render();
    chat.selectClip('t1', 0);
    chat = render();
    release({ project: project(1, 'A'), patch: {} });
    await pending;
    chat = render();
    expect(proposalOf(chat, 't1')?.data.state).not.toBe('ready');
  });

  it('a discarded proposal is never overwritten by a late response', async () => {
    let release;
    apiMocks.editAiPatch.mockImplementation(() => new Promise((r) => { release = r; }));
    let { chat } = await wired();
    const pending = chat.sendMessage('change hook');
    await vi.waitFor(() => expect(release).toBeTypeOf('function'));
    chat = render();
    const id = proposalOf(chat, 't1').id;
    chat.discardProposal('t1', id);
    release({ project: project(1, 'A'), patch: {} });
    await pending;
    chat = render();
    expect(proposalOf(chat, 't1')?.data.state).toBe('discarded');
  });

  it('proposals restored from storage are stale and refuse apply with zero saves', async () => {
    const readyMsg = {
      id: 'm1', ts: 1, role: 'ai', kind: 'proposal',
      data: { state: 'ready', identity: { jobId: 'j', clipIndex: 0 }, baseVersion: 1, project: project(1, 'A'), gen: 99, threadId: 't1' },
    };
    const thinkingMsg = {
      id: 'm2', ts: 2, role: 'ai', kind: 'proposal',
      data: { state: 'thinking', identity: { jobId: 'j', clipIndex: 0 }, gen: 99, threadId: 't1' },
    };
    storage.setItem('nugget-chat-threads-v1', JSON.stringify([
      { id: 't1', title: 't1', phase: 'done', jobId: 'j', clips: [{ title: 'one' }], messages: [readyMsg, thinkingMsg], settings: {} },
    ]));
    storage.setItem(SELECTION_LS_KEY, JSON.stringify({ threadId: 't1', jobId: 'j', clipIndex: 0 }));
    let chat = render();
    const msgs = threadOf(chat, 't1').messages;
    expect(msgs.find((m) => m.id === 'm1').data.state).toBe('stale');
    expect(msgs.find((m) => m.id === 'm2').data.state).toBe('stale');
    // history retained, payload not actionable
    expect(msgs.find((m) => m.id === 'm1').data.project.version).toBe(1);
    const f = makeFixture('A', 0, 'j');
    await f.c.load();
    chat.workspaceRef.current = { controller: f.c };
    await chat.applyProposal('t1', 'm1');
    expect(f.saves).toHaveLength(0);
  });

  it('reselecting identical ids WITHOUT a lifetime change keeps the proposal applicable', async () => {
    apiMocks.editAiPatch.mockResolvedValue({ project: { ...project(1, 'A'), hook: { text: 'WOW' } }, patch: {} });
    let { chat, f } = await wired();
    await chat.sendMessage('change hook');
    chat = render();
    const id = proposalOf(chat, 't1').id;
    expect(proposalOf(chat, 't1').data.state).toBe('ready');
    chat.selectClip('t1', 0); // same identity, no lifetime end
    chat = render();
    expect(proposalOf(chat, 't1').data.state).toBe('ready');
    await chat.applyProposal('t1', id);
    expect(f.saves).toHaveLength(1);
    chat = render();
    expect(proposalOf(chat, 't1').data.state).toBe('applied');
  });

  it('a thread switch ends the lifetime: late response settles stale, apply refuses', async () => {
    let release;
    apiMocks.editAiPatch.mockImplementation(() => new Promise((r) => { release = r; }));
    let { chat } = await wired();
    const pending = chat.sendMessage('change hook');
    await vi.waitFor(() => expect(release).toBeTypeOf('function'));
    const other = chat.createThread();
    chat = render();
    chat.selectThread(other);
    chat = render();
    release({ project: project(1, 'A'), patch: {} });
    await pending;
    chat = render();
    expect(proposalOf(chat, 't1')?.data.state).not.toBe('ready');
  });
});

// =====================================================================
describe('review-01: bulk scope is explicit multi-clip intent only', () => {
  const CTX = { phase: 'done', jobId: 'j', selection: { threadId: 't1', jobId: 'j', clipIndex: 0 } };

  it('plural in-clip operations route as single-clip edits', () => {
    for (const t of [
      'Make all captions in this clip yellow',
      'Edit the selected clip',
      'make all the words in this clip yellow',
      'make each caption bigger',
      'brighten every word on screen',
    ]) {
      expect(routeChatMessage(t, CTX).type).toBe('edit');
    }
  });

  it('explicit multi-clip intent routes bulk (Task 19)', () => {
    for (const t of [
      'make all clips punchier',
      'apply this to every clip please',
      'each of these clips needs louder audio',
      'these selected clips are too dark',
      'edit every single clip',
      'bulk brighten my clips',
    ]) {
      expect(routeChatMessage(t, CTX).type).toBe('bulk');
    }
  });

  it('bulk via the host makes zero provider calls and zero saves until Apply', async () => {
    const render = () => { hooks.index = 0; return useChat(); };
    let chat = render();
    const threadId = chat.createThread();
    chat = render();
    chat.patchThread(threadId, { phase: 'done', jobId: 'job1', clips: [{ index: 0, title: 'Clip 1' }] });
    chat = render();
    chat.selectClip(threadId, 0);
    const f = makeFixture('A', 0, 'job1');
    await f.c.load();
    chat = render();
    chat.workspaceRef.current = { controller: f.c };
    await chat.sendMessage('apply this to every clip please');
    expect(apiMocks.editAiPatch).not.toHaveBeenCalled();
    expect(f.saves).toHaveLength(0);
    chat = render();
    const msgs = chat.threads.find((t) => t.id === threadId).messages;
    const last = msgs[msgs.length - 1];
    expect(last.kind).toBe('bulk');
    expect(last.data.scope).toEqual({ mode: 'all' });
  });
});
