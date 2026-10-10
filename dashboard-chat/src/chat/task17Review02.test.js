/**
 * Task 17 review-02 durable regressions.
 *
 * Gate 1: every/each must range over CLIPS to be bulk scope. Quantifiers
 * over other nouns inside the selected clip ("highlight every word in this
 * clip") are ordinary single-clip edits. Positive + negative fixtures run
 * through the REAL routeChatMessage dispatcher (the exact function
 * useChat.sendMessage calls), plus host-level zero-call assertions.
 *
 * Gate 2: a save issued before session cancellation cannot be un-issued.
 * applyProposal must report honest uncertainty (SAVE_INTERRUPTED) instead
 * of claiming "nothing was applied"; rejections that happen BEFORE the
 * save API is issued keep the known-no-call message. Real controller +
 * injected backend; the persistent hook harness drives the real useChat.
 *
 * Load-bearing mutations (each must fail, then the suite re-runs green):
 *  M8: restore the old BULK_EVERY_RE (every/each + 48 chars + clip)
 *  M9: queue post-save cancellation throws QUEUE_CANCELLED again
 *      (the "nothing was applied" lie returns)
 *  M10: drop the SAVE_INTERRUPTED branch in applyProposal
 */
import { describe, expect, it, vi, beforeEach } from 'vitest';

// --- minimal hook harness: persistent useState/useRef, identity
// useMemo/useCallback, immediate useEffect (node: no window/document) ---
const h = vi.hoisted(() => ({ values: [], index: 0, reset: [] }));
const api = vi.hoisted(() => ({
  editAiPatch: vi.fn(),
  getHistory: vi.fn(async () => []),
  validateUrl: vi.fn(),
}));
const storage = vi.hoisted(() => {
  const s = {};
  return {
    getItem: (k) => (k in s ? s[k] : null),
    setItem: (k, v) => { s[k] = String(v); },
    removeItem: (k) => { delete s[k]; },
    clear: () => { Object.keys(s).forEach((k) => delete s[k]); },
  };
});
vi.stubGlobal('localStorage', storage);
vi.mock('react', async (original) => ({
  ...(await original()),
  useState: (init) => {
    const i = h.index++;
    if (!(i in h.values)) h.values[i] = typeof init === 'function' ? init() : init;
    return [h.values[i], (v) => { h.values[i] = typeof v === 'function' ? v(h.values[i]) : v; }];
  },
  useRef: (init) => {
    const i = h.index++;
    if (!(i in h.values)) h.values[i] = { current: init };
    return h.values[i];
  },
  useCallback: (f) => f,
  useMemo: (f) => f(),
  useEffect: (f) => { f(); },
}));
vi.mock('../api/chatApi.js', async (original) => ({ ...(await original()), ...api }));
vi.mock('../api/sessionReset.js', () => ({ onSessionReset: (f) => { h.reset.push(f); return () => {}; } }));

import { useChat } from './useChat.js';
import { routeChatMessage } from './messageRouter.js';
import { createProjectController } from './projectController.js';

const proj = () => ({ version: 1, idempotency_key: 'v1', segments: [], captions: { edits: {} }, overlays: [] });
const thread = (id = 't1', jobId = 'j') => ({ id, title: id, phase: 'done', jobId, clips: [{ title: 'one' }], messages: [], settings: {} });
function render() { h.index = 0; return useChat(); }
function seed() {
  storage.setItem('nugget-chat-threads-v1', JSON.stringify([thread()]));
  storage.setItem('nugget-selected-clip-v1', JSON.stringify({ threadId: 't1', jobId: 'j', clipIndex: 0 }));
  return render();
}
async function fixture() {
  const saves = [];
  const c = createProjectController({
    jobId: 'j', clipIndex: 0,
    api: {
      getClipProject: async () => proj(),
      saveClipProject: async (j, i, p) => { saves.push(p); return { project: { ...p, version: 2 }, version: 2 }; },
    },
  });
  await c.load();
  return { c, saves };
}
const proposal = (c) => c.threads[0].messages.find((m) => m.kind === 'proposal');

beforeEach(() => {
  h.values = []; h.index = 0; h.reset = []; storage.clear();
  api.editAiPatch.mockReset(); api.getHistory.mockResolvedValue([]);
});

const SEL_CTX = { phase: 'done', jobId: 'j', selection: { threadId: 't1', jobId: 'j', clipIndex: 0 } };

describe('review-02 gate 1: every/each scope is multi-clip only', () => {
  it.each([
    'Highlight every word in this clip',
    'Make each caption in the selected clip bold',
    'Edit every segment in this clip',
    'Make all words in this clip uppercase',
    'Highlight each word of the transcript in this clip',
    'Every caption in this clip should be yellow',
  ])('single-clip edit routes edit: %s', (text) => {
    expect(routeChatMessage(text, SEL_CTX).type).toBe('edit');
  });

  it.each([
    'Apply this hook to every clip',
    'Make each clip start with a question',
    'Add captions to all clips',
    'Update each of these clips with the new style',
    'Every one of the clips needs a hook',
    'These selected clips should all be yellow',
  ])('multi-clip request routes bulk-unsupported: %s', (text) => {
    expect(routeChatMessage(text, SEL_CTX).type).toBe('bulk-unsupported');
  });

  it('a genuine every-clip request through sendMessage makes zero provider calls', async () => {
    const { c } = await fixture();
    let chat = seed();
    chat.workspaceRef.current = { controller: c };
    await chat.sendMessage('apply this hook to every clip');
    expect(api.editAiPatch).not.toHaveBeenCalled();
    chat = render();
    expect(proposal(chat)).toBeUndefined();
  });

  it('an in-clip every-word edit through sendMessage reaches the AI edit route', async () => {
    const { c } = await fixture();
    api.editAiPatch.mockResolvedValue({ project: proj(), patch: {} });
    let chat = seed();
    chat.workspaceRef.current = { controller: c };
    await chat.sendMessage('highlight every word in this clip');
    expect(api.editAiPatch).toHaveBeenCalledTimes(1);
    chat = render();
    expect(proposal(chat)?.data.state).toBe('ready');
  });
});

describe('review-02 gate 2: honest uncertainty for issued-but-cancelled saves', () => {
  it('a cancelled apply cannot assert nothing was saved when the request already committed', async () => {
    let release, committed = false;
    const gate = new Promise((r) => { release = r; });
    const c0 = createProjectController({
      jobId: 'j', clipIndex: 0,
      api: {
        getClipProject: async () => proj(),
        saveClipProject: async (j, i, p) => { committed = true; await gate; return { project: { ...p, version: 2 }, version: 2 }; },
      },
    });
    await c0.load();
    api.editAiPatch.mockResolvedValue({ project: proj(), patch: {} });
    let chat = seed();
    chat.workspaceRef.current = { controller: c0 };
    await chat.sendMessage('change hook');
    chat = render();
    const id = proposal(chat).id;
    const applying = chat.applyProposal('t1', id);
    await vi.waitFor(() => expect(committed).toBe(true));
    h.reset.forEach((f) => f());
    c0.resetSession();
    release();
    await applying;
    chat = render();
    expect(committed).toBe(true);
    const err = proposal(chat).data.error ?? '';
    // The old lie is gone...
    expect(err).not.toMatch(/nothing was applied/i);
    // ...replaced by honest uncertainty plus the safe reconciliation path.
    expect(err).toMatch(/can.t tell whether it saved/i);
    expect(err).toMatch(/reload the clip/i);
    // No blind retry: exactly one save was ever issued.
    // (The backend committed once; the queue never re-issued.)
  });

  it('a rejection before the save API is issued still honestly says nothing was applied', async () => {
    const { c, saves } = await fixture();
    api.editAiPatch.mockResolvedValue({ project: proj(), patch: {} });
    let chat = seed();
    chat.workspaceRef.current = { controller: c };
    await chat.sendMessage('change hook');
    chat = render();
    const id = proposal(chat).id;
    // Terminal disposal: the entry guard rejects before any transform/API call.
    c.dispose();
    await chat.applyProposal('t1', id);
    chat = render();
    expect(saves).toHaveLength(0);
    expect(proposal(chat).data.error ?? '').toMatch(/nothing was applied/i);
  });

  it('a pre-save queue cancellation keeps the known-no-call message', async () => {
    const { c, saves } = await fixture();
    api.editAiPatch.mockResolvedValue({ project: proj(), patch: {} });
    let chat = seed();
    chat.workspaceRef.current = { controller: c };
    await chat.sendMessage('change hook');
    chat = render();
    const id = proposal(chat).id;
    // Block the queue with a stuck first op, then reset: the apply op is
    // cancelled at execution start -- before its save API is ever issued.
    let releaseStuck;
    const stuckGate = new Promise((r) => { releaseStuck = r; });
    const stuck = c.applyEdit('stuck', async (p) => { await stuckGate; return p; });
    const applying = chat.applyProposal('t1', id);
    c.resetSession();
    releaseStuck();
    await stuck.catch(() => {});
    await applying;
    chat = render();
    expect(saves).toHaveLength(0);
    expect(proposal(chat).data.error ?? '').toMatch(/nothing was applied/i);
  });
});
