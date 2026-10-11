/**
 * Task 16 — single authoritative project controller for the chat editor.
 *
 * Before Task 16 the outer EditorView owned a project + edit queue while the
 * nested ClipEditorView independently fetched, mutated, saved, and exported
 * its own copy: a transcript/AI edit could be missing from the nested
 * preview/export, and the nested view's stale save could overwrite a later
 * change. This controller is the one authority both views share.
 *
 * It owns, per (jobId, clipIndex):
 *  - loading (with 404 -> backfill -> retry, like the standalone editor),
 *  - the current project and its last server-acknowledged version,
 *  - the serialized edit queue (editor-timeline/undo.js): manual edits,
 *    AI patch applies, undo/redo all serialize here with optimistic
 *    concurrency (expectedVersion on every save; 409 -> reject + reload,
 *    never overwrite),
 *  - version restore / reset-to-AI-cut (draft-only, keeps the restored
 *    version's idempotency key so an unchanged export dedups),
 *  - export: drains the queue, snapshots the acknowledged project+version,
 *    and composes EXACTLY that snapshot — edits made while the compose is
 *    in flight queue behind it and cannot change the export's identity,
 *  - generation tokens: load/restore/reset/dispose bump a generation so
 *    late async responses can never update another clip's (or user's) state.
 *
 * Origin/idempotency: user-originated applyEdit calls stamp origin='user'
 * and mint a fresh idempotency_key (so the next export is a NEW render,
 * never a dedup hit). Version restore/reset keep the restored version's key.
 * Undo/redo restore snapshots with their original keys (existing queue
 * behavior — honest stacks and honest saved state).
 *
 * Framework-agnostic (no React): EditorView instantiates one per clip and
 * subscribes; the nested ClipEditorView renders controller state in
 * controlled mode and forwards mutations through it.
 */

import { createEditQueue } from '../editor-timeline/undo.js';

function clone(p) {
  return JSON.parse(JSON.stringify(p));
}

function newKey() {
  try {
    if (typeof crypto !== 'undefined' && crypto.randomUUID) return crypto.randomUUID();
  } catch {
    /* fall through */
  }
  return `${Date.now()}-${Math.random().toString(36).slice(2)}`;
}

function isStaleError(e) {
  return e?.status === 409 || /stale project version/i.test(e?.message || '');
}

// Cancellation (session reset / controller dispose) is expected, not a
// failure: callers observe a rejection, but nothing toasts and no state
// is resurrected.
function cancelledOp(what) {
  const e = new Error(`${what} cancelled (session reset or controller disposed)`);
  e.code = 'QUEUE_CANCELLED';
  return e;
}

function isCancelled(e) {
  return e?.code === 'QUEUE_CANCELLED';
}

// getClipProject resolves to either the bare project or (older callers)
// { project }; normalize both.
function normalizeProject(res) {
  return res?.project ?? res ?? null;
}

export function createProjectController({ jobId, clipIndex, api, onToast }) {
  if (!api) throw new Error('createProjectController: api is required');
  const toast = onToast || (() => {});

  // Generation token: bumped on load/reset/dispose so late async responses
  // are dropped instead of updating another clip's state. Ordinary
  // serialized operations (version restore / reset-to-AI-cut) do NOT bump
  // it: they queue behind in-flight work instead of cancelling it.
  let gen = 0;
  const alive = (token) => token === gen;

  // Disposal is terminal for new operations: a disposed controller rejects
  // every new mutation/query before any transform or API call, so a delayed
  // callback (upload/AI response) can never save through it. load() is the
  // explicit lifecycle restart and revives a disposed controller -- the host
  // calls load() after a StrictMode remount whose cleanup disposed the
  // shared controller, so a terminal load() would wedge the editor in dev.
  let disposed = false;
  function disposedError(what) {
    const e = new Error(`${what} rejected: project controller is disposed`);
    e.code = 'DISPOSED';
    return e;
  }
  // Returns a rejected promise when disposed, else null.
  function rejectIfDisposed(what) {
    if (disposed) return Promise.reject(disposedError(what));
    return null;
  }

  let state = {
    status: 'loading', // loading | ready | error
    project: null,
    ackVersion: null, // last server-acknowledged version (expectedVersion source)
    dirty: false, // draft content differs from ackVersion (after restore/reset)
    saving: false,
    exporting: false,
    loadError: '',
    versions: [],
    versionsLoading: false,
    lastExport: null, // { identity: { version, idempotency_key }, result }
  };
  const listeners = new Set();
  const getState = () => ({ ...state });
  const set = (patch) => {
    state = { ...state, ...patch };
    const snap = getState();
    listeners.forEach((fn) => {
      try {
        fn(snap);
      } catch {
        /* a listener error must never break the controller */
      }
    });
  };
  const subscribe = (fn) => {
    listeners.add(fn);
    return () => {
      listeners.delete(fn);
    };
  };

  const queue = createEditQueue({
    getProject: () => state.project,
    saveProject: (p, opts) => api.saveClipProject(jobId, clipIndex, p, opts),
    // The authoritative concurrency version for normal saves. After a
    // version restore the draft keeps its historical version while the
    // server sits at ackVersion -- saves must target the acknowledged
    // version, not the stale draft one. Explicit opts.expectedVersion
    // (the AI base-version guard) always wins.
    getExpectedVersion: () => state.ackVersion,
    onUpdate: (_label, saved) => {
      const proj = normalizeProject(saved);
      set({
        project: proj ? clone(proj) : state.project,
        ackVersion: saved?.version ?? proj?.version ?? state.ackVersion,
        dirty: false,
        saving: false,
      });
    },
    onError: (label, e) => {
      set({ saving: false });
      const stale = isStaleError(e);
      toast(
        stale
          ? 'Someone else saved first — reloading the latest version.'
          : `Couldn't save (${label}): ${e?.message || 'unknown error'}`
      );
      if (stale) load().catch(() => {});
    },
  });

  /** Load the latest project; 404 -> backfill once, then retry. */
  async function load() {
    // Explicit lifecycle (re)start: revives a disposed controller.
    disposed = false;
    const token = ++gen;
    set({ status: 'loading', loadError: '' });
    try {
      let res;
      try {
        res = await api.getClipProject(jobId, clipIndex);
      } catch (e) {
        if (e && e.status === 404) {
          await api.backfillClipProjects(jobId);
          res = await api.getClipProject(jobId, clipIndex);
        } else {
          throw e;
        }
      }
      if (!alive(token)) return null; // superseded by a newer load
      const proj = normalizeProject(res);
      set({
        status: 'ready',
        project: proj,
        ackVersion: proj?.version ?? null,
        dirty: false,
        loadError: '',
      });
      return proj;
    } catch (e) {
      if (!alive(token)) return null;
      set({ status: 'error', loadError: e?.message || 'failed to load project' });
      throw e;
    }
  }

  /**
   * User-originated mutation through the serialized queue. Stamps
   * origin='user' + a fresh idempotency_key unless opts.keepIdentity.
   * `mutate` is the queue's pure project transform.
   */
  function applyEdit(label, mutate, opts = {}) {
    const rejected = rejectIfDisposed('applyEdit');
    if (rejected) return rejected;
    return queue.applyEdit(
      label,
      async (before) => {
        const next = await mutate(before);
        if (!opts.keepIdentity && next && typeof next === 'object') {
          next.origin = 'user';
          next.idempotency_key = newKey();
        }
        return next;
      },
      opts
    );
  }

  function undo() {
    const rejected = rejectIfDisposed('undo');
    if (rejected) return rejected;
    return queue.undo();
  }

  function redo() {
    const rejected = rejectIfDisposed('redo');
    if (rejected) return rejected;
    return queue.redo();
  }

  /**
   * Persist the current draft. Only meaningful when dirty (after a version
   * restore/reset); every other edit already saved through applyEdit.
   * Always carries expectedVersion = the last acknowledged version.
   */
  function saveDraft() {
    const rejected = rejectIfDisposed('saveDraft');
    if (rejected) return rejected;
    if (!state.project || !state.dirty) return Promise.resolve(null);
    const token = gen;
    return queue
      .drain(async () => {
        if (!alive(token)) throw cancelledOp('save draft');
        set({ saving: true });
        const saved = await api.saveClipProject(jobId, clipIndex, state.project, {
          expectedVersion: state.ackVersion,
        });
        if (!alive(token)) throw cancelledOp('save draft');
        const proj = normalizeProject(saved);
        set({
          project: proj ? clone(proj) : state.project,
          ackVersion: saved?.version ?? proj?.version ?? state.ackVersion,
          dirty: false,
          saving: false,
        });
        return saved;
      })
      .catch((e) => {
        if (isCancelled(e) || !alive(token)) throw e; // stale: silent
        set({ saving: false });
        const stale = isStaleError(e);
        toast(
          stale
            ? 'Someone else saved first — reloading the latest version.'
            : `Couldn't save (save draft): ${e?.message || 'unknown error'}`
        );
        if (stale) load().catch(() => {});
        throw e;
      });
  }

  /**
   * Export: drain the queue, snapshot the acknowledged project+version,
   * and compose EXACTLY that snapshot. An edit made while the compose is
   * in flight serializes behind this drain and cannot change the
   * in-flight export's identity. Returns { ...composeResult, identity }.
   */
  function exportProject() {
    const rejected = rejectIfDisposed('exportProject');
    if (rejected) return rejected;
    if (!state.project) return Promise.reject(new Error('no project loaded'));
    const token = gen;
    return queue.drain(async () => {
      // A reset/dispose captured under an older lifetime must not let a
      // completed export replace the current preview or record lastExport.
      if (!alive(token)) throw cancelledOp('export');
      set({ exporting: true });
      try {
        if (state.dirty) {
          const saved = await api.saveClipProject(jobId, clipIndex, state.project, {
            expectedVersion: state.ackVersion,
          });
          if (!alive(token)) throw cancelledOp('export');
          const proj = normalizeProject(saved);
          set({
            project: proj ? clone(proj) : state.project,
            ackVersion: saved?.version ?? proj?.version ?? state.ackVersion,
            dirty: false,
          });
        }
        const snapshot = clone(state.project);
        const identity = {
          version: snapshot?.version ?? null,
          idempotency_key: snapshot?.idempotency_key ?? null,
        };
        const res = await api.composeProject(jobId, clipIndex, snapshot);
        if (!alive(token)) throw cancelledOp('export');
        const out = { ...(res || {}), identity };
        set({ lastExport: { identity, result: res || null } });
        return out;
      } catch (e) {
        if (isCancelled(e) || !alive(token)) throw e; // stale: silent, no toast
        const stale = isStaleError(e);
        toast(
          stale
            ? 'Someone else saved first — reloading the latest version.'
            : `Export failed: ${e?.message || 'unknown error'}`
        );
        if (stale) load().catch(() => {});
        throw e;
      } finally {
        if (alive(token)) set({ exporting: false });
      }
    });
  }

  // Draft-only restore: keeps the restored version's idempotency key (an
  // unchanged export dedups to the identical file — a true undo) and marks
  // dirty. ackVersion stays at the last acknowledged version so the
  // follow-up save carries a valid expectedVersion, not a stale one.
  async function restoreLocked(version, token) {
    // Check BEFORE the follow-up API call, not only after its response: a
    // reset/dispose during version listing must not start a historical
    // project fetch. Already-issued requests keep their post-response check.
    if (!alive(token)) throw cancelledOp('restore');
    const res = await api.getClipProject(jobId, clipIndex, version);
    if (!alive(token)) return null;
    const proj = normalizeProject(res);
    set({ project: proj, dirty: true });
    return proj;
  }

  /** Load a recorded version into the draft (dirty, unsaved). */
  function restoreVersion(version) {
    const rejected = rejectIfDisposed('restoreVersion');
    if (rejected) return rejected;
    // Ordinary serialized restore: does NOT bump the generation. It queues
    // behind in-flight work (e.g. an export awaiting compose) instead of
    // cancelling it. resetSession()/dispose() still cancel it through their
    // own invalidation.
    const token = gen;
    return queue.drain(() => restoreLocked(version, token));
  }

  /** Reset the draft to the earliest AI cut (dirty, unsaved). */
  function resetToAiCut() {
    const rejected = rejectIfDisposed('resetToAiCut');
    if (rejected) return rejected;
    // Same serialized semantics as restoreVersion: queues behind in-flight
    // work instead of invalidating it.
    const token = gen;
    return queue.drain(async () => {
      const r = await api.listClipProjectVersions(jobId, clipIndex);
      const vs = r?.versions || [];
      const auto = vs
        .filter((v) => v.origin === 'auto')
        .sort((a, b) => a.version - b.version)[0];
      return restoreLocked(auto ? auto.version : 1, token);
    });
  }

  /** Refresh the version-history list (passive: never bumps the generation). */
  function listVersions() {
    const rejected = rejectIfDisposed('listVersions');
    if (rejected) return rejected;
    const token = gen;
    set({ versionsLoading: true });
    return api.listClipProjectVersions(jobId, clipIndex).then(
      (r) => {
        if (!alive(token)) return state.versions;
        const vs = r?.versions || [];
        set({ versions: vs, versionsLoading: false });
        return vs;
      },
      () => {
        if (alive(token)) set({ versionsLoading: false });
        return state.versions;
      }
    );
  }

  /** AI trim: ask the backend for drop ranges (generation-guarded). */
  function aiTrim(instruction) {
    const rejected = rejectIfDisposed('aiTrim');
    if (rejected) return rejected;
    const token = gen;
    return api.editClipAI(jobId, clipIndex, instruction).then((r) => {
      if (!alive(token)) throw new Error('clip changed while the AI was thinking');
      return r;
    });
  }

  function uploadAudio(file) {
    const rejected = rejectIfDisposed('uploadAudio');
    if (rejected) return rejected;
    // Lifetime-bound like aiTrim: capture the generation around the awaited
    // response. A reset/dispose while the upload is in flight rejects the
    // stale completion, so a panel continuation can never attach old audio
    // to the next session through this controller.
    const token = gen;
    return api.uploadAudio(jobId, file).then((r) => {
      if (!alive(token)) throw cancelledOp('uploadAudio');
      return r;
    });
  }

  /**
   * Session reset (sign-out / signed-in user change): drop undo/redo
   * history, the cached project, and invalidate in-flight responses —
   * the existing session-reset behavior (queue.clear, project null,
   * remount via the host's reset key).
   */
  function resetSession() {
    if (disposed) return; // already terminally shut down; nothing to reset
    gen += 1;
    queue.clear();
    set({
      status: 'loading',
      project: null,
      ackVersion: null,
      dirty: false,
      saving: false,
      exporting: false,
      loadError: '',
      versions: [],
      versionsLoading: false,
      lastExport: null,
    });
  }

  /**
   * Drain pending saves through the queue without composing (review-02).
   * Lets any in-flight/queued save finish and serialize before a caller
   * (e.g. the Hyperframes export path) proceeds, so the export composes
   * the acknowledged snapshot. Rejects if the controller was disposed.
   */
  function drain() {
    const rejected = rejectIfDisposed('drain');
    if (rejected) return rejected;
    return queue.drain(() => undefined);
  }

  /**
   * Clip switch / unmount: invalidate in-flight work, drop listeners, and
   * reject every future operation. Terminal -- only load() restarts.
   */
  function dispose() {
    if (disposed) return;
    disposed = true;
    gen += 1;
    queue.clear(); // pending queue work never starts; in-flight results are dropped
    listeners.clear();
  }

  return {
    jobId,
    clipIndex,
    getState,
    subscribe,
    load,
    applyEdit,
    undo,
    redo,
    canUndo: queue.canUndo,
    canRedo: queue.canRedo,
    saveDraft,
    exportProject,
    drain,
    restoreVersion,
    resetToAiCut,
    listVersions,
    aiTrim,
    uploadAudio,
    resetSession,
    dispose,
  };
}
