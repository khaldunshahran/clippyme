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

// getClipProject resolves to either the bare project or (older callers)
// { project }; normalize both.
function normalizeProject(res) {
  return res?.project ?? res ?? null;
}

export function createProjectController({ jobId, clipIndex, api, onToast }) {
  if (!api) throw new Error('createProjectController: api is required');
  const toast = onToast || (() => {});

  // Generation token: bumped on every load/restore/reset/dispose so late
  // async responses are dropped instead of updating another clip's state.
  let gen = 0;
  const alive = (token) => token === gen;

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
    return queue.undo();
  }

  function redo() {
    return queue.redo();
  }

  /**
   * Persist the current draft. Only meaningful when dirty (after a version
   * restore/reset); every other edit already saved through applyEdit.
   * Always carries expectedVersion = the last acknowledged version.
   */
  function saveDraft() {
    if (!state.project || !state.dirty) return Promise.resolve(null);
    set({ saving: true });
    return queue
      .drain(async () => {
        const saved = await api.saveClipProject(jobId, clipIndex, state.project, {
          expectedVersion: state.ackVersion,
        });
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
    if (!state.project) return Promise.reject(new Error('no project loaded'));
    set({ exporting: true });
    return queue.drain(async () => {
      try {
        if (state.dirty) {
          const saved = await api.saveClipProject(jobId, clipIndex, state.project, {
            expectedVersion: state.ackVersion,
          });
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
        const out = { ...(res || {}), identity };
        set({ lastExport: { identity, result: res || null } });
        return out;
      } catch (e) {
        const stale = isStaleError(e);
        toast(
          stale
            ? 'Someone else saved first — reloading the latest version.'
            : `Export failed: ${e?.message || 'unknown error'}`
        );
        if (stale) load().catch(() => {});
        throw e;
      } finally {
        set({ exporting: false });
      }
    });
  }

  // Draft-only restore: keeps the restored version's idempotency key (an
  // unchanged export dedups to the identical file — a true undo) and marks
  // dirty. ackVersion stays at the last acknowledged version so the
  // follow-up save carries a valid expectedVersion, not a stale one.
  async function restoreLocked(version, token) {
    const res = await api.getClipProject(jobId, clipIndex, version);
    if (!alive(token)) return null;
    const proj = normalizeProject(res);
    set({ project: proj, dirty: true });
    return proj;
  }

  /** Load a recorded version into the draft (dirty, unsaved). */
  function restoreVersion(version) {
    const token = ++gen;
    return queue.drain(() => restoreLocked(version, token));
  }

  /** Reset the draft to the earliest AI cut (dirty, unsaved). */
  function resetToAiCut() {
    const token = ++gen;
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
    const token = gen;
    return api.editClipAI(jobId, clipIndex, instruction).then((r) => {
      if (!alive(token)) throw new Error('clip changed while the AI was thinking');
      return r;
    });
  }

  function uploadAudio(file) {
    return api.uploadAudio(jobId, file);
  }

  /**
   * Session reset (sign-out / signed-in user change): drop undo/redo
   * history, the cached project, and invalidate in-flight responses —
   * the existing session-reset behavior (queue.clear, project null,
   * remount via the host's reset key).
   */
  function resetSession() {
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

  /** Clip switch / unmount: invalidate in-flight work, drop listeners. */
  function dispose() {
    gen += 1;
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
    restoreVersion,
    resetToAiCut,
    listVersions,
    aiTrim,
    uploadAudio,
    resetSession,
    dispose,
  };
}
