/**
 * Single serialized mutation queue for the transcript editor.
 *
 * EVERY project mutation -- word edit, split, drop, copilot apply, undo,
 * redo -- goes through applyEdit(). Calls are chained on one promise so
 * they never overlap; a failed call never touches the undo stack.
 *
 * Snapshots: the project is deep-cloned BEFORE the API call and pushed
 * onto the undo stack ONLY on success. Undo/redo save with optimistic
 * concurrency (expected_version); a 409 means someone else saved first,
 * and the UI must surface it (refresh, don't silently clobber).
 *
 * Lifetime: clear() drops undo/redo history AND bumps the lifetime epoch.
 * Operations captured under an older epoch never start; an operation that
 * was already running refuses to apply its results (no state resurrection,
 * no undo pollution) and never issues another API call. Cancellation is
 * silent (no error toast) -- it is expected, not a failure.
 */

export function createEditQueue({
  getProject,
  saveProject,
  onUpdate,
  onError,
  getExpectedVersion,
}) {
  let chain = Promise.resolve();
  const undoStack = [];
  const redoStack = [];
  const MAX_DEPTH = 50;
  // Lifetime epoch. Bumped by clear() (session reset / controller dispose).
  let epoch = 0;

  function cancelled(label) {
    const e = new Error(`cancelled: ${label} (queue cleared)`);
    e.code = 'QUEUE_CANCELLED';
    return e;
  }

  // A save whose request was already issued when the session ended.
  // Cancellation only drops the confirmation -- it cannot undo the request,
  // so callers must report honest uncertainty, never "nothing was applied".
  function saveInterrupted(label) {
    const e = new Error(`interrupted: ${label} (save request issued; confirmation cancelled by session reset or dispose)`);
    e.code = 'SAVE_INTERRUPTED';
    return e;
  }

  // Raw serialization primitive. Keeps the chain alive across failures.
  // `fn` receives the epoch token captured for this operation; it must
  // re-check it before every state mutation and API call.
  function enqueue(fn) {
    const token = epoch;
    const p = chain.then(() => {
      if (token !== epoch) throw cancelled('queued operation');
      return fn(token);
    });
    chain = p.catch(() => {});
    return p;
  }

  function clone(p) {
    return JSON.parse(JSON.stringify(p));
  }

  /** The authoritative concurrency version for normal saves. */
  function defaultExpectedVersion(before) {
    return getExpectedVersion?.() ?? before.version;
  }

  /**
   * Run a mutation. `mutate` is a pure project transform:
   *   (projectClone) => mutatedProject   (may be async)
   * The queue deep-clones the current project, runs `mutate`, then saves
   * with optimistic concurrency (expected_version = the authoritative
   * acknowledged version unless overridden via opts.expectedVersion --
   * the AI base-version guard). The undo snapshot is the pre-mutation
   * state and is pushed ONLY when the save succeeds.
   */
  function applyEdit(label, mutate, opts = {}) {
    return enqueue(async (token) => {
      const current = () => token === epoch;
      const before = clone(getProject());
      let next;
      try {
        next = await mutate(clone(before));
      } catch (e) {
        if (current()) onError?.(label, e); // transform failed: nothing to undo
        throw e;
      }
      if (!current()) throw cancelled(label); // cleared mid-mutation: never save
      let saved;
      try {
        saved = await saveProject(next, {
          expectedVersion: opts.expectedVersion ?? defaultExpectedVersion(before),
        });
      } catch (e) {
        if (current()) onError?.(label, e); // save failed (incl. 409): snapshot discarded
        throw e;
      }
      // Cleared after the save request was issued: the server may already
      // have committed. Report honest uncertainty (SAVE_INTERRUPTED) --
      // cancellation cannot undo an issued request, so callers must never
      // claim "nothing was applied" for this case.
      if (!current()) throw saveInterrupted(label);
      undoStack.push({ label, project: before });
      if (undoStack.length > MAX_DEPTH) undoStack.shift();
      redoStack.length = 0;
      onUpdate?.(label, saved);
      return saved;
    });
  }

  /** Save helper with optimistic concurrency. Throws on 409. */
  async function saveWithVersion(project, expectedVersion) {
    return saveProject(project, { expectedVersion });
  }

  function undo() {
    return enqueue(async (token) => {
      const current = () => token === epoch;
      const entry = undoStack[undoStack.length - 1];
      if (!entry) return null;
      const currentProject = clone(getProject());
      let saved;
      try {
        saved = await saveWithVersion(
          entry.project,
          defaultExpectedVersion(currentProject)
        );
      } catch (e) {
        // Entry stays on the stack: the user can refresh and retry.
        if (current()) onError?.(`undo (${entry.label})`, e);
        throw e;
      }
      if (!current()) throw cancelled(`undo (${entry.label})`);
      undoStack.pop();
      redoStack.push({ label: entry.label, project: currentProject });
      onUpdate?.(`undo: ${entry.label}`, saved);
      return saved;
    });
  }

  function redo() {
    return enqueue(async (token) => {
      const current = () => token === epoch;
      const entry = redoStack[redoStack.length - 1];
      if (!entry) return null;
      const currentProject = clone(getProject());
      let saved;
      try {
        saved = await saveWithVersion(
          entry.project,
          defaultExpectedVersion(currentProject)
        );
      } catch (e) {
        if (current()) onError?.(`redo (${entry.label})`, e);
        throw e;
      }
      if (!current()) throw cancelled(`redo (${entry.label})`);
      redoStack.pop();
      undoStack.push({ label: entry.label, project: currentProject });
      onUpdate?.(`redo: ${entry.label}`, saved);
      return saved;
    });
  }

  /**
   * Drop all undo/redo history (e.g. on sign-out or signed-in user change)
   * and invalidate pending/in-flight queue work. Already-issued server
   * requests are not undone, but their responses can no longer recreate
   * cleared state, pollute undo history, or launch another request.
   */
  function clear() {
    undoStack.length = 0;
    redoStack.length = 0;
    epoch += 1;
  }

  /**
   * Task 16 drain primitive: run `fn` after all previously queued work has
   * settled. The project controller uses it for export (drain the queue,
   * then snapshot the acknowledged project) and for version restore/reset.
   * Failures propagate to the caller; the chain itself stays alive. A drain
   * queued before clear() is cancelled like any other operation.
   */
  function drain(fn) {
    return enqueue((token) => fn(token));
  }

  return {
    applyEdit,
    undo,
    redo,
    clear,
    drain,
    canUndo: () => undoStack.length > 0,
    canRedo: () => redoStack.length > 0,
    // Test seam.
    _stacks: () => ({ undo: undoStack.length, redo: redoStack.length }),
  };
}
