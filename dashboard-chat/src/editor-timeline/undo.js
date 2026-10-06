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
 */

export function createEditQueue({ getProject, saveProject, onUpdate, onError }) {
  let chain = Promise.resolve();
  const undoStack = [];
  const redoStack = [];
  const MAX_DEPTH = 50;

  // Raw serialization primitive. Keeps the chain alive across failures.
  function enqueue(fn) {
    const p = chain.then(fn);
    chain = p.catch(() => {});
    return p;
  }

  function clone(p) {
    return JSON.parse(JSON.stringify(p));
  }

  /**
   * Run a mutation. `mutate` is a pure project transform:
   *   (projectClone) => mutatedProject   (may be async)
   * The queue deep-clones the current project, runs `mutate`, then saves
   * with optimistic concurrency (expected_version = pre-mutation version
   * unless overridden via opts). The undo snapshot is the pre-mutation
   * state and is pushed ONLY when the save succeeds.
   */
  function applyEdit(label, mutate, opts = {}) {
    return enqueue(async () => {
      const before = clone(getProject());
      let next;
      try {
        next = await mutate(clone(before));
      } catch (e) {
        onError?.(label, e); // transform failed: nothing to undo
        throw e;
      }
      let saved;
      try {
        saved = await saveProject(next, {
          expectedVersion: opts.expectedVersion ?? before.version,
        });
      } catch (e) {
        onError?.(label, e); // save failed (incl. 409): snapshot discarded
        throw e;
      }
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
    return enqueue(async () => {
      const entry = undoStack[undoStack.length - 1];
      if (!entry) return null;
      const current = clone(getProject());
      let saved;
      try {
        saved = await saveWithVersion(entry.project, current.version);
      } catch (e) {
        // Entry stays on the stack: the user can refresh and retry.
        onError?.(`undo (${entry.label})`, e);
        throw e;
      }
      undoStack.pop();
      redoStack.push({ label: entry.label, project: current });
      onUpdate?.(`undo: ${entry.label}`, saved);
      return saved;
    });
  }

  function redo() {
    return enqueue(async () => {
      const entry = redoStack[redoStack.length - 1];
      if (!entry) return null;
      const current = clone(getProject());
      let saved;
      try {
        saved = await saveWithVersion(entry.project, current.version);
      } catch (e) {
        onError?.(`redo (${entry.label})`, e);
        throw e;
      }
      redoStack.pop();
      undoStack.push({ label: entry.label, project: current });
      onUpdate?.(`redo: ${entry.label}`, saved);
      return saved;
    });
  }

  return {
    applyEdit,
    undo,
    redo,
    canUndo: () => undoStack.length > 0,
    canRedo: () => redoStack.length > 0,
    // Test seam.
    _stacks: () => ({ undo: undoStack.length, redo: redoStack.length }),
  };
}
