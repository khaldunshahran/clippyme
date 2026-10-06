import { describe, it, expect, beforeEach } from 'vitest';
import { createEditQueue } from '../editor-timeline/undo.js';
import {
  onSessionReset,
  emitSessionReset,
  _resetSessionResetForTests,
} from './sessionReset.js';

/**
 * Item 3: on sign-out and on signed-in user change, per-user state must be
 * dropped — the undo manager, cached project data, and editor state.
 *
 * Scenario: user A makes an edit (undo stack non-empty), signs out, user B
 * signs in. Afterwards the undo stack must be empty and no cached project
 * data from A may remain.
 */

function makeQueue() {
  let project = { version: 1, words: [] };
  const saved = [];
  const queue = createEditQueue({
    getProject: () => project,
    saveProject: async (p) => {
      const next = { ...p, version: project.version + 1 };
      project = next;
      saved.push(next);
      return { project: next };
    },
    onUpdate: () => {},
    onError: () => {},
  });
  return { queue, getProject: () => project, saved };
}

describe('session reset (sign-out / user change)', () => {
  beforeEach(() => {
    _resetSessionResetForTests();
  });

  it('A edits, signs out, B signs in: undo stack empty, no cached project from A', async () => {
    const { queue, getProject } = makeQueue();

    // A makes an edit -> undo stack non-empty.
    await queue.applyEdit('edit word', (p) => ({ ...p, words: [{ id: 1 }] }));
    expect(queue._stacks().undo).toBe(1);
    expect(queue.canUndo()).toBe(true);

    // Simulate what EditorView registers: clear queue + drop cached project.
    let cachedProject = getProject();
    const unsubscribe = onSessionReset(() => {
      queue.clear();
      cachedProject = null;
    });

    // A signs out -> AuthGate emits the reset.
    emitSessionReset();

    expect(queue._stacks()).toEqual({ undo: 0, redo: 0 });
    expect(queue.canUndo()).toBe(false);
    expect(queue.canRedo()).toBe(false);
    expect(cachedProject).toBeNull();

    // B signs in with a different user id -> reset fires again (idempotent).
    emitSessionReset();
    expect(queue._stacks()).toEqual({ undo: 0, redo: 0 });

    unsubscribe();
  });

  it('reset handlers are isolated and unsubscribable', () => {
    let calls = 0;
    const off = onSessionReset(() => { calls += 1; });
    emitSessionReset();
    expect(calls).toBe(1);
    off();
    emitSessionReset();
    expect(calls).toBe(1);
  });

  it('a throwing handler does not block the others', () => {
    let ok = false;
    onSessionReset(() => { throw new Error('boom'); });
    onSessionReset(() => { ok = true; });
    emitSessionReset();
    expect(ok).toBe(true);
  });

  it('undo queue clear() drops redo history too', async () => {
    const { queue } = makeQueue();
    await queue.applyEdit('edit 1', (p) => p);
    await queue.undo();
    expect(queue._stacks()).toEqual({ undo: 0, redo: 1 });
    queue.clear();
    expect(queue._stacks()).toEqual({ undo: 0, redo: 0 });
    expect(queue.canRedo()).toBe(false);
  });
});
