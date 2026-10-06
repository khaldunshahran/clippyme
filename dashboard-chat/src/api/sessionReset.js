/**
 * Per-user state reset registry.
 *
 * Components that hold state belonging to the signed-in user (the transcript
 * editor's undo manager, cached project data, editor selection) register a
 * reset callback here. AuthGate emits a reset on sign-out and whenever the
 * signed-in user ID changes, so user A's undo history / cached project can
 * never leak into user B's session.
 *
 * Handlers must be idempotent and must not throw (a throwing handler is
 * caught so one bad callback can't block the others).
 */

const handlers = new Set();

/** Register a reset callback. Returns an unsubscribe function. */
export function onSessionReset(fn) {
  if (typeof fn !== 'function') return () => {};
  handlers.add(fn);
  return () => handlers.delete(fn);
}

/** Run all registered reset callbacks. */
export function emitSessionReset() {
  for (const fn of handlers) {
    try {
      fn();
    } catch {
      // ignore — one bad handler must not block the rest
    }
  }
}

/** Test seam: drop all registered handlers between cases. */
export function _resetSessionResetForTests() {
  handlers.clear();
}
