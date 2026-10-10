/**
 * Task 17 — explicit clip selection persistence.
 *
 * SAFE METADATA ONLY: the persisted shape is exactly
 * { threadId, jobId, clipIndex }. Controller instances, edit queues,
 * API tokens and any other runtime state are never serialized here.
 */

export const SELECTION_LS_KEY = 'nugget-selected-clip-v1';

export function isValidSelection(sel) {
  return !!(
    sel &&
    typeof sel === 'object' &&
    typeof sel.threadId === 'string' &&
    sel.threadId.length > 0 &&
    typeof sel.jobId === 'string' &&
    sel.jobId.length > 0 &&
    Number.isInteger(sel.clipIndex) &&
    sel.clipIndex >= 0
  );
}

/**
 * Load the persisted selection. Returns the safe {threadId, jobId,
 * clipIndex} triple, or null when absent, malformed, or carrying extra
 * fields beyond the three ids (defensive: only the ids are ever read).
 */
export function loadSelection() {
  try {
    const raw = localStorage.getItem(SELECTION_LS_KEY);
    if (!raw) return null;
    const sel = JSON.parse(raw);
    if (!isValidSelection(sel)) return null;
    return { threadId: sel.threadId, jobId: sel.jobId, clipIndex: sel.clipIndex };
  } catch {
    return null;
  }
}

/**
 * Persist only the three ids. Returns true on success, false when the
 * shape is invalid or storage is unavailable. Never writes anything else.
 */
export function saveSelection(sel) {
  try {
    if (!isValidSelection(sel)) return false;
    localStorage.setItem(
      SELECTION_LS_KEY,
      JSON.stringify({ threadId: sel.threadId, jobId: sel.jobId, clipIndex: sel.clipIndex })
    );
    return true;
  } catch {
    return false;
  }
}

export function clearStoredSelection() {
  try {
    localStorage.removeItem(SELECTION_LS_KEY);
  } catch {
    /* storage unavailable — nothing to clear */
  }
}
