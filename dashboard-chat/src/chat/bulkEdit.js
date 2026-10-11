/**
 * Task 19 — bulk editing pure helpers (React-free).
 *
 * Scope freezing, intent summaries and operation inspection live here so
 * the card and the tests share one implementation. The API surface is
 * injected nowhere: these are pure functions over plain data.
 */

// The typed operations the bulk composer offers. Deterministic presets —
// no model call is needed to build an intent.
export const BULK_OPERATION_DEFS = {
  caption_appearance: {
    label: 'Caption style',
    blurb: 'Shared caption appearance (style and/or position).',
    fields: ['caption_style', 'caption_position'],
  },
  hook_appearance: {
    label: 'Hook style',
    blurb: 'Shared hook overlay appearance (style and/or position).',
    fields: ['style', 'position'],
  },
  hook_timing: {
    label: 'Hook timing',
    blurb: 'Shared hook overlay timing, adapted to each clip\u2019s duration.',
    fields: ['start', 'end'],
  },
  hyperframes_finishing: {
    label: 'Hyperframes finishing (optional pilot)',
    blurb: 'Apply one Hyperframes template as the finishing selection.',
    fields: ['template', 'caption_size', 'caption_position'],
  },
};

// Closed value sets mirrored from the backend (clippyme/domain/bulk_edit.py).
// The card only offers these; the server re-validates strictly.
export const BULK_VALUE_SETS = {
  caption_style: ['classic_white', 'hormozi_bold', 'neon_glow', 'mrbeast_box', 'minimal_clean', 'fire_impact'],
  caption_position: ['top', 'middle', 'bottom'],
  hook_style: ['viral'],
  hook_position: ['top', 'middle', 'bottom'],
  hf_template: ['clean-captions', 'emphasized-word', 'timed-opening'],
  hf_caption_size: ['small', 'medium', 'large'],
  hf_caption_position: ['lower-third', 'middle'],
};

/**
 * Freeze the ordered unique clip indices for a scope.
 * mode 'all': every clip index in the thread's clip list (the enumerated
 *   current-job snapshot -- clips added later are never included).
 * mode 'selected': the explicit indices, deduped and sorted.
 */
export function freezeScope(clips, mode, indices) {
  const all = (clips || [])
    .map((c, i) => (Number.isInteger(c?.index) ? c.index : i))
    .filter((n) => Number.isInteger(n) && n >= 0);
  if (mode === 'all') return [...new Set(all)].sort((a, b) => a - b);
  if (mode === 'selected') {
    const picked = (indices || []).filter((n) => Number.isInteger(n) && n >= 0);
    return [...new Set(picked)].sort((a, b) => a - b);
  }
  throw new Error(`unknown bulk scope mode: ${mode}`);
}

/** Human-readable one-line summary of a normalized intent. */
export function summarizeIntent(intent) {
  if (!intent || typeof intent !== 'object') return '';
  const def = BULK_OPERATION_DEFS[intent.op];
  const label = def ? def.label : intent.op;
  const params = intent.params || {};
  const bits = Object.entries(params).map(([k, v]) => `${k}=${v}`);
  const mode = intent.on_override === 'replace' ? ' (replace overrides)' : '';
  return `${label}: ${bits.join(', ')}${mode}`;
}

/** Items as a clip-ordered array (the API returns a dict keyed by index). */
export function bulkItemsArray(operation) {
  const items = operation?.items;
  if (!items) return [];
  const arr = Array.isArray(items) ? items : Object.values(items);
  return arr
    .filter((i) => i && Number.isInteger(i.clip_index))
    .sort((a, b) => a.clip_index - b.clip_index);
}

/** Outcomes that are still actionable by retry. */
export function retryableItems(operation) {
  return bulkItemsArray(operation).filter((i) => i.outcome === 'failed' || i.outcome === 'pending');
}

/** Successfully applied items (undo candidates). */
export function appliedItems(operation) {
  return bulkItemsArray(operation).filter((i) => i.outcome === 'applied');
}

/** Applied items not yet undone (per the recorded undo outcomes). */
export function undoableItems(operation) {
  const done = new Set(
    Object.entries(operation?.undo?.items || {})
      .filter(([, r]) => r?.outcome === 'undone')
      .map(([k]) => Number(k))
  );
  return appliedItems(operation).filter((i) => !done.has(i.clip_index));
}

// Card states where an unapplied proposal is still live. Anything else is
// terminal or explicitly discarded.
const BULK_OPEN_STATES = new Set(['idle', 'preparing', 'proposal', 'applying', 'retrying', 'undoing']);
export function isBulkOpenState(state) {
  return BULK_OPEN_STATES.has(state);
}

// Per-outcome human labels for the results table.
export const BULK_OUTCOME_LABELS = {
  planned: 'planned',
  applied: 'applied',
  pending: 'pending — blocked',
  conflict: 'conflict — clip moved',
  failed: 'failed',
  unsupported: 'unsupported',
  preserved: 'preserved override',
  unchanged: 'already set',
  excluded: 'excluded',
};
export function bulkOutcomeLabel(outcome) {
  return BULK_OUTCOME_LABELS[outcome] || outcome || '?';
}

/**
 * Minimal prepared-operation reference, persisted per originating
 * job+thread so a page refresh can reconnect the card to its existing
 * ledger instead of minting a new idempotency key / operation.
 *
 * Only {operation_id, idempotency_key} are stored -- never project
 * snapshots, patches or grants. localStorage access is guarded so the
 * helpers are safe in non-browser environments (tests).
 */
export function bulkOpRefKey(jobId, threadId) {
  return `nugget-bulk-op:${jobId}:${threadId}`;
}
export function readBulkOpRef(jobId, threadId) {
  try {
    if (typeof localStorage === 'undefined' || !jobId || !threadId) return null;
    const raw = localStorage.getItem(bulkOpRefKey(jobId, threadId));
    if (!raw) return null;
    const ref = JSON.parse(raw);
    if (ref && typeof ref.operation_id === 'string' && typeof ref.idempotency_key === 'string') {
      return { operation_id: ref.operation_id, idempotency_key: ref.idempotency_key };
    }
  } catch { /* storage unavailable or corrupt: treat as no reference */ }
  return null;
}
export function writeBulkOpRef(jobId, threadId, ref) {
  try {
    if (typeof localStorage === 'undefined' || !jobId || !threadId) return;
    localStorage.setItem(
      bulkOpRefKey(jobId, threadId),
      JSON.stringify({ operation_id: ref.operation_id, idempotency_key: ref.idempotency_key }),
    );
  } catch { /* best-effort persistence only */ }
}
export function clearBulkOpRef(jobId, threadId) {
  try {
    if (typeof localStorage === 'undefined' || !jobId || !threadId) return;
    localStorage.removeItem(bulkOpRefKey(jobId, threadId));
  } catch { /* best-effort persistence only */ }
}
