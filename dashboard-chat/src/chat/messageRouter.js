/**
 * Task 17 — the real chat dispatcher for useChat.sendMessage.
 *
 * React-free and pure: given the message text and the thread context it
 * returns exactly one route. sendMessage executes the route; this module
 * never touches state, the network, or any provider.
 *
 * Order is load-bearing (each branch is exclusive):
 *  (a) running job (clipping/submitting + jobId)       -> 'steer'
 *      (existing steering preserved)
 *  (b) /hyperframes command                            -> 'hyperframes'
 *  (c) bulk scope ("all/every/each/these/selected clips", "bulk")
 *                                                      -> 'bulk-unsupported' (never mutates)
 *  (d) explicit clip selection                         -> 'edit'
 *      (even when the text contains a URL — embedded links stay edits;
 *       starting a new source needs the explicit New-source action or a
 *       new conversation)
 *  (e) no selection + URL                              -> 'new-source'
 *      (existing validation flow; clears job state AND selection)
 *  (f) no selection + phase done + no URL              -> 'need-selection'
 *      (actionable "select a clip" — never guess clip0)
 *  (g) phase idle                                      -> 'idle-hint' (existing)
 *  else                                                -> 'none'
 *      (validating/caption-pick/settings/error — stay silent, as before)
 */
import { extractUrl } from '../api/chatApi.js';

const HYPERFRAMES_RE = /^\s*\/hyperframes\b/i;
const BULK_WORD_RE = /\bbulk\b/i;
// "all clips", "every clip", "each of these clips", "these selected clips"…
const BULK_SCOPE_RE = /\b(all|every|each|these|selected)\b[^.!?]{0,48}\bclips?\b/i;

function hasSelection(selection) {
  return !!(
    selection &&
    typeof selection.jobId === 'string' &&
    selection.jobId.length > 0 &&
    Number.isInteger(selection.clipIndex) &&
    selection.clipIndex >= 0
  );
}

export function routeChatMessage(text, ctx = {}) {
  const { phase = 'idle', jobId = null, selection = null } = ctx;
  const clean = (text || '').trim();

  // (a) A running job owns the composer: steer it, whatever the text says.
  if ((phase === 'clipping' || phase === 'submitting') && jobId) {
    return { type: 'steer' };
  }
  // (b) The exact /hyperframes slash command.
  if (HYPERFRAMES_RE.test(clean)) {
    return { type: 'hyperframes' };
  }
  // (c) Bulk scope is honestly unsupported — never mutate several clips.
  if (BULK_WORD_RE.test(clean) || BULK_SCOPE_RE.test(clean)) {
    return { type: 'bulk-unsupported' };
  }
  // (d) Explicit selection: an ordinary edit, even with an embedded link.
  if (hasSelection(selection)) {
    return { type: 'edit', selection };
  }
  // (e) No selection + a URL: the existing new-source flow.
  const url = extractUrl(clean);
  if (url) {
    return { type: 'new-source', url };
  }
  // (f) Clips are done but nothing is selected: ask, don't guess clip0.
  if (phase === 'done') {
    return { type: 'need-selection' };
  }
  // (g) Fresh thread: the existing idle hint.
  if (phase === 'idle') {
    return { type: 'idle-hint' };
  }
  // validating | caption-pick | settings | error …: stay silent as before.
  return { type: 'none' };
}
