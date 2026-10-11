/**
 * UI-01 offline fixture — stubbed ../api/realApi.js.
 *
 * A fake versioned backend: the REAL createProjectController + the REAL
 * serialized edit queue run against this. Saves can be gated (deferred)
 * via __setSaveGate so the test driver can observe in-flight states.
 * No network, no production data.
 */
const clone = (p) => JSON.parse(JSON.stringify(p));

function makeProject(version) {
  return {
    origin: version === 1 ? 'auto' : 'user',
    version,
    idempotency_key: `fixture-key-v${version}`,
    source: { duration: 30, width: 1080, height: 1920 },
    segments: [{ id: 'seg0', start: 0, end: 30, crop: { x: 236, y: 0, w: 608, h: 1080 } }],
    captions: {
      style: 'classic_white',
      position: 'bottom',
      words: [
        { id: 'w0', w: 'Stop', start: 0, end: 0.5 },
        { id: 'w1', w: 'scrubbing.', start: 0.5, end: 1.2 },
        { id: 'w2', w: 'Start', start: 1.2, end: 1.7 },
        { id: 'w3', w: 'posting.', start: 1.7, end: 2.4 },
      ],
      edits: {},
    },
    overlays: [
      { type: 'hook', text: 'Stop scrubbing. Start posting.', style: 'viral', start: 0, end: 3, position: 'top' },
    ],
    audio: [],
  };
}

// One fake backend per (jobId, clipIndex) — like the real server.
const stores = new Map();
function storeFor(jobId, clipIndex) {
  const k = `${jobId}:${clipIndex}`;
  if (!stores.has(k)) {
    const latest = makeProject(3);
    stores.set(k, {
      latest,
      history: { 1: makeProject(1), 2: makeProject(2), 3: clone(latest) },
      nextVersion: 3,
    });
  }
  return stores.get(k);
}

let saveGate = null;

export const __calls = { get: 0, save: 0, compose: 0, versions: 0 };
export function __setSaveGate(p) { saveGate = p; }
export function __resetBackend() {
  stores.clear();
  saveGate = null;
  __calls.get = 0; __calls.save = 0; __calls.compose = 0; __calls.versions = 0;
}

export async function getClipProject(jobId, clipIndex, version = null) {
  __calls.get += 1;
  const s = storeFor(jobId, clipIndex);
  if (version != null && s.history[version]) return clone(s.history[version]);
  return clone(s.latest);
}

export async function saveClipProject(jobId, clipIndex, project, opts = {}) {
  __calls.save += 1;
  if (saveGate) await saveGate;
  const s = storeFor(jobId, clipIndex);
  s.nextVersion += 1;
  s.latest = clone(project);
  s.latest.version = s.nextVersion;
  s.history[s.nextVersion] = clone(s.latest);
  return { version: s.nextVersion, project: clone(s.latest) };
}

export async function composeProject(jobId, clipIndex, project) {
  __calls.compose += 1;
  return {
    version: project.version,
    // Relative on purpose: the fixture is served from file://, where a
    // leading slash would escape the dist folder and the (real, playable)
    // synthetic media would never load.
    composed_url: `fixture-render-v${project.version}.mp4`,
    deduped: false,
  };
}

export async function backfillClipProjects() { /* no-op */ }

export async function listClipProjectVersions() {
  __calls.versions += 1;
  return {
    versions: [
      { version: 1, origin: 'auto', created_at: '2026-10-11T10:00:00', output_url: null },
      { version: 2, origin: 'user', created_at: '2026-10-11T11:00:00', output_url: null },
      { version: 3, origin: 'user', created_at: '2026-10-11T12:00:00', output_url: 'fixture-render-v3.mp4' },
    ],
  };
}

export async function editClipAI() { return { drop_ranges: [], explanation: '' }; }
export async function uploadAudio() { return { file: 'x.mp3' }; }
export async function downloadClip() { return null; }
export async function getClipTranscript() { return { words: [] }; }

// Bulk-ledger API surface (imported by BulkEditCard; unused by the fixture).
export async function bulkPrepare() { return { ok: true, items: [] }; }
export async function bulkApply() { return { ok: true }; }
export async function bulkRetry() { return { ok: true }; }
export async function bulkStatus() { return { ok: true }; }
export async function bulkUndo() { return { ok: true }; }
export const newBulkIdempotencyKey = () => `fixture-bulk-${Date.now()}`;

export const safeResolveUrl = (u) => u;
// The src helpers mirror the PRODUCTION selection policy in
// api/realApi.js verbatim (review-02): clipVideoSrc is the raw
// video_url; clipPreviewSrc prefers an applied previewUrl, then the
// COMPOSED file, then the raw source. The editor's draft player must
// therefore NOT use clipPreviewSrc — draftPreviewSource (clipEditor)
// picks the raw source. An earlier stub returned 'fixture-draft.mp4'
// for both helpers, which hid exactly that integration issue.
export function clipVideoSrc(clip, bust) {
  const full = safeResolveUrl(clip?.video_url || '');
  return bust ? `${full}${full.includes('?') ? '&' : '?'}v=${bust}` : full;
}
export function clipPreviewSrc(clip, state) {
  if (state?.previewUrl) {
    const full = safeResolveUrl(state.previewUrl);
    const b = state.previewBust;
    return b ? `${full}${full.includes('?') ? '&' : '?'}v=${b}` : full;
  }
  if (clip?.composed_video_url) {
    const full = safeResolveUrl(clip.composed_video_url);
    const b = state?.reframeBust;
    return b ? `${full}${full.includes('?') ? '&' : '?'}v=${b}` : full;
  }
  return clipVideoSrc(clip, state?.reframeBust);
}
