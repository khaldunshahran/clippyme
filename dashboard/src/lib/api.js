
import { getApiUrl } from '../config';
import { apiFetch } from './apiToken';

/** Submit POSTs can carry large file uploads over the tunnel — give them a
 *  longer budget than the 30s read default, while still never hanging forever. */
export const SUBMIT_TIMEOUT_MS = 300_000;

/** Fresh idempotency key per submit click; the backend dedupes retries. */
export function newIdempotencyKey() {
  try {
    if (typeof crypto !== 'undefined' && crypto.randomUUID) return crypto.randomUUID();
  } catch { /* fall through to the fallback */ }
  return `idemp-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`;
}

// Gateway statuses mapped to plain-language text: when the tunnel or a proxy
// in front of the backend fails, the body is usually an HTML error page —
// never paste that into the UI.
const GATEWAY_MESSAGES = {
  502: 'The backend is unreachable (bad gateway) — it may be restarting. Try again in a moment.',
  503: 'The backend is temporarily unavailable — try again in a moment.',
  504: 'The backend took too long to respond (gateway timeout) — try again.',
};

export async function throwFromResponse(res) {
  const text = await res.text();
  let msg = text;
  try {
    const parsed = JSON.parse(text);
    const detail = parsed?.detail ?? parsed?.message;
    msg = typeof detail === 'string' ? detail : detail ? JSON.stringify(detail) : text;
  } catch {
    // Non-JSON body: use the captured text.
  }
  msg = String(msg || '');
  // Strip HTML tags (Cloudflare 502/504 pages) and collapse whitespace so a
  // full error document never lands in a log panel or toast.
  const wasHtml = /<[a-z][^>]*>/i.test(msg);
  msg = msg.replace(/<[^>]*>/g, ' ').replace(/\s+/g, ' ').trim();
  if (msg.length > 300) msg = `${msg.slice(0, 297)}...`;
  // Gateway failures usually arrive as an HTML page or a bare status phrase —
  // prefer the plain-language text in both cases, but keep a real JSON detail
  // from the backend when it gave us one.
  if (GATEWAY_MESSAGES[res.status] && (wasHtml || msg.length < 12)) {
    msg = GATEWAY_MESSAGES[res.status];
  }
  if (!msg) msg = `HTTP ${res.status}`;
  const error = new Error(msg);
  error.status = res.status;
  error.retryable = res.status === 408 || res.status === 429 || res.status >= 500;
  throw error;
}

export async function pollJob(jobId, { signal } = {}) {
  const res = await apiFetch(getApiUrl(`/api/status/${encodeURIComponent(jobId)}`), { signal });
  if (!res.ok) await throwFromResponse(res);
  return res.json();
}

function pickLanguage(pre) {
  const lang = (pre?.language || '').trim();
  if (!lang || lang === 'multi' || lang === 'auto') return undefined;
  return lang;
}

export async function submitProcessJob(data, apiKey, { signal, timeoutMs = SUBMIT_TIMEOUT_MS, idempotencyKey } = {}) {
  const key = (apiKey || (typeof localStorage !== 'undefined' ? localStorage.getItem('gemini_key') : '') || '').trim();
  const headers = key ? { 'X-Gemini-Key': key } : {};
  // Fresh idempotency key per submit click so a double-click / retry resumes
  // the same backend job instead of spawning a duplicate on the GPU.
  headers['Idempotency-Key'] = idempotencyKey || newIdempotencyKey();
  let body;
  const language = pickLanguage(data.preselections);
  const reframeMode = data.reframe_mode || data.preselections?.reframe_mode;
  const aspect = data.aspect || data.preselections?.aspect;
  const noZoom = data.no_zoom === true || data.preselections?.no_zoom === true;
  const letterboxZoom = Number(data.letterbox_zoom ?? data.preselections?.letterbox_zoom) || 0;
  const skipAnalysis = data.skip_analysis === true || data.preselections?.skip_analysis === true;
  const model = (data.model || data.preselections?.model || '').trim();
  const minDuration = Number(data.min_duration ?? data.preselections?.min_duration) || null;
  const maxDuration = Number(data.max_duration ?? data.preselections?.max_duration) || null;
  const minClips = Number(data.min_clips ?? data.preselections?.min_clips) || null;
  const maxClips = Number(data.max_clips ?? data.preselections?.max_clips) || null;
  const clipType = (data.clip_type || data.preselections?.clip_type || '').trim() || null;
  const durationMode = (data.duration_mode || data.preselections?.duration_mode || '').trim() || null;
  const highlights = data.highlights === true || data.preselections?.highlights === true;

  if (data.type === 'url') {
    headers['Content-Type'] = 'application/json';
    const jsonBody = { url: data.payload };
    if (data.instructions) jsonBody.instructions = data.instructions;
    if (highlights) jsonBody.highlights = true;
    if (reframeMode) jsonBody.reframe_mode = reframeMode;
    if (letterboxZoom) jsonBody.letterbox_zoom = letterboxZoom;
    if (aspect) jsonBody.aspect = aspect;
    if (language) jsonBody.language = language;
    if (noZoom) jsonBody.no_zoom = true;
    if (skipAnalysis) jsonBody.skip_analysis = true;
    if (model) jsonBody.model = model;
    if (minDuration) jsonBody.min_duration = minDuration;
    if (maxDuration) jsonBody.max_duration = maxDuration;
    if (minClips) jsonBody.min_clips = minClips;
    if (maxClips) jsonBody.max_clips = maxClips;
    if (clipType) jsonBody.clip_type = clipType;
    if (durationMode) jsonBody.duration_mode = durationMode;
    body = JSON.stringify(jsonBody);
  } else {
    if (data.payload?.size > 16 * 1024 * 1024 * 1024) throw new Error('File too large. Maximum size is 16 GB.');
    const formData = new FormData();
    formData.append('file', data.payload);
    if (data.instructions) formData.append('instructions', data.instructions);
    if (highlights) formData.append('highlights', 'true');
    if (reframeMode) formData.append('reframe_mode', reframeMode);
    if (letterboxZoom) formData.append('letterbox_zoom', String(letterboxZoom));
    if (aspect) formData.append('aspect', aspect);
    if (language) formData.append('language', language);
    if (noZoom) formData.append('no_zoom', 'true');
    if (skipAnalysis) formData.append('skip_analysis', 'true');
    if (model) formData.append('model', model);
    if (minDuration) formData.append('min_duration', String(minDuration));
    if (maxDuration) formData.append('max_duration', String(maxDuration));
    if (minClips) formData.append('min_clips', String(minClips));
    if (maxClips) formData.append('max_clips', String(maxClips));
    if (clipType) formData.append('clip_type', clipType);
    if (durationMode) formData.append('duration_mode', durationMode);
    body = formData;
  }

  const res = await apiFetch(getApiUrl('/api/process'), { method: 'POST', headers, body, signal, timeoutMs });
  if (!res.ok) await throwFromResponse(res);
  return res.json();
}

export async function submitBatchJob(data, apiKey, { signal, timeoutMs = SUBMIT_TIMEOUT_MS, idempotencyKey } = {}) {
  const batchBody = { urls: data.urls, instructions: data.instructions };
  if (data.preselections?.reframe_mode) batchBody.reframe_mode = data.preselections.reframe_mode;
  if (Number(data.preselections?.letterbox_zoom)) batchBody.letterbox_zoom = Number(data.preselections.letterbox_zoom);
  if (data.preselections?.aspect && data.preselections.aspect !== '9:16') batchBody.aspect = data.preselections.aspect;
  const language = pickLanguage(data.preselections);
  if (language) batchBody.language = language;
  if (data.preselections?.no_zoom === true) batchBody.no_zoom = true;
  if (data.preselections?.skip_analysis === true) batchBody.skip_analysis = true;
  if ((data.preselections?.model || '').trim()) batchBody.model = data.preselections.model.trim();
  if (Number(data.preselections?.min_duration)) batchBody.min_duration = Number(data.preselections.min_duration);
  if (Number(data.preselections?.max_duration)) batchBody.max_duration = Number(data.preselections.max_duration);
  if (Number(data.preselections?.min_clips)) batchBody.min_clips = Number(data.preselections.min_clips);
  if (Number(data.preselections?.max_clips)) batchBody.max_clips = Number(data.preselections.max_clips);
  if ((data.preselections?.clip_type || '').trim()) batchBody.clip_type = data.preselections.clip_type.trim();
  if ((data.preselections?.duration_mode || '').trim()) batchBody.duration_mode = data.preselections.duration_mode.trim();
  // Omit X-Gemini-Key when empty (like submitProcessJob): an empty header
  // value would shadow the backend's persisted-key fallback on servers that
  // presence-check the header.
  const key = (apiKey || '').trim();
  const headers = { 'Content-Type': 'application/json', ...(key ? { 'X-Gemini-Key': key } : {}) };
  headers['Idempotency-Key'] = idempotencyKey || newIdempotencyKey();
  const res = await apiFetch(getApiUrl('/api/batch'), {
    method: 'POST',
    headers,
    body: JSON.stringify(batchBody),
    signal,
    timeoutMs,
  });
  if (!res.ok) await throwFromResponse(res);
  return res.json();
}

export async function getStorageBreakdown({ signal } = {}) {
  const res = await apiFetch(getApiUrl('/api/storage/breakdown'), { signal });
  if (!res.ok) await throwFromResponse(res);
  return res.json();
}

export async function triggerStorageCleanup({ mode = 'safe', purgeRawSources = false, signal } = {}) {
  const res = await apiFetch(getApiUrl('/api/storage/cleanup'), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ mode, purge_raw_sources: purgeRawSources }),
    signal,
  });
  if (!res.ok) await throwFromResponse(res);
  return res.json();
}
