// Chat API — thin wrappers over the Nugget backend for the chat UI.
// Uses the same apiFetch/getApiUrl plumbing as the classic dashboard.
import { apiFetch } from './apiToken.js';
import { getApiUrl } from './config.js';

async function req(method, path, body, { timeoutMs = 30000 } = {}) {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), timeoutMs);
  try {
    const res = await apiFetch(getApiUrl(path), {
      method,
      headers: { 'Content-Type': 'application/json' },
      body: body === undefined ? undefined : JSON.stringify(body),
      signal: ctrl.signal,
    });
    const text = await res.text();
    let data = null;
    try { data = text ? JSON.parse(text) : null; } catch { data = { _raw: text }; }
    if (!res.ok) {
      const msg = (data && (data.detail || data.message || data.error)) || `HTTP ${res.status}`;
      const err = new Error(typeof msg === 'string' ? msg : JSON.stringify(msg));
      err.status = res.status;
      err.data = data;
      throw err;
    }
    return data;
  } finally {
    clearTimeout(timer);
  }
}

export const validateUrl = (url) =>
  req('POST', '/api/validate-url', { url }, { timeoutMs: 25000 });

export const submitJob = (payload) =>
  req('POST', '/api/process', payload, { timeoutMs: 60000 });

export const getProgress = (jobId) =>
  req('GET', `/api/progress/${encodeURIComponent(jobId)}`, undefined, { timeoutMs: 10000 });

export const getJobStatus = (jobId) =>
  req('GET', `/api/status/${encodeURIComponent(jobId)}`, undefined, { timeoutMs: 15000 });

export const steerJob = (jobId, prompt) =>
  req('POST', `/api/jobs/${encodeURIComponent(jobId)}/steer`, { prompt }, { timeoutMs: 15000 });

export const getHistory = () =>
  req('GET', '/api/history', undefined, { timeoutMs: 15000 });

/** Phase B2 copilot: NL instruction -> validated ClipProject patch (returned, not saved). */
export const editAiPatch = (jobId, clipIndex, instruction, model) =>
  req('POST', `/api/edit-ai/${encodeURIComponent(jobId)}/${encodeURIComponent(clipIndex)}`,
    { instruction, model: model || undefined, mode: 'patch' }, { timeoutMs: 90000 });

/** Quick trim path (kept from the classic dashboard). */
export const editAiTrim = (jobId, clipIndex, instruction, model) =>
  req('POST', `/api/edit-ai/${encodeURIComponent(jobId)}/${encodeURIComponent(clipIndex)}`,
    { instruction, model: model || undefined, mode: 'trim' }, { timeoutMs: 90000 });

export const getJobClips = async (jobId) => {
  const s = await getJobStatus(jobId);
  const r = s?.result || {};
  let clips = r.clips || r.results || r.data?.clips || [];
  if (!Array.isArray(clips)) clips = [];
  if (clips.length === 0) {
    // Older jobs don't carry clips in /api/status — fall back to /api/history.
    try {
      const h = await getHistory();
      const items = Array.isArray(h) ? h : (h.jobs || h.history || []);
      const hit = items.find((j) => (j.jobId || j.job_id || j.id) === jobId);
      if (hit && Array.isArray(hit.clips) && hit.clips.length) clips = hit.clips;
    } catch { /* keep empty */ }
  }
  return clips;
};

export const getJobTitle = async (jobId) => {
  try {
    const s = await getJobStatus(jobId);
    const r = s?.result || {};
    return r.title || r.source_title || r?.metadata?.title || null;
  } catch { return null; }
};

/** Publish via Zernio (schedule_mode: now | auto | manual). */
export const publishClip = (jobId, clipIndex, body = {}) =>
  req('POST', `/api/publish/${encodeURIComponent(jobId)}/${encodeURIComponent(clipIndex)}`, body, { timeoutMs: 120000 });

/** Normalize a clip object defensively — backend shapes vary. */
export function normClip(clip, index) {
  const c = clip || {};
  const dur = c.duration ?? c.length ?? (
    (c.end != null && c.start != null) ? Math.max(0, c.end - c.start) : null
  );
  return {
    index,
    title: c.title || c.name || `Clip ${index + 1}`,
    hook: c.hook_text || c.hook || c.caption || '',
    duration: dur,
    durationTier: c.duration_tier || null,
    score: c.score ?? c.virality_score ?? null,
    videoUrl: c.composed_video_url || c.video_url || '',
    thumbUrl: c.thumbnail_url || c.thumbnail || c.poster_url || '',
    transcript: c.transcript || null,
    scene: c.scene_analysis || c.analysis || c.description || '',
    raw: c,
  };
}

const URL_RE = /(https?:\/\/[^\s<>"']+)/gi;
export function extractUrl(text) {
  if (!text) return null;
  const m = String(text).match(URL_RE);
  return m ? m[0] : null;
}

export function fmtEta(sec) {
  if (sec == null || !isFinite(sec) || sec < 0) return null;
  const s = Math.round(sec);
  if (s < 60) return `${s}s left`;
  const m = Math.floor(s / 60);
  if (m < 60) return `${m}m ${s % 60}s left`;
  return `${Math.floor(m / 60)}h ${m % 60}m left`;
}

export function fmtDur(sec) {
  if (sec == null || !isFinite(sec)) return '';
  const s = Math.round(sec);
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`;
}
