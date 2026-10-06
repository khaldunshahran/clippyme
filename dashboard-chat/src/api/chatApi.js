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
    let badJson = false;
    try { data = text ? JSON.parse(text) : null; } catch { badJson = !!text; data = { _raw: text }; }
    if (!res.ok) {
      const msg = (data && (data.detail || data.message || data.error)) || `HTTP ${res.status}`;
      const err = new Error(typeof msg === 'string' ? msg : JSON.stringify(msg));
      err.status = res.status;
      err.data = data;
      throw err;
    }
    if (badJson) {
      // e.g. a CDN/proxy served the SPA HTML for an /api route: fail loudly
      // instead of letting callers render garbage (the old silent 0% freeze).
      const berr = new Error(`API returned a non-JSON response (HTTP ${res.status})`);
      berr.status = res.status;
      berr.data = data;
      throw berr;
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

/** Phase D: duplicate a clip → returns { new_index }. */
export const duplicateClip = (jobId, clipIndex) =>
  req('POST', `/api/clips/${encodeURIComponent(jobId)}/${encodeURIComponent(clipIndex)}/duplicate`, {}, { timeoutMs: 60000 });

/** Phase D: schedule a clip as a timed Zernio post. */
export const scheduleClipPost = (jobId, clipIndex, body) =>
  req('POST', `/api/clips/${encodeURIComponent(jobId)}/${encodeURIComponent(clipIndex)}/schedule`, body, { timeoutMs: 120000 });

export const getClipSchedule = (jobId, clipIndex) =>
  req('GET', `/api/clips/${encodeURIComponent(jobId)}/${encodeURIComponent(clipIndex)}/schedule`, undefined, { timeoutMs: 15000 });

/** Phase D: upscale a clip 2x (lanczos + NVENC). Long operation — returns { download_url }. */
export const upscaleClip = (jobId, clipIndex) =>
  req('POST', `/api/clips/${encodeURIComponent(jobId)}/${encodeURIComponent(clipIndex)}/upscale`, {}, { timeoutMs: 600000 });

/** Phase D: download the FCP7 XML timeline for a clip. */
export const exportClipXml = async (jobId, clipIndex) => {
  const res = await apiFetch(getApiUrl(`/api/clips/${encodeURIComponent(jobId)}/${encodeURIComponent(clipIndex)}/export-xml`));
  if (!res.ok) {
    const t = await res.text().catch(() => '');
    throw new Error(`Export failed: HTTP ${res.status}${t ? ` — ${t.slice(0, 140)}` : ''}`);
  }
  const blob = await res.blob();
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `clip_${clipIndex + 1}_timeline.xml`;
  document.body.appendChild(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 8000);
  return true;
};

/** Connected Zernio accounts (for building publish targets). */
export const getZernioAccounts = () =>
  req('GET', '/api/zernio/accounts', undefined, { timeoutMs: 15000 });

/** Map a Zernio accounts list → [{ platform, accountId }] publish targets. Defensive: skips unknowns. */
export function zernioTargets(accounts) {
  const list = Array.isArray(accounts) ? accounts : (accounts?.accounts || []);
  return (Array.isArray(list) ? list : [])
    .map((a) => {
      if (!a || typeof a !== 'object') return null;
      const platform = String(a.platform || a.provider || a.type || a.network || '').toLowerCase();
      const accountId = a.id ?? a.accountId ?? a.account_id ?? a.uuid ?? a.external_id;
      if (!platform || accountId == null || accountId === '') return null;
      return { platform, accountId: String(accountId) };
    })
    .filter(Boolean);
}

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
