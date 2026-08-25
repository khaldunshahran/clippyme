
import { getApiUrl } from '../config';
import { apiFetch } from './apiToken';

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
  const error = new Error(msg || `HTTP ${res.status}`);
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

export async function submitProcessJob(data, apiKey, { signal } = {}) {
  const key = (apiKey || (typeof localStorage !== 'undefined' ? localStorage.getItem('gemini_key') : '') || '').trim();
  const headers = key ? { 'X-Gemini-Key': key } : {};
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
    body = formData;
  }

  const res = await apiFetch(getApiUrl('/api/process'), { method: 'POST', headers, body, signal });
  if (!res.ok) await throwFromResponse(res);
  return res.json();
}

export async function submitBatchJob(data, apiKey, { signal } = {}) {
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
  const res = await apiFetch(getApiUrl('/api/batch'), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', 'X-Gemini-Key': apiKey },
    body: JSON.stringify(batchBody),
    signal,
  });
  if (!res.ok) await throwFromResponse(res);
  return res.json();
}

export async function triggerStorageCleanup({ signal } = {}) {
  const res = await apiFetch(getApiUrl('/api/storage/cleanup'), { method: 'POST', signal });
  if (!res.ok) await throwFromResponse(res);
  return res.json();
}
