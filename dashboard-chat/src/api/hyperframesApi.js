// Hyperframes API surface (Task 18). Thin wrappers over the FastAPI
// /api/hyperframes/* endpoints. Reuses apiFetch (auth) and getApiUrl like
// realApi.js. All errors throw with status so the card can render them.
import { getApiUrl } from './config.js';
import { apiFetch } from './apiToken.js';

async function _call(path, { method = 'GET', body } = {}) {
  const res = await apiFetch(getApiUrl(path), {
    method,
    ...(body ? { headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) } : {}),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    const e = new Error(err.detail || `HTTP ${res.status}`);
    e.status = res.status;
    throw e;
  }
  return res.json();
}

// Capability + supported templates (global, not per-clip):
// { available, reason?, templates: [...], renderer_version }
export function getHyperframesCapability() {
  return _call('/api/hyperframes/capability');
}

// Compile a composition for preview. Optional preview-only template/param
// overrides; when omitted, the project's saved finishing is used.
export function compileHyperframes(jobId, index, body) {
  return _call(`/api/hyperframes/compile/${jobId}/${index}`, { method: 'POST', body });
}

// Start an export render (uses the project's SAVED finishing; no overrides).
// body: { expected_version, compile_id? }.
export function renderHyperframes(jobId, index, body) {
  return _call(`/api/hyperframes/render/${jobId}/${index}`, { method: 'POST', body });
}

// Poll a render job: { status, output_url?, metrics?, ... }.
export function hyperframesRenderStatus(renderId) {
  return _call(`/api/hyperframes/render/${encodeURIComponent(renderId)}`);
}

// Cancel an in-flight render.
export function cancelHyperframesRender(renderId) {
  return _call(`/api/hyperframes/render/${encodeURIComponent(renderId)}/cancel`, { method: 'POST' });
}

// Composition URL for the <hyperframes-player> element (sandboxed,
// allowlisted local assets only: index.html, base.mp4, fonts).
export function hyperframesCompositionUrl(jobId, compileId) {
  return getApiUrl(`/api/hyperframes/composition/${jobId}/${compileId}/index.html`);
}
