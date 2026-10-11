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
// body: { expected_version, compile_id } — compile_id binds the export to the
// exact composition the user previewed (review 01).
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

// Mint a short-lived preview token for one compiled composition (authed
// POST -> { token, expires_in }). The sandboxed preview iframe performs
// plain navigation and cannot attach apiFetch's Bearer <redacted>, so the
// backend serves /api/hyperframes/composition/.../{asset}?token=... without
// get_current_user while the token is valid. The existing Bearer <redacted>
// path is unchanged. Never a Supabase JWT in the URL.
export function getHyperframesPreviewToken(jobId, compileId) {
  return _call(`/api/hyperframes/preview-token/${jobId}/${compileId}`, { method: 'POST' });
}

// Composition URL for the sandboxed preview iframe. `token` (from
// getHyperframesPreviewToken) authorizes the otherwise-authed asset routes;
// without it the route requires the Bearer <redacted>
export function hyperframesCompositionUrl(jobId, compileId, token) {
  const base = `/api/hyperframes/composition/${jobId}/${compileId}/index.html`;
  return getApiUrl(token ? `${base}?token=${encodeURIComponent(token)}` : base);
}
