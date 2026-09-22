// Optional API token for deliberate LAN deployments (CLIPPYME_API_TOKEN)
// and multi-tenant Bearer JWT authentication for Supabase SaaS accounts.
// Tokens live in localStorage so the static frontend needs no build-time secrets.

const API_TOKEN_KEY = 'clippyme_api_token';
const AUTH_TOKEN_KEY = 'clippyme_auth_token';

function storage() {
  // Node (unit tests) has no localStorage; browsers can throw on access in
  // hardened privacy modes. Either way we degrade to "no token".
  try {
    return typeof localStorage !== 'undefined' ? localStorage : null;
  } catch {
    return null;
  }
}

export function getApiToken() {
  try {
    return storage()?.getItem(API_TOKEN_KEY) || '';
  } catch {
    return '';
  }
}

export function setApiToken(token) {
  try {
    const s = storage();
    if (!s) return;
    const trimmed = (token || '').trim();
    if (trimmed) s.setItem(API_TOKEN_KEY, trimmed);
    else s.removeItem(API_TOKEN_KEY);
  } catch {
    // Persist failure just means the user re-enters the token next session.
  }
}

export function getAuthToken() {
  try {
    return storage()?.getItem(AUTH_TOKEN_KEY) || '';
  } catch {
    return '';
  }
}

export function setAuthToken(token) {
  try {
    const s = storage();
    if (!s) return;
    const trimmed = (token || '').trim();
    if (trimmed) s.setItem(AUTH_TOKEN_KEY, trimmed);
    else s.removeItem(AUTH_TOKEN_KEY);
  } catch {
    // Persist failure
  }
}

/** fetch() that attaches Authorization: Bearer and/or X-API-Token when configured. */
export function apiFetch(url, init = {}) {
  const authToken = getAuthToken();
  const apiToken = getApiToken();

  if (!authToken && !apiToken) return fetch(url, init);

  const headers = { ...(init.headers || {}) };
  if (authToken) {
    headers['Authorization'] = `Bearer ${authToken}`;
  }
  if (apiToken) {
    headers['X-API-Token'] = apiToken;
  }
  return fetch(url, { ...init, headers });
}
