// Authenticated fetch for the Nugget backend.
//
// Browser callers ALWAYS use the user's Supabase session JWT:
//   Authorization: Bearer <supabase access_token>
//
// The old X-API-Token / clippyme_api_token path was removed (2026-10-07):
// CLIPPYME_API_TOKEN is non-browser-only on the backend and must never ship
// in the frontend bundle.
//
// 401 handling: exactly one silent token refresh (shared across concurrent
// 401s via a single in-flight promise), then a single retry; if the retry
// still 401s (or the refresh fails) the 'signed-out' auth event fires and the
// app shell clears the session and shows the login screen.
//
// 403 handling: the backend returns a machine-readable `code` in the body:
//   NOT_ALLOWLISTED -> 'forbidden' event -> full-screen "Not authorized" page.
//   ADMIN_ONLY / ORIGIN_REJECTED -> 'forbidden-toast' event -> toast notice.
//   Missing/unknown code -> 'forbidden' (safe default; an older backend only
//   emits 403 for the not-allow-listed case).
import { getAccessToken, refreshSessionNow } from './supabase.js';

let authEventHandler = null;

/**
 * App shell registers a handler: (type, detail) => void.
 * Types: 'signed-out' | 'forbidden' (full-screen) | 'forbidden-toast' (toast).
 * detail carries the backend 403 code for the forbidden variants.
 */
export function setAuthEventHandler(fn) {
  authEventHandler = typeof fn === 'function' ? fn : null;
}

function emitAuthEvent(type, detail) {
  try {
    if (authEventHandler) authEventHandler(type, detail);
  } catch {
    // A throwing handler must not break the fetch chain.
  }
}

// Single shared in-flight refresh: concurrent 401s dedupe onto one promise
// instead of each firing its own refreshSession() call.
let refreshPromise = null;
function sharedRefresh() {
  if (!refreshPromise) {
    refreshPromise = Promise.resolve()
      .then(() => refreshSessionNow())
      .finally(() => {
        refreshPromise = null;
      });
  }
  return refreshPromise;
}

/** Test seam: drop any in-flight shared refresh between cases. */
export function _resetRefreshForTests() {
  refreshPromise = null;
}

function withToken(url, init, token) {
  const headers = { ...(init.headers || {}) };
  if (token) headers['Authorization'] = `Bearer ${token}`;
  const { _authRetried, ...rest } = init; // internal flag, never sent on the wire
  return fetch(url, { ...rest, headers });
}

/** Read the backend's machine-readable 403 code ('' when absent/unparseable). */
async function readForbiddenCode(res) {
  try {
    const data = await res.clone().json();
    const code = data && typeof data.code === 'string' ? data.code : '';
    if (code === 'NOT_ALLOWLISTED' || code === 'ADMIN_ONLY' || code === 'ORIGIN_REJECTED') {
      return code;
    }
  } catch {
    // Not JSON, or no code field — caller applies the safe default.
  }
  return '';
}

/** fetch() that attaches the current Supabase session token. */
export async function apiFetch(url, init = {}) {
  let token = '';
  try {
    token = await getAccessToken();
  } catch {
    token = '';
  }

  let res = await withToken(url, init, token);

  if (res.status === 401 && !init._authRetried) {
    // Exactly one refresh attempt (shared), then exactly one retry.
    let fresh = '';
    try {
      fresh = await sharedRefresh();
    } catch {
      fresh = '';
    }
    if (fresh) {
      res = await withToken(url, { ...init, _authRetried: true }, fresh);
    }
    if (res.status === 401) {
      emitAuthEvent('signed-out');
      return res;
    }
    // A retry that lands on 403 (or 200) falls through to the 403 handling.
  }

  if (res.status === 403) {
    const code = await readForbiddenCode(res);
    if (code === 'ADMIN_ONLY' || code === 'ORIGIN_REJECTED') {
      emitAuthEvent('forbidden-toast', code);
    } else {
      emitAuthEvent('forbidden', code || 'NOT_ALLOWLISTED');
    }
  }

  return res;
}
