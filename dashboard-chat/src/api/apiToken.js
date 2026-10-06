// Authenticated fetch for the Nugget backend.
//
// Browser callers ALWAYS use the user's Supabase session JWT:
//   Authorization: Bearer <supabase access_token>
//
// The old X-API-Token / clippyme_api_token path was removed (2026-10-07):
// CLIPPYME_API_TOKEN is non-browser-only on the backend and must never ship
// in the frontend bundle.
//
// 401 handling: one silent token refresh, then a single retry; if the retry
// still 401s (or the refresh fails) the 'signed-out' auth event fires and the
// app shell clears the session and shows the login screen.
// 403 handling: the token is valid but the user is not on the access
// allow-list — the 'forbidden' auth event fires and the app shows a
// "Not authorized" screen (it must NOT loop back to login).
import { getAccessToken, refreshSessionNow } from './supabase.js';

let authEventHandler = null;

/** App shell registers a handler: (type: 'signed-out' | 'forbidden') => void */
export function setAuthEventHandler(fn) {
  authEventHandler = typeof fn === 'function' ? fn : null;
}

function emitAuthEvent(type) {
  try {
    if (authEventHandler) authEventHandler(type);
  } catch {
    // A throwing handler must not break the fetch chain.
  }
}

function withToken(url, init, token) {
  const headers = { ...(init.headers || {}) };
  if (token) headers['Authorization'] = `Bearer ${token}`;
  const { _authRetried, ...rest } = init; // internal flag, never sent on the wire
  return fetch(url, { ...rest, headers });
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
    // Exactly one silent refresh, then one retry with the fresh token.
    let fresh = '';
    try {
      fresh = await refreshSessionNow();
    } catch {
      fresh = '';
    }
    if (fresh) {
      res = await withToken(url, { ...init, _authRetried: true }, fresh);
    }
    if (res.status === 401) {
      emitAuthEvent('signed-out');
    }
    return res;
  }

  if (res.status === 403) {
    emitAuthEvent('forbidden');
  }

  return res;
}
