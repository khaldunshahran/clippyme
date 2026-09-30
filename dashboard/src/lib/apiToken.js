// Optional API token for deliberate LAN deployments (CLIPPYME_API_TOKEN)
// and multi-tenant Bearer JWT authentication for Supabase SaaS accounts.
// Tokens live in localStorage so the static frontend needs no build-time secrets.

const API_TOKEN_KEY = 'clippyme_api_token';
const AUTH_TOKEN_KEY = 'clippyme_auth_token';

/** Default per-request timeout: reads must never hang the UI forever. */
export const DEFAULT_API_TIMEOUT_MS = 30_000;

/** Thrown when a request exceeds its timeout (distinct from network failure). */
export class TimeoutError extends Error {
  constructor(message = 'Request timed out — the connection to the backend dropped. Try again.') {
    super(message);
    this.name = 'TimeoutError';
    this.timedOut = true;
  }
}

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

// --- Auth token refresh hook ------------------------------------------------
// supabaseClient registers its refresh implementation here (one-way
// registration, so apiToken never imports supabaseClient and no import cycle
// forms). The refresher is `async ({ force }) => accessToken | null`:
//   - force=false: proactively refresh when the token is near expiry
//     (cheap no-op otherwise);
//   - force=true:  refresh unconditionally (used on HTTP 401).
// A null return means "no usable session" — the caller then surfaces a
// re-sign-in prompt instead of retrying.
let tokenRefresher = null;

export function setTokenRefresher(fn) {
  tokenRefresher = typeof fn === 'function' ? fn : null;
}

/**
 * fetch() that attaches Authorization: Bearer and/or X-API-Token when
 * configured, applies a default 30s timeout (AbortController), and handles
 * expired sessions globally:
 *
 * Options (consumed here, never forwarded to fetch):
 *   timeoutMs  — per-request timeout; 0/negative/Infinity disables it.
 *                Defaults to 30s. Long uploads pass a larger value.
 *   signal     — caller's AbortController signal; combined with the timeout
 *                (a caller abort surfaces as AbortError, a timeout as
 *                TimeoutError).
 *
 * On HTTP 401 it attempts exactly one token refresh + retry; when refresh
 * fails it throws a friendly re-sign-in error (status 401, reauthRequired)
 * instead of dumping the raw response.
 */
export async function apiFetch(url, init = {}) {
  const { timeoutMs = DEFAULT_API_TIMEOUT_MS, _retried401, ...rest } = init;

  // Proactive refresh: renew the JWT when it is near expiry BEFORE the
  // request goes out, so long-lived tabs don't 401 mid-session. Never
  // allowed to break the request itself.
  if (tokenRefresher) {
    try {
      await tokenRefresher({ force: false });
    } catch {
      /* fall through with whatever token is stored */
    }
  }

  const authToken = getAuthToken();
  const apiToken = getApiToken();

  const headers = { ...(rest.headers || {}) };
  if (authToken) {
    headers['Authorization'] = `Bearer ${authToken}`;
  }
  if (apiToken) {
    headers['X-API-Token'] = apiToken;
  }

  const callerSignal = rest.signal;
  const controller = new AbortController();
  let timer = null;
  let timedOut = false;
  const ms = Number(timeoutMs);
  if (Number.isFinite(ms) && ms > 0) {
    timer = setTimeout(() => {
      timedOut = true;
      controller.abort();
    }, ms);
  }
  const onCallerAbort = () => controller.abort();
  if (callerSignal) {
    if (callerSignal.aborted) controller.abort();
    else callerSignal.addEventListener('abort', onCallerAbort, { once: true });
  }

  const fetchInit = { ...rest, signal: controller.signal };
  // Preserve the old contract: no `headers` key at all when nothing was
  // injected and the caller passed none.
  if (authToken || apiToken || rest.headers) fetchInit.headers = headers;
  else delete fetchInit.headers;

  let res;
  try {
    res = await fetch(url, fetchInit);
  } catch (e) {
    if (timer) clearTimeout(timer);
    if (callerSignal) callerSignal.removeEventListener('abort', onCallerAbort);
    // A caller-initiated abort stays an AbortError so callers can tell
    // "user cancelled" apart from "the request timed out".
    if (timedOut) throw new TimeoutError();
    throw e;
  }
  if (timer) clearTimeout(timer);
  if (callerSignal) callerSignal.removeEventListener('abort', onCallerAbort);

  // Global 401 path: exactly one refresh + retry, then a friendly prompt.
  if (res.status === 401 && !_retried401 && tokenRefresher) {
    let fresh = null;
    try {
      fresh = await tokenRefresher({ force: true });
    } catch {
      fresh = null;
    }
    if (fresh) {
      return apiFetch(url, { ...init, _retried401: true });
    }
    // The refresher clears the stored session on failure, which emits
    // SIGNED_OUT so the app prompts re-sign-in. The error below is what
    // lands in logs/toasts instead of the raw 401 body.
    const err = new Error('Your session expired — please sign in again.');
    err.status = 401;
    err.reauthRequired = true;
    throw err;
  }

  return res;
}
