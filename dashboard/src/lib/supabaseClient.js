// Supabase Auth client for ClippyMe SaaS multi-tenancy.
// Communicates with Supabase Auth REST endpoints with zero external dependencies.
// Synchronizes the current session JWT with setAuthToken in apiToken.js.

import { setAuthToken } from './apiToken.js';

const SESSION_STORAGE_KEY = 'clippyme_auth_session';

const SUPABASE_URL = (import.meta.env?.VITE_SUPABASE_URL || '').replace(/\/+$/, '');
const SUPABASE_ANON_KEY = import.meta.env?.VITE_SUPABASE_ANON_KEY || '';

function storage() {
  try {
    return typeof localStorage !== 'undefined' ? localStorage : null;
  } catch {
    return null;
  }
}

let authSubscribers = [];

export function isAuthEnabled() {
  return Boolean(SUPABASE_URL && SUPABASE_ANON_KEY);
}

export function getSession() {
  try {
    const raw = storage()?.getItem(SESSION_STORAGE_KEY);
    if (!raw) return null;
    const session = JSON.parse(raw);
    // Check if token has expired
    if (session?.expires_at && session.expires_at * 1000 < Date.now()) {
      clearSession();
      return null;
    }
    return session;
  } catch {
    return null;
  }
}

export function getCurrentUser() {
  const session = getSession();
  return session?.user || null;
}

function saveSession(session) {
  try {
    if (!session) {
      clearSession();
      return;
    }
    storage()?.setItem(SESSION_STORAGE_KEY, JSON.stringify(session));
    if (session.access_token) {
      setAuthToken(session.access_token);
    }
    notifySubscribers('SIGNED_IN', session);
  } catch {
    // Storage quota or privacy mode error
  }
}

export function clearSession() {
  try {
    storage()?.removeItem(SESSION_STORAGE_KEY);
    setAuthToken('');
    notifySubscribers('SIGNED_OUT', null);
  } catch {
    // Storage access error
  }
}

function notifySubscribers(event, session) {
  for (const cb of authSubscribers) {
    try {
      cb(event, session);
    } catch (err) {
      console.error('Auth state subscriber error:', err);
    }
  }
}

export function onAuthStateChange(callback) {
  authSubscribers.push(callback);
  return () => {
    authSubscribers = authSubscribers.filter((cb) => cb !== callback);
  };
}

/**
 * Sign up a new user with email and password via Supabase Auth REST API.
 */
export async function signUp({ email, password }) {
  if (!isAuthEnabled()) {
    throw new Error('Supabase Auth is not configured. Please set VITE_SUPABASE_URL and VITE_SUPABASE_ANON_KEY.');
  }

  const res = await fetch(`${SUPABASE_URL}/auth/v1/signup`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      apikey: SUPABASE_ANON_KEY,
    },
    body: JSON.stringify({ email, password }),
  });

  const data = await res.json();
  if (!res.ok) {
    throw new Error(data.msg || data.error_description || data.message || 'Failed to sign up');
  }

  if (data.access_token) {
    saveSession(data);
  }
  return data;
}

/**
 * Sign in existing user with email and password.
 */
export async function signInWithPassword({ email, password }) {
  if (!isAuthEnabled()) {
    throw new Error('Supabase Auth is not configured. Please set VITE_SUPABASE_URL and VITE_SUPABASE_ANON_KEY.');
  }

  const res = await fetch(`${SUPABASE_URL}/auth/v1/token?grant_type=password`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      apikey: SUPABASE_ANON_KEY,
    },
    body: JSON.stringify({ email, password }),
  });

  const data = await res.json();
  if (!res.ok) {
    throw new Error(data.error_description || data.msg || data.message || 'Invalid login credentials');
  }

  saveSession(data);
  return data;
}

/**
 * Start an OAuth sign-in flow (e.g. 'google') via Supabase Auth.
 * Redirects the browser to the provider; Supabase redirects back to this app.
 */
export function signInWithOAuth(provider) {
  if (!isAuthEnabled()) {
    throw new Error('Supabase Auth is not configured. Please set VITE_SUPABASE_URL and VITE_SUPABASE_ANON_KEY.');
  }
  const redirectTo = window.location.origin + window.location.pathname;
  const url =
    `${SUPABASE_URL}/auth/v1/authorize?provider=${encodeURIComponent(provider)}` +
    `&redirect_to=${encodeURIComponent(redirectTo)}`;
  window.location.href = url;
}

/**
 * Decode the user identity embedded in a Supabase JWT access token.
 * Synchronous so the OAuth redirect can restore the session on first paint.
 */
function userFromAccessToken(accessToken) {
  try {
    const payload = JSON.parse(
      atob(accessToken.split('.')[1].replace(/-/g, '+').replace(/_/g, '/'))
    );
    return {
      id: payload.sub,
      email: payload.email,
      user_metadata: payload.user_metadata || {},
      app_metadata: payload.app_metadata || {},
    };
  } catch {
    return null;
  }
}

/**
 * Handle the OAuth redirect back from Supabase: parses the URL fragment
 * (#access_token=...) into a session and stores it. Cleans the tokens out of
 * the address bar. Returns true when a session was recovered from the URL.
 */
export function handleOAuthRedirect() {
  try {
    const hash = window.location.hash || '';
    if (!hash.includes('access_token=')) return false;
    const params = new URLSearchParams(hash.startsWith('#') ? hash.slice(1) : hash);
    const accessToken = params.get('access_token');
    if (!accessToken) return false;
    const expiresIn = parseInt(params.get('expires_in') || '3600', 10);
    saveSession({
      access_token: accessToken,
      refresh_token: params.get('refresh_token') || '',
      token_type: params.get('token_type') || 'bearer',
      expires_in: expiresIn,
      expires_at: Math.floor(Date.now() / 1000) + expiresIn,
      user: userFromAccessToken(accessToken),
    });
    window.history.replaceState(null, '', window.location.pathname + window.location.search);
    return true;
  } catch {
    return false;
  }
}

/**
 * Sign out and clear stored session tokens.
 */
export async function signOut() {
  const session = getSession();
  if (isAuthEnabled() && session?.access_token) {
    try {
      await fetch(`${SUPABASE_URL}/auth/v1/logout`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          apikey: SUPABASE_ANON_KEY,
          Authorization: `Bearer ${session.access_token}`,
        },
      });
    } catch {
      // Ignore network errors on logout
    }
  }
  clearSession();
}

/**
 * Replace the stored session's user object (e.g. after user_metadata changed
 * server-side) WITHOUT emitting auth events — subscribers already know about
 * this session; this just keeps the cached copy fresh.
 */
export function updateStoredUser(user) {
  try {
    const raw = storage()?.getItem(SESSION_STORAGE_KEY);
    if (!raw) return;
    const session = JSON.parse(raw);
    session.user = { ...(session.user || {}), ...(user || {}) };
    storage()?.setItem(SESSION_STORAGE_KEY, JSON.stringify(session));
  } catch {
    // Storage quota or privacy mode error
  }
}

/**
 * Fetch the fresh user record (including user_metadata) from Supabase Auth.
 * Throws when there is no session or the request fails.
 */
export async function fetchAuthUser() {
  if (!isAuthEnabled()) {
    throw new Error('Supabase Auth is not configured.');
  }
  const session = getSession();
  if (!session?.access_token) {
    throw new Error('Not signed in.');
  }
  const res = await fetch(`${SUPABASE_URL}/auth/v1/user`, {
    headers: {
      apikey: SUPABASE_ANON_KEY,
      Authorization: `Bearer ${session.access_token}`,
    },
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    throw new Error(data.msg || data.error_description || data.message || 'Failed to fetch user');
  }
  updateStoredUser(data.user || data);
  return data.user || data;
}

/**
 * Merge `data` into the signed-in user's user_metadata via Supabase Auth.
 * Only the user themself (via their access token) can read/write their own
 * metadata — this is the per-account key vault's server side.
 * Returns the updated user record.
 */
export async function updateAuthUserData(data) {
  if (!isAuthEnabled()) {
    throw new Error('Supabase Auth is not configured.');
  }
  const session = getSession();
  if (!session?.access_token) {
    throw new Error('Not signed in.');
  }
  const res = await fetch(`${SUPABASE_URL}/auth/v1/user`, {
    method: 'PUT',
    headers: {
      'Content-Type': 'application/json',
      apikey: SUPABASE_ANON_KEY,
      Authorization: `Bearer ${session.access_token}`,
    },
    body: JSON.stringify({ data }),
  });
  const payload = await res.json().catch(() => ({}));
  if (!res.ok) {
    throw new Error(payload.msg || payload.error_description || payload.message || 'Failed to save');
  }
  const user = payload.user || payload;
  updateStoredUser(user);
  return user;
}
