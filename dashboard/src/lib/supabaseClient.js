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
