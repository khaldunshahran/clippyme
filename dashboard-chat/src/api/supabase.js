// Supabase client for frontend auth (Google login -> session JWT).
// Uses the PUBLISHABLE key only. The service_role key must NEVER appear here
// (it bypasses Row Level Security and must stay server-side / out of the repo).

function readEnv(name) {
  try {
    if (typeof import.meta !== 'undefined' && import.meta.env && import.meta.env[name]) {
      return import.meta.env[name];
    }
  } catch {
    // import.meta.env is undefined under plain Node (unit tests).
  }
  try {
    if (typeof process !== 'undefined' && process.env && process.env[name]) {
      return process.env[name];
    }
  } catch {
    // ignore
  }
  return '';
}

export const SUPABASE_URL = readEnv('VITE_SUPABASE_URL');
const PUBLISHABLE_KEY = readEnv('VITE_SUPABASE_PUBLISHABLE_KEY');

export function isSupabaseConfigured() {
  return Boolean(SUPABASE_URL && PUBLISHABLE_KEY);
}

let clientPromise = null;

/** Lazily import + create the Supabase client (keeps unit tests dependency-free). */
export async function getSupabase() {
  if (!clientPromise) {
    clientPromise = import('@supabase/supabase-js').then(({ createClient }) =>
      createClient(SUPABASE_URL || 'http://localhost:0', PUBLISHABLE_KEY || 'unconfigured', {
        auth: {
          persistSession: true,
          autoRefreshToken: true, // session auto-refresh; apiFetch always reads the current token
          detectSessionInUrl: true, // handles the OAuth callback (?code= / #access_token)
          storageKey: 'nugget-chat-auth',
        },
      }),
    );
  }
  return clientPromise;
}

/** Current Supabase access token ('' when signed out or unconfigured). */
export async function getAccessToken() {
  try {
    const supabase = await getSupabase();
    const { data } = await supabase.auth.getSession();
    return data?.session?.access_token || '';
  } catch {
    return '';
  }
}

/** Force a token refresh; returns the new access token or ''. */
export async function refreshSessionNow() {
  try {
    const supabase = await getSupabase();
    const { data, error } = await supabase.auth.refreshSession();
    if (error) return '';
    return data?.session?.access_token || '';
  } catch {
    return '';
  }
}

export async function signInWithGoogle() {
  const supabase = await getSupabase();
  const redirectTo =
    typeof window !== 'undefined' ? window.location.origin + window.location.pathname : undefined;
  const { error } = await supabase.auth.signInWithOAuth({
    provider: 'google',
    options: redirectTo ? { redirectTo } : undefined,
  });
  if (error) throw error;
}

/** Clear the legacy localStorage token keys (the X-API-Token path was removed;
 *  stale values must not linger where old code could pick them up). */
export function clearLegacyTokenKeys() {
  try {
    if (typeof window !== 'undefined' && window.localStorage) {
      window.localStorage.removeItem('clippyme_api_token');
      window.localStorage.removeItem('clippyme_auth_token');
    }
  } catch {
    // ignore (private mode etc.)
  }
}

export async function signOut() {
  try {
    const supabase = await getSupabase();
    await supabase.auth.signOut();
  } catch {
    // ignore — local session state is cleared by the caller anyway
  }
  clearLegacyTokenKeys();
}
