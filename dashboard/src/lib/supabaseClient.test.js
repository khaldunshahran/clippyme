import { afterEach, describe, expect, it, vi } from 'vitest';
/* global Buffer: readonly */
import {
  clearSession,
  getCurrentUser,
  handleOAuthRedirect,
  onAuthStateChange,
} from './supabaseClient';

function fakeJwt(payload) {
  const b64url = (obj) =>
    Buffer.from(JSON.stringify(obj))
      .toString('base64')
      .replace(/\+/g, '-')
      .replace(/\//g, '_')
      .replace(/=+$/, '');
  return `${b64url({ alg: 'HS256', typ: 'JWT' })}.${b64url(payload)}.sig`;
}

describe('handleOAuthRedirect', () => {
  afterEach(() => {
    clearSession();
    window.location.hash = '';
  });

  it('returns false when there is no access token in the fragment', () => {
    window.location.hash = '#error=access_denied';
    expect(handleOAuthRedirect()).toBe(false);
    expect(getCurrentUser()).toBeNull();
  });

  it('recovers a session from the URL fragment and cleans the address bar', () => {
    const token = fakeJwt({ sub: 'user-123', email: 'g@example.com' });
    window.location.hash =
      `#access_token=${token}&token_type=bearer&expires_in=3600&refresh_token=rt-1`;

    expect(handleOAuthRedirect()).toBe(true);

    const user = getCurrentUser();
    expect(user?.id).toBe('user-123');
    expect(user?.email).toBe('g@example.com');
    // Tokens must not linger in the address bar.
    expect(window.location.hash).toBe('');
  });

  it('returns false for a malformed token instead of throwing', () => {
    window.location.hash = '#access_token=not-a-jwt';
    expect(handleOAuthRedirect()).toBe(true); // session saved, user null
    expect(getCurrentUser()).toBeNull();
  });

  it('notifies subscribers synchronously so a pre-registered listener sees the sign-in', () => {
    // Regression: the app subscribes to auth changes BEFORE calling
    // handleOAuthRedirect(), and relies on the SIGNED_IN notification to update
    // its user state. If the notification were async, the sidebar would keep
    // showing "Sign in" after a successful Google login.
    const seen = [];
    const unsub = onAuthStateChange((event, session) => {
      seen.push([event, session?.user?.id]);
    });
    try {
      const token = fakeJwt({ sub: 'user-456', email: 'h@example.com' });
      window.location.hash = `#access_token=${token}&token_type=bearer&expires_in=3600`;
      expect(handleOAuthRedirect()).toBe(true);
      expect(seen).toEqual([['SIGNED_IN', 'user-456']]);
    } finally {
      unsub();
    }
  });
});

describe('refreshAccessToken / ensureFreshAccessToken', () => {
  const SESSION_KEY = 'clippyme_auth_session';

  // The module reads VITE_SUPABASE_URL at import time, so re-import it fresh
  // with the env stubbed for the auth-enabled path.
  async function importWithAuth() {
    vi.resetModules();
    vi.stubEnv('VITE_SUPABASE_URL', 'https://xyz.supabase.co');
    vi.stubEnv('VITE_SUPABASE_ANON_KEY', 'anon-key');
    return import('./supabaseClient');
  }

  function seedSession(overrides = {}) {
    localStorage.setItem(
      SESSION_KEY,
      JSON.stringify({
        access_token: 'old-access',
        refresh_token: 'rt-1',
        token_type: 'bearer',
        expires_in: 3600,
        expires_at: Math.floor(Date.now() / 1000) + 3600,
        user: { id: 'u1', email: 'g@example.com' },
        ...overrides,
      }),
    );
  }

  function okFetch(newToken = 'new-access') {
    return vi.fn(async () => ({
      ok: true,
      json: () =>
        Promise.resolve({
          access_token: newToken,
          refresh_token: 'rt-2',
          token_type: 'bearer',
          expires_in: 3600,
          expires_at: Math.floor(Date.now() / 1000) + 3600,
          user: { id: 'u1', email: 'g@example.com' },
        }),
    }));
  }

  afterEach(() => {
    localStorage.clear();
    vi.unstubAllGlobals();
    vi.unstubAllEnvs();
    vi.resetModules();
  });

  it('returns null without calling fetch when auth is not enabled', async () => {
    const fetchMock = vi.fn();
    vi.stubGlobal('fetch', fetchMock);
    // Static import from the top of this file was evaluated without env.
    const { refreshAccessToken, ensureFreshAccessToken } = await import('./supabaseClient');
    expect(await refreshAccessToken()).toBeNull();
    expect(await ensureFreshAccessToken()).toBeNull();
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it('exchanges the refresh token and persists the new session + auth token', async () => {
    const mod = await importWithAuth();
    seedSession({ expires_at: Math.floor(Date.now() / 1000) - 10 }); // already expired
    const fetchMock = okFetch();
    vi.stubGlobal('fetch', fetchMock);

    const token = await mod.refreshAccessToken();

    expect(token).toBe('new-access');
    expect(fetchMock).toHaveBeenCalledTimes(1);
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toContain('grant_type=refresh_token');
    expect(JSON.parse(init.body).refresh_token).toBe('rt-1');
    // localStorage session AND the apiFetch bearer token are both updated.
    expect(JSON.parse(localStorage.getItem(SESSION_KEY)).access_token).toBe('new-access');
    expect(localStorage.getItem('clippyme_auth_token')).toBe('new-access');
  });

  it('proactively refreshes when the token expires within the skew window', async () => {
    const mod = await importWithAuth();
    seedSession({ expires_at: Math.floor(Date.now() / 1000) + 60 });
    const fetchMock = okFetch();
    vi.stubGlobal('fetch', fetchMock);

    expect(await mod.ensureFreshAccessToken()).toBe('new-access');
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it('does not refresh a healthy session', async () => {
    const mod = await importWithAuth();
    seedSession(); // expires in an hour
    const fetchMock = okFetch();
    vi.stubGlobal('fetch', fetchMock);

    expect(await mod.ensureFreshAccessToken()).toBe('old-access');
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it('clears the session when the refresh token is rejected', async () => {
    const mod = await importWithAuth();
    seedSession({ expires_at: Math.floor(Date.now() / 1000) - 10 });
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => ({
        ok: false,
        json: () => Promise.resolve({ error_description: 'Invalid Refresh Token' }),
      })),
    );

    expect(await mod.refreshAccessToken()).toBeNull();
    expect(localStorage.getItem(SESSION_KEY)).toBeNull();
    expect(localStorage.getItem('clippyme_auth_token')).toBeNull();
  });
});
