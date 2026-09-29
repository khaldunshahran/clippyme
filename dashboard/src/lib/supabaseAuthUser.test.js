import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

const URL = 'https://example.supabase.co';
const ANON = 'anon-key-for-tests';
const SESSION_KEY = 'clippyme_auth_session';

let sb;

function seedSession(user = { id: 'u1', email: 't@example.com' }) {
  localStorage.setItem(
    SESSION_KEY,
    JSON.stringify({
      access_token: 'tok-abc',
      refresh_token: 'ref-abc',
      expires_at: Math.floor(Date.now() / 1000) + 3600,
      user,
    })
  );
}

beforeEach(async () => {
  vi.resetModules();
  localStorage.clear();
  vi.stubEnv('VITE_SUPABASE_URL', URL);
  vi.stubEnv('VITE_SUPABASE_ANON_KEY', ANON);
  sb = await import('./supabaseClient');
  vi.unstubAllEnvs();
  vi.stubGlobal('fetch', vi.fn());
});

afterEach(() => {
  vi.unstubAllEnvs();
  vi.unstubAllGlobals();
});

describe('updateAuthUserData', () => {
  it('PUTs merged metadata with the user token and updates the stored session', async () => {
    seedSession();
    const updatedUser = {
      id: 'u1',
      user_metadata: { nugget_api_keys: { gemini: 'AIza-k' } },
    };
    vi.mocked(fetch).mockResolvedValue({ ok: true, json: async () => updatedUser });

    const user = await sb.updateAuthUserData({ nugget_api_keys: { gemini: 'AIza-k' } });

    expect(fetch).toHaveBeenCalledTimes(1);
    const [url, opts] = vi.mocked(fetch).mock.calls[0];
    expect(url).toBe(`${URL}/auth/v1/user`);
    expect(opts.method).toBe('PUT');
    expect(opts.headers.Authorization).toBe('Bearer tok-abc');
    expect(opts.headers.apikey).toBe(ANON);
    expect(JSON.parse(opts.body)).toEqual({ data: { nugget_api_keys: { gemini: 'AIza-k' } } });

    expect(user).toEqual(updatedUser);
    // Stored session's user copy is refreshed without a reload.
    expect(sb.getCurrentUser().user_metadata.nugget_api_keys.gemini).toBe('AIza-k');
    expect(sb.getCurrentUser().email).toBe('t@example.com'); // untouched fields survive
  });

  it('throws without calling fetch when not signed in', async () => {
    await expect(sb.updateAuthUserData({ a: 1 })).rejects.toThrow('Not signed in');
    expect(fetch).not.toHaveBeenCalled();
  });

  it('surfaces the server error message', async () => {
    seedSession();
    vi.mocked(fetch).mockResolvedValue({
      ok: false,
      json: async () => ({ msg: 'bad request' }),
    });
    await expect(sb.updateAuthUserData({ a: 1 })).rejects.toThrow('bad request');
  });
});

describe('fetchAuthUser', () => {
  it('GETs the user and refreshes the stored session copy', async () => {
    seedSession({ id: 'u1' });
    vi.mocked(fetch).mockResolvedValue({
      ok: true,
      json: async () => ({ id: 'u1', user_metadata: { nugget_api_keys: { gemini: 's' } } }),
    });
    const user = await sb.fetchAuthUser();
    const [url, opts] = vi.mocked(fetch).mock.calls[0];
    expect(url).toBe(`${URL}/auth/v1/user`);
    expect(opts.headers.Authorization).toBe('Bearer tok-abc');
    expect(user.user_metadata.nugget_api_keys.gemini).toBe('s');
    expect(sb.getCurrentUser().user_metadata.nugget_api_keys.gemini).toBe('s');
  });

  it('throws when not signed in', async () => {
    await expect(sb.fetchAuthUser()).rejects.toThrow('Not signed in');
    expect(fetch).not.toHaveBeenCalled();
  });
});

describe('updateStoredUser', () => {
  it('merges into the stored user without emitting auth events', async () => {
    seedSession({ id: 'u1', email: 't@example.com' });
    const events = [];
    const unsub = sb.onAuthStateChange((e) => events.push(e));
    sb.updateStoredUser({ user_metadata: { nugget_api_keys: { gemini: 'k' } } });
    unsub();
    expect(events).toEqual([]); // no SIGNED_IN re-emitted
    expect(sb.getCurrentUser().id).toBe('u1');
    expect(sb.getCurrentUser().user_metadata.nugget_api_keys.gemini).toBe('k');
  });

  it('is a no-op without a stored session', async () => {
    expect(() => sb.updateStoredUser({ id: 'x' })).not.toThrow();
    expect(sb.getCurrentUser()).toBeNull();
  });
});
