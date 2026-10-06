import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';

// Mock the supabase seam so apiFetch is unit-testable under Node.
vi.mock('./supabase.js', () => ({
  getAccessToken: vi.fn(),
  refreshSessionNow: vi.fn(),
}));

import { apiFetch, setAuthEventHandler } from './apiToken.js';
import { getAccessToken, refreshSessionNow } from './supabase.js';

const ok = (body = {}) =>
  new Response(JSON.stringify(body), { status: 200, headers: { 'Content-Type': 'application/json' } });

describe('apiFetch (Supabase session auth)', () => {
  let fetchMock;
  let events;

  beforeEach(() => {
    events = [];
    setAuthEventHandler((t) => events.push(t));
    fetchMock = vi.fn(async () => ok());
    vi.stubGlobal('fetch', fetchMock);
    getAccessToken.mockReset().mockResolvedValue('tok-abc');
    refreshSessionNow.mockReset().mockResolvedValue('');
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    setAuthEventHandler(null);
  });

  it('attaches Authorization: Bearer <session token> and never X-API-Token', async () => {
    await apiFetch('https://x/api/health');
    const [, init] = fetchMock.mock.calls[0];
    expect(init.headers['Authorization']).toBe('Bearer tok-abc');
    expect(init.headers['X-API-Token']).toBeUndefined();
    expect(Object.keys(init.headers).some((k) => k.toLowerCase() === 'x-api-token')).toBe(false);
  });

  it('sends no Authorization header when there is no session', async () => {
    getAccessToken.mockResolvedValue('');
    await apiFetch('https://x/api/health');
    const [, init] = fetchMock.mock.calls[0];
    expect(init.headers['Authorization']).toBeUndefined();
  });

  it('on 401: refreshes once, retries with the fresh token, succeeds', async () => {
    refreshSessionNow.mockResolvedValue('tok-fresh');
    fetchMock
      .mockResolvedValueOnce(new Response('unauthorized', { status: 401 }))
      .mockResolvedValueOnce(ok({ ok: true }));
    const res = await apiFetch('https://x/api/process', { method: 'POST' });
    expect(res.status).toBe(200);
    expect(refreshSessionNow).toHaveBeenCalledTimes(1);
    const [, retryInit] = fetchMock.mock.calls[1];
    expect(retryInit.headers['Authorization']).toBe('Bearer tok-fresh');
    expect(events).toEqual([]);
  });

  it('on 401 with failed refresh: emits signed-out exactly once (no retry loop)', async () => {
    refreshSessionNow.mockResolvedValue('');
    fetchMock.mockResolvedValue(new Response('unauthorized', { status: 401 }));
    const res = await apiFetch('https://x/api/process', { method: 'POST' });
    expect(res.status).toBe(401);
    expect(refreshSessionNow).toHaveBeenCalledTimes(1);
    expect(fetchMock).toHaveBeenCalledTimes(1); // no retry without a fresh token
    expect(events).toEqual(['signed-out']);
  });

  it('on 401 even after refresh+retry: emits signed-out', async () => {
    refreshSessionNow.mockResolvedValue('tok-fresh');
    fetchMock.mockResolvedValue(new Response('unauthorized', { status: 401 }));
    await apiFetch('https://x/api/process', { method: 'POST' });
    expect(fetchMock).toHaveBeenCalledTimes(2); // original + exactly one retry
    expect(events).toEqual(['signed-out']);
  });

  it('on 403: emits forbidden (does not loop to login)', async () => {
    fetchMock.mockResolvedValue(new Response('forbidden', { status: 403 }));
    const res = await apiFetch('https://x/api/process', { method: 'POST' });
    expect(res.status).toBe(403);
    expect(refreshSessionNow).not.toHaveBeenCalled();
    expect(events).toEqual(['forbidden']);
  });
});
