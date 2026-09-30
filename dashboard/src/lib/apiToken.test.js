// Node unit tests for the optional API token helper (LAN deploys) and Bearer auth tokens.
import { test, vi } from 'vitest';
import assert from 'node:assert/strict';

// localStorage shim BEFORE importing the module under test (it feature-detects
// per call, so a global set here is picked up).
const store = new Map();
globalThis.localStorage = {
  getItem: (k) => (store.has(k) ? store.get(k) : null),
  setItem: (k, v) => store.set(k, String(v)),
  removeItem: (k) => store.delete(k),
};

const { getApiToken, setApiToken, getAuthToken, setAuthToken, apiFetch, setTokenRefresher, TimeoutError, DEFAULT_API_TIMEOUT_MS } = await import('./apiToken.js');

test('token round-trip + trim', () => {
  store.clear();
  assert.equal(getApiToken(), '');
  setApiToken('  s3cret  ');
  assert.equal(getApiToken(), 's3cret');
});

test('empty/whitespace token clears storage', () => {
  store.clear();
  setApiToken('s3cret');
  setApiToken('   ');
  assert.equal(getApiToken(), '');
  assert.equal(store.size, 0);
});

test('authToken round-trip + trim', () => {
  store.clear();
  assert.equal(getAuthToken(), '');
  setAuthToken('  jwt-bearer-token  ');
  assert.equal(getAuthToken(), 'jwt-bearer-token');
});

test('empty/whitespace authToken clears storage', () => {
  store.clear();
  setAuthToken('jwt-bearer-token');
  setAuthToken('   ');
  assert.equal(getAuthToken(), '');
  assert.equal(store.size, 0);
});

test('apiFetch without token = plain fetch, no header injected', async () => {
  store.clear();
  let captured;
  globalThis.fetch = (url, init) => { captured = { url, init }; return Promise.resolve('ok'); };
  await apiFetch('/api/history', { method: 'GET' });
  assert.equal(captured.url, '/api/history');
  assert.equal(captured.init.headers, undefined);
});

test('apiFetch with token attaches X-API-Token and keeps existing headers', async () => {
  store.clear();
  setApiToken('s3cret');
  let captured;
  globalThis.fetch = (url, init) => { captured = { url, init }; return Promise.resolve('ok'); };
  await apiFetch('/api/compose/j/0', { method: 'POST', headers: { 'Content-Type': 'application/json' } });
  assert.equal(captured.init.headers['X-API-Token'], 's3cret');
  assert.equal(captured.init.headers['Content-Type'], 'application/json');
  assert.equal(captured.init.method, 'POST');
});

test('apiFetch with authToken attaches Authorization: Bearer header', async () => {
  store.clear();
  setAuthToken('my-supabase-jwt');
  let captured;
  globalThis.fetch = (url, init) => { captured = { url, init }; return Promise.resolve('ok'); };
  await apiFetch('/api/status/123', { method: 'GET' });
  assert.equal(captured.init.headers['Authorization'], 'Bearer my-supabase-jwt');
});

// --- C1: timeouts ------------------------------------------------------------

// A fetch mock that hangs until its signal aborts, then rejects the way a
// real browser fetch does on abort.
function hangingFetch() {
  return (url, init) => new Promise((_, reject) => {
    const signal = init?.signal;
    const onAbort = () => reject(Object.assign(new Error('The operation was aborted.'), { name: 'AbortError' }));
    if (!signal) return; // hangs forever
    if (signal.aborted) return onAbort();
    signal.addEventListener('abort', onAbort, { once: true });
  });
}

test('apiFetch aborts a hung request after the default 30s timeout with TimeoutError', async () => {
  store.clear();
  assert.equal(DEFAULT_API_TIMEOUT_MS, 30_000);
  vi.useFakeTimers();
  try {
    let captured;
    globalThis.fetch = (url, init) => { captured = { url, init }; return hangingFetch()(url, init); };
    const p = apiFetch('/api/history');
    const assertion = assert.rejects(p, (e) => e instanceof TimeoutError && e.name === 'TimeoutError');
    await vi.advanceTimersByTimeAsync(30_000);
    await assertion;
    assert.equal(captured.init.signal.aborted, true);
  } finally {
    vi.useRealTimers();
  }
});

test('a custom timeoutMs is honored', async () => {
  store.clear();
  vi.useFakeTimers();
  try {
    globalThis.fetch = hangingFetch();
    const p = apiFetch('/api/history', { timeoutMs: 5000 });
    const assertion = assert.rejects(p, (e) => e.name === 'TimeoutError');
    await vi.advanceTimersByTimeAsync(5000);
    await assertion;
  } finally {
    vi.useRealTimers();
  }
});

test('timeoutMs: 0 disables the timeout (opt-out for long uploads)', async () => {
  store.clear();
  vi.useFakeTimers();
  try {
    globalThis.fetch = hangingFetch();
    let settled = false;
    apiFetch('/api/history', { timeoutMs: 0 }).then(
      () => { settled = true; },
      () => { settled = true; },
    );
    await vi.advanceTimersByTimeAsync(300_000);
    assert.equal(settled, false);
  } finally {
    vi.useRealTimers();
  }
});

test('a caller abort surfaces as AbortError, not TimeoutError', async () => {
  store.clear();
  const controller = new AbortController();
  globalThis.fetch = hangingFetch();
  const p = apiFetch('/api/history', { signal: controller.signal });
  controller.abort();
  await assert.rejects(p, (e) => e.name === 'AbortError');
});

// --- H3: global 401 path -----------------------------------------------------

test('apiFetch retries once after a 401 when the refresher yields a token', async () => {
  store.clear();
  setAuthToken('stale-jwt');
  // Fake refresher mimics the real one: force refresh persists the new token.
  setTokenRefresher(async ({ force }) => {
    if (force) setAuthToken('fresh-jwt');
    return getAuthToken();
  });
  const seen = [];
  globalThis.fetch = (url, init) => {
    seen.push(init.headers?.['Authorization']);
    if (seen.length === 1) return Promise.resolve({ status: 401, ok: false });
    return Promise.resolve({ status: 200, ok: true });
  };
  try {
    const res = await apiFetch('/api/history');
    assert.equal(res.status, 200);
    assert.deepEqual(seen, ['Bearer stale-jwt', 'Bearer fresh-jwt']);
  } finally {
    setTokenRefresher(null);
  }
});

test('apiFetch throws a friendly re-sign-in error when refresh fails after 401', async () => {
  store.clear();
  setAuthToken('stale-jwt');
  setTokenRefresher(async () => null);
  globalThis.fetch = () => Promise.resolve({ status: 401, ok: false });
  try {
    await assert.rejects(
      apiFetch('/api/history'),
      (e) => e.status === 401 && e.reauthRequired === true && /sign in again/i.test(e.message),
    );
  } finally {
    setTokenRefresher(null);
  }
});
