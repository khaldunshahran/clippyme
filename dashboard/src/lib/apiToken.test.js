// Node unit tests for the optional API token helper (LAN deploys) and Bearer auth tokens.
import { test } from 'vitest';
import assert from 'node:assert/strict';

// localStorage shim BEFORE importing the module under test (it feature-detects
// per call, so a global set here is picked up).
const store = new Map();
globalThis.localStorage = {
  getItem: (k) => (store.has(k) ? store.get(k) : null),
  setItem: (k, v) => store.set(k, String(v)),
  removeItem: (k) => store.delete(k),
};

const { getApiToken, setApiToken, getAuthToken, setAuthToken, apiFetch } = await import('./apiToken.js');

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
