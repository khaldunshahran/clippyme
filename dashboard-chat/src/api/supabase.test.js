import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';

// signOut() lazy-imports @supabase/supabase-js; mock it so the test stays local.
vi.mock('@supabase/supabase-js', () => ({
  createClient: vi.fn(() => ({
    auth: {
      signOut: vi.fn(async () => ({})),
      getSession: vi.fn(async () => ({ data: { session: null } })),
      refreshSession: vi.fn(async () => ({ data: { session: null }, error: null })),
      onAuthStateChange: vi.fn(() => ({ data: { subscription: null } })),
    },
  })),
}));

import { signOut, clearLegacyTokenKeys } from './supabase.js';

function stubWindowWithStorage() {
  const store = new Map();
  const localStorage = {
    getItem: (k) => (store.has(k) ? store.get(k) : null),
    setItem: (k, v) => store.set(k, String(v)),
    removeItem: (k) => store.delete(k),
    clear: () => store.clear(),
  };
  vi.stubGlobal('window', { localStorage });
  return { store, localStorage };
}

describe('signOut legacy key cleanup', () => {
  let realWindow;

  beforeEach(() => {
    realWindow = globalThis.window;
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    if (realWindow === undefined) delete globalThis.window;
    else globalThis.window = realWindow;
  });

  it('clearLegacyTokenKeys removes both legacy keys and nothing else', () => {
    const { store } = stubWindowWithStorage();
    store.set('clippyme_api_token', 'old-token');
    store.set('clippyme_auth_token', 'old-bearer');
    store.set('nugget-chat-auth', 'keep-me');
    clearLegacyTokenKeys();
    expect(store.has('clippyme_api_token')).toBe(false);
    expect(store.has('clippyme_auth_token')).toBe(false);
    expect(store.get('nugget-chat-auth')).toBe('keep-me');
  });

  it('clearLegacyTokenKeys is a no-op without window/localStorage', () => {
    delete globalThis.window;
    expect(() => clearLegacyTokenKeys()).not.toThrow();
  });

  it('signOut clears the legacy keys as well as the Supabase session', async () => {
    const { store } = stubWindowWithStorage();
    store.set('clippyme_api_token', 'old-token');
    store.set('clippyme_auth_token', 'old-bearer');
    await signOut();
    expect(store.has('clippyme_api_token')).toBe(false);
    expect(store.has('clippyme_auth_token')).toBe(false);
  });
});
