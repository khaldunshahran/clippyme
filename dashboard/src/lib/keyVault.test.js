import { beforeEach, describe, expect, it, vi } from 'vitest';

vi.mock('./supabaseClient', () => ({
  getSession: vi.fn(),
  isAuthEnabled: vi.fn(),
  fetchAuthUser: vi.fn(),
  updateAuthUserData: vi.fn(),
}));

let vault;
let sb;

const META = 'nugget_api_keys';

function signedIn(serverKeys = {}) {
  sb.isAuthEnabled.mockReturnValue(true);
  sb.getSession.mockReturnValue({
    access_token: 'tok-123',
    user: { id: 'u1', user_metadata: { [META]: serverKeys } },
  });
}

function signedOut() {
  sb.isAuthEnabled.mockReturnValue(false);
  sb.getSession.mockReturnValue(null);
}

beforeEach(async () => {
  vi.resetModules();
  localStorage.clear();
  vault = await import('./keyVault');
  sb = await import('./supabaseClient');
  signedOut();
  sb.fetchAuthUser.mockReset();
  sb.updateAuthUserData.mockReset();
});

describe('keyVault (signed out)', () => {
  it('returns empty string when nothing is stored', () => {
    expect(vault.getVaultKey('gemini')).toBe('');
  });

  it('writes localStorage immediately and never touches the server', async () => {
    vault.setVaultKey('gemini', 'AIza-local');
    expect(vault.getVaultKey('gemini')).toBe('AIza-local');
    expect(localStorage.getItem('gemini_key')).toBe('AIza-local');
    await vault.__flushVaultPushes();
    expect(sb.updateAuthUserData).not.toHaveBeenCalled();
  });

  it('trims whitespace and clears on empty', async () => {
    vault.setVaultKey('gemini', '  AIza-x  ');
    expect(vault.getVaultKey('gemini')).toBe('AIza-x');
    vault.clearVaultKey('gemini');
    expect(vault.getVaultKey('gemini')).toBe('');
    expect(localStorage.getItem('gemini_key')).toBeNull();
    await vault.__flushVaultPushes();
    expect(sb.updateAuthUserData).not.toHaveBeenCalled();
  });
});

describe('keyVault (signed in)', () => {
  it('pushes the key to user_metadata, merging existing vault keys', async () => {
    signedIn({ gemini: 'old', other: 'keep-me' });
    vault.setVaultKey('gemini', 'AIza-new');
    // Not pushed synchronously (debounced)…
    expect(sb.updateAuthUserData).not.toHaveBeenCalled();
    await vault.__flushVaultPushes();
    expect(sb.updateAuthUserData).toHaveBeenCalledTimes(1);
    expect(sb.updateAuthUserData).toHaveBeenCalledWith({
      [META]: { gemini: 'AIza-new', other: 'keep-me' },
    });
  });

  it('skips the server push when the value is unchanged', async () => {
    signedIn({ gemini: 'AIza-same' });
    sb.fetchAuthUser.mockResolvedValue({
      id: 'u1',
      user_metadata: { [META]: { gemini: 'AIza-same' } },
    });
    localStorage.setItem('gemini_key', 'AIza-same');
    await vault.syncVaultOnSignIn(); // marks lastPushed
    expect(sb.updateAuthUserData).not.toHaveBeenCalled();
    vault.setVaultKey('gemini', 'AIza-same');
    await vault.__flushVaultPushes();
    expect(sb.updateAuthUserData).not.toHaveBeenCalled();
  });

  it('pushes removals so clearing propagates to the account', async () => {
    signedIn({ gemini: 'AIza-gone' });
    vault.clearVaultKey('gemini');
    await vault.__flushVaultPushes();
    expect(sb.updateAuthUserData).toHaveBeenCalledWith({ [META]: {} });
  });

  it('never throws when the server push fails (local cache still wins)', async () => {
    signedIn({});
    sb.updateAuthUserData.mockRejectedValue(new Error('offline'));
    expect(() => vault.setVaultKey('gemini', 'AIza-offline')).not.toThrow();
    await vault.__flushVaultPushes();
    expect(vault.getVaultKey('gemini')).toBe('AIza-offline');
  });
});

describe('syncVaultOnSignIn', () => {
  it('pulls a server key into the empty local cache', async () => {
    signedIn({ gemini: 'AIza-server' });
    sb.fetchAuthUser.mockResolvedValue({
      id: 'u1',
      user_metadata: { [META]: { gemini: 'AIza-server' } },
    });
    await vault.syncVaultOnSignIn();
    expect(vault.getVaultKey('gemini')).toBe('AIza-server');
    expect(sb.updateAuthUserData).not.toHaveBeenCalled();
  });

  it('adopts a local-only key up to the server once', async () => {
    signedIn({});
    localStorage.setItem('gemini_key', 'AIza-local-only');
    sb.fetchAuthUser.mockResolvedValue({ id: 'u1', user_metadata: {} });
    await vault.syncVaultOnSignIn();
    expect(sb.updateAuthUserData).toHaveBeenCalledWith({
      [META]: { gemini: 'AIza-local-only' },
    });
    expect(vault.getVaultKey('gemini')).toBe('AIza-local-only');
  });

  it('server wins on conflict', async () => {
    signedIn({ gemini: 'AIza-server' });
    localStorage.setItem('gemini_key', 'AIza-stale-local');
    sb.fetchAuthUser.mockResolvedValue({
      id: 'u1',
      user_metadata: { [META]: { gemini: 'AIza-server' } },
    });
    await vault.syncVaultOnSignIn();
    expect(vault.getVaultKey('gemini')).toBe('AIza-server');
    expect(sb.updateAuthUserData).not.toHaveBeenCalled();
  });

  it('does nothing when signed out', async () => {
    signedOut();
    await vault.syncVaultOnSignIn();
    expect(sb.fetchAuthUser).not.toHaveBeenCalled();
  });

  it('never throws when the fetch fails', async () => {
    signedIn({});
    localStorage.setItem('gemini_key', 'AIza-keep');
    sb.fetchAuthUser.mockRejectedValue(new Error('offline'));
    await expect(vault.syncVaultOnSignIn()).resolves.toBeUndefined();
    expect(vault.getVaultKey('gemini')).toBe('AIza-keep');
  });
});
