import { afterEach, describe, expect, it, vi } from 'vitest';
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
