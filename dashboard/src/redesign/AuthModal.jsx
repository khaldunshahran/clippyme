import { useState } from 'react';
import { createPortal } from 'react-dom';
import { useModalA11y } from './useModalA11y';
import { Icon, Btn } from './primitives';
import { signInWithPassword, signInWithOAuth, signUp } from '../lib/supabaseClient';

function GoogleMark() {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" aria-hidden="true">
      <path fill="#4285F4" d="M23.5 12.3c0-.9-.1-1.5-.3-2.3H12v4.5h6.5c-.1 1.1-.8 2.7-2.4 3.8l-.1.1 3.5 2.7.2.1c2.2-2 3.8-5 3.8-8.9z" />
      <path fill="#34A853" d="M12 24c3.2 0 6-1.1 7.9-2.9l-3.8-2.9c-1 .7-2.4 1.2-4.1 1.2-3.1 0-5.8-2.1-6.8-5l-.1.1-3.7 2.9v.1C3.4 21.5 7.4 24 12 24z" />
      <path fill="#FBBC05" d="M5.2 14.4c-.2-.7-.4-1.5-.4-2.4s.1-1.7.4-2.4l-.1-.1-3.7-2.9-.1.1C.5 8.3 0 10.1 0 12s.5 3.7 1.3 5.3l3.9-2.9z" />
      <path fill="#EA4335" d="M12 4.7c1.8 0 3 .8 3.7 1.4l3.3-3.2C17 1.1 14.8 0 12 0 7.4 0 3.4 2.5 1.3 6.7l3.9 2.9c1-2.9 3.7-4.9 6.8-4.9z" />
    </svg>
  );
}

export function AuthModal({ isOpen, onClose, onSuccess }) {
  const [mode, setMode] = useState('signin'); // 'signin' | 'signup'
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [info, setInfo] = useState('');

  const panelRef = useModalA11y(onClose, isOpen);

  if (!isOpen) return null;

  const handleGoogle = () => {
    setError('');
    setInfo('');
    try {
      signInWithOAuth('google');
    } catch (err) {
      setError(err?.message || 'Google sign-in is not available right now.');
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!email.trim() || !password.trim()) {
      setError('Please enter both email and password.');
      return;
    }

    setBusy(true);
    setError('');
    setInfo('');

    try {
      if (mode === 'signup') {
        const data = await signUp({ email: email.trim(), password });
        if (data?.user && !data.session && !data.access_token) {
          setInfo('Account created! Please check your email to confirm your account, then sign in.');
          setMode('signin');
        } else {
          onSuccess?.(data);
          onClose();
        }
      } else {
        const data = await signInWithPassword({ email: email.trim(), password });
        onSuccess?.(data);
        onClose();
      }
    } catch (err) {
      setError(err?.message || 'Authentication failed. Please check your credentials.');
    } finally {
      setBusy(false);
    }
  };

  const modalNode = (
    <div
      className="overlay"
      onClick={(e) => {
        if (e.target === e.currentTarget && !busy) onClose();
      }}
    >
      <div
        className="modal"
        ref={panelRef}
        role="dialog"
        aria-modal="true"
        aria-labelledby="auth-modal-title"
        style={{ maxWidth: 420 }}
      >
        <div className="modal-head">
          <h3 id="auth-modal-title">
            {mode === 'signin' ? 'Sign in to Nugget' : 'Create your account'}
          </h3>
          <button
            type="button"
            className="x"
            onClick={onClose}
            aria-label="Close modal"
            disabled={busy}
          >
            <Icon n="x" />
          </button>
        </div>

        <form onSubmit={handleSubmit}>
          <div className="modal-body" style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
            <Btn
              variant="secondary"
              block
              type="button"
              onClick={handleGoogle}
              disabled={busy}
              style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 10 }}
            >
              <GoogleMark />
              Continue with Google
            </Btn>

            <div style={{ display: 'flex', alignItems: 'center', gap: 12, color: 'var(--fg-3)', fontSize: 12 }}>
              <div style={{ flex: 1, height: 1, background: 'var(--line, rgba(255,255,255,0.12))' }} />
              or continue with email
              <div style={{ flex: 1, height: 1, background: 'var(--line, rgba(255,255,255,0.12))' }} />
            </div>

            {error && (
              <div
                style={{
                  background: 'rgba(235, 87, 87, 0.12)',
                  border: '1px solid var(--danger)',
                  borderRadius: 8,
                  padding: '10px 12px',
                  color: 'var(--danger)',
                  fontSize: 13,
                  display: 'flex',
                  alignItems: 'center',
                  gap: 8,
                }}
              >
                <Icon n="triangle-alert" style={{ width: 16, height: 16, flexShrink: 0 }} />
                <span>{error}</span>
              </div>
            )}

            {info && (
              <div
                style={{
                  background: 'rgba(110, 127, 92, 0.12)',
                  border: '1px solid var(--brand-teal, #6E7F5C)',
                  borderRadius: 8,
                  padding: '10px 12px',
                  color: 'var(--brand-teal, #6E7F5C)',
                  fontSize: 13,
                  display: 'flex',
                  alignItems: 'center',
                  gap: 8,
                }}
              >
                <Icon n="circle-check" style={{ width: 16, height: 16, flexShrink: 0 }} />
                <span>{info}</span>
              </div>
            )}

            <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
              <label htmlFor="auth-email" style={{ fontSize: 12, fontWeight: 600, color: 'var(--fg-2)' }}>
                Email address
              </label>
              <input
                id="auth-email"
                type="email"
                autoComplete="email"
                required
                className="input-field"
                placeholder="you@example.com"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                disabled={busy}
              />
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
              <label htmlFor="auth-password" style={{ fontSize: 12, fontWeight: 600, color: 'var(--fg-2)' }}>
                Password
              </label>
              <input
                id="auth-password"
                type="password"
                autoComplete={mode === 'signup' ? 'new-password' : 'current-password'}
                required
                className="input-field"
                placeholder="••••••••"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                disabled={busy}
              />
            </div>

            <div style={{ fontSize: 12, color: 'var(--fg-3)', textAlign: 'center', marginTop: 4 }}>
              {mode === 'signin' ? (
                <>
                  Don&apos;t have an account?{' '}
                  <button
                    type="button"
                    onClick={() => {
                      setMode('signup');
                      setError('');
                      setInfo('');
                    }}
                    style={{
                      background: 'none',
                      border: 'none',
                      color: 'var(--brand-teal, #6E7F5C)',
                      cursor: 'pointer',
                      fontWeight: 600,
                      padding: 0,
                    }}
                  >
                    Create account
                  </button>
                </>
              ) : (
                <>
                  Already have an account?{' '}
                  <button
                    type="button"
                    onClick={() => {
                      setMode('signin');
                      setError('');
                      setInfo('');
                    }}
                    style={{
                      background: 'none',
                      border: 'none',
                      color: 'var(--brand-teal, #6E7F5C)',
                      cursor: 'pointer',
                      fontWeight: 600,
                      padding: 0,
                    }}
                  >
                    Sign in
                  </button>
                </>
              )}
            </div>
          </div>

          <div className="modal-foot">
            <Btn variant="ghost" onClick={onClose} disabled={busy}>
              Cancel
            </Btn>
            <div className="mf-right">
              <Btn
                type="submit"
                variant="primary"
                loading={busy}
                disabled={busy || !email.trim() || !password.trim()}
              >
                {mode === 'signin' ? 'Sign In' : 'Sign Up'}
              </Btn>
            </div>
          </div>
        </form>
      </div>
    </div>
  );

  return typeof document !== 'undefined' ? createPortal(modalNode, document.body) : modalNode;
}
