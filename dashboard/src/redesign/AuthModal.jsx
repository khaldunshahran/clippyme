import { useState } from 'react';
import { createPortal } from 'react-dom';
import { useModalA11y } from './useModalA11y';
import { Icon, Btn } from './primitives';
import { signInWithPassword, signUp } from '../lib/supabaseClient';

export function AuthModal({ isOpen, onClose, onSuccess }) {
  const [mode, setMode] = useState('signin'); // 'signin' | 'signup'
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [info, setInfo] = useState('');

  const panelRef = useModalA11y(onClose, isOpen);

  if (!isOpen) return null;

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
            {error && (
              <div
                style={{
                  background: 'rgba(235, 87, 87, 0.12)',
                  border: '1px solid var(--danger, #eb5757)',
                  borderRadius: 8,
                  padding: '10px 12px',
                  color: 'var(--danger, #eb5757)',
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
                  background: 'rgba(2, 197, 191, 0.12)',
                  border: '1px solid var(--brand-teal, #02C5BF)',
                  borderRadius: 8,
                  padding: '10px 12px',
                  color: 'var(--brand-teal, #02C5BF)',
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
                      color: 'var(--brand-teal, #02C5BF)',
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
                      color: 'var(--brand-teal, #02C5BF)',
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
