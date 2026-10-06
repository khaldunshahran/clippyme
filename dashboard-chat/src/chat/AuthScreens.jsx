import { useEffect, useState } from 'react';
import { LogIn, Loader2, ShieldAlert, LogOut, TriangleAlert } from 'lucide-react';
import { signInWithGoogle, signOut, isSupabaseConfigured } from '../api/supabase.js';

function Shell({ children }) {
  return (
    <div style={{
      display: 'flex', alignItems: 'center', justifyContent: 'center',
      height: '100%', background: 'var(--nc-bg)', color: 'var(--nc-text)',
      padding: 20,
    }}>
      <div className="nc-card nc-anim-pop-in" style={{
        width: 'min(380px, calc(100vw - 40px))', borderRadius: 18,
        border: '1px solid var(--nc-border)', padding: '30px 28px', textAlign: 'center',
      }}>
        {children}
      </div>
    </div>
  );
}

function GoogleGlyph() {
  // Inline "G" mark (no external image dependency).
  return (
    <svg width="17" height="17" viewBox="0 0 24 24" aria-hidden="true">
      <path fill="#4285F4" d="M23.5 12.3c0-.9-.1-1.5-.3-2.3H12v4.3h6.5c-.1 1.1-.8 2.7-2.4 3.8l-.1.1 3.5 2.7.2.1c2.2-2 3.8-5 3.8-8.7z" />
      <path fill="#34A853" d="M12 24c3.2 0 5.9-1.1 7.9-2.9l-3.8-2.9c-1 .7-2.4 1.2-4.1 1.2-3.1 0-5.8-2.1-6.8-5l-.1.1-3.7 2.9v.1C3.4 21.5 7.4 24 12 24z" />
      <path fill="#FBBC05" d="M5.2 14.4c-.2-.7-.4-1.5-.4-2.4s.1-1.7.4-2.4l-.1-.1-3.6-2.8-.1.1C.5 8.5 0 10.1 0 12s.5 3.5 1.4 5.1l3.8-2.7z" />
      <path fill="#EA4335" d="M12 4.7c1.8 0 3 .8 3.7 1.4l3.3-3.2C17.9 1.1 15.2 0 12 0 7.4 0 3.4 2.5 1.4 6.9l3.8 2.8c1-2.9 3.7-5 6.8-5z" />
    </svg>
  );
}

/** Google sign-in. Shown whenever there is no Supabase session. */
export function LoginView() {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const configured = isSupabaseConfigured();

  const onLogin = async () => {
    if (busy || !configured) return;
    setBusy(true);
    setError('');
    try {
      await signInWithGoogle(); // redirects to Google; back via OAuth callback
    } catch (e) {
      setBusy(false);
      setError(e?.message || 'Sign-in failed. Please try again.');
    }
  };

  return (
    <Shell>
      <div style={{ fontSize: 20, fontWeight: 800, marginBottom: 6 }}>Nugget Chat</div>
      <div style={{ fontSize: 13, color: 'var(--nc-text-dim)', marginBottom: 22 }}>
        Sign in to start clipping.
      </div>
      {!configured ? (
        <div style={{
          display: 'flex', gap: 8, alignItems: 'flex-start', textAlign: 'left',
          fontSize: 12.5, color: 'var(--nc-text-dim)', background: 'rgba(255,180,0,0.08)',
          border: '1px solid rgba(255,180,0,0.25)', borderRadius: 10, padding: '10px 12px',
        }}>
          <TriangleAlert size={15} style={{ flexShrink: 0, marginTop: 1 }} />
          <span>Auth isn't configured in this build (missing <code>VITE_SUPABASE_URL</code> / <code>VITE_SUPABASE_PUBLISHABLE_KEY</code>).</span>
        </div>
      ) : (
        <button
          className="nc-btn-primary"
          onClick={onLogin}
          disabled={busy}
          style={{
            width: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 10,
            background: '#fff', color: '#1a1a1a', fontWeight: 700, fontSize: 14,
            borderRadius: 12, padding: '11px 16px', border: 'none', cursor: busy ? 'wait' : 'pointer',
          }}
        >
          {busy
            ? <Loader2 size={17} style={{ animation: 'nc-spin 1s linear infinite' }} />
            : <GoogleGlyph />}
          {busy ? 'Redirecting…' : 'Continue with Google'}
        </button>
      )}
      {error && (
        <div style={{ marginTop: 12, fontSize: 12.5, color: 'var(--nc-red)' }}>{error}</div>
      )}
      <div style={{ marginTop: 18, fontSize: 11.5, color: 'var(--nc-text-faint)', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 6 }}>
        <LogIn size={12} /> Secured with Supabase Auth
      </div>
    </Shell>
  );
}

/** Valid session, but the user is not on the backend access allow-list (HTTP 403). */
export function ForbiddenView({ onSignOut }) {
  const [busy, setBusy] = useState(false);
  const onClick = async () => {
    setBusy(true);
    try { await signOut(); } finally { onSignOut && onSignOut(); }
  };
  return (
    <Shell>
      <div style={{
        width: 46, height: 46, borderRadius: '50%', margin: '0 auto 14px',
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        background: 'rgba(255,90,90,0.12)', border: '1px solid rgba(255,90,90,0.3)',
      }}>
        <ShieldAlert size={22} color="#ff6b6b" />
      </div>
      <div style={{ fontSize: 18, fontWeight: 800, marginBottom: 6 }}>Not authorized</div>
      <div style={{ fontSize: 13, color: 'var(--nc-text-dim)', marginBottom: 22, lineHeight: 1.5 }}>
        You're signed in, but this account isn't on the access list for Nugget.
        Ask the workspace owner to add you, then try again.
      </div>
      <button
        className="nc-btn-primary"
        onClick={onClick}
        disabled={busy}
        style={{
          width: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 8,
          borderRadius: 12, padding: '11px 16px', cursor: busy ? 'wait' : 'pointer',
        }}
      >
        {busy
          ? <Loader2 size={16} style={{ animation: 'nc-spin 1s linear infinite' }} />
          : <LogOut size={16} />}
        {busy ? 'Signing out…' : 'Sign out'}
      </button>
    </Shell>
  );
}

/**
 * Minimal auto-dismissing toast for non-blocking auth notices
 * (e.g. ADMIN_ONLY / ORIGIN_REJECTED 403s). Fixed-position, no takeover.
 */
export function AuthToast({ message, onClose }) {
  useEffect(() => {
    if (!message) return undefined;
    const t = setTimeout(onClose, 4500);
    return () => clearTimeout(t);
  }, [message, onClose]);

  if (!message) return null;

  return (
    <div
      role="alert"
      style={{
        position: 'fixed', left: '50%', bottom: 28, transform: 'translateX(-50%)',
        zIndex: 9999, maxWidth: 'min(440px, calc(100vw - 40px))',
        background: 'rgba(22,22,28,0.97)', color: 'var(--nc-text)',
        border: '1px solid rgba(255,176,32,0.4)', borderRadius: 12,
        padding: '12px 16px', fontSize: 13, lineHeight: 1.5,
        display: 'flex', gap: 10, alignItems: 'flex-start',
        boxShadow: '0 10px 34px rgba(0,0,0,0.5)',
      }}
    >
      <TriangleAlert size={16} style={{ flexShrink: 0, marginTop: 2, color: '#ffb020' }} />
      <span>{message}</span>
    </div>
  );
}
