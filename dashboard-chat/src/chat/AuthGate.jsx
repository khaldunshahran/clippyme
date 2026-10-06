import { useCallback, useEffect, useRef, useState } from 'react';
import { Loader2 } from 'lucide-react';
import { getSupabase, signOut } from '../api/supabase.js';
import { setAuthEventHandler } from '../api/apiToken.js';
import { LoginView, ForbiddenView, AuthToast } from './AuthScreens.jsx';

/**
 * Session gate for the whole app shell.
 * - 'loading': resolving the Supabase session (also handles the OAuth callback
 *   via detectSessionInUrl).
 * - 'login': no session -> Google login screen.
 * - 'forbidden': backend answered 403 NOT_ALLOWLISTED (valid token, not on the
 *   allow-list) -> full-screen "Not authorized".
 * - 'app': session present -> the real app. ADMIN_ONLY / ORIGIN_REJECTED 403s
 *   surface as a toast here instead of a takeover.
 */
export default function AuthGate({ children }) {
  const [state, setState] = useState('loading');
  const [toast, setToast] = useState('');
  const unsubRef = useRef(null);

  const goLogin = useCallback(() => {
    signOut().catch(() => {});
    setState('login');
  }, []);

  const dismissToast = useCallback(() => setToast(''), []);

  useEffect(() => {
    let alive = true;

    // Backend-driven auth events from apiFetch.
    // 'signed-out' (401 refresh-exhausted) -> login.
    // 'forbidden' (403 NOT_ALLOWLISTED) -> full-screen.
    // 'forbidden-toast' (403 ADMIN_ONLY / ORIGIN_REJECTED) -> toast notice.
    setAuthEventHandler((type, detail) => {
      if (!alive) return;
      if (type === 'signed-out') goLogin();
      else if (type === 'forbidden') setState('forbidden');
      else if (type === 'forbidden-toast') {
        setToast(
          detail === 'ADMIN_ONLY'
            ? 'This action needs admin access.'
            : 'Request blocked: this origin is not allowed.',
        );
      }
    });

    (async () => {
      try {
        const supabase = await getSupabase();
        const { data } = await supabase.auth.getSession();
        if (!alive) return;
        setState(data?.session ? 'app' : 'login');
        const { data: sub } = supabase.auth.onAuthStateChange((event) => {
          if (!alive) return;
          if (event === 'SIGNED_IN') setState('app');
          else if (event === 'SIGNED_OUT') setState('login');
        });
        unsubRef.current = sub?.subscription || null;
      } catch {
        if (alive) setState('login');
      }
    })();

    return () => {
      alive = false;
      setAuthEventHandler(null);
      try { unsubRef.current?.unsubscribe(); } catch { /* ignore */ }
      unsubRef.current = null;
    };
  }, [goLogin]);

  if (state === 'loading') {
    return (
      <div style={{
        display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%',
        background: 'var(--nc-bg)', color: 'var(--nc-text-dim)',
      }}>
        <Loader2 size={22} style={{ animation: 'nc-spin 1s linear infinite' }} />
      </div>
    );
  }
  if (state === 'login') return <LoginView />;
  if (state === 'forbidden') return <ForbiddenView onSignOut={goLogin} />;
  return (
    <>
      {children}
      <AuthToast message={toast} onClose={dismissToast} />
    </>
  );
}
