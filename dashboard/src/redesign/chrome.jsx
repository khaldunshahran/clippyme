import { useEffect, useState } from 'react';
import { Icon } from './icon';

const TABS = [
  { id: 'create', label: 'Create', shortLabel: 'Create', icon: 'wand-sparkles' },
  { id: 'trends', label: 'Trend Radar', shortLabel: 'Radar', icon: 'flame' },
  { id: 'channels', label: 'Channels', shortLabel: 'Channels', icon: 'tv' },
  { id: 'analytics', label: 'Analytics', shortLabel: 'Analytics', icon: 'bar-chart' },
  { id: 'highlights', label: 'Highlights', shortLabel: 'Highlights', icon: 'sparkles' },
  { id: 'live', label: 'Live Monitor', shortLabel: 'Live', icon: 'rss' },
  { id: 'history', label: 'History', shortLabel: 'History', icon: 'clock' },
  { id: 'settings', label: 'Settings', shortLabel: 'Settings', icon: 'settings' },
];

function useBrowserOnline() {
  const [online, setOnline] = useState(() => typeof navigator === 'undefined' || navigator.onLine !== false);
  useEffect(() => {
    const update = () => setOnline(navigator.onLine !== false);
    window.addEventListener('online', update);
    window.addEventListener('offline', update);
    return () => {
      window.removeEventListener('online', update);
      window.removeEventListener('offline', update);
    };
  }, []);
  return online;
}

export function TopNav({ tab, setTab, busy, user, onSignIn, onSignOut }) {
  const online = useBrowserOnline();
  const status = !online ? 'Offline' : busy ? 'Working' : user ? 'Cloud' : 'Local';
  const initials = user?.email ? user.email.slice(0, 2).toUpperCase() : 'NG';

  return (
    <>
      <header className="topnav">
        <div className="brand" aria-label="Nugget home">
          <div className="mark" aria-hidden="true" />
          <span>Nugget</span>
        </div>

        <nav className="tabs" aria-label="Primary navigation">
          {TABS.map((item) => (
            <button
              key={item.id}
              type="button"
              className={`tab${tab === item.id ? ' active' : ''}`}
              aria-current={tab === item.id ? 'page' : undefined}
              onClick={() => setTab(item.id)}
            >
              <Icon n={item.icon} />
              <span className="lbl">{item.label}</span>
            </button>
          ))}
        </nav>

        <div className="nav-right">
          <span className={`status-dot${online ? '' : ' offline'}`} role="status" aria-live="polite">
            <i
              aria-hidden="true"
              style={
                busy && online
                  ? { background: 'var(--rust)', boxShadow: '0 0 0 3px rgba(187,90,60,0.25)' }
                  : null
              }
            />
            <span className="sd-lbl">{status}</span>
          </span>

          {user ? (
            <button
              type="button"
              className="avatar"
              onClick={onSignOut}
              title={`Signed in as ${user.email || user.id}. Click to sign out.`}
              aria-label="User account"
              style={{ cursor: 'pointer', border: 'none' }}
            >
              {initials}
            </button>
          ) : onSignIn ? (
            <button
              type="button"
              className="btn btn-sm btn-ghost"
              onClick={onSignIn}
              style={{ fontSize: 13, padding: '4px 10px', height: 28 }}
            >
              Sign in
            </button>
          ) : (
            <div className="avatar" aria-hidden="true">CM</div>
          )}
        </div>
      </header>

      <nav className="mobile-bottom-nav" aria-label="Mobile navigation">
        {TABS.map((item) => (
          <button
            key={item.id}
            type="button"
            className={`mb-tab${tab === item.id ? ' active' : ''}`}
            aria-current={tab === item.id ? 'page' : undefined}
            onClick={() => {
              setTab(item.id);
              if (typeof window !== 'undefined' && window.scrollTo) {
                window.scrollTo({ top: 0, behavior: 'smooth' });
              }
            }}
          >
            <span className="mb-icon">
              <Icon n={item.icon} />
            </span>
            <span className="mb-lbl">{item.shortLabel || item.label}</span>
          </button>
        ))}
      </nav>
    </>
  );
}

export function Hero({ eyebrow, line1, grad, sub }) {
  return (
    <div className="hero">
      <div>
        {eyebrow && <div className="eyebrow">{eyebrow}</div>}
        <h1>
          {line1}
          {grad && <> <em>{grad}</em></>}
        </h1>
        {sub && <p className="lede">{sub}</p>}
      </div>
    </div>
  );
}
