import { useEffect, useState } from 'react';
import { Icon } from './icon';
import { SidebarProgress } from './SidebarProgress';
import { DeployBadge } from './DeployBadge';

export const NAV_GROUPS = [
  {
    label: null,
    items: [
      { id: 'home', label: 'Home', icon: 'sparkles' },
      { id: 'create', label: 'Create', icon: 'plus' },
      { id: 'clips', label: 'Clips', icon: 'scissors' },
    ],
  },
  {
    label: 'Grow',
    items: [
      { id: 'trends', label: 'Trend Radar', icon: 'flame' },
      { id: 'channels', label: 'Channels', icon: 'tv' },
      { id: 'analytics', label: 'Analytics', icon: 'bar-chart' },
      { id: 'live', label: 'Live Monitor', icon: 'rss' },
    ],
  },
  {
    label: 'Library',
    items: [
      { id: 'projects', label: 'Projects', icon: 'clapperboard' },
      { id: 'favorites', label: 'Favorites', icon: 'heart' },
      { id: 'collections', label: 'Collections', icon: 'folder' },
      { id: 'history', label: 'History', icon: 'clock' },
      { id: 'highlights', label: 'Highlights', icon: 'star' },
    ],
  },
];

export const GROW_TABS = new Set(['trends', 'channels', 'analytics', 'live']);
export const SETTINGS_TAB = { id: 'settings', label: 'Settings', icon: 'settings' };

function initialsOf(user) {
  const src = user?.name || user?.email || '';
  return src ? src.slice(0, 2).toUpperCase() : 'NG';
}

function NavItem({ item, active, onGo, badge }) {
  return (
    <button
      type="button"
      className={`sb-item${active ? ' active' : ''}`}
      aria-current={active ? 'page' : undefined}
      onClick={() => onGo(item.id)}
    >
      <Icon n={item.icon} />
      <span className="sb-label">{item.label}</span>
      {badge ? <span className="sb-badge" aria-label={`${badge} active`}>{badge}</span> : null}
    </button>
  );
}

export function SidebarNav({ tab, goTab, user, onSignIn, onSignOut, onOpenSite, busy, job }) {
  const go = (id) => {
    goTab(id);
    if (typeof window !== 'undefined' && window.scrollTo) window.scrollTo({ top: 0 });
  };
  return (
    <aside className="sidebar" aria-label="Primary navigation">
      <button type="button" className="sb-brand" onClick={() => go('home')} aria-label="Nugget home">
        <img src="/logo.svg" alt="Nugget logo" />
        <span>
          <span className="sb-brand-name">Nugget</span>
          <span className="sb-brand-sub">Clip studio</span>
        </span>
      </button>

      {NAV_GROUPS.map((group, gi) => (
        <div key={group.label || `g${gi}`}>
          {group.label && <div className="sb-group-label" aria-hidden="true">{group.label}</div>}
          {group.items.map((item) => (
            <NavItem
              key={item.id}
              item={item}
              active={tab === item.id}
              onGo={go}
              badge={item.id === 'create' && busy ? '●' : null}
            />
          ))}
        </div>
      ))}

      {/* Live generation progress — only renders while a job is active. */}
      <SidebarProgress
        status={job?.status}
        step={job?.step}
        logs={job?.logs}
        clipsCount={job?.clipsCount}
        media={job?.media}
        paused={job?.paused}
        onOpen={() => go('create')}
      />

      <div className="sb-footer">
        <DeployBadge />
        <NavItem item={SETTINGS_TAB} active={tab === 'settings'} onGo={go} />
        {onOpenSite && (
          <button type="button" className="sb-site-link" onClick={onOpenSite}>
            <Icon n="globe" />
            <span>View website</span>
          </button>
        )}
        {user ? (
          <button
            type="button"
            className="sb-user"
            onClick={onSignOut}
            title={`Signed in as ${user.email || user.name || 'account'}. Click to sign out.`}
            aria-label="Account — sign out"
          >
            <span className="sb-avatar" aria-hidden="true">{initialsOf(user)}</span>
            <span className="sb-user-meta">
              <span className="sb-user-name">{user.name || user.email || 'Account'}</span>
              <span className="sb-user-sub">Sign out</span>
            </span>
          </button>
        ) : onSignIn ? (
          <button type="button" className="sb-item" onClick={onSignIn}>
            <Icon n="user-round" />
            <span className="sb-label">Sign in</span>
          </button>
        ) : null}
      </div>
    </aside>
  );
}

const MOBILE_PRIMARY = [
  { id: 'home', label: 'Home', icon: 'sparkles' },
  { id: 'create', label: 'Create', icon: 'plus' },
  { id: 'clips', label: 'Clips', icon: 'scissors' },
  { id: 'grow', label: 'Grow', icon: 'flame', target: 'trends', activeWhen: GROW_TABS },
];

export function MobileNav({ tab, goTab, onMore }) {
  const go = (id) => {
    goTab(id);
    if (typeof window !== 'undefined' && window.scrollTo) window.scrollTo({ top: 0 });
  };
  return (
    <nav className="mnav" aria-label="Mobile navigation">
      {MOBILE_PRIMARY.map((item) => {
        const active = item.activeWhen ? item.activeWhen.has(tab) : tab === item.id;
        return (
          <button
            key={item.id}
            type="button"
            className={`mnav-item${active ? ' active' : ''}`}
            aria-current={active ? 'page' : undefined}
            onClick={() => go(item.target || item.id)}
          >
            <Icon n={item.icon} />
            <span>{item.label}</span>
          </button>
        );
      })}
      <button
        type="button"
        className="mnav-item"
        onClick={onMore}
        aria-haspopup="dialog"
        aria-label="More destinations"
      >
        <Icon n="layers" />
        <span>More</span>
      </button>
    </nav>
  );
}

export function MoreSheet({ tab, goTab, onClose, user, onSignIn, onSignOut, onOpenSite }) {
  useEffect(() => {
    const onKey = (e) => { if (e.key === 'Escape') onClose(); };
    window.addEventListener('keydown', onKey);
    document.body.style.overflow = 'hidden';
    return () => {
      window.removeEventListener('keydown', onKey);
      document.body.style.overflow = '';
    };
  }, [onClose]);

  const go = (id) => {
    onClose();
    goTab(id);
    if (typeof window !== 'undefined' && window.scrollTo) window.scrollTo({ top: 0 });
  };

  return (
    <div role="dialog" aria-modal="true" aria-label="More destinations">
      <div className="mnav-sheet-backdrop" onClick={onClose} aria-hidden="true" />
      <div className="mnav-sheet">
        <div className="mnav-sheet-grip" aria-hidden="true" />
        <div className="mnav-sheet-title">Browse Nugget</div>
        {NAV_GROUPS.map((group, gi) => (
          <div key={group.label || `g${gi}`}>
            {group.label && <div className="sb-group-label">{group.label}</div>}
            {group.items.map((item) => (
              <NavItem key={item.id} item={item} active={tab === item.id} onGo={go} />
            ))}
          </div>
        ))}
        <div className="sb-group-label">Account</div>
        <NavItem item={SETTINGS_TAB} active={tab === 'settings'} onGo={go} />
        {onOpenSite && (
          <button type="button" className="sb-item" onClick={() => { onClose(); onOpenSite(); }}>
            <Icon n="globe" />
            <span className="sb-label">View website</span>
          </button>
        )}
        {user ? (
          <button type="button" className="sb-item" onClick={() => { onClose(); onSignOut(); }}>
            <Icon n="user-round" />
            <span className="sb-label">Sign out{user.email ? ` (${user.email})` : ''}</span>
          </button>
        ) : onSignIn ? (
          <button type="button" className="sb-item" onClick={() => { onClose(); onSignIn(); }}>
            <Icon n="user-round" />
            <span className="sb-label">Sign in</span>
          </button>
        ) : null}
      </div>
    </div>
  );
}

export function NotFoundPage({ onHome, onCreate }) {
  return (
    <div className="nf">
      <div className="nf-card">
        <div className="nf-code" aria-hidden="true">404</div>
        <h1>This page got clipped.</h1>
        <p>
          The link you followed doesn't exist anymore — it may have been moved or mistyped.
          Your clips are safe; let's get you back to them.
        </p>
        <div className="nf-row">
          <button type="button" className="btn btn-primary" onClick={onHome}>
            Back to Home
          </button>
          <button type="button" className="btn btn-ghost" onClick={onCreate}>
            Create a clip
          </button>
        </div>
      </div>
    </div>
  );
}

export function Hero({ eyebrow, line1, grad, sub }) {
  return (
    <div className="hero">
      <div>
        {eyebrow && <div className="eyebrow">{eyebrow}</div>}
        <h1>
          {line1}
          {grad && <><em>{grad}</em></>}
        </h1>
        {sub && <p className="lede">{sub}</p>}
      </div>
    </div>
  );
}
