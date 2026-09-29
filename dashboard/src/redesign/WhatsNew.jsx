// Nugget — What's New banner (Home) + full changelog modal.
//
// The banner shows the newest shipped entry until dismissed; dismissal is
// persisted in localStorage. "See all updates" opens the modal with the full
// history. Entries come from redesign/changelog.js — real shipped features
// only.
import { useState } from 'react';
import { createPortal } from 'react-dom';
import { Icon, Btn } from './primitives';
import { useModalA11y } from './useModalA11y';
import { CHANGELOG, dismissWhatsNew, isWhatsNewDismissed } from './changelog';

export function WhatsNewBanner({ entry, onDismiss, onOpenLog }) {
  if (!entry) return null;
  return (
    <section className="wn-banner fade-in" aria-label="What's new">
      <div className="wn-ic" aria-hidden="true">
        <Icon n="megaphone" />
      </div>
      <div className="wn-body">
        <div className="wn-eyebrow">What&apos;s new · {entry.date}</div>
        <div className="wn-title">{entry.title}</div>
        <div className="wn-sub">{entry.body}</div>
      </div>
      <div className="wn-actions">
        <Btn variant="secondary" size="sm" onClick={onOpenLog}>See all updates</Btn>
        <button type="button" className="mini" aria-label="Dismiss what's new" onClick={onDismiss}>
          <Icon n="x" />
        </button>
      </div>
    </section>
  );
}

function ChangelogEntry({ entry }) {
  return (
    <article className="wn-entry">
      <div className="wn-entry-head">
        <span className="wn-tag">{entry.tag}</span>
        <span className="wn-date">{entry.date}</span>
      </div>
      <h4>{entry.title}</h4>
      <p>{entry.body}</p>
      {entry.bullets?.length > 0 && (
        <ul>
          {entry.bullets.map((b, i) => (
            <li key={i}>{b}</li>
          ))}
        </ul>
      )}
    </article>
  );
}

export function WhatsNewModal({ onClose }) {
  const panelRef = useModalA11y(onClose);
  const node = (
    <div className="overlay" onClick={(e) => { if (e.target === e.currentTarget) onClose(); }}>
      <div className="modal" ref={panelRef} role="dialog" aria-modal="true" aria-labelledby="wn-title">
        <div className="modal-head">
          <div>
            <h3 id="wn-title">What&apos;s new in Nugget</h3>
            <div className="mh-sub">Every shipped update, newest first</div>
          </div>
          <button className="x" onClick={onClose} aria-label="Close"><Icon n="x" /></button>
        </div>
        <div className="modal-body">
          {CHANGELOG.map((e) => (
            <ChangelogEntry key={e.id} entry={e} />
          ))}
        </div>
        <div className="modal-foot">
          <div className="mf-right">
            <Btn variant="primary" onClick={onClose}>Done</Btn>
          </div>
        </div>
      </div>
    </div>
  );
  return typeof document !== 'undefined' ? createPortal(node, document.body) : node;
}

/** Banner wired to localStorage dismissal — drop into Home. */
export function WhatsNew({ onOpenLog: externalOpen }) {
  const entry = CHANGELOG[0];
  const [dismissed, setDismissed] = useState(() => (entry ? isWhatsNewDismissed(entry.id) : true));
  const [logOpen, setLogOpen] = useState(false);
  if (!entry || dismissed) return logOpen ? <WhatsNewModal onClose={() => setLogOpen(false)} /> : null;
  const handleDismiss = () => {
    dismissWhatsNew(entry.id);
    setDismissed(true);
  };
  const handleOpenLog = () => {
    if (externalOpen) externalOpen();
    else setLogOpen(true);
  };
  return (
    <>
      <WhatsNewBanner entry={entry} onDismiss={handleDismiss} onOpenLog={handleOpenLog} />
      {logOpen && <WhatsNewModal onClose={() => setLogOpen(false)} />}
    </>
  );
}
