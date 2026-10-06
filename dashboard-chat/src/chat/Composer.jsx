import { useEffect, useRef, useState } from 'react';
import { ArrowUp, Square, Plus, Settings2, ClipboardPaste } from 'lucide-react';

/**
 * ChatGPT-style composer: large rounded box (~26px), textarea on top,
 * bottom row inside the box: [+] attach on the left, gear + dark send on the right.
 * No chip strip. Aspect/length/captions live in the settings popup + caption carousel.
 */
export default function Composer({ onSend, disabled, phase, onOpenSettings }) {
  const [text, setText] = useState('');
  const [sending, setSending] = useState(false);
  const [menuOpen, setMenuOpen] = useState(false);
  const taRef = useRef(null);
  const menuRef = useRef(null);

  // autofocus the composer on mount
  useEffect(() => { taRef.current?.focus(); }, []);

  // auto-grow
  useEffect(() => {
    const el = taRef.current;
    if (!el) return;
    el.style.height = 'auto';
    el.style.height = Math.min(el.scrollHeight, 160) + 'px';
  }, [text]);

  // close attach menu on outside click
  useEffect(() => {
    if (!menuOpen) return;
    const h = (e) => { if (menuRef.current && !menuRef.current.contains(e.target)) setMenuOpen(false); };
    window.addEventListener('pointerdown', h);
    return () => window.removeEventListener('pointerdown', h);
  }, [menuOpen]);

  const busy = phase === 'validating' || phase === 'submitting';

  const doSend = async () => {
    const v = text.trim();
    if (!v || disabled || busy || sending) return;
    setSending(true);
    setText('');
    try { await onSend(v); } finally { setSending(false); }
  };

  const pasteFromClipboard = async () => {
    setMenuOpen(false);
    try {
      const t = await navigator.clipboard.readText();
      if (t && t.trim()) setText((prev) => (prev ? `${prev} ${t.trim()}` : t.trim()));
      taRef.current?.focus();
    } catch { /* clipboard denied — user can paste manually */ }
  };

  const placeholder = phase === 'clipping'
    ? 'Prompt the AI while it clips — e.g. "focus on the funny moments"…'
    : 'Paste a YouTube, Twitch or any video link…';

  const canSend = text.trim() && !disabled && !busy && !sending;

  const iconBtn = {
    width: 34, height: 34, borderRadius: '50%', flexShrink: 0,
    display: 'inline-flex', alignItems: 'center', justifyContent: 'center',
    background: 'transparent', border: 'none', cursor: 'pointer',
    color: 'var(--nc-text-dim)', transition: 'background .15s ease, color .15s ease',
  };

  return (
    <div className="nc-composer" style={{ padding: '10px 16px 16px', background: 'linear-gradient(to top, var(--nc-bg) 70%, transparent)' }}>
      <div style={{ maxWidth: 820, margin: '0 auto' }}>
        <div
          className="nc-card"
          style={{
            borderRadius: 26, padding: '12px 12px 8px',
            background: 'var(--nc-panel)',
            border: '1px solid var(--nc-border)',
            boxShadow: '0 8px 32px rgba(0,0,0,0.35)',
          }}
        >
          <textarea
            ref={taRef}
            className="nc-scroll nc-composer-input"
            value={text}
            onChange={(e) => setText(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); doSend(); }
            }}
            placeholder={placeholder}
            rows={1}
            disabled={disabled}
            style={{
              width: '100%', background: 'transparent', border: 'none', outline: 'none',
              resize: 'none', color: 'var(--nc-text)', fontSize: 15, lineHeight: 1.55,
              padding: '8px 10px', fontFamily: 'inherit', maxHeight: 160, minHeight: 52,
            }}
          />
          <div style={{ display: 'flex', alignItems: 'center', gap: 2, padding: '2px 4px 2px', position: 'relative' }}>
            <div ref={menuRef} style={{ position: 'relative' }}>
              <button
                style={iconBtn}
                onClick={() => setMenuOpen((o) => !o)}
                aria-label="Attach"
                title="Attach"
                onMouseEnter={(e) => { e.currentTarget.style.background = 'var(--nc-card-hover)'; }}
                onMouseLeave={(e) => { e.currentTarget.style.background = 'transparent'; }}
              >
                <Plus size={19} />
              </button>
              {menuOpen && (
                <div
                  className="nc-card nc-anim-fade-up"
                  style={{
                    position: 'absolute', bottom: 42, left: 0, zIndex: 30,
                    borderRadius: 14, padding: 6, minWidth: 220,
                    background: 'var(--nc-panel)', border: '1px solid var(--nc-border)',
                    boxShadow: '0 12px 40px rgba(0,0,0,0.5)',
                  }}
                >
                  <button
                    onClick={pasteFromClipboard}
                    style={{
                      display: 'flex', alignItems: 'center', gap: 10, width: '100%',
                      background: 'transparent', border: 'none', cursor: 'pointer',
                      color: 'var(--nc-text)', fontSize: 13.5, padding: '9px 10px', borderRadius: 9,
                      textAlign: 'left',
                    }}
                    onMouseEnter={(e) => { e.currentTarget.style.background = 'var(--nc-card-hover)'; }}
                    onMouseLeave={(e) => { e.currentTarget.style.background = 'transparent'; }}
                  >
                    <ClipboardPaste size={16} style={{ color: 'var(--nc-text-dim)', flexShrink: 0 }} />
                    Paste link from clipboard
                  </button>
                </div>
              )}
            </div>
            <div style={{ flex: 1 }} />
            <button
              style={iconBtn}
              onClick={onOpenSettings}
              aria-label="Clip settings"
              title="Clip settings"
              onMouseEnter={(e) => { e.currentTarget.style.background = 'var(--nc-card-hover)'; }}
              onMouseLeave={(e) => { e.currentTarget.style.background = 'transparent'; }}
            >
              <Settings2 size={18} />
            </button>
            <button
              onClick={doSend}
              disabled={!canSend}
              aria-label="Send"
              style={{
                width: 34, height: 34, borderRadius: '50%', padding: 0, flexShrink: 0,
                marginLeft: 4, border: 'none', cursor: canSend ? 'pointer' : 'default',
                display: 'inline-flex', alignItems: 'center', justifyContent: 'center',
                background: canSend ? '#2e2e33' : 'rgba(255,255,255,0.08)',
                color: canSend ? '#fff' : 'var(--nc-text-faint)',
                transition: 'background .15s ease, transform .1s ease',
                transform: 'none',
              }}
              onMouseDown={(e) => { if (canSend) e.currentTarget.style.transform = 'scale(0.92)'; }}
              onMouseUp={(e) => { e.currentTarget.style.transform = 'none'; }}
            >
              {busy ? <Square size={14} /> : <ArrowUp size={17} strokeWidth={2.4} />}
            </button>
          </div>
        </div>
        <div style={{ textAlign: 'center', marginTop: 8, fontSize: 11.5, color: 'var(--nc-text-faint)' }}>
          Nugget checks every link before clipping — nothing runs on bad URLs.
        </div>
      </div>
    </div>
  );
}
