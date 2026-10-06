import { useEffect, useRef, useState } from 'react';
import { ArrowUp, Square } from 'lucide-react';

import { DEFAULT_SETTINGS } from './useChat.js';
import { CAPTION_OPTIONS, captionLabel } from './constants.js';

const CHIP_DEFS = [
  { key: 'aspect', label: 'Aspect', options: ['9:16', '1:1', '16:9'] },
  { key: 'clipLength', label: 'Length', options: ['all', 'shorts', 'mid', 'long', 'custom'] },
  { key: 'captions', label: 'Captions', options: CAPTION_OPTIONS.map((o) => o.id) },
];

const CHIP_LABELS = {
  aspect: (v) => v,
  clipLength: (v) => ({ all: 'Mix', shorts: '<60s', mid: '1\u20133m', long: '3\u201310m', custom: 'Custom' }[v] || v),
  captions: (v) => captionLabel(v),
};

export default function Composer({ onSend, disabled, phase, settings, onSettingsChange, onOpenSettings }) {
  const [text, setText] = useState('');
  const [sending, setSending] = useState(false);
  const taRef = useRef(null);

  // autofocus the composer on mount
  useEffect(() => { taRef.current?.focus(); }, []);

  // auto-grow
  useEffect(() => {
    const el = taRef.current;
    if (!el) return;
    el.style.height = 'auto';
    el.style.height = Math.min(el.scrollHeight, 160) + 'px';
  }, [text]);

  const busy = phase === 'validating' || phase === 'submitting';

  const doSend = async () => {
    const v = text.trim();
    if (!v || disabled || busy || sending) return;
    setSending(true);
    setText('');
    try { await onSend(v); } finally { setSending(false); }
  };

  const cycleChip = (def) => {
    const cur = settings[def.key];
    const idx = def.options.indexOf(cur);
    const next = def.options[(idx + 1) % def.options.length];
    onSettingsChange({ ...settings, [def.key]: next });
  };

  const placeholder = phase === 'clipping'
    ? 'Prompt the AI while it clips — e.g. "focus on the funny moments"…'
    : 'Paste a YouTube, Twitch or any video link…';

  return (
    <div className="nc-composer" style={{ padding: '10px 16px 16px', background: 'linear-gradient(to top, var(--nc-bg) 70%, transparent)' }}>
      <div style={{ maxWidth: 820, margin: '0 auto' }}>
        <div
          className="nc-card"
          style={{
            borderRadius: 20, padding: '10px 10px 6px',
            background: 'var(--nc-panel)',
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
              resize: 'none', color: 'var(--nc-text)', fontSize: 15, lineHeight: 1.5,
              padding: '8px 10px', fontFamily: 'inherit', maxHeight: 160,
            }}
          />
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '2px 4px 4px' }}>
            <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
              {CHIP_DEFS.map((def) => {
                const val = settings[def.key];
                const isActive = String(val) !== String(DEFAULT_SETTINGS[def.key]);
                return (
                  <button
                    key={def.key}
                    className="nc-chip"
                    data-active={isActive}
                    onClick={() => cycleChip(def)}
                    title={`Tap to change ${def.label} — full options in settings`}
                    style={{ fontSize: 12, padding: '6px 10px' }}
                  >
                    {CHIP_LABELS[def.key](val)}
                  </button>
                );
              })}
              <button className="nc-chip" onClick={onOpenSettings} style={{ fontSize: 12, padding: '6px 10px' }} title="All settings">
                ⚙ Settings
              </button>
            </div>
            <button
              className="nc-btn-primary"
              onClick={doSend}
              disabled={!text.trim() || disabled || busy || sending}
              aria-label="Send"
              style={{ width: 38, height: 38, borderRadius: '50%', padding: 0, flexShrink: 0 }}
            >
              {busy ? <Square size={14} /> : <ArrowUp size={17} />}
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
