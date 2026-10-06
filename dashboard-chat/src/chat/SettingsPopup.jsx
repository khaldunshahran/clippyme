import { useEffect, useState } from 'react';
import { X, Clapperboard, Clock } from 'lucide-react';
import {
  MODEL_OPTIONS, GENRE_OPTIONS, STYLE_OPTIONS,
  ASPECT_OPTIONS, LENGTH_OPTIONS, parseTime, formatTime,
} from './constants.js';
import { fmtDur } from '../api/chatApi.js';

function Segmented({ options, value, onChange, renderLabel }) {
  return (
    <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
      {options.map((opt) => {
        const id = typeof opt === 'string' ? opt : opt.id;
        const label = renderLabel ? renderLabel(opt) : (typeof opt === 'string' ? opt : opt.label);
        const active = value === id;
        return (
          <button
            key={id}
            className="nc-chip"
            data-active={active}
            onClick={() => onChange(id)}
            style={{ fontSize: 12.5 }}
          >
            {label}
          </button>
        );
      })}
    </div>
  );
}

export default function SettingsPopup({ validation, initial, onConfirm, onClose }) {
  const [s, setS] = useState(initial);
  const [tfStart, setTfStart] = useState('');
  const [tfEnd, setTfEnd] = useState('');
  const [useTimeframe, setUseTimeframe] = useState(false);

  // reset when a new validation arrives
  useEffect(() => {
    setS(initial);
    setTfStart(''); setTfEnd(''); setUseTimeframe(false);
  }, [validation?.title, initial]);

  // esc closes
  useEffect(() => {
    const h = (e) => { if (e.key === 'Escape') onClose(); };
    window.addEventListener('keydown', h);
    return () => window.removeEventListener('keydown', h);
  }, [onClose]);

  const set = (patch) => setS((prev) => ({ ...prev, ...patch }));
  const dur = validation?.duration;

  const confirm = () => {
    const out = { ...s };
    if (useTimeframe) {
      const start = parseTime(tfStart) ?? 0;
      const end = parseTime(tfEnd);
      if (end != null && end > start) {
        out.timeframe = {
          start,
          end: dur ? Math.min(end, dur) : end, // clamp to source duration
        };
      } else {
        out.timeframe = null;
      }
    } else {
      out.timeframe = null;
    }
    onConfirm(out);
  };

  const v = validation || {};

  return (
    <div className="nc-modal-backdrop" onClick={onClose}>
      <div
        className="nc-modal nc-card"
        onClick={(e) => e.stopPropagation()}
        role="dialog" aria-label="Clip settings"
        style={{ width: 520, background: 'var(--nc-panel)', borderRadius: 18, overflow: 'hidden' }}
      >
        {/* header */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 10, padding: '14px 18px', borderBottom: '1px solid var(--nc-border)' }}>
          <Clapperboard size={17} style={{ color: 'var(--nc-accent)' }} />
          <div style={{ flex: 1, fontSize: 15, fontWeight: 700 }}>Clip settings</div>
          <button className="nc-icon-btn" onClick={onClose} aria-label="Close"><X size={17} /></button>
        </div>

        {/* source */}
        <div className="nc-scroll" style={{ overflowY: 'auto', padding: '16px 18px', display: 'flex', flexDirection: 'column', gap: 16 }}>
          <div style={{ display: 'flex', gap: 12, alignItems: 'center' }}>
            {v.thumbnail
              ? <img src={v.thumbnail} alt="" style={{ width: 112, height: 63, objectFit: 'cover', borderRadius: 10, background: '#000' }} draggable={false} />
              : <div className="nc-skeleton" style={{ width: 112, height: 63, borderRadius: 10 }} />}
            <div style={{ minWidth: 0 }}>
              <div style={{ fontSize: 13.5, fontWeight: 600, overflow: 'hidden', textOverflow: 'ellipsis', display: '-webkit-box', WebkitLineClamp: 2, WebkitBoxOrient: 'vertical' }}>
                {v.title || 'Video'}
              </div>
              {dur != null && (
                <div style={{ fontSize: 12, color: 'var(--nc-text-faint)', marginTop: 4, display: 'flex', alignItems: 'center', gap: 4 }}>
                  <Clock size={12} /> {fmtDur(dur)} source
                </div>
              )}
            </div>
          </div>

          <div>
            <span className="nc-label">AI model</span>
            <select className="nc-select" style={{ width: '100%' }} value={s.model} onChange={(e) => set({ model: e.target.value })}>
              {MODEL_OPTIONS.map((o) => <option key={o.id} value={o.id}>{o.label}</option>)}
            </select>
          </div>

          <div>
            <span className="nc-label">Clip genre</span>
            <Segmented options={GENRE_OPTIONS} value={s.genre} onChange={(id) => set({ genre: id })} />
          </div>

          <div>
            <span className="nc-label">Clip style</span>
            <Segmented options={STYLE_OPTIONS} value={s.clipStyle} onChange={(id) => set({ clipStyle: id })} />
          </div>

          <div>
            <span className="nc-label">Aspect ratio</span>
            <Segmented options={ASPECT_OPTIONS} value={s.aspect} onChange={(id) => set({ aspect: id })} />
          </div>

          <div>
            <span className="nc-label">Clip length</span>
            <Segmented options={LENGTH_OPTIONS} value={s.clipLength} onChange={(id) => set({ clipLength: id })} />
            {s.clipLength === 'custom' && (
              <div className="nc-anim-fade-in" style={{ display: 'flex', gap: 8, alignItems: 'center', marginTop: 10 }}>
                <input className="nc-input" type="number" min={5} max={900} value={s.customMin ?? 15}
                  onChange={(e) => set({ customMin: e.target.value })} style={{ width: 90 }} aria-label="Min seconds" />
                <span style={{ color: 'var(--nc-text-faint)' }}>\u2013</span>
                <input className="nc-input" type="number" min={5} max={900} value={s.customMax ?? 60}
                  onChange={(e) => set({ customMax: e.target.value })} style={{ width: 90 }} aria-label="Max seconds" />
                <span style={{ fontSize: 12, color: 'var(--nc-text-faint)' }}>seconds</span>
              </div>
            )}
          </div>

          <div>
            <label style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 13.5, cursor: 'pointer', marginBottom: 8 }}>
              <input type="checkbox" checked={useTimeframe} onChange={(e) => setUseTimeframe(e.target.checked)} />
              Only clip part of the source
            </label>
            {useTimeframe && (
              <div className="nc-anim-fade-in" style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
                <input className="nc-input" placeholder="0:00" value={tfStart} onChange={(e) => setTfStart(e.target.value)} style={{ width: 110 }} aria-label="Timeframe start" />
                <span style={{ color: 'var(--nc-text-faint)' }}>→</span>
                <input className="nc-input" placeholder={dur ? formatTime(dur) : '1:05:34'} value={tfEnd} onChange={(e) => setTfEnd(e.target.value)} style={{ width: 110 }} aria-label="Timeframe end" />
                {dur != null && <span style={{ fontSize: 12, color: 'var(--nc-text-faint)' }}>source is {fmtDur(dur)}</span>}
              </div>
            )}
          </div>
        </div>

        {/* footer */}
        <div style={{ display: 'flex', gap: 10, padding: '14px 18px', borderTop: '1px solid var(--nc-border)' }}>
          <button className="nc-btn-ghost" style={{ flex: 1 }} onClick={onClose}>Cancel</button>
          <button className="nc-btn-primary" style={{ flex: 2 }} onClick={confirm}>
            <Clapperboard size={15} /> Start clipping
          </button>
        </div>
      </div>
    </div>
  );
}
