import { useEffect, useState } from 'react';
import { Captions, Check, Star, ArrowRight, Sparkles } from 'lucide-react';
import { CAPTION_OPTIONS, CAPTION_DEFAULT_LS_KEY, captionLabel } from './constants.js';
import { SUBTITLE_PRESETS } from '../editor/data.js';

const presetVisual = (id) => SUBTITLE_PRESETS.find((p) => p.id === id) || null;

function PreviewThumb({ presetId, selected }) {
  const pv = presetId ? presetVisual(presetId) : null;
  const style = pv?.style || {};
  const hi = pv?.hi || '#fff';
  return (
    <div
      style={{
        position: 'relative', width: '100%', aspectRatio: '16 / 10', borderRadius: 10,
        background: 'linear-gradient(135deg, #14161d 0%, #1d2029 60%, #23262f 100%)',
        border: `1px solid ${selected ? 'var(--nc-accent)' : 'var(--nc-border)'}`,
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        overflow: 'hidden', transition: 'border-color .18s ease, transform .18s ease',
      }}
    >
      {presetId ? (
        <div style={{ padding: '0 10px', textAlign: 'center', lineHeight: 1.35, fontSize: 13, ...style }}>
          this is <span style={{ color: hi }}>how</span> captions look
        </div>
      ) : (
        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 6, color: 'var(--nc-text-faint)' }}>
          <Sparkles size={20} style={{ color: 'var(--nc-accent)' }} />
          <div style={{ fontSize: 11.5, padding: '0 12px', textAlign: 'center', lineHeight: 1.4 }}>System picks<br />the best fit</div>
        </div>
      )}
      {selected && (
        <div style={{
          position: 'absolute', top: 6, right: 6, width: 20, height: 20, borderRadius: '50%',
          background: 'var(--nc-accent)', display: 'flex', alignItems: 'center', justifyContent: 'center',
        }}>
          <Check size={13} style={{ color: '#fff' }} strokeWidth={3} />
        </div>
      )}
    </div>
  );
}

export default function CaptionCarousel({ validation, selected, locked, onSelect, onContinue }) {
  const [isDefault, setIsDefault] = useState(false);

  // Apply the stored default once, when nothing is chosen yet.
  useEffect(() => {
    try {
      const d = localStorage.getItem(CAPTION_DEFAULT_LS_KEY) || '';
      setIsDefault(!!d && d === (selected || ''));
      if (!selected && d) onSelect(d);
    } catch {}
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const v = validation || {};
  const selId = selected || '';
  const defId = (() => { try { return localStorage.getItem(CAPTION_DEFAULT_LS_KEY) || ''; } catch { return ''; } })();

  const saveDefault = () => {
    try { localStorage.setItem(CAPTION_DEFAULT_LS_KEY, selId); } catch {}
    setIsDefault(true);
  };

  if (locked) {
    return (
      <div className="nc-card nc-anim-fade-up" style={{ padding: '10px 14px', maxWidth: 480, display: 'flex', alignItems: 'center', gap: 8 }}>
        <Captions size={15} style={{ color: 'var(--nc-accent)', flexShrink: 0 }} />
        <div style={{ fontSize: 13, color: 'var(--nc-text-dim)' }}>
          Captions: <b style={{ color: 'var(--nc-text)' }}>{captionLabel(selId)}</b>
          {selId === '' && <span style={{ color: 'var(--nc-text-faint)' }}> (auto)</span>}
        </div>
      </div>
    );
  }

  const cards = [{ id: '', label: 'Auto' }, ...CAPTION_OPTIONS.filter((o) => o.id !== '')];

  return (
    <div className="nc-card nc-anim-fade-up" style={{ padding: 16, maxWidth: 640, borderColor: 'rgba(88,101,242,0.35)' }}>
      <div style={{ display: 'flex', gap: 12, alignItems: 'center', marginBottom: 4 }}>
        {v.thumbnail
          ? <img src={v.thumbnail} alt="" style={{ width: 84, height: 47, objectFit: 'cover', borderRadius: 8, background: '#000' }} draggable={false} />
          : <div className="nc-skeleton" style={{ width: 84, height: 47, borderRadius: 8 }} />}
        <div style={{ minWidth: 0, flex: 1 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 7 }}>
            <Captions size={15} style={{ color: 'var(--nc-accent)', flexShrink: 0 }} />
            <div style={{ fontSize: 14, fontWeight: 700 }}>Pick a caption style</div>
          </div>
          <div style={{ fontSize: 12.5, color: 'var(--nc-text-faint)', marginTop: 3 }}>
            Skip and the system chooses for you
            {defId ? <> &nbsp;·&nbsp; your default is <b style={{ color: 'var(--nc-text-dim)' }}>{captionLabel(defId)}</b></> : null}
          </div>
        </div>
      </div>

      <div
        className="nc-scroll"
        style={{
          display: 'flex', gap: 10, overflowX: 'auto', padding: '10px 2px 12px',
          scrollSnapType: 'x mandatory', scrollBehavior: 'smooth',
        }}
      >
        {cards.map((c) => {
          const active = selId === c.id;
          return (
            <button
              key={c.id || 'auto'}
              onClick={() => onSelect(c.id)}
              style={{
                flex: '0 0 148px', scrollSnapAlign: 'start', background: 'transparent',
                border: 'none', padding: 0, cursor: 'pointer', textAlign: 'left',
              }}
              aria-pressed={active}
            >
              <div style={{ transform: active ? 'translateY(-2px)' : 'none', transition: 'transform .18s ease' }}>
                <PreviewThumb presetId={c.id} selected={active} />
              </div>
              <div style={{
                marginTop: 7, fontSize: 12.5, fontWeight: active ? 700 : 500,
                color: active ? 'var(--nc-text)' : 'var(--nc-text-dim)', textAlign: 'center',
              }}>
                {c.id === '' ? 'Auto' : (c.label || captionLabel(c.id))}
              </div>
            </button>
          );
        })}
      </div>

      <div style={{ display: 'flex', gap: 10, alignItems: 'center', marginTop: 2 }}>
        <button
          className="nc-chip"
          onClick={saveDefault}
          disabled={isDefault}
          title="Remember this style for future jobs"
          style={{ fontSize: 12.5, padding: '8px 12px', opacity: isDefault ? 0.55 : 1 }}
        >
          <Star size={13} style={{ marginRight: 5, verticalAlign: -2 }} />
          {isDefault ? 'Default saved' : 'Set as default'}
        </button>
        <div style={{ flex: 1 }} />
        <button className="nc-btn-primary" onClick={onContinue} style={{ padding: '9px 20px', fontSize: 13.5 }}>
          Continue <ArrowRight size={15} style={{ marginLeft: 4, verticalAlign: -2 }} />
        </button>
      </div>
    </div>
  );
}
