import { useEffect } from 'react';
import { X } from 'lucide-react';
import { ClipCard } from './ClipRow.jsx';

export default function ClipGrid({ clips, title, onClose, onOpenClip, onEditClip, onScheduleClip }) {
  useEffect(() => {
    const h = (e) => { if (e.key === 'Escape') onClose(); };
    window.addEventListener('keydown', h);
    return () => window.removeEventListener('keydown', h);
  }, [onClose]);

  return (
    <div className="nc-modal-backdrop nc-anim-fade-in" onClick={onClose} style={{ alignItems: 'stretch', justifyContent: 'stretch' }}>
      <div
        className="nc-modal"
        onClick={(e) => e.stopPropagation()}
        role="dialog" aria-label="All clips"
        style={{
          background: 'var(--nc-bg)', border: '1px solid var(--nc-border)',
          borderRadius: 18, margin: 'auto', width: 'min(1060px, calc(100vw - 40px))',
          maxHeight: 'calc(100vh - 60px)',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', padding: '16px 20px', borderBottom: '1px solid var(--nc-border)' }}>
          <div style={{ flex: 1 }}>
            <div style={{ fontSize: 16, fontWeight: 700 }}>All clips</div>
            <div style={{ fontSize: 12.5, color: 'var(--nc-text-faint)', marginTop: 2 }}>
              {title} · {clips.length} clip{clips.length === 1 ? '' : 's'}
            </div>
          </div>
          <button className="nc-icon-btn" onClick={onClose} aria-label="Close"><X size={18} /></button>
        </div>
        <div className="nc-scroll" style={{ overflowY: 'auto', padding: 20 }}>
          <div style={{
            display: 'grid', gap: 18,
            gridTemplateColumns: 'repeat(auto-fill, minmax(160px, 1fr))',
          }}>
            {clips.map((c) => (
              <div key={c.index} className="nc-anim-fade-up" style={{ animationDelay: `${Math.min(c.index, 8) * 0.03}s` }}>
                <ClipCard
                  clip={c}
                  onOpen={() => onOpenClip(c.index)}
                  onEdit={() => onEditClip(c.index)}
                  onSchedule={() => onScheduleClip(c.index)}
                />
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
