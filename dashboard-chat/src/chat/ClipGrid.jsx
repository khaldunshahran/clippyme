import { useEffect } from 'react';
import { X } from 'lucide-react';
import { ClipCard } from './ClipRow.jsx';

/**
 * Task 19 -- optional multi-select mode for bulk editing.
 * selectionMode: when true, each card shows a checkbox; the header offers
 * "All" plus "Bulk edit (N)", which calls onBulkEdit(selectedIndices).
 * Selection state lives in the host (App); the grid is stateless.
 */
export default function ClipGrid({ clips, title, onClose, onOpenClip, onEditClip, onScheduleClip,
  selectionMode = false, selectedIndices = [], onToggleSelectionMode = null,
  onToggleSelect = null, onSelectAll = null, onBulkEdit = null }) {
  const selected = new Set(selectedIndices);
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
          {onToggleSelectionMode && (
            <button
              className="nc-chip"
              onClick={onToggleSelectionMode}
              aria-pressed={selectionMode}
              title="Select multiple clips for bulk editing"
              style={{ marginRight: 8 }}
            >
              {selectionMode ? 'Done selecting' : 'Select'}
            </button>
          )}
          {selectionMode && (
            <>
              <button className="nc-chip" onClick={onSelectAll} style={{ marginRight: 8 }}>
                All
              </button>
              <button
                className="nc-chip"
                onClick={() => onBulkEdit && onBulkEdit([...selected])}
                disabled={selected.size === 0}
                title={selected.size === 0 ? 'Select at least one clip' : `Bulk-edit ${selected.size} clips`}
                style={{ marginRight: 8 }}
              >
                Bulk edit ({selected.size})
              </button>
            </>
          )}
          <button className="nc-icon-btn" onClick={onClose} aria-label="Close"><X size={18} /></button>
        </div>
        <div className="nc-scroll" style={{ overflowY: 'auto', padding: 20 }}>
          <div style={{
            display: 'grid', gap: 18,
            gridTemplateColumns: 'repeat(auto-fill, minmax(160px, 1fr))',
          }}>
            {clips.map((c) => (
              <div key={c.index} className="nc-anim-fade-up" style={{ position: 'relative', animationDelay: `${Math.min(c.index, 8) * 0.03}s` }}>
                {selectionMode && (
                  <label
                    style={{
                      position: 'absolute', top: 8, left: 8, zIndex: 2,
                      width: 26, height: 26, borderRadius: 8,
                      background: selected.has(c.index) ? 'var(--nc-accent)' : 'rgba(0,0,0,0.6)',
                      border: '1px solid var(--nc-border-strong)',
                      display: 'flex', alignItems: 'center', justifyContent: 'center',
                      cursor: 'pointer',
                    }}
                    title={selected.has(c.index) ? 'Deselect clip' : 'Select clip for bulk editing'}
                  >
                    <input
                      type="checkbox"
                      checked={selected.has(c.index)}
                      onChange={() => onToggleSelect && onToggleSelect(c.index)}
                      aria-label={`Select clip ${c.index + 1} for bulk editing`}
                      style={{ width: 15, height: 15, cursor: 'pointer', accentColor: 'var(--nc-accent)' }}
                    />
                  </label>
                )}
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
