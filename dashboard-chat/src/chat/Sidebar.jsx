import { Plus, MessageSquare, Trash2, Loader2 } from 'lucide-react';

export default function Sidebar({ threads, activeId, onSelect, onNew, onDelete, collapsed }) {
  return (
    <aside
      className={`nc-sidebar ${collapsed ? 'nc-sidebar-collapsed' : ''}`}
      style={{
        width: collapsed ? 0 : 264, minWidth: collapsed ? 0 : 264,
        background: 'var(--nc-bg-soft)',
        borderRight: '1px solid var(--nc-border)',
        display: 'flex', flexDirection: 'column',
        overflow: 'hidden',
        transition: 'width 0.25s cubic-bezier(0.22,1,0.36,1), min-width 0.25s cubic-bezier(0.22,1,0.36,1)',
      }}
    >
      <div style={{ padding: '14px 12px 8px' }}>
        <button className="nc-btn-ghost" style={{ width: '100%', justifyContent: 'flex-start' }} onClick={onNew}>
          <Plus size={16} /> New chat
        </button>
      </div>
      <div className="nc-scroll" style={{ flex: 1, overflowY: 'auto', padding: '4px 8px 12px' }}>
        {threads.length === 0 && (
          <div style={{ padding: '12px', color: 'var(--nc-text-faint)', fontSize: 13 }}>
            No chats yet. Paste a video link to start clipping.
          </div>
        )}
        {threads.map((t) => {
          const isActive = t.id === activeId;
          return (
            <div
              key={t.id}
              onClick={() => onSelect(t.id)}
              className="nc-thread-row"
              data-active={isActive}
              style={{
                display: 'flex', alignItems: 'center', gap: 8,
                padding: '9px 10px', borderRadius: 10, cursor: 'pointer',
                background: isActive ? 'rgba(255,255,255,0.07)' : 'transparent',
                marginBottom: 2,
              }}
              onMouseEnter={(e) => { if (!isActive) e.currentTarget.style.background = 'rgba(255,255,255,0.04)'; }}
              onMouseLeave={(e) => { if (!isActive) e.currentTarget.style.background = 'transparent'; }}
            >
              {t.phase === 'clipping'
                ? <Loader2 size={15} style={{ animation: 'nc-spin 1.2s linear infinite', color: 'var(--nc-accent)', flexShrink: 0 }} />
                : <MessageSquare size={15} style={{ color: 'var(--nc-text-faint)', flexShrink: 0 }} />}
              <span style={{
                flex: 1, fontSize: 13.5, color: isActive ? 'var(--nc-text)' : 'var(--nc-text-dim)',
                overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
              }}>
                {t.title || 'New chat'}
              </span>
              <button
                className="nc-icon-btn nc-thread-del"
                style={{ width: 26, height: 26, opacity: 0 }}
                onClick={(e) => { e.stopPropagation(); onDelete(t.id); }}
                aria-label="Delete chat"
              >
                <Trash2 size={13} />
              </button>
            </div>
          );
        })}
      </div>
      <style>{`.nc-thread-row:hover .nc-thread-del{opacity:0.7 !important}`}</style>
      <div style={{ padding: 12, borderTop: '1px solid var(--nc-border)', fontSize: 11.5, color: 'var(--nc-text-faint)' }}>
        Nugget Chat · clips from any link
      </div>
    </aside>
  );
}
