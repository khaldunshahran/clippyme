import { useEffect, useRef } from 'react';
import { Loader2, TriangleAlert, Info, Link2 } from 'lucide-react';
import ProgressPill from './ProgressPill.jsx';
import ClipRow from './ClipRow.jsx';

function ValidateMsg({ data }) {
  const { state, url, validation, error } = data || {};
  if (state === 'checking') {
    return (
      <div className="nc-card nc-anim-fade-up" style={{ padding: 14, display: 'flex', gap: 10, alignItems: 'center', maxWidth: 480 }}>
        <Loader2 size={16} style={{ animation: 'nc-spin 1.2s linear infinite', color: 'var(--nc-accent)', flexShrink: 0 }} />
        <div style={{ fontSize: 13.5, color: 'var(--nc-text-dim)' }}>
          Checking that link…
          <div style={{ fontSize: 12, color: 'var(--nc-text-faint)', marginTop: 2, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', maxWidth: 380 }}>{url}</div>
        </div>
      </div>
    );
  }
  if (state === 'error') {
    return (
      <div className="nc-card nc-anim-fade-up" style={{ padding: 14, display: 'flex', gap: 10, alignItems: 'center', maxWidth: 480, borderColor: 'rgba(243,18,96,0.4)' }}>
        <TriangleAlert size={16} style={{ color: 'var(--nc-red)', flexShrink: 0 }} />
        <div style={{ fontSize: 13.5 }}>Validation failed: {error}</div>
      </div>
    );
  }
  // done
  const v = validation || {};
  if (v.valid && v.downloadable) {
    return (
      <div className="nc-card nc-anim-fade-up" style={{ padding: 12, display: 'flex', gap: 12, alignItems: 'center', maxWidth: 480, borderColor: 'rgba(52,199,123,0.35)' }}>
        {v.thumbnail
          ? <img src={v.thumbnail} alt="" style={{ width: 96, height: 54, objectFit: 'cover', borderRadius: 8, background: '#000' }} draggable={false} />
          : <div className="nc-skeleton" style={{ width: 96, height: 54, borderRadius: 8 }} />}
        <div style={{ minWidth: 0 }}>
          <div style={{ fontSize: 13, fontWeight: 600, overflow: 'hidden', textOverflow: 'ellipsis', display: '-webkit-box', WebkitLineClamp: 2, WebkitBoxOrient: 'vertical' }}>
            {v.title || 'Video'}
          </div>
          <div style={{ fontSize: 12, color: 'var(--nc-green)', marginTop: 4, display: 'flex', alignItems: 'center', gap: 5 }}>
            <Link2 size={12} /> Link looks good — set up your clips
          </div>
        </div>
      </div>
    );
  }
  return (
    <div className="nc-card nc-anim-fade-up" style={{ padding: 14, maxWidth: 480, borderColor: 'rgba(243,18,96,0.4)' }}>
      <div style={{ display: 'flex', gap: 10, alignItems: 'center' }}>
        <TriangleAlert size={16} style={{ color: 'var(--nc-red)', flexShrink: 0 }} />
        <div style={{ fontSize: 13.5, fontWeight: 600 }}>That link won't work</div>
      </div>
      {v.reason && <div style={{ fontSize: 13, color: 'var(--nc-text-dim)', marginTop: 6 }}>{v.reason}</div>}
    </div>
  );
}

export default function ChatThread({ thread, onOpenClip, onEditClip, onScheduleClip, onViewAll }) {
  const bottomRef = useRef(null);
  const msgs = thread?.messages || [];

  // stick to bottom on new messages (instant — no smooth scroll jank)
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ block: 'end' });
  }, [msgs.length, thread?.phase]);

  if (!thread) return null;

  return (
    <div className="nc-scroll" style={{ flex: 1, overflowY: 'auto', padding: '20px 16px 8px' }}>
      <div style={{ maxWidth: 820, margin: '0 auto', display: 'flex', flexDirection: 'column', gap: 14 }}>
        {msgs.map((m) => {
          if (m.role === 'user') {
            return (
              <div key={m.id} className="nc-anim-fade-up" style={{ display: 'flex', justifyContent: 'flex-end' }}>
                <div style={{
                  background: 'var(--nc-card-hover)', border: '1px solid var(--nc-border)',
                  borderRadius: '18px 18px 6px 18px', padding: '10px 14px',
                  maxWidth: '75%', fontSize: 14.5, lineHeight: 1.5, wordBreak: 'break-word',
                }}>
                  {m.text}
                </div>
              </div>
            );
          }
          // AI messages
          switch (m.kind) {
            case 'validate':
              return <div key={m.id}><ValidateMsg data={m.data} /></div>;
            case 'progress': {
              const st = m.data?.state;
              return (
                <div key={m.id}>
                  <ProgressPill
                    state={st}
                    progress={thread.progress}
                    title={thread.validation?.title}
                    thumbnail={thread.validation?.thumbnail}
                  />
                </div>
              );
            }
            case 'clips':
              return (
                <div key={m.id}>
                  <ClipRow
                    clips={thread.clips}
                    onOpenClip={onOpenClip}
                    onEditClip={onEditClip}
                    onScheduleClip={onScheduleClip}
                    onViewAll={onViewAll}
                  />
                </div>
              );
            case 'notice':
              return (
                <div key={m.id} className="nc-anim-fade-up" style={{ display: 'flex', gap: 8, alignItems: 'flex-start', maxWidth: 640 }}>
                  <Info size={15} style={{ color: 'var(--nc-accent)', flexShrink: 0, marginTop: 2 }} />
                  <div style={{ fontSize: 13.5, color: 'var(--nc-text-dim)', lineHeight: 1.5 }}>{m.text}</div>
                </div>
              );
            case 'error':
              return (
                <div key={m.id} className="nc-anim-fade-up" style={{ display: 'flex', gap: 8, alignItems: 'flex-start', maxWidth: 640 }}>
                  <TriangleAlert size={15} style={{ color: 'var(--nc-red)', flexShrink: 0, marginTop: 2 }} />
                  <div style={{ fontSize: 13.5, color: 'var(--nc-text-dim)', lineHeight: 1.5 }}>{m.text}</div>
                </div>
              );
            default:
              return (
                <div key={m.id} className="nc-anim-fade-up" style={{ maxWidth: 640, fontSize: 14.5, lineHeight: 1.6, color: 'var(--nc-text)' }}>
                  {m.text}
                </div>
              );
          }
        })}
        {/* live progress pill follows the thread while clipping (updates in place) */}
        {thread.phase === 'clipping' && !msgs.some((m) => m.kind === 'progress') && (
          <div><ProgressPill progress={thread.progress} title={thread.validation?.title} thumbnail={thread.validation?.thumbnail} state="clipping" /></div>
        )}
        <div ref={bottomRef} style={{ height: 4 }} />
      </div>
    </div>
  );
}
