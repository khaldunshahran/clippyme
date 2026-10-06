import { useState } from 'react';
import { CalendarClock, Download, Scissors, Play, LayoutGrid } from 'lucide-react';
import { fmtDur } from '../api/chatApi.js';
import { safeResolveUrl, downloadClip } from '../api/realApi.js';

/**
 * Portrait clip card: thumbnail (aspect box reserved — no layout shift),
 * hook/caption text, duration badge, schedule/download/edit icons, title below.
 */
export function ClipCard({ clip, onOpen, onEdit, onSchedule }) {
  const [imgOk, setImgOk] = useState(true);
  const showImg = imgOk && clip.thumbUrl;

  const doDownload = (e) => {
    e.stopPropagation();
    try { downloadClip(clip.raw || clip, clip.index); }
    catch {
      const a = document.createElement('a');
      a.href = safeResolveUrl(clip.videoUrl);
      a.download = '';
      document.body.appendChild(a); a.click(); a.remove();
    }
  };

  return (
    <div className="nc-anim-fade-up" style={{ width: 168 }}>
      <div
        onClick={onOpen}
        style={{
          position: 'relative', width: 168, aspectRatio: '9/16',
          borderRadius: 14, overflow: 'hidden', cursor: 'pointer',
          background: '#101014', border: '1px solid var(--nc-border)',
          transition: 'transform 0.18s cubic-bezier(0.22,1,0.36,1), border-color 0.18s ease',
        }}
        onMouseEnter={(e) => { e.currentTarget.style.transform = 'translateY(-3px)'; e.currentTarget.style.borderColor = 'var(--nc-border-strong)'; }}
        onMouseLeave={(e) => { e.currentTarget.style.transform = 'none'; e.currentTarget.style.borderColor = 'var(--nc-border)'; }}
      >
        {showImg ? (
          <img
            src={clip.thumbUrl} alt="" draggable={false}
            onError={() => setImgOk(false)}
            style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', objectFit: 'cover' }}
          />
        ) : (
          <div style={{ position: 'absolute', inset: 0, display: 'flex', alignItems: 'center', justifyContent: 'center', background: 'linear-gradient(160deg,#1b1b20,#0e0e11)' }}>
            <Play size={30} style={{ color: 'rgba(255,255,255,0.25)' }} />
          </div>
        )}
        {/* gradient + hook text */}
        <div style={{ position: 'absolute', inset: 0, background: 'linear-gradient(to top, rgba(0,0,0,0.82) 0%, transparent 45%, rgba(0,0,0,0.25) 100%)', pointerEvents: 'none' }} />
        {clip.duration != null && (
          <span style={{
            position: 'absolute', top: 8, right: 8,
            background: 'rgba(0,0,0,0.65)', backdropFilter: 'blur(4px)',
            fontSize: 11, fontWeight: 600, padding: '3px 8px', borderRadius: 99,
            fontVariantNumeric: 'tabular-nums',
          }}>
            {fmtDur(clip.duration)}
          </span>
        )}
        {clip.hook && (
          <div style={{
            position: 'absolute', left: 10, right: 10, bottom: 44,
            fontSize: 12.5, fontWeight: 700, lineHeight: 1.35,
            display: '-webkit-box', WebkitLineClamp: 3, WebkitBoxOrient: 'vertical', overflow: 'hidden',
            textShadow: '0 1px 8px rgba(0,0,0,0.8)',
          }}>
            {clip.hook}
          </div>
        )}
        {/* action row */}
        <div style={{ position: 'absolute', left: 6, right: 6, bottom: 6, display: 'flex', justifyContent: 'space-around' }}>
          {[
            { icon: CalendarClock, label: 'Schedule', fn: onSchedule },
            { icon: Download, label: 'Download', fn: doDownload },
            { icon: Scissors, label: 'Edit', fn: onEdit },
          ].map(({ icon: Icon, label, fn }) => (
            <button
              key={label}
              className="nc-icon-btn"
              title={label}
              aria-label={label}
              onClick={(e) => { e.stopPropagation(); fn && fn(e); }}
              style={{ background: 'rgba(0,0,0,0.55)', backdropFilter: 'blur(4px)', borderRadius: 9, width: 32, height: 32, color: '#fff' }}
            >
              <Icon size={15} />
            </button>
          ))}
        </div>
      </div>
      <div style={{ marginTop: 8, fontSize: 12.5, fontWeight: 500, color: 'var(--nc-text-dim)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
        {clip.title}
      </div>
    </div>
  );
}

/** Horizontally scrollable clip row + View all. */
export default function ClipRow({ clips, onOpenClip, onEditClip, onScheduleClip, onViewAll }) {
  if (!clips || clips.length === 0) return null;
  return (
    <div className="nc-anim-fade-up" style={{ maxWidth: '100%' }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 10 }}>
        <div style={{ fontSize: 14, fontWeight: 700 }}>
          Your clips <span style={{ color: 'var(--nc-text-faint)', fontWeight: 500 }}>· {clips.length}</span>
        </div>
        <button className="nc-chip" onClick={onViewAll} style={{ fontSize: 12.5 }}>
          <LayoutGrid size={13} /> View all
        </button>
      </div>
      <div className="nc-clip-row nc-scroll">
        {clips.map((c) => (
          <ClipCard
            key={c.index}
            clip={c}
            onOpen={() => onOpenClip(c.index)}
            onEdit={(e) => onEditClip(c.index, e)}
            onSchedule={(e) => onScheduleClip(c.index, e)}
          />
        ))}
      </div>
    </div>
  );
}
