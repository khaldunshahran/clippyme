import { useState, useCallback } from 'react';
import { Icon, Badge } from './primitives';

export function getClipPlatformCopy(clip, platform = 'tiktok') {
  if (!clip) return { caption: '', hashtags: [], title: '' };
  const pKey = platform === 'ig' ? 'instagram' : (platform === 'yt' ? 'youtube' : platform);
  const platData = clip.platforms?.[pKey];

  if (pKey === 'tiktok') {
    const caption = platData?.caption || clip.video_description_for_tiktok || clip.caption || clip.title || '';
    const hashtags = platData?.hashtags || clip.hashtags || ['#fyp', '#trending', '#viral'];
    return {
      caption,
      hashtags,
      title: clip.title || clip.video_title_for_youtube_short || '',
    };
  }
  if (pKey === 'instagram') {
    const caption = platData?.caption || clip.video_description_for_instagram || clip.caption || clip.title || '';
    const hashtags = platData?.hashtags || clip.hashtags || ['#reels', '#viral', '#trending'];
    return {
      caption,
      hashtags,
      title: clip.title || clip.video_title_for_youtube_short || '',
    };
  }
  if (pKey === 'youtube') {
    const title = platData?.title || clip.video_title_for_youtube_short || clip.title || '';
    const caption = platData?.description || clip.video_description || clip.caption || title;
    const hashtags = platData?.hashtags || clip.hashtags || ['#shorts', '#trending'];
    return {
      caption,
      hashtags,
      title,
    };
  }
  return {
    caption: clip.caption || '',
    hashtags: clip.hashtags || [],
    title: clip.title || '',
  };
}

export function PlatformMockupOverlay({
  platform = 'tiktok',
  clip = {},
  speakerName = '',
  channel = '',
}) {
  const normPlat = platform === 'ig' ? 'instagram' : (platform === 'yt' ? 'youtube' : platform);
  const copy = getClipPlatformCopy(clip, normPlat);
  const speaker = speakerName || clip.speaker_name || '';
  const authorHandle = channel || clip.channel || 'nugget';
  const score = Math.round(clip.viral_score || 85);

  return (
    <div className={`feed-mockup platform-${normPlat}`} style={{ pointerEvents: 'none' }}>
      <div className="mockup-header">
        {normPlat === 'tiktok' && (
          <>
            <span style={{ opacity: 0.6 }}>Following</span>
            <span className="active" style={{ borderBottom: '2px solid #fff', paddingBottom: 2 }}>For You</span>
          </>
        )}
        {normPlat === 'instagram' && (
          <span className="active" style={{ fontWeight: 800, letterSpacing: '0.05em' }}>Reels</span>
        )}
        {normPlat === 'youtube' && (
          <span className="active" style={{ fontWeight: 800, letterSpacing: '0.05em' }}>Shorts</span>
        )}
      </div>

      <div className="mockup-body">
        <div className="mockup-meta">
          <div className="mockup-author" style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <span>@{authorHandle}</span>
            {speaker && (
              <span
                style={{
                  fontSize: '9px',
                  background: 'rgba(2,197,191,0.25)',
                  color: '#fff',
                  padding: '1px 5px',
                  borderRadius: 8,
                  fontWeight: 600,
                }}
              >
                {speaker}
              </span>
            )}
          </div>
          <div
            className="mockup-desc"
            title={copy.caption}
            style={{
              fontSize: '11px',
              lineHeight: 1.35,
              display: '-webkit-box',
              WebkitLineClamp: 2,
              WebkitBoxOrient: 'vertical',
              overflow: 'hidden',
              textShadow: '0 1px 3px rgba(0,0,0,0.85)',
            }}
          >
            {normPlat === 'youtube' ? `${copy.title} — ${copy.caption}` : copy.caption}
          </div>
          {copy.hashtags?.length > 0 && (
            <div className="mockup-tags" style={{ fontSize: '10px', color: '#55d6ff', fontWeight: 600 }}>
              {copy.hashtags.slice(0, 4).join(' ')}
            </div>
          )}
          <div className="mockup-music" style={{ fontSize: '10px', opacity: 0.85, display: 'flex', alignItems: 'center', gap: 4 }}>
            <span>♫</span>
            <span>{normPlat === 'youtube' ? 'Original Audio' : `Original sound · ${speaker || authorHandle}`}</span>
          </div>
        </div>

        <div className="mockup-sidebar" style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 10 }}>
          <div className="mockup-avatar" style={{ width: 30, height: 30, fontSize: 12 }}>⚡</div>
          <div className="mockup-btn" style={{ fontSize: '9px' }}>
            <span style={{ fontSize: 15 }}>{normPlat === 'youtube' ? '👍' : '❤️'}</span>
            <span>{score}k</span>
          </div>
          <div className="mockup-btn" style={{ fontSize: '9px' }}>
            <span style={{ fontSize: 15 }}>💬</span>
            <span>842</span>
          </div>
          <div className="mockup-btn" style={{ fontSize: '9px' }}>
            <span style={{ fontSize: 15 }}>{normPlat === 'youtube' ? '👎' : '🔖'}</span>
            <span>{normPlat === 'youtube' ? 'Dislike' : 'Save'}</span>
          </div>
          <div className="mockup-btn" style={{ fontSize: '9px' }}>
            <span style={{ fontSize: 15 }}>↗️</span>
            <span>Share</span>
          </div>
          <div className="mockup-disc" style={{ width: 24, height: 24, fontSize: 10 }}>🎵</div>
        </div>
      </div>
    </div>
  );
}

export function PlatformCaptionSection({
  clip,
  activePlatform,
  onChangePlatform,
  showMockup,
  onToggleMockup,
  pushToast,
}) {
  const [copied, setCopied] = useState(false);
  const copy = getClipPlatformCopy(clip, activePlatform);
  const speaker = clip.speaker_name || '';

  const doCopy = useCallback((e) => {
    e?.stopPropagation?.();
    const fullText = activePlatform === 'youtube'
      ? `${copy.title}\n\n${copy.caption}`
      : copy.caption;
    if (navigator.clipboard?.writeText) {
      navigator.clipboard.writeText(fullText);
      setCopied(true);
      pushToast?.('success', `Copied ${activePlatform.toUpperCase()} caption!`);
      setTimeout(() => setCopied(false), 2000);
    }
  }, [copy, activePlatform, pushToast]);

  return (
    <div
      className="platform-caption-card"
      onClick={(e) => e.stopPropagation()}
      style={{
        marginTop: 10,
        padding: '10px 12px',
        borderRadius: 8,
        background: 'var(--surface-deep, rgba(0,0,0,0.03))',
        border: '1px solid var(--border, rgba(51,46,38,0.08))',
        display: 'flex',
        flexDirection: 'column',
        gap: 8,
      }}
    >
      {/* Top Header: Platform Selector Tabs & Quick Buttons */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 6 }}>
        <div style={{ display: 'flex', gap: 4 }}>
          {[
            { id: 'tiktok', label: 'TikTok 🎵' },
            { id: 'instagram', label: 'Reels 📷' },
            { id: 'youtube', label: 'Shorts ▶️' },
          ].map((p) => {
            const active = activePlatform === p.id;
            return (
              <button
                key={p.id}
                type="button"
                onClick={() => onChangePlatform(p.id)}
                style={{
                  border: '1px solid ' + (active ? 'var(--brand-teal, #02c5bf)' : 'transparent'),
                  background: active ? 'rgba(2,197,191,0.12)' : 'transparent',
                  color: active ? 'var(--brand-teal, #02c5bf)' : 'var(--fg-3, #777)',
                  fontSize: '11px',
                  fontWeight: active ? 700 : 500,
                  padding: '2px 7px',
                  borderRadius: 6,
                  cursor: 'pointer',
                  transition: 'all 0.15s ease',
                }}
              >
                {p.label}
              </button>
            );
          })}
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
          <button
            type="button"
            title="Toggle platform feed preview overlay"
            onClick={onToggleMockup}
            style={{
              display: 'inline-flex',
              alignItems: 'center',
              gap: 4,
              fontSize: '10.5px',
              fontWeight: 600,
              padding: '2px 7px',
              borderRadius: 6,
              border: '1px solid ' + (showMockup ? 'var(--brand-teal, #02c5bf)' : 'var(--border, rgba(51,46,38,0.1))'),
              background: showMockup ? 'var(--brand-teal, #02c5bf)' : 'transparent',
              color: showMockup ? '#fff' : 'var(--fg-2, #555)',
              cursor: 'pointer',
            }}
          >
            <Icon n="smartphone" style={{ width: 12, height: 12 }} />
            {showMockup ? 'Exit Mockup' : 'Mockup'}
          </button>

          <button
            type="button"
            title="Copy platform caption & tags"
            onClick={doCopy}
            style={{
              display: 'inline-flex',
              alignItems: 'center',
              gap: 4,
              fontSize: '10.5px',
              fontWeight: 600,
              padding: '2px 7px',
              borderRadius: 6,
              border: '1px solid var(--border, rgba(51,46,38,0.1))',
              background: copied ? 'rgba(46,160,67,0.15)' : 'transparent',
              color: copied ? 'var(--success, #2ea043)' : 'var(--fg-2, #555)',
              cursor: 'pointer',
            }}
          >
            <Icon n={copied ? 'check' : 'copy'} style={{ width: 12, height: 12 }} />
            {copied ? 'Copied' : 'Copy'}
          </button>
        </div>
      </div>

      {/* Speaker Badge if available */}
      {speaker && (
        <div style={{ display: 'flex', alignItems: 'center', gap: 5, fontSize: '11px', color: 'var(--brand-teal, #02c5bf)', fontWeight: 600 }}>
          <span>🗣️</span>
          <span>{speaker}</span>
        </div>
      )}

      {/* Caption Content Box */}
      <div
        style={{
          fontSize: '11.5px',
          lineHeight: '1.4',
          color: 'var(--fg-1, #222)',
          background: 'var(--surface-card, #fff)',
          padding: '8px 10px',
          borderRadius: 6,
          border: '1px solid var(--border-subtle, rgba(0,0,0,0.06))',
          maxHeight: 90,
          overflowY: 'auto',
          whiteSpace: 'pre-wrap',
          wordBreak: 'break-word',
        }}
      >
        {activePlatform === 'youtube' && copy.title && (
          <div style={{ fontWeight: 700, marginBottom: 4, color: 'var(--fg-0, #111)' }}>
            {copy.title}
          </div>
        )}
        <div>{copy.caption || 'No caption generated yet.'}</div>
      </div>
    </div>
  );
}
