// ClippyMe redesign — PublishModal: real concurrent publish to Zernio.
// Features in-modal video player, interactive social feed mockups (TikTok/Reels/Shorts),
// dynamic platform character limits, AI metadata generation, 3-mode scheduling,
// TikTok advanced settings, and batch multi-clip inspection.
import { useState, useEffect, useRef } from 'react';
import { createPortal } from 'react-dom';
import { Icon, Social, Btn, Switch, PlatPill, PLATFORMS, Badge, Segmented } from './primitives';
import { LazyVideo } from './LazyVideo';
import { clipVideoSrc, publishClip, getZernio, generateClipMetadata } from './realApi';
import { seedToggles, seedHookParams, seedSubtitleParams, seedLogoParams, seedBannerParams } from '../lib/seedClipParams';
import { localDatePlus } from '../lib/scheduleDates';
import { useModalA11y } from './useModalA11y';
import { getClipPlatformCopy } from './PlatformMockupOverlay';

// redesign plat id → backend platform + account key.
export const PLAT = {
  tiktok: { platform: 'tiktok', acct: 'tiktok', icon: 'tiktok', label: 'TikTok', maxCaption: 2200 },
  ig: { platform: 'instagram', acct: 'instagram', icon: 'instagram', label: 'Reels', maxCaption: 2200 },
  yt: { platform: 'youtube', acct: 'youtube', icon: 'youtube', label: 'Shorts', maxCaption: 5000, maxTitle: 100 },
  fb: { platform: 'facebook', acct: 'facebook', icon: 'facebook', label: 'Facebook', maxCaption: 63206 },
};

function PubRow({ clip, idx, st, plats }) {
  const status = typeof st === 'object' && st ? st.state : st;
  const errMsg = typeof st === 'object' && st ? st.error : null;
  const tasks = Object.keys(plats).filter((k) => plats[k]);
  const done = status === 'done';
  const error = status === 'error';
  return (
    <div className={'pubrow' + (done ? ' done' : '')}>
      <div className="pthumb" style={{ background: '#000', overflow: 'hidden' }}>
        <LazyVideo src={clipVideoSrc(clip)} muted playsInline rootMargin="120px"
          style={{ width: '100%', height: '100%', objectFit: 'cover' }} />
      </div>
      <div className="pinfo">
        <div className="pttl">{clip.video_title_for_youtube_short || clip.title || `Clip ${idx + 1}`}</div>
        <div className="pplats">
          {tasks.map((p) => (
            <div className="pp" key={p}>
              <Social n={PLAT[p].icon} color={done ? '02C5BF' : '7E7E8F'} size={13} />
              <div className="ptrack"><i className={p} style={{ width: done ? '100%' : status === 'uploading' ? '70%' : '0%', transition: 'width .4s' }}></i></div>
            </div>
          ))}
          <span className={'pstat' + (done ? ' done' : status === 'uploading' ? '' : ' wait')}
            style={error ? { color: 'var(--danger)' } : undefined}
            title={error && errMsg ? errMsg : undefined}>
            {error ? (errMsg ? `failed: ${errMsg.slice(0, 60)}` : 'failed') : done ? 'live' : status === 'uploading' ? 'uploading' : 'queued'}
          </span>
        </div>
      </div>
      <div className="pcheck"><Icon n={done ? 'check' : error ? 'x' : 'loader'} /></div>
    </div>
  );
}

export function PublishModal({
  clips = [],
  jobId,
  clipStates = {},
  preselections,
  onClose,
  onPublished,
  pushToast,
  onCustomPublish,
}) {
  const all = clips.length > 1;
  const [activeClipIdx, setActiveClipIdx] = useState(0);
  const activeClip = clips[activeClipIdx] || clips[0] || {};

  const [previewMode, setPreviewMode] = useState('player'); // 'player' | 'mockup'
  const [mockupPlat, setMockupPlat] = useState('tiktok'); // 'tiktok' | 'ig' | 'yt'

  const [zernio, setZernio] = useState(null);
  const [plats, setPlats] = useState({ tiktok: false, ig: false, yt: false, fb: false });

  // 3-mode scheduling: 'smart' (auto prime-time), 'now' (immediate), 'custom' (datetime)
  const [scheduleMode, setScheduleMode] = useState('smart');
  const [customDateTime, setCustomDateTime] = useState(() => {
    const d = new Date(Date.now() + 3600000 * 3);
    return d.toISOString().slice(0, 16);
  });

  const [titleText, setTitleText] = useState(activeClip.video_title_for_youtube_short || activeClip.title || '');
  const [captionPlat, setCaptionPlat] = useState('tiktok');
  const [platCaptions, setPlatCaptions] = useState({
    tiktok: getClipPlatformCopy(activeClip, 'tiktok').caption,
    ig: getClipPlatformCopy(activeClip, 'instagram').caption,
    yt: getClipPlatformCopy(activeClip, 'youtube').caption,
  });
  const [caption, setCaption] = useState(
    getClipPlatformCopy(activeClip, 'tiktok').caption || activeClip.video_description || activeClip.caption || ''
  );
  const [speakerName, setSpeakerName] = useState(activeClip.speaker_name || '');
  const [hashtags, setHashtags] = useState(
    activeClip.hashtags?.length ? activeClip.hashtags : ['#shorts', '#trending', '#viral']
  );

  const [aiThumbnail, setAiThumbnail] = useState(false);
  const [aspectRatio, setAspectRatio] = useState('9:16');

  // TikTok Advanced Controls
  const [tiktokPrivacy, setTiktokPrivacy] = useState('PUBLIC_TO_EVERYONE');
  const [tiktokComments, setTiktokComments] = useState(true);
  const [tiktokDuet, setTiktokDuet] = useState(true);
  const [tiktokStitch, setTiktokStitch] = useState(true);
  const [tiktokAiDisclosure, setTiktokAiDisclosure] = useState(true);
  const [tiktokBrandPartnership] = useState(false);
  const [tiktokExpanded, setTiktokExpanded] = useState(false);

  const [generatingAi, setGeneratingAi] = useState(false);
  const [stage, setStage] = useState('setup'); // setup | uploading | done
  const [progress, setProgress] = useState({});

  useEffect(() => {
    getZernio()
      .then((z) => {
        setZernio(z);
        if (z?.accounts) {
          setPlats({
            tiktok: !!z.accounts.tiktok,
            ig: !!z.accounts.instagram,
            yt: !!z.accounts.youtube,
            fb: !!z.accounts.facebook,
          });
        }
      })
      .catch(() => setZernio({ configured: false }));
  }, []);

  // Update active clip text when switching in batch preview
  useEffect(() => {
    const c = clips[activeClipIdx] || clips[0] || {};
    const tkCopy = getClipPlatformCopy(c, 'tiktok');
    const igCopy = getClipPlatformCopy(c, 'instagram');
    const ytCopy = getClipPlatformCopy(c, 'youtube');

    const nextPlatCaps = {
      tiktok: tkCopy.caption,
      ig: igCopy.caption,
      yt: ytCopy.caption,
    };
    setPlatCaptions(nextPlatCaps);
    setTitleText(ytCopy.title || c.video_title_for_youtube_short || c.title || '');
    const activeText = captionPlat === 'tiktok' ? tkCopy.caption : (captionPlat === 'ig' ? igCopy.caption : ytCopy.caption);
    setCaption(activeText || c.video_description || c.caption || '');
    setSpeakerName(c.speaker_name || '');
    if (c.hashtags?.length) {
      setHashtags(c.hashtags);
    }
  }, [activeClipIdx, clips, captionPlat]);

  const panelRef = useModalA11y(onClose);
  const mountedRef = useRef(true);
  useEffect(() => () => { mountedRef.current = false; }, []);

  const accounts = zernio?.accounts || {};
  const toggle = (k) => setPlats((p) => ({ ...p, [k]: !p[k] }));
  const platTargets = () => Object.keys(plats)
    .filter((k) => plats[k] && accounts[PLAT[k].acct])
    .map((k) => ({ platform: PLAT[k].platform, accountId: accounts[PLAT[k].acct] }));
  const targets = platTargets();
  const ready = zernio?.configured && targets.length > 0;

  const handleGenerateAi = async () => {
    if (generatingAi) return;
    const currentClip = clips[activeClipIdx] || clips[0] || {};
    const effectiveJobId = jobId || currentClip.job_id;
    const apiIdx = Number.isInteger(currentClip.original_index)
      ? currentClip.original_index
      : (currentClip._apiIdx ?? currentClip._idx ?? 0);

    setGeneratingAi(true);
    try {
      if (effectiveJobId) {
        const res = await generateClipMetadata(effectiveJobId, apiIdx);
        if (res.title) setTitleText(res.title);
        if (res.speaker_name) setSpeakerName(res.speaker_name);
        if (res.hashtags?.length) setHashtags(res.hashtags);
        if (res.platforms) {
          const tk = res.platforms.tiktok?.caption || res.caption || '';
          const ig = res.platforms.instagram?.caption || res.caption || '';
          const yt = res.platforms.youtube?.description || res.caption || '';
          setPlatCaptions({ tiktok: tk, ig: ig, yt: yt });
          if (captionPlat === 'tiktok') setCaption(tk);
          else if (captionPlat === 'ig') setCaption(ig);
          else if (captionPlat === 'yt') setCaption(yt);
          else if (res.caption) setCaption(res.caption);
        } else if (res.caption) {
          setCaption(res.caption);
        }
        pushToast?.('success', 'AI Caption, Tags & Speaker generated with video brain awareness!');
      } else {
        const title = currentClip.title || currentClip.video_title_for_youtube_short || 'Viral Moment';
        const speaker = currentClip.speaker_name ? `Featuring ${currentClip.speaker_name}` : '';
        const tags = currentClip.hashtags?.length ? currentClip.hashtags : ['#viral', '#trending', '#shorts', '#clip'];
        const text = `${title}! ${speaker ? `${speaker}. ` : ''}What are your thoughts on this?\n\n${tags.join(' ')}`;
        setCaption(text);
        if (tags.length) setHashtags(tags);
        pushToast?.('success', 'Clip context & tags generated!');
      }
    } catch (err) {
      pushToast?.('warn', `AI Generation failed: ${err?.message || err}`);
    } finally {
      setGeneratingAi(false);
    }
  };

  const toggleTag = (tag) => {
    if (caption.includes(tag)) {
      setCaption((c) => c.replace(new RegExp(`\\s*${tag}`, 'g'), '').trim());
    } else {
      setCaption((c) => (c ? `${c} ${tag}` : tag));
    }
  };

  const buildBody = (clip, idx, batchPos = 0) => {
    const cs = clipStates[idx] || {};
    const toggles = cs.toggles ?? seedToggles(preselections);
    const any = Object.values(toggles).some(Boolean);
    const hookParams = cs.hookParams ?? seedHookParams(clip, preselections);
    const subtitleParams = cs.subtitleParams ?? seedSubtitleParams(preselections);
    const logoParams = cs.logoParams ?? seedLogoParams(preselections);
    const gradeParams = cs.gradeParams ?? { preset: preselections?.grade?.preset || 'none' };
    const bannerParams = cs.bannerParams ?? seedBannerParams(preselections);

    const isSelectedActive = idx === (clips[activeClipIdx]?._idx ?? clips[0]?._idx);
    const resolvedTitle = (isSelectedActive && titleText ? titleText : (clip.video_title_for_youtube_short || clip.title || `Clip ${idx + 1}`)).slice(0, 100);

    let resolvedCaption = (isSelectedActive && caption && caption.trim()) ? caption : (clip.video_description || clip.caption || resolvedTitle);
    if (all && !isSelectedActive) {
      if (clip.video_description) {
        resolvedCaption = clip.video_description;
      } else {
        let desc = clip.video_description_for_tiktok || clip.video_title_for_youtube_short || `Clip ${idx + 1}`;
        if (clip.speaker_name) {
          desc = `Featuring ${clip.speaker_name} — ${desc}`;
        }
        const tags = clip.hashtags?.length ? clip.hashtags : hashtags;
        resolvedCaption = `${desc}\n\n${tags.join(' ')}`;
      }
    }

    let schedulePayload = {};
    if (scheduleMode === 'now') {
      schedulePayload = { schedule_mode: 'now' };
    } else if (scheduleMode === 'smart') {
      schedulePayload = { schedule_mode: 'auto', start_date: localDatePlus(batchPos) };
    } else if (scheduleMode === 'custom') {
      schedulePayload = {
        schedule_mode: 'custom',
        scheduled_for: customDateTime ? new Date(customDateTime).toISOString() : new Date().toISOString(),
      };
    }

    return {
      title: resolvedTitle,
      caption: resolvedCaption,
      platforms: targets,
      ...schedulePayload,
      timezone: zernio?.timezone || 'Europe/Rome',
      generate_ai_thumbnail: aiThumbnail,
      thumbnail_aspect_ratio: aspectRatio,
      tiktok_settings: plats.tiktok && accounts.tiktok ? {
        privacy_level: tiktokPrivacy,
        allow_comment: tiktokComments,
        allow_duet: tiktokDuet,
        allow_stitch: tiktokStitch,
        content_preview_confirmed: true,
        express_consent_given: true,
        is_ai_generated: tiktokAiDisclosure,
        brand_partnership: tiktokBrandPartnership,
      } : undefined,
      ...(any ? {
        compose_first: true,
        toggles,
        hook_params: toggles.hook ? hookParams : {},
        subtitle_params: toggles.subtitles ? subtitleParams : {},
        logo_params: toggles.logo ? logoParams : {},
        grade_params: toggles.grade ? gradeParams : {},
        banner_params: toggles.banner ? bannerParams : {},
        drop_ranges: toggles.smartcut ? (cs.dropRanges || []) : [],
      } : {}),
    };
  };

  const run = async (overrideMode = null) => {
    const effectiveScheduleMode = overrideMode || scheduleMode;
    setStage('uploading');
    const init = {};
    clips.forEach((c) => { init[c._idx] = { state: 'uploading' }; });
    setProgress(init);

    const results = await Promise.allSettled(clips.map(async (clip, batchPos) => {
      const idx = clip._idx;
      const apiIdx = clip._apiIdx ?? idx;
      try {
        let body = buildBody(clip, idx, batchPos);
        if (effectiveScheduleMode === 'now') {
          body.schedule_mode = 'now';
          delete body.start_date;
          delete body.scheduled_for;
        }
        if (all && !clip.video_description && (jobId || clip.job_id)) {
          try {
            const ai = await generateClipMetadata(jobId || clip.job_id, apiIdx);
            if (ai?.caption) body.caption = ai.caption;
          } catch {
            // Best effort metadata generation fallback
          }
        }

        if (typeof onCustomPublish === 'function') {
          await onCustomPublish(clip, body);
        } else {
          await publishClip(jobId, apiIdx, body);
        }

        setProgress((p) => ({ ...p, [idx]: { state: 'done' } }));
        onPublished?.(idx);
        return true;
      } catch (e) {
        setProgress((p) => ({ ...p, [idx]: { state: 'error', error: e?.message || 'Publish failed' } }));
        return false;
      }
    }));

    const ok = results.filter((r) => r.status === 'fulfilled' && r.value).length;
    const fail = clips.length - ok;
    setTimeout(() => {
      if (!mountedRef.current) return;
      setStage('done');
      pushToast?.(fail === 0 ? 'success' : 'warn', `Published ${ok}/${clips.length}${fail ? `, ${fail} failed` : ''}`);
    }, 500);
  };

  const activeVideoUrl = clipVideoSrc(activeClip);

  // Platform limits calculations
  const activePlatKeys = Object.keys(plats).filter((k) => plats[k]);
  const ytTitleOverLimit = plats.yt && titleText.length > 100;
  const anyCaptionOverLimit = activePlatKeys.some((k) => PLAT[k]?.maxCaption && caption.length > PLAT[k].maxCaption);

  const title = stage === 'done' ? (scheduleMode === 'now' ? 'Published' : 'Scheduled')
    : all ? `Publish ${clips.length} clips` : `Publish · ${titleText || activeClip?.video_title_for_youtube_short || ''}`;

  const modalNode = (
    <div className="overlay" onClick={(e) => { if (e.target === e.currentTarget) onClose(); }}>
      <div className="modal publish-wide" ref={panelRef}
        role="dialog" aria-modal="true" aria-labelledby="publish-modal-title">
        <div className="modal-head">
          <div>
            <h3 id="publish-modal-title">{title}</h3>
            {stage === 'uploading' && <div className="mh-sub">uploading concurrently · daily-limit checks server-side</div>}
            {stage === 'setup' && (
              <div className="mh-sub">
                {all ? `Batch review (${clips.length} clips)` : 'Preview, AI metadata & schedule'}
              </div>
            )}
          </div>
          <button className="x" onClick={onClose} aria-label="Close"><Icon n="x" /></button>
        </div>

        {stage === 'setup' && (
          <>
            <div className="modal-body">
              {!zernio ? <div className="cm-small">Loading Zernio…</div> : !zernio.configured ? (
                <div className="empty" style={{ padding: '24px 12px' }}>
                  <div className="ei"><Icon n="rss" /></div>
                  <h3>Zernio not connected</h3>
                  <p>Add your Zernio API key + account IDs in Settings to publish.</p>
                </div>
              ) : (
                <>
                  {/* Multi-clip thumbnail carousel when more than 1 clip */}
                  {all && (
                    <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 16, overflowX: 'auto', paddingBottom: 6 }}>
                      <span style={{ fontSize: '11px', fontWeight: 700, color: 'var(--fg-3)', textTransform: 'uppercase', letterSpacing: '0.04em', flexShrink: 0 }}>
                        Reviewing:
                      </span>
                      {clips.map((c, i) => (
                        <button
                          key={c._idx || i}
                          type="button"
                          onClick={() => setActiveClipIdx(i)}
                          style={{
                            padding: '4px 10px',
                            borderRadius: '16px',
                            border: '1.5px solid ' + (activeClipIdx === i ? 'var(--brand-teal)' : 'rgba(51,46,38,0.1)'),
                            background: activeClipIdx === i ? 'rgba(2,197,191,0.12)' : 'var(--surface-deep)',
                            color: activeClipIdx === i ? 'var(--brand-teal)' : 'var(--ink)',
                            fontSize: '11.5px',
                            fontWeight: activeClipIdx === i ? 700 : 500,
                            cursor: 'pointer',
                            display: 'inline-flex',
                            alignItems: 'center',
                            gap: 6,
                            flexShrink: 0,
                          }}
                        >
                          <span>Clip {i + 1}</span>
                          {c.viral_score && (
                            <span style={{ fontSize: '10px', opacity: 0.85 }}>🔥 {c.viral_score}</span>
                          )}
                        </button>
                      ))}
                    </div>
                  )}

                  <div className="pub-preview-split">
                    {/* Left Column: Video Player & Social Feed Mockup Simulator */}
                    <div className="pub-preview-pane">
                      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 4 }}>
                        <span className="field-label" style={{ margin: 0 }}>Clip Preview</span>
                        <Segmented
                          options={[
                            { id: 'player', label: 'Player' },
                            { id: 'mockup', label: 'Feed' },
                          ]}
                          value={previewMode}
                          onChange={setPreviewMode}
                        />
                      </div>

                      {previewMode === 'mockup' && (
                        <div style={{ display: 'flex', gap: 6, marginBottom: 4 }}>
                          {[
                            { id: 'tiktok', label: 'TikTok' },
                            { id: 'ig', label: 'Reels' },
                            { id: 'yt', label: 'Shorts' },
                          ].map((pm) => (
                            <button
                              key={pm.id}
                              type="button"
                              onClick={() => setMockupPlat(pm.id)}
                              style={{
                                flex: 1,
                                padding: '3px 6px',
                                borderRadius: 6,
                                border: '1px solid ' + (mockupPlat === pm.id ? 'var(--brand-teal)' : 'rgba(51,46,38,0.1)'),
                                background: mockupPlat === pm.id ? 'rgba(2,197,191,0.15)' : 'transparent',
                                color: mockupPlat === pm.id ? 'var(--brand-teal)' : 'var(--fg-3)',
                                fontSize: '10.5px',
                                fontWeight: 600,
                                cursor: 'pointer',
                              }}
                            >
                              {pm.label}
                            </button>
                          ))}
                        </div>
                      )}

                      <div className="pub-video-frame">
                        {previewMode === 'player' ? (
                          <video
                            key={activeVideoUrl}
                            src={activeVideoUrl}
                            controls
                            playsInline
                            preload="metadata"
                            style={{ width: '100%', height: '100%', objectFit: 'contain' }}
                          />
                        ) : (
                          <>
                            <video
                              key={activeVideoUrl + '-mockup'}
                              src={activeVideoUrl}
                              autoPlay
                              loop
                              muted
                              playsInline
                              style={{ width: '100%', height: '100%', objectFit: 'cover' }}
                            />
                            <div className="feed-mockup">
                              <div className="mockup-header">
                                {mockupPlat === 'tiktok' && (
                                  <>
                                    <span>Following</span>
                                    <span className="active">For You</span>
                                  </>
                                )}
                                {mockupPlat === 'ig' && <span className="active">Reels</span>}
                                {mockupPlat === 'yt' && <span className="active">Shorts</span>}
                              </div>

                              <div className="mockup-body">
                                <div className="mockup-meta">
                                  <div className="mockup-author">
                                    @{activeClip.channel || zernio?.accounts?.[mockupPlat] || 'nugget'}
                                    {speakerName && <Badge tone="in" style={{ fontSize: '9px', padding: '1px 5px' }}>{speakerName}</Badge>}
                                  </div>
                                  <div className="mockup-desc">
                                    {(() => {
                                      const mCopy = getClipPlatformCopy(activeClip, mockupPlat);
                                      const mText = platCaptions[mockupPlat] || mCopy.caption || caption || titleText || 'Engaging viral clip highlight';
                                      return mockupPlat === 'yt' && mCopy.title ? `${mCopy.title} — ${mText}` : mText;
                                    })()}
                                  </div>
                                  {(() => {
                                    const mCopy = getClipPlatformCopy(activeClip, mockupPlat);
                                    const mTags = mCopy.hashtags?.length ? mCopy.hashtags : hashtags;
                                    return mTags.length > 0 ? (
                                      <div className="mockup-tags">
                                        {mTags.slice(0, 5).join(' ')}
                                      </div>
                                    ) : null;
                                  })()}
                                  <div className="mockup-music">
                                    <Icon n="sparkles" style={{ width: 10, height: 10 }} />
                                    <span>Original sound · {speakerName || 'Nugget'}</span>
                                  </div>
                                </div>

                                <div className="mockup-sidebar">
                                  <div className="mockup-avatar">⚡</div>
                                  <div className="mockup-btn">
                                    <span style={{ fontSize: 16 }}>❤️</span>
                                    <span>{activeClip.viral_score ? `${activeClip.viral_score}k` : '18.4k'}</span>
                                  </div>
                                  <div className="mockup-btn">
                                    <span style={{ fontSize: 16 }}>💬</span>
                                    <span>842</span>
                                  </div>
                                  <div className="mockup-btn">
                                    <span style={{ fontSize: 16 }}>🔖</span>
                                    <span>Save</span>
                                  </div>
                                  <div className="mockup-btn">
                                    <span style={{ fontSize: 16 }}>↗️</span>
                                    <span>Share</span>
                                  </div>
                                  <div className="mockup-disc">🎵</div>
                                </div>
                              </div>
                            </div>
                          </>
                        )}
                      </div>
                      <div style={{ fontSize: '11px', color: 'var(--fg-4)', textAlign: 'center' }}>
                        {activeClip.duration ? `Duration: ${Math.round(activeClip.duration)}s` : ''}
                        {activeClip.viral_score ? ` · Viral score: ${activeClip.viral_score}/100` : ''}
                      </div>
                    </div>

                    {/* Right Column: Metadata, Social Destinations & Scheduling */}
                    <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
                      {/* Social Destinations */}
                      <div className="field" style={{ marginBottom: 0 }}>
                        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 8 }}>
                          <span className="field-label" style={{ margin: 0 }}>Social Platforms</span>
                          <button
                            type="button"
                            className="btn ghost-btn"
                            style={{
                              padding: '4px 10px',
                              fontSize: '12px',
                              height: '28px',
                              color: 'var(--brand-teal)',
                              display: 'inline-flex',
                              alignItems: 'center',
                              gap: '6px',
                              border: '1px solid var(--brand-teal-dim, rgba(2,197,191,0.2))',
                              borderRadius: '6px',
                              cursor: 'pointer',
                              background: 'transparent',
                            }}
                            onClick={handleGenerateAi}
                            disabled={generatingAi}
                          >
                            <Icon n={generatingAi ? "loader" : "sparkles"} style={{ width: 14, height: 14 }} />
                            {generatingAi ? "AI Generating…" : "✨ AI Caption & Tags"}
                          </button>
                        </div>
                        <div className="plats">
                          {PLATFORMS.map((p) => {
                            const has = !!accounts[PLAT[p.id].acct];
                            return <PlatPill key={p.id} {...p} on={plats[p.id] && has}
                              onClick={() => has ? toggle(p.id) : pushToast?.('warn', `No ${PLAT[p.id].label} account saved`)} />;
                          })}
                        </div>
                      </div>

                      {/* Video Title (YouTube Shorts / General) */}
                      <div className="field" style={{ marginBottom: 0 }}>
                        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 4 }}>
                          <span className="field-label" style={{ margin: 0 }}>Video Title</span>
                          <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                            {speakerName && (
                              <span style={{ fontSize: '11px', color: 'var(--brand-teal)', background: 'rgba(2,197,191,0.1)', padding: '2px 7px', borderRadius: '10px', fontWeight: 600 }}>
                                🗣️ {speakerName}
                              </span>
                            )}
                            <span className={'char-pill' + (ytTitleOverLimit ? ' error' : titleText.length > 85 ? ' warn' : '')}>
                              {titleText.length}/100
                            </span>
                          </div>
                        </div>
                        <input
                          className="key-input"
                          style={{
                            width: '100%',
                            fontFamily: 'var(--font-sans)',
                            fontSize: '13px',
                            padding: '8px 12px',
                            borderColor: ytTitleOverLimit ? 'var(--danger)' : undefined,
                          }}
                          value={titleText}
                          maxLength={120}
                          onChange={(e) => setTitleText(e.target.value)}
                          placeholder="Viral video title for YouTube Shorts..."
                        />
                        {ytTitleOverLimit && (
                          <div style={{ color: 'var(--danger)', fontSize: '11px', marginTop: 3 }}>
                            YouTube Shorts max title length is 100 characters.
                          </div>
                        )}
                      </div>

                      {/* Caption & Description */}
                      <div className="field" style={{ marginBottom: 0 }}>
                        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 4 }}>
                          <span className="field-label" style={{ margin: 0 }}>Caption & Description</span>
                          <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                            {activePlatKeys.map((pk) => {
                              const max = PLAT[pk]?.maxCaption;
                              if (!max) return null;
                              const isOver = caption.length > max;
                              return (
                                <span key={pk} className={'char-pill' + (isOver ? ' error' : caption.length > max - 100 ? ' warn' : '')}>
                                  {PLAT[pk].label}: {caption.length}/{max}
                                </span>
                              );
                            })}
                          </div>
                        </div>

                        {/* Platform-Specific Caption Switcher */}
                        <div style={{ display: 'flex', gap: 6, marginBottom: 6 }}>
                          {[
                            { id: 'tiktok', label: 'TikTok 🎵' },
                            { id: 'ig', label: 'Reels 📷' },
                            { id: 'yt', label: 'Shorts ▶️' },
                          ].map((pm) => (
                            <button
                              key={pm.id}
                              type="button"
                              onClick={() => {
                                setCaptionPlat(pm.id);
                                setMockupPlat(pm.id);
                                const nextVal = platCaptions[pm.id] || getClipPlatformCopy(activeClip, pm.id).caption || '';
                                setCaption(nextVal);
                              }}
                              style={{
                                padding: '3px 8px',
                                borderRadius: 6,
                                border: '1px solid ' + (captionPlat === pm.id ? 'var(--brand-teal)' : 'rgba(51,46,38,0.1)'),
                                background: captionPlat === pm.id ? 'rgba(2,197,191,0.15)' : 'transparent',
                                color: captionPlat === pm.id ? 'var(--brand-teal)' : 'var(--fg-3)',
                                fontSize: '11px',
                                fontWeight: 600,
                                cursor: 'pointer',
                              }}
                            >
                              {pm.label}
                            </button>
                          ))}
                        </div>

                        <textarea
                          className="ta"
                          rows={3}
                          value={caption}
                          onChange={(e) => {
                            const val = e.target.value;
                            setCaption(val);
                            setPlatCaptions((prev) => ({ ...prev, [captionPlat]: val }));
                          }}
                          placeholder="Write or generate an engaging description with trending tags..."
                          style={{ borderColor: anyCaptionOverLimit ? 'var(--danger)' : undefined }}
                        />
                      </div>

                      {/* Trending Tags Chips */}
                      {hashtags.length > 0 && (
                        <div className="field" style={{ marginBottom: 0 }}>
                          <span className="field-label" style={{ fontSize: '11.5px', color: 'var(--fg-3)', marginBottom: 6, display: 'block' }}>
                            🔥 Trending & Topic Tags (click to toggle in caption):
                          </span>
                          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px' }}>
                            {hashtags.map((tag) => {
                              const active = caption.includes(tag);
                              return (
                                <button
                                  key={tag}
                                  type="button"
                                  onClick={() => toggleTag(tag)}
                                  style={{
                                    border: '1px solid ' + (active ? 'var(--brand-teal)' : 'var(--border)'),
                                    background: active ? 'rgba(2,197,191,0.15)' : 'var(--surface-deep)',
                                    color: active ? 'var(--brand-teal)' : 'var(--fg-2)',
                                    padding: '3px 8px',
                                    borderRadius: '12px',
                                    fontSize: '11px',
                                    fontWeight: active ? 600 : 400,
                                    cursor: 'pointer',
                                    transition: 'all 0.15s',
                                  }}
                                >
                                  {active ? '✓ ' : '+ '}{tag}
                                </button>
                              );
                            })}
                          </div>
                        </div>
                      )}

                      {/* 3-Mode Scheduling */}
                      <div className="field" style={{ marginBottom: 0 }}>
                        <span className="field-label" style={{ marginBottom: 6 }}>Publish Timing</span>
                        <Segmented
                          full
                          options={[
                            { id: 'smart', label: 'Prime Time (Auto)' },
                            { id: 'now', label: 'Publish Now' },
                            { id: 'custom', label: 'Custom Time' },
                          ]}
                          value={scheduleMode}
                          onChange={setScheduleMode}
                        />

                        {scheduleMode === 'custom' && (
                          <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginTop: 8, padding: '8px 12px', background: 'var(--surface-deep)', borderRadius: 10 }}>
                            <input
                              type="datetime-local"
                              className="key-input"
                              value={customDateTime}
                              min={new Date().toISOString().slice(0, 16)}
                              onChange={(e) => setCustomDateTime(e.target.value)}
                              style={{ flex: 1, fontSize: '12.5px' }}
                            />
                            <span style={{ fontSize: '11px', color: 'var(--fg-4)' }}>
                              {zernio?.timezone || 'Europe/Rome'}
                            </span>
                          </div>
                        )}
                      </div>

                      {/* AI Thumbnail Cover Toggle */}
                      <div className="opt" style={{ borderBottom: 0, padding: 0, marginTop: 2 }}>
                        <div className="otxt">
                          <div className="ot">AI Thumbnail / Cover (Nano Banana)</div>
                          <div className="od">Auto-generate and attach high-CTR visual cover</div>
                        </div>
                        <div className="r"><Switch on={aiThumbnail} onChange={setAiThumbnail} /></div>
                      </div>

                      {aiThumbnail && (
                        <div style={{ padding: '8px 12px', background: 'var(--surface-deep)', borderRadius: 10 }}>
                          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                            <span className="field-label" style={{ margin: 0, fontSize: '11.5px' }}>Aspect Ratio:</span>
                            <div style={{ display: 'flex', gap: 6 }}>
                              {[
                                { id: '9:16', label: '9:16 (Shorts/Reels)' },
                                { id: '1:1', label: '1:1 (Feed)' },
                                { id: '16:9', label: '16:9 (Landscape)' },
                              ].map((r) => (
                                <button
                                  key={r.id}
                                  type="button"
                                  onClick={() => setAspectRatio(r.id)}
                                  style={{
                                    border: '1px solid ' + (aspectRatio === r.id ? 'var(--brand-teal)' : 'var(--border)'),
                                    background: aspectRatio === r.id ? 'rgba(2,197,191,0.15)' : 'transparent',
                                    color: aspectRatio === r.id ? 'var(--brand-teal)' : 'var(--fg-2)',
                                    padding: '2px 8px',
                                    borderRadius: '6px',
                                    fontSize: '11px',
                                    fontWeight: aspectRatio === r.id ? 600 : 400,
                                    cursor: 'pointer',
                                  }}
                                >
                                  {r.label}
                                </button>
                              ))}
                            </div>
                          </div>
                        </div>
                      )}

                      {/* TikTok Advanced Settings Drawer */}
                      {plats.tiktok && (
                        <div style={{ border: '1px solid rgba(51,46,38,0.08)', borderRadius: 12, padding: 10, background: 'var(--surface-deep)' }}>
                          <button
                            type="button"
                            onClick={() => setTiktokExpanded((e) => !e)}
                            style={{
                              width: '100%',
                              background: 'transparent',
                              border: 0,
                              cursor: 'pointer',
                              display: 'flex',
                              alignItems: 'center',
                              justifyContent: 'space-between',
                              color: 'var(--ink)',
                              padding: 0,
                            }}
                          >
                            <span style={{ fontSize: '12px', fontWeight: 700, display: 'flex', alignItems: 'center', gap: 6 }}>
                              <Social n="tiktok" size={13} />
                              TikTok Advanced Settings
                            </span>
                            <Icon n={tiktokExpanded ? 'chevron-up' : 'chevron-down'} style={{ width: 14, height: 14 }} />
                          </button>

                          {tiktokExpanded && (
                            <div style={{ marginTop: 10, display: 'flex', flexDirection: 'column', gap: 8 }}>
                              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                                <span style={{ fontSize: '11.5px', color: 'var(--fg-3)' }}>Privacy:</span>
                                <select
                                  className="key-input"
                                  value={tiktokPrivacy}
                                  onChange={(e) => setTiktokPrivacy(e.target.value)}
                                  style={{ fontSize: '11.5px', padding: '3px 8px' }}
                                >
                                  <option value="PUBLIC_TO_EVERYONE">Public (Everyone)</option>
                                  <option value="MUTUAL_FOLLOW_FRIENDS">Friends Only</option>
                                  <option value="SELF_ONLY">Private (Only Me)</option>
                                </select>
                              </div>
                              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                                <span style={{ fontSize: '11.5px', color: 'var(--fg-3)' }}>Allow Comments</span>
                                <Switch on={tiktokComments} onChange={setTiktokComments} />
                              </div>
                              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                                <span style={{ fontSize: '11.5px', color: 'var(--fg-3)' }}>Allow Duet</span>
                                <Switch on={tiktokDuet} onChange={setTiktokDuet} />
                              </div>
                              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                                <span style={{ fontSize: '11.5px', color: 'var(--fg-3)' }}>Allow Stitch</span>
                                <Switch on={tiktokStitch} onChange={setTiktokStitch} />
                              </div>
                              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                                <span style={{ fontSize: '11.5px', color: 'var(--fg-3)' }}>AI-Generated Content Tag</span>
                                <Switch on={tiktokAiDisclosure} onChange={setTiktokAiDisclosure} />
                              </div>
                            </div>
                          )}
                        </div>
                      )}
                    </div>
                  </div>
                </>
              )}
            </div>

            <div className="modal-foot">
              <Btn variant="ghost" onClick={onClose}>Cancel</Btn>
              <div className="mf-right">
                <Btn
                  variant="secondary"
                  icon="send"
                  disabled={!ready}
                  onClick={() => run('now')}
                >
                  Publish Now
                </Btn>
                <Btn
                  variant="grad"
                  icon="calendar-clock"
                  disabled={!ready}
                  onClick={() => run()}
                >
                  {scheduleMode === 'smart' ? 'Schedule Prime Time' : scheduleMode === 'custom' ? 'Schedule Slot' : 'Publish Now'}
                </Btn>
              </div>
            </div>
          </>
        )}

        {stage === 'uploading' && (
          <div className="modal-body">
            <div className="pubgrid">
              {clips.map((c) => <PubRow key={c._idx} clip={c} idx={c._idx} st={progress[c._idx]} plats={plats} />)}
            </div>
          </div>
        )}

        {stage === 'done' && (
          <div className="modal-body" style={{ textAlign: 'center', padding: '36px 24px' }}>
            <div style={{ width: 60, height: 60, borderRadius: '50%', background: 'var(--success-bg)', display: 'flex', alignItems: 'center', justifyContent: 'center', margin: '0 auto 18px' }}>
              <Icon n={scheduleMode === 'now' ? 'party-popper' : 'calendar-check'} style={{ width: 28, height: 28, color: 'var(--brand-teal)' }} />
            </div>
            <div style={{ fontWeight: 700, fontSize: 18 }}>
              {all ? `${clips.length} clips ` : 'Clip '}
              {scheduleMode === 'now' ? 'published' : 'scheduled'}
            </div>
            <p style={{ color: 'var(--fg-3)', fontSize: 13.5, marginTop: 8, lineHeight: 1.5 }}>
              {scheduleMode === 'now'
                ? 'Sent to Zernio for immediate live publication.'
                : 'Queued via Zernio for prime-time automated distribution.'}
            </p>
            <div style={{ marginTop: 22 }}>
              <Btn variant="secondary" onClick={onClose}>Done</Btn>
            </div>
          </div>
        )}
      </div>
    </div>
  );

  return typeof document !== 'undefined' ? createPortal(modalNode, document.body) : modalNode;
}
