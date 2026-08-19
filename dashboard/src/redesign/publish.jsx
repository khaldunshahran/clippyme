// ClippyMe redesign — PublishModal: real concurrent publish to Zernio.
// Every selected clip is published in parallel (Promise.allSettled) — the fix
// for the old sequential stall — each row showing live queued→uploading→
// live/error status. Per-clip compose_first honours the clip's toggles.
import { useState, useEffect, useRef } from 'react';
import { Icon, Social, Btn, Switch, PlatPill, PLATFORMS } from './primitives';
import { LazyVideo } from './LazyVideo';
import { clipVideoSrc, publishClip, getZernio, generateClipMetadata } from './realApi';
import { seedToggles, seedHookParams, seedSubtitleParams, seedLogoParams, seedBannerParams } from '../lib/seedClipParams';
import { localDatePlus } from '../lib/scheduleDates';
import { useModalA11y } from './useModalA11y';

// redesign plat id → backend platform + account key. Exported so other
// surfaces publishing to Zernio (live.jsx) don't re-derive this mapping.
export const PLAT = {
  tiktok: { platform: 'tiktok', acct: 'tiktok', icon: 'tiktok', label: 'TikTok' },
  ig: { platform: 'instagram', acct: 'instagram', icon: 'instagram', label: 'Reels' },
  yt: { platform: 'youtube', acct: 'youtube', icon: 'youtube', label: 'Shorts' },
};

function PubRow({ clip, idx, st, plats }) {
  // `st` is either a status string or { state, error } so we can surface the
  // real failure reason instead of a bare "failed".
  const status = typeof st === 'object' && st ? st.state : st;
  const errMsg = typeof st === 'object' && st ? st.error : null;
  const tasks = Object.keys(plats).filter((k) => plats[k]);
  const done = status === 'done';
  const error = status === 'error';
  return (
    <div className={'pubrow' + (done ? ' done' : '')}>
      <div className="pthumb" style={{ background: '#000', overflow: 'hidden' }}>
        {/* LazyVideo, not a bare <video>: a 20-clip batch publish must not fire
            20 concurrent video fetches the moment the modal opens. */}
        <LazyVideo src={clipVideoSrc(clip)} muted playsInline rootMargin="120px"
          style={{ width: '100%', height: '100%', objectFit: 'cover' }} />
      </div>
      <div className="pinfo">
        <div className="pttl">{clip.video_title_for_youtube_short || `Clip ${idx + 1}`}</div>
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

export function PublishModal({ clips, jobId, clipStates = {}, preselections, onClose, onPublished, pushToast }) {
  const all = clips.length > 1;
  const firstClip = clips[0] || {};
  const [zernio, setZernio] = useState(null);
  const [plats, setPlats] = useState({ tiktok: false, ig: false, yt: false });
  const [schedule, setSchedule] = useState(true);
  const [titleText, setTitleText] = useState(firstClip.video_title_for_youtube_short || '');
  const [caption, setCaption] = useState(
    firstClip.video_description || firstClip.video_description_for_tiktok || firstClip.tiktok_caption || firstClip.video_title_for_youtube_short || ''
  );
  const [speakerName, setSpeakerName] = useState(firstClip.speaker_name || '');
  const [hashtags, setHashtags] = useState(
    firstClip.hashtags?.length ? firstClip.hashtags : ['#shorts', '#trending', '#viral']
  );
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
          });
        }
      })
      .catch(() => setZernio({ configured: false }));
  }, []);

  // Accessibility: focus trap + Escape-to-close + focus restore.
  const panelRef = useModalA11y(onClose);

  // Guard the post-publish setTimeout so it never calls setState after the
  // modal has been unmounted (e.g. parent closes it while the delay is in flight).
  const mountedRef = useRef(true);
  useEffect(() => () => { mountedRef.current = false; }, []);

  const accounts = zernio?.accounts || {};
  const toggle = (k) => setPlats((p) => ({ ...p, [k]: !p[k] }));
  const platTargets = () => Object.keys(plats)
    .filter((k) => plats[k] && accounts[PLAT[k].acct])
    .map((k) => ({ platform: PLAT[k].platform, accountId: accounts[PLAT[k].acct] }));
  const targets = platTargets();
  const handleGenerateAi = async () => {
    if (generatingAi) return;
    if (!jobId) {
      pushToast?.('warn', 'No active job ID found for metadata generation');
      return;
    }
    setGeneratingAi(true);
    const apiIdx = Number.isInteger(firstClip.original_index) ? firstClip.original_index : (firstClip._apiIdx ?? firstClip._idx ?? 0);
    try {
      const res = await generateClipMetadata(jobId, apiIdx);
      if (res.title) setTitleText(res.title);
      if (res.speaker_name) setSpeakerName(res.speaker_name);
      if (res.hashtags?.length) setHashtags(res.hashtags);
      if (res.caption) setCaption(res.caption);
      pushToast?.('success', 'AI Caption, Tags & Speaker generated!');
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

  // `batchPos` is the clip's position within this batch (0-based). When
  // scheduling, each clip gets its own day (start_date = today + batchPos) so
  // a per-platform daily cap doesn't reject the tail of the batch — replicates
  // the one-clip-per-day spacing from the original publisher.
  const buildBody = (clip, idx, batchPos = 0) => {
    const cs = clipStates[idx] || {};
    const toggles = cs.toggles ?? seedToggles(preselections);
    const any = Object.values(toggles).some(Boolean);
    const hookParams = cs.hookParams ?? seedHookParams(clip, preselections);
    const subtitleParams = cs.subtitleParams ?? seedSubtitleParams(preselections);
    const logoParams = cs.logoParams ?? seedLogoParams(preselections);
    const gradeParams = cs.gradeParams ?? { preset: preselections?.grade?.preset || 'none' };
    const bannerParams = cs.bannerParams ?? seedBannerParams(preselections);
    const resolvedTitle = (titleText && !all ? titleText : (clip.video_title_for_youtube_short || `Clip ${idx + 1}`)).slice(0, 100);
    let resolvedCaption = (caption && caption.trim()) || resolvedTitle;
    if (all) {
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
    return {
      title: resolvedTitle,
      caption: resolvedCaption,
      platforms: targets,
      schedule_mode: schedule ? 'auto' : 'now',
      ...(schedule ? { start_date: localDatePlus(batchPos) } : {}),
      timezone: zernio?.timezone || 'Europe/Rome',
      tiktok_settings: plats.tiktok && accounts.tiktok ? {
        privacy_level: 'PUBLIC_TO_EVERYONE', allow_comment: true, allow_duet: true,
        allow_stitch: true, content_preview_confirmed: true, express_consent_given: true,
      } : undefined,
      ...(any ? { compose_first: true, toggles, hook_params: toggles.hook ? hookParams : {}, subtitle_params: toggles.subtitles ? subtitleParams : {}, logo_params: toggles.logo ? logoParams : {}, grade_params: toggles.grade ? gradeParams : {}, banner_params: toggles.banner ? bannerParams : {}, drop_ranges: toggles.smartcut ? (cs.dropRanges || []) : [] } : {}),
    };
  };

  const run = async () => {
    setStage('uploading');
    const init = {};
    clips.forEach((c) => { init[c._idx] = { state: 'uploading' }; });
    setProgress(init);
    const results = await Promise.allSettled(clips.map(async (clip, batchPos) => {
      const idx = clip._idx;
      // Resolve to the backend's ABSOLUTE `shorts` position for the actual
      // publish call — `idx` (array position) stays the key into local
      // clipStates/progress, which are unaffected by a manual-publish gap.
      const apiIdx = clip._apiIdx ?? idx;
      try {
        let body = buildBody(clip, idx, batchPos);
        // In batch publish, generate fresh unique AI metadata for clips that don't have rich descriptions
        if (all && !clip.video_description) {
          try {
            const ai = await generateClipMetadata(jobId, apiIdx);
            if (ai?.caption) body.caption = ai.caption;
            if (ai?.title) body.title = ai.title;
          } catch (_) {}
        }
        await publishClip(jobId, apiIdx, body);
        setProgress((p) => ({ ...p, [idx]: { state: 'done' } }));
        onPublished?.(idx);
        return true;
      } catch (e) {
        // Surface the real reason (e.g. a Zernio daily-limit 429) instead of a
        // bare "failed", so the user knows to retry that platform tomorrow.
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

  const title = stage === 'done' ? (schedule ? 'Scheduled' : 'Published')
    : all ? `Publish ${clips.length} clips` : `Publish · ${titleText || firstClip?.video_title_for_youtube_short || ''}`;

  return (
    // Backdrop click is a mouse-only convenience; keyboard users close via
    // Esc (useModalA11y). currentTarget guard replaces stopPropagation.
    <div className="overlay" onClick={(e) => { if (e.target === e.currentTarget) onClose(); }}>
      <div className={'modal' + (all ? ' wide' : '')} ref={panelRef}
        role="dialog" aria-modal="true" aria-labelledby="publish-modal-title">
        <div className="modal-head">
          <div>
            <h3 id="publish-modal-title">{title}</h3>
            {stage === 'uploading' && <div className="mh-sub">uploading concurrently · daily-limit checks server-side</div>}
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
                  <div className="field">
                    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 8 }}>
                      <span className="field-label" style={{ margin: 0 }}>Social Platforms</span>
                      <button
                        type="button"
                        className="btn ghost-btn"
                        style={{ padding: '4px 10px', fontSize: '12px', height: '28px', color: 'var(--brand-teal)', display: 'inline-flex', alignItems: 'center', gap: '6px', border: '1px solid var(--brand-teal-dim, rgba(2,197,191,0.2))', borderRadius: '6px', cursor: 'pointer', background: 'transparent' }}
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

                  {!all && (
                    <div className="field">
                      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 4 }}>
                        <span className="field-label" style={{ margin: 0 }}>Video Title</span>
                        {speakerName && (
                          <span style={{ fontSize: '11.5px', color: 'var(--brand-teal)', background: 'rgba(2,197,191,0.1)', padding: '2px 8px', borderRadius: '12px', fontWeight: 600 }}>
                            🗣️ {speakerName}
                          </span>
                        )}
                      </div>
                      <input
                        className="key-input"
                        style={{ width: '100%', fontFamily: 'var(--font-sans)', fontSize: '13.5px', padding: '8px 12px' }}
                        value={titleText}
                        maxLength={100}
                        onChange={(e) => setTitleText(e.target.value)}
                        placeholder="Viral video title for YouTube Shorts..."
                      />
                    </div>
                  )}

                  <div className="field">
                    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 4 }}>
                      <span className="field-label" style={{ margin: 0 }}>Caption & Description</span>
                      <span style={{ fontSize: '11px', color: 'var(--fg-3)' }}>{caption.length}/2200 chars</span>
                    </div>
                    <textarea
                      className="ta"
                      rows={4}
                      value={caption}
                      onChange={(e) => setCaption(e.target.value)}
                      placeholder="Write or generate an engaging description with trending tags..."
                    />
                  </div>

                  {hashtags.length > 0 && (
                    <div className="field" style={{ marginTop: -4 }}>
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
                                background: active ? 'rgba(2,197,191,0.15)' : 'var(--bg-card)',
                                color: active ? 'var(--brand-teal)' : 'var(--fg-2)',
                                padding: '3px 8px',
                                borderRadius: '12px',
                                fontSize: '11.5px',
                                fontWeight: active ? 600 : 400,
                                cursor: 'pointer',
                                transition: 'all 0.2s',
                              }}
                            >
                              {active ? '✓ ' : '+ '}{tag}
                            </button>
                          );
                        })}
                      </div>
                    </div>
                  )}

                  <div className="opt" style={{ borderBottom: 0, marginTop: 4 }}>
                    <div className="oico"><Icon n="calendar-clock" /></div>
                    <div className="otxt"><div className="ot">Schedule for prime time</div><div className="od">SmartScheduler picks the slot · off = publish now</div></div>
                    <div className="r"><Switch on={schedule} onChange={setSchedule} /></div>
                  </div>
                </>
              )}
            </div>
            <div className="modal-foot">
              <Btn variant="ghost" onClick={onClose}>Cancel</Btn>
              <div className="mf-right">
                <Btn variant="secondary" icon="send" disabled={!ready} onClick={() => { setSchedule(false); run(); }}>Publish now</Btn>
                <Btn variant="grad" icon="calendar-clock" disabled={!ready} onClick={run}>{schedule ? 'Schedule' : 'Queue'}</Btn>
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
              <Icon n={schedule ? 'calendar-check' : 'party-popper'} style={{ width: 28, height: 28, color: 'var(--brand-teal)' }} />
            </div>
            <div style={{ fontWeight: 700, fontSize: 18 }}>{all ? `${clips.length} clips ` : 'Clip '}{schedule ? 'scheduled' : 'published'}</div>
            <p style={{ color: 'var(--fg-3)', fontSize: 13.5, marginTop: 8, lineHeight: 1.5 }}>
              {schedule ? 'Queued via Zernio for the next prime-time slot.' : 'Sent to Zernio for immediate publish.'}
            </p>
            <div style={{ marginTop: 22 }}><Btn variant="secondary" onClick={onClose}>Done</Btn></div>
          </div>
        )}
      </div>
    </div>
  );
}
