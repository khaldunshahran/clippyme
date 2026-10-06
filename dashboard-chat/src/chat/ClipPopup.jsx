import { useEffect, useMemo, useRef, useState } from 'react';
import {
  X, ChevronLeft, ChevronRight, Download, CalendarClock, Scissors,
  Share2, FileCode2, ArrowUpFromLine, WandSparkles, Crop, Copy,
  Loader2, Check,
} from 'lucide-react';
import { safeResolveUrl, downloadClip, getClipTranscript } from '../api/realApi.js';
import {
  publishClip, editAiTrim, fmtDur, duplicateClip, upscaleClip,
  exportClipXml, getZernioAccounts, zernioTargets,
} from '../api/chatApi.js';
import { getApiUrl } from '../api/config.js';

export default function ClipPopup({ clips, index, jobId, onClose, onPrev, onNext, onEdit, onDuplicate }) {
  const clip = clips[index];
  const [transcript, setTranscript] = useState(null);
  const [txLoading, setTxLoading] = useState(false);
  const [publishing, setPublishing] = useState(false);
  const [publishDone, setPublishDone] = useState(null);
  const [aiOpen, setAiOpen] = useState(false);
  const [aiPrompt, setAiPrompt] = useState('');
  const [aiBusy, setAiBusy] = useState(false);
  const [aiResult, setAiResult] = useState(null);
  const [scheduleMode, setScheduleMode] = useState('now');
  const [manualWhen, setManualWhen] = useState('');
  const [showPublish, setShowPublish] = useState(false);
  const [busy, setBusy] = useState(null); // 'xml' | 'upscale' | 'duplicate'
  const [actMsg, setActMsg] = useState(null); // { ok, text }
  const videoRef = useRef(null);

  // keyboard: arrows flip, esc closes
  useEffect(() => {
    const h = (e) => {
      if (e.key === 'Escape') onClose();
      if (e.key === 'ArrowLeft') onPrev();
      if (e.key === 'ArrowRight') onNext();
    };
    window.addEventListener('keydown', h);
    return () => window.removeEventListener('keydown', h);
  }, [onClose, onPrev, onNext]);

  // load transcript per clip
  useEffect(() => {
    if (!clip) return;
    setTranscript(null); setTxLoading(true);
    getClipTranscript(jobId, clip.index)
      .then((t) => setTranscript(t))
      .catch(() => setTranscript({ segments: [] }))
      .finally(() => setTxLoading(false));
  }, [jobId, clip?.index]);

  // reset per-clip action state when flipping
  useEffect(() => {
    setBusy(null); setActMsg(null); setPublishDone(null);
  }, [clip?.index]);

  const videoSrc = useMemo(
    () => (clip?.videoUrl ? safeResolveUrl(clip.videoUrl) : ''),
    [clip],
  );

  if (!clip) return null;

  const doDownload = () => {
    try { downloadClip(clip.raw || clip, clip.index); }
    catch {
      const a = document.createElement('a');
      a.href = videoSrc; a.download = '';
      document.body.appendChild(a); a.click(); a.remove();
    }
  };

  const resolveTargets = async () => {
    const r = await getZernioAccounts();
    const targets = zernioTargets(r?.accounts || r);
    if (!targets.length) throw new Error('No connected social accounts found in Zernio.');
    return targets;
  };

  const doPublish = async () => {
    setPublishing(true); setPublishDone(null);
    try {
      const targets = await resolveTargets();
      const body = {
        title: (clip.title || '').slice(0, 100),
        caption: clip.hook || clip.title || '',
        platforms: targets,
        schedule_mode: scheduleMode,
      };
      if (scheduleMode === 'manual') {
        if (!manualWhen) throw new Error('Pick a date & time for the scheduled post.');
        body.scheduled_for = new Date(manualWhen).toISOString();
      }
      const r = await publishClip(jobId, clip.index, body);
      const when = r?.scheduled_for ? ` — scheduled for ${r.scheduled_for}` : '';
      setPublishDone({ ok: true, text: (r?.message || 'Published — check your connected accounts.') + when });
    } catch (e) {
      setPublishDone({ ok: false, text: `Publish failed: ${e.message}` });
    } finally {
      setPublishing(false);
    }
  };

  const doXml = async () => {
    if (busy) return;
    setBusy('xml'); setActMsg(null);
    try {
      await exportClipXml(jobId, clip.index);
      setActMsg({ ok: true, text: 'Timeline XML downloaded.' });
    } catch (e) {
      setActMsg({ ok: false, text: `Export failed: ${e.message}` });
    } finally {
      setBusy(null);
    }
  };

  const doUpscale = async () => {
    if (busy) return;
    setBusy('upscale'); setActMsg(null);
    try {
      const r = await upscaleClip(jobId, clip.index);
      const url = getApiUrl(r.download_url || r.file || '');
      const a = document.createElement('a');
      a.href = url; a.download = (r.file || `upscaled_clip_${clip.index + 1}.mp4`).split('/').pop();
      document.body.appendChild(a); a.click(); a.remove();
      setActMsg({ ok: true, text: 'Upscaled — download started.' });
    } catch (e) {
      setActMsg({ ok: false, text: `Upscale failed: ${e.message}` });
    } finally {
      setBusy(null);
    }
  };

  const doDuplicate = async () => {
    if (busy) return;
    setBusy('duplicate'); setActMsg(null);
    try {
      const r = await duplicateClip(jobId, clip.index);
      setActMsg({ ok: true, text: `Duplicated as clip ${Number(r.new_index) + 1}.` });
      if (onDuplicate) onDuplicate(r.new_index);
    } catch (e) {
      setActMsg({ ok: false, text: `Duplicate failed: ${e.message}` });
    } finally {
      setBusy(null);
    }
  };

  const doAiTrim = async () => {
    if (!aiPrompt.trim() || aiBusy) return;
    setAiBusy(true); setAiResult(null);
    try {
      const r = await editAiTrim(jobId, clip.index, aiPrompt.trim());
      setAiResult({ ok: true, text: r?.explanation || 'Done — open the editor to review the trim.' });
    } catch (e) {
      setAiResult({ ok: false, text: `AI edit failed: ${e.message}` });
    } finally {
      setAiBusy(false);
    }
  };

  const actBtn = (id, Icon, label, fn) => {
    const isBusy = busy === id;
    return (
      <button
        key={id}
        className="nc-btn-ghost"
        disabled={!!busy}
        onClick={fn}
        title={label}
        style={{ justifyContent: 'flex-start', fontSize: 12.5, padding: '9px 12px', opacity: busy && !isBusy ? 0.55 : 1 }}
      >
        {isBusy ? <Loader2 size={14} style={{ animation: 'nc-spin 1s linear infinite' }} /> : <Icon size={14} />}
        <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
          {isBusy ? 'Working…' : label}
        </span>
      </button>
    );
  };

  const ACTIONS = [
    { id: 'publish', icon: Share2, label: 'Publish on Social', fn: () => setShowPublish((v) => !v), custom: true },
    { id: 'xml', icon: FileCode2, label: 'Export XML', fn: doXml },
    { id: 'download', icon: Download, label: 'Download HD', fn: doDownload, custom: true },
    { id: 'upscale', icon: ArrowUpFromLine, label: 'Upscale & download', fn: doUpscale },
    { id: 'edit', icon: Scissors, label: 'Edit clip', fn: () => onEdit(clip.index), custom: true },
    { id: 'ai', icon: WandSparkles, label: 'AI tools', fn: () => setAiOpen((v) => !v), custom: true },
    { id: 'aspect', icon: Crop, label: 'Aspect ratio', fn: () => onEdit(clip.index), custom: true },
    { id: 'duplicate', icon: Copy, label: 'Duplicate', fn: doDuplicate },
  ];

  const segments = transcript?.segments || [];

  return (
    <div className="nc-modal-backdrop" onClick={onClose} style={{ padding: 12 }}>
      {/* prev/next arrows */}
      <button
        className="nc-icon-btn" onClick={(e) => { e.stopPropagation(); onPrev(); }}
        aria-label="Previous clip"
        style={{
          position: 'fixed', left: 14, top: '50%', transform: 'translateY(-50%)', zIndex: 61,
          width: 52, height: 52, borderRadius: '50%',
          background: 'rgba(255,255,255,0.08)', border: '1px solid var(--nc-border-strong)',
          backdropFilter: 'blur(8px)',
        }}
      >
        <ChevronLeft size={26} />
      </button>
      <button
        className="nc-icon-btn" onClick={(e) => { e.stopPropagation(); onNext(); }}
        aria-label="Next clip"
        style={{
          position: 'fixed', right: 14, top: '50%', transform: 'translateY(-50%)', zIndex: 61,
          width: 52, height: 52, borderRadius: '50%',
          background: 'rgba(255,255,255,0.08)', border: '1px solid var(--nc-border-strong)',
          backdropFilter: 'blur(8px)',
        }}
      >
        <ChevronRight size={26} />
      </button>

      <div
        className="nc-modal nc-card"
        onClick={(e) => e.stopPropagation()}
        role="dialog" aria-label={clip.title}
        style={{ width: 'min(1020px, calc(100vw - 120px))', background: 'var(--nc-bg-soft)', borderRadius: 18, overflow: 'hidden' }}
      >
        <div style={{ display: 'flex', alignItems: 'center', padding: '12px 16px', borderBottom: '1px solid var(--nc-border)' }}>
          <div style={{ flex: 1, minWidth: 0 }}>
            <div style={{ fontSize: 14.5, fontWeight: 700, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{clip.title}</div>
            <div style={{ fontSize: 12, color: 'var(--nc-text-faint)' }}>Clip {index + 1} of {clips.length}</div>
          </div>
          <button className="nc-icon-btn" onClick={onClose} aria-label="Close"><X size={18} /></button>
        </div>

        <div style={{ display: 'flex', minHeight: 0, flexWrap: 'wrap' }}>
          {/* preview */}
          <div style={{ flex: '1 1 380px', background: '#000', display: 'flex', alignItems: 'center', justifyContent: 'center', minHeight: 320 }}>
            {videoSrc ? (
              <video
                key={videoSrc}
                ref={videoRef}
                src={videoSrc}
                controls playsInline preload="metadata"
                style={{ width: '100%', maxHeight: '62vh', background: '#000' }}
              />
            ) : (
              <div style={{ color: 'var(--nc-text-faint)', fontSize: 13 }}>Preview unavailable</div>
            )}
          </div>

          {/* details */}
          <div className="nc-scroll" style={{ flex: '1 1 320px', maxWidth: 420, overflowY: 'auto', maxHeight: '62vh', padding: '16px 18px', display: 'flex', flexDirection: 'column', gap: 14 }}>
            {clip.hook && (
              <div>
                <div className="nc-label">Hook</div>
                <div style={{ fontSize: 14, fontWeight: 600, lineHeight: 1.45 }}>"{clip.hook}"</div>
              </div>
            )}
            {clip.scene && (
              <div>
                <div className="nc-label">Scene analysis</div>
                <div style={{ fontSize: 13, color: 'var(--nc-text-dim)', lineHeight: 1.55 }}>{clip.scene}</div>
              </div>
            )}
            <div>
              <div className="nc-label">Transcript</div>
              {txLoading ? (
                <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                  {[0, 1, 2].map((i) => <div key={i} className="nc-skeleton" style={{ height: 12, width: `${85 - i * 15}%` }} />)}
                </div>
              ) : segments.length > 0 ? (
                <div className="nc-scroll" style={{ maxHeight: 180, overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: 6 }}>
                  {segments.map((sg, i) => (
                    <button
                      key={sg.index ?? i}
                      onClick={() => { if (videoRef.current && sg.start != null) { videoRef.current.currentTime = sg.start; videoRef.current.play().catch(() => {}); } }}
                      style={{
                        display: 'flex', gap: 10, textAlign: 'left', background: 'transparent',
                        border: 'none', cursor: 'pointer', padding: '4px 2px', fontFamily: 'inherit',
                      }}
                      title="Jump to this moment"
                    >
                      <span style={{ fontSize: 11.5, color: 'var(--nc-accent)', fontVariantNumeric: 'tabular-nums', flexShrink: 0, marginTop: 2 }}>
                        {fmtDur(sg.start)}
                      </span>
                      <span style={{ fontSize: 13, color: 'var(--nc-text-dim)', lineHeight: 1.45 }}>{sg.text}</span>
                    </button>
                  ))}
                </div>
              ) : (
                <div style={{ fontSize: 12.5, color: 'var(--nc-text-faint)' }}>No transcript available for this clip.</div>
              )}
            </div>

            {/* actions */}
            <div>
              <div className="nc-label">Actions</div>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
                {ACTIONS.map(({ id, icon: Icon, label, fn, custom }) => (
                  custom ? (
                    <button
                      key={id}
                      className="nc-btn-ghost"
                      onClick={fn}
                      title={label}
                      style={{ justifyContent: 'flex-start', fontSize: 12.5, padding: '9px 12px' }}
                    >
                      <Icon size={14} />
                      <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{label}</span>
                    </button>
                  ) : actBtn(id, Icon, label, fn)
                ))}
              </div>
              {actMsg && (
                <div style={{ marginTop: 8, fontSize: 12.5, color: actMsg.ok ? 'var(--nc-green)' : 'var(--nc-red)', display: 'flex', gap: 6, alignItems: 'center' }}>
                  {actMsg.ok && <Check size={13} />}{actMsg.text}
                </div>
              )}

              {showPublish && (
                <div className="nc-anim-fade-in" style={{ marginTop: 10, padding: 12, border: '1px solid var(--nc-border)', borderRadius: 12, background: 'var(--nc-card)' }}>
                  <div className="nc-label">Publish via connected accounts</div>
                  <select className="nc-select" value={scheduleMode} onChange={(e) => setScheduleMode(e.target.value)} style={{ width: '100%', marginBottom: 10 }}>
                    <option value="now">Publish now</option>
                    <option value="auto">Auto-schedule (best time)</option>
                    <option value="manual">Schedule for a specific time</option>
                  </select>
                  {scheduleMode === 'manual' && (
                    <input
                      type="datetime-local"
                      className="nc-input"
                      value={manualWhen}
                      onChange={(e) => setManualWhen(e.target.value)}
                      style={{ width: '100%', marginBottom: 10 }}
                    />
                  )}
                  <button className="nc-btn-primary" style={{ width: '100%' }} disabled={publishing} onClick={doPublish}>
                    {publishing ? <Loader2 size={15} style={{ animation: 'nc-spin 1s linear infinite' }} /> : <Share2 size={15} />}
                    {publishing ? 'Publishing…' : 'Confirm publish'}
                  </button>
                  {publishDone && (
                    <div style={{ marginTop: 8, fontSize: 12.5, color: publishDone.ok ? 'var(--nc-green)' : 'var(--nc-red)', display: 'flex', gap: 6, alignItems: 'center' }}>
                      {publishDone.ok && <Check size={13} />}{publishDone.text}
                    </div>
                  )}
                </div>
              )}

              {aiOpen && (
                <div className="nc-anim-fade-in" style={{ marginTop: 10, padding: 12, border: '1px solid var(--nc-border)', borderRadius: 12, background: 'var(--nc-card)' }}>
                  <div className="nc-label">Quick AI trim</div>
                  <div style={{ display: 'flex', gap: 8 }}>
                    <input
                      className="nc-input" style={{ flex: 1 }}
                      placeholder='e.g. "cut the silent intro"'
                      value={aiPrompt} onChange={(e) => setAiPrompt(e.target.value)}
                      onKeyDown={(e) => { if (e.key === 'Enter') doAiTrim(); }}
                    />
                    <button className="nc-btn-primary" disabled={aiBusy || !aiPrompt.trim()} onClick={doAiTrim} style={{ padding: '9px 14px' }}>
                      {aiBusy ? <Loader2 size={15} style={{ animation: 'nc-spin 1s linear infinite' }} /> : 'Go'}
                    </button>
                  </div>
                  {aiResult && (
                    <div style={{ marginTop: 8, fontSize: 12.5, color: aiResult.ok ? 'var(--nc-green)' : 'var(--nc-red)' }}>{aiResult.text}</div>
                  )}
                  <div style={{ marginTop: 8, fontSize: 12, color: 'var(--nc-text-faint)' }}>
                    Deeper edits live in the full editor's copilot.
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
