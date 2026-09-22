import { memo, useCallback, useMemo, useState } from 'react';
import { Icon, Btn, Badge, RingGauge, PebbleWaveform } from './primitives';
import { LazyVideo } from './LazyVideo';
import { clipPreviewSrc, fmtDuration, downloadClip, exportClip } from './realApi';
import { PlatformMockupOverlay, PlatformCaptionSection } from './PlatformMockupOverlay';

const REFRAME_ICON = { auto: 'crop', subject: 'scan-face', object: 'scan-face', disabled: 'square' };
const REFRAME_LABEL = { auto: 'Auto', subject: 'Subject', object: 'Subject', disabled: 'Off' };
const delay = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

function getTier(score) {
  if (score >= 90) return { name: 'Legendary', cls: 'legendary', accent: 'var(--gold)' };
  if (score >= 80) return { name: 'Epic', cls: 'epic', accent: 'var(--plum)' };
  if (score >= 70) return { name: 'Rare', cls: 'rare', accent: 'var(--sage)' };
  return { name: 'Common', cls: 'common', accent: 'var(--grey)' };
}

const ClipCard = memo(function ClipCard({
  clip,
  index,
  jobId,
  state,
  preselections,
  onUpdate,
  onEdit,
  onApplyToAll,
  selectMode,
  onPublish,
  pushToast,
}) {
  const [downloading, setDownloading] = useState(false);
  const [activePlatform, setActivePlatform] = useState('tiktok');
  const [showMockup, setShowMockup] = useState(false);
  const selected = state?.selected !== false;
  const score = Math.round(clip.viral_score || 0);
  const tier = getTier(score);
  const mode = state?.reframeMode || clip.reframe_mode || 'auto';
  const title = clip.video_title_for_youtube_short || `Clip ${index + 1}`;
  const processing = !!state?.processing;

  const selectionProps = selectMode
    ? {
        role: 'checkbox',
        tabIndex: 0,
        'aria-checked': selected,
        onClick: () => onUpdate(index, { selected: !selected }),
        onKeyDown: (event) => {
          if (event.key === 'Enter' || event.key === ' ') {
            event.preventDefault();
            onUpdate(index, { selected: !selected });
          }
        },
      }
    : {};

  const doDownload = async (event) => {
    event.stopPropagation();
    if (downloading || processing) return;
    setDownloading(true);
    try {
      const kind = await exportClip(jobId, index, clip, state, preselections);
      pushToast?.('success', kind === 'composed' ? 'Composed clip downloaded' : 'Clip downloaded');
    } catch {
      pushToast?.('warn', 'Compose failed; downloading the raw clip instead');
      downloadClip(clip, index);
    } finally {
      setDownloading(false);
    }
  };

  return (
    <article {...selectionProps} className={`card raised ${selectMode && selected ? 'pressed-md' : ''}`}>
      <div className="card-top">
        <span className={`tier-badge ${tier.cls}`}>{tier.name}</span>
        <span className="timestamp">{fmtDuration(clip.start, clip.end)}</span>
      </div>

      <div className="clip-media">
        <LazyVideo
          src={clipPreviewSrc(clip, state)}
          controls={!selectMode}
          playsInline
          muted={selectMode}
          aria-label={`Preview ${title}`}
          style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', objectFit: 'cover', zIndex: 0 }}
        />
        {showMockup && (
          <PlatformMockupOverlay
            platform={activePlatform}
            clip={clip}
            speakerName={clip.speaker_name}
            channel={clip.channel}
          />
        )}
        <div className="clip-top">
          <span className="score">
            <Icon n="flame" />
            {score}
          </span>
          {selectMode ? (
            <span className="clip-check" aria-hidden="true">
              <Icon n="check" />
            </span>
          ) : (
            <span className="rf-badge" title={`Reframe: ${REFRAME_LABEL[mode] || mode}`}>
              <Icon n={REFRAME_ICON[mode] || 'crop'} />
              {REFRAME_LABEL[mode] || 'Auto'}
            </span>
          )}
        </div>
        <div className="clip-bottom">
          {state?.publishedAt && <span className="clip-pub"><Icon n="check" />published</span>}
          {(clip.analytics || state?.analytics) && (
            <span
              className="clip-pub"
              style={{
                background: 'rgba(50, 160, 120, 0.25)',
                color: 'var(--ink)',
                borderColor: 'rgba(50, 160, 120, 0.4)',
              }}
              title={`${((clip.analytics || state?.analytics).views || 0).toLocaleString()} views · ${((clip.analytics || state?.analytics).shares || 0).toLocaleString()} shares`}
            >
              <Icon n="bar-chart" />
              {((clip.analytics || state?.analytics).views || 0).toLocaleString()} views
            </span>
          )}
        </div>
        {processing && (
          <div className="clip-busy" role="status">
            <Icon n="loader" />
            <span>Reprocessing…</span>
          </div>
        )}
      </div>

      <h4 title={title}>{title}</h4>

      {/* Clip Intelligence: Narrative Arc & Hook */}
      {(clip.viral_hook_text || clip.hook || clip.viral_reason || clip.duration_tier) && (
        <div
          style={{
            background: 'var(--surface-deep)',
            border: '1px solid rgba(51, 46, 38, 0.08)',
            borderRadius: 10,
            padding: '8px 12px',
            marginBottom: 10,
            fontSize: '11.5px',
            lineHeight: 1.45,
            color: 'var(--ink)',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 6, marginBottom: 4 }}>
            <span style={{ fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.4px', fontSize: '9.5px', color: 'var(--ink-dim)' }}>
              Clip Intelligence
            </span>
            {clip.duration_tier && (
              <span
                style={{
                  background: clip.duration_tier === 'mid' ? 'rgba(232, 101, 74, 0.14)' : (clip.duration_tier === 'extended' ? 'rgba(94, 127, 94, 0.16)' : 'rgba(51, 46, 38, 0.08)'),
                  color: clip.duration_tier === 'mid' ? 'var(--rust, #e8654a)' : (clip.duration_tier === 'extended' ? 'var(--moss, #5e7f5e)' : 'var(--ink)'),
                  padding: '1px 6px',
                  borderRadius: '4px',
                  fontWeight: 700,
                  fontSize: '9.5px',
                }}
              >
                {clip.duration_tier === 'mid' ? 'Story Arc' : clip.duration_tier === 'extended' ? 'Deep Dive' : 'Short'}
              </span>
            )}
          </div>

          {(clip.viral_hook_text || clip.hook) && (
            <div style={{ marginBottom: 4, fontWeight: 600 }}>
              <span style={{ color: 'var(--rust, #e8654a)', marginRight: 4 }}>Hook:</span>
              “{clip.viral_hook_text || clip.hook}”
            </div>
          )}

          {clip.viral_reason && (
            <div style={{ color: 'var(--ink-dim)', fontSize: '11px' }}>
              <span style={{ fontWeight: 600, color: 'var(--ink)', marginRight: 4 }}>Why it works:</span>
              {clip.viral_reason}
            </div>
          )}
        </div>
      )}

      {/* Pebble Waveform Density Bar */}
      <PebbleWaveform count={16} activeCount={Math.round((score / 100) * 16)} />

      {/* Sub-Score Gauges */}
      <div className="card-bottom">
        <RingGauge score={score} max={100} size="sm" accent={tier.accent} label="Virality Score" />
        <div className="mini-scores" style={{ display: 'flex', gap: 12 }}>
          <div style={{ textAlign: 'center' }}>
            <RingGauge score={Math.min(100, Math.round(score * 1.02))} max={100} size="mini" accent={tier.accent} />
            <div style={{ fontFamily: 'var(--font-mono)', fontSize: 10, color: 'var(--ink-dim)', marginTop: 4, fontWeight: 700 }}>HOOK</div>
          </div>
          <div style={{ textAlign: 'center' }}>
            <RingGauge score={Math.min(100, Math.round(score * 0.96))} max={100} size="mini" accent={tier.accent} />
            <div style={{ fontFamily: 'var(--font-mono)', fontSize: 10, color: 'var(--ink-dim)', marginTop: 4, fontWeight: 700 }}>COH.</div>
          </div>
          <div style={{ textAlign: 'center' }}>
            <RingGauge score={Math.min(100, Math.round(score * 0.98))} max={100} size="mini" accent={tier.accent} />
            <div style={{ fontFamily: 'var(--font-mono)', fontSize: 10, color: 'var(--ink-dim)', marginTop: 4, fontWeight: 700 }}>PAY.</div>
          </div>
        </div>
      </div>

      {/* Platform Captions & Situational Awareness */}
      {!selectMode && (
        <PlatformCaptionSection
          clip={clip}
          activePlatform={activePlatform}
          onChangePlatform={setActivePlatform}
          showMockup={showMockup}
          onToggleMockup={(e) => {
            e.stopPropagation();
            setShowMockup((prev) => !prev);
          }}
          pushToast={pushToast}
        />
      )}

      {!selectMode && (
        <button
          type="button"
          className="clip-edit"
          disabled={processing}
          onClick={(event) => {
            event.stopPropagation();
            onEdit(clip, index);
          }}
          aria-label={`Edit and reprocess ${title}`}
        >
          <Icon n={processing ? 'loader' : 'sliders-horizontal'} />
          {processing ? 'Reprocessing…' : 'Edit & Reprocess'}
        </button>
      )}

      {/* Actions */}
      {!selectMode && (
        <div className="clip-foot">
          <button
            type="button"
            className="mini"
            title="Apply these settings to all clips"
            aria-label="Apply settings to all clips"
            disabled={processing}
            onClick={(event) => {
              event.stopPropagation();
              if (!processing && window.confirm("Apply this clip's settings to every other clip?")) {
                onApplyToAll(index);
              }
            }}
          >
            <Icon n="copy" />
          </button>
          <button
            type="button"
            className="mini"
            title="Download with edits"
            aria-label={`Download ${title}`}
            disabled={downloading || processing}
            onClick={doDownload}
          >
            <Icon n={downloading ? 'loader' : 'download'} />
          </button>
          <button
            type="button"
            className="mini"
            title="Publish to socials"
            aria-label={`Publish ${title}`}
            disabled={processing}
            onClick={(event) => {
              event.stopPropagation();
              onPublish({ ...clip, _idx: index, _apiIdx: clip.original_index ?? index });
            }}
          >
            <Icon n="send" />
          </button>
          <button
            type="button"
            className="mini"
            title="Remove from grid"
            aria-label={`Remove ${title}`}
            onClick={(event) => {
              event.stopPropagation();
              if (window.confirm('Remove this clip from the grid? The file stays on disk.')) {
                onUpdate(index, { deleted: true });
                pushToast?.('info', 'Clip removed');
              }
            }}
          >
            <Icon n="trash-2" />
          </button>
        </div>
      )}
    </article>
  );
});

export function ResultsView({
  clips,
  jobId,
  preselections,
  clipStates = {},
  onUpdateClipState,
  doneIn,
  onBack,
  onPublish,
  onPublishAll,
  onEdit,
  onApplyToAll,
  onEditSelected,
  embedded,
  pushToast,
}) {
  const [selectMode, setSelectMode] = useState(false);
  const [exporting, setExporting] = useState(false);
  const visible = useMemo(
    () => clips.map((clip, index) => ({ c: clip, i: index })).filter(({ i }) => !clipStates[i]?.deleted),
    [clips, clipStates]
  );
  const selected = useMemo(() => visible.filter(({ i }) => clipStates[i]?.selected !== false), [visible, clipStates]);
  const topScore = useMemo(() => (visible.length ? Math.max(...visible.map(({ c }) => Math.round(c.viral_score || 0))) : 0), [visible]);
  const allSelected = visible.length > 0 && selected.length === visible.length;

  const setSelectedAll = useCallback(
    (value) => visible.forEach(({ i }) => onUpdateClipState(i, { selected: value })),
    [visible, onUpdateClipState]
  );
  const publishMany = useCallback(
    (list) => onPublishAll(list.map(({ c, i }) => ({ ...c, _idx: i, _apiIdx: c.original_index ?? i }))),
    [onPublishAll]
  );
  const exportMany = async (list) => {
    if (exporting || !list.length) return;
    setExporting(true);
    let composed = 0;
    let rawFallback = 0;
    try {
      for (const { c, i } of list) {
        try {
          await exportClip(jobId, i, c, clipStates[i], preselections);
          composed += 1;
        } catch {
          downloadClip(c, i);
          rawFallback += 1;
        }
        await delay(150);
      }
      pushToast?.(
        rawFallback ? 'warn' : 'success',
        rawFallback
          ? `Exported ${list.length} clips; ${rawFallback} used the raw fallback`
          : `Exported ${composed}/${list.length} clips`
      );
    } finally {
      setExporting(false);
    }
  };

  return (
    <main className="container fade-in">
      <div className="section-head">
        <div style={{ display: 'flex', alignItems: 'center', gap: 14 }}>
          {!embedded && <Btn variant="ghost" size="sm" icon="arrow-left" onClick={onBack} title="Start over" aria-label="Start over" />}
          <div>
            <div className="eyebrow">Ranked Moments</div>
            <h2>{visible.length} Scored Clips Ready</h2>
          </div>
          {doneIn && <Badge tone="teal" icon="check">done in {doneIn}</Badge>}
        </div>

        <div style={{ display: 'flex', gap: 10 }}>
          <Btn
            variant="ghost"
            size="sm"
            icon={selectMode ? 'x' : 'check-square'}
            onClick={() =>
              setSelectMode((current) => {
                if (!current) visible.forEach(({ i }) => onUpdateClipState(i, { selected: false }));
                return !current;
              })
            }
          >
            {selectMode ? 'Cancel' : 'Select'}
          </Btn>
          {!selectMode && (
            <Btn variant="ghost" size="sm" icon="download" loading={exporting} disabled={!visible.length} onClick={() => exportMany(visible)}>
              {exporting ? 'Exporting…' : 'Export all'}
            </Btn>
          )}
          {!selectMode && (
            <Btn variant="primary" size="sm" icon="send" disabled={!visible.length} onClick={() => publishMany(visible)}>
              Publish all
            </Btn>
          )}
        </div>
      </div>

      <div style={{ fontFamily: 'var(--font-mono)', fontSize: 13, color: 'var(--ink-dim)', marginBottom: 28 }}>
        Sorted by virality score · top moment <b>{topScore}/100</b>
      </div>

      {selectMode && (
        <div
          role="toolbar"
          aria-label="Selected clip actions"
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: 12,
            background: 'var(--surface)',
            boxShadow: 'var(--clay-raised-md)',
            borderRadius: 'var(--r-panel)',
            padding: '14px 20px',
            marginBottom: 28,
            flexWrap: 'wrap',
          }}
        >
          <span style={{ fontFamily: 'var(--font-mono)', fontWeight: 700, fontSize: 13 }} aria-live="polite">
            {selected.length} selected
          </span>
          <Btn variant="ghost" size="sm" icon="check-check" onClick={() => setSelectedAll(!allSelected)}>
            {allSelected ? 'Deselect all' : 'Select all'}
          </Btn>
          <div style={{ marginLeft: 'auto', display: 'flex', gap: 8 }}>
            <Btn variant="ghost" size="sm" icon="sliders-horizontal" disabled={!selected.length} onClick={() => onEditSelected(selected)}>
              Edit {selected.length || ''}
            </Btn>
            <Btn variant="ghost" size="sm" icon="download" loading={exporting} disabled={!selected.length} onClick={() => exportMany(selected)}>
              {exporting ? 'Exporting…' : 'Export'}
            </Btn>
            <Btn variant="primary" size="sm" icon="send" disabled={!selected.length} onClick={() => publishMany(selected)}>
              Publish {selected.length || ''}
            </Btn>
          </div>
        </div>
      )}

      {visible.length ? (
        <div className="grid" role="list" aria-label="Generated clips">
          {visible.map(({ c, i }) => (
            <ClipCard
              key={c.original_index ?? i}
              clip={c}
              index={i}
              jobId={jobId}
              state={clipStates[i]}
              preselections={preselections}
              onUpdate={onUpdateClipState}
              selectMode={selectMode}
              onPublish={onPublish}
              onEdit={onEdit}
              onApplyToAll={onApplyToAll}
              pushToast={pushToast}
            />
          ))}
        </div>
      ) : (
        <div className="pod raised" role="status">
          <div className="pod-inner pressed">
            <div className="up-icon"><Icon n="film" /></div>
            <h3>No visible clips</h3>
            <p>All clips were removed from this view. Start over to generate a new set.</p>
          </div>
        </div>
      )}
    </main>
  );
}
