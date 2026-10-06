// clipEditor.jsx — Phase 3a: dedicated clip editor page (MVP).
//
// Edits a versioned clip-project.json: trim (+split), caption text/style/
// position, hook text/style — with an instant <video> + DOM-overlay preview.
// The server renders final pixels on Export (POST /api/compose {project}).
// Undo = load an earlier version into the draft, then save/export.
//
// Preview honesty: the <video> plays the master/composed clip (source-slice
// timeline, 1:1 with project source coords); captions/hooks render as DOM
// overlays synced to the playhead. Export is the pixel-exact check.

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Icon, Btn } from './primitives';
import { SUBTITLE_PRESETS } from './data';
import {
  getClipProject, saveClipProject, composeProject, backfillClipProjects,
  listClipProjectVersions, editClipAI, safeResolveUrl, clipPreviewSrc,
  uploadAudio,
} from './realApi';

// ---------------------------------------------------------------- helpers

const fmtT = (s) => {
  if (!isFinite(s) || s < 0) return '0:00.0';
  const m = Math.floor(s / 60);
  const sec = (s - m * 60).toFixed(1).padStart(4, '0');
  return `${m}:${sec}`;
};
const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
const r2 = (v) => Math.round(v * 100) / 100;
const newKey = () => {
  try {
    if (typeof crypto !== 'undefined' && crypto.randomUUID) return crypto.randomUUID();
  } catch { /* fall through */ }
  return `${Date.now()}-${Math.random().toString(36).slice(2)}`;
};

// Complement of drop ranges inside [0, duration] → kept spans.
function keptSpans(duration, drops) {
  const ds = (drops || []).filter((d) => d && d[1] > d[0]).sort((a, b) => a[0] - b[0]);
  const spans = [];
  let cur = 0;
  for (const [s, e] of ds) {
    if (s > cur) spans.push([cur, Math.min(s, duration)]);
    cur = Math.max(cur, e);
  }
  if (cur < duration) spans.push([cur, duration]);
  return spans.filter(([s, e]) => e - s > 0.01);
}

// Center 9:16 crop box in source pixels (fallback when a segment has no crop).
function defaultCrop(p) {
  const W = (p && p.source && p.source.width) || 1920;
  const H = (p && p.source && p.source.height) || 1080;
  const w = Math.round((H * 9) / 16);
  return { x: Math.round((W - w) / 2), y: 0, w, h: H };
}

// Word active at time t (with text edits applied). Returns {i, text, edited}.
function wordAt(words, edits, t) {
  for (let i = 0; i < words.length; i++) {
    const w = words[i];
    if (t >= w.start && t < w.end) {
      const over = edits ? edits[String(i)] : undefined;
      return { i, text: over !== undefined ? over : w.w, edited: over !== undefined };
    }
  }
  return null;
}

function validateDraft(p) {
  const errs = [];
  const dur = (p && p.source && p.source.duration) || 0;
  const segs = (p && p.segments) || [];
  if (!segs.length) errs.push('At least one segment is required.');
  segs.forEach((s, i) => {
    if (!(s.end > s.start)) errs.push(`Segment ${i + 1}: end must be after start.`);
    if (s.start < 0 || (dur > 0 && s.end > dur + 0.01))
      errs.push(`Segment ${i + 1}: out of range (0 – ${fmtT(dur)}).`);
  });
  return errs;
}

const CAP_POS_STYLE = {
  bottom: { left: '6%', right: '6%', bottom: '8%' },
  top: { left: '6%', right: '6%', top: '11%' },
  middle: { left: '6%', right: '6%', top: '42%' },
};

function presetCss(id) {
  const p = SUBTITLE_PRESETS.find((x) => x.id === id);
  return (p && p.style) || SUBTITLE_PRESETS[0].style;
}

const inputStyle = {
  background: 'var(--surface-deep, #f4f1ea)',
  border: '1px solid rgba(51,46,38,.18)',
  borderRadius: 8,
  padding: '6px 10px',
  fontSize: 13,
  color: 'var(--ink, #222)',
  width: '100%',
  boxSizing: 'border-box',
};

function Field({ label, children }) {
  return (
    <label style={{ display: 'block', marginBottom: 12 }}>
      <span style={{ display: 'block', fontSize: 11, fontWeight: 700, textTransform: 'uppercase',
        letterSpacing: '.5px', color: 'var(--ink-dim, #666)', marginBottom: 6 }}>{label}</span>
      {children}
    </label>
  );
}

// ------------------------------------------------------- timeline + scrubber

function Timeline({ duration, segments, selSeg, onSelectSeg, onSeek, onTrim, playhead, onPlayhead }) {
  const trackRef = useRef(null);
  const dragRef = useRef(null);

  const pct = (t) => `${clamp((t / duration) * 100, 0, 100)}%`;

  const beginDrag = (e, edge) => {
    if (!segments[selSeg]) return;
    e.preventDefault();
    e.stopPropagation();
    const rect = trackRef.current.getBoundingClientRect();
    const orig = { start: segments[selSeg].start, end: segments[selSeg].end };
    dragRef.current = { edge, startX: e.clientX, rectW: rect.width, orig };
    const move = (ev) => {
      const d = dragRef.current;
      if (!d) return;
      const dt = ((ev.clientX - d.startX) / d.rectW) * duration;
      if (d.edge === 'l') onTrim(clamp(r2(d.orig.start + dt), 0, r2(d.orig.end - 0.2)), d.orig.end);
      else onTrim(d.orig.start, clamp(r2(d.orig.end + dt), r2(d.orig.start + 0.2), r2(duration)));
    };
    const up = () => {
      dragRef.current = null;
      window.removeEventListener('pointermove', move);
      window.removeEventListener('pointerup', up);
    };
    window.addEventListener('pointermove', move);
    window.addEventListener('pointerup', up);
  };

  const clickSeek = (e) => {
    if (dragRef.current) return;
    const rect = trackRef.current.getBoundingClientRect();
    const t = clamp(((e.clientX - rect.left) / rect.width) * duration, 0, duration);
    onSeek(t);
    if (onPlayhead) onPlayhead(t);
  };

  return (
    <div style={{ padding: '10px 2px 2px' }}>
      <div
        ref={trackRef}
        onClick={clickSeek}
        style={{ position: 'relative', height: 56, background: 'rgba(51,46,38,.08)',
          borderRadius: 10, cursor: 'pointer', overflow: 'hidden' }}
        role="slider" aria-label="Timeline scrubber" tabIndex={0}
        onKeyDown={(e) => {
          if (e.key === 'ArrowLeft') { onSeek(clamp(playhead - 1, 0, duration)); }
          if (e.key === 'ArrowRight') { onSeek(clamp(playhead + 1, 0, duration)); }
        }}
      >
        {segments.map((s, i) => {
          const sel = i === selSeg;
          return (
            <div
              key={s.id || i}
              onClick={(e) => { e.stopPropagation(); onSelectSeg(i); }}
              title={`Segment ${i + 1}: ${fmtT(s.start)} – ${fmtT(s.end)}`}
              style={{ position: 'absolute', top: 8, bottom: 8, left: pct(s.start),
                width: `calc(${pct(s.end - s.start)} - 0px)`, minWidth: 4,
                background: sel ? 'rgba(103,90,221,.55)' : 'rgba(103,90,221,.28)',
                border: sel ? '1.5px solid var(--plum, #675ADD)' : '1px solid rgba(103,90,221,.4)',
                borderRadius: 6, cursor: 'pointer' }}
            >
              {sel && (
                <>
                  <div
                    onPointerDown={(e) => beginDrag(e, 'l')}
                    style={{ position: 'absolute', left: -5, top: -4, bottom: -4, width: 12,
                      cursor: 'ew-resize', display: 'flex', alignItems: 'center', justifyContent: 'center' }}
                    title="Drag to trim start"
                  >
                    <div style={{ width: 5, height: '70%', borderRadius: 3, background: '#fff',
                      boxShadow: '0 1px 4px rgba(0,0,0,.35)' }} />
                  </div>
                  <div
                    onPointerDown={(e) => beginDrag(e, 'r')}
                    style={{ position: 'absolute', right: -5, top: -4, bottom: -4, width: 12,
                      cursor: 'ew-resize', display: 'flex', alignItems: 'center', justifyContent: 'center' }}
                    title="Drag to trim end"
                  >
                    <div style={{ width: 5, height: '70%', borderRadius: 3, background: '#fff',
                      boxShadow: '0 1px 4px rgba(0,0,0,.35)' }} />
                  </div>
                </>
              )}
            </div>
          );
        })}
        {/* playhead */}
        <div style={{ position: 'absolute', top: 0, bottom: 0, left: pct(playhead), width: 2,
          background: '#e6428d', pointerEvents: 'none', zIndex: 3 }}>
          <div style={{ position: 'absolute', top: -1, left: -5, width: 12, height: 12, borderRadius: '50%',
            background: '#e6428d' }} />
        </div>
      </div>
      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11,
        color: 'var(--ink-dim, #666)', fontFamily: 'var(--font-mono, monospace)', marginTop: 4 }}>
        <span>{fmtT(playhead)}</span>
        <span>{segments[selSeg] ? `${fmtT(segments[selSeg].start)} – ${fmtT(segments[selSeg].end)}` : ''}</span>
        <span>{fmtT(duration)}</span>
      </div>
    </div>
  );
}

// ------------------------------------------------------------------ panels

function TrimPanel({ project, selSeg, setSelSeg, onSegTimes, onSplit, onDeleteSeg,
  onMoveSeg, onTransition, onCropNudge, onCropXY, onCropReset, playhead }) {
  const segs = project.segments;
  const seg = segs[selSeg];
  const dur = project.source.duration || 0;
  const srcW = (project.source && project.source.width) || 1920;
  const srcH = (project.source && project.source.height) || 1080;
  if (!seg) return null;
  const num = (v, fn) => (
    <input type="number" step={0.1} min={0} max={dur} value={r2(v)}
      onChange={(e) => { const n = parseFloat(e.target.value); if (isFinite(n)) fn(n); }}
      style={{ ...inputStyle, width: 90, fontFamily: 'var(--font-mono, monospace)' }} />
  );
  const crop = seg.crop || defaultCrop(project);
  const arrowBtn = { width: 30, height: 30, borderRadius: 8, border: '1px solid rgba(51,46,38,.2)',
    background: '#fff', cursor: 'pointer', fontSize: 15, lineHeight: 1, color: 'var(--ink, #222)' };
  return (
    <div>
      <Field label={`Segments (${segs.length}) — order is the export order`}>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
          {segs.map((s, i) => (
            <div key={s.id || i}
              style={{ display: 'flex', alignItems: 'center', gap: 8, padding: '6px 8px',
                borderRadius: 10, cursor: 'pointer',
                border: i === selSeg ? '1.5px solid var(--plum, #675ADD)' : '1px solid rgba(51,46,38,.14)',
                background: i === selSeg ? 'rgba(103,90,221,.07)' : 'transparent' }}
              onClick={() => setSelSeg(i)} title="Select segment">
              <b style={{ fontFamily: 'var(--font-mono, monospace)', fontSize: 13 }}>#{i + 1}</b>
              <span style={{ fontFamily: 'var(--font-mono, monospace)', fontSize: 12,
                color: 'var(--ink, #222)' }}>
                {fmtT(s.start)} – {fmtT(s.end)} · {fmtT(s.end - s.start)}
              </span>
              <span style={{ flex: 1 }} />
              {i > 0 ? (
                <select value={s.transition_in || 'cut'} title="Transition into this segment"
                  onClick={(e) => e.stopPropagation()}
                  onChange={(e) => onTransition(i, e.target.value)}
                  style={{ ...inputStyle, width: 'auto', padding: '4px 6px', fontSize: 12 }}>
                  <option value="cut">Cut</option>
                  <option value="fade">Fade</option>
                </select>
              ) : (
                <span style={{ fontSize: 11, color: 'var(--ink-dim, #666)' }}>start</span>
              )}
              <button type="button" title="Move earlier" disabled={i === 0}
                onClick={(e) => { e.stopPropagation(); onMoveSeg(i, -1); }}
                style={{ width: 28, height: 28, borderRadius: 8, border: '1px solid rgba(51,46,38,.2)',
                  background: '#fff', cursor: 'pointer', fontSize: 14, color: 'var(--ink, #222)',
                  opacity: i === 0 ? 0.35 : 1 }}>&uarr;</button>
              <button type="button" title="Move later" disabled={i === segs.length - 1}
                onClick={(e) => { e.stopPropagation(); onMoveSeg(i, 1); }}
                style={{ width: 28, height: 28, borderRadius: 8, border: '1px solid rgba(51,46,38,.2)',
                  background: '#fff', cursor: 'pointer', fontSize: 14, color: 'var(--ink, #222)',
                  opacity: i === segs.length - 1 ? 0.35 : 1 }}>&darr;</button>
            </div>
          ))}
        </div>
      </Field>
      <div style={{ display: 'flex', gap: 12, alignItems: 'flex-end', flexWrap: 'wrap' }}>
        <Field label="Start (s)">
          {num(seg.start, (n) => onSegTimes(clamp(r2(n), 0, r2(seg.end - 0.2)), seg.end))}
        </Field>
        <Field label="End (s)">
          {num(seg.end, (n) => onSegTimes(seg.start, clamp(r2(n), r2(seg.start + 0.2), r2(dur))))}
        </Field>
        <div style={{ fontSize: 12, color: 'var(--ink-dim, #666)', paddingBottom: 16 }}>
          Duration <b style={{ fontFamily: 'var(--font-mono, monospace)' }}>{fmtT(seg.end - seg.start)}</b>
        </div>
      </div>
      <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
        <Btn variant="ghost" size="sm" icon="scissors" onClick={onSplit}
          title="Split the selected segment at the playhead">
          Split at {fmtT(playhead)}
        </Btn>
        {segs.length > 1 && (
          <Btn variant="ghost" size="sm" icon="trash-2" onClick={onDeleteSeg}>
            Delete segment
          </Btn>
        )}
      </div>
      <Field label={`Crop (9:16) — segment ${selSeg + 1}`}>
        <div style={{ display: 'flex', gap: 14, alignItems: 'center', flexWrap: 'wrap' }}>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 30px)', gap: 3 }}>
            <span />
            <button type="button" title="Nudge up" onClick={() => onCropNudge(0, -24)} style={arrowBtn}>&uarr;</button>
            <span />
            <button type="button" title="Nudge left" onClick={() => onCropNudge(-24, 0)} style={arrowBtn}>&larr;</button>
            <button type="button" title="Reset to center crop" onClick={onCropReset} style={{ ...arrowBtn, fontSize: 12 }}>&#9678;</button>
            <button type="button" title="Nudge right" onClick={() => onCropNudge(24, 0)} style={arrowBtn}>&rarr;</button>
            <span />
            <button type="button" title="Nudge down" onClick={() => onCropNudge(0, 24)} style={arrowBtn}>&darr;</button>
            <span />
          </div>
          <div style={{ display: 'flex', gap: 10, alignItems: 'flex-end' }}>
            <Field label="X">
              <input type="number" step={4} min={0} max={Math.max(0, srcW - crop.w)} value={Math.round(crop.x)}
                onChange={(e) => { const n = parseInt(e.target.value, 10); if (isFinite(n)) onCropXY(n, crop.y); }}
                style={{ ...inputStyle, width: 76, fontFamily: 'var(--font-mono, monospace)' }} />
            </Field>
            <Field label="Y">
              <input type="number" step={4} min={0} max={Math.max(0, srcH - crop.h)} value={Math.round(crop.y)}
                onChange={(e) => { const n = parseInt(e.target.value, 10); if (isFinite(n)) onCropXY(crop.x, n); }}
                style={{ ...inputStyle, width: 76, fontFamily: 'var(--font-mono, monospace)' }} />
            </Field>
          </div>
        </div>
        <div style={{ fontSize: 11, color: 'var(--ink-dim, #666)', marginTop: 6,
          fontFamily: 'var(--font-mono, monospace)' }}>
          {Math.round(crop.w)}×{Math.round(crop.h)} px of {srcW}×{srcH} source
        </div>
      </Field>
      <p style={{ fontSize: 11.5, color: 'var(--ink-dim, #666)', marginTop: 12, lineHeight: 1.5 }}>
        Drag the white handles on the timeline, or type exact times. Preview loops the
        selected segment so you hear exactly what the export will contain.
      </p>
    </div>
  );
}

// ---------------------------------------------------------- music panel
function MusicPanel({ project, uploading, onUpload, onVolume, onOffset, onRemove }) {
  const track = (project.audio && project.audio[0]) || {};
  if (!track.file) {
    return (
      <div>
        <Field label="Music track">
          <input type="file" accept="audio/*" disabled={uploading}
            onChange={(e) => { const f = e.target.files && e.target.files[0]; if (f) onUpload(f); e.target.value = ''; }}
            style={inputStyle} />
        </Field>
        {uploading && <p style={{ fontSize: 12.5, color: 'var(--ink-dim, #666)' }}>Uploading…</p>}
        <p style={{ fontSize: 11.5, color: 'var(--ink-dim, #666)', marginTop: 10, lineHeight: 1.5 }}>
          The track is mixed under the clip audio at export (mp3/wav/m4a/aac/ogg/flac,
          max 20MB). No track — the clip keeps its original audio.
        </p>
      </div>
    );
  }
  return (
    <div>
      <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 12,
        padding: '8px 10px', border: '1px solid rgba(51,46,38,.14)', borderRadius: 10 }}>
        <span style={{ fontSize: 13, fontWeight: 600, flex: 1, overflow: 'hidden',
          textOverflow: 'ellipsis', whiteSpace: 'nowrap' }} title={track.file}>
          {String(track.file).split('/').pop()}
        </span>
        <Btn variant="ghost" size="sm" icon="trash-2" onClick={onRemove}>Remove</Btn>
      </div>
      <Field label={`Volume (${Math.round((track.volume ?? 1) * 100)}%)`}>
        <input type="range" min={0} max={1.5} step={0.05} value={track.volume ?? 1}
          onChange={(e) => onVolume(parseFloat(e.target.value))}
          style={{ width: '100%' }} />
      </Field>
      <Field label="Offset (seconds)">
        <input type="number" step={0.5} min={0} value={r2(track.offset || 0)}
          onChange={(e) => { const n = parseFloat(e.target.value); if (isFinite(n) && n >= 0) onOffset(r2(n)); }}
          style={{ ...inputStyle, width: 110, fontFamily: 'var(--font-mono, monospace)' }} />
      </Field>
      <p style={{ fontSize: 11.5, color: 'var(--ink-dim, #666)', marginTop: 10, lineHeight: 1.5 }}>
        Mixed at export — the preview plays the original audio only.
      </p>
    </div>
  );
}

function CaptionsPanel({ project, onStyle, onPosition, onWordEdit, onClearEdits }) {
  const caps = project.captions || { style: 'classic_white', position: 'bottom', words: [], edits: {} };
  const words = caps.words || [];
  const edits = caps.edits || {};
  const [editing, setEditing] = useState(null);
  const [draft, setDraft] = useState('');
  const editedCount = Object.keys(edits).length;

  const commit = (i) => {
    const orig = words[i] ? words[i].w : '';
    onWordEdit(i, draft.trim() === '' ? orig : draft, orig);
    setEditing(null);
  };

  return (
    <div>
      <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap' }}>
        <Field label="Style">
          <select value={caps.style || 'classic_white'} onChange={(e) => onStyle(e.target.value)} style={inputStyle}>
            {SUBTITLE_PRESETS.map((p) => <option key={p.id} value={p.id}>{p.label}</option>)}
          </select>
        </Field>
        <Field label="Position">
          <select value={caps.position || 'bottom'} onChange={(e) => onPosition(e.target.value)} style={inputStyle}>
            <option value="top">Top</option>
            <option value="middle">Middle</option>
            <option value="bottom">Bottom</option>
          </select>
        </Field>
      </div>
      <Field label={`Words (${words.length})${editedCount ? ` · ${editedCount} corrected` : ''}`}>
        <div style={{ maxHeight: 220, overflowY: 'auto', border: '1px solid rgba(51,46,38,.14)',
          borderRadius: 10, padding: 8, display: 'flex', flexWrap: 'wrap', gap: 5,
          background: 'rgba(255,255,255,.5)' }}>
          {words.map((w, i) => {
            const over = edits[String(i)];
            const shown = over !== undefined ? over : w.w;
            const isEd = editing === i;
            return isEd ? (
              <input key={i} autoFocus value={draft}
                onChange={(e) => setDraft(e.target.value)}
                onBlur={() => commit(i)}
                onKeyDown={(e) => { if (e.key === 'Enter') commit(i); if (e.key === 'Escape') setEditing(null); }}
                style={{ ...inputStyle, width: 110, padding: '2px 6px', fontSize: 12 }} />
            ) : (
              <button key={i} type="button" title={`${fmtT(w.start)} – click to correct`}
                onClick={() => { setEditing(i); setDraft(shown); }}
                style={{ fontSize: 12, padding: '3px 8px', borderRadius: 999, cursor: 'pointer',
                  border: over !== undefined ? '1.5px solid var(--gold, #c90)' : '1px solid rgba(51,46,38,.2)',
                  background: over !== undefined ? 'rgba(247,188,89,.25)' : 'transparent',
                  color: 'var(--ink, #222)' }}>
                {shown}
              </button>
            );
          })}
          {!words.length && <span style={{ fontSize: 12, color: 'var(--ink-dim, #666)' }}>No words in this project.</span>}
        </div>
      </Field>
      {editedCount > 0 && (
        <Btn variant="ghost" size="sm" onClick={onClearEdits}>Clear {editedCount} correction{editedCount === 1 ? '' : 's'}</Btn>
      )}
    </div>
  );
}

function HookPanel({ project, onHookChange, onAddHook, onRemoveHook }) {
  const hook = (project.overlays || []).find((o) => o.type === 'hook');
  if (!hook) {
    return (
      <div>
        <p style={{ fontSize: 13, color: 'var(--ink-dim, #666)' }}>No hook overlay on this clip.</p>
        <Btn variant="ghost" size="sm" icon="plus" onClick={onAddHook}>Add hook</Btn>
      </div>
    );
  }
  const dur = project.source.duration || 0;
  return (
    <div>
      <Field label="Hook text">
        <input value={hook.text || ''} onChange={(e) => onHookChange({ text: e.target.value })}
          style={inputStyle} placeholder="The opener line…" />
      </Field>
      <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap' }}>
        <Field label="Style">
          <select value={hook.style || 'viral'} onChange={(e) => onHookChange({ style: e.target.value })} style={inputStyle}>
            <option value="viral">Viral (persistent top)</option>
            <option value="classic">Classic</option>
          </select>
        </Field>
        <Field label="Start (s)">
          <input type="number" step={0.1} min={0} max={dur} value={r2(hook.start || 0)}
            onChange={(e) => { const n = parseFloat(e.target.value); if (isFinite(n)) onHookChange({ start: clamp(r2(n), 0, dur) }); }}
            style={{ ...inputStyle, width: 90, fontFamily: 'var(--font-mono, monospace)' }} />
        </Field>
        <Field label="End (s)">
          <input type="number" step={0.1} min={0} max={dur} value={r2(hook.end || 0)}
            onChange={(e) => { const n = parseFloat(e.target.value); if (isFinite(n)) onHookChange({ end: clamp(r2(n), 0, dur) }); }}
            style={{ ...inputStyle, width: 90, fontFamily: 'var(--font-mono, monospace)' }} />
        </Field>
      </div>
      <Btn variant="ghost" size="sm" icon="trash-2" onClick={onRemoveHook}>Remove hook</Btn>
    </div>
  );
}

function AiTrimPanel({ aiText, setAiText, aiBusy, aiMsg, onAsk }) {
  return (
    <div>
      <Field label="Describe what to cut">
        <textarea value={aiText} onChange={(e) => setAiText(e.target.value)}
          placeholder='e.g. "cut the intro" or "remove the sponsorship read"'
          rows={3} style={{ ...inputStyle, resize: 'vertical' }} />
      </Field>
      <Btn variant="primary" size="sm" icon="wand-sparkles" loading={aiBusy}
        disabled={!aiText.trim() || aiBusy} onClick={onAsk}>
        {aiBusy ? 'Thinking…' : 'Cut with AI'}
      </Btn>
      {aiMsg && <p style={{ fontSize: 12.5, color: 'var(--ink-dim, #666)', marginTop: 10, lineHeight: 1.5 }}>{aiMsg}</p>}
      <p style={{ fontSize: 11.5, color: 'var(--ink-dim, #666)', marginTop: 10, lineHeight: 1.5 }}>
        The AI returns spans to cut; the editor converts them into segments. Review the
        timeline before exporting — nothing is final until you save.
      </p>
    </div>
  );
}

// ---------------------------------------------------------- versions panel

function VersionsPanel({ versions, currentVersion, onLoad, onPreviewVersion, loading }) {
  return (
    <div>
      <div style={{ fontSize: 11, fontWeight: 700, textTransform: 'uppercase', letterSpacing: '.5px',
        color: 'var(--ink-dim, #666)', marginBottom: 10, display: 'flex', alignItems: 'center', gap: 6 }}>
        <Icon n="clock" /> Version history
      </div>
      {loading && <div style={{ fontSize: 12, color: 'var(--ink-dim, #666)' }}>Loading…</div>}
      {!loading && !versions.length && (
        <div style={{ fontSize: 12, color: 'var(--ink-dim, #666)' }}>No saved versions yet.</div>
      )}
      <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
        {versions.map((v) => {
          const isCur = v.version === currentVersion;
          return (
            <div key={v.version} style={{ border: '1px solid rgba(51,46,38,.14)', borderRadius: 10,
              padding: '8px 10px', background: isCur ? 'rgba(103,90,221,.08)' : 'transparent' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4 }}>
                <b style={{ fontFamily: 'var(--font-mono, monospace)', fontSize: 13 }}>v{v.version}</b>
                <span style={{ fontSize: 10.5, fontWeight: 700, padding: '2px 8px', borderRadius: 999,
                  background: v.origin === 'auto' ? 'rgba(50,160,120,.18)' : 'rgba(103,90,221,.18)',
                  color: 'var(--ink, #222)' }}>
                  {v.origin === 'auto' ? 'AI cut' : v.origin === 'legacy' ? 'Legacy' : 'Edited'}
                </span>
                {isCur && <span style={{ fontSize: 10.5, color: 'var(--ink-dim, #666)' }}>· current draft</span>}
              </div>
              <div style={{ fontSize: 11, color: 'var(--ink-dim, #666)', marginBottom: 6,
                fontFamily: 'var(--font-mono, monospace)' }}>
                {v.created_at ? v.created_at.replace('T', ' ') : '—'}
                {v.output_file ? ` · ${v.output_file}` : ' · draft (not rendered)'}
              </div>
              <div style={{ display: 'flex', gap: 6 }}>
                <button type="button" onClick={() => onLoad(v.version)}
                  style={{ fontSize: 12, fontWeight: 700, padding: '4px 10px', borderRadius: 8, cursor: 'pointer',
                    border: '1px solid rgba(51,46,38,.25)', background: '#fff', color: 'var(--ink, #222)' }}>
                  Load
                </button>
                {v.output_url && (
                  <button type="button" onClick={() => onPreviewVersion(v)}
                    style={{ fontSize: 12, padding: '4px 10px', borderRadius: 8, cursor: 'pointer',
                      border: '1px solid rgba(51,46,38,.15)', background: 'transparent', color: 'var(--ink-dim, #666)' }}>
                    Watch render
                  </button>
                )}
              </div>
            </div>
          );
        })}
      </div>
      <p style={{ fontSize: 11.5, color: 'var(--ink-dim, #666)', marginTop: 10, lineHeight: 1.5 }}>
        Undo = load an older version, then save or export.
      </p>
    </div>
  );
}

// ---------------------------------------------------------- main editor view

export function ClipEditorView({ jobId, clipIndex, clip, onBack, pushToast }) {
  const [status, setStatus] = useState('loading'); // loading | ready | error
  const [loadError, setLoadError] = useState('');
  const [project, setProject] = useState(null);
  const [savedVersion, setSavedVersion] = useState(null);
  const [dirty, setDirty] = useState(false);
  const [versions, setVersions] = useState([]);
  const [versionsLoading, setVersionsLoading] = useState(false);
  const [videoSrc, setVideoSrc] = useState('');
  const [playhead, setPlayhead] = useState(0);
  const [selSeg, setSelSeg] = useState(0);
  const [loopSeg, setLoopSeg] = useState(true);
  const [panel, setPanel] = useState('trim');
  const [saving, setSaving] = useState(false);
  const [exporting, setExporting] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [aiText, setAiText] = useState('');
  const [aiBusy, setAiBusy] = useState(false);
  const [aiMsg, setAiMsg] = useState('');
  const videoRef = useRef(null);

  const title = (clip && (clip.video_title_for_youtube_short || clip.title)) || `Clip ${clipIndex + 1}`;

  const refreshVersions = useCallback(async () => {
    setVersionsLoading(true);
    try {
      const r = await listClipProjectVersions(jobId, clipIndex);
      setVersions(r.versions || []);
    } catch { /* keep previous list */ }
    finally { setVersionsLoading(false); }
  }, [jobId, clipIndex]);

  // Load: latest project; backfill first if this clip has none (old job).
  useEffect(() => {
    let alive = true;
    (async () => {
      try {
        let proj;
        try {
          proj = await getClipProject(jobId, clipIndex);
        } catch (e) {
          if (e && e.status === 404) {
            await backfillClipProjects(jobId);
            proj = await getClipProject(jobId, clipIndex);
          } else { throw e; }
        }
        if (!alive) return;
        setProject(proj);
        setSavedVersion(proj.version);
        setVideoSrc(clipPreviewSrc(clip, null));
        setStatus('ready');
        refreshVersions();
      } catch (e) {
        if (!alive) return;
        setLoadError((e && e.message) || 'Could not load the clip project.');
        setStatus('error');
      }
    })();
    return () => { alive = false; };
    // clip is a stable prop from the results list; intentionally not a dep.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [jobId, clipIndex]);

  // Keep the selected segment in range when segments change.
  useEffect(() => {
    if (project && selSeg >= project.segments.length) setSelSeg(0);
  }, [project, selSeg]);

  // Every user edit: deep-mutate the draft, mark origin=user, mint a fresh
  // idempotency key (so the next export is a NEW render, never a dedup hit).
  const mutate = useCallback((fn) => {
    setProject((prev) => {
      if (!prev) return prev;
      const next = JSON.parse(JSON.stringify(prev));
      fn(next);
      next.origin = 'user';
      next.idempotency_key = newKey();
      return next;
    });
    setDirty(true);
  }, []);

  const duration = (project && project.source && project.source.duration) || 0;
  const segs = (project && project.segments) || [];
  const seg = segs[selSeg];
  const caps = (project && project.captions) || {};
  const words = caps.words || [];
  const edits = caps.edits || {};
  const hook = ((project && project.overlays) || []).find((o) => o.type === 'hook');
  const activeWord = useMemo(() => wordAt(words, edits, playhead), [words, edits, playhead]);

  // --- preview playback ------------------------------------------------
  const onTimeUpdate = () => {
    const v = videoRef.current;
    if (!v) return;
    const t = v.currentTime;
    setPlayhead(t);
    const s = segs[selSeg];
    if (loopSeg && s && !v.paused && t >= s.end) v.currentTime = s.start;
  };
  const onSeek = (t) => {
    const v = videoRef.current;
    if (v) { try { v.currentTime = t; } catch { /* ignore */ } }
    setPlayhead(t);
  };

  // --- trim -------------------------------------------------------------
  const onSegTimes = (start, end) => mutate((p) => {
    const s = p.segments[selSeg];
    if (s) { s.start = start; s.end = end; }
  });
  const onSplit = () => {
    const s = segs[selSeg];
    if (!s) return;
    if (!(playhead > s.start + 0.2 && playhead < s.end - 0.2)) {
      pushToast?.('warn', 'Move the playhead inside the segment to split there.');
      return;
    }
    const t = r2(playhead);
    mutate((p) => {
      const cur = p.segments[selSeg];
      const right = { id: `seg-${Date.now()}`, start: t, end: cur.end,
        crop: cur.crop ? { ...cur.crop } : defaultCrop(p), transition_in: 'cut' };
      cur.end = t;
      p.segments.splice(selSeg + 1, 0, right);
    });
    pushToast?.('success', `Split into ${segs.length + 1} segments`);
  };
  const onDeleteSeg = () => {
    if (segs.length < 2) return;
    if (!window.confirm('Delete this segment?')) return;
    mutate((p) => { p.segments.splice(selSeg, 1); });
    setSelSeg(0);
  };

  // --- 3b: reorder / transitions / crop nudge --------------------------------
  const onMoveSeg = (i, dir) => {
    const j = i + dir;
    if (j < 0 || j >= segs.length) return;
    mutate((p) => { const [s] = p.segments.splice(i, 1); p.segments.splice(j, 0, s); });
    setSelSeg(j);
  };
  const onTransition = (i, t) => mutate((p) => {
    const s = p.segments[i];
    if (s) s.transition_in = t === 'fade' ? 'fade' : 'cut';
  });
  const clampCrop = (p, x, y) => {
    const W = (p.source && p.source.width) || 1920;
    const H = (p.source && p.source.height) || 1080;
    const s = p.segments[selSeg];
    if (!s) return;
    const c = s.crop || defaultCrop(p);
    const w = Math.min(c.w, W), h = Math.min(c.h, H);
    s.crop = { x: clamp(Math.round(x), 0, Math.max(0, W - w)),
               y: clamp(Math.round(y), 0, Math.max(0, H - h)), w, h };
  };
  const onCropNudge = (dx, dy) => mutate((p) => {
    const s = p.segments[selSeg];
    const c = (s && s.crop) || defaultCrop(p);
    clampCrop(p, c.x + dx, c.y + dy);
  });
  const onCropXY = (x, y) => mutate((p) => clampCrop(p, x, y));
  const onCropReset = () => mutate((p) => {
    const s = p.segments[selSeg];
    if (s) s.crop = defaultCrop(p);
  });

  // --- 3b: music ---------------------------------------------------------------
  const onAudioUpload = async (file) => {
    if (uploading) return;
    setUploading(true);
    try {
      const { file: rel } = await uploadAudio(jobId, file);
      mutate((pp) => { pp.audio = [{ file: rel, volume: 1.0, offset: 0.0 }]; });
      pushToast?.('success', 'Music attached — mixed at export');
    } catch (e) {
      pushToast?.('error', 'Upload failed: ' + ((e && e.message) || e).slice(0, 140));
    } finally {
      setUploading(false);
    }
  };
  const onAudioVolume = (v) => mutate((pp) => {
    pp.audio = pp.audio && pp.audio.length ? pp.audio : [{ file: null, volume: 1.0, offset: 0.0 }];
    pp.audio[0].volume = clamp(v, 0, 1.5);
  });
  const onAudioOffset = (v) => mutate((pp) => {
    pp.audio = pp.audio && pp.audio.length ? pp.audio : [{ file: null, volume: 1.0, offset: 0.0 }];
    pp.audio[0].offset = Math.max(0, v);
  });
  const onAudioRemove = () => mutate((pp) => {
    pp.audio = [{ file: null, volume: 1.0, offset: 0.0 }];
  });

  // --- captions ----------------------------------------------------------
  const onStyle = (style) => mutate((p) => { p.captions.style = style; });
  const onPosition = (position) => mutate((p) => { p.captions.position = position; });
  const onWordEdit = (i, text, orig) => mutate((p) => {
    const ed = (p.captions.edits = p.captions.edits || {});
    if (text === orig) delete ed[String(i)];
    else ed[String(i)] = text;
  });
  const onClearEdits = () => mutate((p) => { p.captions.edits = {}; });

  // --- hook ---------------------------------------------------------------
  const onHookChange = (patch) => mutate((p) => {
    const h = (p.overlays || []).find((o) => o.type === 'hook');
    if (h) Object.assign(h, patch);
  });
  const onAddHook = () => mutate((p) => {
    const dur = (p.source && p.source.duration) || 10;
    p.overlays = p.overlays || [];
    p.overlays.push({ type: 'hook', text: '', style: 'viral',
      start: 0, end: r2(Math.min(3, dur)), position: 'top' });
  });
  const onRemoveHook = () => mutate((p) => {
    p.overlays = (p.overlays || []).filter((o) => o.type !== 'hook');
  });

  // --- AI trim ("cut the intro" magic button) ------------------------------
  const onAskAi = async () => {
    const instr = aiText.trim();
    if (!instr || aiBusy) return;
    setAiBusy(true);
    setAiMsg('');
    try {
      const { drop_ranges = [], explanation = '' } = await editClipAI(jobId, clipIndex, instr);
      if (!drop_ranges.length) { setAiMsg(explanation || 'Nothing to cut for that instruction.'); return; }
      const spans = keptSpans(duration, drop_ranges).filter(([s, e]) => e - s >= 0.5);
      if (!spans.length) { setAiMsg('That would cut the whole clip — nothing changed.'); return; }
      if (!window.confirm(`Cut ${drop_ranges.length} span${drop_ranges.length === 1 ? '' : 's'} → ${spans.length} segment${spans.length === 1 ? '' : 's'}?`)) return;
      mutate((p) => {
        const crop0 = p.segments[0] && p.segments[0].crop;
        p.segments = spans.map(([s, e], i) => ({
          id: `seg-${Date.now()}-${i}`, start: r2(s), end: r2(e),
          crop: crop0 ? { ...crop0 } : defaultCrop(p), transition_in: 'cut',
        }));
      });
      setSelSeg(0);
      setAiMsg(explanation || `Cut into ${spans.length} segment${spans.length === 1 ? '' : 's'} — review the timeline, then save.`);
    } catch (e) {
      setAiMsg((e && e.message) || 'AI trim failed.');
    } finally {
      setAiBusy(false);
    }
  };

  // --- save / export / versions --------------------------------------------
  const onSave = async () => {
    if (!project || saving) return null;
    const errs = validateDraft(project);
    if (errs.length) { pushToast?.('error', errs[0]); return null; }
    setSaving(true);
    try {
      const { version, project: saved } = await saveClipProject(jobId, clipIndex, project);
      setProject(saved);
      setSavedVersion(version);
      setDirty(false);
      await refreshVersions();
      pushToast?.('success', `Draft saved as v${version}`);
      return saved;
    } catch (e) {
      pushToast?.('error', 'Save failed: ' + ((e && e.message) || e).slice(0, 140));
      return null;
    } finally {
      setSaving(false);
    }
  };

  const onExport = async () => {
    if (!project || exporting || saving) return;
    const errs = validateDraft(project);
    if (errs.length) { pushToast?.('error', errs[0]); return; }
    setExporting(true);
    try {
      // Export always saves first: the render is versioned, and the
      // version→file mapping stays truthful (no stale-version overwrites).
      let toRender = project;
      if (dirty) {
        const saved = await onSave();
        if (!saved) return;
        toRender = saved;
      }
      const res = await composeProject(jobId, clipIndex, toRender);
      setSavedVersion(res.version);
      const bust = `${res.composed_url.includes('?') ? '&' : '?'}v=${Date.now()}`;
      setVideoSrc(safeResolveUrl(res.composed_url) + bust);
      setPlayhead(0);
      await refreshVersions();
      pushToast?.('success', res.deduped
        ? `v${res.version} unchanged — reused the existing render`
        : `Exported v${res.version} — previewing the exact render`);
    } catch (e) {
      pushToast?.('error', 'Export failed: ' + ((e && e.message) || e).slice(0, 140));
    } finally {
      setExporting(false);
    }
  };

  // Undo: load an older version into the draft (keeps ITS idempotency key, so
  // exporting it unchanged dedups to the identical file — a true undo).
  const onLoadVersion = async (v) => {
    if (dirty && !window.confirm(`Discard unsaved changes and load v${v}?`)) return;
    try {
      const proj = await getClipProject(jobId, clipIndex, v);
      setProject(proj);
      setSavedVersion(proj.version);
      setSelSeg(0);
      setDirty(true);
      pushToast?.('info', `Loaded v${v} into the draft — save or export to keep it`);
    } catch (e) {
      pushToast?.('error', 'Load failed: ' + ((e && e.message) || e).slice(0, 120));
    }
  };

  const onPreviewVersion = (v) => {
    if (!v.output_url) return;
    setVideoSrc(safeResolveUrl(v.output_url) + `?v=${Date.now()}`);
    setPlayhead(0);
  };

  const onResetAi = async () => {
    if (dirty && !window.confirm('Discard all edits and reset to the AI cut?')) return;
    const auto = (versions || [])
      .filter((v) => v.origin === 'auto')
      .sort((a, b) => a.version - b.version)[0];
    const v = auto ? auto.version : 1;
    try {
      const proj = await getClipProject(jobId, clipIndex, v);
      setProject(proj);
      setSavedVersion(proj.version);
      setSelSeg(0);
      setDirty(true);
      pushToast?.('info', `Reset to the AI cut (v${proj.version}) — save or export to keep it`);
    } catch (e) {
      pushToast?.('error', 'Reset failed: ' + ((e && e.message) || e).slice(0, 120));
    }
  };

  const onBackClick = () => {
    if (dirty && !window.confirm('Leave the editor? Unsaved changes will be lost.')) return;
    onBack();
  };

  // --- render --------------------------------------------------------------
  if (status === 'loading') {
    return (
      <main className="container fade-in" style={{ paddingTop: 48, textAlign: 'center' }}>
        <Icon n="loader" cls="ico-spin" />
        <p style={{ color: 'var(--ink-dim, #666)', marginTop: 12 }}>Loading the clip project…</p>
      </main>
    );
  }
  if (status === 'error' || !project) {
    return (
      <main className="container fade-in" style={{ paddingTop: 48, textAlign: 'center' }}>
        <p style={{ color: 'var(--ink, #222)' }}>Couldn&apos;t open the editor.</p>
        <p style={{ color: 'var(--ink-dim, #666)', fontSize: 13 }}>{loadError}</p>
        <Btn variant="ghost" icon="arrow-left" onClick={onBack} style={{ marginTop: 12 }}>Back</Btn>
      </main>
    );
  }

  const capStyle = presetCss(caps.style);
  const hookVisible = hook && hook.text && playhead >= (hook.start || 0) && playhead <= (hook.end || 0);
  const PANELS = [
    { id: 'trim', label: 'Trim', icon: 'scissors' },
    { id: 'captions', label: 'Captions', icon: 'captions' },
    { id: 'hook', label: 'Hook', icon: 'type' },
    { id: 'music', label: 'Music', icon: 'audio-lines' },
    { id: 'ai', label: 'AI cut', icon: 'wand-sparkles' },
  ];

  return (
    <main className="container fade-in" style={{ maxWidth: 1400, paddingBottom: 48 }}>
      {/* header */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 12, padding: '18px 0', flexWrap: 'wrap' }}>
        <Btn variant="ghost" size="sm" icon="arrow-left" onClick={onBackClick}>Back</Btn>
        <div style={{ minWidth: 0, flex: 1 }}>
          <div className="eyebrow">Clip editor</div>
          <h2 style={{ margin: 0, fontSize: 20, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }} title={title}>
            {title}
          </h2>
        </div>
        {dirty && (
          <span style={{ fontSize: 11.5, fontWeight: 700, color: 'var(--gold, #c90)' }}>● unsaved changes</span>
        )}
        <span style={{ fontSize: 12, color: 'var(--ink-dim, #666)', fontFamily: 'var(--font-mono, monospace)' }}>
          v{savedVersion ?? project.version}
        </span>
        <Btn variant="ghost" size="sm" icon="refresh-cw" onClick={onResetAi} title="Reload the AI-generated cut">
          Reset to AI cut
        </Btn>
        <Btn variant="ghost" size="sm" icon="check" loading={saving} disabled={!dirty || saving} onClick={onSave}>
          {saving ? 'Saving…' : 'Save draft'}
        </Btn>
        <Btn variant="primary" size="sm" icon="download" loading={exporting} disabled={exporting} onClick={onExport}>
          {exporting ? 'Exporting…' : 'Export'}
        </Btn>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'minmax(0, 1.12fr) minmax(330px, 0.88fr)', gap: 24, alignItems: 'start' }}>
        {/* left: preview + timeline */}
        <div>
          <div style={{ position: 'relative', width: 'min(100%, 360px)', margin: '0 auto',
            aspectRatio: '9/16', background: '#000', borderRadius: 14, overflow: 'hidden',
            boxShadow: 'var(--clay-raised-md, 0 8px 30px rgba(0,0,0,.18))' }}>
            <video
              key={videoSrc}
              ref={videoRef}
              src={videoSrc}
              controls
              playsInline
              preload="auto"
              onTimeUpdate={onTimeUpdate}
              style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', objectFit: 'cover', zIndex: 0 }}
            />
            {/* hook overlay */}
            {hookVisible && (
              <div style={{ position: 'absolute', top: '4%', left: '6%', right: '6%', zIndex: 2,
                textAlign: 'center', color: '#fff', pointerEvents: 'none',
                fontFamily: "Anton, Impact, 'Arial Black', sans-serif", fontSize: 24, lineHeight: 1.15,
                textShadow: '-1.5px -1.5px 0 #000,1.5px -1.5px 0 #000,-1.5px 1.5px 0 #000,1.5px 1.5px 0 #000' }}>
                {hook.text}
              </div>
            )}
            {/* caption overlay */}
            {activeWord && (
              <div style={{ position: 'absolute', zIndex: 2, pointerEvents: 'none', textAlign: 'center',
                ...(CAP_POS_STYLE[caps.position] || CAP_POS_STYLE.bottom) }}>
                <span style={{ ...capStyle, fontSize: 26, fontWeight: 800, padding: '2px 10px',
                  borderRadius: 6, display: 'inline-block', lineHeight: 1.25 }}>
                  {activeWord.text}
                </span>
              </div>
            )}
          </div>

          <div style={{ maxWidth: 560, margin: '14px auto 0' }}>
            <Timeline
              duration={duration}
              segments={segs}
              selSeg={selSeg}
              onSelectSeg={setSelSeg}
              onSeek={onSeek}
              onTrim={onSegTimes}
              playhead={playhead}
              onPlayhead={setPlayhead}
            />
            <label style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 12.5,
              color: 'var(--ink-dim, #666)', marginTop: 8, cursor: 'pointer' }}>
              <input type="checkbox" checked={loopSeg} onChange={(e) => setLoopSeg(e.target.checked)} />
              Loop the selected segment in preview
            </label>
          </div>
        </div>

        {/* right: inspector + versions */}
        <div>
          <div style={{ display: 'flex', gap: 4, marginBottom: 14, borderBottom: '1px solid rgba(51,46,38,.12)',
            paddingBottom: 8 }} role="tablist">
            {PANELS.map((t) => (
              <button key={t.id} type="button" role="tab" aria-selected={panel === t.id}
                onClick={() => setPanel(t.id)}
                style={{ display: 'flex', alignItems: 'center', gap: 6, padding: '7px 12px',
                  borderRadius: 8, border: 'none', cursor: 'pointer', fontSize: 13, fontWeight: 600,
                  background: panel === t.id ? 'rgba(103,90,221,.14)' : 'transparent',
                  color: panel === t.id ? 'var(--plum, #675ADD)' : 'var(--ink-dim, #666)' }}>
                <Icon n={t.icon} /><span>{t.label}</span>
              </button>
            ))}
          </div>

          <div style={{ background: 'var(--surface, #fff)', borderRadius: 14,
            padding: 18, boxShadow: 'var(--clay-raised-md, 0 4px 18px rgba(0,0,0,.08))', marginBottom: 18 }}>
            {panel === 'trim' && (
              <TrimPanel project={project} selSeg={selSeg} setSelSeg={setSelSeg}
                onSegTimes={onSegTimes} onSplit={onSplit} onDeleteSeg={onDeleteSeg}
                onMoveSeg={onMoveSeg} onTransition={onTransition}
                onCropNudge={onCropNudge} onCropXY={onCropXY} onCropReset={onCropReset}
                playhead={playhead} />
            )}
            {panel === 'captions' && (
              <CaptionsPanel project={project} onStyle={onStyle} onPosition={onPosition}
                onWordEdit={onWordEdit} onClearEdits={onClearEdits} />
            )}
            {panel === 'hook' && (
              <HookPanel project={project} onHookChange={onHookChange}
                onAddHook={onAddHook} onRemoveHook={onRemoveHook} />
            )}
            {panel === 'music' && (
              <MusicPanel project={project} uploading={uploading}
                onUpload={onAudioUpload} onVolume={onAudioVolume}
                onOffset={onAudioOffset} onRemove={onAudioRemove} />
            )}
            {panel === 'ai' && (
              <AiTrimPanel aiText={aiText} setAiText={setAiText} aiBusy={aiBusy}
                aiMsg={aiMsg} onAsk={onAskAi} />
            )}
          </div>

          <div style={{ background: 'var(--surface, #fff)', borderRadius: 14,
            padding: 18, boxShadow: 'var(--clay-raised-md, 0 4px 18px rgba(0,0,0,.08))' }}>
            <VersionsPanel versions={versions} currentVersion={savedVersion}
              onLoad={onLoadVersion} onPreviewVersion={onPreviewVersion} loading={versionsLoading} />
          </div>
        </div>
      </div>
    </main>
  );
}
