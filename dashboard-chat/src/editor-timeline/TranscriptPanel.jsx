import { useEffect, useMemo, useRef, useState } from 'react';
import {
  keptDuration,
  outputToSource,
  sourceToOutput,
} from './timebase.js';
import {
  applyRemoveWords,
  applySplitAtWord,
  applyWordEdit,
  displayWord,
  slice1Menu,
} from './wordMenu.js';
import './transcript.css';

/**
 * Slice-1 transcript panel: word-level transcript with click-to-seek,
 * inline word edit, split & trim, and remove-caption-&-video.
 *
 * All word times are SOURCE time (see timebase.js). Click-to-seek converts
 * to the preview's output timeline via sourceToOutput(); the render
 * pipeline (compose.py) applies the same conversion, so preview and
 * export agree.
 */
export default function TranscriptPanel({
  project,
  videoSrc,
  applyEdit,
  pushToast,
  disabled,
}) {
  const words = useMemo(
    () => project?.captions?.words || [],
    [project]
  );
  const segments = useMemo(() => project?.segments || [], [project]);
  const videoRef = useRef(null);
  const [sel, setSel] = useState([]); // selected word ids
  const [anchor, setAnchor] = useState(0); // shift-click anchor (word index)
  const [menu, setMenu] = useState(null); // {x, y} in viewport coords
  const [editingId, setEditingId] = useState(null);
  const [editText, setEditText] = useState('');
  const [activeId, setActiveId] = useState(null); // word under playhead
  const [busy, setBusy] = useState(false);
  const commitGuard = useRef(false); // Enter+blur must not double-commit

  const startEdit = (w, i) => {
    setAnchor(i);
    setSel([w.id]);
    commitGuard.current = false;
    setEditingId(w.id);
    setEditText(displayWord(project, w, i));
  };

  const kept = useMemo(() => keptDuration(segments), [segments]);

  // Active-word highlight while the preview plays (output -> source).
  useEffect(() => {
    const v = videoRef.current;
    if (!v) return;
    let raf = 0;
    const tick = () => {
      const srcT = outputToSource(v.currentTime || 0, segments);
      let found = null;
      if (srcT != null) {
        found = words.find((w) => w.start <= srcT && srcT < w.end)?.id || null;
      }
      setActiveId((prev) => (prev === found ? prev : found));
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [segments, words]);

  // Close the context menu on outside click / Escape.
  useEffect(() => {
    if (!menu) return;
    const close = () => setMenu(null);
    const onKey = (e) => e.key === 'Escape' && setMenu(null);
    window.addEventListener('mousedown', close);
    window.addEventListener('keydown', onKey);
    return () => {
      window.removeEventListener('mousedown', close);
      window.removeEventListener('keydown', onKey);
    };
  }, [menu ]);

  const seekToWord = (w) => {
    const v = videoRef.current;
    if (!v) return;
    const t = sourceToOutput(w.start, segments);
    if (t == null) {
      pushToast?.('That part was removed from the timeline');
      return;
    }
    try {
      v.currentTime = Math.max(0, Math.min(t, (v.duration || t + 1) - 0.05));
      v.play().catch(() => {});
    } catch {
      /* not yet loaded */
    }
  };

  const onWordClick = (e, w, i) => {
    if (editingId) return;
    if (e.shiftKey) {
      // Shift-click range selection (condition 7).
      const a = Math.min(anchor, i);
      const b = Math.max(anchor, i);
      setSel(words.slice(a, b + 1).map((x) => x.id));
    } else {
      setAnchor(i);
      setSel([w.id]);
      seekToWord(w);
    }
  };

  const onWordContext = (e, w, i) => {
    e.preventDefault();
    e.stopPropagation();
    if (!sel.includes(w.id)) {
      setAnchor(i);
      setSel([w.id]);
    }
    setMenu({ x: e.clientX, y: e.clientY });
  };

  const selectedWords = useMemo(
    () => words.filter((w) => sel.includes(w.id)),
    [words, sel]
  );

  const runWithBusy = async (fn) => {
    setBusy(true);
    try {
      await fn();
    } finally {
      setBusy(false);
    }
  };

  const doAction = (id) => {
    const targets = selectedWords.length ? selectedWords : [];
    setMenu(null);
    if (id === 'edit') {
      const w = targets[0];
      if (!w) return;
      startEdit(w, words.indexOf(w));
      return;
    }
    runWithBusy(async () => {
      if (id === 'split') {
        const w = targets[0];
        if (!w) return;
        await applyEdit('split at word', (before) => {
          const word = (before.captions.words || []).find((x) => x.id === w.id);
          if (!word) throw new Error('word no longer exists');
          return applySplitAtWord(before, word);
        });
        pushToast?.('Split at ' + fmtTime(w.start));
      } else if (id === 'remove') {
        if (!targets.length) return;
        await applyEdit(
          `remove ${targets.length} word${targets.length > 1 ? 's' : ''}`,
          (before) => {
            const ws = targets
              .map((t) => (before.captions.words || []).find((x) => x.id === t.id))
              .filter(Boolean);
            return applyRemoveWords(before, ws);
          }
        );
        setSel([]);
        pushToast?.('Removed from timeline');
      }
    }).catch(() => {});
  };

  const commitEdit = () => {
    if (commitGuard.current) return; // Enter already committed; blur is a no-op
    commitGuard.current = true;
    const id = editingId;
    const text = editText.trim();
    setEditingId(null);
    if (!id || !text) return;
    const before = displayWord(project, words.find((w) => w.id === id),
      words.findIndex((w) => w.id === id));
    if (text === before) return;
    runWithBusy(async () => {
      await applyEdit('edit word', (p) => applyWordEdit(p, id, text));
      pushToast?.('Word updated');
    }).catch(() => {});
  };

  const menuItems = slice1Menu({ hasRange: selectedWords.length > 1 });

  return (
    <div className="tp" data-testid="transcript-panel">
      <div className="tp-video-wrap">
        {videoSrc ? (
          <video
            ref={videoRef}
            className="tp-video"
            src={videoSrc}
            playsInline
            preload="metadata"
          />
        ) : (
          <div className="tp-video tp-video-empty">No preview</div>
        )}
        <div className="tp-kept" title="Kept timeline duration">
          {fmtTime(kept)} kept
        </div>
      </div>

      <div className="tp-head">
        <span className="tp-title">Transcript</span>
        <span className="tp-count">{words.length} words</span>
      </div>

      <div className="tp-words" data-testid="transcript-words">
        {words.map((w, i) => {
          const cls = [
            'tp-w',
            sel.includes(w.id) ? 'sel' : '',
            activeId === w.id ? 'active' : '',
            sourceToOutput(w.start, segments) == null ? 'dropped' : '',
          ].join(' ');
          if (editingId === w.id) {
            return (
              <input
                key={w.id}
                data-testid="word-editor"
                className="tp-wedit"
                autoFocus
                value={editText}
                disabled={disabled || busy}
                onChange={(e) => setEditText(e.target.value)}
                onKeyDown={(e) => {
                  // Condition 6: the global Ctrl/Cmd+Z undo must not fire
                  // while the inline editor has focus -- stop it here.
                  e.stopPropagation();
                  if (e.key === 'Enter') commitEdit();
                  if (e.key === 'Escape') {
                    commitGuard.current = true; // don't commit on blur
                    setEditingId(null);
                  }
                }}
                onBlur={commitEdit}
                onClick={(e) => e.stopPropagation()}
              />
            );
          }
          return (
            <span
              key={w.id}
              className={cls}
              title={`${fmtTime(w.start)} → ${fmtTime(w.end)} (source)`}
              onClick={(e) => onWordClick(e, w, i)}
              onContextMenu={(e) => onWordContext(e, w, i)}
              onDoubleClick={() => startEdit(w, i)}
            >
              {displayWord(project, w, i)}
            </span>
          );
        })}
        {!words.length && (
          <div className="tp-empty">No transcript words in this project.</div>
        )}
      </div>

      {menu && (
        <div
          className="tp-menu"
          data-testid="word-menu"
          style={{
            left: Math.min(menu.x, window.innerWidth - 220),
            top: Math.min(menu.y, window.innerHeight - 180),
          }}
          onMouseDown={(e) => e.stopPropagation()}
        >
          {menuItems.map((m) => (
            <button
              key={m.id}
              className={'tp-menu-item' + (m.danger ? ' danger' : '')}
              disabled={disabled || busy}
              onClick={() => doAction(m.id)}
            >
              {m.label}
              {m.hint && <span className="tp-menu-hint">{m.hint}</span>}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

function fmtTime(t) {
  t = Math.max(0, Number(t) || 0);
  const m = Math.floor(t / 60);
  const s = t - m * 60;
  return `${m}:${s < 10 ? '0' : ''}${s.toFixed(1)}`;
}
