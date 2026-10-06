import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { ArrowLeft } from 'lucide-react';
import '../editor/tokens.css';
import '../editor/app.css';
import { ClipEditorView } from '../editor/clipEditor.jsx';
import CopilotPanel from './CopilotPanel.jsx';
import TranscriptPanel from '../editor-timeline/TranscriptPanel.jsx';
import { createEditQueue } from '../editor-timeline/undo.js';
import {
  clipVideoSrc,
  getClipProject,
  saveClipProject,
} from '../api/realApi.js';

/**
 * Deep editor, slice 1: transcript-first editing.
 *
 * Left column: TranscriptPanel (word-level transcript, click-to-seek,
 * inline word edit, split & trim, remove caption & video). Main area: the
 * shipped clipEditor (unchanged) until later slices replace it.
 *
 * ALL project mutations -- panel actions, copilot applies, undo, redo --
 * go through one serialized edit queue (editor-timeline/undo.js) with
 * optimistic concurrency (expected_version -> 409 on conflict).
 */
export default function EditorView({ jobId, clipIndex, clip, onBack }) {
  const [project, setProject] = useState(null);
  const [loadError, setLoadError] = useState(null);
  const [toast, setToast] = useState(null);
  const projectRef = useRef(null);
  projectRef.current = project;

  const pushToast = useCallback((msg) => {
    setToast(msg);
    clearTimeout(pushToast._t);
    pushToast._t = setTimeout(() => setToast(null), 2600);
    console.log('[editor]', msg);
  }, []);

  const refreshProject = useCallback(async () => {
    try {
      const res = await getClipProject(jobId, clipIndex);
      setProject(res?.project || res);
      setLoadError(null);
    } catch (e) {
      setLoadError(e?.message || 'failed to load project');
    }
  }, [jobId, clipIndex]);

  // The single serialized mutation queue (condition 4).
  const queue = useMemo(
    () =>
      createEditQueue({
        getProject: () => projectRef.current,
        saveProject: (p, opts) => saveClipProject(jobId, clipIndex, p, opts),
        onUpdate: (_label, saved) => {
          if (saved?.project) setProject(saved.project);
        },
        onError: (label, e) => {
          const stale = e?.status === 409 || /stale project version/i.test(e?.message || '');
          pushToast(
            stale
              ? 'Someone else saved first — reloading the latest version.'
              : `Couldn't save (${label}): ${e?.message || 'unknown error'}`
          );
          if (stale) refreshProject();
        },
      }),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [jobId, clipIndex]
  );

  useEffect(() => {
    refreshProject();
  }, [refreshProject]);

  // Global undo/redo with the inline-editor guard (condition 6):
  // never fire when focus is inside an input/textarea/select/contenteditable.
  useEffect(() => {
    const onKey = (e) => {
      const mod = e.ctrlKey || e.metaKey;
      if (!mod || e.key.toLowerCase() !== 'z') return;
      const t = e.target;
      if (
        t &&
        (t.tagName === 'INPUT' ||
          t.tagName === 'TEXTAREA' ||
          t.tagName === 'SELECT' ||
          t.isContentEditable)
      ) {
        return;
      }
      e.preventDefault();
      if (e.shiftKey) {
        queue.redo().catch(() => {});
      } else {
        queue.undo().catch(() => {});
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [queue]);

  const videoSrc = useMemo(() => {
    try {
      return clipVideoSrc(clip?.raw || clip) || '';
    } catch {
      return '';
    }
  }, [clip]);

  return (
    <div
      className="nc-anim-fade-in"
      style={{
        position: 'fixed', inset: 0, zIndex: 70, background: 'var(--nc-bg)',
        display: 'flex', flexDirection: 'column',
      }}
    >
      <div style={{
        display: 'flex', alignItems: 'center', gap: 10, padding: '10px 14px',
        borderBottom: '1px solid var(--nc-border)', background: 'var(--nc-bg-soft)',
        flexShrink: 0,
      }}>
        <button className="nc-icon-btn" onClick={onBack} aria-label="Back to chat">
          <ArrowLeft size={17} />
        </button>
        <div style={{ fontSize: 14, fontWeight: 700, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
          Editing — {clip?.title || `Clip ${clipIndex + 1}`}
        </div>
        <div style={{ marginLeft: 'auto', fontSize: 12, color: 'var(--nc-text-faint)' }}>
          {project ? `v${project.version}` : ''}
        </div>
      </div>

      <div style={{ flex: 1, minHeight: 0, display: 'flex', position: 'relative' }}>
        {/* Slice-1 transcript column */}
        <div style={{ width: 300, flexShrink: 0, borderRight: '1px solid var(--nc-border)', minHeight: 0, display: 'flex', flexDirection: 'column' }}>
          {loadError ? (
            <div style={{ padding: 16, fontSize: 13, color: 'var(--nc-red)' }}>
              {loadError}
            </div>
          ) : (
            <TranscriptPanel
              project={project}
              videoSrc={videoSrc}
              applyEdit={queue.applyEdit}
              pushToast={pushToast}
              disabled={!project}
            />
          )}
        </div>

        {/* Existing deep editor (unchanged) */}
        <div className="nc-editor-scope" style={{ flex: 1, minHeight: 0, position: 'relative', overflow: 'auto' }}>
          <ClipEditorView
            jobId={jobId}
            clipIndex={clipIndex}
            clip={clip?.raw || clip}
            onBack={onBack}
            pushToast={pushToast}
          />
          {/* Condition 4: copilot edits route through the same queue. */}
          <CopilotPanel
            jobId={jobId}
            clipIndex={clipIndex}
            applyEdit={queue.applyEdit}
          />
        </div>
      </div>

      {toast && (
        <div
          style={{
            position: 'fixed', left: '50%', bottom: 24, transform: 'translateX(-50%)',
            zIndex: 100, background: '#17171c', border: '1px solid var(--nc-border)',
            color: 'var(--nc-text)', fontSize: 12.5, fontWeight: 600,
            padding: '9px 16px', borderRadius: 99, boxShadow: '0 8px 28px rgba(0,0,0,.5)',
          }}
        >
          {toast}
        </div>
      )}
    </div>
  );
}
