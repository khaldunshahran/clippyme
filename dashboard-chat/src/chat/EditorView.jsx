import { useCallback, useEffect, useMemo, useState } from 'react';
import { ArrowLeft } from 'lucide-react';
import '../editor/tokens.css';
import '../editor/app.css';
import { ClipEditorView } from '../editor/clipEditor.jsx';
import CopilotPanel from './CopilotPanel.jsx';
import TranscriptPanel from '../editor-timeline/TranscriptPanel.jsx';
import { createProjectController } from './projectController.js';
import { onSessionReset } from '../api/sessionReset.js';
import {
  clipVideoSrc,
  getClipProject,
  saveClipProject,
  composeProject,
  backfillClipProjects,
  listClipProjectVersions,
  editClipAI,
  uploadAudio,
} from '../api/realApi.js';

/**
 * Deep editor, slice 1: transcript-first editing.
 *
 * Left column: TranscriptPanel (word-level transcript, click-to-seek,
 * inline word edit, split & trim, remove caption & video). Main area: the
 * nested clip editor.
 *
 * Task 16: ONE authoritative project controller per clip owns the project,
 * its acknowledged version, and the single serialized edit queue
 * (editor-timeline/undo.js) with optimistic concurrency (expected_version
 * -> 409 on conflict). Manual edits, AI patch applies, version
 * restore/reset, undo/redo, and save/export all route through it; the
 * nested ClipEditorView runs in controlled mode and never keeps its own
 * copy.
 */
export default function EditorView({ jobId, clipIndex, clip, onBack }) {
  const [snap, setSnap] = useState(null);
  const [toast, setToast] = useState(null);
  // Bumped on session reset so the TranscriptPanel remounts and drops its
  // selection / editing state along with the cleared project.
  const [resetKey, setResetKey] = useState(0);

  const pushToast = useCallback((msg) => {
    setToast(msg);
    clearTimeout(pushToast._t);
    pushToast._t = setTimeout(() => setToast(null), 2600);
    console.log('[editor]', msg);
  }, []);

  // The single project authority for this clip (condition 4). Recreated on
  // clip switch; the old controller is disposed so its late responses can
  // never touch the new clip's state.
  const controller = useMemo(
    () =>
      createProjectController({
        jobId,
        clipIndex,
        api: {
          getClipProject,
          saveClipProject,
          composeProject,
          backfillClipProjects,
          listClipProjectVersions,
          editClipAI,
          uploadAudio,
        },
        onToast: pushToast,
      }),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [jobId, clipIndex]
  );

  useEffect(() => {
    const unsubscribe = controller.subscribe(setSnap);
    controller.load().catch(() => {});
    controller.listVersions();
    return () => {
      unsubscribe();
      controller.dispose();
    };
  }, [controller]);

  // Session reset (sign-out or signed-in user change): drop the undo/redo
  // history, the cached project, and force the transcript panel to remount
  // so selection/editing state can't leak across users.
  useEffect(() => {
    return onSessionReset(() => {
      controller.resetSession();
      setResetKey((k) => k + 1);
    });
  }, [controller]);

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
        controller.redo().catch(() => {});
      } else {
        controller.undo().catch(() => {});
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [controller]);

  const project = snap?.project ?? null;
  const loadError = snap?.loadError || '';

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
              key={resetKey}
              project={project}
              videoSrc={videoSrc}
              applyEdit={controller.applyEdit}
              pushToast={pushToast}
              disabled={!project}
            />
          )}
        </div>

        {/* Nested deep editor: controlled by the project controller above. */}
        <div className="nc-editor-scope" style={{ flex: 1, minHeight: 0, position: 'relative', overflow: 'auto' }}>
          <ClipEditorView
            jobId={jobId}
            clipIndex={clipIndex}
            clip={clip?.raw || clip}
            onBack={onBack}
            pushToast={pushToast}
            controller={controller}
          />
          {/* Condition 4: copilot edits route through the same queue.
              Keyed by clip identity + session: a clip switch or session
              reset remounts the panel, dropping any stale AI result. */}
          <CopilotPanel
            key={`${jobId}:${clipIndex}:${resetKey}`}
            jobId={jobId}
            clipIndex={clipIndex}
            applyEdit={controller.applyEdit}
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
