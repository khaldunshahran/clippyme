import { useEffect, useMemo } from 'react';
import { ArrowLeft } from 'lucide-react';
import '../editor/tokens.css';
import '../editor/app.css';
import { ClipEditorView } from '../editor/clipEditor.jsx';
import CopilotPanel from './CopilotPanel.jsx';
import TranscriptPanel from '../editor-timeline/TranscriptPanel.jsx';
import { useWorkspaceController } from './useWorkspaceController.js';
import { clipVideoSrc } from '../api/realApi.js';

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
 *
 * Task 17: the editor renders INLINE in the workspace side panel (no more
 * fixed full-screen overlay) and receives the shared project authority as
 * props {controller, snap, resetKey, toast} from the host's
 * useWorkspaceController — the SAME controller main-chat proposals,
 * undo/redo and export route through. In this mode it never
 * creates/disposes/loads its own controller.
 *
 * Back-compat (pinned by the Task 16 host-lifecycle tests): when no
 * controller prop is passed, it creates exactly one per (jobId, clipIndex)
 * via useWorkspaceController with the same lifecycle as before.
 */
export default function EditorView({
  jobId, clipIndex, clip, onBack,
  controller: controllerProp = null,
  snap: snapProp = null,
  resetKey: resetKeyProp = 0,
  toast: toastProp = null,
}) {
  const internal = useWorkspaceController({ jobId, clipIndex, disabled: !!controllerProp });
  const controller = controllerProp || internal.controller;
  const snap = controllerProp ? snapProp : internal.snap;
  const resetKey = controllerProp ? resetKeyProp : internal.resetKey;
  const toast = controllerProp ? toastProp : internal.toast;

  // Global undo/redo with the inline-editor guard (condition 6):
  // never fire when focus is inside an input/textarea/select/contenteditable.
  useEffect(() => {
    if (!controller) return undefined;
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

  if (!controller) return null;

  return (
    <div
      className="nc-anim-fade-in"
      style={{
        flex: 1, minHeight: 0, background: 'var(--nc-bg)',
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
              pushToast={internal.pushToast}
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
            pushToast={internal.pushToast}
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
