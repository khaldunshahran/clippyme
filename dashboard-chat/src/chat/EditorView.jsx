import { useEffect, useMemo, useState } from 'react';
import '../editor/tokens.css';
import '../editor/app.css';
import { ClipEditorView } from '../editor/clipEditor.jsx';
import { Btn } from '../editor/primitives.jsx';
import CopilotPanel from './CopilotPanel.jsx';
import TranscriptPanel from '../editor-timeline/TranscriptPanel.jsx';
import { useWorkspaceController } from './useWorkspaceController.js';
import { clipVideoSrc } from '../api/realApi.js';

/**
 * Deep editor, slice 1: transcript-first editing.
 *
 * UI-01: the workspace is editor-first. The transcript is a collapsible
 * tool rail (not a fixed 300px column); the nested clip editor owns the
 * single workspace header (clip identity, acknowledged version,
 * pending/saved state, export action). The old EditorView header (Back +
 * "Editing — title" + version) was redundant with it and is removed.
 *
 * In hosted mode (controller prop provided by the app) the floating AI
 * copilot is replaced by a contextual "Ask Nugget to edit" action that
 * focuses the main chat composer — main chat is the primary AI entry
 * point. Standalone/back-compat mounts (no controller prop) keep the
 * floating copilot.
 *
 * Task 16: ONE authoritative project controller per clip owns the project,
 * its acknowledged version, and the single serialized edit queue
 * (editor-timeline/undo.js) with optimistic concurrency (expected_version
 * -> 409 on conflict). Manual edits, AI patch applies, version
 * restore/reset, undo/redo, and save/export all route through it; the
 * nested ClipEditorView runs in controlled mode and never keeps its own
 * copy.
 *
 * Task 17: the editor renders INLINE in the workspace panel (no more
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
  // UI-01 hosted integration. onAskNugget focuses the main chat composer;
  // chatCollapsed/onToggleChat hide the chat rail to focus the editor.
  onAskNugget = null,
  chatCollapsed = false,
  onToggleChat = null,
}) {
  const internal = useWorkspaceController({ jobId, clipIndex, disabled: !!controllerProp });
  const controller = controllerProp || internal.controller;
  const snap = controllerProp ? snapProp : internal.snap;
  const resetKey = controllerProp ? resetKeyProp : internal.resetKey;
  const toast = controllerProp ? toastProp : internal.toast;

  // UI-01: transcript is a collapsible rail. Open by default on wide
  // screens; the header toggle (and keyboard) controls it everywhere.
  const [transcriptOpen, setTranscriptOpen] = useState(
    () => (typeof window === 'undefined' ? true : window.innerWidth >= 1200)
  );

  // Global undo/redo with the inline-editor guard (condition 6):
  // never fire when focus is inside an input/textarea/select/contenteditable.
  useEffect(() => {
    if (!controller) return undefined;
    const onKey = (e) => {
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
      const mod = e.ctrlKey || e.metaKey;
      if (!mod || e.key.toLowerCase() !== 'z') return;
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

  // Hosted = the app provided the shared project authority. Standalone =
  // back-compat mode (own controller, floating copilot kept).
  const hosted = !!controllerProp;

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

  // Workspace header extras: transcript rail toggle, contextual AI entry
  // (hosted only — focuses the main chat composer), chat rail toggle.
  const headerExtra = (
    <>
      <Btn
        variant="ghost"
        size="sm"
        icon="captions"
        onClick={() => setTranscriptOpen((v) => !v)}
        aria-pressed={transcriptOpen}
        aria-controls="nc-transcript-panel"
        title={transcriptOpen ? 'Hide the transcript rail' : 'Show the transcript rail'}
      >
        Transcript
      </Btn>
      {hosted && onAskNugget && (
        <Btn
          variant="ghost"
          size="sm"
          icon="sparkles"
          onClick={onAskNugget}
          title="Focus the chat composer to ask Nugget for an edit"
          aria-label="Ask Nugget to edit — focuses the chat composer"
        >
          Ask Nugget to edit
        </Btn>
      )}
      {onToggleChat && (
        <Btn
          variant="ghost"
          size="sm"
          onClick={onToggleChat}
          aria-pressed={chatCollapsed}
          title={chatCollapsed ? 'Show the chat column' : 'Hide the chat column to focus the editor'}
        >
          {chatCollapsed ? 'Show chat' : 'Hide chat'}
        </Btn>
      )}
    </>
  );

  return (
    <div className="nc-anim-fade-in nc-ws">
      <div className="nc-ws-body">
        {/* Transcript tool rail: collapsible; the transcript seek path
            (click-to-seek, word edit, split & trim) is unchanged. */}
        {transcriptOpen && (
          <div className="nc-ws-transcript" id="nc-transcript-panel">
            {loadError ? (
              <div style={{ padding: 16, fontSize: 13, color: 'var(--rust-deep, #9b4126)' }}>
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
        )}

        {/* Nested deep editor: controlled by the project controller above.
            Its header is the single workspace header (identity, version,
            pending/saved state, export). */}
        <div className="nc-editor-scope nc-ws-editor">
          <ClipEditorView
            jobId={jobId}
            clipIndex={clipIndex}
            clip={clip?.raw || clip}
            onBack={onBack}
            pushToast={internal.pushToast}
            controller={controller}
            headerExtra={headerExtra}
          />
          {/* Standalone/back-compat only: the floating copilot is kept.
              Hosted mode uses "Ask Nugget to edit" (main chat composer)
              instead — main chat is the primary AI entry point. */}
          {!hosted && (
            <CopilotPanel
              key={`${jobId}:${clipIndex}:${resetKey}`}
              jobId={jobId}
              clipIndex={clipIndex}
              applyEdit={controller.applyEdit}
            />
          )}
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
