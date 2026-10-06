import { useEffect } from 'react';
import { ArrowLeft } from 'lucide-react';
import '../editor/tokens.css';
import '../editor/app.css';
import { ClipEditorView } from '../editor/clipEditor.jsx';
import CopilotPanel from './CopilotPanel.jsx';

/**
 * Deep editor: mounts the shipped clipEditor (unchanged) inside the
 * scoped clay theme (.nc-editor-scope), plus the floating AI copilot.
 */
export default function EditorView({ jobId, clipIndex, clip, onBack }) {
  const pushToast = (msg) => {
    // clipEditor's toast hook -> lightweight console + alert-free fallback
    console.log('[editor]', msg);
  };

  return (
    <div className="nc-anim-fade-in" style={{ position: 'fixed', inset: 0, zIndex: 70, background: 'var(--nc-bg)', display: 'flex', flexDirection: 'column' }}>
      <div style={{
        display: 'flex', alignItems: 'center', gap: 10, padding: '10px 14px',
        borderBottom: '1px solid var(--nc-border)', background: 'var(--nc-bg-soft)', flexShrink: 0,
      }}>
        <button className="nc-icon-btn" onClick={onBack} aria-label="Back to chat">
          <ArrowLeft size={17} />
        </button>
        <div style={{ fontSize: 14, fontWeight: 700, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
          Editing — {clip?.title || `Clip ${clipIndex + 1}`}
        </div>
        <div style={{ marginLeft: 'auto', fontSize: 12, color: 'var(--nc-text-faint)' }}>
          AI copilot is floating bottom-right
        </div>
      </div>
      <div className="nc-editor-scope" style={{ flex: 1, minHeight: 0, position: 'relative', overflow: 'auto' }}>
        <ClipEditorView
          jobId={jobId}
          clipIndex={clipIndex}
          clip={clip?.raw || clip}
          onBack={onBack}
          pushToast={pushToast}
        />
        <CopilotPanel jobId={jobId} clipIndex={clipIndex} />
      </div>
    </div>
  );
}
