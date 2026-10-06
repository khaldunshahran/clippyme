import { useCallback, useState } from 'react';
import { Menu } from 'lucide-react';
import { useChat } from './chat/useChat.js';
import Sidebar from './chat/Sidebar.jsx';
import WelcomeView from './chat/WelcomeView.jsx';
import ChatThread from './chat/ChatThread.jsx';
import Composer from './chat/Composer.jsx';
import SettingsPopup from './chat/SettingsPopup.jsx';
import ClipGrid from './chat/ClipGrid.jsx';
import ClipPopup from './chat/ClipPopup.jsx';
import EditorView from './chat/EditorView.jsx';

export default function App() {
  const chat = useChat();
  const {
    threads, active, activeId,
    settingsOpen, setSettingsOpen,
    gridOpen, setGridOpen,
    popupClip, setPopupClip,
    editorTarget, setEditorTarget,
    createThread, selectThread, deleteThread,
    sendMessage, confirmSettings, patchThread,
    DEFAULT_SETTINGS,
  } = chat;

  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [scheduling, setScheduling] = useState(null); // clip index being "scheduled" (Phase D)

  const ensureThread = useCallback(() => {
    if (activeId) return activeId;
    return createThread();
  }, [activeId, createThread]);

  const openClip = useCallback((threadId, index) => {
    setPopupClip({ threadId, index });
  }, [setPopupClip]);

  const stepClip = useCallback((dir) => {
    setPopupClip((pc) => {
      if (!pc) return pc;
      const t = threads.find((x) => x.id === pc.threadId);
      const n = t?.clips?.length || 0;
      if (!n) return pc;
      return { ...pc, index: (pc.index + dir + n) % n };
    });
  }, [threads, setPopupClip]);

  const openEditor = useCallback((threadId, index) => {
    const t = threads.find((x) => x.id === threadId);
    if (!t?.jobId) return;
    setPopupClip(null);
    setEditorTarget({ threadId, jobId: t.jobId, clipIndex: index, clip: t.clips[index] });
  }, [threads, setEditorTarget, setPopupClip]);

  const scheduleClip = useCallback((threadId, index) => {
    // No schedule endpoint exists yet (Phase D gap) — honest placeholder.
    setScheduling({ threadId, index });
    setTimeout(() => setScheduling(null), 2600);
  }, []);

  const showWelcome = !active || (active.messages.length === 0 && active.phase === 'idle');
  const popupThread = popupClip ? threads.find((t) => t.id === popupClip.threadId) : null;

  return (
    <div style={{ display: 'flex', height: '100%', background: 'var(--nc-bg)', color: 'var(--nc-text)', overflow: 'hidden' }}>
      <Sidebar
        threads={threads}
        activeId={activeId}
        onSelect={selectThread}
        onNew={createThread}
        onDelete={deleteThread}
        collapsed={!sidebarOpen}
      />

      <main style={{ flex: 1, display: 'flex', flexDirection: 'column', minWidth: 0 }}>
        {/* top bar */}
        <header style={{
          display: 'flex', alignItems: 'center', gap: 8,
          padding: '10px 14px', borderBottom: '1px solid var(--nc-border)', flexShrink: 0,
        }}>
          <button className="nc-icon-btn" onClick={() => setSidebarOpen((v) => !v)} aria-label="Toggle sidebar">
            <Menu size={17} />
          </button>
          <div style={{ fontSize: 14, fontWeight: 700 }}>
            {active?.title && activeId ? active.title : 'Nugget Chat'}
          </div>
          {active?.phase === 'clipping' && (
            <span style={{
              fontSize: 11, fontWeight: 700, color: 'var(--nc-accent)',
              background: 'var(--nc-accent-soft)', borderRadius: 99, padding: '3px 10px',
              display: 'inline-flex', alignItems: 'center', gap: 6,
            }}>
              <span style={{ width: 6, height: 6, borderRadius: '50%', background: 'var(--nc-accent)', animation: 'nc-pulse-dot 1.6s ease-in-out infinite' }} />
              CLIPPING
            </span>
          )}
        </header>

        {showWelcome ? (
          <WelcomeView onExample={(url) => { const id = ensureThread(); sendMessage(url); }} />
        ) : (
          <ChatThread
            thread={active}
            onOpenClip={(i) => openClip(activeId, i)}
            onEditClip={(i) => openEditor(activeId, i)}
            onScheduleClip={(i) => scheduleClip(activeId, i)}
            onViewAll={() => setGridOpen(true)}
          />
        )}

        <Composer
          onSend={sendMessage}
          disabled={false}
          phase={active?.phase || 'idle'}
          settings={active?.settings || DEFAULT_SETTINGS}
          onSettingsChange={(s) => activeId && patchThread(activeId, { settings: s })}
          onOpenSettings={() => {
            ensureThread();
            setSettingsOpen(true);
          }}
        />
      </main>

      {/* settings popup */}
      {settingsOpen && active && (
        <SettingsPopup
          validation={active.validation}
          initial={active.settings}
          onClose={() => {
            setSettingsOpen(false);
            // closing without confirming returns the thread to idle
            if (active.phase === 'settings') patchThread(active.id, { phase: 'idle' });
          }}
          onConfirm={(s) => confirmSettings(active.id, s)}
        />
      )}

      {/* view-all grid */}
      {gridOpen && active && active.clips.length > 0 && (
        <ClipGrid
          clips={active.clips}
          title={active.title}
          onClose={() => setGridOpen(false)}
          onOpenClip={(i) => { setGridOpen(false); openClip(activeId, i); }}
          onEditClip={(i) => { setGridOpen(false); openEditor(activeId, i); }}
          onScheduleClip={(i) => scheduleClip(activeId, i)}
        />
      )}

      {/* clip detail popup */}
      {popupClip && popupThread && (
        <ClipPopup
          clips={popupThread.clips}
          index={popupClip.index}
          jobId={popupThread.jobId}
          onClose={() => setPopupClip(null)}
          onPrev={() => stepClip(-1)}
          onNext={() => stepClip(1)}
          onEdit={(i) => openEditor(popupThread.id, i)}
        />
      )}

      {/* deep editor */}
      {editorTarget && (
        <EditorView
          jobId={editorTarget.jobId}
          clipIndex={editorTarget.clipIndex}
          clip={editorTarget.clip}
          onBack={() => setEditorTarget(null)}
        />
      )}

      {/* schedule placeholder toast */}
      {scheduling && (
        <div className="nc-anim-pop-in" style={{
          position: 'fixed', bottom: 90, left: '50%', transform: 'translateX(-50%)', zIndex: 80,
          background: 'var(--nc-panel)', border: '1px solid var(--nc-border-strong)',
          borderRadius: 12, padding: '10px 16px', fontSize: 13,
          boxShadow: '0 12px 40px rgba(0,0,0,0.5)',
        }}>
          📅 Scheduling arrives in Phase D — clip saved for now.
        </div>
      )}
    </div>
  );
}
