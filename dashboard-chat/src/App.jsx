import { useCallback, useState } from 'react';
import { Menu, CalendarClock, Loader2, X, Check } from 'lucide-react';
import { useChat } from './chat/useChat.js';
import Sidebar from './chat/Sidebar.jsx';
import WelcomeView from './chat/WelcomeView.jsx';
import ChatThread from './chat/ChatThread.jsx';
import Composer from './chat/Composer.jsx';
import SettingsPopup from './chat/SettingsPopup.jsx';
import ClipGrid from './chat/ClipGrid.jsx';
import ClipPopup from './chat/ClipPopup.jsx';
import EditorView from './chat/EditorView.jsx';
import AuthGate from './chat/AuthGate.jsx';
import { scheduleClipPost, getZernioAccounts, zernioTargets, getJobClips, normClip } from './api/chatApi.js';

const DEPLOY_SHA = (typeof import.meta !== 'undefined' && import.meta.env && import.meta.env.VITE_DEPLOY_SHA) || '';
const DEPLOY_TIME = (typeof import.meta !== 'undefined' && import.meta.env && import.meta.env.VITE_DEPLOY_TIME) || '';

function DeployBadge() {
  if (!DEPLOY_SHA) return null;
  const short = DEPLOY_SHA.slice(0, 7);
  const title = `nugget-chat deploy\n${DEPLOY_SHA}${DEPLOY_TIME ? `\nBuilt ${DEPLOY_TIME}` : ''}`;
  return (
    <span
      title={title}
      style={{
        fontSize: 10.5, fontWeight: 700, letterSpacing: 0.4,
        color: 'var(--nc-text-faint)', background: 'rgba(255,255,255,0.05)',
        border: '1px solid var(--nc-border)', borderRadius: 99, padding: '3px 9px',
        fontVariantNumeric: 'tabular-nums', whiteSpace: 'nowrap', cursor: 'default',
      }}
    >
      DEPLOY {short}
    </span>
  );
}

function AppShell() {
  const chat = useChat();
  const {
    threads, active, activeId,
    settingsOpen, setSettingsOpen,
    gridOpen, setGridOpen,
    popupClip, setPopupClip,
    editorTarget, setEditorTarget,
    createThread, selectThread, deleteThread,
    sendMessage, confirmSettings, proceedToSettings, patchThread,
    DEFAULT_SETTINGS,
  } = chat;

  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [schedTarget, setSchedTarget] = useState(null); // { threadId, jobId, index } — schedule dialog open
  const [schedWhen, setSchedWhen] = useState('');
  const [schedBusy, setSchedBusy] = useState(false);
  const [schedMsg, setSchedMsg] = useState(null);

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

  const refreshClips = useCallback(async (threadId, jobId) => {
    try {
      const raw = await getJobClips(jobId);
      patchThread(threadId, { clips: raw.map((c, i) => normClip(c, i)) });
    } catch { /* keep existing clips on refresh failure */ }
  }, [patchThread]);

  const handleDuplicate = useCallback((threadId, jobId) => {
    refreshClips(threadId, jobId);
  }, [refreshClips]);

  const openSchedule = useCallback((threadId, index) => {
    const t = threads.find((x) => x.id === threadId);
    if (!t?.jobId) return;
    // default: tomorrow 18:00 local
    const d = new Date();
    d.setDate(d.getDate() + 1); d.setHours(18, 0, 0, 0);
    const pad = (n) => String(n).padStart(2, '0');
    setSchedWhen(`${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}T${pad(d.getHours())}:${pad(d.getMinutes())}`);
    setSchedMsg(null);
    setSchedTarget({ threadId, jobId: t.jobId, index });
  }, [threads]);

  const doSchedule = useCallback(async () => {
    if (!schedTarget || schedBusy) return;
    if (!schedWhen) { setSchedMsg({ ok: false, text: 'Pick a date & time.' }); return; }
    setSchedBusy(true); setSchedMsg(null);
    try {
      const r = await getZernioAccounts();
      const targets = zernioTargets(r?.accounts || r);
      if (!targets.length) throw new Error('No connected social accounts found in Zernio.');
      const res = await scheduleClipPost(schedTarget.jobId, schedTarget.index, {
        scheduled_for: new Date(schedWhen).toISOString(),
        platforms: targets,
        schedule_mode: 'manual',
      });
      setSchedMsg({ ok: true, text: `Scheduled for ${res.scheduled_for || schedWhen}.` });
      setTimeout(() => { setSchedTarget(null); setSchedMsg(null); }, 2200);
    } catch (e) {
      setSchedMsg({ ok: false, text: `Schedule failed: ${e.message}` });
    } finally {
      setSchedBusy(false);
    }
  }, [schedTarget, schedWhen, schedBusy]);

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
          <div style={{ marginLeft: 'auto', display: 'flex', alignItems: 'center' }}>
            <DeployBadge />
          </div>
        </header>

        {showWelcome ? (
          <WelcomeView onExample={(url) => { const id = ensureThread(); sendMessage(url); }} />
        ) : (
          <ChatThread
            thread={active}
            onOpenClip={(i) => openClip(activeId, i)}
            onEditClip={(i) => openEditor(activeId, i)}
            onScheduleClip={(i) => openSchedule(activeId, i)}
            onViewAll={() => setGridOpen(true)}
            onCaptionSelect={(id) => activeId && patchThread(activeId, { settings: { ...(active?.settings || {}), captions: id } })}
            onCaptionContinue={() => activeId && proceedToSettings(activeId)}
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
          onScheduleClip={(i) => openSchedule(activeId, i)}
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
          onDuplicate={() => handleDuplicate(popupThread.id, popupThread.jobId)}
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

      {/* schedule dialog */}
      {schedTarget && (
        <div className="nc-modal-backdrop" onClick={() => { if (!schedBusy) setSchedTarget(null); }} style={{ zIndex: 70 }}>
          <div
            className="nc-modal nc-card nc-anim-pop-in"
            onClick={(e) => e.stopPropagation()}
            role="dialog" aria-label="Schedule clip"
            style={{ width: 'min(380px, calc(100vw - 40px))', background: 'var(--nc-bg-soft)', borderRadius: 16, padding: 18 }}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4 }}>
              <CalendarClock size={16} />
              <div style={{ fontSize: 14, fontWeight: 700 }}>Schedule clip {schedTarget.index + 1}</div>
              <button className="nc-icon-btn" onClick={() => setSchedTarget(null)} aria-label="Close" style={{ marginLeft: 'auto' }}>
                <X size={16} />
              </button>
            </div>
            <div style={{ fontSize: 12.5, color: 'var(--nc-text-dim)', marginBottom: 12 }}>
              Posts to your connected accounts at this time via Zernio.
            </div>
            <input
              type="datetime-local"
              className="nc-input"
              value={schedWhen}
              onChange={(e) => setSchedWhen(e.target.value)}
              style={{ width: '100%', marginBottom: 12 }}
            />
            <button className="nc-btn-primary" style={{ width: '100%' }} disabled={schedBusy} onClick={doSchedule}>
              {schedBusy ? <Loader2 size={15} style={{ animation: 'nc-spin 1s linear infinite' }} /> : <CalendarClock size={15} />}
              {schedBusy ? 'Scheduling…' : 'Confirm schedule'}
            </button>
            {schedMsg && (
              <div style={{ marginTop: 8, fontSize: 12.5, color: schedMsg.ok ? 'var(--nc-green)' : 'var(--nc-red)', display: 'flex', gap: 6, alignItems: 'center' }}>
                {schedMsg.ok && <Check size={13} />}{schedMsg.text}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

export default function App() {
  return (
    <AuthGate>
      <AppShell />
    </AuthGate>
  );
}
