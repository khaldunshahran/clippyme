/**
 * UI-01 offline fixture — stubbed useChat.
 *
 * Provides a small static thread with two clips and a selectable clip.
 * The REAL AppShell, Sidebar, ChatThread/WelcomeView, SelectionBar and
 * Composer all render against this. No network, no production data.
 */
import { useCallback, useEffect, useRef, useState } from 'react';

function makeThreads() {
  return [
    {
      id: 't1',
      title: 'Fixture thread',
      jobId: 'j1',
      phase: 'idle',
      messages: [{ id: 'm1', role: 'user', text: 'Fixture: make the hook punchier' }],
      clips: [
        { index: 0, title: 'Fixture clip one', raw: null },
        { index: 1, title: 'Fixture clip two', raw: null },
      ],
      settings: {},
      validation: null,
    },
  ];
}

export function useChat() {
  const [threads] = useState(makeThreads);
  const [activeId] = useState('t1');
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [gridOpen, setGridOpen] = useState(false);
  const [popupClip, setPopupClip] = useState(null);
  const params = new URLSearchParams(window.location.search);
  const [selectedClip, setSelectedClip] = useState(() =>
    params.get('sel') === '0' ? null : { threadId: 't1', jobId: 'j1', clipIndex: 0 }
  );
  const workspaceRef = useRef(null);

  const active = threads.find((t) => t.id === activeId) || null;

  const selectClip = useCallback((threadId, index) => {
    const t = makeThreads()[0];
    setSelectedClip({ threadId, jobId: t.jobId, clipIndex: index });
  }, []);
  const clearSelection = useCallback(() => setSelectedClip(null), []);
  const createThread = useCallback(() => 't1', []);
  const selectThread = useCallback(() => {}, []);
  const deleteThread = useCallback(() => {}, []);
  const sendMessage = useCallback(async () => {}, []);
  const applyProposal = useCallback(() => {}, []);
  const discardProposal = useCallback(() => {}, []);
  const confirmSettings = useCallback(() => {}, []);
  const proceedToSettings = useCallback(() => {}, []);
  const patchThread = useCallback(() => {}, []);
  const startBulkEdit = useCallback(() => {}, []);

  // Driver hook: lets the scripted browser test switch clips.
  useEffect(() => {
    window.__ui01Chat = {
      selectClip,
      clearSelection,
      getSelectedClip: () => selectedClip,
    };
  }, [selectClip, clearSelection, selectedClip]);

  return {
    threads,
    active,
    activeId,
    settingsOpen,
    setSettingsOpen,
    gridOpen,
    setGridOpen,
    popupClip,
    setPopupClip,
    selectedClip,
    selectClip,
    clearSelection,
    workspaceRef,
    createThread,
    selectThread,
    deleteThread,
    sendMessage,
    applyProposal,
    discardProposal,
    confirmSettings,
    proceedToSettings,
    patchThread,
    startBulkEdit,
    DEFAULT_SETTINGS: {},
  };
}
