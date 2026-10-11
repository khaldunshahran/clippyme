/**
 * UI-01 offline fixture — stubbed useChat.
 *
 * Provides a small static thread with three clips and a selectable clip.
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
        // Clip one is production-shaped (review-03): video_url is the
        // pipeline's REFRAMED RESULT file and composed_video_url a
        // rendered artifact — neither is the raw source. The draft
        // player must select the project's recorded source slice
        // (project.source.file, served at videos/j1/ in the dist).
        {
          index: 0,
          title: 'Fixture clip one',
          raw: {
            video_title_for_youtube_short: 'Fixture clip one',
            video_url: 'fixture-result-v3.mp4',
            composed_video_url: 'fixture-composed-v3.mp4',
          },
        },
        {
          index: 1,
          title: 'Fixture clip two',
          raw: {
            video_title_for_youtube_short: 'Fixture clip two',
            video_url: 'fixture-source-2.mp4',
            composed_video_url: null,
          },
        },
        // Clip three (review-03): no result file, only the composed
        // artifact, and its project records no source file (legacy
        // shape) — the player must present the composed file honestly
        // and keep source seeking disabled while it shows.
        {
          index: 2,
          title: 'Fixture clip three',
          raw: {
            video_title_for_youtube_short: 'Fixture clip three',
            video_url: null,
            composed_video_url: 'fixture-composed-v3.mp4',
          },
        },
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
