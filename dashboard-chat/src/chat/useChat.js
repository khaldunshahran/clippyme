import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  validateUrl, submitJob, getProgress, steerJob, getHistory,
  getJobClips, extractUrl, normClip,
} from '../api/chatApi.js';

const LS_KEY = 'nugget-chat-threads-v1';
const POLL_MS = 2000;

const uid = () => `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;

export const DEFAULT_SETTINGS = {
  model: '',            // '' = backend default
  genre: '',            // -> clip_type
  clipStyle: '',        // -> instructions hint
  aspect: '9:16',
  clipLength: 'all',    // all | shorts | mid | long | custom (system duration modes)
  captions: '',         // caption_style_default preset ('' = system auto-chooses)
  timeframe: null,      // {start, end} seconds | null
  customMin: 15,        // custom clip-length bounds (seconds)
  customMax: 60,
};

// System duration modes (old dashboard create.jsx realApi.js):
// shorts 15-60s, mid 60-180s, long 180-600s, all = diverse mix, custom = user bounds.
const LENGTH_PRESETS = {
  all: { duration_mode: 'all' },
  shorts: { duration_mode: 'shorts', min_duration: 15, max_duration: 60 },
  mid: { duration_mode: 'mid', min_duration: 60, max_duration: 180 },
  long: { duration_mode: 'long', min_duration: 180, max_duration: 600 },
  custom: { duration_mode: 'custom' },
};

function newThread() {
  return {
    id: uid(),
    title: 'New chat',
    createdAt: Date.now(),
    updatedAt: Date.now(),
    phase: 'idle', // idle|validating|settings|submitting|clipping|done|error
    url: null,
    validation: null, // {valid, downloadable, title, duration, thumbnail, reason}
    settings: { ...DEFAULT_SETTINGS },
    jobId: null,
    progress: null,
    clips: [],
    messages: [],
    steerLog: [],
    error: null,
  };
}

function loadThreads() {
  try {
    const raw = localStorage.getItem(LS_KEY);
    if (!raw) return [];
    const arr = JSON.parse(raw);
    return Array.isArray(arr) ? arr : [];
  } catch { return []; }
}

export function useChat() {
  const [threads, setThreads] = useState(loadThreads);
  const [activeId, setActiveId] = useState(() => loadThreads()[0]?.id || null);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [gridOpen, setGridOpen] = useState(false);
  const [popupClip, setPopupClip] = useState(null); // {threadId, index}
  const [editorTarget, setEditorTarget] = useState(null); // {threadId, jobId, clipIndex, clip}
  const pollRef = useRef({});

  const active = useMemo(
    () => threads.find((t) => t.id === activeId) || null,
    [threads, activeId],
  );

  // persist
  useEffect(() => {
    try { localStorage.setItem(LS_KEY, JSON.stringify(threads.slice(0, 50))); } catch {}
  }, [threads]);

  const patchThread = useCallback((id, patch) => {
    setThreads((ts) => ts.map((t) => (t.id === id ? { ...t, ...patch, updatedAt: Date.now() } : t)));
  }, []);

  const pushMsg = useCallback((id, msg) => {
    const m = { id: uid(), ts: Date.now(), ...msg };
    setThreads((ts) => ts.map((t) => (t.id === id
      ? { ...t, updatedAt: Date.now(), messages: [...t.messages, m] }
      : t)));
    return m.id;
  }, []);

  const createThread = useCallback(() => {
    const t = newThread();
    setThreads((ts) => [t, ...ts]);
    setActiveId(t.id);
    return t.id;
  }, []);

  const selectThread = useCallback((id) => {
    setActiveId(id);
    setGridOpen(false);
    setPopupClip(null);
  }, []);

  const deleteThread = useCallback((id) => {
    setThreads((ts) => {
      const next = ts.filter((t) => t.id !== id);
      setActiveId((cur) => (cur === id ? next[0]?.id || null : cur));
      return next;
    });
  }, []);

  // ---------------- validation ----------------
  const runValidation = useCallback(async (threadId, url) => {
    patchThread(threadId, { phase: 'validating', url, validation: null, error: null });
    const msgId = pushMsg(threadId, { role: 'ai', kind: 'validate', data: { state: 'checking', url } });
    try {
      const v = await validateUrl(url);
      const validation = {
        valid: !!v.valid,
        downloadable: !!v.downloadable,
        title: v.title || null,
        duration: v.duration ?? null,
        thumbnail: v.thumbnail || null,
        reason: v.reason || null,
      };
      patchThread(threadId, { validation });
      setThreads((ts) => ts.map((t) => (t.id === threadId
        ? { ...t, messages: t.messages.map((m) => (m.id === msgId ? { ...m, data: { state: 'done', url, validation } } : m)) }
        : t)));
      if (validation.valid && validation.downloadable) {
        const title = validation.title || url;
        patchThread(threadId, { phase: 'caption-pick', title: title.length > 42 ? `${title.slice(0, 42)}…` : title });
        pushMsg(threadId, { role: 'ai', kind: 'caption-picker', data: { validation } });
      } else {
        pushMsg(threadId, {
          role: 'ai', kind: 'error',
          text: validation.reason
            ? `I couldn't use that link: ${validation.reason}`
            : 'That link doesn\'t look downloadable. Try a different video URL.',
        });
        patchThread(threadId, { phase: 'idle' });
      }
    } catch (e) {
      setThreads((ts) => ts.map((t) => (t.id === threadId
        ? { ...t, messages: t.messages.map((m) => (m.id === msgId ? { ...m, data: { state: 'error', url, error: e.message } } : m)) }
        : t)));
      pushMsg(threadId, { role: 'ai', kind: 'error', text: `Validation failed: ${e.message}` });
      patchThread(threadId, { phase: 'idle' });
    }
  }, [patchThread, pushMsg]);

  // ---------------- submit ----------------
  const buildPayload = (thread) => {
    const s = thread.settings;
    const len = { ...(LENGTH_PRESETS[s.clipLength] || LENGTH_PRESETS.all) };
    if (s.clipLength === 'custom') {
      const mn = Math.min(900, Math.max(5, Number(s.customMin) || 15));
      const mx = Math.min(900, Math.max(5, Number(s.customMax) || 60));
      len.min_duration = Math.min(mn, mx);
      len.max_duration = Math.max(mn, mx);
    }
    const payload = {
      url: thread.url,
      aspect: s.aspect || '9:16',
      ...len,
    };
    if (s.model) payload.model = s.model;
    if (s.genre) payload.clip_type = s.genre;
    const hints = [];
    if (s.clipStyle) hints.push(`Clip style: ${s.clipStyle}.`);
    if (s.genre) hints.push(`Genre: ${s.genre}.`);
    if (hints.length) payload.instructions = hints.join(' ');
    if (s.captions) payload.caption_style_default = s.captions;
    if (s.timeframe && s.timeframe.end > s.timeframe.start) {
      payload.source_timeframe = { start: s.timeframe.start, end: s.timeframe.end };
    }
    return payload;
  };

  const stopPoll = useCallback((threadId) => {
    const h = pollRef.current[threadId];
    if (h) { clearInterval(h); delete pollRef.current[threadId]; }
  }, []);

  const startPoll = useCallback((threadId, jobId) => {
    stopPoll(threadId);
    const tick = async () => {
      try {
        const p = await getProgress(jobId);
        patchThread(threadId, { progress: p });
        const failed = p && (p.stage === 'failed' || p.stage === 'error');
        if (failed) {
          stopPoll(threadId);
          pushMsg(threadId, { role: 'ai', kind: 'error', text: `The clip job failed${p.detail ? `: ${p.detail}` : ''}. Nothing was rendered \u2014 you can try again with a different link.` });
          patchThread(threadId, { phase: 'error', error: 'job failed' });
          return;
        }
        const done = p && (p.progress >= 100 || p.stage === 'completed' || p.stage === 'done');
        if (done) {
          stopPoll(threadId);
          try {
            const raw = await getJobClips(jobId);
            const clips = raw.map((c, i) => normClip(c, i));
            patchThread(threadId, { phase: 'done', clips });
            if (clips.length > 0) {
              pushMsg(threadId, { role: 'ai', kind: 'clips', data: { jobId } });
            } else {
              pushMsg(threadId, { role: 'ai', kind: 'notice', text: 'The job finished, but I couldn\'t find any clips in its results. Check the classic dashboard\'s history for this job.' });
            }
          } catch (e) {
            pushMsg(threadId, { role: 'ai', kind: 'error', text: `Clips finished but I couldn't load them: ${e.message}` });
            patchThread(threadId, { phase: 'done' });
          }
        }
      } catch {
        // transient poll failure — keep polling; the pill shows last-known state
      }
    };
    tick();
    pollRef.current[threadId] = setInterval(tick, POLL_MS);
  }, [patchThread, pushMsg, stopPoll]);

  const proceedToSettings = useCallback((threadId) => {
    patchThread(threadId, { phase: 'settings' });
    setSettingsOpen(true);
  }, [patchThread]);

  const confirmSettings = useCallback(async (threadId, settings) => {
    const thread = threads.find((t) => t.id === threadId);
    if (!thread || !thread.url) return;
    setSettingsOpen(false);
    patchThread(threadId, { settings: { ...settings }, phase: 'submitting', error: null });
    pushMsg(threadId, { role: 'ai', kind: 'progress', data: { state: 'submitting' } });
    try {
      const payload = buildPayload({ ...thread, settings });
      const res = await submitJob(payload);
      const jobId = res.job_id;
      if (!jobId) throw new Error('backend did not return a job id');
      const v = thread.validation || {};
      patchThread(threadId, {
        jobId, phase: 'clipping',
        title: (v.title || thread.title || 'Clipping job').slice(0, 48),
      });
      startPoll(threadId, jobId);
    } catch (e) {
      pushMsg(threadId, { role: 'ai', kind: 'error', text: `Couldn't start the job: ${e.message}` });
      patchThread(threadId, { phase: 'error', error: e.message });
    }
  }, [threads, patchThread, pushMsg, startPoll]);

  // ---------------- send / steer ----------------
  const sendMessage = useCallback(async (text) => {
    let threadId = activeId;
    if (!threadId) threadId = createThread();
    const thread = threads.find((t) => t.id === threadId) || { phase: 'idle' };
    const clean = (text || '').trim();
    if (!clean) return;
    pushMsg(threadId, { role: 'user', kind: 'text', text: clean });

    // While clipping, a message is a steering prompt for the AI.
    if ((thread.phase === 'clipping' || thread.phase === 'submitting') && thread.jobId) {
      try {
        const r = await steerJob(thread.jobId, clean);
        if (r && r.accepted) {
          pushMsg(threadId, { role: 'ai', kind: 'notice', text: 'Got it — I\'ll steer the clip selection with that in mind.' });
        } else {
          pushMsg(threadId, { role: 'ai', kind: 'notice', text: r?.reason || 'That arrived past the steering point — it will apply to your next job.' });
        }
      } catch (e) {
        // 409 = past steering point; surface honestly
        const reason = e?.data?.reason || e.message;
        pushMsg(threadId, { role: 'ai', kind: 'notice', text: `Couldn't steer this job: ${reason}` });
      }
      return;
    }

    const url = extractUrl(clean);
    if (url) {
      // A fresh link starts a fresh flow on this thread (reset prior job state).
      stopPoll(threadId);
      patchThread(threadId, { jobId: null, clips: [], progress: null });
      runValidation(threadId, url);
    } else if (thread.phase === 'idle') {
      pushMsg(threadId, { role: 'ai', kind: 'text', text: 'Paste a video link (YouTube, Twitch, …) and I\'ll check it, then we\'ll set it up together.' });
    }
  }, [activeId, threads, createThread, pushMsg, patchThread, runValidation, stopPoll]);

  // Resume polling for threads that were mid-clip when the app loaded.
  useEffect(() => {
    threads.forEach((t) => {
      if (t.phase === 'clipping' && t.jobId && !pollRef.current[t.id]) startPoll(t.id, t.jobId);
    });
    return () => Object.values(pollRef.current).forEach(clearInterval);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Backfill sidebar from server history (titles for threads lacking them).
  useEffect(() => {
    let cancelled = false;
    getHistory().then((h) => {
      if (cancelled || !h) return;
      const items = Array.isArray(h) ? h : (h.jobs || h.history || []);
      if (!items.length) return;
      setThreads((ts) => ts.map((t) => {
        if (t.jobId && (t.title === 'New chat' || t.title === 'Clipping job')) {
          const hit = items.find((j) => (j.job_id || j.id) === t.jobId);
          const title = hit && (hit.title || hit.url);
          if (title) return { ...t, title: String(title).slice(0, 48) };
        }
        return t;
      }));
    }).catch(() => {});
    return () => { cancelled = true; };
  }, []);

  return {
    threads, active, activeId,
    settingsOpen, setSettingsOpen,
    gridOpen, setGridOpen,
    popupClip, setPopupClip,
    editorTarget, setEditorTarget,
    createThread, selectThread, deleteThread,
    sendMessage, confirmSettings, proceedToSettings, patchThread,
    DEFAULT_SETTINGS,
  };
}
