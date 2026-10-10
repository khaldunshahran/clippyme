import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  validateUrl, submitJob, getProgress, steerJob, getHistory,
  getJobClips, extractUrl, normClip, editAiPatch,
} from '../api/chatApi.js';
import { onSessionReset } from '../api/sessionReset.js';
import { routeChatMessage } from './messageRouter.js';
import { requestChatProposal, applyChatProposal } from './proposals.js';
import {
  loadSelection, saveSelection, clearStoredSelection, isValidSelection,
} from './selection.js';

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

// A persisted selection is only restored when its thread still exists with
// the same job and clip — otherwise it is dropped, never resurrected.
function loadValidSelection() {
  const sel = loadSelection();
  if (!isValidSelection(sel)) return null;
  const t = loadThreads().find((x) => x.id === sel.threadId);
  if (!t || t.jobId !== sel.jobId || !Array.isArray(t.clips) || !t.clips[sel.clipIndex]) {
    clearStoredSelection();
    return null;
  }
  return sel;
}

// A proposal message is actionable only while its selection session is
// alive. These are the non-terminal proposal states.
const PROPOSAL_OPEN = (s) => s === 'thinking' || s === 'ready';

// A proposal whose lifetime cannot be established is never left actionable:
// it stays in history as 'stale' instead of 'thinking'/'ready'.
// Conversation history is retained; the payload is not actionable.
function staleOpenProposals(thread, reason) {
  if (!thread || !Array.isArray(thread.messages)) return thread;
  let changed = false;
  const messages = thread.messages.map((m) => {
    if (m && m.kind === 'proposal' && PROPOSAL_OPEN(m.data?.state)) {
      changed = true;
      return { ...m, data: { ...(m.data || {}), state: 'stale', staleReason: reason } };
    }
    return m;
  });
  return changed ? { ...thread, updatedAt: Date.now(), messages } : thread;
}

// Coherent restore: the persisted selection and the active thread are ONE
// identity, and proposals restored from storage are never actionable (their
// lifetime belongs to a previous page session). A valid restored selection
// activates its OWN thread — never the newest thread paired with another
// thread's clip/editor. An invalid selection is dropped, never resurrected.
function loadRestoredIdentity() {
  const threads = loadThreads().map((t) => staleOpenProposals(t, 'session-restart'));
  const selection = loadValidSelection();
  return {
    threads,
    selection,
    activeId: selection?.threadId || threads[0]?.id || null,
  };
}

export function useChat() {
  const [restored] = useState(loadRestoredIdentity);
  const [threads, setThreads] = useState(restored.threads);
  const [activeId, setActiveId] = useState(restored.activeId);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [gridOpen, setGridOpen] = useState(false);
  const [popupClip, setPopupClip] = useState(null); // {threadId, index}
  // Task 17 — explicit single-clip selection: {threadId, jobId, clipIndex}.
  // Safe metadata only (persisted via selection.js); the live project
  // controller for it is owned by App's useWorkspaceController and shared
  // through workspaceRef (never stored in state or localStorage).
  const [selectedClip, setSelectedClip] = useState(restored.selection);
  // Selection-session generation (review-01): every proposal is born with
  // the current generation; any end of the selection's lifetime (session
  // reset, clear, thread/selection switch, new conversation) bumps it. A
  // proposal from an older generation can never become ready or be applied
  // — not even by reselecting identical ids.
  const selGenRef = useRef(0);
  const workspaceRef = useRef(null); // { controller } — wired by App
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

  // Patch one message's data in place (proposal state machine).
  const updateMsg = useCallback((threadId, msgId, patch) => {
    setThreads((ts) => ts.map((t) => (t.id === threadId
      ? {
        ...t,
        updatedAt: Date.now(),
        messages: t.messages.map((m) => {
          if (m.id !== msgId) return m;
          const p = typeof patch === 'function' ? patch(m) : patch;
          return { ...m, ...p, data: { ...(m.data || {}), ...((p && p.data) || {}) } };
        }),
      }
      : t)));
  }, []);


  // Mark open proposals ('thinking'/'ready') as 'stale'. threadIds=null
  // sweeps every thread (session reset). History is retained; the payloads
  // stop being actionable. A terminal card is never touched.
  const invalidateProposals = useCallback((threadIds, reason) => {
    setThreads((ts) => {
      let changed = false;
      const next = ts.map((t) => {
        if (threadIds && !threadIds.includes(t.id)) return t;
        const r = staleOpenProposals(t, reason);
        if (r !== t) changed = true;
        return r;
      });
      return changed ? next : ts;
    });
  }, []);

  const clearSelection = useCallback(() => {
    const prev = selectedClip;
    setSelectedClip(null);
    clearStoredSelection();
    if (prev) {
      // The selection's lifetime ends here: its pending/ready proposals die
      // with it and can never be revived by reselecting identical ids.
      selGenRef.current += 1;
      invalidateProposals([prev.threadId], 'selection-changed');
    }
  }, [selectedClip, invalidateProposals]);

  const selectClip = useCallback((threadId, clipIndex) => {
    const t = threads.find((x) => x.id === threadId);
    if (!t?.jobId || !Array.isArray(t.clips) || !t.clips[clipIndex]) return;
    const prev = selectedClip;
    const same = !!prev
      && prev.threadId === threadId
      && prev.jobId === t.jobId
      && prev.clipIndex === clipIndex;
    const sel = { threadId, jobId: t.jobId, clipIndex };
    setSelectedClip(sel);
    saveSelection(sel);
    if (!same) {
      // A new selection session: the previous lifetime (and its proposals)
      // ends, so identical ids can never revive an old proposal.
      selGenRef.current += 1;
      if (prev) invalidateProposals([prev.threadId], 'selection-changed');
    }
    setActiveId(threadId);
    setGridOpen(false);
    setPopupClip(null);
  }, [threads, selectedClip, invalidateProposals]);

  const createThread = useCallback(() => {
    const t = newThread();
    setThreads((ts) => [t, ...ts]);
    setActiveId(t.id);
    // A new conversation starts with no selection — the previous
    // selection's lifetime (and its proposals) ends here.
    if (selectedClip) {
      selGenRef.current += 1;
      invalidateProposals([selectedClip.threadId], 'selection-changed');
    }
    setSelectedClip(null);
    clearStoredSelection();
    return t.id;
  }, [selectedClip, invalidateProposals]);

  const selectThread = useCallback((id) => {
    setActiveId(id);
    setGridOpen(false);
    setPopupClip(null);
    // A selection belongs to exactly one thread: switching threads drops a
    // selection that belongs to another thread, ending its lifetime.
    if (selectedClip && selectedClip.threadId !== id) {
      const prevThread = selectedClip.threadId;
      setSelectedClip(null);
      clearStoredSelection();
      selGenRef.current += 1;
      invalidateProposals([prevThread], 'selection-changed');
    }
  }, [selectedClip, invalidateProposals]);

  const deleteThread = useCallback((id) => {
    if (selectedClip && selectedClip.threadId === id) {
      setSelectedClip(null);
      clearStoredSelection();
      // The thread (and its proposals) is gone; still end the lifetime so
      // an in-flight request for it can never settle as ready.
      selGenRef.current += 1;
    }
    setThreads((ts) => {
      const next = ts.filter((t) => t.id !== id);
      setActiveId((cur) => (cur === id ? next[0]?.id || null : cur));
      return next;
    });
  }, [selectedClip]);

  // Session reset (sign-out / user change): the workspace controller resets
  // itself via its own registration; here the selection is dropped and every
  // open proposal in every thread is invalidated, so a proposal can never be
  // applied against another user's clip — or revive afterwards.
  useEffect(() => onSessionReset(() => {
    selGenRef.current += 1;
    setSelectedClip(null);
    clearStoredSelection();
    invalidateProposals(null, 'session-reset');
  }), [invalidateProposals]);

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
    let fails = 0;
    let failNoticed = false;
    const tick = async () => {
      try {
        const p = await getProgress(jobId);
        fails = 0;
        failNoticed = false;
        patchThread(threadId, { progress: p });
        const failed = p && (p.stage === 'failed' || p.stage === 'error');
        if (failed) {
          stopPoll(threadId);
          pushMsg(threadId, { role: 'ai', kind: 'error', text: `The clip job failed${p.detail ? `: ${p.detail}` : ''}. Nothing was rendered \u2014 you can try again with a different link.` });
          patchThread(threadId, { phase: 'error', error: p.detail ? `The clip job failed: ${p.detail}` : 'The clip job failed.' });
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
        // Transient poll failure: keep polling, but after 3 in a row say so
        // instead of failing silently forever (e.g. a proxy serving HTML).
        fails += 1;
        if (fails >= 3 && !failNoticed) {
          failNoticed = true;
          pushMsg(threadId, { role: 'ai', kind: 'notice', text: "I'm having trouble reaching the clip server - I'll keep trying in the background." });
        }
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
    // No separate progress message: the validation card morphs in place
    // (ValidateMsg renders ProgressBody once phase is submitting/clipping).
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

  // ---------------- chat proposals ----------------
  // Ordinary edit for the selected clip: generate a validated patch proposal
  // (never saved), then let the user apply it explicitly. The proposal is
  // bound to the captured identity + base version AND the selection-session
  // generation from the start: a reset or selection change in between ends
  // its lifetime, and the card settles to 'stale' instead of 'ready'.
  const runChatEdit = useCallback(async (thread, selection, instruction) => {
    const threadId = thread.id;
    const identity = { jobId: selection.jobId, clipIndex: selection.clipIndex };
    const birthGen = selGenRef.current;
    const clipTitle = thread.clips?.[selection.clipIndex]?.title || null;
    const controller = workspaceRef.current?.controller || null;
    if (!controller) {
      pushMsg(threadId, { role: 'ai', kind: 'error', text: 'The editor isn\'t ready for that clip yet — try again in a moment.' });
      return;
    }
    const msgId = pushMsg(threadId, {
      role: 'ai', kind: 'proposal',
      data: { state: 'thinking', text: instruction, identity, clipTitle, gen: birthGen, threadId },
    });
    // Settle the card only while it is still pending AND its selection
    // session is still the live one. A terminal card (discarded by the user,
    // or invalidated by a reset/selection change) is never overwritten by a
    // late response.
    const settle = (patch) => updateMsg(threadId, msgId, (m) => {
      if (!m || m.kind !== 'proposal' || m.data?.state !== 'thinking') return {};
      if (selGenRef.current !== birthGen) {
        return { data: { state: 'stale', staleReason: 'selection-changed' } };
      }
      return patch;
    });
    try {
      const res = await requestChatProposal({
        editAiPatchFn: editAiPatch,
        jobId: identity.jobId,
        clipIndex: identity.clipIndex,
        instruction,
      });
      settle({
        data: {
          state: 'ready',
          explanation: res.explanation,
          patch: res.patch,
          project: res.project,
          identity: res.identity,
          baseVersion: res.baseVersion,
          gen: birthGen,
          threadId,
        },
      });
    } catch (e) {
      settle({
        data: { state: 'error', error: e?.message || 'The AI edit failed. Nothing was changed — try again.' },
      });
    }
  }, [pushMsg, updateMsg]);

  // Apply a ready proposal through the shared controller. Identity, version
  // AND selection-session lifetime are re-checked here: a clip/thread switch,
  // a session reset, or reselecting identical ids after any of those refuses
  // instead of touching the wrong (or a dead) target.
  const applyProposal = useCallback(async (threadId, msgId) => {
    const thread = threads.find((t) => t.id === threadId);
    const msg = thread?.messages.find((m) => m.id === msgId);
    const d = msg?.data;
    if (!msg || msg.kind !== 'proposal' || !d || d.state !== 'ready') return;
    // Lifetime: the proposal belongs to the selection session that created
    // it. Anything else — including identical ids after a reset — refuses
    // with zero api calls and marks the card honestly.
    const sel = selectedClip;
    const lifetimeOk = d.gen === selGenRef.current
      && sel && sel.threadId === threadId && sel.threadId === d.threadId
      && sel.jobId === d.identity?.jobId && sel.clipIndex === d.identity?.clipIndex;
    if (!lifetimeOk) {
      updateMsg(threadId, msgId, { data: { state: 'stale', staleReason: 'selection-changed' } });
      return;
    }
    updateMsg(threadId, msgId, { data: { state: 'applying' } });
    const controller = workspaceRef.current?.controller || null;
    try {
      await applyChatProposal({
        controller,
        identity: d.identity,
        baseVersion: d.baseVersion,
        project: d.project,
        label: `chat: ${(d.text || 'edit').slice(0, 60)}`,
      });
      updateMsg(threadId, msgId, { data: { state: 'applied' } });
    } catch (e) {
      if (e?.code === 'STALE_SELECTION') {
        updateMsg(threadId, msgId, { data: { state: 'stale', staleReason: 'selection-changed' } });
        pushMsg(threadId, { role: 'ai', kind: 'notice', text: 'The selection changed since that proposal was made, so nothing was applied. Select the clip again and send the edit to regenerate it.' });
      } else if (e?.status === 409 || /stale project version/i.test(e?.message || '')) {
        updateMsg(threadId, msgId, { data: { state: 'stale', staleReason: 'version', baseVersion: d.baseVersion } });
        pushMsg(threadId, {
          role: 'ai', kind: 'notice',
          text: `That proposal was based on v${d.baseVersion} but the clip has newer edits — nothing was overwritten. Send the edit again and I'll regenerate it against the latest version.`,
        });
      } else if (e?.code === 'DISPOSED' || e?.code === 'QUEUE_CANCELLED') {
        updateMsg(threadId, msgId, { data: { state: 'error', error: 'The editor was reset while applying — nothing was applied. Try again.' } });
      } else {
        updateMsg(threadId, msgId, { data: { state: 'error', error: `Apply failed: ${e?.message || 'unknown error'}. Nothing was overwritten.` } });
      }
    }
  }, [threads, selectedClip, updateMsg, pushMsg]);



  const discardProposal = useCallback((threadId, msgId) => {
    updateMsg(threadId, msgId, { data: { state: 'discarded' } });
  }, [updateMsg]);


  // ---------------- send / steer ----------------
  const sendMessage = useCallback(async (text) => {
    let threadId = activeId;
    if (!threadId) threadId = createThread();
    const thread = threads.find((t) => t.id === threadId) || { phase: 'idle' };
    const clean = (text || '').trim();
    if (!clean) return;
    pushMsg(threadId, { role: 'user', kind: 'text', text: clean });

    // A selection belongs to exactly one thread AND its live job/clip:
    // thread, job and clip are validated together — never leak a selection
    // across threads or route it against a stale job.
    const sel = selectedClip;
    const selection = (
      sel
      && sel.threadId === threadId
      && thread.jobId === sel.jobId
      && Array.isArray(thread.clips)
      && thread.clips[sel.clipIndex]
    ) ? sel : null;
    const route = routeChatMessage(clean, { phase: thread.phase, jobId: thread.jobId, selection });

    // (a) While clipping, a message is a steering prompt for the AI.
    if (route.type === 'steer') {
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

    // (b) /hyperframes: honest pending-adapter state. Zero provider calls,
    // zero mutations — the message is the whole response.
    if (route.type === 'hyperframes') {
      pushMsg(threadId, {
        role: 'ai', kind: 'hyperframes',
        data: {
          command: clean,
          scope: selection
            ? { threadId: selection.threadId, jobId: selection.jobId, clipIndex: selection.clipIndex }
            : null,
        },
      });
      return;
    }

    // (c) Bulk scope: honest feedback, never mutate multiple clips.
    if (route.type === 'bulk-unsupported') {
      pushMsg(threadId, {
        role: 'ai', kind: 'notice',
        text: 'I can only edit one clip at a time right now, so I left all clips untouched. Select a single clip and describe the change — I\'ll propose the edit for that clip before anything is saved.',
      });
      return;
    }

    // (d) Ordinary edit for the explicitly selected clip.
    if (route.type === 'edit') {
      await runChatEdit(thread, route.selection, clean);
      return;
    }

    // (e) A fresh link starts a fresh flow on this thread: reset prior job
    // state AND the selection (a new source owns the thread again).
    if (route.type === 'new-source') {
      stopPoll(threadId);
      patchThread(threadId, { jobId: null, clips: [], progress: null });
      clearSelection();
      runValidation(threadId, route.url);
      return;
    }

    // (f) Clips done, nothing selected: actionable, never guess clip0.
    if (route.type === 'need-selection') {
      pushMsg(threadId, {
        role: 'ai', kind: 'notice',
        text: 'Your clips are ready — select a clip to edit it (tap Edit on any clip), then tell me what to change, e.g. “make the hook punchier”.',
      });
      return;
    }

    // (g) Fresh thread: the existing idle hint.
    if (route.type === 'idle-hint') {
      pushMsg(threadId, { role: 'ai', kind: 'text', text: 'Paste a video link (YouTube, Twitch, …) and I\'ll check it, then we\'ll set it up together.' });
      return;
    }

    // (none) validating / caption-pick / settings / error: stay silent, as before.
  }, [activeId, threads, selectedClip, createThread, pushMsg, patchThread, runValidation, stopPoll, clearSelection, runChatEdit]);

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
    selectedClip, selectClip, clearSelection,
    workspaceRef,
    createThread, selectThread, deleteThread,
    sendMessage, applyProposal, discardProposal,
    confirmSettings, proceedToSettings, patchThread,
    DEFAULT_SETTINGS,
  };
}
