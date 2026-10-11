/**
 * UI-01 offline fixture — scripted browser test (?ui01test=1 or
 * ?ui01test=mobile).
 *
 * Drives the REAL AppShell (real React root, real hooks, real
 * createProjectController + real serialized edit queue, stubbed API) in a
 * real Chromium. Every assertion reads the live DOM. Results are rendered
 * into #ui01-test-results and the document title is set to
 * UI01-TEST-DONE when the sequence finishes.
 *
 * ?ui01test=1      desktop-width lifecycle (1440): mount, draft-source
 *                  selection, export/preview identity, rendered-artifact
 *                  timebase on real playable media, controller-only save,
 *                  deferred save, restore ack/draft, historical render,
 *                  Ask-Nugget, clip switch, unmount/remount.
 *                  Media: the fixture dist must contain the synthetic
 *                  playable mp4s (see README — generated with ffmpeg
 *                  into the dist folder after the vite build).
 * ?ui01test=mobile narrow-width (390): Chat/Editor tab switching preserves
 *                  mounted composer text + controller identity; real focus.
 */
const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function poll(fn, { timeout = 12000, step = 120, label = 'condition' } = {}) {
  const t0 = Date.now();
  for (;;) {
    let v;
    try { v = fn(); } catch { v = null; }
    if (v) return v;
    if (Date.now() - t0 > timeout) throw new Error(`timeout waiting for: ${label}`);
    await sleep(step);
  }
}

const headBadge = (title) => $(`.nc-ce-head span[title="${title}"]`);
const badgeText = (title) => headBadge(title)?.textContent ?? null;
const buttonByText = (text, scope = document) =>
  $$('button', scope).find((b) => (b.textContent || '').includes(text)) || null;

function setNativeValue(el, value) {
  const setter = Object.getOwnPropertyDescriptor(
    window.HTMLTextAreaElement.prototype, 'value'
  ).set;
  setter.call(el, value);
  el.dispatchEvent(new Event('input', { bubbles: true }));
}

export async function runDriver(mode) {
  const results = [];
  const rec = (name, pass, detail = '') => {
    results.push({ name, pass: !!pass, detail: String(detail) });
    // eslint-disable-next-line no-console
    console.log(`[ui01-test] ${pass ? 'PASS' : 'FAIL'} ${name}${detail ? ` — ${detail}` : ''}`);
  };
  const stats = () => window.__ui01ControllerStats();

  const finish = () => {
    const passed = results.filter((r) => r.pass).length;
    const el = $('#ui01-test-results');
    if (el) {
      el.textContent = JSON.stringify({ mode, passed, total: results.length, results }, null, 1);
    }
    document.title = `UI01-TEST-DONE ${passed}/${results.length}`;
  };

  try {
    if (mode === 'mobile') await runMobile(rec, poll, stats);
    else await runDesktop(rec, poll, stats);
  } catch (e) {
    rec('driver completed without exception', false, e?.message || String(e));
  }
  finish();
}

async function runDesktop(rec, poll, stats) {
  // 1 — mount: real header state.
  await poll(() => badgeText('Acknowledged project version') === 'v3', { label: 'ack badge v3' });
  rec('mount: acknowledged version badge shows v3', badgeText('Acknowledged project version') === 'v3');
  rec('mount: save badge shows Saved', badgeText('Project save state') === 'Saved');
  rec('mount: preview badge shows Draft preview',
    badgeText('What the preview player is showing') === 'Draft preview');
  // Draft overlays are drawn from the current draft project (positive control).
  rec('mount: draft hook/caption overlays render in draft preview',
    $$('[data-ui01="hook-overlay"],[data-ui01="caption-overlay"]').length >= 1);
  // The fixture clip carries BOTH video_url and composed_video_url
  // (review-02): the draft player must select the raw source — the
  // production clipPreviewSrc policy would have picked the composed
  // (burned) file here.
  const mountSrc = $('.nc-ce-grid video')?.getAttribute('src') || '';
  rec('mount: draft player selects the raw source (video_url), not the composed file',
    mountSrc.includes('fixture-source.mp4') && !mountSrc.includes('composed'),
    mountSrc);

  // 2 — controller identity: exactly one per selection.
  rec('mount: exactly one controller created for the selection',
    stats().created === 1 && stats().disposed === 0,
    JSON.stringify(stats()));
  const c0 = window.__ui01Controllers[0];

  // 3 — export through the real controller/queue: exact-render badge,
  // artifact src, no draft overlays on exported pixels.
  buttonByText('Export').click();
  await poll(() => badgeText('What the preview player is showing') === 'Exact v3 render',
    { label: 'Exact v3 render badge' });
  rec('export: preview badge shows Exact v3 render', true);
  const videoSrc = $('.nc-ce-grid video')?.getAttribute('src') || '';
  rec('export: player src is the exported artifact', videoSrc.includes('fixture-render-v3.mp4'), videoSrc);
  rec('export: draft overlays suppressed on the render',
    $$('[data-ui01="hook-overlay"],[data-ui01="caption-overlay"]').length === 0);

  // 3.5 — rendered playback timebase (review-02): the exported file is
  // a real playable mp4 and must play on the artifact's OWN timeline —
  // the draft segment loop must never rewind it. Two probes:
  //  (i) play-through: play from 0 and confirm the playhead advances
  //      steadily past 2s;
  //  (ii) seek: re-initialize the media element with load() first —
  //      headless Chrome can leave a React-remounted element (the
  //      export swaps the player via key={videoSrc}) in a state where
  //      seeks resolve to 0 until load() is called; that is a harness
  //      quirk of the synthetic setup, not app behavior (the app seeks
  //      only through user controls). After load(), seek to 2s and
  //      confirm the playhead lands and advances from there.
  const renderVideo = $('.nc-ce-grid video');
  await poll(() => renderVideo.readyState >= 2, { label: 'render media loaded' });
  rec('render playback: artifact media is playable (readyState)',
    renderVideo.readyState >= 2, `readyState=${renderVideo.readyState}`);
  await renderVideo.play();
  await poll(() => renderVideo.currentTime >= 2.3, { label: 'render play-through past 2.3s' });
  rec('render playback: playhead advances on the artifact timebase (no draft loop)',
    renderVideo.currentTime >= 2.3 && !renderVideo.paused,
    `t=${renderVideo.currentTime.toFixed(2)} paused=${renderVideo.paused}`);
  renderVideo.pause();
  renderVideo.load();
  await poll(() => renderVideo.readyState >= 2, { label: 'render media reloaded' });
  renderVideo.currentTime = 2;
  await poll(() => renderVideo.currentTime >= 1.9, { label: 'render seeked to 2s' });
  rec('render playback: seek lands on the artifact timeline (after load() re-init)',
    renderVideo.currentTime >= 1.9, `t=${renderVideo.currentTime.toFixed(2)}`);

  // 4 — controller-only saved edit (chat/transcript path, NOT the nested
  // mutate): the exact-render badge must invalidate.
  await c0.applyEdit('chat-side edit', (p) => {
    p.captions.edits['0'] = 'STOP!';
    return p;
  });
  await poll(() => badgeText('Acknowledged project version') === 'v4', { label: 'ack badge v4' });
  const afterSave = badgeText('What the preview player is showing');
  rec('controller-only save invalidates the exact-render badge',
    afterSave !== 'Exact v3 render', afterSave);
  rec('controller-only save: ack badge follows to v4',
    badgeText('Acknowledged project version') === 'v4');
  rec('controller-only save: save badge Saved', badgeText('Project save state') === 'Saved');

  // 5 — Ask Nugget restores a collapsed desktop chat, then focuses the
  // real composer.
  buttonByText('Hide chat').click();
  await poll(() => $('.nc-workspace-row.chat-collapsed'), { label: 'chat-collapsed class' });
  rec('hide chat collapses the desktop rail (CSS only)', true);
  buttonByText('Ask Nugget to edit').click();
  await poll(() => !$('.nc-workspace-row.chat-collapsed'), { label: 'chat-collapsed removed' });
  rec('ask-nugget expands the collapsed chat before focusing', true);
  await poll(() => document.activeElement?.classList?.contains('nc-composer-input'),
    { label: 'composer focused' });
  rec('ask-nugget focuses the real composer textarea',
    document.activeElement?.classList?.contains('nc-composer-input') === true);

  // 6 — deferred save through the real queue with a gated API: the header
  // shows Saving… while the save is in flight.
  await c0.restoreVersion(1);
  await poll(() => (badgeText('Project save state') || '').includes('draft v1'),
    { label: 'dirty draft v1 badge' });
  rec('restore: save badge distinguishes the restored draft',
    (badgeText('Project save state') || '').includes('draft v1'),
    badgeText('Project save state'));
  rec('restore: ack badge keeps the server-acknowledged version',
    badgeText('Acknowledged project version') === 'v4',
    badgeText('Acknowledged project version'));
  let releaseSave;
  window.__ui01Api.__setSaveGate(new Promise((r) => { releaseSave = r; }));
  const saving = c0.saveDraft();
  await poll(() => badgeText('Project save state') === 'Saving…', { label: 'Saving… badge' });
  rec('deferred save: header shows Saving… while the API is gated', true);
  releaseSave();
  await saving;
  await poll(() => badgeText('Project save state') === 'Saved', { label: 'Saved badge' });
  rec('deferred save: completes to Saved with ack bump',
    badgeText('Acknowledged project version') === 'v5',
    badgeText('Acknowledged project version'));

  // 7 — restore again: ack stays, draft distinguished.
  await c0.restoreVersion(2);
  await poll(() => (badgeText('Project save state') || '').includes('draft v2'),
    { label: 'draft v2 badge' });
  rec('second restore: ack badge still v5, draft v2 distinguished',
    badgeText('Acknowledged project version') === 'v5' &&
    (badgeText('Project save state') || '').includes('draft v2'),
    `ack=${badgeText('Acknowledged project version')} save=${badgeText('Project save state')}`);

  // 8 — historical render: honestly labeled, no draft overlays.
  buttonByText('Watch render').click();
  await poll(() => badgeText('What the preview player is showing') === 'Historical v3 render',
    { label: 'Historical v3 render badge' });
  rec('watch render: labeled as a historical render', true);
  rec('watch render: draft overlays suppressed',
    $$('[data-ui01="hook-overlay"],[data-ui01="caption-overlay"]').length === 0);
  rec('watch render: player src is that version\u2019s artifact',
    ($('.nc-ce-grid video')?.getAttribute('src') || '').includes('fixture-render-v3.mp4'));

  // 8.5 — historical-render timebase under a short draft segment
  // (review-02): shorten the CURRENT draft to segment [0,1) through the
  // controller (a chat-side edit). The player still shows the v3 render;
  // with loop selected-segment on, a draft-bound handler would rewind
  // the artifact to 0 as soon as it plays past 1s.
  await c0.applyEdit('shorten draft segment', (p) => {
    p.segments = [{ ...p.segments[0], start: 0, end: 1 }];
    return p;
  });
  await poll(() => badgeText('Acknowledged project version') === 'v6', { label: 'ack badge v6' });
  rec('historical: badge stays Historical v3 after the draft edit',
    badgeText('What the preview player is showing') === 'Historical v3 render',
    badgeText('What the preview player is showing'));
  const histVideo = $('.nc-ce-grid video');
  await poll(() => histVideo.readyState >= 2, { label: 'historical media loaded' });
  // Play-through is the load-bearing probe here: with the draft loop
  // wrongly applied, the playhead could never pass the [0,1) segment
  // end — it would rewind to 0 at t>=1 for as long as it plays.
  await histVideo.play();
  await poll(() => histVideo.currentTime >= 1.9, { label: 'historical play-through past 1.9s' });
  rec('historical playback: not rewound by the current draft segment [0,1) loop',
    histVideo.currentTime >= 1.9 && !histVideo.paused,
    `t=${histVideo.currentTime.toFixed(2)} paused=${histVideo.paused}`);
  histVideo.pause();
  histVideo.load();
  await poll(() => histVideo.readyState >= 2, { label: 'historical media reloaded' });
  histVideo.currentTime = 2;
  await poll(() => histVideo.currentTime >= 1.9, { label: 'historical seeked to 2s' });
  rec('historical playback: seek lands on the artifact timeline (after load() re-init)',
    histVideo.currentTime >= 1.9, `t=${histVideo.currentTime.toFixed(2)}`);

  // 9 — clip switch: old controller disposed, preview ownership cleared.
  const before = stats();
  window.__ui01Chat.selectClip('t1', 1);
  await poll(() => ($('.nc-ce-title')?.textContent || '').includes('Fixture clip two'),
    { label: 'clip two title' });
  const after = stats();
  rec('clip switch: old controller disposed, new one created',
    after.created === before.created + 1 && after.disposed === before.disposed + 1,
    JSON.stringify({ before, after }));
  rec('clip switch: preview ownership cleared to Draft preview',
    badgeText('What the preview player is showing') === 'Draft preview',
    badgeText('What the preview player is showing'));

  // 10 — unmount/remount: real setup-cleanup-setup disposes and recreates.
  const beforeRemount = stats();
  window.__ui01.remount();
  await poll(() => badgeText('Acknowledged project version') != null, { label: 'remount ready' });
  await poll(() => stats().created === beforeRemount.created + 1, { label: 'recreated controller' });
  const afterRemount = stats();
  rec('unmount/remount: cleanup disposes, setup creates a fresh controller',
    afterRemount.disposed === beforeRemount.disposed + 1,
    JSON.stringify({ beforeRemount, afterRemount }));
}

async function runMobile(rec, poll, stats) {
  await poll(() => $('.nc-mobile-toggle'), { label: 'mobile toggle visible' });
  const ta = await poll(() => $('textarea.nc-composer-input'), { label: 'composer textarea' });

  // Type unsent text into the REAL composer.
  setNativeValue(ta, 'make the hook punchier');
  rec('composer accepts real text input', ta.value === 'make the hook punchier');
  const c0 = window.__ui01Controllers[0];
  rec('one controller for the selection', stats().created === 1 && !!c0);

  // Switch to the editor view: chat stays MOUNTED (display:none), state kept.
  buttonByText('Editor').click();
  await poll(() => $('.nc-workspace-row')?.dataset.mview === 'editor', { label: 'editor mview' });
  const chatCol = $('.nc-chat-col');
  const taAfter = $('textarea.nc-composer-input');
  rec('narrow switch keeps the chat column mounted',
    !!chatCol && !!taAfter && document.contains(chatCol));
  rec('narrow switch hides the chat column with CSS only',
    chatCol && getComputedStyle(chatCol).display === 'none',
    chatCol ? getComputedStyle(chatCol).display : 'missing');
  rec('unsent composer text survives the switch', taAfter && taAfter.value === 'make the hook punchier');
  rec('controller identity survives the switch',
    window.__ui01Controllers[0] === c0 && stats().created === 1);

  // Back to chat: real focus works.
  buttonByText('Chat').click();
  await poll(() => $('.nc-workspace-row')?.dataset.mview === 'chat', { label: 'chat mview' });
  const ta2 = $('textarea.nc-composer-input');
  ta2.focus();
  rec('composer takes real focus after switching back',
    document.activeElement === ta2);
}
