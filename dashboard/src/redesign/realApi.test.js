import { test } from 'vitest';
import assert from 'node:assert/strict';

// Vitest runs with a jsdom environment, so window.location.origin is real —
// the old plain-Node `globalThis.window` stub is gone.
import { optsToPreselections, clipVideoSrc, clipPreviewSrc, fmtDuration, exportClip, getConfig, saveConfig, getModels, configPresence } from './realApi.js';

// --- optsToPreselections: the Create-tab → backend translation layer --------

test('reframe mode: legacy object alias normalizes to subject', () => {
  assert.equal(optsToPreselections({ reframeMode: 'object' }).reframe_mode, 'subject');
  assert.equal(optsToPreselections({ reframeMode: 'subject' }).reframe_mode, 'subject');
  assert.equal(optsToPreselections({ reframeMode: 'disabled' }).reframe_mode, 'disabled');
});

test('reframe mode: legacy boolean fallback, default auto', () => {
  assert.equal(optsToPreselections({}).reframe_mode, 'auto');
  assert.equal(optsToPreselections({ reframe: false }).reframe_mode, 'disabled');
});

test('model override: blank is omitted, value passes through trimmed', () => {
  assert.equal(optsToPreselections({}).model, undefined);
  assert.equal(optsToPreselections({ model: '  ' }).model, undefined);
  assert.equal(optsToPreselections({ model: ' gemini-2.5-pro ' }).model, 'gemini-2.5-pro');
});

test('karaoke subtitles carry colours but not classic typography', () => {
  const p = optsToPreselections({
    subtitles: true, subMode: 'karaoke', subPreset: 'hormozi_bold',
    subColor: '#FDE700', subStroke: '#111111', subFontSize: 48,
  });
  assert.equal(p.subtitles.preset, 'hormozi_bold');
  assert.equal(p.subtitles.font_color, '#FDE700');
  assert.equal(p.subtitles.outline_color, '#111111');
  assert.equal(p.subtitles.font_size, 48);
  assert.equal('font' in p.subtitles, false, 'classic-only font key leaked into karaoke');
});

test('karaoke font_size 0 means Auto and is omitted', () => {
  const p = optsToPreselections({ subtitles: true, subMode: 'karaoke', subFontSize: 0 });
  assert.equal('font_size' in p.subtitles, false);
});

test('classic subtitles carry font/border/background', () => {
  const p = optsToPreselections({
    subtitles: true, subMode: 'classic', subFont: 'Anton-Regular',
    subColor: '#581BBA', subOutlineW: 3, subBg: true,
  });
  assert.equal(p.subtitles.font, 'Anton-Regular');
  assert.equal(p.subtitles.font_color, '#581BBA');
  assert.equal(p.subtitles.border_width, 3);
  assert.equal(p.subtitles.bg_opacity, 0.6);
  assert.equal(p.subtitles.bg_color, '#000000');
});

test('subtitles off → false; grade none → false; logo off → false', () => {
  const p = optsToPreselections({ subtitles: false, gradePreset: 'none', logo: false });
  assert.equal(p.subtitles, false);
  assert.equal(p.grade, false);
  assert.equal(p.logo, false);
});

test('banner off → banner: false in preselections', () => {
  const p = optsToPreselections({ banner: false });
  assert.equal(p.banner, false);
});

test('banner on → banner object with platform and handle in preselections', () => {
  const p = optsToPreselections({ banner: true, bannerPlatform: 'kick', bannerHandle: 'streamer' });
  assert.deepEqual(p.banner, { enabled: true, platform: 'kick', handle: 'streamer', y_pct: 0.85 });
});

test('grade preset flows through when set', () => {
  assert.deepEqual(optsToPreselections({ gradePreset: 'vivid_pop' }).grade, { preset: 'vivid_pop' });
});

// --- URL safety: a malicious API response must never become an executable src

test('clipVideoSrc neutralizes javascript: and data: schemes', () => {
  for (const evil of ['javascript:alert(1)', 'data:text/html,<script>x</script>']) {
    const src = clipVideoSrc({ video_url: evil });
    assert.equal(src.startsWith('javascript:'), false);
    assert.equal(src.startsWith('data:'), false);
    assert.equal(src.startsWith('/'), true, `expected inert relative path, got ${src}`);
  }
});

test('clipVideoSrc appends the cache-buster correctly', () => {
  assert.equal(clipVideoSrc({ video_url: '/videos/j/clip_1.mp4' }, 99).endsWith('?v=99'), true);
  assert.equal(clipVideoSrc({ video_url: '/videos/j/c.mp4?x=1' }, 99).endsWith('&v=99'), true);
});

test('clipPreviewSrc prefers the composed previewUrl over the raw clip', () => {
  const clip = { video_url: '/videos/j/clip_1.mp4' };
  const raw = clipPreviewSrc(clip, {});
  assert.equal(raw.includes('clip_1.mp4'), true);
  const composed = clipPreviewSrc(clip, { previewUrl: '/videos/j/composed_clip_0.mp4', previewBust: 7 });
  assert.equal(composed.includes('composed_clip_0.mp4'), true);
  assert.equal(composed.endsWith('?v=7'), true);
});

test('fmtDuration renders m:ss with zero-padded seconds', () => {
  assert.equal(fmtDuration(0, 65), '1:05');
  assert.equal(fmtDuration(10, 10), '0:00');
  assert.equal(fmtDuration(0, 599.6), '10:00');
});

// --- I-1 regression: per-clip endpoints must resolve by the backend's
// ABSOLUTE `shorts` position (`original_index`), not the frontend array
// position — they diverge once a manual-publish gap skips a
// deleted_after_publish clip (job_results._build_clips).

test('exportClip composes against clip.original_index, not the array position', async () => {
  const calls = [];
  const originalFetch = globalThis.fetch;
  globalThis.fetch = (url) => {
    calls.push(url);
    return Promise.resolve({ ok: true, json: () => Promise.resolve({ composed_url: '/videos/j/composed.mp4' }) });
  };
  try {
    // Clip B sits at array position 1 but is absolute shorts position 2 (a
    // deleted_after_publish clip at position 1 was skipped upstream).
    const clip = { original_index: 2, video_url: '/videos/j/clip_3.mp4' };
    const state = { toggles: { subtitles: true } };
    await exportClip('job-1', 1, clip, state, {});
    assert.equal(calls.length, 1);
    assert.match(String(calls[0]), /\/api\/compose\/job-1\/2$/);
  } finally {
    globalThis.fetch = originalFetch;
  }
});

// --- H1/M3: getConfig never throws -------------------------------------------
// null means "unknown / unreachable" — callers must not mistake it for "no
// keys configured".

function stubFetch(impl) {
  const original = globalThis.fetch;
  globalThis.fetch = impl;
  return () => { globalThis.fetch = original; };
}

test('getConfig returns null on network failure instead of throwing', async () => {
  const restore = stubFetch(() => Promise.reject(new TypeError('Failed to fetch')));
  try {
    assert.equal(await getConfig(), null);
  } finally { restore(); }
});

test('getConfig returns null on HTTP error status', async () => {
  const restore = stubFetch(() => Promise.resolve({ ok: false, status: 403 }));
  try {
    assert.equal(await getConfig(), null);
  } finally { restore(); }
});

test('getConfig returns null when the body is not valid JSON', async () => {
  const restore = stubFetch(() => Promise.resolve({
    ok: true,
    json: () => Promise.reject(new Error('Unexpected token <')),
  }));
  try {
    assert.equal(await getConfig(), null);
  } finally { restore(); }
});

test('getConfig returns the parsed body on success', async () => {
  const body = { server_has_gemini: true, has_hf_token: false };
  const restore = stubFetch(() => Promise.resolve({ ok: true, json: () => Promise.resolve(body) }));
  try {
    assert.deepEqual(await getConfig(), body);
  } finally { restore(); }
});

// --- configPresence: the boolean contract -------------------------------------

test('configPresence reads the new has_* flags', () => {
  const p = configPresence({
    server_has_gemini: true, has_hf_token: true, has_deepgram_key: false,
    has_elevenlabs_key: false, TWITCH_CLIENT_ID: 'abc', has_twitch_secret: true,
  });
  assert.equal(p.gemini, true);
  assert.equal(p.hf, true);
  assert.equal(p.deepgram, false);
  assert.equal(p.elevenlabs, false);
  assert.equal(p.twitchId, true);
  assert.equal(p.twitchSecret, true);
});

test('configPresence accepts the legacy UPPER_CASE aliases', () => {
  // Old shape: GEMINI_API_KEY carried the key string itself.
  assert.equal(configPresence({ GEMINI_API_KEY: 'AIza...' }).gemini, true);
  assert.equal(configPresence({ GEMINI_API_KEY: '' }).gemini, false);
  // New shape: the aliases are booleans.
  assert.equal(configPresence({ GEMINI_API_KEY: true }).gemini, true);
  assert.equal(configPresence({ server_has_gemini: false, GEMINI_API_KEY: true }).gemini, false);
});

test('configPresence is all-false on empty/undefined config', () => {
  const p = configPresence({});
  assert.deepEqual(p, { gemini: false, hf: false, deepgram: false, elevenlabs: false, twitchId: false, twitchSecret: false });
  assert.equal(configPresence(undefined).gemini, false);
  assert.equal(configPresence(null).hf, false);
});

// --- L1: saveConfig surfaces the server's error detail ------------------------

test('saveConfig throws the server detail instead of a generic message', async () => {
  const restore = stubFetch(() => Promise.resolve({
    ok: false, status: 422,
    json: () => Promise.resolve({ detail: 'HF_TOKEN looks invalid' }),
  }));
  try {
    await assert.rejects(saveConfig({ HF_TOKEN: 'x' }), /HF_TOKEN looks invalid/);
  } finally { restore(); }
});

test('saveConfig falls back to the status when the server gives no detail', async () => {
  const restore = stubFetch(() => Promise.resolve({
    ok: false, status: 500, json: () => Promise.resolve({}),
  }));
  try {
    await assert.rejects(saveConfig({}), /500/);
  } finally { restore(); }
});

// --- M5: getModels omits X-Gemini-Key when empty --------------------------------

test('getModels omits the X-Gemini-Key header when the key is empty', async () => {
  let captured;
  const restore = stubFetch((url, init) => {
    captured = init;
    return Promise.resolve({ ok: true, json: () => Promise.resolve({ models: [] }) });
  });
  try {
    await getModels('');
    assert.ok(!('X-Gemini-Key' in (captured?.headers || {})), 'header must be omitted, not empty');
    await getModels('  k  ');
    assert.equal(captured.headers['X-Gemini-Key'], 'k');
  } finally { restore(); }
});
