/**
 * UI-01 offline fixture — entry point.
 *
 * Renders the REAL AppShell (real React root, real hooks, real
 * useWorkspaceController -> real createProjectController + real queue)
 * with stubbed API/chat/session modules (see vite.ui01fixture.config.js
 * aliases). No network, no production data, no auth.
 *
 * Query params:
 *   sel=0                 no clip selected
 *   actions=a,b           post-mount UI actions: closeTranscript, hideChat,
 *                         mobileEditor (clicks the real controls)
 *   ui01test=1            run the scripted desktop lifecycle test
 *   ui01test=mobile       run the scripted narrow-viewport test
 *   label=...             human label for the screenshot badge
 *
 * Screenshot mode (no ui01test): performs `actions`, runs the control
 * verification, renders the badge, sets document.title=UI01-SHOT-READY.
 */
import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
// Production-faithful shell (review-02): the REAL global chat CSS/theme
// (index.css — the same import production src/main.jsx makes). Without
// it the chat column rendered unstyled (white, serif, default buttons)
// and the screenshots could not establish production theme/contrast.
import '../index.css';
import { AppShell } from '../App.jsx';
import { runDriver } from './driver.js';
import { runVerify } from './verify.js';
import { __setSaveGate, __calls } from './stubs/realApi.js';
import { __ui01ControllerStats } from './stubs/projectControllerShim.js';

const params = new URLSearchParams(window.location.search);
const variant = {
  sel: params.get('sel') || '1',
  actions: (params.get('actions') || '').split(',').filter(Boolean),
  label: params.get('label') || 'selected',
  test: params.get('ui01test') || '',
};

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];
const buttonByText = (text, scope = document) =>
  $$('button', scope).find((b) => (b.textContent || '').trim().includes(text)) || null;

function FixtureBanner() {
  return (
    <div
      style={{
        background: '#3d2b1f', color: '#ffd9a8', fontSize: 11, fontWeight: 800,
        letterSpacing: '.6px', padding: '5px 12px', flexShrink: 0, zIndex: 200,
      }}
    >
      UI-01 OFFLINE FIXTURE — mock project/controller, no network, not live data
    </div>
  );
}

function FixtureApp() {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100vh', overflow: 'hidden' }}>
      <FixtureBanner />
      <div style={{ flex: 1, minHeight: 0, display: 'flex', flexDirection: 'column' }}>
        <AppShell />
      </div>
      <div
        id="ui01-verify-badge"
        style={{
          position: 'fixed', left: 8, bottom: 8, zIndex: 300, fontSize: 11,
          fontWeight: 800, padding: '6px 10px', borderRadius: 8,
          background: 'rgba(20,60,30,.92)', color: '#d8ffd8',
          fontFamily: 'monospace', maxWidth: 'calc(100vw - 16px)',
        }}
      />
      <pre
        id="ui01-test-results"
        style={{
          position: 'fixed', right: 8, top: 40, zIndex: 300, fontSize: 10,
          maxWidth: 420, maxHeight: '70vh', overflow: 'auto',
          background: 'rgba(10,10,14,.92)', color: '#cfe3ff', padding: 10,
          borderRadius: 8, fontFamily: 'monospace', whiteSpace: 'pre-wrap',
          display: variant.test ? 'block' : 'none',
        }}
      />
    </div>
  );
}

async function poll(fn, { timeout = 15000, step = 120, label = '' } = {}) {
  const t0 = Date.now();
  for (;;) {
    let v = null;
    try { v = fn(); } catch { v = null; }
    if (v) return v;
    if (Date.now() - t0 > timeout) throw new Error(`fixture timeout: ${label}`);
    await sleep(step);
  }
}

async function performActions() {
  // Wait for the app to be up: header (selected) or composer (any).
  await poll(() => document.querySelector('.nc-ce-head') || document.querySelector('textarea.nc-composer-input'),
    { label: 'app ready' });
  // Record real-AppShell sidebar presence BEFORE any variant action runs;
  // the closeSidebar action is an intentional user toggle, not a missing shell.
  window.__ui01.sidebarAtMount =
    !!document.querySelector('.nc-sidebar, aside, [class*="sidebar"]');
  await sleep(400); // let controller load + versions list settle
  for (const a of variant.actions) {
    if (a === 'closeTranscript') {
      const btn = document.querySelector('button[aria-controls="nc-transcript-panel"]');
      if (btn && btn.getAttribute('aria-pressed') === 'true') btn.click();
    } else if (a === 'openTranscript') {
      const btn = document.querySelector('button[aria-controls="nc-transcript-panel"]');
      if (btn && btn.getAttribute('aria-pressed') !== 'true') btn.click();
    } else if (a === 'hideChat') {
      buttonByText('Hide chat')?.click();
    } else if (a === 'mobileEditor') {
      buttonByText('Editor')?.click();
    } else if (a === 'mobileChat') {
      buttonByText('Chat')?.click();
    } else if (a === 'closeSidebar') {
      // Real AppShell sidebar is 264px and user-collapsible via the header
      // "Toggle sidebar" button; at 390px a mobile user closes it first.
      // Disable the 0.25s width transition so headless virtual time does not
      // catch the flex layout mid-animation.
      const sb = document.querySelector('.nc-sidebar');
      if (sb) sb.style.transition = 'none';
      document.querySelector('button[aria-label="Toggle sidebar"]')?.click();
    }
    await sleep(350);
  }
  await sleep(400);
}

async function main() {
  // Driver hooks for the scripted tests.
  window.__ui01Api = { __setSaveGate, __calls };
  window.__ui01ControllerStats = __ui01ControllerStats;

  const rootEl = document.getElementById('root');
  const root = createRoot(rootEl);
  window.__ui01 = {
    root,
    remount() {
      root.unmount();
      const r2 = createRoot(rootEl);
      r2.render(<StrictMode><FixtureApp /></StrictMode>);
      window.__ui01.root = r2;
    },
  };
  root.render(<StrictMode><FixtureApp /></StrictMode>);

  if (variant.test) {
    await poll(() => window.__ui01Controllers && window.__ui01Controllers.length > 0,
      { label: 'controller created' });
    await sleep(600);
    await runDriver(variant.test);
  } else {
    await performActions();
    runVerify(variant);
  }
}

main().catch((e) => {
  // eslint-disable-next-line no-console
  console.error('[ui01-fixture] fatal', e);
  document.title = `UI01-FIXTURE-ERROR ${e?.message || e}`;
});
