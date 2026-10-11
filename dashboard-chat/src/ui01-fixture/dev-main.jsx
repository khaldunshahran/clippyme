// UI-01 fixture — DEVELOPMENT-React entry (review-02).
//
// Why this exists: the main fixture bundle is built by Vite in
// production mode, and in a production React bundle <StrictMode> is a
// no-op — it never double-invokes effects, so it CANNOT demonstrate
// the development setup -> cleanup -> setup sequence (the production
// driver correctly asserts disposed=0 on initial mount). This entry
// is built with `--mode development` (see the fixture README), which
// bundles the development React build; StrictMode then mounts every
// effect twice (setup -> cleanup -> setup) and the counts below are
// real, load-bearing evidence — on the production bundle the first
// check fails (setups would be exactly 1, cleanups 0).
//
// What it proves (headless Chrome run, ?ui01dev=1):
//   * a StrictMode probe effect's setup/cleanup counts after mount;
//   * exactly one LIVE workspace controller after the StrictMode
//     double-mount (created - disposed === 1 via the shim stats);
//   * a deferred (gated) controller save stays in flight across a
//     Hide chat -> Show chat layout switch — same stats, still
//     "Saving…" — then completes and bumps the acknowledged version;
//   * an explicit remount (window.__ui01.remount()) produces another
//     full setup -> cleanup -> setup cycle and leaves exactly one
//     live controller.
//
// The shell, stub backend and controller are the same ones the main
// fixture uses (aliases in vite.ui01fixture.config.js), so the save
// path exercised is the real projectController + save queue.

import { StrictMode, useEffect } from 'react';
import { createRoot } from 'react-dom/client';
import '../index.css';
import { AppShell } from '../App.jsx';
import { __setSaveGate, __calls } from './stubs/realApi.js';
import { __ui01ControllerStats } from './stubs/projectControllerShim.js';

const strict = (window.__ui01Strict = { setups: 0, cleanups: 0 });

// An effect with no dependencies: React runs it on every mount. Under
// development StrictMode the initial mount is setup -> cleanup -> setup.
function StrictProbe() {
  useEffect(() => {
    strict.setups += 1;
    return () => {
      strict.cleanups += 1;
    };
  }, []);
  return null;
}

function DevApp() {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100vh', overflow: 'hidden' }}>
      <div
        style={{
          background: '#3d2b1f', color: '#ffd9a8', fontSize: 11, fontWeight: 800,
          letterSpacing: '.6px', padding: '5px 12px', flexShrink: 0, zIndex: 200,
        }}
      >
        UI-01 OFFLINE FIXTURE (DEVELOPMENT REACT) — mock project/controller, no network, not live data
      </div>
      <div style={{ flex: 1, minHeight: 0, display: 'flex', flexDirection: 'column' }}>
        <AppShell />
      </div>
      <pre
        id="ui01-dev-results"
        style={{
          position: 'fixed', right: 8, top: 40, zIndex: 300, fontSize: 10,
          maxWidth: 460, maxHeight: '70vh', overflow: 'auto',
          background: 'rgba(10,10,14,.92)', color: '#cfe3ff', padding: 10,
          borderRadius: 8, fontFamily: 'monospace', whiteSpace: 'pre-wrap',
        }}
      />
    </div>
  );
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const $ = (s) => document.querySelector(s);
const $$ = (s) => [...document.querySelectorAll(s)];
const buttonByText = (text) =>
  $$('button').find((b) => (b.textContent || '').trim() === text) || null;
const headBadge = (title) => $(`.nc-ce-head span[title="${title}"]`);
const badgeText = (title) => headBadge(title)?.textContent ?? null;
const stats = () => window.__ui01ControllerStats();

async function poll(fn, { timeout = 15000, step = 100, label = 'condition' } = {}) {
  const t0 = Date.now();
  for (;;) {
    let v;
    try {
      v = fn();
    } catch {
      v = null;
    }
    if (v) return v;
    if (Date.now() - t0 > timeout) throw new Error(`timeout waiting for: ${label}`);
    await sleep(step);
  }
}

async function runDev() {
  const results = [];
  const rec = (name, pass, detail = '') => {
    results.push({ name, pass: !!pass, detail: String(detail) });
  };
  try {
    // -- StrictMode double-effect on initial mount --------------------
    await poll(() => badgeText('Acknowledged project version') === 'v3', {
      label: 'ack badge v3',
    });
    await poll(() => strict.setups >= 2, { label: 'StrictMode double setup' });
    rec(
      'strict: development StrictMode double-invokes effects on mount (setup-cleanup-setup)',
      strict.setups >= 2 && strict.cleanups >= 1,
      `setups=${strict.setups} cleanups=${strict.cleanups}`,
    );
    await poll(() => stats().created - stats().disposed === 1, {
      label: 'one live controller',
    });
    rec(
      'strict: exactly one LIVE workspace controller after the double-mount',
      stats().created - stats().disposed === 1,
      JSON.stringify(stats()),
    );

    // -- Deferred save across a layout switch --------------------------
    const ctl = window.__ui01Controllers[window.__ui01Controllers.length - 1];
    await ctl.restoreVersion(1);
    await poll(() => (badgeText('Project save state') || '').includes('draft v1'), {
      label: 'dirty draft v1 badge',
    });
    let releaseSave;
    window.__ui01Api.__setSaveGate(new Promise((r) => { releaseSave = r; }));
    const statsBeforeSave = stats();
    const saving = ctl.saveDraft();
    await poll(() => badgeText('Project save state') === 'Saving…', {
      label: 'Saving… badge',
    });
    rec('save: deferred save is in flight (gated backend)', true);

    buttonByText('Hide chat').click();
    await poll(() => buttonByText('Show chat'), { label: 'Show chat toggle' });
    rec(
      'save: layout switch (Hide chat) creates/disposes no controller',
      stats().created === statsBeforeSave.created
        && stats().disposed === statsBeforeSave.disposed,
      JSON.stringify(stats()),
    );
    rec(
      'save: deferred save still in flight after the layout switch',
      badgeText('Project save state') === 'Saving…',
      badgeText('Project save state'),
    );
    buttonByText('Show chat').click();
    await poll(() => buttonByText('Hide chat'), { label: 'Hide chat restored' });

    releaseSave();
    await saving;
    await poll(() => badgeText('Project save state') === 'Saved', { label: 'Saved badge' });
    rec(
      'save: deferred save completes after release and bumps the acknowledged version',
      badgeText('Acknowledged project version') === 'v4',
      badgeText('Acknowledged project version'),
    );

    // -- Explicit remount: full setup -> cleanup -> setup cycle --------
    const strictBefore = { ...strict };
    window.__ui01.remount();
    await poll(() => strict.setups >= strictBefore.setups + 2, {
      label: 'remount double setup',
    });
    rec(
      'remount: unmount cleans up and remount double-invokes again (StrictMode)',
      strict.cleanups >= strictBefore.cleanups + 2,
      `setups ${strictBefore.setups}->${strict.setups} cleanups ${strictBefore.cleanups}->${strict.cleanups}`,
    );
    await poll(() => badgeText('Acknowledged project version') != null, {
      label: 'remounted ack badge',
    });
    await poll(() => stats().created - stats().disposed === 1, {
      label: 'one live controller after remount',
    });
    rec(
      'remount: exactly one live controller after the remount cycle',
      stats().created - stats().disposed === 1,
      JSON.stringify(stats()),
    );
  } catch (err) {
    rec('dev driver completed without exception', false, err?.message || String(err));
  }
  const passed = results.filter((r) => r.pass).length;
  document.title = `UI01DEV ${passed}/${results.length} ${passed === results.length ? 'PASS' : 'FAIL'}`;
  const pre = $('#ui01-dev-results');
  if (pre) {
    pre.textContent =
      `UI01 DEV FIXTURE (development React): ${passed}/${results.length} PASS\n` +
      results
        .map((r) => `${r.pass ? 'PASS' : 'FAIL'}  ${r.name}${r.detail ? `  [${r.detail}]` : ''}`)
        .join('\n');
  }
}

function main() {
  window.__ui01Api = { __setSaveGate, __calls };
  window.__ui01ControllerStats = __ui01ControllerStats;
  const rootEl = document.getElementById('root');
  const root = createRoot(rootEl);
  window.__ui01 = {
    root,
    remount() {
      root.unmount();
      const r2 = createRoot(rootEl);
      r2.render(
        <StrictMode>
          <StrictProbe />
          <DevApp />
        </StrictMode>,
      );
      window.__ui01.root = r2;
    },
  };
  root.render(
    <StrictMode>
      <StrictProbe />
      <DevApp />
    </StrictMode>,
  );
  if (new URLSearchParams(window.location.search).has('ui01dev')) {
    runDev().catch((err) => {
      document.title = 'UI01DEV ERROR';
      const pre = $('#ui01-dev-results');
      if (pre) pre.textContent = `UI01 DEV FIXTURE ERROR: ${err?.message || err}`;
    });
  }
}

main();
