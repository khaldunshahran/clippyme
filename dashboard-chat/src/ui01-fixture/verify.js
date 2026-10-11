/**
 * UI-01 offline fixture — screenshot-mode verification.
 *
 * After the variant's actions settle, checks every required control is
 * visible or reachable by intentional labeled navigation: real bounding
 * boxes inside the viewport (or inside an intentional scroll strip),
 * real keyboard focus, and a non-overlapping clip identity. Results are
 * rendered into #ui01-verify-badge (visible in the screenshot) and the
 * document title becomes UI01-SHOT-READY.
 */
const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

function visible(el) {
  if (!el || !document.contains(el)) return false;
  const cs = getComputedStyle(el);
  if (cs.display === 'none' || cs.visibility === 'hidden' || parseFloat(cs.opacity) === 0) return false;
  const r = el.getBoundingClientRect();
  return r.width > 0 && r.height > 0;
}

function inViewport(el, tol = 3) {
  const r = el.getBoundingClientRect();
  return (
    r.left >= -tol &&
    r.top >= -tol &&
    r.right <= window.innerWidth + tol &&
    r.bottom <= window.innerHeight + tol
  );
}

function scrollableAncestor(el) {
  let n = el.parentElement;
  while (n && n !== document.body) {
    const cs = getComputedStyle(n);
    if ((cs.overflowX === 'auto' || cs.overflowX === 'scroll') && n.scrollWidth > n.clientWidth + 2) {
      return n;
    }
    if ((cs.overflowY === 'auto' || cs.overflowY === 'scroll') && n.scrollHeight > n.clientHeight + 2) {
      return n;
    }
    n = n.parentElement;
  }
  return null;
}

// Visible now, or reachable by scrolling an intentional scroll container.
function reachable(el) {
  return visible(el) && (inViewport(el) || !!scrollableAncestor(el));
}

function canFocus(el) {
  if (!el || el.disabled) return el && el.disabled ? 'skipped-disabled' : false;
  el.focus();
  return document.activeElement === el;
}

function overlaps(a, b) {
  const r1 = a.getBoundingClientRect();
  const r2 = b.getBoundingClientRect();
  return !(r1.right <= r2.left || r2.right <= r1.left || r1.bottom <= r2.top || r2.bottom <= r1.top);
}

const buttonByText = (text, scope = document) =>
  $$('button', scope).find((b) => (b.textContent || '').trim().includes(text)) || null;

export function runVerify(variant) {
  const checks = [];
  const check = (name, pass, detail = '') => {
    checks.push({ name, pass: !!pass, detail: String(detail) });
    // eslint-disable-next-line no-console
    console.log(`[ui01-verify] ${pass ? 'PASS' : 'FAIL'} ${name}${detail ? ` — ${detail}` : ''}`);
  };

  const narrow = window.innerWidth <= 900;
  const selected = variant.sel !== '0';

  // Page-level: no horizontal overflow (the old failure mode).
  check('page: no horizontal overflow',
    document.documentElement.scrollWidth <= window.innerWidth + 1,
    `sw=${document.documentElement.scrollWidth} iw=${window.innerWidth}`);

  if (selected) {
    // Header identity group.
    const title = $('.nc-ce-title');
    check('header: clip title visible', visible(title) && inViewport(title));
    const badges = [
      '.nc-ce-head span[title="Acknowledged project version"]',
      '.nc-ce-head span[title="Project save state"]',
      '.nc-ce-head span[title="What the preview player is showing"]',
    ].map((s) => $(s));
    check('header: all three status badges visible', badges.every((b) => visible(b) && inViewport(b)));
    check('header: title does not overlap badges',
      title && badges.every((b) => b && !overlaps(title, b)));
    check('header: Back reachable + focusable',
      reachable(buttonByText('Back')) && canFocus(buttonByText('Back')) === true);

    // Header action strip: every action reachable and keyboard-focusable.
    const actions = ['Transcript', 'Ask Nugget to edit', 'Reset to AI cut', 'Save draft', 'Export'];
    if (!narrow) actions.splice(2, 0, 'Hide chat');
    const strip = $('.nc-ce-actions');
    for (const name of actions) {
      const btn = name === 'Ask Nugget to edit'
        ? $('button[aria-label^="Ask Nugget to edit"]')
        : buttonByText(name, strip || document);
      const r = btn && reachable(btn);
      const f = btn && canFocus(btn);
      check(`header action reachable: ${name}`, !!r, btn ? '' : 'missing');
      check(`header action focusable: ${name}`, f === true || f === 'skipped-disabled', String(f));
    }
    if (narrow) {
      const toggle = $('.nc-ws-chat-toggle');
      check('mobile: ineffective desktop Hide-chat is hidden',
        !toggle || getComputedStyle(toggle).display === 'none');
    }
    // The strip itself must be an intentional scroll container when crowded.
    if (strip && strip.scrollWidth > strip.clientWidth + 2) {
      const cs = getComputedStyle(strip);
      check('header: crowded actions live in a scrollable labeled strip',
        (cs.overflowX === 'auto' || cs.overflowX === 'scroll') && strip.getAttribute('role') === 'toolbar',
        `sw=${strip.scrollWidth} cw=${strip.clientWidth}`);
    } else {
      check('header: actions fit without scrolling', true, 'no overflow');
    }

    // Panel tabs: every tab inside the viewport horizontally (the AI-cut clip).
    const tabs = $$('.nc-ce-tabs [role="tab"]');
    check('tabs: all five panel tabs present', tabs.length === 5, `found ${tabs.length}`);
    const tabsOk = tabs.every((t) => {
      const r = t.getBoundingClientRect();
      return visible(t) && r.left >= -3 && r.right <= window.innerWidth + 3;
    });
    check('tabs: no tab clipped horizontally', tabsOk,
      tabs.map((t) => { const r = t.getBoundingClientRect(); return `${t.textContent.trim()}@${Math.round(r.right)}`; }).join(' '));

    // Timeline + preview.
    check('timeline scrubber reachable', reachable($('.nc-ce-grid [role="slider"]')));
    const video = $('.nc-ce-grid video');
    check('primary preview player present', !!video);

    // Mobile view: the narrow toggle hides one column by design (CSS
    // display:none, React stays mounted). Chat-column checks only apply
    // when the chat column is the visible one.
    const mview = ($('.nc-workspace-row')?.getAttribute('data-mview')) || 'chat';
    const chatVisible = !narrow || mview === 'chat';

    // Transcript rail state (lives in the editor panel). Defaults closed
    // below 1200px viewport; the fixture opens it explicitly via the
    // openTranscript action for the open-variant shots.
    if (variant.actions.includes('closeTranscript')) {
      check('transcript: closed on request', !$('.nc-ws-transcript'));
    } else if (variant.actions.includes('openTranscript') && (!narrow || mview === 'editor')) {
      check('transcript: rail visible when open', visible($('.nc-ws-transcript')));
    }

    // Chat column state.
    if (variant.actions.includes('hideChat') && !narrow) {
      check('chat: rail hidden by collapse', !!$('.nc-workspace-row.chat-collapsed'));
    } else {
      check('chat: column mounted', !!$('.nc-chat-col'));
    }

    // Selection bar + composer (real AppShell chrome) — only when the
    // chat column is visible.
    if (chatVisible) {
      const selBar = $('.nc-selection-bar');
      check('selection bar visible', visible(selBar));
      check('selection bar: New source reachable', reachable(buttonByText('New source', selBar)));
      const ta = $('textarea.nc-composer-input');
      check('composer textarea reachable + focusable', reachable(ta) && canFocus(ta) === true);
    }
    // Real-AppShell proof: the sidebar was mounted before variant actions
    // ran. A variant that intentionally closes it via the real user toggle
    // (closeSidebar) still counts — the toggle itself is AppShell chrome.
    const sbAtMount = !!(window.__ui01 && window.__ui01.sidebarAtMount);
    const sbNow = !!$('.nc-sidebar, aside, [class*="sidebar"]') || !!$('aside');
    check('sidebar mounted (real AppShell)',
      sbAtMount && (sbNow || variant.actions.includes('closeSidebar')),
      sbAtMount ? (sbNow ? 'open' : 'closed by real user toggle') : 'missing at mount');
  } else {
    // No-selection state: chat full width, no editor.
    check('no-selection: editor panel absent', !$('.nc-workspace-panel'));
    check('no-selection: composer reachable + focusable',
      reachable($('textarea.nc-composer-input')) && canFocus($('textarea.nc-composer-input')) === true);
  }

  const passed = checks.filter((c) => c.pass).length;
  const badge = $('#ui01-verify-badge');
  if (badge) {
    badge.textContent =
      `OFFLINE FIXTURE · ${window.innerWidth}px · ${variant.label} — CHECKS ${passed}/${checks.length} ` +
      (passed === checks.length ? 'PASS' : 'FAIL');
    badge.dataset.pass = String(passed === checks.length);
    badge.dataset.failed =
      checks.filter((c) => !c.pass).map((c) => c.name).join(' | ');
  }
  document.title = 'UI01-SHOT-READY';
  return { passed, total: checks.length, checks };
}
