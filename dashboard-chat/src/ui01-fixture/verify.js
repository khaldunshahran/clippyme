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

function luminance(rgb) {
  const m = String(rgb).match(/[\d.]+/g);
  if (!m || m.length < 3) return null;
  const [r, g, b] = m.slice(0, 3).map((v) => {
    const c = parseFloat(v) / 255;
    return c <= 0.03928 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
  });
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}

function contrastRatio(fg, bg) {
  const l1 = luminance(fg);
  const l2 = luminance(bg);
  if (l1 == null || l2 == null) return 0;
  const [hi, lo] = l1 >= l2 ? [l1, l2] : [l2, l1];
  return (hi + 0.05) / (lo + 0.05);
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
  // View state: the narrow Chat/Editor toggle hides one column by design
  // (CSS display:none; React stays mounted). Editor checks only apply
  // when the editor column is the visible one; chat checks only when
  // the chat column is visible (not toggled away, not collapsed).
  const mview = ($('.nc-workspace-row')?.getAttribute('data-mview')) || 'chat';
  const editorVisible = !narrow || mview === 'editor';
  const chatCollapsedVariant = variant.actions.includes('hideChat') && !narrow;
  const chatVisible = (!narrow || mview === 'chat') && !chatCollapsedVariant;

  // Page-level: no horizontal overflow (the old failure mode).
  check('page: no horizontal overflow',
    document.documentElement.scrollWidth <= window.innerWidth + 1,
    `sw=${document.documentElement.scrollWidth} iw=${window.innerWidth}`);

  // Production-faithful theme (review-02): the fixture imports the real
  // index.css — the same import production src/main.jsx makes. These
  // computed styles prove the production theme actually applied; an
  // unstyled shell (white background, browser-default serif/buttons)
  // fails them, so screenshots from this shell can speak for production
  // theme, dimensions and contrast.
  const bodyBg = getComputedStyle(document.body).backgroundColor;
  check('theme: production global CSS applied (dark chat background)',
    bodyBg === 'rgb(10, 10, 11)', bodyBg);
  const bodyFont = getComputedStyle(document.body).fontFamily || '';
  check('theme: production font stack applied (not browser-default serif)',
    bodyFont.includes('Segoe UI') && !/^\s*"?Times/.test(bodyFont),
    bodyFont.slice(0, 60));

  if (selected) {
  if (editorVisible) {
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
    if (!narrow) actions.splice(2, 0, chatCollapsedVariant ? 'Show chat' : 'Hide chat');
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

    // Edge containment (regression): every required header control must
    // sit fully inside the workspace panel — nothing clipped on either
    // edge. The old scrollable action strip could be scrolled so the
    // first buttons sat LEFT of the panel edge ("gget to edit") while
    // the reachability checks above still passed, because they only
    // measured the right edge / scrollability, never the left edge.
    const panel = $('.nc-workspace-panel');
    if (panel && strip && visible(strip)) {
      const pr = panel.getBoundingClientRect();
      const required = [
        buttonByText('Back'),
        title,
        ...badges,
        ...$$('button', strip),
        ...tabs,
      ].filter((el) => el && visible(el));
      const minLeft = Math.min(...required.map((el) => el.getBoundingClientRect().left));
      const maxRight = Math.max(...required.map((el) => el.getBoundingClientRect().right));
      check('header: no required control clipped at the panel left edge',
        minLeft >= pr.left - 1,
        `minLeft=${Math.round(minLeft)} panelLeft=${Math.round(pr.left)}`);
      check('header: no required control clipped at the panel right edge',
        maxRight <= pr.right + 1,
        `maxRight=${Math.round(maxRight)} panelRight=${Math.round(pr.right)}`);
      check('header: action strip rests unscrolled (scrollLeft stays 0)',
        strip.scrollLeft === 0, `scrollLeft=${strip.scrollLeft}`);
    }

    // Timeline + preview.
    check('timeline scrubber reachable', reachable($('.nc-ce-grid [role="slider"]')));
    const video = $('.nc-ce-grid video');
    check('primary preview player present', !!video);

    // Editor content-area sizing (review-02): the grid/header respond
    // to the .nc-ws-editor width — what remains AFTER the transcript
    // rail — so the primary player must keep a usable width and the
    // grid shape must follow the EDITOR width, not the outer panel's.
    // (The panel-based query once qualified a two-column layout while
    // the player was squeezed to ~116px beside the 300px rail.)
    const editorEl = $('.nc-ws-editor');
    const gridEl = $('.nc-ce-grid');
    if (editorEl && gridEl && video) {
      const ew = editorEl.clientWidth;
      const vw = video.getBoundingClientRect().width;
      check('editor: primary preview keeps a usable width',
        vw >= 240, `video=${Math.round(vw)} editor=${ew}`);
      const kids = gridEl.children;
      if (kids.length >= 2) {
        const r0 = kids[0].getBoundingClientRect();
        const r1 = kids[1].getBoundingClientRect();
        const sideBySide = r1.left > r0.left + 50 && Math.abs(r1.top - r0.top) < 60;
        if (ew >= 740) {
          check('editor: two-column grid when the EDITOR itself fits it',
            sideBySide, `editor=${ew}`);
        } else if (ew <= 700) {
          check('editor: single column when the EDITOR cannot fit two columns',
            !sideBySide, `editor=${ew}`);
        }
      }
    }
  } else {
    // Narrow chat view: the editor column is hidden by the Chat/Editor
    // toggle (CSS display:none) — React and controller state stay
    // mounted, which the mobile driver test proves separately.
    check('mobile chat view: editor column hidden by the toggle (React stays mounted)',
      !!$('.nc-workspace-panel') && !visible($('.nc-workspace-panel')));
    check('mobile chat view: chat column visible', visible($('.nc-chat-col')));
  }

    // (mview / editorVisible / chatVisible were computed at the top.)

    // Transcript rail state (lives in the editor panel). Defaults closed
    // below 1200px viewport; the fixture opens it explicitly via the
    // openTranscript action for the open-variant shots.
    if (variant.actions.includes('closeTranscript')) {
      check('transcript: closed on request', !$('.nc-ws-transcript'));
    } else if (variant.actions.includes('openTranscript') && (!narrow || mview === 'editor')) {
      check('transcript: rail visible when open', visible($('.nc-ws-transcript')));
      // Stacked layouts (review-02): the transcript is a tool view
      // BELOW the editor — its top must start at or below the editor
      // grid's bottom (a shrunk editor box once let the grid paint
      // over the rail; element rects alone would not show that, the
      // grid's painted bottom would).
      const bodyEl = $('.nc-ws-body');
      const gridEl2 = $('.nc-ce-grid');
      const railEl = $('.nc-ws-transcript');
      if (bodyEl && gridEl2 && railEl && getComputedStyle(bodyEl).flexDirection === 'column') {
        const gb = gridEl2.getBoundingClientRect().bottom;
        const rt = railEl.getBoundingClientRect().top;
        check('stacked: transcript tool view below the editor content (no paint overlap)',
          rt >= gb - 2, `railTop=${Math.round(rt)} gridBottom=${Math.round(gb)}`);
      }
      // Readable transcript text on the production theme (review-02):
      // word ink vs its EFFECTIVE background — the active word carries
      // its own highlight background (white ink on the highlight), so
      // the surface behind a word is its own background when it has
      // one, else the rail surface.
      const tpw = $('.nc-ws-transcript .tp-w');
      const tpSurface = $('.nc-ws-transcript .tp');
      if (tpw && tpSurface) {
        const wordCs = getComputedStyle(tpw);
        const wordBg = wordCs.backgroundColor || '';
        const alpha = parseFloat((wordBg.match(/[\d.]+/g) || [])[3] ?? '1');
        const effBg = alpha > 0 ? wordBg : getComputedStyle(tpSurface).backgroundColor;
        const ratio = contrastRatio(wordCs.color, effBg);
        check('theme: transcript words readable on their effective background (contrast >= 3)',
          ratio >= 3, `ratio=${ratio.toFixed(2)} bg=${effBg}`);
      }
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
