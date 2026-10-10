/**
 * Task 16 review-03: lifetime-bound audio uploads, reset-to-AI-cut
 * follow-up cancellation, and UTF-8 punctuation.
 *
 * 1. UPLOAD LIFETIME. uploadAudio captures the generation around the awaited
 *    response (like aiTrim): a reset/dispose while the upload is in flight
 *    rejects the stale completion, so a panel continuation can never attach
 *    old audio to the next session. Removing the post-response check must
 *    fail these tests. Valid new uploads still resolve and attach.
 *
 * 2. RESET-TO-AI-CUT FOLLOW-UP. restoreLocked checks the lifetime BEFORE
 *    its follow-up getClipProject, not only after the response: a
 *    reset/dispose during version listing must not start a historical
 *    project fetch. Removing the pre-call check must fail these tests.
 *
 * 3. UTF-8 PUNCTUATION. The visible copy uses real — … → · characters;
 *    reintroducing the CP437 mojibake (ΓÇö/ΓÇª/ΓåÆ/┬╖) must fail these tests.
 *
 * 4. PANEL HONESTY. The real onAudioUpload handler awaits the music
 *    mutation before reporting success: a rejected save cannot claim the
 *    audio was attached. Session cancellation stays silent (expected, not
 *    a failure). Removing the await must fail the rejection test.
 */
import { describe, expect, it, vi } from 'vitest';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

// --- minimal hook harness (coordinator-probe style): persistent useState ---
const hooks = vi.hoisted(() => ({ values: [], index: 0 }));
vi.mock('react', async (original) => {
  const actual = await original();
  return {
    ...actual,
    useState: (initial) => {
      const i = hooks.index++;
      if (!(i in hooks.values)) hooks.values[i] = initial;
      return [
        hooks.values[i],
        (v) => {
          hooks.values[i] = typeof v === 'function' ? v(hooks.values[i]) : v;
        },
      ];
    },
    useMemo: (fn) => fn(),
    useCallback: (fn) => fn,
    // ClipEditorView uses a video ref; a plain object is enough in node.
    useRef: (v) => ({ current: v }),
    useEffect: (fn) => {
      try {
        const r = fn();
        if (r && typeof r.catch === 'function') r.catch(() => {});
      } catch {
        /* node env: window/document absent; effects are best-effort */
      }
    },
  };
});
vi.mock('../api/realApi.js', () => ({
  getClipProject: vi.fn(),
  saveClipProject: vi.fn(),
  composeProject: vi.fn(),
  backfillClipProjects: vi.fn(),
  listClipProjectVersions: vi.fn(),
  editClipAI: vi.fn(),
  safeResolveUrl: (u) => u,
  clipPreviewSrc: () => '',
  uploadAudio: vi.fn(),
}));

import { createProjectController } from './projectController.js';
import { ClipEditorView } from '../editor/clipEditor.jsx';
import * as React from 'react';
vi.stubGlobal('React', React);

const clone = (o) => JSON.parse(JSON.stringify(o));
const p = () => ({
  version: 1,
  idempotency_key: 'one',
  segments: [],
  captions: {},
  overlays: [],
});

// --- element-tree helpers for the real component ---
function nodes(tree, predicate) {
  const out = [];
  const walk = (n) => {
    if (!n || typeof n !== 'object') return;
    if (predicate(n)) out.push(n);
    const kids = n.props && n.props.children;
    (Array.isArray(kids) ? kids : [kids]).forEach(walk);
  };
  walk(tree);
  return out;
}
function textOf(n) {
  const out = [];
  const walk = (x) => {
    if (typeof x === 'string') out.push(x);
    else if (x && typeof x === 'object') {
      const kids = x.props && x.props.children;
      (Array.isArray(kids) ? kids : [kids]).forEach(walk);
    }
  };
  walk(n);
  return out.join('');
}
function renderEditor(props) {
  hooks.index = 0;
  return ClipEditorView(props);
}
function openMusicPanel(props) {
  let tree = renderEditor(props);
  const tab = nodes(
    tree,
    (n) =>
      n.type === 'button' &&
      n.props &&
      n.props.role === 'tab' &&
      textOf(n).includes('Music')
  )[0];
  expect(tab, 'music tab exists').toBeTruthy();
  tab.props.onClick();
  tree = renderEditor(props);
  const up = nodes(
    tree,
    (n) => n.props && typeof n.props.onUpload === 'function'
  );
  expect(up.length, 'music panel exposes onUpload').toBeGreaterThan(0);
  return up[0].props.onUpload;
}

describe('review-03: upload lifetime binding', () => {
  it('upload started before reset cannot attach music after the next load', async () => {
    let release;
    const saves = [];
    const api = {
      getClipProject: async () => p(),
      uploadAudio: () => new Promise((r) => (release = r)),
      saveClipProject: async (j, i, project) => {
        saves.push(project);
        return { project: { ...project, version: 2 }, version: 2 };
      },
    };
    const c = createProjectController({ jobId: 'j', clipIndex: 0, api });
    await c.load();
    const oldHandler = (async () => {
      const { file } = await c.uploadAudio('old-session-file');
      await c.applyEdit('music', (project) => ({ ...project, audio: [{ file }] }));
    })().catch(() => {});
    c.resetSession();
    await c.load();
    release({ file: 'audio/old.mp3' });
    await oldHandler;
    expect(saves).toHaveLength(0);
    expect(c.getState().project.audio).toBeUndefined();
  });

  it('upload started before dispose rejects after restart', async () => {
    let release;
    const saves = [];
    const api = {
      getClipProject: async () => p(),
      uploadAudio: () => new Promise((r) => (release = r)),
      saveClipProject: async (j, i, project) => {
        saves.push(project);
        return { project: { ...project, version: 2 }, version: 2 };
      },
    };
    const c = createProjectController({ jobId: 'j', clipIndex: 0, api });
    await c.load();
    let err = null;
    const up = c.uploadAudio('f').catch((e) => {
      err = e;
    });
    c.dispose();
    await c.load(); // restart revives the controller, but the old upload is stale
    release({ file: 'audio/old.mp3' });
    await up;
    expect(err).toMatchObject({ code: 'QUEUE_CANCELLED' });
    expect(saves).toHaveLength(0);
    expect(c.getState().project.audio).toBeUndefined();
  });

  it('a fresh upload still resolves and attaches normally', async () => {
    const saves = [];
    const api = {
      getClipProject: async () => p(),
      uploadAudio: async () => ({ file: 'audio/new.mp3' }),
      saveClipProject: async (j, i, project) => {
        saves.push(project);
        return { project: { ...project, version: 2 }, version: 2 };
      },
    };
    const c = createProjectController({ jobId: 'j', clipIndex: 0, api });
    await c.load();
    const { file } = await c.uploadAudio('new-file');
    expect(file).toBe('audio/new.mp3');
    await c.applyEdit('music', (project) => ({ ...project, audio: [{ file }] }));
    expect(saves).toHaveLength(1);
    expect(c.getState().project.audio).toEqual([{ file: 'audio/new.mp3' }]);
  });
});

describe('review-03: resetToAiCut follow-up cancellation', () => {
  it('resetToAiCut cannot issue a follow-up get after disposal during version listing', async () => {
    let release;
    const gets = [];
    const api = {
      getClipProject: async (j, i, v) => {
        gets.push(v);
        return p();
      },
      listClipProjectVersions: () => new Promise((r) => (release = r)),
    };
    const c = createProjectController({ jobId: 'j', clipIndex: 0, api });
    await c.load();
    const op = c.resetToAiCut().catch(() => {});
    await vi.waitFor(() => expect(release).toBeTypeOf('function'));
    c.dispose();
    release({ versions: [{ version: 1, origin: 'auto' }] });
    await op;
    expect(gets).toHaveLength(1);
  });

  it('resetToAiCut cannot issue a follow-up get after session reset during version listing', async () => {
    let release;
    const gets = [];
    const api = {
      getClipProject: async (j, i, v) => {
        gets.push(v);
        return p();
      },
      listClipProjectVersions: () => new Promise((r) => (release = r)),
    };
    const c = createProjectController({ jobId: 'j', clipIndex: 0, api });
    await c.load();
    const op = c.resetToAiCut().catch(() => {});
    await vi.waitFor(() => expect(release).toBeTypeOf('function'));
    c.resetSession();
    release({ versions: [{ version: 1, origin: 'auto' }] });
    await op;
    expect(gets).toHaveLength(1);
  });

  it('resetToAiCut still restores the earliest AI cut when the lifetime is intact', async () => {
    const gets = [];
    const api = {
      getClipProject: async (j, i, v) => {
        gets.push(v);
        return { ...p(), version: v == null ? 1 : v };
      },
      listClipProjectVersions: async () => ({
        versions: [
          { version: 3, origin: 'user' },
          { version: 2, origin: 'auto' },
          { version: 1, origin: 'auto' },
        ],
      }),
    };
    const c = createProjectController({ jobId: 'j', clipIndex: 0, api });
    await c.load();
    const proj = await c.resetToAiCut();
    expect(gets).toEqual([undefined, 1]);
    expect(proj.version).toBe(1);
    expect(c.getState().dirty).toBe(true);
  });
});

describe('review-03: UTF-8 punctuation', () => {
  const dir = path.dirname(fileURLToPath(import.meta.url));
  const src = (f) => fs.readFileSync(path.join(dir, f), 'utf8');

  it('Task 16 sources contain no CP437 mojibake', () => {
    for (const f of ['EditorView.jsx', 'CopilotPanel.jsx', 'projectController.js']) {
      const s = src(f);
      for (const b of ['ΓÇö', 'ΓÇª', 'ΓåÆ', '┬╖']) {
        expect(s, `${f} contains mojibake ${b}`).not.toContain(b);
      }
    }
  });

  it('conflict toasts use a real em dash', async () => {
    const toasts = [];
    const server = p();
    const api = {
      getClipProject: async () => clone(server),
      saveClipProject: async () => {
        const e = new Error('stale project version');
        e.status = 409;
        throw e;
      },
    };
    const c = createProjectController({
      jobId: 'j',
      clipIndex: 0,
      api,
      onToast: (m) => toasts.push(m),
    });
    await c.load();
    await expect(
      c.applyEdit('x', (pr) => ({ ...pr }))
    ).rejects.toMatchObject({ status: 409 });
    const all = toasts.join('\n');
    expect(all).toContain('Someone else saved first — reloading the latest version.');
    expect(all).not.toContain('ΓÇö');
  });

  it('visible copy uses real punctuation characters', () => {
    expect(src('CopilotPanel.jsx')).toContain("'Applying…'");
    expect(src('CopilotPanel.jsx')).toContain(
      'Applied — re-export from the editor to render it.'
    );
    expect(src('CopilotPanel.jsx')).toContain('${k} → ${short}');
    expect(src('EditorView.jsx')).toContain('Editing — ');
  });
});

describe('review-03: onAudioUpload awaits the music mutation', () => {
  function musicFixture({ failSave = false } = {}) {
    const toasts = [];
    let server = p();
    const api = {
      getClipProject: async () => clone(server),
      uploadAudio: async (j, file) => ({ file: `audio/${file}` }),
      saveClipProject: async (j, i, proj) => {
        if (failSave) throw new Error('save exploded');
        server = { ...clone(proj), version: server.version + 1 };
        return { project: clone(server), version: server.version };
      },
    };
    const c = createProjectController({
      jobId: 'j',
      clipIndex: 0,
      api,
      onToast: (m) => toasts.push(m),
    });
    return { c, toasts };
  }

  async function renderMusicPanel(failSave) {
    const { c } = musicFixture({ failSave });
    await c.load();
    const pushToasts = [];
    hooks.values = [];
    hooks.index = 0;
    const props = {
      jobId: 'j',
      clipIndex: 0,
      clip: { title: 't' },
      controller: c,
      pushToast: (...a) => pushToasts.push(a),
    };
    const onUpload = openMusicPanel(props);
    return { c, pushToasts, onUpload };
  }

  it('reports success only after the music save is acknowledged', async () => {
    const { c, pushToasts, onUpload } = await renderMusicPanel(false);
    await onUpload('vocals.wav');
    expect(pushToasts).toContainEqual([
      'success',
      'Music attached — mixed at export',
    ]);
    expect(c.getState().project.audio).toEqual([
      { file: 'audio/vocals.wav', volume: 1.0, offset: 0.0 },
    ]);
  });

  it('does not claim success when the music save is rejected', async () => {
    const { pushToasts, onUpload } = await renderMusicPanel(true);
    await onUpload('vocals.wav');
    expect(pushToasts.some((t) => t[0] === 'success')).toBe(false);
    expect(
      pushToasts.some(
        (t) => t[0] === 'error' && String(t[1]).startsWith('Upload failed:')
      )
    ).toBe(true);
  });
});
