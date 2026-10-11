/**
 * PROMOTED coordinator probe (Task 19 review01).
 *
 * Original: work/muse-handoff/task-19/coordinator-review01.test.js.
 * Production-handler gates: stale Apply refusal, all-scope frozen
 * snapshot, retry-after-job-switch controller safety. Same house
 * harness as bulkEditCard.test.js.
 */
/**
 * Task 19 — BulkEditCard tests.
 *
 * House pattern (hyperframesCard.test.js): minimal react hook harness with
 * persistent useState/useRef, immediate useEffect, mocked api transport,
 * and direct component calls with a manual element-tree walk.
 *
 * The card composes <Field>/<SelectParam> subcomponents, which the harness
 * does not render: the walk matches their props.testId convention directly.
 * SelectParam's onChange receives the raw value (not a DOM event).
 *
 * Contracts covered:
 *  1. Frozen proposal: prepare shows the intent summary, baselines and the
 *     per-clip plan (changes / preserved overrides / adaptations /
 *     exclusions) for the frozen scope.
 *  2. Explicit Apply only: routing/preparing never saves; apply posts only
 *     (jobId, operation_id, idempotency key).
 *  3. Partial results stay partial: per-clip outcomes render; only failed/
 *     pending items are retryable (conflict is terminal); applied items
 *     keep their acknowledged version and are never reapplied.
 *  4. Undo: targets applied clips; per-clip undo outcomes render
 *     (undone -> vN, conflict on later edits).
 *  5. Lifetime: a selection change after prepare renders the proposal
 *     stale (not actionable); start over mints a fresh idempotency key.
 *  6. Interrupted apply: no false failure claim -- confirmation-unknown
 *     until reconcile; reconcile adopts the server's recorded truth
 *     (including recovered items).
 *  7. Controller reconciliation: pending saves on an in-scope active clip
 *     are drained before the base snapshot; the controller reloads after
 *     apply touches its clip.
 *
 * Load-bearing mutations (run manually, each must fail, then re-run green):
 *  M1: prepare posts without the idempotency key -> key assertion fails.
 *  M2: apply re-sends per-item patches instead of (operation_id, key) ->
 *      apply body assertion fails.
 *  M3: retry button shown for conflict items -> retry eligibility fails.
 *  M4: lifetime check removed from the proposal render -> stale test fails.
 *  M5: interrupted apply claims failure instead of unknown -> reconcile
 *      test fails.
 */
import { describe, expect, it, vi, beforeEach } from 'vitest';
import React from 'react';
vi.stubGlobal('React', React);

const hooks = vi.hoisted(() => ({ values: [], index: 0 }));
const bulkApi = vi.hoisted(() => ({
  bulkPrepare: vi.fn(),
  bulkApply: vi.fn(),
  bulkRetry: vi.fn(),
  bulkUndo: vi.fn(),
  bulkStatus: vi.fn(),
  newBulkIdempotencyKey: vi.fn(() => 'test-key-1'),
}));

vi.mock('react', async (original) => {
  const actual = await original();
  return {
    ...actual,
    useState: (initial) => {
      const i = hooks.index++;
      if (!(i in hooks.values)) hooks.values[i] = typeof initial === 'function' ? initial() : initial;
      return [
        hooks.values[i],
        (v) => { hooks.values[i] = typeof v === 'function' ? v(hooks.values[i]) : v; },
      ];
    },
    useRef: (initial) => {
      const i = hooks.index++;
      if (!(i in hooks.values)) hooks.values[i] = { current: initial };
      return hooks.values[i];
    },
    useMemo: (fn) => fn(),
    useCallback: (fn) => fn,
    useEffect: (fn) => {
      try {
        const r = fn();
        if (r && typeof r.catch === 'function') r.catch(() => {});
      } catch { /* node env: best-effort */ }
    },
  };
});
vi.mock('../api/realApi.js', () => ({ ...bulkApi }));

import BulkEditCard from './BulkEditCard.jsx';

// --- element tree helpers ---
// The harness calls function components directly without rendering nested
// ones, so <SelectParam testId="..."> elements are matched by props.testId
// as well as the DOM props['data-testid'] convention.
function walk(node, out = []) {
  if (Array.isArray(node)) { node.forEach((k) => walk(k, out)); return out; }
  if (!node || typeof node !== 'object') return out;
  out.push(node);
  const kids = node.props?.children;
  if (Array.isArray(kids)) kids.forEach((k) => walk(k, out));
  else if (kids) walk(kids, out);
  return out;
}
function findByTestId(root, id) {
  return (
    walk(root).find(
      (n) => n.props && (n.props['data-testid'] === id || n.props.testId === id),
    ) || null
  );
}
function textOf(root) {
  const parts = [];
  for (const n of walk(root)) {
    const c = n.props?.children;
    if (typeof c === 'string' || typeof c === 'number') parts.push(String(c));
    else if (Array.isArray(c)) {
      for (const k of c) {
        if (typeof k === 'string' || typeof k === 'number') parts.push(String(k));
      }
    }
  }
  return parts.join(' ');
}

const baseProps = (over = {}) => ({
  data: {
    jobId: 'job1', threadId: 't1',
    scope: { mode: 'all' }, text: 'make them bold',
  },
  clips: [{ index: 0 }, { index: 1 }, { index: 2 }],
  activeThreadId: 't1',
  activeJobId: 'job1',
  controller: null,
  lifetimeKey: 'L1',
  api: bulkApi,
  ...over,
});

function render(props) {
  hooks.index = 0;
  return BulkEditCard(props);
}
async function click(tree, testId) {
  const el = findByTestId(tree, testId);
  expect(el, `missing [data-testid=${testId}]`).toBeTruthy();
  await el.props.onClick();
}
// SelectParam's onChange receives the raw value, not a DOM event.
function pickCaptionStyle(tree, value = 'hormozi_bold') {
  const sel = findByTestId(tree, 'bulk-param-caption_style');
  expect(sel, 'missing bulk-param-caption_style').toBeTruthy();
  sel.props.onChange(value);
}

const plannedItem = (clip_index, extra = {}) => ({
  clip_index,
  expected_version: 2,
  outcome: 'planned',
  patch: { 'captions.style': 'hormozi_bold' },
  changes: [{ field: 'caption_style', from: 'classic_white', to: 'hormozi_bold', replaced_override: false }],
  preserved: [],
  adapted: [],
  excluded: null,
  summary: '',
  result_version: null,
  ...extra,
});
const preparedOp = (items) => ({
  operation_id: 'op1',
  idempotency_key: 'test-key-1',
  status: 'proposed',
  scope: { mode: 'all', clip_indices: [0, 1, 2], count: 3 },
  intent: { op: 'caption_appearance', params: { caption_style: 'hormozi_bold' }, on_override: 'preserve' },
  baselines: { caption_style: 'classic_white' },
  base_versions: { 0: 2, 1: 2, 2: 2 },
  items: Object.fromEntries(items.map((i) => [String(i.clip_index), i])),
});

beforeEach(() => {
  hooks.values = [];
  hooks.index = 0;
  Object.values(bulkApi).forEach((f) => f.mockReset && f.mockReset());
  bulkApi.newBulkIdempotencyKey.mockReturnValue('test-key-1');
});

describe('composer (idle)', () => {
  it('shows the frozen scope label and the operation picker', () => {
    const tree = render(baseProps());
    expect(findByTestId(tree, 'bulk-scope-label')).toBeTruthy();
    expect(textOf(findByTestId(tree, 'bulk-scope-label'))).toMatch(/all 3 clips/);
    expect(findByTestId(tree, 'bulk-op')).toBeTruthy();
    expect(findByTestId(tree, 'bulk-prepare')).toBeTruthy();
  });

  it('selected scope lists the explicit clip numbers', () => {
    const tree = render(baseProps({
      data: { jobId: 'job1', threadId: 't1', scope: { mode: 'selected', clipIndices: [2, 0] }, text: '' },
    }));
    expect(textOf(findByTestId(tree, 'bulk-scope-label'))).toMatch(/2 selected clips \(1, 3\)/);
  });

  it('requires at least one param value before preparing', async () => {
    const tree = render(baseProps());
    await click(tree, 'bulk-prepare');
    const tree2 = render(baseProps());
    expect(textOf(tree2)).toMatch(/at least one value/i);
    expect(bulkApi.bulkPrepare).not.toHaveBeenCalled();
  });
});

describe('prepare -> proposal', () => {
  it('freezes the proposal and shows intent, baselines and per-clip plan rows', async () => {
    bulkApi.bulkPrepare.mockResolvedValue(preparedOp([
      plannedItem(0), plannedItem(1),
      plannedItem(2, {
        outcome: 'preserved', patch: {},
        changes: [],
        preserved: [{ field: 'caption_style', current: 'neon_glow', baseline: 'classic_white', reason: 'individual override preserved' }],
      }),
    ]));
    let tree = render(baseProps());
    pickCaptionStyle(tree);
    tree = render(baseProps());
    await click(tree, 'bulk-prepare');
    tree = render(baseProps());
    expect(bulkApi.bulkPrepare).toHaveBeenCalledTimes(1);
    const [jobId, body] = bulkApi.bulkPrepare.mock.calls[0];
    expect(jobId).toBe('job1');
    expect(body.scope).toEqual({ mode: 'all', clip_indices: [0, 1, 2] });
    expect(body.intent).toEqual({
      op: 'caption_appearance',
      params: { caption_style: 'hormozi_bold' },
      on_override: 'preserve',
    });
    expect(body.idempotencyKey).toBe('test-key-1');
    // Proposal view: intent summary, baselines, per-clip rows.
    expect(textOf(tree)).toMatch(/Caption style: caption_style=hormozi_bold/);
    expect(textOf(tree)).toMatch(/caption_style=classic_white/);
    expect(findByTestId(tree, 'bulk-item-0')).toBeTruthy();
    expect(findByTestId(tree, 'bulk-item-2')).toBeTruthy();
    expect(textOf(findByTestId(tree, 'bulk-outcome-2'))).toMatch(/preserved override/);
    expect(findByTestId(tree, 'bulk-apply')).toBeTruthy();
  });

  it('drains the active clip controller before the base snapshot', async () => {
    bulkApi.bulkPrepare.mockResolvedValue(preparedOp([plannedItem(0)]));
    const drain = vi.fn(async () => {});
    const controller = { clipIndex: 1, drain, load: vi.fn(async () => {}) };
    let tree = render(baseProps({ controller }));
    pickCaptionStyle(tree);
    tree = render(baseProps({ controller }));
    await click(tree, 'bulk-prepare');
    expect(drain).toHaveBeenCalledTimes(1);
    expect(bulkApi.bulkPrepare).toHaveBeenCalledTimes(1);
  });
});

describe('apply -> results', () => {
  async function toProposal() {
    bulkApi.bulkPrepare.mockResolvedValue(preparedOp([plannedItem(0), plannedItem(1)]));
    let tree = render(baseProps());
    pickCaptionStyle(tree);
    tree = render(baseProps());
    await click(tree, 'bulk-prepare');
    return render(baseProps());
  }

  it('applies the frozen operation and shows per-clip outcomes', async () => {
    const applied = preparedOp([
      { ...plannedItem(0), outcome: 'applied', result_version: 3 },
      { ...plannedItem(1), outcome: 'applied', result_version: 3 },
    ]);
    applied.status = 'applied';
    bulkApi.bulkApply.mockResolvedValue(applied);
    let tree = await toProposal();
    await click(tree, 'bulk-apply');
    tree = render(baseProps());
    expect(bulkApi.bulkApply).toHaveBeenCalledTimes(1);
    // Apply posts only the frozen reference -- never per-item patches.
    expect(bulkApi.bulkApply.mock.calls[0]).toEqual(['job1', 'op1', 'test-key-1']);
    expect(textOf(findByTestId(tree, 'bulk-outcome-0'))).toMatch(/applied/);
    expect(textOf(tree)).toMatch(/2\s*\/\s*2\s+applied/);
    // No retry when nothing is unresolved; undo is offered.
    expect(findByTestId(tree, 'bulk-retry')).toBeNull();
    expect(findByTestId(tree, 'bulk-undo')).toBeTruthy();
  });

  it('partial results: failed is retryable, conflict is terminal, applied is never reapplied', async () => {
    const partial = preparedOp([
      { ...plannedItem(0), outcome: 'applied', result_version: 3 },
      { ...plannedItem(1), outcome: 'failed', error: 'save timed out' },
    ]);
    partial.status = 'partial';
    bulkApi.bulkApply.mockResolvedValue(partial);
    const afterRetry = preparedOp([
      { ...plannedItem(0), outcome: 'applied', result_version: 3 },
      { ...plannedItem(1), outcome: 'conflict', error: 'clip is at v3, proposal froze v2', current_version: 3 },
    ]);
    afterRetry.status = 'partial';
    bulkApi.bulkRetry.mockResolvedValue(afterRetry);

    let tree = await toProposal();
    await click(tree, 'bulk-apply');
    tree = render(baseProps());
    expect(textOf(findByTestId(tree, 'bulk-outcome-1'))).toMatch(/failed/);
    expect(findByTestId(tree, 'bulk-retry')).toBeTruthy();

    await click(tree, 'bulk-retry');
    tree = render(baseProps());
    expect(bulkApi.bulkRetry).toHaveBeenCalledTimes(1);
    expect(bulkApi.bulkRetry.mock.calls[0]).toEqual(['job1', 'op1', 'test-key-1']);
    // Clip 1 moved between prepare and retry: conflict is terminal, so the
    // retry button is gone. Clip 0 kept its acknowledged version.
    expect(textOf(findByTestId(tree, 'bulk-outcome-1'))).toMatch(/conflict/);
    expect(findByTestId(tree, 'bulk-retry')).toBeNull();
    expect(textOf(findByTestId(tree, 'bulk-outcome-0'))).toMatch(/applied/);
    expect(textOf(findByTestId(tree, 'bulk-item-0'))).toMatch(/v3/);
  });

  it('undo targets applied clips and shows per-clip undo outcomes', async () => {
    const applied = preparedOp([
      { ...plannedItem(0), outcome: 'applied', result_version: 3 },
      { ...plannedItem(1), outcome: 'applied', result_version: 3 },
    ]);
    applied.status = 'applied';
    bulkApi.bulkApply.mockResolvedValue(applied);
    const undone = {
      ...applied,
      undo: {
        status: 'partial',
        items: {
          0: { clip_index: 0, outcome: 'undone', version: 4 },
          1: { clip_index: 1, outcome: 'conflict', error: 'a later edit exists', current_version: 5 },
        },
      },
    };
    bulkApi.bulkUndo.mockResolvedValue(undone);

    let tree = await toProposal();
    await click(tree, 'bulk-apply');
    tree = render(baseProps());
    await click(tree, 'bulk-undo');
    tree = render(baseProps());
    expect(bulkApi.bulkUndo).toHaveBeenCalledTimes(1);
    expect(bulkApi.bulkUndo.mock.calls[0]).toEqual(['job1', 'op1', 'test-key-1']);
    expect(textOf(findByTestId(tree, 'bulk-undo-0'))).toMatch(/undone/);
    expect(textOf(findByTestId(tree, 'bulk-undo-1'))).toMatch(/conflict/);
  });

  it('reloads the workspace controller when its clip was touched', async () => {
    const applied = preparedOp([{ ...plannedItem(1), outcome: 'applied', result_version: 3 }]);
    applied.status = 'applied';
    bulkApi.bulkApply.mockResolvedValue(applied);
    const load = vi.fn(async () => {});
    const controller = { clipIndex: 1, drain: vi.fn(async () => {}), load };
    bulkApi.bulkPrepare.mockResolvedValue(preparedOp([plannedItem(1)]));

    let tree = render(baseProps({ controller }));
    pickCaptionStyle(tree);
    tree = render(baseProps({ controller }));
    await click(tree, 'bulk-prepare');
    tree = render(baseProps({ controller }));
    await click(tree, 'bulk-apply');
    expect(load).toHaveBeenCalledTimes(1);
  });
});

describe('lifetime', () => {
  it('a selection change after prepare renders the proposal stale', async () => {
    bulkApi.bulkPrepare.mockResolvedValue(preparedOp([plannedItem(0)]));
    let tree = render(baseProps());
    pickCaptionStyle(tree);
    tree = render(baseProps());
    await click(tree, 'bulk-prepare');
    // Same card, new lifetime (selection changed).
    tree = render(baseProps({ lifetimeKey: 'L2' }));
    expect(findByTestId(tree, 'bulk-apply')).toBeNull();
    expect(textOf(tree)).toMatch(/stale/i);
    expect(findByTestId(tree, 'bulk-start-over')).toBeTruthy();
  });

  it('start over mints a fresh idempotency key', async () => {
    bulkApi.newBulkIdempotencyKey
      .mockReturnValueOnce('test-key-1')
      .mockReturnValueOnce('test-key-2');
    bulkApi.bulkPrepare.mockResolvedValue(preparedOp([plannedItem(0)]));
    let tree = render(baseProps());
    pickCaptionStyle(tree);
    tree = render(baseProps());
    await click(tree, 'bulk-prepare');
    tree = render(baseProps({ lifetimeKey: 'L2' }));
    await click(tree, 'bulk-start-over');
    tree = render(baseProps({ lifetimeKey: 'L2' }));
    pickCaptionStyle(tree);
    tree = render(baseProps({ lifetimeKey: 'L2' }));
    await click(tree, 'bulk-prepare');
    const keys = bulkApi.bulkPrepare.mock.calls.map(([, b]) => b.idempotencyKey);
    expect(keys).toEqual(['test-key-1', 'test-key-2']);
  });
});

describe('interrupted apply', () => {
  async function toProposal() {
    bulkApi.bulkPrepare.mockResolvedValue(preparedOp([plannedItem(0), plannedItem(1)]));
    let tree = render(baseProps());
    pickCaptionStyle(tree);
    tree = render(baseProps());
    await click(tree, 'bulk-prepare');
    return render(baseProps());
  }

  it('reports confirmation-unknown and reconciles with the server truth', async () => {
    bulkApi.bulkApply.mockRejectedValue(new Error('network down'));
    const reconciled = preparedOp([
      { ...plannedItem(0), outcome: 'applied', result_version: 3, recovered: true },
      { ...plannedItem(1), outcome: 'applied', result_version: 3 },
    ]);
    reconciled.status = 'applied';
    bulkApi.bulkStatus.mockResolvedValue(reconciled);

    let tree = await toProposal();
    await click(tree, 'bulk-apply');
    tree = render(baseProps());
    // Never claims failure: confirmation unknown + reconcile offered.
    expect(textOf(tree)).toMatch(/confirmation unknown/i);
    expect(findByTestId(tree, 'bulk-reconcile')).toBeTruthy();
    expect(textOf(tree)).not.toMatch(/nothing was applied by this request/i);

    await click(tree, 'bulk-reconcile');
    tree = render(baseProps());
    expect(bulkApi.bulkStatus).toHaveBeenCalledWith('job1', 'op1');
    expect(textOf(findByTestId(tree, 'bulk-outcome-0'))).toMatch(/applied/);
    expect(textOf(tree)).not.toMatch(/confirmation unknown/i);
  });

  it('an answered 4xx rejection is an honest failure, not unknown', async () => {
    const err = new Error('idempotency key does not match this operation');
    err.status = 409;
    bulkApi.bulkApply.mockRejectedValue(err);
    let tree = await toProposal();
    await click(tree, 'bulk-apply');
    tree = render(baseProps());
    expect(textOf(tree)).toMatch(/nothing was applied by this request/i);
    expect(findByTestId(tree, 'bulk-reconcile')).toBeNull();
  });
});

// Coordinator production-handler gates (Task19 review01).
async function coordinatorProposal(props=baseProps()) {
  bulkApi.bulkPrepare.mockResolvedValue(preparedOp([plannedItem(0)]));
  let tree=render(props); pickCaptionStyle(tree); tree=render(props);
  await click(tree,'bulk-prepare'); return render(props);
}
it('coordinator: stale captured Apply handler refuses the request',async()=>{
  const p=baseProps(); const tree=await coordinatorProposal(p);
  const oldApply=findByTestId(tree,'bulk-apply').props.onClick;
  render({...p,lifetimeKey:'L2'});
  await oldApply();
  expect(bulkApi.bulkApply).not.toHaveBeenCalled();
});
it('coordinator: all scope sends the displayed frozen clip snapshot',async()=>{
  await coordinatorProposal();
  expect(bulkApi.bulkPrepare.mock.calls[0][1].scope.clip_indices).toEqual([0,1,2]);
});
it('coordinator: retry response after job switch cannot reload the old controller',async()=>{
  const ctl={clipIndex:0,drain:vi.fn(async()=>{}),load:vi.fn(async()=>{})};
  const p=baseProps({controller:ctl});
  const tree=await coordinatorProposal(p);
  const partial=preparedOp([{...plannedItem(0),outcome:'failed',error:'temporary'}]);
  partial.status='partial';bulkApi.bulkApply.mockResolvedValue(partial);
  await click(tree,'bulk-apply');
  let finish;bulkApi.bulkRetry.mockImplementation(()=>new Promise(r=>{finish=r;}));
  const waiting=findByTestId(render(p),'bulk-retry').props.onClick();
  render({...p,activeThreadId:'t2',activeJobId:'job2',lifetimeKey:'L2'});
  finish(preparedOp([{...plannedItem(0),outcome:'applied',result_version:3}]));
  await waiting;
  expect(ctl.load).not.toHaveBeenCalled();
});
