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
import { describe, expect, it, vi, beforeEach, afterEach } from 'vitest';
import React from 'react';
vi.stubGlobal('React', React);

const hooks = vi.hoisted(() => ({ values: [], index: 0, effects: [], cleanups: [] }));
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
        hooks.effects.push(fn); const r = fn(); if(typeof r === "function") hooks.cleanups.push(r);
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

describe('reload reconnect (review01)', () => {
  const flush = () => new Promise((r) => setTimeout(r, 0));
  // Direct globalThis assignment (not vi.stubGlobal): unstubbing all vi
  // globals would also drop the React global the harness relies on.
  afterEach(() => { delete globalThis.localStorage; });

  function stubStorage(initial = {}) {
    const store = { ...initial };
    globalThis.localStorage = {
      getItem: (k) => (k in store ? store[k] : null),
      setItem: (k, v) => { store[k] = String(v); },
      removeItem: (k) => { delete store[k]; },
    };
    return store;
  }

  it('reconciles the persisted operation reference instead of minting a new key', async () => {
    stubStorage({
      'nugget-bulk-op:job1:t1': JSON.stringify({
        operation_id: 'op9', idempotency_key: 'restored-key-9',
      }),
    });
    const serverOp = preparedOp([
      { ...plannedItem(0), outcome: 'applied', result_version: 3 },
      { ...plannedItem(1), outcome: 'failed', error: 'boom' },
    ]);
    serverOp.operation_id = 'op9';
    serverOp.status = 'partial';
    bulkApi.bulkStatus.mockResolvedValue(serverOp);
    bulkApi.bulkRetry.mockResolvedValue(serverOp);

    render(baseProps());
    await flush();
    const tree = render(baseProps());
    expect(bulkApi.bulkStatus).toHaveBeenCalledWith('job1', 'op9');
    expect(bulkApi.bulkPrepare).not.toHaveBeenCalled();
    expect(findByTestId(tree, 'bulk-result-items')).toBeTruthy();
    expect(textOf(findByTestId(tree, 'bulk-outcome-0'))).toMatch(/applied/);
    // Ledger actions reuse the restored key, not a fresh one.
    await click(tree, 'bulk-retry');
    expect(bulkApi.bulkRetry).toHaveBeenCalledWith('job1', 'op9', 'restored-key-9');
  });

  it('a dead reference clears and returns to idle without a new operation', async () => {
    const store = stubStorage({
      'nugget-bulk-op:job1:t1': JSON.stringify({
        operation_id: 'op-gone', idempotency_key: 'dead-key',
      }),
    });
    bulkApi.bulkStatus.mockRejectedValue(Object.assign(new Error('not found'), { status: 404 }));

    render(baseProps());
    await flush();
    const tree = render(baseProps());
    expect(bulkApi.bulkPrepare).not.toHaveBeenCalled();
    expect(store['nugget-bulk-op:job1:t1']).toBeUndefined();
    // Back at the composer: no result items, prepare still available.
    expect(findByTestId(tree, 'bulk-result-items')).toBeNull();
    expect(findByTestId(tree, 'bulk-prepare')).toBeTruthy();
  });
});

describe('stale/double handlers (review01)', () => {
  async function toProposal() {
    bulkApi.bulkPrepare.mockResolvedValue(preparedOp([plannedItem(0)]));
    let tree = render(baseProps());
    pickCaptionStyle(tree);
    tree = render(baseProps());
    await click(tree, 'bulk-prepare');
    return render(baseProps());
  }

  it('a delayed/double apply handler starts only one apply', async () => {
    const tree = await toProposal();
    let release;
    bulkApi.bulkApply.mockImplementation(() => new Promise((r) => { release = r; }));
    const applyHandler = findByTestId(tree, 'bulk-apply').props.onClick;
    const p1 = applyHandler();
    const p2 = applyHandler();
    release(preparedOp([{ ...plannedItem(0), outcome: 'applied', result_version: 3 }]));
    await p1;
    await p2;
    expect(bulkApi.bulkApply).toHaveBeenCalledTimes(1);
  });
});

// Coordinator review02: actual effect setup/cleanup replay and reload states.
it('review02: StrictMode setup cleanup setup preserves a usable card', async()=>{
  const p=baseProps(); hooks.effects=[];hooks.cleanups=[];
  let tree=render(p);
  const setups=[...hooks.effects];const cleanups=[...hooks.cleanups];
  cleanups.forEach(fn=>fn());setups.forEach(fn=>fn());
  bulkApi.bulkPrepare.mockResolvedValue(preparedOp([plannedItem(0)]));
  pickCaptionStyle(tree);tree=render(p);await click(tree,'bulk-prepare');
  expect(findByTestId(render(p),'bulk-apply')).toBeTruthy();
});
function reviewStorage(){
  const key='nugget-bulk-op:job1:t1';const store={[key]:JSON.stringify({operation_id:'op1',idempotency_key:'test-key-1'})};
  globalThis.localStorage={getItem:k=>store[k]??null,setItem:(k,v)=>store[k]=v,removeItem:k=>delete store[k]};
  return {key,store};
}
it('review02: reload of a proposed operation can explicitly apply that same operation',async()=>{
  reviewStorage();bulkApi.bulkStatus.mockResolvedValue(preparedOp([plannedItem(0)]));
  try {
    render(baseProps());await flush();const tree=render(baseProps());
    expect(findByTestId(tree,'bulk-apply')).toBeTruthy();
    bulkApi.bulkApply.mockResolvedValue(preparedOp([{...plannedItem(0),outcome:'applied',result_version:3}]));
    await click(tree,'bulk-apply');
    expect(bulkApi.bulkApply.mock.calls[0]).toEqual(['job1','op1','test-key-1']);
    expect(bulkApi.bulkPrepare).not.toHaveBeenCalled();
  }finally{delete globalThis.localStorage;}
});
it('review02: transient status failure retains the durable operation reference',async()=>{
  const {key,store}=reviewStorage();bulkApi.bulkStatus.mockRejectedValue(new Error('network interrupted'));
  try {
    render(baseProps());await flush();render(baseProps());
    expect(store[key]).toBeTruthy();
    expect(bulkApi.bulkPrepare).not.toHaveBeenCalled();
  }finally{delete globalThis.localStorage;}
});
async function flush(){for(let i=0;i<8;i++) await Promise.resolve();}

it('review02: delayed first-generation restore response stays ignored after replay', async () => {
  // Durable control for generation-scoped effect ownership: setup ->
  // cleanup -> setup, then the STALE first-generation response resolves
  // before the live one. Only the live generation may become a proposal.
  reviewStorage(); // op1/test-key-1 in storage
  const deferreds = [];
  bulkApi.bulkStatus.mockImplementation(() => new Promise((r) => { deferreds.push(r); }));
  try {
    const p = baseProps();
    hooks.effects = []; hooks.cleanups = [];
    render(p); // setup1 (gen 1): fires fetch A
    const setups = [...hooks.effects];
    const cleanups = [...hooks.cleanups];
    cleanups.forEach((fn) => fn()); // cleanup1: gen 1 invalidated
    setups.forEach((fn) => fn());   // setup2 (gen 2): fires fetch B
    expect(deferreds.length).toBe(2);
    const stale = preparedOp([plannedItem(0)]);
    stale.operation_id = 'op-stale';
    deferreds[0](stale); // delayed first-generation response
    deferreds[1](preparedOp([plannedItem(0)])); // live generation response
    await flush();
    const tree = render(p);
    // The stale response never became state; the live one is an actionable proposal.
    expect(findByTestId(tree, 'bulk-apply')).toBeTruthy();
    bulkApi.bulkApply.mockResolvedValue(
      preparedOp([{ ...plannedItem(0), outcome: 'applied', result_version: 3 }])
    );
    await click(tree, 'bulk-apply');
    // Same operation/key the reference carried -- not the stale payload.
    expect(bulkApi.bulkApply.mock.calls[0]).toEqual(['job1', 'op1', 'test-key-1']);
    expect(bulkApi.bulkPrepare).not.toHaveBeenCalled();
  } finally { delete globalThis.localStorage; }
});
