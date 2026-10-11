import { useEffect, useRef, useState } from 'react';
import {
  bulkPrepare as apiBulkPrepare,
  bulkApply as apiBulkApply,
  bulkRetry as apiBulkRetry,
  bulkUndo as apiBulkUndo,
  bulkStatus as apiBulkStatus,
  newBulkIdempotencyKey as apiNewKey,
} from '../api/realApi.js';
import {
  BULK_OPERATION_DEFS,
  BULK_VALUE_SETS,
  freezeScope,
  summarizeIntent,
  bulkItemsArray,
  retryableItems,
  undoableItems,
  appliedItems,
  isBulkOpenState,
  bulkOutcomeLabel,
  readBulkOpRef,
  writeBulkOpRef,
  clearBulkOpRef,
} from './bulkEdit.js';

// Task 19 — the bulk editing card.
//
// One shared caption/hook style change across several clips of a job:
// select the operation + params, prepare a frozen proposal (exact job,
// ordered unique clip IDs, acknowledged base versions, per-clip plan),
// inspect it, apply it, inspect partial results, retry unresolved items,
// and undo applied clips without overwriting later edits.
//
// Lifetime (mirrors the Hyperframes card):
// - an UNAPPLIED proposal (idle/preparing/proposal/applying) is only
//   actionable while the card's lifetime is live: the app's lifetimeKey
//   (moves on selection change / session reset) plus the active thread+job
//   must still match the card's own. Anything else renders 'stale' --
//   switching jobs, resetting the session or changing the selection
//   invalidates the unapplied proposal.
// - retry/undo/status act on the server-persisted operation (frozen scope +
//   version guards), so they need the card's thread+job to be active but
//   not the full selection lifetime.
// - the single-clip workspace controller is reconciled, never bypassed:
//   pending saves on an in-scope active clip are drained before the base
//   snapshot is taken, and the controller reloads after apply/undo so its
//   acknowledged version (and any finishing preview bound to the old
//   version) cannot go stale silently.

const defaultApi = {
  bulkPrepare: apiBulkPrepare,
  bulkApply: apiBulkApply,
  bulkRetry: apiBulkRetry,
  bulkUndo: apiBulkUndo,
  bulkStatus: apiBulkStatus,
  newBulkIdempotencyKey: apiNewKey,
};

const boxStyle = {
  background: 'rgba(255,255,255,0.04)', border: '1px solid rgba(255,255,255,0.12)',
  borderRadius: 8, padding: 10, marginTop: 6, fontSize: 13,
};
const btnStyle = {
  marginRight: 8, padding: '5px 12px', borderRadius: 6,
  border: '1px solid rgba(255,255,255,0.25)', background: 'rgba(255,255,255,0.08)',
  color: 'inherit', cursor: 'pointer',
};
const selectStyle = { color: '#000', marginLeft: 6 };
const inputStyle = { color: '#000', marginLeft: 6, width: 90 };

function Field({ label, children }) {
  return (
    <label style={{ display: 'block', marginBottom: 6 }}>
      {label}:{children}
    </label>
  );
}

function SelectParam({ value, onChange, options, testId, allowEmpty }) {
  return (
    <select data-testid={testId} value={value} onChange={(e) => onChange(e.target.value)} style={selectStyle}>
      {allowEmpty && <option value="">—</option>}
      {options.map((o) => <option key={o} value={o}>{o}</option>)}
    </select>
  );
}

export default function BulkEditCard({
  data = {},
  clips = [],
  activeThreadId = null,
  activeJobId = null,
  controller = null,
  lifetimeKey = null,
  api = defaultApi,
}) {
  const { jobId, threadId, scope = {}, text } = data;
  const scopeMode = scope.mode || 'all';
  const frozenIndices = (() => {
    try {
      return freezeScope(clips, scopeMode, scope.clipIndices);
    } catch {
      return [];
    }
  })();

  const [opId, setOpId] = useState('caption_appearance');
  const [params, setParams] = useState({});
  const [onOverride, setOnOverride] = useState('preserve');
  const [state, setState] = useState('idle');
  const [operation, setOperation] = useState(null);
  const [error, setError] = useState(null);
  const [interrupted, setInterrupted] = useState(false);
  const [busy, setBusy] = useState(false);

  const keyRef = useRef(null);
  if (!keyRef.current) keyRef.current = api.newBulkIdempotencyKey();
  // Token captured when the unapplied proposal's lifetime began (prepare
  // start). The render-time stale check compares the CURRENT props against
  // this token -- comparing the live ref against itself would always pass.
  const preparedTokRef = useRef(null);

  // Lifetime: captured at every async start, re-checked after every await.
  const liveRef = useRef(null);
  liveRef.current = { lifetimeKey, activeThreadId, activeJobId, threadId, jobId };
  const captureLifetime = () => ({ ...liveRef.current });
  const lifetimeAlive = (tok) => {
    const cur = liveRef.current;
    return cur.lifetimeKey === tok.lifetimeKey
      && cur.activeThreadId === tok.activeThreadId
      && cur.activeThreadId === cur.threadId
      && cur.activeJobId === tok.activeJobId
      && cur.activeJobId === cur.jobId;
  };
  // Ledger actions (retry/undo/status) need the card's thread+job active;
  // the frozen operation + version guards make them selection-independent.
  const ledgerAlive = () => {
    const cur = liveRef.current;
    return cur.activeThreadId === cur.threadId && cur.activeJobId === cur.jobId;
  };
  // The stale view is only for an UNAPPLIED proposal whose lifetime
  // ended. With no prepared token (fresh idle / discarded) there is no
  // proposal to invalidate, so the card is always live.
  const hasUnappliedProposal = preparedTokRef.current != null;
  const proposalLive = !isBulkOpenState(state)
    || !hasUnappliedProposal
    || lifetimeAlive(preparedTokRef.current);

  const aliveRef = useRef(true);
  const inflightRef = useRef(false);
  // Real lifecycle cleanup: a delayed response after unmount must not
  // touch state. (The test harness runs effects immediately and ignores
  // the returned cleanup, so the component stays usable under it.)
  useEffect(() => () => { aliveRef.current = false; }, []);

  // Reload: a prepared-operation reference persisted for this job+thread
  // reconnects the card to its existing ledger instead of minting a new
  // idempotency key and a duplicate operation. Only the operation/key
  // reference is restored -- never project snapshots or grants.
  const restoreRef = useRef(false);
  useEffect(() => {
    if (restoreRef.current) return;
    restoreRef.current = true;
    const ref = readBulkOpRef(jobId, threadId);
    if (!ref) return;
    keyRef.current = ref.idempotency_key;
    setState('restoring');
    (async () => {
      try {
        const op = await api.bulkStatus(jobId, ref.operation_id);
        if (!aliveRef.current) return;
        setOperation(op);
        setInterrupted(false);
        setState('results');
      } catch {
        clearBulkOpRef(jobId, threadId);
        if (aliveRef.current) setState('idle');
      }
    })();
  }, []);

  const setParam = (k, v) => setParams((p) => ({ ...p, [k]: v }));

  const buildIntent = () => {
    const clean = {};
    for (const [k, v] of Object.entries(params)) {
      if (v === '' || v == null) continue;
      clean[k] = k === 'start' || k === 'end' ? Number(v) : v;
    }
    const intent = { op: opId, params: clean };
    if (opId !== 'hyperframes_finishing') intent.on_override = onOverride;
    return intent;
  };

  const controllerInScope = () => {
    if (!controller || controller.clipIndex == null) return false;
    return frozenIndices.includes(controller.clipIndex);
  };

  const startOver = () => {
    keyRef.current = api.newBulkIdempotencyKey();
    preparedTokRef.current = null;
    clearBulkOpRef(jobId, threadId);
    setOperation(null);
    setError(null);
    setInterrupted(false);
    setParams({});
    setOnOverride('preserve');
    setState('idle');
  };

  const prepare = async () => {
    const tok = captureLifetime();
    preparedTokRef.current = tok;
    setState('preparing');
    setError(null);
    setInterrupted(false);
    try {
      const intent = buildIntent();
      if (!Object.keys(intent.params).length) {
        throw new Error('Pick at least one value for this operation first.');
      }
      // Drain the active clip's serialized queue before the server takes
      // the base snapshot, so the frozen base version is the acknowledged
      // one -- never a stale draft.
      if (controllerInScope() && typeof controller.drain === 'function') {
        await controller.drain();
      }
      if (!lifetimeAlive(tok)) return;
      // The frozen scope is the client's acknowledged enumeration -- for
      // mode "all" that is the displayed clip list, not a bare flag the
      // server would re-enumerate on its own. The server binds it to the
      // current job, so no undisclosed extra clip can enter.
      const op = await api.bulkPrepare(jobId, {
        scope: { mode: scopeMode, clip_indices: frozenIndices },
        intent,
        idempotencyKey: keyRef.current,
      });
      if (!aliveRef.current || !lifetimeAlive(tok)) return;
      setOperation(op);
      writeBulkOpRef(jobId, threadId, {
        operation_id: op.operation_id,
        idempotency_key: keyRef.current,
      });
      setState('proposal');
    } catch (e) {
      if (!aliveRef.current) return;
      if (!lifetimeAlive(tok)) return; // stale view via proposalLive
      setError(e?.message || 'Prepare failed.');
      setState('idle');
    }
  };

  const apply = async () => {
    if (!operation) return;
    // The proposal belongs to the generation captured at prepare start.
    // A handler captured from an older render (stale lifetime, job
    // switch) must refuse instead of sending a stale apply, and a
    // delayed/double handler must not start a second apply.
    const prepTok = preparedTokRef.current;
    if (!prepTok || !lifetimeAlive(prepTok)) return;
    if (inflightRef.current) return;
    inflightRef.current = true;
    try {
      const tok = captureLifetime();
      setState('applying');
      setError(null);
      setInterrupted(false);
      try {
        const op = await api.bulkApply(jobId, operation.operation_id, keyRef.current);
        if (!aliveRef.current || !lifetimeAlive(tok)) return;
        setOperation(op);
        setState('results');
        await reconcileController(op, tok, controller);
      } catch (e) {
        if (!aliveRef.current) return;
        // Lifetime ended mid-apply: leave the open state; proposalLive
        // renders the stale view. (There is no literal 'stale' state -- it
        // matches no render branch by design.)
        if (!lifetimeAlive(tok)) return;
        const answered = e && e.status >= 400 && e.status < 500;
        if (answered) {
          // The server answered and rejected the request: nothing landed
          // under this call. (Per-item conflicts arrive as 200 results.)
          setError(`Apply failed: ${e.message}. Nothing was applied by this request.`);
          setState('error');
          return;
        }
        // The request was issued; the response was lost. Never claim it
        // failed -- report confirmation-unknown until reconciled.
        setInterrupted(true);
        setError(
          'Confirmation unknown: the apply request was sent but its response ' +
          'was interrupted. Nothing is claimed yet -- reconcile to see what ' +
          'actually landed.'
        );
        setState('results');
      }
    } finally {
      inflightRef.current = false;
    }
  };

  const reconcile = async () => {
    if (!operation) return;
    const tok = captureLifetime();
    const ctl = controller;
    setBusy(true);
    setError(null);
    try {
      const op = await api.bulkStatus(jobId, operation.operation_id);
      if (!aliveRef.current || !ledgerAlive() || !lifetimeAlive(tok)) return;
      setOperation(op);
      setInterrupted(false);
      setState('results');
      await reconcileController(op, tok, ctl);
    } catch (e) {
      if (aliveRef.current) setError(`Reconcile failed: ${e?.message || 'unknown error'}`);
    } finally {
      if (aliveRef.current) setBusy(false);
    }
  };

  const retry = async () => {
    if (!operation || !ledgerAlive()) return;
    const tok = captureLifetime();
    const ctl = controller;
    setBusy(true);
    setError(null);
    try {
      const op = await api.bulkRetry(jobId, operation.operation_id, keyRef.current);
      if (!aliveRef.current || !ledgerAlive() || !lifetimeAlive(tok)) return;
      setOperation(op);
      setInterrupted(false);
      setState('results');
      await reconcileController(op, tok, ctl);
    } catch (e) {
      if (aliveRef.current) setError(`Retry failed: ${e?.message || 'unknown error'}`);
    } finally {
      if (aliveRef.current) setBusy(false);
    }
  };

  const undo = async () => {
    if (!operation || !ledgerAlive()) return;
    const tok = captureLifetime();
    const ctl = controller;
    setBusy(true);
    setError(null);
    try {
      const op = await api.bulkUndo(jobId, operation.operation_id, keyRef.current);
      if (!aliveRef.current || !ledgerAlive() || !lifetimeAlive(tok)) return;
      setOperation(op);
      setState('results');
      await reconcileController(op, tok, ctl);
    } catch (e) {
      if (aliveRef.current) setError(`Undo failed: ${e?.message || 'unknown error'}`);
    } finally {
      if (aliveRef.current) setBusy(false);
    }
  };

  // After apply/retry/undo, reload the workspace controller when the active
  // clip was touched: its acknowledged version moved, and any finishing
  // preview bound to the old version is stale by construction (the
  // Hyperframes card's version binding rejects it). The caller's generation
  // must still be live and the controller must be the current one -- a
  // delayed response after a job switch can never revive a disposed
  // controller.
  const reconcileController = async (op, tok, ctl) => {
    const target = ctl === undefined ? controller : ctl;
    if (!target || typeof target.load !== 'function') return;
    if (tok && !lifetimeAlive(tok)) return;
    if (target !== controller) return;
    const touched = bulkItemsArray(op).some(
      (i) => i.clip_index === target.clipIndex
        && (i.outcome === 'applied' || (op.undo?.items?.[i.clip_index]?.outcome === 'undone'))
    );
    if (touched) {
      try { await target.load(); } catch { /* the controller toasts itself */ }
    }
  };

  const discard = () => {
    preparedTokRef.current = null;
    clearBulkOpRef(jobId, threadId);
    setOperation(null);
    setError(null);
    setInterrupted(false);
    setState('idle');
  };

  const renderParams = () => {
    if (opId === 'caption_appearance') {
      return (
        <>
          <Field label="Caption style">
            <SelectParam testId="bulk-param-caption_style" value={params.caption_style || ''}
              onChange={(v) => setParam('caption_style', v)} options={BULK_VALUE_SETS.caption_style} allowEmpty />
          </Field>
          <Field label="Caption position">
            <SelectParam testId="bulk-param-caption_position" value={params.caption_position || ''}
              onChange={(v) => setParam('caption_position', v)} options={BULK_VALUE_SETS.caption_position} allowEmpty />
          </Field>
        </>
      );
    }
    if (opId === 'hook_appearance') {
      return (
        <>
          <Field label="Hook style">
            <SelectParam testId="bulk-param-style" value={params.style || ''}
              onChange={(v) => setParam('style', v)} options={BULK_VALUE_SETS.hook_style} allowEmpty />
          </Field>
          <Field label="Hook position">
            <SelectParam testId="bulk-param-position" value={params.position || ''}
              onChange={(v) => setParam('position', v)} options={BULK_VALUE_SETS.hook_position} allowEmpty />
          </Field>
        </>
      );
    }
    if (opId === 'hook_timing') {
      return (
        <>
          <Field label="Hook start (s)">
            <input data-testid="bulk-param-start" type="number" min="0" step="0.1" style={inputStyle}
              value={params.start ?? ''} onChange={(e) => setParam('start', e.target.value)} />
          </Field>
          <Field label="Hook end (s)">
            <input data-testid="bulk-param-end" type="number" min="0" step="0.1" style={inputStyle}
              value={params.end ?? ''} onChange={(e) => setParam('end', e.target.value)} />
          </Field>
          <div style={{ fontSize: 12, opacity: 0.7, marginBottom: 6 }}>
            End is clamped to each clip&rsquo;s duration and disclosed per clip.
          </div>
        </>
      );
    }
    return (
      <>
        <Field label="Template">
          <SelectParam testId="bulk-param-template" value={params.template || BULK_VALUE_SETS.hf_template[0]}
            onChange={(v) => setParam('template', v)} options={BULK_VALUE_SETS.hf_template} />
        </Field>
        <Field label="Caption size">
          <SelectParam testId="bulk-param-caption_size" value={params.caption_size || 'medium'}
            onChange={(v) => setParam('caption_size', v)} options={BULK_VALUE_SETS.hf_caption_size} />
        </Field>
        <Field label="Caption position">
          <SelectParam testId="bulk-param-hf_caption_position" value={params.caption_position || 'lower-third'}
            onChange={(v) => setParam('caption_position', v)} options={BULK_VALUE_SETS.hf_caption_position} />
        </Field>
        <div style={{ fontSize: 12, opacity: 0.7, marginBottom: 6 }}>
          Optional pilot: only the reviewed templates; unsupported clips are reported per clip, never silently rendered.
        </div>
      </>
    );
  };

  const renderItemRow = (item) => {
    const bits = [];
    for (const c of item.changes || []) {
      bits.push(`${c.field}: ${JSON.stringify(c.from)} → ${JSON.stringify(c.to)}${c.replaced_override ? ' (override replaced)' : ''}`);
    }
    for (const p of item.preserved || []) {
      bits.push(`${p.field}: kept ${JSON.stringify(p.current)} (individual override)`);
    }
    for (const a of item.adapted || []) {
      bits.push(`${a.field}: ${a.requested} → ${a.applied} (${a.reason})`);
    }
    return (
      <div key={item.clip_index} data-testid={`bulk-item-${item.clip_index}`}
        style={{ padding: '6px 0', borderTop: '1px solid rgba(255,255,255,0.08)', fontSize: 12.5 }}>
        <div>
          <strong>Clip {item.clip_index + 1}</strong>
          {' '}· base v{item.expected_version}
          {' '}· <span data-testid={`bulk-outcome-${item.clip_index}`}>{bulkOutcomeLabel(item.outcome)}</span>
          {item.result_version != null && ` → v${item.result_version}`}
          {item.recovered && ' (recovered)'}
        </div>
        {bits.length > 0 && <div style={{ opacity: 0.75, marginTop: 2 }}>{bits.join('; ')}</div>}
        {item.excluded?.reason && <div style={{ opacity: 0.75, marginTop: 2 }}>excluded: {item.excluded.reason}</div>}
        {item.error && <div style={{ color: '#f88', marginTop: 2 }}>{item.error}</div>}
        {item.summary && !bits.length && !item.excluded && (
          <div style={{ opacity: 0.75, marginTop: 2 }}>{item.summary}</div>
        )}
      </div>
    );
  };

  const renderUndoRows = () => {
    const undoItems = Object.values(operation?.undo?.items || {});
    if (!undoItems.length) return null;
    return (
      <div style={{ marginTop: 8 }}>
        <div style={{ fontWeight: 700, marginBottom: 4 }}>
          Undo results{operation.undo?.status ? ` (${operation.undo.status})` : ''}
        </div>
        {undoItems.map((r) => (
          <div key={r.clip_index} data-testid={`bulk-undo-${r.clip_index}`} style={{ fontSize: 12.5, padding: '4px 0' }}>
            Clip {r.clip_index + 1} · {r.outcome === 'undone' ? `undone → v${r.version}` : r.outcome}
            {r.error && <div style={{ color: '#f88' }}>{r.error}</div>}
          </div>
        ))}
      </div>
    );
  };

  if (!jobId) {
    return <div data-testid="bulk-card" style={boxStyle}>Bulk editing needs a finished clip job first.</div>;
  }

  // An unapplied proposal whose lifetime ended renders stale -- never
  // actionable again.
  if (!proposalLive && isBulkOpenState(state)) {
    return (
      <div data-testid="bulk-card" style={boxStyle}>
        <div style={{ fontSize: 12, opacity: 0.75 }}>bulk edit</div>
        <div style={{ marginTop: 6 }}>
          The selection changed since this proposal was prepared, so it is stale and cannot be applied.
        </div>
        <div style={{ marginTop: 8 }}>
          <button data-testid="bulk-start-over" style={btnStyle} onClick={startOver}>Start over</button>
        </div>
      </div>
    );
  }

  const items = operation ? bulkItemsArray(operation) : [];
  const canRetry = operation && retryableItems(operation).length > 0;
  const canUndo = operation && undoableItems(operation).length > 0;
  const scopeLabelFor = (scopeObj) => {
    const idx = Array.isArray(scopeObj?.clip_indices) ? scopeObj.clip_indices : [];
    return scopeObj?.mode === 'all'
      ? `all ${idx.length} clips in this job`
      : `${idx.length} selected clip${idx.length === 1 ? '' : 's'} (${idx.map((i) => i + 1).join(', ')})`;
  };
  // Idle composer: the client-side frozen scope. After prepare, the scope
  // and count render from the server's recorded operation -- the
  // acknowledged frozen enumeration, never the live view.
  const composerScopeLabel = scopeLabelFor({ mode: scopeMode, clip_indices: frozenIndices });
  const operationScopeLabel = operation ? scopeLabelFor(operation.scope) : composerScopeLabel;

  return (
    <div data-testid="bulk-card">
      <div style={{ fontSize: 12, opacity: 0.75 }}>bulk edit — version-aware</div>

      {state === 'idle' && (
        <div style={boxStyle}>
          {text && <div style={{ marginBottom: 8, opacity: 0.85 }}>&ldquo;{text}&rdquo;</div>}
          <div style={{ marginBottom: 8 }}>
            Scope: <strong data-testid="bulk-scope-label">{composerScopeLabel}</strong>
            <div style={{ fontSize: 12, opacity: 0.7, marginTop: 2 }}>
              The proposal freezes this exact clip list and each clip&rsquo;s acknowledged version.
            </div>
          </div>
          <Field label="Operation">
            <select data-testid="bulk-op" value={opId}
              onChange={(e) => { setOpId(e.target.value); setParams({}); }} style={selectStyle}>
              {Object.entries(BULK_OPERATION_DEFS).map(([id, def]) => (
                <option key={id} value={id}>{def.label}</option>
              ))}
            </select>
          </Field>
          <div style={{ fontSize: 12, opacity: 0.7, marginBottom: 8 }}>
            {BULK_OPERATION_DEFS[opId].blurb}
          </div>
          {renderParams()}
          {opId !== 'hyperframes_finishing' && (
            <Field label="Individual overrides">
              <select data-testid="bulk-on-override" value={onOverride}
                onChange={(e) => setOnOverride(e.target.value)} style={selectStyle}>
                <option value="preserve">preserve (default)</option>
                <option value="replace">replace (fields listed per clip)</option>
              </select>
            </Field>
          )}
          <div style={{ marginTop: 8 }}>
            <button data-testid="bulk-prepare" style={btnStyle} onClick={prepare}>
              Prepare proposal
            </button>
          </div>
          {error && <div style={{ color: '#f88', marginTop: 8 }} data-testid="bulk-error">{error}</div>}
        </div>
      )}

      {state === 'preparing' && (
        <div style={boxStyle}>Freezing the proposal — reading each clip&rsquo;s acknowledged version…</div>
      )}

      {state === 'restoring' && (
        <div style={boxStyle}>Reconnecting to the prepared operation — reading the server&rsquo;s recorded truth…</div>
      )}

      {state === 'proposal' && operation && (
        <div style={boxStyle}>
          <div style={{ marginBottom: 6 }}>
            <strong>Proposal</strong> · {operationScopeLabel}
          </div>
          <div style={{ fontSize: 12.5, opacity: 0.8, marginBottom: 6 }} data-testid="bulk-intent-summary">
            {summarizeIntent(operation.intent)}
          </div>
          {Object.keys(operation.baselines || {}).length > 0 && (
            <div style={{ fontSize: 12, opacity: 0.7, marginBottom: 6 }}>
              Common baseline: {Object.entries(operation.baselines).map(([k, v]) => `${k}=${v}`).join(', ')}
            </div>
          )}
          <div data-testid="bulk-proposal-items">
            {items.map(renderItemRow)}
          </div>
          <div style={{ marginTop: 8 }}>
            <button data-testid="bulk-apply" style={btnStyle} onClick={apply}>Apply to {items.length} clips</button>
            <button data-testid="bulk-discard" style={btnStyle} onClick={discard}>Discard</button>
          </div>
          {error && <div style={{ color: '#f88', marginTop: 8 }} data-testid="bulk-error">{error}</div>}
        </div>
      )}

      {state === 'applying' && (
        <div style={boxStyle}>Applying — each clip saves under its frozen version…</div>
      )}

      {state === 'results' && operation && (
        <div style={boxStyle}>
          <div style={{ marginBottom: 6 }}>
            <strong>Results</strong> · {operation.status}
            {' '}· {appliedItems(operation).length}/{items.length} applied
            {' '}· {operationScopeLabel}
          </div>
          {interrupted && (
            <div style={{ color: '#fc6', marginBottom: 6 }} data-testid="bulk-interrupted">
              Confirmation unknown — the request was sent but the response was lost.
              <div style={{ marginTop: 6 }}>
                <button data-testid="bulk-reconcile" style={btnStyle} onClick={reconcile} disabled={busy}>
                  {busy ? 'Reconciling…' : 'Reconcile with the server'}
                </button>
              </div>
            </div>
          )}
          <div data-testid="bulk-result-items">
            {items.map(renderItemRow)}
          </div>
          {renderUndoRows()}
          <div style={{ marginTop: 8 }}>
            {canRetry && ledgerAlive() && (
              <button data-testid="bulk-retry" style={btnStyle} onClick={retry} disabled={busy}>
                {busy ? 'Retrying…' : `Retry ${retryableItems(operation).length} unresolved`}
              </button>
            )}
            {canUndo && ledgerAlive() && (
              <button data-testid="bulk-undo" style={btnStyle} onClick={undo} disabled={busy}>
                {busy ? 'Undoing…' : `Undo ${undoableItems(operation).length} applied`}
              </button>
            )}
            <button data-testid="bulk-done" style={btnStyle} onClick={discard}>Done</button>
          </div>
          {error && <div style={{ color: '#f88', marginTop: 8 }} data-testid="bulk-error">{error}</div>}
        </div>
      )}

      {state === 'error' && (
        <div style={boxStyle}><span style={{ color: '#f88' }}>{error || 'Something went wrong.'}</span>
          <div style={{ marginTop: 8 }}>
            <button data-testid="bulk-start-over" style={btnStyle} onClick={startOver}>Start over</button>
          </div>
        </div>
      )}
    </div>
  );
}
