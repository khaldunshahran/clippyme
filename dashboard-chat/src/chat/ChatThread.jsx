import { useEffect, useRef } from 'react';
import { Loader2, TriangleAlert, Info, Link2, Check, X, Sparkles } from 'lucide-react';
import { ProgressBody } from './ProgressPill.jsx';
import ClipRow from './ClipRow.jsx';
import CaptionCarousel from './CaptionCarousel.jsx';
import HyperframesCard from './HyperframesCard.jsx';
import BulkEditCard from './BulkEditCard.jsx';

/**
 * Single card for a validated link. It MORPHS in place through the job
 * lifecycle instead of spawning a second progress card:
 *  - validating/caption-pick/settings → "Link looks good — set up your clips"
 *  - submitting/clipping/done          → progress bar + % + ETA (same card)
 *  - error                             → honest error state (same card)
 * No duplicate thumbnails, no layout jump: only the bottom line swaps,
 * with a subtle fade (transform/opacity).
 */
function ValidateMsg({ data, thread }) {
  const { state, url, validation, error } = data || {};
  if (state === 'checking') {
    return (
      <div className="nc-card nc-anim-fade-up" style={{ padding: 14, display: 'flex', gap: 10, alignItems: 'center', maxWidth: 480 }}>
        <Loader2 size={16} style={{ animation: 'nc-spin 1.2s linear infinite', color: 'var(--nc-accent)', flexShrink: 0 }} />
        <div style={{ fontSize: 13.5, color: 'var(--nc-text-dim)' }}>
          Checking that link…
          <div style={{ fontSize: 12, color: 'var(--nc-text-faint)', marginTop: 2, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', maxWidth: 380 }}>{url}</div>
        </div>
      </div>
    );
  }
  if (state === 'error') {
    return (
      <div className="nc-card nc-anim-fade-up" style={{ padding: 14, display: 'flex', gap: 10, alignItems: 'center', maxWidth: 480, borderColor: 'rgba(243,18,96,0.4)' }}>
        <TriangleAlert size={16} style={{ color: 'var(--nc-red)', flexShrink: 0 }} />
        <div style={{ fontSize: 13.5 }}>Validation failed: {error}</div>
      </div>
    );
  }
  // done
  const v = validation || {};
  if (v.valid && v.downloadable) {
    const phase = thread?.phase;
    const failed = phase === 'error';
    const morph = phase === 'submitting' || phase === 'clipping' || phase === 'done';
    return (
      <div className="nc-card nc-anim-fade-up" style={{ padding: 12, display: 'flex', gap: 12, alignItems: 'center', maxWidth: 480, borderColor: failed ? 'rgba(243,18,96,0.4)' : 'rgba(52,199,123,0.35)' }}>
        {v.thumbnail
          ? <img src={v.thumbnail} alt="" style={{ width: 96, height: 54, objectFit: 'cover', borderRadius: 8, background: '#000' }} draggable={false} />
          : <div className="nc-skeleton" style={{ width: 96, height: 54, borderRadius: 8 }} />}
        <div style={{ minWidth: 0, flex: 1 }}>
          <div style={{ fontSize: 13, fontWeight: 600, overflow: 'hidden', textOverflow: 'ellipsis', display: '-webkit-box', WebkitLineClamp: 2, WebkitBoxOrient: 'vertical' }}>
            {v.title || 'Video'}
          </div>
          <div key={failed ? 'err' : morph ? 'progress' : 'idle'} className="nc-anim-fade-up" style={{ marginTop: morph || failed ? 10 : 4 }}>
            {failed ? (
              <div style={{ fontSize: 12, color: 'var(--nc-red)', display: 'flex', alignItems: 'center', gap: 5 }}>
                <TriangleAlert size={12} style={{ flexShrink: 0 }} />
                <span>{thread?.error || 'The clip job failed.'}</span>
              </div>
            ) : morph ? (
              <ProgressBody progress={thread?.progress} status={phase === 'done' ? 'done' : undefined} />
            ) : (
              <div style={{ fontSize: 12, color: 'var(--nc-green)', display: 'flex', alignItems: 'center', gap: 5 }}>
                <Link2 size={12} /> Link looks good — set up your clips
              </div>
            )}
          </div>
        </div>
      </div>
    );
  }
  return (
    <div className="nc-card nc-anim-fade-up" style={{ padding: 14, maxWidth: 480, borderColor: 'rgba(243,18,96,0.4)' }}>
      <div style={{ display: 'flex', gap: 10, alignItems: 'center' }}>
        <TriangleAlert size={16} style={{ color: 'var(--nc-red)', flexShrink: 0 }} />
        <div style={{ fontSize: 13.5, fontWeight: 600 }}>That link won't work</div>
      </div>
      {v.reason && <div style={{ fontSize: 13, color: 'var(--nc-text-dim)', marginTop: 6 }}>{v.reason}</div>}
    </div>
  );
}

function patchSummary(patch) {
  if (!patch || typeof patch !== 'object') return 'project patch';
  const keys = Object.keys(patch).filter((k) => k !== 'schema');
  if (!keys.length) return 'project patch';
  return keys.map((k) => {
    const v = patch[k];
    const short = typeof v === 'string' && v.length > 42 ? `${v.slice(0, 42)}…` : JSON.stringify(v)?.slice(0, 42);
    return `${k} → ${short}`;
  }).join(' · ');
}

/**
 * Task 17 — a chat edit proposal for the selected clip.
 * States: thinking | ready | applying | applied | stale | error | discarded.
 * A ready proposal shows its originating clip + base version, is explicitly
 * NOT saved yet, and offers Apply / Discard.
 */
function ProposalMsg({ data, onApply, onDiscard }) {
  const d = data || {};
  const state = d.state || 'thinking';
  const clipN = d.identity && Number.isInteger(d.identity.clipIndex) ? d.identity.clipIndex + 1 : null;
  const scopeLine = clipN != null
    ? `Clip ${clipN}${d.clipTitle ? ` — ${d.clipTitle}` : ''}${d.baseVersion != null ? ` · based on v${d.baseVersion}` : ''}`
    : null;

  const frame = (children, accent) => (
    <div className="nc-card nc-anim-fade-up" style={{ padding: 14, maxWidth: 520, borderColor: accent || 'var(--nc-border)' }}>
      {children}
    </div>
  );

  if (state === 'thinking') {
    return frame(
      <div style={{ display: 'flex', gap: 10, alignItems: 'center' }}>
        <Loader2 size={16} style={{ animation: 'nc-spin 1.2s linear infinite', color: 'var(--nc-accent)', flexShrink: 0 }} />
        <div style={{ fontSize: 13.5, color: 'var(--nc-text-dim)' }}>
          Working on your edit{clipN != null ? ` for clip ${clipN}` : ''}…
        </div>
      </div>
    );
  }

  if (state === 'ready') {
    return frame(
      <div>
        <div style={{ display: 'flex', gap: 8, alignItems: 'flex-start' }}>
          <Sparkles size={15} style={{ color: 'var(--nc-accent)', flexShrink: 0, marginTop: 2 }} />
          <div style={{ fontSize: 13.5, lineHeight: 1.55 }}>{d.explanation || 'Patch ready.'}</div>
        </div>
        {d.patch && (
          <div style={{
            marginTop: 8, padding: '8px 10px', borderRadius: 10,
            background: 'rgba(255,255,255,0.04)', border: '1px solid var(--nc-border)',
            color: 'var(--nc-text-dim)', fontSize: 12,
          }}>
            Changed: {patchSummary(d.patch)}
          </div>
        )}
        <div style={{ marginTop: 8, fontSize: 12, color: 'var(--nc-text-faint)', lineHeight: 1.5 }}>
          {scopeLine && <div>{scopeLine}</div>}
          <div>Not saved yet — review it in the workspace, then apply.</div>
        </div>
        <div style={{ display: 'flex', gap: 8, marginTop: 10 }}>
          <button className="nc-btn-primary" style={{ fontSize: 13, padding: '8px 16px' }} onClick={onApply}>
            <Check size={14} /> Apply to project
          </button>
          <button className="nc-btn-ghost" style={{ fontSize: 13, padding: '8px 16px' }} onClick={onDiscard}>
            <X size={14} /> Discard
          </button>
        </div>
      </div>,
      'rgba(10,129,217,0.4)'
    );
  }

  if (state === 'applying') {
    return frame(
      <div style={{ display: 'flex', gap: 10, alignItems: 'center' }}>
        <Loader2 size={16} style={{ animation: 'nc-spin 1.2s linear infinite', color: 'var(--nc-accent)', flexShrink: 0 }} />
        <div style={{ fontSize: 13.5, color: 'var(--nc-text-dim)' }}>Applying to the project…</div>
      </div>
    );
  }

  if (state === 'applied') {
    return frame(
      <div style={{ display: 'flex', gap: 8, alignItems: 'flex-start' }}>
        <Check size={15} style={{ color: 'var(--nc-green)', flexShrink: 0, marginTop: 2 }} />
        <div style={{ fontSize: 13.5, lineHeight: 1.55 }}>
          Applied{clipN != null ? ` to clip ${clipN}` : ''} — the workspace shows the updated project. Undo is available in the editor.
        </div>
      </div>,
      'rgba(52,199,123,0.35)'
    );
  }

  if (state === 'stale') {
    const why = d.staleReason === 'version'
      ? `This proposal was based on v${d.baseVersion}, but the clip has newer edits — nothing was overwritten. Send the edit again to regenerate it.`
      : 'The selection changed since this proposal was made — nothing was applied. Select the clip again and send the edit to regenerate it.';
    return frame(
      <div style={{ display: 'flex', gap: 8, alignItems: 'flex-start' }}>
        <TriangleAlert size={15} style={{ color: 'var(--nc-accent)', flexShrink: 0, marginTop: 2 }} />
        <div style={{ fontSize: 13.5, lineHeight: 1.55, color: 'var(--nc-text-dim)' }}>{why}</div>
      </div>,
      'rgba(10,129,217,0.4)'
    );
  }

  if (state === 'error') {
    return frame(
      <div style={{ display: 'flex', gap: 8, alignItems: 'flex-start' }}>
        <TriangleAlert size={15} style={{ color: 'var(--nc-red)', flexShrink: 0, marginTop: 2 }} />
        <div style={{ fontSize: 13.5, lineHeight: 1.55, color: 'var(--nc-text-dim)' }}>{d.error || 'The edit failed. Nothing was changed.'}</div>
      </div>,
      'rgba(243,18,96,0.4)'
    );
  }

  // discarded
  return (
    <div className="nc-anim-fade-up" style={{ fontSize: 12.5, color: 'var(--nc-text-faint)' }}>
      Proposal discarded.
    </div>
  );
}

/**
 * Task 17 (pending adapter) + Task 18 (real capability card): the
 * /hyperframes command message is pushed with zero provider calls and zero
 * mutations (contract: ordinary edits stay available). The card itself
 * carries the real capability proposal/apply/preview/render states, and
 * contract 8 is enforced via activeSelectionKey: delayed async responses
 * can never replace another selection's preview.
 */
function HyperframesMsg({ data, activeSelectionKey, controller, controllerKey, lifetimeKey }) {
  const d = data || {};
  const scope = d.scope || {};
  const jobId = scope.jobId ?? null;
  const index = Number.isInteger(scope.clipIndex) ? scope.clipIndex : null;
  const msgKey = jobId != null && index != null ? `${jobId}:${index}` : null;
  // The shared workspace controller belongs to the app's current selection
  // (controllerKey). A card scoped to any other clip must NOT route through
  // it: it would mutate the wrong clip's project. Gate here AND in the card
  // via isLive, so a stale card can never apply to another selection.
  const routedController =
    controller && controllerKey != null && msgKey != null && controllerKey === msgKey
      ? controller
      : null;
  return (
    <HyperframesCard
      data={{
        command: d.command || '/hyperframes',
        jobId,
        index,
        scope: msgKey ? 'single' : 'global',
      }}
      activeSelectionKey={activeSelectionKey}
      controller={routedController}
      lifetimeKey={lifetimeKey}
    />
  );
}

/**
 * Task 19 -- the bulk editing card. The message carries the frozen scope
 * seed {jobId, threadId, scope:{mode, clipIndices?}, text}; the card itself
 * freezes the proposal (base versions + per-clip plan) server-side. The
 * workspace controller is passed for drain-before-prepare / reload-after
 * -apply of an in-scope active clip only -- the card never saves through
 * anything else, and unapplied proposals die with the selection lifetime.
 */
function BulkMsg({ data, thread, controller, lifetimeKey }) {
  const d = data || {};
  return (
    <BulkEditCard
      data={{
        jobId: d.jobId ?? thread?.jobId ?? null,
        threadId: d.threadId ?? thread?.id ?? null,
        scope: d.scope || { mode: 'all' },
        text: d.text || '',
      }}
      clips={thread?.clips || []}
      activeThreadId={thread?.id ?? null}
      activeJobId={thread?.jobId ?? null}
      controller={controller}
      lifetimeKey={lifetimeKey}
    />
  );
}

export default function ChatThread({ thread, activeSelectionKey, controller = null, controllerKey = null, lifetimeKey = null, onOpenClip, onEditClip, onScheduleClip, onViewAll, onCaptionSelect, onCaptionContinue, onApplyProposal, onDiscardProposal }) {
  const bottomRef = useRef(null);
  const msgs = thread?.messages || [];

  // stick to bottom on new messages (instant — no smooth scroll jank)
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ block: 'end' });
  }, [msgs.length, thread?.phase]);

  if (!thread) return null;

  return (
    <div className="nc-scroll" style={{ flex: 1, overflowY: 'auto', padding: '20px 16px 8px' }}>
      <div style={{ maxWidth: 820, margin: '0 auto', display: 'flex', flexDirection: 'column', gap: 14 }}>
        {msgs.map((m) => {
          if (m.role === 'user') {
            return (
              <div key={m.id} className="nc-anim-fade-up" style={{ display: 'flex', justifyContent: 'flex-end' }}>
                <div style={{
                  background: 'var(--nc-card-hover)', border: '1px solid var(--nc-border)',
                  borderRadius: '18px 18px 6px 18px', padding: '10px 14px',
                  maxWidth: '75%', fontSize: 14.5, lineHeight: 1.5, wordBreak: 'break-word',
                }}>
                  {m.text}
                </div>
              </div>
            );
          }
          // AI messages
          switch (m.kind) {
            case 'validate':
              return <div key={m.id}><ValidateMsg data={m.data} thread={thread} /></div>;
            case 'caption-picker':
              return (
                <div key={m.id}>
                  <CaptionCarousel
                    validation={m.data?.validation}
                    selected={thread.settings?.captions || ''}
                    locked={thread.phase !== 'caption-pick'}
                    onSelect={(id) => onCaptionSelect && onCaptionSelect(id)}
                    onContinue={() => onCaptionContinue && onCaptionContinue()}
                  />
                </div>
              );
            case 'progress':
              // Legacy: progress used to be a separate message (pre-merge).
              // The validation card now morphs in place; render nothing so
              // old persisted threads don't show a duplicate card.
              return null;
            case 'clips':
              return (
                <div key={m.id}>
                  <ClipRow
                    clips={thread.clips}
                    onOpenClip={onOpenClip}
                    onEditClip={onEditClip}
                    onScheduleClip={onScheduleClip}
                    onViewAll={onViewAll}
                  />
                </div>
              );
            case 'proposal':
              return (
                <div key={m.id}>
                  <ProposalMsg
                    data={m.data}
                    onApply={() => onApplyProposal && onApplyProposal(m.id)}
                    onDiscard={() => onDiscardProposal && onDiscardProposal(m.id)}
                  />
                </div>
              );
            case 'hyperframes':
              return <div key={m.id}><HyperframesMsg data={m.data} activeSelectionKey={activeSelectionKey} controller={controller} controllerKey={controllerKey} lifetimeKey={lifetimeKey} /></div>;
            case 'bulk':
              return <div key={m.id}><BulkMsg data={m.data} thread={thread} controller={controller} lifetimeKey={lifetimeKey} /></div>;
            case 'notice':
              return (
                <div key={m.id} className="nc-anim-fade-up" style={{ display: 'flex', gap: 8, alignItems: 'flex-start', maxWidth: 640 }}>
                  <Info size={15} style={{ color: 'var(--nc-accent)', flexShrink: 0, marginTop: 2 }} />
                  <div style={{ fontSize: 13.5, color: 'var(--nc-text-dim)', lineHeight: 1.5 }}>{m.text}</div>
                </div>
              );
            case 'error':
              return (
                <div key={m.id} className="nc-anim-fade-up" style={{ display: 'flex', gap: 8, alignItems: 'flex-start', maxWidth: 640 }}>
                  <TriangleAlert size={15} style={{ color: 'var(--nc-red)', flexShrink: 0, marginTop: 2 }} />
                  <div style={{ fontSize: 13.5, color: 'var(--nc-text-dim)', lineHeight: 1.5 }}>{m.text}</div>
                </div>
              );
            default:
              return (
                <div key={m.id} className="nc-anim-fade-up" style={{ maxWidth: 640, fontSize: 14.5, lineHeight: 1.6, color: 'var(--nc-text)' }}>
                  {m.text}
                </div>
              );
          }
        })}
        <div ref={bottomRef} style={{ height: 4 }} />
      </div>
    </div>
  );
}
