import { useState } from 'react';
import { Bot, X, ChevronDown, Loader2, Check, Sparkles } from 'lucide-react';
import { editAiPatch } from '../api/chatApi.js';
import { saveClipProject } from '../api/realApi.js';

/**
 * Floating, collapsible AI copilot inside the editor.
 * Sends prompt-based edits via edit-ai mode=patch, reports what changed,
 * and lets the user apply the validated patch to the stored project.
 *
 * Condition 4: applying routes through the editor's serialized edit queue
 * (props.applyEdit) so copilot edits are undoable and never race panel
 * edits. Falls back to a direct save only when no queue is provided.
 */
export default function CopilotPanel({ jobId, clipIndex, applyEdit }) {
  const [open, setOpen] = useState(true);
  const [prompt, setPrompt] = useState('');
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState(null); // {ok, explanation, patch, project, error}
  const [applying, setApplying] = useState(false);
  const [applied, setApplied] = useState(false);

  const run = async () => {
    const instruction = prompt.trim();
    if (!instruction || busy) return;
    setBusy(true); setResult(null); setApplied(false);
    try {
      const r = await editAiPatch(jobId, clipIndex, instruction);
      setResult({
        ok: true,
        explanation: r?.explanation || 'Patch ready.',
        patch: r?.patch || null,
        project: r?.project || null,
      });
    } catch (e) {
      setResult({ ok: false, error: e?.data?.detail || e.message });
    } finally {
      setBusy(false);
    }
  };

  const applyPatch = async () => {
    if (!result?.project || applying) return;
    setApplying(true);
    try {
      const patched = result.project;
      const label = `copilot: ${(prompt.trim() || 'edit').slice(0, 60)}`;
      if (applyEdit) {
        // Through the serialized queue: snapshot + undo + optimistic
        // concurrency (expected_version = the version the patch was
        // computed against).
        await applyEdit(label, () => patched, {
          expectedVersion: patched.version,
        });
      } else {
        await saveClipProject(jobId, clipIndex, patched);
      }
      setApplied(true);
    } catch (e) {
      setResult({ ok: false, error: `Apply failed: ${e.message}` });
    } finally {
      setApplying(false);
    }
  };

  const patchSummary = (patch) => {
    if (!patch || typeof patch !== 'object') return 'project patch';
    const keys = Object.keys(patch).filter((k) => k !== 'schema');
    if (!keys.length) return 'project patch';
    return keys.map((k) => {
      const v = patch[k];
      const short = typeof v === 'string' && v.length > 42 ? `${v.slice(0, 42)}…` : JSON.stringify(v)?.slice(0, 42);
      return `${k} → ${short}`;
    }).join(' · ');
  };

  if (!open) {
    return (
      <button
        className="nc-anim-pop-in"
        onClick={() => setOpen(true)}
        aria-label="Open AI copilot"
        style={{
          position: 'absolute', right: 18, bottom: 18, zIndex: 40,
          width: 52, height: 52, borderRadius: '50%', border: 'none', cursor: 'pointer',
          background: 'linear-gradient(140deg, var(--nc-accent), #6cb8f5)',
          color: '#fff', display: 'flex', alignItems: 'center', justifyContent: 'center',
          boxShadow: '0 10px 32px rgba(10,129,217,0.45)',
          transition: 'transform 0.15s ease',
        }}
        onMouseEnter={(e) => { e.currentTarget.style.transform = 'scale(1.07)'; }}
        onMouseLeave={(e) => { e.currentTarget.style.transform = 'scale(1)'; }}
      >
        <Bot size={22} />
      </button>
    );
  }

  return (
    <div
      className="nc-anim-pop-in"
      style={{
        position: 'absolute', right: 18, bottom: 18, zIndex: 40,
        width: 340, maxWidth: 'calc(100% - 36px)',
        background: 'rgba(20,20,23,0.96)', backdropFilter: 'blur(12px)',
        border: '1px solid var(--nc-border-strong)', borderRadius: 16,
        boxShadow: '0 18px 60px rgba(0,0,0,0.55)',
        overflow: 'hidden',
      }}
    >
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, padding: '10px 12px', borderBottom: '1px solid var(--nc-border)', cursor: 'pointer' }} onClick={() => setOpen(false)}>
        <span style={{
          width: 26, height: 26, borderRadius: 8, display: 'inline-flex', alignItems: 'center', justifyContent: 'center',
          background: 'var(--nc-accent-soft)', color: 'var(--nc-accent)',
        }}>
          <Sparkles size={14} />
        </span>
        <span style={{ flex: 1, fontSize: 13.5, fontWeight: 700 }}>AI copilot</span>
        <ChevronDown size={15} style={{ color: 'var(--nc-text-faint)' }} />
      </div>

      <div style={{ padding: 12, display: 'flex', flexDirection: 'column', gap: 10 }}>
        <div style={{ display: 'flex', gap: 8 }}>
          <input
            className="nc-input"
            style={{ flex: 1, fontSize: 13 }}
            placeholder='e.g. "make the hook punchier"'
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            onKeyDown={(e) => { if (e.key === 'Enter') run(); }}
            disabled={busy}
          />
          <button className="nc-btn-primary" style={{ padding: '8px 14px', fontSize: 13 }} disabled={busy || !prompt.trim()} onClick={run}>
            {busy ? <Loader2 size={15} style={{ animation: 'nc-spin 1s linear infinite' }} /> : 'Edit'}
          </button>
        </div>

        {result?.ok && (
          <div className="nc-anim-fade-in" style={{ fontSize: 12.5, lineHeight: 1.55 }}>
            <div style={{ display: 'flex', gap: 6, alignItems: 'flex-start', color: 'var(--nc-text)' }}>
              <Check size={14} style={{ color: 'var(--nc-green)', flexShrink: 0, marginTop: 2 }} />
              <span>{result.explanation}</span>
            </div>
            {result.patch && (
              <div style={{
                marginTop: 8, padding: '8px 10px', borderRadius: 10,
                background: 'rgba(255,255,255,0.04)', border: '1px solid var(--nc-border)',
                color: 'var(--nc-text-dim)', fontSize: 12,
              }}>
                Changed: {patchSummary(result.patch)}
              </div>
            )}
            {!applied ? (
              <button className="nc-btn-primary" style={{ width: '100%', marginTop: 10, fontSize: 13 }} disabled={applying} onClick={applyPatch}>
                {applying ? <Loader2 size={14} style={{ animation: 'nc-spin 1s linear infinite' }} /> : <Check size={14} />}
                {applying ? 'Applying…' : 'Apply to project'}
              </button>
            ) : (
              <div style={{ marginTop: 10, fontSize: 12.5, color: 'var(--nc-green)', display: 'flex', alignItems: 'center', gap: 6 }}>
                <Check size={14} /> Applied — re-export from the editor to render it.
              </div>
            )}
          </div>
        )}

        {result && !result.ok && (
          <div className="nc-anim-fade-in" style={{ fontSize: 12.5, color: 'var(--nc-red)', lineHeight: 1.5 }}>
            {result.error}
          </div>
        )}

        <div style={{ fontSize: 11, color: 'var(--nc-text-faint)', lineHeight: 1.5 }}>
          Try: hook text, caption style, word fixes, crop nudges, color grades.
        </div>
      </div>
    </div>
  );
}
