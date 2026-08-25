// ClippyMe redesign — the seven EditClipModal tab bodies as dumb components.
// State stays lifted in EditClipModal (tabs are conditionally rendered, so a
// tab owning state would lose it on every switch; apply() needs it all);
// these render it and report edits up. Markup moved verbatim from the modal.
import { useState } from 'react';
import { Icon, Btn, Segmented, Switch } from './primitives';
import { HookStyleControls, HookPreview } from './hookStyle';
import { SubtitleControls } from './subtitleControls';
import { LogoControls, GradeControls } from './layerControls';
import { BannerControls } from './bannerControls';
import { dubClip } from './realApi';

export const REFRAME_OPTS = [
  { id: 'auto', label: 'Auto' },
  { id: 'subject', label: 'Subject' },
  { id: 'split', label: 'Split' },
  { id: 'screencast', label: 'Screen' },
  { id: 'disabled', label: 'Off' },
];

export function ReframeTab({ mode, onChange }) {
  return (
    <div className="field" style={{ marginTop: 4 }}>
      <span className="field-label">Reframe Layout</span>
      <Segmented full value={mode} onChange={onChange} options={REFRAME_OPTS} />
      <div className="eo-d" style={{ marginTop: 6 }}>Auto face-track · Subject FrameShift · Split 2-speaker · Screen slides · Off letterbox</div>
    </div>
  );
}

export function SmartCutTab({ on, onChange, bulk }) {
  return (
    <>
      <div className="edit-opt">
        <div className="eo-ico"><Icon n="scissors" /></div>
        <div className="eo-txt"><div className="eo-t">Smart Cut</div><div className="eo-d">Auto-remove silence &amp; filler words</div></div>
        <Switch on={on} onChange={onChange} />
      </div>
      <div className="eo-d" style={{ marginTop: 8 }}>
        Detects and trims dead air + fillers automatically. To cut specific
        sentences or words, use the {bulk ? 'Trim section on a single clip' : <b>Trim</b>} tab.
      </div>
    </>
  );
}

// `trim` is the useManualTrim bundle: { segments, segErr, dropped, toggleDrop,
// dropRanges, hasDrops, ai: { text, setText, busy, msg, ask } }.
export function TrimTab({ trim }) {
  const { segments, segErr, dropped, toggleDrop, dropRanges, hasDrops, ai } = trim;
  return (
    <div className="cf-row" style={{ marginBottom: 0 }}>
      <span className="field-label" style={{ marginBottom: 9, display: 'flex', justifyContent: 'space-between' }}>
        <span>Manual trim</span>
        {hasDrops && <span className="eo-d">{dropRanges.length} dropped</span>}
      </span>
      <div className="eo-d" style={{ marginBottom: 8 }}>
        Tap any line to cut it, or describe the edit below. Trimming also runs Smart Cut&apos;s auto silence pass.
      </div>
      <div style={{ display: 'flex', gap: 6, marginBottom: 6 }}>
        <input className="input-field" style={{ flex: 1 }}
          placeholder="e.g. cut the intro and the part where he stumbles"
          value={ai.text} onChange={(e) => ai.setText(e.target.value)}
          onKeyDown={(e) => { if (e.key === 'Enter') ai.ask(); }}
          disabled={ai.busy || !segments || segments.length === 0} />
        <Btn onClick={ai.ask} disabled={ai.busy || !ai.text.trim() || !segments || segments.length === 0}>
          <Icon n={ai.busy ? 'loader' : 'wand-sparkles'} style={{ width: 14, height: 14 }} />
          {ai.busy ? 'Thinking…' : 'AI trim'}
        </Btn>
      </div>
      {ai.msg && <div className="eo-d" style={{ marginBottom: 8 }}>{ai.msg}</div>}
      {segments === null && !segErr && <div className="eo-d">Loading transcript…</div>}
      {segErr && <div className="eo-d">Transcript unavailable — auto Smart Cut still applies.</div>}
      {segments && segments.length === 0 && <div className="eo-d">No transcript segments for this clip.</div>}
      {segments && segments.length > 0 && (
        <div className="trim-list">
          {segments.map((s) => {
            const off = dropped.has(s.index);
            return (
              <button key={s.index} type="button"
                className={'trim-seg' + (off ? ' cut' : '')}
                onClick={() => toggleDrop(s.index)}
                title={off ? 'Will be cut — tap to keep' : 'Kept — tap to cut'}>
                <Icon n={off ? 'scissors' : 'check'} style={{ width: 13, height: 13, flexShrink: 0 }} />
                <span className="trim-txt" title={s.text}>
                  {s.text && s.text.length > 140 ? s.text.slice(0, 140) + '…' : s.text}
                </span>
                <span className="trim-time">{s.start.toFixed(1)}s</span>
              </button>
            );
          })}
        </div>
      )}
    </div>
  );
}

export function CaptionsTab({ on, onToggle, subs, onSubsChange }) {
  return (
    <>
      <div className="edit-opt">
        <div className="eo-ico"><Icon n="captions" /></div>
        <div className="eo-txt"><div className="eo-t">Subtitles</div><div className="eo-d">Burn karaoke or classic captions</div></div>
        <Switch on={on} onChange={onToggle} />
      </div>
      {on && <SubtitleControls variant="edit" value={subs} onChange={onSubsChange} />}
    </>
  );
}

export function HookTab({ on, onToggle, bulk, text, onText, style, onStyle }) {
  return (
    <>
      <div className="edit-opt">
        <div className="eo-ico"><Icon n="type" /></div>
        <div className="eo-txt"><div className="eo-t">Text hook</div><div className="eo-d">A scroll-stopping opener overlaid on the clip</div></div>
        <Switch on={on} onChange={onToggle} />
      </div>
      {on && (
        <div className="cfg-drawer fade-in">
          {bulk ? (
            <div className="eo-d" style={{ marginBottom: 10 }}>
              Applying the hook <b>style</b> to all selected clips. Each clip keeps its own hook text.
            </div>
          ) : (
            <div className="cf-row">
              <span className="field-label" style={{ marginBottom: 9, display: 'flex' }}>Hook text</span>
              <textarea className="ta" rows="2" value={text} placeholder="e.g. THIS changed everything"
                onChange={(e) => onText(e.target.value)}></textarea>
            </div>
          )}
          <div style={{ marginTop: bulk ? 0 : 10 }}><HookPreview text={bulk ? 'Your hook text' : text} style={style} /></div>
          <HookStyleControls style={style} set={onStyle} />
        </div>
      )}
    </>
  );
}

export function LogoTab({ on, onToggle, logo, onChange }) {
  return (
    <>
      <div className="edit-opt">
        <div className="eo-ico"><Icon n="stamp" /></div>
        <div className="eo-txt"><div className="eo-t">Brand logo</div><div className="eo-d">Burn your uploaded logo onto the clip</div></div>
        <Switch on={on} onChange={onToggle} />
      </div>
      {on && (
        <div className="cfg-drawer fade-in">
          <LogoControls position={logo.position} size={logo.size} onChange={onChange} />
        </div>
      )}
    </>
  );
}

export function BannerTab({ on, onToggle, banner, onChange }) {
  return (
    <>
      <div className="edit-opt">
        <div className="eo-ico"><Icon n="rss" /></div>
        <div className="eo-txt"><div className="eo-t">Attribution banner</div><div className="eo-d">Platform logo + handle burned bottom of clip</div></div>
        <Switch on={on} onChange={onToggle} />
      </div>
      {on && (
        <div className="cfg-drawer fade-in">
          <BannerControls value={banner} onChange={onChange} />
        </div>
      )}
    </>
  );
}

// Owns the on/off semantics: the Switch maps off ⇄ 'none' and on ⇄ the
// default look; the Segmented refines the preset while on.
export function GradeTab({ preset, onChange }) {
  const on = !!preset && preset !== 'none';
  return (
    <>
      <div className="edit-opt">
        <div className="eo-ico"><Icon n="palette" /></div>
        <div className="eo-txt"><div className="eo-t">Colour grade</div><div className="eo-d">Cinematic colour pass burned before overlays</div></div>
        <Switch on={on} onChange={(v) => onChange(v ? 'warm_cinematic' : 'none')} />
      </div>
      {on && (
        <div className="cfg-drawer fade-in">
          <div className="cf-row" style={{ marginBottom: 0 }}>
            <span className="field-label" style={{ marginBottom: 9, display: 'flex' }}>Look</span>
            <GradeControls preset={preset} onChange={(p) => onChange(p.preset)} />
          </div>
        </div>
      )}
    </>
  );
}

export function DubbingTab({ jobId, clipIndex, onDubSuccess }) {
  const [lang, setLang] = useState('es');
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState('');
  const [dubbedUrl, setDubbedUrl] = useState(null);

  const handleDub = async () => {
    setBusy(true);
    setMsg('Dubbing clip in background via ElevenLabs...');
    try {
      const res = await dubClip(jobId, clipIndex, lang);
      setDubbedUrl(res.dubbed_url);
      setMsg(`✅ Dubbing complete! Dubbed video created: ${res.dubbed_filename}`);
      if (onDubSuccess) onDubSuccess(res);
    } catch (err) {
      setMsg(`❌ Dubbing error: ${err.message}`);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="cf-row" style={{ marginBottom: 0 }}>
      <span className="field-label" style={{ marginBottom: 9, display: 'flex' }}>AI Voice Dubbing (ElevenLabs)</span>
      <div className="eo-d" style={{ marginBottom: 12 }}>
        Translate and re-voice this clip in 30+ languages while preserving voice characteristics.
      </div>
      <div style={{ display: 'flex', gap: 8, alignItems: 'center', marginBottom: 12 }}>
        <select
          className="input-field"
          style={{ flex: 1 }}
          value={lang}
          onChange={(e) => setLang(e.target.value)}
          disabled={busy}
        >
          <option value="es">Spanish (Español)</option>
          <option value="fr">French (Français)</option>
          <option value="de">German (Deutsch)</option>
          <option value="it">Italian (Italiano)</option>
          <option value="pt">Portuguese (Português)</option>
          <option value="hi">Hindi (हिन्दी)</option>
          <option value="ja">Japanese (日本語)</option>
          <option value="ko">Korean (한국어)</option>
          <option value="zh">Chinese (中文)</option>
          <option value="ar">Arabic (العربية)</option>
          <option value="ru">Russian (Русский)</option>
        </select>
        <Btn primary disabled={busy} onClick={handleDub}>
          {busy ? 'Dubbing…' : 'Start Dubbing'}
        </Btn>
      </div>
      {msg && <div className="eo-d" style={{ color: msg.startsWith('❌') ? '#ef4444' : '#10b981', marginTop: 8 }}>{msg}</div>}
      {dubbedUrl && (
        <div style={{ marginTop: 12 }}>
          <a href={dubbedUrl} download className="btn sm" target="_blank" rel="noreferrer">
            Download Dubbed Video
          </a>
        </div>
      )}
    </div>
  );
}
