// ClippyMe redesign — HistoryView + SettingsView + ApiKeyModal, wired to the
// real backend (history list/restore/delete; config keys, cookies, Zernio).
import { useState, useEffect, useRef } from 'react';
import { useModalA11y } from './useModalA11y';
import { Icon, Btn, Badge, Switch, Segmented, Panel } from './primitives';
import { Hero } from './chrome';
import {
  getConfig, saveConfig, getModels, cookiesStatus, uploadCookies, deleteCookies,
  getZernio, saveZernio, discoverZernioAccounts,
  getWatchdog, saveWatchdog, testWatchdogAlert,
  listFonts, uploadFont, deleteFont, logoStatus, uploadLogo, deleteLogo,
  researchProduct, generateUgcScripts, fetchViralTitles, refineViralTitles, fetchChapters, generateThumbnail,
} from './realApi';
import { SUB_FONTS } from './data';
import { getApiToken, setApiToken } from '../lib/apiToken';
import { relTime } from '../lib/relTime';
import { triggerStorageCleanup } from '../lib/api';

// Curated fallback when live discovery is unavailable (no key yet / offline).
// Mirrors the allow-list prefixes (gemini-2.5- / gemini-3) the backend accepts.
const FALLBACK_MODELS = [
  { name: 'gemini-3.5-flash', display_name: 'Gemini 3.5 Flash — recommended' },
  { name: 'gemini-2.5-flash', display_name: 'Gemini 2.5 Flash — budget' },
  { name: 'gemini-3.1-pro-preview', display_name: 'Gemini 3.1 Pro — max quality' },
  { name: 'gemini-2.5-pro', display_name: 'Gemini 2.5 Pro — max quality' },
];

export function HistoryView({ history, availableIds, onOpen, onDelete, onClear }) {
  if (!history.length) {
    return (
      <div className="container narrow fade-in">
        <Hero eyebrow="History" line1="Nothing here yet." sub="Every job you run lands here, ready to reopen, re-export, or publish again." />
        <div className="empty">
          <div className="ei"><Icon n="clock" /></div>
          <h3>No jobs yet</h3>
          <p>Head to Create, paste a link, and your finished clips will land here.</p>
        </div>
      </div>
    );
  }
  return (
    <div className="container narrow fade-in">
      <div className="results-head" style={{ marginBottom: 18 }}>
        <h2>History</h2>
        <Badge tone="out">{history.length} jobs</Badge>
        <div className="rh-right">
          <Btn variant="ghost" size="sm" icon="trash-2" onClick={onClear}>Clear all</Btn>
        </div>
      </div>
      <Panel pad={false} className="hlist">
        {history.map((h) => {
          // `availableIds` is the set of jobs whose files still exist on disk
          // (null = backend not reached yet → assume available, don't disable).
          // An entry whose files were wiped by a rebuild is shown muted + flagged
          // "files removed" instead of looking clickable and dead-ending.
          const onDisk = !availableIds || availableIds.has(h.jobId);
          const ok = h.status === 'complete' && onDisk;
          const removed = !!availableIds && !availableIds.has(h.jobId);
          return (
            <div className="hrow" key={h.jobId}
              role={ok ? 'button' : undefined} tabIndex={ok ? 0 : undefined}
              aria-label={ok ? `Open job ${h.title || h.source || h.jobId}` : undefined}
              onClick={() => ok && onOpen(h)}
              onKeyDown={(e) => { if (ok && (e.key === 'Enter' || e.key === ' ')) { e.preventDefault(); onOpen(h); } }}
              style={{ cursor: ok ? 'pointer' : 'default', opacity: removed ? 0.55 : 1 }}>
              <div className="hthumb" style={{ background: removed ? 'var(--bg-4)' : 'var(--grad-viral)' }}>{h.clipCount ?? 0}</div>
              <div style={{ minWidth: 0 }}>
                <div className="ht" title={h.title || h.source || h.jobId}
                  style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{h.title || h.source || h.jobId}</div>
                <div className="hm">
                  <Icon n={h.sourceType === 'url' ? 'globe' : 'file-video'} style={{ width: 11, height: 11, verticalAlign: '-1px', marginRight: 5 }} />
                  {removed ? 'Files removed (rebuild/cleanup) · delete to dismiss'
                    : `${h.clipCount || 0} clips${h.cost != null ? ` · $${Number(h.cost).toFixed(2)}` : ''} · ${relTime(h.timestamp)}`}
                </div>
              </div>
              <div className="hr">
                {!removed && h.publishedCount > 0 && (
                  <Badge tone="teal" icon="send">{h.publishedCount} published</Badge>
                )}
                {removed ? <Badge tone="out" icon="triangle-alert">unavailable</Badge>
                  : h.status === 'complete' ? <Badge tone="teal" icon="check">complete</Badge>
                    : h.status === 'error' ? <Badge tone="danger" icon="triangle-alert">error</Badge>
                      : <Badge tone="amber" icon="clock">{h.status || 'pending'}</Badge>}
                <button type="button" className="mini" title="Delete" aria-label="Delete job" onClick={(e) => { e.stopPropagation(); onDelete(h.jobId); }}><Icon n="trash-2" /></button>
                {ok && <Icon n="chevron-right" style={{ width: 18, height: 18, color: 'var(--fg-4)' }} />}
              </div>
            </div>
          );
        })}
      </Panel>
    </div>
  );
}

function KeyRow({ icon, name, desc, value, onChange, onSave, onClear, placeholder, present }) {
  const [reveal, setReveal] = useState(false);
  // Only persist (and toast) when the field actually changed during this focus
  // session — tabbing past an already-set key shouldn't spam saves/toasts.
  const focusVal = useRef(value);
  return (
    <div className="keyrow">
      <div className="ki"><Icon n={icon} /></div>
      <div style={{ minWidth: 0 }}>
        <div className="kt">{name}</div>
        <div className="kd">{desc}</div>
      </div>
      <div className="kr">
        <input className="key-input" type={reveal ? 'text' : 'password'} value={value}
          aria-label={name}
          placeholder={placeholder} onChange={(e) => onChange(e.target.value)}
          onFocus={() => { focusVal.current = value; }}
          onBlur={() => { if (value !== focusVal.current) onSave(); }} />
        <button type="button" className="mini" title={reveal ? 'Hide' : 'Show'} aria-label={reveal ? 'Hide key' : 'Show key'} onClick={() => setReveal(!reveal)}><Icon n={reveal ? 'eye-off' : 'eye'} /></button>
        {/* Backend-confirmed state only — raw input text (typed but not yet
            saved) must never flip this badge. */}
        {present ? <Badge tone="teal" icon="check">set</Badge> : <Badge tone="out">empty</Badge>}
        {present && onClear && (
          <button type="button" className="mini" title="Clear saved key" aria-label={`Clear ${name} key`} onClick={onClear}><Icon n="x" /></button>
        )}
      </div>
    </div>
  );
}

export function SettingsView({ apiKey, onApiKey, cookiesConfigured, onCookiesChange, pushToast }) {
  const [gemini, setGemini] = useState(apiKey || '');
  const [deepgram, setDeepgram] = useState('');
  const [elevenlabs, setElevenlabs] = useState('');
  const [hf, setHf] = useState('');
  const [twitchId, setTwitchId] = useState('');
  const [twitchSecret, setTwitchSecret] = useState('');
  const [apiToken, setApiTokenState] = useState(() => getApiToken());
  const [present, setPresent] = useState({});
  const [zernio, setZernioState] = useState(null);
  const [zKey, setZKey] = useState('');
  const [accts, setAccts] = useState({ tiktok: '', instagram: '', youtube: '' });
  const [cookies, setCookies] = useState(!!cookiesConfigured);
  const [logoOn, setLogoOn] = useState(false);
  const [fonts, setFonts] = useState([]);
  const [provider, setProvider] = useState('whisper');
  const [model, setModel] = useState('');
  const [models, setModels] = useState(FALLBACK_MODELS);
  const [loadingModels, setLoadingModels] = useState(false);

  const [watchdog, setWatchdogState] = useState(null);
  const [wdEnabled, setWdEnabled] = useState(true);
  const [wdAi, setWdAi] = useState(true);
  const [wdProvider, setWdProvider] = useState('ntfy');
  const [wdTopic, setWdTopic] = useState('');
  const [wdTgToken, setWdTgToken] = useState('');
  const [wdTgChat, setWdTgChat] = useState('');
  const [wdDiscUrl, setWdDiscUrl] = useState('');
  const [wdWhUrl, setWdWhUrl] = useState('');
  const [testingWd, setTestingWd] = useState(false);
  const [cleaningStorage, setCleaningStorage] = useState(false);

  // Pull the live model list from the backend (uses the saved key if the
  // header is empty). Merges discovery with the curated fallback + the
  // currently-selected model so the dropdown is never empty and never drops
  // the active choice.
  const loadModels = async (key) => {
    setLoadingModels(true);
    try {
      const { models: live } = await getModels(key || gemini || apiKey || '');
      const seen = new Set();
      const merged = [];
      [...(live || []), ...FALLBACK_MODELS].forEach((m) => {
        if (m?.name && !seen.has(m.name)) { seen.add(m.name); merged.push(m); }
      });
      if (merged.length) setModels(merged);
    } catch { /* keep fallback */ }
    finally { setLoadingModels(false); }
  };

  // Source of truth for "is this key set" is always the backend's response,
  // never the (optimistic) input text — refetched after every save/clear so
  // the badge can't drift from what's actually persisted.
  const refreshConfig = async () => {
    const c = await getConfig();
    if (!c) { pushToast?.('warn', 'Could not refresh key status'); return; }
    setPresent({
      gemini: !!c.GEMINI_API_KEY, hf: !!c.HF_TOKEN, deepgram: !!c.DEEPGRAM_API_KEY, elevenlabs: !!c.ELEVENLABS_API_KEY,
      twitchId: !!c.TWITCH_CLIENT_ID, twitchSecret: !!c.TWITCH_CLIENT_SECRET,
    });
    if (c.TRANSCRIPTION_PROVIDER) setProvider(c.TRANSCRIPTION_PROVIDER);
    if (c.GEMINI_MODEL) setModel(c.GEMINI_MODEL);
  };

  useEffect(() => {
    refreshConfig().then(loadModels);
    getZernio().then((z) => { setZernioState(z); if (z.accounts) setAccts({ tiktok: '', instagram: '', youtube: '', facebook: '', ...z.accounts }); }).catch(() => {});
    getWatchdog().then((w) => {
      if (w) {
        setWatchdogState(w);
        setWdEnabled(w.enabled !== false);
        setWdAi(w.ai_diagnosis !== false);
        if (w.provider) setWdProvider(w.provider);
        if (w.ntfy_topic) setWdTopic(w.ntfy_topic);
        if (w.telegram_chat_id) setWdTgChat(w.telegram_chat_id);
      }
    }).catch(() => {});
    cookiesStatus().then((s) => setCookies(!!s.configured)).catch(() => {});
    logoStatus().then((s) => setLogoOn(!!s.configured)).catch(() => {});
    listFonts().then(({ fonts: f }) => setFonts(Array.isArray(f) ? f : [])).catch(() => {});
    // Mount-once bootstrap; loadModels reads the latest key via closure on call.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const saveWatchdogCfg = async (overrides = {}) => {
    try {
      const payload = {
        enabled: overrides.enabled ?? wdEnabled,
        ai_diagnosis: overrides.ai_diagnosis ?? wdAi,
        provider: wdProvider,
        ntfy_topic: wdTopic.trim(),
        telegram_chat_id: wdTgChat.trim(),
      };
      if (wdTgToken.trim()) payload.telegram_bot_token = wdTgToken.trim();
      if (wdDiscUrl.trim()) payload.discord_webhook_url = wdDiscUrl.trim();
      if (wdWhUrl.trim()) payload.webhook_url = wdWhUrl.trim();
      const updated = await saveWatchdog(payload);
      setWatchdogState(updated);
      setWdTgToken('');
      setWdDiscUrl('');
      setWdWhUrl('');
      pushToast?.('success', 'Watchdog settings saved');
    } catch (err) {
      pushToast?.('error', err.message || 'Failed to save Watchdog settings');
    }
  };

  const sendTestAlert = async () => {
    setTestingWd(true);
    try {
      const payload = {
        enabled: wdEnabled,
        ai_diagnosis: wdAi,
        provider: wdProvider,
        ntfy_topic: wdTopic.trim(),
        telegram_chat_id: wdTgChat.trim(),
      };
      if (wdTgToken.trim()) payload.telegram_bot_token = wdTgToken.trim();
      if (wdDiscUrl.trim()) payload.discord_webhook_url = wdDiscUrl.trim();
      if (wdWhUrl.trim()) payload.webhook_url = wdWhUrl.trim();
      await testWatchdogAlert(payload);
      pushToast?.('success', 'Test alert sent! Check your phone.');
    } catch (err) {
      pushToast?.('error', err.message || 'Test alert failed. Check settings.');
    } finally {
      setTestingWd(false);
    }
  };

  const saveKeys = async (patch) => {
    try { await saveConfig(patch); pushToast?.('success', 'Saved'); await refreshConfig(); }
    catch { pushToast?.('error', 'Save failed'); }
  };

  const saveZernioCfg = async () => {
    try {
      const payload = { accounts: accts };
      if (zKey.trim()) payload.api_key = zKey.trim();
      const z = await saveZernio(payload);
      setZernioState(z); setZKey('');
      pushToast?.('success', 'Zernio saved');
    } catch { pushToast?.('error', 'Zernio save failed'); }
  };

  const discover = async () => {
    try {
      // Discovery runs against the *saved* key, so persist a freshly-typed one
      // first — otherwise the backend 400s with "API key not configured".
      if (zKey.trim()) { await saveZernio({ api_key: zKey.trim(), accounts: accts }); setZKey(''); }
      const { accounts } = await discoverZernioAccounts();
      const next = { ...accts };
      (accounts || []).forEach((a) => {
        const p = (a.platform || '').toLowerCase();
        const id = a._id || a.id;
        if (p.includes('tiktok')) next.tiktok = id;
        else if (p.includes('insta')) next.instagram = id;
        else if (p.includes('you')) next.youtube = id;
        else if (p.includes('face') || p.includes('fb')) next.facebook = id;
      });
      setAccts(next);
      pushToast?.('success', `Discovered ${(accounts || []).length} accounts`);
    } catch { pushToast?.('error', 'Discover failed. Check the API key.'); }
  };

  const onCookieFile = async (e) => {
    const f = e.target.files?.[0];
    if (!f) return;
    try { await uploadCookies(f); setCookies(true); onCookiesChange?.(true); pushToast?.('success', 'Cookies uploaded'); }
    catch { pushToast?.('error', 'Cookie upload failed'); }
  };
  const removeCookies = async () => {
    try { await deleteCookies(); setCookies(false); onCookiesChange?.(false); pushToast?.('info', 'Cookies removed'); }
    catch { pushToast?.('error', 'Remove failed'); }
  };

  const onLogoFile = async (e) => {
    const f = e.target.files?.[0];
    if (!f) return;
    try { await uploadLogo(f); setLogoOn(true); pushToast?.('success', 'Logo uploaded'); }
    catch (err) { pushToast?.('error', String(err.message || 'Logo upload failed').slice(0, 80)); }
  };
  const removeLogo = async () => {
    try { await deleteLogo(); setLogoOn(false); pushToast?.('info', 'Logo removed'); }
    catch { pushToast?.('error', 'Remove failed'); }
  };

  // Only user-uploaded faces are deletable; bundled ones are part of the app.
  const bundled = new Set(SUB_FONTS.map(([v]) => v));
  const onFontFile = async (e) => {
    const f = e.target.files?.[0];
    if (!f) return;
    try { const { fonts: nf } = await uploadFont(f); setFonts(nf || fonts); pushToast?.('success', 'Font added'); }
    catch (err) { pushToast?.('error', String(err.message || 'Font upload failed').slice(0, 80)); }
  };
  const removeFont = async (name) => {
    try { const { fonts: nf } = await deleteFont(name); setFonts(nf || fonts.filter((n) => n !== name)); pushToast?.('info', 'Font removed'); }
    catch { pushToast?.('error', 'Remove failed'); }
  };

  return (
    <div className="container narrow fade-in">
      <Hero eyebrow="Settings" line1="Keys & connections." sub="Everything is stored locally. Your keys never leave your machine." />

      <Panel title="API keys" sub="Required for transcription & moment detection" icon="key-round" style={{ marginBottom: 18 }}>
        <KeyRow icon="sparkles" name="Gemini" desc="Viral-moment detection" value={gemini} present={present.gemini}
          onChange={(v) => { setGemini(v); onApiKey?.(v); }} onSave={() => saveKeys({ GEMINI_API_KEY: gemini })}
          onClear={() => { setGemini(''); onApiKey?.(''); saveKeys({ GEMINI_API_KEY: '' }); }} placeholder="AIza…" />
        <KeyRow icon="audio-lines" name="Deepgram" desc="Nova-3 transcription" value={deepgram} present={present.deepgram}
          onChange={setDeepgram} onSave={() => saveKeys({ DEEPGRAM_API_KEY: deepgram })}
          onClear={() => { setDeepgram(''); saveKeys({ DEEPGRAM_API_KEY: '' }); }} placeholder="dg_…" />
        <KeyRow icon="audio-lines" name="ElevenLabs" desc="Scribe transcription · audio-event tags · AI Dubbing" value={elevenlabs} present={present.elevenlabs}
          onChange={setElevenlabs} onSave={() => saveKeys({ ELEVENLABS_API_KEY: elevenlabs })}
          onClear={() => { setElevenlabs(''); saveKeys({ ELEVENLABS_API_KEY: '' }); }} placeholder="sk_…" />
        <KeyRow icon="scan-face" name="Hugging Face token" desc="Speaker diarization models" value={hf} present={present.hf}
          onChange={setHf} onSave={() => saveKeys({ HF_TOKEN: hf })}
          onClear={() => { setHf(''); saveKeys({ HF_TOKEN: '' }); }} placeholder="hf_…" />
        <KeyRow icon="rss" name="Twitch client ID" desc="Live Monitor: Twitch channel detection" value={twitchId} present={present.twitchId}
          onChange={setTwitchId} onSave={() => saveKeys({ TWITCH_CLIENT_ID: twitchId })}
          onClear={() => { setTwitchId(''); saveKeys({ TWITCH_CLIENT_ID: '' }); }} placeholder="Helix app client id" />
        <KeyRow icon="rss" name="Twitch client secret" desc="Live Monitor: Twitch channel detection" value={twitchSecret} present={present.twitchSecret}
          onChange={setTwitchSecret} onSave={() => saveKeys({ TWITCH_CLIENT_SECRET: twitchSecret })}
          onClear={() => { setTwitchSecret(''); saveKeys({ TWITCH_CLIENT_SECRET: '' }); }} placeholder="Helix app client secret" />
        <KeyRow icon="key-round" name="API token" desc="Only for LAN deploys with CLIPPYME_API_TOKEN set — stored in this browser, sent as X-API-Token" value={apiToken} present={!!getApiToken()}
          onChange={setApiTokenState} onSave={() => { setApiToken(apiToken); pushToast?.('success', apiToken.trim() ? 'API token saved' : 'API token cleared'); }} placeholder="Shared secret (leave empty + Save to clear)" />
        <div className="opt" style={{ borderBottom: 0 }}>
          <div className="oico"><Icon n="audio-lines" /></div>
          <div className="otxt"><div className="ot">Transcription engine</div><div className="od">Cloud STT falls back to local Whisper if its key is missing</div></div>
          <div className="r"><Segmented value={provider}
            onChange={(id) => { setProvider(id); saveKeys({ TRANSCRIPTION_PROVIDER: id }); }}
            options={[{ id: 'deepgram', label: 'Deepgram' }, { id: 'elevenlabs', label: 'ElevenLabs' }, { id: 'whisper', label: 'Whisper' }]} /></div>
        </div>
        {provider === 'deepgram' && !present.deepgram && (
          <div className="od" style={{ color: 'var(--warn, #f5a623)', padding: '0 0 8px 44px' }}>⚠ No Deepgram key saved — pipeline will use local Whisper.</div>
        )}
        {provider === 'elevenlabs' && !present.elevenlabs && (
          <div className="od" style={{ color: 'var(--warn, #f5a623)', padding: '0 0 8px 44px' }}>⚠ No ElevenLabs key saved — pipeline will use local Whisper.</div>
        )}
        <div className="opt" style={{ borderBottom: 0 }}>
          <div className="oico"><Icon n="sparkles" /></div>
          <div className="otxt"><div className="ot">Gemini model</div><div className="od">Viral-moment detection model · applied to new jobs</div></div>
          <div className="r" style={{ gap: 8 }}>
            <select className="key-input" style={{ width: 'auto', minWidth: 200, fontFamily: 'var(--font-sans)' }}
              value={model}
              onChange={(e) => { setModel(e.target.value); saveKeys({ GEMINI_MODEL: e.target.value }); }}>
              {!model && <option value="">Default (gemini-3.5-flash)</option>}
              {model && !models.some((m) => m.name === model) && <option value={model}>{model}</option>}
              {models.map((m) => <option key={m.name} value={m.name}>{m.display_name || m.name}</option>)}
            </select>
            <Btn variant="ghost" size="sm" icon="refresh-cw" onClick={() => loadModels()} disabled={loadingModels}>
              {loadingModels ? '…' : 'Refresh'}
            </Btn>
          </div>
        </div>
      </Panel>

      <Panel title="Publishing" sub="Push finished clips to socials via Zernio" icon="send" style={{ marginBottom: 18 }}>
        <div className="zernio-card" style={{ flexDirection: 'column', alignItems: 'stretch', gap: 14 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
            <div className="zico"><Icon n="rss" /></div>
            <div style={{ flex: 1, minWidth: 0 }}>
              <div className="kt">Zernio</div>
              <div className="kd">{zernio?.configured ? `Connected${zernio.api_key_masked ? ' · ' + zernio.api_key_masked : ''}` : 'Add your API key + account IDs to schedule posts'}</div>
            </div>
            {zernio?.configured && <span className="conn"><Icon n="circle-check" />Connected</span>}
          </div>
          <input className="key-input" style={{ width: '100%' }} type="password" value={zKey}
            aria-label="Zernio API key"
            placeholder={zernio?.configured ? 'Replace API key (optional)' : 'Zernio API key (sk_…)'} onChange={(e) => setZKey(e.target.value)} />
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(130px, 1fr))', gap: 8 }}>
            {['tiktok', 'instagram', 'youtube', 'facebook'].map((p) => (
              <input key={p} className="key-input" style={{ width: '100%', fontFamily: 'var(--font-sans)' }}
                aria-label={`${p} account id`}
                value={accts[p] || ''} placeholder={`${p} account id`} onChange={(e) => setAccts((a) => ({ ...a, [p]: e.target.value }))} />
            ))}
          </div>
          <div style={{ display: 'flex', gap: 10 }}>
            <Btn variant="secondary" size="sm" icon="rss" onClick={discover}>Discover from Zernio</Btn>
            <Btn variant="primary" size="sm" icon="check" onClick={saveZernioCfg}>Save</Btn>
          </div>
        </div>
      </Panel>

      <Panel title="AI Watchdog & Alerts" sub="Autonomous error diagnosis + push notifications to your phone" icon="activity" style={{ marginBottom: 18 }}>
        <div className="zernio-card" style={{ flexDirection: 'column', alignItems: 'stretch', gap: 14 }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
              <div className="zico"><Icon n="bell" /></div>
              <div>
                <div className="kt">Mobile Push Alerts</div>
                <div className="kd">Send instant alerts to your phone if a task fails or needs action</div>
              </div>
            </div>
            <Switch checked={wdEnabled} onChange={(v) => { setWdEnabled(v); saveWatchdogCfg({ enabled: v }); }} />
          </div>

          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', borderTop: '1px solid var(--border-subtle, rgba(255,255,255,0.08))', paddingTop: 10 }}>
            <div>
              <div className="ot" style={{ fontSize: 13 }}>AI Error Diagnosis (Doctor)</div>
              <div className="od">Use Gemini to diagnose root causes & propose fixes automatically</div>
            </div>
            <Switch checked={wdAi} onChange={(v) => { setWdAi(v); saveWatchdogCfg({ ai_diagnosis: v }); }} />
          </div>

          <div style={{ borderTop: '1px solid var(--border-subtle, rgba(255,255,255,0.08))', paddingTop: 10 }}>
            <div className="od" style={{ marginBottom: 8, fontWeight: 600 }}>Alert Channel / Provider</div>
            <Segmented
              value={wdProvider}
              onChange={(p) => { setWdProvider(p); saveWatchdogCfg({ provider: p }); }}
              options={[
                { id: 'ntfy', label: 'ntfy.sh (App)' },
                { id: 'telegram', label: 'Telegram' },
                { id: 'discord', label: 'Discord' },
                { id: 'webhook', label: 'Webhook' },
              ]}
            />
          </div>

          {wdProvider === 'ntfy' && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
              <div className="od">Install the free <b>ntfy</b> app on your phone, pick any private topic name, and subscribe:</div>
              <input
                className="key-input"
                style={{ width: '100%' }}
                placeholder="Topic name (e.g. clippyme-alerts-khaldun)"
                value={wdTopic}
                onChange={(e) => setWdTopic(e.target.value)}
              />
            </div>
          )}

          {wdProvider === 'telegram' && (
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
              <input
                className="key-input"
                type="password"
                placeholder={watchdog?.has_telegram_token ? 'Replace Bot Token' : 'Telegram Bot Token'}
                value={wdTgToken}
                onChange={(e) => setWdTgToken(e.target.value)}
              />
              <input
                className="key-input"
                placeholder="Telegram Chat ID"
                value={wdTgChat}
                onChange={(e) => setWdTgChat(e.target.value)}
              />
            </div>
          )}

          {wdProvider === 'discord' && (
            <input
              className="key-input"
              style={{ width: '100%' }}
              type="password"
              placeholder={watchdog?.has_discord_webhook ? 'Replace Discord Webhook URL' : 'https://discord.com/api/webhooks/...'}
              value={wdDiscUrl}
              onChange={(e) => setWdDiscUrl(e.target.value)}
            />
          )}

          {wdProvider === 'webhook' && (
            <input
              className="key-input"
              style={{ width: '100%' }}
              placeholder={watchdog?.has_webhook_url ? 'Replace Webhook URL' : 'https://example.com/webhook'}
              value={wdWhUrl}
              onChange={(e) => setWdWhUrl(e.target.value)}
            />
          )}

          <div style={{ display: 'flex', gap: 10, marginTop: 4 }}>
            <Btn variant="primary" size="sm" icon="check" onClick={() => saveWatchdogCfg()}>Save Settings</Btn>
            <Btn variant="secondary" size="sm" icon="send" disabled={testingWd} onClick={sendTestAlert}>
              {testingWd ? 'Sending…' : 'Send Test Alert'}
            </Btn>
          </div>
        </div>
      </Panel>

      <Panel title="Brand assets" sub="Logo overlay + custom subtitle fonts" icon="stamp" style={{ marginBottom: 18 }}>
        <div className="opt">
          <div className="oico"><Icon n="image" /></div>
          <div className="otxt"><div className="ot">Brand logo</div><div className="od">{logoOn ? 'Configured · burned on clips when the Logo layer is on' : 'Upload a transparent PNG to overlay on clips'}</div></div>
          <div className="r" style={{ gap: 8 }}>
            <label className="btn btn-secondary btn-sm" style={{ cursor: 'pointer' }}>
              <Icon n="upload" />Upload
              <input type="file" accept="image/png,.png" hidden onChange={onLogoFile} />
            </label>
            {logoOn && <Btn variant="ghost" size="sm" icon="trash-2" onClick={removeLogo}>Remove</Btn>}
          </div>
        </div>
        <div className="opt" style={{ borderBottom: 0, alignItems: 'flex-start' }}>
          <div className="oico"><Icon n="baseline" /></div>
          <div className="otxt" style={{ flex: 1 }}>
            <div className="ot">Subtitle fonts</div>
            <div className="od">Upload a .ttf/.otf (e.g. Stratos) to use in classic captions</div>
            {fonts.filter((n) => !bundled.has(n)).length > 0 && (
              <div className="s-sub" style={{ marginTop: 10 }}>
                {fonts.filter((n) => !bundled.has(n)).map((n) => (
                  <span key={n} className="chip" style={{ display: 'inline-flex', alignItems: 'center', gap: 6 }}>
                    {n}
                    <button type="button" className="mini" aria-label={`Remove ${n}`} title="Remove" onClick={() => removeFont(n)}><Icon n="x" /></button>
                  </span>
                ))}
              </div>
            )}
          </div>
          <div className="r">
            <label className="btn btn-secondary btn-sm" style={{ cursor: 'pointer' }}>
              <Icon n="upload" />Upload
              <input type="file" accept=".ttf,.otf,.ttc,font/ttf,font/otf" hidden onChange={onFontFile} />
            </label>
          </div>
        </div>
      </Panel>

      <Panel title="Storage & Disk Cleanup" sub="Manage local video files and purge published/failed task storage" icon="hard-drive" style={{ marginBottom: 18 }}>
        <div className="opt" style={{ borderBottom: 0 }}>
          <div className="oico"><Icon n="trash-2" /></div>
          <div className="otxt">
            <div className="ot">Disk cleanup</div>
            <div className="od">Purge large source videos after publishing and delete partial download leftovers</div>
          </div>
          <div className="r" style={{ gap: 8 }}>
            <Btn
              variant="secondary"
              size="sm"
              icon="sparkles"
              disabled={cleaningStorage}
              onClick={async () => {
                setCleaningStorage(true);
                try {
                  const res = await triggerStorageCleanup();
                  pushToast?.('success', `Freed ${res.freed_mb || 0} MB across ${res.removed_files || 0} files`);
                } catch (e) {
                  pushToast?.('error', `Cleanup failed: ${e.message}`);
                } finally {
                  setCleaningStorage(false);
                }
              }}
            >
              {cleaningStorage ? 'Cleaning…' : 'Clean Storage Now'}
            </Btn>
          </div>
        </div>
      </Panel>

      <Panel title="Downloads" sub="For age- or region-restricted sources" icon="cookie">
        <div className="opt" style={{ borderBottom: 0 }}>
          <div className="oico"><Icon n="cookie" /></div>
          <div className="otxt"><div className="ot">YouTube cookies</div><div className="od">{cookies ? 'Configured · restricted videos OK' : 'Not set · public videos only'}</div></div>
          <div className="r" style={{ gap: 8 }}>
            <label className="btn btn-secondary btn-sm" style={{ cursor: 'pointer' }}>
              <Icon n="upload" />Upload
              <input type="file" accept=".txt" hidden onChange={onCookieFile} />
            </label>
            {cookies && <Btn variant="ghost" size="sm" icon="trash-2" onClick={removeCookies}>Remove</Btn>}
          </div>
        </div>
      </Panel>
    </div>
  );
}

export function ApiKeyModal({ onClose, onGoToSettings }) {
  const panelRef = useModalA11y(onClose);
  return (
    // Backdrop click is a mouse-only convenience; keyboard users close via
    // Esc (useModalA11y). currentTarget guard replaces stopPropagation.
    <div className="overlay" onClick={(e) => { if (e.target === e.currentTarget) onClose(); }}>
      <div className="modal" ref={panelRef}
        role="dialog" aria-modal="true" aria-labelledby="apikey-modal-title">
        <div className="modal-head"><h3 id="apikey-modal-title">Add your Gemini key</h3><button className="x" onClick={onClose} aria-label="Close"><Icon n="x" /></button></div>
        <div className="modal-body">
          <p style={{ color: 'var(--fg-2)', fontSize: 14, lineHeight: 1.55 }}>
            ClippyMe needs a Gemini key to score the transcript and find viral moments. It&apos;s stored locally and never leaves your machine.
          </p>
        </div>
        <div className="modal-foot">
          <Btn variant="ghost" onClick={onClose}>Later</Btn>
          <div className="mf-right"><Btn variant="primary" icon="settings" onClick={onGoToSettings}>Open settings</Btn></div>
        </div>
      </div>
    </div>
  );
}

export function AiShortsView() {
  const [input, setInput] = useState('');
  const [busy, setBusy] = useState(false);
  const [research, setResearch] = useState(null);
  const [scripts, setScripts] = useState([]);
  const [err, setErr] = useState('');

  const handleResearch = async () => {
    if (!input.trim()) return;
    setBusy(true);
    setErr('');
    setResearch(null);
    setScripts([]);
    try {
      const res = await researchProduct(input);
      setResearch(res);
      const sRes = await generateUgcScripts(res);
      setScripts(sRes.scripts || []);
    } catch (e) {
      setErr(e.message || 'Generation failed');
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="container narrow fade-in">
      <Hero eyebrow="AI Shorts" line1="UGC Video Creator" sub="Turn any product URL or prompt into viral marketing scripts, talking heads, and shorts." />
      <Panel title="Product or URL" sub="Enter your website, SaaS URL, or product description" icon="sparkles" style={{ marginBottom: 18 }}>
        <div style={{ display: 'flex', gap: 8, flexDirection: 'column' }}>
          <textarea
            className="input-field"
            rows={3}
            placeholder="e.g. https://myproduct.com or 'A smart timer app that boosts deep work focus'"
            value={input}
            onChange={(e) => setInput(e.target.value)}
          />
          <div style={{ display: 'flex', justifyContent: 'flex-end', marginTop: 6 }}>
            <Btn variant="primary" icon={busy ? 'loader' : 'wand-sparkles'} disabled={busy || !input.trim()} onClick={handleResearch}>
              {busy ? 'Researching & Writing…' : 'Generate Viral Scripts'}
            </Btn>
          </div>
        </div>
        {err && <div className="eo-d" style={{ color: '#ef4444', marginTop: 10 }}>❌ {err}</div>}
      </Panel>

      {research && (
        <Panel title="Market & Product Research" sub={research.product_name || 'Insights'} icon="search" style={{ marginBottom: 18 }}>
          <div style={{ fontSize: 13, lineHeight: 1.6, color: 'var(--fg-2)' }}>
            <div><b>Target Audience:</b> {research.target_audience}</div>
            <div style={{ marginTop: 4 }}><b>Core Problem:</b> {research.core_problem}</div>
            <div style={{ marginTop: 4 }}><b>Unique Solution:</b> {research.unique_solution}</div>
          </div>
        </Panel>
      )}

      {scripts.length > 0 && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
          <h3>Generated Viral Scripts ({scripts.length})</h3>
          {scripts.map((s, idx) => (
            <Panel key={idx} title={s.title || `Script ${idx + 1}`} sub={s.hook_text_overlay} icon="film">
              <div style={{ fontSize: 14, lineHeight: 1.6, marginBottom: 10 }}>
                <b>Spoken Voiceover:</b>
                <p style={{ marginTop: 4, background: 'var(--bg-3)', padding: 10, borderRadius: 6 }}>{s.full_script_text}</p>
              </div>
              {s.segments && (
                <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                  <span className="field-label">Timeline Breakdown</span>
                  {s.segments.map((seg, sIdx) => (
                    <div key={sIdx} className="eo-d" style={{ background: 'var(--bg-2)', padding: '6px 10px', borderRadius: 4 }}>
                      <b>[{seg.segment_type}]</b> {seg.voiceover} <span style={{ color: 'var(--fg-muted)' }}>({seg.visual_description})</span>
                    </div>
                  ))}
                </div>
              )}
            </Panel>
          ))}
        </div>
      )}
    </div>
  );
}

export function YoutubeStudioView() {
  const [transcript, setTranscript] = useState('');
  const [busy, setBusy] = useState(false);
  const [titlesData, setTitlesData] = useState(null);
  const [chatPrompt, setChatPrompt] = useState('');
  const [chatBusy, setChatBusy] = useState(false);
  const [thumbTitle, setThumbTitle] = useState('');
  const [thumbExtra, setThumbExtra] = useState('');
  const [thumbRatio, setThumbRatio] = useState('16:9');
  const [thumbBusy, setThumbBusy] = useState(false);
  const [thumbPath, setThumbPath] = useState('');
  const [err, setErr] = useState('');

  const handleGetTitles = async () => {
    if (!transcript.trim()) return;
    setBusy(true);
    setErr('');
    setTitlesData(null);
    try {
      const res = await fetchViralTitles(transcript);
      setTitlesData(res);
      if (res.titles?.length) {
        setThumbTitle(res.titles[0]);
      }
    } catch (e) {
      setErr(e.message || 'Failed to generate titles');
    } finally {
      setBusy(false);
    }
  };

  const handleRefine = async () => {
    if (!chatPrompt.trim() || !titlesData) return;
    setChatBusy(true);
    try {
      const res = await refineViralTitles(titlesData.summary || transcript.slice(0, 1000), chatPrompt);
      if (res.titles) {
        setTitlesData((prev) => ({ ...prev, titles: res.titles }));
        setChatPrompt('');
      }
    } catch (e) {
      setErr(e.message || 'Refinement failed');
    } finally {
      setChatBusy(false);
    }
  };

  const handleGenerateThumbnail = async () => {
    const title = thumbTitle.trim() || (titlesData?.titles?.[0] ?? '');
    if (!title) {
      setErr('Please enter a video title for thumbnail generation');
      return;
    }
    setThumbBusy(true);
    setErr('');
    try {
      const res = await generateThumbnail(title, transcript.slice(0, 1000), thumbExtra, thumbRatio);
      if (res.thumbnail_path) {
        const basename = res.thumbnail_path.split(/[\\/]/).pop();
        setThumbPath(`/thumbnails/${basename}`);
      }
    } catch (e) {
      setErr(e.message || 'Thumbnail generation failed');
    } finally {
      setThumbBusy(false);
    }
  };

  return (
    <div className="container narrow fade-in">
      <Hero eyebrow="YouTube Studio" line1="Viral Titles & Thumbnails" sub="Generate high-CTR titles, chapters, and thumbnail concepts for your videos." />
      <Panel title="Video Transcript or Summary" sub="Paste video content to analyze CTR potential" icon="file-text" style={{ marginBottom: 18 }}>
        <textarea
          className="input-field"
          rows={4}
          placeholder="Paste transcript or key video points here..."
          value={transcript}
          onChange={(e) => setTranscript(e.target.value)}
        />
        <div style={{ display: 'flex', justifyContent: 'flex-end', marginTop: 8 }}>
          <Btn variant="primary" icon={busy ? 'loader' : 'sparkles'} disabled={busy || !transcript.trim()} onClick={handleGetTitles}>
            {busy ? 'Analyzing...' : 'Generate 10 Viral Titles'}
          </Btn>
        </div>
        {err && <div className="eo-d" style={{ color: '#ef4444', marginTop: 10 }}>❌ {err}</div>}
      </Panel>

      {titlesData && (
        <Panel title="Suggested Viral Titles" sub="10 CTR-optimized candidates with curiosity gaps" icon="list" style={{ marginBottom: 18 }}>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8, marginBottom: 14 }}>
            {titlesData.titles?.map((t, idx) => (
              <div key={idx} style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '8px 12px', background: 'var(--bg-2)', borderRadius: 6 }}>
                <span style={{ fontWeight: 500, fontSize: 14 }}>{idx + 1}. {t}</span>
                <div style={{ display: 'flex', gap: 6 }}>
                  <Btn size="sm" variant="secondary" icon="sparkles" onClick={() => setThumbTitle(t)}>Use for Thumbnail</Btn>
                  <Btn size="sm" variant="ghost" icon="copy" onClick={() => navigator.clipboard.writeText(t)}>Copy</Btn>
                </div>
              </div>
            ))}
          </div>

          <div style={{ borderTop: '1px solid var(--border)', paddingTop: 12 }}>
            <span className="field-label">Refine Titles with AI</span>
            <div style={{ display: 'flex', gap: 8, marginTop: 6 }}>
              <input
                className="input-field"
                style={{ flex: 1 }}
                placeholder="e.g. Make them sound more urgent or focused on beginners..."
                value={chatPrompt}
                onChange={(e) => setChatPrompt(e.target.value)}
                onKeyDown={(e) => { if (e.key === 'Enter') handleRefine(); }}
              />
              <Btn variant="secondary" disabled={chatBusy || !chatPrompt.trim()} onClick={handleRefine}>
                {chatBusy ? 'Refining…' : 'Refine'}
              </Btn>
            </div>
          </div>
        </Panel>
      )}

      <Panel title="AI Thumbnail & Cover Generator (Nano Banana)" sub="Generate eye-catching high-CTR cover images with 16:9, 9:16 or 1:1 aspect ratios" icon="image" style={{ marginBottom: 18 }}>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
          <div>
            <span className="field-label" style={{ display: 'block', marginBottom: 4 }}>Video / Thumbnail Title</span>
            <input
              className="input-field"
              placeholder="e.g. Why 99% Of People Fail At This..."
              value={thumbTitle}
              onChange={(e) => setThumbTitle(e.target.value)}
            />
          </div>

          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <span className="field-label" style={{ margin: 0 }}>Aspect Ratio:</span>
            <div style={{ display: 'flex', gap: 8 }}>
              {[
                { id: '16:9', label: '16:9 (YouTube)' },
                { id: '9:16', label: '9:16 (Shorts/Reels)' },
                { id: '1:1', label: '1:1 (Square/Feed)' },
              ].map((r) => (
                <button
                  key={r.id}
                  type="button"
                  onClick={() => setThumbRatio(r.id)}
                  style={{
                    border: '1px solid ' + (thumbRatio === r.id ? 'var(--brand-teal)' : 'var(--border)'),
                    background: thumbRatio === r.id ? 'rgba(2,197,191,0.15)' : 'transparent',
                    color: thumbRatio === r.id ? 'var(--brand-teal)' : 'var(--fg-2)',
                    padding: '4px 10px',
                    borderRadius: '6px',
                    fontSize: '12px',
                    fontWeight: thumbRatio === r.id ? 600 : 400,
                    cursor: 'pointer',
                  }}
                >
                  {r.label}
                </button>
              ))}
            </div>
          </div>

          <div>
            <span className="field-label" style={{ display: 'block', marginBottom: 4 }}>Design / Vibe Instructions (Optional)</span>
            <input
              className="input-field"
              placeholder="e.g. Dramatic lighting, shocked expression, neon accents..."
              value={thumbExtra}
              onChange={(e) => setThumbExtra(e.target.value)}
            />
          </div>

          <div style={{ display: 'flex', justifyContent: 'flex-end', marginTop: 4 }}>
            <Btn variant="primary" icon={thumbBusy ? 'loader' : 'sparkles'} disabled={thumbBusy || !thumbTitle.trim()} onClick={handleGenerateThumbnail}>
              {thumbBusy ? 'Generating Cover…' : 'Generate Thumbnail'}
            </Btn>
          </div>

          {thumbPath && (
            <div style={{ marginTop: 14, textAlign: 'center', background: 'var(--bg-2)', padding: 14, borderRadius: 8, border: '1px solid var(--border)' }}>
              <div style={{ marginBottom: 8, fontWeight: 600, fontSize: 13, color: 'var(--brand-teal)' }}>Generated AI Thumbnail ({thumbRatio})</div>
              <img
                src={thumbPath}
                alt="AI Generated Thumbnail"
                style={{
                  maxWidth: '100%',
                  maxHeight: thumbRatio === '9:16' ? 360 : 280,
                  borderRadius: 6,
                  boxShadow: '0 4px 16px rgba(0,0,0,0.4)',
                }}
              />
            </div>
          )}
        </div>
      </Panel>
    </div>
  );
}
