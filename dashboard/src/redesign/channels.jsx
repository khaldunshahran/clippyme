// ClippyMe redesign — Channels: Multi-Channel Hub and Auto-Routing.
// Allows creators to manage multiple niche channels (e.g. US Politics,
// Global Breaking & Disasters, Pop Culture) with custom attribution handles,
// subtitle styles, and auto-publishing destinations.
import { useState, useEffect, useCallback } from 'react';
import { createPortal } from 'react-dom';
import { Hero } from './chrome';
import { Icon, Btn, Badge, Panel, Social, Switch } from './primitives';
import { useModalA11y } from './useModalA11y';
import {
  getChannels,
  createChannel,
  updateChannel,
  deleteChannel,
} from './realApi';

const DEFAULT_NICHES = [
  'politics',
  'breaking_world',
  'entertainment',
  'culture',
  'technology',
  'business',
  'sports',
  'science',
];

const SUB_PRESETS = [
  { id: 'hormozi_bold', label: 'Hormozi Bold (High Viral)' },
  { id: 'classic', label: 'Classic Clean' },
  { id: 'beast', label: 'MrBeast Style (Yellow Punch)' },
  { id: 'minimal', label: 'Minimalist' },
];

const REFRAME_MODES = [
  { id: 'auto', label: 'Auto (Speaker Tracking)' },
  { id: 'center', label: 'Center Crop' },
  { id: 'crop_9_16', label: 'Fill 9:16 Vertical' },
];

const PLATFORM_OPTIONS = [
  { id: 'youtube', label: 'YouTube Shorts', icon: 'youtube' },
  { id: 'tiktok', label: 'TikTok', icon: 'tiktok' },
  { id: 'instagram', label: 'Instagram Reels', icon: 'instagram' },
];

function ChannelModal({
  channel,
  mode = 'create',
  onClose,
  onSaved,
  pushToast,
}) {
  const isEdit = mode === 'edit';
  const panelRef = useModalA11y(onClose, true);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');

  const [name, setName] = useState(channel?.name || '');
  const [description, setDescription] = useState(channel?.description || '');
  const [nichesText, setNichesText] = useState((channel?.niches || []).join(', '));
  const [bannerPlatform, setBannerPlatform] = useState(channel?.banner_platform || 'youtube');
  const [bannerHandle, setBannerHandle] = useState(channel?.banner_handle || '');
  const [subPreset, setSubPreset] = useState(channel?.sub_preset || 'hormozi_bold');
  const [subColor, setSubColor] = useState(channel?.sub_color || '#FFFFFF');
  const [reframeMode, setReframeMode] = useState(channel?.reframe_mode || 'auto');
  const [pubTargets, setPubTargets] = useState(
    channel?.publishing_targets || { youtube: true, tiktok: true, instagram: false }
  );

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!name.trim()) {
      setError('Please provide a channel name.');
      return;
    }

    setBusy(true);
    setError('');

    const niches = nichesText
      .split(',')
      .map((s) => s.trim().toLowerCase())
      .filter(Boolean);

    const payload = {
      name: name.trim(),
      description: description.trim(),
      niches: niches.length ? niches : ['general'],
      banner_platform: bannerPlatform,
      banner_handle: bannerHandle.trim() || undefined,
      sub_preset: subPreset,
      sub_color: subColor,
      reframe_mode: reframeMode,
      publishing_targets: pubTargets,
    };

    try {
      if (isEdit && channel?.id) {
        await updateChannel(channel.id, payload);
        pushToast?.('success', `Channel "${name}" updated successfully.`);
      } else {
        await createChannel(payload);
        pushToast?.('success', `Channel "${name}" created.`);
      }
      onSaved?.();
      onClose();
    } catch (err) {
      setError(err?.message || 'Failed to save channel.');
    } finally {
      setBusy(false);
    }
  };

  const modalNode = (
    <div
      className="overlay"
      onClick={(e) => {
        if (e.target === e.currentTarget && !busy) onClose();
      }}
    >
      <div
        className="modal"
        ref={panelRef}
        role="dialog"
        aria-modal="true"
        aria-labelledby="channel-modal-title"
        style={{ maxWidth: 560 }}
      >
        <div className="modal-head">
          <h3 id="channel-modal-title">
            {isEdit ? `Edit Channel: ${channel?.name}` : 'Add New Channel Profile'}
          </h3>
          <button
            type="button"
            className="x"
            onClick={onClose}
            aria-label="Close modal"
            disabled={busy}
          >
            <Icon n="x" />
          </button>
        </div>

        <form onSubmit={handleSubmit}>
          <div
            className="modal-body"
            style={{ display: 'flex', flexDirection: 'column', gap: 16, padding: '20px 24px' }}
          >
            {error && (
              <div
                style={{
                  background: 'rgba(235, 87, 87, 0.12)',
                  border: '1px solid var(--danger, #eb5757)',
                  borderRadius: 8,
                  padding: '10px 12px',
                  color: 'var(--danger, #eb5757)',
                  fontSize: '0.85rem',
                }}
              >
                {error}
              </div>
            )}

            {/* Channel Name */}
            <div>
              <label
                htmlFor="channel-name"
                style={{ display: 'block', fontSize: '0.85rem', fontWeight: 600, marginBottom: 4 }}
              >
                Channel Name *
              </label>
              <input
                id="channel-name"
                className="input-field"
                type="text"
                placeholder="e.g. US Politics Daily"
                value={name}
                onChange={(e) => setName(e.target.value)}
                required
                style={{ width: '100%', padding: '8px 12px', borderRadius: 8 }}
              />
            </div>

            {/* Description */}
            <div>
              <label
                htmlFor="channel-desc"
                style={{ display: 'block', fontSize: '0.85rem', fontWeight: 600, marginBottom: 4 }}
              >
                Audience / Description
              </label>
              <textarea
                id="channel-desc"
                className="input-field"
                rows={2}
                placeholder="Brief summary of topics, target audience, and editorial angle..."
                value={description}
                onChange={(e) => setDescription(e.target.value)}
                style={{ width: '100%', padding: '8px 12px', borderRadius: 8, resize: 'vertical' }}
              />
            </div>

            {/* Niches / Keywords */}
            <div>
              <label
                htmlFor="channel-niches"
                style={{ display: 'block', fontSize: '0.85rem', fontWeight: 600, marginBottom: 4 }}
              >
                Topic Niches & Matching Keywords (comma-separated)
              </label>
              <input
                id="channel-niches"
                className="input-field"
                type="text"
                placeholder="e.g. politics, congress, elections, us-news"
                value={nichesText}
                onChange={(e) => setNichesText(e.target.value)}
                style={{ width: '100%', padding: '8px 12px', borderRadius: 8 }}
              />
              <span style={{ fontSize: '0.75rem', color: 'var(--ink-dim)', marginTop: 4, display: 'block' }}>
                Trend Radar automatically routes incoming topics matching these keywords to this channel.
              </span>
            </div>

            {/* Banner Attribution */}
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
              <div>
                <label
                  htmlFor="channel-platform"
                  style={{ display: 'block', fontSize: '0.85rem', fontWeight: 600, marginBottom: 4 }}
                >
                  Banner Platform
                </label>
                <select
                  id="channel-platform"
                  className="input-field"
                  value={bannerPlatform}
                  onChange={(e) => setBannerPlatform(e.target.value)}
                  style={{ width: '100%', padding: '8px 10px', borderRadius: 8 }}
                >
                  {PLATFORM_OPTIONS.map((p) => (
                    <option key={p.id} value={p.id}>
                      {p.label}
                    </option>
                  ))}
                </select>
              </div>

              <div>
                <label
                  htmlFor="channel-handle"
                  style={{ display: 'block', fontSize: '0.85rem', fontWeight: 600, marginBottom: 4 }}
                >
                  Channel Handle
                </label>
                <input
                  id="channel-handle"
                  className="input-field"
                  type="text"
                  placeholder="@ChannelHandle"
                  value={bannerHandle}
                  onChange={(e) => setBannerHandle(e.target.value)}
                  style={{ width: '100%', padding: '8px 12px', borderRadius: 8 }}
                />
              </div>
            </div>

            {/* Styling Presets */}
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 12 }}>
              <div>
                <label
                  htmlFor="channel-sub-preset"
                  style={{ display: 'block', fontSize: '0.85rem', fontWeight: 600, marginBottom: 4 }}
                >
                  Subtitle Style
                </label>
                <select
                  id="channel-sub-preset"
                  className="input-field"
                  value={subPreset}
                  onChange={(e) => setSubPreset(e.target.value)}
                  style={{ width: '100%', padding: '8px 10px', borderRadius: 8 }}
                >
                  {SUB_PRESETS.map((p) => (
                    <option key={p.id} value={p.id}>
                      {p.label}
                    </option>
                  ))}
                </select>
              </div>

              <div>
                <label
                  htmlFor="channel-sub-color"
                  style={{ display: 'block', fontSize: '0.85rem', fontWeight: 600, marginBottom: 4 }}
                >
                  Subtitle Color
                </label>
                <input
                  id="channel-sub-color"
                  className="input-field"
                  type="text"
                  placeholder="#FFFFFF"
                  value={subColor}
                  onChange={(e) => setSubColor(e.target.value)}
                  style={{ width: '100%', padding: '8px 10px', borderRadius: 8 }}
                />
              </div>

              <div>
                <label
                  htmlFor="channel-reframe"
                  style={{ display: 'block', fontSize: '0.85rem', fontWeight: 600, marginBottom: 4 }}
                >
                  Default Reframe
                </label>
                <select
                  id="channel-reframe"
                  className="input-field"
                  value={reframeMode}
                  onChange={(e) => setReframeMode(e.target.value)}
                  style={{ width: '100%', padding: '8px 10px', borderRadius: 8 }}
                >
                  {REFRAME_MODES.map((m) => (
                    <option key={m.id} value={m.id}>
                      {m.label}
                    </option>
                  ))}
                </select>
              </div>
            </div>

            {/* Publishing Targets Toggles */}
            <div>
              <span style={{ display: 'block', fontSize: '0.85rem', fontWeight: 600, marginBottom: 8 }}>
                Publishing Destinations
              </span>
              <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap' }}>
                {PLATFORM_OPTIONS.map((p) => (
                  <label
                    key={p.id}
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: 8,
                      fontSize: '0.85rem',
                      cursor: 'pointer',
                      background: 'var(--surface-deep, rgba(0,0,0,0.1))',
                      padding: '6px 12px',
                      borderRadius: 8,
                      border: '1px solid var(--border-dim, rgba(255,255,255,0.05))',
                    }}
                  >
                    <Switch
                      checked={!!pubTargets[p.id]}
                      onChange={(on) =>
                        setPubTargets((prev) => ({ ...prev, [p.id]: on }))
                      }
                      label={`Enable ${p.label}`}
                    />
                    <Social n={p.icon} size={14} color="var(--ink)" />
                    <span>{p.label}</span>
                  </label>
                ))}
              </div>
            </div>
          </div>

          <div
            className="modal-foot"
            style={{
              display: 'flex',
              justifyContent: 'flex-end',
              gap: 10,
              padding: '16px 24px',
              borderTop: '1px solid var(--border-dim, rgba(255,255,255,0.06))',
            }}
          >
            <Btn variant="ghost" size="sm" onClick={onClose} disabled={busy}>
              Cancel
            </Btn>
            <Btn variant="primary" size="sm" type="submit" loading={busy}>
              {isEdit ? 'Save Changes' : 'Create Channel'}
            </Btn>
          </div>
        </form>
      </div>
    </div>
  );

  return typeof document !== 'undefined' ? createPortal(modalNode, document.body) : null;
}

export function ChannelsView({
  pushToast,
  onSelectChannelForRadar,
}) {
  const [channels, setChannels] = useState([]);
  const [loading, setLoading] = useState(true);
  const [modalState, setModalState] = useState({ open: false, mode: 'create', channel: null });
  const [deletingId, setDeletingId] = useState(null);

  const fetchChannelsList = useCallback(async () => {
    try {
      setLoading(true);
      const data = await getChannels();
      setChannels(data.channels || []);
    } catch (err) {
      pushToast?.('error', `Failed to load channels: ${err.message || err}`);
    } finally {
      setLoading(false);
    }
  }, [pushToast]);

  useEffect(() => {
    fetchChannelsList();
  }, [fetchChannelsList]);

  const handleCreate = () => {
    setModalState({ open: true, mode: 'create', channel: null });
  };

  const handleEdit = (channel) => {
    setModalState({ open: true, mode: 'edit', channel });
  };

  const handleDelete = async (channel) => {
    if (!window.confirm(`Are you sure you want to delete channel "${channel.name}"?`)) {
      return;
    }
    setDeletingId(channel.id);
    try {
      await deleteChannel(channel.id);
      pushToast?.('info', `Channel "${channel.name}" deleted.`);
      fetchChannelsList();
    } catch (err) {
      pushToast?.('error', `Delete failed: ${err.message || err}`);
    } finally {
      setDeletingId(null);
    }
  };

  return (
    <div className="container" style={{ paddingBottom: 64 }}>
      <Hero
        eyebrow="Multi-Channel Hub"
        title={<span>Manage <em>Channel Profiles</em></span>}
        desc="Segment your content empire into dedicated channels (US Politics, Global Breaking, Pop Culture) with tailored styling presets, custom banner handles, and auto-routing."
      />

      {/* Toolbar */}
      <div
        className="panel raised-sm"
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          gap: 16,
          padding: '14px 20px',
          borderRadius: 14,
          marginBottom: 24,
          flexWrap: 'wrap',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <Badge tone="out">{channels.length} Active Channels</Badge>
          <span style={{ fontSize: '0.82rem', color: 'var(--ink-dim)' }}>
            Each channel controls topic routing, subtitle styles & attribution handles
          </span>
        </div>

        <Btn
          variant="primary"
          size="sm"
          icon="plus"
          onClick={handleCreate}
        >
          + New Channel
        </Btn>
      </div>

      {/* Channel Cards Grid */}
      {loading ? (
        <div style={{ padding: '48px 0', textAlign: 'center', color: 'var(--ink-dim)' }}>
          <Icon n="loader" style={{ fontSize: 24, marginBottom: 8 }} />
          <div>Loading channel profiles...</div>
        </div>
      ) : channels.length === 0 ? (
        <Panel pad style={{ textAlign: 'center', padding: '40px 20px' }}>
          <div style={{ fontSize: '2rem', marginBottom: 8 }}>📺</div>
          <h3>No Channels Found</h3>
          <p style={{ color: 'var(--ink-dim)', maxWidth: 460, margin: '0 auto 16px' }}>
            Create your first channel profile to configure niche topic routing and custom branding.
          </p>
          <Btn variant="primary" size="sm" icon="plus" onClick={handleCreate}>
            Create Channel
          </Btn>
        </Panel>
      ) : (
        <div
          style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fill, minmax(340px, 1fr))',
            gap: 20,
          }}
        >
          {channels.map((ch) => {
            const isDeleting = deletingId === ch.id;
            const bannerHandle = ch.banner_handle || `@${ch.name.replace(/\s+/g, '')}`;
            const targetEntries = Object.entries(ch.publishing_targets || {}).filter(([, on]) => on);

            return (
              <div
                key={ch.id}
                className="panel raised-sm"
                style={{
                  display: 'flex',
                  flexDirection: 'column',
                  gap: 14,
                  padding: 20,
                  borderRadius: 16,
                  position: 'relative',
                  border: '1px solid var(--border-dim, rgba(255,255,255,0.06))',
                }}
              >
                {/* Header: Name + Handle */}
                <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', gap: 12 }}>
                  <div>
                    <h3 style={{ fontSize: '1.2rem', fontWeight: 700, margin: '0 0 4px', color: 'var(--ink)' }}>
                      {ch.name}
                    </h3>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: '0.82rem', color: 'var(--accent, #02C5BF)' }}>
                      <Social n={ch.banner_platform || 'youtube'} size={14} color="var(--accent, #02C5BF)" />
                      <span style={{ fontWeight: 600 }}>{bannerHandle}</span>
                    </div>
                  </div>

                  <Badge tone="accent" icon="tv">
                    {ch.banner_platform ? ch.banner_platform.toUpperCase() : 'HUB'}
                  </Badge>
                </div>

                {/* Description */}
                {ch.description && (
                  <p style={{ fontSize: '0.85rem', color: 'var(--ink-dim)', margin: 0, lineHeight: 1.45 }}>
                    {ch.description}
                  </p>
                )}

                {/* Niches Badges */}
                <div>
                  <div style={{ fontSize: '0.74rem', fontWeight: 600, color: 'var(--ink-dim)', marginBottom: 6, textTransform: 'uppercase', letterSpacing: '0.04em' }}>
                    Routing Niches
                  </div>
                  <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
                    {(ch.niches || []).map((niche) => (
                      <Badge key={niche} tone="dim">
                        #{niche}
                      </Badge>
                    ))}
                  </div>
                </div>

                {/* Style & Reframe Preset Summary */}
                <div
                  style={{
                    background: 'var(--surface-deep, rgba(0,0,0,0.12))',
                    borderRadius: 10,
                    padding: '10px 12px',
                    display: 'grid',
                    gridTemplateColumns: '1fr 1fr',
                    gap: 8,
                    fontSize: '0.78rem',
                  }}
                >
                  <div>
                    <span style={{ color: 'var(--ink-dim)', display: 'block' }}>Subtitle Style</span>
                    <strong style={{ color: 'var(--ink)' }}>{ch.sub_preset || 'Hormozi Bold'}</strong>
                  </div>
                  <div>
                    <span style={{ color: 'var(--ink-dim)', display: 'block' }}>Reframe</span>
                    <strong style={{ color: 'var(--ink)' }}>{ch.reframe_mode || 'Auto'}</strong>
                  </div>
                </div>

                {/* Publishing Targets */}
                {targetEntries.length > 0 && (
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: '0.76rem', color: 'var(--ink-dim)' }}>
                    <span>Destinations:</span>
                    <div style={{ display: 'flex', gap: 6 }}>
                      {targetEntries.map(([plat]) => (
                        <span
                          key={plat}
                          style={{
                            display: 'inline-flex',
                            alignItems: 'center',
                            gap: 4,
                            background: 'var(--surface, rgba(255,255,255,0.05))',
                            padding: '2px 6px',
                            borderRadius: 6,
                            color: 'var(--ink)',
                            fontWeight: 600,
                          }}
                        >
                          <Social n={plat} size={11} color="var(--ink)" />
                          {plat}
                        </span>
                      ))}
                    </div>
                  </div>
                )}

                {/* Action Buttons */}
                <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginTop: 'auto', paddingTop: 8 }}>
                  <Btn
                    variant="secondary"
                    size="sm"
                    icon="sliders-horizontal"
                    onClick={() => handleEdit(ch)}
                    style={{ flex: 1 }}
                  >
                    Edit Channel
                  </Btn>

                  {onSelectChannelForRadar && (
                    <Btn
                      variant="ghost"
                      size="sm"
                      icon="flame"
                      onClick={() => onSelectChannelForRadar(ch.id)}
                      title="View Trend Radar for this channel"
                    >
                      Radar
                    </Btn>
                  )}

                  <Btn
                    variant="ghost"
                    size="sm"
                    icon="trash-2"
                    loading={isDeleting}
                    disabled={isDeleting}
                    onClick={() => handleDelete(ch)}
                    title="Delete channel"
                  />
                </div>
              </div>
            );
          })}
        </div>
      )}

      {/* Create / Edit Modal */}
      {modalState.open && (
        <ChannelModal
          channel={modalState.channel}
          mode={modalState.mode}
          onClose={() => setModalState({ open: false, mode: 'create', channel: null })}
          onSaved={fetchChannelsList}
          pushToast={pushToast}
        />
      )}
    </div>
  );
}
