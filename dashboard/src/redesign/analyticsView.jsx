// ClippyMe redesign — Analytics: Published Clip Performance & AI Continuous Learning Loop.
// Tracks real-world views, likes, shares, and retention across TikTok, Instagram,
// and YouTube Shorts. Shows recorded measurements honestly; audience-based
// selection rules are not active yet.
import { useState, useEffect, useCallback } from 'react';
import { createPortal } from 'react-dom';
import { Hero } from './chrome';
import { Icon, Btn, Badge, Panel, Social } from './primitives';
import { useModalA11y } from './useModalA11y';
import {
  getAnalyticsSummary,
  syncAnalytics,
  trackClipAnalytics,
  getAnalyticsInsights,
} from './realApi';

function UpdateMetricsModal({
  clip,
  onClose,
  onSaved,
  pushToast,
}) {
  const panelRef = useModalA11y(onClose, true);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');

  const [views, setViews] = useState(clip?.metrics?.views ?? clip?.views ?? 0);
  const [likes, setLikes] = useState(clip?.metrics?.likes ?? clip?.likes ?? 0);
  const [shares, setShares] = useState(clip?.metrics?.shares ?? clip?.shares ?? 0);
  const [comments, setComments] = useState(clip?.metrics?.comments ?? clip?.comments ?? 0);
  const _initialRetention = clip?.metrics?.retention_rate ?? clip?.retention_rate;
  const [retentionPct, setRetentionPct] = useState(
    _initialRetention != null ? String(Math.round(_initialRetention * 100)) : ''
  );
  const [retentionTouched, setRetentionTouched] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setBusy(true);
    setError('');

    try {
      const metrics = {
        views: Number(views) || 0,
        likes: Number(likes) || 0,
        shares: Number(shares) || 0,
        comments: Number(comments) || 0,
      };
      if (retentionTouched && retentionPct !== '') {
        metrics.retention_rate = Math.min(1, Math.max(0, Number(retentionPct) / 100));
      }

      await trackClipAnalytics(clip.clip_id, metrics);
      pushToast?.('success', 'Performance metrics saved.');
      onSaved?.();
      onClose?.();
    } catch (err) {
      setError(err.message || 'Failed to update metrics');
    } finally {
      setBusy(false);
    }
  };

  const modalNode = (
    <div
      className="modal-overlay"
      role="dialog"
      aria-modal="true"
      aria-labelledby="metrics-modal-title"
      style={{
        position: 'fixed',
        inset: 0,
        backgroundColor: 'rgba(0, 0, 0, 0.65)',
        backdropFilter: 'blur(6px)',
        zIndex: 9999,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        padding: 16,
      }}
    >
      <div
        ref={panelRef}
        className="modal-panel raised"
        style={{
          width: '100%',
          maxWidth: 480,
          background: 'var(--surface)',
          borderRadius: 'var(--r-lg, 16px)',
          padding: 24,
          boxShadow: 'var(--clay-card, 0 20px 40px rgba(0,0,0,0.3))',
          color: 'var(--ink)',
        }}
      >
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
          <h2 id="metrics-modal-title" style={{ fontSize: '1.25rem', fontWeight: 700, margin: 0 }}>
            Update Clip Metrics
          </h2>
          <button
            type="button"
            className="btn btn-ghost"
            onClick={onClose}
            aria-label="Close dialog"
            style={{ padding: '6px 10px', borderRadius: '50%' }}
          >
            <Icon n="x" />
          </button>
        </div>

        <p style={{ fontSize: '0.875rem', color: 'var(--ink-dim)', marginBottom: 20, lineHeight: 1.4 }}>
          Saved metrics are stored as recorded measurements. They do not change clip selection rules; audience-based learning is not active yet.
        </p>

        <div style={{ background: 'var(--bg-subtle, rgba(0,0,0,0.04))', padding: '12px 14px', borderRadius: 8, marginBottom: 16 }}>
          <div style={{ fontWeight: 600, fontSize: '0.9rem', marginBottom: 4 }}>
            {clip?.title || 'Selected Clip'}
          </div>
          <div style={{ fontSize: '0.78rem', color: 'var(--ink-dim)' }}>
            Clip ID: <code style={{ fontSize: '0.75rem' }}>{clip?.clip_id}</code> · Duration: {clip?.duration ? `${clip.duration.toFixed(1)}s` : 'Unknown'}
          </div>
        </div>

        {error && (
          <div
            role="alert"
            style={{
              padding: '10px 14px',
              borderRadius: 8,
              background: 'rgba(217, 83, 79, 0.15)',
              color: '#d9534f',
              fontSize: '0.85rem',
              marginBottom: 16,
            }}
          >
            {error}
          </div>
        )}

        <form onSubmit={handleSubmit}>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12, marginBottom: 16 }}>
            <div>
              <label htmlFor="m-views" style={{ display: 'block', fontSize: '0.8rem', fontWeight: 600, marginBottom: 4 }}>
                Total Views
              </label>
              <input
                id="m-views"
                type="number"
                min="0"
                className="input-field"
                value={views}
                onChange={(e) => setViews(e.target.value)}
                style={{ width: '100%', padding: '8px 12px', borderRadius: 8 }}
                required
              />
            </div>

            <div>
              <label htmlFor="m-likes" style={{ display: 'block', fontSize: '0.8rem', fontWeight: 600, marginBottom: 4 }}>
                Likes
              </label>
              <input
                id="m-likes"
                type="number"
                min="0"
                className="input-field"
                value={likes}
                onChange={(e) => setLikes(e.target.value)}
                style={{ width: '100%', padding: '8px 12px', borderRadius: 8 }}
              />
            </div>

            <div>
              <label htmlFor="m-shares" style={{ display: 'block', fontSize: '0.8rem', fontWeight: 600, marginBottom: 4 }}>
                Shares & Saves
              </label>
              <input
                id="m-shares"
                type="number"
                min="0"
                className="input-field"
                value={shares}
                onChange={(e) => setShares(e.target.value)}
                style={{ width: '100%', padding: '8px 12px', borderRadius: 8 }}
              />
            </div>

            <div>
              <label htmlFor="m-comments" style={{ display: 'block', fontSize: '0.8rem', fontWeight: 600, marginBottom: 4 }}>
                Comments
              </label>
              <input
                id="m-comments"
                type="number"
                min="0"
                className="input-field"
                value={comments}
                onChange={(e) => setComments(e.target.value)}
                style={{ width: '100%', padding: '8px 12px', borderRadius: 8 }}
              />
            </div>
          </div>

          <div style={{ marginBottom: 20 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
              <label htmlFor="m-retention" style={{ fontSize: '0.8rem', fontWeight: 600 }}>
                Audience Retention Rate
              </label>
              <span style={{ fontSize: '0.85rem', fontWeight: 700, color: 'var(--rust, #bb5a3c)' }}>
                {retentionPct === '' ? 'Unknown' : `${retentionPct}%`}
              </span>
            </div>
            <input
              id="m-retention"
              type="number"
              min="0"
              max="100"
              placeholder="Unknown"
              value={retentionPct}
              onChange={(e) => { setRetentionPct(e.target.value); setRetentionTouched(true); }}
              style={{ width: '100%' }}
            />
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.72rem', color: 'var(--ink-dim)', marginTop: 2 }}>
              <span>Low (Dropoff)</span>
              <span>Average (60-70%)</span>
              <span>Viral (80%+)</span>
            </div>
          </div>

          <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 10 }}>
            <Btn variant="ghost" size="sm" onClick={onClose} disabled={busy}>
              Cancel
            </Btn>
            <Btn variant="primary" size="sm" type="submit" loading={busy}>
              Save Metrics
            </Btn>
          </div>
        </form>
      </div>
    </div>
  );

  return typeof document !== 'undefined' ? createPortal(modalNode, document.body) : null;
}

export function AnalyticsView({ pushToast }) {
  const [summary, setSummary] = useState(null);
  const [insights, setInsights] = useState(null);
  const [loading, setLoading] = useState(true);
  const [syncing, setSyncing] = useState(false);
  const [editingClip, setEditingClip] = useState(null);
  const [showPromptDetails, setShowPromptDetails] = useState(false);

  const fetchAnalytics = useCallback(async () => {
    try {
      setLoading(true);
      const [sumData, insData] = await Promise.all([
        getAnalyticsSummary(),
        getAnalyticsInsights().catch(() => ({ patterns: {}, prompt_snippet: '' })),
      ]);
      setSummary(sumData);
      setInsights(insData);
    } catch (err) {
      pushToast?.('error', `Failed to load analytics: ${err.message || err}`);
    } finally {
      setLoading(false);
    }
  }, [pushToast]);

  useEffect(() => {
    fetchAnalytics();
  }, [fetchAnalytics]);

  const handleSyncZernio = async () => {
    setSyncing(true);
    try {
      const res = await syncAnalytics();
      const count = res.synced_count ?? res.synced ?? 0;
      if (count > 0) {
        pushToast?.('success', `Synced ${count} posts from Zernio Analytics API.`);
      } else {
        pushToast?.('info', res.message || 'Zernio sync completed.');
      }
      await fetchAnalytics();
    } catch (err) {
      pushToast?.('error', `Zernio sync error: ${err.message || err}`);
    } finally {
      setSyncing(false);
    }
  };

  const totalViews = summary?.total_views || 0;
  const totalShares = summary?.total_shares || 0;
  const totalLikes = summary?.total_likes || 0;
  const totalComments = summary?.total_comments || 0;
  const totalPublished = summary?.total_published || 0;
  const avgRetention = summary?.avg_retention ? Math.round(summary.avg_retention * 100) : 0;
  const clips = summary?.clips || summary?.all_clips || [];
  const patterns = insights?.patterns || {};
  const hasLearningData = patterns?.has_data;

  return (
    <div className="container" style={{ paddingBottom: 64 }}>
      <Hero
        eyebrow="Performance Intelligence"
        title={<span>Published Clip <em>Analytics & Learning</em></span>}
        desc="Track real-world views, likes, shares, and retention across platforms. Recorded measurements are shown as observed; audience-based selection rules are not active yet."
      />

      {/* Top Action Bar */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 24, flexWrap: 'wrap', gap: 12 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <Badge tone="rust" icon="bar-chart">
            Metrics Tracking
          </Badge>
          <span style={{ fontSize: '0.8rem', color: 'var(--ink-dim)' }}>
            Zernio Analytics API Auto-Sync Enabled
          </span>
        </div>

        <div style={{ display: 'flex', gap: 10 }}>
          <Btn
            variant="secondary"
            size="sm"
            icon="refresh-cw"
            loading={syncing}
            onClick={handleSyncZernio}
            title="Query Zernio Analytics API for live view, like, and share counts across all connected platforms"
          >
            Sync Zernio Analytics
          </Btn>
        </div>
      </div>

      {/* KPI Cards Grid */}
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))',
          gap: 16,
          marginBottom: 28,
        }}
      >
        <Panel pad className="raised" style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', color: 'var(--ink-dim)', fontSize: '0.85rem' }}>
            <span>Published Clips</span>
            <Icon n="scissors" />
          </div>
          <div style={{ fontSize: '2rem', fontWeight: 800, color: 'var(--ink)' }}>
            {totalPublished}
          </div>
          <div style={{ fontSize: '0.75rem', color: 'var(--ink-dim)' }}>
            Tracked across TikTok, IG & YT
          </div>
        </Panel>

        <Panel pad className="raised" style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', color: 'var(--ink-dim)', fontSize: '0.85rem' }}>
            <span>Total Views</span>
            <Icon n="eye" />
          </div>
          <div style={{ fontSize: '2rem', fontWeight: 800, color: 'var(--rust, #bb5a3c)' }}>
            {totalViews.toLocaleString()}
          </div>
          <div style={{ fontSize: '0.75rem', color: 'var(--ink-dim)' }}>
            {totalLikes.toLocaleString()} total likes
          </div>
        </Panel>

        <Panel pad className="raised" style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', color: 'var(--ink-dim)', fontSize: '0.85rem' }}>
            <span>Viral Shares & Saves</span>
            <Icon n="trending-up" />
          </div>
          <div style={{ fontSize: '2rem', fontWeight: 800, color: 'var(--ink)' }}>
            {totalShares.toLocaleString()}
          </div>
          <div style={{ fontSize: '0.75rem', color: 'var(--ink-dim)' }}>
            {totalComments.toLocaleString()} discussion comments
          </div>
        </Panel>

        <Panel pad className="raised" style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', color: 'var(--ink-dim)', fontSize: '0.85rem' }}>
            <span>Avg Retention</span>
            <Icon n="flame" />
          </div>
          <div style={{ fontSize: '2rem', fontWeight: 800, color: 'var(--ink)' }}>
            {avgRetention}%
          </div>
          <div style={{ fontSize: '0.75rem', color: avgRetention >= 75 ? '#2e7d32' : 'var(--ink-dim)' }}>
            {summary?.avg_retention != null ? 'Average of recorded values' : 'No retention recorded yet'}
          </div>
        </Panel>
      </div>

      {/* Continuous Learning Loop Banner */}
      <Panel
        pad
        className="raised"
        style={{
          marginBottom: 28,
          borderLeft: '4px solid var(--rust, #bb5a3c)',
          background: 'linear-gradient(135deg, var(--surface) 0%, rgba(187,90,60,0.05) 100%)',
        }}
      >
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: 12, marginBottom: 12 }}>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4 }}>
              <span style={{ fontSize: '1.05rem', fontWeight: 700, color: 'var(--ink)' }}>
                🧠 Performance Observations
              </span>
              <Badge tone="rust">
                Not Active in Selection
              </Badge>
            </div>
            <p style={{ fontSize: '0.875rem', color: 'var(--ink-dim)', margin: 0, maxWidth: 720, lineHeight: 1.4 }}>
              Recorded measurements from your published clips, shown as observed. Audience-based selection rules are not active yet; generic editorial policy still applies to new clips.</p>
          </div>

          <Btn
            variant="ghost"
            size="sm"
            icon={showPromptDetails ? 'chevron-right' : 'sparkles'}
            onClick={() => setShowPromptDetails(!showPromptDetails)}
          >
            {showPromptDetails ? 'Hide Prompt Injection' : 'View Injected Prompt'}
          </Btn>
        </div>

        {hasLearningData ? (
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: 14, marginTop: 14, paddingTop: 14, borderTop: '1px solid rgba(0,0,0,0.06)' }}>
            <div>
              <span style={{ fontSize: '0.75rem', textTransform: 'uppercase', letterSpacing: 0.5, fontWeight: 700, color: 'var(--ink-dim)', display: 'block', marginBottom: 2 }}>
                Measured Clips
              </span>
              <span style={{ fontSize: '0.95rem', fontWeight: 700, color: 'var(--ink)' }}>
                {patterns.sample_size || 0} clips with recorded views
              </span>
              <span style={{ fontSize: '0.75rem', color: 'var(--ink-dim)', display: 'block' }}>
                Descriptive observations only; no causal rules inferred.
              </span>
            </div>

            <div>
              <span style={{ fontSize: '0.75rem', textTransform: 'uppercase', letterSpacing: 0.5, fontWeight: 700, color: 'var(--ink-dim)', display: 'block', marginBottom: 2 }}>
                Recorded Totals
              </span>
              <span style={{ fontSize: '0.95rem', fontWeight: 700, color: 'var(--ink)' }}>
                {(patterns.totals?.views ?? 0).toLocaleString()} views
              </span>
              <span style={{ fontSize: '0.75rem', color: 'var(--ink-dim)', display: 'block' }}>
                {(patterns.totals?.shares ?? 0).toLocaleString()} shares / {(patterns.totals?.likes ?? 0).toLocaleString()} likes recorded
              </span>
            </div>

            <div>
              <span style={{ fontSize: '0.75rem', textTransform: 'uppercase', letterSpacing: 0.5, fontWeight: 700, color: 'var(--ink-dim)', display: 'block', marginBottom: 2 }}>
                Retention
              </span>
              <span style={{ fontSize: '0.95rem', fontWeight: 700, color: 'var(--ink)' }}>
                {patterns.retention?.average_recorded != null
                  ? `${Math.round(patterns.retention.average_recorded * 100)}% avg`
                  : 'Unknown'}
              </span>
              <span style={{ fontSize: '0.75rem', color: 'var(--ink-dim)', display: 'block' }}>
                {patterns.retention?.recorded_count
                  ? `Average of ${patterns.retention.recorded_count} explicitly recorded values`
                  : 'No retention values recorded yet'}
              </span>
            </div>
          </div>
        ) : (

          <div style={{ fontSize: '0.85rem', color: 'var(--ink-dim)', marginTop: 8 }}>
            Publish clips to gather performance data. Recorded measurements will appear here as observed; audience-based selection rules are not active yet.
          </div>
        )}

        {showPromptDetails && insights?.prompt_snippet && (
          <div style={{ marginTop: 16, paddingTop: 14, borderTop: '1px dashed rgba(0,0,0,0.1)' }}>
            <div style={{ fontSize: '0.75rem', fontWeight: 700, color: 'var(--ink-dim)', marginBottom: 6 }}>
              LIVE PROMPT INJECTION SENT TO GEMINI:
            </div>
            <pre
              style={{
                fontSize: '0.78rem',
                fontFamily: 'monospace',
                background: 'rgba(0,0,0,0.04)',
                padding: '12px 16px',
                borderRadius: 8,
                whiteSpace: 'pre-wrap',
                margin: 0,
                color: 'var(--ink)',
                maxHeight: 180,
                overflowY: 'auto',
              }}
            >
              {insights.prompt_snippet}
            </pre>
          </div>
        )}
      </Panel>

      {/* Published Clips Leaderboard */}
      <Panel
        title="Published Clips & Performance Leaderboard"
        sub="All clips pushed to social channels, sorted by total reach and engagement."
        icon="trending-up"
        pad
        className="raised"
      >
        {loading ? (
          <div style={{ padding: '40px 0', textAlign: 'center', color: 'var(--ink-dim)' }}>
            <Icon n="loader" />
            <div style={{ marginTop: 8 }}>Loading analytics leaderboard…</div>
          </div>
        ) : clips.length === 0 ? (
          <div style={{ padding: '40px 0', textAlign: 'center', color: 'var(--ink-dim)' }}>
            <Icon n="film" style={{ fontSize: 32, opacity: 0.5, marginBottom: 8 }} />
            <div style={{ fontWeight: 600, marginBottom: 4 }}>No published clips tracked yet</div>
            <div style={{ fontSize: '0.85rem' }}>
              Publish clips from the Results tab or sync from Zernio to populate your performance dashboard.
            </div>
          </div>
        ) : (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
            {clips.map((clip) => {
              const platforms = clip.platforms || (clip.platform ? [clip.platform] : ['tiktok']);
              const durationFormatted = clip.duration ? `${clip.duration.toFixed(1)}s` : '—';
              const rawRetention = clip.metrics?.retention_rate ?? clip.retention_rate;
              const retentionKnown = clip.metrics?.retention_recorded ?? rawRetention != null;
              const retentionDisplay = (retentionKnown && rawRetention != null)
                              ? `${Math.round(rawRetention * 100)}%` : 'Unknown';
              const viewsCount = (clip.metrics?.views ?? clip.views) || 0;
              const likesCount = (clip.metrics?.likes ?? clip.likes) || 0;
              const sharesCount = (clip.metrics?.shares ?? clip.shares) || 0;
              const tierLabel = clip.narrative_tier || clip.duration_tier;

              return (
                <div
                  key={clip.clip_id}
                  style={{
                    display: 'grid',
                    gridTemplateColumns: 'minmax(280px, 2fr) minmax(100px, 1fr) minmax(180px, 1.5fr) auto',
                    gap: 16,
                    alignItems: 'center',
                    padding: '14px 16px',
                    borderRadius: 12,
                    background: 'var(--surface)',
                    boxShadow: 'var(--clay-raised-sm, 0 2px 6px rgba(0,0,0,0.06))',
                    border: '1px solid rgba(0,0,0,0.05)',
                  }}
                >
                  {/* Clip Info */}
                  <div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4 }}>
                      <span style={{ fontWeight: 700, fontSize: '0.95rem', color: 'var(--ink)' }}>
                        {clip.title}
                      </span>
                      {tierLabel && (
                        <span
                          style={{
                            fontSize: '0.7rem',
                            fontWeight: 700,
                            padding: '2px 6px',
                            borderRadius: 4,
                            background: 'rgba(187,90,60,0.12)',
                            color: 'var(--rust, #bb5a3c)',
                          }}
                        >
                          {tierLabel}
                        </span>
                      )}
                    </div>
                    <div style={{ fontSize: '0.78rem', color: 'var(--ink-dim)', display: 'flex', alignItems: 'center', gap: 10 }}>
                      <span>Duration: <strong>{durationFormatted}</strong></span>
                      <span>·</span>
                      <span style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                        Destinations:
                        {platforms.map((p) => (
                          <Social key={p} n={p} size={13} color="var(--ink-dim)" />
                        ))}
                      </span>
                    </div>
                  </div>

                  {/* Retention Rate */}
                  <div>
                    <div style={{ fontSize: '0.72rem', textTransform: 'uppercase', color: 'var(--ink-dim)', fontWeight: 600 }}>
                      Retention
                    </div>
                    <div style={{ fontSize: '1.1rem', fontWeight: 800, color: (rawRetention || 0) >= 0.8 ? '#2e7d32' : 'var(--ink)' }}>
                      {retentionDisplay}
                    </div>
                  </div>

                  {/* Reach & Engagement Stats */}
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 8, textAlign: 'center' }}>
                    <div>
                      <div style={{ fontSize: '0.7rem', color: 'var(--ink-dim)' }}>Views</div>
                      <div style={{ fontWeight: 700, fontSize: '0.9rem' }}>{viewsCount.toLocaleString()}</div>
                    </div>
                    <div>
                      <div style={{ fontSize: '0.7rem', color: 'var(--ink-dim)' }}>Likes</div>
                      <div style={{ fontWeight: 700, fontSize: '0.9rem' }}>{likesCount.toLocaleString()}</div>
                    </div>
                    <div>
                      <div style={{ fontSize: '0.7rem', color: 'var(--ink-dim)' }}>Shares</div>
                      <div style={{ fontWeight: 700, fontSize: '0.9rem' }}>{sharesCount.toLocaleString()}</div>
                    </div>
                  </div>

                  {/* Actions */}
                  <div>
                    <Btn
                      variant="ghost"
                      size="sm"
                      icon="sliders-horizontal"
                      onClick={() => setEditingClip(clip)}
                      title="Adjust views, likes, or retention manually"
                    >
                      Update
                    </Btn>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </Panel>

      {/* Edit Metrics Modal */}
      {editingClip && (
        <UpdateMetricsModal
          clip={editingClip}
          onClose={() => setEditingClip(null)}
          onSaved={fetchAnalytics}
          pushToast={pushToast}
        />
      )}
    </div>
  );
}
