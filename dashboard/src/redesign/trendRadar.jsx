// ClippyMe redesign — Trend Radar: AI-powered US trend discovery and 1-Click video sourcing.
// Aggregates real-time news & search spikes, analyzes virality angles with Gemini,
// and pairs topics with primary long-form YouTube source videos for instant 1-click clipping.
import { useState, useEffect, useMemo, useCallback } from 'react';
import { Hero } from './chrome';
import { Icon, Btn, Badge, Panel, Segmented } from './primitives';
import { getTrends, triggerTrendScan, clipTrendVideo, getChannels } from './realApi';
import { relTime } from '../lib/relTime';

const CATEGORIES = [
  { id: 'all', label: 'All Topics' },
  { id: 'politics', label: 'US Politics' },
  { id: 'breaking_world', label: 'Breaking & World' },
  { id: 'entertainment', label: 'Entertainment' },
  { id: 'culture', label: 'Culture' },
];

function ViralityPill({ score = 75 }) {
  const isUltra = score >= 90;
  const isHigh = score >= 80;
  const tone = isUltra ? 'danger' : isHigh ? 'warn' : 'accent';
  return (
    <span className={`badge badge-${tone}`} style={{ fontWeight: 700, gap: 4 }}>
      <Icon n="flame" />
      <span>{score}% Viral Potential</span>
    </span>
  );
}

function TrendCard({
  topic,
  onClip,
  onCustomize,
  clippingId,
  channels = [],
  selectedChannelId,
  onSelectChannel,
}) {
  const isClipping = clippingId === topic.id;
  const video = topic.matched_video;
  const categoryLabel = CATEGORIES.find((c) => c.id === topic.category)?.label || topic.category || 'Trending';

  return (
    <div className="panel raised-sm" style={{ display: 'flex', flexDirection: 'column', gap: 14, padding: 18, borderRadius: 16 }}>
      {/* Header with Category & Virality Score */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 8, flexWrap: 'wrap' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <Badge tone="dim">{categoryLabel}</Badge>
          {topic.clipped && <Badge tone="accent" icon="check">Clipped</Badge>}
        </div>
        <ViralityPill score={topic.virality_score} />
      </div>

      {/* Topic Headline */}
      <h3 style={{ fontSize: '1.15rem', fontWeight: 600, lineHeight: 1.35, margin: 0 }}>
        {topic.topic_title}
      </h3>

      {/* AI Viral Hook Angle Callout */}
      {topic.viral_hook && (
        <div
          style={{
            background: 'var(--surface-deep, rgba(0,0,0,0.15))',
            borderRadius: 10,
            padding: '10px 12px',
            borderLeft: '3px solid var(--accent, #02C5BF)',
            fontSize: '0.85rem',
            lineHeight: 1.45,
            color: 'var(--ink-dim)',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: 5, fontWeight: 700, color: 'var(--ink)', marginBottom: 3 }}>
            <Icon n="sparkles" />
            <span>Viral Hook Angle</span>
          </div>
          <div>{topic.viral_hook}</div>
        </div>
      )}

      {/* Matched YouTube Long-Form Video Preview */}
      {video ? (
        <div
          style={{
            display: 'flex',
            gap: 12,
            background: 'var(--surface, rgba(255,255,255,0.02))',
            border: '1px solid var(--border-dim, rgba(255,255,255,0.06))',
            borderRadius: 12,
            overflow: 'hidden',
            padding: 8,
          }}
        >
          {/* Thumbnail Container */}
          <div style={{ position: 'relative', width: 128, minWidth: 128, height: 74, borderRadius: 8, overflow: 'hidden', background: '#000' }}>
            <img
              src={video.thumbnail_url}
              alt={video.title}
              style={{ width: '100%', height: '100%', objectFit: 'cover' }}
              loading="lazy"
            />
            {video.duration_string && (
              <span
                style={{
                  position: 'absolute',
                  bottom: 4,
                  right: 4,
                  background: 'rgba(0,0,0,0.85)',
                  color: '#fff',
                  fontSize: '0.68rem',
                  fontWeight: 700,
                  padding: '2px 4px',
                  borderRadius: 4,
                  fontFamily: 'var(--font-mono)',
                }}
              >
                {video.duration_string}
              </span>
            )}
          </div>

          {/* Video Metadata */}
          <div style={{ display: 'flex', flexDirection: 'column', justifyContent: 'center', minWidth: 0, flex: 1, gap: 4 }}>
            <a
              href={video.video_url}
              target="_blank"
              rel="noopener noreferrer"
              title={video.title}
              style={{
                fontSize: '0.86rem',
                fontWeight: 600,
                color: 'var(--ink)',
                textDecoration: 'none',
                overflow: 'hidden',
                textOverflow: 'ellipsis',
                display: '-webkit-box',
                WebkitLineClamp: 2,
                WebkitBoxOrient: 'vertical',
                lineHeight: 1.25,
              }}
            >
              {video.title}
            </a>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: '0.75rem', color: 'var(--ink-dim)' }}>
              <span style={{ fontWeight: 600, color: 'var(--rust)' }}>{video.channel}</span>
              {video.view_count > 0 && (
                <span>• {video.view_count.toLocaleString()} views</span>
              )}
            </div>
          </div>
        </div>
      ) : (
        <div style={{ fontSize: '0.8rem', color: 'var(--ink-dim)', fontStyle: 'italic' }}>
          Searching for source footage...
        </div>
      )}

      {/* Target Channel Routing Selector */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          gap: 8,
          background: 'var(--surface-deep, rgba(0,0,0,0.12))',
          padding: '8px 12px',
          borderRadius: 10,
          fontSize: '0.8rem',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: 6, color: 'var(--ink-dim)', fontWeight: 600 }}>
          <Icon n="tv" style={{ width: 14, height: 14, color: 'var(--accent, #02C5BF)' }} />
          <span>Target Channel:</span>
        </div>
        {channels.length > 0 ? (
          <select
            className="input-field"
            aria-label="Target Channel"
            value={selectedChannelId || ''}
            onChange={(e) => onSelectChannel?.(topic.id, e.target.value)}
            style={{
              padding: '4px 8px',
              fontSize: '0.78rem',
              fontWeight: 600,
              borderRadius: 6,
              background: 'var(--surface)',
              border: '1px solid var(--border-dim, rgba(255,255,255,0.1))',
              color: 'var(--ink)',
              maxWidth: 220,
            }}
          >
            {channels.map((ch) => (
              <option key={ch.id} value={ch.id}>
                {ch.name} ({ch.banner_handle || ch.id})
              </option>
            ))}
          </select>
        ) : (
          <span style={{ color: 'var(--ink-dim)', fontStyle: 'italic' }}>Auto-routed</span>
        )}
      </div>

      {/* Action Buttons */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginTop: 'auto', paddingTop: 6 }}>
        <Btn
          variant="primary"
          size="sm"
          icon={topic.clipped ? 'check' : 'wand-sparkles'}
          loading={isClipping}
          disabled={!video || isClipping}
          onClick={() => onClip(topic, selectedChannelId)}
          style={{ flex: 1 }}
        >
          {topic.clipped ? 'Clip Again' : '⚡ 1-Click Clip'}
        </Btn>

        {video && (
          <Btn
            variant="secondary"
            size="sm"
            icon="sliders-horizontal"
            onClick={() => onCustomize(topic)}
            title="Open in Create tab to adjust styling preset before clipping"
          >
            Customize
          </Btn>
        )}
      </div>
    </div>
  );
}

export function TrendRadarView({
  apiKey = '',
  onStartClip,
  onCustomizeClip,
  onGoToChannels,
  pushToast,
}) {
  const [topics, setTopics] = useState([]);
  const [channels, setChannels] = useState([]);
  const [topicChannelMap, setTopicChannelMap] = useState({});
  const [lastScanned, setLastScanned] = useState(null);
  const [loading, setLoading] = useState(true);
  const [scanning, setScanning] = useState(false);
  const [category, setCategory] = useState('all');
  const [clippingId, setClippingId] = useState(null);

  const fetchTrendsList = useCallback(async () => {
    try {
      setLoading(true);
      const data = await getTrends({ category });
      setTopics(data.topics || []);
      setLastScanned(data.last_scanned);
    } catch (err) {
      pushToast?.('error', `Failed to load trends: ${err.message || err}`);
    } finally {
      setLoading(false);
    }
  }, [category, pushToast]);

  const fetchChannelsData = useCallback(async () => {
    try {
      const data = await getChannels();
      setChannels(data.channels || []);
    } catch {
      // Non-critical fallback
    }
  }, []);

  useEffect(() => {
    fetchTrendsList();
  }, [fetchTrendsList]);

  useEffect(() => {
    fetchChannelsData();
  }, [fetchChannelsData]);

  const resolveChannelId = useCallback((topic) => {
    if (topicChannelMap[topic.id]) return topicChannelMap[topic.id];
    if (!channels.length) return null;
    const cat = (topic.category || '').toLowerCase();
    const matched = channels.find((c) => (c.niches || []).some((n) => n.toLowerCase() === cat));
    return (matched || channels[0])?.id || null;
  }, [topicChannelMap, channels]);

  const handleScanNow = async () => {
    if (scanning) return;
    setScanning(true);
    pushToast?.('info', 'Scanning live US headlines and search trends with Gemini...');
    try {
      const data = await triggerTrendScan(apiKey);
      setTopics(data.topics || []);
      setLastScanned(data.last_scanned);
      pushToast?.('success', `Trend scan complete: found ${data.topics_count || (data.topics || []).length} trending topics!`);
    } catch (err) {
      pushToast?.('error', `Trend scan failed: ${err.message || err}`);
    } finally {
      setScanning(false);
    }
  };

  const handleOneClickClip = async (topic, chosenChannelId) => {
    const videoUrl = topic.matched_video?.video_url;
    if (!videoUrl) return;

    setClippingId(topic.id);
    try {
      const channelId = chosenChannelId || resolveChannelId(topic);
      const res = await clipTrendVideo({
        topicId: topic.id,
        channelId,
        videoUrl,
        instructions: `Trending topic: ${topic.topic_title}. ${topic.viral_hook || ''}`,
        presetId: topic.suggested_preset || 'viral',
      }, apiKey);

      pushToast?.('success', `Video enqueued for clipping! Job ID: ${res.job_id.slice(0, 8)}`);

      // Update topic card to clipped in state
      setTopics((prev) =>
        prev.map((t) => (t.id === topic.id ? { ...t, clipped: true, job_id: res.job_id } : t))
      );

      if (onStartClip) {
        onStartClip(res.job_id, videoUrl);
      }
    } catch (err) {
      pushToast?.('error', `Failed to clip video: ${err.message || err}`);
    } finally {
      setClippingId(null);
    }
  };

  const handleCustomize = (topic) => {
    const videoUrl = topic.matched_video?.video_url;
    if (!videoUrl) return;
    if (onCustomizeClip) {
      onCustomizeClip({
        url: videoUrl,
        instructions: `Trending topic: ${topic.topic_title}. ${topic.viral_hook || ''}`,
        preset: topic.suggested_preset || 'viral',
      });
    }
  };

  const filteredTopics = useMemo(() => {
    if (category === 'all') return topics;
    return topics.filter((t) => (t.category || '').toLowerCase() === category.toLowerCase());
  }, [topics, category]);

  return (
    <div className="container" style={{ paddingBottom: 64 }}>
      <Hero
        eyebrow="Trend Radar"
        title={<span>Live <em>Trend Radar</em></span>}
        desc="Real-time US breaking news, politics, and viral stories paired with high-potential YouTube long-form videos for instant 1-click clipping."
      />

      {/* Toolbar */}
      <div
        className="panel raised-sm"
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          gap: 16,
          padding: '12px 18px',
          borderRadius: 14,
          marginBottom: 24,
          flexWrap: 'wrap',
        }}
      >
        {/* Category Filter */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 6, overflowX: 'auto', paddingBottom: 2 }}>
          {CATEGORIES.map((cat) => (
            <button
              key={cat.id}
              type="button"
              className={`btn btn-sm ${category === cat.id ? 'btn-primary' : 'btn-ghost'}`}
              onClick={() => setCategory(cat.id)}
              style={{ borderRadius: 20 }}
            >
              {cat.label}
            </button>
          ))}
        </div>

        {/* Right Status & Actions */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          {lastScanned && (
            <span style={{ fontSize: '0.78rem', color: 'var(--ink-dim)' }}>
              Updated {relTime(lastScanned)}
            </span>
          )}

          {onGoToChannels && (
            <Btn
              variant="ghost"
              size="sm"
              icon="tv"
              onClick={onGoToChannels}
              title="Manage Channel Profiles"
            >
              Channels ({channels.length})
            </Btn>
          )}

          <Btn
            variant="secondary"
            size="sm"
            icon="refresh-cw"
            loading={scanning}
            onClick={handleScanNow}
          >
            {scanning ? 'Scanning...' : 'Scan Trends Now'}
          </Btn>
        </div>
      </div>

      {/* Content Feed */}
      {loading && topics.length === 0 ? (
        <div className="panel raised-sm" style={{ padding: 48, textAlign: 'center', borderRadius: 16 }}>
          <div style={{ display: 'inline-flex', marginBottom: 12 }}>
            <Icon n="loader" cls="ico-spin" style={{ width: 28, height: 28, color: 'var(--rust)' }} />
          </div>
          <p style={{ margin: 0, color: 'var(--ink-dim)' }}>Scanning US news feeds & search trends...</p>
        </div>
      ) : filteredTopics.length === 0 ? (
        <div className="panel raised-sm" style={{ padding: 56, textAlign: 'center', borderRadius: 16 }}>
          <div style={{ display: 'inline-flex', marginBottom: 16, color: 'var(--ink-dim)' }}>
            <Icon n="flame" style={{ width: 42, height: 42 }} />
          </div>
          <h3 style={{ margin: '0 0 8px', fontSize: '1.2rem', fontWeight: 600 }}>No trending topics found</h3>
          <p style={{ margin: '0 0 20px', color: 'var(--ink-dim)', maxWidth: 420, marginLeft: 'auto', marginRight: 'auto' }}>
            {category !== 'all'
              ? `No current stories match the "${category}" category. Try selecting "All Topics" or run a fresh scan.`
              : 'Trigger a scan to research breaking US politics, world news, and viral culture stories.'}
          </p>
          <Btn variant="primary" icon="refresh-cw" loading={scanning} onClick={handleScanNow}>
            Scan US Trends Now
          </Btn>
        </div>
      ) : (
        <div
          style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fill, minmax(360px, 1fr))',
            gap: 20,
          }}
        >
          {filteredTopics.map((topic) => (
            <TrendCard
              key={topic.id}
              topic={topic}
              channels={channels}
              selectedChannelId={resolveChannelId(topic)}
              onSelectChannel={(tId, chId) =>
                setTopicChannelMap((prev) => ({ ...prev, [tId]: chId }))
              }
              onClip={handleOneClickClip}
              onCustomize={handleCustomize}
              clippingId={clippingId}
            />
          ))}
        </div>
      )}
    </div>
  );
}
