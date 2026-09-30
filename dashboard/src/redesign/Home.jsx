import { Icon, Social } from './icon';
import { Btn, Badge } from './primitives';
import { WhatsNew } from './WhatsNew';

/* ------------------------------------------------------------------
   Home — the logged-in dashboard for Nugget.
   Default landing tab inside the app: greeting, stats, resume,
   top clips, publishing queue. Pure presentational component —
   every button navigates or acts, no dead ends.
   ------------------------------------------------------------------ */

/* ---------- small helpers ---------- */

function asArray(v) {
  return Array.isArray(v) ? v : [];
}

function asObject(v) {
  return v && typeof v === 'object' ? v : {};
}

function toTime(v) {
  if (v == null) return NaN;
  const t = new Date(v).getTime();
  return Number.isNaN(t) ? NaN : t;
}

/** "2h ago", "yesterday", "3d ago" — and future variants "in 3h". */
function relTime(ts) {
  const t = toTime(ts);
  if (Number.isNaN(t)) return '';
  const now = Date.now();
  const diff = now - t;
  const abs = Math.abs(diff);

  const min = Math.floor(abs / 60000);
  const hr = Math.floor(abs / 3600000);
  const day = Math.floor(abs / 86400000);

  if (min < 1) return diff >= 0 ? 'just now' : 'any moment now';
  if (diff < 0) {
    if (min < 60) return `in ${min}m`;
    if (hr < 24) return `in ${hr}h`;
    if (day === 1) return 'tomorrow';
    if (day < 7) return `in ${day}d`;
    return `in ${Math.floor(day / 7)}w`;
  }
  if (min < 60) return `${min}m ago`;
  if (hr < 24) return `${hr}h ago`;
  if (day === 1) return 'yesterday';
  if (day < 7) return `${day}d ago`;
  if (day < 30) return `${Math.floor(day / 7)}w ago`;
  return `${Math.floor(day / 30)}mo ago`;
}

/** "1:23" from seconds (or passthrough when already a string). */
function fmtDuration(d) {
  if (d == null || d === '') return '';
  if (typeof d === 'string') return d;
  const n = Number(d);
  if (Number.isNaN(n)) return '';
  const m = Math.floor(n / 60);
  const s = Math.floor(n % 60);
  return `${m}:${String(s).padStart(2, '0')}`;
}

function firstName(user) {
  if (!user || typeof user.name !== 'string') return 'creator';
  const f = user.name.trim().split(/\s+/)[0];
  return f || 'creator';
}

/** Human title for the live job from processingMedia. */
function mediaTitle(media) {
  if (!media) return 'Your video';
  const p = media.payload;
  if (typeof p === 'string') {
    const s = p.trim();
    if (!s) return 'Your video';
    if (/^https?:\/\//i.test(s)) {
      try {
        return new URL(s).hostname.replace(/^www\./, '');
      } catch {
        return s.length > 42 ? `${s.slice(0, 42)}…` : s;
      }
    }
    return s.length > 42 ? `${s.slice(0, 42)}…` : s;
  }
  if (p && typeof p === 'object' && typeof p.name === 'string' && p.name) return p.name;
  return 'Your video';
}

/* ============================== component ============================== */

export function Home(props) {
  const {
    user,
    history,
    clips,
    clipStates,
    status,
    processingMedia,
    loading,
    goTab,
    onContinueJob,
    onResumeHistoryJob,
  } = props || {};

  const go = (tab) => {
    if (typeof goTab === 'function') goTab(tab);
  };
  const continueJob = () => {
    if (typeof onContinueJob === 'function') onContinueJob();
    else go('create');
  };
  const resumeJob = (job) => {
    if (typeof onResumeHistoryJob === 'function') onResumeHistoryJob(job);
    else go('create');
  };

  const hist = asArray(history);
  const clipList = asArray(clips);
  const states = asObject(clipStates);
  const stateEntries = Object.entries(states);

  /* ---------------- loading skeletons ---------------- */
  if (loading) {
    return (
      <div className="hm" aria-busy="true" aria-label="Loading dashboard">
        <div className="hm-greet-row">
          <div className="hm-greet">
            <div className="sk" style={{ width: 280, height: 36, borderRadius: 12 }} />
            <div className="sk" style={{ width: 220, height: 16, borderRadius: 8, marginTop: 8 }} />
          </div>
          <div className="hm-actions">
            <div className="sk" style={{ width: 130, height: 42, borderRadius: 999 }} />
            <div className="sk" style={{ width: 130, height: 42, borderRadius: 999 }} />
          </div>
        </div>
        <div className="hm-stats">
          {[0, 1, 2, 3].map((i) => (
            <div key={i} className="sk" style={{ height: 118 }} />
          ))}
        </div>
        <div className="hm-section">
          <div className="sk" style={{ width: 220, height: 24, borderRadius: 8, marginBottom: 14 }} />
          <div className="hm-clips">
            {[0, 1, 2].map((i) => (
              <div key={i} className="sk" style={{ aspectRatio: '16 / 12' }} />
            ))}
          </div>
        </div>
      </div>
    );
  }

  /* ---------------- derived data ---------------- */
  const weekAgo = Date.now() - 7 * 86400000;
  const recentComplete = hist.filter(
    (h) => h && h.status === 'complete' && toTime(h.createdAt) >= weekAgo
  );
  const clipsThisWeek =
    clipList.length +
    recentComplete.reduce(
      (sum, h) => sum + (Number(h.totalClips) > 0 ? Number(h.totalClips) : 0),
      0
    );

  const scores = clipList
    .map((c) => Number(c && c.viral_score))
    .filter((n) => Number.isFinite(n));
  const avgScore = scores.length ? Math.round(scores.reduce((a, b) => a + b, 0) / scores.length) : null;
  const maxScore = scores.length ? Math.round(Math.max(...scores)) : null;

  const publishedCount = stateEntries.filter(([, s]) => s && s.publishedAt).length;

  const stuckJobs = hist.filter((h) => h && h.status === 'processing');
  const inProgress = (status === 'processing' ? 1 : 0) + stuckJobs.length;

  const hour = new Date().getHours();
  const daypart = hour < 12 ? 'morning' : hour < 17 ? 'afternoon' : 'evening';
  const name = firstName(user);

  const subline =
    status === 'processing'
      ? 'Your clips are rendering — grab a coffee.'
      : stuckJobs.length > 0
        ? 'A render is still running in the background.'
        : 'Ready when you are — turn a long video into clips.';

  const topClips = clipList
    .slice()
    .sort((a, b) => (Number(b.viral_score) || 0) - (Number(a.viral_score) || 0))
    .slice(0, 3);

  const clipKey = (c, i) =>
    String((c && (c.key || c.id || c.clipId)) ?? `clip-${i}`);

  const publishedQueue = stateEntries
    .filter(([, s]) => s && s.publishedAt)
    .sort(([, a], [, b]) => toTime(b.publishedAt) - toTime(a.publishedAt))
    .slice(0, 5);
  const scheduledQueue = stateEntries
    .filter(([, s]) => s && !s.publishedAt && s.scheduledAt)
    .sort(([, a], [, b]) => toTime(a.scheduledAt) - toTime(b.scheduledAt));

  const queueTitle = (key, state) => {
    if (state && state.title) return state.title;
    const match = clipList.find((c) => clipKey(c, -1) === key);
    if (match && match.title) return match.title;
    const idx = clipList.findIndex((c) => clipKey(c, -1) === key);
    return idx >= 0 ? `Clip ${idx + 1}` : 'Clip';
  };

  const clipPlatforms = (c, i) => {
    const key = clipKey(c, i);
    const s = states[key];
    const plats = [];
    if (s && s.platform) plats.push(s.platform);
    return plats;
  };

  const showContinue = status === 'processing' || stuckJobs.length > 0;

  /* ---------------- render ---------------- */
  return (
    <div className="hm">
      <WhatsNew />

      {/* 1. Greeting + quick actions */}
      <div className="hm-greet-row">
        <div className="hm-greet">
          <h1>Good {daypart}, {name}.</h1>
          <p>{subline}</p>
        </div>
        <div className="hm-actions">
          <Btn variant="primary" icon="plus" onClick={() => go('create')}>
            New clip
          </Btn>
          <Btn variant="ghost" icon="flame" onClick={() => go('trends')}>
            Trend radar
          </Btn>
          <Btn variant="ghost" icon="scissors" onClick={() => go('clips')}>
            My clips
          </Btn>
        </div>
      </div>

      {/* 2. Stats */}
      <div className="hm-stats" role="group" aria-label="This week at a glance">
        <button
          type="button"
          className="hm-stat"
          style={{ '--stat-accent': 'var(--rust)' }}
          onClick={() => go('clips')}
          aria-label={`Clips this week: ${clipsThisWeek.toLocaleString()}. Go to My clips.`}
        >
          <span className="hm-stat-top">
            <Icon n="scissors" aria-hidden="true" />
            Clips this week
          </span>
          <span className="hm-stat-val">{clipsThisWeek.toLocaleString()}</span>
          <span className="hm-stat-sub">fresh moments</span>
        </button>

        <button
          type="button"
          className="hm-stat"
          style={{ '--stat-accent': 'var(--gold)' }}
          onClick={() => go('clips')}
          aria-label={`Average virality: ${avgScore == null ? 'no scores yet' : avgScore}. Go to My clips.`}
        >
          <span className="hm-stat-top">
            <Icon n="trending-up" aria-hidden="true" />
            Avg. virality
          </span>
          <span className="hm-stat-val">{avgScore == null ? '—' : avgScore.toLocaleString()}</span>
          <span className="hm-stat-sub">
            {maxScore == null ? 'no scores yet' : `top clip ${maxScore.toLocaleString()}`}
          </span>
        </button>

        <button
          type="button"
          className="hm-stat"
          style={{ '--stat-accent': 'var(--sage)' }}
          onClick={() => go('clips')}
          aria-label={`Published: ${publishedCount.toLocaleString()}. Go to My clips.`}
        >
          <span className="hm-stat-top">
            <Icon n="send" aria-hidden="true" />
            Published
          </span>
          <span className="hm-stat-val">{publishedCount.toLocaleString()}</span>
          <span className="hm-stat-sub">across platforms</span>
        </button>

        <button
          type="button"
          className="hm-stat"
          style={{ '--stat-accent': 'var(--plum)' }}
          onClick={() => go('create')}
          aria-label={`${inProgress.toLocaleString()} jobs in progress. Go to create.`}
        >
          <span className="hm-stat-top">
            <Icon n="clock" aria-hidden="true" />
            In progress
          </span>
          <span className="hm-stat-val">{inProgress.toLocaleString()}</span>
          <span className="hm-stat-sub">rendering now</span>
        </button>
      </div>

      {/* 3. Continue where you left off */}
      {showContinue && (
        <section className="hm-section" aria-labelledby="hm-continue-h">
          <div className="hm-section-head">
            <h2 id="hm-continue-h">Continue where you left off</h2>
          </div>

          {status === 'processing' && (
            <button
              type="button"
              className="hm-job"
              onClick={continueJob}
              aria-label={`Continue rendering ${mediaTitle(processingMedia)}`}
            >
              <span className="hm-job-ic" style={{ position: 'relative' }}>
                <Icon n="scissors" aria-hidden="true" />
                <span
                  className="ajb-dot"
                  style={{ position: 'absolute', top: 7, right: 7 }}
                  aria-hidden="true"
                />
              </span>
              <span className="hm-job-body">
                <span className="hm-job-title">{mediaTitle(processingMedia)}</span>
                <span className="hm-job-sub">Rendering in the background</span>
                <span className="hm-progress" aria-hidden="true">
                  <i style={{ width: '60%' }} />
                </span>
              </span>
              <span className="hm-job-go" aria-hidden="true">
                <Icon n="chevron-right" />
              </span>
            </button>
          )}

          {stuckJobs.map((job, i) => (
            <button
              type="button"
              key={(job && job.jobId) || `stuck-${i}`}
              className="hm-job"
              onClick={() => resumeJob(job)}
              aria-label={`Resume render: ${job && job.title ? job.title : 'untitled job'}`}
            >
              <span className="hm-job-ic">
                <Icon n="clock" aria-hidden="true" />
              </span>
              <span className="hm-job-body">
                <span className="hm-job-title">
                  {job && job.title ? job.title : 'Untitled job'}
                </span>
                <span className="hm-job-sub">Paused render — pick up where you left off</span>
                <span className="hm-progress" aria-hidden="true">
                  <i style={{ width: '45%' }} />
                </span>
              </span>
              <span className="hm-job-go" aria-hidden="true">
                <Icon n="chevron-right" />
              </span>
            </button>
          ))}
        </section>
      )}

      {/* 4. Top clips */}
      <section className="hm-section" aria-labelledby="hm-top-h">
        <div className="hm-section-head">
          <h2 id="hm-top-h">Top clips</h2>
          {topClips.length > 0 && (
            <button
              type="button"
              className="hm-link"
              onClick={() => go('clips')}
              aria-label="View all clips"
            >
              View all <Icon n="chevron-right" aria-hidden="true" />
            </button>
          )}
        </div>

        {topClips.length === 0 ? (
          <div className="empty">
            <div className="empty-ic" aria-hidden="true">
              <Icon n="scissors" />
            </div>
            <h3>No clips yet</h3>
            <p>Paste a link or upload a video and Nugget will cut the moments worth keeping.</p>
            <Btn variant="primary" icon="plus" onClick={() => go('create')}>
              Create your first clip
            </Btn>
          </div>
        ) : (
          <div className="hm-clips">
            {topClips.map((clip, i) => {
              const title = (clip && clip.title) || 'Untitled clip';
              const score = Number(clip && clip.viral_score);
              const dur = fmtDuration(clip && clip.duration);
              const plats = clipPlatforms(clip, i);
              return (
                <button
                  type="button"
                  key={clipKey(clip, i)}
                  className="hm-clip"
                  onClick={() => go('clips')}
                  aria-label={`Open ${title} in My clips`}
                >
                  {clip && clip.thumb ? (
                    <span className="hm-clip-thumb">
                      <img src={clip.thumb} alt={title} loading="lazy" />
                      <span className="hm-clip-score">
                        <Badge tone="accent">{Number.isFinite(score) ? score : '—'}</Badge>
                      </span>
                      <span className="hm-clip-play" aria-hidden="true">
                        <span>
                          <Icon n="play" />
                        </span>
                      </span>
                    </span>
                  ) : (
                    <span
                      className="hm-clip-thumb"
                      style={{
                        background:
                          'linear-gradient(135deg, var(--surface), var(--surface-deep))',
                      }}
                    >
                      <span className="hm-clip-score">
                        <Badge tone="accent">{Number.isFinite(score) ? score : '—'}</Badge>
                      </span>
                      <span className="hm-clip-play" aria-hidden="true">
                        <span>
                          <Icon n="play" />
                        </span>
                      </span>
                    </span>
                  )}
                  <span className="hm-clip-body">
                    <span className="hm-clip-title">{title}</span>
                    <span className="hm-clip-meta">
                      {dur && <span>{dur}</span>}
                      {plats.map((p) => (
                        <Social key={p} n={p} size={13} color="var(--ink-dim)" />
                      ))}
                    </span>
                  </span>
                </button>
              );
            })}
          </div>
        )}
      </section>

      {/* 5. Publishing queue */}
      <section className="hm-section" aria-labelledby="hm-queue-h">
        <div className="hm-section-head">
          <div>
            <h2 id="hm-queue-h">Publishing queue</h2>
            {publishedQueue.length > 0 && (
              <p
                style={{
                  margin: '4px 0 0',
                  fontSize: 13,
                  color: 'var(--ink-dim)',
                }}
              >
                Latest publish activity across your connected channels.
              </p>
            )}
          </div>
        </div>

        {publishedQueue.length === 0 && scheduledQueue.length === 0 ? (
          <div className="empty">
            <div className="empty-ic" aria-hidden="true">
              <Icon n="send" />
            </div>
            <h3>Nothing published yet</h3>
            <p>When you publish a clip — or schedule one — it&apos;ll show up here.</p>
            <Btn variant="ghost" icon="scissors" onClick={() => go('clips')}>
              Browse clips
            </Btn>
          </div>
        ) : (
          <div>
            {publishedQueue.map(([key, s], i) => (
              <div className="hm-queue-item" key={key || `pub-${i}`}>
                <span className="hm-q-ic" aria-hidden="true">
                  {s && s.platform ? (
                    <Social n={s.platform} size={17} color="var(--sage)" />
                  ) : (
                    <Icon n="send" />
                  )}
                </span>
                <span className="hm-queue-body">
                  <span className="hm-queue-title">{queueTitle(key, s, i)}</span>
                  <span className="hm-queue-sub">
                    Published{s && s.platform ? ` to ${s.platform}` : ''}
                  </span>
                </span>
                <span className="hm-queue-when">{relTime(s.publishedAt)}</span>
              </div>
            ))}
            {scheduledQueue.map(([key, s], i) => (
              <div className="hm-queue-item" key={key || `sched-${i}`}>
                <span className="hm-q-ic" aria-hidden="true">
                  <Icon n="calendar-clock" />
                </span>
                <span className="hm-queue-body">
                  <span className="hm-queue-title">{queueTitle(key, s, i)}</span>
                  <span className="hm-queue-sub">
                    Scheduled{s && s.platform ? ` for ${s.platform}` : ''}
                  </span>
                </span>
                <span className="hm-queue-when">{relTime(s.scheduledAt)}</span>
              </div>
            ))}
          </div>
        )}
      </section>
    </div>
  );
}
