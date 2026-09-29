// Nugget — Projects tab. Groups finished jobs by their source video so each
// video you clipped from has one home. Derived purely from the localStorage
// history list (real jobs only — never fabricated).
import { Icon, Btn, Badge } from './primitives';
import { Hero } from './chrome';
import { relTime } from '../lib/relTime';

/** Normalize a source string so re-runs of the same video group together. */
function normalizeSource(source) {
  const s = String(source || '').trim();
  if (!s) return '';
  if (/^https?:\/\//i.test(s)) {
    try {
      const u = new URL(s);
      // Drop query/fragment — the same video re-pasted with tracking params
      // is still the same project.
      return `${u.origin}${u.pathname}`.toLowerCase().replace(/\/$/, '');
    } catch {
      return s.toLowerCase();
    }
  }
  return s.toLowerCase();
}

/**
 * Group history entries into projects by source video.
 * @returns {Array} newest-first: { id, source, sourceType, title, jobCount,
 *   clipCount, latestTimestamp, status, jobs }
 */
export function deriveProjects(history) {
  const groups = new Map();
  for (const h of Array.isArray(history) ? history : []) {
    if (!h || typeof h.jobId !== 'string') continue;
    const key = normalizeSource(h.source) || h.jobId;
    let g = groups.get(key);
    if (!g) {
      g = {
        id: key,
        source: h.source || h.jobId,
        sourceType: h.sourceType || 'url',
        title: h.title || h.source || h.jobId,
        jobCount: 0,
        clipCount: 0,
        latestTimestamp: 0,
        status: 'complete',
        jobs: [],
      };
      groups.set(key, g);
    }
    g.jobs.push(h);
    g.jobCount += 1;
    g.clipCount += Number(h.clipCount) || 0;
    const ts = Number(h.timestamp) || 0;
    if (ts > g.latestTimestamp) {
      g.latestTimestamp = ts;
      g.title = h.title || h.source || g.title;
      g.sourceType = h.sourceType || g.sourceType;
    }
    // Status: processing dominates, then error, else complete.
    if (h.status === 'processing') g.status = 'processing';
    else if (h.status === 'error' && g.status !== 'processing') g.status = 'error';
  }
  return [...groups.values()].sort((a, b) => b.latestTimestamp - a.latestTimestamp);
}

/** The most recently completed job in a project — the one we open. */
export function projectOpenJob(project) {
  const jobs = [...(project?.jobs || [])].sort(
    (a, b) => (Number(b.timestamp) || 0) - (Number(a.timestamp) || 0)
  );
  return jobs.find((j) => j.status === 'complete') || jobs[0] || null;
}

function StatusBadge({ status }) {
  if (status === 'complete') return <Badge tone="teal" icon="check">complete</Badge>;
  if (status === 'error') return <Badge tone="danger" icon="triangle-alert">error</Badge>;
  return <Badge tone="amber" icon="loader">rendering</Badge>;
}

export function ProjectsView({ projects, onSelect, onDelete, backendReachable }) {
  if (!projects || projects.length === 0) {
    return (
      <div className="container narrow fade-in">
        <Hero eyebrow="Projects" line1="No projects yet."
          sub="Every video you turn into clips becomes a project, grouped automatically." />
        <div className="empty">
          <div className="ei"><Icon n="clapperboard" /></div>
          <h3>Nothing to group yet</h3>
          <p>
            {backendReachable === false
              ? 'The backend is unreachable — connect it and run your first job to see projects here.'
              : 'Head to Create, paste a link, and your finished video will land here as a project.'}
          </p>
        </div>
      </div>
    );
  }
  return (
    <div className="container fade-in">
      <div className="results-head" style={{ marginBottom: 18 }}>
        <h2>Projects</h2>
        <Badge tone="out">{projects.length} project{projects.length === 1 ? '' : 's'}</Badge>
      </div>
      <div className="proj-grid">
        {projects.map((p) => {
          const job = projectOpenJob(p);
          const openable = !!job;
          return (
            <div
              key={p.id}
              className={`panel raised proj-card${openable ? ' clickable' : ''}`}
              onClick={openable ? () => onSelect(p) : undefined}
              role={openable ? 'button' : undefined}
              tabIndex={openable ? 0 : undefined}
              onKeyDown={openable ? (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); onSelect(p); } } : undefined}
              aria-label={openable ? `Open project ${p.title}` : undefined}
            >
              <div className="proj-thumb" aria-hidden="true">
                <Icon n={p.sourceType === 'url' ? 'globe' : 'file-video'} />
                <span className="proj-count">{p.clipCount}</span>
              </div>
              <div className="proj-body">
                <div className="proj-title" title={p.title}>{p.title}</div>
                <div className="proj-meta">
                  {p.clipCount} clip{p.clipCount === 1 ? '' : 's'}
                  {p.jobCount > 1 && ` · ${p.jobCount} runs`}
                  {p.latestTimestamp > 0 && ` · ${relTime(p.latestTimestamp)}`}
                </div>
                <div className="proj-foot">
                  <StatusBadge status={p.status} />
                  {onDelete && job && (
                    <button
                      type="button" className="mini" title="Delete project" aria-label={`Delete project ${p.title}`}
                      onClick={(e) => { e.stopPropagation(); onDelete(p); }}
                    >
                      <Icon n="trash-2" />
                    </button>
                  )}
                </div>
              </div>
              <Icon n="chevron-right" cls="proj-chev" />
            </div>
          );
        })}
      </div>
      <div className="od" style={{ marginTop: 16 }}>
        Opening a project loads its clips — the same clips you&apos;d see in History.
      </div>
    </div>
  );
}
