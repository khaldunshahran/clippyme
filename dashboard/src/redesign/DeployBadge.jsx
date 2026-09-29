// Nugget — Deploy badge. Shows the deployed build's git short SHA in the
// sidebar footer so the build on screen is always verifiable. Falls back to
// a subtle "dev" when the deploy env vars aren't set (local dev).
import { Icon } from './icon';

function deployInfo() {
  const short = (import.meta.env.VITE_DEPLOY_SHA || '').trim();
  if (!short) return null;
  return {
    short,
    full: (import.meta.env.VITE_DEPLOY_SHA_FULL || short).trim(),
    time: (import.meta.env.VITE_DEPLOY_TIME || '').trim(),
    subject: (import.meta.env.VITE_DEPLOY_SUBJECT || '').trim(),
  };
}

export function DeployBadge() {
  const info = deployInfo();
  const detail = info
    ? [`Full SHA: ${info.full}`, info.time && `Committed: ${info.time}`, info.subject && `“${info.subject}”`]
        .filter(Boolean)
        .join('\n')
    : 'Local development build — no deploy SHA set.';
  return (
    <button
      type="button"
      className="deploy-badge"
      title={detail}
      aria-label={info ? `Deployed build ${info.full}` : 'Development build'}
      onClick={() => {
        // Tap-to-reveal on touch devices where hover doesn't exist.
        try {
          // eslint-disable-next-line no-alert
          alert(detail);
        } catch { /* noop */ }
      }}
    >
      <Icon n="git-commit" />
      <span className="deploy-badge-id">{info ? info.short : 'dev'}</span>
    </button>
  );
}
