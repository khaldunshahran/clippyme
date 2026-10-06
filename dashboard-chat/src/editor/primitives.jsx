import { useId, useRef, useEffect } from 'react';
import { Icon, Social } from './icon';

export { Icon, Social };

export function Btn({
  variant = 'primary',
  size,
  block,
  icon,
  iconRight,
  children,
  loading = false,
  disabled,
  type = 'button',
  className = '',
  ...props
}) {
  const classes = [
    'btn',
    variant && `btn-${variant}`,
    size && `btn-${size}`,
    block && 'btn-block',
    className,
  ].filter(Boolean).join(' ');

  return (
    <button
      {...props}
      type={type}
      className={classes}
      disabled={disabled || loading}
      aria-busy={loading || undefined}
    >
      {(loading || icon) && <Icon n={loading ? 'loader' : icon} />}
      <span>{children}</span>
      {!loading && iconRight && <Icon n={iconRight} />}
    </button>
  );
}

export function Badge({ tone = 'out', icon, children, className = '', ...props }) {
  return (
    <span {...props} className={`badge badge-${tone}${className ? ` ${className}` : ''}`}>
      {icon && <Icon n={icon} />}
      {children}
    </span>
  );
}

export function Switch({ on, checked, onChange, disabled, label = 'Toggle option', ...props }) {
  const isOn = !!(on ?? checked);
  return (
    <button
      {...props}
      type="button"
      role="switch"
      aria-checked={isOn}
      aria-label={label}
      disabled={disabled}
      className={`sw${isOn ? ' on' : ''}`}
      onClick={(event) => {
        event.stopPropagation();
        onChange?.(!isOn);
      }}
    >
      <i aria-hidden="true" />
    </button>
  );
}

export function Segmented({ options, value, onChange, full, blue, label = 'Choose an option' }) {
  const refs = useRef([]);
  const move = (event, index) => {
    if (!['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown'].includes(event.key)) return;
    event.preventDefault();
    const direction = event.key === 'ArrowLeft' || event.key === 'ArrowUp' ? -1 : 1;
    const next = (index + direction + options.length) % options.length;
    onChange(options[next].id);
    refs.current[next]?.focus();
  };

  return (
    <div className={`seg${full ? ' full' : ''}${blue ? ' blue' : ''}`} role="group" aria-label={label}>
      {options.map((option, index) => (
        <button
          key={option.id}
          ref={(node) => { refs.current[index] = node; }}
          type="button"
          aria-pressed={value === option.id}
          className={value === option.id ? 'on' : ''}
          onClick={() => onChange(option.id)}
          onKeyDown={(event) => move(event, index)}
        >
          {option.icon && <Icon n={option.icon} />}
          {option.label}
        </button>
      ))}
    </div>
  );
}

export function Stepper({ value, set, min = 1, max = 12, label = 'Value' }) {
  return (
    <div className="stepper" role="group" aria-label={label}>
      <button
        type="button"
        disabled={value <= min}
        onClick={() => set(Math.max(min, value - 1))}
        aria-label={`Decrease ${label}`}
      >
        –
      </button>
      <output aria-live="polite">{value}</output>
      <button
        type="button"
        disabled={value >= max}
        onClick={() => set(Math.min(max, value + 1))}
        aria-label={`Increase ${label}`}
      >
        +
      </button>
    </div>
  );
}

export function Panel({
  title,
  sub,
  icon,
  headRight,
  pad = true,
  children,
  className,
  style,
  as: Tag = 'section',
  headingAs: Heading = 'h3',
  headingLevel = 2,
}) {
  const titleId = useId();
  return (
    <Tag
      className={`panel raised${className ? ` ${className}` : ''}`}
      style={style}
      aria-labelledby={title ? titleId : undefined}
    >
      {title && (
        <div className="panel-head">
          {icon && (
            <div className="ico" aria-hidden="true">
              <Icon n={icon} />
            </div>
          )}
          <div>
            <Heading id={titleId} aria-level={headingLevel}>
              {title}
            </Heading>
            {sub && <div className="sub">{sub}</div>}
          </div>
          {headRight && <div className="right">{headRight}</div>}
        </div>
      )}
      <div className={pad ? 'panel-pad' : ''}>{children}</div>
    </Tag>
  );
}

export const PLATFORMS = [
  { id: 'tiktok', icon: 'tiktok', label: 'TikTok' },
  { id: 'ig', icon: 'instagram', label: 'Reels' },
  { id: 'yt', icon: 'youtube', label: 'Shorts' },
  { id: 'fb', icon: 'facebook', label: 'Facebook' },
];

export function PlatPill({ id, icon, label, on, onClick }) {
  return (
    <button
      type="button"
      className={`plat${on ? ` on ${id}` : ''}`}
      aria-pressed={!!on}
      onClick={onClick}
    >
      <Social n={icon} color={on ? '332E26' : '736B5B'} />
      {label}
    </button>
  );
}

export function RingGauge({
  score = 0,
  max = 100,
  size = 'med',
  accent = 'var(--gold)',
  label,
  sublabel,
  className = '',
  style,
}) {
  const pct = Math.min(100, Math.max(0, Math.round((score / max) * 100)));
  const ringRef = useRef(null);

  useEffect(() => {
    if (ringRef.current) {
      ringRef.current.style.setProperty('--pct', `${pct}%`);
    }
  }, [pct]);

  return (
    <div
      ref={ringRef}
      role="progressbar"
      aria-valuenow={score}
      aria-valuemin={0}
      aria-valuemax={max}
      aria-label={label || 'Score gauge'}
      className={`ring ${size} ${className}`}
      style={{ '--accent': accent, '--pct': `${pct}%`, ...style }}
    >
      <div className="ring-hole">
        <b>{score}</b>
        {sublabel && <span>{sublabel}</span>}
      </div>
      <span className="sr-only" style={{ position: 'absolute', width: 1, height: 1, overflow: 'hidden', clip: 'rect(0,0,0,0)' }}>
        {label ? `${label}: ${score}/${max}` : `${score} out of ${max}`}
      </span>
    </div>
  );
}

export function StatusDot({ status = 'online', label, className = '' }) {
  const isOnline = status === 'online' || status === 'watching';
  const isBusy = status === 'busy' || status === 'capturing' || status === 'working';
  return (
    <span
      className={`status-dot${!isOnline && !isBusy ? ' offline' : ''} ${className}`}
      role="status"
      aria-live="polite"
    >
      <i
        aria-hidden="true"
        style={
          isBusy
            ? { background: 'var(--rust)', boxShadow: '0 0 0 3px rgba(187,90,60,0.25)' }
            : undefined
        }
      />
      <span>{label || status}</span>
    </span>
  );
}

export function PebbleWaveform({ count = 18, activeCount = 0, className = '' }) {
  const pebbles = useRef([]);
  if (pebbles.current.length === 0) {
    for (let i = 0; i < count; i++) {
      const d = Math.round(6 + (Math.sin(i * 0.4) * 5 + 6));
      pebbles.current.push({ d, peak: d > 13 });
    }
  }

  return (
    <div className={`pebbles ${className}`} aria-hidden="true">
      {pebbles.current.map((p, idx) => (
        <i
          key={idx}
          className={idx < activeCount || p.peak ? 'peak' : ''}
          style={{ width: `${p.d}px`, height: `${p.d}px` }}
        />
      ))}
    </div>
  );
}
